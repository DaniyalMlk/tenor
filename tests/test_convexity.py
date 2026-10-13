"""Tests for the futures convexity adjustment.

Nothing here checks a closed form against itself. The risk-neutral mean of the
short rate is checked against a Simpson quadrature of the model's own drift,
built from ``theta(t)`` rather than from the ``alpha(t)`` identity the module
uses, so the two share no algebra. The mean and variance of the integrated rate
are checked against quadratures of the mean function and of the
Ornstein-Uhlenbeck covariance, the latter over a triangle so the kink on the
diagonal sits on the boundary rather than through the middle of a panel. The
term convention's growth factor is checked against a quadrature over the
terminal short rate using :meth:`~tenor.hullwhite.HullWhite.bond_price`, which
is a separate code path from :meth:`~tenor.hullwhite.HullWhite.log_a` plus a
lognormal moment. And one Euler simulation with an explicit ``theta`` closes the
loop end to end, at a volatility chosen so the measure-change term is resolved
rather than buried in the noise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import date

import pytest

from tenor.bootstrap import bootstrap
from tenor.convexity import (
    Adjustment,
    RateMoments,
    Settlement,
    _integrated_decay,
    continuous_forward_rate,
    convexity_adjustment,
    expected_settlement_rate,
    forward_settlement_rate,
    ho_lee_limit,
    integrated_rate_moments,
    risk_neutral_drift,
    short_rate_moments,
    term_growth_moments,
)
from tenor.curve import DiscountCurve, flat_curve
from tenor.daycount import Basis, year_fraction
from tenor.hullwhite import BadModel, HullWhite
from tenor.instruments import Deposit, Future, ho_lee_convexity

REFERENCE = date(2024, 1, 1)
RATE = 0.04


def level() -> DiscountCurve:
    """A flat 4% continuously compounded curve, so ``f(0, t)`` is known."""
    return flat_curve(REFERENCE, date(2064, 1, 1), RATE, basis=Basis.ACT_365F)


def _add_years(start: date, years: float) -> date:
    return date.fromordinal(start.toordinal() + round(years * 365.0))


def sloped() -> DiscountCurve:
    """A rising curve, so nothing here depends on a flat instantaneous forward."""
    zeros = {
        0.5: 0.030,
        1.0: 0.032,
        2.0: 0.035,
        3.0: 0.037,
        5.0: 0.040,
        7.0: 0.042,
        10.0: 0.043,
        15.0: 0.0435,
        20.0: 0.044,
        30.0: 0.0445,
    }
    return DiscountCurve.from_zeros(
        REFERENCE,
        [(_add_years(REFERENCE, time), rate) for time, rate in zeros.items()],
        basis=Basis.ACT_365F,
    )


def simpson(f: Callable[[float], float], lo: float, hi: float, panels: int) -> float:
    """Composite Simpson with an even number of panels."""
    assert panels % 2 == 0
    step = (hi - lo) / panels
    total = f(lo) + f(hi)
    for index in range(1, panels):
        total += f(lo + index * step) * (4.0 if index % 2 else 2.0)
    return total * step / 3.0


# -- the measure change itself ------------------------------------------------


def test_the_drift_is_zero_now_and_saturates_later() -> None:
    model = HullWhite(0.1, 0.01)
    assert risk_neutral_drift(model, 0.0) == 0.0
    ceiling = model.sigma**2 / (2.0 * model.a**2)
    assert risk_neutral_drift(model, 1000.0) == pytest.approx(ceiling, rel=1e-15)
    walk = [risk_neutral_drift(model, t) for t in (0.5, 1.0, 2.0, 5.0, 10.0, 40.0)]
    assert walk == sorted(walk)
    assert all(0.0 < value < ceiling for value in walk)


def test_the_drift_is_second_order_in_time_so_the_front_contract_has_none() -> None:
    """``sigma^2 t^2 / 2`` for small ``t``, which is why the near strip is flat."""
    model = HullWhite(0.1, 0.01)
    for time in (1e-3, 1e-4):
        expected = 0.5 * model.sigma**2 * time**2
        assert risk_neutral_drift(model, time) == pytest.approx(expected, rel=2e-4)


def test_a_negative_time_is_refused() -> None:
    with pytest.raises(BadModel, match="cannot be negative"):
        risk_neutral_drift(HullWhite(0.1, 0.01), -1.0)


def test_the_short_rate_mean_matches_a_quadrature_of_the_model_drift() -> None:
    """``E[r(T)]`` against ``r(0) e^-aT + int e^-a(T-s) theta(s) ds``.

    ``theta`` is written out from the model's own stochastic differential
    equation on a flat curve, where ``f(0,t)`` is constant and its derivative is
    zero. That integral is a different object from the ``alpha(t)`` closed form
    the module uses, so agreement is a check and not a tautology. The floor on
    the tolerance is the curve's instantaneous forward, which is a bumped
    difference and is itself only good to about three parts in ten billion.
    """
    curve = level()
    for speed in (0.01, 0.03, 0.3, 1.0):
        model = HullWhite(speed, 0.012)

        def theta(time: float, speed: float = speed, model: HullWhite = model) -> float:
            return speed * RATE + model.sigma**2 * -math.expm1(-2.0 * speed * time) / (
                2.0 * speed
            )

        for horizon in (0.5, 2.0, 5.0, 10.0, 25.0):
            def discounted(
                time: float, horizon: float = horizon, speed: float = speed
            ) -> float:
                return math.exp(-speed * (horizon - time)) * theta(time)

            integral = simpson(discounted, 0.0, horizon, 4000)
            quadrature = RATE * math.exp(-speed * horizon) + integral
            assert short_rate_moments(curve, model, horizon).mean == pytest.approx(
                quadrature, rel=5e-9
            )


def test_the_variance_is_the_models_own_and_the_measure_does_not_move_it() -> None:
    curve = level()
    model = HullWhite(0.08, 0.009)
    for horizon in (0.25, 3.0, 12.0):
        assert short_rate_moments(curve, model, horizon).variance == model.short_rate_variance(
            horizon
        )


def test_the_mean_is_the_curves_forward_plus_the_drift_on_a_sloped_curve() -> None:
    curve = sloped()
    model = HullWhite(0.05, 0.011)
    for horizon in (1.0, 4.0, 9.0):
        moments = short_rate_moments(curve, model, horizon)
        assert moments.mean == pytest.approx(
            curve.instantaneous_forward(horizon) + risk_neutral_drift(model, horizon),
            rel=1e-14,
        )


# -- the integrated rate ------------------------------------------------------


def test_the_integrated_mean_matches_a_quadrature_of_the_mean_function() -> None:
    curve = level()
    for speed in (0.02, 0.3):
        model = HullWhite(speed, 0.015)
        def mean_at(time: float, model: HullWhite = model) -> float:
            return short_rate_moments(curve, model, time).mean

        for start, length in ((0.0, 0.25), (2.0, 0.5), (5.0, 0.25), (9.0, 1.0)):
            quadrature = simpson(mean_at, start, start + length, 400)
            assert integrated_rate_moments(curve, model, start, start + length).mean == (
                pytest.approx(quadrature, rel=1e-8)
            )


def test_the_integrated_variance_matches_the_ornstein_uhlenbeck_covariance() -> None:
    """``Var(int r) = int int Cov(r(u), r(w))``, integrated over a triangle.

    The covariance ``(sigma^2 / 2a) (e^-a|u-w| - e^-a(u+w))`` has a kink on the
    diagonal. Integrating over the square puts that kink through the interior of
    every panel and a Simpson rule then converges at a crawl — it lands about
    five parts in a million out at eighty panels a side. Using the symmetry to
    integrate over the lower triangle instead puts the kink on the boundary and
    the same rule reaches eleven digits, which is the panel-edge rule from the
    quadrature work in this package applied to a two-dimensional domain.
    """
    curve = level()
    for speed in (0.03, 0.3):
        model = HullWhite(speed, 0.013)
        variance = model.sigma**2 / (2.0 * speed)

        def covariance(first: float, second: float, v: float = variance, s: float = speed) -> float:
            return v * (math.exp(-s * abs(first - second)) - math.exp(-s * (first + second)))

        for start, length in ((0.0, 0.5), (2.0, 0.5), (5.0, 0.25)):
            end = start + length

            def strip(upper: float, floor: float = start) -> float:
                if upper <= floor:
                    return 0.0

                def row(lower: float, fixed: float = upper) -> float:
                    return covariance(fixed, lower)

                return 2.0 * simpson(row, floor, upper, 240)

            assert integrated_rate_moments(curve, model, start, end).variance == pytest.approx(
                simpson(strip, start, end, 240), rel=1e-7
            )


def test_the_variance_splits_into_what_was_known_and_what_happens_inside() -> None:
    """The first piece is ``B(L)^2 Var(r(T1))`` and vanishes for a window starting now."""
    curve = level()
    model = HullWhite(0.06, 0.012)
    length = 0.25
    now = integrated_rate_moments(curve, model, 0.0, length)
    later = integrated_rate_moments(curve, model, 5.0, 5.0 + length)
    carried = model.b(length) ** 2 * model.short_rate_variance(5.0)
    assert later.variance - now.variance == pytest.approx(carried, rel=1e-13)
    assert now.variance > 0.0


def test_the_series_branch_and_the_direct_branch_meet_at_the_threshold() -> None:
    """The two branches of the integrated decay agree where they hand over.

    Reaching into the private helper is deliberate: comparing two reversion
    speeds either side of the cutoff would mix the branch change with the
    genuine dependence of the answer on the speed, which is of the same size.
    The direct antiderivative is written out here so the comparison is against
    arithmetic this file owns.
    """

    def direct(speed: float, time: float) -> float:
        decay = -math.expm1(-speed * time) / speed
        return time - 2.0 * decay - math.expm1(-2.0 * speed * time) / (2.0 * speed)

    def series(speed: float, time: float) -> float:
        x = speed * time
        return (x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0 - x**6 / 24.0) / speed

    threshold = 1.0e-3
    for time in (0.0833, 0.25, 1.0, 5.0):
        speed = threshold / time
        # Just below the cutoff the module uses the series; the direct form still
        # has ten digits there, which is why the cutoff is where it is.
        assert _integrated_decay(speed * 0.999, time) == pytest.approx(
            direct(speed * 0.999, time), rel=1e-9
        )
        # Just above it the module uses the direct form and the series is still
        # good to a relative O(x^4).
        assert _integrated_decay(speed * 1.001, time) == pytest.approx(
            series(speed * 1.001, time), rel=1e-9
        )


def test_the_integrated_decay_is_the_integral_it_claims_to_be() -> None:
    """Against a Simpson quadrature of ``(1 - e^-au)^2``, away from the cutoff."""
    for speed in (0.03, 0.3, 2.0):
        for time in (0.25, 2.0, 10.0):
            def decay(u: float, speed: float = speed) -> float:
                return (-math.expm1(-speed * u)) ** 2

            quadrature = simpson(decay, 0.0, time, 2000)
            assert _integrated_decay(speed, time) == pytest.approx(quadrature, rel=1e-9)


# -- the term convention's growth factor --------------------------------------


def test_the_term_growth_factor_matches_a_quadrature_of_the_bond_price() -> None:
    """``E[1/P(T1,T2)]`` against a quadrature over the terminal short rate.

    The quadrature calls :meth:`~tenor.hullwhite.HullWhite.bond_price`, which
    exponentiates ``log_a`` once and multiplies; the closed form takes the
    logarithm of the same object and applies a lognormal moment. The two reach
    the same number from different arithmetic.
    """
    curve = level()
    for speed, vol in ((0.03, 0.01), (0.25, 0.02)):
        model = HullWhite(speed, vol)
        for start, end in ((1.0, 1.25), (5.0, 5.5), (10.0, 11.0)):
            rate = short_rate_moments(curve, model, start)
            spread = math.sqrt(rate.variance)
            density = 1.0 / math.sqrt(2.0 * math.pi)

            def weighted(
                z: float,
                model: HullWhite = model,
                start: float = start,
                end: float = end,
                rate: RateMoments = rate,
                spread: float = spread,
                density: float = density,
            ) -> float:
                price = model.bond_price(curve, start, end, rate.mean + spread * z)
                return density * math.exp(-0.5 * z * z) / price

            integral = simpson(weighted, -9.0, 9.0, 4000)
            assert term_growth_moments(curve, model, start, end).lognormal_mean == (
                pytest.approx(integral, rel=1e-11)
            )


def test_the_growth_factor_is_the_forward_ratio_when_nothing_is_random() -> None:
    curve = level()
    model = HullWhite(0.05, 0.0)
    ratio = curve.discount_at(2.0) / curve.discount_at(2.5)
    assert term_growth_moments(curve, model, 2.0, 2.5).lognormal_mean == pytest.approx(
        ratio, rel=1e-13
    )
    assert integrated_rate_moments(curve, model, 2.0, 2.5).lognormal_mean == pytest.approx(
        ratio, rel=1e-13
    )


# -- an end-to-end simulation -------------------------------------------------


def _normals(count: int, seed: int) -> list[float]:
    """Box-Muller from ``random.Random``, so the draws do not depend on CPython's
    own normal variate algorithm and the test is reproducible across versions."""
    source = random.Random(seed)
    out: list[float] = []
    while len(out) < count:
        first, second = source.random(), source.random()
        radius = math.sqrt(-2.0 * math.log(1.0 - first))
        angle = 2.0 * math.pi * second
        out.append(radius * math.cos(angle))
        out.append(radius * math.sin(angle))
    return out[:count]


def test_an_euler_simulation_reproduces_the_drift_and_the_integrated_moments() -> None:
    """One simulation under the risk-neutral measure, with ``theta`` written out.

    The volatility is 20% absolute rather than the one basis point a day a real
    curve carries, because the measure-change term is quadratic in the
    volatility and the sampling noise is linear in it: at a realistic
    volatility the term to be resolved is a hundredth of a standard error and
    half a million paths would be needed to see it. At this volatility it is
    five standard errors wide with sixteen hundred antithetic pairs, which is
    the smallest honest version of the check.
    """
    curve = level()
    speed, vol = 0.1, 0.2
    model = HullWhite(speed, vol)
    start, end = 2.0, 2.5
    steps, pairs = 500, 1600
    step = end / steps
    root = math.sqrt(step)
    draws = _normals(steps * pairs, 20241013)

    terminal: list[float] = []
    integrals: list[float] = []
    crossing = round(start / step)
    for pair in range(pairs):
        for sign in (1.0, -1.0):
            rate = RATE
            running = 0.0
            held = 0.0
            for index in range(steps):
                time = index * step
                theta = speed * RATE + vol**2 * -math.expm1(-2.0 * speed * time) / (2.0 * speed)
                nxt = rate + (theta - speed * rate) * step + vol * root * sign * draws[
                    pair * steps + index
                ]
                if index >= crossing:
                    running += 0.5 * (rate + nxt) * step
                if index + 1 == crossing:
                    held = nxt
                rate = nxt
            terminal.append(held)
            integrals.append(running)

    count = len(terminal)
    mean_rate = sum(terminal) / count
    spread = math.sqrt(sum((x - mean_rate) ** 2 for x in terminal) / (count - 1) / count)
    expected = short_rate_moments(curve, model, start)
    drift = risk_neutral_drift(model, start)
    # The whole point: the simulated mean sits above the curve's forward by the
    # drift, and the drift is several standard errors wide.
    assert drift > 4.0 * spread
    assert mean_rate == pytest.approx(expected.mean, abs=3.0 * spread)
    assert abs(mean_rate - RATE) > 0.5 * drift

    mean_integral = sum(integrals) / count
    variance = sum((x - mean_integral) ** 2 for x in integrals) / (count - 1)
    predicted = integrated_rate_moments(curve, model, start, end)
    error = math.sqrt(variance / count)
    assert mean_integral == pytest.approx(predicted.mean, abs=3.5 * error)
    assert variance == pytest.approx(predicted.variance, rel=0.2)


# -- the Ho-Lee comparison ----------------------------------------------------


def _continuous_adjustment(
    curve: DiscountCurve, model: HullWhite, start: float, end: float, settlement: Settlement
) -> float:
    """The adjustment as a continuous rate, which is what the limits are stated in."""
    reference = math.log(curve.discount_at(start) / curve.discount_at(end))
    if settlement is Settlement.AVERAGED:
        return (integrated_rate_moments(curve, model, start, end).mean - reference) / (end - start)
    moments = (
        term_growth_moments(curve, model, start, end)
        if settlement is Settlement.TERM
        else integrated_rate_moments(curve, model, start, end)
    )
    return (moments.mean + 0.5 * moments.variance - reference) / (end - start)


@pytest.mark.parametrize("settlement", list(Settlement))
def test_the_limit_is_reached_as_the_reversion_vanishes(settlement: Settlement) -> None:
    """And the remainder is first order in the reversion speed, which is why the
    small-argument series in the integrated decay is needed to see it at all."""
    curve = level()
    for start, length in ((0.5, 0.25), (2.0, 0.5), (5.0, 0.25), (1.0, 1.0)):
        end = start + length
        limit = ho_lee_limit(0.01, start, end, settlement=settlement)
        errors = []
        for speed in (1e-3, 1e-4, 1e-5):
            value = _continuous_adjustment(curve, HullWhite(speed, 0.01), start, end, settlement)
            errors.append(abs(value / limit - 1.0))
        assert errors[-1] < 1e-4
        assert errors == sorted(errors, reverse=True)
        assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)


def test_the_term_limit_is_the_textbook_formula_plus_the_jensen_term() -> None:
    """Exactly ``ho_lee_convexity`` plus ``0.5 sigma^2 T1 L``, to the last bit."""
    for vol in (0.005, 0.01, 0.02):
        for start, length in ((0.25, 0.25), (1.0, 1.0), (5.0, 0.25), (10.0, 0.5)):
            end = start + length
            jensen = 0.5 * vol**2 * start * length
            assert ho_lee_limit(vol, start, end) - jensen == pytest.approx(
                ho_lee_convexity(vol, start, end), rel=1e-15, abs=0.0
            )


def test_the_averaged_limit_beats_the_textbook_one_by_the_window_squared() -> None:
    for start, length in ((1.0, 0.25), (5.0, 0.5), (10.0, 1.0)):
        end = start + length
        gap = ho_lee_limit(0.01, start, end, settlement=Settlement.AVERAGED) - ho_lee_convexity(
            0.01, start, end
        )
        assert gap == pytest.approx(0.01**2 * length**2 / 6.0, rel=1e-12)


def test_all_three_conventions_collapse_as_the_window_shrinks() -> None:
    """There is only one rate in an instant, so there is no convention to choose."""
    start = 4.0
    for length in (1e-3, 1e-4):
        end = start + length
        values = [ho_lee_limit(0.01, start, end, settlement=s) for s in Settlement]
        expected = 0.5 * 0.01**2 * start**2
        for value in values:
            assert value == pytest.approx(expected, rel=2.0 * length)


def test_the_ho_lee_error_changes_sign_inside_one_futures_strip() -> None:
    """The documented table, and the crossover found by bisection rather than read
    off the grid it was tabulated on."""
    curve = level()
    table = {
        (0.25, 0.01): 0.508,
        (0.25, 0.3): 0.322,
        (1.0, 0.03): 0.171,
        (1.0, 0.3): -0.137,
        (5.0, 0.03): -0.091,
        (10.0, 0.3): -0.897,
    }
    for (start, speed), expected in table.items():
        end = start + 0.25
        ratio = (
            convexity_adjustment(curve, HullWhite(speed, 0.01), start, end).total
            / ho_lee_convexity(0.01, start, end)
        )
        assert ratio - 1.0 == pytest.approx(expected, abs=5e-4)

    for speed, crossover in ((0.01, 5.31), (0.03, 2.82), (0.1, 1.39)):
        low, high = 0.05, 40.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            end = middle + 0.25
            ratio = (
                convexity_adjustment(curve, HullWhite(speed, 0.01), middle, end).total
                / ho_lee_convexity(0.01, middle, end)
            )
            low, high = (middle, high) if ratio > 1.0 else (low, middle)
        assert 0.5 * (low + high) == pytest.approx(crossover, abs=0.01)


def test_the_limit_refuses_a_negative_volatility() -> None:
    with pytest.raises(BadModel, match="non-negative"):
        ho_lee_limit(-0.01, 1.0, 1.25)


# -- the three conventions against each other ---------------------------------


GRID = [
    (start, length, speed, vol)
    for start in (0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 20.0)
    for length in (0.0833, 0.25, 0.5, 1.0)
    for speed in (0.005, 0.03, 0.1, 0.3, 1.0)
    for vol in (0.002, 0.01, 0.03)
]


def test_no_adjustment_is_ever_negative() -> None:
    """Which is what :class:`~tenor.instruments.Future` refuses to accept, and the
    reason is economic: margin pays the holder when rates rise."""
    curve = level()
    for start, length, speed, vol in GRID:
        model = HullWhite(speed, vol)
        for settlement in Settlement:
            adjustment = convexity_adjustment(
                curve, model, start, start + length, settlement=settlement
            )
            assert adjustment.total >= 0.0
            assert adjustment.drift >= 0.0
            assert adjustment.jensen >= 0.0


def test_the_compounded_convention_never_adjusts_less_than_the_term_one() -> None:
    """Both reference the same forward; the compounded one accumulates the whole
    path where the term one reads a single bond price."""
    curve = level()
    for start, length, speed, vol in GRID:
        model = HullWhite(speed, vol)
        term = convexity_adjustment(curve, model, start, start + length).total
        compounded = convexity_adjustment(
            curve, model, start, start + length, settlement=Settlement.COMPOUNDED
        ).total
        assert compounded >= term


def test_the_averaged_convention_has_no_jensen_term_and_is_the_drift_piece() -> None:
    """Its adjustment is the compounded one's drift divided by the window, exactly
    — the average is linear in the rate, so curvature contributes nothing."""
    curve = level()
    model = HullWhite(0.04, 0.012)
    for start, length in ((0.0, 0.25), (3.0, 0.5), (7.0, 1.0)):
        end = start + length
        averaged = convexity_adjustment(curve, model, start, end, settlement=Settlement.AVERAGED)
        assert averaged.jensen == 0.0
        assert averaged.drift == averaged.total
        reference = math.log(curve.discount_at(start) / curve.discount_at(end))
        drift = (integrated_rate_moments(curve, model, start, end).mean - reference) / length
        assert averaged.total == pytest.approx(drift, rel=1e-14)


def test_the_drift_and_jensen_pieces_stand_in_the_documented_ratio() -> None:
    """``1 + T1/L`` in the zero-reversion limit, and close to it in practice."""
    curve = level()
    model = HullWhite(0.03, 0.01)
    for start, length, expected in ((5.0, 0.25, 21.034), (5.0, 1.0, 6.061), (1.0, 1.0, 2.015)):
        adjustment = convexity_adjustment(curve, model, start, start + length)
        assert adjustment.drift / adjustment.jensen == pytest.approx(expected, rel=1e-3)


def test_a_term_contract_fixing_today_has_no_convexity_and_a_compounded_one_does() -> None:
    curve = level()
    model = HullWhite(0.03, 0.01)
    term = convexity_adjustment(curve, model, 0.0, 0.25, settlement=Settlement.TERM)
    assert term.total == 0.0
    assert term.futures == term.forward
    compounded = convexity_adjustment(curve, model, 0.0, 0.25, settlement=Settlement.COMPOUNDED)
    assert compounded.total > 0.0
    assert compounded.total * 1e4 == pytest.approx(0.0209, abs=5e-4)


def test_zero_volatility_gives_the_forward_exactly() -> None:
    curve = level()
    model = HullWhite(0.07, 0.0)
    for settlement in Settlement:
        adjustment = convexity_adjustment(curve, model, 3.0, 3.5, settlement=settlement)
        assert adjustment.total == 0.0
        assert adjustment.futures == adjustment.forward


def test_the_adjustment_grows_with_expiry_and_volatility_and_falls_with_reversion() -> None:
    curve = level()
    expiries = [
        convexity_adjustment(curve, HullWhite(0.03, 0.01), t, t + 0.25).total
        for t in (0.25, 1.0, 2.0, 5.0, 10.0)
    ]
    assert expiries == sorted(expiries)
    vols = [
        convexity_adjustment(curve, HullWhite(0.03, v), 5.0, 5.25).total
        for v in (0.002, 0.005, 0.01, 0.02)
    ]
    assert vols == sorted(vols)
    speeds = [
        convexity_adjustment(curve, HullWhite(a, 0.01), 5.0, 5.25).total
        for a in (0.01, 0.03, 0.1, 0.3, 1.0)
    ]
    assert speeds == sorted(speeds, reverse=True)
    # Quadratic in the volatility, to the extent the exponential allows.
    doubled = convexity_adjustment(curve, HullWhite(0.03, 0.004), 5.0, 5.25).total
    single = convexity_adjustment(curve, HullWhite(0.03, 0.002), 5.0, 5.25).total
    assert doubled / single == pytest.approx(4.0, rel=1e-3)


# -- the reference the adjustment is measured from ----------------------------


def test_the_simple_and_continuous_references_differ_by_more_than_the_near_convexity() -> None:
    """2.01 basis points on a quarterly window, against 0.73 of convexity at a year."""
    curve = level()
    gap = forward_settlement_rate(curve, 5.0, 5.25) - continuous_forward_rate(curve, 5.0, 5.25)
    assert gap * 1e4 == pytest.approx(2.0067, abs=1e-3)
    near = convexity_adjustment(curve, HullWhite(0.03, 0.01), 1.0, 1.25).total
    assert near * 1e4 == pytest.approx(0.7316, abs=1e-3)
    assert gap > near


def test_the_term_and_compounded_conventions_share_their_forward() -> None:
    curve = sloped()
    for start, end in ((1.0, 1.25), (4.0, 5.0)):
        simple = forward_settlement_rate(curve, start, end)
        assert forward_settlement_rate(curve, start, end, settlement=Settlement.COMPOUNDED) == (
            simple
        )
        assert forward_settlement_rate(curve, start, end, settlement=Settlement.AVERAGED) == (
            continuous_forward_rate(curve, start, end)
        )
        assert simple == pytest.approx(
            (curve.discount_at(start) / curve.discount_at(end) - 1.0) / (end - start), rel=1e-15
        )


def test_an_accrual_that_differs_from_the_window_scales_the_rate() -> None:
    """A three-month window quoted on ACT/360 accrues over 365/360 of its own length,
    and the rate is lower by exactly that ratio while the growth factor is not."""
    curve = level()
    model = HullWhite(0.03, 0.01)
    window = 0.25
    matched = expected_settlement_rate(curve, model, 5.0, 5.0 + window)
    stretched = expected_settlement_rate(
        curve, model, 5.0, 5.0 + window, accrual=window * 365.0 / 360.0
    )
    assert stretched == pytest.approx(matched * 360.0 / 365.0, rel=1e-14)
    # The average does not accrue, so the accrual is ignored rather than applied.
    average = expected_settlement_rate(curve, model, 5.0, 5.25, settlement=Settlement.AVERAGED)
    assert expected_settlement_rate(
        curve, model, 5.0, 5.25, accrual=0.1, settlement=Settlement.AVERAGED
    ) == average


# -- the pieces add up --------------------------------------------------------


def test_the_total_is_the_two_pieces_and_the_two_rates() -> None:
    curve = sloped()
    model = HullWhite(0.05, 0.011)
    for settlement in Settlement:
        adjustment = convexity_adjustment(curve, model, 3.0, 3.5, settlement=settlement)
        assert adjustment.total == adjustment.futures - adjustment.forward
        assert adjustment.total == pytest.approx(adjustment.drift + adjustment.jensen, rel=1e-12)
        assert adjustment.futures == pytest.approx(
            expected_settlement_rate(curve, model, 3.0, 3.5, settlement=settlement), rel=1e-15
        )
        assert adjustment.forward == pytest.approx(
            forward_settlement_rate(curve, 3.0, 3.5, settlement=settlement), rel=1e-15
        )


def test_the_moments_lognormal_mean_is_the_exponential_of_the_two() -> None:
    moments = RateMoments(0.2, 0.05)
    assert moments.lognormal_mean == pytest.approx(math.exp(0.225), rel=1e-15)
    assert RateMoments(0.0, 0.0).lognormal_mean == 1.0


def test_an_adjustment_is_a_plain_record() -> None:
    adjustment = Adjustment(0.05, 0.049, 0.0008, 0.0002)
    assert adjustment.total == pytest.approx(0.001, rel=1e-12)


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (1.0, 1.0, "has no length"),
        (2.0, 1.5, "has no length"),
        (-0.5, 1.0, "cannot start"),
    ],
)
def test_a_window_that_is_not_a_window_is_refused(start: float, end: float, message: str) -> None:
    curve = level()
    with pytest.raises(BadModel, match=message):
        convexity_adjustment(curve, HullWhite(0.03, 0.01), start, end)


@pytest.mark.parametrize("accrual", [0.0, -0.25, float("nan")])
def test_a_non_positive_accrual_is_refused(accrual: float) -> None:
    curve = level()
    with pytest.raises(BadModel, match="accrual year fraction"):
        expected_settlement_rate(curve, HullWhite(0.03, 0.01), 1.0, 1.25, accrual=accrual)
    with pytest.raises(BadModel, match="accrual year fraction"):
        forward_settlement_rate(curve, 1.0, 1.25, accrual=accrual)


# -- the adjustment inside a bootstrap ----------------------------------------


def _quarterly_strip(count: int) -> list[date]:
    starts = [date(2024, 3, 20)]
    while len(starts) < count:
        previous = starts[-1]
        year, month = previous.year, previous.month + 3
        if month > 12:
            year, month = year + 1, month - 12
        starts.append(date(year, month, previous.day))
    return starts


def test_the_adjustment_recovers_the_curve_the_quotes_came_from() -> None:
    """The end-to-end check, with the curve known in advance.

    A flat 4% curve and a model generate futures *prices*. The curve is then
    thrown away and rebuilt from those prices twice: once with the adjustment
    supplied and once with it left at zero, which is what a bootstrapper that
    treats a futures rate as a forward rate does. The adjusted build returns the
    curve it started from; the naive one is biased high, and the bias
    accumulates down the strip rather than staying at the size of one
    adjustment — 3.99 basis points on the five-year zero where the largest
    single contract adjustment in the strip is 11.6.
    """
    truth = level()
    model = HullWhite(0.03, 0.01)
    starts = _quarterly_strip(20)

    quoted: list[Future] = []
    adjusted: list[Future] = []
    largest = 0.0
    for start in starts:
        year, month = start.year, start.month + 3
        if month > 12:
            year, month = year + 1, month - 12
        end = date(year, month, start.day)
        expiry, closing = truth.time_to(start), truth.time_to(end)
        accrual = year_fraction(start, end, Basis.ACT_360)
        rate = expected_settlement_rate(truth, model, expiry, closing, accrual=accrual)
        adjustment = convexity_adjustment(
            truth, model, expiry, closing, accrual=accrual
        ).total
        largest = max(largest, adjustment)
        price = 100.0 * (1.0 - rate)
        label = start.isoformat()
        quoted.append(Future(start, end, price, Basis.ACT_360, label=label))
        adjusted.append(
            Future(start, end, price, Basis.ACT_360, convexity=adjustment, label=label)
        )

    stub_accrual = year_fraction(REFERENCE, starts[0], Basis.ACT_360)
    stub_rate = (1.0 / truth.discount_at(truth.time_to(starts[0])) - 1.0) / stub_accrual
    stub = Deposit(REFERENCE, starts[0], stub_rate, Basis.ACT_360, "stub")

    naive = bootstrap(REFERENCE, [stub, *quoted], basis=Basis.ACT_365F)
    fixed = bootstrap(REFERENCE, [stub, *adjusted], basis=Basis.ACT_365F)
    assert naive.worst_error < 1e-12
    assert fixed.worst_error < 1e-12

    def zero_error(built: DiscountCurve, time: float) -> float:
        return (-math.log(built.discount_at(time)) / time - RATE) * 1e4

    assert largest * 1e4 == pytest.approx(11.64, abs=0.01)
    for time in (1.0, 2.0, 3.0, 4.0, 5.0):
        assert abs(zero_error(fixed.curve, time)) < 1e-6
    naive_errors = [zero_error(naive.curve, time) for time in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert naive_errors == sorted(naive_errors)
    assert all(error > 0.0 for error in naive_errors)
    assert naive_errors[-1] == pytest.approx(3.990, abs=0.01)


def test_the_three_conventions_on_one_contract_are_the_documented_numbers() -> None:
    """A three-month contract five years out, on a flat 4% curve at a hundred basis
    points: 11.931, 11.952 and 11.285 basis points, with the average's whole
    adjustment being its drift."""
    curve = level()
    model = HullWhite(0.03, 0.01)
    figures = {
        settlement: convexity_adjustment(curve, model, 5.0, 5.25, settlement=settlement)
        for settlement in Settlement
    }
    assert figures[Settlement.TERM].total * 1e4 == pytest.approx(11.931, abs=1e-3)
    assert figures[Settlement.COMPOUNDED].total * 1e4 == pytest.approx(11.952, abs=1e-3)
    assert figures[Settlement.AVERAGED].total * 1e4 == pytest.approx(11.285, abs=1e-3)
    assert figures[Settlement.COMPOUNDED].drift * 1e4 == pytest.approx(11.400, abs=1e-3)
