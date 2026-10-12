"""A forward curve fitted by least squares, and what the fit does not determine.

**The closed forms are checked against things that are not them.** The zero rate
is the average of the instantaneous forward over ``[0, t]``, so a composite
Simpson rule on the forward has to reproduce it; the integral of the hump
function has a series at small argument that agrees with the direct form to
within the direct form's own cancellation; and the derivative of the integral
has to be the forward it was the integral of. None of those checks uses the
module's own algebra twice.

**Recovery is the test with an exact answer in it.** Zero rates and bond prices
generated from a known parameter set are fitted back to that parameter set, so
the fit is measured against the truth rather than against its own residuals.
Both routes recover the levels to seven figures and the decay to five.

**The optimiser is measured against the problem it has to solve.** Svensson's
objective has a diagonal valley, and coordinate sweeps stop in it: on a curve
fitted to its own exact zero rates they return decays of 1.73 and 5.99 against a
truth of 1.6 and 9.0, and a root mean square of 3.2e-05 that looks like a
respectable fit. The simplex started from the same point reaches the truth and
1e-14. That comparison is a test, not a footnote, because the wrong answer is
the plausible-looking one.

**Then the parts that are not determined, which is the point of the phase.** The
decay band widens with the tolerance: a factor of 1.59 at one basis point and a
factor of 104 at ten. Across the wider band the slope and hump coefficients move
by about 4,700 while the level parameter moves by 0.028 — so the long end is
pinned, the shape is not, and every point of that band reprices the same quotes
within ten basis points. A reader who takes ``b1`` from a Nelson-Siegel fit as
"the slope of the curve" is reading a number the quotes did not determine.

**The conditioning is measured rather than asserted.** Solving the normal
equations instead of reflecting is wrong by 6.6e-10 on a Svensson design with
its decays a thousandth apart, against 6.4e-14 for the reflections — four orders
of magnitude, on a matrix with four columns.

**And the weighting is measured in both directions.** Yield weighting cuts the
worst yield error from 3.94 basis points to 2.60, all of it at the one-year
point, and makes the worst *price* error larger, 0.063 to 0.120 per 100. A test
that measured only the first would read as a free improvement.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import date, timedelta

import pytest

from tenor import (
    Basis,
    Bond,
    CurveFit,
    DecaySet,
    Family,
    Frequency,
    Interpolation,
    ParametricCurve,
    Weighting,
    fit_bonds,
    fit_zeros,
    solve_least_squares,
)
from tenor.parametric import (
    SERIES_LIMIT,
    BadFit,
    _condition,
    _forward_basis,
    _golden,
    _hump_integral,
    _integral_basis,
    _normal_equations,
    _simplex,
    _zero_basis,
)

REFERENCE = date(2026, 1, 15)
SETTLEMENT = date(2026, 1, 20)

NS_LEVELS = (0.045, -0.018, 0.028)
NS_DECAYS = (2.4,)
SV_LEVELS = (0.045, -0.018, 0.028, -0.012)
SV_DECAYS = (1.6, 9.0)

SAMPLE_TIMES = (0.05, 0.25, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0)


def nelson_siegel() -> ParametricCurve:
    return ParametricCurve(REFERENCE, Family.NELSON_SIEGEL, NS_LEVELS, NS_DECAYS)


def svensson() -> ParametricCurve:
    return ParametricCurve(REFERENCE, Family.SVENSSON, SV_LEVELS, SV_DECAYS)


def simpson(curve: ParametricCurve, upper: float, step_target: float = 0.0025) -> float:
    """Composite Simpson on the instantaneous forward. Nothing of the module's
    closed forms is used, so agreement is evidence about the closed form.

    The panel count is set from the interval rather than fixed, because Simpson's
    error is ``(b - a) h**4`` and a fixed count at fifty years is a quadrature
    error of 1e-10 — large enough to be mistaken for the closed form's.
    """
    panels = max(200, 2 * int(upper / step_target / 2))
    step = upper / panels
    total = curve.forward(0.0) + curve.forward(upper)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * curve.forward(index * step)
    return total * step / 3.0


# -- the closed forms against quadrature ---------------------------------------


@pytest.mark.parametrize("builder", [nelson_siegel, svensson])
def test_integral_is_the_integral_of_the_forward(
    builder: Callable[[], ParametricCurve],
) -> None:
    curve = builder()
    for upper in SAMPLE_TIMES:
        assert simpson(curve, upper) == pytest.approx(curve.integral(upper), abs=1e-13)


@pytest.mark.parametrize("builder", [nelson_siegel, svensson])
def test_zero_rate_is_the_average_forward(
    builder: Callable[[], ParametricCurve],
) -> None:
    curve = builder()
    for upper in SAMPLE_TIMES:
        assert simpson(curve, upper) / upper == pytest.approx(
            curve.zero(upper), abs=1e-13
        )


@pytest.mark.parametrize("builder", [nelson_siegel, svensson])
def test_integral_differentiates_back_to_the_forward(
    builder: Callable[[], ParametricCurve],
) -> None:
    curve = builder()
    step = 1e-5
    for middle in (0.5, 1.0, 4.0, 12.0):
        slope = (curve.integral(middle + step) - curve.integral(middle - step)) / (
            2.0 * step
        )
        assert slope == pytest.approx(curve.forward(middle), abs=1e-9)


def test_hump_series_beats_the_direct_form_where_it_cancels() -> None:
    """``1 - (1 + x) e**-x`` is a difference of two terms of order ``x`` whose
    answer is of order ``x**2``, so the direct form loses about ``log10(1 / x)``
    digits and the series loses none."""
    for argument, floor in ((1e-9, 1e-8), (1e-7, 1e-10), (1e-5, 1e-12)):
        direct = -math.expm1(-argument) - argument * math.exp(-argument)
        series = _hump_integral(argument)
        relative = abs(direct - series) / series
        assert relative > floor, (argument, relative)
    # And above the switch they agree to rounding, so the join is not a step.
    for argument in (SERIES_LIMIT * 1.0001, 1e-3, 0.1, 1.0, 10.0):
        direct = -math.expm1(-argument) - argument * math.exp(-argument)
        assert _hump_integral(argument) == pytest.approx(direct, rel=1e-13)


def test_the_basis_functions_have_their_limits_at_zero() -> None:
    assert _integral_basis(0.0, SV_DECAYS, Family.SVENSSON) == [0.0, 0.0, 0.0, 0.0]
    assert _zero_basis(0.0, SV_DECAYS, Family.SVENSSON) == [1.0, 1.0, 0.0, 0.0]
    assert _forward_basis(0.0, SV_DECAYS, Family.SVENSSON) == [1.0, 1.0, 0.0, 0.0]


def test_a_negative_time_is_refused_rather_than_returned() -> None:
    with pytest.raises(BadFit, match="negative time"):
        _hump_integral(-0.5)


# -- the curve -----------------------------------------------------------------


def test_the_anchor_needs_no_special_case() -> None:
    curve = svensson()
    assert curve.integral(0.0) == 0.0
    assert curve.discount_at(0.0) == 1.0
    assert curve.discount(REFERENCE) == 1.0
    assert curve.zero(0.0) == pytest.approx(curve.short_rate, rel=1e-15)


def test_the_forward_tends_to_the_level_parameter() -> None:
    curve = svensson()
    assert curve.forward(400.0) == pytest.approx(curve.long_rate, abs=1e-15)
    assert curve.zero(4000.0) == pytest.approx(curve.long_rate, abs=1e-4)


def test_the_short_rate_is_the_level_plus_the_slope() -> None:
    curve = nelson_siegel()
    assert curve.short_rate == pytest.approx(NS_LEVELS[0] + NS_LEVELS[1], rel=1e-15)
    assert curve.forward(0.0) == pytest.approx(curve.short_rate, rel=1e-15)


def test_sampling_into_a_pillar_curve_is_exact_at_the_pillars() -> None:
    curve = svensson()
    days = [REFERENCE + timedelta(days=step) for step in (0, 182, 365, 1095, 3650)]
    sampled = curve.to_discount_curve(days)
    for day in days:
        assert sampled.discount(day) == pytest.approx(curve.discount(day), rel=1e-15)


def test_sampling_error_between_pillars_falls_with_density() -> None:
    """Log-linear sampling of a smooth form: 2.0e-03 at yearly pillars, 1.8e-04
    at quarterly, 2.2e-05 at monthly. A reason to sample densely rather than a
    reason to avoid the conversion, since the pillars themselves are exact."""
    curve = svensson()
    errors = []
    for spacing in (365, 90, 30):
        days = [
            REFERENCE + timedelta(days=step)
            for step in range(0, 3651, spacing)
        ]
        if days[-1] != REFERENCE + timedelta(days=3650):
            days.append(REFERENCE + timedelta(days=3650))
        sampled = curve.to_discount_curve(days)
        worst = max(
            abs(sampled.discount(REFERENCE + timedelta(days=step)) - curve.discount(
                REFERENCE + timedelta(days=step)
            ))
            for step in range(0, 3651, 7)
        )
        errors.append(worst)
    assert errors[0] > errors[1] > errors[2]
    assert 1e-3 < errors[0] < 1e-2
    assert errors[2] < 5e-5


def test_the_price_matches_the_pillar_curve_it_samples_into() -> None:
    curve = svensson()
    bond = Bond(
        effective=date(2025, 7, 15),
        maturity=date(2034, 7, 15),
        coupon=0.04,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.THIRTY_360_BOND,
    )
    direct = curve.price(bond, SETTLEMENT)
    errors = []
    for spacing in (7, 3):
        days = [REFERENCE + timedelta(days=step) for step in range(0, 3651, spacing)]
        sampled = curve.to_discount_curve(days)
        errors.append(abs(bond.price_from_curve(sampled, SETTLEMENT) - direct))
    assert errors[0] < 2e-4
    # Second order in the spacing, so the ratio is about the square of 7 / 3.
    assert 3.5 < errors[0] / errors[1] < 7.0


def test_an_empty_sample_is_refused() -> None:
    with pytest.raises(BadFit, match="at least one date"):
        nelson_siegel().to_discount_curve([])


@pytest.mark.parametrize(
    ("family", "levels", "decays", "message"),
    [
        (Family.NELSON_SIEGEL, (0.04, 0.0), (2.0,), "3 level parameters"),
        (Family.SVENSSON, NS_LEVELS, SV_DECAYS, "4 level parameters"),
        (Family.NELSON_SIEGEL, NS_LEVELS, SV_DECAYS, "1 decay parameter"),
        (Family.NELSON_SIEGEL, NS_LEVELS, (0.0,), "must be positive"),
        (Family.NELSON_SIEGEL, NS_LEVELS, (-1.0,), "must be positive"),
        (Family.NELSON_SIEGEL, (0.04, float("nan"), 0.0), (2.0,), "not finite"),
    ],
)
def test_a_curve_validates_its_own_shape(
    family: Family,
    levels: tuple[float, ...],
    decays: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(BadFit, match=message):
        ParametricCurve(REFERENCE, family, levels, decays)


def test_the_family_knows_its_own_parameter_counts() -> None:
    assert (Family.NELSON_SIEGEL.levels, Family.NELSON_SIEGEL.decays) == (3, 1)
    assert (Family.SVENSSON.levels, Family.SVENSSON.decays) == (4, 2)


# -- least squares -------------------------------------------------------------


def test_reflections_solve_an_exactly_determined_system() -> None:
    rows = [[2.0, 1.0], [1.0, 3.0], [0.0, 0.0]]
    targets = [5.0, 10.0, 0.0]
    assert solve_least_squares(rows, targets) == pytest.approx((1.0, 3.0), rel=1e-14)


def test_reflections_agree_with_the_normal_equations_when_it_is_safe_to_ask() -> None:
    rows = [list(_zero_basis(t, (2.5,), Family.NELSON_SIEGEL)) for t in SAMPLE_TIMES]
    targets = [
        sum(b * g for b, g in zip(NS_LEVELS, row, strict=True)) for row in rows
    ]
    assert solve_least_squares(rows, targets) == pytest.approx(
        _normal_equations(rows, targets), rel=1e-12
    )


def test_the_normal_equations_cost_four_orders_on_a_collinear_design() -> None:
    """The reflections see the condition number; the normal equations see its
    square. With the two Svensson decays a thousandth apart that is the
    difference between fourteen digits and nine."""
    decays = (2.0, 2.002)
    rows = [list(_zero_basis(t, decays, Family.SVENSSON)) for t in SAMPLE_TIMES]
    targets = [
        sum(b * g for b, g in zip(SV_LEVELS, row, strict=True)) for row in rows
    ]
    condition = _condition(rows)
    assert 1e4 < condition < 1e5
    reflected = max(
        abs(a - b)
        for a, b in zip(solve_least_squares(rows, targets), SV_LEVELS, strict=True)
    )
    normal = max(
        abs(a - b)
        for a, b in zip(_normal_equations(rows, targets), SV_LEVELS, strict=True)
    )
    assert reflected < 1e-12
    assert normal > 1e-10
    assert normal / reflected > 100.0


def test_the_condition_number_is_worst_at_both_ends() -> None:
    """Non-monotone in the ratio of the decays, so a sweep that stopped at one
    end would conclude the form is fine. The two humps coincide as the ratio
    approaches one and the second hump leaves the quoted range as it grows."""
    conditions = {}
    for ratio in (1.01, 1.2, 3.0, 8.0, 20.0):
        decays = (2.0, 2.0 * ratio)
        rows = [list(_zero_basis(t, decays, Family.SVENSSON)) for t in SAMPLE_TIMES]
        conditions[ratio] = _condition(rows)
    assert conditions[1.01] > conditions[1.2] > conditions[3.0]
    assert conditions[20.0] > conditions[3.0]


@pytest.mark.parametrize(
    ("rows", "targets", "message"),
    [
        ([[1.0], [1.0]], [1.0], "rows against"),
        ([], [], "at least one row"),
        ([[1.0, 2.0], [1.0]], [1.0, 1.0], "not the same length"),
        ([[1.0, 2.0]], [1.0], "cannot determine"),
        ([[0.0, 0.0], [0.0, 1.0]], [1.0, 1.0], "is zero below the diagonal"),
    ],
)
def test_least_squares_refuses_what_it_cannot_answer(
    rows: list[list[float]], targets: list[float], message: str
) -> None:
    with pytest.raises(BadFit, match=message):
        solve_least_squares(rows, targets)


# -- recovery ------------------------------------------------------------------


def exact_zero_observations(curve: ParametricCurve) -> list[tuple[date, float]]:
    out = []
    for years in (0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30):
        day = REFERENCE + timedelta(days=round(365.25 * years))
        out.append((day, curve.zero(curve.time(day))))
    return out


def test_a_known_nelson_siegel_curve_is_recovered_from_its_own_zero_rates() -> None:
    truth = nelson_siegel()
    fit = fit_zeros(REFERENCE, exact_zero_observations(truth))
    assert fit.rms < 1e-8
    assert fit.curve.levels == pytest.approx(NS_LEVELS, abs=1e-6)
    assert fit.curve.decays[0] == pytest.approx(NS_DECAYS[0], rel=1e-4)
    assert fit.weighting is None
    assert fit.iterations == 1


def test_a_known_svensson_curve_is_recovered_from_its_own_zero_rates() -> None:
    """Exactly, both decays and all four levels, from a coarse grid — but only
    because the refinement ends in a simplex. See the test below."""
    truth = svensson()
    fit = fit_zeros(
        REFERENCE, exact_zero_observations(truth), family=Family.SVENSSON, points=24
    )
    assert fit.rms < 1e-12
    assert fit.curve.decays == pytest.approx(SV_DECAYS, rel=1e-6)
    assert fit.curve.levels == pytest.approx(SV_LEVELS, abs=1e-9)
    for t in (0.5, 2.0, 7.0, 25.0):
        assert fit.curve.zero(t) == pytest.approx(truth.zero(t), abs=1e-12)


def test_coordinate_sweeps_alone_stop_in_svenssons_diagonal_valley() -> None:
    """Lengthening one decay and shortening the other barely moves the curve, so
    a search that moves one coordinate at a time stops at the valley wall. On a
    Svensson curve fitted to its own exact zero rates the sweeps stop at a root
    mean square of 3.2e-05 with decays of 1.73 and 5.99 against a truth of 1.6
    and 9.0 — a plausible-looking answer that is simply the wrong one. Measured
    here so the simplex is a fix for something rather than an ornament."""
    truth = svensson()
    observations = exact_zero_observations(truth)
    times = [truth.time(day) for day, _ in observations]
    targets = [rate for _, rate in observations]

    def objective(decays: tuple[float, ...]) -> float:
        rows = [list(_zero_basis(t, decays, Family.SVENSSON)) for t in times]
        levels = solve_least_squares(rows, targets)
        residuals = [
            sum(b * g for b, g in zip(levels, row, strict=True)) - y
            for row, y in zip(rows, targets, strict=True)
        ]
        return math.sqrt(sum(e * e for e in residuals) / len(residuals))

    start = (1.9, 3.8)
    swept = list(start)
    for _ in range(4):
        for axis in range(2):

            def one(value: float, axis: int = axis) -> float:
                trial = list(swept)
                trial[axis] = value
                return objective(tuple(trial))

            swept[axis] = _golden(one, 0.1, 30.0)
    walked = _simplex(objective, start, 0.1, 30.0)
    assert objective(tuple(swept)) > 1e-6
    assert objective(walked) < 1e-14
    assert walked == pytest.approx(SV_DECAYS, rel=1e-6)


def exact_bond_quotes(curve: ParametricCurve) -> list[tuple[Bond, float]]:
    quotes = []
    for years, coupon in (
        (1, 0.025),
        (2, 0.030),
        (3, 0.035),
        (5, 0.040),
        (7, 0.042),
        (10, 0.045),
        (20, 0.047),
        (30, 0.048),
    ):
        bond = Bond(
            effective=date(2025, 7, 15),
            maturity=date(2026 + years, 7, 15),
            coupon=coupon,
            frequency=Frequency.SEMI_ANNUAL,
            basis=Basis.THIRTY_360_BOND,
        )
        clean = curve.price(bond, SETTLEMENT) - bond.accrued(SETTLEMENT)
        quotes.append((bond, clean))
    return quotes


def test_a_known_curve_is_recovered_from_its_own_bond_prices() -> None:
    truth = nelson_siegel()
    fit = fit_bonds(REFERENCE, exact_bond_quotes(truth), SETTLEMENT, points=24)
    assert fit.rms < 1e-7
    assert max(abs(r) for r in fit.residuals) < 1e-4
    assert fit.curve.levels == pytest.approx(NS_LEVELS, abs=1e-5)
    assert fit.curve.decays[0] == pytest.approx(NS_DECAYS[0], rel=1e-3)
    assert 1 <= fit.iterations <= 20


def test_the_jacobian_is_the_derivative_it_claims_to_be() -> None:
    """Gauss-Newton on a wrong Jacobian still converges, slowly, which is why
    this is checked against finite differences rather than inferred from the
    iteration count."""
    truth = nelson_siegel()
    quotes = exact_bond_quotes(truth)
    decays = (2.4,)
    levels = (0.05, -0.02, 0.03)
    step = 1e-7
    for index in range(3):
        up = list(levels)
        down = list(levels)
        up[index] += step
        down[index] -= step
        curve_up = ParametricCurve(REFERENCE, Family.NELSON_SIEGEL, tuple(up), decays)
        curve_down = ParametricCurve(
            REFERENCE, Family.NELSON_SIEGEL, tuple(down), decays
        )
        for bond, _ in quotes:
            numerical = (
                curve_up.price(bond, SETTLEMENT) - curve_down.price(bond, SETTLEMENT)
            ) / (2.0 * step)
            base = ParametricCurve(REFERENCE, Family.NELSON_SIEGEL, levels, decays)
            flows = bond.cashflows(SETTLEMENT)
            anchor = _integral_basis(
                base.time(SETTLEMENT), decays, Family.NELSON_SIEGEL
            )
            analytic = 0.0
            for flow in flows:
                terms = _integral_basis(
                    base.time(flow.day), decays, Family.NELSON_SIEGEL
                )
                discounted = flow.amount * math.exp(
                    -(base.integral(base.time(flow.day)) - base.integral(
                        base.time(SETTLEMENT)
                    ))
                )
                analytic -= discounted * (terms[index] - anchor[index])
            assert analytic == pytest.approx(numerical, rel=1e-6)


# -- what the fit does not determine -------------------------------------------


def noisy_bond_quotes(sigma: float = 0.10, seed: int = 7) -> list[tuple[Bond, float]]:
    """Clean prices from a known curve plus quote noise, because an exactly
    consistent set of quotes hides the whole subject of this phase."""
    truth = nelson_siegel()
    generator = random.Random(seed)
    quotes = []
    for years, coupon in (
        (1, 0.025),
        (2, 0.030),
        (3, 0.035),
        (4, 0.038),
        (5, 0.040),
        (7, 0.042),
        (10, 0.045),
        (15, 0.046),
        (20, 0.047),
        (30, 0.048),
    ):
        bond = Bond(
            effective=date(2025, 7, 15),
            maturity=date(2026 + years, 7, 15),
            coupon=coupon,
            frequency=Frequency.SEMI_ANNUAL,
            basis=Basis.THIRTY_360_BOND,
        )
        clean = truth.price(bond, SETTLEMENT) - bond.accrued(SETTLEMENT)
        quotes.append((bond, clean + generator.gauss(0.0, sigma)))
    return quotes


def test_the_decay_band_contains_the_truth_and_widens_with_the_tolerance() -> None:
    fits = {
        tolerance: fit_zeros(
            REFERENCE,
            exact_zero_observations(nelson_siegel()),
            points=24,
            tolerance=tolerance,
        )
        for tolerance in (1e-7, 1e-5, 1e-4)
    }
    previous: DecaySet | None = None
    for tolerance in (1e-7, 1e-5, 1e-4):
        band = fits[tolerance].identified
        assert band.lowest[0] <= NS_DECAYS[0] <= band.highest[0]
        if previous is not None:
            assert band.lowest[0] <= previous.lowest[0]
            assert band.highest[0] >= previous.highest[0]
        previous = band
    assert fits[1e-4].identified.widest > fits[1e-7].identified.widest


def test_a_basis_point_of_tolerance_leaves_the_hump_position_loose() -> None:
    """The arg-min of a flat function is not a measurement. At a basis point the
    decay runs from 2.05 to 3.00 on these quotes — the fitted 2.50 is the middle
    of a band half as wide as itself, not a hump position."""
    fit = fit_bonds(REFERENCE, noisy_bond_quotes(), SETTLEMENT, points=24, tolerance=1e-4)
    band = fit.identified
    assert band.bounded == (True,)
    assert 1.4 < band.widest < 1.8
    assert band.lowest[0] < fit.curve.decays[0] < band.highest[0]
    assert 1.8 < band.lowest[0] < 2.1
    assert 2.9 < band.highest[0] < 3.3


def test_at_ten_basis_points_the_curve_is_determined_and_its_parameters_are_not() -> None:
    """The decay runs from 0.10 to 10.77 — a factor of 104 — and across that band
    the slope and hump coefficients move by about 4,700 while every point of it
    reprices the same quotes within ten basis points. The *level* parameter moves
    by 0.028, because the long end is the one thing these quotes pin. Reading
    ``b1`` off a Nelson-Siegel fit as "the slope of the curve" is reading a
    number the quotes did not determine."""
    fit = fit_bonds(REFERENCE, noisy_bond_quotes(), SETTLEMENT, points=24, tolerance=1e-3)
    band = fit.identified
    assert band.widest > 50.0
    assert band.lowest[0] < 0.2
    assert band.highest[0] > 8.0
    assert max(band.level_spread[1:]) > 1e3
    assert band.level_spread[0] < 0.05


def test_svensson_trades_its_two_humps_against_each_other() -> None:
    """Both level parameters swing by more than two across the band while the
    first decay is unidentified, which is the collinearity the condition number
    sees from the other side."""
    fit = fit_zeros(
        REFERENCE,
        exact_zero_observations(svensson()),
        family=Family.SVENSSON,
        points=24,
        tolerance=1e-4,
    )
    band = fit.identified
    assert band.lowest[0] < band.highest[0]
    assert band.lowest[1] < band.highest[1]
    assert max(band.level_spread) > min(band.level_spread)


# -- weighting -----------------------------------------------------------------


def yield_errors(fit: CurveFit, quotes: list[tuple[Bond, float]]) -> list[float]:
    out = []
    for (bond, clean), residual in zip(quotes, fit.residuals, strict=True):
        dirty = clean + bond.accrued(SETTLEMENT)
        own = bond.yield_from_dirty(dirty, SETTLEMENT).value
        out.append(residual / (bond.modified_duration(own, SETTLEMENT) * dirty))
    return out


def test_price_weighting_spends_its_accuracy_at_the_long_end() -> None:
    """A one-year bond's yield moves its price a fortieth as much as a
    thirty-year bond's does, so weighting price errors equally is weighting the
    short end out of the fit. Measured: the worst yield error falls from 10.5
    basis points to 4.7 when the residuals are put in yield units first."""
    quotes = noisy_bond_quotes()
    by_price = fit_bonds(
        REFERENCE,
        quotes,
        SETTLEMENT,
        points=24,
        weighting=Weighting.PRICE,
        tolerance=1e-2,
    )
    by_yield = fit_bonds(
        REFERENCE, quotes, SETTLEMENT, points=24, weighting=Weighting.YIELD
    )
    price_errors = yield_errors(by_price, quotes)
    yield_fit_errors = yield_errors(by_yield, quotes)
    assert max(abs(e) for e in price_errors) > 3.5e-4
    assert max(abs(e) for e in yield_fit_errors) < 3.0e-4
    assert max(abs(e) for e in price_errors) > 1.4 * max(
        abs(e) for e in yield_fit_errors
    )
    # The short end is where the whole difference lands.
    assert abs(price_errors[0]) > 3.0 * abs(yield_fit_errors[0])
    # And the long end, which price weighting was protecting, barely moves.
    assert abs(price_errors[-1]) < 2e-5


def test_yield_weighting_is_not_free_and_the_price_errors_say_so() -> None:
    quotes = noisy_bond_quotes()
    by_price = fit_bonds(
        REFERENCE,
        quotes,
        SETTLEMENT,
        points=24,
        weighting=Weighting.PRICE,
        tolerance=1e-2,
    )
    by_yield = fit_bonds(
        REFERENCE, quotes, SETTLEMENT, points=24, weighting=Weighting.YIELD
    )
    worst_price_under_price = max(abs(r) for r in by_price.residuals)
    worst_price_under_yield = max(abs(r) for r in by_yield.residuals)
    assert worst_price_under_yield > worst_price_under_price
    assert 1.5 < worst_price_under_yield / worst_price_under_price < 2.5
    assert by_price.weighting is Weighting.PRICE
    assert by_yield.weighting is Weighting.YIELD


def test_the_fit_names_the_quote_it_missed_by_most() -> None:
    quotes = noisy_bond_quotes()
    fit = fit_bonds(REFERENCE, quotes, SETTLEMENT, points=24)
    index = max(range(len(fit.weighted)), key=lambda i: abs(fit.weighted[i]))
    assert fit.worst_quote == quotes[index][0].name
    assert fit.worst == pytest.approx(abs(fit.weighted[index]), rel=1e-15)


def test_a_fit_with_no_labels_still_names_its_worst_quote() -> None:
    fit = fit_zeros(REFERENCE, exact_zero_observations(nelson_siegel()), points=12)
    bare = CurveFit(
        curve=fit.curve,
        residuals=fit.residuals,
        weighted=fit.weighted,
        weighting=None,
        identified=fit.identified,
        condition=fit.condition,
        iterations=fit.iterations,
    )
    assert bare.worst_quote.startswith("quote ")


# -- refusals ------------------------------------------------------------------


def test_too_few_quotes_for_the_form_is_refused() -> None:
    observations = exact_zero_observations(nelson_siegel())[:2]
    with pytest.raises(BadFit, match="cannot be fitted to 2 observations"):
        fit_zeros(REFERENCE, observations)
    with pytest.raises(BadFit, match="cannot be fitted to 3 quotes"):
        fit_bonds(
            REFERENCE,
            exact_bond_quotes(nelson_siegel())[:3],
            SETTLEMENT,
            family=Family.SVENSSON,
        )


def test_a_mismatched_weight_vector_is_refused() -> None:
    observations = exact_zero_observations(nelson_siegel())
    with pytest.raises(BadFit, match="weights against"):
        fit_zeros(REFERENCE, observations, weights=[1.0, 1.0])


@pytest.mark.parametrize(("low", "high"), [(0.0, 10.0), (5.0, 1.0), (-1.0, 2.0)])
def test_a_degenerate_search_range_is_refused(low: float, high: float) -> None:
    with pytest.raises(BadFit, match="0 < low < high"):
        fit_zeros(
            REFERENCE,
            exact_zero_observations(nelson_siegel()),
            low=low,
            high=high,
        )


def test_a_single_grid_point_is_refused() -> None:
    with pytest.raises(BadFit, match="at least two grid points"):
        fit_zeros(REFERENCE, exact_zero_observations(nelson_siegel()), points=1)


def test_weights_can_move_the_fit() -> None:
    """A weight of nearly nothing on the thirty-year point is a different fit,
    which is the check that the weights reach the arithmetic at all."""
    observations = exact_zero_observations(nelson_siegel())
    observations[-1] = (observations[-1][0], observations[-1][1] + 0.01)
    plain = fit_zeros(REFERENCE, observations, points=24)
    discounted = fit_zeros(
        REFERENCE,
        observations,
        points=24,
        weights=[1.0] * (len(observations) - 1) + [1e-6],
    )
    assert abs(plain.residuals[-1]) < abs(discounted.residuals[-1])
    assert abs(discounted.curve.zero(30.0) - nelson_siegel().zero(30.0)) < abs(
        plain.curve.zero(30.0) - nelson_siegel().zero(30.0)
    )


def test_the_sampled_curve_prices_the_quotes_it_was_fitted_to() -> None:
    """End to end: fit bonds, convert to a pillar curve, and price the same
    bonds through the ordinary bond route. Nothing downstream knows the curve
    came from a formula."""
    quotes = noisy_bond_quotes()
    fit = fit_bonds(REFERENCE, quotes, SETTLEMENT, points=24)
    days = [REFERENCE + timedelta(days=step) for step in range(0, 11200, 7)]
    sampled = fit.curve.to_discount_curve(days)
    for bond, clean in quotes:
        through_curve = bond.price_from_curve(sampled, SETTLEMENT)
        direct = fit.curve.price(bond, SETTLEMENT)
        assert through_curve == pytest.approx(direct, abs=1e-4)
        assert through_curve == pytest.approx(
            clean + bond.accrued(SETTLEMENT), abs=0.2
        )


def test_a_pillar_curve_built_from_a_fit_reports_the_same_zero_rates() -> None:
    curve = nelson_siegel()
    days = [REFERENCE + timedelta(days=step) for step in range(30, 11000, 30)]
    sampled = curve.to_discount_curve(days, interpolation=Interpolation.LINEAR_ZERO)
    for day in days[::40]:
        implied = -math.log(sampled.discount(day)) / curve.time(day)
        assert implied == pytest.approx(curve.zero(curve.time(day)), rel=1e-14)
