"""The two-factor Gaussian model: the reductions, the identities, the measurements.

Three groups carry this suite.

:class:`TestCollapsesToHullWhite` is the strongest check available and the
reason ``a == b`` is a permitted setting rather than a refused one. With equal
mean reversions the model *is* the one-factor model, with an effective
volatility of ``sqrt(sigma**2 + eta**2 + 2 rho sigma eta)``, and every formula
here has to reduce to the corresponding one in :mod:`tenor.hullwhite`. That
test crosses modules rather than checking this module's algebra against itself,
and it is what caught the call/put convention going the wrong way.

:class:`TestAgainstSimulation` checks the correlation closed form against an
exact draw from the terminal law. The draw shares no algebra with the formula —
one is a Cholesky factor applied to normals, the other a ratio of quadratic
forms — so agreement inside a standard error is evidence about the derivation.

:class:`TestWhatTheSecondFactorBuys` holds the measurements that justify the
module, including the one that says the instruments in it cannot identify its
most interesting parameter.
"""

from __future__ import annotations

import datetime
import math
import statistics
from datetime import date

import pytest

from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis
from tenor.g2 import (
    G2,
    Correlation,
    bond_option,
    cap_price,
    caplet,
    factor_moments,
    fit_fast_volatility,
    simulate_factors,
    zero_rate_correlation,
)
from tenor.hullwhite import BadModel, HullWhite, zero_bond_option
from tenor.options import Payoff
from tenor.solve import NoRoot

REFERENCE = date(2026, 1, 2)
BASIS = Basis.ACT_365F

# The same seventeen-pillar humped curve `test_hullwhite` uses, so that a
# disagreement between the two models is about the models.
ZEROS = [
    (0.08, 0.0310),
    (0.25, 0.0318),
    (0.5, 0.0329),
    (0.75, 0.0338),
    (1.0, 0.0345),
    (1.5, 0.0356),
    (2.0, 0.0364),
    (3.0, 0.0374),
    (4.0, 0.0379),
    (5.0, 0.0381),
    (7.0, 0.0380),
    (10.0, 0.0374),
    (12.0, 0.0369),
    (15.0, 0.0362),
    (20.0, 0.0352),
    (25.0, 0.0345),
    (30.0, 0.0340),
]

# A fast factor and a slow one with a strongly negative correlation, which is
# the ordinary shape of a fitted set.
MODEL = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=-0.90)

# Equal speeds, which is the one-factor model in disguise.
TWINS = [
    G2(a=0.08, sigma=0.009, b=0.08, eta=0.0, rho=0.0),
    G2(a=0.08, sigma=0.006, b=0.08, eta=0.005, rho=-0.3),
    G2(a=0.30, sigma=0.010, b=0.30, eta=0.012, rho=0.5),
    G2(a=0.12, sigma=0.004, b=0.12, eta=0.009, rho=0.7),
]


def at(years: float) -> date:
    return REFERENCE + datetime.timedelta(days=round(years * 365.0))


@pytest.fixture
def curve() -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE,
        [(at(years), rate) for years, rate in ZEROS],
        basis=BASIS,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )


class TestTheExactFit:
    """Nothing is fitted, so the error is not small, it is zero."""

    def test_the_curve_reprices_with_no_error_at_all(self, curve: DiscountCurve) -> None:
        for model in [MODEL, *TWINS, G2(a=1.5, sigma=0.03, b=0.01, eta=0.002, rho=0.95)]:
            assert model.curve_error(curve) == 0.0

    def test_a_zero_volatility_model_is_the_curve(self, curve: DiscountCurve) -> None:
        flat = G2(a=0.3, sigma=0.0, b=0.05, eta=0.0, rho=0.5)
        assert flat.curve_error(curve) == 0.0
        for time in (1.0, 5.0, 20.0):
            assert flat.bond_price(curve, 0.0, time, 0.0, 0.0) == curve.discount_at(time)

    def test_the_shift_starts_at_the_curve_s_own_forward(self, curve: DiscountCurve) -> None:
        """``phi(0) = f(0,0)`` exactly: the convexity term vanishes at zero."""
        for model in [MODEL, *TWINS]:
            assert model.short_rate_mean(curve, 0.0) == pytest.approx(
                curve.instantaneous_forward(0.0), abs=1e-15
            )

    def test_the_shift_rises_with_the_driver_correlation(self, curve: DiscountCurve) -> None:
        """Which is where ``rho`` shows up in a quantity with no option in it."""
        shifts = [
            G2(a=0.5, sigma=0.011, b=0.05, eta=0.007, rho=rho).short_rate_mean(curve, 5.0)
            for rho in (-0.9, -0.5, 0.0, 0.5, 0.9)
        ]
        assert shifts == sorted(shifts)
        forward = curve.instantaneous_forward(5.0)
        assert all(shift > forward for shift in shifts)

    def test_the_variance_term_vanishes_at_zero_tenor(self) -> None:
        for model in [MODEL, *TWINS]:
            assert model.variance_term(0.0) == 0.0

    def test_the_variance_term_grows_with_the_tenor(self) -> None:
        for model in [MODEL, *TWINS]:
            values = [model.variance_term(tenor) for tenor in (0.5, 1.0, 5.0, 10.0, 30.0)]
            assert values == sorted(values)


class TestCollapsesToHullWhite:
    """Equal mean reversions make this the one-factor model. Exactly."""

    @pytest.mark.parametrize("model", TWINS, ids=lambda m: f"a={m.a}_rho={m.rho}")
    def test_the_bond_option_volatility_reduces(self, model: G2) -> None:
        one_factor = HullWhite(a=model.a, sigma=model.effective_volatility)
        for expiry, maturity in ((1.0, 5.0), (3.0, 10.0), (5.0, 20.0), (0.5, 1.0), (0.0, 7.0)):
            assert model.bond_option_volatility(expiry, maturity) == pytest.approx(
                one_factor.bond_volatility(expiry, maturity), abs=1e-16
            )

    @pytest.mark.parametrize("model", TWINS, ids=lambda m: f"a={m.a}_rho={m.rho}")
    def test_the_bond_option_price_reduces(self, model: G2, curve: DiscountCurve) -> None:
        one_factor = HullWhite(a=model.a, sigma=model.effective_volatility)
        for expiry, maturity, strike in (
            (1.0, 5.0, 0.85),
            (3.0, 10.0, 0.70),
            (5.0, 20.0, 0.50),
            (0.5, 1.0, 0.98),
        ):
            for payoff in (Payoff.PAYER, Payoff.RECEIVER):
                assert bond_option(
                    curve,
                    model,
                    expiry=expiry,
                    maturity=maturity,
                    strike=strike,
                    payoff=payoff,
                ) == pytest.approx(
                    zero_bond_option(curve, one_factor, expiry, maturity, strike, payoff),
                    abs=1e-15,
                )

    @pytest.mark.parametrize("model", TWINS, ids=lambda m: f"a={m.a}_rho={m.rho}")
    def test_the_bond_price_reduces_under_the_state_map(
        self, model: G2, curve: DiscountCurve
    ) -> None:
        """``r(t) = phi(t) + x(t) + y(t)`` is the mapping, and it has to be used.

        Comparing the two models at the same *numerical* state is meaningless:
        the one-factor module's argument is the short rate and this module's
        are the two factors, which sum to the short rate less ``phi``. The
        first version of this test passed the curve's instantaneous forward
        plus one factor and found a 1.5e-03 relative gap, which is the size of
        the convexity term in ``phi`` rather than a defect.
        """
        one_factor = HullWhite(a=model.a, sigma=model.effective_volatility)
        for valuation, maturity in ((1.0, 5.0), (2.0, 30.0), (0.25, 0.5), (7.5, 12.0)):
            shift = model.short_rate_mean(curve, valuation)
            for x, y in ((-0.010, 0.004), (0.0, 0.0), (0.013, -0.006)):
                assert model.bond_price(curve, valuation, maturity, x, y) == pytest.approx(
                    one_factor.bond_price(curve, valuation, maturity, shift + x + y),
                    rel=1e-14,
                )

    @pytest.mark.parametrize("model", TWINS, ids=lambda m: f"a={m.a}_rho={m.rho}")
    def test_the_correlation_reduces_to_one(self, model: G2) -> None:
        """Which is the limitation the second factor exists to remove."""
        for horizon, first, second in ((1.0, 2.0, 11.0), (0.5, 1.0, 30.0), (5.0, 6.0, 25.0)):
            assert zero_rate_correlation(model, horizon, first, second).value == pytest.approx(
                1.0, abs=1e-12
            )

    def test_the_effective_volatility_is_the_sum_of_the_drivers(self) -> None:
        model = G2(a=0.1, sigma=0.006, b=0.1, eta=0.008, rho=-0.5)
        expected = math.sqrt(0.006**2 + 0.008**2 + 2.0 * -0.5 * 0.006 * 0.008)
        assert model.effective_volatility == pytest.approx(expected)
        # And a perfectly anticorrelated pair of equal size would cancel, which
        # is why rho is open at the endpoints rather than closed.
        near = G2(a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=-0.999999)
        assert near.effective_volatility < 1e-4


class TestBondOption:
    """Parity, monotonicity, and the degenerate cases."""

    def test_put_call_parity_is_an_identity(self, curve: DiscountCurve) -> None:
        for expiry, maturity, strike in (
            (1.0, 5.0, 0.85),
            (2.0, 10.0, 0.75),
            (5.0, 7.0, 0.95),
            (0.25, 30.0, 0.40),
        ):
            call = bond_option(
                curve,
                MODEL,
                expiry=expiry,
                maturity=maturity,
                strike=strike,
                payoff=Payoff.RECEIVER,
            )
            put = bond_option(
                curve, MODEL, expiry=expiry, maturity=maturity, strike=strike
            )
            expected = curve.discount_at(maturity) - strike * curve.discount_at(expiry)
            assert call - put == pytest.approx(expected, abs=1e-15)

    def test_both_sides_are_non_negative_and_above_intrinsic(
        self, curve: DiscountCurve
    ) -> None:
        for strike in (0.4, 0.6, 0.8, 0.95, 1.0):
            call = bond_option(
                curve, MODEL, expiry=2.0, maturity=10.0, strike=strike, payoff=Payoff.RECEIVER
            )
            put = bond_option(curve, MODEL, expiry=2.0, maturity=10.0, strike=strike)
            intrinsic = curve.discount_at(10.0) - strike * curve.discount_at(2.0)
            assert call >= max(intrinsic, 0.0) - 1e-15
            assert put >= max(-intrinsic, 0.0) - 1e-15

    def test_a_call_falls_and_a_put_rises_with_the_strike(self, curve: DiscountCurve) -> None:
        strikes = [0.4, 0.55, 0.7, 0.85, 1.0]
        calls = [
            bond_option(
                curve, MODEL, expiry=2.0, maturity=10.0, strike=k, payoff=Payoff.RECEIVER
            )
            for k in strikes
        ]
        puts = [
            bond_option(curve, MODEL, expiry=2.0, maturity=10.0, strike=k) for k in strikes
        ]
        assert calls == sorted(calls, reverse=True)
        assert puts == sorted(puts)

    def test_zero_volatility_is_the_discounted_intrinsic(self, curve: DiscountCurve) -> None:
        flat = G2(a=0.3, sigma=0.0, b=0.05, eta=0.0, rho=0.5)
        assert flat.bond_option_volatility(2.0, 10.0) == 0.0
        forward = curve.discount_at(10.0) / curve.discount_at(2.0)
        for strike in (forward * 0.8, forward * 1.2):
            call = bond_option(
                curve, flat, expiry=2.0, maturity=10.0, strike=strike, payoff=Payoff.RECEIVER
            )
            put = bond_option(curve, flat, expiry=2.0, maturity=10.0, strike=strike)
            intrinsic = curve.discount_at(10.0) - strike * curve.discount_at(2.0)
            assert call == pytest.approx(max(intrinsic, 0.0))
            assert put == pytest.approx(max(-intrinsic, 0.0))

    def test_a_zero_expiry_has_no_randomness_either(self, curve: DiscountCurve) -> None:
        assert MODEL.bond_option_volatility(0.0, 10.0) == 0.0
        call = bond_option(
            curve, MODEL, expiry=0.0, maturity=10.0, strike=0.5, payoff=Payoff.RECEIVER
        )
        assert call == pytest.approx(curve.discount_at(10.0) - 0.5)

    def test_an_option_on_a_bond_maturing_at_expiry_is_worthless_or_certain(
        self, curve: DiscountCurve
    ) -> None:
        """The bond is worth one at its own maturity, so there is nothing random."""
        assert MODEL.bond_option_volatility(3.0, 3.0) == 0.0
        call = bond_option(
            curve, MODEL, expiry=3.0, maturity=3.0, strike=0.9, payoff=Payoff.RECEIVER
        )
        assert call == pytest.approx(0.1 * curve.discount_at(3.0))

    def test_volatility_grows_with_expiry_and_with_the_bond_s_tenor(self) -> None:
        by_expiry = [MODEL.bond_option_volatility(t, 30.0) for t in (0.25, 1.0, 5.0, 10.0)]
        assert by_expiry == sorted(by_expiry)
        by_tenor = [MODEL.bond_option_volatility(2.0, s) for s in (3.0, 5.0, 10.0, 30.0)]
        assert by_tenor == sorted(by_tenor)


class TestCapsAndFloors:
    """A caplet is a put on a bond, which is why this model prices caps exactly."""

    def test_a_caplet_is_the_bond_put_it_claims_to_be(self, curve: DiscountCurve) -> None:
        start, end, strike = 2.0, 2.5, 0.036
        span = end - start
        quantity = 1.0 + span * strike
        direct = quantity * bond_option(
            curve, MODEL, expiry=start, maturity=end, strike=1.0 / quantity
        )
        assert caplet(
            curve, MODEL, start=start, end=end, strike=strike
        ) == pytest.approx(direct)

    def test_cap_floor_parity_is_the_forward_rate_agreement(
        self, curve: DiscountCurve
    ) -> None:
        """A cap less a floor at the same strike is the swap, which is model-free.

        Each period's caplet less floorlet is worth
        ``accrual * (forward - strike)`` discounted, and summing gives the
        value of paying fixed. Nothing in it depends on the two factors, so a
        failure means the quantity or the bond strike in :func:`caplet` is
        wrong.
        """
        dates = [1.0 + 0.5 * index for index in range(11)]
        strike = 0.037
        cap = cap_price(curve, MODEL, dates=dates, strike=strike)
        floor = cap_price(curve, MODEL, dates=dates, strike=strike, floor=True)
        swap = 0.0
        for index in range(len(dates) - 1):
            near = curve.discount_at(dates[index])
            far = curve.discount_at(dates[index + 1])
            span = dates[index + 1] - dates[index]
            swap += near - far - span * strike * far
        assert cap - floor == pytest.approx(swap, abs=1e-14)

    def test_a_cap_is_the_sum_of_its_caplets(self, curve: DiscountCurve) -> None:
        dates = [1.0, 1.5, 2.0, 2.5, 3.0]
        total = math.fsum(
            caplet(curve, MODEL, start=dates[i], end=dates[i + 1], strike=0.038)
            for i in range(len(dates) - 1)
        )
        assert cap_price(curve, MODEL, dates=dates, strike=0.038) == pytest.approx(total)

    def test_a_cap_falls_with_its_strike_and_a_floor_rises(
        self, curve: DiscountCurve
    ) -> None:
        dates = [1.0 + 0.5 * index for index in range(11)]
        strikes = [0.02, 0.03, 0.04, 0.05, 0.06]
        caps = [cap_price(curve, MODEL, dates=dates, strike=k) for k in strikes]
        floors = [cap_price(curve, MODEL, dates=dates, strike=k, floor=True) for k in strikes]
        assert caps == sorted(caps, reverse=True)
        assert floors == sorted(floors)

    def test_an_explicit_accrual_matters_more_than_it_looks(
        self, curve: DiscountCurve
    ) -> None:
        """A day count's year fraction is not the gap between the dates, and the
        difference is not second order.

        Moving the accrual from 0.5 to 0.5055 — the 30/360 against the
        actual/365 figure for the same half-year, a difference of 1.1% —
        moves a two-year caplet struck at 3.6% by **7.0%**, from 0.00198766
        to 0.00184770. The accrual enters through ``1 + accrual * strike``,
        which is the quantity *and* the reciprocal of the bond strike, so it
        moves a near-the-money option's moneyness rather than scaling it.
        Passing the wrong one is not a rounding error.
        """
        gap = caplet(curve, MODEL, start=2.0, end=2.5, strike=0.036)
        other = caplet(curve, MODEL, start=2.0, end=2.5, strike=0.036, accrual=0.5055)
        assert gap == pytest.approx(0.00198766, rel=1e-4)
        assert other == pytest.approx(0.00184770, rel=1e-4)
        # A longer accrual raises the quantity, lowers the equivalent bond
        # strike, and so lowers the put the caplet is.
        assert other < gap
        assert other / gap - 1.0 == pytest.approx(-0.070, abs=0.002)


class TestAgainstSimulation:
    """An exact draw from the terminal law, sharing no algebra with the formulas."""

    @pytest.mark.parametrize("horizon", [0.5, 2.0, 10.0])
    def test_the_factor_moments_are_the_drawn_ones(self, horizon: float) -> None:
        """Each row draws its own seed.

        One seed shared across the rows makes that draw's sampling error look
        like a systematic bias: at 400,000 paths the three moments all came out
        0.1% to 0.5% high in every row, identically, which reads as a defect
        and is one draw.
        """
        paths = 200_000
        moments = factor_moments(MODEL, horizon)
        draws = simulate_factors(MODEL, horizon, paths=paths, seed=int(horizon * 1000) + 7)
        xs = [x for x, _ in draws]
        ys = [y for _, y in draws]
        # Relative standard error of a variance is sqrt(2/n), about 0.32% here.
        tolerance = 4.0 * math.sqrt(2.0 / paths)
        assert statistics.variance(xs) == pytest.approx(moments.variance_x, rel=tolerance)
        assert statistics.variance(ys) == pytest.approx(moments.variance_y, rel=tolerance)
        assert statistics.covariance(xs, ys) == pytest.approx(
            moments.covariance, rel=4.0 * tolerance
        )
        assert statistics.fmean(xs) == pytest.approx(0.0, abs=5.0 * math.sqrt(
            moments.variance_x / paths
        ))

    @pytest.mark.parametrize(
        ("horizon", "first", "second"),
        [(1.0, 2.0, 11.0), (1.0, 3.0, 6.0), (2.0, 2.5, 30.0), (5.0, 6.0, 15.0)],
    )
    def test_the_correlation_closed_form_is_the_drawn_one(
        self, curve: DiscountCurve, horizon: float, first: float, second: float
    ) -> None:
        paths = 200_000
        closed = zero_rate_correlation(MODEL, horizon, first, second).value
        draws = simulate_factors(
            MODEL, horizon, paths=paths, seed=int(horizon * 100 + first * 10 + second)
        )
        logs_first = []
        logs_second = []
        for x, y in draws:
            logs_first.append(math.log(MODEL.bond_price(curve, horizon, first, x, y)))
            logs_second.append(math.log(MODEL.bond_price(curve, horizon, second, x, y)))
        drawn = statistics.correlation(logs_first, logs_second)
        error = (1.0 - closed * closed) / math.sqrt(paths)
        assert abs(drawn - closed) < 4.0 * error

    def test_the_bond_price_is_lognormal_with_the_option_s_own_volatility(
        self, curve: DiscountCurve
    ) -> None:
        """Ties the two closed forms together through the simulation.

        The option formula's ``Sigma(T, S)`` is the standard deviation of
        ``log`` of the forward bond price under the ``T``-forward measure. Under
        the risk-neutral measure the standard deviation of ``log P(T, S)`` is
        the same number — the measure change shifts the mean, not the variance —
        so the drawn one has to match it.
        """
        paths = 200_000
        expiry, maturity = 2.0, 10.0
        draws = simulate_factors(MODEL, expiry, paths=paths, seed=4242)
        logs = [
            math.log(MODEL.bond_price(curve, expiry, maturity, x, y)) for x, y in draws
        ]
        assert statistics.stdev(logs) == pytest.approx(
            MODEL.bond_option_volatility(expiry, maturity), rel=4.0 * math.sqrt(0.5 / paths)
        )

    def test_the_draw_is_exact_rather_than_stepped(self) -> None:
        """No step count to choose, so doubling the paths is the only refinement."""
        coarse = simulate_factors(MODEL, 3.0, paths=50_000, seed=1)
        fine = simulate_factors(MODEL, 3.0, paths=200_000, seed=2)
        truth = factor_moments(MODEL, 3.0)
        coarse_error = abs(
            statistics.variance([x for x, _ in coarse]) / truth.variance_x - 1.0
        )
        fine_error = abs(statistics.variance([x for x, _ in fine]) / truth.variance_x - 1.0)
        assert fine_error < coarse_error

    def test_it_is_reproducible(self) -> None:
        assert simulate_factors(MODEL, 1.0, paths=100, seed=5) == simulate_factors(
            MODEL, 1.0, paths=100, seed=5
        )
        assert simulate_factors(MODEL, 1.0, paths=100, seed=5) != simulate_factors(
            MODEL, 1.0, paths=100, seed=6
        )


class TestWhatTheSecondFactorBuys:
    """The measurements, including the one that says a cap cannot see them."""

    def test_the_decorrelation_against_the_one_factor_s_forced_one(self) -> None:
        """The figures in the module docstring, asserted so they cannot drift."""
        assert zero_rate_correlation(MODEL, 1.0, 2.0, 11.0).value == pytest.approx(
            0.3592, abs=5e-4
        )
        assert zero_rate_correlation(MODEL, 1.0, 1.5, 11.0).value == pytest.approx(
            0.1549, abs=5e-4
        )
        assert zero_rate_correlation(MODEL, 1.0, 2.0, 6.0).value == pytest.approx(
            0.5589, abs=5e-4
        )

    def test_the_decorrelation_closes_as_the_speeds_meet(self) -> None:
        """And is not monotone on the way, which is worth knowing before fitting."""
        values = [
            zero_rate_correlation(
                G2(a=0.50, sigma=0.011, b=slow, eta=0.007, rho=-0.90), 1.0, 2.0, 11.0
            ).value
            for slow in (0.05, 0.1, 0.2, 0.35, 0.49, 0.50)
        ]
        assert values[-1] == pytest.approx(1.0, abs=1e-12)
        assert values[-2] > 0.999
        assert values != sorted(values)
        assert min(values) == pytest.approx(0.3387, abs=5e-4)

    def test_the_correlation_falls_as_the_maturities_separate(self) -> None:
        values = [
            zero_rate_correlation(MODEL, 1.0, 2.0, second).value
            for second in (2.5, 3.0, 5.0, 11.0, 30.0)
        ]
        assert values == sorted(values, reverse=True)
        assert values[0] < 1.0

    def test_identical_maturities_are_perfectly_correlated(self) -> None:
        assert zero_rate_correlation(MODEL, 1.0, 5.0, 5.0).value == pytest.approx(1.0)

    def test_a_cap_cannot_identify_the_correlation(self, curve: DiscountCurve) -> None:
        """Seven calibrations a cap market cannot tell apart, bending differently.

        The structural reason is that a cap is a strip of options on *single*
        bonds, so its price depends only on each bond's own volatility and
        never on the joint law of two maturities. The measurement is what makes
        that concrete: refitting ``sigma`` to hold a one-year to ten-year
        semi-annual cap at 3.8% fixed, seven values of ``rho`` give the same
        cap price to rounding and correlations spanning almost the whole
        interval.
        """
        dates = [1.0 + 0.5 * index for index in range(19)]
        strike = 0.038
        base = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=0.0)
        target = cap_price(curve, base, dates=dates, strike=strike)

        def refit(rho: float) -> G2:
            template = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=rho)
            root = fit_fast_volatility(
                curve, template, dates=dates, strike=strike, target=target
            )
            return G2(a=0.50, sigma=root.value, b=0.05, eta=0.007, rho=rho)

        correlations = []
        for rho in (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
            fitted = refit(rho)
            # The cap is matched to rounding, not to a tolerance worth stating.
            assert cap_price(curve, fitted, dates=dates, strike=strike) == pytest.approx(
                target, rel=1e-12
            )
            correlations.append(zero_rate_correlation(fitted, 1.0, 2.0, 11.0).value)
        assert correlations == sorted(correlations)
        assert correlations[0] == pytest.approx(0.0674, abs=2e-3)
        assert correlations[-1] == pytest.approx(0.9965, abs=2e-3)
        assert correlations[-1] - correlations[0] > 0.9

    def test_the_factor_correlation_is_not_the_driver_correlation(self) -> None:
        """The levels are less correlated than the drivers, at every horizon.

        Two mean reversions weight the same history differently, so the
        integrated factors cannot inherit ``rho``. Worth asserting because the
        two numbers are easy to confuse and the result object carries both.
        """
        for horizon in (0.25, 1.0, 5.0, 30.0):
            moments = factor_moments(MODEL, horizon)
            assert abs(moments.correlation) < abs(MODEL.rho)
            assert moments.correlation < 0.0

    def test_the_result_carries_what_it_was_computed_from(self) -> None:
        result = zero_rate_correlation(MODEL, 1.0, 2.0, 11.0)
        assert isinstance(result, Correlation)
        assert result.horizon == 1.0
        assert result.first == 2.0
        assert result.second == 11.0
        assert result.moments == factor_moments(MODEL, 1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.value = 0.0  # type: ignore[misc]


class TestValidation:
    """What gets refused."""

    @pytest.mark.parametrize("speed", [0.0, -0.1, math.inf, math.nan])
    def test_a_non_positive_speed(self, speed: float) -> None:
        with pytest.raises(BadModel, match="mean reversion a"):
            G2(a=speed, sigma=0.01, b=0.05, eta=0.007, rho=0.0)
        with pytest.raises(BadModel, match="mean reversion b"):
            G2(a=0.5, sigma=0.01, b=speed, eta=0.007, rho=0.0)

    @pytest.mark.parametrize("volatility", [-0.001, math.inf, math.nan])
    def test_a_negative_volatility(self, volatility: float) -> None:
        with pytest.raises(BadModel, match="volatility sigma"):
            G2(a=0.5, sigma=volatility, b=0.05, eta=0.007, rho=0.0)
        with pytest.raises(BadModel, match="volatility eta"):
            G2(a=0.5, sigma=0.01, b=0.05, eta=volatility, rho=0.0)

    @pytest.mark.parametrize("rho", [-1.0, 1.0, -1.5, 2.0, math.nan])
    def test_a_correlation_outside_the_open_interval(self, rho: float) -> None:
        with pytest.raises(BadModel, match="strictly inside"):
            G2(a=0.5, sigma=0.01, b=0.05, eta=0.007, rho=rho)

    def test_equal_speeds_are_allowed(self) -> None:
        """Deliberately: it is the one-factor model and it is the cross-check."""
        model = G2(a=0.1, sigma=0.01, b=0.1, eta=0.005, rho=0.0)
        assert model.a == model.b

    def test_dates_out_of_order(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="cannot be priced at"):
            MODEL.bond_price(curve, 5.0, 2.0, 0.0, 0.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            MODEL.bond_price(curve, -1.0, 2.0, 0.0, 0.0)
        with pytest.raises(BadModel, match="cannot underlie an option"):
            MODEL.bond_option_volatility(5.0, 2.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            MODEL.bond_option_volatility(-1.0, 2.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            MODEL.variance_term(-1.0)

    def test_a_bad_strike(self, curve: DiscountCurve) -> None:
        for strike in (0.0, -0.5, math.nan):
            with pytest.raises(BadModel, match="strike must be positive"):
                bond_option(curve, MODEL, expiry=1.0, maturity=5.0, strike=strike)

    def test_a_caplet_with_no_length(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="positive length"):
            caplet(curve, MODEL, start=2.0, end=2.0, strike=0.03)
        with pytest.raises(BadModel, match="positive length"):
            caplet(curve, MODEL, start=2.0, end=1.5, strike=0.03)
        with pytest.raises(BadModel, match="accrual factor must be positive"):
            caplet(curve, MODEL, start=2.0, end=2.5, strike=0.03, accrual=0.0)

    def test_a_strike_that_makes_the_bond_strike_undefined(
        self, curve: DiscountCurve
    ) -> None:
        with pytest.raises(BadModel, match="equivalent bond strike is undefined"):
            caplet(curve, MODEL, start=2.0, end=2.5, strike=-10.0)

    def test_a_cap_needs_two_dates(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="at least two dates"):
            cap_price(curve, MODEL, dates=[1.0], strike=0.03)

    def test_mismatched_accruals(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="accrual factors were given"):
            cap_price(curve, MODEL, dates=[1.0, 1.5, 2.0], strike=0.03, accruals=[0.5])

    def test_a_negative_horizon_for_the_correlation(self) -> None:
        with pytest.raises(BadModel, match="horizon must be positive"):
            zero_rate_correlation(MODEL, 0.0, 1.0, 2.0)
        with pytest.raises(BadModel, match="horizon must be positive"):
            zero_rate_correlation(MODEL, -1.0, 1.0, 2.0)

    def test_a_maturity_before_the_horizon(self) -> None:
        with pytest.raises(BadModel, match="first maturity"):
            zero_rate_correlation(MODEL, 5.0, 2.0, 10.0)
        with pytest.raises(BadModel, match="second maturity"):
            zero_rate_correlation(MODEL, 5.0, 6.0, 4.0)

    def test_a_model_with_no_volatility_has_no_correlation_to_report(self) -> None:
        flat = G2(a=0.5, sigma=0.0, b=0.05, eta=0.0, rho=0.5)
        with pytest.raises(BadModel, match="no correlation to report"):
            zero_rate_correlation(flat, 1.0, 2.0, 11.0)

    def test_a_negative_time_for_the_moments(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="non-negative"):
            factor_moments(MODEL, -1.0)
        with pytest.raises(BadModel, match="non-negative"):
            MODEL.short_rate_mean(curve, -1.0)
        with pytest.raises(BadModel, match="non-negative"):
            factor_moments(MODEL, math.nan)

    def test_no_paths(self) -> None:
        with pytest.raises(BadModel, match="positive number of draws"):
            simulate_factors(MODEL, 1.0, paths=0)
        with pytest.raises(BadModel, match="positive number of draws"):
            simulate_factors(MODEL, 1.0, paths=-5)

    def test_a_zero_horizon_has_no_randomness(self) -> None:
        moments = factor_moments(MODEL, 0.0)
        assert moments.variance_x == 0.0
        assert moments.variance_y == 0.0
        assert moments.covariance == 0.0
        assert moments.correlation == 0.0
        assert simulate_factors(MODEL, 0.0, paths=3) == [(0.0, 0.0)] * 3

    def test_a_model_with_only_the_second_factor_still_draws(self) -> None:
        """The first factor's scale is zero, so the conditioning has nothing to use."""
        half = G2(a=0.5, sigma=0.0, b=0.05, eta=0.007, rho=0.5)
        draws = simulate_factors(half, 2.0, paths=5000, seed=3)
        assert all(x == 0.0 for x, _ in draws)
        assert statistics.variance([y for _, y in draws]) == pytest.approx(
            factor_moments(half, 2.0).variance_y, rel=0.1
        )


class TestFittingTheFastVolatility:
    """The search, and the non-monotonicity that makes a bisection wrong."""

    def test_it_recovers_a_volatility_it_was_given(self, curve: DiscountCurve) -> None:
        dates = [1.0 + 0.5 * index for index in range(11)]
        for sigma in (0.004, 0.011, 0.03):
            model = G2(a=0.50, sigma=sigma, b=0.05, eta=0.007, rho=-0.3)
            target = cap_price(curve, model, dates=dates, strike=0.038)
            root = fit_fast_volatility(
                curve, model, dates=dates, strike=0.038, target=target
            )
            assert root.value == pytest.approx(sigma, rel=1e-9)
            assert root.converged
            assert abs(root.residual) < 1e-12

    def test_a_cap_is_not_monotone_in_the_fast_volatility(
        self, curve: DiscountCurve
    ) -> None:
        """The reason the bracket is searched for rather than assumed.

        The bond option variance is ``sigma**2 A + eta**2 B + 2 rho sigma eta
        C`` with all three coefficients positive, so at a negative ``rho`` it
        is a parabola in ``sigma`` with its minimum strictly inside the domain.
        Raising ``sigma`` from nothing there makes the cap *cheaper*, and a
        bisection from a floor to a ceiling converges to the floor while
        reporting success.
        """
        dates = [1.0 + 0.5 * index for index in range(11)]
        prices = [
            cap_price(
                curve,
                G2(a=0.50, sigma=sigma, b=0.05, eta=0.007, rho=-0.9),
                dates=dates,
                strike=0.038,
            )
            for sigma in (1e-9, 0.002, 0.005, 0.01, 0.02, 0.05)
        ]
        assert prices != sorted(prices)
        assert prices[1] < prices[0]
        # And the same sweep at a positive rho is monotone, which is why the
        # defect only shows up on half the parameter space.
        rising = [
            cap_price(
                curve,
                G2(a=0.50, sigma=sigma, b=0.05, eta=0.007, rho=0.9),
                dates=dates,
                strike=0.038,
            )
            for sigma in (1e-9, 0.002, 0.005, 0.01, 0.02, 0.05)
        ]
        assert rising == sorted(rising)

    def test_a_target_below_the_cheapest_attainable_price_is_refused(
        self, curve: DiscountCurve
    ) -> None:
        """And the refusal names the cheapest price, because that is the answer."""
        dates = [1.0 + 0.5 * index for index in range(11)]
        model = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=-0.9)
        cheapest = min(
            cap_price(
                curve,
                G2(a=0.50, sigma=sigma, b=0.05, eta=0.007, rho=-0.9),
                dates=dates,
                strike=0.038,
            )
            for sigma in [1e-9 * (1e9) ** (i / 79.0) for i in range(80)]
        )
        with pytest.raises(NoRoot, match="cheapest price on the grid"):
            fit_fast_volatility(
                curve,
                model,
                dates=dates,
                strike=0.038,
                target=cheapest * 0.5,
            )

    def test_it_picks_the_upper_branch(self, curve: DiscountCurve) -> None:
        """Where more volatility means a dearer cap, which is the sensible root.

        At a negative ``rho`` a target above the minimum has two roots. The one
        below the parabola's vertex prices a cap that gets *cheaper* as the
        market gets more volatile, which is not a calibration anybody wants.
        """
        dates = [1.0 + 0.5 * index for index in range(11)]
        model = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=-0.9)
        at_zero = cap_price(
            curve,
            G2(a=0.50, sigma=1e-9, b=0.05, eta=0.007, rho=-0.9),
            dates=dates,
            strike=0.038,
        )
        # A target just above the sigma -> 0 price has a root on each branch.
        root = fit_fast_volatility(
            curve, model, dates=dates, strike=0.038, target=at_zero * 1.0001
        )
        bumped = cap_price(
            curve,
            G2(a=0.50, sigma=root.value * 1.05, b=0.05, eta=0.007, rho=-0.9),
            dates=dates,
            strike=0.038,
        )
        assert bumped > at_zero * 1.0001

    def test_bad_search_arguments(self, curve: DiscountCurve) -> None:
        dates = [1.0, 1.5, 2.0]
        model = G2(a=0.50, sigma=0.011, b=0.05, eta=0.007, rho=0.0)
        with pytest.raises(BadModel, match="upper must be finite and positive"):
            fit_fast_volatility(
                curve, model, dates=dates, strike=0.038, target=0.001, upper=0.0
            )
        with pytest.raises(BadModel, match="nodes must be at least 3"):
            fit_fast_volatility(
                curve, model, dates=dates, strike=0.038, target=0.001, nodes=2
            )

    def test_it_fits_a_floor_too(self, curve: DiscountCurve) -> None:
        dates = [1.0 + 0.5 * index for index in range(11)]
        model = G2(a=0.50, sigma=0.014, b=0.05, eta=0.007, rho=0.2)
        target = cap_price(curve, model, dates=dates, strike=0.030, floor=True)
        root = fit_fast_volatility(
            curve, model, dates=dates, strike=0.030, target=target, floor=True
        )
        assert root.value == pytest.approx(0.014, rel=1e-8)
