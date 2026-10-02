"""Hull-White: four identities, two independent routes, and one unidentified pair.

The model is small enough that almost everything about it can be checked
against something exact rather than against itself.

The curve repricing is an identity built into ``A(t, T)``, so it has to hold
to the last bit and not to a tolerance. Put-call parity on a bond option is an
identity in the model's algebra. The forward-measure mean is over-determined —
every probe maturity implies it — so the probes have to agree. And Jamshidian's
decomposition has a second route to be measured against, which matters because
the decomposition is the one step here that is a theorem rather than a
substitution.

The parts that are *not* identities are measured instead: how far mean
reversion and volatility trade off against each other, and how flat the normal
volatility the model generates is compared with the lognormal one.
"""

from __future__ import annotations

import datetime
import math
from datetime import date

import pytest

from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis, year_fraction
from tenor.hullwhite import (
    BadModel,
    CouponBondOption,
    Flow,
    HullWhite,
    Method,
    Quote,
    calibrate,
    coupon_bond_option,
    forward_measure,
    reprices_curve,
    swaption_price,
    zero_bond_option,
)
from tenor.options import Convention, Payoff, Swaption, implied_volatility
from tenor.schedule import Frequency
from tenor.solve import solve

REFERENCE = date(2026, 1, 2)
BASIS = Basis.ACT_365F

# Seventeen pillars with a gentle hump: rising to five years, flat to seven,
# falling after. A monotone curve would hide an instantaneous forward read off
# the wrong side of a pillar, which is exactly what A(t, T) depends on.
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

MODEL = HullWhite(a=0.08, sigma=0.009)


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


def bullet(
    expiry: float, coupon: float, periods: int = 20, payoff: Payoff = Payoff.PAYER
) -> CouponBondOption:
    """An option on a ten-year semi-annual bullet starting at the expiry."""
    flows = tuple(
        Flow(expiry + 0.5 * index, coupon * 0.5 + (1.0 if index == periods else 0.0))
        for index in range(1, periods + 1)
    )
    return CouponBondOption(expiry, flows, 1.0, payoff)


def swaption(
    expiry: float, tenor: float, strike: float, payoff: Payoff = Payoff.PAYER
) -> Swaption:
    return Swaption(
        expiry=at(expiry),
        effective=at(expiry),
        maturity=at(expiry + tenor),
        strike=strike,
        payoff=payoff,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.THIRTY_360_BOND,
    )


def annuity(curve: DiscountCurve, option: Swaption) -> float:
    return sum(
        year_fraction(period.start, period.end, option.basis)
        * curve.discount(period.payment)
        for period in option.swap.fixed_schedule()
    )


def par_rate(curve: DiscountCurve, expiry: float, tenor: float) -> float:
    """The forward par rate on the swap's *own* dates.

    The notional legs sit on the schedule's adjusted start and last payment
    rather than on the nominal effective and maturity, because that is where
    the money actually moves and the parity identity is only exact there.
    """
    option = swaption(expiry, tenor, 0.03)
    schedule = option.swap.fixed_schedule()
    start = curve.discount(schedule[0].adjusted_start)
    end = curve.discount(schedule[-1].payment)
    return (start - end) / annuity(curve, option)


class TestModel:
    def test_b_is_the_integral_of_the_decay(self) -> None:
        """``B(tenor)`` is ``int_0^tenor e^{-a s} ds``, by a Riemann sum."""
        for a in (0.01, 0.08, 0.5):
            model = HullWhite(a=a, sigma=0.01)
            for tenor in (0.5, 5.0, 30.0):
                steps = 200_000
                step = tenor / steps
                total = sum(
                    math.exp(-a * (index + 0.5) * step) * step for index in range(steps)
                )
                assert model.b(tenor) == pytest.approx(total, rel=1e-9)

    def test_b_is_the_tenor_at_short_horizons(self) -> None:
        """Because ``(1 - e^{-ax})/a -> x``, which is the no-reversion limit."""
        model = HullWhite(a=0.08, sigma=0.01)
        assert model.b(0.0) == 0.0
        assert model.b(1e-8) == pytest.approx(1e-8, rel=1e-7)

    def test_short_rate_variance_saturates(self) -> None:
        """It rises to ``sigma^2 / 2a`` and stops, which is the reversion working."""
        model = HullWhite(a=0.1, sigma=0.012)
        limit = model.sigma**2 / (2.0 * model.a)
        values = [model.short_rate_variance(t) for t in (0.5, 5.0, 50.0, 500.0)]
        assert values == sorted(values)
        assert values[-1] == pytest.approx(limit, rel=1e-12)
        assert values[0] < 0.2 * limit

    def test_bond_volatility_factorises(self) -> None:
        model = HullWhite(a=0.08, sigma=0.009)
        for expiry in (1.0, 10.0):
            for maturity in (expiry, expiry + 2.0, expiry + 20.0):
                expected = model.b(maturity - expiry) * math.sqrt(
                    model.short_rate_variance(expiry)
                )
                assert model.bond_volatility(expiry, maturity) == pytest.approx(expected)
        assert model.bond_volatility(5.0, 5.0) == 0.0

    @pytest.mark.parametrize(
        ("a", "sigma", "message"),
        [
            (0.0, 0.01, "positive"),
            (-0.1, 0.01, "positive"),
            (math.inf, 0.01, "positive"),
            (0.08, -0.01, "non-negative"),
            (0.08, math.nan, "non-negative"),
        ],
    )
    def test_refuses_bad_parameters(self, a: float, sigma: float, message: str) -> None:
        with pytest.raises(BadModel, match=message):
            HullWhite(a=a, sigma=sigma)

    def test_the_zero_reversion_limit_is_refused_by_name(self) -> None:
        with pytest.raises(BadModel, match="Ho-Lee"):
            HullWhite(a=0.0, sigma=0.01)

    def test_refuses_a_bond_maturing_before_its_option(self) -> None:
        with pytest.raises(BadModel, match="cannot underlie"):
            MODEL.bond_volatility(5.0, 4.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            MODEL.b(-1.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            MODEL.short_rate_variance(-1.0)


class TestRepricing:
    """The curve has to come back exactly, because the identity is in ``A``."""

    @pytest.mark.parametrize("a", [0.01, 0.05, 0.08, 0.2, 0.5])
    @pytest.mark.parametrize("sigma", [0.0, 0.005, 0.02])
    def test_reprices_every_pillar_exactly(
        self, curve: DiscountCurve, a: float, sigma: float
    ) -> None:
        assert reprices_curve(curve, HullWhite(a=a, sigma=sigma)) == 0.0

    def test_the_bond_price_is_monotone_in_the_rate(self, curve: DiscountCurve) -> None:
        prices = [
            MODEL.bond_price(curve, 5.0, 15.0, rate)
            for rate in (-0.02, 0.0, 0.02, 0.04, 0.08)
        ]
        assert prices == sorted(prices, reverse=True)

    def test_a_bond_maturing_at_the_expiry_is_worth_one(
        self, curve: DiscountCurve
    ) -> None:
        for rate in (-0.01, 0.03, 0.09):
            assert MODEL.bond_price(curve, 7.0, 7.0, rate) == pytest.approx(1.0, rel=1e-15)


class TestBondOption:
    @pytest.mark.parametrize("expiry", [0.5, 2.0, 5.0, 10.0])
    @pytest.mark.parametrize("gap", [1.0, 5.0, 20.0])
    def test_put_call_parity_is_exact(
        self, curve: DiscountCurve, expiry: float, gap: float
    ) -> None:
        """``call - put = P(0,S) - K P(0,T)``, with nothing approximate in it."""
        maturity = expiry + gap
        for ratio in (0.9, 1.0, 1.1):
            strike = ratio * curve.discount_at(maturity) / curve.discount_at(expiry)
            call = zero_bond_option(curve, MODEL, expiry, maturity, strike, Payoff.RECEIVER)
            put = zero_bond_option(curve, MODEL, expiry, maturity, strike, Payoff.PAYER)
            identity = curve.discount_at(maturity) - strike * curve.discount_at(expiry)
            assert call - put == pytest.approx(identity, abs=1e-15)

    def test_zero_volatility_is_intrinsic(self, curve: DiscountCurve) -> None:
        model = HullWhite(a=0.08, sigma=0.0)
        forward = curve.discount_at(10.0) / curve.discount_at(5.0)
        for ratio in (0.8, 1.0, 1.2):
            strike = ratio * forward
            call = zero_bond_option(curve, model, 5.0, 10.0, strike, Payoff.RECEIVER)
            put = zero_bond_option(curve, model, 5.0, 10.0, strike, Payoff.PAYER)
            discounted = curve.discount_at(5.0)
            assert call == pytest.approx(max(forward - strike, 0.0) * discounted, abs=1e-16)
            assert put == pytest.approx(max(strike - forward, 0.0) * discounted, abs=1e-16)

    def test_value_rises_with_volatility(self, curve: DiscountCurve) -> None:
        forward = curve.discount_at(12.0) / curve.discount_at(2.0)
        values = [
            zero_bond_option(
                curve, HullWhite(a=0.08, sigma=sigma), 2.0, 12.0, forward, Payoff.PAYER
            )
            for sigma in (0.0, 0.002, 0.006, 0.012, 0.02)
        ]
        assert values == sorted(values)
        assert values[0] == 0.0

    def test_refuses_a_non_positive_strike(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="must be positive"):
            zero_bond_option(curve, MODEL, 1.0, 5.0, 0.0)
        with pytest.raises(BadModel, match="cannot be negative"):
            zero_bond_option(curve, MODEL, -1.0, 5.0, 0.9)


class TestForwardMeasure:
    """The mean is over-determined, so every probe has to give the same answer."""

    @pytest.mark.parametrize("expiry", [1.0, 5.0, 10.0, 20.0])
    def test_probes_agree(self, curve: DiscountCurve, expiry: float) -> None:
        measure = forward_measure(curve, MODEL, expiry, (1.0, 3.0, 5.0, 10.0))
        # The terms cancel identically, so the only residual is the rounding
        # of the division by B that solves for the mean: 6.9e-18 at worst.
        assert measure.spread < 1e-17
        assert measure.variance == pytest.approx(MODEL.short_rate_variance(expiry))
        assert measure.standard_deviation == pytest.approx(math.sqrt(measure.variance))

    @pytest.mark.parametrize("expiry", [1.0, 5.0, 10.0, 20.0])
    def test_the_mean_is_the_instantaneous_forward(
        self, curve: DiscountCurve, expiry: float
    ) -> None:
        """Not approximately: the convexity terms cancel identically.

        ``B**2 v / 2`` from the lognormal expectation is exactly the term
        inside ``A``, so the implied mean is ``f(0, T)`` with nothing left
        over, at every volatility.
        """
        for sigma in (0.0, 0.004, 0.009, 0.03):
            measure = forward_measure(curve, HullWhite(a=0.08, sigma=sigma), expiry)
            assert measure.mean == pytest.approx(
                curve.instantaneous_forward(expiry), abs=1e-15
            )

    def test_forward_bond_prices_are_martingales_under_it(
        self, curve: DiscountCurve
    ) -> None:
        """``E^T[P(T,S)] = P(0,S)/P(0,T)``, which is the property it was built on."""
        expiry = 7.0
        measure = forward_measure(curve, MODEL, expiry)
        for offset in (0.5, 2.0, 8.0, 23.0):
            maturity = expiry + offset
            tenor = MODEL.b(offset)
            expected = math.exp(
                MODEL.log_a(curve, expiry, maturity)
                - tenor * measure.mean
                + 0.5 * tenor * tenor * measure.variance
            )
            assert expected == pytest.approx(
                curve.discount_at(maturity) / curve.discount_at(expiry), rel=1e-14
            )

    def test_refuses_a_bad_expiry_or_probe(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="positive expiry"):
            forward_measure(curve, MODEL, 0.0)
        with pytest.raises(BadModel, match="probe offset"):
            forward_measure(curve, MODEL, 5.0, (0.0,))


class TestDecomposition:
    """Jamshidian against an integral that shares none of its algebra."""

    @pytest.mark.parametrize("expiry", [1.0, 5.0, 10.0])
    @pytest.mark.parametrize("coupon", [0.02, 0.0374, 0.06])
    @pytest.mark.parametrize("payoff", [Payoff.PAYER, Payoff.RECEIVER])
    def test_two_routes_agree(
        self, curve: DiscountCurve, expiry: float, coupon: float, payoff: Payoff
    ) -> None:
        option = bullet(expiry, coupon, payoff=payoff)
        decomposed = coupon_bond_option(curve, MODEL, option, Method.JAMSHIDIAN)
        integrated = coupon_bond_option(curve, MODEL, option, Method.QUADRATURE)
        assert decomposed == pytest.approx(integrated, rel=1e-12)
        assert decomposed > 0.0

    def test_parity_holds_on_the_coupon_bond_too(self, curve: DiscountCurve) -> None:
        """Call less put is the forward bond price less the strike, discounted."""
        for expiry in (1.0, 5.0, 10.0):
            option = bullet(expiry, 0.0374)
            call = coupon_bond_option(
                curve, MODEL, bullet(expiry, 0.0374, payoff=Payoff.RECEIVER)
            )
            put = coupon_bond_option(curve, MODEL, option)
            forward = sum(
                flow.amount * curve.discount_at(flow.time) for flow in option.flows
            )
            identity = forward - option.strike * curve.discount_at(expiry)
            assert call - put == pytest.approx(identity, abs=1e-14)

    def test_the_reference_rate_prices_the_bond_at_the_strike(
        self, curve: DiscountCurve
    ) -> None:
        option = bullet(5.0, 0.0374)
        measure = forward_measure(curve, MODEL, 5.0)
        reference = solve(
            lambda rate: option.value_at(curve, MODEL, rate) - option.strike,
            low=measure.mean - 1.0,
            high=measure.mean + 1.0,
        ).value
        assert option.value_at(curve, MODEL, reference) == pytest.approx(
            option.strike, abs=1e-14
        )

    def test_a_bond_with_a_negative_flow_refuses_the_decomposition(
        self, curve: DiscountCurve
    ) -> None:
        option = CouponBondOption(
            2.0, (Flow(3.0, -1.0), Flow(5.0, 1.05)), 0.1, Payoff.RECEIVER
        )
        assert not option.is_monotone
        with pytest.raises(BadModel, match="every cash flow to be positive"):
            coupon_bond_option(curve, MODEL, option, Method.JAMSHIDIAN)
        # The integral has no such requirement and still prices it.
        assert coupon_bond_option(curve, MODEL, option, Method.QUADRATURE) > 0.0

    def test_zero_volatility_is_intrinsic(self, curve: DiscountCurve) -> None:
        model = HullWhite(a=0.08, sigma=0.0)
        option = bullet(5.0, 0.0374)
        forward = sum(
            flow.amount * curve.discount_at(flow.time) for flow in option.flows
        ) / curve.discount_at(5.0)
        expected = max(option.strike - forward, 0.0) * curve.discount_at(5.0)
        for method in Method:
            assert coupon_bond_option(curve, MODEL, option, method) > 0.0
            assert coupon_bond_option(curve, model, option, method) == pytest.approx(
                expected, abs=1e-15
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"expiry": 0.0}, "expiry must be positive"),
            ({"flows": ()}, "at least one cash flow"),
            ({"strike": -1.0}, "non-negative"),
        ],
    )
    def test_refuses_a_bad_option(self, kwargs: dict[str, object], message: str) -> None:
        base: dict[str, object] = {
            "expiry": 1.0,
            "flows": (Flow(2.0, 1.0),),
            "strike": 1.0,
            "payoff": Payoff.PAYER,
        }
        base.update(kwargs)
        with pytest.raises(BadModel, match=message):
            CouponBondOption(**base)  # type: ignore[arg-type]

    def test_refuses_a_flow_that_does_not_survive_the_expiry(self) -> None:
        with pytest.raises(BadModel, match="does not survive"):
            CouponBondOption(5.0, (Flow(4.0, 1.0),), 1.0)

    def test_refuses_flows_out_of_order(self) -> None:
        with pytest.raises(BadModel, match="time order"):
            CouponBondOption(1.0, (Flow(5.0, 0.5), Flow(3.0, 1.0)), 1.0)

    def test_refuses_a_non_finite_amount(self) -> None:
        with pytest.raises(BadModel, match="must be finite"):
            CouponBondOption(1.0, (Flow(2.0, math.nan),), 1.0)


class TestSwaption:
    def test_both_routes_and_both_sides_agree_at_the_money(
        self, curve: DiscountCurve
    ) -> None:
        rate = par_rate(curve, 10.0, 10.0)
        payer = swaption_price(curve, MODEL, swaption(10.0, 10.0, rate), REFERENCE, BASIS)
        receiver = swaption_price(
            curve, MODEL, swaption(10.0, 10.0, rate, Payoff.RECEIVER), REFERENCE, BASIS
        )
        assert abs(payer.gap) < 1e-12
        assert abs(receiver.gap) < 1e-12
        # A payer and a receiver struck at the forward are worth the same,
        # because the swap between them is worth nothing.
        assert payer.price == pytest.approx(receiver.price, abs=1e-15)
        assert payer.reference_rate is not None

    def test_a_payer_falls_with_the_strike_and_a_receiver_rises(
        self, curve: DiscountCurve
    ) -> None:
        """Paying a higher fixed rate is worse, and receiving one is better."""
        rate = par_rate(curve, 5.0, 10.0)
        payers = []
        receivers = []
        for ratio in (0.7, 0.9, 1.0, 1.1, 1.3):
            payers.append(
                swaption_price(
                    curve, MODEL, swaption(5.0, 10.0, rate * ratio), REFERENCE, BASIS
                ).price
            )
            receivers.append(
                swaption_price(
                    curve,
                    MODEL,
                    swaption(5.0, 10.0, rate * ratio, Payoff.RECEIVER),
                    REFERENCE,
                    BASIS,
                ).price
            )
        assert payers == sorted(payers, reverse=True)
        assert receivers == sorted(receivers)

    def test_payer_less_receiver_is_the_forward_swap(self, curve: DiscountCurve) -> None:
        """The same identity :mod:`tenor.options` is held to, on a different model."""
        rate = par_rate(curve, 5.0, 10.0)
        for ratio in (0.8, 1.0, 1.25):
            strike = rate * ratio
            payer = swaption_price(
                curve, MODEL, swaption(5.0, 10.0, strike), REFERENCE, BASIS
            ).price
            receiver = swaption_price(
                curve, MODEL, swaption(5.0, 10.0, strike, Payoff.RECEIVER), REFERENCE, BASIS
            ).price
            swap = annuity(curve, swaption(5.0, 10.0, strike)) * (strike - rate)

            assert receiver - payer == pytest.approx(swap, abs=1e-14)

    def test_a_mid_curve_swaption_prices_by_quadrature_only(
        self, curve: DiscountCurve
    ) -> None:
        """Its notional exchange is a negative flow, so monotonicity is not given."""
        option = Swaption(
            expiry=at(2.0),
            effective=at(5.0),
            maturity=at(15.0),
            strike=0.038,
            payoff=Payoff.PAYER,
            frequency=Frequency.SEMI_ANNUAL,
            basis=Basis.THIRTY_360_BOND,
        )
        trade = swaption_price(curve, MODEL, option, REFERENCE, BASIS)
        assert trade.jamshidian is None
        assert trade.reference_rate is None
        assert trade.gap == 0.0
        assert trade.price > 0.0
        assert not trade.option.is_monotone

    def test_refuses_an_expired_swaption(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="before the valuation date"):
            swaption_price(
                curve, MODEL, swaption(10.0, 10.0, 0.03), at(10.0) + datetime.timedelta(1)
            )


class TestIdentification:
    """What one quote can and cannot say about two parameters."""

    def test_the_volatility_moves_by_a_factor_of_six_along_the_ridge(
        self, curve: DiscountCurve
    ) -> None:
        """Every mean reversion reprices the quote, so the fit cannot choose.

        Measured: volatilities from 0.492% at a reversion of 0.01 to 3.226% at
        0.30, a factor of 6.55, each repricing the quote to 2.2e-16. This is
        why :func:`calibrate` takes the reversion as an argument.
        """
        rate = par_rate(curve, 10.0, 10.0)
        option = swaption(10.0, 10.0, rate)
        target = swaption_price(curve, MODEL, option, REFERENCE, BASIS).price
        fitted = []
        for reversion in (0.01, 0.02, 0.05, 0.08, 0.15, 0.30):
            result = calibrate(
                curve,
                [Quote(option, target)],
                REFERENCE,
                mean_reversion=reversion,
                basis=BASIS,
            )
            fitted.append(result.model.sigma)
            assert result.worst < 1e-14
        assert fitted == sorted(fitted)
        assert max(fitted) / min(fitted) == pytest.approx(6.55, abs=0.05)
        # And the one that was used to generate the quote comes back.
        assert fitted[3] == pytest.approx(MODEL.sigma, rel=1e-10)

    def test_a_second_expiry_separates_them(self, curve: DiscountCurve) -> None:
        """The ridge is flat in one price and steep in another.

        The pair fitted to a ten-year expiry prices a one-year expiry from
        23.4% below to 57.2% above the reversion-0.08 answer, so the two
        expiries together pin a pair that either alone cannot.
        """
        long_rate = par_rate(curve, 10.0, 10.0)
        long_option = swaption(10.0, 10.0, long_rate)
        target = swaption_price(curve, MODEL, long_option, REFERENCE, BASIS).price
        short_option = swaption(1.0, 10.0, par_rate(curve, 1.0, 10.0))
        prices = []
        for reversion in (0.01, 0.30):
            fit = calibrate(
                curve,
                [Quote(long_option, target)],
                REFERENCE,
                mean_reversion=reversion,
                basis=BASIS,
            )
            prices.append(
                swaption_price(curve, fit.model, short_option, REFERENCE, BASIS).price
            )
        base = swaption_price(curve, MODEL, short_option, REFERENCE, BASIS).price
        assert prices[0] / base - 1.0 == pytest.approx(-0.2339, abs=0.002)
        assert prices[1] / base - 1.0 == pytest.approx(0.5720, abs=0.002)

    def test_a_strip_matches_its_total_premium(self, curve: DiscountCurve) -> None:
        quotes = []
        for expiry in (1.0, 5.0, 10.0):
            option = swaption(expiry, 10.0, par_rate(curve, expiry, 10.0))
            quotes.append(
                Quote(option, swaption_price(curve, MODEL, option, REFERENCE, BASIS).price)
            )
        fit = calibrate(curve, quotes, REFERENCE, mean_reversion=0.08, basis=BASIS)
        # The quotes came from this model, so the compromise is no compromise.
        assert fit.model.sigma == pytest.approx(MODEL.sigma, rel=1e-9)
        assert fit.worst < 1e-9
        assert len(fit.errors) == 3
        assert fit.iterations > 0

    def test_a_strip_at_the_wrong_reversion_leaves_a_real_residual(
        self, curve: DiscountCurve
    ) -> None:
        """One volatility cannot fit three expiries generated at another reversion."""
        quotes = []
        for expiry in (1.0, 5.0, 10.0):
            option = swaption(expiry, 10.0, par_rate(curve, expiry, 10.0))
            quotes.append(
                Quote(option, swaption_price(curve, MODEL, option, REFERENCE, BASIS).price)
            )
        fit = calibrate(curve, quotes, REFERENCE, mean_reversion=0.30, basis=BASIS)
        assert fit.worst > 0.05
        # The errors change sign across the strip, which is what a term
        # structure the model cannot reach looks like.
        assert min(fit.errors) < 0.0 < max(fit.errors)

    def test_refuses_an_empty_strip(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="at least one quote"):
            calibrate(curve, [], REFERENCE, mean_reversion=0.08)

    def test_refuses_a_negative_premium(self, curve: DiscountCurve) -> None:
        with pytest.raises(BadModel, match="non-negative"):
            Quote(swaption(5.0, 10.0, 0.03), -1.0)


class TestAgainstTheAnnuityMeasure:
    """What the model says in the units :mod:`tenor.options` quotes."""

    def test_the_normal_volatility_is_nearly_flat(self, curve: DiscountCurve) -> None:
        """A Gaussian short rate makes a nearly Gaussian swap rate.

        Across strikes from 60% to 140% of the forward the normal volatility
        moves 1.06 basis points on a level of 45.55 — 2.33% wide — against the
        lognormal convention's 41.91% on the same prices, eighteen times as
        much. It is also monotone rather than a smile, which is a skew and not
        a curvature.
        """
        rate = par_rate(curve, 10.0, 10.0)
        ratios = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
        spans = {}
        for convention in (Convention.NORMAL, Convention.LOGNORMAL):
            vols = []
            for ratio in ratios:
                strike = rate * ratio
                payoff = Payoff.PAYER if strike >= rate else Payoff.RECEIVER
                option = swaption(10.0, 10.0, strike, payoff)
                price = swaption_price(curve, MODEL, option, REFERENCE, BASIS).price
                vols.append(
                    implied_volatility(
                        price / annuity(curve, option),
                        forward=rate,
                        strike=strike,
                        time=year_fraction(REFERENCE, option.expiry, BASIS),
                        payoff=payoff,
                        convention=convention,
                    )
                )
            middle = vols[ratios.index(1.0)]
            spans[convention] = (max(vols) - min(vols)) / middle
            if convention is Convention.NORMAL:
                assert vols == sorted(vols)
                assert middle * 1e4 == pytest.approx(45.55, abs=0.05)
        assert spans[Convention.NORMAL] == pytest.approx(0.0233, abs=0.0005)
        assert spans[Convention.LOGNORMAL] == pytest.approx(0.4191, abs=0.0010)
        assert spans[Convention.LOGNORMAL] / spans[Convention.NORMAL] > 15.0
