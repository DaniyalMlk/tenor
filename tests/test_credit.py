"""Default risk: the survival curve, the swap, and the rule of thumb measured."""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor.bond import Bond
from tenor.calendar import WEEKENDS_ONLY
from tenor.credit import (
    BadCredit,
    CreditDefaultSwap,
    SurvivalCurve,
    _expm1_ratio,
    _ramp_integral,
    _survival_integral,
    bootstrap_hazards,
    implied_hazard,
    risky_bond_price,
    triangle_hazard,
    triangle_spread,
)
from tenor.curve import flat_curve
from tenor.daycount import Basis
from tenor.schedule import Frequency

REFERENCE = date(2026, 6, 15)
FIVE_YEAR = date(2031, 6, 20)
HORIZON = date(2066, 6, 15)
RECOVERY = 0.4


def discount(rate: float = 0.03):  # type: ignore[no-untyped-def]
    return flat_curve(REFERENCE, HORIZON, rate, basis=Basis.ACT_365F)


def flat_hazard(hazard: float, maturity: date = FIVE_YEAR) -> SurvivalCurve:
    return SurvivalCurve.flat(
        REFERENCE, maturity, hazard, basis=Basis.ACT_365F, recovery=RECOVERY
    )


def swap(
    maturity: date = FIVE_YEAR,
    *,
    basis: Basis = Basis.ACT_365F,
    frequency: Frequency = Frequency.QUARTERLY,
    accrual: bool = True,
) -> CreditDefaultSwap:
    built = CreditDefaultSwap.standard(
        REFERENCE, maturity, 0.0, calendar=WEEKENDS_ONLY, basis=basis, frequency=frequency
    )
    if accrual:
        return built
    return CreditDefaultSwap(built.schedule, built.basis, 0.0, 1.0, accrual_on_default=False)


class TestTheIntegralHelpers:
    """Both are exponential integrals written to survive their own limits."""

    @pytest.mark.parametrize("x", [1e-3, 0.1, 1.0, 5.0, 40.0])
    def test_the_decay_ratio_matches_its_definition(self, x: float) -> None:
        assert _expm1_ratio(x) == pytest.approx((1.0 - math.exp(-x)) / x, rel=1e-12)

    def test_the_decay_ratio_is_one_at_zero(self) -> None:
        assert _expm1_ratio(0.0) == 1.0

    @pytest.mark.parametrize("x", [1e-3, 0.1, 1.0, 5.0, 40.0])
    def test_the_ramp_matches_a_direct_quadrature(self, x: float) -> None:
        # integral_0^1 u exp(-x u) du, by a fine Simpson rule, which shares no
        # code with the closed form.
        n = 20_000
        total = 0.0
        for i in range(n + 1):
            u = i / n
            weight = 1 if i in (0, n) else (4 if i % 2 else 2)
            total += weight * u * math.exp(-x * u)
        assert _ramp_integral(x) == pytest.approx(total / (3.0 * n), rel=1e-9)

    def test_the_ramp_is_one_half_at_zero(self) -> None:
        # Which is exactly the half-period approximation everybody writes.
        assert _ramp_integral(0.0) == 0.5

    def test_neither_loses_its_digits_near_zero(self) -> None:
        # The naive forms do. At 1e-9 the subtraction leaves seven digits.
        naive = (1.0 - math.exp(-1e-9)) / 1e-9
        assert abs(naive - 1.0) > 1e-10
        assert abs(_expm1_ratio(1e-9) - 1.0) < 1e-9


class TestSurvivalCurve:
    def test_survival_is_one_today_and_falls(self) -> None:
        curve = flat_hazard(0.03)
        assert curve.survival(REFERENCE) == 1.0
        previous = 1.0
        for year in range(1, 11):
            value = curve.survival(date(2026 + year, 6, 15))
            assert 0.0 < value < previous
            previous = value

    def test_a_flat_hazard_is_an_exponential(self) -> None:
        curve = flat_hazard(0.05)
        for years in (0.5, 1.0, 3.7, 12.0):
            assert curve.survival_at(years) == pytest.approx(
                math.exp(-0.05 * years), rel=1e-14
            )

    def test_a_zero_hazard_never_defaults(self) -> None:
        curve = flat_hazard(0.0)
        assert curve.survival_at(50.0) == 1.0
        assert curve.default_probability(date(2076, 6, 15)) == 0.0

    def test_the_integrated_hazard_is_piecewise_linear(self) -> None:
        curve = SurvivalCurve.from_hazards(
            REFERENCE,
            [(date(2027, 6, 15), 0.01), (date(2029, 6, 15), 0.04), (date(2031, 6, 15), 0.02)],
            basis=Basis.ACT_365F,
            recovery=RECOVERY,
        )
        first = curve.times[0]
        # Inside the first piece, doubling the time doubles the integral.
        assert curve.integrated_hazard(first / 2.0) == pytest.approx(
            0.5 * curve.integrated_hazard(first), rel=1e-12
        )
        # And the second piece starts from where the first ended.
        second = curve.times[1]
        middle = 0.5 * (first + second)
        assert curve.integrated_hazard(middle) == pytest.approx(
            curve.integrated_hazard(first) + 0.04 * (middle - first), rel=1e-12
        )

    def test_the_last_hazard_extends_beyond_the_last_pillar(self) -> None:
        curve = SurvivalCurve.from_hazards(
            REFERENCE,
            [(date(2027, 6, 15), 0.01), (date(2031, 6, 15), 0.05)],
            basis=Basis.ACT_365F,
            recovery=RECOVERY,
        )
        last = curve.times[-1]
        assert curve.hazard_at(last + 10.0) == 0.05
        assert curve.integrated_hazard(last + 3.0) == pytest.approx(
            curve.integrated_hazard(last) + 0.15, rel=1e-12
        )

    def test_the_hazard_is_right_continuous_at_a_pillar(self) -> None:
        # At a pillar the rate that governs the *next* interval is returned,
        # which is the one every integral below asks for.
        curve = SurvivalCurve.from_hazards(
            REFERENCE,
            [(date(2027, 6, 15), 0.01), (date(2031, 6, 15), 0.05)],
            basis=Basis.ACT_365F,
            recovery=RECOVERY,
        )
        boundary = curve.times[0]
        assert curve.hazard_at(boundary - 1e-6) == 0.01
        assert curve.hazard_at(boundary) == 0.05

    @pytest.mark.parametrize("recovery", [-0.01, 1.0, 1.5])
    def test_an_impossible_recovery_is_refused(self, recovery: float) -> None:
        with pytest.raises(BadCredit, match="recovery"):
            SurvivalCurve.flat(
                REFERENCE, FIVE_YEAR, 0.02, basis=Basis.ACT_365F, recovery=recovery
            )

    def test_a_negative_hazard_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="negative hazard makes survival rise"):
            SurvivalCurve.flat(
                REFERENCE, FIVE_YEAR, -0.01, basis=Basis.ACT_365F, recovery=RECOVERY
            )

    def test_an_empty_curve_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="at least one hazard pillar"):
            SurvivalCurve.from_hazards(REFERENCE, [], basis=Basis.ACT_365F, recovery=0.4)

    def test_a_pillar_on_or_before_the_reference_date_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="not after the reference date"):
            SurvivalCurve.from_hazards(
                REFERENCE, [(REFERENCE, 0.02)], basis=Basis.ACT_365F, recovery=0.4
            )

    def test_a_negative_time_is_refused(self) -> None:
        curve = flat_hazard(0.02)
        with pytest.raises(BadCredit, match="non-negative time"):
            curve.survival_at(-1.0)


class TestTheClosedFormIntegral:
    """The protection leg, against a brute-force sum that shares no algebra."""

    @staticmethod
    def brute(curve: SurvivalCurve, curve_discount: object, end: float, steps: int) -> float:
        # A midpoint rule on integral DF(s) h(s) S(s) ds. Crude, convergent,
        # and written from the definition rather than from the closed form.
        width = end / steps
        total = 0.0
        for i in range(steps):
            middle = (i + 0.5) * width
            total += (
                curve_discount.discount_at(middle)  # type: ignore[attr-defined]
                * curve.hazard_at(middle)
                * curve.survival_at(middle)
                * width
            )
        return total

    @pytest.mark.parametrize("hazard", [0.0, 0.01, 0.1, 0.5])
    @pytest.mark.parametrize("rate", [0.0, 0.03, 0.10])
    def test_it_matches_a_fine_midpoint_rule(self, hazard: float, rate: float) -> None:
        curve = flat_hazard(hazard)
        curve_discount = discount(rate)
        exact = _survival_integral(curve, curve_discount, 0.0, 5.0)
        assert exact == pytest.approx(
            self.brute(curve, curve_discount, 5.0, 40_000), rel=2e-8, abs=1e-12
        )

    def test_it_matches_across_a_hazard_step(self) -> None:
        curve = SurvivalCurve.from_hazards(
            REFERENCE,
            [(date(2028, 6, 15), 0.01), (date(2031, 6, 15), 0.20)],
            basis=Basis.ACT_365F,
            recovery=RECOVERY,
        )
        curve_discount = discount(0.04)
        exact = _survival_integral(curve, curve_discount, 0.0, 5.0)
        assert exact == pytest.approx(
            self.brute(curve, curve_discount, 5.0, 60_000), rel=1e-5
        )

    def test_a_zero_length_interval_integrates_to_nothing(self) -> None:
        assert _survival_integral(flat_hazard(0.05), discount(), 2.0, 2.0) == 0.0

    def test_a_zero_hazard_integrates_to_nothing(self) -> None:
        assert _survival_integral(flat_hazard(0.0), discount(), 0.0, 5.0) == 0.0

    def test_the_total_default_probability_is_recovered_at_zero_rates(self) -> None:
        # With no discounting the integral is just the default probability.
        curve = flat_hazard(0.07)
        got = _survival_integral(curve, discount(0.0), 0.0, 5.0)
        assert got == pytest.approx(1.0 - curve.survival_at(5.0), rel=1e-12)


class TestTheSwap:
    def test_a_swap_at_its_par_spread_is_worth_nothing(self) -> None:
        curve = flat_hazard(0.03)
        curve_discount = discount()
        par = swap().par_spread(curve, curve_discount)
        priced = CreditDefaultSwap(swap().schedule, Basis.ACT_365F, par)
        assert priced.value(curve, curve_discount) == pytest.approx(0.0, abs=1e-14)

    def test_the_par_spread_rises_with_the_hazard_rate(self) -> None:
        curve_discount = discount()
        spreads = [
            swap().par_spread(flat_hazard(h), curve_discount)
            for h in (0.005, 0.02, 0.08, 0.3)
        ]
        assert spreads == sorted(spreads)

    def test_the_par_spread_rises_as_recovery_falls(self) -> None:
        curve_discount = discount()
        previous = 0.0
        for recovery in (0.8, 0.5, 0.2, 0.0):
            curve = SurvivalCurve.flat(
                REFERENCE, FIVE_YEAR, 0.03, basis=Basis.ACT_365F, recovery=recovery
            )
            value = swap().par_spread(curve, curve_discount)
            assert value > previous
            previous = value

    def test_a_zero_hazard_has_a_zero_par_spread(self) -> None:
        assert swap().par_spread(flat_hazard(0.0), discount()) == 0.0

    def test_the_value_is_the_protection_leg_less_the_premium_leg(self) -> None:
        curve, curve_discount = flat_hazard(0.04), discount()
        priced = CreditDefaultSwap(
            swap().schedule, Basis.ACT_365F, 0.01, notional=5_000_000.0
        )
        legs = priced.protection_leg(curve, curve_discount) - priced.premium_leg(
            curve, curve_discount
        )
        assert priced.value(curve, curve_discount) == pytest.approx(
            5_000_000.0 * legs, rel=1e-14
        )

    def test_the_risky_annuity_is_the_premium_leg_per_unit_of_coupon(self) -> None:
        curve, curve_discount = flat_hazard(0.04), discount()
        priced = CreditDefaultSwap(swap().schedule, Basis.ACT_365F, 0.017)
        assert priced.premium_leg(curve, curve_discount) == pytest.approx(
            0.017 * priced.risky_annuity(curve, curve_discount), rel=1e-14
        )

    def test_the_risky_annuity_falls_as_the_hazard_rises(self) -> None:
        curve_discount = discount()
        annuities = [
            swap().risky_annuity(flat_hazard(h), curve_discount)
            for h in (0.0, 0.05, 0.2, 0.8)
        ]
        assert annuities == sorted(annuities, reverse=True)

    def test_a_certain_default_leaves_no_fair_coupon(self) -> None:
        # It takes an absurd hazard rate to reach: the risky annuity falls
        # like one over the hazard, so driving it under a rounding error
        # needs a default intensity of order a trillion a year. The guard is
        # a backstop against a division that would otherwise return a
        # plausible-looking enormous spread, not a path anything takes.
        assert swap().risky_annuity(flat_hazard(1e6), discount()) == pytest.approx(
            1e-6, rel=1e-3
        )
        with pytest.raises(BadCredit, match="no coupon makes this swap fair"):
            swap().par_spread(flat_hazard(1e13), discount())

    @pytest.mark.parametrize(("field", "value"), [("coupon", -0.01), ("notional", 0.0)])
    def test_an_impossible_contract_is_refused(self, field: str, value: float) -> None:
        arguments = {"coupon": 0.01, "notional": 1.0}
        arguments[field] = value
        with pytest.raises(BadCredit):
            CreditDefaultSwap(swap().schedule, Basis.ACT_365F, **arguments)  # type: ignore[arg-type]


class TestTheCreditTriangle:
    """The rule of thumb, and exactly where it is wrong."""

    def test_it_is_exact_when_there_is_no_discounting(self) -> None:
        """The result worth knowing, and it is not the one the folklore gives.

        With accrual on default and a zero interest rate the credit triangle is
        not an approximation: it is exact, for any hazard rate and any premium
        frequency. Integrating the accrual term by parts turns each period's
        contribution into the integral of the survival probability over that
        period, the discrete premium dates cancel out entirely, and the annuity
        collapses to ``(1 - S(T)) / h`` — which divides the protection leg to
        give exactly ``h (1 - R)``.
        """
        zero = discount(0.0)
        for hazard in (0.005, 0.02, 0.1, 0.4, 1.0):
            for frequency in Frequency:
                exact = swap(frequency=frequency).par_spread(flat_hazard(hazard), zero)
                assert exact == pytest.approx(
                    triangle_spread(hazard, RECOVERY), rel=1e-12
                )

    @pytest.mark.parametrize(
        ("frequency", "period"),
        [(Frequency.ANNUAL, 1.0), (Frequency.SEMI_ANNUAL, 0.5), (Frequency.QUARTERLY, 0.25)],
    )
    @pytest.mark.parametrize("rate", [0.01, 0.03, 0.10])
    def test_discounting_lifts_the_spread_by_the_rate_times_half_a_period(
        self, frequency: Frequency, period: float, rate: float
    ) -> None:
        # So the triangle's error is not about the credit at all. It is about
        # the gap between when a default happens and when the premium it
        # interrupts would have been paid.
        exact = swap(frequency=frequency).par_spread(flat_hazard(0.02), discount(rate))
        excess = exact / triangle_spread(0.02, RECOVERY) - 1.0
        assert excess == pytest.approx(rate * period / 2.0, rel=0.03)

    @pytest.mark.parametrize(
        ("frequency", "period"),
        [(Frequency.ANNUAL, 1.0), (Frequency.SEMI_ANNUAL, 0.5), (Frequency.QUARTERLY, 0.25)],
    )
    @pytest.mark.parametrize("hazard", [0.02, 0.10])
    def test_dropping_the_accrual_raises_it_by_the_hazard_times_half_a_period(
        self, frequency: Frequency, period: float, hazard: float
    ) -> None:
        # The mirror image of the last one, and the same shape. A contract
        # without accrual on default pays the protection seller less, so the
        # fair spread is higher.
        curve, curve_discount = flat_hazard(hazard), discount(0.03)
        with_it = swap(frequency=frequency).par_spread(curve, curve_discount)
        without = swap(frequency=frequency, accrual=False).par_spread(curve, curve_discount)
        assert without > with_it
        assert without / with_it - 1.0 == pytest.approx(hazard * period / 2.0, rel=0.12)

    def test_the_day_count_gap_composes_with_the_discount_one(self) -> None:
        # Market premiums accrue on actual/360 and the curve counts
        # actual/365, so a premium year is 365/360 of a hazard year and the
        # fair spread is lower in the same proportion. Against a matched
        # curve that is 1.389%, and it lands at 1.389% minus the 0.375% the
        # discounting adds back.
        curve, curve_discount = flat_hazard(0.02), discount(0.03)
        market = swap(basis=Basis.ACT_360).par_spread(curve, curve_discount)
        matched = swap(basis=Basis.ACT_365F).par_spread(curve, curve_discount)
        assert matched / market - 1.0 == pytest.approx(365.0 / 360.0 - 1.0, rel=0.02)
        triangle = triangle_spread(0.02, RECOVERY)
        assert triangle / market - 1.0 == pytest.approx(
            (365.0 / 360.0 - 1.0) - 0.03 * 0.25 / 2.0, rel=0.05
        )

    def test_the_triangle_round_trips(self) -> None:
        for spread in (0.001, 0.01, 0.05):
            assert triangle_spread(triangle_hazard(spread, 0.4), 0.4) == pytest.approx(
                spread, rel=1e-14
            )

    @pytest.mark.parametrize("recovery", [-0.1, 1.0])
    def test_the_triangle_refuses_an_impossible_recovery(self, recovery: float) -> None:
        with pytest.raises(BadCredit, match="recovery"):
            triangle_spread(0.02, recovery)
        with pytest.raises(BadCredit, match="recovery"):
            triangle_hazard(0.01, recovery)

    def test_the_triangle_refuses_a_negative_input(self) -> None:
        with pytest.raises(BadCredit, match="not negative"):
            triangle_spread(-0.01, 0.4)
        with pytest.raises(BadCredit, match="not negative"):
            triangle_hazard(-0.01, 0.4)


class TestImpliedHazard:
    @pytest.mark.parametrize("basis_points", [1, 25, 100, 300, 1000, 3000])
    def test_it_reproduces_the_spread_it_was_given(self, basis_points: int) -> None:
        spread = basis_points / 1e4
        curve_discount = discount()
        hazard = implied_hazard(
            swap(), spread, curve_discount, recovery=RECOVERY, basis=Basis.ACT_365F
        )
        assert swap().par_spread(flat_hazard(hazard), curve_discount) == pytest.approx(
            spread, rel=1e-10
        )

    def test_it_agrees_with_the_triangle_at_zero_rates(self) -> None:
        hazard = implied_hazard(
            swap(), 0.03, discount(0.0), recovery=RECOVERY, basis=Basis.ACT_365F
        )
        assert hazard == pytest.approx(triangle_hazard(0.03, RECOVERY), rel=1e-10)

    def test_a_zero_spread_implies_no_hazard(self) -> None:
        assert implied_hazard(
            swap(), 0.0, discount(), recovery=RECOVERY, basis=Basis.ACT_365F
        ) == pytest.approx(0.0, abs=1e-12)

    def test_a_negative_spread_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="not negative"):
            implied_hazard(swap(), -0.01, discount(), recovery=RECOVERY)


class TestBootstrap:
    MATURITIES = (
        date(2027, 6, 20),
        date(2029, 6, 20),
        date(2031, 6, 20),
        date(2033, 6, 20),
        date(2036, 6, 20),
    )
    SPREADS = (0.0060, 0.0090, 0.0120, 0.0150, 0.0185)

    def curve(self) -> SurvivalCurve:
        return bootstrap_hazards(
            [swap(maturity) for maturity in self.MATURITIES],
            self.SPREADS,
            discount(),
            recovery=RECOVERY,
            basis=Basis.ACT_365F,
        )

    def test_it_reprices_every_quote(self) -> None:
        # The check that matters. A bootstrap that does not reprice its own
        # inputs is the most common way a credit curve goes quietly wrong,
        # because everything downstream of it stays plausible.
        curve = self.curve()
        for maturity, quoted in zip(self.MATURITIES, self.SPREADS, strict=True):
            assert swap(maturity).par_spread(curve, discount()) == pytest.approx(
                quoted, rel=1e-10
            )

    def test_it_produces_one_hazard_rate_per_quote(self) -> None:
        curve = self.curve()
        assert len(curve.pillars) == len(self.MATURITIES)

    def test_a_pillar_sits_at_the_last_date_its_quote_touches(self) -> None:
        """Not at the maturity, which is a different date and a real bug.

        The final premium pays on the rolled maturity, which can be a day or
        three after the protection ends. A pillar at the maturity leaves that
        coupon discounted at the *next* quote's hazard rate and the curve then
        misses the quote that produced it -- by about three parts in a hundred
        million, which reads as solver noise and is not.
        """
        curve = self.curve()
        for pillar, maturity in zip(curve.pillars, self.MATURITIES, strict=True):
            last_payment = swap(maturity).schedule.payment_dates[-1]
            assert pillar.day == max(maturity, last_payment)
        # And at least one of them is genuinely later, or this proves nothing.
        assert any(
            pillar.day > maturity
            for pillar, maturity in zip(curve.pillars, self.MATURITIES, strict=True)
        )

    def test_an_upward_sloping_curve_has_rising_forward_hazards(self) -> None:
        hazards = [pillar.hazard for pillar in self.curve().pillars]
        assert hazards == sorted(hazards)
        assert all(hazard > 0.0 for hazard in hazards)

    def test_survival_falls_throughout(self) -> None:
        curve = self.curve()
        previous = 1.0
        for maturity in self.MATURITIES:
            value = curve.survival(maturity)
            assert 0.0 < value < previous
            previous = value

    def test_an_earlier_quote_is_untouched_by_a_later_one(self) -> None:
        # The defining property of a sequential bootstrap.
        short = bootstrap_hazards(
            [swap(m) for m in self.MATURITIES[:2]],
            self.SPREADS[:2],
            discount(),
            recovery=RECOVERY,
            basis=Basis.ACT_365F,
        )
        full = self.curve()
        for one, two in zip(short.pillars, full.pillars[:2], strict=True):
            assert one.hazard == pytest.approx(two.hazard, rel=1e-12)

    def test_a_flat_spread_curve_gives_a_nearly_flat_hazard_curve(self) -> None:
        curve = bootstrap_hazards(
            [swap(m) for m in self.MATURITIES],
            [0.012] * len(self.MATURITIES),
            discount(),
            recovery=RECOVERY,
            basis=Basis.ACT_365F,
        )
        hazards = [pillar.hazard for pillar in curve.pillars]
        assert max(hazards) / min(hazards) < 1.02

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(BadCredit, match="they pair up"):
            bootstrap_hazards(
                [swap(m) for m in self.MATURITIES], self.SPREADS[:2], discount(), recovery=0.4
            )

    def test_an_empty_bootstrap_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="at least one quote"):
            bootstrap_hazards([], [], discount(), recovery=0.4)

    def test_out_of_order_maturities_are_refused(self) -> None:
        with pytest.raises(BadCredit, match="quotes are bootstrapped in order"):
            bootstrap_hazards(
                [swap(self.MATURITIES[2]), swap(self.MATURITIES[0])],
                (0.012, 0.006),
                discount(),
                recovery=RECOVERY,
                basis=Basis.ACT_365F,
            )

    def test_a_negative_spread_is_refused(self) -> None:
        with pytest.raises(BadCredit, match="not negative"):
            bootstrap_hazards(
                [swap(self.MATURITIES[0])], (-0.001,), discount(), recovery=RECOVERY
            )


class TestRiskyBond:
    BOND = Bond(
        effective=date(2026, 6, 15),
        maturity=date(2031, 6, 15),
        coupon=0.06,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.THIRTY_360_BOND,
    )

    def riskless(self) -> float:
        curve_discount = discount()
        curve = flat_hazard(0.0)
        return math.fsum(
            flow.amount * curve_discount.discount_at(curve.time_to(flow.day))
            for flow in self.BOND.cashflows(REFERENCE)
        )

    def test_a_zero_hazard_gives_the_riskless_price(self) -> None:
        assert risky_bond_price(self.BOND, flat_hazard(0.0), discount()) == pytest.approx(
            self.riskless(), rel=1e-13
        )

    def test_default_risk_only_ever_lowers_the_price(self) -> None:
        riskless = self.riskless()
        previous = riskless
        for hazard in (0.005, 0.02, 0.08, 0.3):
            value = risky_bond_price(self.BOND, flat_hazard(hazard), discount())
            assert value < previous
            previous = value
        assert previous > 0.0

    def test_a_higher_recovery_is_worth_more(self) -> None:
        previous = 0.0
        for recovery in (0.0, 0.2, 0.4, 0.8):
            curve = SurvivalCurve.flat(
                REFERENCE, FIVE_YEAR, 0.05, basis=Basis.ACT_365F, recovery=recovery
            )
            value = risky_bond_price(self.BOND, curve, discount())
            assert value > previous
            previous = value

    def test_a_certain_immediate_default_is_worth_the_recovery(self) -> None:
        # In the limit of an enormous hazard rate the bond pays the recovery
        # fraction of face, now.
        curve = SurvivalCurve.flat(
            REFERENCE, FIVE_YEAR, 5_000.0, basis=Basis.ACT_365F, recovery=0.4
        )
        assert risky_bond_price(self.BOND, curve, discount()) == pytest.approx(
            0.4 * 100.0, rel=1e-3
        )

    def test_recovering_face_is_not_recovering_the_cashflows(self) -> None:
        # The distinction the docstring makes, made visible: a high-coupon
        # bond promises far more than face, so recovering a fraction of face
        # is worth much less than the same fraction of what was promised.
        curve = SurvivalCurve.flat(
            REFERENCE, FIVE_YEAR, 0.05, basis=Basis.ACT_365F, recovery=0.4
        )
        value = risky_bond_price(self.BOND, curve, discount())
        naive = 0.4 * self.riskless() + 0.6 * value
        assert naive > value


class TestTheFiguresInTheReadme:
    """Every number the README quotes about the triangle, recomputed."""

    def test_the_quarterly_discount_effect(self) -> None:
        exact = swap(frequency=Frequency.QUARTERLY).par_spread(
            flat_hazard(0.02), discount(0.03)
        )
        excess = exact / triangle_spread(0.02, RECOVERY) - 1.0
        assert 100.0 * excess == pytest.approx(0.374, abs=2e-3)
        predicted = 100.0 * 0.03 * 0.25 / 2.0
        assert predicted == pytest.approx(0.375, abs=1e-9)

    def test_the_quarterly_accrual_effect_at_a_ten_per_cent_hazard(self) -> None:
        curve, curve_discount = flat_hazard(0.10), discount(0.03)
        with_it = swap(frequency=Frequency.QUARTERLY).par_spread(curve, curve_discount)
        without = swap(frequency=Frequency.QUARTERLY, accrual=False).par_spread(
            curve, curve_discount
        )
        assert 100.0 * (without / with_it - 1.0) == pytest.approx(1.257, abs=5e-3)
        predicted = 100.0 * 0.10 * 0.25 / 2.0
        assert predicted == pytest.approx(1.250, abs=1e-9)

    def test_the_day_count_composition(self) -> None:
        curve, curve_discount = flat_hazard(0.02), discount(0.03)
        market = swap(basis=Basis.ACT_360).par_spread(curve, curve_discount)
        triangle = triangle_spread(0.02, RECOVERY)
        ratio = 100.0 * (365.0 / 360.0 - 1.0)
        assert ratio == pytest.approx(1.389, abs=1e-3)
        assert 100.0 * (triangle / market - 1.0) == pytest.approx(1.01, abs=2e-2)

    def test_the_bootstrap_reprices_to_a_fraction_of_a_basis_point(self) -> None:
        curve = TestBootstrap().curve()
        worst = max(
            abs(swap(maturity).par_spread(curve, discount()) - quoted)
            for maturity, quoted in zip(
                TestBootstrap.MATURITIES, TestBootstrap.SPREADS, strict=True
            )
        )
        assert 1e4 * worst < 1e-13

    def test_the_forward_is_half_as_much_again_as_the_average_at_ten_years(self) -> None:
        curve = TestBootstrap().curve()
        last = curve.pillars[-1]
        average = curve.integrated_hazard(last.time) / last.time
        assert last.hazard / average == pytest.approx(1.5, abs=0.2)
        triangle = triangle_hazard(TestBootstrap.SPREADS[-1], RECOVERY)
        assert 100.0 * (average / triangle - 1.0) == pytest.approx(9.0, abs=2.0)
