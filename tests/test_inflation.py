"""Index-linked bonds: every convention, and what breaks if it is swapped.

The tests that matter here are the ones that fail when a convention is replaced
by the plausible alternative — the lagged index by the current print, the exact
Fisher inversion by the difference of two yields, the indexed invoice by a clean
price times a ratio. Each of those substitutions produces a number in the right
neighbourhood, which is why they need tests rather than inspection.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor import (
    Basis,
    Bond,
    Frequency,
    flat_curve,
)
from tenor.inflation import (
    BadIndex,
    BadLinker,
    IndexInterpolation,
    LinkedBond,
    ReferenceIndex,
    Unpublished,
    breakeven_from_prices,
    breakeven_inflation,
    implied_inflation_from_price,
    index_accretion,
    real_yield_from_breakeven,
)


def series(
    *,
    start: tuple[int, int] = (2022, 1),
    months: int = 44,
    level: float = 100.0,
    annual: float = 0.02,
) -> dict[tuple[int, int], float]:
    """A contiguous monthly index on a constant compound path."""
    values: dict[tuple[int, int], float] = {}
    current = level
    year, month = start
    for _ in range(months):
        values[(year, month)] = current
        current *= (1.0 + annual) ** (1.0 / 12.0)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return values


def index(**kwargs: object) -> ReferenceIndex:
    return ReferenceIndex(series(**kwargs))  # type: ignore[arg-type]


def linker(**kwargs: object) -> LinkedBond:
    defaults: dict[str, object] = {
        "effective": date(2022, 4, 1),
        "maturity": date(2032, 4, 1),
        "coupon": 0.015,
        "base_index": 100.5,
        "frequency": Frequency.SEMI_ANNUAL,
    }
    defaults.update(kwargs)
    return LinkedBond(**defaults)  # type: ignore[arg-type]


class TestReferenceIndexLag:
    def test_reference_months_are_three_back_and_two_back(self) -> None:
        # The September reference index interpolates between June and July, which
        # is the whole point of the lag: the September print does not exist yet
        # when September settles.
        first, second = index().reference_months(date(2025, 9, 15))
        assert first == (2025, 6)
        assert second == (2025, 7)

    def test_january_wraps_the_year_backwards(self) -> None:
        first, second = index().reference_months(date(2025, 1, 20))
        assert first == (2024, 10)
        assert second == (2024, 11)

    def test_the_first_of_the_month_is_exactly_the_earlier_figure(self) -> None:
        # A settlement on the 1st needs nothing from the second month, and asking
        # for it would refuse a date that is fully determined.
        values = series()
        reference = ReferenceIndex(values)
        assert reference.reference(date(2025, 8, 1)) == values[(2025, 5)]

    def test_interpolation_is_linear_in_day_of_month(self) -> None:
        values = series()
        reference = ReferenceIndex(values)
        low = values[(2025, 5)]
        high = values[(2025, 6)]
        # 16 August: 15 days elapsed of a 31-day month.
        expected = low + 15.0 / 31.0 * (high - low)
        assert reference.reference(date(2025, 8, 16)) == pytest.approx(expected)

    def test_the_month_length_is_the_settlement_month_not_the_index_month(
        self,
    ) -> None:
        # The convention divides by the length of the *settlement* month, so the
        # 15th of February is 14/28ths through its step and the 15th of March is
        # 14/31sts through its own. Using the index month's length instead is a
        # plausible reading and a different number.
        values = series(months=60)
        reference = ReferenceIndex(values)
        february = reference.reference(date(2025, 2, 15))
        low = values[(2024, 11)]
        high = values[(2024, 12)]
        assert february == pytest.approx(low + 14.0 / 28.0 * (high - low))
        march = reference.reference(date(2025, 3, 15))
        low = values[(2024, 12)]
        high = values[(2025, 1)]
        assert march == pytest.approx(low + 14.0 / 31.0 * (high - low))

    def test_monthly_interpolation_steps_once_a_month(self) -> None:
        values = series()
        stepped = ReferenceIndex(values, interpolation=IndexInterpolation.MONTHLY)
        for day in (1, 9, 17, 28):
            assert stepped.reference(date(2025, 8, day)) == values[(2025, 5)]
        assert stepped.reference(date(2025, 9, 1)) == values[(2025, 6)]

    def test_a_longer_lag_reaches_further_back(self) -> None:
        eight = ReferenceIndex(series(months=60), lag_months=8)
        first, _ = eight.reference_months(date(2025, 9, 15))
        assert first == (2025, 1)

    def test_last_known_date_is_the_lag_past_the_last_print(self) -> None:
        # A series through August 2025 with a three-month lag determines the
        # reference index up to 1 December 2025 and no further, because 2 December
        # would interpolate towards September, which has not been published.
        reference = index()
        assert reference.last_month == (2025, 8)
        assert reference.last_known_date() == date(2025, 11, 1)
        assert reference.is_known(date(2025, 11, 1))
        assert not reference.is_known(date(2025, 11, 2))


class TestReferenceIndexRefusals:
    def test_an_empty_series(self) -> None:
        with pytest.raises(BadIndex, match="no figures"):
            ReferenceIndex({})

    def test_a_gap_is_refused_rather_than_interpolated_across(self) -> None:
        values = series(months=12)
        del values[(2022, 6)]
        with pytest.raises(BadIndex, match="2022-06 is missing"):
            ReferenceIndex(values)

    def test_a_non_positive_level(self) -> None:
        values = series(months=6)
        values[(2022, 3)] = 0.0
        with pytest.raises(BadIndex, match="has to be positive"):
            ReferenceIndex(values)

    def test_a_non_finite_level(self) -> None:
        values = series(months=6)
        values[(2022, 3)] = float("nan")
        with pytest.raises(BadIndex, match="has to be positive"):
            ReferenceIndex(values)

    def test_a_negative_lag(self) -> None:
        with pytest.raises(BadIndex, match="cannot look forward"):
            ReferenceIndex(series(months=6), lag_months=-1)

    def test_an_unpublished_month_names_the_range(self) -> None:
        with pytest.raises(Unpublished, match="2022-01 to 2025-08"):
            index().reference(date(2026, 6, 15))

    def test_a_zero_base_index(self) -> None:
        with pytest.raises(BadIndex, match="base index"):
            index().ratio(date(2025, 9, 15), 0.0)

    def test_realised_inflation_needs_an_interval(self) -> None:
        with pytest.raises(BadIndex, match="not after"):
            index().realised_inflation(date(2025, 9, 15), date(2025, 9, 15))


class TestRealisedAndProjected:
    def test_realised_inflation_recovers_the_generating_rate(self) -> None:
        # The series compounds at exactly 2% a year, and the reference index is a
        # lagged, interpolated version of it — so the measured rate over two
        # years should come back to 2% to within the interpolation error.
        measured = index().realised_inflation(date(2023, 6, 1), date(2025, 6, 1))
        assert measured == pytest.approx(0.02, abs=2e-4)

    def test_projection_is_the_published_value_where_one_exists(self) -> None:
        reference = index()
        day = date(2025, 9, 15)
        assert reference.projected(day, rate=0.05) == reference.reference(day)

    def test_projection_grows_from_the_last_known_date(self) -> None:
        reference = index()
        anchor = reference.last_known_date()
        target = date(2027, 5, 1)
        years = (target - anchor).days / 365.0
        expected = reference.reference(anchor) * (1.03**years)
        assert reference.projected(target, rate=0.03) == pytest.approx(expected)

    def test_a_higher_projection_gives_a_higher_level(self) -> None:
        reference = index()
        target = date(2028, 1, 15)
        assert reference.projected(target, rate=0.04) > reference.projected(
            target, rate=0.01
        )

    def test_projecting_backwards_is_refused(self) -> None:
        reference = index()
        with pytest.raises(BadIndex, match="before the projection anchor"):
            reference.projected(
                date(2021, 1, 15), rate=0.02, anchor=date(2025, 11, 1)
            )


class TestIndexRatioAndSettlement:
    def test_the_ratio_is_the_reference_over_the_base(self) -> None:
        reference = index()
        bond = linker(base_index=100.0)
        settlement = date(2025, 9, 15)
        assert bond.index_ratio(reference, settlement) == pytest.approx(
            reference.reference(settlement) / 100.0
        )

    def test_the_invoice_indexes_the_dirty_price_not_the_clean_one(self) -> None:
        # Indexing the clean price and adding unindexed accrued is the mistake
        # this asserts against. It is small on a new issue — the ratio is near
        # one — and grows with cumulative inflation, so it survives testing on
        # recent paper.
        reference = index()
        bond = linker()
        settlement = date(2025, 9, 15)
        clean = bond.real_clean_price(0.010, settlement)
        accrued = bond.real_accrued(settlement)
        ratio = bond.index_ratio(reference, settlement)
        assert bond.settlement_amount(clean, reference, settlement) == pytest.approx(
            (clean + accrued) * ratio
        )
        wrong = clean * ratio + accrued
        assert abs(
            bond.settlement_amount(clean, reference, settlement) - wrong
        ) > 1e-4

    def test_at_issue_the_ratio_is_one_and_the_spaces_coincide(self) -> None:
        reference = index()
        settlement = date(2025, 9, 15)
        base = reference.reference(settlement)
        bond = linker(base_index=base)
        assert bond.index_ratio(reference, settlement) == pytest.approx(1.0)
        clean = bond.real_clean_price(0.010, settlement)
        assert bond.settlement_amount(clean, reference, settlement) == pytest.approx(
            clean + bond.real_accrued(settlement)
        )


class TestRealSpace:
    def test_real_price_and_yield_round_trip(self) -> None:
        bond = linker()
        settlement = date(2025, 9, 15)
        clean = bond.real_clean_price(0.0125, settlement)
        root = bond.real_yield_from_clean(clean, settlement)
        assert root.converged
        assert root.value == pytest.approx(0.0125, abs=1e-10)

    def test_real_price_needs_no_index_at_all(self) -> None:
        # A real yield discounts real flows, so nothing about the price in real
        # terms depends on the index series. Worth a test because it is the reason
        # the quoted convention is real in the first place.
        bond = linker()
        settlement = date(2025, 9, 15)
        assert bond.real_clean_price(0.01, settlement) > 0.0
        assert bond.real_modified_duration(0.01, settlement) > 0.0
        assert bond.real_convexity(0.01, settlement) > 0.0

    def test_a_par_real_yield_prices_at_par_on_a_coupon_date(self) -> None:
        bond = linker(coupon=0.02)
        assert bond.real_clean_price(0.02, date(2026, 4, 1)) == pytest.approx(
            100.0, abs=1e-9
        )

    def test_real_duration_is_shorter_for_a_higher_coupon(self) -> None:
        settlement = date(2025, 9, 15)
        low = linker(coupon=0.005).real_modified_duration(0.015, settlement)
        high = linker(coupon=0.04).real_modified_duration(0.015, settlement)
        assert high < low

    def test_periodic_coupon_is_the_real_rate_over_the_frequency(self) -> None:
        assert linker(coupon=0.015).periodic_coupon == pytest.approx(0.75)


class TestNominalCashflows:
    def test_a_flow_past_the_history_needs_a_projection(self) -> None:
        bond = linker()
        with pytest.raises(Unpublished, match="pass projection="):
            bond.nominal_cashflows(index(), date(2025, 9, 15))

    def test_near_flows_are_published_and_far_ones_are_not(self) -> None:
        flows = linker().nominal_cashflows(
            index(), date(2025, 9, 15), projection=0.02
        )
        assert not flows[0].projected
        assert flows[-1].projected
        assert flows[-1].redemption

    def test_nominal_amount_is_the_real_one_times_the_ratio(self) -> None:
        flows = linker().nominal_cashflows(
            index(), date(2025, 9, 15), projection=0.02
        )
        for flow in flows:
            assert flow.nominal_amount == pytest.approx(
                flow.real_amount * flow.index_ratio
            )

    def test_a_higher_projection_raises_every_projected_flow(self) -> None:
        settlement = date(2025, 9, 15)
        low = linker().nominal_cashflows(index(), settlement, projection=0.01)
        high = linker().nominal_cashflows(index(), settlement, projection=0.04)
        for cheap, dear in zip(low, high, strict=True):
            if dear.projected:
                assert dear.nominal_amount > cheap.nominal_amount
            else:
                assert dear.nominal_amount == pytest.approx(cheap.nominal_amount)

    def test_every_amount_is_finite(self) -> None:
        for flow in linker().nominal_cashflows(
            index(), date(2025, 9, 15), projection=0.02
        ):
            assert math.isfinite(flow.nominal_amount)
            assert math.isfinite(flow.index_ratio)


class TestDeflationFloor:
    def _deflating(self) -> ReferenceIndex:
        return ReferenceIndex(series(months=44, annual=-0.03))

    def test_the_floor_binds_and_says_so(self) -> None:
        bond = linker(maturity=date(2025, 10, 1), base_index=100.0)
        ratio, floored = bond.redemption_ratio(self._deflating())
        assert floored
        assert ratio == 1.0

    def test_without_the_floor_the_principal_shrinks(self) -> None:
        bond = linker(
            maturity=date(2025, 10, 1), base_index=100.0, deflation_floor=False
        )
        ratio, floored = bond.redemption_ratio(self._deflating())
        assert not floored
        assert ratio < 1.0

    def test_the_final_coupon_and_the_principal_are_separate_flows(self) -> None:
        # The fixed-coupon bond bundles the last coupon into the redemption
        # payment. Indexing that bundle with a floored ratio floors the last
        # coupon too, which is a different and more valuable bond — and it can
        # only show up on a bond in cumulative deflation, so nothing else here
        # would have caught it.
        deflating = self._deflating()
        bond = linker(maturity=date(2025, 10, 1), base_index=100.0, coupon=0.02)
        flows = bond.nominal_cashflows(deflating, date(2025, 6, 15))
        at_maturity = [flow for flow in flows if flow.day == date(2025, 10, 1)]
        assert len(at_maturity) == 2
        coupon = next(flow for flow in at_maturity if not flow.redemption)
        principal = next(flow for flow in at_maturity if flow.redemption)
        assert coupon.real_amount == pytest.approx(1.0)
        assert principal.real_amount == pytest.approx(100.0)
        assert principal.index_ratio == 1.0
        assert coupon.index_ratio < 1.0
        unfloored = linker(
            maturity=date(2025, 10, 1),
            base_index=100.0,
            coupon=0.02,
            deflation_floor=False,
        ).nominal_cashflows(deflating, date(2025, 6, 15))
        raw = next(flow for flow in unfloored if flow.redemption).index_ratio
        assert coupon.index_ratio == pytest.approx(raw)

    def test_the_floor_is_on_the_principal_and_not_the_coupons(self) -> None:
        # The distinguishing feature of the structure: a bond deep in cumulative
        # deflation redeems at par while its coupons keep getting smaller. A floor
        # applied to every flow would be a different and more valuable bond.
        deflating = self._deflating()
        bond = linker(
            maturity=date(2025, 10, 1), base_index=100.0, coupon=0.02
        )
        flows = bond.nominal_cashflows(deflating, date(2024, 6, 15))
        coupons = [flow for flow in flows if not flow.redemption]
        redemption = next(flow for flow in flows if flow.redemption)
        assert redemption.index_ratio == 1.0
        assert len(coupons) >= 2
        for coupon in coupons:
            assert coupon.index_ratio < 1.0

    def test_no_floor_in_inflation(self) -> None:
        ratio, floored = linker(base_index=100.0).redemption_ratio(
            index(), projection=0.02
        )
        assert not floored
        assert ratio > 1.0


class TestBreakeven:
    def test_the_exact_form_inverts_the_fisher_relation(self) -> None:
        breakeven = breakeven_inflation(0.045, 0.02)
        assert (1.0 + 0.02) * (1.0 + breakeven) == pytest.approx(1.045)

    def test_the_quoted_form_is_the_plain_difference(self) -> None:
        assert breakeven_inflation(0.045, 0.02, exact=False) == pytest.approx(0.025)

    def test_the_gap_between_them_is_the_cross_term(self) -> None:
        # 4.90bp at a 4.5% nominal against a 2.0% real, and 19.23bp at 9% against
        # 4% — the difference is second order in the product, so it widens with
        # the level of both and is not a rounding matter at either.
        for nominal, real, expected_bp in ((0.045, 0.02, 4.90), (0.09, 0.04, 19.23)):
            quoted = breakeven_inflation(nominal, real, exact=False)
            exact = breakeven_inflation(nominal, real)
            assert (quoted - exact) * 1e4 == pytest.approx(expected_bp, abs=0.01)

    def test_the_two_agree_when_the_real_yield_is_zero(self) -> None:
        assert breakeven_inflation(0.03, 0.0) == pytest.approx(
            breakeven_inflation(0.03, 0.0, exact=False)
        )

    def test_real_yield_from_breakeven_is_the_inverse(self) -> None:
        breakeven = breakeven_inflation(0.05, 0.018)
        assert real_yield_from_breakeven(0.05, breakeven) == pytest.approx(0.018)

    def test_a_real_yield_at_minus_one_has_no_inverse(self) -> None:
        with pytest.raises(ValueError, match="no inverse"):
            breakeven_inflation(0.03, -1.0)

    def test_inflation_at_minus_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at or below -100%"):
            real_yield_from_breakeven(0.03, -1.0)

    def test_breakeven_between_two_priced_bonds(self) -> None:
        settlement = date(2025, 9, 15)
        nominal = Bond(
            effective=date(2022, 4, 1),
            maturity=date(2032, 4, 1),
            coupon=0.04,
            frequency=Frequency.SEMI_ANNUAL,
        )
        bond = linker()
        real_clean = bond.real_clean_price(0.010, settlement)
        nominal_clean = nominal.clean_price(0.042, settlement)
        breakeven = breakeven_from_prices(
            nominal,
            bond,
            nominal_clean=nominal_clean,
            real_clean=real_clean,
            settlement=settlement,
        )
        # The two yields are 4.2% and 1.0%, so the exact breakeven is a shade
        # under the 3.2% the difference would suggest.
        assert breakeven == pytest.approx((1.042 / 1.010) - 1.0, abs=1e-6)
        assert breakeven < 0.032

    def test_a_quoted_breakeven_between_the_same_bonds_is_the_difference(
        self,
    ) -> None:
        settlement = date(2025, 9, 15)
        nominal = Bond(
            effective=date(2022, 4, 1),
            maturity=date(2032, 4, 1),
            coupon=0.04,
            frequency=Frequency.SEMI_ANNUAL,
        )
        bond = linker()
        quoted = breakeven_from_prices(
            nominal,
            bond,
            nominal_clean=nominal.clean_price(0.042, settlement),
            real_clean=bond.real_clean_price(0.010, settlement),
            settlement=settlement,
            exact=False,
        )
        assert quoted == pytest.approx(0.032, abs=1e-6)


class TestImpliedInflation:
    def test_the_implied_rate_reprices_the_bond(self) -> None:
        settlement = date(2025, 9, 15)
        reference = index()
        bond = linker()
        curve = flat_curve(
            settlement, date(2035, 1, 1), 0.042, basis=Basis.ACT_365F
        )
        price = bond.nominal_price_from_curve(
            curve, reference, settlement, projection=0.023
        )
        root = implied_inflation_from_price(
            bond, curve, reference, settlement, nominal_price=price
        )
        assert root.converged
        assert root.value == pytest.approx(0.023, abs=1e-8)

    def test_a_dearer_bond_implies_more_inflation(self) -> None:
        settlement = date(2025, 9, 15)
        reference = index()
        bond = linker()
        curve = flat_curve(
            settlement, date(2035, 1, 1), 0.042, basis=Basis.ACT_365F
        )
        base = bond.nominal_price_from_curve(
            curve, reference, settlement, projection=0.02
        )
        dearer = implied_inflation_from_price(
            bond, curve, reference, settlement, nominal_price=base * 1.05
        )
        assert dearer.value > 0.02

    def test_a_steeper_nominal_curve_lowers_the_nominal_price(self) -> None:
        settlement = date(2025, 9, 15)
        reference = index()
        bond = linker()
        cheap = flat_curve(
            settlement, date(2035, 1, 1), 0.06, basis=Basis.ACT_365F
        )
        dear = flat_curve(
            settlement, date(2035, 1, 1), 0.02, basis=Basis.ACT_365F
        )
        assert bond.nominal_price_from_curve(
            cheap, reference, settlement, projection=0.02
        ) < bond.nominal_price_from_curve(
            dear, reference, settlement, projection=0.02
        )


class TestAccretion:
    def test_the_published_share_falls_with_the_horizon(self) -> None:
        # Measured in the module: from 15 September with a series through August,
        # the reference index is known to 1 November, so about half of the next
        # quarter's accretion is arithmetic and about an eighth of the next
        # year's.
        reference = index()
        settlement = date(2025, 9, 15)
        shares = []
        for end in (date(2025, 12, 15), date(2026, 3, 15), date(2026, 9, 15)):
            split = index_accretion(
                reference, settlement, end, 100.5, projection=0.02
            )
            shares.append(split.known_fraction)
        assert shares[0] == pytest.approx(0.514, abs=0.01)
        assert shares[1] == pytest.approx(0.257, abs=0.01)
        assert shares[2] == pytest.approx(0.127, abs=0.01)
        assert shares == sorted(shares, reverse=True)

    def test_known_and_projected_compound_to_the_total(self) -> None:
        split = index_accretion(
            index(), date(2025, 9, 15), date(2027, 1, 15), 100.5, projection=0.025
        )
        assert (1.0 + split.known) * (1.0 + split.projected) == pytest.approx(
            1.0 + split.total
        )

    def test_a_horizon_inside_the_history_is_entirely_known(self) -> None:
        split = index_accretion(
            index(), date(2025, 6, 15), date(2025, 10, 1), 100.5, projection=0.02
        )
        assert split.known_fraction == pytest.approx(1.0)
        assert split.projected == pytest.approx(0.0)

    def test_the_projection_rate_does_not_move_the_known_part(self) -> None:
        low = index_accretion(
            index(), date(2025, 9, 15), date(2027, 1, 15), 100.5, projection=0.01
        )
        high = index_accretion(
            index(), date(2025, 9, 15), date(2027, 1, 15), 100.5, projection=0.05
        )
        assert low.known == pytest.approx(high.known)
        assert high.projected > low.projected

    def test_an_interval_with_no_accretion_refuses_a_share(self) -> None:
        # Zero total accretion makes the published share 0/0, and "none of it is
        # known" is a different statement from "there is nothing to know".
        flat = ReferenceIndex(series(months=44, annual=0.0))
        split = index_accretion(
            flat, date(2025, 9, 15), date(2025, 10, 15), 100.0, projection=0.0
        )
        with pytest.raises(BadIndex, match="no accretion"):
            _ = split.known_fraction

    def test_a_backwards_interval(self) -> None:
        with pytest.raises(BadIndex, match="not after"):
            index_accretion(
                index(), date(2025, 9, 15), date(2025, 9, 1), 100.0, projection=0.02
            )


class TestLinkerRefusals:
    def test_maturity_before_issue(self) -> None:
        with pytest.raises(BadLinker, match="not after"):
            linker(effective=date(2030, 1, 1), maturity=date(2025, 1, 1))

    def test_a_non_positive_base_index(self) -> None:
        with pytest.raises(BadLinker, match="every index ratio divides by it"):
            linker(base_index=0.0)

    def test_a_non_positive_redemption(self) -> None:
        with pytest.raises(BadLinker, match="not a principal"):
            linker(redemption=0.0)

    def test_the_name_falls_back_to_the_terms(self) -> None:
        assert "1.500%" in linker().name
        assert linker(label="TII 1.5 32").name == "TII 1.5 32"
