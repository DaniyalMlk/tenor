"""Tests for asset swap spreads.

The weight is carried by one test: the spread the module returns has to make
an *independently written* valuation of the package come out at zero. That
valuation is spelled out here in terms of cash flows rather than reusing the
module's rearrangement, so a sign error in the derivation cannot agree with
itself.

Everything else is a consequence that can be stated exactly. The two
conventions stand in a ratio of par to the dirty price; the par-par spread is
an affine function of the price and the Z-spread is not; a bond priced on the
curve has a spread of exactly zero; and the whole structure's value is
independent of the floating leg's frequency and day count, so a table of legs
has to show one amount of money divided by several annuities and nothing else.

The mismatch cases are the point of carrying a residual at all. A leg whose
calendar has a holiday the bond's does not, and a leg that starts forward, both
break the cancellation the textbook formula rests on, and the tests pin how
much each is worth.
"""

from __future__ import annotations

from datetime import date

import pytest

from tenor.assetswap import (
    AssetSwap,
    BadAssetSwap,
    Convention,
    FloatingLeg,
    asset_swap_spread,
    compare_spreads,
    leg_for,
    leg_sensitivity,
    par_par_spread,
    price_from_spread,
    proceeds_spread,
)
from tenor.bond import BadBond, Bond
from tenor.bootstrap import bootstrap
from tenor.calendar import calendar_from
from tenor.curve import DiscountCurve, OffCurve
from tenor.daycount import Basis
from tenor.instruments import Deposit, Swap
from tenor.schedule import Frequency
from tenor.solve import brent
from tenor.spread import discounted_with_spread, z_spread

REFERENCE = date(2021, 1, 5)


def sloped() -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2021, 7, 5), 0.002, Basis.ACT_360),
        Swap(REFERENCE, date(2023, 1, 5), 0.006),
        Swap(REFERENCE, date(2026, 1, 5), 0.015),
        Swap(REFERENCE, date(2031, 1, 6), 0.022),
        Swap(REFERENCE, date(2041, 1, 7), 0.027),
    ]
    return bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F).curve


def sample_bond(coupon: float = 0.04) -> Bond:
    return Bond(date(2019, 1, 15), date(2031, 1, 15), coupon)


def package_value(
    bond: Bond,
    dirty: float,
    curve: DiscountCurve,
    settlement: date,
    leg: FloatingLeg,
    spread: float,
    convention: Convention,
) -> float:
    """What the buyer's whole position is worth at inception, from the flows.

    Written out rather than rearranged. In both conventions the buyer ends up
    long the bond and short a swap that pays the bond's fixed flows; the
    conventions differ in what is handed over at the start and therefore in
    the notional the floating leg runs on.
    """
    base = curve.discount(settlement)

    def forward(day: date) -> float:
        return curve.discount(day) / base

    flows = bond.cashflows(settlement)
    coupons = sum(
        (flow.amount - (bond.redemption if flow.redemption else 0.0)) * forward(flow.day)
        for flow in flows
    )
    redemption = bond.redemption * forward(flows[-1].day)
    floating = forward(leg.first_accrual) - forward(leg.final_payment)
    annuity = leg.annuity(curve, settlement)

    if convention is Convention.PAR_PAR:
        notional = bond.redemption
        # Pay par, take delivery of the bond. The swap pays the coupons away
        # and brings back the index plus the spread on a notional of par; the
        # bond's own redemption is what repays the par that went out.
        return (
            -notional
            + dirty
            + notional * floating
            + spread * notional * annuity
            - coupons
        )
    notional = dirty
    # Pay the market price, take delivery of the bond. The swap pays the
    # coupons *and the redemption* away and brings back the index plus the
    # spread on the price, with the price itself returned at maturity.
    return (
        -dirty
        + dirty
        + notional * floating
        + spread * notional * annuity
        + notional * forward(flows[-1].day)
        - coupons
        - redemption
    )


class TestTheStructureIsWorthZero:
    """The only test that is not a consequence of the others."""

    @pytest.mark.parametrize("convention", list(Convention))
    @pytest.mark.parametrize("price", [70.0, 85.0, 100.0, 116.0, 130.0])
    @pytest.mark.parametrize("coupon", [0.0, 0.02, 0.04, 0.08])
    def test_independent_valuation_vanishes(
        self, convention: Convention, price: float, coupon: float
    ) -> None:
        curve, bond = sloped(), sample_bond(coupon)
        leg = leg_for(bond, REFERENCE)
        struck = asset_swap_spread(
            bond, price, curve, REFERENCE, leg=leg, convention=convention
        )
        value = package_value(
            bond,
            struck.dirty_price,
            curve,
            REFERENCE,
            leg,
            struck.spread,
            convention,
        )
        assert abs(value) < 1e-12, (convention, price, coupon, value)

    @pytest.mark.parametrize("convention", list(Convention))
    def test_a_different_spread_does_not_vanish(self, convention: Convention) -> None:
        curve, bond = sloped(), sample_bond()
        leg = leg_for(bond, REFERENCE)
        struck = asset_swap_spread(
            bond, 90.0, curve, REFERENCE, leg=leg, convention=convention
        )
        off = package_value(
            bond,
            struck.dirty_price,
            curve,
            REFERENCE,
            leg,
            struck.spread + 0.0001,
            convention,
        )
        # One basis point on the notional, over the annuity. Not a vacuous
        # comparison: the test above would pass for a solver stuck at zero.
        assert abs(off) == pytest.approx(
            0.0001 * struck.notional * struck.annuity, rel=1e-12
        )

    def test_a_forward_starting_leg_still_prices_to_zero(self) -> None:
        curve, bond = sloped(), sample_bond()
        leg = FloatingLeg(effective=date(2021, 7, 15), maturity=bond.maturity)
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE, leg=leg)
        value = package_value(
            bond,
            struck.dirty_price,
            curve,
            REFERENCE,
            leg,
            struck.spread,
            Convention.PAR_PAR,
        )
        assert abs(value) < 1e-12, value


class TestOnTheCurve:
    """A bond priced exactly on the curve earns nothing over it."""

    @pytest.mark.parametrize("convention", list(Convention))
    @pytest.mark.parametrize("coupon", [0.01, 0.04, 0.09])
    def test_spread_is_exactly_zero(self, convention: Convention, coupon: float) -> None:
        curve, bond = sloped(), sample_bond(coupon)
        clean = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
        struck = asset_swap_spread(
            bond, clean, curve, REFERENCE, convention=convention
        )
        assert abs(struck.spread) < 1e-15, struck.spread

    def test_the_i_spread_is_not(self) -> None:
        # The contrast tenor.spread draws for the Z-spread holds here too, and
        # it is the reason the asset swap is worth computing rather than
        # reading off a yield.
        curve, bond = sloped(), sample_bond()
        clean = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
        measured = compare_spreads(bond, clean, curve, REFERENCE)
        assert abs(measured.par_par) < 1e-15
        assert abs(measured.z_spread) < 1e-12
        assert measured.i_spread * 10000 == pytest.approx(-3.484, abs=0.01)


class TestTheTwoConventions:
    """One amount of money, two notionals."""

    @pytest.mark.parametrize("price", [60.0, 80.0, 100.0, 110.0, 140.0])
    def test_ratio_is_par_over_the_dirty_price(self, price: float) -> None:
        curve, bond = sloped(), sample_bond()
        par_par = asset_swap_spread(bond, price, curve, REFERENCE)
        proceeds = asset_swap_spread(
            bond, price, curve, REFERENCE, convention=Convention.PROCEEDS
        )
        assert proceeds.spread / par_par.spread == pytest.approx(
            100.0 / par_par.dirty_price, rel=1e-14
        )

    def test_the_published_figures(self) -> None:
        curve, bond = sloped(), sample_bond()
        at_eighty = compare_spreads(bond, 80.0, curve, REFERENCE)
        assert at_eighty.par_par * 10000 == pytest.approx(391.8, abs=0.1)
        assert at_eighty.proceeds * 10000 == pytest.approx(478.4, abs=0.1)
        assert at_eighty.proceeds_over_par_par == pytest.approx(1.221, abs=0.001)
        at_onetwenty = compare_spreads(bond, 120.0, curve, REFERENCE)
        assert at_onetwenty.proceeds_over_par_par == pytest.approx(0.820, abs=0.001)

    def test_the_gap_is_widest_where_the_bond_is_cheapest(self) -> None:
        curve, bond = sloped(), sample_bond()
        gaps = [
            abs(
                compare_spreads(bond, price, curve, REFERENCE).proceeds
                - compare_spreads(bond, price, curve, REFERENCE).par_par
            )
            for price in (60.0, 70.0, 80.0, 90.0)
        ]
        assert gaps == sorted(gaps, reverse=True), gaps


class TestShape:
    """Affine in the price, where the Z-spread is convex."""

    @pytest.mark.parametrize("price", [80.0, 90.0, 100.0, 110.0, 120.0])
    def test_par_par_is_affine(self, price: float) -> None:
        curve, bond = sloped(), sample_bond()
        step = 5.0
        second = (
            par_par_spread(bond, price + step, curve, REFERENCE)
            - 2 * par_par_spread(bond, price, curve, REFERENCE)
            + par_par_spread(bond, price - step, curve, REFERENCE)
        ) / step**2
        assert abs(second) < 1e-17, second

    def test_the_slope_is_minus_one_over_the_annuity(self) -> None:
        curve, bond = sloped(), sample_bond()
        struck = asset_swap_spread(bond, 100.0, curve, REFERENCE)
        slope = (
            par_par_spread(bond, 101.0, curve, REFERENCE)
            - par_par_spread(bond, 100.0, curve, REFERENCE)
        )
        assert slope == pytest.approx(-1.0 / (100.0 * struck.annuity), rel=1e-12)

    @pytest.mark.parametrize("price", [80.0, 100.0, 120.0])
    def test_the_z_spread_is_not_affine(self, price: float) -> None:
        curve, bond = sloped(), sample_bond()
        step = 5.0

        def z(at: float) -> float:
            return z_spread(bond, at, curve, REFERENCE).value

        second = (z(price + step) - 2 * z(price) + z(price - step)) / step**2
        # In basis points per squared price point, so the published figures
        # read directly. Positive everywhere: it is convex, not merely curved.
        assert second * 10000 > 0.08
        expected = {80.0: 0.2225, 100.0: 0.1353, 120.0: 0.0906}[price]
        assert second * 10000 == pytest.approx(expected, abs=0.002)


class TestAgainstTheZSpread:
    """Two measures that agree at two prices and nowhere else."""

    def test_they_cross_twice(self) -> None:
        curve, bond = sloped(), sample_bond()

        def gap(price: float) -> float:
            measured = compare_spreads(bond, price, curve, REFERENCE)
            return measured.par_par_less_z

        on_curve = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
        assert gap(80.0) < 0.0
        assert gap(110.0) > 0.0
        assert abs(gap(on_curve)) < 1e-12
        crossing = brent(gap, 100.0, 116.0, tolerance=1e-14)
        assert crossing.converged
        assert crossing.value == pytest.approx(104.661, abs=0.001)
        assert crossing.value < on_curve

    def test_the_gap_at_eighty(self) -> None:
        curve, bond = sloped(), sample_bond()
        measured = compare_spreads(bond, 80.0, curve, REFERENCE)
        assert measured.par_par_less_z * 10000 == pytest.approx(-61.5, abs=0.2)


class TestNotACreditSpread:
    """Same issuer curve, same maturity, same Z-spread, five quotes."""

    @pytest.mark.parametrize(
        ("level", "dispersion"), [(0.01, 33.7), (0.02, 65.2), (0.03, 94.8)]
    )
    def test_coupon_moves_the_quote(self, level: float, dispersion: float) -> None:
        curve = sloped()
        spreads = []
        for coupon in (0.01, 0.02, 0.04, 0.06, 0.08):
            bond = sample_bond(coupon)
            dirty = discounted_with_spread(bond, curve, REFERENCE, level)
            clean = dirty - bond.accrued(REFERENCE)
            assert z_spread(bond, clean, curve, REFERENCE).value == pytest.approx(
                level, rel=1e-9
            )
            spreads.append(par_par_spread(bond, clean, curve, REFERENCE) * 10000)
        assert spreads == sorted(spreads), spreads
        assert max(spreads) - min(spreads) == pytest.approx(dispersion, abs=0.1)

    def test_the_dispersion_grows_with_the_level(self) -> None:
        curve = sloped()

        def spread_of(level: float, coupon: float) -> float:
            bond = sample_bond(coupon)
            dirty = discounted_with_spread(bond, curve, REFERENCE, level)
            return par_par_spread(bond, dirty - bond.accrued(REFERENCE), curve, REFERENCE)

        widths = [
            spread_of(level, 0.08) - spread_of(level, 0.01)
            for level in (0.005, 0.01, 0.02, 0.03)
        ]
        assert widths == sorted(widths), widths


class TestTheLegsConvention:
    """One amount of money, several annuities."""

    CONVENTIONS = (
        (Frequency.QUARTERLY, Basis.ACT_360),
        (Frequency.SEMI_ANNUAL, Basis.ACT_360),
        (Frequency.ANNUAL, Basis.ACT_360),
        (Frequency.QUARTERLY, Basis.ACT_365F),
        (Frequency.SEMI_ANNUAL, Basis.THIRTY_360_BOND),
    )

    def test_the_money_is_the_same(self) -> None:
        curve, bond = sloped(), sample_bond()
        rows = leg_sensitivity(bond, 90.0, curve, REFERENCE, self.CONVENTIONS)
        amounts = [row.spread * 100.0 * row.annuity for row in rows]
        for amount in amounts[1:]:
            assert amount == pytest.approx(amounts[0], rel=1e-12)

    def test_the_published_table(self) -> None:
        curve, bond = sloped(), sample_bond()
        rows = leg_sensitivity(bond, 90.0, curve, REFERENCE, self.CONVENTIONS)
        quoted = [round(row.spread * 10000, 2) for row in rows]
        assert quoted == [284.25, 285.04, 286.63, 288.20, 289.13]
        assert max(quoted) - min(quoted) == pytest.approx(4.87, abs=0.01)

    def test_a_coarser_leg_quotes_wider(self) -> None:
        # Fewer, longer accruals discount later on average, so the annuity is
        # smaller and the same money divides into a bigger number.
        curve, bond = sloped(), sample_bond()
        rows = leg_sensitivity(bond, 90.0, curve, REFERENCE, self.CONVENTIONS[:3])
        assert [row.annuity for row in rows] == sorted(
            [row.annuity for row in rows], reverse=True
        )
        assert [row.spread for row in rows] == sorted([row.spread for row in rows])

    def test_the_convention_cannot_be_read_off_the_proceeds_spread_either(self) -> None:
        curve, bond = sloped(), sample_bond()
        rows = leg_sensitivity(
            bond,
            90.0,
            curve,
            REFERENCE,
            self.CONVENTIONS,
            convention=Convention.PROCEEDS,
        )
        quoted = [row.spread * 10000 for row in rows]
        assert max(quoted) - min(quoted) == pytest.approx(5.30, abs=0.02)


class TestTheResidual:
    """Where the textbook cancellation holds, and what it costs where it does not."""

    def test_zero_when_the_leg_lines_up(self) -> None:
        curve, bond = sloped(), sample_bond()
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE)
        assert abs(struck.residual) < 1e-13, struck.residual
        textbook = (struck.curve_price - struck.dirty_price) / (100.0 * struck.annuity)
        assert struck.spread == pytest.approx(textbook, rel=1e-13)

    def test_one_holiday_on_the_redemption_payment(self) -> None:
        curve, bond = sloped(), sample_bond()
        leg = leg_for(bond, REFERENCE, calendar=calendar_from("US", [date(2031, 1, 15)]))
        assert leg.final_payment == date(2031, 1, 16)
        assert bond.cashflows(REFERENCE)[-1].day == date(2031, 1, 15)
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE, leg=leg)
        assert struck.residual == pytest.approx(0.00732, abs=1e-5)
        textbook = (struck.curve_price - struck.dirty_price) / (100.0 * struck.annuity)
        assert (textbook - struck.spread) * 10000 == pytest.approx(0.079, abs=0.002)

    def test_a_leg_that_starts_six_months_forward(self) -> None:
        curve, bond = sloped(), sample_bond()
        leg = FloatingLeg(effective=date(2021, 7, 15), maturity=bond.maturity)
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE, leg=leg)
        assert struck.residual == pytest.approx(-0.1204, abs=1e-4)
        textbook = (struck.curve_price - struck.dirty_price) / (100.0 * struck.annuity)
        assert (textbook - struck.spread) * 10000 == pytest.approx(-1.374, abs=0.01)

    def test_the_conventions_ratio_breaks_too(self) -> None:
        curve, bond = sloped(), sample_bond()
        leg = leg_for(bond, REFERENCE, calendar=calendar_from("US", [date(2031, 1, 15)]))
        par_par = asset_swap_spread(bond, 90.0, curve, REFERENCE, leg=leg)
        proceeds = asset_swap_spread(
            bond, 90.0, curve, REFERENCE, leg=leg, convention=Convention.PROCEEDS
        )
        ratio = proceeds.spread / par_par.spread
        exact = 100.0 / par_par.dirty_price
        assert ratio != pytest.approx(exact, rel=1e-9)
        assert ratio == pytest.approx(exact, rel=1e-4)


class TestTheInverse:
    """A quote goes back to a price without a root find."""

    @pytest.mark.parametrize("convention", list(Convention))
    @pytest.mark.parametrize("price", [65.0, 90.0, 100.0, 125.0])
    def test_round_trip(self, convention: Convention, price: float) -> None:
        curve, bond = sloped(), sample_bond()
        spread = asset_swap_spread(
            bond, price, curve, REFERENCE, convention=convention
        ).spread
        back = price_from_spread(
            bond, spread, curve, REFERENCE, convention=convention
        )
        assert back == pytest.approx(price, abs=1e-10)

    @pytest.mark.parametrize("convention", list(Convention))
    def test_round_trip_through_a_mismatched_leg(self, convention: Convention) -> None:
        curve, bond = sloped(), sample_bond()
        leg = FloatingLeg(effective=date(2021, 4, 15), maturity=bond.maturity)
        spread = asset_swap_spread(
            bond, 95.0, curve, REFERENCE, leg=leg, convention=convention
        ).spread
        back = price_from_spread(
            bond, spread, curve, REFERENCE, leg=leg, convention=convention
        )
        assert back == pytest.approx(95.0, abs=1e-10)

    def test_dirty_in_dirty_out(self) -> None:
        curve, bond = sloped(), sample_bond()
        dirty = 95.0 + bond.accrued(REFERENCE)
        spread = asset_swap_spread(bond, dirty, curve, REFERENCE, clean=False).spread
        assert price_from_spread(
            bond, spread, curve, REFERENCE, clean=False
        ) == pytest.approx(dirty, abs=1e-10)
        assert spread == pytest.approx(
            par_par_spread(bond, 95.0, curve, REFERENCE), rel=1e-14
        )


class TestDecomposition:
    """What the quote is made of."""

    @pytest.mark.parametrize("price", [75.0, 100.0, 125.0])
    def test_the_parts_add_up(self, price: float) -> None:
        curve, bond = sloped(), sample_bond()
        struck = asset_swap_spread(bond, price, curve, REFERENCE)
        assert struck.curve_component + struck.price_component == pytest.approx(
            struck.spread, rel=1e-13
        )

    def test_a_discount_bond_is_mostly_its_coupon(self) -> None:
        curve, bond = sloped(), sample_bond()
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE)
        share = struck.curve_component / struck.spread
        assert 0.65 < share < 0.75, share

    def test_the_price_component_changes_sign_at_par(self) -> None:
        curve, bond = sloped(), sample_bond()
        accrued = bond.accrued(REFERENCE)
        at_par = asset_swap_spread(bond, 100.0 - accrued, curve, REFERENCE)
        assert at_par.price_component == pytest.approx(0.0, abs=1e-15)
        assert asset_swap_spread(bond, 90.0, curve, REFERENCE).price_component > 0.0
        assert asset_swap_spread(bond, 110.0, curve, REFERENCE).price_component < 0.0

    def test_the_convenience_wrappers_agree(self) -> None:
        curve, bond = sloped(), sample_bond()
        assert par_par_spread(bond, 90.0, curve, REFERENCE) == pytest.approx(
            asset_swap_spread(bond, 90.0, curve, REFERENCE).spread, rel=0.0
        )
        assert proceeds_spread(bond, 90.0, curve, REFERENCE) == pytest.approx(
            asset_swap_spread(
                bond, 90.0, curve, REFERENCE, convention=Convention.PROCEEDS
            ).spread,
            rel=0.0,
        )

    def test_basis_points(self) -> None:
        curve, bond = sloped(), sample_bond()
        struck = asset_swap_spread(bond, 90.0, curve, REFERENCE)
        assert struck.spread_basis_points == pytest.approx(struck.spread * 10000)
        assert isinstance(struck, AssetSwap)


class TestRefusals:
    """What the module declines to answer."""

    def test_a_leg_that_ends_before_it_starts(self) -> None:
        with pytest.raises(BadAssetSwap, match="ends before it starts"):
            FloatingLeg(effective=date(2026, 1, 5), maturity=date(2024, 1, 5))

    def test_a_negative_price(self) -> None:
        curve, bond = sloped(), sample_bond()
        with pytest.raises(BadAssetSwap, match="not something a buyer can pay for"):
            asset_swap_spread(bond, -120.0, curve, REFERENCE)

    def test_a_bond_with_nothing_left(self) -> None:
        # Deliberately not guarded here. A bond that has matured cannot report
        # accrued interest either, and the error that comes back names the
        # maturity date, which is more use than one naming the structure.
        curve = sloped()
        matured = Bond(date(2019, 1, 15), date(2020, 1, 15), 0.04)
        leg = FloatingLeg(effective=REFERENCE, maturity=date(2026, 1, 5))
        with pytest.raises(BadBond, match="matured on 2020-01-15"):
            asset_swap_spread(matured, 100.0, curve, REFERENCE, leg=leg)

    def test_a_proceeds_spread_that_no_price_supports(self) -> None:
        curve, bond = sloped(), sample_bond()
        with pytest.raises(BadAssetSwap, match="denominator"):
            price_from_spread(
                bond, -1.0, curve, REFERENCE, convention=Convention.PROCEEDS
            )

    def test_settlement_past_the_curve(self) -> None:
        curve, bond = sloped(), sample_bond()
        leg = leg_for(bond, REFERENCE)
        with pytest.raises(OffCurve):
            leg.annuity(curve, date(2099, 1, 5))
