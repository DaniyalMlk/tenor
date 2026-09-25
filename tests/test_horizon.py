"""Carry, roll-down, and the identity that separates them.

The central claim is not a tolerance but an equality: on an arbitrage-free
curve, coupon income plus the forward price change *is* the financing cost.
Most of what follows establishes that it holds wherever it should and then
uses it to pin down what roll-down can be.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tenor import Basis, Bond, Compounding, DiscountCurve, Frequency
from tenor.curve import Interpolation, OffCurve
from tenor.horizon import (
    BadHorizon,
    HorizonReturn,
    forward_curve,
    horizon_return,
    rolled_curve,
    rolling_horizon_returns,
)

REFERENCE = date(2024, 1, 2)
HORIZON = date(2025, 1, 2)

UPWARD = [
    (date(2024, 7, 2), 0.0285),
    (date(2025, 1, 2), 0.0300),
    (date(2026, 1, 2), 0.0340),
    (date(2029, 1, 2), 0.0380),
    (date(2034, 1, 2), 0.0420),
    (date(2044, 1, 2), 0.0440),
]
INVERTED = [(day, 0.070 - rate) for day, rate in UPWARD]
FLAT = [(day, 0.040) for day, _ in UPWARD]


def curve_of(
    points: list[tuple[date, float]],
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
) -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE, points, basis=Basis.ACT_365F, interpolation=interpolation
    )


def bond_of(coupon: float, maturity: date = date(2034, 1, 2)) -> Bond:
    return Bond(
        effective=date(2024, 1, 2),
        maturity=maturity,
        coupon=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.THIRTY_360_BOND,
    )


COUPONS = [0.00, 0.01, 0.036, 0.05, 0.09]


# --------------------------------------------------------------------------
# The identity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("coupon", COUPONS)
@pytest.mark.parametrize("points", [UPWARD, INVERTED, FLAT], ids=["up", "inverted", "flat"])
def test_carry_is_the_financing_cost_exactly(
    coupon: float, points: list[tuple[date, float]]
) -> None:
    """Whatever the coupon and whatever the curve's shape.

    A premium bond, a discount bond and a zero-coupon bond reach this by
    completely different routes — the first is mostly coupon income, the last
    is entirely price appreciation — and they land on the same number because
    the curve discounts consistently, not because the cases were handled.
    """
    result = horizon_return(bond_of(coupon), curve_of(points), REFERENCE, HORIZON)
    assert result.carry == pytest.approx(result.financing_cost, abs=1e-11)
    assert result.is_arbitrage_free()


def test_the_identity_holds_across_a_year_of_settlement_dates() -> None:
    """Including settlements inside a coupon period, where accrued interest is
    non-zero and the number of coupons in the window changes."""
    curve = curve_of(UPWARD)
    bond = bond_of(0.05)
    for offset in range(0, 360, 17):
        settlement = REFERENCE + timedelta(days=offset)
        horizon = settlement + timedelta(days=180)
        result = horizon_return(bond, curve, settlement, horizon)
        assert result.is_arbitrage_free(), f"failed at {settlement}"


def test_the_identity_survives_every_interpolation() -> None:
    bond = bond_of(0.04)
    for scheme in Interpolation:
        curve = curve_of(UPWARD, scheme)
        result = horizon_return(bond, curve, REFERENCE, HORIZON)
        assert result.is_arbitrage_free(), scheme


def test_income_less_financing_is_the_start_minus_forward_price() -> None:
    """The market's other "carry", and the algebraic identity behind it."""
    result = horizon_return(bond_of(0.05), curve_of(UPWARD), REFERENCE, HORIZON)
    assert result.income_less_financing == pytest.approx(
        result.start_price - result.forward_price, abs=1e-11
    )


def test_the_excess_over_financing_is_exactly_the_roll_down() -> None:
    for coupon in COUPONS:
        result = horizon_return(bond_of(coupon), curve_of(UPWARD), REFERENCE, HORIZON)
        assert result.excess_over_financing == pytest.approx(result.roll_down, abs=1e-11)


def test_total_return_is_its_two_parts() -> None:
    result = horizon_return(bond_of(0.05), curve_of(UPWARD), REFERENCE, HORIZON)
    assert result.total_return == pytest.approx(result.carry + result.roll_down)
    assert result.total_return_bps == pytest.approx(
        1e4 * result.total_return / result.start_price
    )


# --------------------------------------------------------------------------
# Roll-down, and what the curve's slope does to it.
# --------------------------------------------------------------------------


def test_roll_down_is_zero_on_a_flat_curve() -> None:
    """Nothing to roll down. The bond ages into the same yield it had."""
    for coupon in COUPONS:
        result = horizon_return(bond_of(coupon), curve_of(FLAT), REFERENCE, HORIZON)
        assert result.roll_down == pytest.approx(0.0, abs=1e-9)


def test_roll_down_is_positive_on_an_upward_sloping_curve() -> None:
    for coupon in COUPONS:
        result = horizon_return(bond_of(coupon), curve_of(UPWARD), REFERENCE, HORIZON)
        assert result.roll_down > 0.0


def test_roll_down_is_negative_on_an_inverted_curve() -> None:
    for coupon in COUPONS:
        result = horizon_return(bond_of(coupon), curve_of(INVERTED), REFERENCE, HORIZON)
        assert result.roll_down < 0.0


def test_the_sign_of_roll_down_follows_the_slope_and_not_the_level() -> None:
    """A curve shifted up in parallel changes the financing cost and leaves
    roll-down where it was, to within the convexity of the shift."""
    bond = bond_of(0.04)
    base = horizon_return(bond, curve_of(UPWARD), REFERENCE, HORIZON)
    lifted = horizon_return(
        bond, curve_of([(day, rate + 0.01) for day, rate in UPWARD]), REFERENCE, HORIZON
    )
    assert lifted.financing_cost > base.financing_cost
    assert lifted.roll_down > 0.0
    # Both are rolling down the same 40bp of slope; the level moves the price
    # the roll is measured on, so the two differ but not in order of magnitude.
    assert 0.4 < lifted.roll_down / base.roll_down < 1.0


def test_a_steeper_curve_rolls_further() -> None:
    bond = bond_of(0.04)
    gentle = horizon_return(bond, curve_of(UPWARD), REFERENCE, HORIZON)
    steep = horizon_return(
        bond,
        curve_of([(day, 0.028 + 2.0 * (rate - 0.0285)) for day, rate in UPWARD]),
        REFERENCE,
        HORIZON,
    )
    assert steep.roll_down > gentle.roll_down


def test_income_less_financing_is_a_poor_guide_to_the_actual_return() -> None:
    """The finding that justifies reporting both definitions.

    A high-coupon and a low-coupon bond of the same maturity earn almost the
    same holding-period return, because the high coupon is paid for by a price
    falling towards par. Income minus financing says otherwise by a wide
    margin, and that is what makes it misleading rather than merely different.
    """
    curve = curve_of(UPWARD)
    premium = horizon_return(bond_of(0.09), curve, REFERENCE, HORIZON)
    discount = horizon_return(bond_of(0.00), curve, REFERENCE, HORIZON)
    spread_in_income = premium.income_less_financing - discount.income_less_financing
    spread_in_return = abs(premium.total_return_bps - discount.total_return_bps)
    assert spread_in_income > 5.0, "the two look very different on income"
    assert spread_in_return < 40.0, "and land within 40bp of each other in truth"


# --------------------------------------------------------------------------
# The two curves, which are not the same curve.
# --------------------------------------------------------------------------


def test_the_forward_curve_reprices_todays_curve_for_later_value() -> None:
    curve = curve_of(UPWARD)
    forward = forward_curve(curve, HORIZON)
    assert forward.reference == HORIZON
    for day in (date(2029, 1, 2), date(2034, 1, 2), date(2044, 1, 2)):
        assert forward.discount(day) == pytest.approx(
            curve.discount(day) / curve.discount(HORIZON), rel=1e-12
        )


def test_the_forward_curve_carries_no_information_the_original_lacked() -> None:
    """A forward rate between two dates beyond the horizon is the same on
    both curves, which is what makes the forward curve a renormalisation."""
    curve = curve_of(UPWARD)
    forward = forward_curve(curve, HORIZON)
    near, far = date(2029, 1, 2), date(2034, 1, 2)
    assert forward.forward_rate(near, far) == pytest.approx(
        curve.forward_rate(near, far), rel=1e-12
    )


def test_the_rolled_curve_keeps_the_zero_rate_at_each_tenor() -> None:
    curve = curve_of(UPWARD)
    rolled = rolled_curve(curve, HORIZON)
    assert rolled.reference == HORIZON
    offset = HORIZON - REFERENCE
    for pillar in curve.pillars:
        if pillar.is_anchor:
            continue
        original = curve.zero_rate(pillar.day, Compounding.CONTINUOUS)
        moved = rolled.zero_rate(pillar.day + offset, Compounding.CONTINUOUS)
        assert moved == pytest.approx(original, rel=1e-12)


def test_the_rolled_and_forward_curves_differ_unless_the_curve_is_flat() -> None:
    """The distinction the whole module rests on. On a flat curve they agree,
    which is exactly why roll-down is zero there."""
    sloped = curve_of(UPWARD)
    day = date(2033, 1, 2)
    assert rolled_curve(sloped, HORIZON).discount(day) != pytest.approx(
        forward_curve(sloped, HORIZON).discount(day), rel=1e-6
    )
    flat = curve_of(FLAT)
    assert rolled_curve(flat, HORIZON).discount(day) == pytest.approx(
        forward_curve(flat, HORIZON).discount(day), rel=1e-9
    )


def test_pricing_on_the_forward_curve_reproduces_the_forward_price() -> None:
    """The forward price is computed directly rather than by rebuilding a
    curve; this checks the two routes agree, which is what justifies taking
    the cheaper and exactly-consistent one."""
    curve = curve_of(UPWARD)
    bond = bond_of(0.05)
    result = horizon_return(bond, curve, REFERENCE, HORIZON)
    rebuilt = bond.price_from_curve(forward_curve(curve, HORIZON), HORIZON)
    assert rebuilt == pytest.approx(result.forward_price, rel=1e-9)


def test_a_curve_cannot_be_rolled_or_forwarded_backwards() -> None:
    curve = curve_of(UPWARD)
    earlier = REFERENCE - timedelta(days=1)
    with pytest.raises(BadHorizon, match="before"):
        rolled_curve(curve, earlier)
    with pytest.raises(BadHorizon, match="before"):
        forward_curve(curve, earlier)


def test_a_forward_curve_at_the_very_end_of_the_curve_is_refused() -> None:
    """There is a discount factor at the last pillar, so the curve itself
    allows the query; what is missing is anything left to build a curve out
    of, and that refusal belongs here."""
    curve = curve_of(UPWARD)
    with pytest.raises(BadHorizon, match="nothing left of it"):
        forward_curve(curve, curve.horizon)


def test_a_forward_curve_past_the_end_of_the_curve_is_refused_by_the_curve() -> None:
    """Past the last pillar the curve declines to extrapolate, and its own
    message is the better one — it says what to do about it."""
    curve = curve_of(UPWARD)
    with pytest.raises(OffCurve, match="past the curve's last pillar"):
        forward_curve(curve, date(2045, 1, 2))


# --------------------------------------------------------------------------
# Coupons inside the window.
# --------------------------------------------------------------------------


def test_the_coupons_in_the_window_are_the_ones_that_fall_in_it() -> None:
    result = horizon_return(bond_of(0.05), curve_of(UPWARD), REFERENCE, HORIZON)
    assert len(result.coupons) == 2
    assert all(REFERENCE < one.day <= HORIZON for one in result.coupons)
    assert all(one.amount == pytest.approx(2.5) for one in result.coupons)


def test_a_coupon_is_worth_more_at_the_horizon_than_when_it_was_paid() -> None:
    """Reinvested at the curve's own forward rate, which is the only
    assumption under which the identity holds."""
    result = horizon_return(bond_of(0.05), curve_of(UPWARD), REFERENCE, HORIZON)
    early = min(result.coupons, key=lambda one: one.day)
    assert early.value_at_horizon > early.amount
    assert result.coupon_income == pytest.approx(
        sum(one.value_at_horizon for one in result.coupons)
    )


def test_a_zero_coupon_bond_earns_nothing_but_price_and_still_balances() -> None:
    """A 0% bond issued on a semi-annual schedule still *has* payment dates;
    they pay nothing. Dropping them would be inventing a rule about which
    scheduled flows count, so they are reported with an amount of zero and
    contribute zero.
    """
    result = horizon_return(bond_of(0.0), curve_of(UPWARD), REFERENCE, HORIZON)
    assert len(result.coupons) == 2
    assert all(one.amount == 0.0 for one in result.coupons)
    assert result.coupon_income == 0.0
    assert result.is_arbitrage_free()
    # All of its carry is therefore the price rising towards par.
    assert result.forward_price_change > 0.0
    assert result.forward_price_change == pytest.approx(result.financing_cost, abs=1e-11)


def test_a_premium_bond_is_pulled_down_towards_par() -> None:
    result = horizon_return(bond_of(0.09), curve_of(UPWARD), REFERENCE, HORIZON)
    assert result.start_price > 100.0
    assert result.forward_price_change < 0.0
    assert result.coupon_income > 0.0


def test_a_horizon_shorter_than_the_first_coupon_has_no_receipts() -> None:
    result = horizon_return(
        bond_of(0.05), curve_of(UPWARD), REFERENCE, REFERENCE + timedelta(days=30)
    )
    assert result.coupons == ()
    assert result.is_arbitrage_free()


# --------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------


def test_a_horizon_at_or_before_settlement_is_refused() -> None:
    curve = curve_of(UPWARD)
    with pytest.raises(BadHorizon, match="runs forwards"):
        horizon_return(bond_of(0.05), curve, REFERENCE, REFERENCE)
    with pytest.raises(BadHorizon, match="runs forwards"):
        horizon_return(bond_of(0.05), curve, HORIZON, REFERENCE)


def test_a_horizon_past_maturity_is_refused_and_says_why() -> None:
    """Not zero, and not the redemption amount. A redeemed bond has no price
    at the horizon, so there is no price change to decompose."""
    curve = curve_of(UPWARD)
    bond = bond_of(0.05, maturity=date(2026, 1, 2))
    with pytest.raises(BadHorizon, match="no price at the"):
        horizon_return(bond, curve, REFERENCE, date(2026, 1, 2))
    with pytest.raises(BadHorizon, match="no price at the"):
        horizon_return(bond, curve, REFERENCE, date(2027, 1, 2))


def test_settling_before_the_curves_reference_is_refused() -> None:
    curve = curve_of(UPWARD)
    with pytest.raises(BadHorizon, match="before the curve"):
        horizon_return(
            bond_of(0.05), curve, REFERENCE - timedelta(days=5), HORIZON
        )


# --------------------------------------------------------------------------
# A sequence of horizons.
# --------------------------------------------------------------------------


def test_rolling_horizons_lengthen_and_each_one_balances() -> None:
    results = rolling_horizon_returns(
        bond_of(0.04), curve_of(UPWARD), REFERENCE, days=90, steps=8
    )
    assert len(results) == 8
    assert [one.horizon for one in results] == sorted(one.horizon for one in results)
    assert all(one.is_arbitrage_free() for one in results)


def test_roll_down_is_not_linear_in_the_holding_period() -> None:
    """The bond moves through the curve's shape rather than along a line, so
    doubling the horizon does not double the roll."""
    results = rolling_horizon_returns(
        bond_of(0.04), curve_of(UPWARD), REFERENCE, days=180, steps=2
    )
    one_period, two_periods = results
    assert two_periods.roll_down > one_period.roll_down
    assert two_periods.roll_down != pytest.approx(2.0 * one_period.roll_down, rel=0.02)


def test_a_sequence_stops_at_maturity_rather_than_raising() -> None:
    results = rolling_horizon_returns(
        bond_of(0.04, maturity=date(2025, 1, 2)), curve_of(UPWARD), REFERENCE,
        days=90, steps=10,
    )
    assert 0 < len(results) < 10
    assert all(one.horizon < date(2025, 1, 2) for one in results)


@pytest.mark.parametrize(("days", "steps"), [(0, 4), (-1, 4), (90, 0), (90, -2)])
def test_a_degenerate_sequence_is_refused(days: int, steps: int) -> None:
    with pytest.raises(BadHorizon):
        rolling_horizon_returns(
            bond_of(0.04), curve_of(UPWARD), REFERENCE, days=days, steps=steps
        )


def test_the_dataclass_is_frozen_so_a_reported_figure_cannot_drift() -> None:
    result = horizon_return(bond_of(0.05), curve_of(UPWARD), REFERENCE, HORIZON)
    assert isinstance(result, HorizonReturn)
    with pytest.raises(AttributeError):
        result.start_price = 0.0  # type: ignore[misc]
