"""Deliverable bond futures.

The conversion factor is checked two ways. The closed form is what the library
computes; the check builds the *rounded* bond with this package's own schedule
generator and prices it at the notional yield, which reaches the same number
through date arithmetic and discounting rather than through algebra. That is the
only kind of check worth having here: a formula transcribed once and asserted
against itself passes whatever it says.
"""

from __future__ import annotations

from datetime import date

import pytest

from tenor.bond import Accrual, Bond
from tenor.calendar import Rolling
from tenor.daycount import Basis
from tenor.futures import (
    BadDelivery,
    BondFuture,
    conversion_factor,
    rounded_life,
)
from tenor.schedule import Frequency, add_months

FIRST_DELIVERY = date(2026, 12, 1)


def deliverable(maturity: date, coupon: float, *, label: str = "") -> Bond:
    return Bond(
        effective=date(2010, 1, 1),
        maturity=maturity,
        coupon=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.ACT_ACT_ISDA,
        accrual=Accrual.PERIOD_FRACTION,
        label=label,
    )


def factor_by_pricing(
    bond: Bond, first_delivery: date, *, notional_coupon: float = 0.06
) -> float:
    """The conversion factor, reached by pricing the rounded bond.

    The exchange's definition in the most literal form available: take the
    remaining life the rounding convention gives, build a bond with exactly
    that life paying this bond's coupon, and ask what it is worth at the
    notional yield on the first delivery day.

    Two deliberate choices. ``Rolling.NONE``, because the conversion factor is
    arithmetic on nominal dates and has no calendar in it at all -- this is the
    one place in the package where an unadjusted schedule is the convention
    rather than a shortcut. And 30/360 with day-count accrual, under which
    every period generated at whole-month boundaries is exactly half a year, so
    the year fractions the pricing formula needs are exact and the comparison
    is against the closed form rather than against a rounding of it.
    """
    life = rounded_life(first_delivery, bond.maturity)
    effective = add_months(
        first_delivery, life.to_next_coupon - 6, keep_end_of_month=False
    )
    synthetic = Bond(
        effective=effective,
        maturity=life.maturity,
        coupon=bond.coupon,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.THIRTY_360_BOND,
        rolling=Rolling.NONE,
        accrual=Accrual.DAY_COUNT,
    )
    return synthetic.clean_price(notional_coupon, first_delivery) / 100.0


# -- the rounding convention -------------------------------------------------


@pytest.mark.parametrize(
    ("maturity", "years", "months"),
    [
        (date(2036, 12, 1), 10, 0),
        (date(2037, 2, 28), 10, 0),  # two months of odd life rounds to none
        (date(2037, 3, 1), 10, 3),
        (date(2037, 5, 31), 10, 3),  # five months rounds to three
        (date(2037, 6, 1), 10, 6),
        (date(2037, 9, 1), 10, 9),
        (date(2037, 11, 30), 10, 9),  # eleven months rounds to nine
        (date(2037, 12, 1), 11, 0),
    ],
)
def test_the_life_rounds_down_to_a_whole_quarter(
    maturity: date, years: int, months: int
) -> None:
    life = rounded_life(FIRST_DELIVERY, maturity)
    assert (life.whole_years, life.months) == (years, months)


def test_a_day_short_of_a_month_is_a_month_short() -> None:
    """The day-of-month comparison, which is the only fiddly part.

    A bond maturing on the 15th, seen from the 1st, has completed the month; one
    maturing on the 1st of the next month, seen from the 15th, has not.
    """
    assert rounded_life(date(2026, 12, 15), date(2037, 3, 1)).months == 0
    assert rounded_life(date(2026, 12, 15), date(2037, 3, 15)).months == 3
    assert rounded_life(date(2026, 12, 15), date(2037, 3, 14)).months == 0


def test_the_next_coupon_is_never_more_than_two_quarters_away() -> None:
    for months, expected in ((0, 0), (3, 3), (6, 6), (9, 3)):
        life = rounded_life(
            FIRST_DELIVERY, add_months(FIRST_DELIVERY, 120 + months, keep_end_of_month=False)
        )
        assert life.months == months
        assert life.to_next_coupon == expected
        assert life.coupons_after_next == 20 + (1 if months == 9 else 0)


def test_a_matured_bond_is_not_deliverable() -> None:
    with pytest.raises(BadDelivery, match="not deliverable"):
        rounded_life(FIRST_DELIVERY, date(2026, 11, 30))


# -- the factor itself -------------------------------------------------------


@pytest.mark.parametrize("years", [2, 5, 10, 20, 30])
def test_a_notional_coupon_bond_with_whole_years_has_a_factor_of_one(years: int) -> None:
    """The check the formula's shape is easiest to get wrong against.

    A bond paying exactly the notional coupon and maturing on a whole number of
    years from the first delivery day is, by construction, worth par at the
    notional yield. Anything other than 1.0 here means a period has been
    miscounted or accrued interest has been subtracted the wrong way round.
    """
    bond = deliverable(add_months(FIRST_DELIVERY, 12 * years, keep_end_of_month=False), 0.06)
    assert conversion_factor(bond, FIRST_DELIVERY, digits=None) == pytest.approx(
        1.0, abs=1e-15
    )


def test_a_notional_coupon_bond_with_a_quarter_of_odd_life_is_not_quite_one() -> None:
    """And the residue is the convention rather than an error.

    With three months of odd life the next coupon is half a period away, so the
    whole stream is discounted by ``1.03 ** -0.5`` and a half coupon of accrued
    interest is taken off. Those two do not cancel: the factor comes to
    0.999889, eleven hundredths of a basis point below par, and a basket of
    six per cent bonds with different stubs therefore has factors that differ
    in the fourth decimal for no reason to do with the bonds.
    """
    bond = deliverable(add_months(FIRST_DELIVERY, 123, keep_end_of_month=False), 0.06)
    assert conversion_factor(bond, FIRST_DELIVERY, digits=None) == pytest.approx(
        0.9998891565, abs=5e-11
    )


def test_at_the_notional_coupon_the_factor_ignores_the_maturity() -> None:
    """A degenerate case with a reason: the annuity and the redemption cancel.

    At ``c == y`` the bracket is ``c/2 + p + (1 - p)``, in which the discount
    factor to the rounded maturity drops out. So every six per cent bond with a
    three-month stub has the same factor whatever its maturity, and a nine-month
    stub gives the same number as a three-month one. Neither holds a basis point
    away from the notional coupon, which is why it is a test rather than a
    shortcut in the code.
    """
    three = [
        conversion_factor(
            deliverable(add_months(FIRST_DELIVERY, 12 * y + 3, keep_end_of_month=False), 0.06),
            FIRST_DELIVERY,
            digits=None,
        )
        for y in (3, 7, 19, 29)
    ]
    assert three == pytest.approx([three[0]] * 4, abs=1e-15)
    nine = conversion_factor(
        deliverable(add_months(FIRST_DELIVERY, 129, keep_end_of_month=False), 0.06),
        FIRST_DELIVERY,
        digits=None,
    )
    assert nine == pytest.approx(three[0], abs=1e-15)

    # Off the notional coupon it fails, which is the point.
    off = [
        conversion_factor(
            deliverable(add_months(FIRST_DELIVERY, 12 * y + 3, keep_end_of_month=False), 0.02),
            FIRST_DELIVERY,
            digits=None,
        )
        for y in (3, 29)
    ]
    assert off[0] - off[1] > 0.2


@pytest.mark.parametrize("coupon", [0.0, 0.00625, 0.0125, 0.02, 0.03, 0.045, 0.06, 0.0775, 0.09, 0.125])
@pytest.mark.parametrize(
    "maturity",
    [
        date(2029, 2, 15),
        date(2031, 5, 31),
        date(2036, 8, 15),
        date(2046, 11, 15),
        date(2056, 12, 1),
        date(2037, 3, 1),
        date(2037, 6, 30),
        date(2037, 9, 30),
    ],
)
def test_the_closed_form_matches_pricing_the_rounded_bond(
    coupon: float, maturity: date
) -> None:
    """Eighty cases, all four stub lengths, coupons from zero to 12.5%.

    The two routes share the notional coupon and nothing else: one is five
    lines of algebra, the other generates a schedule, walks it and discounts.
    """
    bond = deliverable(maturity, coupon)
    assert conversion_factor(bond, FIRST_DELIVERY, digits=None) == pytest.approx(
        factor_by_pricing(bond, FIRST_DELIVERY), abs=1e-12
    )


def test_a_zero_coupon_deliverable_is_the_discount_factor() -> None:
    """With no coupon the whole formula collapses to one discount factor."""
    bond = deliverable(add_months(FIRST_DELIVERY, 120, keep_end_of_month=False), 0.0)
    assert conversion_factor(bond, FIRST_DELIVERY, digits=None) == pytest.approx(
        1.03**-20, rel=1e-14
    )


def test_the_factor_rises_with_the_coupon_and_crosses_one_at_the_notional() -> None:
    maturity = date(2046, 11, 15)
    factors = [
        conversion_factor(deliverable(maturity, c), FIRST_DELIVERY, digits=None)
        for c in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
    ]
    assert factors == sorted(factors)
    assert factors[2] < 1.0 < factors[4]


def test_a_longer_bond_is_further_from_one_in_whichever_direction_it_started() -> None:
    """Duration, seen through the factor.

    A discount bond's factor falls with maturity and a premium bond's rises,
    because the factor is a price at a fixed yield and the gap between the
    bond's coupon and that yield is worth more for longer.
    """
    short = add_months(FIRST_DELIVERY, 36, keep_end_of_month=False)
    long = add_months(FIRST_DELIVERY, 360, keep_end_of_month=False)
    assert conversion_factor(deliverable(long, 0.02), FIRST_DELIVERY, digits=None) < (
        conversion_factor(deliverable(short, 0.02), FIRST_DELIVERY, digits=None)
    )
    assert conversion_factor(deliverable(long, 0.10), FIRST_DELIVERY, digits=None) > (
        conversion_factor(deliverable(short, 0.10), FIRST_DELIVERY, digits=None)
    )


def test_the_published_precision_is_the_default_and_can_be_turned_off() -> None:
    bond = deliverable(date(2046, 11, 15), 0.0325)
    exact = conversion_factor(bond, FIRST_DELIVERY, digits=None)
    assert conversion_factor(bond, FIRST_DELIVERY) == round(exact, 4)
    assert conversion_factor(bond, FIRST_DELIVERY, digits=6) == round(exact, 6)
    assert conversion_factor(bond, FIRST_DELIVERY) != exact


def test_a_different_notional_coupon_is_the_same_arithmetic() -> None:
    """A contract fixing 3% rather than 6% is one substitution, not a new model."""
    bond = deliverable(add_months(FIRST_DELIVERY, 120, keep_end_of_month=False), 0.03)
    assert conversion_factor(
        bond, FIRST_DELIVERY, notional_coupon=0.03, digits=None
    ) == pytest.approx(1.0, abs=1e-15)
    assert conversion_factor(
        bond, FIRST_DELIVERY, notional_coupon=0.03, digits=None
    ) == pytest.approx(
        factor_by_pricing(bond, FIRST_DELIVERY, notional_coupon=0.03), abs=1e-12
    )


def test_a_negative_coupon_and_a_zero_notional_are_both_refused() -> None:
    with pytest.raises(BadDelivery, match="negative coupon"):
        conversion_factor(deliverable(date(2036, 12, 1), -0.01), FIRST_DELIVERY)
    with pytest.raises(BadDelivery, match="divides by zero"):
        conversion_factor(
            deliverable(date(2036, 12, 1), 0.05), FIRST_DELIVERY, notional_coupon=0.0
        )


# -- the contract ------------------------------------------------------------


def test_the_contract_refuses_delivery_before_its_own_month() -> None:
    with pytest.raises(BadDelivery, match="before the delivery month"):
        BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 11, 30))


def test_the_invoice_is_the_factor_times_the_price_plus_accrued() -> None:
    """And the accrued interest is not scaled by the factor.

    The factor normalises the principal the contract is written on. The coupon
    the short gives up is the coupon the bond pays, so it enters the invoice
    unscaled -- which on a high-coupon bond with a factor of 1.3 is a
    difference of most of a point.
    """
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    bond = deliverable(date(2046, 11, 15), 0.0475, label="4.75% of 2046")
    factor = future.conversion_factor(bond)
    invoice = future.invoice_price(bond, 98.5)
    assert invoice == pytest.approx(98.5 * factor + bond.accrued(future.delivery))
    assert bond.accrued(future.delivery) > 0.5
    assert invoice != pytest.approx(factor * (98.5 + bond.accrued(future.delivery)))


def test_the_contract_names_itself_when_unlabelled() -> None:
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    assert future.name == "future delivering 2026-12-31"
    assert BondFuture(
        first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), label="TYZ6"
    ).name == "TYZ6"
