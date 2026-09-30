"""Deliverable bond futures.

The conversion factor is checked two ways. The closed form is what the library
computes; the check builds the *rounded* bond with this package's own schedule
generator and prices it at the notional yield, which reaches the same number
through date arithmetic and discounting rather than through algebra. That is the
only kind of check worth having here: a formula transcribed once and asserted
against itself passes whatever it says.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tenor.bond import Accrual, Bond
from tenor.calendar import Rolling
from tenor.daycount import Basis, year_fraction
from tenor.futures import (
    BadDelivery,
    BondFuture,
    cheapest_to_deliver,
    conversion_factor,
    delivery_basis,
    delivery_switch,
    futures_dv01,
    implied_repo_rate,
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


@pytest.mark.parametrize(
    "coupon", [0.0, 0.00625, 0.0125, 0.02, 0.03, 0.045, 0.06, 0.0775, 0.09, 0.125]
)
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


# -- the cash and carry ------------------------------------------------------

SETTLEMENT = date(2026, 11, 20)
CONTRACT = BondFuture(
    first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), label="USZ6"
)
REPO = 0.042


def test_net_basis_is_the_implied_repo_rate_expressed_as_a_price() -> None:
    """The identity the two conventional definitions rest on.

    ``net basis = (repo - implied repo) * financed balance``, exactly. Both
    sides are computed from the same inputs by different routes -- one adds up
    income and financing, the other divides a break-even -- so agreement to
    machine precision is a statement about the definitions rather than about
    floating point.
    """
    for coupon, maturity, price in (
        (0.0175, date(2046, 8, 15), 61.9),
        (0.0475, date(2046, 11, 15), 98.5),
        (0.0625, date(2043, 5, 15), 119.4),
        (0.0900, date(2039, 2, 15), 151.2),
    ):
        basis = delivery_basis(
            deliverable(maturity, coupon),
            CONTRACT,
            clean_price=price,
            futures_price=106.0,
            settlement=SETTLEMENT,
            repo=REPO,
        )
        assert basis.net_basis == pytest.approx(
            (basis.repo - basis.implied_repo) * basis.financed_balance, rel=1e-12
        )
        assert basis.net_basis == pytest.approx(
            basis.gross_basis - basis.carry, rel=1e-12, abs=1e-12
        )
        assert basis.net_basis == pytest.approx(
            (basis.breakeven_futures_price - 106.0) * basis.conversion_factor,
            rel=1e-12,
        )


def test_the_net_basis_is_zero_at_the_implied_repo_rate() -> None:
    """The same identity from the other end, and the definition of the rate."""
    bond = deliverable(date(2046, 11, 15), 0.0475)
    first = delivery_basis(
        bond,
        CONTRACT,
        clean_price=98.5,
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    again = delivery_basis(
        bond,
        CONTRACT,
        clean_price=98.5,
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=first.implied_repo,
    )
    assert again.net_basis == pytest.approx(0.0, abs=1e-12)
    assert again.implied_repo == pytest.approx(first.implied_repo, rel=1e-14)
    # And the implied repo rate does not depend on the repo rate assumed, which
    # the carry and the net basis both do.
    assert first.gross_basis == pytest.approx(again.gross_basis, rel=1e-14)
    assert first.carry != pytest.approx(again.carry)


def test_a_coupon_inside_the_holding_period_is_received_and_reinvested() -> None:
    """Two bonds, same everything, one paying a coupon before delivery.

    The one that pays carries better by the coupon plus the interest it earns
    from its own payment date, and finances a smaller balance because the coupon
    comes back before the money is due.
    """
    inside = deliverable(date(2046, 12, 15), 0.05)
    outside = deliverable(date(2047, 3, 15), 0.05)
    paid = delivery_basis(
        inside,
        CONTRACT,
        clean_price=100.0,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    unpaid = delivery_basis(
        outside,
        CONTRACT,
        clean_price=100.0,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert [day for day, _ in paid.interim_coupons] == [date(2026, 12, 15)]
    assert unpaid.interim_coupons == ()
    coupon = paid.interim_coupons[0][1]
    assert coupon == pytest.approx(2.5)

    # The comparison has to be against this bond's own unrelieved balance, not
    # against the other bond's. Two bonds with different coupon dates have
    # different accrued interest at settlement -- 2.16 against 0.90 here -- and
    # that difference is larger than the coupon relief, so the bond that pays a
    # coupon inside the period finances the *bigger* balance of the two. A first
    # attempt asserted the opposite and was measuring accrued interest.
    stub = year_fraction(date(2026, 12, 15), CONTRACT.delivery, Basis.ACT_360)
    assert paid.financed_balance == pytest.approx(
        paid.dirty_price * paid.holding_period - coupon * stub, rel=1e-12
    )
    assert paid.financed_balance < paid.dirty_price * paid.holding_period
    assert unpaid.financed_balance == pytest.approx(
        unpaid.dirty_price * unpaid.holding_period, rel=1e-14
    )
    assert paid.dirty_price - 100.0 > unpaid.dirty_price - 100.0 + 1.0

    # And carry moves the *other* way from the way a coupon receipt suggests,
    # which is the finding here rather than an accident of the pair chosen.
    # Coupon income over a fixed 41 days is the same 41 days of five per cent
    # whenever the coupon lands -- 0.5613 against 0.5663, the gap being the
    # day-count noise of measuring 41 days against two different period
    # lengths. So what separates the two is the balance financed: the bond
    # about to pay carries 2.158 of accrued interest into the trade against the
    # other's 0.912, and the financing on that 1.247 difference is 0.00596,
    # against the 0.00467 the received coupon earns back over its 16-day stub.
    # Being close to a coupon date is a cost, not a benefit.
    assert paid.carry < unpaid.carry
    reinvestment = REPO * coupon * stub
    financing_gap = REPO * (paid.dirty_price - unpaid.dirty_price) * paid.holding_period
    income_gap = (
        paid.accrued_at_delivery - (paid.dirty_price - 100.0) + coupon
    ) - (unpaid.accrued_at_delivery - (unpaid.dirty_price - 100.0))
    assert paid.carry - unpaid.carry == pytest.approx(
        income_gap + reinvestment - financing_gap, rel=1e-12
    )
    assert reinvestment == pytest.approx(0.004667, abs=5e-7)
    assert financing_gap == pytest.approx(0.005964, abs=5e-7)


def test_the_redemption_never_reaches_the_interim_coupons() -> None:
    """The guard, and the search for the case it rules out.

    ``Bond`` bundles the final coupon into the redemption flow, so a bond
    redeeming inside the holding period would contribute 102.5 to the coupons
    received rather than 2.5 -- an error of two orders of magnitude, arriving as
    a wildly negative net basis rather than as an exception. It cannot happen,
    because a bond that has redeemed cannot be delivered, and this is the sweep
    that looks for a counterexample: every maturity in the delivery month and
    the two months around it, at a daily step.
    """
    for offset in range(-30, 75):
        maturity = date(2026, 12, 1) + timedelta(days=offset)
        if maturity <= CONTRACT.first_delivery:
            continue
        bond = deliverable(maturity, 0.05)
        if maturity <= CONTRACT.delivery:
            with pytest.raises(BadDelivery, match="has redeemed by then"):
                delivery_basis(
                    bond,
                    CONTRACT,
                    clean_price=100.0,
                    futures_price=100.0,
                    settlement=SETTLEMENT,
                    repo=REPO,
                )
            continue
        basis = delivery_basis(
            bond,
            CONTRACT,
            clean_price=100.0,
            futures_price=100.0,
            settlement=SETTLEMENT,
            repo=REPO,
        )
        for _, amount in basis.interim_coupons:
            assert amount < 0.5 * bond.redemption


def test_carry_is_income_less_financing_and_is_signed() -> None:
    """A positive carry means the coupon beats the repo rate, which is the
    normal state of a long bond, and a negative one means it does not.

    The crossing is at a repo rate equal to the running yield, near enough: a
    five per cent bond at par carries positively against 4.2% repo and
    negatively against 6%.
    """
    bond = deliverable(date(2046, 11, 15), 0.05)
    cheap = delivery_basis(
        bond,
        CONTRACT,
        clean_price=100.0,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=0.042,
    )
    dear = delivery_basis(
        bond,
        CONTRACT,
        clean_price=100.0,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=0.060,
    )
    assert cheap.carry > 0.0 > dear.carry
    assert cheap.gross_basis == pytest.approx(dear.gross_basis, rel=1e-14)
    assert dear.net_basis > cheap.net_basis


def test_the_accrued_interest_grows_over_the_holding_period() -> None:
    bond = deliverable(date(2046, 11, 15), 0.0475)
    basis = delivery_basis(
        bond,
        CONTRACT,
        clean_price=98.5,
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert basis.dirty_price == pytest.approx(98.5 + bond.accrued(SETTLEMENT))
    assert basis.accrued_at_delivery > bond.accrued(SETTLEMENT)
    assert basis.invoice_price == pytest.approx(
        114.0 * basis.conversion_factor + basis.accrued_at_delivery
    )
    assert basis.holding_period == pytest.approx(41 / 360.0, rel=1e-14)


def test_the_implied_repo_rate_is_solved_by_division_and_agrees_with_a_search() -> None:
    """The closed form against a bisection on the break-even it comes from.

    Worth doing once: the formula is a rearrangement, and a rearrangement is
    exactly the kind of step that can be done backwards without any test
    noticing. The bisection knows only the cash flows.
    """
    bond = deliverable(date(2046, 12, 15), 0.05)
    basis = delivery_basis(
        bond,
        CONTRACT,
        clean_price=100.0,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    stubs = [
        (year_fraction(day, CONTRACT.delivery, Basis.ACT_360), amount)
        for day, amount in basis.interim_coupons
    ]

    def shortfall(rate: float) -> float:
        cost = basis.dirty_price * (1.0 + rate * basis.holding_period)
        proceeds = basis.invoice_price + sum(
            amount * (1.0 + rate * stub) for stub, amount in stubs
        )
        return proceeds - cost

    low, high = -5.0, 5.0
    assert shortfall(low) * shortfall(high) < 0.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if shortfall(low) * shortfall(middle) <= 0.0:
            high = middle
        else:
            low = middle
    assert basis.implied_repo == pytest.approx(0.5 * (low + high), rel=1e-11)


def test_the_arithmetic_refuses_inputs_that_cannot_mean_anything() -> None:
    bond = deliverable(date(2046, 11, 15), 0.0475)
    with pytest.raises(BadDelivery, match="not positive, so the invoice"):
        delivery_basis(
            bond,
            CONTRACT,
            clean_price=98.5,
            futures_price=0.0,
            settlement=SETTLEMENT,
            repo=REPO,
        )
    with pytest.raises(BadDelivery, match="not before delivery"):
        delivery_basis(
            bond,
            CONTRACT,
            clean_price=98.5,
            futures_price=114.0,
            settlement=date(2027, 1, 5),
            repo=REPO,
        )
    with pytest.raises(BadDelivery, match="nothing to finance"):
        implied_repo_rate(
            dirty_price=0.0,
            invoice_price=100.0,
            interim_coupons=(),
            holding_period=0.1,
        )
    with pytest.raises(BadDelivery, match="not positive; delivery must"):
        implied_repo_rate(
            dirty_price=100.0,
            invoice_price=100.0,
            interim_coupons=(),
            holding_period=0.0,
        )


def test_a_financed_balance_of_nothing_is_refused_relatively() -> None:
    """The guard is relative to the position, not against zero.

    The balance vanishes only if the interim coupons weighted by their own
    stubs match the whole dirty price weighted by the holding period, which
    needs a coupon of the order of the price. A guard at an absolute tolerance
    would either never fire or would fire on a legitimate short holding period,
    where the balance is small in absolute terms and perfectly well determined.
    """
    with pytest.raises(BadDelivery, match="carry the whole holding cost"):
        implied_repo_rate(
            dirty_price=100.0,
            invoice_price=100.0,
            interim_coupons=((0.5, 20.0),),
            holding_period=0.1,
        )
    # A balance of a ten-thousandth of a point is small and is not degenerate.
    tiny = implied_repo_rate(
        dirty_price=100.0,
        invoice_price=100.02,
        interim_coupons=(),
        holding_period=1e-6,
    )
    assert tiny == pytest.approx(0.02 / (100.0 * 1e-6), rel=1e-12)


def test_the_contract_method_and_the_function_are_the_same_call() -> None:
    bond = deliverable(date(2046, 11, 15), 0.0475)
    through_method = CONTRACT.basis_of(
        bond,
        clean_price=98.5,
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    through_function = delivery_basis(
        bond,
        CONTRACT,
        clean_price=98.5,
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert through_method == through_function


# -- the cheapest to deliver -------------------------------------------------

#: Four long bonds with similar maturities and very different coupons, so the
#: conversion factors span a ratio of two while the durations barely differ.
#: That is the basket in which the three criteria have the best chance of
#: disagreeing, which is why it is the one they are tested on.
LONG_BASKET = (
    (date(2046, 8, 15), 0.0175, "1.750% of 2046"),
    (date(2046, 11, 15), 0.0300, "3.000% of 2046"),
    (date(2047, 2, 15), 0.0450, "4.500% of 2047"),
    (date(2047, 5, 15), 0.0625, "6.250% of 2047"),
)


def long_basket() -> tuple[list[Bond], list[float]]:
    """The basket, priced off one flat yield so nothing is arbitrarily rich."""
    bonds = [deliverable(maturity, coupon, label=label) for maturity, coupon, label in LONG_BASKET]
    return bonds, [bond.clean_price(0.0455, SETTLEMENT) for bond in bonds]


def basket_implied_price(bonds: list[Bond], prices: list[float]) -> float:
    probe = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=100.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    return probe.implied_futures_price


def test_at_the_basket_implied_price_the_cheapest_bond_has_no_net_basis() -> None:
    """Which is what makes that price the one the contract should trade at.

    Every other deliverable is dearer, so its net basis is positive and its
    implied repo rate is below the financing rate assumed. That is the whole
    content of "cheapest to deliver": one bond breaks even and the rest lose.
    """
    bonds, prices = long_basket()
    fair = basket_implied_price(bonds, prices)
    ranked = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=fair,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert ranked.unanimous
    cheapest = ranked.by_futures_price
    assert cheapest.net_basis == pytest.approx(0.0, abs=1e-10)
    assert cheapest.implied_repo == pytest.approx(REPO, abs=1e-12)
    for entry in ranked.basis:
        if entry.bond.name == cheapest.bond.name:
            continue
        assert entry.net_basis > 0.0
        assert entry.implied_repo < REPO


def test_the_financed_balance_is_nearly_proportional_to_the_conversion_factor() -> None:
    """The measurement that decides which criteria can disagree.

    The natural expectation is that the implied repo rate and the net basis part
    company, because a deep-discount long bond finances around half the balance
    of a high-coupon short one. The factors here do span a ratio of 2.00. But
    the ratio that converts a rate into a price is the factor *over* the
    balance, and the balance is nearly the dirty price, and the dirty price is
    nearly the factor times the futures price -- so the factor cancels and the
    ratio spans 4.65%. The rate and the price rank a basket alike; it is the net
    basis, a price per 100 of the bond's own face, that is on its own scale.
    """
    bonds, prices = long_basket()
    ranked = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=basket_implied_price(bonds, prices),
        settlement=SETTLEMENT,
        repo=REPO,
    )
    factors = [entry.conversion_factor for entry in ranked.basis]
    ratios = [entry.conversion_factor / entry.financed_balance for entry in ranked.basis]
    assert max(factors) / min(factors) == pytest.approx(1.997, abs=0.002)
    assert max(ratios) / min(ratios) == pytest.approx(1.0465, abs=0.0005)


def test_the_quote_must_be_five_points_from_fair_before_the_rankings_split() -> None:
    """Quantifying how safe the wrong unit is.

    Walking the quote down from the basket-implied price, the net-basis ranking
    holds with the other two for 5.00 points -- 4.20% -- and then picks the
    4.500% of 2047 while both other criteria stay with the 6.250%. By then every
    deliverable shows several points of arbitrage, so the disagreement is real
    and not a situation anybody trades in. Comparing raw net bases across a
    basket is comparing points per 100 of different bonds' face, and it gets
    away with it because a liquid contract does not trade five points from fair.
    """
    bonds, prices = long_basket()
    fair = basket_implied_price(bonds, prices)

    def verdicts(price: float) -> tuple[str, str, str]:
        ranked = cheapest_to_deliver(
            bonds,
            CONTRACT,
            clean_prices=prices,
            futures_price=price,
            settlement=SETTLEMENT,
            repo=REPO,
        )
        return (
            ranked.by_implied_repo.bond.name,
            ranked.by_net_basis.bond.name,
            ranked.by_futures_price.bond.name,
        )

    agreed = verdicts(fair)
    assert len(set(agreed)) == 1
    # The split is at exactly 5.00 points, so the walk has to reach past it: a
    # bound of 500 steps stops one hundredth short and finds nothing.
    for step in range(1, 900):
        if verdicts(fair - 0.01 * step) != agreed:
            gap = 0.01 * step
            break
    else:  # pragma: no cover - the split is at five points, well inside the walk
        raise AssertionError("no split found within five points of fair value")
    assert gap == pytest.approx(5.00, abs=0.01)
    assert 100.0 * gap / fair == pytest.approx(4.20, abs=0.01)
    split = verdicts(fair - gap)
    assert split == ("6.250% of 2047", "4.500% of 2047", "6.250% of 2047")


def test_dividing_the_net_basis_by_the_factor_recovers_the_price_criterion() -> None:
    """Exactly, not approximately -- so the odd unit has an exact remedy.

    ``net basis / factor = break-even futures price - quote``, and the quote is
    common to the basket, so ranking on the left is ranking on the right. Tested
    at the quote where the raw net basis disagrees, because that is the only
    place the claim has any content.
    """
    bonds, prices = long_basket()
    quote = basket_implied_price(bonds, prices) - 5.0
    ranked = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=quote,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    for entry in ranked.basis:
        assert entry.net_basis / entry.conversion_factor == pytest.approx(
            entry.breakeven_futures_price - quote, rel=1e-12
        )
    by_scaled = min(
        ranked.basis, key=lambda one: one.net_basis / one.conversion_factor
    )
    assert by_scaled.bond.name == ranked.by_futures_price.bond.name
    assert by_scaled.bond.name != ranked.by_net_basis.bond.name


def test_the_break_even_price_does_not_depend_on_the_quote() -> None:
    """Which is what makes it the criterion available without one."""
    bonds, prices = long_basket()
    first = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=90.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    second = cheapest_to_deliver(
        bonds,
        CONTRACT,
        clean_prices=prices,
        futures_price=130.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert [one.breakeven_futures_price for one in first.basis] == pytest.approx(
        [one.breakeven_futures_price for one in second.basis], rel=1e-14
    )
    assert first.by_futures_price.bond.name == second.by_futures_price.bond.name


def test_a_basket_of_one_is_unanimous_and_is_its_own_cheapest() -> None:
    bond = deliverable(date(2046, 11, 15), 0.0475, label="only one")
    ranked = cheapest_to_deliver(
        [bond],
        CONTRACT,
        clean_prices=[98.5],
        futures_price=114.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    assert ranked.unanimous
    assert ranked.by_net_basis.bond.name == "only one"
    assert ranked.implied_futures_price == ranked.basis[0].breakeven_futures_price


def test_a_basket_and_its_quotes_have_to_line_up() -> None:
    bonds, prices = long_basket()
    with pytest.raises(BadDelivery, match="have to line up"):
        cheapest_to_deliver(
            bonds,
            CONTRACT,
            clean_prices=prices[:-1],
            futures_price=118.0,
            settlement=SETTLEMENT,
            repo=REPO,
        )
    with pytest.raises(BadDelivery, match="empty basket"):
        cheapest_to_deliver(
            [],
            CONTRACT,
            clean_prices=[],
            futures_price=118.0,
            settlement=SETTLEMENT,
            repo=REPO,
        )


# -- the switch --------------------------------------------------------------


def switch_yield(basket: list[Bond], future: BondFuture, repo: float) -> float:
    """Bisect for the yield at which the cheapest bond changes."""
    low, high = 0.01, 0.15

    def cheapest(level: float) -> str:
        return delivery_switch(
            basket, future, settlement=SETTLEMENT, repo=repo, yields=[level]
        )[0].cheapest

    below = cheapest(low)
    assert cheapest(high) != below, "the basket does not switch over this range"
    for _ in range(200):
        middle = 0.5 * (low + high)
        if cheapest(middle) == below:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def whole_year_pair() -> list[Bond]:
    """Two notional-coupon bonds, both maturing a whole number of years out.

    Whole years from the first delivery day make the rounding convention a
    no-op, so each conversion factor is the exact price of the actual bond. It
    also forces both coupon schedules onto the same months, so accrued interest,
    dirty price and therefore carry per unit of factor are common to the pair.
    Those are the conditions under which the switch should be exactly at the
    notional coupon, and separating them from the general case is the point of
    having this pair alongside the odd-maturity one.
    """
    return [
        deliverable(date(2032, 12, 1), 0.06, label="short"),
        deliverable(date(2056, 12, 1), 0.06, label="long"),
    ]


@pytest.mark.parametrize("repo", [0.0, 0.042, 0.09])
@pytest.mark.parametrize("digits", [None, 4])
def test_the_switch_is_exactly_at_the_notional_coupon_when_the_factors_are_exact(
    repo: float, digits: int | None
) -> None:
    """Six decimal places, and neither carry nor the published rounding moves it.

    The conversion factors were computed at the notional coupon, so at that
    yield every bond's price over its factor is the same number and the basket
    is exactly indifferent. Above it the longest bond is cheapest and below it
    the shortest, because price over factor falls fastest where there is most
    duration. The exchange did not choose where the switch is; the notional
    coupon did.
    """
    future = BondFuture(
        first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), digits=digits
    )
    assert switch_yield(whole_year_pair(), future, repo) == pytest.approx(
        0.06, abs=1e-8
    )


def test_below_the_notional_coupon_the_short_bond_is_cheapest_and_above_it_the_long() -> None:
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    levels = delivery_switch(
        whole_year_pair(),
        future,
        settlement=SETTLEMENT,
        repo=REPO,
        yields=[0.02, 0.04, 0.055, 0.065, 0.08, 0.10],
    )
    assert [one.cheapest for one in levels] == [
        "short",
        "short",
        "short",
        "long",
        "long",
        "long",
    ]
    # The margin is what the short's choice is worth conditional on landing
    # there, and it widens with distance from the notional coupon in both
    # directions.
    below = [one.margin for one in levels[:3]]
    above = [one.margin for one in levels[3:]]
    assert below == sorted(below, reverse=True)
    assert above == sorted(above)
    assert min(below + above) > 0.0


def test_at_the_notional_coupon_the_basket_is_indifferent() -> None:
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    level = delivery_switch(
        whole_year_pair(), future, settlement=SETTLEMENT, repo=REPO, yields=[0.06]
    )[0]
    assert level.margin == pytest.approx(0.0, abs=1e-9)
    assert level.cheapest != level.runner_up


def test_the_rounding_convention_displaces_the_switch_by_a_tenth_of_a_basis_point() -> None:
    """The general case, measured rather than waved away.

    Real maturities do not land on whole years from the first delivery day, so
    the factor is the price of a bond up to a quarter shorter than the one being
    delivered and the indifference point moves. On a 2033/2056 pair it moves to
    6.0011% with exact factors and no carry. Carry and the published
    four-decimal factor then pull it back rather than further out -- to 6.0003%
    and 6.0002% -- which is a coincidence of this pair and not a rule, so the
    assertion is on the numbers and the ordering is left alone.
    """
    odd = [
        deliverable(date(2033, 2, 15), 0.06, label="short"),
        deliverable(date(2056, 11, 15), 0.06, label="long"),
    ]
    exact_factors = BondFuture(
        first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), digits=None
    )
    published = BondFuture(
        first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), digits=4
    )
    assert switch_yield(odd, exact_factors, 0.0) == pytest.approx(0.06001127, abs=5e-8)
    assert switch_yield(odd, exact_factors, REPO) == pytest.approx(0.06000313, abs=5e-8)
    assert switch_yield(odd, published, REPO) == pytest.approx(0.06000189, abs=5e-8)
    # A tenth of a basis point either way. Worth knowing it is not zero, and
    # worth knowing it is not a basis point.
    assert abs(switch_yield(odd, published, REPO) - 0.06) < 2e-6


def test_a_switch_walk_needs_a_basket() -> None:
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    with pytest.raises(BadDelivery, match="empty basket"):
        delivery_switch(
            [], future, settlement=SETTLEMENT, repo=REPO, yields=[0.05]
        )


def test_a_basket_of_one_never_switches_and_reports_itself_as_runner_up() -> None:
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    level = delivery_switch(
        [deliverable(date(2046, 11, 15), 0.0475, label="alone")],
        future,
        settlement=SETTLEMENT,
        repo=REPO,
        yields=[0.045],
    )[0]
    assert level.cheapest == level.runner_up == "alone"
    assert level.margin == 0.0


def test_the_contract_dv01_is_the_bond_dv01_over_the_factor() -> None:
    """And a factor below one *raises* it, which is the wrong way round from
    the intuition that a conversion factor scales risk down.

    The invoice divides by the factor, so a bond whose factor is 0.5153 moves
    the futures price nearly twice as far as it moves itself. Hedging a bond
    position with the contract at a ratio of one to one, on the grounds that the
    factor is less than one and therefore conservative, is under-hedging by that
    whole ratio.
    """
    future = BondFuture(first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31))
    bond = deliverable(date(2046, 8, 15), 0.0175, label="1.750% of 2046")
    basis = delivery_basis(
        bond,
        future,
        clean_price=bond.clean_price(0.0455, SETTLEMENT),
        futures_price=118.0,
        settlement=SETTLEMENT,
        repo=REPO,
    )
    contract = futures_dv01(basis, settlement=SETTLEMENT, yield_level=0.0455)
    own = bond.pv01(0.0455, SETTLEMENT)
    assert basis.conversion_factor < 1.0
    assert contract == pytest.approx(own / basis.conversion_factor, rel=1e-14)
    assert abs(contract) > abs(own) * 1.9
