"""Deliverable bond futures: conversion factors, basis and the cheapest to deliver.

:class:`~tenor.instruments.Future` in this package is a *rate* contract — one
accrual period, cash settled, no choice for either side. A deliverable bond
future is a different instrument. The short hands over an actual bond from a
published basket, chooses which one and chooses when inside the delivery month,
and the contract normalises those choices with a **conversion factor**: the
price of that bond, per unit of face, at a notional yield the exchange fixes and
the market then moves away from.

Everything interesting about the contract follows from that last clause.

**The conversion factor is a bond price, so it is computed rather than quoted.**
The exchange convention is the price of the deliverable at the notional coupon,
as of the first day of the delivery month, with the remaining life rounded down
to a whole number of quarters. :func:`conversion_factor` implements the closed
form. The rounding is the only part that is arbitrary; the rest is the street
discounting formula, and the test suite checks it against a bond built with this
package's own schedule generator and priced at the notional yield, which is a
genuinely separate code path rather than the same algebra spelled twice.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .bond import Bond
from .daycount import Basis
from .schedule import add_months

__all__ = [
    "BadDelivery",
    "BondFuture",
    "RoundedLife",
    "conversion_factor",
    "rounded_life",
]

#: The notional coupon of the contract. Six per cent for the US Treasury
#: complex, the Bund complex and the long gilt; kept a default rather than a
#: constant because a contract that fixes a different one is otherwise
#: identical arithmetic.
DEFAULT_NOTIONAL_COUPON = 0.06

#: Digits the exchange publishes the factor to. Delivery settles on the
#: published number, so the invoice uses the rounded factor and not the exact
#: one -- a distinction worth about a tenth of a tick on a long bond.
DEFAULT_DIGITS = 4


class BadDelivery(ValueError):
    """A basis calculation that cannot mean anything."""


@dataclass(frozen=True)
class RoundedLife:
    """The deliverable's remaining life, as the conversion factor sees it.

    Attributes:
        whole_years: Complete years from the first delivery day to maturity.
        months: Whole months beyond those years, rounded *down* to a quarter,
            so one of 0, 3, 6 or 9.
        maturity: The rounded maturity itself. Not used by the closed form,
            and the thing to build an independent check against.
    """

    whole_years: int
    months: int
    maturity: date

    @property
    def to_next_coupon(self) -> int:
        """``v``: months from the first delivery day to the next coupon.

        The rounded bond pays every six months counting back from its rounded
        maturity, so with nine months of odd life the next coupon is three
        months away and not nine.
        """
        return self.months if self.months < 7 else self.months - 6

    @property
    def coupons_after_next(self) -> int:
        """Whole coupon periods from that next coupon to the rounded maturity."""
        return 2 * self.whole_years + (0 if self.months < 7 else 1)


def rounded_life(first_delivery: date, maturity: date) -> RoundedLife:
    """Split the remaining life into whole years and a whole quarter.

    Rounding *down* is the convention and it is not a tie-break rule: a bond
    with four months of odd life is treated as having three, which makes its
    factor that of a slightly shorter bond. That is a real, deliberate
    distortion of a quarter of a coupon period at the front of the basket, and
    it is the reason two bonds maturing weeks apart can have factors that look
    inconsistent with each other.
    """
    if maturity <= first_delivery:
        raise BadDelivery(
            f"a bond maturing {maturity.isoformat()} is not deliverable into a "
            f"contract whose delivery month begins {first_delivery.isoformat()}"
        )
    months = (maturity.year - first_delivery.year) * 12 + (
        maturity.month - first_delivery.month
    )
    if maturity.day < first_delivery.day:
        months -= 1
    whole_years, odd = divmod(months, 12)
    quarters = (odd // 3) * 3
    rounded = add_months(
        add_months(first_delivery, 12 * whole_years, keep_end_of_month=False),
        quarters,
        keep_end_of_month=False,
    )
    return RoundedLife(whole_years=whole_years, months=quarters, maturity=rounded)


def conversion_factor(
    bond: Bond,
    first_delivery: date,
    *,
    notional_coupon: float = DEFAULT_NOTIONAL_COUPON,
    digits: int | None = DEFAULT_DIGITS,
) -> float:
    """The price of ``bond`` per unit of face at the notional yield.

    The exchange formula, which is the street price of the *rounded* bond
    discounted at the notional coupon semi-annually::

        a = (1 + y/2) ** -(v/6)
        b = (c/2) * (6 - v)/6
        p = (1 + y/2) ** -m
        d = (c/y) * (1 - p)
        factor = a * (c/2 + p + d) - b

    with ``v`` the months to the next coupon of the rounded bond, ``m`` the
    whole periods from that coupon to the rounded maturity, ``c`` the bond's
    coupon and ``y`` the notional one. ``d`` is the annuity: ``(c/2)`` times
    ``(1 - p)/(y/2)`` is ``(c/y)(1 - p)``, which is where the notional coupon
    divides rather than the bond's own. ``b`` subtracts accrued interest, so
    the result is a clean price; with no odd months it subtracts a whole
    coupon, which is what makes the leading ``c/2`` correct in that case too.

    A bond at exactly the notional coupon and a whole number of years to run
    has a factor of exactly one -- the check worth remembering, because it is
    the one the formula's shape is easiest to get wrong against. It is *not*
    one when the odd months are three or nine: a six per cent bond with a
    quarter of odd life comes to 0.999889, and that residue is the rounding
    convention showing through rather than an error.

    ``digits`` rounds to the published precision, which is what the invoice is
    computed on. Pass ``None`` for the unrounded value.
    """
    if bond.coupon < 0.0:
        raise BadDelivery(
            f"{bond.name} has a coupon of {bond.coupon!r}; a deliverable bond's "
            "conversion factor is a price and a negative coupon is not priced here"
        )
    if notional_coupon <= 0.0:
        raise BadDelivery(
            f"a notional coupon of {notional_coupon!r} is not positive, so the "
            "annuity the factor is built on divides by zero"
        )
    life = rounded_life(first_delivery, bond.maturity)
    periodic = 1.0 + notional_coupon / 2.0
    # A float raised to a float is complex in general, so the annotation has to
    # be narrowed here, the same way `bond._discount` narrows it for the
    # street discounting formula.
    a = float(periodic ** -(life.to_next_coupon / 6.0))
    b = (bond.coupon / 2.0) * ((6 - life.to_next_coupon) / 6.0)
    p = float(periodic**-life.coupons_after_next)
    d = (bond.coupon / notional_coupon) * (1.0 - p)
    factor = a * (bond.coupon / 2.0 + p + d) - b
    if digits is None:
        return factor
    return round(factor, digits)


@dataclass(frozen=True)
class BondFuture:
    """A contract on a basket of deliverable bonds.

    Attributes:
        first_delivery: First day of the delivery month. The conversion factor
            is computed as of this date whatever day delivery happens on, so
            it is contract data rather than a choice.
        delivery: The delivery date the basis arithmetic assumes. The short
            picks it inside the month, and picking the last day is worth a few
            days of carry, so it is stated rather than derived.
        notional_coupon: The yield the factors are computed at.
        digits: Published precision of the factor.
        money_basis: Day count the financing accrues on. Money market, so
            ACT/360 for dollars and ACT/365 fixed for sterling -- and never the
            bond's own basis, which is a different convention for a different
            purpose.
    """

    first_delivery: date
    delivery: date
    notional_coupon: float = DEFAULT_NOTIONAL_COUPON
    digits: int | None = DEFAULT_DIGITS
    money_basis: Basis = Basis.ACT_360
    label: str = ""

    def __post_init__(self) -> None:
        if self.delivery < self.first_delivery:
            raise BadDelivery(
                f"delivery on {self.delivery.isoformat()} is before the delivery "
                f"month begins on {self.first_delivery.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"future delivering {self.delivery.isoformat()}"

    def conversion_factor(self, bond: Bond) -> float:
        """This contract's factor for ``bond``."""
        return conversion_factor(
            bond,
            self.first_delivery,
            notional_coupon=self.notional_coupon,
            digits=self.digits,
        )

    def invoice_price(self, bond: Bond, futures_price: float) -> float:
        """What the long pays per 100 of face: factor times price, plus accrued.

        Accrued interest is the deliverable's own, at the delivery date, on the
        bond's convention. It is not scaled by the conversion factor -- the
        factor normalises the *principal* the contract is written on, and the
        coupon the short gives up is the coupon the bond pays.
        """
        return futures_price * self.conversion_factor(bond) + bond.accrued(self.delivery)
