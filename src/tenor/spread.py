"""How much more than the curve a bond pays, measured two ways.

A credit spread is a single number standing in for the difference between two
whole curves, so how it is extracted decides what it means.

**I-spread** is the bond's yield less the par swap rate at its maturity. One
number minus one number. It is quick, it is quoted, and it discards the shape
of the curve completely: two bonds with identical cash flows and the same
maturity have the same I-spread benchmark whether the curve between here and
there is flat, steep or humped. On a steep curve a short bond's I-spread
overstates what it earns over funding, because the single benchmark point sits
at the far end of a curve the bond's earlier coupons never see.

**Z-spread** is the constant addition to *every* zero rate that makes the whole
curve reproduce the bond's price. Every cash flow is discounted at its own
point on the curve, plus the spread. It is the honest one, and it costs a root
find.

The cleanest way to see the difference is to take a bond with *no spread at
all* — priced exactly on the curve — and ask each measure what it earns. The
Z-spread answers zero, by construction. The I-spread does not.

On the curve in the tests, which runs from 21bp at six months to 2.79% at twenty
years, a bond priced exactly on it shows an I-spread of -0.14bp at two years,
-0.46bp at five, -1.6bp at ten and **-6.0bp at fifteen**. None of that is
spread. It is the benchmark being a single par rate at the maturity while the
bond's coupons are discounted across the whole curve, and it grows with maturity
and with distance from par. On a flat curve every one of those numbers is zero
to within two hundredths of a basis point, which is why the flat case is a poor
test of anything here.

**The spread runs from settlement, not from the curve's reference date.** A
spread is what the holder earns over the curve for holding the bond, and they
do not hold it before they own it. Accruing it from the curve's reference date
instead inflates every flow's discounting by the same settlement period, which
is a small error on a spot-settling bond and a systematic one on anything with
a long settlement.
"""

from __future__ import annotations

import math
from datetime import date

from .bond import Bond
from .curve import DiscountCurve
from .instruments import Swap
from .schedule import Frequency
from .solve import Root, brent


class BadSpread(ValueError):
    """A spread that cannot be extracted."""


def discounted_with_spread(
    bond: Bond, curve: DiscountCurve, settlement: date, spread: float
) -> float:
    """The bond's dirty price on ``curve`` with ``spread`` on every zero rate.

    Forward to settlement, and the spread accrues from settlement, so a spread
    of zero returns exactly :meth:`Bond.price_from_curve`.
    """
    base = curve.discount(settlement)
    start = curve.time_to(settlement)
    total = 0.0
    for flow in bond.cashflows(settlement):
        time = curve.time_to(flow.day)
        total += (
            flow.amount
            * curve.discount(flow.day)
            / base
            * math.exp(-spread * (time - start))
        )
    return total


def z_spread(
    bond: Bond, price: float, curve: DiscountCurve, settlement: date, *, clean: bool = True
) -> Root:
    """The constant spread over every zero rate that reproduces ``price``.

    ``clean`` says which price was quoted, and defaults to the clean one
    because that is what a screen shows. The solve runs against the dirty
    price either way, for the same reason a yield does.

    The spread is continuously compounded, matching the curve's own internal
    convention, so it is directly comparable with a zero rate off the same
    curve and not with a semi-annual yield.
    """
    target = price + bond.accrued(settlement) if clean else price
    if target <= 0.0:
        raise BadSpread(
            f"{bond.name} cannot have a dirty price of {target!r}; every remaining "
            "flow is positive, so no spread discounts them to zero or below"
        )

    def objective(spread: float) -> float:
        return discounted_with_spread(bond, curve, settlement, spread) - target

    low, high = -0.5, 1.0
    if objective(low) * objective(high) > 0.0:
        low, high = -2.0, 20.0
        if objective(low) * objective(high) > 0.0:
            raise BadSpread(
                f"no spread between {low} and {high} prices {bond.name} at "
                f"{price!r}: the error is {objective(low):.6g} at one end and "
                f"{objective(high):.6g} at the other"
            )
    return brent(objective, low, high, tolerance=1e-15)


def par_rate_at(
    curve: DiscountCurve,
    settlement: date,
    maturity: date,
    *,
    frequency: Frequency = Frequency.SEMI_ANNUAL,
) -> float:
    """The par swap rate the curve implies to ``maturity``.

    The benchmark an I-spread is measured against. Built as an actual swap and
    asked for its par rate rather than approximated by the zero rate, because
    the two differ by the coupon effect — on a steep curve, by more than the
    spreads people quote to three decimal places.
    """
    probe = Swap(settlement, maturity, 0.0, frequency)
    return probe.par_rate(curve)


def i_spread(
    bond: Bond, price: float, curve: DiscountCurve, settlement: date, *, clean: bool = True
) -> float:
    """The bond's yield less the par swap rate at its maturity.

    One number minus one number, in the bond's own compounding. Quick, quoted,
    and blind to everything the curve does before the bond matures.
    """
    target = price if clean else price - bond.accrued(settlement)
    solved = bond.yield_from_clean(target, settlement)
    return solved.value - par_rate_at(
        curve, settlement, bond.maturity, frequency=bond.frequency
    )


__all__ = [
    "BadSpread",
    "discounted_with_spread",
    "i_spread",
    "par_rate_at",
    "z_spread",
]
