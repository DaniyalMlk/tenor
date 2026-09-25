"""What a bond earns between now and a horizon, if nothing happens.

"Nothing happens" turns out to mean two different things, and the gap between
them is the entire expected excess return of a bond position.

The first meaning is *arbitrage-free*: the curve evolves to its own forwards.
Today's curve says what a one-year rate will be in one year's time, and if the
market turns out to agree, then a bond held for a year is worth exactly its
forward price. Under that assumption a bond earns its funding cost and nothing
else — the coupon income and the pull of the price towards its forward offset
the financing exactly. This is not an approximation; it is an identity, and
:attr:`HorizonReturn.carry` is asserted equal to
:attr:`HorizonReturn.financing_cost` to machine precision.

The second meaning is what a trader means by an unchanged curve: the zero rate
at each *tenor* is the same tomorrow as today. A five-year yield of 4% stays a
five-year yield of 4%. Under that assumption a bond does not stay where it is;
it ages, and next year the five-year bond is a four-year bond, priced off the
four-year point. On an upward-sloping curve that point is lower, so the bond is
repriced at a lower yield and gains. That is roll-down.

The two curves are different objects and the distinction is the whole module.
The forward curve at a horizon is ``P(t)/P(horizon)`` — today's curve seen from
the horizon. The rolled curve keeps the shape as a function of time to maturity
and moves its reference date. Building the first and calling it the second is
the natural mistake, and it makes roll-down come out as exactly zero every
time, which looks plausible and is the answer to a different question.

A note on the word "carry", which the market uses for at least two quantities.
Here :attr:`HorizonReturn.carry` is coupon income plus the forward price change
— the no-arbitrage total, identically the financing cost. The other common
definition, income minus financing, is :attr:`HorizonReturn.income_less_financing`.
Both are reported because both are used, and neither is called "carry" without
saying which one it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from .bond import BadBond, Bond
from .curve import DiscountCurve, OffCurve
from .rates import Compounding, rate_from_discount

#: How closely the no-arbitrage identity must hold before
#: :meth:`HorizonReturn.is_arbitrage_free` calls it satisfied. Per 100 of
#: notional, so this is a hundredth of a basis point of price.
IDENTITY_TOLERANCE = 1e-9


class BadHorizon(ValueError):
    """A holding period that cannot be evaluated as it stands."""


@dataclass(frozen=True)
class CouponReceipt:
    """A coupon paid inside the holding period, and what it grows to."""

    day: date
    amount: float
    #: Value at the horizon, reinvested at the curve's own forward rate from
    #: the payment date. Any other reinvestment assumption is a view, and a
    #: view does not belong inside a no-arbitrage decomposition.
    value_at_horizon: float


@dataclass(frozen=True)
class HorizonReturn:
    """A holding-period return, split into the part that is free and the part
    that is not.

    Every amount is per 100 of face value, and every price is a dirty price.
    Clean prices are the wrong unit here: accrued interest is part of what a
    holder earns over the period, and netting it out of the price without
    adding it back to income loses it.
    """

    settlement: date
    horizon: date
    #: Dirty price on the valuation curve, for value on the settlement date.
    start_price: float
    #: Dirty price at the horizon if the curve evolves to its own forwards.
    forward_price: float
    #: Dirty price at the horizon if the zero rate at each tenor is unchanged.
    rolled_price: float
    #: Coupons paid strictly after settlement and on or before the horizon.
    coupons: tuple[CouponReceipt, ...]
    #: Simple rate from settlement to the horizon, off the valuation curve.
    financing_rate: float
    #: Year fraction of the holding period, on the curve's basis.
    period: float

    @property
    def coupon_income(self) -> float:
        """Coupons received in the window, grown to the horizon."""
        return math.fsum(one.value_at_horizon for one in self.coupons)

    @property
    def financing_cost(self) -> float:
        """Cost of funding the starting price to the horizon."""
        return self.start_price * self.financing_rate * self.period

    @property
    def forward_price_change(self) -> float:
        """Forward price less starting price.

        Negative for a bond trading above its funding rate, which is most of
        them: a premium bond is pulled towards par, and that pull is not a loss
        — it is the other side of the coupon income, and the two together are
        the financing cost exactly.
        """
        return self.forward_price - self.start_price

    @property
    def carry(self) -> float:
        """Coupon income plus the forward price change.

        Identically the financing cost, on any arbitrage-free curve. This is
        the quantity that is *not* a source of return: a bond held to a horizon
        on a curve that evolves to its forwards earns its funding cost, and a
        strategy built on collecting it is collecting nothing.
        """
        return self.coupon_income + self.forward_price_change

    @property
    def income_less_financing(self) -> float:
        """The market's other "carry": income minus the cost of funds.

        Equal to ``start_price - forward_price``, which is worth knowing. It
        is large and positive for a high-coupon bond and that largeness means
        nothing on its own, because the same bond's price is falling towards
        par by the same amount.
        """
        return self.coupon_income - self.financing_cost

    @property
    def roll_down(self) -> float:
        """Rolled price less forward price.

        The whole of the expected excess return. Positive when the curve
        slopes upward over the bond's remaining life, because ageing moves it
        to a lower yield; negative on an inverted curve; zero on a flat one.
        """
        return self.rolled_price - self.forward_price

    @property
    def total_return(self) -> float:
        """Profit over the period under an unchanged spot curve, per 100."""
        return self.carry + self.roll_down

    @property
    def excess_over_financing(self) -> float:
        """Total return less the cost of funding it.

        Equal to :attr:`roll_down`, and that equality is the finding rather
        than a coincidence of the arithmetic.
        """
        return self.total_return - self.financing_cost

    @property
    def total_return_bps(self) -> float:
        """Total return as basis points of the starting price."""
        return 1e4 * self.total_return / self.start_price

    def is_arbitrage_free(self, tolerance: float = IDENTITY_TOLERANCE) -> bool:
        """Whether carry and the financing cost agree.

        They must, on a curve that discounts consistently. A failure here means
        the curve is not self-consistent, not that the bond is special.
        """
        return abs(self.carry - self.financing_cost) <= tolerance


def forward_curve(curve: DiscountCurve, horizon: date) -> DiscountCurve:
    """Today's curve as seen from ``horizon``.

    Every discount factor is divided by the horizon's, which is a
    renormalisation rather than a new view: the forward curve contains no
    information the original did not, and pricing on it is pricing today's
    curve for value at a later date.

    Pillars at or before the horizon are dropped, because a curve referenced to
    the horizon cannot hold a point behind itself.
    """
    if horizon < curve.reference:
        raise BadHorizon(
            f"a forward curve starts on or after its own curve's reference date, "
            f"but {horizon.isoformat()} is before {curve.reference.isoformat()}"
        )
    base = curve.discount(horizon)
    points = [
        (pillar.day, pillar.discount / base)
        for pillar in curve.pillars
        if pillar.day > horizon
    ]
    if not points:
        raise BadHorizon(
            f"the curve ends at {curve.horizon.isoformat()}, so there is nothing "
            f"left of it after {horizon.isoformat()} to build a forward curve from"
        )
    return DiscountCurve.from_discounts(
        horizon, points, basis=curve.basis, interpolation=curve.interpolation
    )


def rolled_curve(curve: DiscountCurve, horizon: date) -> DiscountCurve:
    """The curve with the same zero rate at each tenor, referenced from ``horizon``.

    This is the "unchanged curve" of carry-and-roll-down, and it is a genuinely
    different object from :func:`forward_curve`. Each pillar keeps its distance
    from the reference date and its discount factor; only the reference moves.

    The shift is applied in *days* rather than by adding a period, so a pillar
    that was 1826 days out stays 1826 days out. Rolling by calendar months
    instead would move the pillars by unequal amounts and quietly change the
    curve's shape, which is the one thing this function exists not to do.
    """
    if horizon < curve.reference:
        raise BadHorizon(
            f"a curve cannot be rolled backwards: {horizon.isoformat()} is before "
            f"{curve.reference.isoformat()}"
        )
    offset = horizon - curve.reference
    points = [
        (pillar.day + offset, pillar.discount)
        for pillar in curve.pillars
        if not pillar.is_anchor
    ]
    if not points:
        raise BadHorizon("a curve with no pillar beyond its reference cannot be rolled")
    return DiscountCurve.from_discounts(
        horizon, points, basis=curve.basis, interpolation=curve.interpolation
    )


def horizon_return(
    bond: Bond,
    curve: DiscountCurve,
    settlement: date,
    horizon: date,
) -> HorizonReturn:
    """Decompose a bond's holding-period return into carry and roll-down.

    ``curve`` is the valuation curve at ``settlement``. The horizon must fall
    strictly after settlement and strictly before the bond's maturity: a bond
    that has redeemed has no price to roll to, and a decomposition of its
    return into a price change is meaningless rather than zero.

    Coupons paid inside the window are reinvested at the curve's own forward
    rates. That is the only assumption under which the no-arbitrage identity
    holds, and it is the point of the exercise — reinvesting at a chosen rate
    instead would make :attr:`HorizonReturn.carry` differ from the financing
    cost by the size of the view, which is a real number about the view and not
    about the bond.
    """
    if horizon <= settlement:
        raise BadHorizon(
            f"a holding period runs forwards: the horizon {horizon.isoformat()} is "
            f"not after settlement {settlement.isoformat()}"
        )
    if horizon >= bond.maturity:
        raise BadHorizon(
            f"the horizon {horizon.isoformat()} is on or after the bond's maturity "
            f"{bond.maturity.isoformat()}. A redeemed bond has no price at the "
            "horizon, so there is no price change to decompose — value the "
            "redemption directly instead."
        )
    if settlement < curve.reference:
        raise BadHorizon(
            f"settlement {settlement.isoformat()} is before the curve's reference "
            f"date {curve.reference.isoformat()}"
        )

    period = curve.time_to(horizon) - curve.time_to(settlement)
    if period <= 0.0:
        raise BadHorizon(
            f"the holding period measures as {period!r} years on the curve's "
            f"{curve.basis.value} basis, which cannot be used as a denominator"
        )

    settle_factor = curve.discount(settlement)
    horizon_factor = curve.discount(horizon)
    start_price = bond.price_from_curve(curve, settlement)

    # Coupons strictly after settlement and on or before the horizon. Strictly
    # after, because a coupon paid *on* the settlement date belongs to the
    # seller and is not in the dirty price being decomposed.
    receipts: list[CouponReceipt] = []
    remaining_pv = 0.0
    for flow in bond.cashflows(settlement):
        if settlement < flow.day <= horizon:
            grown = flow.amount * curve.discount(flow.day) / horizon_factor
            receipts.append(
                CouponReceipt(day=flow.day, amount=flow.amount, value_at_horizon=grown)
            )
        else:
            remaining_pv += flow.amount * curve.discount(flow.day)

    # The forward price is what is left of the bond, grown to the horizon. It
    # is deliberately not computed as `bond.price_from_curve(forward_curve(...))`
    # — that would give the same number by a longer route and would inherit the
    # interpolation error of a rebuilt curve into an identity that should hold
    # exactly.
    forward_price = remaining_pv / horizon_factor

    rolled = rolled_curve(curve, horizon)
    rolled_price = bond.price_from_curve(rolled, horizon)

    financing_rate = rate_from_discount(
        horizon_factor / settle_factor, period, Compounding.SIMPLE
    )

    return HorizonReturn(
        settlement=settlement,
        horizon=horizon,
        start_price=start_price,
        forward_price=forward_price,
        rolled_price=rolled_price,
        coupons=tuple(receipts),
        financing_rate=financing_rate,
        period=period,
    )


def rolling_horizon_returns(
    bond: Bond,
    curve: DiscountCurve,
    settlement: date,
    *,
    days: int = 90,
    steps: int = 4,
) -> tuple[HorizonReturn, ...]:
    """The same decomposition at a sequence of lengthening horizons.

    Useful for seeing where roll-down actually comes from: it is not linear in
    the holding period, because the bond moves through the curve's shape rather
    than along a straight line.

    Stops early rather than raising once the horizon would reach the bond's
    maturity or run off the end of the curve, so a long-dated request over a
    short bond returns what it could compute.
    """
    if days <= 0:
        raise BadHorizon(f"a step is a positive number of days, got {days!r}")
    if steps <= 0:
        raise BadHorizon(f"a sequence has at least one step, got {steps!r}")
    out: list[HorizonReturn] = []
    for step in range(1, steps + 1):
        horizon = settlement + timedelta(days=days * step)
        if horizon >= bond.maturity:
            break
        try:
            out.append(horizon_return(bond, curve, settlement, horizon))
        except (BadHorizon, BadBond, OffCurve):
            break
    return tuple(out)


__all__ = [
    "IDENTITY_TOLERANCE",
    "BadHorizon",
    "CouponReceipt",
    "HorizonReturn",
    "forward_curve",
    "horizon_return",
    "rolled_curve",
    "rolling_horizon_returns",
]
