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

**The implied repo rate is linear, so it is not solved.** The cash-and-carry
break-even is an equation in the financing rate, and the financing rate appears
once in the cost of carrying the bond and once in the reinvestment of any coupon
that falls inside the holding period. Both are simple interest, so the equation
is linear and :func:`implied_repo_rate` divides rather than iterating. Using the
package's root finder here would be slower, less accurate and would invite the
reader to think there is a fixed point where there is not.

**Net basis and the implied repo rate are the same number twice.** Exactly::

    net basis = (repo - implied repo) * financed balance

where the financed balance is the dirty price accrued over the holding period,
less each interim coupon accrued from its own payment date. That identity is
asserted to machine precision in the tests. It also settles which of the
conventional definitions of cheapest to deliver can disagree with which, and the
answer is not the one the algebra suggests.

**Net basis is the odd criterion, and it is nearly always safe anyway.** The
three conventions rank the basket on the same gap between a bond's break-even
futures price and the quote, scaled three different ways: unscaled, times the
conversion factor (net basis), and times the factor over the financed balance
(implied repo). The expectation is that the rate and the price part company,
because a deep-discount long bond finances around half the balance of a
high-coupon short one. Measured on a four-bond long basket, that is wrong: the
factors span a ratio of 2.00 and the factor *over* the balance spans 4.65%,
because the balance is very nearly the price and the price is very nearly the
factor times the futures price, so the factor cancels. The implied repo rate and
the break-even futures price therefore rank a basket alike.

It is the net basis that is measured in a different unit -- points per 100 of
*the bond's* face, not per contract -- and dividing it by the conversion factor
recovers the futures-price criterion exactly. On that basket the quote has to be
5.00 points, or 4.20%, below the basket-implied price before the net-basis
ranking picks a different bond, at which point every deliverable shows several
points of arbitrage and the question is academic. So comparing raw net bases
across a basket does compare quantities in different units, and gets away with
it because a liquid contract never trades five points from fair.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .bond import Bond
from .daycount import Basis, year_fraction
from .schedule import add_months

__all__ = [
    "BadDelivery",
    "BondFuture",
    "CheapestToDeliver",
    "DeliveryBasis",
    "RoundedLife",
    "cheapest_to_deliver",
    "conversion_factor",
    "delivery_basis",
    "implied_repo_rate",
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

    def basis_of(
        self,
        bond: Bond,
        *,
        clean_price: float,
        futures_price: float,
        settlement: date,
        repo: float,
    ) -> DeliveryBasis:
        """The full cash-and-carry for one deliverable."""
        return delivery_basis(
            bond,
            self,
            clean_price=clean_price,
            futures_price=futures_price,
            settlement=settlement,
            repo=repo,
        )


@dataclass(frozen=True)
class DeliveryBasis:
    """One deliverable's cash-and-carry, and the rate that makes it break even.

    Every price here is per 100 of the bond's face, and every rate is a simple
    interest rate on :attr:`BondFuture.money_basis`.
    """

    bond: Bond
    conversion_factor: float
    clean_price: float
    dirty_price: float
    accrued_at_delivery: float
    #: Coupons paid strictly after settlement and no later than delivery.
    interim_coupons: tuple[tuple[date, float], ...]
    invoice_price: float
    #: Year fraction from settlement to delivery on the money basis.
    holding_period: float
    #: ``dirty * holding period`` less each interim coupon over its own stub.
    #: The balance the financing is charged on, and the factor that converts a
    #: rate into a price.
    financed_balance: float
    #: Clean price less the futures price times the factor.
    gross_basis: float
    #: Coupon income and its reinvestment, less the cost of financing.
    carry: float
    #: ``gross_basis - carry``.
    net_basis: float
    repo: float
    implied_repo: float
    #: The futures price at which :attr:`net_basis` would be zero. Independent
    #: of the quote, so this is the one criterion that survives not having one.
    breakeven_futures_price: float

    @property
    def is_cheapest_at(self) -> float:
        """Shorthand for the ranking quantity: the futures price it justifies."""
        return self.breakeven_futures_price


def _interim_coupons(
    bond: Bond, settlement: date, delivery: date
) -> tuple[tuple[date, float], ...]:
    """Coupons paid in ``(settlement, delivery]``.

    Half-open at the settlement end for the same reason
    :meth:`~tenor.bond.Bond.cashflows` is: a coupon paid on the settlement date
    belongs to the seller. Closed at the delivery end because a coupon paid on
    the delivery date is paid to whoever still holds the bond that morning,
    which is the short.

    The redemption cannot appear here and is not filtered out. ``Bond`` bundles
    the final coupon into the redemption flow, so a filter would have to take
    the principal back off the amount, and subtracting something that is never
    there reads as a case that has been handled. It is not a case: the caller
    refuses a bond maturing on or before delivery, so every flow this sees is
    strictly before the last one. :func:`delivery_basis` carries that guard and
    the tests carry the search for a counterexample.
    """
    return tuple(
        (flow.day, flow.amount)
        for flow in bond.cashflows(settlement)
        if settlement < flow.day <= delivery
    )


def implied_repo_rate(
    *,
    dirty_price: float,
    invoice_price: float,
    interim_coupons: Sequence[tuple[float, float]],
    holding_period: float,
) -> float:
    """The financing rate at which buying and delivering breaks even.

    Buy the bond dirty, finance it to delivery, collect any coupon that falls
    inside and reinvest it to delivery at the same rate, deliver, receive the
    invoice::

        dirty * (1 + r * T) = invoice + sum C_i * (1 + r * T_i)

    linear in ``r``, so::

        r = (invoice + sum C_i - dirty) / (dirty * T - sum C_i * T_i)

    ``interim_coupons`` is pairs of ``(year fraction from payment to delivery,
    amount)``.

    The denominator is the financed balance, and it is guarded relative to its
    own leading term rather than against zero: it vanishes only if the interim
    coupons weighted by their stubs match the whole dirty price weighted by the
    holding period, which needs a coupon of the order of the price and is
    therefore a malformed input rather than a small number.
    """
    if dirty_price <= 0.0:
        raise BadDelivery(
            f"a dirty price of {dirty_price!r} is not positive, so there is nothing "
            "to finance"
        )
    if holding_period <= 0.0:
        raise BadDelivery(
            f"a holding period of {holding_period!r} is not positive; delivery must "
            "be after settlement for a carry to exist"
        )
    scale = dirty_price * holding_period
    balance = scale - math.fsum(
        amount * stub for stub, amount in interim_coupons
    )
    if abs(balance) <= 1e-12 * scale:
        raise BadDelivery(
            f"the financed balance is {balance!r} against a position of {scale!r}: "
            "the interim coupons carry the whole holding cost, so no financing rate "
            "makes this trade break even"
        )
    proceeds = invoice_price + math.fsum(amount for _, amount in interim_coupons)
    return (proceeds - dirty_price) / balance


def delivery_basis(
    bond: Bond,
    future: BondFuture,
    *,
    clean_price: float,
    futures_price: float,
    settlement: date,
    repo: float,
) -> DeliveryBasis:
    """Gross basis, carry, net basis and the implied repo rate for one bond.

    Carry is coupon income less the cost of financing the position, both over
    the holding period: accrued interest earned, plus any coupon actually paid
    inside the period, plus the simple interest that coupon earns from its
    payment date to delivery, less simple interest on the dirty price.

    Net basis is gross basis less carry, and it is also
    ``(repo - implied repo) * financed balance``. The two routes agree to
    machine precision and the tests check that they do, which is worth more
    than either on its own: the identity is the statement that the implied repo
    rate and the net basis are one number expressed twice, and a sign error in
    either definition breaks it.
    """
    if futures_price <= 0.0:
        raise BadDelivery(
            f"a futures price of {futures_price!r} is not positive, so the invoice "
            "it implies is not a price"
        )
    if settlement >= future.delivery:
        raise BadDelivery(
            f"settlement on {settlement.isoformat()} is not before delivery on "
            f"{future.delivery.isoformat()}"
        )
    if bond.maturity <= future.delivery:
        # Not a redundant guard. The conversion factor only needs the bond to
        # outlive the *first* day of the delivery month, and the short picks the
        # delivery day inside it, so a bond maturing on the 15th has a perfectly
        # good factor and cannot be delivered on the 31st. Refusing it here is
        # also what makes the interim coupons safe to take at face value.
        raise BadDelivery(
            f"{bond.name} matures on {bond.maturity.isoformat()}, on or before "
            f"delivery on {future.delivery.isoformat()}; it has redeemed by then"
        )
    factor = future.conversion_factor(bond)
    accrued_now = bond.accrued(settlement)
    accrued_then = bond.accrued(future.delivery)
    dirty = clean_price + accrued_now
    coupons = _interim_coupons(bond, settlement, future.delivery)
    stubs = tuple(
        (year_fraction(day, future.delivery, future.money_basis), amount)
        for day, amount in coupons
    )
    holding = year_fraction(settlement, future.delivery, future.money_basis)
    invoice = futures_price * factor + accrued_then
    balance = dirty * holding - math.fsum(amount * stub for stub, amount in stubs)

    paid = math.fsum(amount for _, amount in coupons)
    reinvestment = repo * math.fsum(amount * stub for stub, amount in stubs)
    financing = repo * dirty * holding
    carry = (accrued_then - accrued_now) + paid + reinvestment - financing
    gross = clean_price - futures_price * factor
    return DeliveryBasis(
        bond=bond,
        conversion_factor=factor,
        clean_price=clean_price,
        dirty_price=dirty,
        accrued_at_delivery=accrued_then,
        interim_coupons=coupons,
        invoice_price=invoice,
        holding_period=holding,
        financed_balance=balance,
        gross_basis=gross,
        carry=carry,
        net_basis=gross - carry,
        repo=repo,
        implied_repo=implied_repo_rate(
            dirty_price=dirty,
            invoice_price=invoice,
            interim_coupons=stubs,
            holding_period=holding,
        ),
        breakeven_futures_price=(clean_price - carry) / factor,
    )


@dataclass(frozen=True)
class CheapestToDeliver:
    """The basket, ranked, and whether the three conventions agree.

    Attributes:
        basis: Every deliverable's cash-and-carry, in the order given.
        by_implied_repo: The bond with the highest implied repo rate.
        by_net_basis: The bond with the lowest net basis.
        by_futures_price: The bond justifying the lowest futures price, which
            is the definition that does not need a quote.
        unanimous: Whether all three name the same bond.
    """

    basis: tuple[DeliveryBasis, ...]
    by_implied_repo: DeliveryBasis
    by_net_basis: DeliveryBasis
    by_futures_price: DeliveryBasis
    unanimous: bool

    @property
    def implied_futures_price(self) -> float:
        """The futures price the basket justifies: the lowest any bond does."""
        return self.by_futures_price.breakeven_futures_price


def cheapest_to_deliver(
    basket: Sequence[Bond],
    future: BondFuture,
    *,
    clean_prices: Sequence[float],
    futures_price: float,
    settlement: date,
    repo: float,
) -> CheapestToDeliver:
    """Rank a basket by all three conventional criteria.

    All three rank on the gap between a bond's break-even futures price and the
    quote, scaled differently: unscaled for the price criterion, times the
    conversion factor for the net basis, and times the factor over the financed
    balance for the implied repo rate. So the orders differ only as much as
    those scalings do, and they do not differ equally. Factor over balance is
    nearly constant across a basket -- 4.65% of spread against the factors' own
    ratio of 2.00, on the long basket in the tests -- because the balance is
    nearly the price and the price is nearly the factor times the futures price.
    The rate and the price therefore agree in practice, which is the opposite of
    what the algebra suggests, and the net basis is the criterion on its own
    scale.

    None of that makes any of them wrong. A desk financing at a single rate
    across the basket wants the rate comparison; one asked what the contract
    should trade at wants the price. :attr:`CheapestToDeliver.unanimous` says
    whether the question mattered on this basket, and on any quote within five
    points of fair value it does not.
    """
    if len(basket) != len(clean_prices):
        raise BadDelivery(
            f"{len(basket)} bonds and {len(clean_prices)} prices; the basket and its "
            "quotes have to line up"
        )
    if not basket:
        raise BadDelivery("an empty basket has no cheapest bond")
    entries = tuple(
        delivery_basis(
            bond,
            future,
            clean_price=price,
            futures_price=futures_price,
            settlement=settlement,
            repo=repo,
        )
        for bond, price in zip(basket, clean_prices, strict=True)
    )
    by_repo = max(entries, key=lambda one: one.implied_repo)
    by_net = min(entries, key=lambda one: one.net_basis)
    by_price = min(entries, key=lambda one: one.breakeven_futures_price)
    names = {by_repo.bond.name, by_net.bond.name, by_price.bond.name}
    return CheapestToDeliver(
        basis=entries,
        by_implied_repo=by_repo,
        by_net_basis=by_net,
        by_futures_price=by_price,
        unanimous=len(names) == 1,
    )
