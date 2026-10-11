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

**The delivery option is described, not valued.** :func:`delivery_switch` walks
a parallel yield shift and reports which bond is cheapest at each level, which
is where the option's payoff comes from: at the notional yield every conversion
factor is fair and the basket is indifferent, and away from it the ranking is by
duration, so the cheapest bond switches from the shortest to the longest as
yields rise through the notional coupon. Putting a *value* on that switch needs
a model of how the curve moves, which this package does not have, so this module
does not pretend to one.

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
    "DeliveryOption",
    "RoundedLife",
    "SwitchLevel",
    "cheapest_to_deliver",
    "conversion_factor",
    "delivery_basis",
    "delivery_option",
    "delivery_switch",
    "futures_dv01",
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


@dataclass(frozen=True)
class SwitchLevel:
    """Which bond is cheapest at one yield level.

    Attributes:
        yield_level: The flat yield every bond in the basket was priced at.
        cheapest: Its name.
        futures_price: The price that bond justifies.
        runner_up: The next cheapest, or the same name for a basket of one.
        margin: How much cheaper, in price points. Zero at the notional
            coupon, where the factors are fair and the basket is indifferent.
    """

    yield_level: float
    cheapest: str
    futures_price: float
    runner_up: str
    margin: float


def delivery_switch(
    basket: Sequence[Bond],
    future: BondFuture,
    *,
    settlement: date,
    repo: float,
    yields: Sequence[float],
) -> tuple[SwitchLevel, ...]:
    """Walk a flat yield across the basket and report the cheapest at each level.

    Each bond is priced at the same yield, which is the point: the conversion
    factors were computed at the notional coupon, so at that yield every bond's
    price over its factor is the same number and the basket is exactly
    indifferent. Away from it the ranking is by duration, because price over
    factor falls fastest for the bond with the most of it. So the cheapest bond
    is the shortest in the basket below the notional coupon and the longest
    above it, and the switch is at the notional coupon rather than anywhere the
    market chose.

    This is the geometry of the short's option, not its value. Valuing it needs
    a model of how the curve moves between now and delivery, and a flat yield
    walked in parallel is not one -- it says nothing about the probability of
    arriving anywhere on it. The margin column is what the option is worth
    *conditional* on landing there, which is the most this can honestly say.
    """
    if not basket:
        raise BadDelivery("an empty basket has no cheapest bond")
    levels: list[SwitchLevel] = []
    for level in yields:
        ranked = sorted(
            (
                delivery_basis(
                    bond,
                    future,
                    clean_price=bond.clean_price(level, settlement),
                    # Any positive quote serves: the break-even futures price
                    # this ranks on does not depend on it, and the constructor
                    # refuses a non-positive one.
                    futures_price=100.0,
                    settlement=settlement,
                    repo=repo,
                )
                for bond in basket
            ),
            key=lambda one: one.breakeven_futures_price,
        )
        best = ranked[0]
        next_best = ranked[1] if len(ranked) > 1 else best
        levels.append(
            SwitchLevel(
                yield_level=level,
                cheapest=best.bond.name,
                futures_price=best.breakeven_futures_price,
                runner_up=next_best.bond.name,
                margin=next_best.breakeven_futures_price
                - best.breakeven_futures_price,
            )
        )
    return tuple(levels)


def futures_dv01(
    basis: DeliveryBasis, *, settlement: date, yield_level: float, shift: float = 1e-4
) -> float:
    """The contract's sensitivity: the cheapest bond's, over the factor.

    A one basis point move in the deliverable's yield moves its price by its
    own pv01 and the futures price by that over the conversion factor, because
    the invoice divides by the factor. A factor below one therefore *raises*
    the contract's sensitivity above the bond's, which is the opposite of the
    intuition that a factor scales risk down.

    Holding the cheapest bond fixed is the approximation, and it is the same
    approximation the number is used under: it is the hedge ratio for a small
    move, and a move large enough to change the cheapest bond is a move this
    does not describe.
    """
    return basis.bond.pv01(yield_level, settlement, shift=shift) / basis.conversion_factor


@dataclass(frozen=True)
class DeliveryOption:
    """The value of the short's option to choose, not merely its geometry.

    Attributes:
        yield_deviation: Standard deviation of the yield level at delivery,
            ``volatility * sqrt(years to delivery)``.
        intrinsic: The futures price the basket justifies if the level does not
            move: the lowest price over conversion factor at the forward level.
        intrinsic_cheapest: The bond that justifies it.
        futures_price: ``E[min over the basket]`` once the level is allowed to
            move, which is the price the contract should trade at.
        convexity: What the cheapest bond's own convexity adds, holding the
            choice fixed. Non-negative, by Jensen's inequality on a convex
            price-yield function.
        switch: What the right to change bonds subtracts. Non-positive, because
            the minimum over the basket is at most any one member of it.
        net: ``futures_price - intrinsic``, identically ``convexity + switch``.
            Its sign is a question about this basket rather than a general fact.
        probabilities: Chance each bond is the cheapest at delivery, in the
            order the basket was given. Exact: the regions are the intervals
            between the kinks, so these are differences of the normal
            distribution function rather than a quadrature.
        kinks: Yield levels at which the cheapest bond changes, increasing.
    """

    forward_level: float
    yield_deviation: float
    intrinsic: float
    intrinsic_cheapest: str
    futures_price: float
    convexity: float
    switch: float
    net: float
    probabilities: tuple[tuple[str, float], ...]
    kinks: tuple[float, ...]


_GAUSS_NODES = 16
_GAUSS_PANELS = 24
_GAUSS_WIDTH = 9.0


def _legendre(nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss-Legendre nodes and weights on ``(-1, 1)``, by Newton on the polynomial.

    Written here rather than imported because this library has no numerical
    dependency and is not about to acquire one for sixteen numbers. The
    starting guesses are Tricomi's, which are good enough that three Newton
    steps reach the last bit.
    """
    points: list[float] = []
    weights: list[float] = []
    for index in range(1, nodes + 1):
        guess = math.cos(math.pi * (index - 0.25) / (nodes + 0.5))
        value = guess
        derivative = 0.0
        for _ in range(100):
            previous, current = 1.0, value
            for degree in range(2, nodes + 1):
                previous, current = current, (
                    (2 * degree - 1) * value * current - (degree - 1) * previous
                ) / degree
            derivative = nodes * (value * current - previous) / (value * value - 1.0)
            step = current / derivative
            value -= step
            if abs(step) < 1e-16:
                break
        points.append(-value)
        weights.append(2.0 / ((1.0 - value * value) * derivative * derivative))
    order = sorted(range(nodes), key=lambda k: points[k])
    return tuple(points[k] for k in order), tuple(weights[k] for k in order)


def _normal_density(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _normalised_prices(
    basket: Sequence[Bond],
    factors: Sequence[float],
    spreads: Sequence[float],
    delivery: date,
    level: float,
) -> list[float]:
    """Each bond's clean price at delivery over its conversion factor."""
    return [
        bond.clean_price(level + spread, delivery) / factor
        for bond, factor, spread in zip(basket, factors, spreads, strict=True)
    ]


def _envelope_kinks(
    basket: Sequence[Bond],
    factors: Sequence[float],
    spreads: Sequence[float],
    delivery: date,
    low: float,
    high: float,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Where the cheapest bond changes on ``[low, high]``, and which bond rules where.

    Every candidate is a crossing of *some pair* of normalised prices, so the
    pairs are solved rather than the envelope scanned. Each pair's difference is
    a difference of two smooth decreasing functions; where the endpoints bracket
    a sign change there is a crossing, and bisection finds it to the last
    representable step. The envelope's kinks are then the subset of those
    crossings at which the argument of the minimum actually changes, decided by
    evaluating it at the midpoint of each segment.

    Doing it this way rather than by scanning matters more than it looks. The
    intermediate bonds of a long basket are cheapest only inside a window about
    a tenth of a basis point wide, and a scan fine enough to see that at one
    volatility is not fine enough at another -- so the reported kinks, and the
    delivery probabilities that come from them, would move with the volatility
    for no reason but the grid.
    """
    size = len(basket)
    candidates: list[float] = []
    for first in range(size):
        for second in range(first + 1, size):

            def gap(
                point: float, first: int = first, second: int = second
            ) -> float:
                prices = _normalised_prices(basket, factors, spreads, delivery, point)
                return prices[first] - prices[second]

            left, right = low, high
            left_sign = gap(left)
            if left_sign == 0.0:
                candidates.append(left)
                continue
            if left_sign * gap(right) > 0.0:
                continue
            for _ in range(200):
                middle = 0.5 * (left + right)
                if middle <= left or middle >= right:
                    break
                if gap(middle) * left_sign > 0.0:
                    left = middle
                else:
                    right = middle
            candidates.append(0.5 * (left + right))
    edges = [low, *sorted(candidates), high]
    owners: list[int] = []
    boundaries: list[float] = []
    for index in range(len(edges) - 1):
        middle = 0.5 * (edges[index] + edges[index + 1])
        prices = _normalised_prices(basket, factors, spreads, delivery, middle)
        owner = min(range(size), key=lambda k: prices[k])
        if owners and owner == owners[-1]:
            continue
        if owners:
            boundaries.append(edges[index])
        owners.append(owner)
    return tuple(boundaries), tuple(owners)


def delivery_option(
    basket: Sequence[Bond],
    future: BondFuture,
    *,
    settlement: date,
    forward_level: float,
    volatility: float,
    spreads: Sequence[float] | None = None,
) -> DeliveryOption:
    """Value the short's right to choose which bond to deliver.

    :func:`delivery_switch` walks a flat yield across the basket and says which
    bond is cheapest at each level, which is the option's geometry. This puts a
    measure on that line: the yield level at delivery is normal around
    ``forward_level`` with a standard deviation of ``volatility`` times the root
    of the time to delivery, each bond keeps its own spread to that level, and
    the futures price is the expectation of the lowest normalised price.

    **Two effects, pulling opposite ways.** The right to choose can only lower
    the price, since a minimum over the basket is at most any one member of it.
    Each bond's price is convex in its yield, so letting the level move raises
    the expectation of whichever bond is delivered. The second is not a delivery
    option at all and does not vanish on a basket of one; the two are reported
    separately for that reason, and their sum is what the quote has to clear.
    The net is negative on a basket with a real choice in it and positive on a
    basket of one, so "the delivery option" as a single number is a sign error
    waiting to happen.

    **Priced at delivery, not at settlement.** :func:`delivery_switch` walks
    the yield with the basket priced at *settlement*, because it is describing
    the basis a desk is looking at today. The choice is made at delivery and the
    invoice is computed there, so this prices the basket at the delivery date,
    and the two sets of crossings are not the same: on a long basket they are
    nine to ten basis points apart, against a window in which the intermediate
    bond is cheapest that is a tenth of a basis point wide. Borrowing one date's
    crossings for the other's measure would put every region in the wrong place.

    **The kinks are the panel edges.** The integrand is smooth except where the
    cheapest bond changes, and a quadrature panel straddling one of those loses
    most of its order. The regions between kinks are found first and integrated
    separately, which also makes the delivery probabilities exact rather than
    quadrature: they are differences of the normal distribution function over
    intervals.

    Args:
        basket: The deliverable bonds.
        future: The contract, which supplies the conversion factors.
        settlement: Today, for the time to delivery.
        forward_level: Expected yield level at delivery, as a decimal.
        volatility: Annualised standard deviation of that level in yield units
            -- 0.009 for 90 basis points a year. Zero is allowed and returns
            the intrinsic.
        spreads: Each bond's own spread to the level, as decimals. Zero if
            omitted. Without them a long basket's intermediate bond is cheapest
            only inside a narrow window -- 1.195 basis points on the unrounded
            factors and 0.131 on the published four-digit ones, so the rounding
            narrows it by a factor of nine and the bond's chance of being
            delivered with it.

    Raises:
        BadDelivery: If the basket is empty, the spreads do not line up with it,
            the volatility is negative, or delivery is not after settlement.
    """
    if not basket:
        raise BadDelivery("an empty basket gives the short nothing to choose between")
    offsets = tuple(0.0 for _ in basket) if spreads is None else tuple(spreads)
    if len(offsets) != len(basket):
        raise BadDelivery(
            f"{len(basket)} bonds and {len(offsets)} spreads; they have to line up"
        )
    if not math.isfinite(volatility) or volatility < 0.0:
        raise BadDelivery(
            f"a yield volatility of {volatility!r} is not a non-negative number"
        )
    if future.delivery <= settlement:
        raise BadDelivery(
            f"delivery on {future.delivery.isoformat()} is not after settlement on "
            f"{settlement.isoformat()}, so there is no time for the level to move"
        )
    factors = tuple(future.conversion_factor(bond) for bond in basket)
    years = year_fraction(settlement, future.delivery, future.money_basis)
    deviation = volatility * math.sqrt(years)

    at_forward = _normalised_prices(basket, factors, offsets, future.delivery, forward_level)
    cheapest = min(range(len(basket)), key=lambda k: at_forward[k])
    intrinsic = at_forward[cheapest]
    if deviation == 0.0:
        return DeliveryOption(
            forward_level=forward_level,
            yield_deviation=0.0,
            intrinsic=intrinsic,
            intrinsic_cheapest=basket[cheapest].name,
            futures_price=intrinsic,
            convexity=0.0,
            switch=0.0,
            net=0.0,
            probabilities=tuple(
                (bond.name, 1.0 if index == cheapest else 0.0)
                for index, bond in enumerate(basket)
            ),
            kinks=(),
        )

    low = forward_level - _GAUSS_WIDTH * deviation
    high = forward_level + _GAUSS_WIDTH * deviation
    kinks, owners = _envelope_kinks(basket, factors, offsets, future.delivery, low, high)
    edges = (low, *kinks, high)
    nodes, weights = _legendre(_GAUSS_NODES)

    def integrate(left: float, right: float, index: int) -> float:
        """``int P_index(y) phi(y) dy`` over ``[left, right]``, in panels."""
        count = max(1, math.ceil(_GAUSS_PANELS * (right - left) / (high - low)))
        total = 0.0
        for panel in range(count):
            start = left + (right - left) * panel / count
            stop = left + (right - left) * (panel + 1) / count
            half = 0.5 * (stop - start)
            centre = 0.5 * (stop + start)
            for node, weight in zip(nodes, weights, strict=True):
                level = centre + half * node
                price = (
                    basket[index].clean_price(level + offsets[index], future.delivery)
                    / factors[index]
                )
                total += (
                    half * weight * price * _normal_density((level - forward_level) / deviation)
                    / deviation
                )
        return total

    expected = 0.0
    probabilities = [0.0] * len(basket)
    for region, index in enumerate(owners):
        left, right = edges[region], edges[region + 1]
        expected += integrate(left, right, index)
        probabilities[index] += _standard_normal_cdf(
            (right - forward_level) / deviation
        ) - _standard_normal_cdf((left - forward_level) / deviation)
    held = integrate(low, high, cheapest)
    convexity = held - intrinsic
    return DeliveryOption(
        forward_level=forward_level,
        yield_deviation=deviation,
        intrinsic=intrinsic,
        intrinsic_cheapest=basket[cheapest].name,
        futures_price=expected,
        convexity=convexity,
        switch=expected - held,
        net=expected - intrinsic,
        probabilities=tuple(
            (bond.name, probabilities[index]) for index, bond in enumerate(basket)
        ),
        kinks=kinks,
    )
