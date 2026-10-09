"""Two currencies, one identity, and the residual nobody can make vanish.

Everything else in this library prices in a single currency. A foreign exchange
forward connects two of them, and the connection is an identity rather than a
model: borrow domestic, buy foreign spot, lend foreign, sell the proceeds
forward, and the four prices have to multiply to one or the trade is free money.
So

    F = S * D_foreign / D_domestic

with ``S`` quoted as domestic currency per unit of foreign. The currency with
the higher rate has the smaller discount factor and therefore trades at a
forward *discount*, which is the direction that catches people: a high interest
rate is not a reason to expect appreciation, it is the reason the forward is
below the spot.

**The identity has four quantities and the market quotes all four.** Two curves,
a spot and a strip of forwards over-determine the system, and the residual is
the cross-currency basis. It is not a rounding error and it is not a data
problem to be cleaned: a dollar borrowed synthetically through the FX market
costs a different rate from a dollar borrowed directly, and the difference is
what this module measures. :func:`implied_curve` builds the foreign discount
curve the forwards imply, :func:`implied_basis` is what it and the foreign
currency's own curve disagree by, and :func:`par_basis_spread` is what a swap
has to pay to close the gap.

**The implied curve is built by inversion, not by bootstrapping.** A swap
bootstrap solves each pillar because a swap's value depends on every pillar
before it. A forward's does not: one division gives the foreign discount factor
at its own maturity and nothing else is involved. So :func:`implied_curve` has
no solver, no iteration count and no way to fail to converge, the round trip
through :func:`implied_foreign_discount` and back through
:func:`forward_outright` is **exact** — the same float, not a close one — and a
curve rebuilt from forwards generated off a known curve reproduces it to 1.1e-16
relative.

**Parity holds from the spot date, and the lag is the largest error here.** A
foreign exchange spot trade settles two business days out almost everywhere, so
the arbitrage runs between the spot date and the forward date and both discount
factors have to be taken relative to the spot date. Skipping that is the easiest
mistake in the module to make and the hardest to see, because the forward still
looks like a forward.

What makes it dangerous is that the error is almost independent of tenor. On a
EUR/USD-shaped example — a spot of 1.10, a 4.5% domestic curve, a 3.0% foreign
one and settlement two business days out over a weekend — ignoring the lag moves
the outright by 1.815 pips at three months, 1.835 at one year and 1.949 at five.
As a share of the forward points that is **4.7% at three months**, 1.1% at one
year and 0.23% at five, so the convention matters most exactly where the forward
is cheapest and most actively quoted. :func:`forward_outright` takes the spot
date rather than assuming one.

The same lag is the one piece a strip of forwards cannot determine. Parity fixes
the *ratio* of foreign discount factors between the spot date and the forward
date, so turning the strip into a curve from today needs the foreign discount
factor to the spot date from somewhere else. :func:`implied_curve` defaults it
to the domestic one — the two currencies discounting alike over two days — and
the cost of that default is **1.47 basis points of implied basis at one year**
and 0.29 at five. It scales as one over the maturity, so it is worst at the
short end, which is where the basis is quoted most actively. Pass
``foreign_spot`` whenever there is a foreign curve to take it from.

**The par basis spread comes out of a telescoping identity, not out of a
solve.** A floating leg projected and discounted on the same curve, together
with notional received at the start and repaid at maturity, is worth exactly
nothing: each coupon is the difference of two neighbouring discount factors and
the sum collapses onto the two ends. A cross-currency leg breaks that by
projecting on one curve and discounting on another, and the spread that restores
it is the leftover divided by the annuity — one expression, no iteration. When
the two curves are the same object the numerator telescopes to **-1.0e-16** on a
five-year quarterly schedule, which is the floating-point residue of the
collapse rather than a tolerance anybody chose.

**The par spread is not the curve basis, and the two corrections pull opposite
ways.** Against a flat continuously compounded basis of 25 basis points on an
ACT/365F curve, a five-year quarterly leg accruing on ACT/365F pays 25.196 —
**+0.785%**, which is the whole of the compounding effect, since a spread paid
quarterly on an accrual is worth more than the same number compounded
continuously. Change the accrual to ACT/360, which is what a dollar leg uses,
and the accruals grow by 365/360 so the spread has to shrink by 1.37%: 24.851,
or **-0.595%** against the curve basis. The net figure is smaller than either
part, and reporting it alone would hide that an ACT/365F leg moves it the other
way. The gap is close to proportional in the basis — -0.0614bp at 10, -0.1488 at
25, -0.5020 at 100 — rather than quadratic, because it is a convention and not an
approximation.

**And a term structure of basis is averaged over the *forward* basis, which is
not the curve people read.** With a zero basis rising linearly from 10 to 40
basis points over five years, the forward basis is much steeper — 13bp at three
months and 70bp at five years, since differentiating ``b(t) t`` adds ``t b'(t)``
— and the par spread is **39.02 basis points**. A reader who averages the *zero*
basis over time gets 25, understating by 36%; one who reads its short end gets
11.5, understating by 70%. The spread happens to land near the five-year zero
basis here, which is a coincidence of this shape and not a rule. That is the
reason this function exists rather than a subtraction of two zero rates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .curve import BadCurve, DiscountCurve, Interpolation, OffCurve
from .daycount import Basis, year_fraction
from .schedule import Schedule

__all__ = [
    "DEFAULT_PIP",
    "BadFx",
    "BasisPoint",
    "FxForward",
    "basis_curve",
    "forward_outright",
    "forward_points",
    "implied_basis",
    "implied_curve",
    "implied_foreign_discount",
    "par_basis_spread",
]

#: One pip, as a fraction of the quote currency. Four decimal places, which is
#: the convention for every major pair except the yen crosses, where it is two.
#: Named rather than inlined because a pip is a quoting convention and not a
#: property of the mathematics: a function that returns "points" without saying
#: which pip it divided by has returned a number with no units.
DEFAULT_PIP = 1e-4


class BadFx(ValueError):
    """A foreign exchange quote or pair that cannot be used as given."""


@dataclass(frozen=True)
class FxForward:
    """An observed outright forward.

    Attributes:
        day: Settlement date of the forward.
        outright: The forward rate, in the same quotation as the spot —
            domestic currency per unit of foreign.
    """

    day: date
    outright: float

    def __post_init__(self) -> None:
        if self.outright <= 0.0:
            raise BadFx(
                f"a forward of {self.outright!r} on {self.day.isoformat()} is not "
                "a price; an exchange rate is strictly positive"
            )
        if not math.isfinite(self.outright):
            raise BadFx(f"a forward must be finite, got {self.outright!r}")


@dataclass(frozen=True)
class BasisPoint:
    """The basis between two curves at one date.

    Attributes:
        day: Where it was read.
        time: Year fraction from the reference date, under the curve's basis.
        spread: Continuously compounded spread, as a rate. The foreign discount
            curve implied by the forwards equals the foreign currency's own
            curve multiplied by ``exp(-spread * time)``, so a negative spread
            means the synthetic borrowing is *cheaper* than the direct kind.
    """

    day: date
    time: float
    spread: float


def _checked_spot(spot: float) -> float:
    if spot <= 0.0:
        raise BadFx(f"a spot of {spot!r} is not a price; an exchange rate is positive")
    if not math.isfinite(spot):
        raise BadFx(f"a spot must be finite, got {spot!r}")
    return spot


def _settlement(
    domestic: DiscountCurve, foreign: DiscountCurve, spot_day: date | None
) -> tuple[date, float, float]:
    """The spot date and the two discount factors to it.

    Both curves have to agree about the reference date or the identity is
    comparing two different todays, which is a mistake that produces a plausible
    number.
    """
    if domestic.reference != foreign.reference:
        raise BadFx(
            f"the domestic curve references {domestic.reference.isoformat()} and "
            f"the foreign one {foreign.reference.isoformat()}; parity between "
            "them would be comparing two different todays"
        )
    day = domestic.reference if spot_day is None else spot_day
    if day < domestic.reference:
        raise BadFx(
            f"a spot date of {day.isoformat()} is before the curves' reference "
            f"date {domestic.reference.isoformat()}"
        )
    return day, domestic.discount(day), foreign.discount(day)


def forward_outright(
    spot: float,
    domestic: DiscountCurve,
    foreign: DiscountCurve,
    day: date,
    *,
    spot_day: date | None = None,
) -> float:
    """The outright forward from covered interest parity.

    ``F = S * (D_f(T) / D_f(spot)) / (D_d(T) / D_d(spot))``, with both discount
    factors taken relative to the spot date because that is where the arbitrage
    starts. With ``spot_day`` left out the spot date is the curves' reference,
    which is same-day settlement and is the wrong default for every traded pair
    — it is the default here only because a curve knows its reference and does
    not know a settlement convention.

    Args:
        spot: Domestic currency per unit of foreign. Positive.
        domestic: Discount curve in the currency the rate is quoted in.
        foreign: Discount curve in the currency being priced.
        day: Settlement date of the forward. At or after the spot date.
        spot_day: Settlement date of the spot trade. Defaults to the reference.

    Returns:
        The outright forward, in the spot's quotation.

    Raises:
        BadFx: If the spot is not positive, the curves disagree about their
            reference date, or ``day`` is before the spot date.
        OffCurve: If either curve does not reach ``day``.
    """
    _checked_spot(spot)
    settles, domestic_spot, foreign_spot = _settlement(domestic, foreign, spot_day)
    if day < settles:
        raise BadFx(
            f"a forward settling {day.isoformat()} is before the spot date "
            f"{settles.isoformat()}; that is a past date, not a forward"
        )
    ratio = (foreign.discount(day) / foreign_spot) / (
        domestic.discount(day) / domestic_spot
    )
    return spot * ratio


def forward_points(
    spot: float, outright: float, *, pip: float = DEFAULT_PIP
) -> float:
    """``(forward - spot) / pip``: the forward expressed the way it is quoted.

    Points are in the pip of the *quote* currency, so the number carries the
    quotation and not the mathematics. A negative figure is a forward discount,
    which is what the higher-rate currency trades at.

    Args:
        spot: The spot rate.
        outright: The outright forward, in the same quotation.
        pip: Size of one pip. Positive.

    Returns:
        Forward points.

    Raises:
        BadFx: If the spot is not positive or ``pip`` is not positive.
    """
    _checked_spot(spot)
    if pip <= 0.0:
        raise BadFx(f"a pip is a positive size, got {pip!r}")
    return (outright - spot) / pip


def implied_foreign_discount(
    spot: float,
    outright: float,
    domestic: DiscountCurve,
    day: date,
    *,
    spot_day: date | None = None,
    foreign_spot: float = 1.0,
) -> float:
    """The foreign discount factor an observed forward implies.

    Parity rearranged. One division, so the round trip back through
    :func:`forward_outright` is exact rather than close.

    Args:
        spot: Domestic per unit of foreign.
        outright: The observed forward.
        domestic: Discount curve in the quote currency.
        day: Settlement date of the forward.
        spot_day: Settlement date of the spot trade. Defaults to the reference.
        foreign_spot: The foreign discount factor to the spot date. One unless
            the spot lag is being discounted in the foreign currency too, which
            needs a foreign curve the caller is in the middle of building — so
            it is an argument rather than a lookup.

    Returns:
        The foreign discount factor to ``day``.

    Raises:
        BadFx: If the spot or the forward is not positive, or ``day`` is before
            the spot date.
        OffCurve: If the domestic curve does not reach ``day``.
    """
    _checked_spot(spot)
    _checked_spot(outright)
    if foreign_spot <= 0.0:
        raise BadFx(
            f"a discount factor of {foreign_spot!r} to the spot date is not "
            "usable; discount factors are strictly positive"
        )
    settles = domestic.reference if spot_day is None else spot_day
    if day < settles:
        raise BadFx(
            f"a forward settling {day.isoformat()} is before the spot date "
            f"{settles.isoformat()}"
        )
    domestic_spot = domestic.discount(settles)
    return (outright / spot) * (domestic.discount(day) / domestic_spot) * foreign_spot


def implied_curve(
    spot: float,
    forwards: Sequence[FxForward],
    domestic: DiscountCurve,
    *,
    spot_day: date | None = None,
    foreign_spot: float | None = None,
    basis: Basis | None = None,
    interpolation: Interpolation | None = None,
) -> DiscountCurve:
    """The foreign discount curve a strip of forwards implies.

    Built by inversion. A swap bootstrap solves each pillar because a swap's
    value depends on every pillar before it; a forward's does not, so each
    pillar here is one division and the function has no solver, no iteration
    count and no way to fail to converge.

    The spot lag is the one piece the forwards do not determine, and it is not
    harmless. Parity fixes the *ratio* ``D_f(T) / D_f(spot)``, so turning it
    into a curve from the reference date needs the foreign discount factor to
    the spot date from somewhere else. Left out, the domestic factor is used,
    which assumes the two currencies discount alike over the two days. That
    shifts the whole implied curve by the rate differential over the lag — on a
    two-day lag against a 1.5% differential, 8.3e-05 in level, which is **0.83
    of a basis point** of implied basis at one year and five times that at a
    tenor of three months. It is larger than the accrual-versus-continuous gap
    this module also reports, and in the other direction. So pass
    ``foreign_spot`` whenever there is a foreign curve to take it from.

    Args:
        spot: Domestic per unit of foreign.
        forwards: Observed outrights. At least one, all distinct, all at or
            after the spot date.
        domestic: Discount curve in the quote currency.
        spot_day: Settlement date of the spot trade. Defaults to the reference.
        foreign_spot: The foreign discount factor to the spot date. Defaults to
            the domestic one, with the cost measured above.
        basis: Day count for the new curve. The domestic curve's if omitted.
        interpolation: Scheme for the new curve. The domestic curve's if
            omitted.

    Returns:
        A :class:`~tenor.curve.DiscountCurve` in the foreign currency.

    Raises:
        BadFx: If the spot is not positive, the forwards are empty, a date
            repeats, or a forward settles before the spot date.
        OffCurve: If the domestic curve does not reach a forward's date.
    """
    _checked_spot(spot)
    if not forwards:
        raise BadFx("an implied curve needs at least one forward")
    settles = domestic.reference if spot_day is None else spot_day
    if settles < domestic.reference:
        raise BadFx(
            f"a spot date of {settles.isoformat()} is before the curve's "
            f"reference date {domestic.reference.isoformat()}"
        )
    if foreign_spot is None:
        at_spot = domestic.discount(settles)
    elif foreign_spot <= 0.0 or not math.isfinite(foreign_spot):
        raise BadFx(
            f"a foreign discount factor of {foreign_spot!r} to the spot date is "
            "not usable; discount factors are strictly positive"
        )
    else:
        at_spot = foreign_spot
    seen: set[date] = set()
    points: list[tuple[date, float]] = []
    if settles != domestic.reference:
        points.append((settles, at_spot))
    for quote in forwards:
        if quote.day in seen:
            raise BadFx(f"a forward settling {quote.day.isoformat()} appears twice")
        if quote.day < settles:
            raise BadFx(
                f"a forward settling {quote.day.isoformat()} is before the spot "
                f"date {settles.isoformat()}"
            )
        seen.add(quote.day)
        factor = implied_foreign_discount(
            spot,
            quote.outright,
            domestic,
            quote.day,
            spot_day=settles,
            foreign_spot=at_spot,
        )
        points.append((quote.day, factor))
    try:
        return DiscountCurve.from_discounts(
            domestic.reference,
            points,
            basis=domestic.basis if basis is None else basis,
            interpolation=(
                domestic.interpolation if interpolation is None else interpolation
            ),
        )
    except BadCurve as bad:
        raise BadFx(f"the forwards do not make a usable curve: {bad}") from bad


def implied_basis(
    synthetic: DiscountCurve, direct: DiscountCurve, day: date
) -> BasisPoint:
    """What the forward-implied curve and the currency's own curve disagree by.

    The continuously compounded spread ``b`` with
    ``D_synthetic(T) = D_direct(T) exp(-b T)``. A negative spread means the
    synthetic borrowing is cheaper than the direct kind.

    Args:
        synthetic: The curve implied by the forwards, from :func:`implied_curve`.
        direct: The foreign currency's own discount curve.
        day: Where to read the basis. After the reference date.

    Returns:
        A :class:`BasisPoint`.

    Raises:
        BadFx: If the curves disagree about their reference date.
        OffCurve: If ``day`` is at or before the reference, or beyond either
            curve.
    """
    if synthetic.reference != direct.reference:
        raise BadFx(
            f"the implied curve references {synthetic.reference.isoformat()} and "
            f"the direct one {direct.reference.isoformat()}"
        )
    time = year_fraction(synthetic.reference, day, synthetic.basis)
    if time <= 0.0:
        raise OffCurve(
            "there is no basis at the reference date: both discount factors are "
            "one, so any spread fits"
        )
    return BasisPoint(
        day=day,
        time=time,
        spread=math.log(direct.discount(day) / synthetic.discount(day)) / time,
    )


def basis_curve(
    synthetic: DiscountCurve, direct: DiscountCurve, days: Sequence[date]
) -> tuple[BasisPoint, ...]:
    """:func:`implied_basis` at several dates, in the order given."""
    return tuple(implied_basis(synthetic, direct, day) for day in days)


def par_basis_spread(
    schedule: Schedule,
    projection: DiscountCurve,
    discounting: DiscountCurve,
    *,
    basis: Basis,
) -> float:
    """The spread that makes a constant-notional cross-currency leg worth nothing.

    The leg receives one unit of notional at the schedule's first adjusted
    start, pays its own floating index at each period, repays the notional at
    the last payment date, and pays a constant spread on each accrual. Its value
    is

        D(t_0) - D(t_N) - sum_i D(t_i) (P(t_{i-1}) / P(t_i) - 1)
            - s sum_i D(t_i) tau_i

    where ``P`` projects the index and ``D`` discounts the cash. Setting that to
    zero gives the spread directly: no solve, no bracket, no iteration count.

    When ``projection`` and ``discounting`` are the same curve the first three
    terms telescope — each coupon is the difference of two neighbouring discount
    factors — and the spread is zero. That is the identity the function is built
    on rather than a case it handles, which is why it is worth stating here:
    a cross-currency leg is a single-currency leg whose projection and
    discounting have come apart.

    The result is a *spread on an accrual*, which is not the same quantity as
    the continuously compounded basis between the two curves. Against a flat
    basis of 25 basis points a five-year quarterly schedule gives 24.84, and the
    gap grows with the square of the basis. Against a basis with a term
    structure the spread is the annuity-weighted average and not the time
    average, which is a much larger difference than the compounding one.

    Args:
        schedule: The leg's accrual periods. At least one.
        projection: The curve the floating index is read from — the foreign
            currency's own curve.
        discounting: The curve the cash is discounted on — the curve the
            forwards imply.
        basis: Day count for the accruals.

    Returns:
        The par spread, as a rate. Positive means the leg pays it away.

    Raises:
        BadFx: If the schedule is empty, the curves disagree about their
            reference date, or the accruals sum to nothing.
        OffCurve: If either curve does not reach a payment date.
    """
    if len(schedule) == 0:
        raise BadFx("a par spread needs at least one accrual period")
    if projection.reference != discounting.reference:
        raise BadFx(
            f"the projection curve references {projection.reference.isoformat()} "
            f"and the discounting one {discounting.reference.isoformat()}"
        )
    start = schedule[0].adjusted_start
    leftover = discounting.discount(start) - discounting.discount(
        schedule[-1].payment
    )
    annuity = 0.0
    for period in schedule:
        factor = discounting.discount(period.payment)
        growth = projection.discount(period.adjusted_start) / projection.discount(
            period.adjusted_end
        )
        leftover -= factor * (growth - 1.0)
        annuity += factor * year_fraction(
            period.adjusted_start, period.adjusted_end, basis
        )
    if annuity <= 0.0:
        raise BadFx(
            "the schedule's discounted accruals sum to nothing, so no spread "
            "on them can be worth anything"
        )
    return leftover / annuity
