"""Forecasting and discounting on two different curves.

:class:`~tenor.instruments.Swap` values its floating leg as
``P(start) - P(maturity)``. That identity is exact, it holds for any curve of
any shape, and it holds because the leg is forecast off the same curve it is
discounted on: each period's projected growth is its own discount ratio, so the
sum telescopes and every intermediate date cancels. One subtraction prices the
whole leg and no forward rate is ever computed.

It stopped describing the market in 2008. A three-month deposit rate and an
overnight-indexed rate had been within a basis point or two of each other for a
decade; they separated, and they stayed separated, because the index a swap
forecasts carries term credit and funding risk and the rate a collateralised
swap discounts at does not. Once the two differ the projected growth across a
period is no longer the discount ratio across that period, nothing cancels, and
the leg has to be valued coupon by coupon.

This module is that arithmetic. A :class:`ForecastIndex` says what is being
projected, a :class:`FloatingLeg` projects it off one curve and discounts the
result off another, and :class:`DualSwap` puts a fixed leg against it.
:func:`bootstrap_forecast` solves a projection curve out of par quotes with the
discount curve taken as given, which is the order the market itself works in:
discounting is settled first, from overnight-indexed swaps, and every
projection curve is built on top of it.

**The identity is kept as a test rather than a claim.** Point both curves at
the same object and set the spread to zero, and :meth:`DualSwap.floating_value`
computes a sum of forty projected coupons where
:meth:`~tenor.instruments.Swap.floating_value` computes one subtraction. On the
ten-year quarterly leg in the tests they agree to 7.2e-16 of the leg's value —
not bit for bit, because the two expressions accumulate rounding in different
orders, but to the last few bits, which is what says the coupon-by-coupon path
carries no error of its own.

What that identity needs is narrower than "the curves are the same", and the
first draft of this module got the condition wrong. It needs each period's
projected growth to be discounted at the end of that period, because the
cancellation is between one period's ``P(end)`` and the next one's ``P(start)``.
The rolling convention was the suspect and it is not the culprit: under
modified following a period ending on a Saturday rolls to the Monday, but the
*payment* rolls with it, so the dates still meet and the identity survives
untouched — measured, not reasoned about, and the agreement is the 7.2e-16
above. What breaks it is a payment lag, which moves the payment without moving
the accrual it pays for. At the two-business-day lag an overnight-indexed leg
settles on, the gap is 2.6e-04 of the leg, which is 0.095 basis points of par
rate; at five days it is 0.262. So :attr:`ForecastIndex.payment_lag` exists, and
it defaults to zero, and that zero is load-bearing.

**What the discount curve is actually worth, measured.** The expectation this
module was built against was that separating the curves repriced swaps, since
that is the story the 2008 change is usually told as. For a *par* rate it does
almost nothing. Shifting the projection curve by 100 basis points moves the
ten-year par rate by 101.62 basis points. Shifting the discount curve by the
same 100 basis points moves it by **-0.12 basis points** — the other way, and
smaller by a factor of 843. Both of those follow from the par rate being a
discount-weighted average of the forwards the leg projects: the projection
curve moves every term of the average, and the discount curve only reweights
it, so on an upward-sloping curve heavier discounting tilts the weights towards
the earlier and lower forwards and the average falls a little.

The place the discount curve is worth something is a swap that is not at par. A
ten-year swap struck 100 basis points away from the market is worth the rate
difference times the annuity, and the annuity is all discount curve: the same
100 basis point discount shift moves that mark by 4.67% of itself, against the
0.031% it moves the par rate's level. So "which curve do you discount on" is a
third-order question for a new trade and a first-order one for a book of old
ones, which is the reverse of the order the choice is usually introduced in.

**The basis passes through at 1.0146, and that figure factorises exactly.**
Raise every projected forward by a flat 20 basis points, leave the discount
curve alone, and the ten-year par swap rate rises by 20.29 basis points. The
pass-through is not one for two reasons that pull in opposite directions. The
legs accrue on different conventions — the floating leg on actual/360 and the
fixed leg on 30/360 — so the ratio of their annuities is 1.0191. And a flat
shift to a continuously compounded curve lifts a *simply compounded* quarterly
forward by slightly less than itself, by 0.9956 here. The product of those two
is 1.014643 and the measured pass-through is 1.014643, agreeing to 5.8e-15,
which is the sort of check worth writing down: the decomposition is an identity
rather than a story that happens to come out near the right size.

**A curve that reprices every quote is not the curve that produced them.** The
projection bootstrap recovers quotes to 1e-16 and settles in one sweep under
log-linear discounts and five under monotone convex, which is why the sweep
loop is there at all. Feed it par rates generated from a known curve and ask
how close the result is to that curve: beyond a year, within 0.03 basis points
of zero rate; between the three-month forward quote and the one-year swap,
**5.30 basis points** out at 0.49 years. Nothing in that gap is pinned by a
quote, so the interpolation decides it, and an exact fit to the inputs says
nothing at all about the part of the curve no input touches.

That hole has a cost in the basis curve too, and :class:`IndexForward` is what
closes it. Bootstrapping a six-month projection curve from quoted tenor basis
spreads of 5 to 9 basis points, with nothing pinning its short end, leaves an
implied six-month-over-three-month forward basis of 16.87 basis points over the
first period — spread backwards out of the one-year quote by the interpolation
and three times the quote it came from. Add one forward quote on the long index
at 5.00 basis points and the first period reads 5.00.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from .bootstrap import _bounds, _build, _endpoints
from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import BadCurve, DiscountCurve, Interpolation, OffCurve
from .daycount import Basis, year_fraction
from .rates import Compounding
from .risk import Bucket, KeyRate, Weighting, key_rates, shifted_by, tent_weights
from .schedule import Frequency, Schedule, generate
from .solve import NoRoot, Root, brent

__all__ = [
    "BadForecast",
    "BasisSwap",
    "BucketResponse",
    "DualSwap",
    "DualValuation",
    "FlatLegOf",
    "FloatingLeg",
    "ForecastBootstrapped",
    "ForecastIndex",
    "ForecastQuote",
    "IndexForward",
    "ProjectedCoupon",
    "SplitRisk",
    "SpreadLegOf",
    "bootstrap_forecast",
    "discount_key_rates",
    "forecast_key_rates",
    "forward_spread",
    "projected_forward",
    "shifted_forecast",
    "split_buckets",
    "split_risk",
]


class BadForecast(ValueError):
    """A projected leg, index or quote that does not describe a trade."""


# -- the index ----------------------------------------------------------------


@dataclass(frozen=True)
class ForecastIndex:
    """What a floating leg is projecting: a tenor and the conventions on it.

    The tenor is the *index's* reset frequency, which is also the leg's payment
    frequency for every ordinary money-market index: a leg on three-month money
    resets and pays quarterly. The basis is the index's own, which is
    actual/360 for most of them and actual/365 for sterling, and it is not the
    fixed leg's basis — the two differing is what keeps a basis from passing
    through to a par rate exactly.
    """

    tenor: Frequency = Frequency.QUARTERLY
    basis: Basis = Basis.ACT_360
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    #: Business days between a period's end and the money moving. Zero for an
    #: ordinary money-market leg, which pays on the accrual end -- and that
    #: zero is what the telescoping identity rests on.
    payment_lag: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.payment_lag < 0:
            raise BadForecast(
                f"index {self.name} has a payment lag of {self.payment_lag!r} days. "
                "A lag moves the payment later than the accrual it pays for; a "
                "negative one would pay for a period before it has run."
            )

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        months = self.tenor.months
        return f"{months}m index"

    @property
    def business_days(self) -> Calendar:
        """The calendar to roll on, defaulting to weekends only."""
        return self.calendar if self.calendar is not None else WEEKENDS_ONLY


def projected_forward(
    projection: DiscountCurve, start: date, end: date, basis: Basis
) -> float:
    """The simply compounded forward rate across ``[start, end)``.

    Not :func:`~tenor.rates.forward_rate`, which takes two discount factors and
    two year fractions and answers under a stated compounding. This one takes
    the curve and the two dates, computes the year fraction in the *index's*
    day count, and is simply compounded because that is what a money-market
    index quotes. The two agree where the conventions line up and the reason to
    have both is that an index forward is a statement about an index rather than
    about a curve.

    ``(P(start) / P(end) - 1) / tau``, which is the rate that grows one unit at
    ``start`` into the curve's own implied redemption at ``end``. Negative where
    the curve is upward sloping in discount factors, which is a real state of
    the market and not an error.
    """
    if end <= start:
        raise BadForecast(
            f"a forward rate over {start.isoformat()} to {end.isoformat()} runs "
            "backwards; the end of the period is after its start"
        )
    accrual = year_fraction(start, end, basis)
    if accrual <= 0.0:
        raise BadForecast(
            f"the period {start.isoformat()} to {end.isoformat()} has an accrual "
            f"of {accrual!r} under {basis.value}, so a rate over it is not defined"
        )
    growth = projection.discount(start) / projection.discount(end)
    return (growth - 1.0) / accrual


def forward_spread(
    high: DiscountCurve, low: DiscountCurve, start: date, end: date, basis: Basis
) -> float:
    """``high``'s forward over the period, less ``low``'s. The basis, read off."""
    return projected_forward(high, start, end, basis) - projected_forward(low, start, end, basis)


def shifted_forecast(curve: DiscountCurve, spread: float) -> DiscountCurve:
    """A projection curve whose continuous forwards exceed ``curve``'s by ``spread``.

    Multiplying every discount factor by ``exp(-spread * t)`` adds exactly
    ``spread`` to the instantaneous forward rate at every point, so this is the
    cleanest available statement of "the same curve, plus a flat basis". The
    *simply compounded* forward over a quarter then exceeds the original's by
    slightly more than ``spread``, by the compounding over the period, which is
    half a basis point on a 20 basis point basis and is a real effect rather
    than a construction artefact.
    """
    points = [
        (one.day, one.discount * math.exp(-spread * one.time)) for one in curve.pillars
    ]
    return DiscountCurve.from_discounts(
        curve.reference,
        points,
        basis=curve.basis,
        interpolation=curve.interpolation,
    )


# -- the floating leg ---------------------------------------------------------


@dataclass(frozen=True)
class ProjectedCoupon:
    """One period of a floating leg, and every number that went into it."""

    start: date
    end: date
    payment: date
    accrual: float
    #: The index forward read off the projection curve.
    forward: float
    #: The leg's spread over the index, carried here so the rate is readable.
    spread: float
    #: The discount factor from the *discount* curve, at the payment date.
    discount: float

    @property
    def rate(self) -> float:
        return self.forward + self.spread

    @property
    def cashflow(self) -> float:
        return self.rate * self.accrual

    @property
    def value(self) -> float:
        return self.cashflow * self.discount


@dataclass(frozen=True)
class FloatingLeg:
    """A leg paying an index plus a spread, projected and discounted apart.

    Accrual and the forward are both taken over the *adjusted* period — the
    rolled business days the index actually fixes and accrues over — while
    discounting is at the payment date. The forward and the accrual share one
    year fraction, which is what makes the two cancel when the curves coincide;
    using different ones there would leave a residue proportional to the day
    count difference and nothing in the output would show it.
    """

    effective: date
    maturity: date
    index: ForecastIndex = ForecastIndex()
    spread: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadForecast(
                f"leg {self.name} matures on {self.maturity.isoformat()}, which is "
                f"not after it starts on {self.effective.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"{self.index.name} leg to {self.maturity.isoformat()}"

    def schedule(self) -> Schedule:
        return generate(
            self.effective,
            self.maturity,
            self.index.tenor,
            calendar=self.index.business_days,
            rolling=self.index.rolling,
            payment_lag=self.index.payment_lag,
        )

    def coupons(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> tuple[ProjectedCoupon, ...]:
        """Every period, projected off ``projection`` and discounted off ``discount``."""
        schedule = self.schedule()
        first = schedule[0].adjusted_start
        if first < projection.reference:
            raise BadForecast(
                f"leg {self.name} starts accruing on {first.isoformat()}, before the "
                f"projection curve's reference date {projection.reference.isoformat()}. "
                "The first fixing is then history, which a curve cannot supply; price "
                "it with the known fixing instead."
            )
        last = schedule[-1].payment
        horizon = discount.pillars[-1].day
        if last > horizon:
            lagged = ""
            if last != schedule[-1].adjusted_end:
                lagged = (
                    f" The accrual ends on {schedule[-1].adjusted_end.isoformat()}, "
                    f"which the curve does cover, so it is the "
                    f"{self.index.payment_lag}-business-day payment lag that puts the "
                    "flow outside it."
                )
            raise BadForecast(
                f"leg {self.name} pays last on {last.isoformat()}, past the discount "
                f"curve's last pillar {horizon.isoformat()}.{lagged} A curve quoted "
                "to a leg's maturity does not reach a leg's final payment; quote the "
                "pillar past the payment instead."
            )
        made = []
        for period in schedule:
            accrual = year_fraction(
                period.adjusted_start, period.adjusted_end, self.index.basis
            )
            made.append(
                ProjectedCoupon(
                    start=period.adjusted_start,
                    end=period.adjusted_end,
                    payment=period.payment,
                    accrual=accrual,
                    forward=projected_forward(
                        projection,
                        period.adjusted_start,
                        period.adjusted_end,
                        self.index.basis,
                    ),
                    spread=self.spread,
                    discount=discount.discount(period.payment),
                )
            )
        return tuple(made)

    def value(self, discount: DiscountCurve, projection: DiscountCurve) -> float:
        """The leg's present value, per unit of notional."""
        return sum(one.value for one in self.coupons(discount, projection))

    def annuity(self, discount: DiscountCurve) -> float:
        """What one unit of spread is worth: the accrual-weighted discount sum.

        The projection curve does not appear. A spread is paid on the leg's own
        accrual whatever the index does, so its value is a pure discounting
        question — which is why a basis swap's spread leg is the one with a
        clean sensitivity.
        """
        total = 0.0
        for period in self.schedule():
            accrual = year_fraction(
                period.adjusted_start, period.adjusted_end, self.index.basis
            )
            total += accrual * discount.discount(period.payment)
        return total

    @property
    def first_accrual(self) -> date:
        return self.schedule()[0].adjusted_start

    @property
    def final_payment(self) -> date:
        return self.schedule()[-1].payment


# -- swaps --------------------------------------------------------------------


@dataclass(frozen=True)
class DualSwap:
    """A par swap forecast on one curve and discounted on another.

    The sign convention is the receiver of the floating leg, so
    :meth:`par_error` is the floating leg less the fixed one — the same
    convention :class:`~tenor.instruments.Swap` uses, so the two can be
    subtracted from each other without thinking about it.
    """

    effective: date
    maturity: date
    rate: float
    index: ForecastIndex = ForecastIndex()
    spread: float = 0.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadForecast(
                f"swap {self.name} matures on {self.maturity.isoformat()}, which is "
                f"not after it starts on {self.effective.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"{self.index.name} swap to {self.maturity.isoformat()}"

    @property
    def floating(self) -> FloatingLeg:
        return FloatingLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=self.index,
            spread=self.spread,
            label=f"{self.name} floating leg",
        )

    def fixed_schedule(self) -> Schedule:
        return generate(
            self.effective,
            self.maturity,
            self.frequency,
            calendar=self.calendar if self.calendar is not None else WEEKENDS_ONLY,
            rolling=self.rolling,
        )

    def annuity(self, discount: DiscountCurve) -> float:
        """The fixed leg's annuity: accrual on unadjusted boundaries, discounted
        at the payment date, exactly as the single-curve swap does it."""
        total = 0.0
        for period in self.fixed_schedule():
            accrual = year_fraction(period.start, period.end, self.basis)
            total += accrual * discount.discount(period.payment)
        return total

    def floating_value(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        return self.floating.value(discount, projection)

    def fixed_value(self, discount: DiscountCurve) -> float:
        return self.rate * self.annuity(discount)

    def par_rate(self, discount: DiscountCurve, projection: DiscountCurve) -> float:
        """The fixed rate that makes the swap worth nothing."""
        annuity = self.annuity(discount)
        if annuity <= 0.0:
            raise BadForecast(
                f"swap {self.name} has an annuity of {annuity!r}, so no fixed rate "
                "makes it worth nothing"
            )
        return self.floating_value(discount, projection) / annuity

    def par_error(self, discount: DiscountCurve, projection: DiscountCurve) -> float:
        return self.floating_value(discount, projection) - self.fixed_value(discount)

    def forecast_error(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        return self.par_error(discount, projection)

    @property
    def final_date(self) -> date:
        """The last date any curve has to reach: the later leg's final payment."""
        fixed = self.fixed_schedule()[-1].payment
        return max(fixed, self.floating.final_payment)


@dataclass(frozen=True)
class BasisSwap:
    """Two floating legs on different index tenors, with a spread on one.

    The market quotes a tenor basis as a spread on the shorter-tenor leg
    against the longer one flat, so that is the shape here: ``spread_index``
    carries ``spread`` and ``flat_index`` carries nothing. The sign convention
    is the receiver of the spread leg.

    Both legs discount on the same curve, which is the point of the trade: the
    two legs are collateralised identically, so the basis is a statement about
    the two *projection* curves and nothing else.
    """

    effective: date
    maturity: date
    spread: float
    spread_index: ForecastIndex = ForecastIndex(tenor=Frequency.QUARTERLY)
    flat_index: ForecastIndex = ForecastIndex(tenor=Frequency.SEMI_ANNUAL)
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadForecast(
                f"basis swap {self.name} matures on {self.maturity.isoformat()}, "
                f"which is not after it starts on {self.effective.isoformat()}"
            )
        if self.spread_index.tenor == self.flat_index.tenor:
            raise BadForecast(
                f"basis swap {self.name} has both legs on the "
                f"{self.spread_index.tenor.months}-month tenor. A basis between an "
                "index and itself is zero by construction and carries no "
                "information about either curve."
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"{self.spread_index.tenor.months}m/{self.flat_index.tenor.months}m "
            f"basis to {self.maturity.isoformat()}"
        )

    @property
    def spread_leg(self) -> FloatingLeg:
        return FloatingLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=self.spread_index,
            spread=self.spread,
            label=f"{self.name} spread leg",
        )

    @property
    def flat_leg(self) -> FloatingLeg:
        return FloatingLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=self.flat_index,
            spread=0.0,
            label=f"{self.name} flat leg",
        )

    def par_error(
        self,
        discount: DiscountCurve,
        spread_projection: DiscountCurve,
        flat_projection: DiscountCurve,
    ) -> float:
        return self.spread_leg.value(discount, spread_projection) - self.flat_leg.value(
            discount, flat_projection
        )

    def par_spread(
        self,
        discount: DiscountCurve,
        spread_projection: DiscountCurve,
        flat_projection: DiscountCurve,
    ) -> float:
        """The spread that makes the two legs worth the same.

        Linear in the spread, so this is one division rather than a solve: the
        spread leg is its index projection plus ``spread`` times the leg's own
        annuity, and only the second term moves.
        """
        annuity = self.spread_leg.annuity(discount)
        if annuity <= 0.0:
            raise BadForecast(
                f"basis swap {self.name} has a spread annuity of {annuity!r}, so no "
                "spread makes the legs agree"
            )
        bare = FloatingLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=self.spread_index,
            spread=0.0,
        )
        gap = self.flat_leg.value(discount, flat_projection) - bare.value(
            discount, spread_projection
        )
        return gap / annuity

    @property
    def final_date(self) -> date:
        return max(self.spread_leg.final_payment, self.flat_leg.final_payment)


@dataclass(frozen=True)
class IndexForward:
    """A single forward rate on the index: one period, one quote.

    What a forward rate agreement on the index says, and the only kind of quote
    that pins the short end of a projection curve — a par swap says nothing
    about the curve before its first reset. The discount curve does not enter:
    both sides of the statement are projection-curve discount factors, so this
    quote is the same whatever the swap is collateralised with.
    """

    start: date
    end: date
    rate: float
    basis: Basis = Basis.ACT_360
    label: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise BadForecast(
                f"index forward {self.name} ends on {self.end.isoformat()}, which is "
                f"not after it starts on {self.start.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"forward {self.start.isoformat()}/{self.end.isoformat()}"

    @property
    def accrual(self) -> float:
        return year_fraction(self.start, self.end, self.basis)

    def forecast_error(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        growth = 1.0 + self.rate * self.accrual
        return projection.discount(self.end) * growth - projection.discount(self.start)

    @property
    def final_date(self) -> date:
        return self.end


# -- basis swaps as one-unknown quotes ----------------------------------------


@runtime_checkable
class ForecastQuote(Protocol):
    """Anything that states how far a projection curve is from a market quote.

    The discount curve is an input rather than an unknown, which is the whole
    shape of the multi-curve build: discounting is settled first, from
    overnight-indexed swaps, and every projection curve is solved against it.
    """

    @property
    def name(self) -> str: ...

    @property
    def final_date(self) -> date: ...

    def forecast_error(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float: ...


@dataclass(frozen=True)
class FlatLegOf:
    """A basis swap read as a quote on its flat leg's curve, the other known."""

    swap: BasisSwap
    spread_projection: DiscountCurve

    @property
    def name(self) -> str:
        return f"{self.swap.name} (flat leg)"

    @property
    def final_date(self) -> date:
        return self.swap.final_date

    def forecast_error(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        return self.swap.par_error(discount, self.spread_projection, projection)


@dataclass(frozen=True)
class SpreadLegOf:
    """A basis swap read as a quote on its spread leg's curve, the other known."""

    swap: BasisSwap
    flat_projection: DiscountCurve

    @property
    def name(self) -> str:
        return f"{self.swap.name} (spread leg)"

    @property
    def final_date(self) -> date:
        return self.swap.final_date

    def forecast_error(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        return self.swap.par_error(discount, projection, self.flat_projection)


# -- bootstrapping the projection curve ---------------------------------------


@dataclass(frozen=True)
class ForecastBootstrapped:
    """A projection curve, the discount curve it was solved against, and the fit."""

    curve: DiscountCurve
    discount: DiscountCurve
    quotes: tuple[ForecastQuote, ...]
    sweeps: int
    solutions: tuple[Root, ...]

    def par_errors(self) -> tuple[float, ...]:
        return tuple(
            one.forecast_error(self.discount, self.curve) for one in self.quotes
        )

    @property
    def worst_error(self) -> float:
        return max(abs(error) for error in self.par_errors())

    def reprices(self, tolerance: float = 1e-12) -> bool:
        return self.worst_error <= tolerance

    def __len__(self) -> int:
        return len(self.quotes)


def _ordered_quotes(
    reference: date, quotes: Sequence[ForecastQuote]
) -> list[ForecastQuote]:
    if not quotes:
        raise BadForecast("a projection curve needs at least one quote")
    for one in quotes:
        if one.final_date <= reference:
            raise BadForecast(
                f"quote {one.name} ends on {one.final_date.isoformat()}, on or before "
                f"the reference date {reference.isoformat()}; it has already settled"
            )
    ordered = sorted(quotes, key=lambda one: one.final_date)
    seen: dict[date, str] = {}
    for one in ordered:
        if one.final_date in seen:
            raise BadForecast(
                f"quotes {seen[one.final_date]} and {one.name} both end on "
                f"{one.final_date.isoformat()}. Two quotes pinning one pillar either "
                "agree, in which case one of them is redundant, or disagree, in which "
                "case no curve satisfies both."
            )
        seen[one.final_date] = one.name
    return ordered


def _solve_forecast_pillar(
    *,
    reference: date,
    discount: DiscountCurve,
    days: Sequence[date],
    factors: Sequence[float],
    index: int,
    quote: ForecastQuote,
    basis: Basis,
    interpolation: Interpolation,
    tolerance: float,
) -> Root:
    """Solve for the projection factor at ``days[index]``, holding the rest.

    The bracket is the single-curve bootstrap's own, imported rather than
    restated: the range a discount factor may take given its neighbours is a
    property of the interpolation scheme and has nothing to do with which curve
    is being built, and two copies of that reasoning would drift apart the
    first time one of them was corrected.
    """
    working = list(factors)

    def residual(candidate: float) -> float:
        working[index] = candidate
        curve = _build(reference, days, working, basis, interpolation)
        return quote.forecast_error(discount, curve)

    low, high = _bounds(factors, index, interpolation)
    try:
        low_value, high_value = _endpoints(residual, low, high)
    except (BadCurve, OffCurve) as bad:
        raise BadForecast(
            f"{quote.name} could not be priced while solving for its own forecast "
            f"discount factor: {bad}"
        ) from bad

    if low_value * high_value > 0.0:
        raise BadForecast(
            f"no projection discount factor between {low:.12g} and {high:.12g} "
            f"prices {quote.name} to par: the error is {low_value:.6g} at one end "
            f"and {high_value:.6g} at the other, with no crossing between. That "
            "range is the one the interpolation admits given the neighbouring "
            "pillars, so the quote disagrees with the ones around it rather than "
            "merely being awkward to solve."
        )
    try:
        return brent(residual, low, high, tolerance=min(tolerance, 1e-15))
    except NoRoot as error:  # pragma: no cover - the bracket is checked above
        raise BadForecast(f"{quote.name}: {error}") from error


def bootstrap_forecast(
    reference: date,
    quotes: Sequence[ForecastQuote],
    *,
    discount: DiscountCurve,
    basis: Basis,
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    tolerance: float = 1e-14,
    max_sweeps: int = 50,
) -> ForecastBootstrapped:
    """Solve a projection curve that reprices every quote, discounting on ``discount``.

    Sequential then swept, for the same reason the single-curve bootstrap is:
    under an interpolation that is not local, solving a later pillar moves the
    earlier ones, so one pass leaves the early quotes mispriced and the fit has
    to be iterated to a fixed point.
    """
    if max_sweeps < 1:
        raise BadForecast(
            f"a curve takes at least one sweep to build, got {max_sweeps!r}"
        )
    if discount.reference != reference:
        raise BadForecast(
            f"the discount curve is dated {discount.reference.isoformat()} and the "
            f"projection curve is being built for {reference.isoformat()}. Two curves "
            "on different reference dates cannot discount each other's flows."
        )
    ordered = _ordered_quotes(reference, quotes)
    days = [one.final_date for one in ordered]
    horizon = discount.pillars[-1].day
    if days[-1] > horizon:
        beyond = [one.name for one in ordered if one.final_date > horizon]
        raise BadForecast(
            f"{len(beyond)} of {len(ordered)} quotes settle past the discount "
            f"curve's last pillar {horizon.isoformat()}, the furthest on "
            f"{days[-1].isoformat()}: {', '.join(beyond)}. A projection pillar is "
            "seeded from the discount factor at its own date, so the discount curve "
            "has to reach every quote's final payment -- which is later than its "
            "maturity whenever the index pays with a lag."
        )
    factors = [discount.discount(day) for day in days]
    solutions: list[Root] = [
        Root(factor, math.nan, 0, (factor, factor), converged=False)
        for factor in factors
    ]
    errors = [math.inf] * len(ordered)

    for sweep in range(1, max_sweeps + 1):
        for position, quote in enumerate(ordered):
            extent = position + 1 if sweep == 1 else len(ordered)
            solutions[position] = _solve_forecast_pillar(
                reference=reference,
                discount=discount,
                days=days[:extent],
                factors=factors[:extent],
                index=position,
                quote=quote,
                basis=basis,
                interpolation=interpolation,
                tolerance=tolerance,
            )
            factors[position] = solutions[position].value

        curve = _build(reference, days, factors, basis, interpolation)
        errors = [abs(one.forecast_error(discount, curve)) for one in ordered]
        if max(errors) <= tolerance:
            return ForecastBootstrapped(
                curve=curve,
                discount=discount,
                quotes=tuple(ordered),
                sweeps=sweep,
                solutions=tuple(solutions),
            )

    worst = max(range(len(ordered)), key=lambda position: errors[position])
    raise BadForecast(
        f"the projection curve did not settle after {max_sweeps} sweeps. The worst "
        f"remaining error is {errors[worst]:.3e} on {ordered[worst].name}, against a "
        f"tolerance of {tolerance:.3e}."
    )


# -- risk, split between the two curves ---------------------------------------

#: Anything that turns a *pair* of curves into a number: discount first,
#: projection second, in the order every method in this module takes them.
DualValuation = Callable[[DiscountCurve, DiscountCurve], float]


def discount_key_rates(
    value: DualValuation,
    discount: DiscountCurve,
    projection: DiscountCurve,
    buckets: Sequence[Bucket],
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> tuple[KeyRate, ...]:
    """Key rates against the discount curve, holding the projection curve still.

    A key rate duration is a *relative* sensitivity, so this needs a valuation
    that is worth something: a par swap is worth nothing and
    :func:`~tenor.risk.key_rates` refuses it, correctly. For a position at par
    the money responses in :func:`split_buckets` are the thing that exists.
    """
    return key_rates(
        lambda curve: value(curve, projection),
        discount,
        buckets,
        shift=shift,
        compounding=compounding,
    )


def forecast_key_rates(
    value: DualValuation,
    discount: DiscountCurve,
    projection: DiscountCurve,
    buckets: Sequence[Bucket],
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> tuple[KeyRate, ...]:
    """Key rates against the projection curve, holding the discount curve still."""
    return key_rates(
        lambda curve: value(discount, curve),
        projection,
        buckets,
        shift=shift,
        compounding=compounding,
    )


@dataclass(frozen=True)
class SplitRisk:
    """A valuation's response to each curve moving, and to both together.

    :attr:`crossed` is the amount by which the two separate responses fail to
    add up to the joint one. It is the second derivative's contribution and it
    is not zero: a swap's value is bilinear in the two curves, so shifting both
    at once is not the sum of shifting each. On the ten-year swap in the tests
    it is about 0.3% of the joint response at a basis point and it scales with
    the square of the shift, which is the test that identifies it as the cross
    term rather than as an error.
    """

    discount: float
    forecast: float
    joint: float

    @property
    def crossed(self) -> float:
        return self.joint - self.discount - self.forecast


def _shifted(
    curve: DiscountCurve,
    amount: float,
    weight: Weighting,
    compounding: Compounding,
) -> DiscountCurve:
    return shifted_by(curve, amount, weight, compounding)


def _money_response(
    value: DualValuation,
    discount: DiscountCurve,
    projection: DiscountCurve,
    weight: Weighting,
    *,
    shift: float,
    compounding: Compounding,
    which: str,
) -> float:
    """A central difference in money, not in duration, so par positions work."""
    if which == "discount":
        up = value(_shifted(discount, shift, weight, compounding), projection)
        down = value(_shifted(discount, -shift, weight, compounding), projection)
    elif which == "forecast":
        up = value(discount, _shifted(projection, shift, weight, compounding))
        down = value(discount, _shifted(projection, -shift, weight, compounding))
    else:
        up = value(
            _shifted(discount, shift, weight, compounding),
            _shifted(projection, shift, weight, compounding),
        )
        down = value(
            _shifted(discount, -shift, weight, compounding),
            _shifted(projection, -shift, weight, compounding),
        )
    return (up - down) / 2.0


def split_risk(
    value: DualValuation,
    discount: DiscountCurve,
    projection: DiscountCurve,
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> SplitRisk:
    """Central differences in the discount curve, the projection curve and both.

    In money rather than in years, because the position this is wanted for is
    usually a swap at or near par, whose value is too close to zero for a
    relative sensitivity to mean anything.
    """
    if shift <= 0.0:
        raise BadForecast(
            f"a risk shift of {shift!r} is not positive; a central difference needs "
            "a step to take"
        )
    def flat(_time: float) -> float:
        return 1.0

    return SplitRisk(
        discount=_money_response(
            value,
            discount,
            projection,
            flat,
            shift=shift,
            compounding=compounding,
            which="discount",
        ),
        forecast=_money_response(
            value,
            discount,
            projection,
            flat,
            shift=shift,
            compounding=compounding,
            which="forecast",
        ),
        joint=_money_response(
            value,
            discount,
            projection,
            flat,
            shift=shift,
            compounding=compounding,
            which="joint",
        ),
    )


@dataclass(frozen=True)
class BucketResponse:
    """One bucket's money response on each curve separately."""

    bucket: Bucket
    discount: float
    forecast: float


def split_buckets(
    value: DualValuation,
    discount: DiscountCurve,
    projection: DiscountCurve,
    buckets: Sequence[Bucket],
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> tuple[BucketResponse, ...]:
    """Attribute each curve's money response across ``buckets``.

    The tent shapes add to one at every pillar, so each curve's bucket
    responses add back to that curve's parallel response in :func:`split_risk`
    — exactly in the shifts, and to the accuracy of the central difference in
    the responses. Both halves of that are tested.
    """
    weights = tent_weights([one.time for one in buckets])
    made = []
    for bucket, weight in zip(buckets, weights, strict=True):
        made.append(
            BucketResponse(
                bucket=bucket,
                discount=_money_response(
                    value,
                    discount,
                    projection,
                    weight,
                    shift=shift,
                    compounding=compounding,
                    which="discount",
                ),
                forecast=_money_response(
                    value,
                    discount,
                    projection,
                    weight,
                    shift=shift,
                    compounding=compounding,
                    which="forecast",
                ),
            )
        )
    return tuple(made)
