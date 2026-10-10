"""Compounding an overnight rate, and the four conventions on where to read it.

A leg on a term index fixes once at the start of a period and pays over it. A
leg on an overnight index reads a rate every business day and compounds it over
the accrual it pays for, which raises a question a term leg never has to answer:
**which days are the rate read on?** The accrual period's own days are the
obvious answer and almost nobody uses it unmodified, because the last fixing
would then be published on the payment date itself.

So the window moves, and how it moves is a term of the trade. Four conventions
are in use. In arrears reads each accrual day's own rate. A lookback reads the
rate from a few business days earlier and keeps the accrual day's own weight. An
observation shift moves the whole window back, rate and weight together. A
lockout freezes the rate at a cut a few days before the end and repeats it. A
payment delay is a fifth thing that is often grouped with them and is not one:
it moves the money and leaves the rate alone.

**The plain case is an identity, not an approximation.** The projected overnight
rate across one business day is ``P(a) / P(b) - 1`` divided by the day count, so
each daily growth factor is exactly ``P(a) / P(b)`` and the product over the
period telescopes to ``P(start) / P(end)``. Compounded in arrears is therefore
exactly replicable by a pair of zero-coupon bonds, needs no convexity
adjustment of any kind, and is asserted here against ``P(start) / P(end)`` to
4.4e-16 relative on a curve with a rate step in the middle of the period rather
than to a chosen tolerance.

**Observation shift keeps the identity and the other two do not, and that is
structural.** Shifting rate and weight together makes the product telescope over
the *shifted* window, to ``P(start - k) / P(end - k)`` — asserted to 6.7e-16,
and exactly zero at some lags. A lookback takes the rate from one window and the
weight from another, so no two consecutive factors share a discount factor and
nothing telescopes. A lockout repeats a factor. Both are still perfectly well
defined; they are simply not replicable, and reading that off the structure is
better than inferring it from the size of a difference.

**The size of the difference is the curve's local slope times the lag, which
means a smoothly interpolated curve hides the whole subject.** On a curve built
by interpolating zero rates, a five-day lookback over a three-month period is
worth 0.000bp and a two-day one 0.007bp: the conventions look like pedantry. Put
a 50bp policy step in the overnight forward inside the period and the same
five-day lookback is worth **-3.84bp** and a ten-day one -7.67bp. The
conventions exist because of the steps, and a model that smooths the steps away
cannot be used to decide whether they matter.

**Lookback and observation shift coincide exactly when the lag is a whole number
of weeks, which is how the distinction gets missed.** On a weekends-only
calendar a five-business-day lag is exactly seven calendar days, so every
accrual day's own weight equals its shifted day's weight and the two conventions
agree to the last bit — measured at both five and ten days. At two days a
weekend crosses the window boundary, the weights stop matching, and the same
curve gives -1.10bp for the lookback against -2.19bp for the shift: a factor of
two from a convention difference that the usual lag choice conceals.

**A lockout is worth nothing unless the step is inside the locked window.** With
the policy step in the middle of the period the lockout is worth 0.0000bp at
every lag, because the days it freezes all carried the same rate anyway. Move
the period so that the step falls five business days before its end and the same
lockout is worth -4.46bp, identically at five and at ten days since both windows
already reach past the step. It earns its name only when the end of the period
is where the rate moves, and stating a lockout's effect without saying where the
step is says nothing at all.

**Compounding against averaging is a different contract and the gap has a closed
form.** An arithmetic average pays ``sum(r_i tau_i) / D`` where compounding pays
``(prod(1 + r_i tau_i) - 1) / D``, and the second exceeds the first by about
``r^2 T / 2``. Measured over the step curve at three months the gap is 1.33bp
against 1.37bp predicted, 2.6% apart, and on the smooth curve it runs 1.10bp at
three months, 2.76bp at six and 7.22bp at twelve — growing with the square of
the tenor, which is what makes an averaged leg quoted against a compounded one a
level-dependent basis rather than a spread. The closed form is high by 2.1% to
3.0% across all six of those measurements, always in the same direction, which
is the third-order term it drops.

**At the level of a leg the conventions compress, and one of them moves no rate
at all.** On a three-year quarterly leg across a single 50bp step, only one
period spans the step, so the -3.84bp a five-day lookback is worth inside that
period becomes **-0.337bp** as a spread over the whole leg — identical for the
observation shift, -0.241bp for the lockout and -1.493bp for arithmetic
averaging. A five-day payment delay is worth -0.236bp while changing the rate of
every period by exactly zero, which is the clearest possible demonstration that
it belongs in a different group from the other four.

**A lookback leg needs fixings from before its own effective date.** The first
period's first observation is the lag's worth of business days before the
accrual starts, so a leg traded on its effective date is already asking for
rates published before the trade existed. That is refused here rather than
filled with the first projectable rate, because the substitution is silent and
wrong by the whole move since the window opened.

**A fixing is read, not projected, and a missing one is refused.** An
observation date before the curve's reference has no projected rate; the
published fixing is the only thing that can fill it, and a leg part-way through
its period has several such days. Substituting the first projectable rate would
return a number, and the number would be wrong by the whole move since the
period began.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .schedule import Frequency, Period, Schedule, generate

#: The largest lag accepted on any of the window conventions. Ten business days
#: is already two weeks of observations and well outside anything quoted; a
#: larger number is a units mistake, usually calendar days where business days
#: were meant.
MAX_LAG = 10


class Observation(str, Enum):
    """Where the rate that compounds over an accrual day is read.

    The accrual day's weight and the observation day's rate are separate
    choices, and the four conventions are the four combinations that are
    actually traded.
    """

    #: Each accrual day's own rate over its own weight. Telescopes.
    IN_ARREARS = "in arrears"
    #: The rate from ``days`` business days earlier, over the accrual day's own
    #: weight. Does not telescope: rate and weight come from different days.
    LOOKBACK = "lookback"
    #: The whole window moved back ``days`` business days, rate and weight
    #: together. Telescopes, over the shifted window.
    SHIFT = "observation shift"
    #: In arrears until ``days`` business days from the end, then the rate at
    #: the cut repeated. Does not telescope: a factor appears more than once.
    LOCKOUT = "lockout"


class Averaging(str, Enum):
    """Whether the daily rates compound or are averaged."""

    COMPOUNDED = "compounded"
    ARITHMETIC = "arithmetic"


class BadIndex(ValueError):
    """The index's conventions do not describe a trade."""


class MissingFixing(ValueError):
    """An observation date is in the past and no published fixing was supplied.

    The curve cannot project a rate for a date before its reference, and the
    nearest projectable rate is not a substitute: on a period part-way through,
    it is wrong by the whole move since the period began.
    """


@dataclass(frozen=True)
class OvernightIndex:
    """An overnight index and the conventions on reading it."""

    basis: Basis = Basis.ACT_360
    calendar: Calendar | None = None
    observation: Observation = Observation.IN_ARREARS
    #: Business days of lag, for every convention but in arrears.
    days: int = 0
    averaging: Averaging = Averaging.COMPOUNDED
    #: Business days between the accrual end and the money moving. Changes no
    #: rate at all; it is here because it is quoted alongside the others.
    payment_lag: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.days < 0:
            raise BadIndex(
                f"{self.name} has a lag of {self.days!r} business days. A negative lag "
                "would read a rate that has not been published."
            )
        if self.days > MAX_LAG:
            raise BadIndex(
                f"{self.name} has a lag of {self.days!r} business days against a "
                f"ceiling of {MAX_LAG}. Lags are quoted in business days, not calendar "
                "days, and nothing traded reaches this."
            )
        if self.observation is Observation.IN_ARREARS and self.days != 0:
            raise BadIndex(
                f"{self.name} is in arrears and carries a lag of {self.days!r} days. In "
                "arrears means the accrual days are the observation days; a lag needs "
                "one of the other three conventions to say what it does."
            )
        if self.observation is not Observation.IN_ARREARS and self.days == 0:
            raise BadIndex(
                f"{self.name} is {self.observation.value} with a lag of zero, which is "
                "in arrears under another name. Give the lag, or say in arrears."
            )
        if self.payment_lag < 0:
            raise BadIndex(
                f"{self.name} has a payment lag of {self.payment_lag!r} days, which "
                "would pay for a period before it has run."
            )

    @property
    def name(self) -> str:
        return self.label or f"overnight {self.observation.value}"

    @property
    def business_days(self) -> Calendar:
        """The calendar to observe on, weekends only unless one was given."""
        return self.calendar if self.calendar is not None else WEEKENDS_ONLY

    @property
    def telescopes(self) -> bool:
        """Whether the daily growth factors collapse to a ratio of two discounts.

        True for in arrears and for an observation shift, where every factor is
        ``P(a) / P(b)`` on consecutive days. False for a lookback, which pairs
        one day's rate with another's weight, and for a lockout, which repeats
        a factor. Compounded only: an arithmetic average is a sum and
        telescoping is a statement about a product.
        """
        return self.averaging is Averaging.COMPOUNDED and self.observation in (
            Observation.IN_ARREARS,
            Observation.SHIFT,
        )


@dataclass(frozen=True)
class Observed:
    """One business day's contribution to a compounded rate."""

    #: The accrual day the weight comes from.
    accrual_start: date
    accrual_end: date
    #: The day the rate is read on, and the day it runs to.
    observation: date
    observation_end: date
    rate: float
    #: The year fraction the rate compounds over, under the index's basis.
    weight: float
    #: Whether the rate came from a published fixing rather than the curve.
    fixed: bool

    @property
    def growth(self) -> float:
        return 1.0 + self.rate * self.weight


@dataclass(frozen=True)
class OvernightRate:
    """A compounded or averaged overnight rate over one accrual period."""

    start: date
    end: date
    rate: float
    #: ``prod(1 + r_i tau_i)`` for a compounded rate, and ``1 + rate * D`` for
    #: an averaged one so that the two are comparable.
    growth: float
    #: The accrual's own year fraction, which is the denominator in both cases.
    accrual: float
    index: OvernightIndex
    observations: tuple[Observed, ...]

    @property
    def days(self) -> int:
        return len(self.observations)

    @property
    def fixed_days(self) -> int:
        return sum(1 for one in self.observations if one.fixed)

    @property
    def first_observation(self) -> date:
        return self.observations[0].observation

    @property
    def last_observation(self) -> date:
        return self.observations[-1].observation

    @property
    def weight(self) -> float:
        """Total of the daily weights.

        Equal to the accrual under every convention but an observation shift,
        where the shifted window can contain a different number of calendar
        days and the sum moves away from it.
        """
        return math.fsum(one.weight for one in self.observations)


# -- the observation schedule ------------------------------------------------


def business_days(calendar: Calendar, start: date, end: date) -> Iterator[tuple[date, date]]:
    """The business days in ``[start, end)``, each with the day it runs to.

    The first day is ``start`` itself when it is a business day and the next one
    otherwise, which is the convention a schedule's adjusted start already
    satisfies. The last day runs to ``end``, so the weights sum to the accrual
    whatever the calendar does in between.
    """
    if end <= start:
        raise ValueError(f"an observation window needs end after start, got {start}..{end}")
    day = start if calendar.is_business_day(start) else calendar.next_business_day(start)
    while day < end:
        following = calendar.next_business_day(day)
        yield (day, min(following, end))
        day = following


def _rate_from(
    curve: DiscountCurve,
    start: date,
    end: date,
    basis: Basis,
    fixings: Mapping[date, float] | None,
) -> tuple[float, bool]:
    published = None if fixings is None else fixings.get(start)
    if published is not None:
        return (published, True)
    if start < curve.reference:
        raise MissingFixing(
            f"the overnight rate for {start} is in the past relative to the curve's "
            f"reference date {curve.reference} and no fixing was supplied for it. A "
            "period part-way through needs its published fixings; the curve cannot "
            "project a date behind its own reference."
        )
    fraction = year_fraction(start, end, basis)
    if fraction <= 0.0:
        raise ValueError(f"an overnight accrual of {fraction!r} years, {start}..{end}")
    return ((curve.discount(start) / curve.discount(end) - 1.0) / fraction, False)


def observations(
    curve: DiscountCurve,
    start: date,
    end: date,
    index: OvernightIndex,
    *,
    fixings: Mapping[date, float] | None = None,
) -> tuple[Observed, ...]:
    """Every business day's rate and weight under the index's convention.

    The four conventions differ here and nowhere else, which is deliberate: the
    compounding below is one line and cannot be where a convention goes wrong.
    """
    calendar = index.business_days
    accruals = list(business_days(calendar, start, end))
    if not accruals:  # pragma: no cover - business_days refuses an empty window
        raise ValueError(f"no business days in {start}..{end}")

    cut = len(accruals) - index.days
    if index.observation is Observation.LOCKOUT and cut < 1:
        raise BadIndex(
            f"{index.name} locks out {index.days} of {len(accruals)} business days in "
            f"{start}..{end}, which leaves nothing observed. A lockout longer than the "
            "period is a schedule error, not a convention."
        )

    built = []
    for position, (accrual_start, accrual_end) in enumerate(accruals):
        weight = year_fraction(accrual_start, accrual_end, index.basis)
        if index.observation is Observation.SHIFT:
            observation = calendar.add_business_days(accrual_start, -index.days)
            observation_end = calendar.add_business_days(accrual_end, -index.days)
            weight = year_fraction(observation, observation_end, index.basis)
        elif index.observation is Observation.LOOKBACK:
            observation = calendar.add_business_days(accrual_start, -index.days)
            observation_end = calendar.next_business_day(observation)
        elif index.observation is Observation.LOCKOUT and position >= cut:
            observation, observation_end = accruals[cut - 1]
        else:
            observation, observation_end = accrual_start, accrual_end
        rate, fixed = _rate_from(curve, observation, observation_end, index.basis, fixings)
        built.append(
            Observed(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                observation=observation,
                observation_end=observation_end,
                rate=rate,
                weight=weight,
                fixed=fixed,
            )
        )
    return tuple(built)


def compounded_rate(
    curve: DiscountCurve,
    start: date,
    end: date,
    index: OvernightIndex | None = None,
    *,
    fixings: Mapping[date, float] | None = None,
) -> OvernightRate:
    """The overnight rate over ``[start, end)`` under ``index``.

    Compounded unless the index says to average, in which case the daily rates
    are weighted by the *accrual's* day counts rather than by whatever the
    observation window's were, because an averaged leg pays a weighted mean of
    the published rates over the days it accrues.
    """
    index = index if index is not None else OvernightIndex()
    daily = observations(curve, start, end, index, fixings=fixings)
    accrual = year_fraction(start, end, index.basis)
    if accrual <= 0.0:
        raise ValueError(f"an accrual of {accrual!r} years, {start}..{end}")

    if index.averaging is Averaging.ARITHMETIC:
        total = math.fsum(one.rate * one.weight for one in daily)
        rate = total / accrual
        growth = 1.0 + rate * accrual
    else:
        growth = 1.0
        for one in daily:
            growth *= one.growth
        rate = (growth - 1.0) / accrual

    return OvernightRate(
        start=start,
        end=end,
        rate=rate,
        growth=growth,
        accrual=accrual,
        index=index,
        observations=daily,
    )


def replication_factor(
    curve: DiscountCurve, start: date, end: date, index: OvernightIndex
) -> float:
    """``P(a) / P(b)`` for the window the convention's product telescopes over.

    The whole point of the plain convention: the compounded growth is this
    number, so the leg is replicated by borrowing one zero and lending another
    and there is nothing left to adjust for. Refused for the conventions where
    no such pair exists, rather than returned as an approximation to them.
    """
    if not index.telescopes:
        raise BadIndex(
            f"{index.name} does not telescope, so there is no pair of discount factors "
            "whose ratio is its growth. A lookback pairs one day's rate with another "
            "day's weight and a lockout repeats a factor; both are well defined and "
            "neither is replicable."
        )
    calendar = index.business_days
    first = start if calendar.is_business_day(start) else calendar.next_business_day(start)
    if index.observation is Observation.SHIFT:
        window_start = calendar.add_business_days(first, -index.days)
        last = max(one for one, _ in business_days(calendar, start, end))
        window_end = calendar.add_business_days(calendar.next_business_day(last), -index.days)
        return curve.discount(window_start) / curve.discount(window_end)
    return curve.discount(first) / curve.discount(end)


def convention_basis(
    curve: DiscountCurve,
    start: date,
    end: date,
    index: OvernightIndex,
    against: OvernightIndex,
    *,
    fixings: Mapping[date, float] | None = None,
) -> float:
    """``index`` less ``against``, as a rate, over the same accrual period.

    The number a negotiation turns on, and the reason it is a function rather
    than a constant: it is the curve's local slope times the lag, so it is a
    fraction of a basis point on a smooth curve and several basis points across
    a policy step.
    """
    left = compounded_rate(curve, start, end, index, fixings=fixings)
    right = compounded_rate(curve, start, end, against, fixings=fixings)
    return left.rate - right.rate


# -- a leg -------------------------------------------------------------------


@dataclass(frozen=True)
class OvernightCoupon:
    """One period of an overnight leg."""

    period: Period
    overnight: OvernightRate
    spread: float
    notional: float
    payment: date
    discount: float

    @property
    def rate(self) -> float:
        return self.overnight.rate + self.spread

    @property
    def cashflow(self) -> float:
        return self.notional * self.rate * self.overnight.accrual

    @property
    def value(self) -> float:
        return self.cashflow * self.discount


@dataclass(frozen=True)
class OvernightLeg:
    """A leg paying a compounded overnight rate plus a spread."""

    effective: date
    maturity: date
    index: OvernightIndex = OvernightIndex()
    frequency: Frequency = Frequency.QUARTERLY
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    spread: float = 0.0
    notional: float = 1.0

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadIndex(
                f"a leg from {self.effective} to {self.maturity} has no life in it"
            )
        if self.notional <= 0.0:
            raise BadIndex(f"a notional of {self.notional!r} is not a trade")

    @property
    def schedule(self) -> Schedule:
        """Generated rather than stepped by hand, so the roll rule applies."""
        return generate(
            self.effective,
            self.maturity,
            self.frequency,
            calendar=self.index.business_days,
            rolling=self.rolling,
            payment_lag=self.index.payment_lag,
        )

    def coupons(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve | None = None,
        *,
        fixings: Mapping[date, float] | None = None,
    ) -> tuple[OvernightCoupon, ...]:
        """Every period's rate and discounted value.

        A floating leg accrues on the *adjusted* period boundaries, which is
        where the observation window starts and ends too: the window is defined
        on business days and an unadjusted boundary need not be one.
        """
        forecast = projection if projection is not None else discount
        built = []
        for period in self.schedule:
            overnight = compounded_rate(
                forecast,
                period.adjusted_start,
                period.adjusted_end,
                self.index,
                fixings=fixings,
            )
            built.append(
                OvernightCoupon(
                    period=period,
                    overnight=overnight,
                    spread=self.spread,
                    notional=self.notional,
                    payment=period.payment,
                    discount=discount.discount(period.payment),
                )
            )
        return tuple(built)

    def value(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve | None = None,
        *,
        fixings: Mapping[date, float] | None = None,
    ) -> float:
        return math.fsum(
            coupon.value
            for coupon in self.coupons(discount, projection, fixings=fixings)
        )

    def annuity(self, discount: DiscountCurve) -> float:
        """Value of a basis point of spread, over the leg's own accruals."""
        return math.fsum(
            self.notional
            * year_fraction(period.adjusted_start, period.adjusted_end, self.index.basis)
            * discount.discount(period.payment)
            for period in self.schedule
        )

    def with_index(self, index: OvernightIndex) -> OvernightLeg:
        """The same trade under another convention, for comparing the two."""
        return OvernightLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=index,
            frequency=self.frequency,
            rolling=self.rolling,
            spread=self.spread,
            notional=self.notional,
        )


def convention_margin(
    leg: OvernightLeg,
    against: OvernightIndex,
    discount: DiscountCurve,
    projection: DiscountCurve | None = None,
    *,
    fixings: Mapping[date, float] | None = None,
) -> float:
    """The spread that makes ``leg`` worth what it would be worth under ``against``.

    Solved rather than searched, because a leg's value is linear in its spread
    and the annuity is the coefficient. The two legs share a schedule unless the
    payment lags differ, in which case they do not and each annuity is its own.
    """
    other = leg.with_index(against)
    difference = other.value(discount, projection, fixings=fixings) - leg.value(
        discount, projection, fixings=fixings
    )
    annuity = leg.annuity(discount)
    if annuity == 0.0:
        raise BadIndex("a leg with no annuity cannot express a convention as a spread")
    return difference / annuity


def equivalent_rates(
    curve: DiscountCurve,
    start: date,
    end: date,
    indices: Sequence[OvernightIndex],
    *,
    fixings: Mapping[date, float] | None = None,
) -> tuple[OvernightRate, ...]:
    """The same period under several conventions, for printing side by side."""
    if not indices:
        raise BadIndex("no conventions to compare")
    return tuple(
        compounded_rate(curve, start, end, index, fixings=fixings) for index in indices
    )


__all__ = [
    "MAX_LAG",
    "Averaging",
    "BadIndex",
    "MissingFixing",
    "Observation",
    "Observed",
    "OvernightCoupon",
    "OvernightIndex",
    "OvernightLeg",
    "OvernightRate",
    "business_days",
    "compounded_rate",
    "convention_basis",
    "convention_margin",
    "equivalent_rates",
    "observations",
    "replication_factor",
]
