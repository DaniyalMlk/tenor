"""Payment schedules: the dates a coupon accrues between and is paid on.

A schedule looks like a loop over months and is not. Three decisions inside it
change the cash flows, and all three are invisible in the output:

**Which end it is generated from.** Backwards from maturity is the market
convention and it is not a stylistic choice. A bond maturing on 15 August paying
semi-annually pays on the 15th of February and August whatever its issue date
was; generating forwards from issue puts the payments on the issue day of the
month and leaves the odd period at the *end*, next to redemption, which is the
one place a stub is least often intended.

**End of month.** If the anchor is the last day of its month, every subsequent
date should be the last day of its month: 28 February to 31 March to 30 April,
not to the 28th of every month thereafter. The rule bites exactly where the
anchor is a month end that is not the 31st, since a 31st already clamps onto
each month's end by itself — so an implementation can look correct on every
January-anchored test and be wrong on every February one.

Separately, dates are always measured from the original anchor rather than by
stepping a month at a time. Stepping loses month ends permanently: 31 January
advanced one month at a time gives 28 February, then 28 March, and never
recovers.

**Stubs.** An odd first or last period is a fact about the trade, not a rounding
error, so it is reported on the schedule rather than left for the reader to spot
by comparing period lengths. A stub that nobody noticed is a coupon that does
not match the counterparty's.

The unadjusted dates are kept alongside the adjusted ones. Accrual under the
thirty-day conventions is computed on unadjusted dates while payment happens on
the adjusted ones, and a schedule that throws the unadjusted dates away cannot
compute its own accruals afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from .calendar import Calendar, Rolling


class Frequency(int, Enum):
    """Payments per year."""

    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12

    @property
    def months(self) -> int:
        return 12 // int(self)


class Stub(str, Enum):
    """Where an odd period sits, if there is one."""

    NONE = "none"
    SHORT_FIRST = "short first"
    LONG_FIRST = "long first"
    SHORT_LAST = "short last"
    LONG_LAST = "long last"


class BadSchedule(ValueError):
    """The dates cannot form a schedule."""


def end_of_month(year: int, month: int) -> date:
    """The last day of a month, without importing a calendar library."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def add_months(anchor: date, months: int, *, keep_end_of_month: bool) -> date:
    """Move a date by whole months.

    **Always measured from the original anchor, never by stepping a month at a
    time.** Iterating is the obvious implementation and it loses month ends
    permanently: 31 January stepped forward one month at a time gives 28
    February, then 28 March, 28 April, and every date after the first short
    month is wrong. Measuring ``n`` months from the anchor each time cannot do
    that, whatever ``keep_end_of_month`` is set to.

    ``keep_end_of_month`` decides what a month-end anchor means. With it, an
    anchor on the last day of its month lands on the last day of every target
    month: 28 February plus one is 31 March, plus six is 31 August. Without it,
    the day number is kept: 28 March, 28 August.

    It therefore has no effect at all for an anchor on the 31st, since the
    clamp to the target month's length already lands on the month end — which
    is why testing the flag on a 31st shows nothing and testing it on a 28
    February shows everything.

    A day that does not exist in the target month is clamped to that month's
    last day regardless, because there is no other sensible answer.
    """
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    last = end_of_month(year, month).day
    if keep_end_of_month and anchor.day == end_of_month(anchor.year, anchor.month).day:
        return date(year, month, last)
    return date(year, month, min(anchor.day, last))


@dataclass(frozen=True)
class Period:
    """One accrual period and the date it pays on."""

    #: Unadjusted period boundaries. Accrual under the thirty-day conventions is
    #: computed on these, not on the adjusted ones.
    start: date
    end: date
    #: The same boundaries rolled onto business days.
    adjusted_start: date
    adjusted_end: date
    #: When the money moves, which is the adjusted end unless a lag was applied.
    payment: date
    #: Whether this period is the odd one out, and which kind.
    stub: Stub = Stub.NONE

    @property
    def is_stub(self) -> bool:
        return self.stub is not Stub.NONE


@dataclass(frozen=True)
class Schedule:
    """The full set of accrual periods for an instrument."""

    periods: tuple[Period, ...]
    frequency: Frequency
    rolling: Rolling
    calendar: Calendar

    def __len__(self) -> int:
        return len(self.periods)

    def __iter__(self) -> Iterator[Period]:
        return iter(self.periods)

    def __getitem__(self, index: int) -> Period:
        return self.periods[index]

    @property
    def effective(self) -> date:
        return self.periods[0].start

    @property
    def maturity(self) -> date:
        return self.periods[-1].end

    @property
    def payment_dates(self) -> tuple[date, ...]:
        return tuple(one.payment for one in self.periods)

    @property
    def stubs(self) -> tuple[Period, ...]:
        """Every odd period, reported rather than left to be noticed."""
        return tuple(one for one in self.periods if one.is_stub)

    @property
    def has_stub(self) -> bool:
        return bool(self.stubs)


def generate(
    effective: date,
    maturity: date,
    frequency: Frequency,
    *,
    calendar: Calendar,
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING,
    keep_end_of_month: bool = True,
    payment_lag: int = 0,
    from_end: bool = True,
) -> Schedule:
    """Build a payment schedule between two dates.

    Generated backwards from ``maturity`` by default, which is the market
    convention: a bond pays on the anniversary of its redemption, and any odd
    period lands at the front where a stub is normally intended. ``from_end``
    false generates forwards from ``effective`` instead, leaving the stub at the
    end, which is occasionally what a term sheet says and never what it means by
    accident.

    ``payment_lag`` is in business days after the adjusted period end.
    """
    if maturity <= effective:
        raise BadSchedule(
            f"maturity {maturity.isoformat()} is not after the effective date "
            f"{effective.isoformat()}; a schedule needs a period to cover"
        )
    if payment_lag < 0:
        raise BadSchedule(f"a payment lag is not negative, got {payment_lag!r}")

    boundaries = (
        _backwards(effective, maturity, frequency, keep_end_of_month)
        if from_end
        else _forwards(effective, maturity, frequency, keep_end_of_month)
    )
    regular = frequency.months
    periods = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        adjusted_start = calendar.adjust(start, rolling)
        adjusted_end = calendar.adjust(end, rolling)
        payment = (
            calendar.add_business_days(adjusted_end, payment_lag)
            if payment_lag
            else adjusted_end
        )
        periods.append(
            Period(
                start=start,
                end=end,
                adjusted_start=adjusted_start,
                adjusted_end=adjusted_end,
                payment=payment,
                stub=_stub_kind(
                    start,
                    end,
                    regular,
                    keep_end_of_month,
                    first=index == 0,
                    last=index == len(boundaries) - 2,
                    from_end=from_end,
                ),
            )
        )
    return Schedule(
        periods=tuple(periods),
        frequency=frequency,
        rolling=rolling,
        calendar=calendar,
    )


def _backwards(
    effective: date, maturity: date, frequency: Frequency, keep_end_of_month: bool
) -> list[date]:
    dates = [maturity]
    step = 1
    while True:
        previous = add_months(
            maturity, -frequency.months * step, keep_end_of_month=keep_end_of_month
        )
        if previous <= effective:
            break
        dates.append(previous)
        step += 1
    dates.append(effective)
    dates.reverse()
    return dates


def _forwards(
    effective: date, maturity: date, frequency: Frequency, keep_end_of_month: bool
) -> list[date]:
    dates = [effective]
    step = 1
    while True:
        following = add_months(
            effective, frequency.months * step, keep_end_of_month=keep_end_of_month
        )
        if following >= maturity:
            break
        dates.append(following)
        step += 1
    dates.append(maturity)
    return dates


def _stub_kind(
    start: date,
    end: date,
    regular_months: int,
    keep_end_of_month: bool,
    *,
    first: bool,
    last: bool,
    from_end: bool,
) -> Stub:
    """Classify a period against the regular length for this frequency.

    Only the end the schedule was *not* generated from can carry a stub, so the
    others are not examined — which matters because month-length variation makes
    a regular period fail a naive day-count comparison about a third of the
    time.
    """
    if from_end and not first:
        return Stub.NONE
    if not from_end and not last:
        return Stub.NONE
    expected = add_months(end, -regular_months, keep_end_of_month=keep_end_of_month)
    if from_end:
        if start == expected:
            return Stub.NONE
        return Stub.SHORT_FIRST if start > expected else Stub.LONG_FIRST
    expected_end = add_months(start, regular_months, keep_end_of_month=keep_end_of_month)
    if end == expected_end:
        return Stub.NONE
    return Stub.SHORT_LAST if end < expected_end else Stub.LONG_LAST


__all__ = [
    "BadSchedule",
    "Frequency",
    "Period",
    "Schedule",
    "Stub",
    "add_months",
    "end_of_month",
    "generate",
]
