"""Business days, holiday calendars, and rolling a date off a weekend.

A payment date that falls on a Saturday is not a payment date. Moving it is the
whole of this module, and the rule chosen for the move changes the accrual
period, which changes the coupon — so the convention is an argument here too.

**Modified following is the default nowhere and the answer almost everywhere.**
Plain following rolls forward to the next business day, which is right until it
pushes a payment into the next month and the period lengths stop matching the
schedule. Modified following rolls forward unless that crosses a month end, in
which case it rolls back instead. That one exception is what keeps a quarterly
schedule inside its quarters, and an implementation missing it produces
schedules that are correct for years and then are not.

**Calendars compose by union, not by intersection.** A cross-currency trade
settles only when both centres are open, so the joint calendar holds every
holiday of each. Intersecting them — keeping only the days both are closed —
is the intuitive-sounding mistake and produces a calendar with almost no
holidays at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class Rolling(str, Enum):
    """What to do with a date that is not a business day."""

    #: Leave it where it is. For conventions that accrue on calendar days.
    NONE = "none"
    #: Forward to the next business day.
    FOLLOWING = "following"
    #: Forward, unless that crosses into the next month; then backward.
    MODIFIED_FOLLOWING = "modified following"
    #: Backward to the previous business day.
    PRECEDING = "preceding"
    #: Backward, unless that crosses into the previous month; then forward.
    MODIFIED_PRECEDING = "modified preceding"


@dataclass(frozen=True)
class Calendar:
    """Weekends and a set of holidays.

    Weekend days are given as ``date.weekday()`` values, so a Gulf calendar
    running Friday-Saturday is expressible rather than special-cased. Defaults
    to Saturday and Sunday because that is what the great majority of the
    world's settlement runs on, and the default is visible in the constructor
    rather than buried.
    """

    name: str = "weekends"
    holidays: frozenset[date] = field(default_factory=frozenset)
    weekend: frozenset[int] = field(default_factory=lambda: frozenset({5, 6}))

    def __post_init__(self) -> None:
        for day in self.weekend:
            if not 0 <= day <= 6:
                raise ValueError(
                    f"{day!r} is not a weekday number; they run 0 (Monday) to 6 (Sunday)"
                )
        if len(self.weekend) >= 7:
            raise ValueError(
                f"{self.name}: every day of the week is a weekend, so no date is ever "
                "a business day and nothing can settle"
            )

    def is_business_day(self, day: date) -> bool:
        return day.weekday() not in self.weekend and day not in self.holidays

    def joined(self, other: Calendar) -> Calendar:
        """The calendar on which both centres are open.

        A union of the holidays, because a trade settling in two centres needs
        both open. Intersecting is the intuitive-sounding mistake: it keeps only
        the days *both* are closed and so produces a calendar that is open on
        almost every holiday either centre observes.
        """
        return Calendar(
            name=f"{self.name}+{other.name}",
            holidays=self.holidays | other.holidays,
            weekend=self.weekend | other.weekend,
        )

    def next_business_day(self, day: date) -> date:
        moved = day
        for _ in range(_GUARD):
            moved += timedelta(days=1)
            if self.is_business_day(moved):
                return moved
        raise ValueError(_stuck(self, day))

    def previous_business_day(self, day: date) -> date:
        moved = day
        for _ in range(_GUARD):
            moved -= timedelta(days=1)
            if self.is_business_day(moved):
                return moved
        raise ValueError(_stuck(self, day))

    def adjust(self, day: date, rolling: Rolling) -> date:
        """Move ``day`` to a business day under ``rolling``."""
        if rolling is Rolling.NONE or self.is_business_day(day):
            return day
        if rolling is Rolling.FOLLOWING:
            return self.next_business_day(day)
        if rolling is Rolling.PRECEDING:
            return self.previous_business_day(day)
        if rolling is Rolling.MODIFIED_FOLLOWING:
            forward = self.next_business_day(day)
            return forward if forward.month == day.month else self.previous_business_day(day)
        backward = self.previous_business_day(day)
        return backward if backward.month == day.month else self.next_business_day(day)

    def add_business_days(self, day: date, count: int) -> date:
        """Move ``count`` business days from ``day``, which need not be one.

        Used for settlement lags: a spot date is two business days after the
        trade, counted in business days and not in calendar days plus a fudge.
        """
        moved = day
        step = 1 if count >= 0 else -1
        for _ in range(abs(count)):
            moved = (
                self.next_business_day(moved)
                if step > 0
                else self.previous_business_day(moved)
            )
        return moved

    def business_days_between(self, start: date, end: date) -> int:
        """Business days in ``[start, end)``, which is how accrual counts them."""
        if end < start:
            raise ValueError(
                f"{end.isoformat()} is before {start.isoformat()}; count forwards"
            )
        total = 0
        day = start
        while day < end:
            if self.is_business_day(day):
                total += 1
            day += timedelta(days=1)
        return total


#: Enough days to step over any plausible run of closures. A calendar that
#: cannot find a business day within a fortnight is misconfigured rather than
#: unusual, and looping forever on it would be the worst available outcome.
_GUARD = 14


def _stuck(calendar: Calendar, day: date) -> str:
    return (
        f"{calendar.name}: no business day within {_GUARD} days of "
        f"{day.isoformat()}. The calendar has a run of holidays longer than any "
        "real closure, which usually means holidays were loaded for the wrong "
        "year or the weekend was set wrongly."
    )


def calendar_from(
    name: str, holidays: Iterable[date], *, weekend: Iterable[int] | None = None
) -> Calendar:
    """Build a calendar from any iterable of dates."""
    return Calendar(
        name=name,
        holidays=frozenset(holidays),
        weekend=frozenset(weekend) if weekend is not None else frozenset({5, 6}),
    )


#: Weekends only. Useful as a base to join real holidays onto, and honest about
#: being nothing more than that.
WEEKENDS_ONLY = Calendar()


__all__ = [
    "WEEKENDS_ONLY",
    "Calendar",
    "Rolling",
    "calendar_from",
]
