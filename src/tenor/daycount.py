"""Day count conventions: the fraction of a year between two dates.

This is the least interesting arithmetic in fixed income and the most
consequential. Every accrual, every discount factor and every yield runs through
a year fraction, and the conventions disagree by enough to move a price by
several basis points — which is the size of the spread anybody is trying to
measure in the first place. A bond priced on the wrong basis is not
approximately right.

The disagreements are not subtle once written down. Between 1 January and 1 July
of a non-leap year:

    ACT/360      181 / 360 = 0.502778
    ACT/365F     181 / 365 = 0.495890
    30/360       180 / 360 = 0.500000

That is 69 basis points of a year on a six-month period, and nothing in the
number says which convention produced it. So there is no default anywhere in
this library. A basis is an argument, spelled the way the market spells it, so
the call site reads like the term sheet.

**"30/360" names three different rules and they give three different answers.**
From 28 February 2021 to 31 August 2021:

    30/360 Bond Basis (ISDA 4.16(f))     183 days
    30E/360 Eurobond  (ISDA 4.16(g))     182 days
    30/360 US (SIA, with the EOM rules)  180 days

Three days apart on a six-month accrual, between conventions that are all
written "30/360" on a screen. They are separate members of the enumeration here
rather than one with a flag, because a flag invites a default and there is no
right one.

The rules are transcribed in the order they are applied, which matters. Bond
Basis adjusts a second date of 31 only when the first date is already past the
29th; US rules reach that state through the end-of-February adjustment and
Eurobond rules never condition on the first date at all. An implementation that
adjusts both dates independently is the Eurobond rule wearing another name.

**ACT/ACT ISDA is not a single ratio.** It splits the period at each year
boundary and divides each part by the length of its own year, so a period
spanning a leap year gets both 365 and 366 as denominators. Treating it as
actual days over 365.25, or over the length of the start year, is wrong by up to
a day of accrual, in a direction that depends on where in the calendar the trade
sits.
"""

from __future__ import annotations

from datetime import date
from enum import Enum


class Basis(str, Enum):
    """A day count convention, spelled as the market spells it."""

    #: Actual days over 360. Money markets almost everywhere, and USD swap
    #: floating legs.
    ACT_360 = "ACT/360"
    #: Actual days over 365, fixed, leap year or not. Sterling and much of Asia.
    ACT_365F = "ACT/365F"
    #: Actual days, each part of the period over the length of its own year.
    #: ISDA 2006 section 4.16(b); most fixed swap legs.
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    #: Thirty-day months, ISDA 2006 section 4.16(f). Also called Bond Basis or
    #: 360/360. No end-of-February rule.
    THIRTY_360_BOND = "30/360 Bond Basis"
    #: Thirty-day months, ISDA 2006 section 4.16(g). Also called Eurobond Basis.
    #: Both dates capped at 30, independently, and February is left alone.
    THIRTY_E_360 = "30E/360"
    #: Thirty-day months, US municipal and corporate convention, with the
    #: end-of-February rules the other two do not have.
    THIRTY_360_US = "30/360 US"


class BadPeriod(ValueError):
    """The two dates do not form a period this convention can measure."""


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_year(year: int) -> int:
    return 366 if is_leap(year) else 365


def is_end_of_february(day: date) -> bool:
    if day.month != 2:
        return False
    return day.day == (29 if is_leap(day.year) else 28)


def year_fraction(start: date, end: date, basis: Basis) -> float:
    """The fraction of a year between ``start`` and ``end`` under ``basis``.

    ``end`` before ``start`` is refused rather than returned negative. A negative
    accrual is always a mistake in the caller — usually a schedule generated
    backwards — and returning one lets it travel a long way before anything
    notices.
    """
    if end < start:
        raise BadPeriod(
            f"{end.isoformat()} is before {start.isoformat()}; a year fraction runs "
            "forwards. A negative accrual is a mistake in the schedule rather than a "
            "quantity, so it is refused here instead of propagating."
        )
    if basis is Basis.ACT_360:
        return (end - start).days / 360.0
    if basis is Basis.ACT_365F:
        return (end - start).days / 365.0
    if basis is Basis.ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    return _thirty_days(start, end, basis) / 360.0


def _act_act_isda(start: date, end: date) -> float:
    """Actual days, each calendar year measured against its own length.

    The period is cut at 1 January of every year it crosses. A period inside one
    year is the ordinary ratio; a period crossing a leap boundary picks up 365
    in one part and 366 in the other, which is the point of the convention and
    the thing a single-denominator shortcut cannot reproduce.
    """
    if start.year == end.year:
        return (end - start).days / days_in_year(start.year)
    total = (date(start.year + 1, 1, 1) - start).days / days_in_year(start.year)
    total += float(end.year - start.year - 1)
    total += (end - date(end.year, 1, 1)).days / days_in_year(end.year)
    return total


def _thirty_days(start: date, end: date, basis: Basis) -> int:
    """Days between two dates under one of the thirty-day-month conventions."""
    first, second = start.day, end.day

    if basis is Basis.THIRTY_E_360:
        # ISDA 4.16(g). Both capped at 30, independently, February untouched.
        first = min(first, 30)
        second = min(second, 30)
    elif basis is Basis.THIRTY_360_BOND:
        # ISDA 4.16(f). The second date is adjusted only when the first is
        # already past the 29th, which after the first adjustment means it was
        # a 30th or a 31st. No end-of-February rule at all: that absence is
        # what separates this from the US convention and is worth three days on
        # a February-to-August accrual.
        if first == 31:
            first = 30
        if second == 31 and first > 29:
            second = 30
    else:
        # US convention. The end-of-February rules come first, and they are what
        # pull the first date up to the 30th, which then lets the 31st rule fire
        # on the second date where Bond Basis would leave it alone.
        if is_end_of_february(start) and is_end_of_february(end):
            second = 30
        if is_end_of_february(start):
            first = 30
        if first == 31:
            first = 30
        if second == 31 and first == 30:
            second = 30

    return (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + (second - first)
    )


def day_count(start: date, end: date, basis: Basis) -> int:
    """The numerator of the year fraction: days counted under ``basis``.

    Exposed because accrued interest on a bond is quoted as a number of days,
    and a reader checking a price against a screen needs the same integer the
    screen used rather than a fraction they have to reverse.
    """
    if end < start:
        raise BadPeriod(
            f"{end.isoformat()} is before {start.isoformat()}; a day count runs "
            "forwards"
        )
    if basis in (Basis.ACT_360, Basis.ACT_365F, Basis.ACT_ACT_ISDA):
        return (end - start).days
    return _thirty_days(start, end, basis)


def denominator(start: date, end: date, basis: Basis) -> float:
    """The annual denominator a quoted fraction implies, so it can be checked.

    For ACT/ACT ISDA there is no single denominator — that is the convention's
    defining feature — so this reports the effective one, which is what a reader
    reconciling against a screen actually needs.
    """
    fraction = year_fraction(start, end, basis)
    if fraction == 0.0:
        raise BadPeriod(
            f"{start.isoformat()} to {end.isoformat()} is a zero-length period under "
            f"{basis.value}, so it has no implied denominator"
        )
    return day_count(start, end, basis) / fraction


__all__ = [
    "BadPeriod",
    "Basis",
    "day_count",
    "days_in_year",
    "denominator",
    "is_end_of_february",
    "is_leap",
    "year_fraction",
]
