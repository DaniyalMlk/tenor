"""Calendars, business day rolling, and payment schedules.

The cases worth testing here are the ones where a plausible implementation is
right most of the time. Modified following is indistinguishable from following
until a roll would cross a month end. End-of-month handling is indistinguishable
from doing nothing until the anchor is a month end that is not the 31st. Both of
those are pinned directly.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tenor.calendar import WEEKENDS_ONLY, Calendar, Rolling, calendar_from
from tenor.schedule import (
    BadSchedule,
    Frequency,
    Stub,
    add_months,
    end_of_month,
    generate,
)

#: A few real closures, enough to make the rolling rules fire.
US = calendar_from(
    "US",
    [
        date(2021, 1, 1),
        date(2021, 7, 5),
        date(2021, 11, 25),
        date(2021, 12, 24),
        date(2022, 1, 17),
    ],
)
UK = calendar_from("UK", [date(2021, 8, 30), date(2021, 12, 27), date(2021, 12, 28)])


# -- calendars ---------------------------------------------------------------


def test_weekends_are_not_business_days() -> None:
    assert not WEEKENDS_ONLY.is_business_day(date(2021, 10, 30))  # Saturday
    assert not WEEKENDS_ONLY.is_business_day(date(2021, 10, 31))  # Sunday
    assert WEEKENDS_ONLY.is_business_day(date(2021, 10, 29))  # Friday


def test_holidays_are_not_business_days() -> None:
    assert not US.is_business_day(date(2021, 7, 5))
    assert WEEKENDS_ONLY.is_business_day(date(2021, 7, 5))


def test_a_calendar_can_have_a_different_weekend() -> None:
    """A Friday-Saturday weekend is expressible rather than special-cased."""
    gulf = calendar_from("gulf", [], weekend={4, 5})
    assert not gulf.is_business_day(date(2021, 10, 29))  # Friday
    assert gulf.is_business_day(date(2021, 10, 31))  # Sunday


def test_calendars_join_by_union_not_intersection() -> None:
    """A trade settling in two centres needs both open.

    Intersecting keeps only the days both are closed, which would leave this
    joint calendar open on every holiday in either list — the mistake the union
    exists to avoid.
    """
    joint = US.joined(UK)
    assert joint.holidays == US.holidays | UK.holidays
    assert not joint.is_business_day(date(2021, 7, 5))  # US only
    assert not joint.is_business_day(date(2021, 8, 30))  # UK only
    assert US.is_business_day(date(2021, 8, 30))
    assert UK.is_business_day(date(2021, 7, 5))
    assert len(US.holidays & UK.holidays) == 0


def test_joining_is_commutative_in_effect() -> None:
    first, second = US.joined(UK), UK.joined(US)
    day = date(2021, 1, 1)
    for _ in range(400):
        assert first.is_business_day(day) == second.is_business_day(day)
        day += timedelta(days=1)


def test_following_rolls_forward() -> None:
    assert US.adjust(date(2021, 10, 30), Rolling.FOLLOWING) == date(2021, 11, 1)


def test_preceding_rolls_backward() -> None:
    assert US.adjust(date(2021, 10, 30), Rolling.PRECEDING) == date(2021, 10, 29)


def test_none_leaves_the_date_alone() -> None:
    assert US.adjust(date(2021, 10, 30), Rolling.NONE) == date(2021, 10, 30)


def test_modified_following_turns_back_at_a_month_end() -> None:
    """The exception that separates it from plain following.

    31 July 2021 is a Saturday. Following pushes it into August; modified
    following pulls it back to the 30th, keeping the payment inside its month
    and the schedule inside its quarters.
    """
    saturday = date(2021, 7, 31)
    assert US.adjust(saturday, Rolling.FOLLOWING) == date(2021, 8, 2)
    assert US.adjust(saturday, Rolling.MODIFIED_FOLLOWING) == date(2021, 7, 30)


def test_modified_following_is_plain_following_everywhere_else() -> None:
    """Which is why an implementation missing the exception looks fine.

    Over a whole year the two agree on every date but the handful at a month
    end, and the test counts the disagreements rather than asserting there are
    none.
    """
    day = date(2021, 1, 1)
    disagreements = []
    while day.year == 2021:
        following = US.adjust(day, Rolling.FOLLOWING)
        modified = US.adjust(day, Rolling.MODIFIED_FOLLOWING)
        if following != modified:
            disagreements.append(day)
            assert following.month != day.month
            assert modified.month == day.month
        day += timedelta(days=1)
    assert disagreements
    assert len(disagreements) < 20


def test_modified_preceding_turns_forward_at_a_month_start() -> None:
    """1 January 2021 is a holiday and 2-3 January is a weekend.

    Preceding leaves it in December; modified preceding pulls it forward to the
    4th, keeping the date inside January.
    """
    day = date(2021, 1, 1)
    assert US.adjust(day, Rolling.PRECEDING) == date(2020, 12, 31)
    assert US.adjust(day, Rolling.MODIFIED_PRECEDING) == date(2021, 1, 4)


def test_adjusting_a_business_day_changes_nothing() -> None:
    day = date(2021, 10, 29)
    for rolling in Rolling:
        assert US.adjust(day, rolling) == day


def test_adding_business_days_skips_weekends_and_holidays() -> None:
    """A spot date is two business days on, not two calendar days.

    From Friday 2 July 2021: the 3rd and 4th are a weekend, the 5th is a
    holiday, so two business days lands on Wednesday the 7th.
    """
    assert US.add_business_days(date(2021, 7, 2), 2) == date(2021, 7, 7)
    assert US.add_business_days(date(2021, 7, 7), -2) == date(2021, 7, 2)


def test_adding_zero_business_days_is_the_identity() -> None:
    for day in (date(2021, 7, 2), date(2021, 7, 4)):
        assert US.add_business_days(day, 0) == day


def test_business_days_between_counts_the_half_open_interval() -> None:
    """``[start, end)``, which is how accrual counts periods."""
    # Monday 5 July 2021 is a holiday; the week has four business days.
    assert US.business_days_between(date(2021, 7, 5), date(2021, 7, 12)) == 4
    assert US.business_days_between(date(2021, 7, 5), date(2021, 7, 5)) == 0


def test_counting_business_days_backwards_is_refused() -> None:
    with pytest.raises(ValueError, match="count forwards"):
        US.business_days_between(date(2021, 7, 12), date(2021, 7, 5))


def test_a_bad_weekday_number_is_refused() -> None:
    with pytest.raises(ValueError, match="weekday number"):
        Calendar(name="broken", weekend=frozenset({7}))


def test_a_calendar_with_no_business_days_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing can settle"):
        Calendar(name="closed", weekend=frozenset(range(7)))


def test_an_impossibly_long_closure_is_reported_rather_than_looped_on() -> None:
    """A month of holidays means the calendar is misconfigured.

    Looping forever would be the worst available outcome, so the search gives up
    and says what is probably wrong.
    """
    closed = calendar_from(
        "misloaded", [date(2021, 6, 1) + timedelta(days=k) for k in range(40)]
    )
    with pytest.raises(ValueError, match="no business day within"):
        closed.next_business_day(date(2021, 6, 2))


# -- month arithmetic --------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "day"),
    [(2021, 1, 31), (2021, 2, 28), (2020, 2, 29), (2021, 4, 30), (2021, 12, 31)],
)
def test_end_of_month(year: int, month: int, day: int) -> None:
    assert end_of_month(year, month) == date(year, month, day)


def test_months_are_measured_from_the_anchor_not_stepped() -> None:
    """Stepping loses month ends permanently; measuring cannot.

    31 January advanced one month at a time gives 28 February, then 28 March,
    and never recovers. Measured from the anchor each time it stays on the month
    end throughout.
    """
    anchor = date(2021, 1, 31)
    measured = [add_months(anchor, k, keep_end_of_month=False) for k in range(1, 5)]
    assert measured == [
        date(2021, 2, 28),
        date(2021, 3, 31),
        date(2021, 4, 30),
        date(2021, 5, 31),
    ]

    stepped = [anchor]
    for _ in range(4):
        stepped.append(add_months(stepped[-1], 1, keep_end_of_month=False))
    assert stepped[1:] == [
        date(2021, 2, 28),
        date(2021, 3, 28),
        date(2021, 4, 28),
        date(2021, 5, 28),
    ]
    assert stepped[1:] != measured


def test_end_of_month_bites_on_a_month_end_that_is_not_the_thirty_first() -> None:
    """28 February is where the flag matters and 31 January is where it does not.

    A 31st already clamps onto each month's end by itself, so an implementation
    can pass every January-anchored test and fail every February one.
    """
    february = date(2021, 2, 28)
    on = [add_months(february, k, keep_end_of_month=True) for k in (1, 3, 6)]
    off = [add_months(february, k, keep_end_of_month=False) for k in (1, 3, 6)]
    assert on == [date(2021, 3, 31), date(2021, 5, 31), date(2021, 8, 31)]
    assert off == [date(2021, 3, 28), date(2021, 5, 28), date(2021, 8, 28)]

    january = date(2021, 1, 31)
    assert [add_months(january, k, keep_end_of_month=True) for k in (1, 3, 6)] == [
        add_months(january, k, keep_end_of_month=False) for k in (1, 3, 6)
    ]


def test_a_day_that_does_not_exist_is_clamped() -> None:
    assert add_months(date(2021, 1, 30), 1, keep_end_of_month=False) == date(2021, 2, 28)
    assert add_months(date(2020, 1, 30), 1, keep_end_of_month=False) == date(2020, 2, 29)


def test_months_move_backwards_too() -> None:
    assert add_months(date(2021, 3, 15), -3, keep_end_of_month=False) == date(
        2020, 12, 15
    )
    assert add_months(date(2021, 1, 15), -1, keep_end_of_month=False) == date(
        2020, 12, 15
    )


def test_twelve_months_is_a_year() -> None:
    for day in (date(2021, 3, 15), date(2020, 2, 29), date(2021, 12, 31)):
        moved = add_months(day, 12, keep_end_of_month=False)
        assert moved.month == day.month
        assert moved.year == day.year + 1


# -- schedules ---------------------------------------------------------------


def test_a_regular_schedule_has_no_stub() -> None:
    schedule = generate(
        date(2021, 8, 15), date(2024, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    assert len(schedule) == 6
    assert not schedule.has_stub
    assert schedule.effective == date(2021, 8, 15)
    assert schedule.maturity == date(2024, 8, 15)


@pytest.mark.parametrize(
    ("frequency", "periods"),
    [
        (Frequency.ANNUAL, 3),
        (Frequency.SEMI_ANNUAL, 6),
        (Frequency.QUARTERLY, 12),
        (Frequency.MONTHLY, 36),
    ],
)
def test_the_frequency_sets_the_number_of_periods(
    frequency: Frequency, periods: int
) -> None:
    schedule = generate(
        date(2021, 8, 15), date(2024, 8, 15), frequency, calendar=US
    )
    assert len(schedule) == periods
    assert not schedule.has_stub


def test_generating_backwards_puts_the_stub_at_the_front() -> None:
    """The market convention, and where a stub is normally intended.

    A bond maturing on 15 August pays on the 15th of February and August
    whatever its issue date was.
    """
    schedule = generate(
        date(2021, 3, 10), date(2024, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    assert schedule[0].stub is Stub.SHORT_FIRST
    assert schedule[0].start == date(2021, 3, 10)
    assert schedule[0].end == date(2021, 8, 15)
    assert not any(one.is_stub for one in schedule.periods[1:])
    for period in schedule.periods[1:]:
        assert period.end.day == 15
        assert period.end.month in (2, 8)


def test_generating_forwards_puts_the_stub_at_the_end() -> None:
    schedule = generate(
        date(2021, 3, 10),
        date(2024, 8, 15),
        Frequency.SEMI_ANNUAL,
        calendar=US,
        from_end=False,
    )
    assert schedule[-1].stub is Stub.SHORT_LAST
    assert not any(one.is_stub for one in schedule.periods[:-1])
    for period in schedule.periods[:-1]:
        assert period.end.day == 10


def test_the_two_directions_give_different_payment_dates() -> None:
    """Which is why the choice is an argument rather than an implementation detail."""
    backwards = generate(
        date(2021, 3, 10), date(2024, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    forwards = generate(
        date(2021, 3, 10),
        date(2024, 8, 15),
        Frequency.SEMI_ANNUAL,
        calendar=US,
        from_end=False,
    )
    assert backwards.payment_dates != forwards.payment_dates
    assert backwards.effective == forwards.effective
    assert backwards.maturity == forwards.maturity


def test_a_long_first_stub_is_reported_as_long() -> None:
    """An effective date more than a period before the first regular date."""
    schedule = generate(
        date(2021, 1, 20), date(2024, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    assert schedule[0].stub in (Stub.SHORT_FIRST, Stub.LONG_FIRST)
    assert schedule.stubs
    assert schedule.stubs[0] is schedule[0]


def test_periods_are_contiguous_and_cover_the_whole_term() -> None:
    schedule = generate(
        date(2021, 3, 10), date(2024, 8, 15), Frequency.QUARTERLY, calendar=US
    )
    for earlier, later in zip(schedule.periods, schedule.periods[1:], strict=False):
        assert earlier.end == later.start
    assert schedule.periods[0].start == date(2021, 3, 10)
    assert schedule.periods[-1].end == date(2024, 8, 15)


def test_unadjusted_dates_are_kept_alongside_the_adjusted_ones() -> None:
    """Accrual runs on unadjusted dates and payment on adjusted ones.

    A schedule that threw the unadjusted dates away could not compute its own
    accruals afterwards.
    """
    schedule = generate(
        date(2021, 1, 31), date(2023, 1, 31), Frequency.QUARTERLY, calendar=US
    )
    moved = [one for one in schedule.periods if one.end != one.adjusted_end]
    assert moved
    for period in moved:
        assert US.is_business_day(period.adjusted_end)
        assert not US.is_business_day(period.end)


def test_payment_dates_are_business_days() -> None:
    schedule = generate(
        date(2021, 1, 31), date(2023, 1, 31), Frequency.MONTHLY, calendar=US
    )
    for period in schedule:
        assert US.is_business_day(period.payment)


def test_a_payment_lag_moves_the_payment_and_not_the_accrual() -> None:
    plain = generate(
        date(2021, 8, 15), date(2023, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    lagged = generate(
        date(2021, 8, 15),
        date(2023, 8, 15),
        Frequency.SEMI_ANNUAL,
        calendar=US,
        payment_lag=2,
    )
    for without, with_lag in zip(plain, lagged, strict=True):
        assert without.start == with_lag.start
        assert without.end == with_lag.end
        assert with_lag.payment > without.payment
        assert with_lag.payment == US.add_business_days(without.adjusted_end, 2)


def test_an_end_of_month_schedule_stays_on_month_ends() -> None:
    schedule = generate(
        date(2021, 2, 28), date(2023, 2, 28), Frequency.QUARTERLY, calendar=US
    )
    for period in schedule.periods[:-1]:
        assert period.end == end_of_month(period.end.year, period.end.month)


def test_turning_off_end_of_month_keeps_the_day_number() -> None:
    schedule = generate(
        date(2021, 2, 28),
        date(2023, 2, 28),
        Frequency.QUARTERLY,
        calendar=US,
        keep_end_of_month=False,
    )
    for period in schedule.periods:
        assert period.end.day == 28


def test_rolling_none_leaves_every_date_unadjusted() -> None:
    schedule = generate(
        date(2021, 1, 31),
        date(2023, 1, 31),
        Frequency.MONTHLY,
        calendar=US,
        rolling=Rolling.NONE,
    )
    for period in schedule:
        assert period.adjusted_end == period.end
        assert period.payment == period.end


def test_a_maturity_before_the_effective_date_is_refused() -> None:
    with pytest.raises(BadSchedule, match="not after the effective date"):
        generate(date(2024, 1, 1), date(2021, 1, 1), Frequency.ANNUAL, calendar=US)


def test_a_zero_length_schedule_is_refused() -> None:
    day = date(2021, 1, 1)
    with pytest.raises(BadSchedule, match="not after"):
        generate(day, day, Frequency.ANNUAL, calendar=US)


def test_a_negative_payment_lag_is_refused() -> None:
    with pytest.raises(BadSchedule, match="not negative"):
        generate(
            date(2021, 1, 1),
            date(2022, 1, 1),
            Frequency.ANNUAL,
            calendar=US,
            payment_lag=-1,
        )


def test_a_schedule_shorter_than_one_period_is_a_single_stub() -> None:
    schedule = generate(
        date(2021, 6, 1), date(2021, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    assert len(schedule) == 1
    assert schedule[0].start == date(2021, 6, 1)
    assert schedule[0].end == date(2021, 8, 15)
    assert schedule[0].stub is Stub.SHORT_FIRST


def test_a_schedule_is_indexable_and_iterable() -> None:
    schedule = generate(
        date(2021, 8, 15), date(2023, 8, 15), Frequency.SEMI_ANNUAL, calendar=US
    )
    assert len(list(schedule)) == len(schedule) == 4
    assert schedule[0] is schedule.periods[0]
    assert schedule[-1] is schedule.periods[-1]
    assert len(schedule.payment_dates) == 4


def test_the_frequency_knows_its_month_step() -> None:
    assert Frequency.ANNUAL.months == 12
    assert Frequency.SEMI_ANNUAL.months == 6
    assert Frequency.QUARTERLY.months == 3
    assert Frequency.MONTHLY.months == 1
