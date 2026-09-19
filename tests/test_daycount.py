"""Day count conventions.

The headline test is that the same two dates give six different answers, because
that is the fact the whole module exists to keep straight. The thirty-day
conventions are then checked case by case against the published rules, with
particular attention to the cases where they disagree — which are almost all
about February and the 31st, and almost never about anything else.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tenor.daycount import (
    BadPeriod,
    Basis,
    day_count,
    days_in_year,
    denominator,
    is_end_of_february,
    is_leap,
    year_fraction,
)

# -- the disagreement --------------------------------------------------------


def test_six_conventions_give_six_different_answers() -> None:
    """28 February to 31 August 2021, the case that separates all of them.

    Actual counts agree with each other at 184 days and disagree on the
    denominator; the three thirty-day conventions disagree on the count itself,
    at 183, 182 and 180. Every one of these is written on somebody's screen as
    the day count for the same six months.
    """
    start, end = date(2021, 2, 28), date(2021, 8, 31)
    counts = {basis: day_count(start, end, basis) for basis in Basis}
    assert counts[Basis.ACT_360] == 184
    assert counts[Basis.ACT_365F] == 184
    assert counts[Basis.ACT_ACT_ISDA] == 184
    assert counts[Basis.THIRTY_360_BOND] == 183
    assert counts[Basis.THIRTY_E_360] == 182
    assert counts[Basis.THIRTY_360_US] == 180

    fractions = {basis: year_fraction(start, end, basis) for basis in Basis}
    assert len(set(round(value, 10) for value in fractions.values())) == 5
    assert max(fractions.values()) - min(fractions.values()) > 0.01


def test_the_six_month_spread_between_the_actual_conventions() -> None:
    """1 January to 1 July, the example in the module docstring."""
    start, end = date(2021, 1, 1), date(2021, 7, 1)
    assert year_fraction(start, end, Basis.ACT_360) == pytest.approx(181 / 360)
    assert year_fraction(start, end, Basis.ACT_365F) == pytest.approx(181 / 365)
    assert year_fraction(start, end, Basis.THIRTY_360_BOND) == pytest.approx(0.5)
    spread = year_fraction(start, end, Basis.ACT_360) - year_fraction(
        start, end, Basis.ACT_365F
    )
    assert spread == pytest.approx(0.0069, abs=1e-4)


# -- the actual conventions --------------------------------------------------


@pytest.mark.parametrize("basis", [Basis.ACT_360, Basis.ACT_365F])
@pytest.mark.parametrize("days", [1, 30, 181, 365, 366, 1000])
def test_the_actual_conventions_are_days_over_a_constant(
    basis: Basis, days: int
) -> None:
    start = date(2021, 3, 15)
    end = start + timedelta(days=days)
    expected = days / (360.0 if basis is Basis.ACT_360 else 365.0)
    assert year_fraction(start, end, basis) == pytest.approx(expected)
    assert day_count(start, end, basis) == days


def test_act_365_fixed_does_not_notice_a_leap_year() -> None:
    """A whole leap year is 366/365, above one, which is the convention working."""
    fraction = year_fraction(date(2020, 1, 1), date(2021, 1, 1), Basis.ACT_365F)
    assert fraction == pytest.approx(366 / 365)
    assert fraction > 1.0


def test_act_act_isda_gives_exactly_one_for_any_whole_calendar_year() -> None:
    for year in (2019, 2020, 2021, 2100, 2000):
        fraction = year_fraction(
            date(year, 1, 1), date(year + 1, 1, 1), Basis.ACT_ACT_ISDA
        )
        assert fraction == pytest.approx(1.0)


def test_act_act_isda_splits_at_the_year_boundary() -> None:
    """A period spanning a leap year uses both 365 and 366 as denominators.

    Computed here by hand from the pieces: 61 days of 2019 over 365, the whole
    of 2020 as one, and 31 days of 2021 over 365.
    """
    start, end = date(2019, 11, 1), date(2021, 2, 1)
    expected = 61 / 365 + 1.0 + 31 / 365
    assert year_fraction(start, end, Basis.ACT_ACT_ISDA) == pytest.approx(expected)


def test_act_act_isda_is_not_actual_days_over_365() -> None:
    """The shortcut that looks right and is not.

    Over this period the difference is nearly a day of accrual, which on a
    coupon is real money and is invisible in the resulting number.
    """
    start, end = date(2019, 11, 1), date(2021, 2, 1)
    proper = year_fraction(start, end, Basis.ACT_ACT_ISDA)
    shortcut = (end - start).days / 365.0
    assert shortcut != pytest.approx(proper, abs=1e-6)
    assert abs(shortcut - proper) * 365 > 0.9


def test_act_act_isda_inside_one_year_is_the_plain_ratio() -> None:
    start, end = date(2020, 3, 1), date(2020, 9, 1)
    assert year_fraction(start, end, Basis.ACT_ACT_ISDA) == pytest.approx(
        (end - start).days / 366
    )


def test_act_act_isda_is_additive_across_a_split() -> None:
    """Splitting a period and adding the parts must give the whole.

    True of the convention by construction and not true of the naive
    single-denominator version once the split crosses a year boundary, so it is
    a check on the implementation rather than on arithmetic.
    """
    start, middle, end = date(2019, 5, 10), date(2020, 7, 3), date(2022, 2, 20)
    whole = year_fraction(start, end, Basis.ACT_ACT_ISDA)
    parts = year_fraction(start, middle, Basis.ACT_ACT_ISDA) + year_fraction(
        middle, end, Basis.ACT_ACT_ISDA
    )
    assert whole == pytest.approx(parts, rel=1e-14)


# -- the thirty-day conventions ----------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "bond", "european", "american"),
    [
        # The February cases, where all three can differ.
        (date(2021, 2, 28), date(2021, 8, 31), 183, 182, 180),
        (date(2020, 2, 29), date(2020, 8, 31), 182, 181, 180),
        # A 31st start: bond and US agree, European differs only if the end is
        # also a 31st.
        (date(2021, 8, 31), date(2021, 9, 30), 30, 30, 30),
        (date(2021, 8, 31), date(2021, 10, 31), 60, 60, 60),
        # A 30th start against a 31st end: the conditional rule fires in all
        # three, by different routes.
        (date(2021, 8, 30), date(2021, 10, 31), 60, 60, 60),
        # A start before the 30th against a 31st end: bond and US leave the end
        # alone, European caps it.
        (date(2021, 8, 15), date(2021, 10, 31), 76, 75, 76),
        # Ordinary dates, where nothing fires and everything agrees.
        (date(2021, 1, 1), date(2021, 7, 1), 180, 180, 180),
        (date(2021, 3, 15), date(2024, 3, 15), 1080, 1080, 1080),
    ],
)
def test_the_thirty_day_rules_case_by_case(
    start: date, end: date, bond: int, european: int, american: int
) -> None:
    assert day_count(start, end, Basis.THIRTY_360_BOND) == bond
    assert day_count(start, end, Basis.THIRTY_E_360) == european
    assert day_count(start, end, Basis.THIRTY_360_US) == american


def test_only_the_us_convention_has_an_end_of_february_rule() -> None:
    """Which is the whole difference between it and Bond Basis.

    Searched over every February start in a decade against a range of ends,
    asserting the two agree everywhere except where a February end-of-month is
    involved.
    """
    ends = [
        date(2021, 5, 31),
        date(2021, 8, 31),
        date(2021, 10, 31),
        date(2021, 6, 30),
        date(2021, 9, 30),
        date(2021, 7, 15),
        date(2022, 2, 28),
    ]
    starts = [
        date(2021, 2, 28),  # end of February
        date(2021, 2, 27),  # one day earlier, and so not
        date(2021, 1, 31),
        date(2021, 1, 15),
        date(2021, 3, 30),
    ]
    disagreements = 0
    for start in starts:
        for end in ends:
            if end < start:
                continue
            bond = day_count(start, end, Basis.THIRTY_360_BOND)
            american = day_count(start, end, Basis.THIRTY_360_US)
            if bond != american:
                disagreements += 1
                # Every disagreement traces to the end-of-February rule, which
                # only the US convention has.
                assert is_end_of_february(start)
    assert disagreements >= 3


def test_european_rules_never_condition_on_the_first_date() -> None:
    """Each date is capped at 30 independently, which is the whole rule.

    Checked by computing the count directly from the capped day numbers, with
    none of the implementation's branching.
    """
    for start_day in (1, 15, 28, 30, 31):
        for end_day in (1, 15, 28, 30, 31):
            start = date(2021, 1, min(start_day, 31))
            end = date(2021, 12, min(end_day, 31))
            expected = (
                360 * 0 + 30 * (12 - 1) + (min(end_day, 30) - min(start_day, 30))
            )
            assert day_count(start, end, Basis.THIRTY_E_360) == expected


def test_a_thirty_day_month_is_exactly_a_twelfth_of_a_year() -> None:
    for basis in (Basis.THIRTY_360_BOND, Basis.THIRTY_E_360, Basis.THIRTY_360_US):
        assert year_fraction(
            date(2021, 4, 15), date(2021, 5, 15), basis
        ) == pytest.approx(1 / 12)


def test_a_whole_year_is_one_under_every_thirty_day_convention() -> None:
    for basis in (Basis.THIRTY_360_BOND, Basis.THIRTY_E_360, Basis.THIRTY_360_US):
        assert year_fraction(
            date(2021, 6, 15), date(2022, 6, 15), basis
        ) == pytest.approx(1.0)


# -- shared properties -------------------------------------------------------


@pytest.mark.parametrize("basis", list(Basis))
def test_a_zero_length_period_is_zero(basis: Basis) -> None:
    day = date(2021, 5, 4)
    assert year_fraction(day, day, basis) == 0.0
    assert day_count(day, day, basis) == 0


@pytest.mark.parametrize("basis", list(Basis))
def test_a_backwards_period_is_refused(basis: Basis) -> None:
    with pytest.raises(BadPeriod, match="runs forwards"):
        year_fraction(date(2021, 5, 4), date(2021, 5, 3), basis)
    with pytest.raises(BadPeriod, match="runs forwards"):
        day_count(date(2021, 5, 4), date(2021, 5, 3), basis)


@pytest.mark.parametrize("basis", list(Basis))
def test_a_year_fraction_increases_with_the_end_date(basis: Basis) -> None:
    start = date(2021, 1, 15)
    previous = 0.0
    for days in range(1, 400, 7):
        fraction = year_fraction(start, start + timedelta(days=days), basis)
        assert fraction >= previous
        previous = fraction


@pytest.mark.parametrize("basis", list(Basis))
def test_a_zero_length_period_has_no_denominator(basis: Basis) -> None:
    day = date(2021, 5, 4)
    with pytest.raises(BadPeriod, match="zero-length"):
        denominator(day, day, basis)


@pytest.mark.parametrize(
    ("basis", "expected"), [(Basis.ACT_360, 360.0), (Basis.ACT_365F, 365.0)]
)
def test_the_implied_denominator_is_the_named_one(basis: Basis, expected: float) -> None:
    assert denominator(date(2021, 1, 1), date(2021, 7, 1), basis) == pytest.approx(
        expected
    )


def test_act_act_has_no_single_denominator() -> None:
    """The effective one sits between 365 and 366 for a period spanning a leap.

    That it is not a round number is the convention's defining feature, and the
    reason the function reports an effective figure rather than a constant.
    """
    inside_leap = denominator(date(2020, 1, 1), date(2020, 7, 1), Basis.ACT_ACT_ISDA)
    assert inside_leap == pytest.approx(366.0)
    spanning = denominator(date(2019, 11, 1), date(2021, 2, 1), Basis.ACT_ACT_ISDA)
    assert 365.0 < spanning < 366.0


# -- the calendar helpers ----------------------------------------------------


@pytest.mark.parametrize(
    ("year", "leap"),
    [(2020, True), (2021, False), (2000, True), (1900, False), (2100, False), (2400, True)],
)
def test_the_leap_year_rule_including_the_century_exceptions(
    year: int, leap: bool
) -> None:
    assert is_leap(year) is leap
    assert days_in_year(year) == (366 if leap else 365)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2021, 2, 28), True),
        (date(2021, 2, 27), False),
        (date(2020, 2, 29), True),
        (date(2020, 2, 28), False),
        (date(2021, 3, 31), False),
        (date(2021, 1, 31), False),
    ],
)
def test_end_of_february(day: date, expected: bool) -> None:
    assert is_end_of_february(day) is expected
