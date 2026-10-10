"""Compounded overnight rates and the conventions on the observation window.

**One identity carries the plain case and it is exact.** A projected overnight
rate is a ratio of discount factors, so the compounded growth over a period is
``P(start) / P(end)`` and nothing else. That is asserted to rounding rather than
to a tolerance, on a curve whose overnight forward has a 50bp step inside the
period, which is where an approximation would show.

**The structural claim is tested as structure.** An observation shift keeps the
identity over a shifted window; a lookback and a lockout cannot, because the
first pairs one day's rate with another day's weight and the second repeats a
factor. The tests assert the telescoping where it holds, assert the refusal
where it does not, and only then measure the sizes.

**The sizes are measured on two curves, because one of them hides the subject.**
The conventions are worth a fraction of a basis point on a smoothly interpolated
curve and several basis points across a policy step, and a file that measured
only the smooth one would conclude the conventions do not matter.

The compounding-against-averaging gap is checked against ``r^2 T / 2``, which is
derived from nothing in this module, and the leg-level figures are checked for
the compression they show against the single-period ones. Then the refusals.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from tenor import (
    Averaging,
    BadOvernight,
    Basis,
    DiscountCurve,
    Frequency,
    MissingFixing,
    Observation,
    OvernightIndex,
    OvernightLeg,
    business_days,
    calendar_from,
    compounded_rate,
    convention_basis,
    convention_margin,
    equivalent_rates,
    observations,
    replication_factor,
    year_fraction,
)
from tenor.calendar import WEEKENDS_ONLY

REFERENCE = date(2026, 1, 15)
MEETING = date(2026, 4, 29)
BEFORE, AFTER = 0.0300, 0.0350

#: A period that straddles the policy step, and one whose end does.
SPANNING = (date(2026, 3, 16), date(2026, 6, 16))
ENDING_AFTER = (date(2026, 2, 16), date(2026, 5, 6))

PLAIN = OvernightIndex()


def step_curve(horizon: date = date(2030, 7, 15)) -> DiscountCurve:
    """Discounts built from a piecewise-constant overnight rate with one hike.

    Built day by day from the rate rather than interpolated from zero rates, so
    the step survives into the forwards. An interpolated curve smooths it away,
    which is the point of having both.
    """
    days = [REFERENCE]
    day = REFERENCE
    while day < horizon:
        day = WEEKENDS_ONLY.next_business_day(day)
        days.append(day)
    points = []
    factor = 1.0
    for start, end in pairwise(days):
        rate = BEFORE if start < MEETING else AFTER
        factor /= 1.0 + rate * year_fraction(start, end, Basis.ACT_360)
        points.append((end, factor))
    return DiscountCurve.from_discounts(REFERENCE, points, basis=Basis.ACT_365F)


def smooth_curve() -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE,
        [
            (date(2026, 7, 15), 0.030),
            (date(2027, 1, 15), 0.035),
            (date(2028, 1, 15), 0.042),
            (date(2031, 1, 15), 0.050),
        ],
        basis=Basis.ACT_365F,
    )


STEP = step_curve()
SMOOTH = smooth_curve()


# -- the identity ------------------------------------------------------------


@pytest.mark.parametrize("curve", [STEP, SMOOTH], ids=["step", "smooth"])
def test_compounded_in_arrears_is_a_ratio_of_two_discount_factors(
    curve: DiscountCurve,
) -> None:
    """The one statement the whole module rests on.

    Not a tolerance: the daily growth factors are ``P(a) / P(b)`` exactly, so
    the product is ``P(start) / P(end)`` to the accumulated rounding of sixty-six
    multiplications and no more.
    """
    start, end = SPANNING
    rate = compounded_rate(curve, start, end, PLAIN)
    assert rate.growth == pytest.approx(replication_factor(curve, start, end, PLAIN), rel=5e-15)
    direct = (curve.discount(start) / curve.discount(end) - 1.0) / year_fraction(
        start, end, Basis.ACT_360
    )
    assert rate.rate == pytest.approx(direct, rel=5e-15)
    assert rate.days == 66
    # Every weight belongs to the accrual, so they sum to it exactly.
    assert rate.weight == pytest.approx(rate.accrual, abs=1e-15)


@pytest.mark.parametrize("lag", [2, 5, 10])
def test_an_observation_shift_telescopes_over_the_shifted_window(lag: int) -> None:
    """The identity survives, moved back. Rate and weight shift together, so
    consecutive factors still share a discount factor."""
    start, end = SPANNING
    index = OvernightIndex(observation=Observation.SHIFT, days=lag)
    assert index.telescopes
    rate = compounded_rate(STEP, start, end, index)
    assert rate.growth == pytest.approx(
        replication_factor(STEP, start, end, index), rel=5e-15
    )


@pytest.mark.parametrize(
    "index",
    [
        OvernightIndex(observation=Observation.LOOKBACK, days=5),
        OvernightIndex(observation=Observation.LOCKOUT, days=5),
        OvernightIndex(averaging=Averaging.ARITHMETIC),
    ],
)
def test_the_conventions_that_do_not_telescope_refuse_to_pretend(
    index: OvernightIndex,
) -> None:
    """Better to have no replication factor than an approximate one."""
    assert not index.telescopes
    start, end = SPANNING
    with pytest.raises(BadOvernight, match="does not telescope"):
        replication_factor(STEP, start, end, index)


def test_a_shift_redistributes_the_weights_and_a_lookback_does_not() -> None:
    """The mechanism behind the whole difference between the two conventions.

    A lookback keeps each accrual day's own weight and changes only the rate, so
    the weights match day for day at every lag. A shift takes the weight from the
    observation day, which on a weekends-only calendar means a Friday's three
    days of weight lands on whichever accrual day is two business days after it.
    The total survives — 26 of the 66 days carry a different weight at a two-day
    shift and the sum is unchanged to the last bit — so this is a
    redistribution, and it is the reason the shift moves the rate twice as far
    as the lookback at that lag.
    """
    start, end = SPANNING
    accrual = observations(STEP, start, end, PLAIN)
    for lag in (2, 3, 4):
        shifted = compounded_rate(
            STEP, start, end, OvernightIndex(observation=Observation.SHIFT, days=lag)
        )
        lagged = observations(
            STEP, start, end, OvernightIndex(observation=Observation.LOOKBACK, days=lag)
        )
        moved = sum(
            1
            for ordinary, one in zip(accrual, shifted.observations, strict=True)
            if abs(ordinary.weight - one.weight) > 1e-15
        )
        assert moved == 26
        assert shifted.weight == pytest.approx(shifted.accrual, abs=1e-15)
        assert all(
            ordinary.weight == pytest.approx(one.weight, abs=1e-18)
            for ordinary, one in zip(accrual, lagged, strict=True)
        )


def test_at_five_days_the_shift_does_not_move_a_single_weight() -> None:
    """Which is the whole of why the two conventions coincide at that lag.

    Five business days is exactly seven calendar days, so every accrual day's
    observation day has its own day count. Nothing is redistributed and only the
    rates move, leaving the shift indistinguishable from the lookback.
    """
    start, end = SPANNING
    accrual = observations(STEP, start, end, PLAIN)
    shifted = observations(
        STEP, start, end, OvernightIndex(observation=Observation.SHIFT, days=5)
    )
    assert all(
        ordinary.weight == pytest.approx(one.weight, abs=1e-18)
        for ordinary, one in zip(accrual, shifted, strict=True)
    )


def test_at_one_day_the_weights_stop_summing_to_the_accrual() -> None:
    """A real artefact of the convention rather than a rounding story.

    The period starts on a Monday and ends on a Tuesday, so a one-business-day
    shift moves the start back three calendar days and the end back one. The
    shifted window is two calendar days longer than the accrual, and since the
    accrual is still the denominator, the convention is paying the growth of a
    longer window over a shorter one.
    """
    start, end = SPANNING
    shifted = compounded_rate(
        STEP, start, end, OvernightIndex(observation=Observation.SHIFT, days=1)
    )
    assert shifted.weight - shifted.accrual == pytest.approx(2.0 / 360.0, abs=1e-15)


# -- the window itself -------------------------------------------------------


def test_the_observation_window_covers_the_accrual_without_a_gap() -> None:
    start, end = SPANNING
    window = list(business_days(WEEKENDS_ONLY, start, end))
    assert window[0][0] == start
    assert window[-1][1] == end
    for (_, first_end), (second_start, _) in pairwise(window):
        assert first_end == second_start
    assert sum(
        year_fraction(a, b, Basis.ACT_360) for a, b in window
    ) == pytest.approx(year_fraction(start, end, Basis.ACT_360), abs=1e-15)


def test_a_window_opening_on_a_weekend_starts_on_the_next_business_day() -> None:
    saturday = date(2026, 3, 14)
    assert not WEEKENDS_ONLY.is_business_day(saturday)
    window = list(business_days(WEEKENDS_ONLY, saturday, date(2026, 3, 24)))
    assert window[0][0] == date(2026, 3, 16)


def test_a_lookback_reads_an_earlier_rate_and_keeps_the_accrual_weight() -> None:
    """The definition, checked day by day rather than through a total."""
    start, end = SPANNING
    index = OvernightIndex(observation=Observation.LOOKBACK, days=5)
    plain = observations(STEP, start, end, PLAIN)
    lagged = observations(STEP, start, end, index)
    assert len(plain) == len(lagged)
    for ordinary, shifted in zip(plain, lagged, strict=True):
        assert shifted.accrual_start == ordinary.accrual_start
        assert shifted.weight == pytest.approx(ordinary.weight, abs=1e-15)
        assert shifted.observation < ordinary.observation
        assert WEEKENDS_ONLY.business_days_between(shifted.observation, ordinary.observation) == 5


def test_a_lockout_repeats_the_rate_at_the_cut() -> None:
    start, end = SPANNING
    index = OvernightIndex(observation=Observation.LOCKOUT, days=4)
    daily = observations(STEP, start, end, index)
    frozen = daily[-4:]
    assert len({one.rate for one in frozen}) == 1
    assert frozen[0].observation == daily[-5].observation
    # The weights are still the accrual's, so they still sum to it.
    assert sum(one.weight for one in daily) == pytest.approx(
        year_fraction(start, end, Basis.ACT_360), abs=1e-15
    )


def test_a_lockout_longer_than_the_period_is_refused() -> None:
    with pytest.raises(BadOvernight, match="leaves nothing observed"):
        compounded_rate(
            STEP,
            date(2026, 3, 16),
            date(2026, 3, 23),
            OvernightIndex(observation=Observation.LOCKOUT, days=6),
        )


# -- the measured sizes ------------------------------------------------------


@pytest.mark.parametrize(
    ("lag", "expected"),
    [(2, -1.0959), (5, -3.8354), (10, -7.6703)],
)
def test_a_lookback_across_a_policy_step_is_worth_basis_points(
    lag: int, expected: float
) -> None:
    """And the number is the step times the fraction of the window it moves."""
    start, end = SPANNING
    index = OvernightIndex(observation=Observation.LOOKBACK, days=lag)
    basis = convention_basis(STEP, start, end, index, PLAIN) * 1e4
    assert basis == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("lag", [5, 10])
def test_a_whole_number_of_weeks_makes_the_two_window_conventions_identical(
    lag: int,
) -> None:
    """How the distinction gets missed.

    A five-business-day lag is exactly seven calendar days on a weekends-only
    calendar, so each accrual day's weight equals its shifted day's weight and
    the lookback and the shift agree bit for bit. The usual lag is five.
    """
    start, end = SPANNING
    lookback = compounded_rate(
        STEP, start, end, OvernightIndex(observation=Observation.LOOKBACK, days=lag)
    )
    shifted = compounded_rate(
        STEP, start, end, OvernightIndex(observation=Observation.SHIFT, days=lag)
    )
    assert lookback.rate == pytest.approx(shifted.rate, rel=1e-14)


def test_at_two_days_the_two_conventions_differ_by_a_factor_of_two() -> None:
    """A weekend crosses the boundary, the weights stop matching, and what the
    five-day lag concealed becomes the whole of the difference."""
    start, end = SPANNING
    lookback = convention_basis(
        STEP, start, end, OvernightIndex(observation=Observation.LOOKBACK, days=2), PLAIN
    )
    shifted = convention_basis(
        STEP, start, end, OvernightIndex(observation=Observation.SHIFT, days=2), PLAIN
    )
    assert lookback * 1e4 == pytest.approx(-1.0959, abs=5e-4)
    assert shifted * 1e4 == pytest.approx(-2.1915, abs=5e-4)
    assert shifted / lookback == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("lag", [2, 5, 10])
def test_a_smooth_curve_makes_every_convention_look_like_pedantry(lag: int) -> None:
    """Under a hundredth of a basis point, against several basis points on the
    step curve. A model that interpolates the steps away cannot be used to
    decide whether the conventions matter."""
    start, end = SPANNING
    for observation in (Observation.LOOKBACK, Observation.SHIFT):
        index = OvernightIndex(observation=observation, days=lag)
        assert abs(convention_basis(SMOOTH, start, end, index, PLAIN)) * 1e4 < 0.01


@pytest.mark.parametrize("lag", [2, 5, 10])
def test_a_lockout_is_worth_nothing_with_the_step_in_the_middle(lag: int) -> None:
    start, end = SPANNING
    index = OvernightIndex(observation=Observation.LOCKOUT, days=lag)
    assert convention_basis(STEP, start, end, index, PLAIN) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("lag", "expected"), [(2, 0.0), (5, -4.4591), (10, -4.4591)])
def test_a_lockout_earns_its_name_when_the_step_is_inside_the_locked_window(
    lag: int, expected: float
) -> None:
    """Identically at five and ten days, because both already reach the step."""
    start, end = ENDING_AFTER
    index = OvernightIndex(observation=Observation.LOCKOUT, days=lag)
    assert convention_basis(STEP, start, end, index, PLAIN) * 1e4 == pytest.approx(
        expected, abs=5e-4
    )


@pytest.mark.parametrize(
    ("start", "end", "label"),
    [
        (date(2026, 3, 16), date(2026, 6, 16), "3m"),
        (date(2026, 3, 16), date(2026, 9, 16), "6m"),
        (date(2026, 3, 16), date(2027, 3, 16), "12m"),
    ],
)
@pytest.mark.parametrize("curve", [STEP, SMOOTH], ids=["step", "smooth"])
def test_compounding_beats_averaging_by_half_r_squared_t(
    start: date, end: date, label: str, curve: DiscountCurve
) -> None:
    """A closed form with nothing of this module in it.

    The second-order term of the product against the sum. It is high by 2% to 3%
    in every one of the six cases, always the same way, which is the third-order
    term it drops — a correction that is consistent is a different thing from an
    error that is not.
    """
    compounded = compounded_rate(curve, start, end, PLAIN)
    averaged = compounded_rate(curve, start, end, OvernightIndex(averaging=Averaging.ARITHMETIC))
    gap = compounded.rate - averaged.rate
    predicted = 0.5 * compounded.rate**2 * compounded.accrual
    assert gap > 0.0
    assert 0.02 <= predicted / gap - 1.0 <= 0.035, label


def test_the_averaging_gap_grows_with_the_square_of_the_tenor() -> None:
    gaps = []
    for end in (date(2026, 6, 16), date(2026, 9, 16), date(2027, 3, 16)):
        compounded = compounded_rate(SMOOTH, date(2026, 3, 16), end, PLAIN)
        averaged = compounded_rate(
            SMOOTH, date(2026, 3, 16), end, OvernightIndex(averaging=Averaging.ARITHMETIC)
        )
        gaps.append((compounded.rate - averaged.rate) * 1e4)
    assert gaps[0] == pytest.approx(1.0991, abs=5e-4)
    assert gaps[1] == pytest.approx(2.7598, abs=5e-4)
    assert gaps[2] == pytest.approx(7.2217, abs=5e-4)
    # Quadratic growth, not linear: doubling the tenor more than doubles it.
    assert gaps[1] / gaps[0] > 2.0
    assert gaps[2] / gaps[1] > 2.0


def test_an_averaged_rate_still_reports_a_comparable_growth() -> None:
    start, end = SPANNING
    averaged = compounded_rate(
        STEP, start, end, OvernightIndex(averaging=Averaging.ARITHMETIC)
    )
    assert averaged.growth == pytest.approx(1.0 + averaged.rate * averaged.accrual, abs=1e-15)
    assert averaged.growth < compounded_rate(STEP, start, end, PLAIN).growth


# -- payment delay -----------------------------------------------------------


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_a_payment_delay_changes_no_rate_at_all(lag: int) -> None:
    """Exactly zero, not approximately. It is in a different group from the
    other four conventions and the only way to show that is to measure it."""
    start, end = SPANNING
    delayed = OvernightIndex(payment_lag=lag)
    assert convention_basis(STEP, start, end, delayed, PLAIN) == 0.0


def test_a_payment_delay_moves_the_money_and_so_has_a_spread() -> None:
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    margin = convention_margin(leg, OvernightIndex(payment_lag=5), STEP)
    assert margin * 1e4 == pytest.approx(-0.2363, abs=5e-4)
    assert leg.with_index(OvernightIndex(payment_lag=5)).schedule.payment_dates != (
        leg.schedule.payment_dates
    )


# -- a leg -------------------------------------------------------------------


def test_a_leg_accrues_on_the_adjusted_boundaries() -> None:
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    coupons = leg.coupons(STEP)
    assert len(coupons) == 12
    for coupon in coupons:
        assert coupon.overnight.start == coupon.period.adjusted_start
        assert coupon.overnight.end == coupon.period.adjusted_end
        assert coupon.payment == coupon.period.payment
    # No gaps and no overlaps between consecutive accruals.
    for first, second in pairwise(coupons):
        assert first.overnight.end == second.overnight.start


def test_a_leg_paying_its_own_projected_rate_is_worth_the_telescoped_sum() -> None:
    """Each period's cashflow is ``P(s)/P(e) - 1`` on the notional, discounted
    at its own payment, so with no payment lag the leg is a sum of terms that
    each collapse. Checked against the sum built directly from the curve.
    """
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    direct = sum(
        (STEP.discount(period.adjusted_start) / STEP.discount(period.adjusted_end) - 1.0)
        * STEP.discount(period.payment)
        for period in leg.schedule
    )
    assert leg.value(STEP) == pytest.approx(direct, rel=1e-13)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (OvernightIndex(observation=Observation.LOOKBACK, days=5), -0.3370),
        (OvernightIndex(observation=Observation.SHIFT, days=5), -0.3370),
        (OvernightIndex(observation=Observation.LOCKOUT, days=5), -0.2412),
        (OvernightIndex(averaging=Averaging.ARITHMETIC), -1.4933),
    ],
)
def test_the_conventions_compress_at_the_level_of_a_leg(
    index: OvernightIndex, expected: float
) -> None:
    """Only one period of twelve spans the step, so the -3.84bp a five-day
    lookback is worth inside that period is -0.34bp across the leg."""
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    assert convention_margin(leg, index, STEP) * 1e4 == pytest.approx(expected, abs=5e-4)


def test_the_margin_is_a_solve_and_not_a_search() -> None:
    """A leg's value is linear in its spread, so the margin reproduces the other
    convention's value exactly when it is put back on the leg."""
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    other = OvernightIndex(averaging=Averaging.ARITHMETIC)
    margin = convention_margin(leg, other, STEP)
    repriced = OvernightLeg(
        effective=leg.effective, maturity=leg.maturity, index=leg.index, spread=margin
    )
    assert repriced.value(STEP) == pytest.approx(
        leg.with_index(other).value(STEP), rel=1e-12
    )


def test_a_spread_is_worth_its_annuity() -> None:
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    wider = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2), spread=0.0001)
    assert wider.value(STEP) - leg.value(STEP) == pytest.approx(
        0.0001 * leg.annuity(STEP), rel=1e-12
    )


def test_a_leg_can_project_off_one_curve_and_discount_on_another() -> None:
    leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2))
    both = leg.value(STEP, STEP)
    assert both == pytest.approx(leg.value(STEP), rel=1e-15)
    split = leg.value(STEP, SMOOTH)
    assert split != pytest.approx(both, rel=1e-6)


# -- fixings -----------------------------------------------------------------


def test_a_fixing_is_read_rather_than_projected() -> None:
    start, end = SPANNING
    plain = compounded_rate(STEP, start, end, PLAIN)
    override = {start: plain.observations[0].rate + 0.01}
    tweaked = compounded_rate(STEP, start, end, PLAIN, fixings=override)
    assert tweaked.observations[0].fixed
    assert tweaked.fixed_days == 1
    assert not tweaked.observations[1].fixed
    assert tweaked.rate > plain.rate


def test_fixings_equal_to_the_curve_change_nothing() -> None:
    """Which is the check that the fixing path and the projection path agree."""
    start, end = SPANNING
    plain = compounded_rate(STEP, start, end, PLAIN)
    same = {one.observation: one.rate for one in plain.observations}
    replayed = compounded_rate(STEP, start, end, PLAIN, fixings=same)
    assert replayed.fixed_days == replayed.days
    assert replayed.rate == pytest.approx(plain.rate, rel=1e-15)
    assert replayed.growth == pytest.approx(plain.growth, rel=1e-15)


def test_a_past_observation_with_no_fixing_is_refused() -> None:
    """Rather than filled with the first projectable rate, which would be wrong
    by the whole move since the window opened."""
    with pytest.raises(MissingFixing, match="in the past relative to the curve"):
        compounded_rate(STEP, date(2026, 1, 5), date(2026, 4, 5), PLAIN)


def test_a_lookback_leg_asks_for_fixings_from_before_its_own_effective_date() -> None:
    """The lag's worth of business days before the accrual starts, which for a
    leg traded on its effective date is before the trade existed."""
    leg = OvernightLeg(
        REFERENCE,
        date(2027, 1, 15),
        index=OvernightIndex(observation=Observation.LOOKBACK, days=5),
    )
    with pytest.raises(MissingFixing):
        leg.value(STEP)
    first = leg.schedule[0]
    needed = observations(
        STEP,
        first.adjusted_start,
        first.adjusted_end,
        OvernightIndex(observation=Observation.LOOKBACK, days=5),
        fixings={
            WEEKENDS_ONLY.add_business_days(first.adjusted_start, -step): 0.030
            for step in range(1, 6)
        },
    )
    assert needed[0].fixed
    assert needed[0].observation < leg.effective


# -- refusals ----------------------------------------------------------------


def test_a_lag_on_a_convention_that_has_no_lag_is_refused() -> None:
    with pytest.raises(BadOvernight, match="in arrears and carries a lag"):
        OvernightIndex(days=3)


@pytest.mark.parametrize(
    "observation", [Observation.LOOKBACK, Observation.SHIFT, Observation.LOCKOUT]
)
def test_a_lagged_convention_with_no_lag_is_refused(observation: Observation) -> None:
    with pytest.raises(BadOvernight, match="in arrears under another name"):
        OvernightIndex(observation=observation, days=0)


def test_a_lag_in_the_wrong_units_is_refused() -> None:
    """Thirty is calendar days where business days were meant, and it would
    silently read a rate three weeks stale."""
    with pytest.raises(BadOvernight, match="against a ceiling"):
        OvernightIndex(observation=Observation.LOOKBACK, days=30)
    with pytest.raises(BadOvernight, match="not been published"):
        OvernightIndex(observation=Observation.LOOKBACK, days=-1)


def test_a_negative_payment_lag_is_refused() -> None:
    with pytest.raises(BadOvernight, match="before it has run"):
        OvernightIndex(payment_lag=-1)


def test_a_window_with_no_life_in_it_is_refused() -> None:
    with pytest.raises(ValueError, match="end after start"):
        list(business_days(WEEKENDS_ONLY, date(2026, 3, 16), date(2026, 3, 16)))
    with pytest.raises(ValueError, match="end after start"):
        compounded_rate(STEP, date(2026, 3, 16), date(2026, 3, 10), PLAIN)


def test_a_leg_with_no_life_or_no_notional_is_refused() -> None:
    with pytest.raises(BadOvernight, match="no life in it"):
        OvernightLeg(date(2026, 3, 16), date(2026, 3, 16))
    with pytest.raises(BadOvernight, match="not a trade"):
        OvernightLeg(date(2026, 3, 16), date(2027, 3, 16), notional=0.0)


def test_comparing_nothing_is_refused() -> None:
    with pytest.raises(BadOvernight, match="no conventions to compare"):
        equivalent_rates(STEP, *SPANNING, [])


def test_several_conventions_can_be_priced_side_by_side() -> None:
    start, end = SPANNING
    rates = equivalent_rates(
        STEP,
        start,
        end,
        [
            PLAIN,
            OvernightIndex(observation=Observation.LOOKBACK, days=5),
            OvernightIndex(observation=Observation.SHIFT, days=5),
            OvernightIndex(averaging=Averaging.ARITHMETIC),
        ],
    )
    assert len(rates) == 4
    assert all(one.days == 66 for one in rates)
    assert rates[0].rate > rates[1].rate
    assert rates[0].rate > rates[3].rate


def test_a_holiday_calendar_changes_the_observation_count() -> None:
    """The window is defined on business days, so a holiday removes one and
    lengthens the weight of the day before it."""
    start, end = SPANNING
    holiday = date(2026, 4, 3)
    assert WEEKENDS_ONLY.is_business_day(holiday)
    with_holiday = OvernightIndex(calendar=calendar_from("GB", [holiday]))
    daily = observations(STEP, start, end, with_holiday)
    assert len(daily) == 65
    assert sum(one.weight for one in daily) == pytest.approx(
        year_fraction(start, end, Basis.ACT_360), abs=1e-15
    )


def test_a_leg_rolls_on_its_own_frequency() -> None:
    for frequency, periods in (
        (Frequency.MONTHLY, 36),
        (Frequency.QUARTERLY, 12),
        (Frequency.SEMI_ANNUAL, 6),
        (Frequency.ANNUAL, 3),
    ):
        leg = OvernightLeg(date(2026, 2, 2), date(2029, 2, 2), frequency=frequency)
        assert len(leg.schedule) == periods
        assert leg.value(STEP) == pytest.approx(
            sum(
                (STEP.discount(p.adjusted_start) / STEP.discount(p.adjusted_end) - 1.0)
                * STEP.discount(p.payment)
                for p in leg.schedule
            ),
            rel=1e-13,
        )


def test_a_lockout_longer_than_a_leg_s_stub_is_refused() -> None:
    """Which a leg can produce without the caller choosing it.

    A schedule runs backward from maturity, so an effective date that is not a
    whole number of periods before it leaves a short first stub. A five-day
    lockout on a leg whose stub is two business days long has nothing left to
    observe, and the refusal names the period rather than the leg so the stub is
    identifiable.
    """
    leg = OvernightLeg(
        date(2026, 2, 2),
        date(2029, 2, 4),
        index=OvernightIndex(observation=Observation.LOCKOUT, days=5),
    )
    assert leg.schedule.has_stub
    with pytest.raises(BadOvernight, match="leaves nothing observed"):
        leg.value(STEP)
