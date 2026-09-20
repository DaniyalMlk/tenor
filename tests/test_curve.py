"""Compounding conventions and discount curves.

Two things carry most of the weight here. The round trip between a rate and a
discount factor has to close to floating point, because a curve stores one and
quotes the other and any drift shows up as a curve that cannot reprice its own
inputs. And the two interpolations have to be shown doing what they are
documented to do — which for linear-on-zero means manufacturing a negative
forward rate from arbitrage-free data, since that is the reason the choice is
named on the curve rather than assumed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tenor.curve import (
    BadCurve,
    DiscountCurve,
    Interpolation,
    OffCurve,
    flat_curve,
)
from tenor.daycount import Basis
from tenor.rates import (
    BadRate,
    Compounding,
    Rate,
    convert,
    discount_factor,
    forward_rate,
    rate_from_discount,
)

REFERENCE = date(2021, 1, 4)
ZEROS = [
    (date(2022, 1, 4), 0.010),
    (date(2024, 1, 4), 0.015),
    (date(2026, 1, 4), 0.018),
    (date(2031, 1, 4), 0.022),
]


def build(interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT) -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE, ZEROS, basis=Basis.ACT_365F, interpolation=interpolation
    )


# -- compounding -------------------------------------------------------------


@pytest.mark.parametrize("compounding", list(Compounding))
@pytest.mark.parametrize("time", [0.25, 1.0, 5.0, 30.0])
@pytest.mark.parametrize("rate", [0.0001, 0.02, 0.15])
def test_rate_and_discount_round_trip(
    compounding: Compounding, time: float, rate: float
) -> None:
    factor = discount_factor(rate, time, compounding)
    assert rate_from_discount(factor, time, compounding) == pytest.approx(
        rate, rel=1e-12
    )


def test_the_published_spread_between_conventions() -> None:
    """The table in the module docstring, recomputed.

    A discount factor of 0.90 over five years is eleven and a half basis points
    wide across the conventions, which is more than the bid-offer on most of the
    curve.
    """
    rates = {
        compounding: rate_from_discount(0.90, 5.0, compounding)
        for compounding in Compounding
    }
    assert rates[Compounding.SIMPLE] == pytest.approx(0.022222, abs=1e-6)
    assert rates[Compounding.ANNUAL] == pytest.approx(0.021296, abs=1e-6)
    assert rates[Compounding.SEMI_ANNUAL] == pytest.approx(0.021184, abs=1e-6)
    assert rates[Compounding.CONTINUOUS] == pytest.approx(0.021072, abs=1e-6)
    spread = rates[Compounding.SIMPLE] - rates[Compounding.CONTINUOUS]
    assert spread * 1e4 == pytest.approx(11.5, abs=0.1)


def test_more_frequent_compounding_needs_a_lower_rate() -> None:
    """For a fixed discount factor the rate falls monotonically with frequency."""
    ordered = [
        Compounding.SIMPLE,
        Compounding.ANNUAL,
        Compounding.SEMI_ANNUAL,
        Compounding.QUARTERLY,
        Compounding.MONTHLY,
        Compounding.CONTINUOUS,
    ]
    rates = [rate_from_discount(0.90, 5.0, one) for one in ordered]
    assert rates == sorted(rates, reverse=True)


def test_simple_is_not_annual_with_one_period() -> None:
    """They agree at exactly one year and nowhere else.

    Which is the one point somebody checking an implementation is most likely to
    pick, so the test checks a year and then deliberately checks either side.
    """
    assert discount_factor(0.03, 1.0, Compounding.SIMPLE) == pytest.approx(
        discount_factor(0.03, 1.0, Compounding.ANNUAL)
    )
    for time in (0.5, 2.0, 5.0):
        assert discount_factor(0.03, time, Compounding.SIMPLE) != pytest.approx(
            discount_factor(0.03, time, Compounding.ANNUAL), rel=1e-9
        )


def test_continuous_is_the_limit_of_the_others() -> None:
    """And the approach is first order, which is the checkable part.

    Asserting a tolerance at some large ``n`` only says the number is small.
    The error of periodic compounding against continuous falls as ``1/n``, so
    doubling the frequency halves it — that is a property of the limit rather
    than of the arithmetic, and it pins the relationship rather than one point
    on it.
    """
    target = discount_factor(0.05, 3.0, Compounding.CONTINUOUS)
    errors = {}
    for periods in (1, 2, 4, 12, 365, 10_000, 100_000):
        approximation = (1.0 + 0.05 / periods) ** (-periods * 3.0)
        errors[periods] = abs(approximation - target)

    values = list(errors.values())
    assert values == sorted(values, reverse=True)

    for coarse, fine in ((1, 2), (2, 4), (10_000, 100_000)):
        ratio = errors[coarse] / errors[fine]
        assert ratio == pytest.approx(fine / coarse, rel=0.1)


def test_conversion_preserves_the_discount_factor() -> None:
    for source in Compounding:
        for target in Compounding:
            rate = rate_from_discount(0.88, 4.0, source)
            converted = convert(rate, 4.0, source, target)
            assert discount_factor(converted, 4.0, target) == pytest.approx(
                0.88, rel=1e-12
            )


def test_converting_to_the_same_convention_is_the_identity() -> None:
    for compounding in Compounding:
        assert convert(0.0345, 2.0, compounding, compounding) == 0.0345


def test_a_rate_carries_its_convention() -> None:
    rate = Rate(0.021072, Compounding.CONTINUOUS)
    assert rate.discount(5.0) == pytest.approx(0.90, abs=1e-5)
    annual = rate.to(Compounding.ANNUAL, 5.0)
    assert annual.compounding is Compounding.ANNUAL
    assert annual.value == pytest.approx(0.021296, abs=1e-6)
    assert annual.discount(5.0) == pytest.approx(rate.discount(5.0), rel=1e-12)
    assert "continuous" in str(rate)


def test_a_zero_time_discount_factor_is_one() -> None:
    for compounding in Compounding:
        assert discount_factor(0.05, 0.0, compounding) == 1.0


def test_no_rate_is_implied_at_zero_time() -> None:
    with pytest.raises(BadRate, match="strictly positive time"):
        rate_from_discount(1.0, 0.0, Compounding.ANNUAL)


def test_a_negative_time_is_refused() -> None:
    with pytest.raises(BadRate, match="non-negative time"):
        discount_factor(0.05, -1.0, Compounding.ANNUAL)


def test_a_non_positive_discount_factor_is_refused() -> None:
    for bad in (0.0, -0.5):
        with pytest.raises(BadRate, match="strictly positive"):
            rate_from_discount(bad, 1.0, Compounding.ANNUAL)


def test_a_simple_rate_below_minus_one_over_t_has_no_discount_factor() -> None:
    with pytest.raises(BadRate, match="past zero"):
        discount_factor(-0.5, 5.0, Compounding.SIMPLE)


def test_a_periodic_rate_that_wipes_out_a_period_is_refused() -> None:
    with pytest.raises(BadRate, match="not positive"):
        discount_factor(-3.0, 1.0, Compounding.ANNUAL)


def test_negative_rates_are_ordinary() -> None:
    """They are the market, not an error, and discount to above one."""
    for compounding in Compounding:
        factor = discount_factor(-0.005, 3.0, compounding)
        assert factor > 1.0
        assert rate_from_discount(factor, 3.0, compounding) == pytest.approx(
            -0.005, rel=1e-12
        )


def test_a_standalone_forward_rate_spans_the_interval_not_the_far_date() -> None:
    """The error that vanishes at ``near = 0``, which is where it is tested."""
    near_factor, far_factor = 0.98, 0.94
    rate = forward_rate(near_factor, far_factor, 1.0, 3.0, Compounding.CONTINUOUS)
    assert math.exp(-rate * 2.0) == pytest.approx(far_factor / near_factor, rel=1e-14)
    wrong = -math.log(far_factor / near_factor) / 3.0
    assert rate != pytest.approx(wrong, rel=1e-6)


def test_a_backwards_forward_rate_is_refused() -> None:
    with pytest.raises(BadRate, match="not forwards in time"):
        forward_rate(0.98, 0.94, 3.0, 1.0)


# -- curve construction ------------------------------------------------------


def test_a_curve_reprices_its_own_pillars() -> None:
    curve = build()
    assert curve.reprices_pillars()
    for day, rate in ZEROS:
        assert curve.zero_rate(day) == pytest.approx(rate, rel=1e-12)


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_every_interpolation_reprices_the_pillars(
    interpolation: Interpolation,
) -> None:
    """Parametrised over the enum, so a scheme added later is covered at once."""
    assert build(interpolation).reprices_pillars()


def test_the_reference_date_is_added_as_an_anchor() -> None:
    """Without it the curve cannot price anything before its first traded point."""
    curve = build()
    assert len(curve) == len(ZEROS) + 1
    assert curve.pillars[0].day == REFERENCE
    assert curve.pillars[0].discount == 1.0
    assert curve.pillars[0].is_anchor


def test_an_anchor_supplied_explicitly_is_not_duplicated() -> None:
    curve = DiscountCurve.from_discounts(
        REFERENCE,
        [(REFERENCE, 1.0), (date(2022, 1, 4), 0.99)],
        basis=Basis.ACT_365F,
    )
    assert len(curve) == 2


def test_an_anchor_that_is_not_one_is_refused() -> None:
    with pytest.raises(BadCurve, match="worth one unit"):
        DiscountCurve.from_discounts(
            REFERENCE, [(REFERENCE, 0.99)], basis=Basis.ACT_365F
        )


def test_pillars_are_sorted_however_they_arrive() -> None:
    curve = DiscountCurve.from_zeros(
        REFERENCE, list(reversed(ZEROS)), basis=Basis.ACT_365F
    )
    assert list(curve.dates) == sorted(curve.dates)
    assert curve.reprices_pillars()


def test_a_pillar_before_the_reference_date_is_refused() -> None:
    with pytest.raises(BadCurve, match="runs forwards"):
        DiscountCurve.from_discounts(
            REFERENCE, [(date(2020, 1, 1), 1.01)], basis=Basis.ACT_365F
        )


def test_a_repeated_pillar_is_refused() -> None:
    with pytest.raises(BadCurve, match="appears twice"):
        DiscountCurve.from_discounts(
            REFERENCE,
            [(date(2022, 1, 4), 0.99), (date(2022, 1, 4), 0.98)],
            basis=Basis.ACT_365F,
        )


def test_a_non_positive_discount_pillar_is_refused() -> None:
    with pytest.raises(BadCurve, match="strictly positive"):
        DiscountCurve.from_discounts(
            REFERENCE, [(date(2022, 1, 4), 0.0)], basis=Basis.ACT_365F
        )


def test_an_empty_curve_is_refused() -> None:
    with pytest.raises(BadCurve, match="at least one pillar"):
        DiscountCurve.from_discounts(REFERENCE, [], basis=Basis.ACT_365F)


def test_the_compounding_of_the_input_does_not_survive_construction() -> None:
    """The same curve quoted two ways is the same curve.

    Zero rates are converted to discount factors once, so nothing downstream can
    depend on which convention the quotes arrived in.
    """
    continuous = DiscountCurve.from_zeros(
        REFERENCE, ZEROS, basis=Basis.ACT_365F, compounding=Compounding.CONTINUOUS
    )
    as_semi = [
        (day, convert(rate, (day - REFERENCE).days / 365.0, Compounding.CONTINUOUS,
                      Compounding.SEMI_ANNUAL))
        for day, rate in ZEROS
    ]
    semi = DiscountCurve.from_zeros(
        REFERENCE, as_semi, basis=Basis.ACT_365F, compounding=Compounding.SEMI_ANNUAL
    )
    for left, right in zip(continuous.pillars, semi.pillars, strict=True):
        assert left.discount == pytest.approx(right.discount, rel=1e-12)


# -- a flat curve ------------------------------------------------------------


def test_a_flat_curve_returns_its_rate_everywhere() -> None:
    curve = flat_curve(REFERENCE, date(2031, 1, 4), 0.025, basis=Basis.ACT_365F)
    for days in (30, 365, 1000, 3000):
        assert curve.zero_rate(REFERENCE + timedelta(days=days)) == pytest.approx(
            0.025, rel=1e-12
        )


def test_a_flat_curve_has_a_constant_forward() -> None:
    curve = flat_curve(REFERENCE, date(2031, 1, 4), 0.025, basis=Basis.ACT_365F)
    for time in (0.5, 2.0, 5.0, 9.0):
        assert curve.instantaneous_forward(time) == pytest.approx(0.025, abs=1e-8)


def test_a_flat_curve_needs_a_horizon() -> None:
    with pytest.raises(BadCurve, match="horizon after"):
        flat_curve(REFERENCE, REFERENCE, 0.02, basis=Basis.ACT_365F)


# -- what the interpolations actually do -------------------------------------


def test_log_linear_gives_piecewise_constant_forwards() -> None:
    """Flat within each pillar interval, which is the defining property."""
    curve = build(Interpolation.LOG_LINEAR_DISCOUNT)
    inside_one_to_three = [
        curve.instantaneous_forward(time) for time in (1.2, 1.8, 2.4, 2.9)
    ]
    assert max(inside_one_to_three) - min(inside_one_to_three) < 1e-9
    inside_three_to_five = [
        curve.instantaneous_forward(time) for time in (3.2, 4.0, 4.8)
    ]
    assert max(inside_three_to_five) - min(inside_three_to_five) < 1e-9
    # And the two intervals are genuinely different, so this is not measuring a
    # flat curve.
    assert abs(inside_one_to_three[0] - inside_three_to_five[0]) > 1e-3


def test_linear_zero_gives_forwards_that_move_within_an_interval() -> None:
    curve = build(Interpolation.LINEAR_ZERO)
    inside = [curve.instantaneous_forward(time) for time in (1.2, 1.8, 2.4, 2.9)]
    assert max(inside) - min(inside) > 1e-3
    assert inside == sorted(inside)


def test_linear_zero_manufactures_a_negative_forward_from_clean_inputs() -> None:
    """The reason the interpolation is named on the curve.

    ``z(1y) = 6%`` and ``z(2y) = 3.5%`` give strictly decreasing discount
    factors, so the inputs admit no arbitrage and the average forward over the
    year is a positive 1%. Log-linear returns exactly that, flat. Linear on zero
    rates ramps from +3% down to -1% and crosses zero around 1.7 years, entirely
    from the interpolation.
    """
    pillars = [(date(2022, 1, 4), 0.060), (date(2023, 1, 4), 0.035)]
    linear = DiscountCurve.from_zeros(
        REFERENCE,
        pillars,
        basis=Basis.ACT_365F,
        interpolation=Interpolation.LINEAR_ZERO,
    )
    log_linear = linear.with_interpolation(Interpolation.LOG_LINEAR_DISCOUNT)

    # The inputs contain no arbitrage: discount factors strictly decrease.
    factors = [one.discount for one in linear.pillars]
    assert factors == sorted(factors, reverse=True)
    assert factors[1] == pytest.approx(0.941765, abs=1e-6)
    assert factors[2] == pytest.approx(0.932394, abs=1e-6)

    grid = [1.0 + 0.1 * step for step in range(1, 10)]
    log_forwards = [log_linear.instantaneous_forward(time) for time in grid]
    for value in log_forwards:
        assert value == pytest.approx(0.01, abs=1e-8)

    linear_forwards = [linear.instantaneous_forward(time) for time in grid]
    assert max(linear_forwards) == pytest.approx(0.03, abs=1e-6)
    assert min(linear_forwards) == pytest.approx(-0.01, abs=1e-6)
    assert any(value < 0.0 for value in linear_forwards)
    crossing = [time for time, value in zip(grid, linear_forwards, strict=True) if value < 0]
    assert 1.7 <= min(crossing) <= 1.8


def test_log_linear_forwards_stay_positive_while_discounts_decrease() -> None:
    """The guarantee the method actually carries."""
    curve = build(Interpolation.LOG_LINEAR_DISCOUNT)
    factors = [one.discount for one in curve.pillars]
    assert factors == sorted(factors, reverse=True)
    for time in [0.1 * step for step in range(1, 100)]:
        assert curve.instantaneous_forward(time) > 0.0


def test_the_two_interpolations_agree_at_the_pillars_and_nowhere_else() -> None:
    log_linear = build(Interpolation.LOG_LINEAR_DISCOUNT)
    linear = build(Interpolation.LINEAR_ZERO)
    for pillar in log_linear.pillars:
        assert log_linear.discount_at(pillar.time) == pytest.approx(
            linear.discount_at(pillar.time), rel=1e-12
        )
    assert log_linear.discount_at(2.0) != pytest.approx(
        linear.discount_at(2.0), rel=1e-9
    )


# -- monotone convex ---------------------------------------------------------


def test_monotone_convex_holds_the_forward_positive_where_linear_zero_does_not() -> None:
    """The same counterexample, put to the scheme that is meant to survive it.

    Linear on zero rates ramps from +3% to -1% across this interval. Log-linear
    stays flat at the 1% average. Monotone convex is the only one of the three
    that both varies across the interval and stays non-negative, which is the
    entire reason for its existence.
    """
    pillars = [(date(2022, 1, 4), 0.060), (date(2023, 1, 4), 0.035)]
    monotone = DiscountCurve.from_zeros(
        REFERENCE,
        pillars,
        basis=Basis.ACT_365F,
        interpolation=Interpolation.MONOTONE_CONVEX,
    )
    linear = monotone.with_interpolation(Interpolation.LINEAR_ZERO)

    grid = [1.0 + 0.05 * step for step in range(1, 20)]
    monotone_forwards = [monotone.instantaneous_forward(time) for time in grid]
    linear_forwards = [linear.instantaneous_forward(time) for time in grid]

    assert min(linear_forwards) < 0.0
    assert min(monotone_forwards) >= 0.0
    # And it is not flat: it falls across the interval, as the data says it must.
    assert max(monotone_forwards) - min(monotone_forwards) > 1e-3
    assert monotone_forwards == sorted(monotone_forwards, reverse=True)

    # The figure quoted in the README: +2% at the near pillar, straight down to
    # exactly 0% at the far one. The zero is the positivity clamp, not the
    # data - the unconstrained node forward there extrapolates to -0.25%.
    assert monotone.instantaneous_forward(1.0) == pytest.approx(0.02, abs=1e-9)
    assert monotone.instantaneous_forward(2.0) == pytest.approx(0.0, abs=1e-12)
    for time in grid:
        expected = 0.02 - 0.02 * (time - 1.0)
        assert monotone.instantaneous_forward(time) == pytest.approx(expected, abs=1e-9)


def test_monotone_convex_reprices_the_pillars_to_machine_precision() -> None:
    """Not to a tolerance: the interval shapes integrate to zero by construction."""
    curve = build(Interpolation.MONOTONE_CONVEX)
    for pillar in curve.pillars:
        assert curve.discount_at(pillar.time) == pytest.approx(pillar.discount, abs=1e-15)
        assert curve.discount(pillar.day) == pytest.approx(pillar.discount, abs=1e-15)
    assert curve.reprices_pillars(tolerance=1e-15)


def test_the_analytic_forward_agrees_with_a_central_difference() -> None:
    """Two independent routes to the same number.

    The analytic forward reads the interpolated shape directly; the difference
    goes through the discount factors, which are built from the closed-form
    integral of that shape. A sign error in one region's integral would split
    them.
    """
    curve = build(Interpolation.MONOTONE_CONVEX)
    bump = 1e-6
    for step in range(1, 100):
        time = 10.0 * step / 100.0
        analytic = curve.instantaneous_forward(time)
        differenced = -(
            math.log(curve.discount_at(time + bump))
            - math.log(curve.discount_at(time - bump))
        ) / (2.0 * bump)
        assert analytic == pytest.approx(differenced, abs=1e-8)


def test_monotone_convex_gives_continuous_forwards_across_the_pillars() -> None:
    """Log-linear jumps at every pillar. This one does not."""
    curve = build(Interpolation.MONOTONE_CONVEX)
    log_linear = curve.with_interpolation(Interpolation.LOG_LINEAR_DISCOUNT)
    for pillar in curve.pillars[1:-1]:
        step = 1e-4
        before = curve.instantaneous_forward(pillar.time - step)
        after = curve.instantaneous_forward(pillar.time + step)
        assert before == pytest.approx(after, abs=1e-5)
        jump = abs(
            log_linear.instantaneous_forward(pillar.time + step)
            - log_linear.instantaneous_forward(pillar.time - step)
        )
        assert jump > 1e-4


def test_monotone_convex_refuses_a_curve_containing_an_arbitrage() -> None:
    """Discount factors that rise put the positivity clamp out of reach."""
    rising = DiscountCurve.from_discounts(
        REFERENCE,
        [(date(2022, 1, 4), 0.98), (date(2023, 1, 4), 0.99)],
        basis=Basis.ACT_365F,
        interpolation=Interpolation.MONOTONE_CONVEX,
    )
    with pytest.raises(BadCurve, match="arbitrage"):
        rising.discount_at(1.5)


def test_the_monotone_fit_is_available_whatever_the_curve_interpolates_with() -> None:
    """So two schemes can be compared over identical pillars without rebuilding."""
    log_linear = build(Interpolation.LOG_LINEAR_DISCOUNT)
    fitted = log_linear.monotone
    assert fitted.times == log_linear.times
    assert log_linear.monotone is fitted  # fitted once and kept
    monotone = log_linear.with_interpolation(Interpolation.MONOTONE_CONVEX)
    assert monotone.discount_at(3.3) == pytest.approx(fitted.discount_at(3.3))
    assert log_linear.discount_at(3.3) != pytest.approx(fitted.discount_at(3.3))


def test_monotone_convex_still_refuses_to_extrapolate() -> None:
    curve = build(Interpolation.MONOTONE_CONVEX)
    with pytest.raises(OffCurve, match="past the curve"):
        curve.discount(date(2032, 1, 4))
    # But the horizon itself is queryable, rounding and all.
    assert curve.discount(curve.horizon) == pytest.approx(curve.pillars[-1].discount)


# -- querying ----------------------------------------------------------------


def test_discount_factors_decrease_along_a_positive_curve() -> None:
    curve = build()
    previous = 1.0
    for time in [0.25 * step for step in range(1, 40)]:
        factor = curve.discount_at(time)
        assert factor < previous
        previous = factor


def test_a_forward_rate_reproduces_the_ratio_of_discount_factors() -> None:
    curve = build()
    near, far = date(2023, 1, 4), date(2025, 1, 4)
    rate = curve.forward_rate(near, far, Compounding.CONTINUOUS)
    span = curve.time_to(far) - curve.time_to(near)
    assert math.exp(-rate * span) == pytest.approx(
        curve.discount(far) / curve.discount(near), rel=1e-12
    )


def test_a_forward_from_the_reference_date_is_the_zero_rate() -> None:
    curve = build()
    far = date(2026, 1, 4)
    assert curve.forward_rate(
        REFERENCE, far, Compounding.CONTINUOUS
    ) == pytest.approx(curve.zero_rate(far), rel=1e-12)


def test_a_backwards_forward_is_refused() -> None:
    curve = build()
    with pytest.raises(OffCurve, match="not forwards in time"):
        curve.forward_rate(date(2026, 1, 4), date(2024, 1, 4))


def test_extrapolation_past_the_last_pillar_is_refused() -> None:
    """Holding the last factor flat asserts every forward beyond it is zero."""
    curve = build()
    with pytest.raises(OffCurve, match="past the curve's last pillar"):
        curve.discount(date(2032, 1, 1))
    with pytest.raises(OffCurve, match="past the curve's last pillar"):
        curve.discount_at(12.0)


def test_the_last_pillar_itself_is_on_the_curve() -> None:
    curve = build()
    assert curve.discount(curve.horizon) == pytest.approx(curve.pillars[-1].discount)


def test_a_date_before_the_reference_is_refused() -> None:
    curve = build()
    with pytest.raises(OffCurve, match="before the curve's reference date"):
        curve.discount(date(2020, 12, 1))


def test_there_is_no_zero_rate_at_the_reference_date() -> None:
    curve = build()
    with pytest.raises(OffCurve, match="no zero rate"):
        curve.zero_rate(REFERENCE)
    with pytest.raises(OffCurve, match="no zero rate"):
        curve.zero_rate_at(0.0)


def test_zero_rates_are_available_under_any_compounding() -> None:
    curve = build()
    day = date(2026, 1, 4)
    factor = curve.discount(day)
    time = curve.time_to(day)
    for compounding in Compounding:
        rate = curve.zero_rate(day, compounding)
        assert discount_factor(rate, time, compounding) == pytest.approx(
            factor, rel=1e-12
        )


def test_the_curve_reports_its_own_interpolation_and_basis() -> None:
    curve = build(Interpolation.LINEAR_ZERO)
    assert curve.interpolation is Interpolation.LINEAR_ZERO
    assert curve.basis is Basis.ACT_365F
    assert "linear on zero rates" in curve.interpolation.value
    swapped = curve.with_interpolation(Interpolation.LOG_LINEAR_DISCOUNT)
    assert swapped.pillars == curve.pillars
    assert swapped.interpolation is Interpolation.LOG_LINEAR_DISCOUNT
