"""Tests for the curve risk decompositions.

The claim under test is an identity - key rate durations add to the total
duration - and identities checked at one bump size prove very little, because
the residual of a central difference is second order in the bump and looks like
rounding at any single value. So the residual is measured at three bump sizes
and asserted to fall as the square, which distinguishes "correct, differenced"
from "nearly correct".

The shift shapes are checked to sum to exactly one, at points inside and
outside the bucket range. Outside is where the construction goes wrong if the
end shapes are not held flat, and it is also where nobody looks.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from tenor.bond import Bond
from tenor.bootstrap import Bootstrapped, bootstrap
from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis
from tenor.instruments import Deposit, Future, Swap
from tenor.rates import Compounding
from tenor.risk import (
    BadRisk,
    Bucket,
    Valuation,
    buckets_from,
    bumped,
    curvature,
    instrument_risk,
    key_rates,
    level,
    nearest_bucket,
    shape_duration,
    shifted_by,
    slope,
    tent_weights,
    total_instrument_risk,
)

REFERENCE = date(2021, 1, 4)


def built_curve(
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
) -> Bootstrapped:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2021, 7, 5), 0.002, Basis.ACT_360),
        Swap(REFERENCE, date(2023, 1, 4), 0.006),
        Swap(REFERENCE, date(2026, 1, 5), 0.015),
        Swap(REFERENCE, date(2031, 1, 6), 0.022),
        Swap(REFERENCE, date(2041, 1, 7), 0.027),
    ]
    return bootstrap(
        REFERENCE, quotes, basis=Basis.ACT_365F, interpolation=interpolation
    )


def bond_value(bond: Bond) -> Valuation:
    return lambda curve: bond.price_from_curve(curve, REFERENCE)


# -- the shift shapes ---------------------------------------------------------


def test_the_tent_shapes_sum_to_exactly_one_everywhere() -> None:
    """Including outside the bucket range, which is where it goes wrong."""
    weights = tent_weights([0.5, 2.0, 5.0, 10.0, 20.0])
    probes = [0.0, 0.1, 0.5, 0.9, 2.0, 3.7, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
    for time in probes:
        assert sum(one(time) for one in weights) == pytest.approx(1.0, abs=1e-15)


def test_each_tent_is_one_at_its_own_bucket_and_zero_at_the_others() -> None:
    times = [0.5, 2.0, 5.0, 10.0, 20.0]
    weights = tent_weights(times)
    for index, weight in enumerate(weights):
        for other, time in enumerate(times):
            expected = 1.0 if other == index else 0.0
            assert weight(time) == pytest.approx(expected, abs=1e-15)


def test_the_end_tents_are_held_flat_past_the_ends() -> None:
    """The detail the sum identity depends on.

    Falling to zero outside the range instead would leave the shapes summing
    to less than one there, and the key rate durations short by exactly the
    long-dated exposure they exist to measure.
    """
    weights = tent_weights([2.0, 5.0, 10.0])
    assert weights[0](0.0) == 1.0
    assert weights[0](1.0) == 1.0
    assert weights[0](2.0) == 1.0
    assert weights[-1](10.0) == 1.0
    assert weights[-1](25.0) == 1.0
    # And they still go to zero towards the interior.
    assert weights[0](5.0) == 0.0
    assert weights[-1](5.0) == 0.0


def test_a_single_bucket_is_a_parallel_shift() -> None:
    weights = tent_weights([7.0])
    assert len(weights) == 1
    for time in (0.0, 1.0, 7.0, 40.0):
        assert weights[0](time) == 1.0


def test_buckets_have_to_increase() -> None:
    with pytest.raises(BadRisk, match="must increase"):
        tent_weights([5.0, 2.0])
    with pytest.raises(BadRisk, match="at least one bucket"):
        tent_weights([])


def test_the_named_shapes_are_normalised_as_documented() -> None:
    """Both conventions in use differ by a factor of two, so this is pinned."""
    horizon = 20.0
    flat = level()
    tilt = slope(horizon)
    hump = curvature(horizon)
    for time in (0.0, 5.0, 10.0, 20.0):
        assert flat(time) == 1.0
    assert tilt(0.0) == pytest.approx(-1.0)
    assert tilt(10.0) == pytest.approx(0.0)
    assert tilt(20.0) == pytest.approx(1.0)
    assert hump(0.0) == pytest.approx(1.0)
    assert hump(10.0) == pytest.approx(-1.0)
    assert hump(20.0) == pytest.approx(1.0)
    assert hump(5.0) == pytest.approx(-0.5)


def test_a_shape_needs_a_positive_horizon() -> None:
    with pytest.raises(BadRisk, match="positive horizon"):
        slope(0.0)
    with pytest.raises(BadRisk, match="positive horizon"):
        curvature(-1.0)


def test_a_level_shift_is_the_curves_own_parallel_shift() -> None:
    """Two routes to the same curve, so neither is checking only itself."""
    curve = built_curve().curve
    by_shape = shifted_by(curve, 1e-3, level())
    by_method = curve.shifted(1e-3)
    for left, right in zip(by_shape.pillars, by_method.pillars, strict=True):
        assert left.discount == pytest.approx(right.discount, rel=1e-15)


def test_a_shift_under_another_compounding_matches_the_curves_own() -> None:
    curve = built_curve().curve
    by_shape = shifted_by(curve, 1e-3, level(), Compounding.SEMI_ANNUAL)
    by_method = curve.shifted(1e-3, Compounding.SEMI_ANNUAL)
    for left, right in zip(by_shape.pillars, by_method.pillars, strict=True):
        assert left.discount == pytest.approx(right.discount, rel=1e-15)


def test_a_slope_shift_moves_the_ends_in_opposite_directions() -> None:
    curve = built_curve().curve
    horizon = curve.times[-1]
    steepened = shifted_by(curve, 0.01, slope(horizon))
    assert steepened.zero_rate(date(2021, 7, 5)) < curve.zero_rate(date(2021, 7, 5))
    assert steepened.zero_rate(date(2041, 1, 7)) > curve.zero_rate(date(2041, 1, 7))


# -- key rate durations -------------------------------------------------------


def test_key_rate_durations_sum_to_the_total_duration() -> None:
    """And the residual falls as the square of the bump, which is the evidence.

    At one bump size a small residual proves nothing: it could be a correct
    decomposition differenced, or a decomposition that is slightly wrong. The
    second-order scaling separates them. Measured here: 2.06e-05, 2.06e-07 and
    2.06e-09 relative, at bumps of 1e-3, 1e-4 and 1e-5.
    """
    built = built_curve()
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    buckets = buckets_from(built.curve, [0.5, 2.0, 5.0, 10.0, 20.0])

    residuals = []
    for bump in (1e-3, 1e-4, 1e-5):
        total = shape_duration(value, built.curve, level(), shift=bump)
        parts = key_rates(value, built.curve, buckets, shift=bump)
        residuals.append(abs(sum(one.duration for one in parts) - total) / total)

    assert residuals[0] == pytest.approx(2.06e-5, rel=0.02)
    assert residuals[1] == pytest.approx(2.06e-7, rel=0.02)
    assert residuals[2] == pytest.approx(2.06e-9, rel=0.02)
    for coarse, fine in pairwise(residuals):
        assert coarse / fine == pytest.approx(100.0, rel=0.02)


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_the_sum_identity_holds_under_every_interpolation(
    interpolation: Interpolation,
) -> None:
    """And is measurably looser under monotone convex, for a stateable reason.

    A key rate bump can move a pillar far enough to change which of the four
    shapes monotone convex uses on an interval. The price is continuous across
    that boundary but its second derivative is not, so the central difference
    straddles a kink and the residual is about seven times the local schemes'.
    Real, small, and worth knowing before someone reads the difference as a
    bug in the decomposition.
    """
    built = built_curve(interpolation)
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    buckets = buckets_from(built.curve, [1.0, 3.0, 7.0, 15.0])
    total = shape_duration(value, built.curve, level())
    parts = key_rates(value, built.curve, buckets)
    residual = abs(sum(one.duration for one in parts) - total) / total
    if interpolation is Interpolation.MONOTONE_CONVEX:
        assert residual == pytest.approx(1.1e-6, rel=0.2)
    else:
        assert residual < 3e-7


def test_the_risk_sits_where_the_cash_flows_are() -> None:
    """A fifteen-year bullet is mostly twenty-year and ten-year risk.

    The whole reason for the decomposition: a single duration of 11.2 years
    says nothing about whether this is a fifteen-year bond or a barbell of
    twos and thirties, and these numbers do.
    """
    built = built_curve()
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    buckets = buckets_from(built.curve, [0.5, 2.0, 5.0, 10.0, 20.0])
    parts = key_rates(value, built.curve, buckets)
    durations = {one.bucket.name: one.duration for one in parts}
    assert durations["0.5y"] < 0.1
    assert durations["2y"] < 0.5
    assert durations["10y"] > 3.0
    assert durations["20y"] > 5.0
    assert all(one.duration > 0.0 for one in parts)


def test_a_barbell_and_a_bullet_of_the_same_duration_look_different() -> None:
    """The case the decomposition exists for, and a single duration cannot see."""
    built = built_curve()
    bullet = Bond(REFERENCE, date(2033, 1, 4), 0.04)
    short = Bond(REFERENCE, date(2024, 1, 4), 0.04)
    long = Bond(REFERENCE, date(2041, 1, 4), 0.04)
    buckets = buckets_from(built.curve, [1.0, 3.0, 10.0, 20.0])

    # Weight the barbell so its total duration matches the bullet's.
    target = shape_duration(bond_value(bullet), built.curve, level())
    short_duration = shape_duration(bond_value(short), built.curve, level())
    long_duration = shape_duration(bond_value(long), built.curve, level())
    weight = (long_duration - target) / (long_duration - short_duration)

    # Weighted by *value*, not by price. A portfolio's duration is the
    # value-weighted average of its parts' durations, so holding one unit of
    # each and weighting the prices would put the weights on the wrong
    # quantity and miss the target by 3%.
    short_base = short.price_from_curve(built.curve, REFERENCE)
    long_base = long.price_from_curve(built.curve, REFERENCE)

    def barbell(curve: DiscountCurve) -> float:
        one = short.price_from_curve(curve, REFERENCE) / short_base
        other = long.price_from_curve(curve, REFERENCE) / long_base
        return weight * one + (1.0 - weight) * other

    assert shape_duration(barbell, built.curve, level()) == pytest.approx(
        target, rel=1e-3
    )

    bullet_rates = [one.duration for one in key_rates(bond_value(bullet), built.curve, buckets)]
    barbell_rates = [one.duration for one in key_rates(barbell, built.curve, buckets)]
    assert sum(bullet_rates) == pytest.approx(sum(barbell_rates), rel=1e-3)
    # Same total, and the barbell carries visibly more at both ends.
    assert barbell_rates[0] > 2.0 * bullet_rates[0]
    assert barbell_rates[-1] > bullet_rates[-1]


def test_the_money_value_is_the_duration_times_price_times_the_bump() -> None:
    built = built_curve()
    bond = Bond(REFERENCE, date(2036, 1, 4), 0.05)
    value = bond_value(bond)
    buckets = buckets_from(built.curve, [2.0, 10.0, 20.0])
    price = value(built.curve)
    for one in key_rates(value, built.curve, buckets, shift=1e-4):
        assert one.value == pytest.approx(one.duration * price * 1e-4, rel=1e-14)


def test_a_bucket_off_the_curve_is_refused() -> None:
    """A shift where there are no pillars moves nothing, which is not zero risk."""
    built = built_curve()
    with pytest.raises(BadRisk, match="outside the curve"):
        buckets_from(built.curve, [5.0, 50.0])
    with pytest.raises(BadRisk, match="outside the curve"):
        buckets_from(built.curve, [0.0])


def test_something_worth_nothing_has_no_duration() -> None:
    built = built_curve()
    with pytest.raises(BadRisk, match="worth nothing"):
        shape_duration(lambda _: 0.0, built.curve, level())


def test_the_nearest_bucket_is_reported_for_a_tenor() -> None:
    buckets = [Bucket(1.0), Bucket(5.0), Bucket(10.0)]
    assert nearest_bucket(buckets, 0.2).time == 1.0
    assert nearest_bucket(buckets, 2.0).time == 1.0
    assert nearest_bucket(buckets, 4.0).time == 5.0
    assert nearest_bucket(buckets, 20.0).time == 10.0
    with pytest.raises(BadRisk, match="no buckets"):
        nearest_bucket([], 1.0)


# -- instrument risk ----------------------------------------------------------


def test_bumping_a_quote_raises_its_rate_whatever_the_quoting_convention() -> None:
    """A future is quoted as 100 minus the rate, so its price has to fall.

    Getting this backwards gives a hedge of the right size pointing the wrong
    way, which is worse than no hedge at all.
    """
    deposit = Deposit(REFERENCE, date(2021, 7, 5), 0.003)
    swap = Swap(REFERENCE, date(2031, 1, 6), 0.02)
    future = Future(date(2021, 9, 15), date(2021, 12, 15), 99.55)

    moved_deposit = bumped(deposit, 1e-4)
    assert isinstance(moved_deposit, Deposit)
    assert moved_deposit.rate == pytest.approx(0.0031)
    moved_swap = bumped(swap, 1e-4)
    assert isinstance(moved_swap, Swap)
    assert moved_swap.rate == pytest.approx(0.0201)
    moved = bumped(future, 1e-4)
    assert isinstance(moved, Future)
    assert moved.price < future.price
    assert moved.futures_rate == pytest.approx(future.futures_rate + 1e-4)


def test_every_instrument_carries_some_of_the_risk() -> None:
    built = built_curve()
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    risks = instrument_risk(value, built)
    assert len(risks) == len(built.instruments)
    # The long swaps carry most of it, and the deposit almost none. Not every
    # entry has to be positive - see the negative five-year below - but under
    # this interpolation they are.
    assert all(one.value > 0.0 for one in risks)
    ordered = sorted(risks, key=lambda one: one.value)
    assert ordered[0].name.startswith("deposit")
    assert "2041" in ordered[-1].name


def test_the_instrument_risks_add_up_to_moving_every_quote_at_once() -> None:
    """Two routes to one number, the second not a sum of the first.

    They agree to 7.5e-04 relative. That gap is the bootstrap's own
    nonlinearity - bumping five quotes together is not the sum of bumping them
    one at a time, because each one changes the curve the others are solved
    against - and it is first order in the bump because these are one-sided
    differences, which is how a desk bumps a quote.
    """
    built = built_curve()
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    risks = instrument_risk(value, built)
    together = total_instrument_risk(value, built)
    summed = sum(one.value for one in risks)
    assert summed == pytest.approx(together, rel=1e-3)
    assert abs(summed - together) / together == pytest.approx(7.5e-4, rel=0.1)


def test_instrument_risk_is_not_the_key_rate_decomposition() -> None:
    """Different questions, different answers, and the totals differ too.

    A swap quote is a statement about its whole annuity, so bumping it moves
    every zero rate out to its maturity rather than one bucket's worth. The
    two decompositions are therefore genuinely different numbers, not one set
    regrouped - which is the reason both exist.
    """
    built = built_curve()
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    buckets = buckets_from(built.curve, [0.5, 2.0, 5.0, 10.0, 20.0])
    key = {one.bucket.name: one.value for one in key_rates(value, built.curve, buckets)}
    instrument = {one.name: one.value for one in instrument_risk(value, built)}

    ten_year_key = key["10y"]
    ten_year_swap = instrument["swap to 2031-01-06"]
    assert ten_year_key != pytest.approx(ten_year_swap, rel=0.05)
    # Both decompositions are of a real exposure, so both are positive and of
    # the same order; they are simply not the same split.
    assert ten_year_key > 0.0 and ten_year_swap > 0.0


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_instrument_risk_works_under_every_interpolation(
    interpolation: Interpolation,
) -> None:
    """Each bump is a full rebuild, so monotone convex re-sweeps every time."""
    built = built_curve(interpolation)
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    risks = instrument_risk(value, built)
    assert sum(one.value for one in risks) == pytest.approx(
        total_instrument_risk(value, built), rel=2e-3
    )


def test_the_total_is_interpolation_independent_and_the_hedge_is_not() -> None:
    """The caution anyone hedging off this needs, stated in numbers.

    Across the three interpolations the *total* instrument risk of the same
    bond on the same quotes spans 1.3%. The breakdown spans far more: the
    ten-year entry moves by +64%, the twenty-year by -25%, and the five-year
    changes sign outright.

    A hedge built from these is a hedge built on the interpolation, which is an
    assumption about the gaps between quotes rather than anything the market
    said. The total is what the quotes determine; the split between them is
    what the curve builder decided.
    """
    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    totals = {}
    breakdowns = {}
    for scheme in Interpolation:
        built = built_curve(scheme)
        risks = instrument_risk(value, built)
        totals[scheme] = sum(one.value for one in risks)
        breakdowns[scheme] = [one.value for one in risks]

    spread = max(totals.values()) - min(totals.values())
    assert spread / min(totals.values()) == pytest.approx(0.0132, rel=0.05)

    local = breakdowns[Interpolation.LOG_LINEAR_DISCOUNT]
    smooth = breakdowns[Interpolation.MONOTONE_CONVEX]
    assert smooth[3] / local[3] == pytest.approx(1.64, rel=0.05)
    assert smooth[4] / local[4] == pytest.approx(0.75, rel=0.05)
    assert local[2] > 0.0 > smooth[2]


def test_a_negative_instrument_risk_is_a_real_answer() -> None:
    """Raising a shorter par swap rate can raise a longer bond's price.

    The par rates further out are held fixed while the five-year moves, so the
    forwards beyond five years have to fall to keep them there. For a bond
    whose risk sits mostly past ten years, that fall can outweigh the rise in
    the three-to-seven region - and under monotone convex, where it is about
    three times larger, it does.

    Checked at the curve rather than only at the price, so this pins the
    mechanism and not just the sign.
    """
    built = built_curve(Interpolation.MONOTONE_CONVEX)
    quotes = list(built.instruments)
    quotes[2] = bumped(quotes[2], 1e-4)
    rebuilt = bootstrap(
        REFERENCE,
        quotes,
        basis=Basis.ACT_365F,
        interpolation=Interpolation.MONOTONE_CONVEX,
    )
    # Up in the middle, down at the long end.
    assert rebuilt.curve.zero_rate_at(5.0) > built.curve.zero_rate_at(5.0)
    assert rebuilt.curve.zero_rate_at(15.0) < built.curve.zero_rate_at(15.0)

    value = bond_value(Bond(REFERENCE, date(2036, 1, 4), 0.05))
    assert value(rebuilt.curve) > value(built.curve)
    assert instrument_risk(value, built)[2].value < 0.0


def test_a_futures_strip_carries_risk_with_the_right_sign() -> None:
    """The sign check that a futures quoting convention can invert."""
    quotes: list[Deposit | Future | Swap] = [
        Deposit(REFERENCE, date(2021, 4, 5), 0.002, Basis.ACT_360),
        Future(date(2021, 6, 16), date(2021, 9, 15), 99.70, Basis.ACT_360),
        Future(date(2021, 9, 15), date(2021, 12, 15), 99.55, Basis.ACT_360),
        Swap(REFERENCE, date(2026, 1, 5), 0.015),
    ]
    built = bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F)
    value = bond_value(Bond(REFERENCE, date(2026, 1, 5), 0.03))
    risks = instrument_risk(value, built)
    # The bond matures at the last quote, so every quote is inside its life and
    # a rise in any of them lowers its price. A futures price bumped the wrong
    # way would invert one of these.
    assert all(one.value > 0.0 for one in risks)
    assert sum(one.value for one in risks) == pytest.approx(
        total_instrument_risk(value, built), rel=1e-3
    )
