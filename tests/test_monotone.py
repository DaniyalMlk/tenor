"""Tests for the monotone convex interpolation scheme.

The scheme makes four promises: it reprices its pillars, its forward curve is
continuous, the forwards stay non-negative when the inputs contain no
arbitrage, and a monotone set of discrete forwards produces a monotone forward
curve. Each is checked here as a property over generated inputs rather than at
a handful of points, because all four are the kind of claim that holds at every
case anybody writes by hand and fails in one region nobody thought to try.

The region formulas are also checked individually, in both directions: the
closed-form integral is differentiated back to the shape it integrates, and the
shape is checked at both endpoints. A transposed coefficient in one region is
invisible in a curve-level test, because the other three regions still carry
the pillars.
"""

from __future__ import annotations

import math
import random

import pytest

from tenor.monotone import (
    MonotoneConvex,
    NotMonotone,
    Region,
    _segment,
    classify,
    node_forwards,
)


def _arbitrage_free_curve(
    rng: random.Random, *, intervals: int, floor: float = 0.0, ceiling: float = 0.12
) -> tuple[list[float], list[float], list[float]]:
    """Times, integrated forwards, and the discrete forwards that produced them.

    Built forwards from non-negative discrete forwards rather than backwards
    from random discount factors, so the inputs are arbitrage-free by
    construction and any negative forward the scheme returns is the scheme's
    own doing.
    """
    times = [0.0]
    for _ in range(intervals):
        times.append(times[-1] + rng.uniform(0.05, 3.0))
    discrete = [rng.uniform(floor, ceiling) for _ in range(intervals)]
    integrated = [0.0]
    for index, forward in enumerate(discrete):
        integrated.append(integrated[-1] + forward * (times[index + 1] - times[index]))
    return times, integrated, discrete


# -- the region classification ------------------------------------------------


@pytest.mark.parametrize(
    ("g0", "g1", "expected"),
    [
        (0.0, 0.0, Region.FLAT),
        (0.0, 0.03, Region.QUADRATIC),
        (0.02, 0.0, Region.QUADRATIC),
        # Opposite signs, decided by the ratio alone.
        (0.02, -0.02, Region.QUADRATIC),  # ratio -1, mid-band
        (0.02, -0.04, Region.QUADRATIC),  # ratio -2, the boundary itself
        (0.02, -0.01, Region.QUADRATIC),  # ratio -1/2, the other boundary
        (0.02, -0.05, Region.LEFT_FLAT),  # ratio -2.5
        (0.02, -0.005, Region.RIGHT_FLAT),  # ratio -1/4
        # Negating both deviations must not move the region.
        (-0.02, 0.05, Region.LEFT_FLAT),
        (-0.02, 0.005, Region.RIGHT_FLAT),
        (-0.02, 0.02, Region.QUADRATIC),
        # Same sign, either sign.
        (0.02, 0.01, Region.BOWED),
        (-0.02, -0.01, Region.BOWED),
    ],
)
def test_the_region_is_decided_by_the_ratio(g0: float, g1: float, expected: Region) -> None:
    assert classify(g0, g1) is expected


def test_the_classification_is_symmetric_under_negation() -> None:
    """Negating both deviations mirrors the picture and must not change the shape."""
    rng = random.Random(11)
    for _ in range(2000):
        g0 = rng.uniform(-0.1, 0.1)
        g1 = rng.uniform(-0.1, 0.1)
        assert classify(g0, g1) is classify(-g0, -g1)


def test_every_pair_of_deviations_lands_in_a_region() -> None:
    """The four regions partition the plane, with no pair falling through."""
    rng = random.Random(3)
    seen = set()
    for _ in range(20000):
        g0 = rng.uniform(-0.2, 0.2)
        g1 = rng.uniform(-0.2, 0.2)
        region = classify(g0, g1)
        assert isinstance(region, Region)
        seen.add(region)
    # Region.FLAT needs both deviations at zero, which random sampling will
    # never produce, so it is excluded here and covered by the table above.
    assert seen == set(Region) - {Region.FLAT}


# -- the shapes themselves ----------------------------------------------------


def _representative_segments() -> list[tuple[Region, float, float]]:
    """One deviation pair per region, with room either side of each boundary."""
    return [
        (Region.FLAT, 0.0, 0.0),
        (Region.QUADRATIC, 0.015, -0.018),
        (Region.LEFT_FLAT, 0.012, -0.040),
        (Region.RIGHT_FLAT, 0.020, -0.004),
        (Region.BOWED, 0.010, 0.030),
        (Region.BOWED, -0.010, -0.030),
        (Region.QUADRATIC, 0.0, 0.025),
        (Region.QUADRATIC, 0.025, 0.0),
    ]


@pytest.mark.parametrize(("region", "g0", "g1"), _representative_segments())
def test_each_shape_hits_both_endpoints(region: Region, g0: float, g1: float) -> None:
    segment = _segment(1.0, 3.0, 0.05, g0, g1)
    assert segment.region is region
    assert segment.deviation(0.0) == pytest.approx(g0, abs=1e-15)
    assert segment.deviation(1.0) == pytest.approx(g1, abs=1e-15)


@pytest.mark.parametrize(("region", "g0", "g1"), _representative_segments())
def test_each_shape_integrates_to_zero(region: Region, g0: float, g1: float) -> None:
    """The repricing identity, per region.

    This is the one property the whole method rests on: whatever shape the
    forward takes inside an interval, its average over the interval is the
    discrete forward, so both pillars are reproduced exactly.
    """
    segment = _segment(1.0, 3.0, 0.05, g0, g1)
    assert segment.deviation_integral(0.0) == 0.0
    assert segment.deviation_integral(1.0) == pytest.approx(0.0, abs=1e-16)


@pytest.mark.parametrize(("region", "g0", "g1"), _representative_segments())
def test_the_closed_form_integral_differentiates_back(
    region: Region, g0: float, g1: float
) -> None:
    """The integral is stated in closed form, so nothing checks it but this.

    Differentiated numerically rather than integrated numerically, because the
    shapes have a kink at the split point that quadrature handles badly and
    differencing away from it handles exactly.
    """
    segment = _segment(1.0, 3.0, 0.05, g0, g1)
    split = segment.split
    bump = 1e-6
    for step in range(1, 100):
        x = step / 100.0
        if split is not None and abs(x - split) < 10.0 * bump:
            continue  # the derivative does not exist at the join
        slope = (
            segment.deviation_integral(x + bump) - segment.deviation_integral(x - bump)
        ) / (2.0 * bump)
        assert slope == pytest.approx(segment.deviation(x), abs=1e-9)


def test_the_flat_regions_really_are_flat_over_their_share() -> None:
    """The point of splitting the interval is that part of it stops moving."""
    left_flat = _segment(0.0, 1.0, 0.05, 0.012, -0.040)
    assert left_flat.region is Region.LEFT_FLAT
    split = left_flat.split
    assert split is not None and 0.0 < split < 1.0
    for step in range(11):
        assert left_flat.deviation(split * step / 10.0) == pytest.approx(0.012)

    right_flat = _segment(0.0, 1.0, 0.05, 0.020, -0.004)
    assert right_flat.region is Region.RIGHT_FLAT
    split = right_flat.split
    assert split is not None and 0.0 < split < 1.0
    for step in range(11):
        x = split + (1.0 - split) * step / 10.0
        assert right_flat.deviation(x) == pytest.approx(-0.004)


def test_the_regions_agree_where_they_meet() -> None:
    """At a ratio of exactly -2 the split-interval shape has nothing to split.

    Approaching the boundary from the flat-then-quadratic side, the flat part
    shrinks to nothing and the shape has to converge on the plain quadratic. A
    discontinuity here would mean two nearly identical curves interpolating
    visibly differently.
    """
    quadratic = _segment(0.0, 1.0, 0.05, 0.02, -0.04)
    assert quadratic.region is Region.QUADRATIC
    approaching = _segment(0.0, 1.0, 0.05, 0.02, -0.04 * (1.0 + 1e-9))
    assert approaching.region is Region.LEFT_FLAT
    for step in range(21):
        x = step / 20.0
        assert approaching.deviation(x) == pytest.approx(quadratic.deviation(x), abs=1e-7)


# -- the node forwards --------------------------------------------------------


def test_an_interior_node_weights_towards_the_shorter_side() -> None:
    """The weight on each discrete forward is the *other* interval's length.

    Pinned against a hand-computed case with a three-to-one length ratio,
    because the transposed version still reprices every pillar and is wrong
    everywhere between them. With a three-year interval at 4% followed by a
    one-year interval at 6%, the node between them sits at 5.5%, not 4.5%: it
    is dragged towards the short interval's forward, which is the one that has
    to change quickly to hit its pillar.

    The rates are chosen so the positivity clamp does not bind. Its ceiling is
    twice the smaller neighbouring forward, which is close enough to an
    ordinary interior node to swallow a weighting error in a less careful
    example.
    """
    times = [0.0, 3.0, 4.0]
    discrete = [0.04, 0.06]
    nodes = node_forwards(times, discrete)
    assert nodes[1] < 2.0 * min(discrete), "the clamp must not be what is under test"
    assert nodes[1] == pytest.approx(3.0 / 4.0 * 0.06 + 1.0 / 4.0 * 0.04)
    assert nodes[1] == pytest.approx(0.055)
    transposed = 1.0 / 4.0 * 0.06 + 3.0 / 4.0 * 0.04
    assert nodes[1] != pytest.approx(transposed)


def test_the_nodes_are_clamped_into_the_positivity_range() -> None:
    """Twice the smaller neighbouring discrete forward is a hard ceiling."""
    times = [0.0, 1.0, 2.0]
    discrete = [0.06, 0.01]
    nodes = node_forwards(times, discrete)
    # The raw interior node is the average, 3.5%, well above twice the smaller
    # neighbour, so it is pulled down to exactly that bound.
    assert nodes[1] == pytest.approx(2.0 * 0.01)
    # The right-hand boundary node extrapolates to -0.25% and is floored.
    assert nodes[2] == 0.0
    assert all(node >= 0.0 for node in nodes)


def test_the_boundary_nodes_extrapolate_from_the_unclamped_interior() -> None:
    """The clamp is applied once, at the end, to every node together.

    Clamping the interior nodes first and extrapolating the boundaries from the
    clamped values is a defensible-sounding order that gives a different curve,
    so the order is pinned.
    """
    times = [0.0, 1.0, 2.0, 3.0]
    discrete = [0.04, 0.05, 0.055]
    nodes = node_forwards(times, discrete)
    interior_one = 0.5 * 0.05 + 0.5 * 0.04
    interior_two = 0.5 * 0.055 + 0.5 * 0.05
    assert nodes[1] == pytest.approx(interior_one)
    assert nodes[2] == pytest.approx(interior_two)
    assert nodes[0] == pytest.approx(0.04 - 0.5 * (interior_one - 0.04))
    assert nodes[3] == pytest.approx(0.055 - 0.5 * (interior_two - 0.055))


def test_a_single_interval_is_flat() -> None:
    """One interval has no interior node, so there is nothing to shape it with."""
    nodes = node_forwards([0.0, 5.0], [0.03])
    assert nodes == (0.03, 0.03)
    fitted = MonotoneConvex.fit([0.0, 5.0], [0.0, 0.15])
    assert fitted.segments[0].region is Region.FLAT
    for step in range(11):
        assert fitted.forward_at(5.0 * step / 10.0) == pytest.approx(0.03)


def test_a_negative_discrete_forward_is_refused() -> None:
    """Discount factors that rise are an arbitrage, not a shape problem."""
    with pytest.raises(NotMonotone, match="arbitrage"):
        node_forwards([0.0, 1.0, 2.0], [0.02, -0.01])
    with pytest.raises(NotMonotone, match="arbitrage"):
        MonotoneConvex.fit([0.0, 1.0, 2.0], [0.0, 0.02, 0.01])


# -- fitting ------------------------------------------------------------------


def test_the_fit_is_refused_on_inputs_that_are_not_a_curve() -> None:
    with pytest.raises(NotMonotone, match="at least one interval"):
        MonotoneConvex.fit([0.0], [0.0])
    with pytest.raises(NotMonotone, match="must increase"):
        MonotoneConvex.fit([0.0, 2.0, 1.0], [0.0, 0.04, 0.02])
    with pytest.raises(NotMonotone, match="against"):
        MonotoneConvex.fit([0.0, 1.0], [0.0])


def test_off_the_fitted_range_is_refused() -> None:
    fitted = MonotoneConvex.fit([0.0, 1.0, 2.0], [0.0, 0.03, 0.05])
    with pytest.raises(NotMonotone, match="outside the fitted range"):
        fitted.forward_at(2.5)
    with pytest.raises(NotMonotone, match="outside the fitted range"):
        fitted.integrated_at(-0.5)
    # A rounding error at the horizon is not off the curve. A caller reaches
    # the last pillar's year fraction by a different arithmetic path than the
    # fit did, and the two agree only to the last bit.
    assert fitted.forward_at(2.0 * (1.0 + 1e-15)) == pytest.approx(fitted.nodes[-1])


def test_the_fit_reprices_every_pillar() -> None:
    rng = random.Random(19)
    worst = 0.0
    for _ in range(500):
        times, integrated, _ = _arbitrage_free_curve(rng, intervals=rng.randint(1, 8))
        fitted = MonotoneConvex.fit(times, integrated)
        for index, time in enumerate(times):
            worst = max(worst, abs(fitted.integrated_at(time) - integrated[index]))
    assert worst < 1e-14


def test_the_forward_curve_is_continuous_at_the_pillars() -> None:
    """Both segments meeting at a pillar must return that pillar's node forward.

    This is what the node forwards are *for*. Interpolating each interval
    independently would reprice every pillar just as well and leave a forward
    curve that jumps at each one.

    Checked at the join exactly, by asking each of the two segments meeting
    there for its own end, rather than by stepping either side of the pillar.
    Stepping measures the segment's slope as much as its value, and the slope
    scales with one over the interval length, so a fixed step gives a tolerance
    that means something different on a one-month interval than on a five-year
    one.
    """
    rng = random.Random(23)
    for _ in range(200):
        times, integrated, _ = _arbitrage_free_curve(rng, intervals=rng.randint(2, 7))
        fitted = MonotoneConvex.fit(times, integrated)
        for index in range(1, len(times) - 1):
            before = fitted.segments[index - 1]
            after = fitted.segments[index]
            left_limit = before.discrete + before.deviation(1.0)
            right_limit = after.discrete + after.deviation(0.0)
            assert left_limit == pytest.approx(fitted.nodes[index], abs=1e-15)
            assert right_limit == pytest.approx(fitted.nodes[index], abs=1e-15)


def test_the_forwards_never_go_negative() -> None:
    """The property the scheme exists for, over generated arbitrage-free input.

    The generator builds the curve forwards from non-negative discrete
    forwards, so every input here is arbitrage-free and any negative forward
    would be manufactured by the interpolation, which is exactly the failure
    that linear-on-zero-rates has and this scheme does not.
    """
    rng = random.Random(29)
    worst = math.inf
    for _ in range(600):
        times, integrated, _ = _arbitrage_free_curve(rng, intervals=rng.randint(1, 8))
        fitted = MonotoneConvex.fit(times, integrated)
        span = times[-1]
        for step in range(201):
            worst = min(worst, fitted.forward_at(span * step / 200.0))
    assert worst >= 0.0


def test_monotone_discrete_forwards_give_a_monotone_forward_curve() -> None:
    """The other half of the name, and the reason for the amended regions.

    The plain quadratic through two endpoints with a fixed integral overshoots
    for endpoint pairs far enough apart, and an overshoot in a rising curve is
    a dip. Generated with sorted discrete forwards so the input is monotone by
    construction.
    """
    rng = random.Random(31)
    for _ in range(300):
        times, _, discrete = _arbitrage_free_curve(rng, intervals=rng.randint(2, 7))
        discrete = sorted(discrete)
        integrated = [0.0]
        for index, forward in enumerate(discrete):
            integrated.append(
                integrated[-1] + forward * (times[index + 1] - times[index])
            )
        fitted = MonotoneConvex.fit(times, integrated)
        span = times[-1]
        previous = -math.inf
        for step in range(201):
            current = fitted.forward_at(span * step / 200.0)
            assert current >= previous - 1e-12
            previous = current


def test_the_integrated_forward_is_the_integral_of_the_forward() -> None:
    """``y(t)`` and ``f(t)`` are computed by different code paths.

    One sums a closed-form integral, the other evaluates the shape directly, so
    agreeing is evidence rather than tautology. A sign error in one region's
    integral would show here and nowhere in the repricing tests, which only see
    the endpoints.
    """
    rng = random.Random(37)
    for _ in range(60):
        times, integrated, _ = _arbitrage_free_curve(rng, intervals=rng.randint(2, 5))
        fitted = MonotoneConvex.fit(times, integrated)
        span = times[-1]
        bump = 1e-7 * span
        for step in range(1, 60):
            time = span * step / 60.0
            slope = (
                fitted.integrated_at(time + bump) - fitted.integrated_at(time - bump)
            ) / (2.0 * bump)
            assert slope == pytest.approx(fitted.forward_at(time), abs=1e-7)


def test_a_flat_input_stays_flat() -> None:
    """A constant forward is the case every scheme must get exactly right."""
    times = [0.0, 0.25, 1.0, 2.5, 7.0]
    integrated = [0.035 * time for time in times]
    fitted = MonotoneConvex.fit(times, integrated)
    assert all(segment.region is Region.FLAT for segment in fitted.segments)
    for step in range(71):
        time = 7.0 * step / 70.0
        assert fitted.forward_at(time) == pytest.approx(0.035)
        assert fitted.discount_at(time) == pytest.approx(math.exp(-0.035 * time))


def test_every_region_is_reached_by_real_curves() -> None:
    """A region no input reaches is untested code, whatever the unit tests say."""
    rng = random.Random(41)
    reached = set()
    for _ in range(1500):
        times, integrated, _ = _arbitrage_free_curve(rng, intervals=rng.randint(1, 8))
        fitted = MonotoneConvex.fit(times, integrated)
        reached.update(segment.region for segment in fitted.segments)
    assert reached == set(Region)
