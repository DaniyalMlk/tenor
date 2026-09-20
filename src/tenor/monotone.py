"""The monotone convex interpolation scheme of Hagan and West (2006).

The two schemes already in :mod:`tenor.curve` each give up something. Log-linear
on discount factors keeps forward rates positive but makes them piecewise
constant, which is a picture no market ever produced. Linear on zero rates is
smooth in the quantity people quote and can manufacture a negative forward from
inputs that contain none — the counterexample is worked in the curve docstring.

Monotone convex is the scheme that declines to choose. It interpolates the
*forward rate* rather than the zero rate or the log discount factor, and it does
so under a constraint the other two cannot express: whatever happens inside a
pillar interval, the forward rate must average back to the discrete forward the
interval was built from.

That constraint is the whole method, so it is worth stating precisely. Write
``y(t) = -log P(t)``, the integrated instantaneous forward. Between two pillars
the discrete forward is

    f_i = (y_i - y_{i-1}) / (t_i - t_{i-1})

and any instantaneous forward curve that reprices both pillars must integrate to
exactly that over the interval. Write the instantaneous forward as
``f_i + G(x)`` on ``x in [0, 1]``, and the repricing condition becomes

    integral of G over [0, 1] = 0

which is an identity rather than a tolerance. So the pillars are reproduced to
machine precision no matter which shape ``G`` takes, and the method is free to
spend all of its freedom on the shape.

**What the shape has to do.** ``G`` is pinned at both ends by the node forward
rates — the instantaneous forward *at* each pillar, which is shared with the
neighbouring interval and is what makes the forward curve continuous. Given
those two endpoints and a zero integral, the natural choice is the quadratic
through them, and for most inputs that is exactly what this module uses. But a
quadratic with a fixed integral has no freedom left to stay inside its own
endpoints: for endpoint deviations far enough apart it overshoots, and the
overshoot is what drives a forward rate negative.

Hagan and West's answer is to split the interval instead. When the quadratic
would overshoot, ``G`` is flat over part of the interval and quadratic over the
rest, with the split point chosen so the integral is still zero. That gives four
cases, and which one applies depends only on the ratio of the two endpoint
deviations — not on their size, not on the interval length, not on the level of
rates. The classification here is written in terms of that ratio for exactly
that reason: one number decides it, and the regions meet at ``-2`` and
``-1/2`` where the formulas agree.

**Positivity is a clamp, not an emergent property.** Before any of this, the
node forwards are forced into ``[0, 2 min(f_i, f_{i+1})]``. The upper bound is
what makes the interpolant's undershoot recoverable; the lower bound is what
makes the result non-negative. The clamp assumes the discrete forwards are
themselves non-negative, which is the one input this module refuses rather than
handles: a pillar set implying a negative discrete forward is an arbitrage in
the inputs, and clamping a node into a range whose upper bound is below its
lower bound would return a confident number from it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class NotMonotone(ValueError):
    """Pillars the monotone convex scheme cannot be fitted through."""


class Region(str, Enum):
    """Which of the four shapes an interval's interpolant takes.

    Reported on the segment rather than kept private, because "which region"
    is the first question when a forward curve looks wrong, and the answer is
    otherwise only recoverable by re-deriving the classification by hand.
    """

    #: Both endpoints zero. The forward is flat at the discrete forward.
    FLAT = "flat"
    #: The plain quadratic through both endpoints. Hagan-West region (i).
    QUADRATIC = "quadratic"
    #: Flat at the left value, then quadratic. Hagan-West region (ii).
    LEFT_FLAT = "flat then quadratic"
    #: Quadratic, then flat at the right value. Hagan-West region (iii).
    RIGHT_FLAT = "quadratic then flat"
    #: Quadratic to an interior level and quadratic away from it, used when both
    #: endpoints sit on the same side of the discrete forward. Region (iv).
    BOWED = "bowed"


#: Endpoint deviations closer to zero than this are treated as zero. They arrive
#: as differences of rates, so the scale is a rate: a thousandth of a basis
#: point is far below anything quoted and far above the rounding in the
#: subtraction that produced them.
_NEGLIGIBLE = 1e-14

#: How far outside the fitted range a query may land and still be read as the
#: boundary. Relative to the horizon, and about four orders of magnitude above
#: the rounding in a year fraction at thirty years.
_SNAP = 1e-12


@dataclass(frozen=True)
class Segment:
    """One pillar interval, and the forward rate shape across it.

    ``value`` and ``integral`` are written in the local coordinate
    ``x = (t - start) / (end - start)``, which is what makes the four region
    formulas comparable with the published ones. Callers work in years and go
    through :class:`MonotoneConvex`, which does the scaling.
    """

    start: float
    end: float
    #: The discrete forward over the interval: the average the shape must hit.
    discrete: float
    #: The node forward at the left end, less the discrete forward.
    left_deviation: float
    #: The node forward at the right end, less the discrete forward.
    right_deviation: float
    region: Region
    #: The split point, for the regions that have one. ``None`` otherwise.
    split: float | None = None
    #: The interior level the bowed region bends around. ``None`` otherwise.
    level: float | None = None

    @property
    def span(self) -> float:
        return self.end - self.start

    def deviation(self, x: float) -> float:
        """``G(x)``: the instantaneous forward at ``x``, less the discrete one."""
        g0, g1 = self.left_deviation, self.right_deviation
        if self.region is Region.FLAT:
            return 0.0
        if self.region is Region.QUADRATIC:
            return g0 * (1.0 - 4.0 * x + 3.0 * x * x) + g1 * (-2.0 * x + 3.0 * x * x)
        split = self.split
        assert split is not None
        if self.region is Region.LEFT_FLAT:
            if x <= split:
                return g0
            ratio = (x - split) / (1.0 - split)
            return g0 + (g1 - g0) * ratio * ratio
        if self.region is Region.RIGHT_FLAT:
            if x >= split:
                return g1
            ratio = (split - x) / split
            return g1 + (g0 - g1) * ratio * ratio
        level = self.level
        assert level is not None
        if x < split:
            ratio = (split - x) / split
            return level + (g0 - level) * ratio * ratio
        ratio = (x - split) / (1.0 - split)
        return level + (g1 - level) * ratio * ratio

    def deviation_integral(self, x: float) -> float:
        """The integral of :meth:`deviation` from ``0`` to ``x``.

        In closed form rather than quadrature. Every branch below integrates to
        zero at ``x = 1`` by construction — that is the repricing identity — and
        the test suite asserts it for each region separately rather than
        trusting the algebra.
        """
        g0, g1 = self.left_deviation, self.right_deviation
        if self.region is Region.FLAT:
            return 0.0
        if self.region is Region.QUADRATIC:
            return g0 * (x - 2.0 * x * x + x**3) + g1 * (-(x * x) + x**3)
        split = self.split
        assert split is not None
        if self.region is Region.LEFT_FLAT:
            if x <= split:
                return g0 * x
            return g0 * x + (g1 - g0) * (x - split) ** 3 / (3.0 * (1.0 - split) ** 2)
        if self.region is Region.RIGHT_FLAT:
            if x < split:
                return g1 * x + (g0 - g1) * (split**3 - (split - x) ** 3) / (
                    3.0 * split * split
                )
            whole = g1 * split + (g0 - g1) * split / 3.0
            return whole + g1 * (x - split)
        level = self.level
        assert level is not None
        if x < split:
            return level * x + (g0 - level) * (split**3 - (split - x) ** 3) / (
                3.0 * split * split
            )
        left = level * split + (g0 - level) * split / 3.0
        return (
            left
            + level * (x - split)
            + (g1 - level) * (x - split) ** 3 / (3.0 * (1.0 - split) ** 2)
        )


def classify(left_deviation: float, right_deviation: float) -> Region:
    """Which shape the two endpoint deviations call for.

    Hagan and West state the four regions as eight sign-and-magnitude
    conditions. They collapse to one number. With both deviations non-zero, let
    ``rho = g1 / g0``:

    ======================  =========================
    ``rho``                 region
    ======================  =========================
    ``rho > 0``             bowed, the same side
    ``-1/2 < rho < 0``      quadratic then flat
    ``-2 <= rho <= -1/2``   plain quadratic
    ``rho < -2``            flat then quadratic
    ======================  =========================

    The ratio form is not a simplification for its own sake: it makes the two
    boundaries visible as the numbers they are, and it is symmetric under
    negating both deviations, which the sign-by-sign statement obscures.

    A zero endpoint is the case the published region (iv) formula cannot take —
    its split point degenerates to an interval end and the shape collapses to a
    constant that misses the other endpoint entirely. The plain quadratic
    handles it correctly, and does so while respecting the positivity bound,
    because a clamped node deviation is never larger in magnitude than the
    discrete forward it deviates from.
    """
    g0, g1 = left_deviation, right_deviation
    flat_left = abs(g0) < _NEGLIGIBLE
    flat_right = abs(g1) < _NEGLIGIBLE
    if flat_left and flat_right:
        return Region.FLAT
    if flat_left or flat_right:
        return Region.QUADRATIC
    ratio = g1 / g0
    if ratio > 0.0:
        return Region.BOWED
    if ratio < -2.0:
        return Region.LEFT_FLAT
    if ratio > -0.5:
        return Region.RIGHT_FLAT
    return Region.QUADRATIC


def _segment(start: float, end: float, discrete: float, g0: float, g1: float) -> Segment:
    region = classify(g0, g1)
    split: float | None = None
    level: float | None = None
    if region is Region.LEFT_FLAT:
        split = (g1 + 2.0 * g0) / (g1 - g0)
    elif region is Region.RIGHT_FLAT:
        split = 3.0 * g1 / (g1 - g0)
    elif region is Region.BOWED:
        split = g1 / (g1 + g0)
        level = -g0 * g1 / (g0 + g1)
    return Segment(
        start=start,
        end=end,
        discrete=discrete,
        left_deviation=g0,
        right_deviation=g1,
        region=region,
        split=split,
        level=level,
    )


def node_forwards(times: Sequence[float], discrete: Sequence[float]) -> tuple[float, ...]:
    """The instantaneous forward at each pillar, clamped to keep it positive.

    Interior nodes are the length-weighted average of the two discrete forwards
    meeting there — weighted towards the *shorter* side, which is the part most
    people transpose. The weight on the right-hand discrete forward is the
    *left* interval's length over the total, so a long interval followed by a
    short one puts most of the weight on the short one's forward. That is the
    correct reading of a finite difference of ``y`` at an unevenly spaced node,
    and transposing it produces a curve that still reprices every pillar and is
    wrong everywhere between them.

    The two boundary nodes have only one neighbour, so they are extrapolated
    from it and the adjacent interior node.

    Everything is then clamped into ``[0, 2 min(neighbouring discretes)]``.
    """
    count = len(discrete)
    for index, forward in enumerate(discrete):
        if forward < 0.0:
            raise NotMonotone(
                f"the pillars imply a discrete forward rate of {forward!r} between "
                f"{times[index]!r} and {times[index + 1]!r} years. Monotone convex "
                "keeps forwards non-negative by clamping each node into a range whose "
                "upper bound is twice the neighbouring discrete forwards, and that "
                "range is empty once one of them is negative. The inputs contain an "
                "arbitrage rather than a shape problem: discount factors that rise "
                "over an interval."
            )
    raw: list[float] = [0.0] * (count + 1)
    for index in range(1, count):
        left_span = times[index] - times[index - 1]
        right_span = times[index + 1] - times[index]
        total = left_span + right_span
        raw[index] = (
            left_span * discrete[index] + right_span * discrete[index - 1]
        ) / total
    if count == 1:
        raw[0] = discrete[0]
        raw[1] = discrete[0]
    else:
        raw[0] = discrete[0] - 0.5 * (raw[1] - discrete[0])
        raw[count] = discrete[count - 1] - 0.5 * (raw[count - 1] - discrete[count - 1])

    clamped: list[float] = []
    for index in range(count + 1):
        if index == 0:
            ceiling = 2.0 * discrete[0]
        elif index == count:
            ceiling = 2.0 * discrete[count - 1]
        else:
            ceiling = 2.0 * min(discrete[index - 1], discrete[index])
        clamped.append(min(max(raw[index], 0.0), ceiling))
    return tuple(clamped)


@dataclass(frozen=True)
class MonotoneConvex:
    """A fitted monotone convex interpolant over one set of pillars.

    Holds ``y(t) = -log P(t)`` at the pillars and the per-interval shape, and
    answers in the same quantity, so a curve built on it never converts a rate
    to fit and back to answer.
    """

    times: tuple[float, ...]
    integrated: tuple[float, ...]
    discrete: tuple[float, ...]
    nodes: tuple[float, ...]
    segments: tuple[Segment, ...]

    @classmethod
    def fit(cls, times: Sequence[float], integrated: Sequence[float]) -> MonotoneConvex:
        """Fit through ``y(t_i) = integrated[i]``, with ``times[0] == 0``.

        ``integrated`` is ``-log P``, which is the zero rate times the time
        under continuous compounding. Passing that rather than the rate keeps
        the anchor well defined: ``y(0) = 0`` says nothing about a rate at zero,
        where none exists.
        """
        if len(times) != len(integrated):
            raise NotMonotone(
                f"{len(times)} times against {len(integrated)} values"
            )
        if len(times) < 2:
            raise NotMonotone("monotone convex needs at least one interval to fit")
        for index in range(1, len(times)):
            if times[index] <= times[index - 1]:
                raise NotMonotone(
                    f"times must increase, but {times[index]!r} does not follow "
                    f"{times[index - 1]!r}"
                )
        discrete = tuple(
            (integrated[index] - integrated[index - 1]) / (times[index] - times[index - 1])
            for index in range(1, len(times))
        )
        nodes = node_forwards(times, discrete)
        segments = tuple(
            _segment(
                times[index],
                times[index + 1],
                discrete[index],
                nodes[index] - discrete[index],
                nodes[index + 1] - discrete[index],
            )
            for index in range(len(discrete))
        )
        return cls(
            times=tuple(times),
            integrated=tuple(integrated),
            discrete=discrete,
            nodes=nodes,
            segments=segments,
        )

    def _locate(self, time: float) -> int:
        """The index of the segment containing ``time``.

        A query at the horizon is allowed to miss it by a rounding error.
        Callers reach the last pillar's year fraction by their own arithmetic —
        a date difference over a day count denominator — rather than by the
        addition the fit used, and the two agree only to the last bit or two.
        Refusing the difference would make the horizon unqueryable from the
        outside for no reason a caller could act on. The tolerance is relative,
        because an absolute one is meaningless at thirty years and punitive at
        one month.
        """
        slack = _SNAP * max(1.0, abs(self.times[-1]))
        if time < self.times[0] - slack or time > self.times[-1] + slack:
            raise NotMonotone(
                f"{time!r} is outside the fitted range "
                f"[{self.times[0]!r}, {self.times[-1]!r}]"
            )
        time = min(max(time, self.times[0]), self.times[-1])
        low, high = 0, len(self.segments) - 1
        while low < high:
            middle = (low + high) // 2
            if time <= self.segments[middle].end:
                high = middle
            else:
                low = middle + 1
        return low

    def _place(self, time: float) -> tuple[int, float]:
        """The segment index and the local coordinate within it."""
        index = self._locate(time)
        segment = self.segments[index]
        clamped = min(max(time, segment.start), segment.end)
        return index, (clamped - segment.start) / segment.span

    def integrated_at(self, time: float) -> float:
        """``y(t) = -log P(t)``, interpolated."""
        index, x = self._place(time)
        segment = self.segments[index]
        return (
            self.integrated[index]
            + segment.span * (segment.discrete * x + segment.deviation_integral(x))
        )

    def discount_at(self, time: float) -> float:
        return math.exp(-self.integrated_at(time))

    def forward_at(self, time: float) -> float:
        """The instantaneous forward rate, analytically.

        At a pillar this is the node forward, which both neighbouring segments
        agree on — that agreement is what makes the forward curve continuous,
        and it is asserted in the tests rather than assumed from the
        construction.
        """
        index, x = self._place(time)
        return self.segments[index].discrete + self.segments[index].deviation(x)


__all__ = [
    "MonotoneConvex",
    "NotMonotone",
    "Region",
    "Segment",
    "classify",
    "node_forwards",
]
