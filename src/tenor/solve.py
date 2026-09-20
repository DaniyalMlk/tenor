"""Root finding, with the convergence reported rather than assumed.

Three things in this library are defined by an equation rather than a formula:
the discount factor that makes an instrument price to par, the yield that makes
a bond price to its quote, and the spread that makes a risky bond price to its
curve. All three are one-dimensional root finds, and all three are quoted to the
basis point by people who will not be told "it converged".

So the solver returns how it got there — iterations used, the bracket it started
from, and the residual it stopped at — rather than a bare float. A yield that
took the maximum iteration count and stopped at a residual of 1e-3 is not the
same answer as one that took six and stopped at 1e-16, and a function signature
that returns only ``float`` cannot tell the caller which it just handed them.

**Brent's method rather than Newton.** Newton is faster where it works and its
failure mode is silent: with a bad starting point it walks away from the root
and returns whatever it last computed. Brent keeps a bracket around the root at
all times and cannot leave it, so it either returns a root or says it could not
find one. Both are worth having; only one is worth having by default when the
answer is going to be quoted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


class NoRoot(ValueError):
    """The equation has no root the solver could reach."""


@dataclass(frozen=True)
class Root:
    """A solved root, and the evidence that it is one."""

    value: float
    #: The function's value at :attr:`value`. Zero to the extent it converged.
    residual: float
    iterations: int
    #: The interval the search started from, which is part of the answer: a
    #: root found at the edge of a bracket is worth a second look.
    bracket: tuple[float, float]
    #: False when the solver ran out of iterations and returned its best guess.
    converged: bool = True

    def __str__(self) -> str:
        state = "converged" if self.converged else "did not converge"
        return (
            f"{self.value:.10g} ({state} in {self.iterations} iterations, "
            f"residual {self.residual:.3g})"
        )


def bracket_root(
    function: Callable[[float], float],
    low: float,
    high: float,
    *,
    growth: float = 1.6,
    attempts: int = 60,
) -> tuple[float, float]:
    """Widen ``[low, high]`` until the function changes sign across it.

    Expands whichever end currently has the smaller absolute value, which walks
    towards the root rather than symmetrically away from the starting guess.
    Sixty attempts at a growth of 1.6 covers about ten orders of magnitude,
    which is far more than any rate or spread needs and cheap enough not to
    tune.
    """
    if high <= low:
        raise NoRoot(f"a bracket runs from low to high, got [{low!r}, {high!r}]")
    low_value = function(low)
    high_value = function(high)
    if low_value == 0.0:
        return low, low
    if high_value == 0.0:
        return high, high
    for _ in range(attempts):
        if low_value * high_value < 0.0:
            return low, high
        span = high - low
        if abs(low_value) < abs(high_value):
            low -= growth * span
            low_value = function(low)
        else:
            high += growth * span
            high_value = function(high)
    raise NoRoot(
        f"no sign change found after {attempts} widenings, ending at "
        f"[{low!r}, {high!r}] with values [{low_value!r}, {high_value!r}]. The "
        "function may not cross zero at all, which for a pricing problem "
        "usually means the quote itself is unattainable rather than that the "
        "search was too narrow."
    )


def brent(
    function: Callable[[float], float],
    low: float,
    high: float,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 100,
) -> Root:
    """Solve ``function(x) == 0`` on a bracket that contains a sign change.

    Brent's method: inverse quadratic interpolation where the three most recent
    points support it, secant where they do not, and bisection whenever either
    would step outside the bracket or fail to halve the interval fast enough.
    The bookkeeping that enforces the last part is the whole method — without
    it, interpolation can stall arbitrarily close to the root without reaching
    it, which is exactly the case a naive secant implementation hits on a
    convex price-yield curve.

    ``tolerance`` is on ``x``, not on the residual. A residual tolerance is
    meaningless without knowing the function's scale, and the scale differs by
    orders of magnitude between a discount factor and a bond price.
    """
    a, b = low, high
    fa, fb = function(a), function(b)
    if fa == 0.0:
        return Root(a, 0.0, 0, (low, high))
    if fb == 0.0:
        return Root(b, 0.0, 0, (low, high))
    if fa * fb > 0.0:
        raise NoRoot(
            f"the function does not change sign across [{low!r}, {high!r}]: "
            f"it is {fa!r} at one end and {fb!r} at the other. Brent's method "
            "needs a bracket; use bracket_root to find one."
        )
    # ``b`` is always the best estimate, ``a`` the other end of the bracket,
    # ``c`` the previous ``b`` and ``d`` the one before that. Keeping ``b`` the
    # smaller in absolute value is what makes the interval checks below read as
    # "is the step towards the good end".
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    d = c
    bisected = True

    for iteration in range(1, max_iterations + 1):
        if fb == 0.0 or abs(b - a) <= tolerance:
            return Root(b, fb, iteration - 1, (low, high))

        if fa != fc and fb != fc:
            # Inverse quadratic: fit x as a quadratic in f through all three.
            candidate = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            candidate = b - fb * (b - a) / (fb - fa)

        # The five conditions under which the interpolated step is not
        # trustworthy and the interval is halved instead. The first keeps the
        # step inside the quarter of the bracket nearest the best estimate; the
        # rest force the step to keep halving so interpolation cannot stall
        # arbitrarily close to the root without reaching it, which is exactly
        # what a bare secant does on a convex price-yield curve.
        quarter = (3.0 * a + b) / 4.0
        outside = not (min(quarter, b) < candidate < max(quarter, b))
        if (
            outside
            or (bisected and abs(candidate - b) >= 0.5 * abs(b - c))
            or (not bisected and abs(candidate - b) >= 0.5 * abs(c - d))
            or (bisected and abs(b - c) < tolerance)
            or (not bisected and abs(c - d) < tolerance)
        ):
            candidate = 0.5 * (a + b)
            bisected = True
        else:
            bisected = False

        value = function(candidate)
        d, c, fc = c, b, fb
        if fa * value < 0.0:
            b, fb = candidate, value
        else:
            a, fa = candidate, value
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa

    return Root(b, fb, max_iterations, (low, high), converged=False)


def solve(
    function: Callable[[float], float],
    *,
    low: float,
    high: float,
    tolerance: float = 1e-14,
    max_iterations: int = 100,
) -> Root:
    """Bracket, then solve. The combination every caller here actually wants."""
    left, right = bracket_root(function, low, high)
    if left == right:
        return Root(left, 0.0, 0, (low, high))
    return brent(
        function, left, right, tolerance=tolerance, max_iterations=max_iterations
    )


__all__ = ["NoRoot", "Root", "bracket_root", "brent", "solve"]
