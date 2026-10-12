"""A forward curve with fewer parameters than quotes.

Every other curve in this library passes exactly through its inputs.
:func:`tenor.bootstrap` solves one pillar per instrument, so the parameter count
equals the quote count and the fit is an identity: there is nothing left over to
inspect, because nothing was left over. That is the right construction for a
swap curve, where the quotes are consistent by arrangement and a curve that
missed one of them would be mispriced against the market.

A government bond curve is not that. Thirty bonds of differing coupon, age and
liquidity disagree with one another, there is no curve that prices all of them,
and a construction that insists on passing through every quote turns the
disagreement into shape. What is wanted instead is a smooth form with three or
four parameters, fitted by least squares, so the disagreement lands in the
residuals where it can be read.

**The forms.** Both are written on the *instantaneous forward rate*, which is
where they are simple:

- Nelson-Siegel: ``f(t) = b0 + b1 exp(-x) + b2 x exp(-x)``, with ``x = t / t1``.
  A level, a slope that decays, and a hump whose position is set by ``t1``.
- Svensson: the same with a second hump, ``b3 y exp(-y)`` at ``y = t / t2``,
  because one hump cannot fit a curve that has a dip at two years and a peak at
  ten.

**The integral, not the average.** The zero rate is the average forward over
``[0, t]`` and every textbook writes it that way, with a ``(1 - exp(-x)) / x``
in it. Discounting does not want the average: it wants the integral, and the
integral is the average times ``t``, which puts the ``t`` back in the numerator
and removes the division entirely. So the primitive here is
:meth:`ParametricCurve.integral` and the zero rate is derived from it rather
than the other way round. The difference is only visible at ``t = 0``, where the
average needs a special case and the integral is simply zero — but ``t = 0`` is
the anchor of every curve, so it is visible on every curve.

**What is linear and what is not.** With the decay parameters ``t1`` and ``t2``
held fixed, the zero rate is *linear* in the level parameters. So a fit to
observed zero rates is an exact weighted least squares in three or four
unknowns, and only the one or two decays have to be searched. That is the whole
reason the decays are separated out and searched on a grid: the inner problem
has a closed-form answer, so the outer problem is one- or two-dimensional and
can be examined rather than trusted.

A fit to bond *prices* is not linear — a price is an exponential of the
parameters — so that route is Gauss-Newton with the analytic Jacobian, started
from the linear fit to the bonds' own yields.

**The decays are weakly identified, and the output says so.** The objective is
flat in ``t1`` over a wide range: many decays reprice the same quotes within a
basis point, and reporting the arg-min of a flat function as *the* curve's hump
position is a false precision. Every fit here therefore carries a
:class:`DecaySet` — the range of decays that reprice within a stated tolerance,
and the spread the level parameters take across that range. Svensson is worse
than Nelson-Siegel for the same reason it is more flexible: as ``t1``
approaches ``t2`` its two hump functions become the same function, the design
matrix loses rank, and the level parameters run off in opposite directions while
the curve barely moves.

**Weighting is a choice, so it is named.** Equal weight on price errors is not
equal weight on yield errors. A two-year bond's yield moves its price about a
fiftieth as much as a thirty-year bond's does, so a price-weighted fit spends
its accuracy at the long end and leaves the short end to drift.
:class:`Weighting` makes that explicit and ``YIELD`` divides each price residual
by the quote's own price sensitivity, which puts the residuals in yield units
before they are squared.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from .bond import Bond
from .curve import DiscountCurve, Interpolation
from .daycount import Basis, year_fraction

__all__ = [
    "BadFit",
    "CurveFit",
    "DecaySet",
    "Family",
    "ParametricCurve",
    "Weighting",
    "fit_bonds",
    "fit_zeros",
    "solve_least_squares",
]

#: Below this the small-argument series is used for the hump integral, which is
#: a difference of two quantities of order ``x`` that cancel to order ``x**2``.
SERIES_LIMIT = 1e-4

#: The default grid resolution per decay parameter, logarithmically spaced.
GRID_POINTS = 48


class BadFit(ValueError):
    """The quotes cannot be fitted by this form."""


class Family(str, Enum):
    """Which functional form the forward rate takes."""

    #: Level, decaying slope, one hump.
    NELSON_SIEGEL = "Nelson-Siegel"
    #: The same with a second hump on its own decay.
    SVENSSON = "Svensson"

    @property
    def levels(self) -> int:
        """How many parameters the forward rate is linear in."""
        return 3 if self is Family.NELSON_SIEGEL else 4

    @property
    def decays(self) -> int:
        """How many decay parameters have to be searched."""
        return 1 if self is Family.NELSON_SIEGEL else 2


class Weighting(str, Enum):
    """What the squared residuals are measured in."""

    #: Each bond's price error, per 100 of notional, weighted equally.
    PRICE = "equal weight on price errors"
    #: Each price error divided by that bond's own price sensitivity, so the
    #: residuals are in yield units before they are squared.
    YIELD = "equal weight on yield errors"


# -- the basis functions -------------------------------------------------------


def _mean_decay(x: float) -> float:
    """The average of ``exp(-u)`` over ``[0, x]``, which is ``(1 - e**-x) / x``.

    Only used for reporting the zero rate; ``expm1`` keeps it accurate for small
    ``x`` and the value at zero is one.
    """
    if x == 0.0:
        return 1.0
    return -math.expm1(-x) / x


def _hump_integral(x: float) -> float:
    """``1 - (1 + x) exp(-x)``, the integral of ``u exp(-u)`` over ``[0, x]``.

    Written directly this is a difference of two quantities of order ``x`` whose
    leading terms cancel, leaving an answer of order ``x**2``: at ``x = 1e-8``
    the direct form has eight significant figures left and the series has all of
    them. The series is ``sum (-1)**n (n - 1) x**n / n!`` from ``n = 2``.
    """
    if x < 0.0:
        raise BadFit(f"the hump integral is not defined at a negative time ({x!r})")
    if x < SERIES_LIMIT:
        return x * x * (0.5 - x * (1.0 / 3.0 - x * (1.0 / 8.0 - x / 30.0)))
    return -math.expm1(-x) - x * math.exp(-x)


def _integral_basis(t: float, decays: Sequence[float], family: Family) -> list[float]:
    """The integrals of each forward basis function over ``[0, t]``.

    This is the primitive: a discount factor is ``exp(-sum b_k G_k(t))``, and
    every ``G_k`` is zero at ``t = 0`` with no special case needed.
    """
    first = decays[0]
    x = t / first
    terms = [t, -first * math.expm1(-x), first * _hump_integral(x)]
    if family is Family.SVENSSON:
        second = decays[1]
        y = t / second
        terms.append(second * _hump_integral(y))
    return terms


def _forward_basis(t: float, decays: Sequence[float], family: Family) -> list[float]:
    """The forward basis functions themselves, evaluated at ``t``."""
    x = t / decays[0]
    terms = [1.0, math.exp(-x), x * math.exp(-x)]
    if family is Family.SVENSSON:
        y = t / decays[1]
        terms.append(y * math.exp(-y))
    return terms


def _zero_basis(t: float, decays: Sequence[float], family: Family) -> list[float]:
    """The zero-rate basis functions, which are the integrals divided by ``t``.

    At ``t = 0`` the limits are taken rather than the division attempted: a
    level of one, a slope of one, and humps of zero.
    """
    if t == 0.0:
        return [1.0, 1.0, 0.0] + ([0.0] if family is Family.SVENSSON else [])
    x = t / decays[0]
    terms = [1.0, _mean_decay(x), _mean_decay(x) - math.exp(-x)]
    if family is Family.SVENSSON:
        y = t / decays[1]
        terms.append(_mean_decay(y) - math.exp(-y))
    return terms


# -- the curve -----------------------------------------------------------------


@dataclass(frozen=True)
class ParametricCurve:
    """A forward curve given by a formula rather than by a set of pillars.

    ``levels`` are the parameters the forward rate is linear in and ``decays``
    are the ones it is not. Both are in the order the family writes them.
    """

    reference: date
    family: Family
    levels: tuple[float, ...]
    decays: tuple[float, ...]
    basis: Basis = Basis.ACT_365F

    def __post_init__(self) -> None:
        if len(self.levels) != self.family.levels:
            raise BadFit(
                f"{self.family.value} has {self.family.levels} level parameters, "
                f"not {len(self.levels)}"
            )
        if len(self.decays) != self.family.decays:
            raise BadFit(
                f"{self.family.value} has {self.family.decays} decay parameters, "
                f"not {len(self.decays)}"
            )
        for decay in self.decays:
            if not decay > 0.0 or not math.isfinite(decay):
                raise BadFit(
                    f"a decay parameter is a time in years and must be positive; "
                    f"got {decay!r}"
                )
        for level in self.levels:
            if not math.isfinite(level):
                raise BadFit(f"a level parameter is not finite: {level!r}")

    # -- times ---------------------------------------------------------------

    def time(self, day: date) -> float:
        """Years from the reference date to ``day`` on the curve's own basis."""
        return year_fraction(self.reference, day, self.basis)

    # -- the three readings --------------------------------------------------

    def forward(self, t: float) -> float:
        """The instantaneous forward rate at time ``t``, continuously compounded."""
        terms = _forward_basis(t, self.decays, self.family)
        return sum(b * g for b, g in zip(self.levels, terms, strict=True))

    def integral(self, t: float) -> float:
        """The integral of the instantaneous forward over ``[0, t]``.

        The quantity discounting actually needs. Zero at ``t = 0``.
        """
        terms = _integral_basis(t, self.decays, self.family)
        return sum(b * g for b, g in zip(self.levels, terms, strict=True))

    def zero(self, t: float) -> float:
        """The continuously compounded zero rate to ``t``: the average forward."""
        terms = _zero_basis(t, self.decays, self.family)
        return sum(b * g for b, g in zip(self.levels, terms, strict=True))

    def discount_at(self, t: float) -> float:
        """The discount factor to time ``t``."""
        return math.exp(-self.integral(t))

    def discount(self, day: date) -> float:
        """The discount factor to ``day``."""
        return self.discount_at(self.time(day))

    @property
    def long_rate(self) -> float:
        """The limit of the forward rate as ``t`` grows: the level parameter."""
        return self.levels[0]

    @property
    def short_rate(self) -> float:
        """The instantaneous forward at the reference date."""
        return self.levels[0] + self.levels[1]

    # -- interoperation ------------------------------------------------------

    def to_discount_curve(
        self,
        days: Sequence[date],
        *,
        interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    ) -> DiscountCurve:
        """Sample the form at ``days`` and hand back an ordinary pillar curve.

        Everything in this library that takes a :class:`DiscountCurve` then
        works on a fitted curve without knowing it was fitted. The sampling is
        lossy between pillars, by exactly the amount the chosen interpolation
        differs from the form — which is a reason to sample densely rather than
        a reason to avoid the conversion.
        """
        if not days:
            raise BadFit("a curve needs at least one date to be sampled at")
        points = [(day, self.discount(day)) for day in days]
        return DiscountCurve.from_discounts(
            self.reference, points, basis=self.basis, interpolation=interpolation
        )

    def price(self, bond: Bond, settlement: date) -> float:
        """The bond's dirty price on this curve, per 100, forward to settlement.

        The same convention as :meth:`tenor.Bond.price_from_curve`: each flow's
        discount factor is divided by the settlement date's, so the answer is a
        price on the settlement date and not a present value at the reference
        date.
        """
        base = self.integral(self.time(settlement))
        return sum(
            flow.amount * math.exp(-(self.integral(self.time(flow.day)) - base))
            for flow in bond.cashflows(settlement)
        )


# -- linear algebra, in the sizes this needs -----------------------------------


def solve_least_squares(
    rows: Sequence[Sequence[float]], targets: Sequence[float]
) -> tuple[float, ...]:
    """The least-squares solution of ``rows @ x = targets`` by Householder QR.

    The alternative is to form ``A'A`` and solve that, which is shorter and
    squares the condition number. On a Svensson design with its two decays a
    thousandth apart the design's condition number is 3.1e+04, so the normal
    equations work at 1e+09 and lose nine digits: recovering a known parameter
    set, the reflections are wrong by 6.4e-14 and the normal equations by
    6.6e-10. Four orders of magnitude, on a matrix with four columns, which is
    why the reflections are used even at this size.
    """
    height = len(rows)
    if height != len(targets):
        raise BadFit(f"{height} rows against {len(targets)} targets")
    if height == 0:
        raise BadFit("a least-squares problem needs at least one row")
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise BadFit("the rows of the design matrix are not the same length")
    if height < width:
        raise BadFit(
            f"{height} quotes cannot determine {width} parameters; this form needs "
            f"at least {width}"
        )
    matrix = [list(row) for row in rows]
    rhs = list(targets)
    for column in range(width):
        norm = math.sqrt(sum(matrix[i][column] ** 2 for i in range(column, height)))
        if norm == 0.0:
            raise BadFit(
                f"column {column} of the design matrix is zero below the diagonal; "
                "the parameters are not separately determined by these quotes"
            )
        alpha = -norm if matrix[column][column] >= 0.0 else norm
        vector = [0.0] * height
        for i in range(column, height):
            vector[i] = matrix[i][column]
        vector[column] -= alpha
        scale = sum(vector[i] ** 2 for i in range(column, height))
        if scale == 0.0:
            continue
        for j in range(column, width):
            inner = sum(vector[i] * matrix[i][j] for i in range(column, height))
            factor = 2.0 * inner / scale
            for i in range(column, height):
                matrix[i][j] -= factor * vector[i]
        inner = sum(vector[i] * rhs[i] for i in range(column, height))
        factor = 2.0 * inner / scale
        for i in range(column, height):
            rhs[i] -= factor * vector[i]
    solution = [0.0] * width
    for i in range(width - 1, -1, -1):
        if matrix[i][i] == 0.0:
            raise BadFit(
                "the design matrix is singular; with these decays the form has "
                "fewer independent shapes than parameters"
            )
        total = rhs[i] - sum(matrix[i][j] * solution[j] for j in range(i + 1, width))
        solution[i] = total / matrix[i][i]
    return tuple(solution)


def _normal_equations(
    rows: Sequence[Sequence[float]], targets: Sequence[float]
) -> tuple[float, ...]:
    """The same solution by forming ``A'A`` and running Cholesky on it.

    Kept because the roadmap quotes what it costs on an ill-conditioned design,
    and a measurement nobody can reproduce is not a measurement. Not exported.
    """
    width = len(rows[0])
    gram = [
        [sum(row[i] * row[j] for row in rows) for j in range(width)]
        for i in range(width)
    ]
    moment = [sum(row[i] * t for row, t in zip(rows, targets, strict=True)) for i in range(width)]
    lower = [[0.0] * width for _ in range(width)]
    for i in range(width):
        for j in range(i + 1):
            total = gram[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise BadFit("the normal equations are not positive definite")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    forward = [0.0] * width
    for i in range(width):
        forward[i] = (
            moment[i] - sum(lower[i][k] * forward[k] for k in range(i))
        ) / lower[i][i]
    back = [0.0] * width
    for i in range(width - 1, -1, -1):
        back[i] = (
            forward[i] - sum(lower[k][i] * back[k] for k in range(i + 1, width))
        ) / lower[i][i]
    return tuple(back)


def _jacobi_eigenvalues(symmetric: Sequence[Sequence[float]]) -> list[float]:
    """Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations."""
    size = len(symmetric)
    matrix = [list(row) for row in symmetric]
    for _ in range(100):
        off = max(
            (abs(matrix[i][j]), i, j)
            for i in range(size)
            for j in range(size)
            if i != j
        ) if size > 1 else (0.0, 0, 0)
        magnitude, p, q = off
        if magnitude <= 1e-18 * max(
            1.0, max(abs(matrix[i][i]) for i in range(size))
        ):
            break
        difference = matrix[q][q] - matrix[p][p]
        if difference == 0.0:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * matrix[p][q], difference)
        cosine, sine = math.cos(theta), math.sin(theta)
        for k in range(size):
            kp, kq = matrix[k][p], matrix[k][q]
            matrix[k][p] = cosine * kp - sine * kq
            matrix[k][q] = sine * kp + cosine * kq
        for k in range(size):
            pk, qk = matrix[p][k], matrix[q][k]
            matrix[p][k] = cosine * pk - sine * qk
            matrix[q][k] = sine * pk + cosine * qk
    return sorted(matrix[i][i] for i in range(size))


def _condition(rows: Sequence[Sequence[float]]) -> float:
    """The two-norm condition number of a design matrix.

    Through the eigenvalues of ``A'A``, which is fine for *reporting* a
    conditioning that is already bad — the squaring costs accuracy in the
    diagnosis, not in the answer, and the answer comes from the reflections.
    """
    width = len(rows[0])
    gram = [
        [sum(row[i] * row[j] for row in rows) for j in range(width)]
        for i in range(width)
    ]
    values = _jacobi_eigenvalues(gram)
    smallest = values[0]
    largest = values[-1]
    if smallest <= 0.0:
        return math.inf
    return math.sqrt(largest / smallest)


# -- the fit -------------------------------------------------------------------


@dataclass(frozen=True)
class DecaySet:
    """How far each decay can move before the quotes stop being repriced.

    Measured one axis at a time, with the other decays held at the optimum, by
    stepping outwards until the objective exceeds the optimum by ``tolerance``
    and then bisecting. So it is the *coordinate* band and not a joint region —
    a joint region would be larger, and reporting the smaller of the two as the
    uncertainty is the direction an honest answer errs in.

    ``bounded`` says, per axis and per side, whether the band closed inside the
    search range. A band that ran into the range's edge means the decay is not
    identified by these quotes at all, which is a different statement from a
    wide band and is worth being able to tell apart.
    """

    tolerance: float
    lowest: tuple[float, ...]
    highest: tuple[float, ...]
    level_spread: tuple[float, ...]
    bounded: tuple[bool, ...]
    evaluations: int

    @property
    def widest(self) -> float:
        """The largest ratio of highest to lowest across the decay parameters."""
        return max(
            high / low for low, high in zip(self.lowest, self.highest, strict=True)
        )


@dataclass(frozen=True)
class CurveFit:
    """A fitted curve, what it missed by, and how determined its decays were."""

    curve: ParametricCurve
    #: Model less quote, in the units the quotes were given in.
    residuals: tuple[float, ...]
    #: Model less quote after weighting, which is what was actually minimised.
    weighted: tuple[float, ...]
    weighting: Weighting | None
    identified: DecaySet
    condition: float
    iterations: int
    labels: tuple[str, ...] = ()

    @property
    def rms(self) -> float:
        """Root mean square weighted residual."""
        return math.sqrt(
            sum(r * r for r in self.weighted) / len(self.weighted)
        )

    @property
    def worst(self) -> float:
        """The largest absolute weighted residual."""
        return max(abs(r) for r in self.weighted)

    @property
    def worst_quote(self) -> str:
        """Which quote the largest weighted residual belongs to."""
        index = max(range(len(self.weighted)), key=lambda i: abs(self.weighted[i]))
        if index < len(self.labels):
            return self.labels[index]
        return f"quote {index}"


def _grid(low: float, high: float, points: int) -> list[float]:
    """Logarithmically spaced decays; a hump at one year and at ten are not a
    linear distance apart in any sense that matters to the objective."""
    if not 0.0 < low < high:
        raise BadFit(f"the decay search needs 0 < low < high; got {low!r}, {high!r}")
    if points < 2:
        raise BadFit("the decay search needs at least two grid points")
    step = (math.log(high) - math.log(low)) / (points - 1)
    return [math.exp(math.log(low) + step * i) for i in range(points)]


def _candidates(family: Family, low: float, high: float, points: int) -> list[tuple[float, ...]]:
    axis = _grid(low, high, points)
    if family is Family.NELSON_SIEGEL:
        return [(value,) for value in axis]
    coarse = axis[:: max(1, points // 12)]
    return [(first, second) for first in coarse for second in coarse if second > first]


def _golden(
    objective: Callable[[float], float], low: float, high: float, iterations: int = 25
) -> float:
    """Golden-section minimisation in the log of the decay."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = math.log(low), math.log(high)
    a = right - ratio * (right - left)
    b = left + ratio * (right - left)
    fa, fb = objective(math.exp(a)), objective(math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = objective(math.exp(a))
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = objective(math.exp(b))
    return math.exp((left + right) / 2.0)


def _simplex(
    objective: Callable[[tuple[float, ...]], float],
    start: Sequence[float],
    low: float,
    high: float,
    rounds: int = 300,
) -> tuple[float, ...]:
    """Nelder-Mead in the log of the decays, which the sweeps cannot replace.

    Svensson's objective has a diagonal valley: lengthening one decay and
    shortening the other leaves the curve almost unchanged, so moving one axis
    at a time stops at the valley wall rather than running down it. On a
    Svensson curve fitted to its own exact zero rates the sweeps alone stop at a
    root mean square of 3.2e-05 and report decays of 1.73 and 5.99 against a
    truth of 1.6 and 9.0 — a plausible fit and the wrong parameters. The
    simplex, started from the same point, returns the truth to six figures and a
    residual at rounding. A reflection that moves both coordinates is the whole
    difference.
    """
    floor, ceiling = math.log(low), math.log(high)

    def clamp(point: Sequence[float]) -> tuple[float, ...]:
        return tuple(min(max(value, floor), ceiling) for value in point)

    def score(point: Sequence[float]) -> float:
        return objective(tuple(math.exp(value) for value in clamp(point)))

    size = len(start)
    base = clamp([math.log(value) for value in start])
    vertices = [list(base)]
    for axis in range(size):
        moved = list(base)
        moved[axis] = min(moved[axis] + 0.25, ceiling)
        if moved[axis] == base[axis]:
            moved[axis] = max(base[axis] - 0.25, floor)
        vertices.append(moved)
    scores = [score(vertex) for vertex in vertices]
    for _ in range(rounds):
        order = sorted(range(size + 1), key=lambda i: scores[i])
        vertices = [vertices[i] for i in order]
        scores = [scores[i] for i in order]
        spread = max(
            abs(vertices[i][axis] - vertices[0][axis])
            for i in range(1, size + 1)
            for axis in range(size)
        )
        if spread < 1e-11:
            break
        centroid = [
            sum(vertex[axis] for vertex in vertices[:-1]) / size for axis in range(size)
        ]
        worst = vertices[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst, strict=True)]
        value = score(reflected)
        if value < scores[0]:
            expanded = [c + 2.0 * (c - w) for c, w in zip(centroid, worst, strict=True)]
            expanded_value = score(expanded)
            if expanded_value < value:
                vertices[-1], scores[-1] = expanded, expanded_value
            else:
                vertices[-1], scores[-1] = reflected, value
        elif value < scores[-2]:
            vertices[-1], scores[-1] = reflected, value
        else:
            contracted = [
                c + 0.5 * (w - c) for c, w in zip(centroid, worst, strict=True)
            ]
            contracted_value = score(contracted)
            if contracted_value < scores[-1]:
                vertices[-1], scores[-1] = contracted, contracted_value
            else:
                for i in range(1, size + 1):
                    vertices[i] = [
                        vertices[0][axis] + 0.5 * (vertices[i][axis] - vertices[0][axis])
                        for axis in range(size)
                    ]
                    scores[i] = score(vertices[i])
    best = min(range(size + 1), key=lambda i: scores[i])
    return tuple(math.exp(value) for value in clamp(vertices[best]))


def _refine(
    objective: Callable[[tuple[float, ...]], float],
    start: tuple[float, ...],
    low: float,
    high: float,
) -> tuple[float, ...]:
    """Coordinate-wise golden-section sweeps, then a simplex if there is more
    than one decay to move."""
    current = list(start)
    for _ in range(2):
        for axis in range(len(current)):

            def one(value: float, axis: int = axis) -> float:
                trial = list(current)
                trial[axis] = value
                return objective(tuple(trial))

            current[axis] = _golden(one, low, high)
    if len(current) == 1:
        return tuple(current)
    swept = tuple(current)
    walked = _simplex(objective, swept, low, high)
    return walked if objective(walked) <= objective(swept) else swept


def _edge(
    probe: Callable[[float], float],
    start: float,
    ceiling: float,
    direction: float,
    low: float,
    high: float,
) -> tuple[float, bool]:
    """Step outward from ``start`` until ``probe`` exceeds ``ceiling``, then bisect.

    ``direction`` is +1 for the upper edge and -1 for the lower one. Returns the
    edge and whether it closed inside ``[low, high]``.
    """
    inside = start
    outside: float | None = None
    step = 1.2
    for _ in range(60):
        candidate = inside * (step if direction > 0 else 1.0 / step)
        if candidate >= high or candidate <= low:
            limit = high if direction > 0 else low
            if probe(limit) <= ceiling:
                return limit, False
            outside = limit
            break
        if probe(candidate) > ceiling:
            outside = candidate
            break
        inside = candidate
    if outside is None:
        return inside, False
    for _ in range(40):
        middle = math.sqrt(inside * outside)
        if probe(middle) <= ceiling:
            inside = middle
        else:
            outside = middle
    return inside, True


def _decay_set(
    objective: Callable[[tuple[float, ...]], float],
    levels_at: Callable[[tuple[float, ...]], tuple[float, ...]],
    best: tuple[float, ...],
    value: float,
    tolerance: float,
    low: float,
    high: float,
) -> DecaySet:
    ceiling = value + tolerance
    calls = 0
    lowest: list[float] = []
    highest: list[float] = []
    bounded: list[bool] = []
    corners: list[tuple[float, ...]] = [best]
    for axis in range(len(best)):

        def probe(point: float, axis: int = axis) -> float:
            nonlocal calls
            calls += 1
            trial = list(best)
            trial[axis] = point
            return objective(tuple(trial))

        upper, closed_above = _edge(probe, best[axis], ceiling, 1.0, low, high)
        lower, closed_below = _edge(probe, best[axis], ceiling, -1.0, low, high)
        lowest.append(lower)
        highest.append(upper)
        bounded.append(closed_above and closed_below)
        for edge in (lower, upper):
            corner = list(best)
            corner[axis] = edge
            corners.append(tuple(corner))
    spreads: list[tuple[float, ...]] = []
    for point in corners:
        try:
            spreads.append(levels_at(point))
        except BadFit:  # pragma: no cover - a singular corner is reported as zero spread
            continue
    width = len(spreads[0]) if spreads else 0
    return DecaySet(
        tolerance=tolerance,
        lowest=tuple(lowest),
        highest=tuple(highest),
        level_spread=tuple(
            max(entry[i] for entry in spreads) - min(entry[i] for entry in spreads)
            for i in range(width)
        ),
        bounded=tuple(bounded),
        evaluations=calls,
    )


def fit_zeros(
    reference: date,
    observations: Sequence[tuple[date, float]],
    *,
    family: Family = Family.NELSON_SIEGEL,
    basis: Basis = Basis.ACT_365F,
    weights: Sequence[float] | None = None,
    low: float = 0.1,
    high: float = 30.0,
    points: int = GRID_POINTS,
    tolerance: float = 1e-5,
) -> CurveFit:
    """Fit the form to observed continuously compounded zero rates.

    Linear in the level parameters, so for each candidate decay the levels come
    from one least-squares solve and nothing is iterated. The decays are taken
    from a logarithmic grid and then refined by golden section, and the grid is
    kept so the identified set can be read off it.

    ``tolerance`` is in the units of the weighted residual — here, a rate. The
    default of ``1e-5`` is a tenth of a basis point.
    """
    if len(observations) < family.levels:
        raise BadFit(
            f"{family.value} has {family.levels} level parameters and cannot be "
            f"fitted to {len(observations)} observations"
        )
    times = [year_fraction(reference, day, basis) for day, _ in observations]
    targets = [rate for _, rate in observations]
    if weights is None:
        weights = [1.0] * len(observations)
    if len(weights) != len(observations):
        raise BadFit(f"{len(weights)} weights against {len(observations)} observations")
    roots = [math.sqrt(w) for w in weights]

    def solve(decays: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
        rows = [
            [r * g for g in _zero_basis(t, decays, family)]
            for t, r in zip(times, roots, strict=True)
        ]
        scaled = [r * y for r, y in zip(roots, targets, strict=True)]
        levels = solve_least_squares(rows, scaled)
        residuals = [
            sum(b * g for b, g in zip(levels, _zero_basis(t, decays, family), strict=True)) - y
            for t, y in zip(times, targets, strict=True)
        ]
        weighted = [r * e for r, e in zip(roots, residuals, strict=True)]
        rms = math.sqrt(sum(e * e for e in weighted) / len(weighted))
        return levels, rms

    scored: list[tuple[tuple[float, ...], float, tuple[float, ...]]] = []
    for candidate in _candidates(family, low, high, points):
        try:
            levels, rms = solve(candidate)
        except BadFit:
            continue
        scored.append((candidate, rms, levels))
    if not scored:
        raise BadFit("no candidate decay produced a solvable linear system")
    best_grid = min(scored, key=lambda entry: entry[1])

    def objective(decays: tuple[float, ...]) -> float:
        try:
            return solve(decays)[1]
        except BadFit:
            return math.inf

    decays = _refine(objective, best_grid[0], low, high)
    if objective(decays) > best_grid[1]:
        decays = best_grid[0]
    levels, _ = solve(decays)
    curve = ParametricCurve(reference, family, levels, decays, basis)
    residuals = tuple(curve.zero(t) - y for t, y in zip(times, targets, strict=True))
    weighted = tuple(r * e for r, e in zip(roots, residuals, strict=True))
    rows = [
        [r * g for g in _zero_basis(t, decays, family)]
        for t, r in zip(times, roots, strict=True)
    ]
    return CurveFit(
        curve=curve,
        residuals=residuals,
        weighted=weighted,
        weighting=None,
        identified=_decay_set(
            objective,
            lambda point: solve(point)[0],
            decays,
            objective(decays),
            tolerance,
            low,
            high,
        ),
        condition=_condition(rows),
        iterations=1,
        labels=tuple(day.isoformat() for day, _ in observations),
    )


def fit_bonds(
    reference: date,
    quotes: Sequence[tuple[Bond, float]],
    settlement: date,
    *,
    family: Family = Family.NELSON_SIEGEL,
    basis: Basis = Basis.ACT_365F,
    weighting: Weighting = Weighting.YIELD,
    low: float = 0.1,
    high: float = 30.0,
    points: int = GRID_POINTS,
    tolerance: float = 1e-5,
    iterations: int = 60,
) -> CurveFit:
    """Fit the form to observed *clean* bond prices.

    Clean, because that is how bonds are quoted; the accrued interest is added
    here so the residual is on the dirty price the curve actually produces.

    The price is an exponential of the parameters, so this is Gauss-Newton with
    the analytic Jacobian rather than a single solve. It is started from a
    linear fit to the bonds' own yields placed at their maturities — not a
    curve anyone would use, and close enough that the iteration converges in a
    handful of steps.

    ``tolerance`` is in the units of the weighted residual, which under
    ``Weighting.YIELD`` is a rate and under ``Weighting.PRICE`` is a price per
    100.
    """
    if len(quotes) < family.levels:
        raise BadFit(
            f"{family.value} has {family.levels} level parameters and cannot be "
            f"fitted to {len(quotes)} quotes"
        )
    flows: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    dirty: list[float] = []
    roots: list[float] = []
    pseudo: list[tuple[float, float]] = []
    settlement_time = year_fraction(reference, settlement, basis)
    for bond, clean in quotes:
        cashflows = bond.cashflows(settlement)
        if not cashflows:
            raise BadFit(f"{bond.name} has no cashflows after {settlement.isoformat()}")
        amounts = tuple(flow.amount for flow in cashflows)
        moments = tuple(
            year_fraction(reference, flow.day, basis) for flow in cashflows
        )
        target = clean + bond.accrued(settlement)
        flows.append((amounts, moments))
        dirty.append(target)
        if weighting is Weighting.YIELD:
            own = bond.yield_from_dirty(target, settlement)
            sensitivity = bond.modified_duration(own.value, settlement) * target
            if not sensitivity > 0.0:
                raise BadFit(
                    f"{bond.name} has no price sensitivity to its own yield, so its "
                    "price error cannot be read as a yield error"
                )
            roots.append(1.0 / sensitivity)
            pseudo.append((year_fraction(reference, bond.maturity, basis), own.value))
        else:
            roots.append(1.0)
            own = bond.yield_from_dirty(target, settlement)
            pseudo.append((year_fraction(reference, bond.maturity, basis), own.value))

    def model(levels: tuple[float, ...], decays: tuple[float, ...]) -> list[float]:
        base = sum(
            b * g
            for b, g in zip(
                levels, _integral_basis(settlement_time, decays, family), strict=True
            )
        )
        prices = []
        for amounts, moments in flows:
            total = 0.0
            for amount, moment in zip(amounts, moments, strict=True):
                integral = sum(
                    b * g
                    for b, g in zip(
                        levels, _integral_basis(moment, decays, family), strict=True
                    )
                )
                total += amount * math.exp(-(integral - base))
            prices.append(total)
        return prices

    def jacobian(
        levels: tuple[float, ...], decays: tuple[float, ...]
    ) -> list[list[float]]:
        base_basis = _integral_basis(settlement_time, decays, family)
        base = sum(b * g for b, g in zip(levels, base_basis, strict=True))
        rows = []
        for amounts, moments in flows:
            gradient = [0.0] * family.levels
            for amount, moment in zip(amounts, moments, strict=True):
                terms = _integral_basis(moment, decays, family)
                integral = sum(b * g for b, g in zip(levels, terms, strict=True))
                discounted = amount * math.exp(-(integral - base))
                for k in range(family.levels):
                    gradient[k] -= discounted * (terms[k] - base_basis[k])
            rows.append(gradient)
        return rows

    def start(decays: tuple[float, ...]) -> tuple[float, ...]:
        rows = [list(_zero_basis(t, decays, family)) for t, _ in pseudo]
        return solve_least_squares(rows, [y for _, y in pseudo])

    def run(decays: tuple[float, ...]) -> tuple[tuple[float, ...], float, int]:
        levels = start(decays)
        try:
            prices = model(levels, decays)
        except (OverflowError, ValueError) as bad:
            # The warm start is a linear fit to the bonds' own yields, which on a
            # near-singular decay pair can return level parameters of thousands.
            # Discounting on those overflows before the iteration has taken a
            # single step, so the candidate is infeasible rather than merely bad.
            raise BadFit(
                f"discounting overflows at decays {decays!r}, where the linear "
                f"start returns {levels!r}"
            ) from bad
        weighted = [
            r * (p - d) for r, p, d in zip(roots, prices, dirty, strict=True)
        ]
        loss = sum(e * e for e in weighted)
        previous = loss
        taken = 0
        for _ in range(iterations):
            taken += 1
            try:
                rows = [
                    [r * g for g in gradient]
                    for r, gradient in zip(roots, jacobian(levels, decays), strict=True)
                ]
                step = solve_least_squares(rows, [-e for e in weighted])
            except (OverflowError, ValueError, BadFit):
                # The same story as the warm start, one step later: on a
                # near-singular decay pair the Jacobian's entries pass the
                # largest double. Stopping here scores the candidate on where it
                # got to, which is bad enough to be skipped, rather than failing
                # the whole fit over one grid point.
                break
            damping = 1.0
            improved = False
            for _ in range(30):
                trial = tuple(b + damping * s for b, s in zip(levels, step, strict=True))
                try:
                    trial_prices = model(trial, decays)
                except (OverflowError, ValueError):
                    damping *= 0.5
                    continue
                trial_weighted = [
                    r * (p - d)
                    for r, p, d in zip(roots, trial_prices, dirty, strict=True)
                ]
                trial_loss = sum(e * e for e in trial_weighted)
                if trial_loss < loss:
                    levels, weighted, improved = trial, trial_weighted, True
                    previous, loss = loss, trial_loss
                    break
                damping *= 0.5
            if not improved:
                break
            if previous - loss <= 1e-18 * max(previous, 1e-30):
                break
        return levels, math.sqrt(loss / len(weighted)), taken

    scored: list[tuple[tuple[float, ...], float, tuple[float, ...]]] = []
    for candidate in _candidates(family, low, high, points):
        try:
            levels, rms, _ = run(candidate)
        except BadFit:
            continue
        scored.append((candidate, rms, levels))
    if not scored:
        raise BadFit("no candidate decay produced a solvable linear system")
    best_grid = min(scored, key=lambda entry: entry[1])

    def objective(decays: tuple[float, ...]) -> float:
        try:
            return run(decays)[1]
        except BadFit:
            return math.inf

    decays = _refine(objective, best_grid[0], low, high)
    if objective(decays) > best_grid[1]:
        decays = best_grid[0]
    levels, rms, taken = run(decays)
    curve = ParametricCurve(reference, family, levels, decays, basis)
    prices = model(levels, decays)
    residuals = tuple(
        p - d for p, d in zip(prices, dirty, strict=True)
    )
    weighted = tuple(r * e for r, e in zip(roots, residuals, strict=True))
    rows = [
        [r * g for g in gradient]
        for r, gradient in zip(roots, jacobian(levels, decays), strict=True)
    ]
    return CurveFit(
        curve=curve,
        residuals=residuals,
        weighted=weighted,
        weighting=weighting,
        identified=_decay_set(
            objective,
            lambda point: run(point)[0],
            decays,
            rms,
            tolerance,
            low,
            high,
        ),
        condition=_condition(rows),
        iterations=taken,
        labels=tuple(bond.name for bond, _ in quotes),
    )
