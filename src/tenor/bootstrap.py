"""Building a curve out of quotes.

The input is a strip of instruments — deposits at the front, futures in the
middle, par swaps at the back — and the output is a curve that reprices every
one of them. The method is the obvious one: each instrument pins down the
discount factor at its own maturity, given the discount factors at every
shorter maturity, so they can be solved one at a time in order.

**That obvious method is wrong for a smooth interpolation, and the way it is
wrong is invisible.** Solving instrument ``i`` fixes the curve at its maturity,
but under an interpolation whose shape at one maturity depends on the
neighbouring pillars — monotone convex, any spline — adding pillar ``i + 1``
changes the curve *before* ``t_i`` as well. The earlier instruments stop
repricing. Nothing raises, no discount factor looks unreasonable, and the curve
is out by a basis point or two in the middle of each interval, which is the size
of the thing anybody is trying to measure.

So this bootstrapper sweeps. The first pass is the ordinary sequential one, and
then it re-solves every pillar in turn, repeatedly, until every instrument
reprices to tolerance at the same time. For a local interpolation — log-linear
on discount factors, linear on zero rates — the first sweep is already
consistent and it stops there. For monotone convex over a deposit, futures and
swap strip it takes seven. The sweep count is reported either way, because
"settled on the first pass" and "settled on the seventh" describe curves built
by different amounts of work, and a set of quotes that starts needing many more
sweeps than it used to is saying something.

**The unknown is the discount factor itself**, not a rate parametrising it. That
choice buys an exact bracket: the arbitrage-free range for pillar ``i`` is
``[P(i+1), P(i-1)]``, since a discount factor outside its neighbours means a
negative forward rate over one side or the other. Bracketing in rate space would
mean guessing a range and widening it, and widening it in the wrong direction
builds a trial curve that the interpolation then refuses, which turns a clean
bracket into an exception to catch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from .curve import BadCurve, DiscountCurve, Interpolation, OffCurve
from .daycount import Basis
from .instruments import Instrument, maturity_of, start_of
from .solve import NoRoot, Root, brent


class BootstrapFailed(BadCurve):
    """A curve that could not be built, and the instrument responsible.

    Carries the instrument rather than only naming it in the message, because
    the caller's next move is almost always to drop that quote and rebuild, and
    parsing it back out of a string is not a reasonable thing to ask.
    """

    def __init__(self, message: str, instrument: Instrument | None = None) -> None:
        super().__init__(message)
        self.instrument = instrument


@dataclass(frozen=True)
class Bootstrapped:
    """A curve, the quotes it was built from, and the evidence it fits them."""

    curve: DiscountCurve
    instruments: tuple[Instrument, ...]
    #: Passes over the full instrument set. One means the sequential pass was
    #: already consistent; more means the interpolation moved earlier pillars
    #: as later ones were added, and they had to be re-solved.
    sweeps: int
    #: The final solve for each pillar, in instrument order.
    solutions: tuple[Root, ...]

    def par_errors(self) -> tuple[float, ...]:
        """Each instrument's remaining pricing error, in present value."""
        return tuple(one.par_error(self.curve) for one in self.instruments)

    @property
    def worst_error(self) -> float:
        return max(abs(error) for error in self.par_errors())

    def reprices(self, tolerance: float = 1e-12) -> bool:
        """Whether every input instrument still prices to par off the result.

        The identity the whole exercise exists to satisfy, checked against the
        finished curve rather than against the state each instrument was solved
        in. Those are different claims, and only this one is worth anything.
        """
        return self.worst_error <= tolerance

    def __len__(self) -> int:
        return len(self.instruments)


def bootstrap(
    reference: date,
    instruments: Sequence[Instrument],
    *,
    basis: Basis,
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    tolerance: float = 1e-14,
    max_sweeps: int = 50,
) -> Bootstrapped:
    """Build a discount curve that reprices every instrument given.

    ``tolerance`` is on each instrument's present value, where one unit of
    notional is 1.0 — so 1e-14 is a hundredth of a cent on a million, which is
    as close to exact as double precision allows across a swap's worth of
    additions.
    """
    if max_sweeps < 1:
        raise BootstrapFailed(
            f"a curve takes at least one sweep to build, got {max_sweeps!r}"
        )
    ordered = _ordered(reference, instruments)
    days = [maturity_of(one) for one in ordered]
    factors = _seed(reference, ordered, basis)
    # Placeholders. Every entry is replaced by a real solve before the first
    # sweep ends, and nothing reads them in between, but a NaN residual is a
    # more honest thing to leave lying around than a zero one.
    solutions: list[Root] = [
        Root(factor, math.nan, 0, (factor, factor), converged=False)
        for factor in factors
    ]

    for sweep in range(1, max_sweeps + 1):
        for index, instrument in enumerate(ordered):
            # On the first sweep the curve stops at the instrument being
            # solved: nothing beyond it is known yet. Afterwards every pillar
            # is present, which is what lets an earlier one be re-solved
            # against the shape a later one imposed on it.
            extent = index + 1 if sweep == 1 else len(ordered)
            solutions[index] = _solve_pillar(
                reference=reference,
                days=days[:extent],
                factors=factors[:extent],
                index=index,
                instrument=instrument,
                basis=basis,
                interpolation=interpolation,
                tolerance=tolerance,
            )
            factors[index] = solutions[index].value

        curve = _build(reference, days, factors, basis, interpolation)
        errors = [abs(one.par_error(curve)) for one in ordered]
        if max(errors) <= tolerance:
            return Bootstrapped(
                curve=curve,
                instruments=tuple(ordered),
                sweeps=sweep,
                solutions=tuple(solutions),
            )

    worst = max(range(len(ordered)), key=lambda index: errors[index])
    raise BootstrapFailed(
        f"the curve did not settle after {max_sweeps} sweeps. The worst "
        f"remaining error is {errors[worst]:.3e} on {ordered[worst].name}, "
        f"against a tolerance of {tolerance:.3e}. Under "
        f"'{interpolation.value}' the pillars are not independent, so each "
        "sweep moves the ones already solved; a set of quotes that keeps "
        "moving them is usually one containing two instruments that disagree.",
        ordered[worst],
    )


def _ordered(reference: date, instruments: Sequence[Instrument]) -> list[Instrument]:
    """Sorted by maturity, with everything that is not a curve rejected."""
    if not instruments:
        raise BootstrapFailed("a curve needs at least one instrument to be built from")
    ordered = sorted(instruments, key=maturity_of)
    for index, instrument in enumerate(ordered):
        if maturity_of(instrument) <= reference:
            raise BootstrapFailed(
                f"{instrument.name} matures on "
                f"{maturity_of(instrument).isoformat()}, on or before the curve's "
                f"reference date {reference.isoformat()}, so it says nothing about "
                "the future",
                instrument,
            )
        if start_of(instrument) < reference:
            raise BootstrapFailed(
                f"{instrument.name} starts on {start_of(instrument).isoformat()}, "
                f"before the curve's reference date {reference.isoformat()}. Its "
                "value depends on discount factors the curve does not have.",
                instrument,
            )
        if index and maturity_of(ordered[index - 1]) == maturity_of(instrument):
            raise BootstrapFailed(
                f"{instrument.name} and {ordered[index - 1].name} both mature on "
                f"{maturity_of(instrument).isoformat()}. Two quotes for one "
                "discount factor either agree, in which case one of them is "
                "redundant, or disagree, in which case no curve prices both.",
                instrument,
            )
    return ordered


def _seed(reference: date, ordered: Sequence[Instrument], basis: Basis) -> list[float]:
    """A starting curve, flat at 2%, only ever used as somewhere to begin.

    The answer does not depend on it: every pillar is solved to a bracketed
    root, and the bracket is the arbitrage-free range rather than a
    neighbourhood of the seed. It exists so the first sweep has discount
    factors to interpolate between at maturities it has not reached yet, and so
    later sweeps start from a curve rather than from nothing.
    """
    from .daycount import year_fraction

    return [
        math.exp(-0.02 * year_fraction(reference, maturity_of(one), basis))
        for one in ordered
    ]


def _build(
    reference: date,
    days: Sequence[date],
    factors: Sequence[float],
    basis: Basis,
    interpolation: Interpolation,
) -> DiscountCurve:
    return DiscountCurve.from_discounts(
        reference,
        list(zip(days, factors, strict=True)),
        basis=basis,
        interpolation=interpolation,
    )


def _solve_pillar(
    *,
    reference: date,
    days: Sequence[date],
    factors: Sequence[float],
    index: int,
    instrument: Instrument,
    basis: Basis,
    interpolation: Interpolation,
    tolerance: float,
) -> Root:
    """Solve for the discount factor at ``days[index]``, holding the rest."""
    trial = list(factors)

    def objective(factor: float) -> float:
        trial[index] = factor
        return instrument.par_error(
            _build(reference, days, trial, basis, interpolation)
        )

    low, high = _bounds(factors, index, interpolation)
    try:
        low_value, high_value = _endpoints(objective, low, high)
    except (BadCurve, OffCurve) as bad:
        raise BootstrapFailed(
            f"{instrument.name} could not be priced while solving for its own "
            f"discount factor: {bad}",
            instrument,
        ) from bad

    if low_value * high_value > 0.0:
        raise BootstrapFailed(
            f"no discount factor between {low:.12g} and {high:.12g} prices "
            f"{instrument.name} to par: the pricing error is {low_value:.6g} at "
            f"one end and {high_value:.6g} at the other, with no crossing "
            "between. That range is the arbitrage-free one — a discount factor "
            "outside it implies a negative forward rate against a neighbouring "
            "pillar — so the quote is inconsistent with the ones around it "
            "rather than merely awkward to solve.",
            instrument,
        )
    try:
        return brent(objective, low, high, tolerance=min(tolerance, 1e-15))
    except NoRoot as bad:  # pragma: no cover - the bracket is checked above
        raise BootstrapFailed(f"{instrument.name}: {bad}", instrument) from bad


def _endpoints(
    objective: Callable[[float], float], low: float, high: float
) -> tuple[float, float]:
    return objective(low), objective(high)


def _bounds(
    factors: Sequence[float], index: int, interpolation: Interpolation
) -> tuple[float, float]:
    """The range this pillar's discount factor may take.

    Bounded above by the previous pillar's factor and below by the next one's:
    a discount factor that is not between its neighbours implies a negative
    forward rate on one side. Under monotone convex those bounds are hard,
    since the scheme refuses such a curve outright. Under the local schemes
    they are widened, because a negative forward is a thing those curves will
    happily represent and occasionally a thing the quotes genuinely imply.
    """
    ceiling = factors[index - 1] if index else 1.0
    floor = factors[index + 1] if index + 1 < len(factors) else _FLOOR
    if interpolation is not Interpolation.MONOTONE_CONVEX:
        ceiling = ceiling * _SLACK
        floor = max(floor / _SLACK, _FLOOR)
    if floor >= ceiling:
        floor = min(floor, ceiling * 0.5)
    return floor, ceiling


#: A discount factor low enough to cover any quote and high enough to stay well
#: away from the point where its logarithm stops being representable.
_FLOOR = 1e-12

#: How far outside the arbitrage-free range the local interpolations may solve.
#: A factor of 1.65 is about fifty years of 1% negative rates, which is past
#: anything quoted and short of anything numerically awkward.
_SLACK = 1.65


__all__ = ["BootstrapFailed", "Bootstrapped", "bootstrap"]
