"""Bermudan swaptions: several exercise dates, and no formula for any of them.

:func:`~tenor.hullwhite.swaption_price` prices a European swaption twice over,
by Jamshidian's decomposition and by a direct integral, and both routes rest on
the same structural fact: there is one exercise date, so the payoff is a
function of the short rate at that date and nothing else. Integrate it against
the rate's density and the job is done.

A Bermudan swaption breaks that. The holder may enter the same swap on any of
several dates, and the value of *not* exercising on the first one is the value
of still holding the right on the second, which depends on where the rate goes
between them. The quantity being integrated is itself the answer to the same
problem one date later, so there is nothing to write down and the recursion has
to be run.

**Why not :mod:`tenor.lattice`.** That module also builds a short rate tree and
also handles early exercise, and it is the wrong tool here for a reason that is
not about convenience. Its tree is calibrated to a curve and a volatility of
its own, expresses the exercise right as a schedule of *call prices* held by the
issuer, and has no relationship to the ``(a, sigma)`` a swaption desk has fitted
to its own market. Pricing a Bermudan swaption there would mean holding two
unrelated models of the same short rate and hoping they agree. The tree here is
Hull-White's: the same two parameters, the same ``B(t, T)``, the same
:class:`~tenor.hullwhite.HullWhite` object that priced the Europeans.

**The construction.** The state is ``x = r - phi(t)``, which mean-reverts to
zero at speed ``a`` with volatility ``sigma`` and carries all of the model's
randomness. Over a step of length ``dt`` its conditional mean is ``x e^{-a dt}``
and its conditional variance is ``sigma^2 (1 - e^{-2 a dt}) / (2 a)``; both exact
moments are used rather than their Euler approximations, which cost accuracy for
nothing. Nodes sit at integer multiples of a spacing ``dx``, each node branches
to the three nodes around its own conditional mean, and the branching
probabilities are chosen to match that mean and variance exactly::

    u  = j q - k                      q = e^{-a dt},  k = round(j q)
    pu = (v + u^2 + u) / 2            v = Var / dx^2
    pm = 1 - v - u^2
    pd = (v + u^2 - u) / 2

Rounding to the *nearest* node is what makes this work. It bounds ``|u|`` by one
half, and with ``v`` near one third every probability is then strictly positive
at every node with no truncation anywhere — the usual Hull-White cap on the tree
width exists to repair a construction that branches up-mid-down from a fixed
offset, and is not needed here. Nothing is discarded, so the tree is exact in
the sense that matters for a discounted expectation: total state-price mass at
each step equals the discount factor to that step, to floating point.

**Fitting the curve.** The shift ``theta_i`` on each step is found by forward
induction: carry Arrow-Debreu prices along the tree and choose the shift so that
the next zero on the grid is repriced. That is one division and one logarithm per
step, and the result is a tree that reprices every grid zero exactly rather than
to the order of the discretisation.

**Where the discontinuity goes.** Every exercise date lands on a grid node, and
this is not a detail. An exercise decision is a kink in the value function; a
step that straddles one smears it across the step and loses an order. Equal time
steps cannot do it — exercise dates are swap anniversaries whose day counts
differ by a day or three — so the grid is built interval by interval, each gap
between consecutive exercise dates split into whole sub-steps, with a single
``dx`` shared by the whole tree. The step variances then differ slightly, and
positivity needs ``dx^2 / 4 <= Var <= 3 dx^2 / 4``, which the uniform case
satisfies with room to spare at ``Var = dx^2 / 3``. The bound is checked at build
time against the grid actually produced rather than assumed from the request.

**What is approximated, and what is not.** The exercise value at a node is the
analytic forward swap value under the same model, built from
:meth:`~tenor.hullwhite.HullWhite.bond_price` — no tree quantity enters it. The
only discretisation is in the continuation value, and the convergence that
remains is measured rather than claimed: see the module's tests and the
repository roadmap for the numbers.

**The switch value** is the whole point of the instrument. A Bermudan is worth at
least the best of its co-terminal Europeans, because exercising only on that one
date is one of the strategies available to it, and the excess is what the other
dates add. Reporting the price alone hides whether a desk is paying for
optionality or for a European with extra steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .hullwhite import CouponBondOption, Flow, HullWhite, Method, coupon_bond_option
from .options import Payoff, Swaption
from .schedule import Frequency, Schedule, generate

__all__ = [
    "BadBermudan",
    "BermudanSwaption",
    "BermudanValue",
    "ExerciseStep",
    "TrinomialTree",
    "bermudan_swaption",
    "build_tree",
    "coterminal_bonds",
    "coterminal_europeans",
    "short_rate_level",
]

#: Lower and upper bounds on ``Var / dx**2`` that keep every branching
#: probability non-negative for any residual drift within half a spacing.
_VARIANCE_FLOOR = 0.25
_VARIANCE_CEILING = 0.75


class BadBermudan(ValueError):
    """An instrument, grid or exercise schedule that cannot be priced as stated."""


@dataclass(frozen=True, slots=True)
class TrinomialTree:
    """A Hull-White tree on ``x = r - phi(t)``, fitted to a curve by induction.

    Attributes:
        times: Node times in years from the valuation date, ascending, starting
            at zero. There are ``len(times) - 1`` steps.
        spacing: The state spacing ``dx``, shared by every step.
        shift: ``theta_i`` for the step beginning at ``times[i]``, so the short
            rate on that step at node ``j`` is ``shift[i] + j * spacing``.
        lower: Lowest node index reached at each time.
        upper: Highest node index reached at each time.
        state_prices: Arrow-Debreu price of each node, indexed by time then by
            ``j - lower[i]``. Their sum at each time is the discount factor to
            it.
        variance_ratio: ``Var / dx**2`` on each step, which the construction
            holds between :data:`_VARIANCE_FLOOR` and :data:`_VARIANCE_CEILING`.
    """

    times: tuple[float, ...]
    spacing: float
    shift: tuple[float, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    state_prices: tuple[tuple[float, ...], ...]
    variance_ratio: tuple[float, ...]

    @property
    def steps(self) -> int:
        """Number of steps, one fewer than the number of node times."""
        return len(self.times) - 1

    @property
    def nodes(self) -> int:
        """Total node count, which is what the cost of a backward pass scales with."""
        return sum(hi - lo + 1 for lo, hi in zip(self.lower, self.upper, strict=True))

    def state(self, index: int) -> float:
        """The state variable ``x = j dx`` at node ``index``.

        The state grid is built from ``a`` and ``sigma`` alone and owes nothing to
        the curve or to the shifts, which is why
        :func:`short_rate_level` may be used with it.
        """
        return index * self.spacing

    def short_rate(self, step: int, index: int) -> float:
        """The short rate used to discount the step beginning at ``times[step]``.

        This is the *step* rate, ``theta_i + j dx``, and forward induction fits
        ``theta_i`` so that the step's discounting reproduces the curve. It is
        therefore an average of the instantaneous rate across the step rather
        than its value at the left end, and it is not the right rate at which to
        evaluate a payoff: see :func:`short_rate_level`.
        """
        if not 0 <= step < len(self.shift):
            raise BadBermudan(
                f"step {step!r} is outside the tree's {len(self.shift)} steps"
            )
        return self.shift[step] + index * self.spacing

    def curve_error(self, curve: DiscountCurve) -> float:
        """Largest relative error in repricing the curve's own zeros on the grid.

        Forward induction sets each shift so that this is zero up to floating
        point. A number materially above machine precision means the grid and
        the curve have come apart, not that the tree is coarse.
        """
        worst = 0.0
        for time, prices in zip(self.times, self.state_prices, strict=True):
            target = curve.discount_at(time)
            worst = max(worst, abs(math.fsum(prices) / target - 1.0))
        return worst


@dataclass(frozen=True, slots=True)
class ExerciseStep:
    """One exercise date, and what the decision looked like there.

    Attributes:
        expiry: The business day the right may be taken, after any roll.
        time: Its year fraction from the valuation date.
        step: Index into the tree's ``times``.
        european: The co-terminal European expiring here, priced analytically.
        boundary: Short rate, as phi(t) + x, at which exercising stops being worse than waiting,
            linearly interpolated between the two nodes that bracket it, or
            ``None`` where no node in the tree exercises.
        exercised: Fraction of this date's state-price mass that exercises,
            counted with the paths that already exercised earlier, so these
            shares describe where the decision bites rather than forming a
            distribution over exercise dates.
    """

    expiry: date
    time: float
    step: int
    european: float
    boundary: float | None
    exercised: float


@dataclass(frozen=True, slots=True)
class BermudanValue:
    """A priced Bermudan, with the Europeans it has to beat.

    Attributes:
        price: Value per unit notional.
        steps: One entry per exercise date, in date order.
        tree: The tree it was priced on.
        crossings: Number of exercise dates whose exercise-versus-wait decision
            changed sign more than once across the nodes. Zero everywhere a
            single boundary describes the decision, which is the normal case
            and is reported rather than assumed.
    """

    price: float
    steps: tuple[ExerciseStep, ...]
    tree: TrinomialTree
    crossings: int

    @property
    def europeans(self) -> tuple[float, ...]:
        """The co-terminal European prices, in date order."""
        return tuple(step.european for step in self.steps)

    @property
    def best_european(self) -> float:
        """The most valuable single-date right inside this instrument."""
        return max(self.europeans)

    @property
    def switch_value(self) -> float:
        """What the exercise dates beyond the best single one are worth.

        Non-negative by construction of the instrument: exercising only on the
        best date is one of the strategies a Bermudan holder may follow. A
        negative number here is a discretisation error larger than the
        optionality, which is a reason to refine the grid rather than a price.
        """
        return self.price - self.best_european


@dataclass(frozen=True, slots=True)
class BermudanSwaption:
    """The right to enter one swap on any of several dates.

    The swap is *co-terminal*: whichever date is used, it runs from that date to
    the same ``maturity``, so a later exercise buys a shorter swap. That is the
    traded instrument — a Bermudan swaption written against a callable bond or a
    cancellable swap always ends when the underlying does.

    Every exercise date must fall on a fixed-leg period boundary, so that the
    swap entered there starts on an accrual boundary and its floating leg is
    worth par. A date between boundaries describes a different instrument, part
    mid-curve, and is refused with the boundaries that are available rather than
    silently rolled.

    Attributes:
        expiries: Exercise dates, strictly ascending, all before ``maturity``.
        maturity: The common end of the underlying swap.
        strike: Fixed rate, as a decimal.
        payoff: ``PAYER`` to pay fixed on exercise, ``RECEIVER`` to receive it.
        frequency: Fixed-leg frequency, which also sets where exercise dates may
            fall.
        basis: Fixed-leg accrual basis.
        calendar: Holiday calendar; weekends only by default.
        rolling: Business-day rule for the fixed leg.
        label: Optional name used in messages.
    """

    expiries: tuple[date, ...]
    maturity: date
    strike: float
    payoff: Payoff = Payoff.PAYER
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if not self.expiries:
            raise BadBermudan(
                "a Bermudan swaption needs at least one exercise date; with none "
                "there is no instrument, and with one there is a European, which "
                "tenor.hullwhite.swaption_price prices in closed form"
            )
        for earlier, later in pairwise(self.expiries):
            if later <= earlier:
                raise BadBermudan(
                    f"exercise dates must be strictly ascending, got "
                    f"{earlier.isoformat()} followed by {later.isoformat()}"
                )
        if self.expiries[-1] >= self.maturity:
            raise BadBermudan(
                f"the last exercise date {self.expiries[-1].isoformat()} is not "
                f"before the swap's maturity {self.maturity.isoformat()}; "
                "exercising into a swap with no life left is not an option"
            )
        if not math.isfinite(self.strike):
            raise BadBermudan(f"the strike must be finite, got {self.strike!r}")

    @property
    def name(self) -> str:
        """A readable name, the label where one was given."""
        return self.label or (
            f"{self.payoff.value} to {self.maturity.isoformat()} on "
            f"{len(self.expiries)} dates from {self.expiries[0].isoformat()}"
        )

    def schedule(self) -> Schedule:
        """The fixed leg from the first exercise date to maturity.

        Generated backwards from maturity, so the boundaries are the swap's
        anniversaries and any odd period sits at the front, where the first
        exercise date is.
        """
        return generate(
            self.expiries[0],
            self.maturity,
            self.frequency,
            calendar=self.calendar if self.calendar is not None else WEEKENDS_ONLY,
            rolling=self.rolling,
        )

    def resolve(self, schedule: Schedule) -> tuple[tuple[int, date], ...]:
        """Match each exercise date to a fixed-leg period, in date order.

        An exercise date may be given either as the period's unadjusted
        anniversary — which is how a term sheet reads, "annually on 15 January" —
        or as the business day it rolls to. Either way the date returned is the
        adjusted one, because that is when the swap starts accruing and so when
        its floating leg is worth par. The anniversary falling on a weekend is
        the common case rather than the exception, and refusing it would make the
        caller reproduce the calendar.

        Returns:
            ``(period index, adjusted start)`` for each exercise date.

        Raises:
            BadBermudan: If an exercise date is neither form of a boundary, or if
                two of them resolve to the same period.
        """
        starts: dict[date, int] = {}
        for position, period in enumerate(schedule):
            starts.setdefault(period.start, position)
            starts[period.adjusted_start] = position
        found: list[tuple[int, date]] = []
        seen: set[int] = set()
        for expiry in self.expiries:
            matched = starts.get(expiry)
            if matched is None:
                available = ", ".join(
                    period.adjusted_start.isoformat() for period in schedule
                )
                raise BadBermudan(
                    f"exercise date {expiry.isoformat()} is not a fixed-leg period "
                    f"boundary of {self.name}. Exercising there would enter a swap "
                    f"part way through an accrual, whose floating leg is not worth "
                    f"par at the start. Available boundaries: {available}"
                )
            if matched in seen:
                raise BadBermudan(
                    f"exercise date {expiry.isoformat()} resolves to the same fixed-leg "
                    f"period as an earlier one; the adjusted and unadjusted forms of "
                    "one boundary are the same date, not two exercise rights"
                )
            seen.add(matched)
            found.append((matched, schedule[matched].adjusted_start))
        return tuple(found)

    def reference_swaption(self, expiry: date) -> Swaption:
        """The European :class:`~tenor.options.Swaption` for one exercise date.

        Provided for callers who want the instrument rather than its price —
        to quote it, to risk it, or to price it with
        :func:`~tenor.hullwhite.swaption_price`. It regenerates its own schedule
        from ``expiry``, so it describes exactly the same swap as this
        instrument's own tail only where ``expiry`` needed no business-day roll;
        where the roll moved the boundary, the two differ by the days between.
        The Europeans this module prices and compares against are built from the
        tail flows directly, which cannot drift apart that way.
        """
        return Swaption(
            expiry=expiry,
            effective=expiry,
            maturity=self.maturity,
            strike=self.strike,
            payoff=self.payoff,
            frequency=self.frequency,
            basis=self.basis,
            calendar=self.calendar,
            rolling=self.rolling,
            label=f"{self.name} exercised {expiry.isoformat()}",
        )


def short_rate_level(curve: DiscountCurve, model: HullWhite, time: float) -> float:
    """The deterministic part of the short rate, ``phi(t)``, so that ``r = phi + x``.

    Hull-White's drift fit in closed form::

        phi(t) = f(0, t) + sigma**2 (1 - e^{-a t})**2 / (2 a**2)

    the instantaneous forward plus the convexity the volatility adds. Every
    analytic price in :mod:`tenor.hullwhite` is built on this, which is the
    reason to use it here rather than the tree's own step shift.

    **Why it matters.** The shift ``theta_i`` that forward induction produces is
    an average of ``phi`` across its step, not ``phi`` at the step's left end,
    and the two differ by about half a step times the curve's slope. On a flat
    curve that is nothing — measured at 1.4e-09 — and a payoff evaluated at
    either rate gives the same answer. On a curve rising 15 basis points a year
    it is not: evaluating the exercise value at the step rate instead of at
    ``phi`` left a zero-volatility Bermudan 5.3e-03 above its own best
    discounted intrinsic value, an error 17% of the price and in the direction
    that flatters the instrument. Discounting is the step's job and uses
    ``theta``; a payoff is a statement about one instant and uses ``phi``.
    """
    convexity = -math.expm1(-model.a * time)
    return curve.instantaneous_forward(time) + model.sigma**2 * convexity**2 / (
        2.0 * model.a**2
    )


def build_tree(
    curve: DiscountCurve,
    model: HullWhite,
    times: tuple[float, ...],
    steps_per_year: int,
) -> TrinomialTree:
    """Build a curve-fitted tree whose nodes include every time in ``times``.

    Args:
        curve: The initial curve. Every grid zero is repriced from it.
        model: Mean reversion and volatility.
        times: Times that must land on nodes, ascending and positive. The
            exercise dates, in practice.
        steps_per_year: Target step density. Each gap between consecutive
            required times is split into whole sub-steps at roughly this
            density, so the realised density is at least this and the required
            times are exact.

    Returns:
        The tree, with shifts already fitted.

    Raises:
        BadBermudan: If ``times`` is not ascending and positive, if
            ``steps_per_year`` is not positive, or if the grid's step variances
            fall outside the range that keeps branching probabilities
            non-negative.
    """
    if steps_per_year <= 0:
        raise BadBermudan(f"steps per year must be positive, got {steps_per_year!r}")
    if not times:
        raise BadBermudan("a tree needs at least one required time")
    previous = 0.0
    for time in times:
        if not math.isfinite(time) or time <= previous:
            raise BadBermudan(
                f"required times must be finite, positive and ascending, got {times!r}"
            )
        previous = time
    grid = [0.0]
    for time in times:
        base = grid[-1]
        gap = time - base
        count = max(1, round(gap * steps_per_year))
        step = gap / count
        grid.extend(base + step * k for k in range(1, count))
        grid.append(time)
    # One step past the last required time. The short rate *at* that time is the
    # shift of the step beginning there, and the last exercise date needs it to
    # value the swap it would be exercised into. Without the extra step the
    # final required time would be a node with no step and no shift.
    grid.append(grid[-1] + (grid[-1] - grid[-2]))
    node_times = tuple(grid)
    gaps = [later - earlier for earlier, later in pairwise(node_times)]
    variances = [
        -model.sigma**2 * math.expm1(-2.0 * model.a * gap) / (2.0 * model.a)
        for gap in gaps
    ]
    if model.sigma == 0.0:
        # No randomness, so one state. Letting the width grow as it does below
        # would add nodes of exactly zero probability whose *rates* are a
        # spacing apart, and with no volatility to set the spacing from there is
        # no sensible value for it: a placeholder of one put node rates hundreds
        # of per cent away from the curve and overflowed a bond price to
        # infinity, which subtracted to NaN. A degenerate model gets a
        # degenerate tree.
        spacing = 0.0
        ratios = tuple(0.0 for _ in gaps)
        forwards = [
            -math.log(
                curve.discount_at(later) / curve.discount_at(earlier)
            )
            / (later - earlier)
            for earlier, later in pairwise(node_times)
        ]
        return TrinomialTree(
            times=node_times,
            spacing=spacing,
            shift=tuple(forwards),
            lower=tuple(0 for _ in node_times),
            upper=tuple(0 for _ in node_times),
            state_prices=tuple(
                (curve.discount_at(time),) for time in node_times
            ),
            variance_ratio=ratios,
        )
    else:
        spacing = math.sqrt(3.0 * math.fsum(variances) / len(variances))
        ratios = tuple(variance / spacing**2 for variance in variances)
        worst = min(ratios)
        if worst < _VARIANCE_FLOOR or max(ratios) > _VARIANCE_CEILING:
            raise BadBermudan(
                f"the time grid is too uneven for one state spacing: step variances "
                f"span {worst:.4f} to {max(ratios):.4f} of dx**2, outside the "
                f"[{_VARIANCE_FLOOR}, {_VARIANCE_CEILING}] that keeps every branching "
                "probability non-negative. Raise the step density so the required "
                "times are split more finely, or space them more evenly."
            )
    lower = [0]
    upper = [0]
    shift: list[float] = []
    state_prices: list[tuple[float, ...]] = [(1.0,)]
    current = [1.0]
    for index, gap in enumerate(gaps):
        decay = math.exp(-model.a * gap)
        lo, hi = lower[index], upper[index]
        undiscounted = math.fsum(
            price * math.exp(-(lo + offset) * spacing * gap)
            for offset, price in enumerate(current)
        )
        target = curve.discount_at(node_times[index + 1])
        theta = -math.log(target / undiscounted) / gap
        shift.append(theta)
        centre_lo = round(lo * decay)
        centre_hi = round(hi * decay)
        next_lo, next_hi = centre_lo - 1, centre_hi + 1
        nxt = [0.0] * (next_hi - next_lo + 1)
        ratio = ratios[index]
        for offset, price in enumerate(current):
            node = lo + offset
            flow = price * math.exp(-(theta + node * spacing) * gap)
            centre = round(node * decay)
            residual = node * decay - centre
            square = residual * residual
            up = 0.5 * (ratio + square + residual)
            middle = 1.0 - ratio - square
            down = 0.5 * (ratio + square - residual)
            base = centre - next_lo
            nxt[base + 1] += flow * up
            nxt[base] += flow * middle
            nxt[base - 1] += flow * down
        lower.append(next_lo)
        upper.append(next_hi)
        state_prices.append(tuple(nxt))
        current = nxt
    return TrinomialTree(
        times=node_times,
        spacing=spacing,
        shift=tuple(shift),
        lower=tuple(lower),
        upper=tuple(upper),
        state_prices=tuple(state_prices),
        variance_ratio=ratios,
    )


def _swap_flows(
    option: BermudanSwaption,
    schedule: Schedule,
    first: int,
    reference: date,
    basis: Basis,
) -> tuple[tuple[float, float], ...]:
    """Fixed-leg flows from period ``first`` on, as ``(time, amount)`` pairs.

    The redemption of notional rides on the last period's payment date, which is
    the schedule's own, not the instrument's nominal maturity. Coupons accrue on
    the unadjusted boundaries and discount at the payment date, exactly as the
    single-curve swap and :func:`~tenor.hullwhite.swaption_price` do it.
    """
    periods = schedule.periods[first:]
    flows: list[tuple[float, float]] = []
    for index, period in enumerate(periods):
        accrual = year_fraction(period.start, period.end, option.basis)
        amount = option.strike * accrual
        if index == len(periods) - 1:
            amount += 1.0
        flows.append((year_fraction(reference, period.payment, basis), amount))
    return tuple(flows)


def coterminal_bonds(
    option: BermudanSwaption,
    reference: date,
    basis: Basis = Basis.ACT_365F,
) -> tuple[CouponBondOption, ...]:
    """Each exercise right as an option on the fixed leg it would enter.

    A receiver swaption is a call on the coupon bond struck at par and a payer is
    a put, which is the equivalence
    :func:`~tenor.hullwhite.swaption_price` is built on. Expressing the rights
    this way rather than as :class:`~tenor.options.Swaption` objects means the
    European and the tree's exercise value are the *same* set of flows, so a
    disagreement between them is a pricing difference and never a scheduling one.
    """
    schedule = option.schedule()
    resolved = option.resolve(schedule)
    bonds: list[CouponBondOption] = []
    for first, start in resolved:
        bonds.append(
            CouponBondOption(
                expiry=year_fraction(reference, start, basis),
                flows=tuple(
                    Flow(time, amount)
                    for time, amount in _swap_flows(
                        option, schedule, first, reference, basis
                    )
                ),
                strike=1.0,
                payoff=option.payoff,
            )
        )
    return tuple(bonds)


def coterminal_europeans(
    curve: DiscountCurve,
    model: HullWhite,
    option: BermudanSwaption,
    reference: date,
    basis: Basis = Basis.ACT_365F,
    method: Method = Method.JAMSHIDIAN,
) -> tuple[float, ...]:
    """Price each co-terminal European analytically, in exercise-date order.

    These are the lower bounds a Bermudan has to clear, and they come from
    :func:`~tenor.hullwhite.coupon_bond_option` — Jamshidian's decomposition, or
    a Gauss-Legendre integral under ``Method.QUADRATURE`` — so comparing a
    Bermudan with them tests backward induction against analysis that shares no
    code with it.
    """
    return tuple(
        coupon_bond_option(curve, model, bond, method)
        for bond in coterminal_bonds(option, reference, basis)
    )


def bermudan_swaption(
    curve: DiscountCurve,
    model: HullWhite,
    option: BermudanSwaption,
    reference: date,
    *,
    basis: Basis = Basis.ACT_365F,
    steps_per_year: int = 48,
) -> BermudanValue:
    """Price a Bermudan swaption by backward induction on a Hull-White tree.

    Args:
        curve: The initial curve, which the tree is fitted to.
        model: Mean reversion and volatility, the same pair the Europeans use.
        option: The instrument.
        reference: Valuation date; times are measured from here.
        basis: Day count for turning dates into times.
        steps_per_year: Step density between exercise dates. The dates
            themselves always land on nodes whatever this is.

    Returns:
        The price, the co-terminal Europeans, the switch value over the best of
        them, the exercise boundary at each date, and the tree.

    Raises:
        BadBermudan: If the first exercise date is not after ``reference``, if an
            exercise date is not a fixed-leg boundary, or if the grid cannot
            carry a single state spacing.
    """
    if option.expiries[0] <= reference:
        raise BadBermudan(
            f"{option.name} may first be exercised on "
            f"{option.expiries[0].isoformat()}, on or before the valuation date "
            f"{reference.isoformat()}; that is a decision, not a value"
        )
    schedule = option.schedule()
    resolved = option.resolve(schedule)
    expiries = tuple(start for _, start in resolved)
    times = tuple(year_fraction(reference, start, basis) for start in expiries)
    tree = build_tree(curve, model, times, steps_per_year)
    bonds = coterminal_bonds(option, reference, basis)
    flows = tuple(
        tuple((flow.time, flow.amount) for flow in bond.flows) for bond in bonds
    )
    exercise_steps = tuple(tree.times.index(time) for time in times)
    sign = option.payoff.sign
    europeans = tuple(
        coupon_bond_option(
            curve,
            model,
            bond,
            Method.JAMSHIDIAN if bond.is_monotone else Method.QUADRATURE,
        )
        for bond in bonds
    )

    # Roll back from the last exercise date, where continuation is worthless.
    last = exercise_steps[-1]
    values = [0.0] * (tree.upper[last] - tree.lower[last] + 1)
    boundaries: list[float | None] = [None] * len(times)
    exercised: list[float] = [0.0] * len(times)
    crossings = 0
    for position in range(len(times) - 1, -1, -1):
        step = exercise_steps[position]
        lo, hi = tree.lower[step], tree.upper[step]
        level = short_rate_level(curve, model, tree.times[step])
        intrinsic = [
            sign
            * (
                1.0
                - math.fsum(
                    amount
                    * model.bond_price(
                        curve, tree.times[step], time, level + tree.state(node)
                    )
                    for time, amount in flows[position]
                )
            )
            for node in range(lo, hi + 1)
        ]
        takes = [value < gain for value, gain in zip(values, intrinsic, strict=True)]
        changes = sum(1 for earlier, later in pairwise(takes) if earlier != later)
        if changes > 1:
            crossings += 1
        if changes >= 1:
            # Interpolate the short rate where waiting and exercising are level.
            for offset in range(len(takes) - 1):
                if takes[offset] != takes[offset + 1]:
                    low_gap = intrinsic[offset] - values[offset]
                    high_gap = intrinsic[offset + 1] - values[offset + 1]
                    weight = low_gap / (low_gap - high_gap)
                    boundaries[position] = (
                        level + tree.state(lo + offset) + weight * tree.spacing
                    )
                    break
        prices = tree.state_prices[step]
        mass = math.fsum(prices)
        if mass > 0.0:
            exercised[position] = (
                math.fsum(
                    price
                    for price, take in zip(prices, takes, strict=True)
                    if take
                )
                / mass
            )
        values = [
            gain if take else value
            for value, gain, take in zip(values, intrinsic, takes, strict=True)
        ]
        previous = exercise_steps[position - 1] if position > 0 else 0
        for index in range(step - 1, previous - 1, -1):
            gap = tree.times[index + 1] - tree.times[index]
            decay = math.exp(-model.a * gap)
            lo_now, hi_now = tree.lower[index], tree.upper[index]
            base_next = tree.lower[index + 1]
            ratio = tree.variance_ratio[index]
            theta = tree.shift[index]
            if tree.spacing == 0.0:
                # The degenerate tree has one state, so rolling back is
                # discounting and there are no neighbours to read.
                values = [math.exp(-theta * gap) * values[0]]
                continue
            rolled = [0.0] * (hi_now - lo_now + 1)
            for offset in range(hi_now - lo_now + 1):
                node = lo_now + offset
                centre = round(node * decay)
                residual = node * decay - centre
                square = residual * residual
                up = 0.5 * (ratio + square + residual)
                middle = 1.0 - ratio - square
                down = 0.5 * (ratio + square - residual)
                seat = centre - base_next
                rolled[offset] = math.exp(-(theta + node * tree.spacing) * gap) * (
                    up * values[seat + 1]
                    + middle * values[seat]
                    + down * values[seat - 1]
                )
            values = rolled
    price = values[0]
    return BermudanValue(
        price=price,
        steps=tuple(
            ExerciseStep(
                expiry=expiry,
                time=time,
                step=step,
                european=european,
                boundary=boundary,
                exercised=taken,
            )
            for expiry, time, step, european, boundary, taken in zip(
                expiries,
                times,
                exercise_steps,
                europeans,
                boundaries,
                exercised,
                strict=True,
            )
        ),
        tree=tree,
        crossings=crossings,
    )
