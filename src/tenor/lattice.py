"""A short rate lattice, and the spread of a bond whose flows are not known.

Everything before this module prices a bond whose cash flows are fixed. A
callable bond's are not: whether the issuer redeems early depends on where
rates are when the call date arrives, which depends on a path nobody has yet
taken. A single discounting curve cannot express that, because there is only
one of it.

So the curve is turned into a tree. At each step the short rate goes up or
down with equal risk-neutral probability, and the bond is valued backwards from
maturity: at every node the holder or the issuer takes whichever of exercise
and continuation is worth more to *them*, and the value that propagates back is
the one that decision leaves.

**The calibration has to be exact.** A lattice that reprices the underlying
curve to within a few basis points contaminates every spread measured on it by
that much, and an option-adjusted spread is quoted in basis points. So the
level at each step is not set from a formula — it is solved, one step at a
time, so that the tree's own price for the zero coupon bond maturing at that
step is the curve's discount factor to the last bit. :meth:`Lattice.reprices`
asserts it afterwards rather than trusting the solve.

**Black-Derman-Toy rather than Ho-Lee**, for one reason: the short rate is
lognormal, so it cannot go negative. Ho-Lee's normal rate can, and a tree with
negative rates in the low branches will still produce a plausible-looking OAS
while the exercise decisions in those branches are being made about rates that
do not exist. At the cost of the volatility being proportional rather than
absolute, which is the more usual way it is quoted anyway.

**Exercise goes opposite ways.** A call is the *issuer's* right: they redeem
when continuation is worth more than the call price, so the call caps the
bond's value and its option cost is positive to the holder. A put is the
*holder's*: they redeem when continuation is worth less than the put price, so
it floors the value. Implementing one and negating it for the other gives a
number that still looks like a spread and is wrong in a direction nobody
checks.

**The identity worth the whole module** is that Z-spread minus OAS is the cost
of the embedded option. That is the reason to compute an OAS at all: a
callable bond's Z-spread includes compensation for a short option position and
so overstates what the holder earns. A bond with no option must have an OAS
equal to its Z-spread, which is the first thing the tests check.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from .curve import DiscountCurve
from .solve import Root, brent


class BadLattice(ValueError):
    """A lattice that cannot be built or valued."""


@dataclass(frozen=True)
class Lattice:
    """A calibrated binomial tree of one-period short rates.

    ``rates[i][j]`` is the continuously compounded short rate that applies over
    step ``i``, in the state reached by ``j`` up moves. The tree recombines, so
    step ``i`` has ``i + 1`` states.

    ``state_prices[i][j]`` is the Arrow-Debreu price of one unit paid only in
    that state — what it is worth today. They are kept because the calibration
    needs them and because their sum at step ``i`` is the tree's own price of
    the zero coupon bond maturing there, which is the quantity being matched.
    """

    step: float
    volatility: float
    rates: tuple[tuple[float, ...], ...]
    state_prices: tuple[tuple[float, ...], ...]

    @property
    def steps(self) -> int:
        return len(self.rates)

    def time_at(self, index: int) -> float:
        return index * self.step

    def zero_price(self, index: int) -> float:
        """The tree's price for a zero coupon bond maturing at step ``index``."""
        if not 0 <= index <= self.steps:
            raise BadLattice(
                f"step {index!r} is outside a lattice of {self.steps} steps"
            )
        return sum(self.state_prices[index])

    @classmethod
    def calibrated(
        cls,
        curve: DiscountCurve,
        settlement: date,
        *,
        steps: int,
        step: float,
        volatility: float,
    ) -> Lattice:
        """Build a tree that reprices ``curve`` exactly, step by step.

        The shape is fixed by the volatility — within a step the rate is
        ``level * exp(2 j sigma sqrt(dt))``, so the ratio between adjacent
        states never changes — and only the level is free. One unknown per
        step, one discount factor to match per step, solved in order. Each
        solve uses the state prices already accumulated, so the cost is linear
        in the number of steps rather than quadratic.
        """
        if steps < 1:
            raise BadLattice(f"a lattice needs at least one step, got {steps!r}")
        if step <= 0.0:
            raise BadLattice(f"a step is a positive length of time, got {step!r}")
        if volatility < 0.0:
            raise BadLattice(f"a volatility is not negative, got {volatility!r}")

        base = curve.discount(settlement)
        start = curve.time_to(settlement)
        targets = []
        for index in range(1, steps + 1):
            time = start + index * step
            if time > curve.times[-1] + 1e-12:
                raise BadLattice(
                    f"a lattice of {steps} steps of {step!r} years reaches "
                    f"{start + steps * step:.4f} years, past the curve's last pillar "
                    f"at {curve.times[-1]:.4f}. The tree would be extrapolating, "
                    "which the curve refuses to do directly and should not do by "
                    "proxy."
                )
            targets.append(curve.discount_at(time) / base)

        spacing = math.exp(2.0 * volatility * math.sqrt(step))
        rates: list[tuple[float, ...]] = []
        state: list[tuple[float, ...]] = [(1.0,)]

        for index in range(steps):
            prices = state[index]

            def tree_price(level: float, prices: tuple[float, ...] = prices) -> float:
                total = 0.0
                for position, weight in enumerate(prices):
                    rate = level * spacing**position
                    total += weight * math.exp(-rate * step)
                return total

            target = targets[index]

            def objective(level: float, target: float = target) -> float:
                return tree_price(level) - target

            level = _solve_level(objective, index, target)
            row = tuple(level * spacing**position for position in range(index + 1))
            rates.append(row)

            # Roll the state prices forward one step. Half of each node's value
            # goes each way, discounted at that node's own rate.
            forward = [0.0] * (index + 2)
            for position, weight in enumerate(prices):
                moved = 0.5 * weight * math.exp(-row[position] * step)
                forward[position] += moved
                forward[position + 1] += moved
            state.append(tuple(forward))

        return cls(
            step=step,
            volatility=volatility,
            rates=tuple(rates),
            state_prices=tuple(state),
        )

    def reprices(
        self, curve: DiscountCurve, settlement: date, tolerance: float = 1e-12
    ) -> bool:
        """Whether the tree reproduces the curve's own discount factors.

        Checked against the curve rather than against the targets the
        calibration used, so a mistake in assembling the state prices after
        each solve would show here. The two are the same numbers only if
        everything in between is right.
        """
        base = curve.discount(settlement)
        start = curve.time_to(settlement)
        for index in range(1, self.steps + 1):
            expected = curve.discount_at(start + index * self.step) / base
            if abs(self.zero_price(index) - expected) > tolerance:
                return False
        return True


def _solve_level(
    objective: Callable[[float], float], index: int, target: float
) -> float:
    """Solve for a step's rate level, on a bracket that always contains it.

    The tree's price for the step falls monotonically in the level, from one at
    a rate of zero towards nothing, so any target strictly between zero and one
    is bracketed. Five hundred percent is the upper end; nothing that reprices
    a real curve comes near it, and the bracket costs two evaluations.
    """
    low, high = 1e-10, 5.0
    if objective(low) * objective(high) > 0.0:
        raise BadLattice(
            f"no short rate level between {low} and {high} reproduces the curve's "
            f"discount factor of {target!r} at step {index}. A target above one "
            "means the curve's discount factors increase over this step, which is "
            "an arbitrage rather than a calibration problem."
        )
    return brent(objective, low, high, tolerance=1e-16).value


# -- valuing a bond on the tree ----------------------------------------------


@dataclass(frozen=True)
class Exercise:
    """A right to redeem early, held by one side or the other."""

    #: Step index at which the right may be exercised, and the price it is at.
    schedule: tuple[tuple[int, float], ...]
    #: True when the issuer holds it (a call), False when the holder does.
    issuer: bool = True

    def price_at(self, index: int) -> float | None:
        for step, price in self.schedule:
            if step == index:
                return price
        return None


def lattice_price(
    lattice: Lattice,
    *,
    coupon: float,
    redemption: float = 100.0,
    spread: float = 0.0,
    exercise: Exercise | None = None,
) -> float:
    """Value a bond on the tree, rolling backwards from maturity.

    ``coupon`` is the cash paid per step, so a 5% semi-annual bond on a
    half-year step pays 2.5. Every step pays one, which is what restricts the
    tree's step to the coupon frequency: putting a coupon between two nodes
    would mean discounting it at a rate the tree does not have.

    ``spread`` is added to every node's short rate. A spread of zero prices the
    bond off the curve the lattice was calibrated to, which is the identity the
    tests use to tie this module to the last one.

    The returned value is *cum* the coupon paid at maturity and *ex* any coupon
    paid at the valuation date, matching the convention everywhere else here
    that a flow due today belongs to the seller.
    """
    steps = lattice.steps
    # At maturity the bond is worth its redemption plus the last coupon.
    values = [redemption + coupon] * (steps + 1)

    for index in reversed(range(steps)):
        row = lattice.rates[index]
        nxt = values
        values = []
        for position in range(index + 1):
            discount = math.exp(-(row[position] + spread) * lattice.step)
            continuation = discount * 0.5 * (nxt[position] + nxt[position + 1])
            if index > 0:
                continuation += coupon
            value = continuation
            if exercise is not None:
                strike = exercise.price_at(index)
                if strike is not None:
                    # The issuer redeems when continuing costs them more than
                    # the strike; the holder redeems when continuing is worth
                    # less. Opposite sides, opposite comparisons.
                    payoff = strike + (coupon if index > 0 else 0.0)
                    value = min(value, payoff) if exercise.issuer else max(value, payoff)
            values.append(value)
    return values[0]


def option_adjusted_spread(
    lattice: Lattice,
    price: float,
    *,
    coupon: float,
    redemption: float = 100.0,
    exercise: Exercise | None = None,
) -> Root:
    """The constant spread over every node's short rate that reproduces ``price``.

    Against a *dirty* price, as every spread here is. With no exercise schedule
    this is the Z-spread computed on the tree instead of on the curve, and the
    two agree — which is what makes the difference between them, once an
    exercise schedule is added, attributable to the option and nothing else.
    """

    def objective(spread: float) -> float:
        return (
            lattice_price(
                lattice,
                coupon=coupon,
                redemption=redemption,
                spread=spread,
                exercise=exercise,
            )
            - price
        )

    low, high = -0.5, 1.0
    if objective(low) * objective(high) > 0.0:
        raise BadLattice(
            f"no spread between {low} and {high} prices this bond at {price!r}: "
            f"the error is {objective(low):.6g} at one end and "
            f"{objective(high):.6g} at the other"
        )
    return brent(objective, low, high, tolerance=1e-15)


def option_cost(zero_volatility_spread: float, adjusted_spread: float) -> float:
    """Z-spread less OAS: what the embedded option is worth, in rate terms.

    Positive for a callable bond, because the holder is short the option and
    the Z-spread is paying them for it. Negative for a putable one, where they
    are long. Trivial arithmetic, named because the naming is the point — the
    subtraction is the whole reason an OAS is computed.
    """
    return zero_volatility_spread - adjusted_spread


def steps_between(
    curve: DiscountCurve, settlement: date, maturity: date, step: float
) -> int:
    """How many whole steps of length ``step`` fit before ``maturity``.

    Refuses a maturity that is not a whole number of steps away, rather than
    rounding. A tree whose last node is three weeks from the redemption date
    prices a different bond, and does so without saying so.
    """
    span = curve.time_to(maturity) - curve.time_to(settlement)
    count = round(span / step)
    if count < 1:
        raise BadLattice(
            f"{maturity.isoformat()} is less than one step of {step!r} years from "
            f"{settlement.isoformat()}"
        )
    if abs(count * step - span) > 0.01:
        raise BadLattice(
            f"{settlement.isoformat()} to {maturity.isoformat()} is {span:.4f} "
            f"years, which is not a whole number of {step!r}-year steps "
            f"({count} steps would be {count * step:.4f}). The tree pays a coupon "
            "at every node, so a maturity between nodes is a different bond."
        )
    return count


__all__ = [
    "BadLattice",
    "Exercise",
    "Lattice",
    "lattice_price",
    "option_adjusted_spread",
    "option_cost",
    "steps_between",
]
