"""The swap rate paid outside the measure it is a martingale under.

Every option in :mod:`tenor.options` is priced by making the forward swap rate
a martingale under the fixed leg's annuity, which is the measure a swaption
settles against: the annuity is the numeraire and the strike is compared to a
rate that drifts nowhere. A constant maturity swap breaks that. It observes the
same swap rate and pays it *once*, on a single date, so the expectation wanted
is under the forward measure of that payment date instead.

The swap rate is not a martingale there, and the gap is not small. Writing
``A(t)`` for the annuity and ``P(t, Tp)`` for the discount factor to the
payment date, the change of measure carries the ratio::

    P(0, Tp) E^Tp[g(S_T)] = A(0) E^A[g(S_T) alpha(S_T)]
    alpha(S) = P(T, Tp) / A(T)

:class:`AnnuityMap` is that ``alpha``, as a function of the rate rather than as
a constant, which is the whole of the modelling here. Everything above it is
the Carr-Madan expansion of ``g alpha`` around the forward and a quadrature
over the swaptions the package already prices, so a CMS is valued out of the
same surface a swaption book is marked on rather than out of a separate model.

**A single period reduces exactly, and that is the load-bearing check.** If the
underlying swap has one fixed period and the CMS pays at the end of it, then
``A(T) = delta P(T, Tp)`` identically, so ``alpha`` is ``1 / delta`` — flat in
the rate, every derivative of it zero, and the convexity adjustment
**identically 0.0**. It has to be: that contract is a forward rate paid in
arrears, which needs no adjustment at all. The replication is then the point
mass at the kink and nothing else, so it matches :mod:`tenor.options` to
**5e-16** across four strikes and both payoffs. It is the only check here that
would catch the point mass carrying the wrong sign, which is the class of error
that cost :mod:`tenor.g2` a call and a put.

**The second moment has a closed form, and that is what validates the
quadrature.** Against that same flat mapping, replicating ``S**2`` must return
``F**2 exp(sigma**2 T)`` under Black and ``F**2 + sigma**2 T`` under Bachelier.
No model enters either statement, so it checks the expansion, the Gauss rule
and the panel layout against arithmetic from outside this package — both to
**rounding**. It is also the only functional here that is sensitive to the far
tail, which is how the panel spacing below was found out.

**Uniform panels make a wider ceiling worse, not better.** The lognormal
ceiling is ``F exp(w sigma sqrt(T))``, so it moves out exponentially in ``w``
while the panel count stays put: every extra standard deviation coarsens the
mesh *at the forward*, where the integrand lives. On the second moment, uniform
panels gave relative errors of 1.4e-09, 3.2e-08, 7.9e-04 and 1.8e-01 as the
ceiling went from six standard deviations to twenty — monotonically worse, by
eight orders of magnitude, for covering more of the distribution. Spacing the
edges evenly in ``log K`` holds every ceiling from eight to thirty at rounding,
and sixteen panels then reach 5.1e-13.

**The flat-curve mapping is arbitrageable, twice over.** The standard model
evaluates ``alpha`` by assuming the curve is flat at whatever level the rate
fixes at, which makes ``P(T, Tp)`` and ``A(T)`` explicit powers of ``1 + S/q``.
Raw, it returns the *flat curve's* ratio at the forward rather than the real
one: on the upward-sloping curve in the tests it prices a zero-coupon bond
**0.88% wrong**, a quarter of the convexity adjustment it exists to compute.
:func:`annuity_map` scales that away at the forward — and the constant payoff
is then *still* worth 0.217% more than the bond it is, because matching at one
point does not match an expectation. So :meth:`ConstantMaturity.replicate`
divides by the replicated unit payoff, which enforces the one no-arbitrage
condition available, moves the adjustment by **3.6%**, and is a no-op wherever
the mapping is constant or the volatility is zero — so no reduction above is
disturbed. :func:`measure_error` reports the gap rather than hiding it.

**What the adjustment is worth, measured.** On a ten-year rate fixing in five
years and paid six months later, off that curve at a flat 24% lognormal
volatility, it is **+26.06 basis points** on a forward swap rate of 4.1053% —
0.63% of the rate, and several times any spread being quoted on it.

**It is not quadratic in the volatility**, which is the usual rule of thumb.
The expansion is leading-order in the variance and at low volatility that
holds, but the ratio of the adjustment to ``sigma**2`` runs 0.0382, 0.0395,
0.0452, 0.0570 and 0.0773 at volatilities of 6%, 12%, 24%, 36% and 48%. It
doubles across the range, so scaling a 24% adjustment up to 48% underestimates
it by a third.

**Paying later unwinds the convexity rather than compounding it.** This came
out against the guess. Moving the payment of that fixing from the fixing date
out to ten years past it takes the adjustment **29.28, 26.06, 22.92, 16.83,
5.43, 0.07 and -23.86 basis points** — monotonically down, and through zero.
The sign is the sign of ``alpha'(F)``, and ``alpha'(F)`` vanishes where the
payment date meets the annuity's own annuity-weighted mean payment time: that
mean is **5.1872 years** here and the slope crosses zero at 62 months, which is
5.1667. Structural rather than coincidental — ``alpha`` is one discount factor
over a weighted sum of them, so putting the payment at the weighted mean makes
numerator and denominator respond to the rate alike.

**And it is a smile instrument, which is the practical reason this module
exists.** The adjustment is an integral of swaption prices across every strike,
so it reads the whole surface and not the at-the-money point. Against surfaces
that leave every at-the-money swaption worth exactly what the flat one does, it
comes to 100.0%, 96.9%, 94.3%, 92.1% and 88.5% of the flat number at skews of
0, -0.10, -0.20, -0.30 and -0.50 per unit of rate. An 8% error from a marking
choice no at-the-money quote can see.

**A surface sloping *up* in strike has no answer at all.** A volatility rising
without bound stops the swaption price decaying, so the integral diverges and
whatever comes back is a function of the ceiling: the same fixing against a
+0.30 skew reads 35.6 basis points at a six standard deviation ceiling and
36565 at fourteen. Every one of those is wrong and none looks it, so
:attr:`Replication.truncation` measures the share of the value beyond the
ceiling — 5.5e-24 on the flat case, exactly 0.0 against a downward skew, 120%
here — and :attr:`Quadrature.tolerance` refuses it rather than returning a
number whose only real input was the ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from itertools import pairwise

from .calendar import Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .multicurve import ForecastIndex
from .options import (
    Convention,
    Payoff,
    Swaption,
    bachelier,
    black,
)
from .schedule import Frequency, Period, add_months, generate

__all__ = [
    "STANDARD",
    "AnnuityMap",
    "BadConvexity",
    "ConstantMaturity",
    "ConstantMaturityLeg",
    "Convexity",
    "Fixing",
    "OptionPayoff",
    "PowerPayoff",
    "Quadrature",
    "Replication",
    "Smile",
    "Strike",
    "annuity_map",
    "flat_smile",
    "measure_error",
]


class BadConvexity(ValueError):
    """A replication that cannot be set up, or a mapping asked outside its domain."""


#: A volatility as a function of strike. A flat surface is
#: :func:`flat_smile`; anything steeper is the caller's own grid.
Smile = Callable[[float], float]


def flat_smile(volatility: float) -> Smile:
    """The same volatility at every strike.

    Written as a function rather than special-cased so that nothing downstream
    has to branch on whether a surface was supplied, and so the flat case goes
    down exactly the same path the smile does.
    """
    if volatility < 0.0 or not math.isfinite(volatility):
        raise BadConvexity(f"a volatility of {volatility!r} is not one")

    def at(strike: float) -> float:
        return volatility

    return at


def _as_smile(volatility: float | Smile) -> Smile:
    if callable(volatility):
        return volatility
    return flat_smile(volatility)


# -- the annuity mapping function ---------------------------------------------


@dataclass(frozen=True)
class AnnuityMap:
    """``P(T, Tp) / A(T)`` as a function of the swap rate, with two derivatives.

    The model is the standard one: at the fixing the curve is taken to be flat
    at the level the swap rate came out at, so with ``u = 1 + S/q`` for a fixed
    leg paying ``q`` times a year::

        P(T, Tp) = u ** (-q * delay)
        A(T)     = sum_i accrual_i * u ** (-q * offset_i)

    The offsets and the delay are year fractions measured *from the fixing*, on
    the fixed leg's own basis, so that the single-period case collapses the
    ratio to ``1 / accrual`` whatever ``q`` is.

    ``scale`` multiplies the whole thing. :func:`annuity_map` sets it so that
    the mapping agrees with the real curve at the forward; a ``scale`` of one
    is the raw flat-curve model, which does not, and the module docstring
    measures what that costs.
    """

    #: The fixed leg's accrual factors, in schedule order.
    accruals: tuple[float, ...]
    #: Year fractions from the fixing to each fixed payment, in the same order.
    offsets: tuple[float, ...]
    #: Year fraction from the fixing to the date the CMS pays on.
    delay: float
    #: Fixed-leg payments per year, the compounding of the flat rate.
    frequency: int = int(Frequency.SEMI_ANNUAL)
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.accruals:
            raise BadConvexity(
                "an annuity mapping needs at least one fixed period; none were given"
            )
        if len(self.accruals) != len(self.offsets):
            raise BadConvexity(
                f"the mapping has {len(self.accruals)} accruals and "
                f"{len(self.offsets)} payment offsets, which cannot be paired"
            )
        if any(one <= 0.0 for one in self.accruals):
            raise BadConvexity(
                "a fixed period with a non-positive accrual cannot discount anything"
            )
        if self.delay < 0.0:
            raise BadConvexity(
                f"the payment is {self.delay!r} years after the fixing, which is "
                "before it; a CMS cannot pay out on a rate it has not observed"
            )
        if self.frequency <= 0:
            raise BadConvexity(f"a fixed leg paying {self.frequency!r} times a year is not one")

    @property
    def floor(self) -> float:
        """The rate below which the flat curve has a non-positive discount factor.

        The mapping is a power of ``1 + S/q``, so it exists only strictly above
        ``-q``. Nothing in this module integrates near it, but a normal-
        convention surface with a wide enough ceiling would, and an exception
        naming the bound is better than a complex number.
        """
        return -float(self.frequency)

    def _base(self, rate: float) -> float:
        base = 1.0 + rate / float(self.frequency)
        if base <= 0.0:
            raise BadConvexity(
                f"the annuity mapping was asked for a rate of {rate!r}, at or below "
                f"the {self.floor!r} where the flat curve's discount factor stops "
                "being positive"
            )
        return base

    def _parts(self, rate: float) -> tuple[float, float, float, float, float, float]:
        """``(N, N', N'', D, D', D'')`` at ``rate``, before ``scale``."""
        base = self._base(rate)
        reciprocal = 1.0 / float(self.frequency)
        delay = self.delay
        power = base ** (-float(self.frequency) * delay)
        numerator = power
        first = -delay * power / base
        second = delay * (delay + reciprocal) * power / (base * base)
        total = 0.0
        total_first = 0.0
        total_second = 0.0
        for accrual, offset in zip(self.accruals, self.offsets, strict=True):
            weight = accrual * base ** (-float(self.frequency) * offset)
            total += weight
            total_first -= offset * weight / base
            total_second += offset * (offset + reciprocal) * weight / (base * base)
        return numerator, first, second, total, total_first, total_second

    def value(self, rate: float) -> float:
        """``alpha(S)``."""
        numerator, _, _, total, _, _ = self._parts(rate)
        return self.scale * numerator / total

    def slope(self, rate: float) -> float:
        """``alpha'(S)``."""
        numerator, first, _, total, total_first, _ = self._parts(rate)
        return self.scale * (first * total - numerator * total_first) / (total * total)

    def curvature(self, rate: float) -> float:
        """``alpha''(S)``.

        From ``N = alpha D`` differentiated twice, so that the second
        derivative of the ratio never has to be expanded: ``alpha'' =
        (N'' - 2 alpha' D' - alpha D'') / D``. Three terms instead of the
        quotient rule's five, and no cancellation between large products.
        """
        numerator, first, second, total, total_first, total_second = self._parts(rate)
        alpha = numerator / total
        slope = (first * total - numerator * total_first) / (total * total)
        raw = (second - 2.0 * slope * total_first - alpha * total_second) / total
        return self.scale * raw

    def rescaled(self, target: float, forward: float) -> AnnuityMap:
        """The same shape, scaled so that ``value(forward)`` is ``target``.

        ``target`` is the curve's own ``P(0, Tp) / A(0)``. Without this the
        model disagrees with the curve about a zero-coupon bond.
        """
        here = self.value(forward)
        if here <= 0.0:
            raise BadConvexity(
                f"the mapping is worth {here!r} at a forward of {forward!r}, so it "
                "cannot be scaled onto the curve"
            )
        if target <= 0.0:
            raise BadConvexity(
                f"the curve puts {target!r} on one unit of annuity at the payment "
                "date, which is not a positive number of discount factors"
            )
        return replace(self, scale=self.scale * target / here)


# -- payoffs, and the second derivative each one contributes -------------------


class Strike(str, Enum):
    """Whether an option payoff is live above or below its strike.

    Named after the rate rather than after the option, for the same reason
    :class:`~tenor.options.Payoff` is: a cap on a rate is a call, and calling
    it a call is what inverts when the same code is pointed at a bond.
    """

    ABOVE = "above"
    BELOW = "below"

    @classmethod
    def of(cls, payoff: Payoff) -> Strike:
        return cls.ABOVE if payoff is Payoff.PAYER else cls.BELOW

    @property
    def sign(self) -> float:
        return 1.0 if self is Strike.ABOVE else -1.0


@dataclass(frozen=True)
class RatePayoff:
    """``scale * S + offset``. A CMS swaplet is the default; ``scale`` zero is
    the constant payoff that :func:`measure_error` replicates.

    Smooth everywhere, so it contributes no point mass and the whole of its
    convexity is in the integral.
    """

    scale: float = 1.0
    offset: float = 0.0

    @property
    def name(self) -> str:
        if self.scale == 0.0:
            return f"a flat {self.offset:g}"
        return "the swap rate" if (self.scale, self.offset) == (1.0, 0.0) else "a linear payoff"

    def value(self, rate: float) -> float:
        return self.scale * rate + self.offset

    @property
    def nodes(self) -> tuple[float, ...]:
        return ()

    def mass(self, strike: float, mapping: AnnuityMap) -> float:
        raise BadConvexity("a linear payoff has no kink to carry a point mass")

    def weight(self, rate: float, mapping: AnnuityMap) -> float:
        """``(g alpha)''`` at ``rate``: ``2 g' alpha' + g alpha''``."""
        return 2.0 * self.scale * mapping.slope(rate) + self.value(rate) * mapping.curvature(rate)


@dataclass(frozen=True)
class OptionPayoff:
    """``max(S - K, 0)`` or ``max(K - S, 0)``, by :class:`Strike`.

    The kink is the only point mass in this module. ``g'`` jumps by one across
    the strike whichever way the option points, so ``(g alpha)''`` carries a
    Dirac of ``alpha(K)`` there — positive for both, because the jump in ``g'``
    is from ``0`` to ``+1`` for a cap and from ``-1`` to ``0`` for a floor.
    Away from the strike the payoff is linear, so the continuous part is the
    linear one's with ``g'`` equal to the :class:`Strike`'s sign.
    """

    strike: float
    side: Strike = Strike.ABOVE

    @property
    def name(self) -> str:
        where = "above" if self.side is Strike.ABOVE else "below"
        return f"the rate {where} {self.strike:.4%}"

    def value(self, rate: float) -> float:
        return max(self.side.sign * (rate - self.strike), 0.0)

    @property
    def nodes(self) -> tuple[float, ...]:
        return (self.strike,)

    def mass(self, strike: float, mapping: AnnuityMap) -> float:
        return mapping.value(strike)

    def weight(self, rate: float, mapping: AnnuityMap) -> float:
        live = self.side.sign * (rate - self.strike)
        if live <= 0.0:
            return 0.0
        return 2.0 * self.side.sign * mapping.slope(rate) + live * mapping.curvature(rate)


@dataclass(frozen=True)
class PowerPayoff:
    """``S ** power``, for an integer power of two or more.

    A CMS-squared leg is a real if uncommon structure, and the second moment is
    what any implied CMS volatility is read out of. It is here mostly because
    it is the one payoff in this module with a closed-form annuity-measure
    expectation: against a *constant* mapping the replication of the square
    must return ``F**2 exp(sigma**2 T)`` under Black and ``F**2 + sigma**2 T``
    under Bachelier, with no model in the statement at all. That is the
    independent check on the quadrature, the expansion and the panel split at
    the forward, and nothing else in the module provides one.
    """

    power: int = 2

    def __post_init__(self) -> None:
        if self.power < 2:
            raise BadConvexity(
                f"a power of {self.power!r} is linear or constant; use RatePayoff, "
                "whose second derivative is written out rather than taken of a power"
            )

    @property
    def name(self) -> str:
        return f"the swap rate to the power {self.power}"

    def value(self, rate: float) -> float:
        return float(rate) ** self.power

    @property
    def nodes(self) -> tuple[float, ...]:
        return ()

    def mass(self, strike: float, mapping: AnnuityMap) -> float:
        raise BadConvexity("a power payoff has no kink to carry a point mass")

    def weight(self, rate: float, mapping: AnnuityMap) -> float:
        """``g'' alpha + 2 g' alpha' + g alpha''``."""
        order = self.power
        first = order * rate ** (order - 1)
        second = order * (order - 1) * rate ** (order - 2)
        return (
            second * mapping.value(rate)
            + 2.0 * first * mapping.slope(rate)
            + self.value(rate) * mapping.curvature(rate)
        )


#: Anything this module can replicate. A protocol would admit a caller's own
#: payoff, which is attractive and wrong: the quadrature needs the exact
#: location of every kink and the exact point mass at it, and a payoff that
#: reports either one approximately is a silent error rather than a loud one.
Fixing = RatePayoff | OptionPayoff | PowerPayoff


# -- the replication itself ---------------------------------------------------


@dataclass(frozen=True)
class Quadrature:
    """How far the replication integrates and how finely.

    Collected into one object rather than spread across five keyword arguments
    because every entry point in this module needs all of them and forwarding
    them individually is where a misspelled ``panels`` quietly takes a default.

    ``width`` is the ceiling in standard deviations of the rate under
    ``convention``: lognormal reaches ``F exp(+- width sigma sqrt(T))`` and
    normal ``F +- width sigma sqrt(T)``.
    """

    convention: Convention = Convention.LOGNORMAL
    vol_basis: Basis = Basis.ACT_365F
    width: float = 10.0
    panels: int = 64
    nodes: int = 8
    #: The largest the measured tail may be, relative to the replicated value,
    #: before the replication is refused rather than returned. See
    #: :attr:`Replication.truncation`.
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.width <= 0.0 or not math.isfinite(self.width):
            raise BadConvexity(
                f"a ceiling {self.width!r} standard deviations from the forward is "
                "not one to integrate to"
            )
        if self.panels < 1:
            raise BadConvexity(f"{self.panels!r} panels cannot cover a strike range")
        if self.nodes < 2:
            raise BadConvexity(
                f"a Gauss rule of {self.nodes!r} nodes cannot integrate a curvature"
            )
        if self.tolerance <= 0.0 or not math.isfinite(self.tolerance):
            raise BadConvexity(
                f"a truncation tolerance of {self.tolerance!r} can never be met"
            )


#: The settings everything in this module defaults to: a lognormal surface,
#: volatility on actual/365, a ceiling ten standard deviations out and
#: sixty-four graded panels of eight nodes. Measured on the second moment
#: identity, sixteen panels already reach 5.1e-13 and thirty-two reach
#: rounding, so the default is not where the accuracy comes from -- the grading
#: is -- and it is set here to leave headroom on a steep surface.
STANDARD = Quadrature()


@dataclass(frozen=True)
class Replication:
    """What a static replication came to, and everything that went into it."""

    #: The present value, per unit of notional, after normalisation.
    value: float
    #: The same before it: the raw expansion, which prices a zero-coupon bond
    #: wrong by :attr:`measure` less one.
    raw: float
    #: ``A(0) E^A[alpha] / P(0, Tp)``: what this model and this quadrature make
    #: of one unit paid on the payment date, which has to be one. It is not,
    #: and dividing by it is what :attr:`value` does about that.
    measure: float
    #: ``A(0) g(F) alpha(F)``: what the payoff is worth with no convexity at all.
    intrinsic: float
    #: The point mass at the payoff's kink, if it has one.
    mass: float
    #: The two integrals, payers above the forward and receivers below it.
    above: float
    below: float
    #: The strikes the quadrature ran between.
    lower: float
    upper: float
    #: The integral over one further doubling beyond ``upper``, as a measured
    #: bound on what truncating the ceiling discarded rather than a claim that
    #: nothing was.
    tail: float
    #: Gauss-Legendre panels used, and nodes per panel.
    panels: int
    nodes: int

    @property
    def convexity(self) -> float:
        """Everything that is not the intrinsic value."""
        return self.value - self.intrinsic

    @property
    def arbitrage(self) -> float:
        """``measure - 1``: the model's own inconsistency, in relative terms.

        Zero at zero volatility and zero whenever the mapping is constant, so
        it is the price of the flat-curve assumption rather than of the
        quadrature, and it is worth reading before a number is believed.
        """
        return self.measure - 1.0

    @property
    def truncation(self) -> float:
        """``|tail| / |value|``: how much of the price is beyond the ceiling.

        A replication converges only if the integrand decays faster than the
        swaption prices grow, and whether it does is a property of the
        *surface*, not of this code. An at-the-money or downward-sloping smile
        gives 1e-24 here. A smile sloping *up* in strike gives a number that
        grows with the ceiling, because a volatility rising without bound stops
        the swaption price decaying and the integral has no value to converge
        to; the adjustment then reads 35.6, 316, 4975, 16567 and 36565 basis
        points at ceilings of six, eight, ten, twelve and fourteen standard
        deviations on the ten-year structure in the tests. Every one of those is
        wrong and none of them looks it, which is why
        :attr:`Quadrature.tolerance` refuses them instead.
        """
        if self.value == 0.0:
            return abs(self.tail)
        return abs(self.tail / self.value)


def _deterministic(payoff: Fixing, forward: float, factor: float) -> Replication:
    """What a replication is worth when the rate cannot move.

    At zero volatility the swap rate is known to be the forward, so the payoff
    is a fixed cashflow on a known date and the value is the discount factor
    times it. That is exact and needs neither the mapping nor the quadrature,
    which is the reason to answer it here rather than to let a degenerate
    integration range do it: going through ``alpha(F)`` instead leaves the
    adjustment at -6.9e-18 rather than at zero, and a degenerate range cannot
    cover an option's kink at all, so a strike away from the forward was being
    refused where its intrinsic value was the obvious answer.
    """
    value = factor * payoff.value(forward)
    return Replication(
        value=value,
        raw=value,
        measure=1.0,
        intrinsic=value,
        mass=0.0,
        above=0.0,
        below=0.0,
        lower=forward,
        upper=forward,
        tail=0.0,
        panels=0,
        nodes=0,
    )


def _legendre(degree: int, point: float) -> tuple[float, float]:
    """``(P_n(x), P_n'(x))`` by the three-term recurrence.

    The derivative comes out of the recurrence's own companion identity rather
    than from differencing, so Newton below converges to the last bit instead of
    to the square root of a step size.
    """
    previous, value = 1.0, point
    for order in range(2, degree + 1):
        previous, value = value, (
            (2.0 * order - 1.0) * point * value - (order - 1.0) * previous
        ) / order
    derivative = degree * (point * value - previous) / (point * point - 1.0)
    return value, derivative


def _gauss_legendre(nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights on ``[-1, 1]``, by Newton on the Legendre polynomial.

    Computed rather than tabulated so the node count is an argument the caller
    can raise when a surface is steep, instead of a constant in this file.
    """
    if nodes < 2:
        raise BadConvexity(f"a Gauss rule of {nodes!r} nodes cannot integrate a curvature")
    points: list[float] = []
    weights: list[float] = []
    for index in range(1, nodes + 1):
        guess = math.cos(math.pi * (index - 0.25) / (nodes + 0.5))
        for _ in range(100):
            value, derivative = _legendre(nodes, guess)
            step = value / derivative
            guess -= step
            if abs(step) < 1e-15:
                break
        _, derivative = _legendre(nodes, guess)
        points.append(guess)
        weights.append(2.0 / ((1.0 - guess * guess) * derivative * derivative))
    return tuple(points), tuple(weights)


def _edges(
    lower: float,
    upper: float,
    forward: float,
    breaks: Sequence[float],
    panels: int,
    *,
    geometric: bool,
) -> list[float]:
    """Panel edges, with the forward and every kink *on* one.

    A Gauss rule integrates a polynomial exactly and a kink not at all: the
    payoff's second derivative is discontinuous at the strike and the forward
    is where the payer and receiver legs change over, so both have to be panel
    boundaries rather than points a rule straddles.

    The forward earns an anchor only when it is *inside* the range. Anchoring it
    unconditionally is how the tail estimate beyond the ceiling came to
    re-integrate the entire main range and report it as a truncation bound of
    1.3% of the price -- the one number in the result whose job was to say the
    truncation did not matter.

    **The panels are graded, and uniform ones make a wider ceiling worse rather
    than better.** Under a lognormal surface the ceiling is ``F exp(w sigma
    sqrt(T))``, so it moves out exponentially in ``w`` while the panel count
    stays put: every extra standard deviation coarsens the mesh *at the
    forward*, where the whole integrand lives. Measured on the second moment of
    a five-year rate at a 24% volatility, uniform panels gave a relative error
    of 1.4e-09, 3.2e-08, 7.9e-04 and 1.8e-01 as the ceiling went from six
    standard deviations to twenty -- monotonically worse, by eight orders of
    magnitude, for asking the quadrature to cover more of the tail. Spacing the
    edges evenly in ``log K`` instead makes a wider ceiling add panels out in
    the tail and leaves the resolution near the forward alone, which is the
    behaviour a ``width`` argument is supposed to have.
    """
    inside = [one for one in (forward, *breaks) if lower < one < upper]
    anchors = sorted({lower, upper, *inside})
    if upper - lower <= 0.0:
        return anchors
    grade = geometric and lower > 0.0
    scale: Callable[[float], float] = math.log if grade else (lambda one: one)
    back: Callable[[float], float] = math.exp if grade else (lambda one: one)
    span = scale(upper) - scale(lower)
    edges: list[float] = []
    for left, right in pairwise(anchors):
        measure = scale(right) - scale(left)
        share = max(1, round(panels * measure / span)) if span > 0.0 else 1
        step = measure / share
        edges.extend(back(scale(left) + step * which) for which in range(share))
    edges.append(anchors[-1])
    return edges


def _annuity_option(
    forward: float, strike: float, total: float, side: Strike, convention: Convention
) -> float:
    """The undiscounted annuity-measure value of one swaption, per unit annuity."""
    payoff = Payoff.PAYER if side is Strike.ABOVE else Payoff.RECEIVER
    if convention is Convention.NORMAL:
        return bachelier(forward, strike, total, payoff)
    if strike <= 0.0 or forward <= 0.0:
        raise BadConvexity(
            f"a lognormal volatility cannot price a strike of {strike!r} against a "
            f"forward of {forward!r}; a surface reaching to zero needs the normal "
            "convention"
        )
    return black(forward, strike, total, payoff)


def _integrate(
    payoff: Fixing,
    mapping: AnnuityMap,
    forward: float,
    lower: float,
    upper: float,
    smile: Smile,
    time: float,
    quadrature: Quadrature,
    panels: int,
) -> tuple[float, float]:
    """``(above, below)``: the payer and receiver integrals over ``[lower, upper]``."""
    points, weights = _gauss_legendre(quadrature.nodes)
    edges = _edges(
        lower, upper, forward, payoff.nodes, panels,
        geometric=quadrature.convention is Convention.LOGNORMAL,
    )
    above = 0.0
    below = 0.0
    for left, right in pairwise(edges):
        half = 0.5 * (right - left)
        middle = 0.5 * (right + left)
        if half <= 0.0:
            continue
        side = Strike.ABOVE if middle >= forward else Strike.BELOW
        total = 0.0
        for point, weight in zip(points, weights, strict=True):
            strike = middle + half * point
            curvature = payoff.weight(strike, mapping)
            if curvature == 0.0:
                continue
            volatility = smile(strike)
            if volatility < 0.0 or not math.isfinite(volatility):
                raise BadConvexity(
                    f"the surface returned a volatility of {volatility!r} at a "
                    f"strike of {strike!r}"
                )
            option = _annuity_option(
                forward, strike, volatility * math.sqrt(time), side,
                quadrature.convention,
            )
            total += weight * curvature * option
        if side is Strike.ABOVE:
            above += half * total
        else:
            below += half * total
    return above, below


# -- one CMS fixing -----------------------------------------------------------


@dataclass(frozen=True)
class Convexity:
    """A CMS rate, the forward it came from, and the gap between them."""

    #: ``E^Tp[S_T]``: what the contract pays, in expectation under the measure
    #: it settles in.
    rate: float
    #: ``S_0``: the forward swap rate, a martingale under the annuity and not
    #: under this one.
    forward: float
    #: The present value of the swaplet, per unit of notional.
    value: float
    #: The discount factor to the payment date.
    discount: float
    #: The replication the rate came out of.
    replication: Replication

    @property
    def adjustment(self) -> float:
        """``rate - forward``, in rate terms. Positive for an ordinary curve."""
        return self.rate - self.forward

    @property
    def basis_points(self) -> float:
        return 1e4 * self.adjustment


@dataclass(frozen=True)
class ConstantMaturity:
    """One CMS fixing: a swap rate observed at an expiry and paid on one date.

    ``payment`` is deliberately free of the underlying swap's own schedule.
    A CMS pays on its own leg's dates, which may be the fixing date itself (a
    rate set and paid at once, the largest adjustment per unit of tenor), the
    end of the fixing period, or anything a term sheet says. The payment date
    not being tied to the swap is the reason the adjustment exists at all.
    """

    expiry: date
    effective: date
    maturity: date
    payment: date
    index: ForecastIndex = field(default_factory=ForecastIndex)
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if self.payment < self.expiry:
            raise BadConvexity(
                f"fixing {self.name} pays on {self.payment.isoformat()}, before the "
                f"rate fixes on {self.expiry.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"the {self.effective.isoformat()}-{self.maturity.isoformat()} rate "
            f"fixing {self.expiry.isoformat()}"
        )

    def swaption(self, strike: float, side: Strike = Strike.ABOVE) -> Swaption:
        """The swaption this fixing replicates out of, struck at ``strike``."""
        return Swaption(
            expiry=self.expiry,
            effective=self.effective,
            maturity=self.maturity,
            strike=strike,
            payoff=Payoff.PAYER if side is Strike.ABOVE else Payoff.RECEIVER,
            index=self.index,
            frequency=self.frequency,
            basis=self.basis,
            calendar=self.calendar,
            rolling=self.rolling,
            label=f"{self.name} at {strike:.4%}",
        )

    def forward_rate(self, discount: DiscountCurve, projection: DiscountCurve) -> float:
        """The forward swap rate: the martingale under the annuity, not under the
        measure this contract settles in."""
        return self.swaption(0.0).forward_rate(discount, projection)

    def annuity(self, discount: DiscountCurve) -> float:
        return self.swaption(0.0).annuity(discount)

    def time_to_expiry(self, reference: date, basis: Basis = Basis.ACT_365F) -> float:
        return self.swaption(0.0).time_to_expiry(reference, basis)

    def mapping(self, discount: DiscountCurve, projection: DiscountCurve) -> AnnuityMap:
        """The annuity mapping function, scaled onto this curve at the forward."""
        return annuity_map(self, discount, projection)

    def replicate(
        self,
        payoff: Fixing,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float | Smile,
        *,
        quadrature: Quadrature = STANDARD,
        normalise: bool = True,
    ) -> Replication:
        """Statically replicate ``payoff`` out of this fixing's swaptions.

        The expansion is run twice: once on ``payoff`` and once on the constant
        payoff, whose value has to be ``P(0, Tp)`` because one unit paid on one
        date is a zero-coupon bond. It is not, and ``normalise`` divides the
        first by the second so that it is. That is the only no-arbitrage
        condition available here and it costs one extra quadrature; turning it
        off is for measuring what it was worth, not for pricing.

        The ceiling is reported back on the result together with a measured
        integral over one further doubling beyond it, because a replication
        with a truncated ceiling has an accuracy floor and a caller is entitled
        to see where it is rather than to be told the sum converged.
        """
        smile = _as_smile(volatility)
        time = self.time_to_expiry(discount.reference, quadrature.vol_basis)
        forward = self.forward_rate(discount, projection)
        mapping = self.mapping(discount, projection)
        annuity = self.annuity(discount)
        factor = discount.discount(self.payment)
        if factor <= 0.0:
            raise BadConvexity(
                f"fixing {self.name} pays on {self.payment.isoformat()}, where the "
                f"curve puts a discount factor of {factor!r}"
            )
        if smile(forward) * math.sqrt(time) <= 0.0:
            return _deterministic(payoff, forward, factor)
        lower, upper = self._range(forward, mapping, smile, time, quadrature)
        parts = self._expand(
            payoff, mapping, forward, annuity, lower, upper, smile, time, quadrature
        )
        unit = self._expand(
            RatePayoff(scale=0.0, offset=1.0), mapping, forward, annuity,
            lower, upper, smile, time, quadrature,
        )
        measure = unit[0] / factor
        if measure <= 0.0:
            raise BadConvexity(
                f"fixing {self.name} replicates one unit as {measure!r} of a "
                "zero-coupon bond, so nothing priced against it would mean anything"
            )
        scale = 1.0 / measure if normalise else 1.0
        value, intrinsic, mass, above, below = parts
        tail, _, _, _, _ = self._expand(
            payoff, mapping, forward, annuity, upper, upper + (upper - forward),
            smile, time, replace(quadrature, panels=max(1, quadrature.panels // 4)),
            intrinsic=False,
        )
        replication = Replication(
            value=scale * value,
            raw=value,
            measure=measure,
            intrinsic=scale * intrinsic,
            mass=scale * mass,
            above=scale * above,
            below=scale * below,
            lower=lower,
            upper=upper,
            tail=scale * tail,
            panels=quadrature.panels,
            nodes=quadrature.nodes,
        )
        if replication.truncation > quadrature.tolerance:
            raise BadConvexity(
                f"replicating {payoff.name} for {self.name} leaves "
                f"{replication.truncation:.3%} of the value beyond a ceiling of "
                f"{upper:.4%}, against a tolerance of {quadrature.tolerance:.1e}. "
                "The integral has not converged: a volatility that keeps rising "
                "with the strike stops the swaption price decaying, so the answer "
                "is whatever ceiling was chosen. Cap the surface's wings, or widen "
                "the tolerance and read Replication.truncation yourself."
            )
        return replication

    def _range(
        self,
        forward: float,
        mapping: AnnuityMap,
        smile: Smile,
        time: float,
        quadrature: Quadrature,
    ) -> tuple[float, float]:
        """The strikes to integrate between, in the convention's own units."""
        reference = smile(forward) * math.sqrt(time)
        width = quadrature.width
        if quadrature.convention is Convention.NORMAL:
            return (
                max(forward - width * reference, mapping.floor + 1e-9),
                forward + width * reference,
            )
        if forward <= 0.0:
            raise BadConvexity(
                f"fixing {self.name} has a forward swap rate of {forward!r}, which "
                "has no lognormal volatility; price it under the normal convention "
                "instead"
            )
        return (
            forward * math.exp(-width * reference),
            forward * math.exp(width * reference),
        )

    def _expand(
        self,
        payoff: Fixing,
        mapping: AnnuityMap,
        forward: float,
        annuity: float,
        lower: float,
        upper: float,
        smile: Smile,
        time: float,
        quadrature: Quadrature,
        *,
        intrinsic: bool = True,
    ) -> tuple[float, float, float, float, float]:
        """``(value, intrinsic, mass, above, below)`` for one Carr-Madan expansion."""
        above, below = _integrate(
            payoff, mapping, forward, lower, upper, smile, time,
            quadrature, quadrature.panels,
        )
        mass = 0.0
        for node in payoff.nodes:
            if not lower <= node <= upper:
                if not intrinsic:
                    continue
                raise BadConvexity(
                    f"payoff {payoff.name} kinks at {node!r}, outside the "
                    f"[{lower!r}, {upper!r}] the quadrature covers; widen it"
                )
            side = Strike.ABOVE if node >= forward else Strike.BELOW
            mass += payoff.mass(node, mapping) * _annuity_option(
                forward, node, smile(node) * math.sqrt(time), side,
                quadrature.convention,
            )
        flat = (
            annuity * payoff.value(forward) * mapping.value(forward)
            if intrinsic
            else 0.0
        )
        mass *= annuity
        above *= annuity
        below *= annuity
        return flat + mass + above + below, flat, mass, above, below

    def swaplet(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float | Smile,
        *,
        quadrature: Quadrature = STANDARD,
        normalise: bool = True,
    ) -> Convexity:
        """The CMS rate and the convexity adjustment that produced it."""
        replication = self.replicate(
            RatePayoff(), discount, projection, volatility,
            quadrature=quadrature, normalise=normalise,
        )
        factor = discount.discount(self.payment)
        return Convexity(
            rate=replication.value / factor,
            forward=self.forward_rate(discount, projection),
            value=replication.value,
            discount=factor,
            replication=replication,
        )

    def option(
        self,
        strike: float,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float | Smile,
        *,
        side: Strike = Strike.ABOVE,
        quadrature: Quadrature = STANDARD,
        normalise: bool = True,
    ) -> Replication:
        """A CMS caplet (``Strike.ABOVE``) or floorlet (``Strike.BELOW``)."""
        return self.replicate(
            OptionPayoff(strike=strike, side=side),
            discount,
            projection,
            volatility,
            quadrature=quadrature,
            normalise=normalise,
        )


def annuity_map(
    fixing: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> AnnuityMap:
    """Build the mapping for ``fixing`` and scale it onto ``discount``.

    The accruals and the offsets come from the underlying swap's own fixed
    schedule, measured from the fixing on the fixed leg's basis. The scale is
    then set so that the mapping returns the curve's own ``P(0, Tp) / A(0)`` at
    the forward, which is the no-arbitrage condition for the unit payoff and is
    not automatic — see the module docstring for what it is worth.
    """
    swaption = fixing.swaption(0.0)
    accruals: list[float] = []
    offsets: list[float] = []
    for period in swaption.swap.fixed_schedule():
        accruals.append(year_fraction(period.start, period.end, fixing.basis))
        offsets.append(year_fraction(fixing.expiry, period.payment, fixing.basis))
    raw = AnnuityMap(
        accruals=tuple(accruals),
        offsets=tuple(offsets),
        delay=year_fraction(fixing.expiry, fixing.payment, fixing.basis),
        frequency=int(fixing.frequency),
    )
    annuity = fixing.annuity(discount)
    if annuity <= 0.0:
        raise BadConvexity(
            f"fixing {fixing.name} has an annuity of {annuity!r}, so it has no "
            "forward swap rate to map"
        )
    target = discount.discount(fixing.payment) / annuity
    return raw.rescaled(target, fixing.forward_rate(discount, projection))


def measure_error(
    fixing: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    volatility: float | Smile,
    *,
    quadrature: Quadrature = STANDARD,
) -> float:
    """Relative error in replicating one unit paid on the payment date.

    The constant payoff is worth ``P(0, Tp)``: one unit at one date is a
    zero-coupon bond, and no model is needed to say so. It goes through the
    same mapping, the same expansion and the same quadrature as a real payoff
    and is sensitive to all three, so this is the diagnostic to read before
    believing a convexity adjustment.
    """
    return fixing.replicate(
        RatePayoff(scale=0.0, offset=1.0),
        discount,
        projection,
        volatility,
        quadrature=quadrature,
        normalise=False,
    ).arbitrage


# -- a strip of them ----------------------------------------------------------


@dataclass(frozen=True)
class ConstantMaturityLeg:
    """A leg paying a swap rate of fixed tenor, period after period.

    Each period's rate is a fresh fixing: observed at the period's own start,
    on a swap of ``tenor_years`` running from there, and paid on the period's
    payment date. So the leg is a strip of :class:`ConstantMaturity` fixings
    with a common tenor and nothing else shared, which is exactly what a CMS
    leg is and why a single long swaption cannot stand in for one.
    """

    effective: date
    maturity: date
    tenor_years: int
    index: ForecastIndex = field(default_factory=ForecastIndex)
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadConvexity(
                f"leg {self.name} matures on {self.maturity.isoformat()}, which is "
                f"not after it starts on {self.effective.isoformat()}"
            )
        if self.tenor_years <= 0:
            raise BadConvexity(
                f"leg {self.name} pays a {self.tenor_years!r}-year swap rate, which "
                "is not a tenor"
            )

    @property
    def name(self) -> str:
        return self.label or f"{self.tenor_years}y CMS to {self.maturity.isoformat()}"

    def schedule(self) -> tuple[Period, ...]:
        return tuple(
            generate(
                self.effective,
                self.maturity,
                self.index.tenor,
                calendar=self.index.business_days,
                rolling=self.index.rolling,
                payment_lag=self.index.payment_lag,
            )
        )

    def fixings(self, reference: date) -> tuple[ConstantMaturity, ...]:
        """Every period still carrying optionality, as its own fixing.

        A period whose rate has already fixed by ``reference`` is dropped, for
        the reason :class:`~tenor.options.Cap` drops its first one: there is no
        expectation left to take, and replicating one would be integrating
        swaptions that have expired.
        """
        made: list[ConstantMaturity] = []
        for period in self.schedule():
            if period.adjusted_start <= reference:
                continue
            start = period.adjusted_start
            made.append(
                ConstantMaturity(
                    expiry=start,
                    effective=start,
                    maturity=add_months(
                        start, 12 * self.tenor_years, keep_end_of_month=True
                    ),
                    payment=period.payment,
                    index=self.index,
                    frequency=self.frequency,
                    basis=self.basis,
                    calendar=self.calendar,
                    rolling=self.rolling,
                    label=f"{self.name} fixing {start.isoformat()}",
                )
            )
        if not made:
            raise BadConvexity(
                f"leg {self.name} has no periods left whose rate has not already "
                f"fixed by {reference.isoformat()}"
            )
        return tuple(made)

    def accrual(self, period: Period) -> float:
        """One period's year fraction, on the paying leg's own basis.

        The CMS tenor and the payment frequency are independent — a ten-year
        rate paid quarterly is the ordinary structure — so this is the *leg's*
        accrual and has nothing to do with the underlying swap's.
        """
        return year_fraction(period.adjusted_start, period.adjusted_end, self.index.basis)

    def value(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float | Smile,
        *,
        strike: float | None = None,
        side: Strike = Strike.ABOVE,
        quadrature: Quadrature = STANDARD,
    ) -> float:
        """The leg's present value per unit of notional.

        ``strike`` left out values the CMS leg itself, every period paying its
        own adjusted rate. Given, it values the cap (``Strike.ABOVE``) or floor
        (``Strike.BELOW``) on the same strip.
        """
        periods = {period.adjusted_start: period for period in self.schedule()}
        total = 0.0
        for fixing in self.fixings(discount.reference):
            accrual = self.accrual(periods[fixing.expiry])
            if strike is None:
                total += accrual * fixing.swaplet(
                    discount, projection, volatility, quadrature=quadrature
                ).value
            else:
                total += accrual * fixing.option(
                    strike, discount, projection, volatility,
                    side=side, quadrature=quadrature,
                ).value
        return total
