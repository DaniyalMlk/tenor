"""Hull-White's one-factor model: one set of parameters, analytic prices.

:mod:`tenor.options` prices a swaption on the annuity measure, which means
Black or Bachelier on a forward swap rate with the volatility supplied from
outside. :mod:`tenor.lattice` prices a bond whose flows are not known on a
short rate tree, which is calibrated to the curve but has no closed form and
says nothing about a swaption. Neither is a *model* in the sense of a single
specification that prices a swaption, a cap and a callable bond consistently.
This is.

The short rate is Gaussian with a drift fitted to the initial curve::

    dr = (theta(t) - a r) dt + sigma dW

and ``theta`` never has to be written down, because the whole of the initial
curve enters through the bond price

    P(t, T) = A(t, T) exp(-B(t, T) r(t)),   B(t, T) = (1 - e**(-a(T-t))) / a

    A(t, T) = (P(0,T) / P(0,t)) exp(B(t,T) f(0,t) - sigma**2 (1 - e**(-2at)) B**2 / (4a))

with ``f(0,t)`` the instantaneous forward the curve already supplies.
:func:`reprices_curve` checks the consequence: at ``t = 0`` the formula must
return the curve's own discount factors, and measured across a seventeen-pillar
curve it does so with a worst relative error of **exactly 0.0**, at every mean
reversion from 0.01 to 0.5 and every volatility from zero to 2%, because the
identity is in ``A`` rather than in a fit.

**Put-call parity on a bond option is an identity, so it is asserted.** A call
and a put on ``P(T,S)`` struck at ``K`` differ by the forward bond price less
the strike, discounted: ``P(0,S) - K P(0,T)``. That holds in the model's
algebra with nothing approximate about it, and it is the cheapest test that
the two branches of :func:`zero_bond_option` share the same ``A``, ``B`` and
volatility.

**Jamshidian's decomposition, and something to check it against.** An option
on a coupon bond is not a sum of options on its flows, because the exercise
decision is taken on the whole bond. In a one-factor model it becomes one:
bond prices are monotone in the short rate, so there is a single rate ``r*`` at
which the bond is worth the strike, and above it every flow's discounted value
is below its own value at ``r*``. So the option is a portfolio of zero-coupon
bond options struck at ``P(T, t_i; r*)``, exactly.

Exactly, but only where the bond price is monotone, which needs the option to
expire when the swap starts. A mid-curve swaption — expiring well before its
underlying begins — carries a ``-P(T, effective)`` term with the *smallest*
exponent in the sum, and monotonicity there is a claim about coefficients
rather than a fact. :class:`Method` therefore offers a second route that needs
no monotonicity at all: quadrature over the terminal short rate under the
forward measure, evaluating the bond's actual price at each node rather than
any closed form. Where both apply they agree to **3.1e-14** relative over
eighteen combinations of expiry, coupon and side, and to 1.3e-15 on a
ten-year-into-ten-year swaption — which is what makes the decomposition
believable rather than merely cited.

**The obvious quadrature does not work, and that is worth stating.** A Gaussian
weight invites a Gauss-Hermite rule, and a sixty-four point one gets the
near-the-money prices to five digits and a deep out-of-the-money one wrong by
**1.25e-01 relative**. A Gauss rule converges at that rate on a kink, and an
option payoff is nothing but a kink; the two routes agreed on the *difference*
between a payer and a receiver to 1e-12 while both were wrong, because the
error is in the straddle and cancels in parity. Putting the exercise boundary
at a panel edge of a composite rule turns five digits into fifteen.

**The forward-measure mean is implied, not remembered, and comes out exact.** Pricing by quadrature
needs the law of ``r(T)`` under the ``T``-forward measure. Its variance is
``sigma**2 (1 - e**(-2aT)) / (2a)``, which is the model's own stationary
expression, and its mean follows from the fact that every forward bond price
is a martingale under that measure: ``A(T,S) exp(-B m + B**2 v / 2)`` has to
equal ``P(0,S)/P(0,T)`` for *every* ``S``, which determines ``m`` and
over-determines it. :func:`forward_measure` solves for ``m`` from each probe
maturity separately and reports the spread; across probes from one to ten years
it is **0.0** at every expiry tested, and the reason is that the terms cancel
identically: the ``B**2 v / 2`` from the lognormal expectation is exactly the
convexity term inside ``A``, leaving ``m = f(0, T)``. The forward-measure
expectation of the future short rate *is* the instantaneous forward, with
nothing left over. A drift adjustment copied from a reference would have been a
third of a line and would not have shown that.

**Mean reversion and volatility are not separately identified by one quote,
and the trade-off is steep.** A ten-year-into-ten-year payer struck at its
forward is repriced exactly at every mean reversion from 0.01 to 0.30 — the
refitted volatility runs from 0.492% to 3.226%, a factor of **6.55** for a
factor of 30 in ``a``, and the refit error at each point is at most 2.2e-16.
There is no residual to choose between them. What does separate them is the
*term structure*: the pair fitted to the ten-year expiry prices a
one-year-into-ten-year anywhere from **23.4% below to 57.2% above** the
``a = 0.08`` answer across that same range, so two expiries pin the pair where
one cannot. That is why :func:`calibrate` takes a mean reversion as an argument
rather than fitting it: a function that fitted both to one quote would return
whichever of a one-parameter family its initial guess fell nearest.

**And the model is Gaussian, so the normal volatility it generates is nearly
flat.** Reading the model's own swaption prices back through
:func:`tenor.options.implied_volatility`, the normal volatility across strikes
from 60% to 140% of the forward swap rate moves by 1.06 basis points on a level
of 45.55 — a smile **2.33%** wide, and monotone rather than smiling. The
lognormal convention on the *same prices* moves by 41.91% of its own level over
the same strikes, eighteen times as much. A Gaussian short rate makes a nearly
Gaussian swap rate, which is the sense in which the normal convention is this
model's native one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from itertools import pairwise

from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .options import Payoff, Swaption
from .solve import solve

__all__ = [
    "BadModel",
    "Calibration",
    "CouponBondOption",
    "Flow",
    "ForwardMeasure",
    "HullWhite",
    "Method",
    "Quote",
    "Trade",
    "calibrate",
    "coupon_bond_option",
    "forward_measure",
    "reprices_curve",
    "swaption_price",
    "zero_bond_option",
]

#: Standard deviations either side of the forward-measure mean that the
#: quadrature route covers. Ten is past where a Gaussian density times a bond
#: price contributes anything at double precision.
_QUADRATURE_WIDTH = 10.0

#: Standard deviations of the short rate to search over when inverting the
#: coupon bond price for Jamshidian's reference rate. Twenty is far outside any
#: rate the model gives positive probability and still a finite bracket.
_SEARCH_WIDTH = 20.0


class BadModel(ValueError):
    """Parameters that do not describe a Hull-White process."""


class Method(str, Enum):
    """How to price an option on a coupon bond."""

    JAMSHIDIAN = "jamshidian"
    """Decompose into zero-coupon bond options. Exact, needs monotonicity."""

    QUADRATURE = "quadrature"
    """Integrate the payoff over the terminal short rate. Needs nothing."""


@dataclass(frozen=True, slots=True)
class HullWhite:
    """The two parameters of the short rate process.

    Attributes:
        a: Speed of mean reversion, in reciprocal years. Positive; the
            ``a -> 0`` limit is Ho-Lee and is a different model with a
            different ``B``, so it is refused rather than approximated.
        sigma: Absolute volatility of the short rate, in rate units per root
            year. Non-negative; zero is the deterministic case and every
            formula here reduces to intrinsic value there.
    """

    a: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.a) or self.a <= 0.0:
            raise BadModel(
                f"mean reversion must be a positive finite number, got {self.a!r}. "
                "The zero-reversion limit is Ho-Lee, which has a different B(t, T)."
            )
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise BadModel(
                f"volatility must be a non-negative finite number, got {self.sigma!r}"
            )

    def b(self, tenor: float) -> float:
        """``B(t, T)`` for a gap of ``tenor`` years, which depends only on the gap."""
        if tenor < 0.0:
            raise BadModel(f"a bond tenor cannot be negative, got {tenor!r}")
        return -math.expm1(-self.a * tenor) / self.a

    def short_rate_variance(self, time: float) -> float:
        """``Var(r(t))``, which is also the variance under any forward measure."""
        if time < 0.0:
            raise BadModel(f"time cannot be negative, got {time!r}")
        return -self.sigma**2 * math.expm1(-2.0 * self.a * time) / (2.0 * self.a)

    def bond_volatility(self, expiry: float, maturity: float) -> float:
        """Total volatility of ``log P(T, S)``: ``B(T,S) sqrt(Var(r(T)))``.

        This is the only place the model's randomness enters a bond option, and
        it is the product of a term-structure factor and a rate factor. The
        first is why a long bond is more volatile than a short one and the
        second is why a distant expiry is more volatile than a near one.
        """
        if maturity < expiry:
            raise BadModel(
                f"a bond maturing at {maturity!r} cannot underlie an option expiring "
                f"at {expiry!r}"
            )
        return self.b(maturity - expiry) * math.sqrt(self.short_rate_variance(expiry))

    def log_a(self, curve: DiscountCurve, expiry: float, maturity: float) -> float:
        """``log A(T, S)``, the part of the bond price the initial curve sets.

        Returned as a logarithm because that is how it is used — the bond price
        is an exponential and the two exponents should be added once rather
        than multiplied after separate exponentiation.
        """
        tenor = self.b(maturity - expiry)
        forward = curve.instantaneous_forward(expiry)
        decay = -math.expm1(-2.0 * self.a * expiry)
        return (
            math.log(curve.discount_at(maturity) / curve.discount_at(expiry))
            + tenor * forward
            - self.sigma**2 * decay * tenor * tenor / (4.0 * self.a)
        )

    def bond_price(
        self, curve: DiscountCurve, expiry: float, maturity: float, short_rate: float
    ) -> float:
        """``P(T, S)`` given the short rate at ``T``."""
        return math.exp(
            self.log_a(curve, expiry, maturity) - self.b(maturity - expiry) * short_rate
        )


def reprices_curve(curve: DiscountCurve, model: HullWhite) -> float:
    """The worst relative error in repricing the curve's own pillars at ``t = 0``.

    At ``t = 0`` the model's bond price must be the curve's discount factor,
    because ``A(0, S)`` reduces to ``P(0, S) e**(B(0,S) r(0))`` and ``r(0)`` is
    the curve's own instantaneous forward at zero. Nothing is being fitted here
    — the identity is built into ``A`` — so a failure means the instantaneous
    forward or the interpolation is not what the formula assumes.

    Args:
        curve: The curve the model was given.
        model: The model.

    Returns:
        The largest relative error across the curve's pillars.
    """
    short_rate = curve.instantaneous_forward(0.0)
    worst = 0.0
    for time in curve.times:
        if time <= 0.0:
            continue
        expected = curve.discount_at(time)
        got = model.bond_price(curve, 0.0, time, short_rate)
        worst = max(worst, abs(got / expected - 1.0))
    return worst


def zero_bond_option(
    curve: DiscountCurve,
    model: HullWhite,
    expiry: float,
    maturity: float,
    strike: float,
    payoff: Payoff = Payoff.PAYER,
) -> float:
    """An option on a zero-coupon bond price, in closed form.

    The forward bond price ``P(0,S)/P(0,T)`` is a martingale under the
    ``T``-forward measure and lognormal in this model, so this is Black's
    formula on that forward with the model's own bond volatility, discounted at
    ``P(0,T)``.

    Args:
        curve: The initial curve.
        model: The model.
        expiry: Option expiry, in years.
        maturity: Bond maturity, in years. Not before ``expiry``.
        strike: Strike *bond price*, in [0, 1] for an ordinary case though
            nothing here requires it. Positive.
        payoff: :attr:`~tenor.options.Payoff.PAYER` for a put on the bond,
            which is what a payer swaption decomposes into, and
            :attr:`~tenor.options.Payoff.RECEIVER` for a call.

    Returns:
        The present value, per unit of notional.

    Raises:
        BadModel: If the strike is not positive or the dates are out of order.
    """
    if not math.isfinite(strike) or strike <= 0.0:
        raise BadModel(f"a bond option strike must be positive, got {strike!r}")
    if expiry < 0.0:
        raise BadModel(f"an expiry cannot be negative, got {expiry!r}")
    discount_expiry = curve.discount_at(expiry)
    discount_maturity = curve.discount_at(maturity)
    total_vol = model.bond_volatility(expiry, maturity)
    # A receiver swaption is long the fixed leg, so it decomposes into calls on
    # the bond; a payer is short it and decomposes into puts. The sign below is
    # the bond option's own, which is the opposite of the swaption's.
    call = payoff is Payoff.RECEIVER
    if total_vol == 0.0:
        intrinsic = discount_maturity - strike * discount_expiry
        return max(intrinsic, 0.0) if call else max(-intrinsic, 0.0)
    first = (
        math.log(discount_maturity / (discount_expiry * strike)) / total_vol
        + 0.5 * total_vol
    )
    second = first - total_vol
    if call:
        return discount_maturity * _normal_cdf(first) - strike * discount_expiry * (
            _normal_cdf(second)
        )
    return strike * discount_expiry * _normal_cdf(-second) - discount_maturity * (
        _normal_cdf(-first)
    )


@dataclass(frozen=True, slots=True)
class Flow:
    """One cash flow of the bond underlying an option.

    Attributes:
        time: When it pays, in years from the valuation date.
        amount: What it pays, per unit of notional.
    """

    time: float
    amount: float


@dataclass(frozen=True, slots=True)
class ForwardMeasure:
    """The law of the short rate at an expiry, under that expiry's forward measure.

    Attributes:
        mean: ``E^T[r(T)]``, implied from the martingale property of forward
            bond prices rather than from a drift adjustment.
        variance: ``Var(r(T))``, which the change of measure leaves alone
            because the volatility is deterministic.
        spread: The largest disagreement between the means implied by
            different probe maturities. A diagnostic on ``A`` and ``B``: the
            model over-determines the mean and all the answers must coincide.
        probes: The maturities used.
    """

    mean: float
    variance: float
    spread: float
    probes: tuple[float, ...]

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)


def forward_measure(
    curve: DiscountCurve,
    model: HullWhite,
    expiry: float,
    probes: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0),
) -> ForwardMeasure:
    """The terminal short rate's law under the forward measure to ``expiry``.

    Every forward bond price ``P(T,S)/P(T,T)`` is a martingale under that
    measure, so for each ``S``

        A(T,S) exp(-B(T,S) m + B(T,S)**2 v / 2) = P(0,S) / P(0,T)

    which can be solved for ``m``. One probe suffices; several are used because
    they must all agree, and the amount by which they do not is a direct read
    on whether ``A`` and ``B`` are consistent with each other.

    Args:
        curve: The initial curve.
        model: The model.
        expiry: The measure's maturity, in years. Positive.
        probes: Bond maturities beyond ``expiry`` to imply the mean from, as
            offsets added to ``expiry``.

    Returns:
        A :class:`ForwardMeasure`.

    Raises:
        BadModel: If the expiry is not positive or no probe is usable.
    """
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise BadModel(f"a forward measure needs a positive expiry, got {expiry!r}")
    variance = model.short_rate_variance(expiry)
    means: list[float] = []
    used: list[float] = []
    for offset in probes:
        if offset <= 0.0:
            raise BadModel(f"a probe offset must be positive, got {offset!r}")
        maturity = expiry + offset
        tenor = model.b(offset)
        target = math.log(curve.discount_at(maturity) / curve.discount_at(expiry))
        # log A - B m + B^2 v / 2 = target
        mean = (
            model.log_a(curve, expiry, maturity) + 0.5 * tenor * tenor * variance - target
        ) / tenor
        means.append(mean)
        used.append(maturity)
    if not means:  # pragma: no cover - probes is non-empty by validation above
        raise BadModel("a forward measure needs at least one probe maturity")
    return ForwardMeasure(
        mean=sum(means) / len(means),
        variance=variance,
        spread=max(means) - min(means),
        probes=tuple(used),
    )


@dataclass(frozen=True, slots=True)
class CouponBondOption:
    """An option on a bond whose flows are known, and the two ways to price it.

    Attributes:
        expiry: Option expiry, in years.
        flows: The bond's cash flows after the expiry, per unit of notional.
        strike: The bond price the option is struck at.
        payoff: :attr:`~tenor.options.Payoff.PAYER` for a put on the bond,
            :attr:`~tenor.options.Payoff.RECEIVER` for a call, matching the
            swaption whose decomposition this is.
    """

    expiry: float
    flows: tuple[Flow, ...]
    strike: float
    payoff: Payoff = Payoff.PAYER

    def __post_init__(self) -> None:
        if not math.isfinite(self.expiry) or self.expiry <= 0.0:
            raise BadModel(f"an option expiry must be positive, got {self.expiry!r}")
        if not self.flows:
            raise BadModel("a coupon bond option needs at least one cash flow")
        if not math.isfinite(self.strike) or self.strike < 0.0:
            raise BadModel(
                f"a bond option strike must be non-negative, got {self.strike!r}. "
                "Zero is the mid-curve case, where the notional legs carry the "
                "exercise condition instead of a bond price."
            )
        previous = self.expiry
        for flow in self.flows:
            if flow.time <= self.expiry:
                raise BadModel(
                    f"a cash flow at {flow.time!r} does not survive an expiry at "
                    f"{self.expiry!r}; only flows after the expiry are optional"
                )
            if flow.time < previous:
                raise BadModel("cash flows must be in time order")
            previous = flow.time
            if not math.isfinite(flow.amount):
                raise BadModel(f"a cash flow amount must be finite, got {flow.amount!r}")

    def value_at(self, curve: DiscountCurve, model: HullWhite, short_rate: float) -> float:
        """The bond's price at the expiry, given the short rate there."""
        return sum(
            flow.amount * model.bond_price(curve, self.expiry, flow.time, short_rate)
            for flow in self.flows
        )

    @property
    def is_monotone(self) -> bool:
        """True when every flow is positive, so the bond price falls with the rate.

        The condition Jamshidian's decomposition needs. A bond with a negative
        flow in it — which is how a mid-curve swaption arrives — may still be
        monotone, but it is no longer monotone *by inspection*, and the
        decomposition rests on the inspection.
        """
        return all(flow.amount > 0.0 for flow in self.flows)


def _legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]``, from the recurrence.

    Newton's method on the Legendre polynomial, with the derivative from the
    same three-term recurrence, so nothing here is transcribed. Only the
    positive half is solved for; the rule is symmetric.
    """
    found: list[tuple[float, float]] = []
    for index in range((order + 1) // 2):
        guess = math.cos(math.pi * (index + 0.75) / (order + 0.5))
        for _ in range(100):
            value, derivative = _legendre_at(order, guess)
            step = value / derivative
            guess -= step
            if abs(step) <= 1e-16:
                break
        _, derivative = _legendre_at(order, guess)
        weight = 2.0 / ((1.0 - guess * guess) * derivative * derivative)
        if abs(guess) < 1e-15:
            guess = 0.0
        found.append((guess, weight))
    pairs: list[tuple[float, float]] = []
    for node, weight in found:
        pairs.append((node, weight))
        if node != 0.0:
            pairs.append((-node, weight))
    pairs.sort()
    return tuple(node for node, _ in pairs), tuple(weight for _, weight in pairs)


def _legendre_at(order: int, x: float) -> tuple[float, float]:
    """``P_n(x)`` and ``P_n'(x)``, from ``n P_n = (2n-1) x P_{n-1} - (n-1) P_{n-2}``."""
    behind = 0.0
    here = 1.0
    for degree in range(1, order + 1):
        behind, here = here, (
            (2.0 * degree - 1.0) * x * here - (degree - 1.0) * behind
        ) / degree
    return here, order * (x * here - behind) / (x * x - 1.0)


#: Nodes and weights of the panel rule, built once. Sixteen points per panel
#: and thirty-two panels resolves a bond price against a Gaussian density to
#: the floor of double precision once the payoff's kink is a panel edge.
_PANEL_NODES, _PANEL_WEIGHTS = _legendre(16)
_PANELS = 32


def _exercise_boundary(
    curve: DiscountCurve, model: HullWhite, option: CouponBondOption, centre: float
) -> float:
    """The short rate at which the bond is worth the strike.

    The one quantity both routes share. For the decomposition it is what sets
    every zero-coupon strike; for the quadrature it is only a panel edge, and
    putting it there is the difference between machine precision and five
    digits.
    """
    deviation = math.sqrt(model.short_rate_variance(option.expiry))
    width = max(_SEARCH_WIDTH * deviation, 1.0)
    return solve(
        lambda rate: option.value_at(curve, model, rate) - option.strike,
        low=centre - width,
        high=centre + width,
    ).value


def coupon_bond_option(
    curve: DiscountCurve,
    model: HullWhite,
    option: CouponBondOption,
    method: Method = Method.JAMSHIDIAN,
) -> float:
    """Price an option on a coupon bond.

    Args:
        curve: The initial curve.
        model: The model.
        option: The option.
        method: :attr:`Method.JAMSHIDIAN` for the exact decomposition, which
            needs every flow to be positive, or :attr:`Method.QUADRATURE`
            for the integral, which needs nothing.

    Returns:
        The present value, per unit of notional.

    Raises:
        BadModel: If the decomposition is asked for on a bond whose price is
            not monotone in the short rate by inspection.
    """
    if method is Method.QUADRATURE:
        return _by_quadrature(curve, model, option)
    if not option.is_monotone:
        raise BadModel(
            "Jamshidian's decomposition needs every cash flow to be positive, so "
            "that the bond price falls with the short rate; this bond has a "
            f"negative flow. Price it with {Method.QUADRATURE.value} instead."
        )
    return _by_decomposition(curve, model, option)


def _by_decomposition(
    curve: DiscountCurve, model: HullWhite, option: CouponBondOption
) -> float:
    centre = forward_measure(
        curve, model, option.expiry, (option.flows[-1].time - option.expiry,)
    ).mean
    reference = _exercise_boundary(curve, model, option, centre)
    total = 0.0
    for flow in option.flows:
        strike = model.bond_price(curve, option.expiry, flow.time, reference)
        total += flow.amount * zero_bond_option(
            curve, model, option.expiry, flow.time, strike, option.payoff
        )
    return total


def _by_quadrature(
    curve: DiscountCurve, model: HullWhite, option: CouponBondOption
) -> float:
    """Integrate the payoff against the terminal short rate's own density.

    Independent of :func:`zero_bond_option` and of the decomposition: it
    evaluates the bond's actual price at each node and the payoff there, and
    weights by the Gaussian density. The only thing it shares with the
    decomposition is the exercise boundary, which it uses as a panel edge
    rather than as a strike.

    **The obvious rule does not work.** A Gaussian weight invites a
    Gauss-Hermite rule, and one of sixty-four points gets the near-the-money
    prices to five digits and a deep out-of-the-money one wrong by 1.25e-01
    relative. A Gauss rule converges that slowly on a kink, and an option
    payoff is nothing but a kink. Splitting the range at the kink and using a
    composite rule on each side turns five digits into fifteen.
    """
    measure = forward_measure(
        curve, model, option.expiry, (option.flows[-1].time - option.expiry,)
    )
    deviation = measure.standard_deviation
    if deviation == 0.0:
        intrinsic = option.value_at(curve, model, measure.mean) - option.strike
        sign = -1.0 if option.payoff is Payoff.PAYER else 1.0
        return curve.discount_at(option.expiry) * max(sign * intrinsic, 0.0)
    low = measure.mean - _QUADRATURE_WIDTH * deviation
    high = measure.mean + _QUADRATURE_WIDTH * deviation
    edges = [low + (high - low) * index / _PANELS for index in range(_PANELS + 1)]
    if option.is_monotone:
        boundary = _exercise_boundary(curve, model, option, measure.mean)
        if low < boundary < high:
            edges.append(boundary)
            edges.sort()
    sign = -1.0 if option.payoff is Payoff.PAYER else 1.0
    scale = 1.0 / (deviation * math.sqrt(2.0 * math.pi))
    total = 0.0
    for left, right in pairwise(edges):
        half = 0.5 * (right - left)
        middle = 0.5 * (right + left)
        if half == 0.0:
            continue
        for node, weight in zip(_PANEL_NODES, _PANEL_WEIGHTS, strict=True):
            rate = middle + half * node
            value = option.value_at(curve, model, rate)
            payoff = max(sign * (value - option.strike), 0.0)
            if payoff == 0.0:
                continue
            standard = (rate - measure.mean) / deviation
            total += weight * half * payoff * scale * math.exp(-0.5 * standard * standard)
    return curve.discount_at(option.expiry) * total


@dataclass(frozen=True, slots=True)
class Trade:
    """A swaption as this model sees it, and the price both routes give.

    Attributes:
        option: The coupon bond option the swaption is equivalent to.
        jamshidian: The decomposition's price.
        quadrature: The integral's price, or ``None`` where the bond is not
            monotone and only the integral was run.
        reference_rate: The short rate at which the bond is worth the strike,
            which is the decomposition's one solved quantity.
    """

    option: CouponBondOption
    jamshidian: float | None
    quadrature: float
    reference_rate: float | None = None

    @property
    def price(self) -> float:
        """The decomposition where it applies, the integral otherwise."""
        return self.quadrature if self.jamshidian is None else self.jamshidian

    @property
    def gap(self) -> float:
        """Relative disagreement between the two routes, or 0.0 with only one."""
        if self.jamshidian is None or self.quadrature == 0.0:
            return 0.0
        return self.jamshidian / self.quadrature - 1.0


def swaption_price(
    curve: DiscountCurve,
    model: HullWhite,
    option: Swaption,
    reference: date,
    basis: Basis = Basis.ACT_365F,
) -> Trade:
    """Price a swaption under the model, by both routes where both apply.

    The swap is read off the option's own schedule, so the frequency, basis,
    calendar and rolling rule are the ones the option carries. A receiver
    swaption is a call on the coupon bond struck at par and a payer is a put,
    which is where the two sign conventions meet.

    Only the single-curve case is modelled: one short rate discounts and
    forecasts, so the floating leg is worth par at the expiry when the swap
    starts there. A mid-curve swaption still prices, by quadrature, because the
    notional exchange at the swap's start enters as a negative flow.

    Args:
        curve: The initial curve, used for discounting and for the drift fit.
        model: The model.
        option: The swaption.
        reference: Valuation date. Times are measured from here.
        basis: Day count for converting dates to times.

    Returns:
        A :class:`Trade` carrying both prices.

    Raises:
        BadModel: If the expiry is not after the reference date.
    """
    expiry = year_fraction(reference, option.expiry, basis)
    if expiry <= 0.0:
        raise BadModel(
            f"swaption {option.name} expires on {option.expiry.isoformat()}, on or "
            f"before the valuation date {reference.isoformat()}"
        )
    schedule = option.swap.fixed_schedule()
    flows: list[Flow] = []
    if option.effective > option.expiry:
        # Mid-curve: the floating leg is only worth par at the swap's start, so
        # the notional is exchanged there and comes back at maturity.
        flows.append(
            Flow(year_fraction(reference, option.effective, basis), -1.0)
        )
    for index, period in enumerate(schedule):
        accrual = year_fraction(period.start, period.end, option.basis)
        amount = option.strike * accrual
        if index == len(schedule) - 1:
            amount += 1.0
        flows.append(Flow(year_fraction(reference, period.payment, basis), amount))
    flows.sort(key=lambda flow: flow.time)
    # A mid-curve swaption's exercise condition is on the swap's whole value,
    # so the notional exchange at the swap's start is one of the flows and the
    # strike is zero. Where the swap starts at the expiry that exchange is
    # worth par with certainty and moves to the strike instead.
    strike = 1.0 if option.effective <= option.expiry else 0.0
    bond = CouponBondOption(
        expiry=expiry, flows=tuple(flows), strike=strike, payoff=option.payoff
    )
    quadrature = _by_quadrature(curve, model, bond)
    if not bond.is_monotone:
        return Trade(option=bond, jamshidian=None, quadrature=quadrature)
    centre = forward_measure(curve, model, expiry, (flows[-1].time - expiry,)).mean
    reference_rate = _exercise_boundary(curve, model, bond, centre)
    return Trade(
        option=bond,
        jamshidian=_by_decomposition(curve, model, bond),
        quadrature=quadrature,
        reference_rate=reference_rate,
    )


def _normal_cdf(x: float) -> float:
    """The standard normal distribution function, through ``erfc``."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@dataclass(frozen=True, slots=True)
class Quote:
    """A swaption and the premium the market pays for it.

    Attributes:
        option: The swaption.
        premium: Its price, per unit of notional, on the same scale the model
            returns.
    """

    option: Swaption
    premium: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.premium) or self.premium < 0.0:
            raise BadModel(
                f"a swaption premium must be a non-negative finite number, got "
                f"{self.premium!r}"
            )


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted volatility and how well it fits.

    Attributes:
        model: The fitted model, carrying the mean reversion it was given.
        errors: One relative pricing error per quote, in the order given.
        worst: The largest absolute relative error.
        iterations: Root-finding steps the fit took.
    """

    model: HullWhite
    errors: tuple[float, ...]
    worst: float
    iterations: int


def calibrate(
    curve: DiscountCurve,
    quotes: Sequence[Quote],
    reference: date,
    *,
    mean_reversion: float,
    basis: Basis = Basis.ACT_365F,
    low: float = 1e-5,
    high: float = 0.2,
) -> Calibration:
    """Fit the volatility to a strip of swaption premiums, at a given reversion.

    **Mean reversion is an argument, not a parameter.** One quote cannot
    separate the two: the module docstring records a volatility running from
    0.492% to 3.226% across mean reversions from 0.01 to 0.30, every one of
    them repricing the quote to 2.2e-16. A routine that fitted both would
    return a point on that ridge chosen by its initial guess and would report a
    residual of zero either way, which is worse than useless.

    Every swaption price in this model is strictly increasing in the volatility,
    so the *sum* of the pricing errors is too, and the fit is a root rather
    than a minimisation: the volatility at which the strip's total premium is
    matched. With one quote that reprices it exactly; with several it is a
    stated compromise rather than an opaque one, and :attr:`Calibration.errors`
    shows where the compromise fell.

    Args:
        curve: The initial curve.
        quotes: The swaptions and their premiums. Non-empty.
        reference: Valuation date.
        mean_reversion: The ``a`` to hold fixed. Positive.
        basis: Day count for converting dates to times.
        low: Lower end of the volatility bracket.
        high: Upper end.

    Returns:
        A :class:`Calibration`.

    Raises:
        BadModel: If there are no quotes or the bracket cannot be widened to
            contain the answer.
    """
    if not quotes:
        raise BadModel("a calibration needs at least one quote")

    def total_error(sigma: float) -> float:
        model = HullWhite(a=mean_reversion, sigma=sigma)
        return sum(
            swaption_price(curve, model, quote.option, reference, basis).price
            - quote.premium
            for quote in quotes
        )

    root = solve(total_error, low=low, high=high)
    model = HullWhite(a=mean_reversion, sigma=root.value)
    errors = tuple(
        swaption_price(curve, model, quote.option, reference, basis).price
        / quote.premium
        - 1.0
        if quote.premium > 0.0
        else 0.0
        for quote in quotes
    )
    return Calibration(
        model=model,
        errors=errors,
        worst=max(abs(error) for error in errors),
        iterations=root.iterations,
    )
