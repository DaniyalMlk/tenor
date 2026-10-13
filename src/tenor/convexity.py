"""Rate futures: the gap between a margined contract and the forward.

A deposit, a forward rate agreement and a swap are all discounted. A future is
not. It is marked to market every day and the variation margin is paid in cash,
so the holder of a long position earns on the margin when rates fall and funds
it when rates rise. The consequence is a change of measure: the futures
settlement rate is a martingale under the risk-neutral measure, whereas the
forward rate the same period implies is a martingale under the measure that
discounts to the end of that period. The two expectations of the same random
rate differ, and the difference is the **convexity adjustment**.

:class:`~tenor.instruments.Future` takes that adjustment as an input and refuses
a negative one, because the sign is not a modelling choice: margin pays the
holder when rates rise, which is worth something. This module computes it.

**Three contracts, not one.** What the future settles on is a convention, and
the three in use are different random variables over the same window:

===================== ==================================== =====================
convention            settles on                           forward reference
===================== ==================================== =====================
:attr:`Settlement.TERM`        ``(1/P(T1,T2) - 1) / tau``   simple forward
:attr:`Settlement.COMPOUNDED`  ``(exp(int r) - 1) / tau``   simple forward
:attr:`Settlement.AVERAGED`    ``int r / (T2 - T1)``        continuous forward
===================== ==================================== =====================

The first two have the *same* value under the end-of-period forward measure —
that is the telescoping identity :mod:`tenor.overnight` is built on — and
different values under the risk-neutral one, because one reads a single bond
price at the start of the window and the other accumulates the short rate across
it. The third is **linear** in the short rate, so no Jensen term enters at all
and its entire adjustment is the risk-neutral drift of the rate above the
instantaneous forward. Its no-convexity reference is also different: an
arithmetic average of annualised rates lines up with the *continuously*
compounded forward over the window, not the simple one, and at a three-month
accrual the gap between those two references is larger than the convexity.

**Everything here is exact.** Under Hull-White the short rate is Gaussian and
both the bond price and the integral of the rate are exponentials of Gaussians,
so each expectation is a lognormal mean and no expansion is needed. The closed
forms are:

* ``E[r(t)] = f(0,t) + (sigma^2 / 2a^2) (1 - e^-at)^2`` under the risk-neutral
  measure. The second term is the whole reason a future is not a forward.
* ``Var(r(t)) = sigma^2 (1 - e^-2at) / 2a``, which
  :meth:`~tenor.hullwhite.HullWhite.short_rate_variance` already supplies and
  which is the same under every forward measure — the measure change moves the
  mean and leaves the variance alone.
* the mean and variance of ``int_{T1}^{T2} r(u) du``, in
  :func:`integrated_rate_moments`. The variance splits into ``B(T2-T1)^2``
  times the variance of the rate at the start of the window plus a term that
  depends only on the window's length, which is the decomposition of "what was
  already uncertain when the window opened" against "what happens inside it".

**Ho-Lee's formula adjusts a different rate, and its error is not one-signed.**
:func:`~tenor.instruments.ho_lee_convexity` returns ``0.5 sigma^2 T1 T2``, and
this module's own zero-reversion limit says exactly what that is: the drift part
of the term adjustment, expressed as a *continuously* compounded rate. The
contract settles on a simply compounded rate, and the Jensen term that turns
``E[-log P]`` into ``E[1/P - 1]`` is missing. In the zero-reversion limit the
two pieces stand in the ratio ``1 + T1/L`` for a window of length ``L``, so the
omission is worth 4.8% of the adjustment for a three-month contract five years
out and 50% for a one-year accrual one year out. That error makes the formula
**understate**.

Mean reversion pushes the other way, and harder. Measured on a flat 4% curve at
a hundred basis points of rate volatility, the exact three-month term adjustment
this module returns against the Ho-Lee figure is:

========= =============== =============== ===============
``T1``    ``a = 0.01``    ``a = 0.03``    ``a = 0.3``
========= =============== =============== ===============
0.25y     +50.8%          +49.4%          +32.2%
1y        +19.8%          +17.1%          -13.7%
5y        +0.6%           -9.1%           -72.3%
10y       -6.4%           -23.0%          -89.7%
========= =============== =============== ===============

So the approximation is low by half at the front of the strip and high by a
factor of ten at the back of a strongly reverting one, crossing through exact at
``T1 = 5.31`` years when ``a = 0.01``, 2.82 years when ``a = 0.03`` and 1.39
years when ``a = 0.1``. An error that changes sign inside the range of a single
futures strip cannot be made safe by a tolerance, and the exact formula costs
the same arithmetic. :func:`ho_lee_limit` reproduces ``0.5 sigma^2 T1 T2`` plus
the Jensen term to machine precision, which is how the two are tied together
here rather than by assertion.

**The reference matters more than the adjustment at the front.** On the same
curve the simple forward over a three-month window five years out is 2.01 basis
points above the continuously compounded forward over the same window, where the
convexity adjustment at that point is 11.93 basis points and at one year is
0.73.
Choosing the wrong reference for an averaging contract is therefore a larger
error than ignoring convexity altogether for anything inside two years, and
:func:`forward_settlement_rate` picks the reference from the convention rather
than leaving it to the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .curve import DiscountCurve
from .hullwhite import BadModel, HullWhite

__all__ = [
    "Adjustment",
    "RateMoments",
    "Settlement",
    "continuous_forward_rate",
    "convexity_adjustment",
    "expected_settlement_rate",
    "forward_settlement_rate",
    "ho_lee_limit",
    "integrated_rate_moments",
    "risk_neutral_drift",
    "short_rate_moments",
    "term_growth_moments",
]


class Settlement(str, Enum):
    """What the contract's final settlement price is computed from."""

    TERM = "term"
    """A rate fixed at the start of the window and simply compounded.

    The classic three-month contract on an interbank term rate. The settlement
    rate is ``(1 / P(T1, T2) - 1) / tau``, a function of one bond price, and it
    is known the moment the window opens.
    """

    COMPOUNDED = "compounded"
    """The daily rate compounded in arrears across the window.

    The settlement rate is ``(exp(int r du) - 1) / tau`` in the continuous
    limit. Under the end-of-period forward measure this telescopes to the same
    simple forward the term convention references; under the risk-neutral
    measure it does not, because the whole path contributes.
    """

    AVERAGED = "averaged"
    """The arithmetic average of the daily rate across the window.

    The settlement rate is ``int r du / (T2 - T1)``. Linear in the rate, so the
    only thing separating it from a forward is the drift. The day-count accrual
    does not enter: an average of annualised rates weighted by day counts is a
    plain time average, and dividing it by a year fraction a second time would
    be counting the window's length twice.
    """


@dataclass(frozen=True, slots=True)
class RateMoments:
    """The mean and variance of a Gaussian quantity under the risk-neutral measure.

    Attributes:
        mean: The expectation.
        variance: The variance. Zero is reachable and is not degenerate — it is
            the deterministic model, and every expectation here reduces to the
            forward there.
    """

    mean: float
    variance: float

    @property
    def lognormal_mean(self) -> float:
        """``E[exp(X)]`` for this quantity, ``exp(mean + variance / 2)``."""
        return math.exp(self.mean + 0.5 * self.variance)


@dataclass(frozen=True, slots=True)
class Adjustment:
    """A convexity adjustment and the two pieces it is made of.

    Attributes:
        futures: The expected settlement rate under the risk-neutral measure.
        forward: The settlement rate the curve alone implies, with no model.
        drift: The part of the adjustment that comes from the risk-neutral drift
            of the short rate above the instantaneous forward.
        jensen: The part that comes from the curvature of the settlement
            function — zero for :attr:`Settlement.AVERAGED`, which is linear.
    """

    futures: float
    forward: float
    drift: float
    jensen: float

    @property
    def total(self) -> float:
        """``futures - forward``, the number :class:`~tenor.instruments.Future` wants."""
        return self.futures - self.forward


def _integrated_decay(speed: float, time: float) -> float:
    """``int_0^t (1 - e^-au)^2 du``, which is ``a^2`` times the integral of ``B(u)^2``.

    The direct antiderivative is ``t - 2 B(t) + (1 - e^-2at) / 2a``, three terms
    of size ``t`` whose sum is of size ``a^2 t^3 / 3``. At ``a t = 1e-3`` that is
    a cancellation of seven digits and at ``1e-5`` of eleven, which is enough to
    destroy the zero-reversion limit this module is checked against. Below that
    threshold the series in ``x = a t`` is used instead::

        (1/a) (x^3/3 - x^4/4 + 7 x^5/60 - x^6/24 + O(x^7))

    so the small-argument branch is accurate to a relative ``O(x^4)`` and the
    direct branch takes over while it still has ten digits.
    """
    x = speed * time
    if x < 1.0e-3:
        poly = x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0 - x**6 / 24.0
        return poly / speed
    decay = -math.expm1(-speed * time) / speed
    return time - 2.0 * decay - math.expm1(-2.0 * speed * time) / (2.0 * speed)


def _check_window(start: float, end: float) -> float:
    if not math.isfinite(start) or start < 0.0:
        raise BadModel(f"a futures window cannot start at {start!r}")
    if not math.isfinite(end) or end <= start:
        raise BadModel(
            f"a futures window from {start!r} to {end!r} has no length; the rate has "
            "to accrue over something"
        )
    return end - start


def risk_neutral_drift(model: HullWhite, time: float) -> float:
    """How far ``E[r(t)]`` sits above the curve's instantaneous forward at ``t``.

    ``(sigma^2 / 2a^2) (1 - e^-at)^2``. This single term is the whole difference
    between the risk-neutral and the forward measure for a one-factor Gaussian
    model, and it is why a futures rate exceeds a forward rate rather than the
    other way round. It is zero at ``t = 0``, grows like ``sigma^2 t^2 / 2`` for
    small ``t`` — note the *square*, so the front contract has almost none — and
    saturates at ``sigma^2 / 2a^2`` far out, which is the stationary variance of
    the rate divided by its reversion speed.

    Args:
        model: The Hull-White parameters.
        time: Years to the start of the window. Non-negative.

    Returns:
        The drift, in rate units.

    Raises:
        BadModel: If ``time`` is negative.
    """
    if not math.isfinite(time) or time < 0.0:
        raise BadModel(f"time cannot be negative, got {time!r}")
    decay = -math.expm1(-model.a * time)
    return model.sigma**2 * decay * decay / (2.0 * model.a**2)


def short_rate_moments(curve: DiscountCurve, model: HullWhite, time: float) -> RateMoments:
    """The risk-neutral mean and variance of the short rate at ``time``.

    The mean is the curve's instantaneous forward plus
    :func:`risk_neutral_drift`; the variance is
    :meth:`~tenor.hullwhite.HullWhite.short_rate_variance`, which the measure
    change does not touch.

    Args:
        curve: The discount curve the model is fitted to.
        model: The Hull-White parameters.
        time: Years ahead. Non-negative.

    Returns:
        The moments of ``r(time)``.
    """
    mean = curve.instantaneous_forward(time) + risk_neutral_drift(model, time)
    return RateMoments(mean, model.short_rate_variance(time))


def term_growth_moments(
    curve: DiscountCurve, model: HullWhite, start: float, end: float
) -> RateMoments:
    """The moments of ``log(1 / P(T1, T2))`` under the risk-neutral measure.

    The term convention settles on one bond price, and under Hull-White that
    bond price is ``A(T1,T2) exp(-B(T2-T1) r(T1))``, so its reciprocal's
    logarithm is affine in a Gaussian rate. Both moments follow from
    :func:`short_rate_moments` and the model's own ``A`` and ``B``, which is
    why nothing here re-derives the bond price formula.

    Args:
        curve: The discount curve.
        model: The Hull-White parameters.
        start: Years to the window's start, ``T1``.
        end: Years to the window's end, ``T2``.

    Returns:
        The moments of ``log(1 / P(T1, T2))``.

    Raises:
        BadModel: If the window is empty or starts before today.
    """
    tenor = _check_window(start, end)
    factor = model.b(tenor)
    rate = short_rate_moments(curve, model, start)
    mean = -model.log_a(curve, start, end) + factor * rate.mean
    return RateMoments(mean, factor * factor * rate.variance)


def integrated_rate_moments(
    curve: DiscountCurve, model: HullWhite, start: float, end: float
) -> RateMoments:
    """The moments of ``int_{T1}^{T2} r(u) du`` under the risk-neutral measure.

    The mean is the curve's own log discount ratio over the window plus the
    integral of :func:`risk_neutral_drift` across it, both in closed form. The
    variance splits exactly into two pieces::

        Var = B(T2-T1)^2 Var(r(T1)) + (sigma/a)^2 [L - 2 B(L) + (1 - e^-2aL) / 2a]

    with ``L = T2 - T1``. The first is the uncertainty that already existed when
    the window opened, propagated through the same ``B`` a bond of the window's
    length uses; the second depends only on the window's length and is what
    happens inside it. At ``T1 = 0`` the first vanishes and the second does not,
    which is the difference between a term contract that fixes today — no
    convexity, the rate is known — and a compounded one that starts today and
    still accrues an unknown path.

    Args:
        curve: The discount curve.
        model: The Hull-White parameters.
        start: Years to the window's start.
        end: Years to the window's end.

    Returns:
        The moments of the integrated short rate, which is dimensionless.

    Raises:
        BadModel: If the window is empty or starts before today.
    """
    length = _check_window(start, end)
    a, sigma = model.a, model.sigma
    span = _integrated_decay(a, end) - _integrated_decay(a, start)
    drift = sigma**2 * span / (2.0 * a**2)
    mean = math.log(curve.discount_at(start) / curve.discount_at(end)) + drift

    carried = model.b(length) ** 2 * model.short_rate_variance(start)
    inside = (sigma / a) ** 2 * _integrated_decay(a, length)
    return RateMoments(mean, carried + inside)


def continuous_forward_rate(curve: DiscountCurve, start: float, end: float) -> float:
    """The continuously compounded forward rate over the window, from the curve.

    ``log(P(0,T1) / P(0,T2)) / (T2 - T1)``. This is the reference an averaging
    contract is adjusted away from, not the simple forward: the arithmetic mean
    of the instantaneous rate over a window is the window's continuous forward,
    whereas a simply compounded rate carries the window's own compounding.
    """
    length = _check_window(start, end)
    return math.log(curve.discount_at(start) / curve.discount_at(end)) / length


def forward_settlement_rate(
    curve: DiscountCurve,
    start: float,
    end: float,
    *,
    accrual: float | None = None,
    settlement: Settlement = Settlement.TERM,
) -> float:
    """The settlement rate the curve implies with no model and no adjustment.

    For the term and compounded conventions this is the simple forward
    ``(P(0,T1)/P(0,T2) - 1) / tau``; they agree here and disagree only once a
    measure is chosen. For the averaged convention it is
    :func:`continuous_forward_rate`.

    Args:
        curve: The discount curve.
        start: Years to the window's start.
        end: Years to the window's end.
        accrual: The contract's day-count year fraction for the window. Defaults
            to ``end - start``. Ignored for :attr:`Settlement.AVERAGED`, which
            does not accrue.
        settlement: Which convention the contract settles on.

    Returns:
        The rate, in decimal.
    """
    length = _check_window(start, end)
    if settlement is Settlement.AVERAGED:
        return continuous_forward_rate(curve, start, end)
    tau = length if accrual is None else accrual
    if not math.isfinite(tau) or tau <= 0.0:
        raise BadModel(f"an accrual year fraction must be positive, got {tau!r}")
    return (curve.discount_at(start) / curve.discount_at(end) - 1.0) / tau


def expected_settlement_rate(
    curve: DiscountCurve,
    model: HullWhite,
    start: float,
    end: float,
    *,
    accrual: float | None = None,
    settlement: Settlement = Settlement.TERM,
) -> float:
    """The expected settlement rate under the risk-neutral measure: the futures rate.

    Args:
        curve: The discount curve.
        model: The Hull-White parameters.
        start: Years to the window's start.
        end: Years to the window's end.
        accrual: The contract's day-count year fraction. Defaults to
            ``end - start``; ignored for :attr:`Settlement.AVERAGED`.
        settlement: Which convention the contract settles on.

    Returns:
        The rate the contract's price implies, in decimal.
    """
    length = _check_window(start, end)
    if settlement is Settlement.AVERAGED:
        return integrated_rate_moments(curve, model, start, end).mean / length
    tau = length if accrual is None else accrual
    if not math.isfinite(tau) or tau <= 0.0:
        raise BadModel(f"an accrual year fraction must be positive, got {tau!r}")
    moments = (
        term_growth_moments(curve, model, start, end)
        if settlement is Settlement.TERM
        else integrated_rate_moments(curve, model, start, end)
    )
    return (moments.lognormal_mean - 1.0) / tau


def convexity_adjustment(
    curve: DiscountCurve,
    model: HullWhite,
    start: float,
    end: float,
    *,
    accrual: float | None = None,
    settlement: Settlement = Settlement.TERM,
) -> Adjustment:
    """The convexity adjustment for a rate future, split into its two causes.

    The total is the futures rate less the forward rate and is what
    :class:`~tenor.instruments.Future` wants. The split is the useful part: the
    **drift** piece is the risk-neutral drift of the short rate integrated over
    the relevant span, and is shared by every convention; the **Jensen** piece
    comes from the curvature of the map from the rate to the settlement amount,
    and is what the averaging convention does not have. In the zero-reversion
    limit the two stand in the ratio ``1 + T1/L``, and the exact figures track
    it closely: 21.03 for a three-month contract five years out, 6.06 for a
    one-year accrual five years out and 2.01 for a one-year accrual one year
    out. Ho-Lee's formula is the drift piece alone, so that ratio is also the
    size of what it leaves out.

    Args:
        curve: The discount curve.
        model: The Hull-White parameters.
        start: Years to the window's start.
        end: Years to the window's end.
        accrual: The contract's day-count year fraction. Defaults to the
            window's length; ignored for :attr:`Settlement.AVERAGED`.
        settlement: Which convention the contract settles on.

    Returns:
        The adjustment and its decomposition. Every field is in rate units.
    """
    length = _check_window(start, end)
    forward = forward_settlement_rate(
        curve, start, end, accrual=accrual, settlement=settlement
    )
    futures = expected_settlement_rate(
        curve, model, start, end, accrual=accrual, settlement=settlement
    )
    if settlement is Settlement.AVERAGED:
        return Adjustment(futures, forward, futures - forward, 0.0)

    tau = length if accrual is None else accrual
    moments = (
        term_growth_moments(curve, model, start, end)
        if settlement is Settlement.TERM
        else integrated_rate_moments(curve, model, start, end)
    )
    reference = math.log(curve.discount_at(start) / curve.discount_at(end))
    # The drift piece is what the settlement rate would be if the exponent were
    # shifted by its own excess mean and nothing else; the Jensen piece is the
    # rest, which is the variance term.
    without_jensen = (math.exp(moments.mean) - 1.0) / tau
    drift = without_jensen - (math.exp(reference) - 1.0) / tau
    return Adjustment(futures, forward, drift, futures - without_jensen)


def ho_lee_limit(
    volatility: float, start: float, end: float, *, settlement: Settlement = Settlement.TERM
) -> float:
    """The zero-mean-reversion limit of the adjustment, as a continuous rate.

    Written out rather than taken numerically, because the limit is where this
    module meets :func:`~tenor.instruments.ho_lee_convexity` and the comparison
    is the point. Expressed as a continuously compounded rate over the window —
    so all three conventions are directly comparable with each other and with
    the textbook number — the limits are, with ``L = T2 - T1``:

    ======================== ==========================================
    convention               limit
    ======================== ==========================================
    :attr:`Settlement.TERM`        ``sigma^2 T1 (L + T1/2)``
    :attr:`Settlement.COMPOUNDED`  ``[sigma^2 (T2^3 - T1^3)/6 + sigma^2 (T1 L^2 + L^3/3)/2] / L``
    :attr:`Settlement.AVERAGED`    ``sigma^2 (T1^2 + T1 T2 + T2^2) / 6``
    ======================== ==========================================

    :func:`~tenor.instruments.ho_lee_convexity` returns ``0.5 sigma^2 T1 T2``,
    and the term limit above is exactly that **plus** ``0.5 sigma^2 T1 L`` — the
    Jensen term simple compounding adds and the textbook formula leaves out. The
    textbook number is therefore the adjustment to a rate with no compounding in
    it, and the ratio of the two is ``1 + L / T2``: 4.8% for a three-month
    contract five years out, 50% for a one-year accrual one year out.

    The averaged limit beats the textbook number by ``sigma^2 L^2 / 6``, which
    is 0.08% at a three-month accrual — the two agree to second order in the
    window's length and are not the same formula. All three limits collapse to
    ``sigma^2 T1^2 / 2`` as the window shrinks, because there is then only one
    rate and no convention left to choose.

    Args:
        volatility: Absolute short-rate volatility, 0.01 for a hundred basis
            points a year.
        start: Years to the window's start.
        end: Years to the window's end.
        settlement: Which convention.

    Returns:
        The limiting adjustment as a continuously compounded rate over the
        window, in decimal.

    Raises:
        BadModel: If the window is empty or the volatility is negative.
    """
    length = _check_window(start, end)
    if not math.isfinite(volatility) or volatility < 0.0:
        raise BadModel(f"volatility must be non-negative and finite, got {volatility!r}")
    variance = volatility**2
    if settlement is Settlement.AVERAGED:
        return variance * (start * start + start * end + end * end) / 6.0
    if settlement is Settlement.TERM:
        return variance * start * (length + 0.5 * start)
    drift = variance * (end**3 - start**3) / 6.0
    jensen = 0.5 * variance * (start * length * length + length**3 / 3.0)
    return (drift + jensen) / length
