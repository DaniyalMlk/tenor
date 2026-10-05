"""Two Gaussian factors, because one cannot decorrelate two rates.

:mod:`tenor.hullwhite` is a real model and has one limitation that calibration
cannot touch. With a single factor every bond price is an affine function of
the same scalar, so ``log P(t, T1)`` and ``log P(t, T2)`` are affine in one
Gaussian and their correlation is **exactly one** for every pair of maturities,
at every parameter setting. The short end and the long end move together,
always. For anything whose value depends on the shape of the curve rather than
its level, that is not a small approximation.

The additive two-factor Gaussian model — G2++ — fixes it without giving up a
single closed form::

    r(t) = x(t) + y(t) + phi(t)
    dx = -a x dt + sigma dW1,   x(0) = 0
    dy = -b y dt + eta dW2,     y(0) = 0
    dW1 dW2 = rho dt

``phi`` never has to be written down, for the same reason ``theta`` never does
in the one-factor case: the whole of the initial curve enters through the bond
price, as a ratio of the curve's own discount factors times a correction that
vanishes at ``t = 0``. So :meth:`G2.curve_error` is **identically** zero rather
than zero to a tolerance — there is nothing being fitted.

**Equal mean reversions collapse the model to Hull-White, exactly.** If
``a == b`` then both factors have the same ``B``, the bond price depends on
``x + y`` alone, and ``x + y`` is itself an Ornstein-Uhlenbeck process with
speed ``a`` and volatility ``sqrt(sigma**2 + eta**2 + 2 rho sigma eta)``. So
the model *is* the one-factor model there, with that effective volatility, and
every formula here must reduce to the corresponding one in
:mod:`tenor.hullwhite`. That is the strongest check available for this module
and it crosses modules rather than checking the algebra against itself; the
tests assert the bond price, the option price and the implied correlation all
reduce, to rounding.

It is also the reason ``a == b`` is allowed rather than refused. A caller who
reaches it has written a one-factor model in two-factor notation, which is a
waste rather than an error, and refusing it would remove the test.

**What the second factor is worth, measured.** At a 0.50 fast speed, a 0.05
slow speed and a correlation of -0.9 between the drivers — an ordinary shape
for a fitted set — the one-year and ten-year zero rates come out correlated
**0.3592** at a one-year horizon, against the one factor's forced 1.0. Further
apart is weaker still: 0.1549 for the six-month against the ten-year.

The decorrelation is the work of the *gap* between the two speeds rather than
of the second factor's existence, and it closes as they meet: holding
everything else, moving the slow speed from 0.05 up towards the fast 0.50 gives
0.3592, 0.3387, 0.4255, 0.8862, 0.9999 and finally exactly 1.0000 at 0.50. Not
monotone — there is a shallow minimum near 0.1 — and the endpoint is the
collapse described above, arrived at numerically rather than assumed.

**And a cap cannot see any of it, exactly rather than approximately.** The
reason is structural: a cap is a strip of options on *single* bonds, so its
price depends only on each :meth:`G2.bond_option_volatility`, which is the
variance of one log bond price. The joint law of two maturities never enters,
so no cap or floor in this module carries information about it.

Measured, that is not a small effect. Refitting ``sigma`` so that a one-year to
ten-year semi-annual cap struck at 3.8% reprices to the same number, seven
values of ``rho`` from -0.9 to 0.9 give cap prices equal to within **2.4e-15
relative** — the same number, to rounding — while the one-year-to-ten-year
zero-rate correlation moves from **0.0674 to 0.9965**. Seven calibrations a cap
market cannot distinguish, describing curves that bend in completely different
ways. That is the quantitative form of the usual advice that caps calibrate
volatility and swaptions calibrate correlation, and it is why
:func:`zero_rate_correlation` is part of the public surface rather than a
diagnostic: it is the one thing here that the instruments here cannot pin down.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .curve import DiscountCurve
from .hullwhite import BadModel
from .options import Payoff
from .solve import NoRoot, Root, brent

__all__ = [
    "G2",
    "Correlation",
    "FactorMoments",
    "bond_option",
    "cap_price",
    "caplet",
    "factor_moments",
    "fit_fast_volatility",
    "simulate_factors",
    "zero_rate_correlation",
]


def _norm_cdf(x: float) -> float:
    """The standard normal distribution function, through ``erfc``."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@dataclass(frozen=True, slots=True)
class FactorMoments:
    """The joint law of ``(x(t), y(t))``, which is all the randomness there is.

    Both factors start at zero and are driven by correlated Brownian motions,
    so each is a centred Gaussian and the pair is jointly normal. Everything
    else in this module is a function of these three numbers.

    Attributes:
        variance_x: ``Var(x(t))``.
        variance_y: ``Var(y(t))``.
        covariance: ``Cov(x(t), y(t))``.
    """

    variance_x: float
    variance_y: float
    covariance: float

    @property
    def correlation(self) -> float:
        """``Corr(x(t), y(t))``, which is ``rho`` only in the long run.

        The instantaneous drivers are correlated at ``rho``, and the integrated
        factors are not: the two mean reversions weight the same history
        differently, so the correlation of the levels is strictly inside
        ``rho`` in absolute value for every finite horizon and approaches a
        limit set by the two speeds rather than reaching ``rho``.
        """
        spread = math.sqrt(self.variance_x * self.variance_y)
        if spread == 0.0:
            return 0.0
        return self.covariance / spread


@dataclass(frozen=True, slots=True)
class G2:
    """The five parameters of the two-factor additive Gaussian model.

    Attributes:
        a: Mean reversion of the first factor, in reciprocal years. Positive.
        sigma: Volatility of the first factor. Non-negative.
        b: Mean reversion of the second factor. Positive.
        eta: Volatility of the second factor. Non-negative.
        rho: Correlation between the two drivers, strictly inside ``(-1, 1)``.

    Raises:
        BadModel: If a speed is not positive, a volatility is negative, or
            ``rho`` is outside the open interval. Note that ``a == b`` is
            permitted: it is the one-factor model written in two-factor
            notation, which is a waste rather than an error, and the tests use
            it to check this module against :mod:`tenor.hullwhite`.
    """

    a: float
    sigma: float
    b: float
    eta: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        for name in ("a", "b"):
            speed = float(getattr(self, name))
            if not math.isfinite(speed) or speed <= 0.0:
                raise BadModel(
                    f"mean reversion {name} must be a positive finite number, got "
                    f"{speed!r}. The zero-reversion limit has a different B(t, T)."
                )
        for name in ("sigma", "eta"):
            volatility = float(getattr(self, name))
            if not math.isfinite(volatility) or volatility < 0.0:
                raise BadModel(
                    f"volatility {name} must be a non-negative finite number, got "
                    f"{volatility!r}"
                )
        if not math.isfinite(self.rho) or not -1.0 < self.rho < 1.0:
            raise BadModel(
                f"rho must lie strictly inside (-1, 1), got {self.rho!r}. At the "
                "endpoints the two drivers are the same Brownian motion up to sign "
                "and the covariance matrix is singular."
            )

    @property
    def effective_volatility(self) -> float:
        """The one-factor volatility this model equals when ``a == b``.

        ``sqrt(sigma**2 + eta**2 + 2 rho sigma eta)``, the volatility of the
        sum of the two drivers. Meaningless unless the speeds agree, and
        exposed because that case is this module's cross-check against
        :mod:`tenor.hullwhite`.
        """
        return math.sqrt(
            self.sigma**2 + self.eta**2 + 2.0 * self.rho * self.sigma * self.eta
        )

    def b_factor(self, speed: float, tenor: float) -> float:
        """``(1 - e**(-speed * tenor)) / speed``, the loading on one factor."""
        if tenor < 0.0:
            raise BadModel(f"a bond tenor cannot be negative, got {tenor!r}")
        return -math.expm1(-speed * tenor) / speed

    def variance_term(self, tenor: float) -> float:
        """``V(t, T)``, the model's own variance of the integrated short rate.

        A function of ``T - t`` alone, which is why the bond price's correction
        term can be written with three evaluations of one function. The three
        pieces are the two factors' own contributions and their cross term, and
        the cross term is the only place ``rho`` appears in a bond price.
        """
        if tenor < 0.0:
            raise BadModel(f"a tenor cannot be negative, got {tenor!r}")
        if tenor == 0.0:
            return 0.0
        a, b, sigma, eta, rho = self.a, self.b, self.sigma, self.eta, self.rho
        first = (sigma * sigma / (a * a)) * (
            tenor
            + (2.0 / a) * math.exp(-a * tenor)
            - (1.0 / (2.0 * a)) * math.exp(-2.0 * a * tenor)
            - 1.5 / a
        )
        second = (eta * eta / (b * b)) * (
            tenor
            + (2.0 / b) * math.exp(-b * tenor)
            - (1.0 / (2.0 * b)) * math.exp(-2.0 * b * tenor)
            - 1.5 / b
        )
        cross = (2.0 * rho * sigma * eta / (a * b)) * (
            tenor
            + math.expm1(-a * tenor) / a
            + math.expm1(-b * tenor) / b
            - math.expm1(-(a + b) * tenor) / (a + b)
        )
        return first + second + cross

    def short_rate_mean(self, curve: DiscountCurve, time: float) -> float:
        """``phi(t)``, the deterministic shift that makes the fit exact.

        Never needed to price anything — the whole initial curve enters
        :meth:`bond_price` as a ratio of discount factors, which is the point
        of writing the model this way — and worth having anyway, because
        ``r(t) = x(t) + y(t) + phi(t)`` is the only sentence that connects the
        two factors to a short rate somebody might want to look at.

        It is the curve's own instantaneous forward plus a convexity term, and
        the convexity term is where the correlation shows up: at a negative
        ``rho`` the shift sits *below* the sum of what the two factors would
        contribute separately.

        Raises:
            BadModel: If ``time`` is negative.
        """
        if not math.isfinite(time) or time < 0.0:
            raise BadModel(f"time must be a non-negative finite number, got {time!r}")
        fast = self.b_factor(self.a, time) * self.a
        slow = self.b_factor(self.b, time) * self.b
        return (
            curve.instantaneous_forward(time)
            + (self.sigma**2 / (2.0 * self.a**2)) * fast * fast
            + (self.eta**2 / (2.0 * self.b**2)) * slow * slow
            + (self.rho * self.sigma * self.eta / (self.a * self.b)) * fast * slow
        )

    def bond_price(
        self,
        curve: DiscountCurve,
        valuation: float,
        maturity: float,
        factor_x: float,
        factor_y: float,
    ) -> float:
        """``P(t, S)`` given both factors at ``t``.

        The curve enters as the ratio of its own discount factors, so at
        ``t = 0`` — where both factors are zero and the variance correction
        telescopes to nothing — this returns the curve's discount factor with
        no error at all. :meth:`curve_error` is the assertion of that.

        Raises:
            BadModel: If ``maturity`` precedes ``valuation`` or either is
                negative.
        """
        if valuation < 0.0:
            raise BadModel(f"the valuation time cannot be negative, got {valuation!r}")
        if maturity < valuation:
            raise BadModel(
                f"a bond maturing at {maturity!r} cannot be priced at {valuation!r}"
            )
        gap = maturity - valuation
        correction = 0.5 * (
            self.variance_term(gap)
            - self.variance_term(maturity)
            + self.variance_term(valuation)
        )
        exponent = (
            correction
            - self.b_factor(self.a, gap) * factor_x
            - self.b_factor(self.b, gap) * factor_y
        )
        ratio = curve.discount_at(maturity) / curve.discount_at(valuation)
        return ratio * math.exp(exponent)

    def curve_error(self, curve: DiscountCurve) -> float:
        """The worst relative error in repricing the curve's own pillars at ``t = 0``.

        Expected to be exactly 0.0, and it is: at ``t = 0`` both factors are
        zero by construction and the three variance terms cancel identically,
        leaving the curve's discount factor multiplied by one. Nothing is
        fitted. A non-zero answer here means the curve's interpolation is not
        what the formula assumes, which is a finding about the curve.
        """
        worst = 0.0
        for time in curve.times:
            if time <= 0.0:
                continue
            expected = curve.discount_at(time)
            got = self.bond_price(curve, 0.0, time, 0.0, 0.0)
            worst = max(worst, abs(got / expected - 1.0))
        return worst

    def bond_option_volatility(self, expiry: float, maturity: float) -> float:
        """``Sigma(T, S)``, the total volatility of the forward bond price.

        The one place the model's randomness enters a bond option, and the one
        place to look when comparing this model with the one-factor one: with
        ``a == b`` this is exactly
        ``HullWhite(a, effective_volatility).bond_volatility(T, S)``, which the
        tests assert.

        Raises:
            BadModel: If either argument is negative or ``maturity`` precedes
                ``expiry``.
        """
        if expiry < 0.0:
            raise BadModel(f"an expiry cannot be negative, got {expiry!r}")
        if maturity < expiry:
            raise BadModel(
                f"a bond maturing at {maturity!r} cannot underlie an option expiring "
                f"at {expiry!r}"
            )
        a, b, sigma, eta, rho = self.a, self.b, self.sigma, self.eta, self.rho
        gap = maturity - expiry
        fast = -math.expm1(-a * gap)
        slow = -math.expm1(-b * gap)
        total = (
            (sigma * sigma / (2.0 * a**3)) * fast * fast * -math.expm1(-2.0 * a * expiry)
            + (eta * eta / (2.0 * b**3)) * slow * slow * -math.expm1(-2.0 * b * expiry)
            + (2.0 * rho * sigma * eta / (a * b * (a + b)))
            * fast
            * slow
            * -math.expm1(-(a + b) * expiry)
        )
        # Non-negative as algebra -- it is a variance -- but the cross term is
        # negative whenever rho is, and a sum of three doubles can land a few
        # units of the last place below zero when the first two are tiny.
        return math.sqrt(max(total, 0.0))


def factor_moments(model: G2, time: float) -> FactorMoments:
    """The joint law of the two factors at ``time``.

    Raises:
        BadModel: If ``time`` is negative.
    """
    if not math.isfinite(time) or time < 0.0:
        raise BadModel(f"time must be a non-negative finite number, got {time!r}")
    a, b, sigma, eta, rho = model.a, model.b, model.sigma, model.eta, model.rho
    return FactorMoments(
        variance_x=-sigma * sigma * math.expm1(-2.0 * a * time) / (2.0 * a),
        variance_y=-eta * eta * math.expm1(-2.0 * b * time) / (2.0 * b),
        covariance=-rho * sigma * eta * math.expm1(-(a + b) * time) / (a + b),
    )


@dataclass(frozen=True, slots=True)
class Correlation:
    """The correlation between two zero rates, with the pieces it came from.

    Attributes:
        value: ``Corr(log P(t, T1), log P(t, T2))``, which is also the
            correlation of the two continuously compounded zero rates over
            those spans, since each is the log price divided by a constant.
        horizon: The ``t`` it was measured at.
        first: The earlier maturity.
        second: The later one.
        moments: The factor law at ``t``.
    """

    value: float
    horizon: float
    first: float
    second: float
    moments: FactorMoments


def zero_rate_correlation(
    model: G2, horizon: float, first: float, second: float
) -> Correlation:
    """How correlated two zero rates are at a future date.

    ``log P(t, T)`` is affine in the two factors with maturity-dependent
    loadings, so two maturities give two different linear combinations of the
    same correlated pair, and the correlation between them follows. It does not
    depend on the curve: the curve sets the level of the bond price and the
    factors set every bit of its variation.

    **One factor cannot produce anything but one here.** With a single factor
    both loadings are scalars times the same Gaussian, so the correlation is
    the ratio of a product to the product of absolute values, which is one.
    That is the limitation this model exists to remove, and it is why the only
    honest way to report a number from this function is beside that one.

    Args:
        model: The parameters.
        horizon: When the correlation is measured. Positive; at zero there is
            no randomness yet and no correlation to report.
        first: Maturity of the first bond, after ``horizon``.
        second: Maturity of the second, after ``horizon``.

    Returns:
        A :class:`Correlation`.

    Raises:
        BadModel: If ``horizon`` is not positive, either maturity precedes it,
            or the model has no volatility at all.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise BadModel(
            f"the horizon must be positive, got {horizon!r}: at zero neither factor "
            "has moved and there is no correlation to report"
        )
    for name, maturity in (("first", first), ("second", second)):
        if not math.isfinite(maturity) or maturity < horizon:
            raise BadModel(
                f"{name} maturity {maturity!r} must be finite and at or after the "
                f"horizon {horizon!r}"
            )
    moments = factor_moments(model, horizon)
    loadings = [
        (
            model.b_factor(model.a, maturity - horizon),
            model.b_factor(model.b, maturity - horizon),
        )
        for maturity in (first, second)
    ]
    variance_x, variance_y, covariance = (
        moments.variance_x,
        moments.variance_y,
        moments.covariance,
    )

    def quadratic(p: tuple[float, float], q: tuple[float, float]) -> float:
        return (
            p[0] * q[0] * variance_x
            + p[1] * q[1] * variance_y
            + (p[0] * q[1] + p[1] * q[0]) * covariance
        )

    spread = math.sqrt(quadratic(loadings[0], loadings[0]) * quadratic(loadings[1], loadings[1]))
    if spread == 0.0:
        raise BadModel(
            "both bonds have zero variance at this horizon, which means both "
            "volatilities are zero or both maturities equal the horizon; there is "
            "no correlation to report"
        )
    value = quadratic(loadings[0], loadings[1]) / spread
    return Correlation(
        value=min(max(value, -1.0), 1.0),
        horizon=horizon,
        first=first,
        second=second,
        moments=moments,
    )


def bond_option(
    curve: DiscountCurve,
    model: G2,
    *,
    expiry: float,
    maturity: float,
    strike: float,
    payoff: Payoff = Payoff.PAYER,
) -> float:
    """A European option on a zero-coupon bond, in closed form.

    The forward bond price is lognormal under the ``T``-forward measure with a
    total volatility of :meth:`G2.bond_option_volatility`, so this is Black's
    formula on it.

    The ``payoff`` convention is :func:`tenor.hullwhite.zero_bond_option`'s,
    deliberately: :attr:`~tenor.options.Payoff.PAYER` is the *put* on the bond,
    because that is what a payer swaption decomposes into. Writing this
    function with its own more obvious ``call``/``put`` flag was the first
    version, and it meant the same call against the two models in this package
    returned opposite options — which the cross-check caught by disagreeing
    with Hull-White by exactly the parity amount at every parameter setting.
    One convention per package is worth more than a readable argument name.

    Put-call parity is an identity in the model's algebra —
    ``call - put = P(0,S) - K P(0,T)`` — which makes it the cheapest check that
    both branches share the same volatility, and it is asserted in the tests.

    Args:
        curve: The initial curve.
        model: The parameters.
        expiry: Option expiry, in years. Non-negative.
        maturity: Bond maturity, at or after ``expiry``.
        strike: Strike *bond price*. Positive.
        payoff: ``PAYER`` for the put on the bond, ``RECEIVER`` for the call.

    Returns:
        The present value, per unit of notional.

    Raises:
        BadModel: If the dates are out of order or the strike is not positive.
    """
    if not math.isfinite(strike) or strike <= 0.0:
        raise BadModel(f"a bond option strike must be positive, got {strike!r}")
    total = model.bond_option_volatility(expiry, maturity)
    near = curve.discount_at(expiry)
    far = curve.discount_at(maturity)
    call = payoff is Payoff.RECEIVER
    if total == 0.0:
        # No randomness: the forward bond price is known and the option is
        # worth its discounted intrinsic value. Reached by a zero-volatility
        # model and by a zero expiry, and both are the same statement.
        intrinsic = far - strike * near
        return max(intrinsic, 0.0) if call else max(-intrinsic, 0.0)
    moneyness = math.log(far / (strike * near)) / total + 0.5 * total
    if call:
        return far * _norm_cdf(moneyness) - strike * near * _norm_cdf(moneyness - total)
    return strike * near * _norm_cdf(total - moneyness) - far * _norm_cdf(-moneyness)


def caplet(
    curve: DiscountCurve,
    model: G2,
    *,
    start: float,
    end: float,
    strike: float,
    accrual: float | None = None,
    floor: bool = False,
) -> float:
    """A caplet or floorlet, as the bond option it is.

    A caplet on the simple rate over ``[start, end]`` pays
    ``accrual * max(L - K, 0)`` at ``end``, and
    ``(1 + accrual K) max(1 / (1 + accrual K) - P(start, end), 0)`` is the same
    payoff seen at ``start``. So a caplet is a *put* on the zero-coupon bond
    maturing at ``end``, struck at ``1 / (1 + accrual K)``, in a quantity of
    ``1 + accrual K`` — exact, not an approximation, and the reason this model
    prices caps in closed form while a swaption still needs an integral.

    Args:
        curve: The initial curve.
        model: The parameters.
        start: When the rate is fixed and the bond option expires.
        end: When the rate pays.
        strike: The cap rate, as a simple rate.
        accrual: Year fraction of the period. Defaults to ``end - start``,
            which is the right default only for an actual/actual-like
            convention; pass the day count's own figure otherwise.
        floor: Price the floorlet instead.

    Returns:
        The present value, per unit of notional.

    Raises:
        BadModel: If the dates are out of order, or the strike makes the
            equivalent bond strike non-positive.
    """
    if end <= start:
        raise BadModel(f"a caplet's period must have positive length, got [{start}, {end}]")
    span = (end - start) if accrual is None else float(accrual)
    if not math.isfinite(span) or span <= 0.0:
        raise BadModel(f"the accrual factor must be positive, got {span!r}")
    quantity = 1.0 + span * strike
    if quantity <= 0.0:
        raise BadModel(
            f"a strike of {strike!r} over an accrual of {span!r} gives 1 + accrual * "
            f"strike = {quantity!r}; the equivalent bond strike is undefined there"
        )
    equivalent = 1.0 / quantity
    return quantity * bond_option(
        curve,
        model,
        expiry=start,
        maturity=end,
        strike=equivalent,
        payoff=Payoff.RECEIVER if floor else Payoff.PAYER,
    )


def cap_price(
    curve: DiscountCurve,
    model: G2,
    *,
    dates: Sequence[float],
    strike: float,
    accruals: Sequence[float] | None = None,
    floor: bool = False,
) -> float:
    """A cap or floor, as the strip of caplets it is.

    Args:
        curve: The initial curve.
        model: The parameters.
        dates: Period boundaries, ascending, at least two. The first period
            runs from ``dates[0]`` to ``dates[1]``.
        strike: The cap rate.
        accruals: Year fractions, one per period. Defaults to the date gaps.
        floor: Price the floor instead.

    Returns:
        The present value, per unit of notional.

    Raises:
        BadModel: If fewer than two dates are given, the dates are not
            ascending, or the accrual count does not match.
    """
    if len(dates) < 2:
        raise BadModel(f"a cap needs at least two dates, got {len(dates)}")
    spans: Sequence[float] | None = accruals
    if spans is not None and len(spans) != len(dates) - 1:
        raise BadModel(
            f"{len(dates)} dates make {len(dates) - 1} periods but {len(spans)} "
            "accrual factors were given"
        )
    total = 0.0
    for index in range(len(dates) - 1):
        total += caplet(
            curve,
            model,
            start=dates[index],
            end=dates[index + 1],
            strike=strike,
            accrual=None if spans is None else spans[index],
            floor=floor,
        )
    return total


def simulate_factors(
    model: G2, horizon: float, *, paths: int, seed: int = 0
) -> list[tuple[float, float]]:
    """Draw ``(x(T), y(T))`` exactly, from the terminal law rather than a scheme.

    The pair is jointly normal with known moments, so there is no discretisation
    to get wrong and no step count to choose: a Cholesky factor of the exact
    covariance applied to independent standard normals is the terminal law, not
    an approximation to it. That is what makes this usable as an independent
    check on the closed forms — a simulation that shared their algebra would
    only confirm the arithmetic.

    The normals come from the standard library's Mersenne Twister through
    ``random.Random``, seeded, so a disagreement is reproducible.

    Args:
        model: The parameters.
        horizon: When to draw. Non-negative.
        paths: How many draws. Positive.
        seed: Seed for the generator.

    Returns:
        One ``(x, y)`` pair per path.

    Raises:
        BadModel: If ``paths`` is not positive, or ``horizon`` is negative.
    """
    if paths <= 0:
        raise BadModel(f"paths must be a positive number of draws, got {paths!r}")
    moments = factor_moments(model, horizon)
    generator = random.Random(seed)
    first_scale = math.sqrt(moments.variance_x)
    if first_scale == 0.0:
        # No first factor at all, so nothing to condition on; the second is
        # independent of it by construction.
        second_scale = math.sqrt(moments.variance_y)
        return [(0.0, second_scale * generator.gauss(0.0, 1.0)) for _ in range(paths)]
    loading = moments.covariance / first_scale
    residual = moments.variance_y - loading * loading
    # Negative only by rounding, and only when the pair is near-singular.
    residual_scale = math.sqrt(max(residual, 0.0))
    draws = []
    for _ in range(paths):
        first = generator.gauss(0.0, 1.0)
        second = generator.gauss(0.0, 1.0)
        draws.append((first_scale * first, loading * first + residual_scale * second))
    return draws


def fit_fast_volatility(
    curve: DiscountCurve,
    model: G2,
    *,
    dates: Sequence[float],
    strike: float,
    target: float,
    accruals: Sequence[float] | None = None,
    floor: bool = False,
    upper: float = 1.0,
    nodes: int = 80,
) -> Root:
    """Solve for the fast factor's ``sigma`` that reprices a cap to ``target``.

    Everything else in ``model`` is held, including ``rho``, which is what
    makes this the tool for the identification question: run it across ``rho``
    and the cap price stays put while the correlation structure does not.

    **A cap price is not monotone in ``sigma``, and a bisection that assumes it
    is will fail silently.** The bond option variance is
    ``sigma**2 A + eta**2 B + 2 rho sigma eta C`` with ``A``, ``B``, ``C``
    positive, so at a negative ``rho`` it is a parabola in ``sigma`` with a
    minimum at ``-rho eta C / A``, strictly inside the domain. Raising
    ``sigma`` from zero there *lowers* the cap. A plain bisection from a floor
    to a ceiling then converges to the floor and returns a price a third away
    from the target while reporting success, which is how this was found.

    So the bracket is searched for rather than assumed: the price is evaluated
    on a geometric grid and the first interval where it crosses ``target`` from
    below is handed to Brent. That picks the upper branch, which is the
    economically sensible root — the one where more volatility means a dearer
    cap.

    Args:
        curve: The initial curve.
        model: The parameters, whose ``sigma`` is replaced by the answer.
        dates: Period boundaries, as for :func:`cap_price`.
        strike: The cap rate.
        target: The price to match.
        accruals: Year fractions, as for :func:`cap_price`.
        floor: Fit a floor instead.
        upper: Largest ``sigma`` to consider.
        nodes: Grid resolution for the bracket search.

    Returns:
        A :class:`~tenor.solve.Root`, whose ``value`` is the fitted ``sigma``.

    Raises:
        NoRoot: If the price never reaches ``target`` from below on
            ``(0, upper]``. The message carries the cheapest price available,
            because an unreachable target is a statement about the quote.
        BadModel: If ``upper`` or ``nodes`` are out of range.
    """
    if not math.isfinite(upper) or upper <= 0.0:
        raise BadModel(f"upper must be finite and positive, got {upper!r}")
    if nodes < 3:
        raise BadModel(f"nodes must be at least 3, got {nodes!r}")

    def price_at(sigma: float) -> float:
        trial = G2(a=model.a, sigma=sigma, b=model.b, eta=model.eta, rho=model.rho)
        return cap_price(
            curve, trial, dates=dates, strike=strike, accruals=accruals, floor=floor
        )

    floor_sigma = upper * 1e-9
    grid = [
        floor_sigma * (upper / floor_sigma) ** (index / (nodes - 1)) for index in range(nodes)
    ]
    prices = [price_at(sigma) for sigma in grid]
    for index in range(nodes - 1):
        if prices[index] < target <= prices[index + 1]:
            return brent(lambda sigma: price_at(sigma) - target, grid[index], grid[index + 1])
    raise NoRoot(
        f"no fast volatility in (0, {upper!r}] reprices this to {target!r} from below. "
        f"The cheapest price on the grid is {min(prices)!r} and the dearest is "
        f"{max(prices)!r}. A cap price is not monotone in sigma at a negative rho, "
        "so a target below the cheapest attainable price is unreachable however the "
        "search is widened."
    )
