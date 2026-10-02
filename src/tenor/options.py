"""Swaptions, caps and floors: where the discount curve finally matters.

:mod:`tenor.multicurve` measured something that reads as a paradox. A par swap
rate is almost independent of the curve the swap is discounted on — a 100 basis
point shift to the discount curve moves the ten-year par rate by -0.12 basis
points, against 101.62 for the same shift to the projection curve, a factor of
843. The discount curve enters a par rate only as the weights in an average of
forwards, and reweighting a nearly flat average changes nearly nothing.

This module is where that stops being true, and the reason is structural rather
than numerical. A swaption is worth the *annuity* times a call on the forward
swap rate, and the annuity is nothing but discount factors. On the ten-year into
ten-year payer in the tests, that same 100 basis point shift moves the forward
swap rate by 0.5067 basis points — 0.125% of its level — and moves the option's
value by **-13.57%**, which is 108 times as much. The move decomposes: the
annuity falls 13.845% and the forward rate's 0.125% rise claws a little back.

That is the whole content of the annuity measure. Under the measure whose
numeraire is the annuity the forward swap rate is a martingale, so the swaption
is a European call on it and Black's formula applies with no approximation
beyond the lognormality assumed of the rate. The discount curve has been
factored into the numeraire, which is exactly why it reappears
multiplicatively.

**Both conventions, because the market quotes both.** A negative forward swap
rate has no lognormal volatility at all, and forward rates went negative and
stayed there for a decade in two major currencies, so :class:`Convention` picks
between Black and Bachelier and the normal volatility is in absolute units of
the rate. The two are not conversions of one another. At the money both are
``numeraire * vol * sqrt(2T/pi)`` to leading order, so the normal volatility
pricing the same straddle is about the lognormal one times the forward — and
only about. On a 3.98% forward at a 10% volatility over one year it is 39.78
basis points against the product's 39.80, agreeing to 4.2e-04 relative; over ten
years to 4.2e-03; and at a 20% volatility over ten years to 1.6e-02. The
agreement is the leading term and nothing more, which is worth knowing before
converting a quote.

**The identities are tests, not remarks.** A payer less a receiver at the same
strike is the forward-starting swap, whatever the volatility: both option values
cancel and ``annuity * (forward - strike)`` is left. A cap less a floor is the
same statement for the strip. Both hold to **1.5e-17** and **6.2e-18** of their
own numeraires across every strike and both conventions in the suite, which is
what says the caplet strip and the swaption are built on the same forwards and
the same discount factors rather than on two slightly different readings of the
curve.

**A cap is a portfolio of options and a swaption is an option on the
portfolio**, and the gap between them is large rather than a correction. The
strip's holder chooses period by period, so it is worth more — on the five-year
quarterly structure in the tests, **78.1% more** at a 20% volatility. The
premium *shrinks* as volatility rises, 82.8% at 10% against 74.8% at 40%, which
is the opposite of the way an option value's dependence on volatility usually
runs and is worth stating because the natural guess is that more volatility
means more to choose between.

That inequality — strip at least as valuable as the option on it — is Jensen's,
and it is the sharpest available check that the two prices are built on the same
curve. Over twenty-five strikes and volatilities it holds in twenty-four and
fails in one, by 6.98e-05, and the failure is the interesting part: it is not a
defect but the day-count mismatch between the two numeraires. The fixed leg's
annuity accrues on unadjusted boundaries under its own basis and the floating
periods accrue on adjusted ones under the index's, so the annuity and the sum of
the caplet weights differ by 4.655e-05 relative, and the forward swap rate and
the weight-average forward differ by 4.654e-05 — the same number. Deep in the
money at a low volatility both prices collapse to their intrinsics, the ratio
collapses to those two mismatches, and ``(1 + 4.655e-05)(1 - 1.1635e-04)`` is
0.99993019 against a measured 0.99993019. The inequality is exact in the
mathematics and violated in the conventions, by precisely the amount the
conventions differ.

**The first caplet is excluded by default**, because its rate has already fixed
by the time anyone holds the cap and there is no optionality left in it.
``include_first=True`` prices it, at exactly its intrinsic value — 0.001412984290
against an intrinsic of 0.001412984290 on the structure in the tests, which is
7.8% of the whole cap and not something to lose silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from .calendar import Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .multicurve import DualSwap, FloatingLeg, ForecastIndex, projected_forward
from .schedule import Frequency, Schedule, generate
from .solve import NoRoot, brent

__all__ = [
    "BadOption",
    "Cap",
    "Caplet",
    "Convention",
    "Payoff",
    "Swaption",
    "bachelier",
    "black",
    "implied_volatility",
]

#: One over the square root of two pi, which is the at-the-money straddle's
#: constant in both conventions and the reason they agree to leading order.
_ROOT_TWO_PI = math.sqrt(2.0 * math.pi)


class BadOption(ValueError):
    """An option that does not describe a trade."""


class Payoff(str, Enum):
    """Which side of the strike the holder receives.

    A payer swaption pays fixed and receives floating, so it gains when rates
    rise: it is a *call* on the swap rate. A receiver is the put. Naming them
    after the fixed leg rather than after the option is the market's convention
    and it inverts for a bond, which is the usual source of a sign error here.
    """

    PAYER = "payer"
    RECEIVER = "receiver"

    @property
    def sign(self) -> float:
        return 1.0 if self is Payoff.PAYER else -1.0

    @property
    def rate_option(self) -> str:
        return "call" if self is Payoff.PAYER else "put"


class Convention(str, Enum):
    """Which model the quoted volatility belongs to.

    Lognormal is Black and the volatility is proportional; normal is Bachelier
    and it is in absolute units of the rate. The choice is not cosmetic: a
    negative forward swap rate has no lognormal volatility, which is why the
    normal convention exists at all.
    """

    LOGNORMAL = "lognormal"
    NORMAL = "normal"


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _ROOT_TWO_PI


def black(forward: float, strike: float, total_vol: float, payoff: Payoff) -> float:
    """The undiscounted Black value, per unit of numeraire.

    ``total_vol`` is ``sigma * sqrt(T)``. Zero volatility gives the intrinsic
    value, which is the limit rather than a special case: it is what the formula
    converges to and asserting it costs one branch and removes a division by
    zero.

    Raises:
        BadOption: If the forward or the strike is not positive, which is where
            the lognormal convention has no value at all — and the reason
            :func:`bachelier` exists.
    """
    if total_vol < 0.0 or not math.isfinite(total_vol):
        raise BadOption(f"a total volatility of {total_vol!r} is not a number of one")
    sign = payoff.sign
    if total_vol == 0.0:
        return max(sign * (forward - strike), 0.0)
    if forward <= 0.0 or strike <= 0.0:
        raise BadOption(
            f"Black has no value at a forward of {forward!r} and a strike of "
            f"{strike!r}: the lognormal model cannot reach a non-positive rate, "
            "which is what the normal convention is for"
        )
    first = math.log(forward / strike) / total_vol + 0.5 * total_vol
    second = first - total_vol
    return sign * (
        forward * _normal_cdf(sign * first) - strike * _normal_cdf(sign * second)
    )


def bachelier(forward: float, strike: float, total_vol: float, payoff: Payoff) -> float:
    """The undiscounted Bachelier value, per unit of numeraire.

    ``total_vol`` is in absolute units of the rate, so a 36.8 basis point
    annualised normal volatility over one year is ``0.00368``. Takes a negative
    forward or strike, which is the whole point of it.
    """
    if total_vol < 0.0 or not math.isfinite(total_vol):
        raise BadOption(f"a total volatility of {total_vol!r} is not a number of one")
    sign = payoff.sign
    if total_vol == 0.0:
        return max(sign * (forward - strike), 0.0)
    scaled = sign * (forward - strike) / total_vol
    return total_vol * (scaled * _normal_cdf(scaled) + _normal_pdf(scaled))


def _value(
    forward: float,
    strike: float,
    total_vol: float,
    payoff: Payoff,
    convention: Convention,
) -> float:
    if convention is Convention.LOGNORMAL:
        return black(forward, strike, total_vol, payoff)
    return bachelier(forward, strike, total_vol, payoff)


def _vega(
    forward: float, strike: float, total_vol: float, convention: Convention
) -> float:
    """``d(value)/d(total_vol)``, the same for both payoffs.

    Strictly positive for a positive total volatility, which is what makes the
    inversion's root unique.
    """
    if total_vol <= 0.0:
        return 0.0
    if convention is Convention.LOGNORMAL:
        first = math.log(forward / strike) / total_vol + 0.5 * total_vol
        return forward * _normal_pdf(first)
    return _normal_pdf((forward - strike) / total_vol)


# -- swaptions ----------------------------------------------------------------


@dataclass(frozen=True)
class Swaption:
    """A European option on a forward-starting interest rate swap.

    The expiry is the option's; the underlying swap runs from ``effective`` to
    ``maturity``, and ``effective`` is normally the expiry or a settlement lag
    after it. Nothing here requires them to coincide, and nothing checks that
    the expiry is before the swap starts other than the refusal below, because a
    mid-curve swaption — an option expiring well before its underlying begins —
    is a real instrument.
    """

    expiry: date
    effective: date
    maturity: date
    strike: float
    payoff: Payoff = Payoff.PAYER
    index: ForecastIndex = field(default_factory=ForecastIndex)
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadOption(
                f"swaption {self.name} underlies a swap maturing on "
                f"{self.maturity.isoformat()}, which is not after it starts on "
                f"{self.effective.isoformat()}"
            )
        if self.expiry > self.effective:
            raise BadOption(
                f"swaption {self.name} expires on {self.expiry.isoformat()}, after "
                f"its swap starts on {self.effective.isoformat()}. An option cannot "
                "be exercised into a swap that has already begun accruing."
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"{self.payoff.value} to {self.maturity.isoformat()} "
            f"expiring {self.expiry.isoformat()}"
        )

    @property
    def swap(self) -> DualSwap:
        """The underlying, struck at this option's strike."""
        return DualSwap(
            effective=self.effective,
            maturity=self.maturity,
            rate=self.strike,
            index=self.index,
            frequency=self.frequency,
            basis=self.basis,
            calendar=self.calendar,
            rolling=self.rolling,
            label=f"{self.name} underlying",
        )

    def time_to_expiry(self, reference: date, basis: Basis = Basis.ACT_365F) -> float:
        """Year fraction from ``reference`` to expiry, for scaling a volatility.

        A separate basis from the fixed leg's, because the day count a
        volatility is quoted on is a different convention from the one a coupon
        accrues on, and using the coupon's here moves an option price by the
        ratio of the two.
        """
        if self.expiry <= reference:
            raise BadOption(
                f"swaption {self.name} expires on {self.expiry.isoformat()}, on or "
                f"before the valuation date {reference.isoformat()}; an expired "
                "option has an exercise decision rather than a value"
            )
        return year_fraction(reference, self.expiry, basis)

    def annuity(self, discount: DiscountCurve) -> float:
        """The numeraire: the underlying's fixed-leg annuity."""
        return self.swap.annuity(discount)

    def forward_rate(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        """The forward swap rate, which is the martingale under the annuity measure."""
        return self.swap.par_rate(discount, projection)

    def value(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float,
        *,
        convention: Convention = Convention.LOGNORMAL,
        vol_basis: Basis = Basis.ACT_365F,
    ) -> float:
        """Annuity times a call on the forward swap rate, per unit of notional."""
        if volatility < 0.0 or not math.isfinite(volatility):
            raise BadOption(f"a volatility of {volatility!r} is not one")
        time = self.time_to_expiry(discount.reference, vol_basis)
        total = volatility * math.sqrt(time)
        return self.annuity(discount) * _value(
            self.forward_rate(discount, projection),
            self.strike,
            total,
            self.payoff,
            convention,
        )

    def intrinsic(
        self, discount: DiscountCurve, projection: DiscountCurve
    ) -> float:
        """What it is worth at zero volatility: the swap, if that is positive."""
        annuity = self.annuity(discount)
        gap = self.payoff.sign * (
            self.forward_rate(discount, projection) - self.strike
        )
        return annuity * max(gap, 0.0)


# -- caps and floors ----------------------------------------------------------


@dataclass(frozen=True)
class Caplet:
    """One period of a cap: an option on a single forward rate.

    Attributes:
        start: When the period's rate fixes and starts accruing.
        end: When it stops.
        payment: When the money moves.
        accrual: The period's year fraction under the index basis.
        forward: The projected rate.
        discount: The discount factor at the payment date.
        value: The option's value, per unit of notional.
    """

    start: date
    end: date
    payment: date
    accrual: float
    forward: float
    discount: float
    value: float

    @property
    def weight(self) -> float:
        """``accrual * discount``: what one unit of rate is worth here."""
        return self.accrual * self.discount


@dataclass(frozen=True)
class Cap:
    """A strip of options on an index's forwards, one per period.

    A cap is not a swaption and the difference is not a refinement. A cap is a
    portfolio of options, each exercised on its own; a swaption is one option on
    the portfolio. The strip is worth more, because its holder chooses period by
    period, and the gap is that flexibility — 18.4% of the swaption's value on
    the five-year quarterly structure in the tests at a 20% volatility.

    ``payoff`` is in the swap convention, so a :attr:`Payoff.PAYER` cap is a cap
    proper (a strip of calls on the rate) and a receiver is a floor. Naming them
    the same way as the swaption keeps the parity identity readable: a payer
    less a receiver is the swap, for the strip exactly as for the option on it.
    """

    effective: date
    maturity: date
    strike: float
    payoff: Payoff = Payoff.PAYER
    index: ForecastIndex = field(default_factory=ForecastIndex)
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadOption(
                f"cap {self.name} matures on {self.maturity.isoformat()}, which is "
                f"not after it starts on {self.effective.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"{'cap' if self.payoff is Payoff.PAYER else 'floor'} to "
            f"{self.maturity.isoformat()}"
        )

    @property
    def leg(self) -> FloatingLeg:
        """The floating leg whose forwards this is written on."""
        return FloatingLeg(
            effective=self.effective,
            maturity=self.maturity,
            index=self.index,
            label=f"{self.name} underlying",
        )

    def schedule(self) -> Schedule:
        return generate(
            self.effective,
            self.maturity,
            self.index.tenor,
            calendar=self.index.business_days,
            rolling=self.index.rolling,
            payment_lag=self.index.payment_lag,
        )

    def caplets(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float,
        *,
        convention: Convention = Convention.LOGNORMAL,
        vol_basis: Basis = Basis.ACT_365F,
        include_first: bool = False,
    ) -> tuple[Caplet, ...]:
        """Every period, priced on its own forward.

        The first period is excluded by default and that is the convention
        rather than an omission: its rate has already fixed by the time anyone
        holds the cap, so there is no optionality left in it. Including it
        prices an option on a known number, which is its intrinsic value and
        not a mistake — but it is not what a quoted cap contains.
        """
        if volatility < 0.0 or not math.isfinite(volatility):
            raise BadOption(f"a volatility of {volatility!r} is not one")
        reference = discount.reference
        made: list[Caplet] = []
        for index, period in enumerate(self.schedule()):
            if index == 0 and not include_first:
                continue
            accrual = year_fraction(
                period.adjusted_start, period.adjusted_end, self.index.basis
            )
            forward = projected_forward(
                projection,
                period.adjusted_start,
                period.adjusted_end,
                self.index.basis,
            )
            factor = discount.discount(period.payment)
            if period.adjusted_start <= reference:
                total = 0.0
            else:
                total = volatility * math.sqrt(
                    year_fraction(reference, period.adjusted_start, vol_basis)
                )
            option = _value(forward, self.strike, total, self.payoff, convention)
            made.append(
                Caplet(
                    start=period.adjusted_start,
                    end=period.adjusted_end,
                    payment=period.payment,
                    accrual=accrual,
                    forward=forward,
                    discount=factor,
                    value=accrual * factor * option,
                )
            )
        if not made:
            raise BadOption(
                f"cap {self.name} has no periods left with any optionality in them. "
                "Its only period has already fixed; pass include_first=True to "
                "price it at its intrinsic value instead."
            )
        return tuple(made)

    def value(
        self,
        discount: DiscountCurve,
        projection: DiscountCurve,
        volatility: float,
        *,
        convention: Convention = Convention.LOGNORMAL,
        vol_basis: Basis = Basis.ACT_365F,
        include_first: bool = False,
    ) -> float:
        return sum(
            one.value
            for one in self.caplets(
                discount,
                projection,
                volatility,
                convention=convention,
                vol_basis=vol_basis,
                include_first=include_first,
            )
        )


# -- reading a premium back as a volatility -----------------------------------


def implied_volatility(
    premium: float,
    forward: float,
    strike: float,
    time: float,
    payoff: Payoff,
    *,
    convention: Convention = Convention.LOGNORMAL,
    tolerance: float = 1e-14,
    max_iterations: int = 100,
) -> float:
    """The annualised volatility reproducing ``premium``, per unit of numeraire.

    ``premium`` is the option's value *divided by the numeraire* — the swaption's
    price over its annuity, or a caplet's over its accrual times its discount
    factor. Dividing first is not a convenience: the numeraire is the one part
    of the price that is not model-dependent, and leaving it in would make the
    inversion's conditioning depend on the curve level.

    Bisection on a bracket rather than Newton. Vega vanishes in both wings, so
    an unsafeguarded Newton step from a deep strike leaves the domain, and the
    bracket here is widened until it contains the premium rather than assumed:
    a quoted premium above every finite volatility's value is a datum about the
    quote and not a reason to return the bound.

    Raises:
        BadOption: If the premium is below intrinsic, in which case no
            volatility reproduces it.
    """
    if time <= 0.0 or not math.isfinite(time):
        raise BadOption(f"a time to expiry of {time!r} cannot scale a volatility")
    intrinsic = max(payoff.sign * (forward - strike), 0.0)
    if premium < intrinsic - 1e-15:
        raise BadOption(
            f"a premium of {premium!r} is below the intrinsic value {intrinsic!r}; "
            "no volatility is low enough to produce it"
        )
    if premium <= intrinsic:
        return 0.0
    root = math.sqrt(time)

    def residual(volatility: float) -> float:
        return (
            _value(forward, strike, volatility * root, payoff, convention) - premium
        )

    high = 1.0 if convention is Convention.LOGNORMAL else max(abs(forward), 0.01)
    for _ in range(60):
        if residual(high) >= 0.0:
            break
        high *= 2.0
    else:  # pragma: no cover - needs a premium above every doubling
        raise BadOption(
            f"no volatility below {high:.6g} reaches a premium of {premium!r}"
        )
    try:
        found = brent(residual, 1e-12, high, tolerance=tolerance,
                      max_iterations=max_iterations)
    except NoRoot as error:  # pragma: no cover - the bracket is widened above
        raise BadOption(f"the premium {premium!r} could not be inverted: {error}") from error
    return found.value
