"""Caplet volatilities, bootstrapped from a term structure of flat cap quotes.

:meth:`~tenor.options.Cap.value` takes one volatility for the whole strip,
which is how a cap is *quoted* and not how it is priced. A quoted flat
volatility is the single number that reproduces a cap's premium; what prices
anything else is the volatility of each forward underneath. Getting from one to
the other is a sequential inverse problem, and this module is it.

The bootstrap runs in maturity. The shortest cap's periods are fixed by its own
quote. Each longer cap adds periods, and one root solve sets a single volatility
for the new ones so that the cap reprices to its quote with the earlier periods
held where previous steps put them. Each step is one-dimensional and monotone —
an option is worth more at a higher volatility — so there is nothing to
iterate and no starting guess to choose.

**Stripping is per strike.** Caplet volatilities at different strikes are
different quantities, so a set of at-the-money caps each struck at its own par
rate is not strippable: the strike belongs to the problem and not to the quote,
which is why :class:`CapQuote` does not carry one.

Three things make this more than a loop over maturities.

**The schedules have to nest, and nothing guarantees they do.**
:func:`~tenor.schedule.generate` runs backward from maturity, so a two-year
cap's periods are a prefix of a five-year cap's only when both maturities sit on
the index's own roll grid from the shared effective date. A quote set whose
maturities fall between rolls produces period boundaries that interleave rather
than nest, and the bootstrap has nothing to hold fixed. That is checked against
the realised schedules and refused by name rather than assumed.

**A solution need not exist.** The incremental premium each step has to explain
is the quoted cap less what the earlier periods are already worth, and the new
periods can produce premiums only in the interval between their intrinsic value
and their value at the volatility ceiling. A flat quote curve that falls fast
enough makes the target unattainable. The bound is sharp and it is measured
below.

**The problem gets worse-conditioned with maturity**, which is the part worth
reporting rather than discovering. Each :class:`CapletBucket` carries the
derivative of its own volatility with respect to the quote that set it, so a
caller can see how much of the far end to believe.

Measured on a five-year quarterly structure against a curve rising from 2.90%
to 3.74%, struck at 3.5%, with flat quotes rising from 18% at one year to 26%
at five:

* **Every quote reprices exactly.** Reading the flat volatility back out of the
  stripped curve returns the input to between 1.3e-16 and 6.2e-16 relative.
  That is the only check here that does not go through another approximation.
* **A flat caplet curve quotes flat**, to every digit: feeding 20% to every
  period and reading the flat quote back returns 0.200000000000000 at all five
  maturities. It is the degenerate case the construction has to agree with,
  and it is what catches a volatility-basis mismatch between the bootstrap and
  :func:`flat_volatility` -- the two would still each be self-consistent.
* **The stripped curve is much steeper than the quotes**, because a flat
  volatility is a premium-weighted average of the caplet volatilities under it
  and not an average of the volatilities themselves. The quotes span 18.0% to
  26.0%; the buckets span 18.0% to **31.2%**, so the far bucket sits 5.2
  volatility points above the quote that set it and the slope between the last
  two buckets is **1.77 times** the slope between the last two quotes.
* **The first bucket's volatility is its quote, exactly**, with a sensitivity
  of 1.0000. A one-bucket strip has nothing to average, so the flat volatility
  *is* the caplet volatility there -- which is the one row of the table that
  can be checked by hand.
* **The conditioning degrades by a factor of 3.6 across five years.** A basis
  point on the one-year quote moves its bucket by 1.00 basis points; the
  two-, three-, four- and five-year quotes move theirs by 1.41, 2.12, 2.84 and
  **3.63**. Each bucket is a smaller share of its cap's premium than the last,
  so the same quoting error says less and less about the curve's far end.
* **A hump in the quotes is an overshoot in the buckets.** Quotes of 18, 26, 30,
  26 and 22 per cent strip to 18.0, 29.3, **34.5**, 18.6 and 11.5. The four-point
  fall from 30 to 26 in the quotes is a sixteen-point fall from 34.5 to 18.6 in
  the buckets, because the later periods have to undo an average the earlier
  ones are holding up. Monotone quotes happen to give monotone buckets on this
  structure and nothing guarantees it, so the tests record the measurement
  rather than the expectation.
* **The quotes can fall, but not freely.** Holding the one-year quote at 18%
  and walking the two-year quote down, the strip survives to **5.9832%** and has
  no answer below it: the two-year cap is then worth exactly what its first
  three periods are already worth plus the intrinsic value of the four new ones,
  so their volatility is zero and there is nothing left to take away. A fall of
  1202 basis points over one year is admissible and 1203 is not. That boundary
  is reached exactly rather than approached, which is why it is returned as a
  zero rather than searched for -- a root solve sees no sign change there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .multicurve import ForecastIndex
from .options import Cap, Convention, Payoff, bachelier, black
from .solve import brent

__all__ = [
    "PREMIUM_TOLERANCE",
    "VOLATILITY_CEILING",
    "CapQuote",
    "CapletBucket",
    "CapletVolatility",
    "StrippingError",
    "flat_volatility",
    "strip_caplets",
]

#: Relative tolerance on a premium, used to decide whether an incremental
#: premium that reads slightly below the new periods' intrinsic value is a
#: quote at the feasibility boundary or a quote below it. A premium is a sum of
#: option values, so the scale to measure against is the premium itself rather
#: than the volatility being solved for.
PREMIUM_TOLERANCE = 1.0e-12

#: The largest volatility the bootstrap will attribute to a bucket. A cap
#: premium is increasing in volatility without limit, so the ceiling is what
#: turns "no solution" into a refusal rather than a widening search. Five
#: hundred per cent is far outside any quoted market and inside the range where
#: the lognormal option value is still numerically well behaved.
VOLATILITY_CEILING = 5.0


class StrippingError(ValueError):
    """A quote set the bootstrap cannot turn into a caplet curve.

    Raised for quotes whose schedules do not nest, for a maturity with no new
    periods to solve for, and for an incremental premium no non-negative
    volatility reaches.
    """


@dataclass(frozen=True)
class CapQuote:
    """One flat volatility, quoted for a cap running to ``maturity``.

    No strike: see the module docstring. The strike belongs to
    :func:`strip_caplets`, because volatilities at two strikes are not
    comparable and a quote set at mixed strikes is not a term structure of one
    quantity.

    Attributes:
        maturity: Last payment date of the cap this volatility prices.
        volatility: The flat volatility quoted for it.
    """

    maturity: date
    volatility: float

    def __post_init__(self) -> None:
        if self.volatility < 0.0 or not math.isfinite(self.volatility):
            raise StrippingError(f"a flat volatility of {self.volatility!r} is not one")


@dataclass(frozen=True)
class CapletBucket:
    """The periods one quote added, and the volatility that explains them.

    Attributes:
        maturity: Maturity of the cap whose quote set this bucket.
        first_fixing: Adjusted start of the earliest period in it.
        last_fixing: Adjusted start of the latest.
        periods: How many caplets the bucket covers.
        volatility: The volatility applied to all of them.
        quoted: The flat volatility quoted for that cap, for comparison.
        sensitivity: ``d volatility / d quoted``, measured by re-solving the
            step. One means a basis point on the quote is a basis point on the
            bucket; the figure grows with maturity as the bucket becomes a
            smaller share of the premium.
    """

    maturity: date
    first_fixing: date
    last_fixing: date
    periods: int
    volatility: float
    quoted: float
    sensitivity: float


@dataclass(frozen=True)
class CapletVolatility:
    """A piecewise-constant caplet volatility curve, by fixing date.

    Piecewise constant and not interpolated. A smoother curve through the same
    quotes exists, but it would reprice the quotes only if its shape were
    solved for jointly, and the sawtooth that a constant bucket produces is
    information rather than an artefact: it is the shape the quotes imply when
    nothing beyond them is assumed.

    Attributes:
        strike: The strike these volatilities belong to.
        buckets: In maturity order.
    """

    strike: float
    buckets: tuple[CapletBucket, ...]

    def volatility(self, fixing: date) -> float:
        """The volatility for a period fixing on ``fixing``.

        Flat before the first bucket and after the last, because a caplet
        outside the quoted range has no quote behind it and extrapolating a
        slope would invent one.
        """
        for bucket in self.buckets:
            if fixing <= bucket.last_fixing:
                return bucket.volatility
        return self.buckets[-1].volatility

    @property
    def quoted_span(self) -> tuple[float, float]:
        """Lowest and highest flat volatility in the input."""
        quotes = [bucket.quoted for bucket in self.buckets]
        return min(quotes), max(quotes)

    @property
    def stripped_span(self) -> tuple[float, float]:
        """Lowest and highest bucket volatility."""
        levels = [bucket.volatility for bucket in self.buckets]
        return min(levels), max(levels)


@dataclass(frozen=True, slots=True)
class _Period:
    """One caplet reduced to what pricing it at any volatility needs.

    ``factor`` is ``sqrt(t)`` on the volatility basis, held separately so that
    a period's value at a new volatility is one Black call rather than a
    rebuild of the schedule -- which matters because the bootstrap reprices
    every earlier period at every step.
    """

    fixing: date
    weight: float
    forward: float
    factor: float


def _cap(
    effective: date, maturity: date, strike: float, index: ForecastIndex, payoff: Payoff
) -> Cap:
    return Cap(effective=effective, maturity=maturity, strike=strike, payoff=payoff, index=index)


def _periods(
    cap: Cap, discount: DiscountCurve, projection: DiscountCurve, vol_basis: Basis
) -> tuple[_Period, ...]:
    """Every period with optionality left in it, in order.

    The already-fixed first period is dropped on the same convention
    :meth:`~tenor.options.Cap.caplets` uses: its rate is known, so an option
    on it is its intrinsic value and not part of a quoted cap.
    """
    rows = []
    for one in cap.caplets(discount, projection, 0.0, include_first=False):
        years = year_fraction(discount.reference, one.start, vol_basis)
        rows.append(
            _Period(
                fixing=one.start,
                weight=one.weight,
                forward=one.forward,
                factor=math.sqrt(years) if years > 0.0 else 0.0,
            )
        )
    return tuple(rows)


def _value(
    periods: tuple[_Period, ...],
    strike: float,
    volatility: float,
    payoff: Payoff,
    convention: Convention,
) -> float:
    """Value of the given periods, all at one volatility."""
    pricer = black if convention is Convention.LOGNORMAL else bachelier
    return sum(
        period.weight * pricer(period.forward, strike, volatility * period.factor, payoff)
        for period in periods
    )


def _value_on(
    periods: tuple[_Period, ...],
    strike: float,
    curve: CapletVolatility,
    payoff: Payoff,
    convention: Convention,
) -> float:
    """Value of the given periods, each at its own volatility from ``curve``."""
    pricer = black if convention is Convention.LOGNORMAL else bachelier
    return sum(
        period.weight
        * pricer(period.forward, strike, curve.volatility(period.fixing) * period.factor, payoff)
        for period in periods
    )


def strip_caplets(
    quotes: list[CapQuote] | tuple[CapQuote, ...],
    effective: date,
    strike: float,
    discount: DiscountCurve,
    projection: DiscountCurve,
    *,
    index: ForecastIndex | None = None,
    payoff: Payoff = Payoff.PAYER,
    convention: Convention = Convention.LOGNORMAL,
    vol_basis: Basis = Basis.ACT_365F,
) -> CapletVolatility:
    """Bootstrap a caplet volatility curve from flat cap quotes.

    Args:
        quotes: Flat volatilities, one per maturity. Sorted here; a maturity
            adding no new period is refused.
        effective: Start of every cap in the set.
        strike: The common strike. See the module docstring on why it is here
            and not on the quote.
        discount: Curve the premiums discount on.
        projection: Curve the forwards come from.
        index: The forecast index. Defaults to a fresh
            :class:`~tenor.multicurve.ForecastIndex`.
        payoff: ``PAYER`` for caps, ``RECEIVER`` for floors.
        convention: Lognormal or normal volatilities.
        vol_basis: Basis the volatility's time to fixing is measured on. Has to
            match whatever the quotes were made on, and matching
            :meth:`~tenor.options.Cap.value`'s own default is what makes a flat
            caplet curve quote flat.

    Raises:
        StrippingError: if the schedules do not nest, if a maturity adds no
            periods, or if a step's incremental premium is outside what a
            non-negative volatility under :data:`VOLATILITY_CEILING` attains.
    """
    if not quotes:
        raise StrippingError("a caplet curve needs at least one cap quote")
    ordered = sorted(quotes, key=lambda quote: quote.maturity)
    forecast = ForecastIndex() if index is None else index

    schedules = [
        _periods(
            _cap(effective, quote.maturity, strike, forecast, payoff),
            discount,
            projection,
            vol_basis,
        )
        for quote in ordered
    ]

    # Nesting. generate runs backward from maturity, so a shorter cap's periods
    # are a prefix of a longer one's only when every maturity sits on the roll
    # grid the longest one defines. Checked against the realised schedules.
    longest = schedules[-1]
    for quote, periods in zip(ordered, schedules, strict=True):
        prefix = longest[: len(periods)]
        if [period.fixing for period in periods] != [period.fixing for period in prefix]:
            raise StrippingError(
                f"the cap to {quote.maturity.isoformat()} has periods that do not "
                f"nest inside the cap to {ordered[-1].maturity.isoformat()}: "
                f"{[p.fixing.isoformat() for p in periods][-3:]} against "
                f"{[p.fixing.isoformat() for p in prefix][-3:]}. Caps strip only "
                "when their maturities sit on the index's own roll grid from the "
                "shared effective date."
            )

    buckets: list[CapletBucket] = []

    def step(periods: tuple[_Period, ...], covered: int, flat: float) -> float:
        """The volatility for periods ``covered:`` that reprices this quote.

        Monotone in the volatility, so a bracket on ``[0, ceiling]`` is all the
        solver needs and the ends of that bracket are also the test for whether
        the quote is attainable at all.
        """
        target = _value(periods, strike, flat, payoff, convention)
        settled = _value_on(
            periods[:covered],
            strike,
            CapletVolatility(strike=strike, buckets=tuple(buckets)),
            payoff,
            convention,
        )
        fresh = periods[covered:]
        incremental = target - settled
        floor = _value(fresh, strike, 0.0, payoff, convention)
        ceiling = _value(fresh, strike, VOLATILITY_CEILING, payoff, convention)
        tolerance = PREMIUM_TOLERANCE * max(abs(target), floor, 1.0e-300)
        if incremental < floor - tolerance or incremental > ceiling:
            raise StrippingError(
                f"the cap to {periods[-1].fixing.isoformat()} leaves "
                f"{incremental:.12g} for its {len(fresh)} new periods, which can "
                f"only be worth between {floor:.12g} at a zero volatility and "
                f"{ceiling:.12g} at {VOLATILITY_CEILING:g}. The quote is "
                "unattainable given the shorter quotes already fitted, not "
                "merely hard to solve."
            )
        if incremental <= floor:
            # The feasibility boundary itself, and it is reached exactly rather
            # than approached: the new periods are worth their intrinsic value
            # and the quote has nothing left to pay for optionality. Brent
            # would see no sign change here, so the boundary is returned
            # instead of searched for.
            return 0.0

        def residual(level: float) -> float:
            return _value(fresh, strike, level, payoff, convention) - incremental

        return brent(residual, 0.0, VOLATILITY_CEILING, tolerance=1e-15).value

    covered = 0
    for position, (quote, periods) in enumerate(zip(ordered, schedules, strict=True)):
        if len(periods) <= covered:
            raise StrippingError(
                f"the cap to {quote.maturity.isoformat()} adds no period the cap to "
                f"{ordered[position - 1].maturity.isoformat()} did not already "
                "cover, so its quote has nothing of its own to determine"
            )
        level = step(periods, covered, quote.volatility)
        bump = 1e-6
        bumped = step(periods, covered, quote.volatility + bump)
        buckets.append(
            CapletBucket(
                maturity=quote.maturity,
                first_fixing=periods[covered].fixing,
                last_fixing=periods[-1].fixing,
                periods=len(periods) - covered,
                volatility=level,
                quoted=quote.volatility,
                sensitivity=(bumped - level) / bump,
            )
        )
        covered = len(periods)

    return CapletVolatility(strike=strike, buckets=tuple(buckets))


def flat_volatility(
    curve: CapletVolatility,
    maturity: date,
    effective: date,
    discount: DiscountCurve,
    projection: DiscountCurve,
    *,
    index: ForecastIndex | None = None,
    payoff: Payoff = Payoff.PAYER,
    convention: Convention = Convention.LOGNORMAL,
    vol_basis: Basis = Basis.ACT_365F,
) -> float:
    """The flat volatility that reproduces the premium ``curve`` gives this cap.

    The inverse of one bootstrap step, and the round trip that checks the
    whole construction: a curve stripped from a quote set returns that set.

    Raises:
        StrippingError: if the premium is outside what a flat volatility under
            :data:`VOLATILITY_CEILING` attains.
    """
    forecast = ForecastIndex() if index is None else index
    cap = _cap(effective, maturity, curve.strike, forecast, payoff)
    periods = _periods(cap, discount, projection, vol_basis)
    premium = _value_on(periods, curve.strike, curve, payoff, convention)

    floor = _value(periods, curve.strike, 0.0, payoff, convention)
    ceiling = _value(periods, curve.strike, VOLATILITY_CEILING, payoff, convention)
    tolerance = PREMIUM_TOLERANCE * max(abs(premium), floor, 1.0e-300)
    if premium < floor - tolerance or premium > ceiling:
        raise StrippingError(
            f"a premium of {premium:.12g} is outside the range [{floor:.12g}, "
            f"{ceiling:.12g}] a flat volatility can reach for this cap"
        )
    if premium <= floor:
        return 0.0

    def residual(level: float) -> float:
        return _value(periods, curve.strike, level, payoff, convention) - premium

    return brent(residual, 0.0, VOLATILITY_CEILING, tolerance=1e-15).value
