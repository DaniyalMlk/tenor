"""Breaking a single duration into where on the curve the risk actually is.

A duration is the answer to one question: what happens if the whole curve moves
together. That is the one thing curves reliably do not do. A ten-year bond and
a barbell of twos and thirties can have the same duration and behave nothing
alike, and no amount of precision in the duration will say so.

The breakdown here comes in two kinds, which are not the same numbers grouped
differently.

**Key rate durations** shift the *zero curve*, in a shape localised around one
bucket, and measure the price response. They answer "where on the curve is this
exposed", and they are the natural decomposition of a parallel shift because
their shapes are built to add back up to it.

**Instrument risk** shifts a *quote* — a deposit rate, a futures price, a swap
rate — rebuilds the curve from scratch, and measures the response to that. It
answers "what do I trade to hedge this", which is a different question with a
different answer. A ten-year swap quote does not move the ten-year zero rate
alone; it moves every zero rate out to ten years, because the quote is a
statement about the whole annuity. So the ten-year instrument risk of a bond is
not its ten-year key rate duration, and treating them as interchangeable
produces a hedge that is right in total and wrong everywhere.

**The sum identity is a property of the shapes, not a law.** Triangular shifts
falling to zero at the neighbouring buckets add to one *between* the first and
last bucket and to less than one outside them. Held flat past the two ends they
add to one everywhere, and only then do the key rate durations sum to the total.
Get that wrong and the sum is short at both ends by an amount that looks like a
rounding error and is not one. :func:`tent_weights` holds the ends flat, and
the tests assert the sum numerically rather than inferring it from the
construction.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .bootstrap import Bootstrapped, bootstrap
from .curve import DiscountCurve, Pillar
from .instruments import Deposit, Future, Instrument, Swap
from .rates import Compounding, discount_factor, rate_from_discount

#: A shift shape: given a year fraction, how much of the shift applies there.
Weighting = Callable[[float], float]

#: Anything that turns a curve into a number. A bond's price, a portfolio's
#: value, a swap's mark — the risk functions here do not care which.
Valuation = Callable[[DiscountCurve], float]


class BadRisk(ValueError):
    """A risk decomposition that does not describe anything."""


def shifted_by(
    curve: DiscountCurve,
    amount: float,
    weight: Weighting,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> DiscountCurve:
    """``curve`` with each pillar's zero rate moved by ``amount * weight(t)``.

    The general form of :meth:`DiscountCurve.shifted`, which is this with a
    weight of one everywhere.

    The shift lands on the pillars, not on a continuum, which is what makes the
    sum identity exact rather than approximate: if the weights add to one at
    every pillar then applying them one at a time and adding the *shifts* gives
    back precisely the parallel shift, with no appeal to what the interpolation
    does between pillars. What is left over in the summed *durations* is then
    the pricing function's own curvature and nothing else.
    """
    moved = []
    for one in curve.pillars:
        if one.time == 0.0:
            moved.append(one)
            continue
        step = amount * weight(one.time)
        if compounding is Compounding.CONTINUOUS:
            factor = one.discount * math.exp(-step * one.time)
        else:
            rate = rate_from_discount(one.discount, one.time, compounding)
            factor = discount_factor(rate + step, one.time, compounding)
        moved.append(Pillar(day=one.day, discount=factor, time=one.time))
    return DiscountCurve(
        reference=curve.reference,
        pillars=tuple(moved),
        basis=curve.basis,
        interpolation=curve.interpolation,
    )


# -- named shapes -------------------------------------------------------------


def level() -> Weighting:
    """One everywhere. The parallel shift, written as a shape for symmetry."""
    return lambda _: 1.0


def slope(horizon: float) -> Weighting:
    """Minus one at the front, plus one at ``horizon``, linear between.

    Normalised so the *ends* move by the full amount and the midpoint does not
    move at all, which makes a slope shift of a basis point mean "one basis
    point of steepening either side of the middle" — two basis points across
    the curve. The other common normalisation puts the full amount across the
    whole curve, so the same number means half as much. Neither is wrong and
    they differ by a factor of two, which is why this one says which it is.
    """
    if horizon <= 0.0:
        raise BadRisk(f"a slope shift needs a positive horizon, got {horizon!r}")
    return lambda time: 2.0 * time / horizon - 1.0


def curvature(horizon: float) -> Weighting:
    """Plus one at both ends, minus one in the middle. A butterfly.

    The wings up and the body down, so a positive curvature shift is the curve
    becoming more humped. Normalised the same way as :func:`slope`: the extreme
    points move by the full amount.
    """
    if horizon <= 0.0:
        raise BadRisk(f"a curvature shift needs a positive horizon, got {horizon!r}")
    return lambda time: 2.0 * (2.0 * time / horizon - 1.0) ** 2 - 1.0


# -- key rates ----------------------------------------------------------------


@dataclass(frozen=True)
class Bucket:
    """A point on the curve that risk is attributed to."""

    time: float
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"{self.time:g}y"


@dataclass(frozen=True)
class KeyRate:
    """One bucket's share of the risk."""

    bucket: Bucket
    #: ``-1/P dP/dr`` under this bucket's shift alone, in years.
    duration: float
    #: The money version, per 100 of whatever the valuation returns.
    value: float


def tent_weights(times: Sequence[float]) -> tuple[Weighting, ...]:
    """Triangular shapes over ``times`` that add to exactly one everywhere.

    Each shape is one at its own bucket and falls linearly to zero at the
    neighbouring buckets. The first is held flat at one for everything before
    it and the last flat at one for everything after, which is the part that
    makes them a partition: without it the shapes sum to less than one outside
    the bucket range, the key rate durations come up short, and the shortfall
    is largest for exactly the long-dated exposure the decomposition exists to
    find.
    """
    if not times:
        raise BadRisk("a key rate decomposition needs at least one bucket")
    ordered = list(times)
    for index in range(1, len(ordered)):
        if ordered[index] <= ordered[index - 1]:
            raise BadRisk(
                f"buckets must increase, but {ordered[index]!r} does not follow "
                f"{ordered[index - 1]!r}"
            )
    if len(ordered) == 1:
        return (level(),)

    def make(index: int) -> Weighting:
        here = ordered[index]
        before = ordered[index - 1] if index else None
        after = ordered[index + 1] if index + 1 < len(ordered) else None

        def weight(time: float) -> float:
            if before is None and time <= here:
                return 1.0
            if after is None and time >= here:
                return 1.0
            if before is not None and before < time <= here:
                return (time - before) / (here - before)
            if after is not None and here <= time < after:
                return (after - time) / (after - here)
            return 0.0

        return weight

    return tuple(make(index) for index in range(len(ordered)))


def shape_duration(
    value: Valuation,
    curve: DiscountCurve,
    weight: Weighting,
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> float:
    """``-1/P dP/dr`` under one shift shape, by central difference."""
    base = value(curve)
    if base == 0.0:
        raise BadRisk("a duration is a relative sensitivity, and this is worth nothing")
    up = value(shifted_by(curve, shift, weight, compounding))
    down = value(shifted_by(curve, -shift, weight, compounding))
    return (down - up) / (2.0 * base * shift)


def key_rates(
    value: Valuation,
    curve: DiscountCurve,
    buckets: Sequence[Bucket],
    *,
    shift: float = 1e-4,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> tuple[KeyRate, ...]:
    """Attribute the sensitivity across ``buckets``.

    Each bucket gets a triangular shift of its own and the price response to
    it. Because the shapes add to one at every pillar, these add back to the
    total duration — to the accuracy of the central difference, which is what
    the tests measure rather than assume.
    """
    weights = tent_weights([one.time for one in buckets])
    base = value(curve)
    results = []
    for bucket, weight in zip(buckets, weights, strict=True):
        duration = shape_duration(
            value, curve, weight, shift=shift, compounding=compounding
        )
        results.append(
            KeyRate(bucket=bucket, duration=duration, value=duration * base * shift)
        )
    return tuple(results)


def buckets_from(curve: DiscountCurve, years: Sequence[float]) -> tuple[Bucket, ...]:
    """Buckets at the given tenors, refusing any the curve cannot carry."""
    horizon = curve.times[-1]
    made = []
    for one in years:
        if one <= 0.0 or one > horizon:
            raise BadRisk(
                f"a bucket at {one!r} years is outside the curve, which runs to "
                f"{horizon:.4f}. A shift applied where the curve has no pillars "
                "moves nothing, and a key rate duration of zero there would look "
                "like an absence of risk rather than an absence of curve."
            )

        made.append(Bucket(time=one))
    return tuple(made)


# -- instrument risk ----------------------------------------------------------


@dataclass(frozen=True)
class InstrumentRisk:
    """What one quote moving by a basis point does to a valuation."""

    instrument: Instrument
    #: The change in value from raising this quote's rate by ``shift``.
    value: float

    @property
    def name(self) -> str:
        return self.instrument.name


def bumped(instrument: Instrument, shift: float) -> Instrument:
    """The same instrument quoted ``shift`` higher, in rate terms.

    A future is quoted as ``100 - rate``, so raising its rate lowers its price.
    Getting that backwards produces a hedge of the right size pointing the
    wrong way, which is worse than no hedge, and it is the single easiest sign
    error to make in this file.
    """
    if isinstance(instrument, Future):
        return replace(instrument, price=instrument.price - 100.0 * shift)
    if isinstance(instrument, Deposit | Swap):
        return replace(instrument, rate=instrument.rate + shift)
    raise BadRisk(  # pragma: no cover - the union above is closed
        f"no way to bump the quote of {instrument!r}"
    )


def instrument_risk(
    value: Valuation,
    built: Bootstrapped,
    *,
    shift: float = 1e-4,
) -> tuple[InstrumentRisk, ...]:
    """The value change from moving each input quote by a basis point in turn.

    The curve is rebuilt from scratch for each bump, which is the point: a swap
    quote is a statement about its whole annuity, so moving it moves every zero
    rate out to its maturity. That is why this decomposition differs from the
    key rate one, and why it is the one a hedge is put on in — the numbers are
    in the units of the things actually traded.

    Rebuilding is also why this is the expensive one. Each bump is a full
    bootstrap, which under monotone convex is several sweeps of root finds.
    """
    reference = built.curve.reference
    results = []
    for index, instrument in enumerate(built.instruments):
        quotes = list(built.instruments)
        quotes[index] = bumped(instrument, shift)
        rebuilt = bootstrap(
            reference,
            quotes,
            basis=built.curve.basis,
            interpolation=built.curve.interpolation,
        )
        results.append(
            InstrumentRisk(
                instrument=instrument,
                value=value(built.curve) - value(rebuilt.curve),
            )
        )
    return tuple(results)


def total_instrument_risk(
    value: Valuation, built: Bootstrapped, *, shift: float = 1e-4
) -> float:
    """The value change from moving every quote at once.

    The thing the individual instrument risks have to add up to, and the reason
    to compute it separately rather than by summing them: agreement between two
    routes is evidence, and a sum checked against itself is not.
    """
    reference = built.curve.reference
    rebuilt = bootstrap(
        reference,
        [bumped(one, shift) for one in built.instruments],
        basis=built.curve.basis,
        interpolation=built.curve.interpolation,
    )
    return value(built.curve) - value(rebuilt.curve)


def nearest_bucket(buckets: Sequence[Bucket], time: float) -> Bucket:
    """The bucket a given tenor sits closest to. For reporting, not for risk."""
    if not buckets:
        raise BadRisk("there are no buckets")
    times = [one.time for one in buckets]
    index = bisect.bisect_left(times, time)
    if index == 0:
        return buckets[0]
    if index == len(buckets):
        return buckets[-1]
    before, after = buckets[index - 1], buckets[index]
    return before if time - before.time <= after.time - time else after


__all__ = [
    "BadRisk",
    "Bucket",
    "InstrumentRisk",
    "KeyRate",
    "Valuation",
    "Weighting",
    "buckets_from",
    "bumped",
    "curvature",
    "instrument_risk",
    "key_rates",
    "level",
    "nearest_bucket",
    "shape_duration",
    "shifted_by",
    "slope",
    "tent_weights",
    "total_instrument_risk",
]
