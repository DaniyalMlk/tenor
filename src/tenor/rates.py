"""Compounding conventions: turning a discount factor into a rate and back.

A discount factor is unambiguous. A rate is not, and the same discount factor
quoted under different compounding gives visibly different numbers:

    P = 0.90 over five years

    simple        2.2222%
    annual        2.1296%
    semi-annual   2.1184%
    quarterly     2.1128%
    continuous    2.1072%

Eleven and a half basis points between the extremes at a five-year point, and
two and a quarter between annual and continuous alone. Neither is a rounding
difference; the first is wider than the bid-offer on most of the curve.

So a rate never travels through this library on its own — it is always
paired with the convention that produced it, and the pairing is a type
rather than a naming convention, because a naming convention is something
a caller can get wrong silently.

**Simple compounding is not a limiting case of the others.** It is
``1 / (1 + r t)`` rather than a power, which makes it the odd one out and also
makes it the market convention for anything under a year. Money market deposits
quote simple; swap curves quote annual or continuous; a library that treats
simple as "annual with n = 1" is wrong for every period that is not exactly one
year, and exactly right at the one point somebody is most likely to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Compounding(str, Enum):
    """How a rate accumulates over a period."""

    #: ``1 / (1 + r t)``. Money market deposits, and anything under a year.
    SIMPLE = "simple"
    #: ``(1 + r) ** -t``.
    ANNUAL = "annual"
    #: ``(1 + r/2) ** -2t``. Most government bond markets quote here.
    SEMI_ANNUAL = "semi-annual"
    #: ``(1 + r/4) ** -4t``.
    QUARTERLY = "quarterly"
    #: ``(1 + r/12) ** -12t``.
    MONTHLY = "monthly"
    #: ``exp(-r t)``. What curve mathematics is almost always written in.
    CONTINUOUS = "continuous"

    @property
    def periods(self) -> int | None:
        """Compounding periods per year, or ``None`` for simple and continuous.

        Both ends of the range are outside the family: simple is not a power at
        all, and continuous is its limit rather than a member of it.
        """
        return {
            Compounding.ANNUAL: 1,
            Compounding.SEMI_ANNUAL: 2,
            Compounding.QUARTERLY: 4,
            Compounding.MONTHLY: 12,
        }.get(self)


class BadRate(ValueError):
    """A rate or discount factor that cannot be converted."""


def discount_factor(rate: float, time: float, compounding: Compounding) -> float:
    """The present value of one unit paid in ``time`` years.

    ``time`` is a year fraction, so it comes from
    :func:`~tenor.daycount.year_fraction` and carries whichever day count basis
    was used. This function does not know or care which — but the caller must
    use the same one when going back, or the round trip does not close.
    """
    if time < 0.0:
        raise BadRate(f"a discount factor needs a non-negative time, got {time!r}")
    if time == 0.0:
        return 1.0
    if compounding is Compounding.SIMPLE:
        denominator = 1.0 + rate * time
        if denominator <= 0.0:
            raise BadRate(
                f"a simple rate of {rate!r} over {time!r} years gives a growth factor "
                f"of {denominator!r}. Simple compounding has no discount factor once "
                "the rate is below -1/t, because the money has gone past zero rather "
                "than merely shrunk."
            )
        return 1.0 / denominator
    if compounding is Compounding.CONTINUOUS:
        return math.exp(-rate * time)
    periods = compounding.periods
    assert periods is not None
    growth = 1.0 + rate / periods
    if growth <= 0.0:
        raise BadRate(
            f"a rate of {rate!r} compounded {periods} times a year gives a per-period "
            f"growth factor of {growth!r}, which is not positive"
        )
    return float(growth ** (-periods * time))


def rate_from_discount(
    factor: float, time: float, compounding: Compounding
) -> float:
    """The rate implied by a discount factor over ``time`` years.

    The exact inverse of :func:`discount_factor` under the same convention. That
    the two round-trip is not decoration: a curve stores one of them and quotes
    the other, so any drift between them shows up as a curve that does not
    reprice its own inputs.
    """
    if time <= 0.0:
        raise BadRate(
            f"a rate needs a strictly positive time, got {time!r}. At zero the "
            "discount factor is one whatever the rate is, so there is no rate to "
            "recover."
        )
    if factor <= 0.0:
        raise BadRate(
            f"a discount factor is strictly positive, got {factor!r}. A factor of "
            "zero says the payment is worthless at any rate and a negative one says "
            "it is a liability, and neither is a discounting problem."
        )
    if compounding is Compounding.SIMPLE:
        return (1.0 / factor - 1.0) / time
    if compounding is Compounding.CONTINUOUS:
        return -math.log(factor) / time
    periods = compounding.periods
    assert periods is not None
    return periods * (float(factor ** (-1.0 / (periods * time))) - 1.0)


def convert(
    rate: float, time: float, source: Compounding, target: Compounding
) -> float:
    """The rate under ``target`` that discounts identically to ``rate``.

    Routed through the discount factor rather than through a closed form per
    pair. There are six conventions here and so thirty ordered pairs; writing
    each one out is thirty chances to transpose a sign, against one shared path
    that is already tested in both directions.
    """
    if source is target:
        return rate
    return rate_from_discount(discount_factor(rate, time, source), time, target)


@dataclass(frozen=True)
class Rate:
    """A rate that carries its own compounding convention.

    The point of the type is that a bare float cannot be misread. A 2.11%
    continuous rate and a 2.13% annual rate are the same rate, and a function
    taking ``float`` has no way to tell which it was handed.
    """

    value: float
    compounding: Compounding

    def discount(self, time: float) -> float:
        return discount_factor(self.value, time, self.compounding)

    def to(self, compounding: Compounding, time: float) -> Rate:
        """The same rate expressed under another convention.

        ``time`` is required because the conversion depends on it for every pair
        involving simple compounding — a simple rate's equivalent continuous
        rate is not a fixed multiple of it — and requiring it always is better
        than requiring it sometimes.
        """
        return Rate(convert(self.value, time, self.compounding, compounding), compounding)

    def __str__(self) -> str:
        return f"{self.value:.6%} {self.compounding.value}"


def forward_rate(
    near_factor: float,
    far_factor: float,
    near_time: float,
    far_time: float,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> float:
    """The rate implied between two dates by their discount factors.

    The ratio ``P(near) / P(far)`` is the growth factor over the interval, which
    is then read as a rate under ``compounding`` over the interval's own length.

    Worth stating because the forward is a rate over ``far - near``, not over
    ``far``. Reading it against the wrong span is the standard way a forward
    curve comes out looking plausible and scaled wrongly, and the error
    disappears entirely at ``near = 0``, which is where an example is most
    likely to be checked.
    """
    if far_time <= near_time:
        raise BadRate(
            f"a forward rate runs from {near_time!r} to {far_time!r}, which is not "
            "forwards in time"
        )
    if near_factor <= 0.0 or far_factor <= 0.0:
        raise BadRate("discount factors are strictly positive")
    span = far_time - near_time
    return rate_from_discount(far_factor / near_factor, span, compounding)


__all__ = [
    "BadRate",
    "Compounding",
    "Rate",
    "convert",
    "discount_factor",
    "forward_rate",
    "rate_from_discount",
]
