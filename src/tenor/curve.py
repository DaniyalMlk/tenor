"""Discount curves: a set of pillar dates, and what happens between them.

A curve is a handful of observed points and a rule for the gaps. The points come
from traded instruments and are not in question; the rule is an assumption, and
different assumptions produce visibly different forward rates from identical
inputs. So the interpolation is named on the curve, and a curve will say which
one it is rather than leaving it to be inferred from the shape.

**The two methods here are not two flavours of the same thing.**

*Log-linear on discount factors* is linear in ``log P``, which makes the
instantaneous forward rate *piecewise constant* — flat within each pillar
interval and jumping at the pillars. That is unrealistic as a picture of the
market and extremely well behaved as arithmetic: the forwards are guaranteed
positive whenever the discount factors are decreasing, which is exactly the
condition for the curve to be arbitrage-free in the first place. It is the
method to use when the answer has to be defensible rather than smooth.

*Linear on zero rates* is the one most people reach for, and it does something
the name does not advertise. Interpolating ``z(t)`` linearly makes ``z(t) * t``
quadratic, and the instantaneous forward is its derivative — so the forwards are
*piecewise linear with jumps at the pillars*, and they can go negative from
inputs that are themselves perfectly arbitrage-free.

Two pillars are enough to show it. Take ``z(1y) = 6%`` and ``z(2y) = 3.5%``.
The discount factors are 0.941765 and 0.932394, strictly decreasing, so the
inputs admit no arbitrage and the average forward over the year is a positive
1%. Log-linear returns exactly that 1%, flat across the interval. Linear on zero
rates returns a ramp from +3% just after the first pillar down to **-1%** just
before the second, crossing zero around 1.7 years — a negative rate manufactured
entirely by the interpolation, from data that contained none.

The curve reports its own instantaneous forwards, so this is inspectable rather
than theoretical, and both numbers above are in the test suite.

*Monotone convex* is the third option and declines the trade-off. It
interpolates the forward rate directly, under the constraint that it must
average back to the discrete forward over each interval, so the pillars are
repriced by an identity rather than by luck, the forward curve is continuous
across them, and the positivity that log-linear gets for free is imposed
explicitly instead of hoped for. On the counterexample above it returns forwards
that stay non-negative across the whole interval. The construction is in
:mod:`tenor.monotone`; what it costs is that a pillar set implying a negative
discrete forward is refused rather than interpolated, because the positivity
constraint has no meaning there.

**Extrapolation is refused.** Past the last pillar there is no information, and
returning the last value is not a neutral default — it is the assertion that all
forward rates beyond the curve are zero, which is a strong opinion stated
silently. A caller who wants a flat extension can add a pillar and say so.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from functools import cached_property

from .daycount import Basis, year_fraction
from .monotone import MonotoneConvex, NotMonotone
from .rates import BadRate, Compounding, rate_from_discount


class Interpolation(str, Enum):
    """The rule for the gaps between pillars."""

    #: Linear in ``log P``. Piecewise-constant forwards, positive whenever the
    #: discount factors are decreasing.
    LOG_LINEAR_DISCOUNT = "log-linear on discount factors"
    #: Linear in the zero rate. Piecewise-linear forwards that can go negative
    #: between two positive zero rates.
    LINEAR_ZERO = "linear on zero rates"
    #: Hagan-West. Continuous forwards, held non-negative, averaging back to the
    #: discrete forward over every interval.
    MONOTONE_CONVEX = "monotone convex"


class OffCurve(ValueError):
    """A date the curve has no information about."""


class BadCurve(ValueError):
    """The pillars do not form a curve."""


@dataclass(frozen=True)
class Pillar:
    """One observed point on the curve."""

    day: date
    discount: float
    time: float

    @property
    def is_anchor(self) -> bool:
        return self.time == 0.0


@dataclass(frozen=True)
class DiscountCurve:
    """Discount factors at a set of dates, and a rule for everything between.

    Built through :meth:`from_discounts` or :meth:`from_zeros` rather than
    directly, because both need to compute year fractions from the reference
    date and validate ordering before anything is usable.
    """

    reference: date
    pillars: tuple[Pillar, ...]
    basis: Basis
    interpolation: Interpolation

    # -- construction --------------------------------------------------------

    @classmethod
    def from_discounts(
        cls,
        reference: date,
        points: Sequence[tuple[date, float]],
        *,
        basis: Basis,
        interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    ) -> DiscountCurve:
        """Build from observed discount factors.

        The reference date is added as a pillar with a discount factor of one if
        it is not already present. Without it the curve has no left-hand end and
        cannot price anything before its first traded point, which is most of
        the money market.
        """
        if not points:
            raise BadCurve("a curve needs at least one pillar")
        gathered: list[Pillar] = []
        seen: set[date] = set()
        for day, factor in points:
            if day < reference:
                raise BadCurve(
                    f"pillar {day.isoformat()} is before the reference date "
                    f"{reference.isoformat()}; a discount curve runs forwards"
                )
            if day in seen:
                raise BadCurve(f"pillar {day.isoformat()} appears twice")
            if factor <= 0.0:
                raise BadCurve(
                    f"pillar {day.isoformat()} has a discount factor of {factor!r}; "
                    "discount factors are strictly positive"
                )
            seen.add(day)
            gathered.append(
                Pillar(
                    day=day,
                    discount=factor,
                    time=year_fraction(reference, day, basis),
                )
            )
        if reference not in seen:
            gathered.append(Pillar(day=reference, discount=1.0, time=0.0))
        gathered.sort(key=lambda one: one.day)
        anchor = gathered[0]
        if anchor.time == 0.0 and anchor.discount != 1.0:
            raise BadCurve(
                f"the reference date has a discount factor of {anchor.discount!r}; "
                "one unit today is worth one unit"
            )
        return cls(
            reference=reference,
            pillars=tuple(gathered),
            basis=basis,
            interpolation=interpolation,
        )

    @classmethod
    def from_zeros(
        cls,
        reference: date,
        points: Sequence[tuple[date, float]],
        *,
        basis: Basis,
        compounding: Compounding = Compounding.CONTINUOUS,
        interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    ) -> DiscountCurve:
        """Build from observed zero rates under ``compounding``.

        The rates are converted to discount factors once, here, and the curve
        stores only those. Storing rates and converting on every query would
        make the stored convention part of every later answer, and a curve built
        from semi-annual quotes would give different numbers from the identical
        curve built from continuous ones through nothing but accumulated
        rounding.
        """
        from .rates import discount_factor

        converted = []
        for day, rate in points:
            time = year_fraction(reference, day, basis)
            converted.append(
                (day, 1.0 if time == 0.0 else discount_factor(rate, time, compounding))
            )
        return cls.from_discounts(
            reference, converted, basis=basis, interpolation=interpolation
        )

    # -- the shape ------------------------------------------------------------

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(one.time for one in self.pillars)

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(one.day for one in self.pillars)

    @property
    def horizon(self) -> date:
        """The last date the curve knows anything about."""
        return self.pillars[-1].day

    def __len__(self) -> int:
        return len(self.pillars)

    def time_to(self, day: date) -> float:
        """The year fraction from the reference date, under the curve's basis."""
        if day < self.reference:
            raise OffCurve(
                f"{day.isoformat()} is before the curve's reference date "
                f"{self.reference.isoformat()}"
            )
        return year_fraction(self.reference, day, self.basis)

    # -- querying -------------------------------------------------------------

    def discount(self, day: date) -> float:
        """The discount factor to ``day``.

        Refuses anything past the last pillar. Returning the final factor would
        assert that every forward rate beyond the curve is zero, which is a
        strong opinion to state by omission.
        """
        if day > self.horizon:
            raise OffCurve(
                f"{day.isoformat()} is past the curve's last pillar "
                f"{self.horizon.isoformat()}. The curve has no information there, and "
                "holding the last discount factor flat would assert that every "
                "forward rate beyond it is zero. Add a pillar if that is what you "
                "mean."
            )
        return self.discount_at(self.time_to(day))

    def discount_at(self, time: float) -> float:
        """The discount factor at a year fraction, interpolated.

        Separate from :meth:`discount` because the risk and bootstrapping code
        works in year fractions and converting back to a date to look something
        up would reintroduce the rounding the year fraction just removed.
        """
        if time < 0.0:
            raise OffCurve(f"a discount factor needs a non-negative time, got {time!r}")
        times = self.times
        if time > times[-1] + _TOLERANCE:
            raise OffCurve(
                f"time {time!r} is past the curve's last pillar at {times[-1]!r}"
            )
        index = bisect.bisect_left(times, time)
        if index < len(times) and abs(times[index] - time) <= _TOLERANCE:
            return self.pillars[index].discount
        if index == 0:
            return 1.0
        if self.interpolation is Interpolation.MONOTONE_CONVEX:
            # Unlike the other two, this scheme is not local: the shape inside
            # one interval depends on the neighbouring discrete forwards, so the
            # whole pillar set is fitted once and cached on the curve.
            return self.monotone.discount_at(time)
        left, right = self.pillars[index - 1], self.pillars[index]
        weight = (time - left.time) / (right.time - left.time)
        if self.interpolation is Interpolation.LOG_LINEAR_DISCOUNT:
            # Linear in log P, so the forward is flat across the interval.
            return math.exp(
                (1.0 - weight) * math.log(left.discount)
                + weight * math.log(right.discount)
            )
        # Linear in the zero rate. The left pillar may be the anchor, where the
        # zero rate is undefined rather than zero — the discount factor is one
        # whatever the rate — so the interpolation runs from the next known rate
        # instead of from a fabricated zero.
        right_rate = rate_from_discount(right.discount, right.time, Compounding.CONTINUOUS)
        if left.time == 0.0:
            rate = right_rate
        else:
            left_rate = rate_from_discount(
                left.discount, left.time, Compounding.CONTINUOUS
            )
            rate = (1.0 - weight) * left_rate + weight * right_rate
        return math.exp(-rate * time)

    @cached_property
    def monotone(self) -> MonotoneConvex:
        """The fitted monotone convex interpolant over this curve's pillars.

        Fitted on first use and kept, because the fit is global — every
        interval's shape depends on its neighbours — and a curve is queried far
        more often than it is built. Bootstrapping in particular evaluates a
        trial curve at every instrument on every iteration.

        Available whatever the curve's own interpolation is, which is what lets
        a caller compare two schemes over identical pillars without rebuilding
        anything.
        """
        if len(self.pillars) < 2:
            raise BadCurve(
                "monotone convex needs at least one interval, and this curve has "
                "only its anchor"
            )
        try:
            return MonotoneConvex.fit(
                self.times,
                tuple(-math.log(one.discount) for one in self.pillars),
            )
        except NotMonotone as bad:
            raise BadCurve(str(bad)) from bad

    def zero_rate(
        self, day: date, compounding: Compounding = Compounding.CONTINUOUS
    ) -> float:
        """The zero rate to ``day`` under ``compounding``."""
        time = self.time_to(day)
        if time <= 0.0:
            raise OffCurve(
                "there is no zero rate at the reference date: the discount factor is "
                "one whatever the rate is, so nothing is implied"
            )
        return rate_from_discount(self.discount(day), time, compounding)

    def zero_rate_at(
        self, time: float, compounding: Compounding = Compounding.CONTINUOUS
    ) -> float:
        if time <= 0.0:
            raise OffCurve("there is no zero rate at the reference date")
        return rate_from_discount(self.discount_at(time), time, compounding)

    def forward_rate(
        self,
        near: date,
        far: date,
        compounding: Compounding = Compounding.SIMPLE,
    ) -> float:
        """The rate the curve implies between two future dates.

        Simple compounding by default, because a forward rate is nearly always
        quoted against a money market instrument and those are simple.
        """
        near_time, far_time = self.time_to(near), self.time_to(far)
        if far_time <= near_time:
            raise OffCurve(
                f"a forward runs from {near.isoformat()} to {far.isoformat()}, which "
                "is not forwards in time"
            )
        return rate_from_discount(
            self.discount(far) / self.discount(near),
            far_time - near_time,
            compounding,
        )

    def instantaneous_forward(self, time: float, *, bump: float = 1e-6) -> float:
        """``-d log P / dt``, the limiting forward rate at ``time``.

        Computed by a central difference, which is the honest way to get it from
        an interpolation whose analytic derivative is different in each scheme
        and discontinuous at the pillars. The bump is one-millionth of a year —
        about thirty seconds — so it stays inside a pillar interval everywhere
        the answer is well defined.

        This is the function that reveals what an interpolation is actually
        doing. Log-linear gives a step function; linear-on-zero gives a sawtooth
        that can dip below zero.

        Monotone convex is the exception: its forward rate is the quantity it
        interpolates, so it is returned analytically rather than differenced.
        The two agree to the accuracy of the difference away from the pillars,
        which the tests check rather than assert in a comment.
        """
        if self.interpolation is Interpolation.MONOTONE_CONVEX:
            return self.monotone.forward_at(min(max(time, 0.0), self.times[-1]))
        low = max(time - bump, 0.0)
        high = min(time + bump, self.times[-1])
        if high <= low:
            raise OffCurve(f"no interval around {time!r} lies on the curve")
        return -(math.log(self.discount_at(high)) - math.log(self.discount_at(low))) / (
            high - low
        )

    def reprices_pillars(self, tolerance: float = 1e-12) -> bool:
        """Whether querying the curve at its own pillars returns their factors.

        Trivially true of any sane interpolation and worth asserting anyway: a
        curve that does not reproduce its own inputs is not a curve through
        them, and an off-by-one in the interval search is exactly the kind of
        mistake that leaves it nearly right.
        """
        for pillar in self.pillars:
            if abs(self.discount_at(pillar.time) - pillar.discount) > tolerance:
                return False
        return True

    def with_interpolation(self, interpolation: Interpolation) -> DiscountCurve:
        """The same pillars under a different rule for the gaps."""
        return DiscountCurve(
            reference=self.reference,
            pillars=self.pillars,
            basis=self.basis,
            interpolation=interpolation,
        )


#: Year fractions are computed rather than stored, so a pillar looked up by its
#: own date can miss by a rounding error. A microsecond of a year is far below
#: any real spacing and far above the rounding.
_TOLERANCE = 1e-12


def flat_curve(
    reference: date,
    horizon: date,
    rate: float,
    *,
    basis: Basis,
    compounding: Compounding = Compounding.CONTINUOUS,
) -> DiscountCurve:
    """A curve at a single rate, for tests and for sanity checks.

    Two pillars is enough: under either interpolation a constant rate is
    reproduced exactly everywhere between them, and a curve that does not manage
    that is broken in a way worth finding before anything harder is attempted.
    """
    if horizon <= reference:
        raise BadCurve("a flat curve needs a horizon after its reference date")
    from .rates import discount_factor

    time = year_fraction(reference, horizon, basis)
    return DiscountCurve.from_discounts(
        reference,
        [(horizon, discount_factor(rate, time, compounding))],
        basis=basis,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )


__all__ = [
    "BadCurve",
    "BadRate",
    "DiscountCurve",
    "Interpolation",
    "OffCurve",
    "Pillar",
    "flat_curve",
]
