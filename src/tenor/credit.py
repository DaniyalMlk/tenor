"""Default risk: a survival curve, credit default swaps, and risky bonds.

Everything else in this package discounts a cashflow that arrives. This module
prices the possibility that it does not.

The object underneath is a **survival curve**: the probability that the
reference entity is still paying at each future date. It is held as a
piecewise-constant forward hazard rate, for the same reason a discount curve is
held as discount factors rather than as yields — the quantity that is actually
piecewise-anything is the forward, and interpolating the level instead puts
kinks in the forward where nothing happened.

Three decisions are worth stating.

**The integrals are closed form, not quadrature.** A protection leg is
``(1 - R) integral DF(s) dQ(s)`` where ``Q`` is the default probability, and
the temptation is to put a fine grid under it. There is no need. On any
interval where the hazard rate and the instantaneous forward rate are both
constant, ``DF(s) S(s) = DF(a) S(a) exp(-(r + h)(s - a))`` and the integral is
elementary. Taking the grid to be the union of the curve's own pillars, the
hazard pillars and the coupon dates makes both constant on every piece, and the
answer is then exact for the curve as interpolated rather than convergent to
it. The accrual-on-default term, which is that integral weighted by the time
since the last coupon, is elementary for the same reason and is computed rather
than approximated by a half-period.

**The recovery rate belongs to the curve, not to the trade.** A hazard rate
stripped from quoted spreads is only meaningful beside the recovery assumed
while stripping it: the two are very nearly a product, and a curve that has
forgotten which recovery produced it will silently misprice anything struck on
a different one. So :class:`SurvivalCurve` carries it and refuses to be used
with another.

**The credit triangle is provided and measured.** ``spread = hazard times one
minus recovery`` is the rule of thumb everybody uses, and it is wrong in a
direction and by an amount worth knowing. :func:`triangle_hazard` implements
it, :func:`implied_hazard` solves the real thing, and the README carries the
gap between them across the surface where it matters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from .bond import Bond
from .calendar import Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .schedule import Frequency, Period, Schedule, generate
from .solve import solve

__all__ = [
    "BadCredit",
    "CreditDefaultSwap",
    "HazardPillar",
    "SurvivalCurve",
    "UpfrontSettlement",
    "bootstrap_hazards",
    "implied_hazard",
    "risky_bond_price",
    "triangle_hazard",
    "triangle_spread",
]

_TOLERANCE = 1e-12

# Below this the exponential integrals lose their leading digits to
# cancellation and the series expansion is both faster and more accurate. The
# crossover is where the cubic term of the expansion falls under the rounding
# of the value itself.
_SMALL = 1e-8


class BadCredit(ValueError):
    """A survival curve or a swap that does not describe anything."""


def _expm1_ratio(x: float) -> float:
    """``(1 - exp(-x)) / x``, accurate at zero, where it is 1.

    The integral of a decaying exponential over an interval reduces to this,
    and every protection leg here is a sum of such integrals. Written with
    ``expm1`` so the numerator keeps its digits; the branch at zero is for a
    zero-length interval or a zero hazard, both of which occur.
    """
    if abs(x) < _SMALL:
        # 1 - x/2 + x^2/6, whose next term is below the rounding of the result.
        return 1.0 - 0.5 * x + x * x / 6.0
    return -math.expm1(-x) / x


def _ramp_integral(x: float) -> float:
    """``(1 - exp(-x)(1 + x)) / x^2``, the accrual weight, accurate at zero.

    This is ``integral_0^1 u exp(-x u) du`` times ``x``-free scaling, and it is
    what turns "accrued since the last coupon" into a closed form. At zero it
    is one half, which is the half-period approximation everyone writes instead
    — correct only in the limit of no discounting and no hazard.
    """
    if abs(x) < _SMALL:
        return 0.5 - x / 3.0 + x * x / 8.0
    return (1.0 - math.exp(-x) * (1.0 + x)) / (x * x)


@dataclass(frozen=True)
class HazardPillar:
    """A forward hazard rate in force up to ``day``."""

    day: date
    time: float
    hazard: float


@dataclass(frozen=True)
class SurvivalCurve:
    """Survival probabilities, as a piecewise-constant forward hazard rate.

    The hazard in ``pillars[i]`` applies from ``pillars[i - 1].time`` to
    ``pillars[i].time``, and the last one extends beyond the final pillar. That
    flat extrapolation is a choice and it is the market's: a five-year quote
    says nothing about year seven, and extending the last hazard at least keeps
    the survival probability monotone and the implied spread flat rather than
    drifting somewhere nobody asked for.

    Attributes:
        reference: The curve's today.
        pillars: Forward hazard rates, in increasing date order.
        basis: Day count used to turn dates into year fractions.
        recovery: Fraction of face recovered on default, in [0, 1). Carried
            here because a hazard rate stripped from spreads means nothing
            without it.
    """

    reference: date
    pillars: tuple[HazardPillar, ...]
    basis: Basis
    recovery: float

    def __post_init__(self) -> None:
        if not self.pillars:
            raise BadCredit("a survival curve needs at least one hazard pillar")
        if not (0.0 <= self.recovery < 1.0):
            raise BadCredit(
                f"recovery is {self.recovery!r}; it is a fraction of face in [0, 1). "
                "A recovery of one means default costs nothing and the spread is "
                "zero whatever the hazard rate, which is not a curve."
            )
        previous = 0.0
        for pillar in self.pillars:
            if pillar.hazard < 0.0:
                raise BadCredit(
                    f"the hazard rate to {pillar.day.isoformat()} is "
                    f"{pillar.hazard!r}; a negative hazard makes survival rise"
                )
            if not math.isfinite(pillar.hazard):
                raise BadCredit(f"the hazard rate to {pillar.day.isoformat()} is not finite")
            if pillar.time <= previous:
                raise BadCredit(
                    f"pillar {pillar.day.isoformat()} at {pillar.time!r} does not "
                    f"come after {previous!r}; pillars run forwards and are distinct"
                )
            previous = pillar.time

    @classmethod
    def from_hazards(
        cls,
        reference: date,
        points: Sequence[tuple[date, float]],
        *,
        basis: Basis,
        recovery: float,
    ) -> SurvivalCurve:
        """Build from ``(date, forward hazard)`` pairs."""
        if not points:
            raise BadCredit("a survival curve needs at least one hazard pillar")
        pillars = []
        for day, hazard in points:
            if day <= reference:
                raise BadCredit(
                    f"pillar {day.isoformat()} is not after the reference date "
                    f"{reference.isoformat()}"
                )
            pillars.append(
                HazardPillar(day=day, time=year_fraction(reference, day, basis), hazard=hazard)
            )
        pillars.sort(key=lambda one: one.day)
        return cls(
            reference=reference, pillars=tuple(pillars), basis=basis, recovery=recovery
        )

    @classmethod
    def flat(
        cls, reference: date, maturity: date, hazard: float, *, basis: Basis, recovery: float
    ) -> SurvivalCurve:
        """A single hazard rate, extended forever."""
        return cls.from_hazards(
            reference, [(maturity, hazard)], basis=basis, recovery=recovery
        )

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(pillar.time for pillar in self.pillars)

    def time_to(self, day: date) -> float:
        """The year fraction from the reference date, on the curve's own basis."""
        return year_fraction(self.reference, day, self.basis)

    def hazard_at(self, time: float) -> float:
        """The forward hazard rate in force at ``time``.

        Right-continuous: exactly at a pillar the rate *starting* there is
        returned, which is the one that governs the next interval and is what
        every integral below wants.
        """
        if time < 0.0:
            raise BadCredit(f"a hazard rate needs a non-negative time, got {time!r}")
        for pillar in self.pillars:
            if time < pillar.time - _TOLERANCE:
                return pillar.hazard
        return self.pillars[-1].hazard

    def integrated_hazard(self, time: float) -> float:
        """``integral_0^t h(s) ds``, piecewise linear in ``t``."""
        if time < 0.0:
            raise BadCredit(f"an integrated hazard needs a non-negative time, got {time!r}")
        total = 0.0
        previous = 0.0
        for pillar in self.pillars:
            if time <= pillar.time:
                return total + pillar.hazard * (time - previous)
            total += pillar.hazard * (pillar.time - previous)
            previous = pillar.time
        return total + self.pillars[-1].hazard * (time - previous)

    def survival_at(self, time: float) -> float:
        """The probability of still paying at ``time``."""
        return math.exp(-self.integrated_hazard(time))

    def survival(self, day: date) -> float:
        """The probability of still paying on ``day``."""
        return self.survival_at(self.time_to(day))

    def default_probability(self, day: date) -> float:
        """The probability of having defaulted by ``day``."""
        return -math.expm1(-self.integrated_hazard(self.time_to(day)))

    def grid(self, start: float, end: float) -> list[float]:
        """Times splitting ``[start, end]`` where the hazard rate changes.

        The endpoints are always included. Anything the caller adds — coupon
        dates, discount pillars — is merged on top of this, and on the union
        both the hazard and the forward rate are constant, which is what makes
        the integrals exact rather than approximate.
        """
        if end < start:
            raise BadCredit(f"the interval [{start!r}, {end!r}] runs backwards")
        inner = [t for t in self.times if start + _TOLERANCE < t < end - _TOLERANCE]
        return [start, *inner, end]


def _signed_time(curve: SurvivalCurve, day: date) -> float:
    """Year fraction from the curve's reference date to ``day``, signed.

    :meth:`SurvivalCurve.time_to` refuses a date in the past, because a
    negative accrual is almost always a mistake in a schedule rather than a
    quantity anybody wanted. A contract that has already started is the
    exception: its first period began before the valuation date by
    construction, and how long ago is exactly what the accrual-on-default
    weight needs.

    The magnitude in the past is measured by running the day count the way it
    is defined, forwards from the earlier date, and negating. On the actual
    bases that is exactly the negation of the forward fraction. On the thirty
    day conventions it is not quite: their end-of-month rules are not
    symmetric in the two dates, so the two directions can differ by a day's
    worth of accrual. It does not matter here, because the signed time is used
    only to locate the period's start relative to the valuation date, and the
    accrual it stands for is converted back to the contract's own basis by the
    ratio of the whole period's lengths before any money is computed.
    """
    if day >= curve.reference:
        return curve.time_to(day)
    return -year_fraction(day, curve.reference, curve.basis)


def _survival_integral(
    curve: SurvivalCurve,
    discount: DiscountCurve,
    start: float,
    end: float,
    *,
    accrual_from: float | None = None,
) -> float:
    """``integral DF(s) (-dS(s))`` over ``[start, end]``, in closed form.

    With ``accrual_from`` set, the integrand is weighted by ``s`` minus that
    time, which is the accrual-on-default term: a default halfway through a
    coupon period owes half a coupon.

    Both the hazard and the instantaneous forward rate are taken to be constant
    on each piece of the grid, which they are once the grid is the union of the
    two curves' pillars. The forward rate on a piece is read off the discount
    curve's own endpoints, so the result is exact for the curve as interpolated
    — log-linear interpolation makes it exact for the curve's continuum too,
    and any other scheme makes it exact on the grid and convergent under
    refinement.
    """
    if end <= start + _TOLERANCE:
        return 0.0
    pieces = _merge(curve.grid(start, end), discount.times, start, end)
    total = 0.0
    for left, right in pairwise(pieces):
        width = right - left
        if width <= _TOLERANCE:
            continue
        hazard = curve.hazard_at(0.5 * (left + right))
        df_left = discount.discount_at(left)
        df_right = discount.discount_at(right)
        # The piecewise-constant forward rate that reproduces the two endpoints.
        rate = -math.log(df_right / df_left) / width
        survival = curve.survival_at(left)
        decay = (rate + hazard) * width
        base = df_left * survival * hazard * width
        if accrual_from is None:
            total += base * _expm1_ratio(decay)
        else:
            offset = left - accrual_from
            total += base * (
                offset * _expm1_ratio(decay) + width * _ramp_integral(decay)
            )
    return total


def _merge(
    first: Sequence[float], second: Sequence[float], start: float, end: float
) -> list[float]:
    """The sorted union of two time grids, clipped to ``[start, end]``."""
    points = set(first)
    points.update(t for t in second if start + _TOLERANCE < t < end - _TOLERANCE)
    points.add(start)
    points.add(end)
    ordered = sorted(points)
    merged = [ordered[0]]
    for value in ordered[1:]:
        if value - merged[-1] > _TOLERANCE:
            merged.append(value)
    return merged


@dataclass(frozen=True)
class UpfrontSettlement:
    """What changes hands to enter or unwind a contract, split the way it is quoted.

    The whole amount is :attr:`cash`, and it is what
    :meth:`CreditDefaultSwap.value` returns: the value of every flow still to
    come, which already includes the buyer's promise to pay the part of the
    current coupon that accrued before today. The market does not quote that
    number. It quotes the principal and the accrued premium separately, and
    the buyer pays both, so the split is a split of one amount rather than two
    independent figures — ``cash == principal + accrued`` holds exactly, by
    construction, and the tests say so in both directions.

    The two halves are not measured the same way, which is the part worth
    knowing. :attr:`accrued` is the convention: the coupon times the elapsed
    fraction of the period, undiscounted, and owed whatever happens. The money
    inside :attr:`cash` that stands for the same slice of coupon is discounted
    to the payment date and paid only on survival to it. The gap between the
    two is small and it is not zero, and
    :meth:`CreditDefaultSwap.accrued_value` is the other half of it.
    """

    #: The clean upfront: the settlement amount with the accrued taken out.
    principal: float
    #: Premium accrued from the current period's start to the valuation date.
    accrued: float
    #: ``principal + accrued``, and the value of the remaining contract.
    cash: float


@dataclass(frozen=True)
class CreditDefaultSwap:
    """A single-name credit default swap, from the protection buyer's side.

    Attributes:
        schedule: Premium payment periods.
        basis: Day count for the premium accrual.
        coupon: Contractual running coupon, as a decimal. The standardised
            contracts trade at a fixed coupon with an upfront payment; a par
            swap has ``coupon`` equal to the par spread and zero upfront.
        notional: Face protected.
        accrual_on_default: Whether a default part-way through a period owes
            the accrued premium. True for every standard contract; available
            as false because the difference is worth being able to see.
    """

    schedule: Schedule
    basis: Basis
    coupon: float
    notional: float = 1.0
    accrual_on_default: bool = True

    def __post_init__(self) -> None:
        if self.coupon < 0.0:
            raise BadCredit(f"the coupon is {self.coupon!r}; a premium is not negative")
        if self.notional <= 0.0:
            raise BadCredit(f"the notional is {self.notional!r}; it is positive")

    @classmethod
    def standard(
        cls,
        effective: date,
        maturity: date,
        coupon: float,
        *,
        calendar: Calendar,
        basis: Basis = Basis.ACT_360,
        frequency: Frequency = Frequency.QUARTERLY,
        notional: float = 1.0,
        rolling: Rolling = Rolling.FOLLOWING,
    ) -> CreditDefaultSwap:
        """The market convention: quarterly premiums on actual/360.

        The rolling convention defaults to following rather than the modified
        following the rest of this package uses, because that is what the
        standard contract specifies and the difference is a payment date.
        """
        return cls(
            schedule=generate(
                effective, maturity, frequency, calendar=calendar, rolling=rolling
            ),
            basis=basis,
            coupon=coupon,
            notional=notional,
        )

    # -- where the valuation date sits inside the schedule ---------------------

    def remaining(self, reference: date) -> tuple[Period, ...]:
        """The periods still to pay on ``reference``.

        A period is gone once its money has moved, which is its payment date
        and not its accrual end: under a payment lag a period can have finished
        accruing and still owe its whole coupon. The test is therefore on the
        payment date, and a payment falling on the valuation date counts as
        made.
        """
        return tuple(one for one in self.schedule if one.payment > reference)

    def current_period(self, reference: date) -> Period | None:
        """The period ``reference`` falls inside, or ``None`` if it falls outside.

        ``None`` before the contract starts and after it ends, which are the
        two states in which nothing has accrued.
        """
        for one in self.schedule:
            if one.adjusted_start <= reference < one.adjusted_end:
                return one
        return None

    def accrued(self, reference: date) -> float:
        """Premium accrued since the current period began, for the notional.

        The market convention, which is a simple day count: undiscounted, and
        not weighted by survival, because the buyer owes it whether or not the
        entity survives the rest of the period.
        """
        period = self.current_period(reference)
        if period is None:
            return 0.0
        elapsed = year_fraction(period.adjusted_start, reference, self.basis)
        return self.notional * self.coupon * elapsed

    def _require_live(self, reference: date) -> tuple[Period, ...]:
        periods = self.remaining(reference)
        if not periods:
            raise BadCredit(
                f"this contract has nothing left to settle: its last payment was "
                f"{self.schedule[-1].payment.isoformat()} and the curve is "
                f"referenced {reference.isoformat()}. A fully settled contract has "
                "no annuity to divide by and no fair coupon."
            )
        return periods

    # -- the two legs ----------------------------------------------------------

    def risky_annuity(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """The premium leg per unit of coupon, per unit of notional.

        Also the credit DV01 divided by a basis point, which is what it is
        usually wanted for: multiply by the notional and by 0.0001 to get the
        money value of a one basis point move in the spread.

        Only the periods that have not paid are counted, so a contract that
        started before the curve's reference date prices as the remaining
        contract it is. Two things about the current period are easy to get
        wrong and are worth stating.

        The coupon paid at the next payment date is the **whole** period's
        coupon. The contract does not pro-rate it for the part of the period
        that has already gone by, so neither does this.

        The accrual owed on a default is measured **from the period's own
        start**, not from the valuation date. A default tomorrow owes the
        premium that has accrued since the period began, including the part
        that accrued before today. So the integration runs over the remaining
        stub while the weight's origin stays back at the period start — two
        different clamps, and clamping both is the mistake. It understates the
        leg, by the elapsed accrual times the chance of defaulting inside the
        rest of the period; the README measures it on a seasoned contract.
        """
        total = 0.0
        for period in self._require_live(curve.reference):
            pay = curve.time_to(period.payment)
            accrual = year_fraction(period.adjusted_start, period.adjusted_end, self.basis)
            total += accrual * discount.discount_at(pay) * curve.survival_at(pay)
            if self.accrual_on_default:
                start = _signed_time(curve, period.adjusted_start)
                end = _signed_time(curve, period.adjusted_end)
                # The integral is in year fractions on the curve's basis; the
                # accrual owed is on the contract's. The two bases differ in
                # general, so the fraction of the period elapsed is converted
                # by the ratio of the whole period's lengths rather than
                # assumed equal.
                span = end - start
                if span > _TOLERANCE:
                    weighted = _survival_integral(
                        curve, discount, max(start, 0.0), end, accrual_from=start
                    )
                    total += weighted * accrual / span
        return total

    def protection_leg(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """Present value of the payout on default, per unit of notional.

        Integrated from the valuation date or the effective date, whichever is
        later. A default before the curve's reference date is not a future
        event to be priced, and a contract whose protection has already ended
        is worth nothing rather than an error.
        """
        start = _signed_time(curve, self.schedule.effective)
        end = _signed_time(curve, self.schedule.maturity)
        if end <= 0.0:
            return 0.0
        return (1.0 - curve.recovery) * _survival_integral(
            curve, discount, max(start, 0.0), end
        )

    def accrued_value(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """What the elapsed slice of the current coupon is actually worth, for the notional.

        The counterpart to :meth:`accrued`. That one is the quoted convention:
        a day count, undiscounted, owed come what may. This one is the same
        slice of coupon valued the way the rest of the leg is valued — paid at
        the payment date, and only on survival to it, plus the share of it that
        a default inside the remaining stub would bring forward.

        The two are the same quantity by construction and differ by discounting
        and by default risk. The difference is small at ordinary levels, which
        is why nobody writes it down, and it is the reason the principal in
        :class:`UpfrontSettlement` is not a present value.
        """
        period = self.current_period(curve.reference)
        if period is None or period.payment <= curve.reference:
            return 0.0
        accrual = year_fraction(period.adjusted_start, period.adjusted_end, self.basis)
        if accrual <= _TOLERANCE:
            return 0.0
        elapsed = year_fraction(period.adjusted_start, curve.reference, self.basis)
        share = elapsed / accrual
        pay = curve.time_to(period.payment)
        worth = accrual * discount.discount_at(pay) * curve.survival_at(pay)
        if self.accrual_on_default:
            start = _signed_time(curve, period.adjusted_start)
            end = _signed_time(curve, period.adjusted_end)
            span = end - start
            if span > _TOLERANCE:
                weighted = _survival_integral(
                    curve, discount, max(start, 0.0), end, accrual_from=start
                )
                worth += weighted * accrual / span
        return self.notional * self.coupon * share * worth

    def premium_leg(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """Present value of the premiums, per unit of notional."""
        return self.coupon * self.risky_annuity(curve, discount)

    def value(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """Present value to the protection buyer, for the stated notional.

        Positive means the protection is worth more than the premiums promised
        for it, which is the upfront the buyer pays.
        """
        return self.notional * (
            self.protection_leg(curve, discount) - self.premium_leg(curve, discount)
        )

    def par_spread(self, curve: SurvivalCurve, discount: DiscountCurve) -> float:
        """The coupon at which the swap is worth nothing.

        Raises:
            BadCredit: If the risky annuity is zero, which means the entity is
                certain to have defaulted before the first premium and no
                coupon makes the trade fair.
        """
        annuity = self.risky_annuity(curve, discount)
        if annuity <= _TOLERANCE:
            raise BadCredit(
                "the risky annuity is zero, so no coupon makes this swap fair: "
                "survival to the first premium date is negligible"
            )
        return self.protection_leg(curve, discount) / annuity

    def upfront(self, curve: SurvivalCurve, discount: DiscountCurve) -> UpfrontSettlement:
        """The settlement amount, split into principal and accrued premium.

        The buyer pays both halves, so the principal is the whole amount less
        the accrued rather than a separate valuation. See
        :class:`UpfrontSettlement` for why the two halves are measured
        differently and :meth:`accrued_value` for the size of that.
        """
        cash = self.value(curve, discount)
        owed = self.accrued(curve.reference)
        return UpfrontSettlement(principal=cash - owed, accrued=owed, cash=cash)


def triangle_spread(hazard: float, recovery: float) -> float:
    """The credit triangle: ``spread = hazard * (1 - recovery)``.

    The rule of thumb, exact in the limit of continuous premiums, zero
    interest rates and no accrual on default. It is off by a few per cent of
    itself at ordinary levels and by considerably more at high ones; the
    README measures where.
    """
    if hazard < 0.0:
        raise BadCredit(f"the hazard rate is {hazard!r}; it is not negative")
    if not (0.0 <= recovery < 1.0):
        raise BadCredit(f"recovery is {recovery!r}; it is a fraction of face in [0, 1)")
    return hazard * (1.0 - recovery)


def triangle_hazard(spread: float, recovery: float) -> float:
    """The credit triangle, inverted: ``hazard = spread / (1 - recovery)``."""
    if spread < 0.0:
        raise BadCredit(f"the spread is {spread!r}; it is not negative")
    if not (0.0 <= recovery < 1.0):
        raise BadCredit(f"recovery is {recovery!r}; it is a fraction of face in [0, 1)")
    return spread / (1.0 - recovery)


def implied_hazard(
    swap: CreditDefaultSwap,
    spread: float,
    discount: DiscountCurve,
    *,
    recovery: float,
    reference: date | None = None,
    basis: Basis | None = None,
) -> float:
    """The flat hazard rate at which ``swap`` has ``spread`` as its par spread.

    The honest version of :func:`triangle_hazard`, solved rather than
    approximated. Used on its own for a single quote, and by
    :func:`bootstrap_hazards` for each step of a term structure.

    Args:
        swap: Defines the premium schedule and the day count.
        spread: The quoted par spread, as a decimal.
        discount: Curve to discount on.
        recovery: Fraction of face recovered.
        reference: Curve reference date; defaults to the discount curve's.
        basis: Day count for the survival curve; defaults to the swap's.

    Returns:
        The flat forward hazard rate.

    Raises:
        BadCredit: If the spread is negative, or no hazard rate reproduces it.
    """
    if spread < 0.0:
        raise BadCredit(f"the spread is {spread!r}; a par spread is not negative")
    anchor = discount.reference if reference is None else reference
    counting = swap.basis if basis is None else basis
    maturity = swap.schedule.maturity

    def residual(hazard: float) -> float:
        curve = SurvivalCurve.flat(
            anchor, maturity, hazard, basis=counting, recovery=recovery
        )
        return swap.par_spread(curve, discount) - spread

    guess = triangle_hazard(spread, recovery)
    try:
        root = solve(residual, low=0.0, high=max(4.0 * guess, 1.0))
    except ValueError as error:
        raise BadCredit(
            f"no flat hazard rate reproduces a par spread of {spread!r} at a "
            f"recovery of {recovery!r}: {error}"
        ) from error
    return root.value


def bootstrap_hazards(
    swaps: Sequence[CreditDefaultSwap],
    spreads: Sequence[float],
    discount: DiscountCurve,
    *,
    recovery: float,
    reference: date | None = None,
    basis: Basis | None = None,
) -> SurvivalCurve:
    """Strip a survival curve from par spreads at increasing maturities.

    Sequential, as a discount curve is bootstrapped: each hazard rate is
    solved with every earlier one already fixed, so the ``n``-th quote
    determines the forward hazard between the ``n-1``-th and ``n``-th
    pillars and nothing before it.

    A pillar sits at the last date its quote touches rather than at the
    quote's maturity. Those are not the same date -- the final premium pays on
    the rolled maturity, which can be a day or three later -- and a pillar at
    the maturity leaves that last coupon discounted at the next quote's hazard
    rate, so the curve misses the quote that produced it by about three parts
    in a hundred million. Small enough to read as solver noise, which is why
    it is worth saying that it is not.

    The result reprices every input exactly, which the tests check rather than
    assume — a bootstrap that does not is the most common way a credit curve
    goes quietly wrong, because every downstream number stays plausible.

    Args:
        swaps: One per quote, in increasing maturity order.
        spreads: Par spreads, as decimals, in the same order.
        discount: Curve to discount on.
        recovery: Fraction of face recovered.
        reference: Curve reference date; defaults to the discount curve's.
        basis: Day count for the survival curve; defaults to the first swap's.

    Returns:
        The stripped curve.

    Raises:
        BadCredit: If the inputs disagree in length, are not in increasing
            maturity order, or a step has no solution.
    """
    if len(swaps) != len(spreads):
        raise BadCredit(
            f"{len(swaps)} swaps against {len(spreads)} spreads; they pair up"
        )
    if not swaps:
        raise BadCredit("a bootstrap needs at least one quote")
    anchor = discount.reference if reference is None else reference
    counting = swaps[0].basis if basis is None else basis

    # The pillar for each quote goes at the last date that quote touches, not
    # at its maturity. Those differ: the final premium is paid on the rolled
    # maturity date, which can be a day or three after the protection ends, so
    # a pillar at the maturity leaves the last coupon discounted at the *next*
    # quote's hazard rate. The curve then does not reprice the quote that
    # produced it -- by one day's hazard on one coupon, which is around three
    # parts in a hundred million and looks exactly like solver noise.
    pillars = [
        max(swap.schedule.maturity, swap.schedule.payment_dates[-1]) for swap in swaps
    ]
    for earlier, later in pairwise(pillars):
        if later <= earlier:
            raise BadCredit(
                f"maturity {later.isoformat()} does not come after "
                f"{earlier.isoformat()}; quotes are bootstrapped in order"
            )

    points: list[tuple[date, float]] = []
    for swap, spread, pillar in zip(swaps, spreads, pillars, strict=True):
        if spread < 0.0:
            raise BadCredit(f"the spread is {spread!r}; a par spread is not negative")

        def residual(
            hazard: float,
            swap: CreditDefaultSwap = swap,
            spread: float = spread,
            pillar: date = pillar,
        ) -> float:
            curve = SurvivalCurve.from_hazards(
                anchor,
                [*points, (pillar, hazard)],
                basis=counting,
                recovery=recovery,
            )
            return swap.par_spread(curve, discount) - spread

        guess = triangle_hazard(spread, recovery)
        try:
            root = solve(residual, low=0.0, high=max(8.0 * guess, 1.0))
        except ValueError as error:
            raise BadCredit(
                f"the quote maturing {swap.schedule.maturity.isoformat()} at a spread "
                f"of {spread!r} has no forward hazard rate consistent with the "
                f"earlier quotes: {error}"
            ) from error
        points.append((pillar, root.value))

    return SurvivalCurve.from_hazards(anchor, points, basis=counting, recovery=recovery)


def risky_bond_price(
    bond: Bond,
    curve: SurvivalCurve,
    discount: DiscountCurve,
    *,
    settlement: date | None = None,
) -> float:
    """The present value of a bond that may default, per unit of face.

    Two terms. Each promised cashflow is discounted and weighted by the
    probability of surviving to receive it. Against that, a default pays the
    recovery fraction of *face* at the moment it happens, which is the market
    convention for a bond and is not the same as recovering a fraction of the
    remaining cashflows — the difference is largest for a long, high-coupon
    bond, where the promised payments are worth far more than face.

    Args:
        bond: The instrument.
        curve: Survival and recovery.
        discount: Curve to discount on.
        settlement: Value from this date; defaults to the curve's reference.

    Returns:
        Present value per unit of face, including accrued interest.
    """
    start = curve.reference if settlement is None else settlement
    total = 0.0
    last = curve.time_to(start)
    for flow in bond.cashflows(start):
        time = curve.time_to(flow.day)
        total += flow.amount * discount.discount_at(time) * curve.survival_at(time)
        last = max(last, time)
    # Recovery is a fraction of *face*, and this bond's face is its
    # redemption amount -- 100 in the convention the rest of this package
    # prices in, not 1. Leaving that out silently returns a price two orders
    # of magnitude too low for the recovery term and very nearly right for
    # everything else, which is the sort of error that survives a smoke test.
    recovery_leg = (
        curve.recovery
        * bond.redemption
        * _survival_integral(curve, discount, curve.time_to(start), last)
    )
    return total + recovery_leg
