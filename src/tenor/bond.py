"""Bonds: what they pay, what they cost, and how that cost moves.

This is the part of fixed income where two careful implementations disagree in
the third decimal place and neither is obviously wrong. The disagreements are
never in the arithmetic. They are in four conventions that are usually left
implicit:

1. **What accrued interest is a fraction of.** Either the day count fraction
   from the last coupon, or the fraction of the coupon period. These agree
   exactly under 30/360 and disagree under every ACT basis, which is precisely
   why the disagreement survives: the case people test is the one where it
   cannot show up. Both are here, named, as :class:`Accrual`.

2. **What the exponent counts.** The street convention discounts the ``k``-th
   remaining flow by ``(1 + y/f)`` raised to ``k - 1 + w``, where ``w`` is the
   fraction of the *current* coupon period still to run. Not the year fraction
   to the flow: a bond with a short first period and a bond without one
   discount their second coupons by the same exponent, because the exponent
   counts coupon periods and one of them has a shorter first one.

3. **Which price the yield belongs to.** Yield solves against the *dirty*
   price. Quoting clean and solving against clean is a mistake that gets
   smaller as settlement approaches a coupon date and vanishes on it, so it
   survives every test written on a coupon date.

4. **Whether a basis point moves the yield or the curve.** See
   :meth:`Bond.pv01` and :meth:`Bond.dv01`, which are not the same number.

**Duration is not one quantity.** Macaulay duration is a time — the
present-value-weighted average maturity of the cash flows, in years. Modified
duration is a sensitivity, ``-1/P dP/dy``, and equals Macaulay divided by
``1 + y/f``. Effective duration is also a sensitivity, but to the *curve*
rather than to the bond's own yield, and it is the only one of the three that
still means anything once a bond has an embedded option. The first two are
computed in closed form here and the third by bumping the curve, because that
is what each of them actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .rates import Compounding
from .schedule import Frequency, Period, Schedule, generate
from .solve import Root, brent


class BadBond(ValueError):
    """A bond, or a settlement date, that does not describe a trade."""


def _discount(growth: float, periods: float) -> float:
    """``growth ** -periods``, as a float.

    A float raised to a float is a complex number in general, so the
    annotation has to be narrowed somewhere. Doing it once here rather than at
    each of the four places the street discounting formula appears keeps the
    formula readable, which matters more than usual when the exponent is the
    thing most likely to be wrong.
    """
    return float(growth**-periods)


class Accrual(str, Enum):
    """How accrued interest divides the coupon period.

    The two agree to the last bit under 30/360, where a semi-annual period is
    exactly half a year by construction. Under ACT/365F they differ by the
    ratio of twice the period's actual length to 365 — about 0.8% of accrued
    interest for a period of 181 days, which on a 5% coupon three months in is
    around a cent per 100 of notional.
    """

    #: ``rate * year_fraction(last coupon, settlement)``. Direct, and what most
    #: corporate and 30/360 markets mean.
    DAY_COUNT = "day count fraction"
    #: ``rate / f * (elapsed / full period)``. What ACT/ACT ICMA means, and
    #: what government bond markets quoting on an actual basis mean.
    PERIOD_FRACTION = "fraction of the coupon period"


@dataclass(frozen=True)
class Cashflow:
    """One payment, and where it sits in the discounting."""

    day: date
    amount: float
    #: Coupon periods from settlement, ``k - 1 + w``. The exponent the street
    #: convention raises ``1 + y/f`` to, and not a year fraction.
    periods: float
    #: The same thing in years, ``periods / frequency``, for duration weights.
    years: float
    #: True for the payment that returns the principal.
    redemption: bool = False


@dataclass(frozen=True)
class Bond:
    """A fixed-coupon bullet bond.

    ``coupon`` is the annual rate as a decimal — 0.05 for a 5% bond — and
    ``redemption`` is per 100 of notional, so every price here is per 100 and
    par is 100.0.
    """

    effective: date
    maturity: date
    coupon: float
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    redemption: float = 100.0
    calendar: Calendar = WEEKENDS_ONLY
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    accrual: Accrual = Accrual.DAY_COUNT
    label: str = ""
    _schedule: list[Schedule] = field(
        default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadBond(
                f"{self.name} matures on {self.maturity.isoformat()}, which is not "
                f"after it was issued on {self.effective.isoformat()}"
            )
        if self.redemption <= 0.0:
            raise BadBond(
                f"{self.name} redeems at {self.redemption!r}; a bond that pays "
                "nothing back at maturity is not priced by this"
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"{self.coupon:.3%} of {self.maturity.isoformat()}"
        )

    def schedule(self) -> Schedule:
        """The coupon schedule, generated once and kept.

        A bond is priced, then differentiated, then bumped and repriced twice
        more for every risk number, and each of those regenerates the schedule.
        Caching it in a field rather than recomputing keeps a curve-bumped
        duration from being an order of magnitude more date arithmetic than
        the price it is measuring.
        """
        if not self._schedule:
            self._schedule.append(
                generate(
                    self.effective,
                    self.maturity,
                    self.frequency,
                    calendar=self.calendar,
                    rolling=self.rolling,
                )
            )
        return self._schedule[0]

    @property
    def periodic_coupon(self) -> float:
        """The cash each coupon pays, per 100 of notional."""
        return 100.0 * self.coupon / self.frequency.value

    # -- where settlement sits ------------------------------------------------

    def current_period(self, settlement: date) -> Period:
        """The coupon period containing ``settlement``.

        Settlement on a coupon date belongs to the period that *starts* there:
        the coupon has been paid, accrued interest is zero, and the next
        payment is a full period away.
        """
        if settlement < self.effective:
            raise BadBond(
                f"{self.name} settles on {settlement.isoformat()}, before it was "
                f"issued on {self.effective.isoformat()}"
            )
        if settlement >= self.maturity:
            raise BadBond(
                f"{self.name} matured on {self.maturity.isoformat()}, so there is "
                f"nothing to price on {settlement.isoformat()}"
            )
        for period in self.schedule():
            if period.start <= settlement < period.end:
                return period
        raise BadBond(  # pragma: no cover - the two guards above cover the range
            f"{settlement.isoformat()} falls in no coupon period of {self.name}"
        )

    def period_remaining(self, settlement: date) -> float:
        """``w``: the fraction of the current coupon period still to run.

        One at a coupon date, falling to zero as the next one approaches. This
        is the quantity the street convention's fractional exponent uses, and
        it is a ratio of two year fractions on the same basis, so the basis
        cancels wherever the period is a whole number of months.
        """
        period = self.current_period(settlement)
        full = year_fraction(period.start, period.end, self.basis)
        if full <= 0.0:  # pragma: no cover - schedules do not emit empty periods
            raise BadBond(f"{self.name} has a coupon period of zero length")
        return year_fraction(settlement, period.end, self.basis) / full

    def accrued(self, settlement: date) -> float:
        """Accrued interest per 100, under this bond's stated convention."""
        period = self.current_period(settlement)
        if self.accrual is Accrual.DAY_COUNT:
            return (
                100.0 * self.coupon * year_fraction(period.start, settlement, self.basis)
            )
        return self.periodic_coupon * (1.0 - self.period_remaining(settlement))

    def cashflows(self, settlement: date) -> tuple[Cashflow, ...]:
        """Every payment still to come, with its position in the discounting.

        A flow due exactly on the settlement date has been paid to the seller
        and is not part of what the buyer gets, so it is excluded — which is
        the same rule as :meth:`current_period` uses, stated once in each place
        it has to hold.
        """
        remaining = self.period_remaining(settlement)
        periods = [one for one in self.schedule() if one.end > settlement]
        flows: list[Cashflow] = []
        for index, period in enumerate(periods):
            amount = self.periodic_coupon
            if index == len(periods) - 1:
                amount += self.redemption
            count = index + remaining
            flows.append(
                Cashflow(
                    day=period.payment,
                    amount=amount,
                    periods=count,
                    years=count / self.frequency.value,
                    redemption=index == len(periods) - 1,
                )
            )
        return tuple(flows)

    # -- price and yield ------------------------------------------------------

    def dirty_price(self, rate: float, settlement: date) -> float:
        """Present value of the remaining flows at yield ``rate``, per 100.

        The street convention: each flow discounted by ``(1 + y/f)`` raised to
        its count of coupon periods, the first of which is fractional. This is
        what a yield *means*, and it is not the same as discounting by a year
        fraction at an annual rate — those agree only when the coupon periods
        are exactly ``1/f`` years long, which under an ACT basis they never
        quite are.
        """
        factor = 1.0 + rate / self.frequency.value
        if factor <= 0.0:
            raise BadBond(
                f"a yield of {rate!r} at {self.frequency.value} periods a year gives "
                f"a per-period growth factor of {factor!r}, which is not positive"
            )
        return sum(
            flow.amount * _discount(factor, flow.periods)
            for flow in self.cashflows(settlement)
        )

    def clean_price(self, rate: float, settlement: date) -> float:
        """The quoted price: dirty less accrued interest."""
        return self.dirty_price(rate, settlement) - self.accrued(settlement)

    def yield_from_clean(self, price: float, settlement: date) -> Root:
        """The yield that reproduces a quoted clean price.

        Solved against the dirty price, because that is what the yield
        discounts to. The distinction vanishes at a coupon date, where accrued
        is zero — so a yield solver written against the clean price passes
        every test written on a coupon date and is wrong on every other day.

        Returns the solve rather than the number: a yield quoted to the basis
        point should come with the evidence it converged.
        """
        return self.yield_from_dirty(price + self.accrued(settlement), settlement)

    def yield_from_dirty(self, price: float, settlement: date) -> Root:
        """The yield that reproduces a dirty price."""
        if price <= 0.0:
            raise BadBond(
                f"{self.name} cannot have a dirty price of {price!r}; every "
                "remaining flow is positive, so no yield discounts them to zero "
                "or below"
            )

        def objective(rate: float) -> float:
            return self.dirty_price(rate, settlement) - price

        # The price falls monotonically in the yield, from the undiscounted sum
        # of the flows down towards zero, so a bracket is known rather than
        # searched for. The low end stops short of the pole at ``y = -f``.
        low = -self.frequency.value + 1e-9
        high = 10.0
        return brent(objective, low, high, tolerance=1e-15)

    # -- sensitivity to the bond's own yield ----------------------------------

    def macaulay_duration(self, rate: float, settlement: date) -> float:
        """The present-value-weighted average time to the flows, in years.

        A time, not a sensitivity. For a zero-coupon bond it is exactly the
        time to maturity, whatever the yield — which is the test that catches
        an off-by-one in the exponent.
        """
        factor = 1.0 + rate / self.frequency.value
        flows = self.cashflows(settlement)
        price = sum(flow.amount * _discount(factor, flow.periods) for flow in flows)
        weighted = sum(
            flow.years * flow.amount * _discount(factor, flow.periods)
            for flow in flows
        )
        return weighted / price

    def modified_duration(self, rate: float, settlement: date) -> float:
        """``-1/P dP/dy``, in years, against the dirty price.

        Macaulay divided by ``1 + y/f``. Stated as the derivative it is rather
        than as a scaled duration, because the division is where the two
        quantities stop being the same thing.
        """
        return self.macaulay_duration(rate, settlement) / (
            1.0 + rate / self.frequency.value
        )

    def convexity(self, rate: float, settlement: date) -> float:
        """``1/P d2P/dy2``, in years squared, against the dirty price.

        In closed form. The second derivative of a sum of powers is another sum
        of powers, and differencing the price twice to get it loses about half
        the significant digits to cancellation — which does not matter for a
        number quoted to two decimal places, and does matter when it is being
        used to check the first derivative.
        """
        periodic = 1.0 + rate / self.frequency.value
        frequency = float(self.frequency.value)
        flows = self.cashflows(settlement)
        price = sum(flow.amount * _discount(periodic, flow.periods) for flow in flows)
        total = sum(
            flow.periods
            * (flow.periods + 1.0)
            / (frequency * frequency)
            * flow.amount
            * _discount(periodic, flow.periods + 2.0)
            for flow in flows
        )
        return total / price

    def pv01(self, rate: float, settlement: date, *, shift: float = 1e-4) -> float:
        """The price change from moving *the bond's own yield* by a basis point.

        Per 100 of notional, and positive for a fall in price — the sign
        convention every trading desk quotes it in. Computed by repricing at
        ``y + shift`` rather than as ``modified duration * price * shift``,
        because the two differ by the convexity term and the whole point of
        quoting a basis point value rather than a duration is to have the
        number that includes it.
        """
        return self.dirty_price(rate, settlement) - self.dirty_price(
            rate + shift, settlement
        )

    # -- sensitivity to the curve ---------------------------------------------

    def price_from_curve(self, curve: DiscountCurve, settlement: date) -> float:
        """Dirty price by discounting each flow on the curve, per 100.

        Forward to settlement: each flow's discount factor is divided by the
        settlement date's, so the answer is a price on the settlement date
        rather than a present value today. Skipping that division prices the
        bond for value now, which is a different and smaller number.
        """
        base = curve.discount(settlement)
        return sum(
            flow.amount * curve.discount(flow.day) / base
            for flow in self.cashflows(settlement)
        )

    def curve_yield(self, curve: DiscountCurve, settlement: date) -> Root:
        """The single yield equivalent to pricing this bond on ``curve``."""
        return self.yield_from_dirty(self.price_from_curve(curve, settlement), settlement)

    def dv01(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        shift: float = 1e-4,
        compounding: Compounding = Compounding.CONTINUOUS,
    ) -> float:
        """The price change from moving *the whole curve* by a basis point.

        Per 100 of notional, positive for a fall in price, and not the same
        number as :meth:`pv01`. Treating the two as interchangeable is the
        standard shortcut, and it is harmless right up until one is used to
        hedge the other.

        They differ for two reasons, and the smaller one is the one usually
        given. The familiar reason is shape: a bond's yield is a weighted
        average of the curve over its own flows, so shifting every zero rate by
        a basis point moves that average by a basis point only if the curve is
        flat. On a sloped curve, for a fifteen-year bullet, that is worth about
        two parts in a thousand.

        The larger reason is that a basis point is not one quantity. Move a
        *continuously compounded* zero rate by a basis point and you have moved
        the equivalent semi-annual rate by rather more — by ``exp(z/2)``, which
        is 1.5% at a 3% level. Against a bond whose yield is semi-annual, that
        alone puts the two numbers 1.5% apart *on a perfectly flat curve*,
        which is seven times the shape effect and in a case where the shape
        effect is zero by construction.

        So ``compounding`` says which rate the basis point is applied to.
        Matching it to the bond's own frequency brings the two back to within
        the convexity term; leaving it continuous does not, and the difference
        is not the curve's shape however much it looks like it.
        """
        return self.price_from_curve(curve, settlement) - self.price_from_curve(
            curve.shifted(shift, compounding), settlement
        )

    def effective_duration(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        shift: float = 1e-4,
        compounding: Compounding = Compounding.CONTINUOUS,
    ) -> float:
        """``-1/P dP/dr`` against a parallel curve shift, in years.

        A central difference, which is second-order accurate where a one-sided
        one is first-order: the error term in the one-sided version is
        proportional to the convexity, and convexity is exactly what
        distinguishes a bond's price from a straight line.
        """
        base = self.price_from_curve(curve, settlement)
        up = self.price_from_curve(curve.shifted(shift, compounding), settlement)
        down = self.price_from_curve(curve.shifted(-shift, compounding), settlement)
        return (down - up) / (2.0 * base * shift)

    def effective_convexity(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        shift: float = 1e-4,
        compounding: Compounding = Compounding.CONTINUOUS,
    ) -> float:
        """``1/P d2P/dr2`` against a parallel curve shift, in years squared."""
        base = self.price_from_curve(curve, settlement)
        up = self.price_from_curve(curve.shifted(shift, compounding), settlement)
        down = self.price_from_curve(curve.shifted(-shift, compounding), settlement)
        return (up + down - 2.0 * base) / (base * shift * shift)


__all__ = ["Accrual", "BadBond", "Bond", "Cashflow"]
