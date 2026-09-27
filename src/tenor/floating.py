"""Floating rate notes: projected coupons, a discount margin, and two durations.

A floater is the instrument whose entire behaviour is a statement about the
curve, and it is the one place in fixed income where rate risk and spread risk
are not roughly the same number. They differ here by two orders of magnitude, and
quoting one where the other was meant is a mistake this instrument makes
available in a way a fixed bond does not.

The reason is an identity rather than an approximation. Write ``g_k`` for the
growth the curve implies across period ``k``, so the projected index rate is
``f_k = g_k / tau_k``, and let the note pay ``(f_k + m) tau_k`` for a quoted
margin ``m``. Discount period by period at the projected rate plus a discount
margin ``d``. Then when ``d == m`` each period's coupon is exactly its own
discount denominator minus one, the sum telescopes, and the price is **exactly**
par — for any curve, of any shape, at any level, on any day count basis. Nothing
about that is a limit or a first-order argument, and it is asserted here to the
last few bits rather than to a tolerance.

The consequence is stronger than the textbook statement of it, and measuring it
is what showed that. A floater priced at its own quoted margin has a rate
duration of **exactly zero on any settlement date**, not only on a reset date,
provided the current period's fixing is known. Substituting the telescoping sum
for the periods after the current one collapses the whole price to

    (1 + (fixing + m) tau_1) / (1 + (fixing + m) tau_remaining)

in which no curve appears at all. Rate risk is not something a floater has a
little of because its coupon resets soon; at par it has none, and what it has
away from par comes from somewhere else entirely.

That somewhere else is the margin. A discount margin different from the quoted
one leaves an annuity of the difference, and *that* has the rate sensitivity. On
the five-year quarterly note in the tests, rate duration measures 0.000 at a
75bp margin equal to the quoted one, -0.061 years at 125bp, -0.250 years at
275bp and **+0.061 years** at 25bp. Linear in the margin difference, and the sign
flips with the side of par: a floater trading cheap to its quoted margin gains
when rates rise, because the negative annuity it is carrying gets discounted
harder. Projecting the current fixing instead of knowing it adds a separate
-0.057 years at par three weeks into a quarterly period, which is the cost of a
missing fixings history stated as risk rather than as an apology.

What survives is spread duration, which runs over the note's whole life because
the margin is discounted across every remaining period. Measured against the
modified duration of a fixed bond of the same maturity on the same curve, it
comes out at 1.014 times it at three years, 1.005 at five and 0.974 at ten.

Three decisions are made here rather than left implicit.

**The projected rate is backed out of the discount factors against the coupon's
own accrual fraction**, as ``(P(start) / P(end) - 1) / tau``, rather than taken
from :meth:`~tenor.curve.DiscountCurve.forward_rate`. The curve's forward is a
simple rate over the *curve's* year fraction and the coupon accrues over the
note's; on a curve built ACT/365F under a note on ACT/360 the two index rates
differ by the ratio of the bases, 20.72bp against 21.01bp on the first period of
the test note. The price effect is small — half a basis point of price at a 275bp
margin, and 0.05bp on the discount margin implied from a clean 96 — which is the
honest size of it and still worth not having, because being consistent costs
nothing.

**It is worth knowing that the par identity does not check this.** The
telescoping needs only that the coupon and the discount denominator use the *same*
rate over the *same* fraction; it does not care whether that rate is the right
one. A note projected the wrong way still prices at exactly par on its reset date
at its own quoted margin. So the identity is a test that projection and
discounting agree with each other, and not evidence that either is right.

**The discount margin is applied to the projected forward, period by period, not
added to a zero rate.** Adding a constant to the zero curve is a Z-spread and it
is a different number: it discounts each flow from settlement at a single
compounded rate plus the spread, where the discount margin compounds the margin
across the same periods the coupons accrue over. The market quotes the discount
margin, so that is what :func:`FloatingNote.discount_margin` solves for.

**The current period's coupon is a fixing, not a projection.** It was set at the
last reset and the curve does not know it. Passing ``current_fixing`` uses it;
omitting it projects the rate the curve implies for a period that has already
started, which is an approximation the library will make and name rather than
refuse — it is what a caller without a fixings history has. Three weeks into a
quarterly period, with the real fixing 50bp away from the projection, it is worth
2.9bp of price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .rates import Compounding
from .schedule import Frequency, Period, Schedule, generate
from .solve import Root, brent


class BadNote(ValueError):
    """A floating rate note, or a settlement date, that does not describe a trade."""


class BadMargin(ValueError):
    """A discount margin that cannot be extracted from the price given."""


@dataclass(frozen=True)
class FloatingCoupon:
    """One projected coupon, with everything that went into it.

    Returned rather than summed away because the two questions asked of a
    floater — what rate was assumed, and what was it discounted at — are not
    answerable from the price.
    """

    period: Period
    payment: date
    #: Year fraction the coupon accrues over, on the note's basis.
    accrual: float
    #: The index rate for the period, simple, on the note's accrual basis.
    index_rate: float
    #: False when :attr:`index_rate` came from a supplied fixing.
    projected: bool
    #: Cash paid, per 100 of notional, excluding any redemption.
    coupon: float
    #: Redemption paid on this date, per 100. Non-zero on the last coupon only.
    redemption: float
    #: The margin-adjusted discount factor from settlement to :attr:`payment`.
    discount: float

    @property
    def amount(self) -> float:
        return self.coupon + self.redemption

    @property
    def present_value(self) -> float:
        return self.amount * self.discount


@dataclass(frozen=True)
class FloatingNote:
    """A floating rate note paying an index plus a fixed margin.

    ``quoted_margin`` is the annual spread over the index as a decimal — 0.0075
    for 75 basis points. Prices are per 100 of notional, so par is 100.0 and
    ``redemption`` defaults to it.

    The index is assumed to be the rate for exactly the coupon period, set at the
    start of the period and paid at the end. That is the ordinary term-rate
    floater. A note on a compounded overnight rate in arrears is a different
    instrument and is not this one.
    """

    effective: date
    maturity: date
    quoted_margin: float = 0.0
    frequency: Frequency = Frequency.QUARTERLY
    basis: Basis = Basis.ACT_360
    redemption: float = 100.0
    calendar: Calendar = WEEKENDS_ONLY
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""
    _schedule: list[Schedule] = field(
        default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadNote(
                f"{self.name} matures on {self.maturity.isoformat()}, which is not "
                f"after it was issued on {self.effective.isoformat()}"
            )
        if self.redemption <= 0.0:
            raise BadNote(
                f"{self.name} redeems at {self.redemption!r}; a note that pays "
                "nothing back at maturity is not priced by this"
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"index + {self.quoted_margin * 1e4:.1f}bp of {self.maturity.isoformat()}"
        )

    def schedule(self) -> Schedule:
        """The coupon schedule, generated once and kept.

        Same reason as :meth:`tenor.bond.Bond.schedule`: every risk number
        reprices the note two or three times, and regenerating the date
        arithmetic each time costs more than the pricing does.
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

    # -- where settlement sits ------------------------------------------------

    def current_period(self, settlement: date) -> Period:
        """The coupon period containing ``settlement``.

        Settlement on a reset date belongs to the period starting there: the
        coupon has been paid, accrued interest is zero, and the note is at one of
        the two dates where its rate risk vanishes.
        """
        if settlement < self.effective:
            raise BadNote(
                f"{self.name} settles on {settlement.isoformat()}, before it was "
                f"issued on {self.effective.isoformat()}"
            )
        if settlement >= self.maturity:
            raise BadNote(
                f"{self.name} matured on {self.maturity.isoformat()}, so there is "
                f"nothing to price on {settlement.isoformat()}"
            )
        for period in self.schedule():
            if period.start <= settlement < period.end:
                return period
        raise BadNote(  # pragma: no cover - the two guards above cover the range
            f"{settlement.isoformat()} falls in no coupon period of {self.name}"
        )

    def is_reset_date(self, settlement: date) -> bool:
        """Whether settlement falls on a period boundary.

        The dates where the par identity holds exactly, so worth being able to
        ask rather than inferring from an accrual of zero.
        """
        return settlement == self.current_period(settlement).start

    def period_remaining(self, settlement: date) -> float:
        """The fraction of the current coupon period still to run.

        One on a reset date, falling to zero as the next one approaches. It is
        the multiplier on the current period's discounting and, at the first
        order, the whole of the note's rate duration.
        """
        period = self.current_period(settlement)
        full = year_fraction(period.start, period.end, self.basis)
        if full <= 0.0:  # pragma: no cover - schedules do not emit empty periods
            raise BadNote(f"{self.name} has a coupon period of zero length")
        return year_fraction(settlement, period.end, self.basis) / full

    def index_rate(
        self, curve: DiscountCurve, period: Period, *, fixing: float | None = None
    ) -> float:
        """The simple index rate for ``period``, on this note's accrual basis.

        Backed out of the two discount factors against the coupon's own accrual
        fraction rather than read off the curve's forward, because the curve's
        forward is a simple rate over the curve's year fraction and the coupon
        does not accrue over that one. Where the bases differ the two rates
        differ by their ratio — 20.72bp against 21.01bp for an ACT/360 note on an
        ACT/365F curve — and the error lands in every coupon in the same
        direction. The price effect is half a basis point at a wide margin, and
        the par identity does not catch it at all.
        """
        if fixing is not None:
            return fixing
        accrual = year_fraction(period.start, period.end, self.basis)
        if accrual <= 0.0:  # pragma: no cover - schedules do not emit empty periods
            raise BadNote(f"{self.name} has a coupon period of zero length")
        growth = curve.discount(period.adjusted_start) / curve.discount(
            period.adjusted_end
        )
        return (growth - 1.0) / accrual

    def accrued(
        self,
        settlement: date,
        *,
        curve: DiscountCurve | None = None,
        current_fixing: float | None = None,
    ) -> float:
        """Accrued interest per 100, which needs the fixing to be known.

        The rate for the current period was set at the last reset and is not on
        any curve. Supply ``current_fixing`` for the right answer, or ``curve``
        to project it and accept the approximation, or neither and be refused —
        silently returning an accrual computed from the margin alone would look
        like a number and be wrong by the whole of the index.
        """
        period = self.current_period(settlement)
        if current_fixing is None and curve is None:
            raise BadNote(
                f"{self.name} accrues at its current fixing, which was set on "
                f"{period.start.isoformat()} and is not on any curve. Pass "
                "`current_fixing` for the rate that was actually set, or `curve` to "
                "project it."
            )
        rate = (
            current_fixing
            if current_fixing is not None
            else self.index_rate(curve, period)  # type: ignore[arg-type]
        )
        elapsed = year_fraction(period.start, settlement, self.basis)
        return 100.0 * (rate + self.quoted_margin) * elapsed

    # -- coupons, price and margin --------------------------------------------

    def coupons(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
    ) -> tuple[FloatingCoupon, ...]:
        """Every remaining coupon, projected and discounted at ``margin``.

        ``margin`` is the discount margin: it is added to each period's projected
        index rate and the flows are discounted period by period at the sum. A
        flow due exactly on the settlement date belongs to the seller and is
        excluded, which is the same rule :meth:`current_period` uses.
        """
        remaining = [one for one in self.schedule() if one.end > settlement]
        factor = 1.0
        coupons: list[FloatingCoupon] = []
        for index, period in enumerate(remaining):
            fixing = current_fixing if index == 0 else None
            rate = self.index_rate(curve, period, fixing=fixing)
            accrual = year_fraction(period.start, period.end, self.basis)
            # The current period is discounted over what is left of it only. On a
            # reset date that is the whole period and this reduces to the clean
            # case, which is what makes the par identity exact there and
            # approximate everywhere else.
            discounting = (
                year_fraction(settlement, period.end, self.basis)
                if index == 0
                else accrual
            )
            growth = 1.0 + (rate + margin) * discounting
            if growth <= 0.0:
                raise BadMargin(
                    f"a margin of {margin!r} makes the discounting factor for the "
                    f"period ending {period.end.isoformat()} non-positive; the "
                    "projected rate plus the margin is below -1 over that period"
                )
            factor /= growth
            coupons.append(
                FloatingCoupon(
                    period=period,
                    payment=period.payment,
                    accrual=accrual,
                    index_rate=rate,
                    projected=fixing is None,
                    coupon=100.0 * (rate + self.quoted_margin) * accrual,
                    redemption=self.redemption if index == len(remaining) - 1 else 0.0,
                    discount=factor,
                )
            )
        return tuple(coupons)

    def dirty_price(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
    ) -> float:
        """Present value of the projected flows, per 100.

        At ``margin`` equal to :attr:`quoted_margin` and settlement on a reset
        date this is exactly ``redemption``, whatever the curve is. Away from a
        reset date, with the fixing supplied, it is exactly
        ``(1 + (fixing + m) tau_1) / (1 + (fixing + m) tau_remaining)`` times it —
        still with no curve in it.
        """
        return sum(
            one.present_value
            for one in self.coupons(
                curve, settlement, margin=margin, current_fixing=current_fixing
            )
        )

    def clean_price(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
    ) -> float:
        """The dirty price less accrued interest, which is what a screen shows."""
        return self.dirty_price(
            curve, settlement, margin=margin, current_fixing=current_fixing
        ) - self.accrued(settlement, curve=curve, current_fixing=current_fixing)

    def discount_margin(
        self,
        curve: DiscountCurve,
        price: float,
        settlement: date,
        *,
        clean: bool = True,
        current_fixing: float | None = None,
    ) -> Root:
        """The margin over the projected index that reproduces ``price``.

        Solved against the *dirty* price whichever price was quoted, for the same
        reason a yield is: the discounting has to reproduce what is actually
        paid, and the clean price is a quoting convention sitting on top of it.

        A note trading at par on a reset date returns its own quoted margin, and
        that is the calibration this is worth checking against — no external
        reference is needed because the answer is known by construction.
        """
        target = (
            price
            + self.accrued(settlement, curve=curve, current_fixing=current_fixing)
            if clean
            else price
        )
        if target <= 0.0:
            raise BadMargin(
                f"{self.name} cannot have a dirty price of {target!r}; every "
                "remaining flow is positive on a non-negative projected rate"
            )

        def objective(margin: float) -> float:
            return (
                self.dirty_price(
                    curve, settlement, margin=margin, current_fixing=current_fixing
                )
                - target
            )

        low, high = -0.05, 0.20
        if objective(low) * objective(high) > 0.0:
            low, high = -0.2, 2.0
            if objective(low) * objective(high) > 0.0:
                raise BadMargin(
                    f"no margin between {low} and {high} prices {self.name} at "
                    f"{price!r}: the error is {objective(low):.6g} at one end and "
                    f"{objective(high):.6g} at the other"
                )
        return brent(objective, low, high, tolerance=1e-15)

    # -- risk ------------------------------------------------------------------

    def spread_duration(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
        shift: float = 1e-4,
    ) -> float:
        """``-1/P dP/dm``, in years, by a central difference on the margin.

        The sensitivity that survives on a floater. It runs over the note's whole
        life, because the margin is discounted across every remaining period, so
        it comes out close to the modified duration of a fixed bond of the same
        maturity — 1.014 times it at three years, 1.005 at five and 0.974 at ten
        on the curve in the tests. Which is the opposite of what this
        instrument's rate duration does, and the whole reason both are reported.
        """
        if shift <= 0.0:
            raise BadNote(f"a shift is a positive size, got {shift!r}")
        up = self.dirty_price(
            curve, settlement, margin=margin + shift, current_fixing=current_fixing
        )
        down = self.dirty_price(
            curve, settlement, margin=margin - shift, current_fixing=current_fixing
        )
        base = self.dirty_price(
            curve, settlement, margin=margin, current_fixing=current_fixing
        )
        return -(up - down) / (2.0 * shift) / base

    def rate_duration(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
        shift: float = 1e-4,
        compounding: Compounding = Compounding.CONTINUOUS,
    ) -> float:
        """``-1/P dP/dz``, in years, for a parallel shift of the curve.

        Zero to floating point whenever ``margin`` equals the quoted margin and
        ``current_fixing`` is supplied — on *any* settlement date, not only on a
        reset date. The shift moves every projected coupon and every discount
        denominator by the same growth, the telescoping is untouched, and what is
        left of the price has no curve in it.

        What it measures, then, is not the note's coupon reset. It is the rate
        sensitivity of the annuity left over when the discount margin differs from
        the quoted one, which is linear in that difference and changes sign with
        the side of par: -0.250 years at a 275bp margin against a 75bp coupon
        spread, +0.061 at 25bp. Omitting the fixing adds the sensitivity of the
        current period's projected rate on top, -0.057 years at par three weeks
        into a quarterly period.
        """
        if shift <= 0.0:
            raise BadNote(f"a shift is a positive size, got {shift!r}")
        up = self.dirty_price(
            curve.shifted(shift, compounding),
            settlement,
            margin=margin,
            current_fixing=current_fixing,
        )
        down = self.dirty_price(
            curve.shifted(-shift, compounding),
            settlement,
            margin=margin,
            current_fixing=current_fixing,
        )
        base = self.dirty_price(
            curve, settlement, margin=margin, current_fixing=current_fixing
        )
        return -(up - down) / (2.0 * shift) / base

    def margin_value_of_a_basis_point(
        self,
        curve: DiscountCurve,
        settlement: date,
        *,
        margin: float = 0.0,
        current_fixing: float | None = None,
    ) -> float:
        """Price change per 100 for one basis point on the margin.

        The number a desk hedges with, positive for a widening because a wider
        margin is a lower price.
        """
        base = self.dirty_price(
            curve, settlement, margin=margin, current_fixing=current_fixing
        )
        wider = self.dirty_price(
            curve, settlement, margin=margin + 1e-4, current_fixing=current_fixing
        )
        return base - wider
