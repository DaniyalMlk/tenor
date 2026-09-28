"""Index-linked bonds: the reference index, the real yield, and breakeven.

An index-linked bond looks like a small variation on a fixed-coupon one and is
not. Almost everything that distinguishes it is a convention, each convention is
usually left implicit, and getting any of them wrong produces a number that looks
entirely reasonable.

**The index applied to a date is not the index for that date.** A statistical
office publishes a monthly figure several weeks after the month it describes, so
a bond that settles today cannot reference today's price level. The market
convention is a three-month lag with interpolation by day of month: the reference
index for the 15th of September is the July figure plus 14/30ths of the step from
July to August. Two consequences follow and both matter. The accretion over the
next three months is *already published* — it is arithmetic, not a forecast — and
the accretion over the three months just past has not happened yet as far as the
bond is concerned. :class:`ReferenceIndex` implements the lag and says which of
the two regimes a date falls in.

**The quoted price and the quoted yield are real; the money is not.** The
invoice is the real dirty price multiplied by the index ratio at settlement, and
the coupons are the real rate multiplied by the ratio at each payment date. Every
method here says in its name which space it is in, because a real price and a
nominal price differ by a factor that is 1.0 at issue and drifts away from it for
the life of the bond — so the error is invisible on a new issue and grows.

**Past the last published figure the index has to be projected**, which means a
linker priced beyond that point carries an inflation assumption. It is an
argument here, never a default: :meth:`LinkedBond.real_price_from_yield` needs no
projection at all, because a real yield discounts real flows, and
:meth:`LinkedBond.nominal_cashflows` needs one for every flow past the history
and will say so.

**The redemption is floored in index terms and the coupons are not.** On the US
structure the principal returned is the greater of the index ratio and one, so a
bond in cumulative deflation redeems at par while its coupons continue to shrink.
:meth:`LinkedBond.redemption_ratio` reports whether the floor binds rather than
applying it silently, because a bond whose floor is in the money is a different
instrument from one whose floor is not.

**Breakeven inflation is not the difference of two yields**, though that is how
it is quoted. The Fisher relation is multiplicative: one plus the nominal yield is
one plus the real yield times one plus inflation. Measured at a 4.5% nominal
against a 2.0% real, the quoted difference is 250.0bp and the exact rate is
245.1bp — 4.9bp, which is more than the bid-offer on the spread it is quoted as.
The gap is second order in the product of the two, so it widens with the level of
both: at 9% against 4% it is 19.2bp.

Sign and unit conventions follow :mod:`tenor.bond`: rates are decimals, prices
are per 100 of notional, and par is 100.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from .bond import Accrual, Bond
from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .schedule import Frequency, Period, Schedule
from .solve import Root, brent

#: Months the reference index lags the settlement date by. Three is the US and
#: Canadian convention, and the one modern index-linked gilts moved to in 2005.
DEFAULT_LAG_MONTHS = 3


class BadIndex(ValueError):
    """An index series, or a date, that the reference index cannot be built from."""


class BadLinker(ValueError):
    """An index-linked bond that does not describe a trade."""


class Unpublished(BadIndex):
    """A date whose reference index needs a figure that has not been published.

    Separate from :class:`BadIndex` because it is not a mistake. It is the
    ordinary condition of every payment date past the next quarter, and the
    answer is to project rather than to fix the call.
    """


class IndexInterpolation(str, Enum):
    """How the reference index moves between monthly figures.

    Named for the index rather than plainly, because
    :class:`tenor.curve.Interpolation` already exists and means something else
    entirely — how a discount curve moves between pillars. Two types with the
    same name in one library is a trap for whoever reads the import list.
    """

    #: Linear in day of month, which is what the US, Canadian and post-2005 gilt
    #: structures use. The index ratio changes every day.
    DAILY = "daily"
    #: The month's figure applies to the whole month. The pre-2005 gilt
    #: structure, with its eight-month lag, worked this way: the ratio steps once
    #: a month and is flat in between.
    MONTHLY = "monthly"


def _month_index(year: int, month: int) -> int:
    """Months since year zero, so month arithmetic is integer arithmetic."""
    return year * 12 + (month - 1)


def _from_month_index(value: int) -> tuple[int, int]:
    return value // 12, value % 12 + 1


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    following = date(year, month + 1, 1)
    return (following - date(year, month, 1)).days


@dataclass(frozen=True)
class ReferenceIndex:
    """A published monthly index, and the lagged interpolation built on it.

    ``values`` maps ``(year, month)`` to the figure published for that month. The
    series must be contiguous — a gap in the middle would otherwise be
    interpolated across silently, turning a missing print into a plausible number
    — and it must be positive, because the ratio divides by it.
    """

    values: dict[tuple[int, int], float]
    lag_months: int = DEFAULT_LAG_MONTHS
    interpolation: IndexInterpolation = IndexInterpolation.DAILY

    def __post_init__(self) -> None:
        if not self.values:
            raise BadIndex("an index series with no figures in it")
        if self.lag_months < 0:
            raise BadIndex(
                f"the lag is {self.lag_months} months; a reference index cannot "
                "look forward"
            )
        keys = sorted(_month_index(year, month) for year, month in self.values)
        for key in keys:
            year, month = _from_month_index(key)
            value = self.values[(year, month)]
            if not math.isfinite(value) or value <= 0.0:
                raise BadIndex(
                    f"the figure for {year}-{month:02d} is {value!r}; an index "
                    "level has to be positive and finite"
                )
        for earlier, later in itertools.pairwise(keys):
            if later != earlier + 1:
                missing = _from_month_index(earlier + 1)
                raise BadIndex(
                    f"the series jumps from {_from_month_index(earlier)[0]}-"
                    f"{_from_month_index(earlier)[1]:02d} to "
                    f"{_from_month_index(later)[0]}-"
                    f"{_from_month_index(later)[1]:02d}; "
                    f"{missing[0]}-{missing[1]:02d} is missing and interpolating "
                    "across a gap would turn an absent print into a number"
                )

    @property
    def first_month(self) -> tuple[int, int]:
        return min(self.values)

    @property
    def last_month(self) -> tuple[int, int]:
        return max(self.values)

    def published(self, year: int, month: int) -> float:
        try:
            return self.values[(year, month)]
        except KeyError:
            raise Unpublished(
                f"no index figure for {year}-{month:02d}; the series runs "
                f"{self.first_month[0]}-{self.first_month[1]:02d} to "
                f"{self.last_month[0]}-{self.last_month[1]:02d}"
            ) from None

    def reference_months(self, day: date) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two months a date's reference index interpolates between."""
        base = _month_index(day.year, day.month) - self.lag_months
        return _from_month_index(base), _from_month_index(base + 1)

    def reference(self, day: date) -> float:
        """The reference index for a settlement or payment date.

        Under daily interpolation this is

        ``I(m-lag) + (d - 1) / days_in_month * (I(m-lag+1) - I(m-lag))``

        where ``d`` is the day of the month of ``day`` and ``days_in_month`` is
        the length of *that* month, not of either index month. The convention
        deliberately uses the settlement month's length, so the ratio moves at a
        different daily rate in February than in March even when the underlying
        monthly step is the same.
        """
        first, second = self.reference_months(day)
        start = self.published(*first)
        if self.interpolation is IndexInterpolation.MONTHLY:
            return start
        if day.day == 1:
            # The 1st is exactly the earlier figure, and asking for the later one
            # would refuse a date that needs nothing unpublished.
            return start
        end = self.published(*second)
        length = _days_in_month(day.year, day.month)
        return start + (day.day - 1) / length * (end - start)

    def is_known(self, day: date) -> bool:
        """Whether this date's reference index needs no projection."""
        try:
            self.reference(day)
        except Unpublished:
            return False
        return True

    def last_known_date(self) -> date:
        """The latest date whose reference index is fully published.

        With a three-month lag and a series through August, the reference index
        is known through the 1st of December — the last date that interpolates
        between August and *nothing*, plus every date in November that
        interpolates between July and August. So this returns the 1st of the
        month ``lag`` months after the last published figure.
        """
        year, month = self.last_month
        year, month = _from_month_index(_month_index(year, month) + self.lag_months)
        return date(year, month, 1)

    def projected(self, day: date, *, rate: float, anchor: date | None = None) -> float:
        """The reference index at ``day``, projecting past the published history.

        Beyond :meth:`last_known_date` the index is grown from the last known
        reference value at a constant annual ``rate``, compounded over the ACT/365
        fraction. That is an assumption, and it is the caller's: a flat inflation
        path is the simplest thing that is not silently zero, and anyone who wants
        a seasonal or forward-implied path should build the series instead.
        """
        if self.is_known(day):
            return self.reference(day)
        start = anchor if anchor is not None else self.last_known_date()
        # Ordered before the year fraction rather than after: `year_fraction`
        # refuses a backwards interval itself, with a message about day counts,
        # which names the symptom rather than the problem.
        if day < start:
            raise BadIndex(
                f"{day.isoformat()} is before the projection anchor "
                f"{start.isoformat()} and is not published either"
            )
        base = self.reference(start)
        years = year_fraction(start, day, Basis.ACT_365F)
        # float ** float is complex in general, so the narrowing happens here,
        # as it does in `tenor.bond._discount`.
        return base * float((1.0 + rate) ** years)

    def ratio(self, day: date, base: float) -> float:
        """Reference index at ``day`` over the bond's base index."""
        if base <= 0.0:
            raise BadIndex(f"the base index is {base!r} and has to be positive")
        return self.reference(day) / base

    def realised_inflation(self, start: date, end: date) -> float:
        """Annualised inflation between two dates, from the reference index.

        Stated as an annual rate compounded over the ACT/365 fraction, so it is
        directly comparable with a yield. Over a short interval the number is
        volatile for the usual reason — a monthly print annualised is twelve
        times the noise — and the interval is the caller's to choose.
        """
        if end <= start:
            raise BadIndex(
                f"{end.isoformat()} is not after {start.isoformat()}, so there is "
                "no interval to annualise over"
            )
        growth = self.reference(end) / self.reference(start)
        years = year_fraction(start, end, Basis.ACT_365F)
        return float(growth ** (1.0 / years)) - 1.0


@dataclass(frozen=True)
class LinkedCashflow:
    """One payment of an index-linked bond, in both spaces."""

    day: date
    #: The payment as a fixed-coupon bond would state it, per 100 of notional.
    real_amount: float
    #: The index ratio applied to it.
    index_ratio: float
    #: Coupon periods from settlement, ``k - 1 + w``, as in :mod:`tenor.bond`.
    periods: float
    redemption: bool = False
    #: True when this flow's ratio came from a projection rather than a print.
    projected: bool = False

    @property
    def nominal_amount(self) -> float:
        return self.real_amount * self.index_ratio


@dataclass(frozen=True)
class LinkedBond:
    """An index-linked bullet bond.

    ``coupon`` is the *real* annual rate as a decimal, and ``base_index`` is the
    reference index at the bond's dated date — the denominator of every index
    ratio for its life. It is a property of the bond, fixed at issue, and is
    passed rather than derived so that a bond whose dated date predates the
    supplied index history can still be priced.
    """

    effective: date
    maturity: date
    coupon: float
    base_index: float
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.ACT_ACT_ISDA
    redemption: float = 100.0
    calendar: Calendar = WEEKENDS_ONLY
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    accrual: Accrual = Accrual.PERIOD_FRACTION
    #: Whether the principal is floored at par in index terms, as on US TIPS.
    #: Index-linked gilts have no such floor, so this is a real choice.
    deflation_floor: bool = True
    label: str = ""
    _schedule: list[Schedule] = field(
        default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadLinker(
                f"{self.name} matures on {self.maturity.isoformat()}, which is not "
                f"after it was issued on {self.effective.isoformat()}"
            )
        if self.base_index <= 0.0:
            raise BadLinker(
                f"{self.name} has a base index of {self.base_index!r}; every index "
                "ratio divides by it"
            )
        if self.redemption <= 0.0:
            raise BadLinker(
                f"{self.name} redeems at {self.redemption!r}, which is not a "
                "principal"
            )

    @property
    def name(self) -> str:
        return self.label or (
            f"{self.coupon:.3%} linker of {self.maturity.isoformat()}"
        )

    def _real_bond(self) -> Bond:
        """The fixed-coupon bond this linker is, in real terms.

        Every real-space quantity — real price from real yield, real yield from
        real price, real duration, real convexity — is that bond's, unchanged.
        Restating them here would be two implementations of the street
        discounting convention, and the conventions are the part worth having
        only once.
        """
        return Bond(
            effective=self.effective,
            maturity=self.maturity,
            coupon=self.coupon,
            frequency=self.frequency,
            basis=self.basis,
            redemption=self.redemption,
            calendar=self.calendar,
            rolling=self.rolling,
            accrual=self.accrual,
            label=self.name,
        )

    def schedule(self) -> Schedule:
        return self._real_bond().schedule()

    @property
    def periodic_coupon(self) -> float:
        """The real coupon per period, per 100 of notional."""
        return self._real_bond().periodic_coupon

    def current_period(self, settlement: date) -> Period:
        return self._real_bond().current_period(settlement)

    # -- real space ----------------------------------------------------------

    def real_accrued(self, settlement: date) -> float:
        return self._real_bond().accrued(settlement)

    def real_dirty_price(self, real_yield: float, settlement: date) -> float:
        return self._real_bond().dirty_price(real_yield, settlement)

    def real_clean_price(self, real_yield: float, settlement: date) -> float:
        return self._real_bond().clean_price(real_yield, settlement)

    def real_yield_from_clean(self, price: float, settlement: date) -> Root:
        return self._real_bond().yield_from_clean(price, settlement)

    def real_modified_duration(self, real_yield: float, settlement: date) -> float:
        """Sensitivity of the *real* price to the real yield.

        This is the duration that is quoted for a linker, and it is not the
        sensitivity of the invoice to anything. The invoice moves with the index
        ratio as well, and the ratio's own movement is the instrument's reason to
        exist rather than a risk to be hedged.
        """
        return self._real_bond().modified_duration(real_yield, settlement)

    def real_convexity(self, real_yield: float, settlement: date) -> float:
        return self._real_bond().convexity(real_yield, settlement)

    # -- index space ---------------------------------------------------------

    def index_ratio(self, index: ReferenceIndex, settlement: date) -> float:
        """Reference index at settlement over the base index."""
        return index.ratio(settlement, self.base_index)

    def redemption_ratio(
        self, index: ReferenceIndex, *, projection: float | None = None
    ) -> tuple[float, bool]:
        """The ratio applied to the principal, and whether the floor bound it.

        The floor is on the principal only. A bond in cumulative deflation
        redeems at par and its coupons keep shrinking, which is why the two are
        computed separately everywhere here.
        """
        if projection is None:
            raw = index.ratio(self.maturity, self.base_index)
        else:
            raw = (
                index.projected(self.maturity, rate=projection) / self.base_index
            )
        if self.deflation_floor and raw < 1.0:
            return 1.0, True
        return raw, False

    def nominal_cashflows(
        self,
        index: ReferenceIndex,
        settlement: date,
        *,
        projection: float | None = None,
    ) -> tuple[LinkedCashflow, ...]:
        """Every remaining flow, in real and nominal terms.

        ``projection`` is the annual inflation rate used for any payment date
        past the published history. Omitting it is not a default of zero: a flow
        that needs a projection and has not been given one raises
        :class:`Unpublished`, naming the date. A bond with more than a quarter to
        run always has such flows, so the argument is effectively required — and
        making it explicit is the point, since the number it produces is an
        assumption about the future price level.
        """
        flows = []
        for flow in self._real_bond().cashflows(settlement):
            known = index.is_known(flow.day)
            if known:
                ratio = index.ratio(flow.day, self.base_index)
            elif projection is None:
                raise Unpublished(
                    f"the flow on {flow.day.isoformat()} needs a reference index "
                    f"past the published series, which runs to "
                    f"{index.last_known_date().isoformat()}; pass projection= to "
                    f"state the inflation rate it should be grown at"
                )
            else:
                ratio = (
                    index.projected(flow.day, rate=projection) / self.base_index
                )
            if not flow.redemption:
                flows.append(
                    LinkedCashflow(
                        day=flow.day,
                        real_amount=flow.amount,
                        index_ratio=ratio,
                        periods=flow.periods,
                        projected=not known,
                    )
                )
                continue
            # The fixed-coupon bond bundles the final coupon into the redemption
            # payment, and the two are indexed differently: the floor is on the
            # principal only. Bundled, a floored ratio would floor the last
            # coupon as well, which is a more valuable bond than this one — and it
            # would only show up on a bond in cumulative deflation, so it would
            # not show up at all in most testing. They are emitted separately.
            final_coupon = flow.amount - self.redemption
            if final_coupon > 0.0:
                flows.append(
                    LinkedCashflow(
                        day=flow.day,
                        real_amount=final_coupon,
                        index_ratio=ratio,
                        periods=flow.periods,
                        projected=not known,
                    )
                )
            principal_ratio = (
                max(ratio, 1.0) if self.deflation_floor else ratio
            )
            flows.append(
                LinkedCashflow(
                    day=flow.day,
                    real_amount=self.redemption,
                    index_ratio=principal_ratio,
                    periods=flow.periods,
                    redemption=True,
                    projected=not known,
                )
            )
        return tuple(flows)

    def settlement_amount(
        self, real_clean_price: float, index: ReferenceIndex, settlement: date
    ) -> float:
        """What actually changes hands, per 100 of notional.

        ``(real clean price + real accrued) * index ratio``. Multiplying the
        *clean* price by the ratio and adding unindexed accrued is the mistake
        this method exists to make impossible; it is small on a new issue and
        grows with cumulative inflation, so it survives testing on recent paper.
        """
        ratio = self.index_ratio(index, settlement)
        return (real_clean_price + self.real_accrued(settlement)) * ratio

    def nominal_price_from_curve(
        self,
        curve: DiscountCurve,
        index: ReferenceIndex,
        settlement: date,
        *,
        projection: float,
    ) -> float:
        """Present value of the projected nominal flows off a nominal curve.

        This is the arbitrage-free way round: project the index, then discount
        the resulting money flows at nominal rates. It needs an inflation path and
        a nominal curve, and it produces a *nominal* price — the thing the invoice
        is compared against, not the thing the bond is quoted at.
        """
        total = 0.0
        for flow in self.nominal_cashflows(
            index, settlement, projection=projection
        ):
            total += flow.nominal_amount * curve.discount(flow.day)
        return total


def breakeven_inflation(
    nominal_yield: float, real_yield: float, *, exact: bool = True
) -> float:
    """Inflation that makes a nominal and a real yield equivalent.

    With ``exact`` the Fisher relation is inverted properly,
    ``(1 + n) / (1 + r) - 1``. Without it, the quoted convention: the plain
    difference. The difference between the two is the cross term ``r * pi``, so
    it grows with the level of both — 4.9bp at a 4.5% nominal against a 2.0%
    real, 19.2bp at 9% against 4%.

    Neither is a market expectation of inflation. Both are the expectation plus
    an inflation risk premium plus a liquidity difference between two bonds, and
    the decomposition is not identified by two prices.
    """
    if real_yield <= -1.0:
        raise ValueError(
            f"a real yield of {real_yield!r} is at or below -100%, where the "
            "Fisher relation has no inverse"
        )
    if not exact:
        return nominal_yield - real_yield
    return (1.0 + nominal_yield) / (1.0 + real_yield) - 1.0


def real_yield_from_breakeven(nominal_yield: float, inflation: float) -> float:
    """The real yield implied by a nominal yield and an inflation rate."""
    if inflation <= -1.0:
        raise ValueError(
            f"an inflation rate of {inflation!r} is at or below -100%"
        )
    return (1.0 + nominal_yield) / (1.0 + inflation) - 1.0


def breakeven_from_prices(
    nominal: Bond,
    linker: LinkedBond,
    *,
    nominal_clean: float,
    real_clean: float,
    settlement: date,
    exact: bool = True,
) -> float:
    """Breakeven inflation between two actual bonds.

    Each yield is solved from its own quoted clean price, so this carries every
    convention difference between the two instruments — different day count
    bases, different frequencies, different maturities — which is exactly what a
    quoted breakeven does. The pair being close in maturity is the caller's
    responsibility, and a mismatch shows up here as a term-premium difference
    wearing the name of an inflation expectation.
    """
    nominal_root = nominal.yield_from_clean(nominal_clean, settlement)
    real_root = linker.real_yield_from_clean(real_clean, settlement)
    if not nominal_root.converged or not real_root.converged:
        raise BadLinker(
            "a yield did not converge, so the breakeven between them is not a "
            "number: "
            f"nominal residual {nominal_root.residual:.3g}, "
            f"real residual {real_root.residual:.3g}"
        )
    return breakeven_inflation(nominal_root.value, real_root.value, exact=exact)


def implied_inflation_from_price(
    linker: LinkedBond,
    curve: DiscountCurve,
    index: ReferenceIndex,
    settlement: date,
    *,
    nominal_price: float,
    low: float = -0.05,
    high: float = 0.20,
) -> Root:
    """The flat inflation path that reprices a linker off a nominal curve.

    A different quantity from :func:`breakeven_from_prices` and worth having
    both. That one compares two bonds and inherits everything that differs
    between them; this one takes the nominal curve as given and asks what
    constant inflation rate makes this bond's projected money flows worth its
    invoice. It needs a curve and gives a number that is not contaminated by a
    second bond's liquidity.
    """

    def objective(rate: float) -> float:
        return (
            linker.nominal_price_from_curve(
                curve, index, settlement, projection=rate
            )
            - nominal_price
        )

    return brent(objective, low, high)


@dataclass(frozen=True)
class IndexAccretion:
    """Accretion over an interval, split by whether it is already published."""

    start: date
    end: date
    start_ratio: float
    end_ratio: float
    #: The part of the interval whose reference index is published.
    known_through: date
    known_ratio: float

    @property
    def total(self) -> float:
        """Total accretion as a fraction of the starting ratio."""
        return self.end_ratio / self.start_ratio - 1.0

    @property
    def known(self) -> float:
        return self.known_ratio / self.start_ratio - 1.0

    @property
    def projected(self) -> float:
        return self.end_ratio / self.known_ratio - 1.0

    @property
    def known_fraction(self) -> float:
        """How much of the accretion is arithmetic rather than assumption.

        The interesting number, and the one the lag produces. How large it is
        depends on where in the publication cycle the question is asked, which is
        the part that is easy to state wrongly. With a three-month lag the
        reference index is known out to the first of the month three months after
        the last published figure — so from mid-September with a series through
        August it is known to 1 November, which is six weeks, not three months.
        Measured on a 2% path from that date: 51.4% of the next quarter's
        accretion is already published, 25.7% of the next six months' and 12.7%
        of the next year's. A caller hedging a three-month inflation exposure
        with a linker is trading something about half of whose payoff is already
        determined, and the share falls through the month until the next print.
        """
        total = self.total
        if total == 0.0:
            raise BadIndex(
                "the interval has no accretion in it, so the published share of "
                "it is 0/0; compare the ratios instead"
            )
        return self.known / total


def index_accretion(
    index: ReferenceIndex,
    start: date,
    end: date,
    base: float,
    *,
    projection: float,
) -> IndexAccretion:
    """Split the index accretion between two dates into known and projected."""
    if end <= start:
        raise BadIndex(
            f"{end.isoformat()} is not after {start.isoformat()}"
        )
    known_through = min(index.last_known_date(), end)
    if known_through < start:
        known_through = start
    return IndexAccretion(
        start=start,
        end=end,
        start_ratio=index.projected(start, rate=projection) / base,
        end_ratio=index.projected(end, rate=projection) / base,
        known_through=known_through,
        known_ratio=index.projected(known_through, rate=projection) / base,
    )
