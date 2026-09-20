"""The instruments a curve is actually built from.

Nobody has a set of discount factors. What they have is a screen: money market
deposits out to a year, futures for the year or two after that, and par swap
rates beyond. Each of those is a quote, and each quote is a statement that one
particular combination of discount factors comes to a particular number. This
module is those statements.

Every instrument answers one question — :meth:`par_error`, how far off par it is
against a given curve — and the bootstrapper in :mod:`tenor.bootstrap` does
nothing but drive that to zero. Pricing and bootstrapping are then the same
code, which is what makes "every input reprices" an identity rather than a
second implementation that happens to agree.

**Futures are not forwards, and the difference is not a rounding error.** A
futures contract is margined daily; a forward rate agreement settles once at
maturity. The margin flows are positively correlated with the rate — you receive
cash when rates rise, and reinvest it at the higher rate — so a futures contract
is worth more to its holder than the equivalent forward, and its implied rate
sits above the forward rate. The gap is the convexity adjustment. It is
negligible at the front and emphatically not further out, because it grows with
the *product* of the two maturities and so roughly with the square. On 1% normal
volatility a three-month contract starting in one year is adjusted by 0.6 basis
points, which nobody would miss; the same contract starting in ten years is
adjusted by **51** basis points, which is most of a quarter-point.

That figure is worth stating because it is the one people expect to be small.
Hull's worked example is the same arithmetic: at 1.2% volatility and eight
years, the adjustment is 47.5 basis points. Both numbers are in the test suite.

So :class:`Future` carries the adjustment explicitly and reports it. A
bootstrapper that quietly treats the futures rate as a forward rate is wrong by
an amount nobody can see in the output, and it is wrong in the same direction
every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .calendar import Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .schedule import Frequency, Schedule, generate


class BadInstrument(ValueError):
    """An instrument that does not describe a trade."""


@dataclass(frozen=True)
class Deposit:
    """A money market deposit: one payment, simple interest.

    The rate is simple because the market quotes it simple, over a period of at
    most a year. Reading it as annually compounded understates the discount
    factor by a few tenths of a basis point at six months, which is inside the
    bid-offer and outside the tolerance any repricing test should use.
    """

    start: date
    maturity: date
    rate: float
    basis: Basis = Basis.ACT_360
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.start:
            raise BadInstrument(
                f"deposit {self.name} matures on {self.maturity.isoformat()}, which "
                f"is not after it starts on {self.start.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"deposit to {self.maturity.isoformat()}"

    @property
    def accrual(self) -> float:
        return year_fraction(self.start, self.maturity, self.basis)

    def par_error(self, curve: DiscountCurve) -> float:
        """How far the curve is from repricing this deposit, in present value.

        One unit lent at the start comes back as ``1 + r * tau`` at maturity, so
        the curve prices the deposit correctly when the discounted redemption
        equals the discounted principal.
        """
        growth = 1.0 + self.rate * self.accrual
        return curve.discount(self.maturity) * growth - curve.discount(self.start)


@dataclass(frozen=True)
class Future:
    """A short-term interest rate future, quoted as ``100 - rate``.

    ``convexity`` is in rate terms and is *subtracted* from the futures rate to
    give the forward rate the curve is built from. Zero by default, because a
    convexity adjustment is a model output — it needs a volatility — and
    defaulting to a number would be asserting one. :func:`ho_lee_convexity`
    computes it when a volatility is available.
    """

    start: date
    end: date
    price: float
    basis: Basis = Basis.ACT_360
    convexity: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise BadInstrument(
                f"future {self.name} ends on {self.end.isoformat()}, which is not "
                f"after it starts on {self.start.isoformat()}"
            )
        if self.convexity < 0.0:
            raise BadInstrument(
                f"future {self.name} has a convexity adjustment of "
                f"{self.convexity!r}. The adjustment is the amount by which the "
                "futures rate exceeds the forward rate, and it is never negative: "
                "daily margining pays the holder when rates rise, which is worth "
                "something rather than nothing."
            )

    @property
    def name(self) -> str:
        return self.label or f"future {self.start.isoformat()}/{self.end.isoformat()}"

    @property
    def futures_rate(self) -> float:
        """The rate implied by the quoted price, before any adjustment."""
        return (100.0 - self.price) / 100.0

    @property
    def forward_rate(self) -> float:
        """The rate the curve is built from: the futures rate less convexity."""
        return self.futures_rate - self.convexity

    @property
    def accrual(self) -> float:
        return year_fraction(self.start, self.end, self.basis)

    def par_error(self, curve: DiscountCurve) -> float:
        growth = 1.0 + self.forward_rate * self.accrual
        return curve.discount(self.end) * growth - curve.discount(self.start)


def ho_lee_convexity(volatility: float, start: float, end: float) -> float:
    """The futures-forward convexity adjustment under a Ho-Lee short rate.

    ``0.5 * sigma^2 * t1 * t2``, with ``sigma`` the *absolute* (normal)
    volatility of the short rate — 0.01 for 100 basis points a year, not 20%.
    The two are routinely confused and differ by two orders of magnitude, so
    the units are worth stating twice.

    Ho-Lee rather than anything richer because the adjustment is a small
    correction whose leading term is model-independent: any one-factor model
    with the same short-rate volatility gives the same answer to first order,
    and the differences between them are second order on a number that is
    already a few basis points. Spending a mean-reverting model on it buys
    precision in a correction, which is the wrong place to spend it.
    """
    if volatility < 0.0:
        raise BadInstrument(f"a volatility is not negative, got {volatility!r}")
    if start < 0.0 or end < start:
        raise BadInstrument(
            f"a convexity adjustment runs from {start!r} to {end!r}, which is not "
            "forwards in time"
        )
    return 0.5 * volatility * volatility * start * end


@dataclass(frozen=True)
class Swap:
    """A par interest rate swap: fixed against floating, priced at zero.

    The floating leg is valued as ``P(effective) - P(maturity)``. That identity
    — the whole leg telescopes, whatever its frequency or day count — holds
    when the leg is forecast off the same curve it is discounted on, which is
    the single-curve world this library models. It stopped being true of the
    real market in 2008, when forecasting and discounting separated; what
    remains true is that a single-curve bootstrap is the thing a multi-curve one
    is built out of, and that pretending otherwise here would be modelling a
    basis this library has no inputs for.

    The fixed leg is the swap rate times the annuity: each period's accrual
    under the fixed basis, discounted to its own payment date.
    """

    effective: date
    maturity: date
    rate: float
    frequency: Frequency = Frequency.SEMI_ANNUAL
    basis: Basis = Basis.THIRTY_360_BOND
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING
    label: str = ""

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadInstrument(
                f"swap {self.name} matures on {self.maturity.isoformat()}, which is "
                f"not after it starts on {self.effective.isoformat()}"
            )

    @property
    def name(self) -> str:
        return self.label or f"swap to {self.maturity.isoformat()}"

    def schedule(self) -> Schedule:
        from .calendar import WEEKENDS_ONLY

        return generate(
            self.effective,
            self.maturity,
            self.frequency,
            calendar=self.calendar if self.calendar is not None else WEEKENDS_ONLY,
            rolling=self.rolling,
        )

    def annuity(self, curve: DiscountCurve) -> float:
        """The present value of one unit of fixed rate.

        Accrual runs on the *unadjusted* period boundaries and discounting on
        the payment date. Mixing the two up - accruing on the rolled dates -
        moves a swap's fixed leg by a day of interest whenever a period end
        lands on a weekend, which is most of them.
        """
        total = 0.0
        for period in self.schedule():
            accrual = year_fraction(period.start, period.end, self.basis)
            total += accrual * curve.discount(period.payment)
        return total

    @property
    def final_payment(self) -> date:
        """When the last cash actually moves.

        Not :attr:`maturity`. A swap maturing on a weekend or a holiday has its
        final payment rolled forward, and that rolled date is the one whose
        discount factor the quote pins down. Using the unadjusted maturity
        instead puts the curve's pillar a day or three before the payment it is
        supposed to price, which shows up as a bootstrap that cannot evaluate
        its own last instrument.
        """
        return self.schedule()[-1].payment

    @property
    def first_accrual(self) -> date:
        """When the legs start accruing, rolled onto a business day."""
        return self.schedule()[0].adjusted_start

    def floating_value(self, curve: DiscountCurve) -> float:
        return curve.discount(self.first_accrual) - curve.discount(self.final_payment)

    def fixed_value(self, curve: DiscountCurve) -> float:
        return self.rate * self.annuity(curve)

    def par_rate(self, curve: DiscountCurve) -> float:
        """The fixed rate that would make this swap worth zero on ``curve``."""
        annuity = self.annuity(curve)
        if annuity <= 0.0:
            raise BadInstrument(
                f"swap {self.name} has an annuity of {annuity!r}, so no fixed rate "
                "makes it worth zero"
            )
        return self.floating_value(curve) / annuity

    def par_error(self, curve: DiscountCurve) -> float:
        """The swap's value, which is zero when the curve reprices it."""
        return self.floating_value(curve) - self.fixed_value(curve)


#: What the bootstrapper needs from anything it is handed. Deliberately not a
#: base class: these three share no state and no behaviour, only the question
#: they can answer, and inheritance would invent a relationship to carry it.
Instrument = Deposit | Future | Swap


def maturity_of(instrument: Instrument) -> date:
    """The date whose discount factor an instrument pins down.

    For a swap that is its last *payment* date, not its maturity: the two
    differ whenever maturity lands on a weekend or a holiday, and it is the
    payment the curve has to be able to discount.
    """
    if isinstance(instrument, Future):
        return instrument.end
    if isinstance(instrument, Swap):
        return instrument.final_payment
    return instrument.maturity


def start_of(instrument: Instrument) -> date:
    """The date an instrument's own accrual begins."""
    if isinstance(instrument, Swap):
        return instrument.first_accrual
    return instrument.start


__all__ = [
    "BadInstrument",
    "Deposit",
    "Future",
    "Instrument",
    "Swap",
    "ho_lee_convexity",
    "maturity_of",
    "start_of",
]
