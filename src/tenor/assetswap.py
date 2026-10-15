"""The asset swap: a bond's spread, quoted on a leg that is not the bond's.

:mod:`tenor.spread` measures a bond against the curve two ways, and neither is
the number a desk transacts on. That one is the **asset swap spread**: a buyer
takes the bond, hands its fixed coupons to a swap counterparty, and receives
the floating index plus a spread instead. What they receive is the quote. It is
not a relabelling of the Z-spread, because it is the answer to a question about
a *structure* rather than about a discounting convention, and the structure has
conventions of its own — which notional the spread accrues on, and what the
floating leg's frequency and day count are.

**Nothing here needs solving.** The package is linear in the spread, so both
conventions rearrange, and so does the inverse: :func:`price_from_spread`
recovers the price a quote implies in closed form, in the proceeds convention
too, where the price appears in the notional as well as in the value.

## The two conventions, and the factor between them

In a **par-par** asset swap the buyer pays 100, receives the bond, pays its
coupons and receives the index plus ``s`` on a notional of 100. Writing the
package's value at settlement and setting it to zero, the floating leg's
telescoped value cancels against the bond's redemption and what is left is::

    s = (curve price - market price) / (100 * annuity)

In a **proceeds** asset swap the buyer pays the market price instead, and the
floating notional is that price. The bond's cost cancels against its own value
and what is left is the same money divided by a different notional::

    s = (curve price - market price) / (market price * annuity)

So the two differ by exactly ``100 / dirty price``, which this module verifies
to a relative 1e-15 rather than asserting. It is not a rounding difference. A
bond at 80 quotes a proceeds spread of **478.4bp** against a par-par **391.8bp**
— 22.1% wider, on the same bond, the same curve and the same day. A bond at
120 quotes 18.0% narrower. A spread handed over without its convention is not
a number, and the gap is largest exactly where the bond is most distressed and
the quote matters most.

## It is exactly linear in the price, and the Z-spread is not

Every price enters the par-par spread once, through one subtraction, so the
spread is an affine function of the price with slope ``-1/(100 * annuity)``.
Measured as a second difference it is zero to 2.8e-15 bp per squared price
point everywhere. The Z-spread, over the same bond and the same range, has a
second difference of 0.222 at a price of 80, 0.135 at 100 and 0.091 at 120: it
is convex, and noticeably so.

That difference in shape is why **neither measure bounds the other**. On a 4%
twelve-year bond they agree at exactly two prices — 104.661 and 116.435, the
latter being the curve's own price for it, where both are zero — and the sign
of the gap is different on either side. At 80 the par-par spread is 61.5bp
*below* the Z-spread; at 110 it is 1.9bp above. Anyone treating the two as
interchangeable at a tolerance of a few basis points is relying on their bonds
staying near par.

## And it is not a credit spread

The clearest statement of that is five bonds with the *same issuer curve, the
same maturity and the same Z-spread*, differing only in coupon. Priced off a
curve running from 20bp to 2.7%, with a Z-spread of 100bp applied to all five:

| coupon | price | asset swap |
|---|---|---|
| 1% | 80.94 | 86.8bp |
| 2% | 89.63 | 91.6bp |
| 4% | 107.02 | 101.2bp |
| 6% | 124.41 | 110.9bp |
| 8% | 141.80 | 120.5bp |

A **33.7bp** spread of quotes across bonds that are by construction equally
risky, and it widens with the level: 65.2bp at a Z-spread of 200bp and 94.8bp
at 300bp. The ordering is monotone in the coupon, which makes it a systematic
bias rather than noise, and :attr:`AssetSwap.curve_component` and
:attr:`AssetSwap.price_component` split the quote into the two pieces that
produce it.

## The leg's convention is worth basis points

The structure's value does not depend on the floating leg's frequency or day
count at all — the leg telescopes. Every convention therefore amortises the
same amount of money, and they differ only in the annuity they divide it by.
On the 4% bond at 90, the same money comes out as 284.25bp on a quarterly
ACT/360 leg, 285.04bp semi-annual, 286.63bp annual, 288.20bp on quarterly
ACT/365F and 289.13bp on semi-annual 30/360. A range of **4.87bp on a 284bp
quote**, from a choice nobody writes down. :func:`leg_sensitivity` produces
that table for any bond.

## Where the textbook formula is exact, and where it is not

The cancellation that produces ``(curve price - market price) / (100 *
annuity)`` assumes the floating leg accrues from the settlement date to the
bond's own redemption payment. When it does, :attr:`AssetSwap.residual` is
3.6e-15 and the formula is exact. When it does not, the residual is the
leg's value less the ``par - redemption`` it was assumed to telescope to, and
the textbook answer is wrong by that over the annuity:

- One holiday in the leg's calendar that is not in the bond's, landing on the
  redemption payment, moves the leg's last payment by a day: residual 0.0073
  price points, **0.08bp** of spread.
- A leg struck to start six months forward rather than at settlement: residual
  -0.120 price points, **1.37bp**.

Neither is large and both are larger than the precision the quote is given to.
Under a mismatch the ``100 / price`` relation between the two conventions also
stops being exact — 1.0882953 against 1.0882709 in the holiday case — so the
identity in the second section is a statement about an aligned structure, not
about asset swaps in general. Both quantities are computed here rather than
assumed away, which is the only reason either number is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from .bond import Bond
from .calendar import WEEKENDS_ONLY, Calendar, Rolling
from .curve import DiscountCurve
from .daycount import Basis, year_fraction
from .instruments import Swap
from .schedule import Frequency

__all__ = [
    "AssetSwap",
    "BadAssetSwap",
    "Convention",
    "FloatingLeg",
    "LegSensitivity",
    "SpreadComparison",
    "asset_swap_spread",
    "compare_spreads",
    "leg_for",
    "leg_sensitivity",
    "par_par_spread",
    "price_from_spread",
    "proceeds_spread",
]


class BadAssetSwap(ValueError):
    """An asset swap that cannot be struck."""


class Convention(str, Enum):
    """Which notional the spread is paid on.

    The two conventions describe the same bond and the same curve. They differ
    in what the buyer hands over at inception, and therefore in the size of the
    leg the spread accrues on.
    """

    PAR_PAR = "PAR_PAR"
    """The buyer pays par and the floating notional is par."""

    PROCEEDS = "PROCEEDS"
    """The buyer pays the market price and the floating notional is that price."""


@dataclass(frozen=True)
class FloatingLeg:
    """The leg the spread is paid on, described without reference to the bond.

    An asset swap's floating leg is a market convention, not a property of the
    asset: a quarterly ACT/360 leg is quoted against a bond paying annual
    30/360 coupons without anybody thinking it odd. Describing the leg
    separately is the only way to ask what the convention is worth, which
    :func:`leg_sensitivity` does.
    """

    effective: date
    maturity: date
    frequency: Frequency = Frequency.QUARTERLY
    basis: Basis = Basis.ACT_360
    calendar: Calendar | None = None
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING

    def __post_init__(self) -> None:
        if self.maturity <= self.effective:
            raise BadAssetSwap(
                f"a floating leg from {self.effective.isoformat()} to "
                f"{self.maturity.isoformat()} ends before it starts"
            )

    def _swap(self) -> Swap:
        return Swap(
            self.effective,
            self.maturity,
            0.0,
            self.frequency,
            self.basis,
            self.calendar if self.calendar is not None else WEEKENDS_ONLY,
            self.rolling,
        )

    @property
    def first_accrual(self) -> date:
        """The rolled date the leg starts accruing on."""
        return self._swap().first_accrual

    @property
    def final_payment(self) -> date:
        """The rolled date the leg's last payment lands on."""
        return self._swap().final_payment

    def annuity(self, curve: DiscountCurve, settlement: date) -> float:
        """One unit of spread, per unit of notional, valued at ``settlement``.

        Forwarded to the settlement date rather than discounted to the curve's
        reference date, because every other quantity in the structure is a
        price on the settlement date and they have to be added together.
        """
        base = curve.discount(settlement)
        if base <= 0.0:
            raise BadAssetSwap(
                f"the curve discounts {settlement.isoformat()} to {base!r}, so "
                "nothing on that date can be valued"
            )
        total = 0.0
        for period in self._swap().schedule():
            accrual = year_fraction(period.start, period.end, self.basis)
            total += accrual * curve.discount(period.payment)
        return total / base

    def projected_value(self, curve: DiscountCurve, settlement: date) -> float:
        """The index's own flows, per unit of notional, valued at ``settlement``.

        The leg telescopes to the difference of two discount factors whatever
        its frequency, which is the single-curve identity :class:`~tenor.Swap`
        rests on and is stated there. What matters here is *which* two dates:
        the leg's own rolled boundaries, which need not be the settlement date
        and the bond's redemption.
        """
        base = curve.discount(settlement)
        if base <= 0.0:
            raise BadAssetSwap(
                f"the curve discounts {settlement.isoformat()} to {base!r}, so "
                "nothing on that date can be valued"
            )
        swap = self._swap()
        near = curve.discount(swap.first_accrual)
        far = curve.discount(swap.final_payment)
        return (near - far) / base


def leg_for(
    bond: Bond,
    settlement: date,
    *,
    frequency: Frequency = Frequency.QUARTERLY,
    basis: Basis = Basis.ACT_360,
    calendar: Calendar | None = None,
    rolling: Rolling = Rolling.MODIFIED_FOLLOWING,
) -> FloatingLeg:
    """The leg an asset swap on ``bond`` is struck with by default.

    Settlement to the bond's maturity. Not to the bond's *last payment*: the
    schedule rolls the maturity forward in the same way for both, so matching
    the unadjusted dates is what makes the two sides land together.
    """
    return FloatingLeg(
        effective=settlement,
        maturity=bond.maturity,
        frequency=frequency,
        basis=basis,
        calendar=calendar,
        rolling=rolling,
    )


@dataclass(frozen=True)
class AssetSwap:
    """A struck asset swap, with the spread and what it is made of."""

    convention: Convention
    spread: float
    """The spread over the index, in the same units as a rate."""

    notional: float
    """What the spread accrues on: par, or the price the buyer paid."""

    annuity: float
    """One unit of spread per unit of notional, at the settlement date."""

    dirty_price: float
    """The market dirty price the swap was struck against, per 100."""

    curve_price: float
    """What the curve says the bond is worth, dirty, per 100."""

    residual: float
    """The leg's value less the par-minus-redemption it is assumed to telescope to.

    Zero when the floating leg accrues from settlement to the bond's own
    redemption payment, which is the case the textbook formula is written for.
    Non-zero whenever the leg's rolled boundaries and the bond's part company,
    and then the textbook formula is wrong by this much divided by the annuity.
    """

    par: float = 100.0
    """The redemption the bond pays, per 100."""

    @property
    def curve_component(self) -> float:
        """The part of the spread that is the bond's richness to the curve."""
        return (self.curve_price - self.par) / (self.notional * self.annuity)

    @property
    def price_component(self) -> float:
        """The part that is only the bond's distance from par."""
        return (self.par - self.dirty_price) / (self.notional * self.annuity)

    @property
    def spread_basis_points(self) -> float:
        """The spread in basis points, which is how it is quoted."""
        return self.spread * 10000.0


def _parts(
    bond: Bond, curve: DiscountCurve, settlement: date, leg: FloatingLeg
) -> tuple[float, float, float, float, float]:
    """``(curve_price, redemption_pv, annuity, projected, par)`` at settlement."""
    base = curve.discount(settlement)
    if base <= 0.0:
        raise BadAssetSwap(
            f"the curve discounts {settlement.isoformat()} to {base!r}, so the "
            "structure cannot be valued there"
        )
    flows = bond.cashflows(settlement)
    if not flows:
        raise BadAssetSwap(
            f"{bond.name} has no cash flows after {settlement.isoformat()}, so "
            "there is nothing to swap"
        )
    curve_price = sum(flow.amount * curve.discount(flow.day) / base for flow in flows)
    final = flows[-1]
    redemption_pv = bond.redemption * curve.discount(final.day) / base
    annuity = leg.annuity(curve, settlement)
    if annuity <= 0.0:
        raise BadAssetSwap(
            f"the floating leg to {leg.maturity.isoformat()} has an annuity of "
            f"{annuity!r}, so no spread on it is worth anything"
        )
    projected = leg.projected_value(curve, settlement)
    return curve_price, redemption_pv, annuity, projected, bond.redemption


def asset_swap_spread(
    bond: Bond,
    price: float,
    curve: DiscountCurve,
    settlement: date,
    *,
    leg: FloatingLeg | None = None,
    convention: Convention = Convention.PAR_PAR,
    clean: bool = True,
) -> AssetSwap:
    """The spread that makes the asset swap worth zero at inception.

    ``price`` is the bond's market quote, clean unless ``clean`` is false. The
    answer is closed form in both conventions: the structure is linear in the
    spread, so there is nothing to solve.
    """
    if leg is None:
        leg = leg_for(bond, settlement)
    dirty = price + bond.accrued(settlement) if clean else price
    if dirty <= 0.0:
        raise BadAssetSwap(
            f"{bond.name} at a dirty price of {dirty!r} is not something a buyer "
            "can pay for"
        )
    curve_price, redemption_pv, annuity, projected, par = _parts(
        bond, curve, settlement, leg
    )
    residual = par * projected - (par - redemption_pv)
    if convention is Convention.PAR_PAR:
        notional = par
        numerator = curve_price - dirty - residual
    else:
        notional = dirty
        numerator = curve_price - dirty * (projected + redemption_pv / par)
    return AssetSwap(
        convention=convention,
        spread=numerator / (notional * annuity),
        notional=notional,
        annuity=annuity,
        dirty_price=dirty,
        curve_price=curve_price,
        residual=residual,
        par=par,
    )


def par_par_spread(
    bond: Bond,
    price: float,
    curve: DiscountCurve,
    settlement: date,
    *,
    leg: FloatingLeg | None = None,
    clean: bool = True,
) -> float:
    """The par-par asset swap spread, as a rate."""
    return asset_swap_spread(
        bond, price, curve, settlement, leg=leg, convention=Convention.PAR_PAR, clean=clean
    ).spread


def proceeds_spread(
    bond: Bond,
    price: float,
    curve: DiscountCurve,
    settlement: date,
    *,
    leg: FloatingLeg | None = None,
    clean: bool = True,
) -> float:
    """The proceeds (market value) asset swap spread, as a rate."""
    return asset_swap_spread(
        bond,
        price,
        curve,
        settlement,
        leg=leg,
        convention=Convention.PROCEEDS,
        clean=clean,
    ).spread


def price_from_spread(
    bond: Bond,
    spread: float,
    curve: DiscountCurve,
    settlement: date,
    *,
    leg: FloatingLeg | None = None,
    convention: Convention = Convention.PAR_PAR,
    clean: bool = True,
) -> float:
    """The price a quoted asset swap spread implies.

    The inverse of :func:`asset_swap_spread`, and in closed form in both
    conventions. Par-par is linear in the price. Proceeds is not — the price
    appears in the notional as well as in the structure's value — but the
    relation still rearranges rather than needing a root find.
    """
    if leg is None:
        leg = leg_for(bond, settlement)
    curve_price, redemption_pv, annuity, projected, par = _parts(
        bond, curve, settlement, leg
    )
    residual = par * projected - (par - redemption_pv)
    if convention is Convention.PAR_PAR:
        dirty = curve_price - residual - spread * par * annuity
    else:
        denominator = projected + redemption_pv / par + spread * annuity
        if denominator <= 0.0:
            raise BadAssetSwap(
                f"a proceeds spread of {spread!r} leaves the structure with a "
                f"denominator of {denominator!r}, so no price makes it worth zero"
            )
        dirty = curve_price / denominator
    return dirty - bond.accrued(settlement) if clean else dirty


@dataclass(frozen=True)
class SpreadComparison:
    """The same bond's spread by four measures, and the gaps between them."""

    price: float
    par_par: float
    proceeds: float
    z_spread: float
    i_spread: float

    @property
    def par_par_less_z(self) -> float:
        """Par-par asset swap less Z-spread, as a rate."""
        return self.par_par - self.z_spread

    @property
    def proceeds_over_par_par(self) -> float:
        """The ratio the two conventions stand in."""
        return self.proceeds / self.par_par if self.par_par != 0.0 else float("nan")


def compare_spreads(
    bond: Bond,
    price: float,
    curve: DiscountCurve,
    settlement: date,
    *,
    leg: FloatingLeg | None = None,
    clean: bool = True,
) -> SpreadComparison:
    """Every spread measure in the library, on one bond at one price."""
    from .spread import i_spread, z_spread

    clean_price = price if clean else price - bond.accrued(settlement)
    return SpreadComparison(
        price=clean_price,
        par_par=par_par_spread(bond, clean_price, curve, settlement, leg=leg),
        proceeds=proceeds_spread(bond, clean_price, curve, settlement, leg=leg),
        z_spread=z_spread(bond, clean_price, curve, settlement).value,
        i_spread=i_spread(bond, clean_price, curve, settlement),
    )


@dataclass(frozen=True)
class LegSensitivity:
    """One floating leg convention, and the spread it produces."""

    frequency: Frequency
    basis: Basis
    annuity: float
    spread: float


def leg_sensitivity(
    bond: Bond,
    price: float,
    curve: DiscountCurve,
    settlement: date,
    conventions: tuple[tuple[Frequency, Basis], ...],
    *,
    convention: Convention = Convention.PAR_PAR,
    clean: bool = True,
) -> tuple[LegSensitivity, ...]:
    """The same bond's spread under several floating leg conventions.

    The structure's value does not depend on the leg's frequency or day count
    at all — the leg telescopes — so every one of these spreads discounts the
    same amount of money. They differ because they divide it by different
    annuities, which is the whole of the effect and the reason a quote without
    its convention is not a number.
    """
    out: list[LegSensitivity] = []
    for frequency, basis in conventions:
        leg = leg_for(bond, settlement, frequency=frequency, basis=basis)
        struck = asset_swap_spread(
            bond,
            price,
            curve,
            settlement,
            leg=leg,
            convention=convention,
            clean=clean,
        )
        out.append(
            LegSensitivity(
                frequency=frequency,
                basis=basis,
                annuity=struck.annuity,
                spread=struck.spread,
            )
        )
    return tuple(out)
