"""A worked example that runs the whole library in one pass.

This is not a demonstration that the code runs — the test suite already says
that. It exists to reproduce numbers that live outside this repository, and to
exercise the identities *between* phases, which no single module's tests can
see: quotes to curve, curve to bond, bond to risk, risk to spread, spread to
option cost.

It exits non-zero if any figure moves, so a change to the numerics cannot
quietly shift something that has been in print for years.

Run it with ``python examples/worked.py``.
"""

from __future__ import annotations

import math
import sys
from datetime import date

from tenor import (
    WEEKENDS_ONLY,
    Accrual,
    Basis,
    Bond,
    BondFuture,
    CreditDefaultSwap,
    Deposit,
    Exercise,
    Frequency,
    Interpolation,
    Lattice,
    SurvivalCurve,
    Swap,
    asset_swap_spread,
    bootstrap,
    buckets_from,
    cheapest_to_deliver,
    compare_spreads,
    conversion_factor,
    day_count,
    delivery_switch,
    ho_lee_convexity,
    instrument_risk,
    key_rates,
    lattice_price,
    leg_sensitivity,
    level,
    option_adjusted_spread,
    option_cost,
    par_par_spread,
    proceeds_spread,
    shape_duration,
    year_fraction,
)
from tenor.spread import i_spread, z_spread

REFERENCE = date(2021, 1, 5)
FAILURES: list[str] = []


def check(what: str, got: float, want: float, tolerance: float) -> None:
    """Record a figure and whether it still reproduces."""
    ok = abs(got - want) <= tolerance
    mark = " " if ok else "  <-- MOVED"
    print(f"    {what:<52} {got:>18.10g}{mark}")
    if not ok:
        FAILURES.append(f"{what}: got {got!r}, expected {want!r} +/- {tolerance!r}")


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# -- 1. Conventions, against the published rules ------------------------------

rule("1. Three conventions are all called 30/360, and they disagree")

print(
    """
  ISDA 2006 section 4.16 defines more than one thirty-day-month rule, and the
  market calls several of them "30/360". Over 2021-02-28 to 2021-08-31 they
  give different day counts, which is a difference of three days of accrual on
  a period nobody would think twice about.
"""
)
start, end = date(2021, 2, 28), date(2021, 8, 31)
check(
    "30/360 Bond Basis, 4.16(f), days", day_count(start, end, Basis.THIRTY_360_BOND), 183, 0
)
check("30E/360, 4.16(g), days", day_count(start, end, Basis.THIRTY_E_360), 182, 0)
check("30/360 US, days", day_count(start, end, Basis.THIRTY_360_US), 180, 0)
check("ACT/365F, days", day_count(start, end, Basis.ACT_365F), 184, 0)

# -- 2. Hull's futures convexity adjustment -----------------------------------

rule("2. A published futures convexity adjustment")

print(
    """
  A futures contract is margined daily and a forward settles once, so the
  futures rate sits above the forward rate it stands in for. Hull's worked
  example takes a volatility of 1.2% and a contract eight years out, and gets
  47.5 basis points. The same arithmetic at ten years on 1% volatility gives
  51 basis points, which is the figure people expect to be small.
"""
)
check(
    "Hull: 1.2% vol at 8 years, in bp", 1e4 * ho_lee_convexity(0.012, 8.0, 8.25), 47.52, 1e-6
)
check("1% vol at 1 year, in bp", 1e4 * ho_lee_convexity(0.01, 1.0, 1.25), 0.625, 1e-9)
check("1% vol at 10 years, in bp", 1e4 * ho_lee_convexity(0.01, 10.0, 10.25), 51.25, 1e-9)

# -- 3. The curve, and the identity it is built on ----------------------------

rule("3. A curve from quotes, repricing every one of them")

quotes = [
    Deposit(REFERENCE, date(2021, 7, 5), 0.0020, Basis.ACT_360, "6m deposit"),
    Swap(REFERENCE, date(2023, 1, 5), 0.0060, label="2y swap"),
    Swap(REFERENCE, date(2026, 1, 5), 0.0150, label="5y swap"),
    Swap(REFERENCE, date(2031, 1, 5), 0.0220, label="10y swap"),
    Swap(REFERENCE, date(2041, 1, 7), 0.0270, label="20y swap"),
]
print(
    """
  Built three ways over the same quotes. Every instrument has to reprice to par
  off the finished curve under all three, which for the smooth interpolation
  takes several sweeps: adding a pillar moves the curve before it as well, so
  one sequential pass leaves the earlier instruments mispriced.
"""
)
for scheme in Interpolation:
    built = bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F, interpolation=scheme)
    check(
        f"{scheme.name}: worst repricing error, {built.sweeps} sweep(s)",
        built.worst_error,
        0.0,
        1e-13,
    )

built = bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F)
curve = built.curve
tenors = ((date(2023, 1, 5), "2y"), (date(2031, 1, 5), "10y"), (date(2041, 1, 7), "20y"))
for day, name in tenors:
    check(f"zero rate at {name}, continuous, %", 100.0 * curve.zero_rate(day), 0.0, math.inf)

# -- 4. A bond, against its closed forms --------------------------------------

rule("4. A bond, checked against closed forms written out here")

print(
    """
  The annuity price and the Macaulay duration of a bond settling on a coupon
  date both have closed forms. They are written out below rather than called,
  so this checks the library rather than checking it against itself.
"""
)
bond = Bond(REFERENCE, date(2031, 1, 5), 0.05)
periods, coupon, periodic = 20, 2.5, 0.06 / 2.0
discounted = (1.0 + periodic) ** -periods
annuity = coupon * (1.0 - discounted) / periodic + 100.0 * discounted
check(
    "price at 6%, against the annuity closed form",
    bond.dirty_price(0.06, REFERENCE),
    annuity,
    1e-11,
)

growth = 1.0 + periodic
macaulay = (
    growth / periodic
    - (growth + periods * (0.025 - periodic)) / (0.025 * (growth**periods - 1.0) + periodic)
) / 2.0
check(
    "Macaulay at 6%, against its closed form",
    bond.macaulay_duration(0.06, REFERENCE),
    macaulay,
    1e-11,
)

zero = Bond(REFERENCE, date(2031, 1, 5), 0.0)
check(
    "a zero coupon bond's Macaulay is its maturity",
    zero.macaulay_duration(0.037, REFERENCE),
    10.0,
    1e-12,
)
check(
    "and does not depend on the yield", zero.macaulay_duration(0.11, REFERENCE), 10.0, 1e-12
)

# -- 5. Risk, and the identity it has to satisfy ------------------------------

rule("5. Risk, broken down two ways")


def value(shifted: object) -> float:
    return bond.price_from_curve(shifted, REFERENCE)  # type: ignore[arg-type]


total = shape_duration(value, curve, level())
parts = key_rates(value, curve, buckets_from(curve, [0.5, 2.0, 5.0, 10.0]))
print(
    """
  Key rate durations shift the zero curve around one bucket; instrument risk
  shifts a quote and rebuilds the curve. The first must sum back to the total
  duration, because the shift shapes add to one at every pillar. The second
  need not, and does not: it is a different decomposition of the same exposure.
"""
)
for one in parts:
    check(f"key rate duration at {one.bucket.name}", one.duration, 0.0, math.inf)
check(
    "their sum, against the total duration",
    sum(one.duration for one in parts),
    total,
    1e-5,
)
risks = instrument_risk(value, built)
for one in risks:
    check(f"risk to the {one.name}", one.value, 0.0, math.inf)

# -- 6. Spreads ---------------------------------------------------------------

rule("6. A bond with no spread, measured two ways")

print(
    """
  This bond is priced exactly on the curve, so it has no spread at all. The
  Z-spread says so. The I-spread does not, because its benchmark is a single
  par rate at the maturity while the bond's coupons are discounted across the
  whole curve.
"""
)
fair = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
check(
    "Z-spread of a bond priced on the curve, bp",
    1e4 * z_spread(bond, fair, curve, REFERENCE).value,
    0.0,
    1e-8,
)
check(
    "I-spread of the same bond, bp",
    1e4 * i_spread(bond, fair, curve, REFERENCE),
    -5.17,
    0.05,
)

# The artefact grows with distance from par, so the same maturity at a coupon
# nearer the curve shows much less of it. Neither number is spread.
nearer_par = Bond(REFERENCE, date(2031, 1, 5), 0.03)
nearer_fair = nearer_par.price_from_curve(curve, REFERENCE) - nearer_par.accrued(REFERENCE)
check(
    "I-spread of a 3% coupon of the same maturity, bp",
    1e4 * i_spread(nearer_par, nearer_fair, curve, REFERENCE),
    -1.62,
    0.05,
)
check(
    "Z-spread of that one, bp",
    1e4 * z_spread(nearer_par, nearer_fair, curve, REFERENCE).value,
    0.0,
    1e-8,
)

# -- 6b. The spread the bond is actually swapped into -------------------------

rule("6b. The asset swap spread, in both of its conventions")

print(
    """
  The same bond, swapped into the floating index. The par-par buyer pays 100
  and receives the index plus a spread on 100; the proceeds buyer pays the
  market price and receives it on that. At a dirty price of exactly par the
  two are the same number and nowhere else, and the ratio between them is par
  over the price to the last bit.
"""
)
at_par = asset_swap_spread(bond, 100.0, curve, REFERENCE)
check("par-par spread at a price of 100, bp", 1e4 * at_par.spread, 275.2721, 5e-4)
check(
    "proceeds spread at the same price, bp",
    1e4 * proceeds_spread(bond, 100.0, curve, REFERENCE),
    275.2721,
    5e-4,
)
check(
    "the price component there, bp",
    1e4 * at_par.price_component,
    0.0,
    1e-12,
)
cheap = compare_spreads(bond, 95.0, curve, REFERENCE)
check("par-par spread at 95, bp", 1e4 * cheap.par_par, 329.1637, 5e-4)
check("proceeds spread at 95, bp", 1e4 * cheap.proceeds, 346.4881, 5e-4)
check("the ratio between them", cheap.proceeds_over_par_par, 100.0 / 95.0, 1e-12)
check("Z-spread at 95, bp", 1e4 * cheap.z_spread, 346.3071, 5e-4)

# The bond priced on the curve earns nothing over it, which the I-spread above
# could not say. This is the same statement as the Z-spread's, reached through
# a structure rather than through a discounting convention.
check(
    "asset swap spread of the bond priced on the curve, bp",
    1e4 * par_par_spread(bond, fair, curve, REFERENCE),
    0.0,
    1e-11,
)

# One amount of money, two annuities. Neither quote is wrong.
LEGS = ((Frequency.QUARTERLY, Basis.ACT_360), (Frequency.ANNUAL, Basis.THIRTY_360_BOND))
legs = leg_sensitivity(bond, 95.0, curve, REFERENCE, LEGS)
check("quarterly ACT/360 leg, bp", 1e4 * legs[0].spread, 329.1637, 5e-4)
check("annual 30/360 leg, bp", 1e4 * legs[1].spread, 336.6962, 5e-4)
check(
    "the money the two legs amortise, identical",
    legs[0].spread * legs[0].annuity - legs[1].spread * legs[1].annuity,
    0.0,
    1e-15,
)

# -- 7. The lattice, and what the option costs --------------------------------

rule("7. A callable bond, and the spread the call takes away")

step = 0.5
tree = Lattice.calibrated(curve, REFERENCE, steps=20, step=step, volatility=0.15)
check("lattice repricing of the curve, worst error",
      max(abs(tree.zero_price(i) - curve.discount_at(curve.time_to(REFERENCE) + i * step)
              / curve.discount(REFERENCE)) for i in range(1, 21)), 0.0, 1e-14)

bullet = lattice_price(tree, coupon=2.5)
calls = Exercise(tuple((index, 100.0) for index in range(10, 20)), issuer=True)
callable_price = lattice_price(tree, coupon=2.5, exercise=calls)
print(
    """
  The same price, asked two questions: what spread reproduces it if the call is
  ignored, and what spread reproduces it if the call is modelled. The first is
  the Z-spread, the second the OAS, and the difference is what the option is
  worth. A bullet, with no option, must show a cost of exactly zero.
"""
)
check("bullet on the tree", bullet, 0.0, math.inf)
check("callable at par from year five", callable_price, 0.0, math.inf)

ignoring = option_adjusted_spread(tree, callable_price, coupon=2.5).value
modelling = option_adjusted_spread(tree, callable_price, coupon=2.5, exercise=calls).value
check("Z-spread of the callable, bp", 1e4 * ignoring, 89.0, 1.5)
check("OAS of the callable, bp", 1e4 * modelling, 0.0, 1e-6)
check("the option cost, bp", 1e4 * option_cost(ignoring, modelling), 89.0, 1.5)

bullet_ignoring = option_adjusted_spread(tree, bullet, coupon=2.5).value
bullet_modelling = option_adjusted_spread(tree, bullet, coupon=2.5, exercise=None).value
check(
    "option cost of a bullet, bp",
    1e4 * option_cost(bullet_ignoring, bullet_modelling),
    0.0,
    1e-10,
)

puts = Exercise(tuple((index, 100.0) for index in range(10, 20)), issuer=False)
putable_price = lattice_price(tree, coupon=2.5, exercise=puts)
put_ignoring = option_adjusted_spread(tree, putable_price, coupon=2.5).value
put_modelling = option_adjusted_spread(tree, putable_price, coupon=2.5, exercise=puts).value
put_cost = 1e4 * option_cost(put_ignoring, put_modelling)
check("option cost of a putable, bp (the holder is long it)", put_cost, -2.6, 0.5)
if put_cost >= 0.0:
    FAILURES.append("a putable bond's option cost must be negative")

# -- 9. Deliverable bond futures ----------------------------------------------

rule("8. A delivery basket, and where the cheapest bond switches")

DELIVERY_FIRST = date(2026, 12, 1)
DELIVERY = date(2026, 12, 31)
BASKET_SETTLEMENT = date(2026, 11, 20)
CONTRACT = BondFuture(
    first_delivery=DELIVERY_FIRST, delivery=DELIVERY, label="a long contract"
)


def deliverable_bond(maturity: date, coupon: float, label: str) -> Bond:
    return Bond(
        effective=date(maturity.year - 40, maturity.month, maturity.day),
        maturity=maturity,
        coupon=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.ACT_ACT_ISDA,
        accrual=Accrual.PERIOD_FRACTION,
        label=label,
    )


# A bond at exactly the notional coupon with whole years to run is worth par at
# the notional yield, so its factor is one. This is the check the closed form's
# shape is easiest to fail, and it is independent of everything else here.
check(
    "factor, 6% with 20 whole years to run",
    conversion_factor(
        deliverable_bond(date(2046, 12, 1), 0.06, "at the notional"),
        DELIVERY_FIRST,
        digits=None,
    ),
    1.0,
    1e-14,
)
# And with a quarter of odd life it is not one. The residue is the rounding
# convention: the stream is discounted half a period and half a coupon of
# accrued interest comes off, and the two do not cancel.
check(
    "factor, 6% with a three-month stub",
    conversion_factor(
        deliverable_bond(date(2047, 3, 1), 0.06, "with a stub"),
        DELIVERY_FIRST,
        digits=None,
    ),
    0.9998891565,
    5e-10,
)

BASKET = [
    deliverable_bond(date(2046, 8, 15), 0.0175, "1.750% of 2046"),
    deliverable_bond(date(2046, 11, 15), 0.0300, "3.000% of 2046"),
    deliverable_bond(date(2047, 2, 15), 0.0450, "4.500% of 2047"),
    deliverable_bond(date(2047, 5, 15), 0.0625, "6.250% of 2047"),
]
# The quotes as `examples/basket.txt` carries them, rounded to three decimals
# the way a screen shows them, rather than recomputed from a flat 4.55% yield.
# The rounding is worth 7.8e-5 of a point on the price the basket justifies, and
# pinning the number the command line prints is more useful than pinning one
# nobody can reproduce from the file in the repository.
BASKET_PRICES = [63.782, 79.795, 99.337, 122.497]
for bond, quoted in zip(BASKET, BASKET_PRICES, strict=True):
    if abs(bond.clean_price(0.0455, BASKET_SETTLEMENT) - quoted) > 5e-4:
        FAILURES.append(f"{bond.name} is not the quote in examples/basket.txt")
QUOTE = 118.75
REPO = 0.042

ranked = cheapest_to_deliver(
    BASKET,
    CONTRACT,
    clean_prices=BASKET_PRICES,
    futures_price=QUOTE,
    settlement=BASKET_SETTLEMENT,
    repo=REPO,
)
check("price the basket justifies", ranked.implied_futures_price, 118.9266122, 1e-6)
if not ranked.unanimous:
    FAILURES.append("the three criteria disagree on this basket, and should not")
if ranked.by_net_basis.bond.name != "6.250% of 2047":
    FAILURES.append("the cheapest bond moved")

# Net basis and the implied repo rate are one number written twice. Both sides
# are computed by different routes, so this is a statement about the definitions.
worst_identity = max(
    abs(one.net_basis - (REPO - one.implied_repo) * one.financed_balance)
    for one in ranked.basis
)
check("net basis against the implied repo identity", worst_identity, 0.0, 1e-12)
worst_price_identity = max(
    abs(one.net_basis - (one.breakeven_futures_price - QUOTE) * one.conversion_factor)
    for one in ranked.basis
)
check("net basis against the break-even identity", worst_price_identity, 0.0, 1e-12)

# The factors span a factor of two; the factor over the financed balance barely
# moves, which is why the rate and the price criteria agree and the net basis is
# the one on its own scale.
factors = [one.conversion_factor for one in ranked.basis]
ratios = [one.conversion_factor / one.financed_balance for one in ranked.basis]
check("spread of conversion factors, ratio", max(factors) / min(factors), 1.997, 0.002)
check("spread of factor over balance, ratio", max(ratios) / min(ratios), 1.0465, 5e-4)

# The switch is at the notional coupon. Whole years to maturity make the
# rounding convention a no-op, which is the condition for it to be exact.
PAIR = [
    deliverable_bond(date(2032, 12, 1), 0.06, "short"),
    deliverable_bond(date(2056, 12, 1), 0.06, "long"),
]
low, high = 0.01, 0.15
BELOW = delivery_switch(
    PAIR, CONTRACT, settlement=BASKET_SETTLEMENT, repo=REPO, yields=[low]
)[0].cheapest
for _ in range(200):
    middle = 0.5 * (low + high)
    at = delivery_switch(
        PAIR, CONTRACT, settlement=BASKET_SETTLEMENT, repo=REPO, yields=[middle]
    )[0].cheapest
    if at == BELOW:
        low = middle
    else:
        high = middle
check("yield at which the cheapest bond switches, %", 50.0 * (low + high), 6.0, 1e-6)
if BELOW != "short":
    FAILURES.append("below the notional coupon the shortest bond should be cheapest")

# -- 9. A contract marked from inside its own schedule ------------------------

rule("9. A credit default swap already running")

print(
    """
  Everything above prices an instrument from the day it starts. A credit
  default swap is almost never looked at on that day: the standard contract
  rolls on the IMM dates and is marked every day in between, so the valuation
  date sits inside a premium period that has partly accrued and not yet paid.

  Three things are checked here. The premium leg against a midpoint rule
  written out below, which shares none of the closed form's algebra. The
  protection leg against the same contract struck today, which has to be the
  same number and not merely a close one, because the dates the two differ on
  are ones the integral never reaches. And the settlement amount against its
  own split into principal and accrued.
"""
)

CREDIT = SurvivalCurve.flat(
    REFERENCE, date(2031, 1, 5), 0.02, basis=Basis.ACT_365F, recovery=0.4
)
ROLL = date(2020, 12, 20)
RUNNING = CreditDefaultSwap.standard(
    ROLL,
    date(2026, 3, 20),
    0.01,
    calendar=WEEKENDS_ONLY,
    frequency=Frequency.QUARTERLY,
    notional=10_000_000.0,
)
STRUCK_TODAY = CreditDefaultSwap.standard(
    REFERENCE,
    date(2026, 3, 20),
    0.01,
    calendar=WEEKENDS_ONLY,
    frequency=Frequency.QUARTERLY,
    notional=10_000_000.0,
)


def signed(day: date, name: SurvivalCurve) -> float:
    """Year fraction from the valuation date to ``day``, negative in the past."""
    if day >= REFERENCE:
        return name.time_to(day)
    return -year_fraction(day, REFERENCE, name.basis)


def default_pv_by_quadrature(name: SurvivalCurve, end: float, steps: int) -> float:
    """``integral DF(s) dQ(s)`` from the valuation date, by the same rule."""
    width = end / steps
    total = 0.0
    for step in range(steps):
        moment = (step + 0.5) * width
        total += (
            curve.discount_at(moment)
            * name.hazard_at(moment)
            * name.survival_at(moment)
            * width
        )
    return total


def premium_leg_by_quadrature(contract: CreditDefaultSwap, steps: int) -> float:
    """The premium leg read off the contract rather than off the integral.

    Each remaining period pays its whole coupon at its payment date if the
    entity survives to it. Over the part of the period still to come, a default
    owes the premium accrued since the period *began*, which for the period in
    progress means reaching back before the valuation date.
    """
    total = 0.0
    for period in contract.remaining(REFERENCE):
        pay = CREDIT.time_to(period.payment)
        accrual = year_fraction(period.adjusted_start, period.adjusted_end, contract.basis)
        total += accrual * curve.discount_at(pay) * CREDIT.survival_at(pay)
        start = signed(period.adjusted_start, CREDIT)
        lower = max(start, 0.0)
        end = CREDIT.time_to(period.adjusted_end)
        if end <= lower:
            continue
        width = (end - lower) / steps
        weighted = 0.0
        for step in range(steps):
            moment = lower + (step + 0.5) * width
            weighted += (
                (moment - start)
                * curve.discount_at(moment)
                * CREDIT.hazard_at(moment)
                * CREDIT.survival_at(moment)
                * width
            )
        total += weighted * accrual / (end - start)
    return total


closed = RUNNING.risky_annuity(CREDIT, curve)
check(
    "premium leg of a running contract, relative to a midpoint rule",
    abs(premium_leg_by_quadrature(RUNNING, 2000) / closed - 1.0),
    0.0,
    1e-10,
)
# The other half: at a grid this coarse the rule is outside the bound above, so
# the bound is reading the integral rather than passing on rounding alone.
coarse = abs(premium_leg_by_quadrature(RUNNING, 100) / closed - 1.0)
if coarse <= 1e-10:
    FAILURES.append("the coarse quadrature is inside the fine bound, so it proves nothing")
check("the same rule at a twentieth of the steps", coarse, 3.6e-10, 1e-10)

# An identity, so the answer is zero rather than small: the two contracts
# differ only in an effective date that lies behind the valuation date, and the
# protection integral starts at the valuation date under both.
check(
    "protection before the mark, running less struck-today",
    RUNNING.protection_leg(CREDIT, curve) - STRUCK_TODAY.protection_leg(CREDIT, curve),
    0.0,
    0.0,
)

SETTLEMENT = RUNNING.upfront(CREDIT, curve)
check(
    "settlement less its own principal and accrued",
    SETTLEMENT.principal + SETTLEMENT.accrued - SETTLEMENT.cash,
    0.0,
    0.0,
)
check(
    "accrued premium quoted, on 10mm at 100bp",
    SETTLEMENT.accrued,
    4166.666667,
    1e-5,
)
# The quoted accrued is a day count owed come what happens; the same slice of
# coupon is paid at the payment date and only on survival to it. The gap is the
# reason the principal is not a present value.
check(
    "the accrued is worth less than it is quoted at, %",
    100.0 * (1.0 - RUNNING.accrued_value(CREDIT, curve) / SETTLEMENT.accrued),
    0.2157,
    5e-4,
)

print(
    """
  The last figure worth stating is the convention that is easy to get wrong.
  The accrual owed on a default runs from the period's start, not from the
  valuation date, so the integration clamps to the remaining stub while the
  weight's origin stays back where the period began. Clamping both understates
  the leg and so overstates the par spread. How much depends entirely on the
  chance of defaulting inside the rest of the period, which is why it can be
  ignored on a good name and cannot be on a bad one.
"""
)
for hazard, label, expected in (
    (0.02, "2% hazard", 0.0041536),
    (1.0, "100% hazard", 43.03172),
):
    name = SurvivalCurve.flat(
        REFERENCE, date(2031, 1, 5), hazard, basis=Basis.ACT_365F, recovery=0.4
    )
    first = RUNNING.schedule[0]
    begin = signed(first.adjusted_start, name)
    finish = name.time_to(first.adjusted_end)
    accrual = year_fraction(first.adjusted_start, first.adjusted_end, RUNNING.basis)
    # The gap between the two origins is a product, not a residual: the elapsed
    # accrual times the discounted chance of defaulting inside the stub.
    gap = -begin * default_pv_by_quadrature(name, finish, 4000) * accrual / (finish - begin)
    protection = RUNNING.protection_leg(name, curve)
    annuity = RUNNING.risky_annuity(name, curve)
    moved = 1e4 * (protection / (annuity - gap) - protection / annuity)
    check(f"accrual origin, {label}, bp of par spread", moved, expected, 1e-4)


# -- done ---------------------------------------------------------------------

print()
if FAILURES:
    print(f"{len(FAILURES)} figure(s) did not reproduce:", file=sys.stderr)
    for failure in FAILURES:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(1)
print("All figures reproduced.")
