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
    Basis,
    Bond,
    Deposit,
    Exercise,
    Interpolation,
    Lattice,
    Swap,
    bootstrap,
    buckets_from,
    day_count,
    ho_lee_convexity,
    instrument_risk,
    key_rates,
    lattice_price,
    level,
    option_adjusted_spread,
    option_cost,
    shape_duration,
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

# -- done ---------------------------------------------------------------------

print()
if FAILURES:
    print(f"{len(FAILURES)} figure(s) did not reproduce:", file=sys.stderr)
    for failure in FAILURES:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(1)
print("All figures reproduced.")
