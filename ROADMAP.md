# Roadmap

Phases are ordered but not dated. An item is checked when the code is written,
exercised end to end, and covered by tests that run.

Fixed income is a field where the arithmetic is easy and the conventions decide
the answer. A bond priced with the wrong day count is wrong by a few basis
points, which is the size of the spread anybody is trying to measure — so every
convention here is an argument with a name, never a default, and every one is
checked against a published worked example where one exists.

## Phase 1 — Dates, day counts and schedules

- [x] Day count fractions: ACT/360, ACT/365F, ACT/ACT ISDA, and all three 30/360
- [x] Each checked against the published rules, case by case
- [x] Business day conventions: following, modified following, preceding
- [x] Holiday calendars, composable across currencies
- [x] Payment schedule generation from an effective date, maturity and frequency
- [x] End-of-month and stub handling, with the stub reported rather than implied

"30/360" turned out to name three distinct rules — ISDA Bond Basis, 30E/360 and
the US convention — which give 183, 182 and 180 days for the same six months.
They are separate conventions here rather than one with a flag.

## Phase 2 — Discount curves

- [x] Discount factors, zero rates and forward rates, with the compounding stated
- [x] Interpolation: linear on zero rates, log-linear on discount factors
- [x] Monotone convex interpolation, which keeps forwards positive
- [x] Conversion between any two of discount, zero and forward without drift
- [x] Extrapolation refused rather than silently flattened

Log-linear on discount factors already guaranteed positive forwards wherever the
discount factors decrease, at the cost of forwards that are flat inside each
interval and jump at every pillar. The Hagan-West scheme keeps the guarantee and
drops the cost: forwards that vary across each interval and join continuously
across the pillars. It buys that by imposing positivity as a clamp rather than
inheriting it, which is also its one refusal — pillars implying a negative
discrete forward put the clamp's range out of reach and are rejected as the
arbitrage they are.

## Phase 3 — Bootstrapping

- [x] Curve built from deposits, futures and par swaps
- [x] Every input instrument reprices to par, as an identity the tests assert
- [x] Futures convexity adjustment, applied explicitly and reported
- [x] A curve that cannot be built says which instrument broke it

Solving each instrument in turn for the discount factor at its own maturity is
the standard method and it is wrong for any interpolation whose shape at one
maturity depends on its neighbours. Adding a pillar moves the curve before it as
well, so the instruments already solved stop repricing — silently, by about ten
basis points in the middle of an interval. The sequential pass is therefore
followed by sweeps until every instrument reprices at once, and the number of
sweeps is reported: one for the local schemes, seven for monotone convex over
this strip.

## Phase 4 — Bond analytics

- [x] Price from yield and yield from price, with the solver's convergence reported
- [x] Accrued interest, clean and dirty price
- [x] Macaulay, modified and effective duration; convexity
- [x] DV01 and PV01, and the difference between them stated

The difference between DV01 and PV01 has two sources, not one. Shape — a yield
is a weighted average of the curve over the bond's own flows — is worth +1.01%
on a fifteen-year bullet here. Compounding, because a basis point of
continuously compounded rate is not a basis point of the semi-annual rate that
discounts identically, is worth −1.54%, and applies even on a flat curve where
shape cannot. They have opposite signs, so making both mistakes at once leaves
the two numbers 0.23% apart — closer than either error alone.

## Phase 5 — Curve risk

- [x] Key rate durations against a set of curve buckets
- [x] Key rate durations sum to the total duration, as an identity the tests assert
- [x] Parallel, slope and curvature shifts
- [x] Bucketed DV01 against the bootstrapped instruments

The two decompositions are not one set of numbers regrouped. Key rates shift the
zero curve in a shape around one bucket; instrument risk shifts a quote and
rebuilds the curve, which moves every zero rate out to that quote's maturity.
The second is the one a hedge is put on in, and it is the one that depends on
the interpolation: across the three schemes here the total agrees to 1.3% while
the ten-year entry moves by 64% and the five-year changes sign.

## Phase 6 — Spreads and embedded options

- [x] Z-spread and I-spread over a curve
- [x] A short-rate lattice calibrated to reprice the curve exactly
- [x] American call and put exercise on the lattice
- [x] Option-adjusted spread, and the option cost it implies

I-spread is kept for contrast rather than for use. A bond priced exactly on the
curve - with no spread at all - shows an I-spread of -0.14bp at two years and
-6.0bp at fifteen, purely because the benchmark is a single par rate at the
maturity. Its Z-spread is zero, as it should be.

The lattice steps in uniform half-years where a bond pays on actual dates, so a
spread measured on the tree and the same spread measured on the curve differ by
about a quarter of a basis point. That is recorded rather than tolerated
silently; closing it would mean a tree whose steps follow the coupon dates.

## Phase 7 — Interface, documentation and continuous integration

- [ ] Command line entry point over a curve and an instrument file
- [ ] Worked example reproducing a published figure end to end
- [ ] README covering the conventions and the design decisions
- [ ] Continuous integration across supported Python versions with types and lint
