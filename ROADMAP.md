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
- [ ] Monotone convex interpolation, which keeps forwards positive
- [x] Conversion between any two of discount, zero and forward without drift
- [x] Extrapolation refused rather than silently flattened

Log-linear on discount factors already guarantees positive forwards wherever the
discount factors decrease; the remaining item is the smooth Hagan-West scheme,
which keeps that guarantee without the piecewise-constant forwards.

## Phase 3 — Bootstrapping

- [ ] Curve built from deposits, futures and par swaps
- [ ] Every input instrument reprices to par, as an identity the tests assert
- [ ] Futures convexity adjustment, applied explicitly and reported
- [ ] A curve that cannot be built says which instrument broke it

## Phase 4 — Bond analytics

- [ ] Price from yield and yield from price, with the solver's convergence reported
- [ ] Accrued interest, clean and dirty price
- [ ] Macaulay, modified and effective duration; convexity
- [ ] DV01 and PV01, and the difference between them stated

## Phase 5 — Curve risk

- [ ] Key rate durations against a set of curve buckets
- [ ] Key rate durations sum to the total duration, as an identity the tests assert
- [ ] Parallel, slope and curvature shifts
- [ ] Bucketed DV01 against the bootstrapped instruments

## Phase 6 — Spreads and embedded options

- [ ] Z-spread and I-spread over a curve
- [ ] A short-rate lattice calibrated to reprice the curve exactly
- [ ] American call and put exercise on the lattice
- [ ] Option-adjusted spread, and the option cost it implies

## Phase 7 — Interface, documentation and continuous integration

- [ ] Command line entry point over a curve and an instrument file
- [ ] Worked example reproducing a published figure end to end
- [ ] README covering the conventions and the design decisions
- [ ] Continuous integration across supported Python versions with types and lint
