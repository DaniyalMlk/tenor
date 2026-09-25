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

- [x] Command line entry point over a curve and an instrument file
- [x] Worked example reproducing a published figure end to end
- [x] README covering the conventions and the design decisions
- [x] Continuous integration across supported Python versions with types and lint

The command line requires `--basis` rather than defaulting it. Conventions are
accepted by enum name as well as by the market's spelling, because the market's
spelling has spaces in it.

The worked example exists to reproduce figures from outside this repository -
the three ISDA thirty-day-month rules, Hull's futures convexity adjustment - and
to exercise the identities between phases, which no single module's tests can
see. It exits non-zero if any of them moves.

## Phase 8 — Distribution

- [x] `py.typed` inside the package, so the annotations reach anyone who installs it
- [x] Distribution metadata an index can present: authors, keywords, classifiers,
      project URLs, and the licence as an SPDX expression carrying the LICENSE text
      into the artefact, which the legacy licence table did not
- [x] `tenor --version`, asserted against both the installed metadata and the version
      declared in `pyproject.toml`
- [x] A release driven by a version tag, publishing with the index's trusted
      publishing flow, so no upload credential exists in the repository — and
      refusing to publish when the tag and the declared version disagree
- [x] The sdist and the wheel each installed into a clean environment and made to
      bootstrap the bundled quote screen, on pull requests as well as on a tag
- [ ] A first release on the index, which waits on the publisher being registered
      there for this project

## Phase 8 — Holding-period return

The library could price a bond and measure its risk, and could not answer the
question a position is held for: if nothing happens, what does this earn?

- [x] A forward curve — today's curve seen from a future date
- [x] A rolled curve — the same zero rate at each tenor, reference date moved,
      which is a different object and the one carry-and-roll-down needs
- [x] Coupons inside the window, reinvested at the curve's own forward rates
- [x] The decomposition, with carry asserted equal to the financing cost rather
      than described as roughly equal to it
- [x] Both market meanings of "carry" reported, each named for what it is
- [x] A sequence of lengthening horizons, stopping at maturity rather than
      raising
- [x] A `horizon` command over a quote screen and a bond

"Nothing happens" means two different things and the gap between them is the
entire expected excess return. Under the arbitrage-free reading the curve
evolves to its own forwards and the bond earns its funding cost exactly — an
identity, measured here at 2e-15 per 100 across coupons from 0% to 9%, three
curve shapes and every interpolation. Under the trader's reading the zero rate
at each tenor is unchanged, the bond ages into a different point on it, and the
difference is roll-down.

On the bundled screen a 3% 2031 held for a year returns 279bp: 50bp financing,
249bp roll-down.

The two curves must not be confused. Building the forward curve and using it as
the rolled curve makes roll-down come out as exactly zero, which is plausible
enough to survive review and answers a different question. On a flat curve they
genuinely coincide, which is why roll-down is zero there and nowhere else.

Reporting only income-minus-financing would have been the conventional choice
and it ranks positions backwards: a 9% bond shows +4.83 against a zero-coupon
bond's -2.00, and the zero-coupon bond earns 27bp more over the year.
