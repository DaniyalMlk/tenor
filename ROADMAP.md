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

## Phase 9 — Floating rate notes

The library priced fixed-coupon bullets, solved yields and three kinds of spread,
and measured key rate and shape risk. It could not price the one instrument whose
whole behaviour is a statement about the curve, and the one place where rate risk
and spread risk are not approximately the same number.

- [x] Projected coupons, each period's index rate taken from the discount factors
      against the coupon's own accrual fraction
- [x] The current period's coupon taken from a supplied fixing rather than
      projected, with the approximation named and measured where it is projected
- [x] A price off the curve, clean and dirty, and accrued interest that refuses
      rather than guess a fixing
- [x] A discount margin solved against the observed price, applied period by
      period to the projected forwards rather than added to a zero rate
- [x] Spread duration, rate duration and the margin value of a basis point,
      reported separately
- [x] A command line report printing both durations, with the coupon projection
      behind a flag
- [x] The mypy version pin removed, which was making the checker read installed
      stubs as 3.11 and fail inside numpy's on a 3.12 interpreter

The identity everything here rests on: with the discount margin equal to the
quoted margin, each coupon is exactly its own discount denominator minus one, the
sum telescopes, and the price is exactly par — for any curve of any shape at any
level on any day count basis. Asserted to 5e-13 across four curve shapes, four
bases and three frequencies.

Measuring it showed the consequence is stronger than the usual statement of it.
Collapsing the telescoping sum leaves the whole price equal to
`(1 + (fixing + m) tau_1) / (1 + (fixing + m) tau_remaining)`, which contains no
curve. So rate duration is exactly zero on *any* settlement date once the current
fixing is known, not only on a reset date. A floater does not have a little rate
risk because its coupon resets soon; at par it has none.

What it has away from par is the rate sensitivity of the leftover annuity when the
discount margin differs from the quoted one — linear in that difference and
changing sign with the side of par. Against a quoted 75bp on the five-year note in
the tests: +0.061 years at a 25bp margin, 0.000 at 75bp, −0.061 at 125bp, −0.250
at 275bp. A floater trading cheap to its quoted margin gains when rates rise.
Projecting the current fixing instead of knowing it adds −0.057 years at par three
weeks into a quarterly period, and costs 2.9bp of price when the real fixing is
50bp away from the projection.

Spread duration is 1.014 times the modified duration of a fixed bond of the same
maturity at three years, 1.005 at five and 0.974 at ten.

**The par identity does not validate the projection**, and that is worth recording
because it is the obvious thing to rely on. The telescoping needs the coupon and
the discount denominator to use the same rate over the same fraction; it does not
need that rate to be right. A note projected with its index scaled by an arbitrary
factor prices at exactly par, so there is a test asserting that rather than a
false sense that par covers it. The projection convention is measured separately:
an ACT/360 note on an ACT/365F curve reads 20.72bp against 21.01bp on its first
period, worth half a basis point of price at a wide margin and 0.05bp of implied
margin.

Two defects found by the tests and fixed. A settlement date inside a period that
began before the curve's reference date raised `OffCurve` from inside the curve;
it is not an approximation but an impossibility — there is no discount factor at
the reset to take a forward from — and it now refuses with the fixing named. And a
settlement date past maturity left no flows, so the sum over an empty sequence
priced a matured note at 0.0 and every risk number built on it would have divided
by it.

## Phase 10 — Index-linked bonds

The library priced fixed-coupon bullets, floating rate notes and callables, and
had nothing for the third major cash government instrument. Almost nothing about
an index-linked bond is arithmetic: it is a stack of conventions, and swapping any
one of them for the plausible alternative gives a number in the right
neighbourhood.

- [x] A published monthly index with the three-month lag and day-of-month
      interpolation, refusing a series with a gap in it rather than interpolating
      an absent print into a plausible number
- [x] The monthly-step convention alongside the daily one, and an arbitrary lag,
      since both exist in the market
- [x] Real and index space kept apart in every method name: the quoted price and
      yield real, the invoice the real dirty price times the ratio at settlement
- [x] Projection past the published history as a required argument, never a
      default, with the flows that needed it flagged in the output
- [x] The deflation floor on the principal only, reported rather than applied
      silently, with the final coupon separated from the redemption it is wired
      with
- [x] Breakeven inflation in both the quoted and the exact Fisher form, plus the
      inflation implied by a nominal curve and an invoice
- [x] Accretion between two dates split into what is published and what is
      assumed
- [x] A `linker` command taking an index series file, with the parser naming the
      line as the quote parser does

The number the lag exists to produce is how much of the near-term accretion is
already arithmetic, and it is smaller than the lag suggests. The reference index is
determined to the first of the month three months after the last print, so from
mid-September with a series through August it reaches 1 November — six weeks, not
three months. On a 2% path: 51.4% of the next quarter is published, 25.7% of six
months and 12.7% of a year, falling through each month until the next print.

The interpolation is worth having named. By the 28th of a month the daily
convention reads 107.159 against 107.000 stepped, 15bp of index level, which is
15bp of invoice on a bond at par.

Breakeven measured in both forms: 250.0bp quoted against 245.1bp exact at a 4.5%
nominal on a 2.0% real, and 19.2bp of difference at 9% on 4%. The gap is the cross
term and it is wider than the bid-offer on the spread it is quoted as.

Real duration is not comparable with a nominal bond's: 6.19 years on the 1.5% real
of 2032 against 5.59 for a 4% nominal of the same maturity, which is the coupon
difference and not an inflation effect.

One defect found by the tests. The fixed-coupon bond bundles the final coupon into
the redemption payment, correctly — they are one wire — and flooring that bundle's
index ratio floors the final coupon too, which is a more valuable bond. It can
only appear on a bond in cumulative deflation, so no other test here would have
reached it. The two are now separate flows on the same day. And the guard against
projecting backwards sat after the year fraction, which refuses a backwards
interval itself with a message about day counts; it now runs first and names the
anchor.

## Phase 11 — Default risk

- [x] A survival curve as a piecewise-constant forward hazard rate, carrying the
      recovery assumption it was stripped under
- [x] Protection and premium legs in closed form on the union of the two curves'
      pillars, with accrual on default computed rather than approximated by a
      half-period
- [x] Par spreads, upfront values and the risky annuity
- [x] A sequential bootstrap from par spreads that reprices every quote it was
      given
- [x] Risky bond pricing, with recovery on face rather than on the remaining
      cashflows
- [x] The credit triangle implemented, and the size and shape of its error
      measured rather than repeated
- [x] A command-line entry point reading quoted spreads in basis points

## Phase 12 — Deliverable bond futures

- [x] Conversion factors on the exchange convention, with the remaining life
      rounded down to a whole quarter, checked against pricing the rounded bond
      through the package's own schedule generator rather than against the same
      algebra written twice
- [x] Gross basis, carry and net basis for a cash-and-carry, with the coupons
      that fall inside the holding period received and reinvested rather than
      assumed away
- [x] The implied repo rate in closed form, because the break-even is linear in
      the financing rate, checked against a bisection on the cash flows
- [x] Cheapest to deliver by all three of the conventional criteria, with
      whether they agree reported rather than assumed
- [x] The delivery switch walked across a flat yield, described and deliberately
      not valued
- [x] The contract's own sensitivity: the deliverable's, divided by the factor
- [x] A command-line entry point taking a basket file and a quote

The measurement that changed the design. The expectation going in was that the
implied repo rate and the net basis would rank a basket differently, because a
deep-discount long bond finances around half the balance of a high-coupon short
one. That is backwards. The three criteria rank on the same gap between a bond's
break-even futures price and the quote, scaled by one, by the conversion factor,
and by the factor over the financed balance — and the factor over the balance
barely moves, because the balance is nearly the dirty price and the dirty price
is nearly the factor times the futures price, so the factor cancels. On the
four-bond basket the factors span a ratio of 2.00 and that scaling spans 4.65%.

It is the net basis that is in a different unit, points per 100 of the bond's own
face rather than per contract, and it takes a quote 5.00 points — 4.20% — from
the basket-implied price before it names a different bond. The module docstring
said the opposite before it was measured, which is the second time a plausible
claim in this package has turned out to be reversed.

Two defects the tests found, both in the tests rather than in the code. A bond
paying a coupon inside the holding period was asserted to carry better; it
carries worse, because income over a fixed 41 days is the same 41 days of coupon
whenever it lands, so what separates two such bonds is the accrued interest they
finance — 2.158 against 0.912, costing 0.00596 against the 0.00467 the coupon
earns back. And the walk searching for the point where the rankings split was
bounded one hundredth of a point short of the split, so it found nothing and read
as agreement.

One guard removed rather than kept. `Bond` bundles the final coupon into the
redemption flow, so a bond redeeming inside the holding period would contribute
102.5 to the coupons received rather than 2.5. Subtracting the principal back off
would handle a case that cannot arise: such a bond has redeemed before delivery
and is refused. The refusal is the proof, and the test sweeps every maturity for
a month either side of the delivery month at a daily step looking for the
counterexample.

## Phase 13 — Discounting and forecasting, separated

- [x] A forecast index — tenor, day count, calendar, rolling rule and payment
      lag — and the simply compounded forward it implies, read off a projection
      curve in the index's own day count
- [x] A floating leg projected on one curve, discounted on another, with a
      spread, reporting every period's forward and discount factor rather than
      only the total
- [x] A par swap that answers how far off par it is against the *pair* of
      curves, so pricing and bootstrapping remain the same code
- [x] The telescoping identity as a test: forty projected coupons against the
      single-curve leg's one subtraction, agreeing to 7.2e-16
- [x] A projection bootstrap taking the discount curve as given, sequential then
      swept, with every quote repricing to 1e-16
- [x] A tenor basis swap, and the longer tenor's curve solved out of the quoted
      spreads with the shorter one taken as known
- [x] Risk split into a discount response and a forecast response, in money
      rather than in years, attributed across buckets that add back to the
      parallel shifts
- [x] A command-line entry point over two quote files, reporting the forward
      basis between the curves and the split

What the identity actually needs, which is narrower than it looked. The
cancellation that collapses a floating leg to `P(start) - P(maturity)` is
between one period's `P(end)` and the next period's `P(start)`, so it needs the
payment to land on the accrual end. The rolling convention was the suspect and
it is innocent: under modified following a period ending on a Saturday rolls to
the Monday and the payment rolls with it, so the dates still meet and the
agreement is the 7.2e-16 above. A payment lag is the culprit, because it moves
the payment without moving the accrual. At the two business days an
overnight-indexed leg settles on, the gap is 0.095 basis points of par rate, and
at five days 0.262. So the lag is an explicit field that defaults to zero, and
that zero is load-bearing.

What the discount curve is worth, which is the reverse of the usual telling.
Separating the curves is introduced as the thing that repriced swaps, and for a
*par* rate it does almost nothing: a 100 basis point shift to the projection
curve moves the ten-year par rate by 101.62 basis points and the same shift to
the discount curve moves it by **-0.12** — the other way, and smaller by a
factor of 843. A par rate is a discount-weighted average of the forwards the leg
projects, so the projection curve moves every term and the discount curve only
reweights them, and on an upward-sloping curve heavier discounting tilts the
weights towards the earlier and lower forwards. On a swap that is *not* at par
the position reverses: a ten-year swap struck 100 basis points off the market
moves 4.67% of its own mark under that discount shift, against the 0.031% the
par rate's level moves. Third-order for a new trade, first-order for a book of
old ones.

The basis pass-through factorises exactly, which is the check worth having. A
flat 20 basis point lift to every projected forward raises the ten-year par rate
by 20.29 basis points, a ratio of 1.0146 rather than one. Two reasons pull in
opposite directions: the legs' annuities stand in the ratio 1.0191, actual/360
against 30/360, and a flat shift to a continuously compounded curve lifts a
simply compounded quarterly forward by 0.9956 of itself. The product is
1.014643 against a measured 1.014643, agreeing to 5.8e-15 — a decomposition
rather than a story that comes out near the right size.

And a curve that reprices every quote is not the curve that produced them. Fed
par rates generated off a known projection curve, the bootstrap recovers it to
within 0.03 basis points of zero rate beyond a year and sits **5.30 basis
points** away at 0.49 years, in the gap between the three-month forward quote and
the one-year swap where no input reaches. The same hole shows in the basis curve:
tenor basis spreads of 5 to 9 basis points with nothing pinning the short end
imply a 16.87 basis point forward basis over the first period, three times the
quote it was spread backwards out of. One forward quote on the long index puts it
at 5.00.

Four defects, all in the tests. The forward-rate identity was written with the
two discount factors the wrong way round; two annuity guards were asserted
against a curve at 1e-300, where the product does not underflow and the guard
cannot fire, rather than at 5e-324, where it does; and a difference of two sums
of forty terms was held to one ulp rather than three. Two defects in the code,
both found by running the command line rather than the library: a payment lag
puts a leg's final flow past a curve quoted to its maturity, which arrived as an
unexplained off-curve error out of the pillar seeding, and the forward lines in a
basis file were being dropped when the basis quotes were adapted to the
one-unknown protocol — which no repricing check can see, because the curve
reprices whatever it was actually handed.

## Phase 14 — Swaptions, caps and floors

- [x] European swaptions on the annuity measure, in which the forward swap rate
      is a martingale and Black's formula applies with no approximation beyond
      the lognormality assumed of the rate
- [x] The Bachelier convention alongside it, because a negative forward swap
      rate has no lognormal volatility and two major currencies had one for a
      decade
- [x] Caps and floors as strips of caplets, each on its own forward with its own
      accrual and discount factor, the already-fixed first period excluded by
      default and priced at intrinsic when asked for
- [x] Both parity identities as tests: a payer less a receiver is the
      forward-starting swap, and a cap less a floor is the same statement for
      the strip
- [x] A premium read back as a volatility in either convention, bisected on a
      bracket that is widened rather than assumed
- [x] A command-line entry point pricing the option and the strip side by side

The measurement this phase exists for follows directly from the last one. A par
swap rate is almost independent of the curve it is discounted on — the previous
phase measured a factor of 843 between what the projection curve and the
discount curve are worth to it. A swaption is the *annuity* times a call on that
rate, and the annuity is nothing but discount factors. So the same 100 basis
point shift that moves the ten-year into ten-year forward swap rate by 0.5067
basis points, 0.125% of its level, moves the option's value by **-13.57%** —
108 times as much. The move decomposes exactly as it should: the annuity falls
13.845% and the rate's rise claws 0.125% back.

The two conventions agree at the money only to leading order, and that is worth
measuring rather than assuming, because the conversion looks exact. Both give
`numeraire * vol * sqrt(2T/pi)` to first order, so a normal volatility is about
a lognormal one times the forward — 39.78 basis points against the product's
39.80 at a 10% volatility over one year, agreeing to 4.2e-04; 4.2e-03 over ten
years; and 1.6e-02 at a 20% volatility over ten years. The error grows with
`sigma sqrt(T)` and is a per cent and a half by the time anyone is quoting a
ten-year expiry.

A cap is a portfolio of options and a swaption is an option on the portfolio,
and the gap is large rather than a correction: on the five-year quarterly
structure the strip is worth **78.1% more** at a 20% volatility. The premium
*shrinks* as volatility rises — 82.8% at 10% against 74.8% at 40% — which is the
reverse of the natural guess that more volatility means more to choose between.

The check that found something was Jensen's inequality: the strip is worth at
least the option on it. Over twenty-five strikes and volatilities it holds
twenty-four times and fails once, by 6.98e-05, deep in the money at a low
volatility. That is not a pricing error. The fixed leg's annuity accrues on
unadjusted boundaries under 30/360 and the caplet weights on adjusted ones under
actual/360, so the annuity and the weight sum differ by 4.655e-05 relative and
the forward swap rate and the weight-average forward by 4.654e-05. At a strike
of 0.6 times the forward both prices are their intrinsics and the ratio is
`(1 + 4.655e-05)(1 - 1.1635e-04)`, which is 0.99993019 against a measured
0.99993019. The inequality is exact in the mathematics and violated in the
conventions, by precisely the amount the conventions differ — which is the most
useful thing an inequality can do.
