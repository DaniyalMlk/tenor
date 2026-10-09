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

## Phase 15 — A short rate model, so one parameter set prices everything

- [x] Hull-White's one factor, with the drift fitted to the initial curve
      through `A(t,T)` rather than by solving for `theta`
- [x] The analytic zero-coupon bond option, and put-call parity on it asserted
      rather than assumed
- [x] Jamshidian's decomposition for an option on a coupon bond, and so for a
      swaption
- [x] A second route by quadrature over the terminal short rate, so the
      decomposition has something independent to be measured against
- [x] The forward measure's mean implied from the martingale property of
      forward bond prices, and over-determined on purpose
- [x] Volatility calibration at a given mean reversion, with the trade-off
      between them measured rather than hidden inside a two-parameter fit
- [x] A command-line entry point reporting the identities before the price

Phases 13 and 14 put a swaption on the annuity measure, which takes a
volatility as an input, and phase 6 put a callable bond on a lattice, which has
no closed form and says nothing about a swaption. Neither is a model in the
sense of one specification pricing both. This is the smallest thing that is.

**The curve repricing is an identity, so it is exact.** `A(t,T)` carries the
whole of the initial curve, so at `t = 0` the model's bond price *is* the
curve's discount factor. Across a seventeen-pillar humped curve, every mean
reversion from 0.01 to 0.5 and every volatility from zero to 2%, the worst
relative error is **exactly 0.0**. Nothing is being fitted; a failure there
would mean the instantaneous forward is not what the formula assumes.

**The forward-measure mean came out cleaner than expected.** Pricing by
quadrature needs the law of `r(T)` under the `T`-forward measure, and copying a
drift adjustment from a reference would have been a third of a line and checked
by nothing. Implying it instead, from the fact that every forward bond price is
a martingale under that measure, over-determines it: each probe maturity gives
its own answer and they all have to agree. They agree to **0.0**, and the
reason is that the terms cancel identically — the `B²v/2` from the lognormal
expectation is exactly the convexity term inside `A`, leaving
`E^T[r(T)] = f(0,T)` with nothing left over. The forward-measure expectation of
the future short rate *is* the instantaneous forward.

**The obvious quadrature is wrong by twelve per cent.** A Gaussian weight
invites a Gauss-Hermite rule, and a sixty-four point one priced the
near-the-money options to five digits and a deep out-of-the-money one wrong by
**1.25e-01 relative**. A Gauss rule converges at that rate on a kink and an
option payoff is nothing but a kink. The symptom was misleading in a specific
way: the two routes agreed on the *difference* between a payer and a receiver
to 1e-12 while both were wrong, because the error is in the straddle and
cancels in parity — so every identity still held and only the level was out.
Putting the exercise boundary at a panel edge of a composite Gauss-Legendre
rule turned five digits into fifteen, and the two routes now agree to
**3.1e-14** relative over eighteen combinations of expiry, coupon and side, and
to 1.3e-15 on a ten-year-into-ten-year swaption.

**Mean reversion and volatility are not separately identified by one quote.**
A ten-into-ten payer struck at its forward is repriced *exactly* at every
reversion from 0.01 to 0.30 — the refitted volatility runs 0.492% to 3.226%, a
factor of **6.55** for a factor of 30 in `a`, and the refit error is at most
2.2e-16 at every point. There is no residual to choose between them, which is
why `calibrate` takes the reversion as an argument: a routine fitting both to
one quote would return whichever point on the ridge its initial guess fell
nearest and would report a residual of zero either way. What separates them is
the term structure: the pair fitted to the ten-year expiry prices a
one-year-into-ten-year anywhere from **23.4% below to 57.2% above** the
`a = 0.08` answer across that range.

**And the model's native convention is the normal one, measurably.** Reading
its own swaption prices back through `implied_volatility`, the normal
volatility across strikes from 60% to 140% of the forward moves 1.06 basis
points on a level of 45.55 — a smile **2.33%** wide, and monotone rather than
curved, so it is a skew and not a smile at all. The lognormal convention on the
same prices moves **41.91%** of its own level, eighteen times as much. A
Gaussian short rate makes a nearly Gaussian swap rate.

Two defects in the code, both found by identities.

The notional legs were on the swaption's nominal effective and maturity rather
than on the schedule's own adjusted start and last payment. A rolling
convention moves those by days, and the payer-less-receiver identity broke by
1.02e-04 on a swap worth 0.0491 — 0.21%, which reads as a pricing error rather
than as a date. The par rate a test compares against has to be computed on the
same dates for the same reason, and getting one of the two right is worse than
getting both wrong, because then only one number looks off.

And the default probe maturities for the forward measure reached thirty years
past the expiry, so a twenty-year expiry walked off a thirty-year curve and
arrived as `OffCurve` out of a diagnostic. The defaults are short offsets now.

One defect in the tests: a payer swaption was asserted to rise with its strike.
Paying a higher fixed rate is worse, so it falls, and the first run of the
suite said so.

## Phase 16 — A second factor, because one cannot decorrelate two rates

- [x] The five parameters, each with the constraint it has, and equal speeds
      allowed rather than refused
- [x] The bond price with the exact fit in the formula, not fitted
- [x] The deterministic shift, so the two factors connect to a short rate
- [x] The zero-coupon bond option in closed form, and caps and floors as the
      strips of them they exactly are
- [x] The correlation between two zero rates, in closed form
- [x] Exact simulation of the terminal factor law, as something independent to
      check the closed forms against
- [x] A fit of the fast volatility to a cap price, with the bracket searched
      for rather than assumed
- [x] A command-line entry point that prices a cap and then shows what the cap
      cannot identify

Phase 15 is a real model and has one limitation calibration cannot touch. With
a single factor every bond price is affine in the same scalar, so
`log P(t, T1)` and `log P(t, T2)` are affine in one Gaussian and their
correlation is **exactly one** for every pair of maturities at every parameter
setting. The short end and the long end move together, always. For anything
whose value depends on the shape of the curve rather than its level that is not
a small approximation, and `r = x + y + phi(t)` with two speeds fixes it
without giving up a single closed form.

**The exact fit is better than Hull-White's, and for the same reason.** Both
factors start at zero and the three variance terms telescope at `t = 0`, so the
curve reprices with an error of **identically 0.0** rather than to a tolerance.
Nothing is fitted; the whole initial curve enters as a ratio of its own
discount factors.

**Equal mean reversions collapse the model to the one-factor one, exactly**, and
that is why `a == b` is permitted rather than refused. With both speeds equal
the bond price depends on `x + y` alone, which is itself an Ornstein-Uhlenbeck
process with volatility `sqrt(sigma^2 + eta^2 + 2 rho sigma eta)` — so the
model *is* Hull-White there and every formula has to reduce. Measured against
`tenor.hullwhite`: the bond option volatility to **1e-16**, the option price to
**1e-15** across both payoffs and four strikes, the bond price to **1e-14**, and
the implied correlation to 1.0 within **1e-12**. A caller who reaches `a == b`
has written a one-factor model in two-factor notation, which is a waste rather
than an error, and refusing it would have cost the strongest check in the
module.

It is a check that crosses modules, and it earned its keep immediately: the
bond option was first written with its own `call`/`put` flag, clearer than
`tenor.options.Payoff` and the opposite way round, so the same call against the
two models in this package returned opposite options. The reduction test caught
it by disagreeing with Hull-White by exactly the parity amount at every
parameter setting. One convention per package beats a readable argument name.

**What the second factor buys, measured.** At a 0.50 fast speed, a 0.05 slow
speed and a driver correlation of -0.9, the one-year and ten-year zero rates
come out correlated **0.3592** at a one-year horizon against the one factor's
forced 1.0, and the six-month against the ten-year **0.1549**. The work is done
by the *gap* between the speeds rather than by the second factor's existence,
and it closes as they meet: 0.3592, 0.3387, 0.4255, 0.8862, 0.9999 and exactly
1.0000 as the slow speed walks up to the fast one. Not monotone — there is a
shallow minimum near 0.1 — which is worth knowing before fitting the pair.

**And a cap cannot identify any of it, structurally.** A cap is a strip of
options on *single* bonds, so its price depends only on each bond's own
volatility and never on the joint law of two maturities. Refitting the fast
volatility so that a one-year to ten-year semi-annual cap struck at 3.8%
reprices to the same number, seven driver correlations from -0.9 to 0.9 give
cap prices equal to within **2.4e-15 relative** — the same number, to rounding
— while the one-year-to-ten-year zero-rate correlation runs from **0.0674 to
0.9965**. Seven calibrations a cap market cannot distinguish, describing curves
that bend in completely different ways. That is the quantitative form of the
usual advice that caps calibrate volatility and swaptions calibrate
correlation.

**The refit is where this phase's bug was.** Written as a bisection from a
floor to a ceiling, it returned a price a third away from the target while
reporting success. A cap price is not monotone in the fast volatility: the bond
option variance is `sigma^2 A + eta^2 B + 2 rho sigma eta C` with all three
coefficients positive, so at a negative `rho` it is a parabola in `sigma` with
its minimum strictly inside the domain, and raising `sigma` from nothing makes
the cap *cheaper*. So `fit_fast_volatility` searches for its bracket on a
geometric grid and hands the first crossing from below to Brent, which picks
the branch where more volatility means a dearer cap; a target under the
cheapest attainable price raises `NoRoot` naming that price. The command
reports those rows as unreachable with the reason rather than as numbers, since
an unreachable target is itself part of the answer about what a cap constrains.

Two smaller findings. The accrual factor is not a second-order input: moving it
from 0.5 to 0.5055 — the 30/360 against the actual/365 figure for the same half
year, 1.1% apart — moves a two-year caplet struck at 3.6% by **7.0%**, because
the accrual enters through `1 + accrual * strike`, which is both the quantity
and the reciprocal of the bond strike, so it shifts a near-the-money option's
moneyness rather than scaling it. And the integrated factors are strictly less
correlated than their drivers at every finite horizon — the two mean reversions
weight the same history differently, so the levels cannot inherit `rho`, which
is why the result object carries both numbers.

One test defect of my own, corrected rather than loosened. The bond-price
reduction was first checked by passing Hull-White the curve's instantaneous
forward plus one factor, which found a 1.5e-03 relative gap. That is the size
of the convexity term in `phi`, not a defect: the state map is
`r(t) = phi(t) + x(t) + y(t)`, and the mapping is now the point of the test
rather than an incidental detail of it.

## Phase 17 — A rate paid outside the measure it is a martingale under

- [x] The annuity mapping function as a function of the rate, with its first
      two derivatives, under the flat-curve model
- [x] The mapping scaled onto the curve, and the residual arbitrage reported
      rather than left in
- [x] Static replication of an arbitrary payoff by the Carr-Madan expansion,
      over the swaptions the package already prices
- [x] Graded panels, so widening the ceiling cannot cost accuracy at the
      forward
- [x] CMS swaplets, caplets and floorlets, and the strip of them a CMS leg is
- [x] The convexity adjustment as a number in rate terms, not folded into a
      price
- [x] A surface the integral does not converge against refused, with the
      measured share beyond the ceiling in the message
- [x] A smile taken as a function of strike, since the adjustment reads the
      whole surface
- [x] A command-line entry point, with the payment-date and skew tables that
      say why the adjustment is not one number

Every instrument before this one pays a swap rate against the annuity that rate
is a martingale under, so the measure never had to be left. A CMS leaves it: the
rate is observed once and paid once, on a single date, so the expectation wanted
is under that date's forward measure and the forward read off the curve is a
biased forecast of what the contract pays. On a ten-year rate fixing in five
years, paid six months later, at a flat 24% volatility, the bias is **26.06
basis points** on a 4.1053% forward — 0.63% of the rate.

**A single period reduces exactly, and that is the check that carries the
file.** One fixed period with the CMS paying at the end of it makes
`A(T) = delta P(T, Tp)` identically, so the mapping is `1 / delta`, flat in the
rate, every derivative zero and the adjustment **identically 0.0**. It has to
be — that contract is a forward rate paid in arrears, which needs no adjustment
and never did. The replication is then the point mass at the kink and nothing
else, and it matches `tenor.options` to **5e-16** across four strikes and both
payoffs. It is the only check here that would catch the point mass carrying the
wrong sign, which is exactly the error Phase 16 made with a call and a put.

The quadrature is checked against arithmetic from outside the package rather
than against itself. Against that flat mapping, replicating `S**2` has to return
`F**2 exp(sigma**2 T)` under Black and `F**2 + sigma**2 T` under Bachelier; both
come back at rounding, and that is the only statement here with no model in it.

**Widening the ceiling made it worse, which is the second time this has
happened.** The second moment is the one functional sensitive to the far tail.
The lognormal ceiling is `F exp(w sigma sqrt(T))`, so it moves out exponentially
in the width while the panel count stays put, and every extra standard deviation
coarsens the mesh at the forward where the integrand lives. Uniform panels gave
1.4e-09, 3.2e-08, 7.9e-04 and 1.8e-01 relative as the ceiling went from six
standard deviations to twenty — monotonically worse for covering more of the
distribution, the same mechanism as widening a finite-difference domain at a
fixed node count. Spacing the edges evenly in `log K` holds every ceiling from
eight to thirty at rounding.

**The flat-curve model is arbitrageable twice over, and both are measured.**
Raw, it returns the flat curve's own ratio at the forward rather than the real
one, pricing a zero-coupon bond 0.88% wrong on an upward-sloping curve — a
quarter of the adjustment. Scaled onto the curve at the forward, the constant
payoff is *still* worth 0.217% more than the bond it is, because matching at a
point does not match an expectation. Dividing by the replicated unit payoff
enforces the one no-arbitrage condition there is, moves the adjustment by 3.6%,
and is a no-op wherever the mapping is constant or the volatility is zero, so no
reduction above is disturbed.

**Paying later unwinds the convexity rather than compounding it**, which came
out against the guess written down before it was measured. The adjustment runs
29.28, 26.06, 22.92, 16.83, 5.43, 0.07 and -23.86 basis points as the payment
moves from the fixing out to ten years past it — monotonically down, and through
zero. The sign is the sign of `alpha'(F)`, and `alpha'(F)` vanishes where the
payment date meets the annuity's own annuity-weighted mean payment time: 5.1872
years, against a slope crossing zero at 62 months, which is 5.1667. Structural,
not coincidental — the mapping is one discount factor over a weighted sum of
them, so a payment at the weighted mean makes numerator and denominator respond
to the rate alike.

**And it is a smile instrument, which is the practical reason the module
exists.** The adjustment integrates swaption prices across every strike, so it
reads the whole surface. Against surfaces leaving every at-the-money swaption
worth exactly what the flat one does, it comes to 100.0%, 96.9%, 94.3%, 92.1%
and 88.5% of the flat number at skews of 0 to -0.50 per unit of rate — an 8%
error from a marking choice no at-the-money quote can see. A surface sloping
*up* has no answer at all: a volatility rising without bound stops the swaption
price decaying, the integral diverges, and the same fixing reads 35.6 basis
points at a six standard deviation ceiling and 36565 at fourteen. None of those
looks wrong, so the measured share of the value beyond the ceiling — 5.5e-24
flat, exactly 0.0 against a downward skew, 120% there — is reported and the
replication refused.

One defect of my own, fixed rather than tolerated. A zero-volatility
replication was being routed through the mapping and a degenerate integration
range, which left the adjustment at -6.9e-18 instead of zero and could not cover
an option's kink at all, so a strike away from the forward was refused where its
intrinsic value was the answer. A known rate makes the payoff a fixed cashflow
on a known date; it is now answered as one, with no model and no quadrature.

## Phase 18 — Several exercise dates, and no formula for any of them

- [x] A trinomial tree on the Hull-White state variable, with the exact
      conditional moments of the step rather than their Euler approximations
- [x] Branching to the nodes around each node's own conditional mean, so every
      probability is positive without a cap on the tree width
- [x] Shifts by forward induction on Arrow-Debreu prices, so the grid's zeros are
      repriced rather than approximated
- [x] One state spacing across steps of differing length, so every exercise date
      lands exactly on a node
- [x] Backward induction with an exercise decision, the exercise value read
      analytically at the node
- [x] The co-terminal Europeans, the best of them, and the switch value over it
- [x] The exercise boundary and the share of mass that takes it, at each date
- [x] The error a bond outliving the grid reprices to, reported rather than
      inferred from a price
- [x] A command-line entry point that leads with the switch value and keeps the
      exact diagnostic apart from the approximate one

Every option before this one has a single exercise date, which is what makes
`tenor.hullwhite` able to price it twice in closed form: the payoff is a function
of the short rate at that date, and both routes integrate it against that rate's
density. A Bermudan breaks the structure rather than complicating it. The value
of not exercising today is the value of still holding the right tomorrow, so the
quantity being integrated is the answer to the same problem one date later and
there is nothing to write down.

`tenor.lattice` was not the answer, and not for reasons of convenience. Its tree
carries a volatility of its own, expresses early exercise as a schedule of call
prices held by an issuer, and has no relationship to the `(a, sigma)` fitted to a
swaption market. Pricing a Bermudan swaption there would mean holding two
unrelated models of one short rate and hoping they agreed.

**Rounding to the nearest node is the whole construction.** The textbook
Hull-White tree branches up-mid-down from a fixed offset and needs a cap on its
width, because far from the centre the mean reversion pulls the conditional mean
past the neighbouring node and a probability goes negative. Branching instead to
the three nodes around each node's *own* conditional mean bounds the residual
drift by half a spacing, which makes `pm = 1 - v - u**2` and
`pd = (v + u**2 - u) / 2` positive everywhere for `v` near a third. Nothing is
truncated, so total state-price mass at each step is the discount factor to it
rather than a share of it, and the width stops growing on its own: once `j a dt`
exceeds half a spacing the centre comes back a node and the top stays put.

**Exercise dates land on nodes, which equal time steps cannot deliver.** An
exercise decision is a kink in the value function and a step that straddles one
smears it. The dates are swap anniversaries whose day counts differ by a day or
three, so the grid is built interval by interval with a single spacing shared
across steps of slightly different length. Positivity then needs the step
variance between a quarter and three quarters of `dx**2`, which the uniform case
satisfies at a third with room on both sides, and the bound is checked against
the grid produced rather than assumed from the request.

**Forward induction recovers the convexity term, which was not expected.**
Fitting one rate per step to reproduce the next zero looks like a statement about
the step's average instantaneous forward — a curve quantity with no model in it.
The fitted shift sits above that average by exactly the second term of `phi(t)`:
8.9e-04 against an analytic 8.9e-04 at `a = 0.05, sigma = 0.01` and five years,
agreeing to 1.3e-05 relative at four steps a year and 9.0e-08 at forty-eight, and
identically on flat, sloped and humped curves. The discount-weighted spread of
the state across a step *is* that convexity, so fitting a discount factor finds
it. Exercise values are still read at the analytic `phi`, because that is what
the closed forms are written in and because a decision is about an instant rather
than an interval.

**Two identities hold exactly and are asserted as such.** The grid's zeros
reprice to 2.2e-16, which is the induction's own construction. And a model with
no volatility has no decision to make, so the price collapses to the best
discounted intrinsic value computed off the curve alone — to **5.9e-17** across
twelve curve, payoff and mesh combinations, with no tree quantity in the
benchmark.

That second one was reached through a defect worth recording. A zero-volatility
tree was still being widened, which added nodes of exactly zero probability whose
*rates* were a placeholder spacing apart; a bond price at one of them overflowed
to infinity and the leg subtraction returned **NaN** — a number-shaped
non-number, which is the worst available outcome. A degenerate model now gets a
degenerate tree.

**Convergence is a band, not a point, and the reason is where the kink sits.**
The exercise boundary does not fall on a node, and it moves between nodes as the
mesh changes, so the error against the analytic European oscillates in sign and
size. At 48 steps a year this instrument came within **1.9e-06** of the analytic
price while the worst case over its neighbouring meshes was **1.8e-05**, and a
convergence ratio read off that point came out at **-865**. Measured on bands
instead, the worst case runs 4.8e-05, 1.8e-05 and 6.5e-06 across 24-40, 48-64 and
96-112 steps a year: about 2.7 per doubling.

The underlying rate is first order, and it is visible without going through a
price at all. The state prices at a node, weighted by analytic prices of a bond
that outlives the grid, fall short of today's discount factor by **9.7e-05** at
twelve steps a year and halve at every doubling — a ratio of **2.00** each time,
the same on all three curves, and always negative. That is the rectangle rule the
discounting uses, and forward induction removes only the part of it that does not
depend on the state.

**Two test claims were wrong and were corrected rather than loosened.** A deep
in-the-money payer was asserted to be worth exactly the forward swap it would be
exercised into, since a payoff positive everywhere has nothing left to wait for.
A positive payoff loses its kink, not its choice: the price is above that swap by
**1.33e-03** on 0.202, stable to three figures from 24 to 96 steps a year, and 7
to 9 per cent of the state-price mass waits. The co-terminal swap shortens with
every date, so date one dominates *in expectation*; per path it does not, and
where rates have fallen the right is worth more than the swap.

And the exercise boundary was asserted to fall over time, on the reasoning that a
shorter remaining swap gives up less by being taken. True, and incomplete. Flat,
it falls from 3.92% to 2.95% over seven annual dates. On a curve rising 15 basis
points a year it reverses once, from 3.128% to 3.212%, because the swap starting
a year later pays against higher forwards; on a humped curve it rises over four
consecutive dates. Only the last date is reliably the lowest bar. The assertion
would have passed on the one curve it was tried on.

**Payer-receiver parity is the identity to *not* reach for here.** A European
payer less a European receiver is the forward swap exactly, because the two
payoffs are the positive and negative parts of one number. Two Bermudans are
maxima over different stopping rules, and a difference of maxima is not the
maximum of a difference: measured at 0.00212 against a swap worth 0.00136, so
treating it as a check would report a 56% pricing error on correct prices.

## Phase 19 — Reading a quote convention back into a model input

- [x] A cap quote type carrying a maturity and a flat volatility, with the
      strike on the call rather than on the quote
- [x] A bucket type carrying the periods one quote added and the volatility
      that explains them
- [x] A piecewise-constant caplet volatility curve, looked up by fixing date
      and flat outside the quoted range
- [x] The bootstrap in maturity, one monotone root solve per step
- [x] Schedules checked to nest, since `generate` runs backward from maturity
- [x] Every quote repriced from the stripped curve, as the only check that goes
      through nothing else
- [x] A flat volatility recovered back out of a caplet curve, and a flat curve
      shown to quote flat at every maturity
- [x] The sensitivity of each bucket to the quote that set it, reported
- [x] The width of the set of volatilities that reprice each cap, reported, and
      a bucket wider than a basis point of volatility refused
- [x] The steepening between quoted and stripped curves measured
- [x] The feasibility boundary measured rather than assumed to be about
      negative variance
- [x] Monotonicity recorded as a measurement, not asserted as a property
- [x] A command-line entry point printing the curve beside both conditioning
      columns

`Cap.value` takes one volatility for the whole strip, which is how a cap is
*quoted* and not how anything is priced. A quoted flat volatility is the single
number reproducing a cap's premium; what prices anything else is the volatility
of each forward underneath. This phase is the inverse, and it is a bootstrap in
maturity: the shortest cap's periods come from its own quote, and each longer
cap's new periods get one volatility from a root solve while the earlier ones
stay where previous steps put them.

Three things that are not the loop. The schedules have to nest, and
`schedule.generate` runs backward from maturity, so a shorter cap's periods are
a prefix of a longer one's only when every maturity sits on the index's own roll
grid from the shared effective date — checked against the realised schedules and
refused by name. The strike lives on the call rather than on the quote, because
volatilities at two strikes are not the same quantity and a set of
at-the-money caps struck at their own par rates is not a term structure of one
thing. And each bucket reports its own conditioning, because this problem is
ill-posed in a way that a single number hides.

### The stripped curve outruns the quotes, and by a lot

A flat volatility is a premium-weighted average of the caplet volatilities under
it, not an average of the volatilities themselves, so a rising quote curve has to
be explained by a caplet curve that rises faster. On a five-year quarterly
structure with quotes running 18% to 26%, the buckets run 18.0% to **31.2%**: the
far bucket sits 5.2 volatility points above the quote that set it and the slope
between the last two buckets is 1.77 times the slope between the last two quotes.
The first bucket's volatility is its quote exactly, because a one-bucket strip
has nothing to average — the one row of the table that can be checked by hand.

A hump is worse than a slope. Quotes of 18, 26, 30, 26 and 22 per cent strip to
18.0, 29.3, **34.5**, 18.6 and 11.5, so a four-point fall in the quotes is a
sixteen-point fall in the buckets: the later periods have to undo an average the
earlier ones are holding up. Monotone quotes give monotone buckets on this
structure and nothing guarantees it, which is why the tests record the
measurement rather than the expectation.

### Monotone is not steep, and that is the trap at both ends

The premium rises with the volatility everywhere, so a bracket on `[0, ceiling]`
always contains a root and a solver always returns something. At the far
maturities it returns something badly conditioned: a basis point on the one-year
quote moves its bucket by 1.00 basis points and a basis point on the five-year
quote moves the last bucket by **3.63**, because each bucket is a smaller share
of its cap's premium than the last.

Deep in the money it returns nonsense. An option eleven standard deviations into
the money is worth its intrinsic value to *the same double*, so every volatility
explains the quote equally well. Each bucket therefore carries the width of the
set of volatilities repricing its cap to within the premium tolerance, and a
bucket wider than one basis point of volatility is refused as unidentified rather
than reported. Measured at an 18% quote against forwards of 3.23% to 3.39%: the
answer is pinned to 1.2e-13 at a strike of 3.5%, 1.3e-09 at 2%, 6.2e-06 at 1.5%,
and the lowest strike the quote says anything about at all is **1.3901%**.

That guard also turns out to be what stops a falling quote curve. Holding the
one-year quote at 18% and walking the two-year quote down, the strip survives a
fall of **1202 basis points** and not 1203 — and at the last quote it accepts,
the bucket's indeterminacy has come within 0.3% of the 1.0e-04 limit, with
the premium unattainable one step below. The two refusals coincide because they
are one statement measured two ways: the premium has stopped responding to the
volatility. The usual worry about a sequential volatility bootstrap is a negative
variance, and that is not what binds here.

The last consequence is the least obvious. **A zero quote identifies a zero
volatility only exactly at the money.** Vega vanishes with the volatility away
from the strike — both `d1` and `d2` run off to infinity and take the density
with them — so an interval of volatilities all produce the intrinsic value. At
the money the premium is linear in the volatility with a slope near
`0.4 F sqrt(T)`, and zero is pinned to 2.8e-16.

## Phase 20 — Two currencies, and the quantity that is left over

- [x] Covered interest parity in both directions, with the round trip returning
      the same float rather than a close one
- [x] The spot settlement lag carried explicitly, since parity runs from the
      spot date and not from today
- [x] Forward points, in the pip of the quote currency, with the pip named
      rather than assumed
- [x] The foreign discount curve a strip of forwards implies, built by
      inversion because a forward depends on one pillar and not on every pillar
      before it
- [x] The one quantity the strip cannot determine — the foreign discount factor
      to the spot date — made an argument, with the cost of its default measured
- [x] The basis term structure between the implied curve and the foreign
      currency's own
- [x] The par spread on a constant-notional cross-currency leg, from the
      telescoping of the notional exchange rather than from a valuation solve
- [x] The accrual convention and the compounding correction measured
      separately, because they pull opposite ways and the net hides both
- [x] A command-line entry point whose lag column is printed to be looked at

A foreign exchange forward is an identity rather than a model, which makes this
the only module here whose checks can be assertions on equality. It is also the
first place in the library where the market quotes more than the mathematics
determines: two curves, a spot and a strip of forwards are four quantities with
one relation between them, so something has to give, and what gives is the
assumption that a currency has one discount curve.

### The convention nobody can see is the one that costs

The spot lag is two business days and the error from ignoring it is the rate
differential over those days applied to the spot — so it hardly changes with
maturity while the forward points grow with it. Under a quarter of a per cent of
a five-year forward's points and nearly five per cent of a three-month one's,
which is the wrong way round for noticing. The same two days are what a strip of
forwards cannot pin down, and defaulting them to the domestic curve moves the
implied one-year basis by 1.47 basis points.

### A spread on an accrual is not a continuously compounded rate

Two corrections separate them and they have opposite signs: quarterly payment
against continuous compounding is worth +0.785%, and an ACT/360 accrual against
an ACT/365F curve is worth -1.37%. The net, -0.595%, is smaller than either, so
reporting only the net would suggest the correction has one direction. The gap
is proportional to the basis rather than quadratic in it, which is the signature
of a convention rather than an approximation.

### And a basis curve is averaged the way an annuity averages

The par spread is the annuity-weighted average of the *forward* basis. With a
zero basis rising from 10 to 40 basis points the forward basis reaches 70, and
the spread is 39.02 — against 25 for a time average of the zero basis and 11.5
for its short end.
