# tenor

A fixed-income curve toolkit, in pure Python with no dependencies.

Fixed income is a field where the arithmetic is easy and the conventions decide
the answer. A bond priced on the wrong day count basis is wrong by a few basis
points, which is the size of the spread anybody is trying to measure — so it is
not approximately right, it is answering a different question. This library
treats every convention as an argument with a name, never a default, and checks
each one against the published rules rather than against itself.

All seven phases in `ROADMAP.md` are built: dates, day counts, holiday
calendars and payment schedules; discount curves, compounding conventions and
three interpolation schemes; a bootstrapper over deposits, futures and par
swaps; bond analytics — price, yield, duration, convexity and basis point
values; curve risk — key rate durations, shape shifts and
instrument-by-instrument risk; spreads — Z-spread, a calibrated short rate
lattice, American exercise and option-adjusted spread; and a command line with
a worked example.

## Installing

```bash
pip install tenor
```

> **Not on the package index yet.** The `pip install` line above is what it
> will be; until the first release lands, install from source:
>
> ```bash
> pip install "git+https://github.com/DaniyalMlk/tenor.git"
> ```

Python 3.10 or newer. The library imports only the standard library, so there is
nothing else to resolve and nothing to compile.

## Using it

```python
from datetime import date
from tenor import Basis, year_fraction

year_fraction(date(2021, 2, 28), date(2021, 8, 31), Basis.ACT_365F)          # 0.504110
year_fraction(date(2021, 2, 28), date(2021, 8, 31), Basis.THIRTY_360_BOND)   # 0.508333
year_fraction(date(2021, 2, 28), date(2021, 8, 31), Basis.THIRTY_360_US)     # 0.500000
```

```python
from tenor import Frequency, Rolling, calendar_from, generate

us = calendar_from("US", [date(2021, 7, 5), date(2021, 11, 25)])

schedule = generate(
    date(2021, 3, 10), date(2024, 8, 15), Frequency.SEMI_ANNUAL,
    calendar=us, rolling=Rolling.MODIFIED_FOLLOWING,
)

schedule.payment_dates     # business days, rolled
schedule.has_stub          # True — the first period is odd
schedule.stubs[0].stub     # Stub.SHORT_FIRST
schedule[0].start          # unadjusted, for accrual
schedule[0].adjusted_end   # rolled, for payment
```

```python
from tenor import Basis, Compounding, DiscountCurve, Interpolation

curve = DiscountCurve.from_zeros(
    date(2021, 1, 4),
    [(date(2022, 1, 4), 0.010), (date(2026, 1, 4), 0.018)],
    basis=Basis.ACT_365F,
    interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
)

curve.discount(date(2024, 6, 1))
curve.zero_rate(date(2024, 6, 1), Compounding.SEMI_ANNUAL)
curve.forward_rate(date(2023, 1, 4), date(2024, 1, 4))
curve.instantaneous_forward(2.5)   # what the interpolation is really doing
curve.reprices_pillars()           # True, and asserted rather than assumed

smooth = curve.with_interpolation(Interpolation.MONOTONE_CONVEX)
smooth.instantaneous_forward(2.5)  # continuous across the pillars, and positive
smooth.monotone.segments[1].region # which of the four shapes that interval took
```

```python
from tenor import Deposit, Future, Swap, bootstrap, ho_lee_convexity

built = bootstrap(
    date(2021, 1, 4),
    [
        Deposit(date(2021, 1, 4), date(2021, 7, 5), 0.0035, Basis.ACT_360),
        Future(date(2021, 9, 15), date(2021, 12, 15), 99.55, Basis.ACT_360,
               convexity=ho_lee_convexity(0.01, 0.70, 0.95)),
        Swap(date(2021, 1, 4), date(2031, 1, 6), 0.0195),
    ],
    basis=Basis.ACT_365F,
    interpolation=Interpolation.MONOTONE_CONVEX,
)

built.curve.zero_rate(date(2028, 6, 1))
built.sweeps          # how many passes it took to settle
built.worst_error     # ~1e-15: every quote still prices to par
built.reprices()      # the identity, asserted rather than assumed
built.solutions[2]    # the solve itself: iterations, bracket, residual
```

```python
from tenor import Bond, Compounding

bond = Bond(date(2021, 1, 4), date(2031, 1, 4), 0.05)   # 5% of 2031

bond.accrued(date(2021, 4, 15))              # per 100, on the stated convention
bond.clean_price(0.043, date(2021, 4, 15))   # quoted price
bond.yield_from_clean(96.5, date(2021, 4, 15))  # solved, with its convergence

bond.macaulay_duration(0.043, date(2021, 1, 4))   # a time, in years
bond.modified_duration(0.043, date(2021, 1, 4))   # a sensitivity to its yield
bond.convexity(0.043, date(2021, 1, 4))           # in closed form

curve = built.curve
bond.price_from_curve(curve, date(2021, 1, 4))
bond.pv01(bond.curve_yield(curve, date(2021, 1, 4)).value, date(2021, 1, 4))
bond.dv01(curve, date(2021, 1, 4), compounding=Compounding.SEMI_ANNUAL)
bond.effective_duration(curve, date(2021, 1, 4))
```

```python
from tenor import buckets_from, instrument_risk, key_rates, level, shape_duration, slope

value = lambda c: bond.price_from_curve(c, date(2021, 1, 4))

shape_duration(value, curve, level())             # the total, as one number
shape_duration(value, curve, slope(curve.times[-1]))   # response to a steepening

for one in key_rates(value, curve, buckets_from(curve, [0.5, 10])):
    one.bucket.name, one.duration, one.value      # where the exposure sits
    # (a bucket in a gap between pillars is refused rather than returned as zero)

for one in instrument_risk(value, built):
    one.name, one.value                           # what to trade against it
```

```python
from tenor import FloatingNote

note = FloatingNote(
    date(2022, 1, 17), date(2029, 1, 17), quoted_margin=0.0075,
    frequency=Frequency.QUARTERLY, basis=Basis.ACT_360,
)

note.dirty_price(curve, date(2024, 1, 17), margin=0.0075)   # exactly 100.0
note.rate_duration(curve, date(2024, 1, 17), margin=0.0075) # exactly 0.0
note.spread_duration(curve, date(2024, 1, 17), margin=0.0075)      # 4.77 years
note.discount_margin(curve, 99.0, date(2024, 1, 17)).value         # 0.009608
```

```python
from tenor import LinkedBond, ReferenceIndex, breakeven_inflation

index = ReferenceIndex({(2025, 6): 106.5, (2025, 7): 106.7, (2025, 8): 107.0})
index.last_known_date()                     # 2025-11-01: the lag reaches this far

linker = LinkedBond(
    date(2022, 4, 1), date(2032, 4, 1), coupon=0.015, base_index=100.5,
    frequency=Frequency.SEMI_ANNUAL, basis=Basis.ACT_ACT_ISDA,
)

linker.index_ratio(index, date(2025, 9, 15))            # 1.065498
linker.real_clean_price(0.011, date(2025, 9, 15))       # 102.5187, quoted real
linker.settlement_amount(102.5187, index, date(2025, 9, 15))   # 109.9626, money
linker.real_modified_duration(0.011, date(2025, 9, 15)) # 6.19 years

breakeven_inflation(0.0425, 0.011)                      # 0.031157, Fisher exact
breakeven_inflation(0.0425, 0.011, exact=False)         # 0.031500, as quoted
```

## The command line

```
$ tenor curve examples/quotes.txt --reference 2021-01-05 --basis ACT_365F
reference: 2021-01-05
interpolation: log-linear on discount factors
sweeps: 1
worst_repricing_error: 5.551115123e-17
reprices: True
pillars:
  date=2021-07-05  years=0.49589  discount=0.9989954546  zero=0.002026758939
  ...

$ tenor option examples/quotes.txt --reference 2021-01-05 --basis ACT_365F \
      --maturity 2031-01-05 --coupon 0.05 --volatility 0.15 \
      --first-call 2026-01-05
bullet: 125.5612364
with_option: 116.7589796
option_value: 8.802256883
z_spread: 0.008904973855
oas: 1.208616557e-17
option_cost: 0.008904973855

$ tenor horizon examples/quotes.txt --reference 2021-01-05 --basis ACT_365F \
      --maturity 2031-01-05 --coupon 0.03 --horizon 2022-01-05
start_price: 107.2969819
forward_price: 104.7956417
rolled_price: 107.2879716
coupon_income: 3.005534685
forward_price_change: -2.501340175
financing_cost: 0.5041945095
carry: 0.5041945095
roll_down: 2.49232992
total_return_bps: 279.2738787
arbitrage_free: True

$ tenor floating examples/quotes.txt --reference 2021-01-05 --basis ACT_365F \
      --maturity 2026-01-05 --quoted-margin 0.0075
note: index + 75.0bp of 2026-01-05
on_reset_date: True
discount_margin: 0.0075
dirty: 100
clean: 100
spread_duration: 4.827613483
rate_duration: -7.105427358e-13
margin_value_of_a_basis_point: 0.0482635851
coupon_count: 20

$ tenor linker examples/quotes.txt --reference 2021-01-05 --basis ACT_365F \
      --index examples/cpi.txt --maturity 2031-01-05 --coupon 0.015 \
      --base-index 100.5 --issued 2019-01-05 --projection 0.02 \
      --real-yield 0.005 --nominal-yield 0.022
bond: 1.500% linker of 2031-01-05
index_published_to: 2020-11
reference_index_known_to: 2021-02-01
index_ratio: 1.050937655
real_yield: 0.005
real_clean: 109.7422441
real_modified_duration: 9.338612259
settlement_amount: 115.3322567
redemption_index_ratio: 1.281190635
deflation_floor_binds: False
inflation_implied_by_the_invoice: 0.01718356716
breakeven_exact: 0.01691542289
breakeven_quoted: 0.017
accretion_one_year: {'total': 0.01997172, 'published': 0.00143816,
                     'projected': 0.01850694, 'published_share': 0.07200994,
                     'published_through': '2021-02-01'}
```

`--basis` is required, not defaulted. Also `price`, `risk`, `horizon`,
`floating` and `linker`; `--json` on any of them. Every subcommand reports the identity that goes with its numbers — the
repricing error, the key rate sum against the total duration, whether the
lattice reprices the curve — rather than only the numbers.

`examples/worked.py` runs the whole library in one pass and exits non-zero if
any published figure moves.

## Design

### A floater at its quoted margin has no rate risk at all

Not a little, because its coupon resets soon. None. The identity is this: with the
discount margin equal to the quoted margin, each coupon is exactly its own
discount denominator minus one, so the sum telescopes and the price is exactly par
for any curve of any shape at any level on any day count basis. That is asserted
in the tests to 5e-13 across four curve shapes, four bases and three frequencies,
not to a basis point.

Measuring it showed the consequence is stronger than the textbook statement.
Substituting the telescoping sum for the periods after the current one collapses
the whole price to

    (1 + (fixing + m) tau_1) / (1 + (fixing + m) tau_remaining)

in which no curve appears at all. So the rate duration is exactly zero on *any*
settlement date, not only on a reset date, provided the current period's fixing is
known.

What a floater has away from par is the rate sensitivity of the annuity left over
when the discount margin differs from the quoted one. It is linear in that
difference and it changes sign with the side of par. On the five-year quarterly
note in the tests, against a quoted 75bp:

| discount margin | price | rate duration |
|---|---|---|
| 25bp | 102.42 | **+0.061** years |
| 75bp | 100.00 | 0.000 |
| 125bp | 97.65 | −0.061 |
| 275bp | 90.94 | −0.250 |

A floater trading cheap to its quoted margin *gains* when rates rise, because the
negative annuity it is carrying gets discounted harder. Not having the current
fixing adds a separate −0.057 years at par three weeks into a quarterly period,
which is the cost of a missing fixings history stated as risk rather than as an
apology.

Spread duration is the one that survives, and it is a whole-life number because
the margin is discounted across every remaining period: 1.014 times the modified
duration of a fixed bond of the same maturity at three years, 1.005 at five and
0.974 at ten. Both are reported, and the command line prints both, because
reporting one as "the duration" would be reporting the wrong one.

Two conventions are settled rather than left implicit. The projected index rate is
backed out of the discount factors against **the coupon's own accrual fraction**,
not taken from `DiscountCurve.forward_rate` — the curve's forward is a simple rate
over the curve's year fraction and the coupon does not accrue over that one. On an
ACT/360 note over an ACT/365F curve the two differ by the ratio of the bases,
20.72bp against 21.01bp on the first period of the test note; the price effect is
half a basis point at a wide margin and 0.05bp on the implied discount margin,
which is the honest size of it. And the discount margin is applied to the
projected forward period by period rather than added to a zero rate, because
adding a constant to the zero curve is a Z-spread and that is a different number.

**The par identity does not check either of those.** The telescoping needs only
that the coupon and the discount denominator use the *same* rate over the *same*
fraction; it does not care whether that rate is right. A note projected with its
index scaled by an arbitrary factor still prices at exactly par, and there is a
test asserting that so nobody relies on par to catch a projection error.

Two things a floater cannot do, both refusals rather than plausible numbers.
Accrued interest needs the current fixing, which is not on any curve — returning
the margin's accrual alone would be a number and would be wrong by the whole of
the index, about 85% of it on a note paying 4% over 75bp. And a settlement date
inside a period that began before the curve's reference date cannot be projected
at all, because there is no discount factor at the reset to take a forward from;
that is the ordinary case on any date that is not a reset date, so the message
names the fixing rather than the curve.

### Carry earns nothing; roll-down is the whole of it

A bond held to a horizon on a curve that evolves to its own forwards earns its
funding cost and nothing else. Coupon income and the pull of the price towards
its forward offset the financing exactly. This is an identity, not an
approximation, and `tenor.horizon` asserts it to machine precision — measured
at 2e-15 per 100 of notional across coupons from 0% to 9%, three curve shapes,
every interpolation scheme, and a year of settlement dates.

So the expected excess return of a bond position is not carry. It is the curve
*failing* to evolve to its forwards: the bond ages, its remaining maturity
shortens, and on an unchanged spot curve it is repriced off a lower point. That
is roll-down, and it is all of it.

On the bundled quote screen — a curve running from 20bp to 220bp — a 3% 2031
bond held for a year returns **279 basis points, of which 50 is the financing
cost and 249 is roll-down.**

This needs two curves and they are different objects:

| | what it is | what it is for |
|---|---|---|
| `forward_curve` | every discount factor divided by the horizon's | the no-arbitrage half |
| `rolled_curve` | the same zero rate at each *tenor*, reference moved | the trader's unchanged curve |

Building the first and calling it the second is the natural mistake, and it
makes roll-down come out as exactly zero every time — which looks plausible and
is the answer to a different question. On a flat curve the two coincide, which
is precisely why roll-down is zero there.

The roll is applied in days, not by adding a calendar period. A pillar 1826
days out stays 1826 days out; adding months instead moves pillars by unequal
amounts and quietly changes the curve's shape, which is the one thing this must
not do.

**Why both definitions of "carry" are reported.** The market uses the word for
at least two quantities. Here `carry` is coupon income plus the forward price
change — the no-arbitrage total, identically the financing cost.
`income_less_financing` is the other one. They are not interchangeable and the
second can point the wrong way:

| | income − financing | actual return |
|---|---|---|
| 9% of 2031 | **+4.83** | 443.6 bp |
| 0% of 2031 | **−2.00** | **470.7 bp** |

Nearly seven points of apparent carry separate the two bonds, the zero-coupon
one looks far worse on it, and the zero-coupon one actually earns *more*. The
high coupon is paid for by a price falling towards par by the same amount. A
decomposition whose leading term ranks two positions backwards is worse than no
decomposition, which is why both are reported and neither is called "carry"
without saying which.

Coupons paid inside the window are reinvested at the curve's own forward rates.
That is the only assumption under which the identity holds; reinvesting at a
chosen rate would make carry differ from the financing cost by the size of the
view, which is a fact about the view and not about the bond.

### Half of a linker's next quarter of inflation is already published

An index-linked bond references a price index three months back, interpolated by
day of month, because the index for this month does not exist yet when this month
settles. The consequence is not symmetrical and is easy to state wrongly. The
reference index is determined out to the first of the month three months after the
last published figure — from mid-September with a series through August, that is
1 November, six weeks away rather than three months. On a 2% path from that date,
51.4% of the next quarter's accretion is arithmetic, 25.7% of the next six
months' and 12.7% of the next year's, and the share falls through each month until
the next print lands. `index_accretion` splits the two, because a caller hedging a
three-month inflation exposure with a linker is trading something about half of
whose payoff is already fixed.

The interpolation itself is worth 15bp of index level by the end of a month: on
the series in the tests the 28th of September reads 107.159 interpolated against
107.000 stepped. Both conventions exist in the market — modern linkers interpolate
daily, index-linked gilts before 2005 stepped once a month on an eight-month lag —
so `IndexInterpolation` names the choice rather than assuming one.

### A linker is quoted in real terms and settles in money

The price and yield on the screen are real; the invoice is the real *dirty* price
times the index ratio at settlement. Indexing the clean price and adding
unindexed accrued is the plausible mistake, and it is invisible on a new issue
because the ratio is near one — on the bond above, with 6.5% of cumulative
accretion behind it, it is 4 cents per 100 and growing. Every method here says
which space it is in, and the real ones delegate to the fixed-coupon bond rather
than restating the street discounting convention.

Real duration is not comparable with a nominal bond's. The linker above has a real
modified duration of 6.19 years against 5.59 for a 4% nominal of the same
maturity, and the gap is almost entirely the coupon: a 1.5% real coupon puts more
of its value at the end. It is a sensitivity to the real yield, and the invoice
also moves with the index ratio — which is the instrument's reason to exist rather
than a risk to hedge.

### The deflation floor is on the principal, and the last coupon is not principal

On the US structure the principal returned is the greater of the index ratio and
one, and the coupons are unfloored. So a bond in cumulative deflation redeems at
par while its coupons keep shrinking, and `redemption_ratio` reports whether the
floor binds rather than applying it silently.

This produced a defect worth recording. The fixed-coupon bond bundles the final
coupon into the redemption payment, as it should — they are one wire. Flooring
that bundle's index ratio floors the final coupon as well, which is a different
and more valuable bond. It can only show up on one in cumulative deflation, so
nothing else in the suite would have caught it. The final coupon and the
principal are now separate flows on the same day.

### Breakeven inflation is not the difference of two yields

It is quoted as one, and the Fisher relation is multiplicative. At a 4.5% nominal
against a 2.0% real the quoted difference is 250.0bp and the exact rate is
245.1bp; the gap is the cross term, so it widens with the level of both and
reaches 19.2bp at 9% against 4%. Both forms are available and the exact one is the
default, because 4.9bp is wider than the bid-offer on the spread it is quoted as.

Neither number is an inflation expectation. Both are that plus an inflation risk
premium plus the liquidity difference between two bonds, and two prices do not
identify three things. `implied_inflation_from_price` is the other route — the
constant inflation path that reprices the linker off a nominal curve — and it is
worth having both, because that one takes the curve as given instead of
inheriting a second bond's liquidity.

The two routes agreeing is a useful check on both, since they share no code: on
the bundled example the invoice implies 1.7184% against a two-yield breakeven of
1.6915%, 2.7bp apart, which is the shape of the curve against the single yield
rather than an error in either.

### A gap in an index series is refused

A missing monthly print, interpolated across, becomes a plausible number that
nobody published. `ReferenceIndex` requires a contiguous series and names the
month that is absent. Projecting past the end of the series is a different matter:
it is the ordinary condition of every flow beyond the next quarter, so it raises
`Unpublished` — which the caller answers by supplying an inflation rate, not by
fixing the call.

### "30/360" names three different rules

From 28 February 2021 to 31 August 2021:

| convention | days | year fraction |
| --- | --- | --- |
| ACT/360 | 184 | 0.511111 |
| ACT/365F | 184 | 0.504110 |
| ACT/ACT ISDA | 184 | 0.504110 |
| 30/360 Bond Basis (ISDA 4.16(f)) | 183 | 0.508333 |
| 30E/360 Eurobond (ISDA 4.16(g)) | 182 | 0.505556 |
| 30/360 US (SIA) | 180 | 0.500000 |

Six answers for the same six months, three of them from conventions all written
"30/360" on a screen. They are separate members of the enumeration rather than
one convention with a flag, because a flag invites a default and there is no
right one.

The rules are transcribed in the order they are applied, which is where the
differences live. Bond Basis adjusts a second date of 31 only when the first is
already past the 29th, and has no end-of-February rule at all. The US rules
reach that state *through* the end-of-February adjustment, which is the entire
difference between them and is worth three days on a February-to-August accrual.
Eurobond rules never condition on the first date — an implementation that caps
both dates independently is the Eurobond rule wearing another name, whatever the
docstring says.

### ACT/ACT ISDA is not a single ratio

It splits the period at each year boundary and divides each part by the length
of *its own* year, so a period spanning a leap year gets both 365 and 366 as
denominators. Actual days over 365.25, or over the length of the start year, is
the shortcut; over 1 November 2019 to 1 February 2021 it is wrong by nearly a
full day of accrual, in a direction that depends on where in the calendar the
trade sits.

The property that catches a wrong implementation is additivity: splitting a
period and adding the parts must give the whole, and the single-denominator
version fails that as soon as the split crosses a year boundary.

### The cases where a plausible implementation is still wrong

Three rules in this library are indistinguishable from a simpler, wrong version
almost all of the time, which is exactly why they are pinned directly.

**Modified following.** It is plain following except when a roll would cross
into the next month, where it turns back instead. Over a whole year the two
agree on every date but a handful, so an implementation missing the exception is
correct for months and then is not. The test counts the disagreements and
asserts each one crosses a month boundary, rather than asserting there are none.

**End of month.** If the anchor is the last day of its month, every subsequent
date should be the last day of its month. The rule bites exactly where the
anchor is a month end that is *not* the 31st — a 31st already clamps onto each
month's end by itself — so an implementation can pass every January-anchored
test and fail every February one. Both cases are in the suite.

**Measuring months, not stepping them.** Dates are always computed from the
original anchor. Stepping a month at a time loses month ends permanently: 31
January gives 28 February, then 28 March, 28 April, and never recovers. The
test computes both and asserts they differ.

### Calendars compose by union

A trade settling in two centres needs both open, so the joint calendar holds
every holiday of each. Intersecting them — keeping only the days both are closed
— is the intuitive-sounding mistake and produces a calendar open on almost every
holiday either centre observes. The test uses two calendars that share no
holidays at all, so an intersection would be empty and could not pass by
accident.

A calendar with no business days at all is refused rather than searched, and a
run of closures longer than a fortnight is reported as a misconfiguration —
usually holidays loaded for the wrong year — instead of being looped on forever.

### Schedules are generated backwards, and say where the stub is

A bond maturing on 15 August pays on the 15th of February and August whatever
its issue date was, so the schedule is generated backwards from maturity and any
odd period lands at the front. Generating forwards from the effective date puts
the payments on the issue day of the month and leaves the stub next to
redemption, which is the one place it is least often intended. Both are
available; neither is implied.

A stub is a fact about the trade rather than a rounding error, so it is
classified and reported on the schedule rather than left for the reader to find
by comparing period lengths. A stub nobody noticed is a coupon that does not
match the counterparty's.

Unadjusted boundaries are kept alongside the adjusted ones, because accrual
under the thirty-day conventions runs on unadjusted dates while payment runs on
adjusted ones. A schedule that discards the unadjusted dates cannot compute its
own accruals afterwards.

### A rate without its compounding convention is not a rate

The same discount factor of 0.90 over five years is 2.2222% simple, 2.1296%
annual, 2.1184% semi-annual and 2.1072% continuous. Eleven and a half basis
points across the conventions, which is wider than the bid-offer on most of the
curve, and nothing in a bare float says which one it is. So a rate is paired
with its convention as a type rather than by naming, since a naming convention
is something a caller can get wrong silently.

Simple compounding is not annual with one period. It is `1 / (1 + r t)` rather
than a power, so the two agree at exactly one year and nowhere else — which is
the one point somebody checking an implementation is most likely to pick.

Conversions route through the discount factor rather than through a closed form
per pair. Six conventions make thirty ordered pairs, and thirty chances to
transpose a sign, against one shared path already tested in both directions.

### Interpolation is a choice, so the curve names it

*Log-linear on discount factors* is linear in `log P`, which makes the
instantaneous forward piecewise constant — flat within each pillar interval and
jumping at the pillars. Unrealistic as a picture of the market, extremely well
behaved as arithmetic: the forwards are positive whenever the discount factors
decrease, which is exactly the arbitrage-free condition.

*Linear on zero rates* is the one most people reach for, and it does something
its name does not advertise. Interpolating `z(t)` linearly makes `z(t) * t`
quadratic and the forward its derivative, so forwards are piecewise linear with
jumps at the pillars — and they can go negative from inputs containing no
arbitrage at all.

Two pillars show it. `z(1y) = 6%` and `z(2y) = 3.5%` give discount factors of
0.941765 and 0.932394, strictly decreasing, so the inputs are clean and the
average forward over the year is a positive 1%. Log-linear returns exactly that,
flat. Linear on zero rates returns a ramp from +3% down to **−1%**, crossing
zero around 1.7 years — a negative rate manufactured by the interpolation from
data that contained none. Both figures are in the test suite, and
`instantaneous_forward` makes them inspectable on any curve.

*Monotone convex*, the scheme of Hagan and West, declines the trade-off. It
interpolates the forward rate itself, under the constraint that it must average
back to each interval's discrete forward — so the pillars are repriced by an
identity rather than by luck, the forward curve is continuous across them, and
positivity is imposed rather than hoped for. On the same two pillars its forward
runs from **+2% at the one-year pillar to exactly 0% at the two-year pillar**,
straight down and never below: it moves across the interval, as the data says it
should, and stays where log-linear's flat 1% average says it can. The 0% at the
far end is the positivity clamp doing its work — the unconstrained node forward
there extrapolates to −0.25%.

The construction is worth a sentence, because the interesting part is not the
quadratic. Write the forward on an interval as the discrete forward plus a
deviation `G(x)`, pinned at each end by the node forward there. Repricing both
pillars is then exactly `∫₀¹ G = 0`, and the natural choice is the quadratic
through both endpoints. But a quadratic with a fixed integral has no freedom
left to stay inside its own endpoints, and for endpoint pairs far enough apart
it overshoots — which in a rising curve is a dip and in a falling one can be a
negative forward. The answer is to hold `G` flat over part of the interval and
curve it over the rest, with the split point chosen to keep the integral at
zero. Which of the four shapes applies depends on nothing but the ratio of the
two endpoint deviations, with the regions meeting at −2 and −½, and
`tenor.monotone.classify` is written that way rather than as the eight
sign-and-magnitude conditions the method is usually stated with.

What it costs: pillars implying a negative discrete forward are refused rather
than interpolated. The positivity clamp bounds each node into `[0, 2·min(f)]`
over its neighbouring discrete forwards, and that range is empty once one of
them is negative. Those inputs contain an arbitrage — discount factors that rise
over an interval — so the honest answer is the error rather than a confident
number from a clamp that no longer means anything.

### A sequential bootstrap is wrong for a smooth interpolation

Each instrument pins down the discount factor at its own maturity given every
shorter one, so the obvious method is to solve them one at a time in order. That
is right for a local interpolation and wrong for any scheme whose shape at one
maturity depends on its neighbours — monotone convex, any spline. Adding pillar
`i + 1` changes the curve *before* `t_i` too, and the instruments already solved
stop repricing.

Nothing raises. No discount factor looks unreasonable. The curve is out by
around ten basis points in the middle of an interval, which is the size of the
thing being measured. On the test strip, one sequential pass under monotone
convex leaves the September future mispriced by 2.6e-05 of present value on unit
notional — and leaves both local schemes exactly right, which is how the bug
survives.

So the bootstrapper sweeps: the sequential pass, then repeated passes re-solving
every pillar against the shape the later ones imposed, until all instruments
reprice at once. It reports how many it took — one for the local schemes, seven
for monotone convex — because that number describes how much work the curve
took and a strip that suddenly needs many more is saying something.

The unknown solved for is the discount factor itself rather than a rate
parametrising it, which buys an exact bracket: a pillar's arbitrage-free range
is its two neighbouring discount factors. Solving in rate space means guessing a
range and widening it, and widening the wrong way builds a trial curve that
monotone convex then refuses — an exception to catch in place of a bracket that
was already known.

### Futures are not forwards

A futures contract is margined daily and a forward settles once, so the holder
receives cash exactly when rates rise and reinvests it at the higher rate. The
futures rate therefore sits *above* the forward rate it stands in for, and the
gap grows with the product of the two maturities.

The number is bigger than people expect. On 1% normal volatility a three-month
contract starting in one year is adjusted by 0.6 basis points; the same contract
starting in ten years is adjusted by **51**. `Future` takes the adjustment
explicitly and defaults it to zero rather than computing one, because computing
it needs a volatility and defaulting would be asserting one. `ho_lee_convexity`
provides it when a volatility is available, and reproduces Hull's worked
example — 47.5 basis points at 1.2% and eight years — in the test suite.

### DV01 and PV01 are not the same number, for two reasons

One moves the bond's own yield by a basis point; the other moves the whole
curve. They are routinely treated as interchangeable, which is harmless right
up until one is used to hedge the other.

The reason usually given is shape: a yield is a weighted average of the curve
over the bond's flows, so shifting every zero rate by a basis point moves that
average by a basis point only if the curve is flat. On the test curve, for a
fifteen-year bullet, that is worth **+1.01%**.

The reason not usually given is that a basis point is not one quantity. A basis
point of *continuously compounded* zero rate moves the equivalent semi-annual
rate by `exp(z/2)` times as much — 1.5% at a 3% level. Against a semi-annual
yield that puts the two numbers **−1.54%** apart on a *perfectly flat* curve,
where the shape explanation predicts no gap at all.

And the two have opposite signs. Make both mistakes at once — a continuous
shift measured against a semi-annual yield, on a sloped curve — and the answer
comes out **0.23%** apart, closer than either error alone would put it. The
discrepancy looks most negligible exactly where it is least understood.

`shifted()` and `dv01()` therefore take the compounding as an argument. Match it
to the bond and only shape and convexity remain; leave it continuous and what is
left is not the curve's shape, however much it looks like it. All four numbers
are in the test suite.

### Accrued interest is a fraction of one of two things

Either the day count fraction since the last coupon, or the fraction of the
coupon period. Under 30/360 they agree exactly, because a semi-annual period is
half a year by construction — which is why the disagreement survives, since the
case people test is the one where it cannot appear.

Under ACT/365F they differ by twice the period's actual length over 365. Three
months into a 181-day period on a 5% coupon, that is just over a cent per 100,
about 0.8% of accrued. `Accrual` names both; neither is a default that happens
silently.

### The total risk is the market's; the hedge is the curve builder's

There are two ways to break a duration down and they are not one set of numbers
regrouped.

*Key rate durations* shift the zero curve in a shape localised around one
bucket. They answer where the exposure sits. Their shapes are built to add to
one at every pillar, so they sum back to the total duration — an identity, and
one the tests check by measuring the residual at three bump sizes and asserting
it falls as the square of the bump, rather than by accepting a small number
once.

*Instrument risk* shifts a quote, rebuilds the curve and measures that. It
answers what to trade. A ten-year swap quote does not move the ten-year zero
rate alone; it moves every zero rate out to ten years, because the quote is a
statement about a whole annuity.

The difference is not academic. For the same bond on the same quotes, across
the three interpolations:

| | total | 5y swap | 10y swap | 20y swap |
|---|---|---|---|---|
| log-linear | 0.1509 | +0.0039 | +0.0534 | +0.0926 |
| linear on zeros | 0.1529 | +0.0041 | +0.0762 | +0.0715 |
| monotone convex | 0.1516 | **−0.0066** | +0.0878 | +0.0694 |

The totals span 1.3%. The ten-year entry moves by 64%, the twenty-year by −25%,
and the five-year changes sign. The total is what the quotes determine; the
split between them is what the interpolation decided, which is an assumption
about the gaps between quotes rather than anything the market said.

The negative entry is a real answer, not an artefact. Raising the five-year par
swap rate while the longer par rates are held fixed forces the forwards beyond
five years *down*, and for a bond whose risk sits past ten years that fall can
outweigh the rise nearer in. It happens under every scheme here; under monotone
convex the long-end effect is about three times larger, and that is enough to
flip the sign.

### A bond with no spread has an I-spread anyway

Z-spread adds a constant to every zero rate until the whole curve reproduces the
bond's price. I-spread subtracts one par swap rate from one yield.

The difference is clearest on a bond that has no spread at all — priced exactly
on the curve. Z-spread answers zero, by construction. I-spread, on the test
curve, answers:

| maturity | I-spread | Z-spread |
|---|---|---|
| 2 years | −0.14bp | 0 |
| 5 years | −0.46bp | 0 |
| 10 years | −1.62bp | 0 |
| 15 years | −6.03bp | 0 |

None of that is spread. It is a single benchmark point at the maturity being
compared against coupons discounted across the whole curve, and it grows with
maturity and with distance from par. On a flat curve every figure is zero to
two hundredths of a basis point, which is why a flat curve tests nothing here.

### Z-spread minus OAS is what the option costs

A callable bond's cash flows are not known: whether the issuer redeems depends
on where rates are when the call date arrives. One discounting curve cannot say,
because there is only one of it. So the curve becomes a Black-Derman-Toy tree,
calibrated so that it reprices the curve's own zero coupon bonds to 1e-16 —
exactly, because everything measured on it is quoted in basis points and a
calibration error is indistinguishable from a spread.

Then one price, two questions. What spread reproduces it if the option is
ignored? That is the Z-spread. What spread reproduces it if the option is
modelled? That is the OAS. The difference is the option. For a 5% bond callable
at par from year five, on the test curve at 15% volatility, it is **89 basis
points** — which is what the holder is being paid for being short the call, and
what a Z-spread would have counted as income.

A put runs the other way and its option cost is negative, because the holder is
long it. Implementing one and negating it for the other produces a number that
still looks like a spread, so both directions are tested, along with the two
cases where an option never binds and the bullet must come back exactly.

### Extrapolation is refused

Past the last pillar there is no information. Returning the last discount factor
is not a neutral default — it asserts that every forward rate beyond the curve
is zero, which is a strong opinion stated silently. A caller who wants a flat
extension can add a pillar and say so.

### Negative periods are refused

A year fraction with the end before the start is a mistake in the caller —
almost always a schedule generated backwards — not a negative quantity. Returned
as a number it travels a long way before anything notices, so it is refused at
the point it is asked for.

## Default risk, and what the credit triangle is actually worth

Everything above discounts a cashflow that arrives. `tenor.credit` prices the
possibility that it does not: a survival curve held as a piecewise-constant
forward hazard rate, credit default swaps with both legs, a bootstrap from par
spreads, and bonds discounted for default with recovery on face.

```bash
tenor credit quotes.txt --reference 2026-06-15 --basis ACT_365F \
  --spreads spreads.txt --bond-maturity 2031-06-15 --bond-coupon 0.05
```

### The integrals are closed form

A protection leg is `(1 - R) integral DF(s) dQ(s)`, and the temptation is to
put a fine grid under it. There is no need. On any interval where the hazard
rate and the instantaneous forward rate are both constant,
`DF(s) S(s) = DF(a) S(a) exp(-(r + h)(s - a))` and the integral is elementary.
Take the grid to be the union of the discount curve's pillars, the hazard
pillars and the coupon dates and both are constant on every piece, so the
answer is exact for the curve as interpolated rather than convergent to it.
The accrual-on-default term — the same integral weighted by time since the
last coupon — is elementary for the same reason and is computed, not
approximated by half a period.

### The credit triangle is exact, and then it is not

`spread = hazard x (1 - recovery)` is the rule everybody uses, and the first
surprise is how good it is. **With accrual on default and a zero interest
rate it is not an approximation at all: it is exact**, for any hazard rate and
any premium frequency. Integrating the accrual term by parts turns each
period's contribution into the integral of the survival probability over that
period; the discrete premium dates cancel completely and the annuity collapses
to `(1 - S(T)) / h`, which divides the protection leg to give exactly
`h (1 - R)`.

So the error is not about the credit. Measured, it is two effects of the same
shape, both linear in the length of a premium period:

| | relative effect | measured, quarterly at r = 3% |
|---|---|---|
| discounting lifts the true spread above the triangle | `+ r x period / 2` | +0.374% (predicted 0.375%) |
| dropping accrual on default lifts it further | `+ h x period / 2` | +1.257% at h = 10% (predicted 1.250%) |

Both hold across annual, semi-annual and quarterly premiums and across rates
from 1% to 10%, to within a few per cent of themselves — the residue is second
order. The first says something unintuitive: **the triangle's error depends on
the interest rate and the premium frequency and hardly at all on the hazard
rate**, so it is no worse on a distressed name than on an investment grade one.

Day counts add a third, larger effect that is nothing to do with either. Market
premiums accrue on actual/360 while a curve counts actual/365, so a premium
year is 365/360 of a hazard year and the fair spread is lower in the same
proportion: 1.389%, against which the discounting adds back 0.375%, leaving the
triangle high by 1.01%. The suite asserts that composition rather than the
total.

### Two defects the tests found

The recovery leg on a bond was not scaled by face. Recovery is a fraction of
the redemption amount, which is 100 in the convention this package prices in
and not 1 — so the term was two orders of magnitude too small while everything
else was very nearly right, which is the sort of error that survives a smoke
test.

And the bootstrap did not reprice its own inputs. A pillar placed at a quote's
maturity leaves that quote's final premium — paid on the *rolled* maturity, a
day or three later — discounted at the next quote's hazard rate. The error is
about three parts in a hundred million, which reads exactly like solver noise.
Pillars now sit at the last date their quote touches, and the repricing is
exact to 1e-13 of a basis point.

### One thing the command line will not let you misread

The triangle produces a *flat* hazard rate to a maturity, so the quantity to
compare it against is the average hazard to that date and not the forward rate
over the last bucket. On an upward-sloping curve those diverge: at ten years
the forward is half as much again as the average, and a reader seeing the
forward beside the triangle would conclude the rule of thumb is wildly wrong
when it is out by nine per cent. Both are printed, labelled.

## Deliverable bond futures, and why the switch is at six per cent

A short-term interest rate future here is a *rate* contract: one accrual period,
cash settled, nothing for either side to choose. A deliverable bond future is a
different instrument. The short hands over an actual bond from a published
basket, picks which one and picks when inside the delivery month, and the
contract normalises those choices with a **conversion factor** — the price of
that bond, per unit of face, at a notional yield the exchange fixes and the
market then moves away from.

```
$ tenor futures examples/basket.txt --settlement 2026-11-20 \
      --first-delivery 2026-12-01 --delivery 2026-12-31 \
      --price 118.75 --repo 0.042 --switch 0.045 0.06 0.075
deliverables:
  bond=1.750% of 2046  conversion_factor=0.5153  gross_basis=2.590125  carry=-0.1123241831  net_basis=2.702449183  implied_repo=-0.3273588363  breakeven_futures_price=123.9944191
  bond=3.000% of 2046  conversion_factor=0.6555  gross_basis=1.954375  carry=-0.04210528223  net_basis=1.996480282  implied_repo=-0.177574815  breakeven_futures_price=121.7957365
  bond=4.500% of 2047  conversion_factor=0.8266  gross_basis=1.17825  carry=0.02052300308  net_basis=1.157726997  implied_repo=-0.05912505087  breakeven_futures_price=120.1505892
  bond=6.250% of 2047  conversion_factor=1.029  gross_basis=0.30325  carry=0.121516019  net_basis=0.181733981  implied_repo=0.02898262597  breakeven_futures_price=118.9266122
(the invoice price, accrued interest, financed balance and interim coupon columns
are omitted here; the command prints them, and `--json` is the machine form)
cheapest_to_deliver:
  by_implied_repo: 6.250% of 2047
  by_net_basis: 6.250% of 2047
  by_futures_price: 6.250% of 2047
  unanimous: True
  implied_futures_price: 118.9266122
  quote_less_implied: -0.1766122265
```

It is the only subcommand that takes no curve. The whole calculation is the
bond's quoted price, its conversion factor and a money-market financing rate,
and asking for a file of swap quotes to compute it would be asking for something
it does not read.

### The conversion factor is a bond price, so it is checked by pricing a bond

The exchange formula is five lines of algebra, and a formula transcribed once and
asserted against itself passes whatever it happens to say. So the suite reaches
the same number a second way: it builds the *rounded* bond with this package's
own schedule generator, prices it at the notional yield, and compares. Eighty
cases, all four stub lengths, coupons from zero to 12.5% — worst disagreement
1.8e-15.

Two facts worth having as tests rather than as comments came out of it. A bond
paying exactly the notional coupon with a whole number of years to run has a
factor of exactly one, which is the check the formula's shape is easiest to fail.
And at exactly the notional coupon the discount factor to the rounded maturity
cancels out of the formula, so the factor stops depending on the maturity at all
— a six per cent bond with a three-month stub has the same factor at three years
as at twenty-nine, and 0.999889 rather than 1.0, which is the rounding convention
showing through and not an error. A basis point off the notional coupon, the
maturity matters again and matters strongly.

### Net basis and the implied repo rate are one number written twice

Exactly:

```
net basis = (repo - implied repo) * financed balance
```

where the financed balance is the dirty price over the holding period less each
interim coupon over its own stub. Both sides are computed by different routes —
one adds up income and financing, the other divides a break-even — and they agree
to 1e-12, which is a statement about the definitions rather than about floating
point. The same identity through the third convention,
`net basis = (break-even futures price - quote) * conversion factor`, holds to
the same precision.

The implied repo rate is not solved for. The cash-and-carry break-even is linear
in the financing rate, which appears once in the cost of carrying the bond and
once in the reinvestment of any coupon falling inside the holding period, both as
simple interest. So it divides. The suite checks the rearrangement against a
bisection on the cash flows themselves, because a rearrangement is exactly the
kind of step that can be done backwards without any test noticing.

### The criterion that disagrees is not the one you would expect

Three conventions name the cheapest bond: the highest implied repo rate, the
lowest net basis, and the lowest futures price a bond can justify. All three rank
on the same gap between a bond's break-even price and the quote, scaled by one,
by the conversion factor, and by the factor over the financed balance.

The natural expectation is that the rate and the price part company, because a
deep-discount long bond finances around half the balance of a high-coupon short
one. Measured on the four-bond basket above, that is backwards. The conversion
factors span a ratio of **2.00**; the factor *over* the balance spans **4.65%**,
because the balance is nearly the dirty price and the dirty price is nearly the
factor times the futures price, so the factor cancels. The rate and the price
rank a basket alike.

It is the **net basis** that is on its own scale — points per 100 of *the bond's*
face rather than per contract. Walking the quote down from the basket-implied
price, it takes **5.00 points, or 4.20%,** before the net-basis ranking picks a
different bond from the other two, and by then every deliverable shows several
points of arbitrage. So comparing raw net bases across a basket does compare
quantities in different units, and gets away with it because a liquid contract
never trades five points from fair. Dividing the net basis by the conversion
factor recovers the price criterion exactly.

### Being close to a coupon date is a cost, not a benefit

Two bonds, same coupon, one paying inside the holding period and one not. The
one that pays looks better: a coupon in hand, reinvested to delivery. It carries
worse.

Coupon income over a fixed 41 days is the same 41 days of coupon whenever it
lands — 0.5613 against 0.5663 here, the gap being the day-count noise of
measuring 41 days against two different period lengths. What separates the two is
the balance financed, and the bond about to pay carries 2.158 of accrued interest
into the trade against the other's 0.912. Financing that 1.247 difference costs
0.00596; the received coupon earns 0.00467 back over its 16-day stub. The test
that found this asserted the opposite twice before it was a measurement.

### The switch is at the notional coupon, to six decimal places

The factors were computed at the notional coupon, so at that yield every bond's
price over its factor is the same number and the basket is exactly indifferent.
Away from it the ranking is by duration, because price over factor falls fastest
where there is most of it. So the cheapest bond is the shortest in the basket
below the notional coupon and the longest above it, and the exchange did not
choose where that happens — the notional coupon did.

Measured on a pair of notional-coupon bonds maturing a whole number of years from
the first delivery day, which makes the rounding convention a no-op and forces
both coupon schedules onto the same months, the crossing is at **6.000000%**, and
neither the repo rate nor rounding the factor to four decimals moves it. With
real odd maturities the rounding convention displaces it: 6.0011% with exact
factors and no carry, pulled back to 6.0002% once carry and the published factor
are included — which is a coincidence of that pair and is stated as one.

The switch is described and not valued. The margin column is what the short's
choice is worth *conditional* on arriving at a level, and a parallel walk says
nothing about the chance of arriving anywhere, so a number called the value of
the delivery option would need a model of how the curve moves. This package does
not have one and does not pretend to.

### A conversion factor below one raises the contract's risk

`futures_dv01` divides the deliverable's pv01 by the conversion factor, because
the invoice divides by it. So a bond with a factor of 0.5153 moves the futures
price nearly twice as far as it moves itself. Hedging a bond position one for one
with the contract, on the grounds that a factor below one is conservative,
under-hedges by that whole ratio.

### A bond that has redeemed cannot be delivered

The conversion factor only needs the bond to outlive the *first* day of the
delivery month, so a bond maturing on the 15th has a perfectly good factor and
cannot be handed over on the 31st. That bond is refused rather than priced — and
the guard is load-bearing, because `Bond` bundles the final coupon into the
redemption flow, so such a bond would contribute 102.5 to the coupons received
inside the holding period rather than 2.5. Nothing subtracts the principal back
off: a subtraction that can never fire reads as a case somebody handled. The test
sweeps every maturity for a month either side of the delivery month at a daily
step, looking for the counterexample.

## Two curves, and what the second one is actually worth

A floating leg forecast off the curve it is discounted on collapses to one
subtraction. Each period's projected growth is its own discount ratio, so the
sum telescopes, every intermediate date cancels, and `P(start) - P(maturity)`
prices the whole leg without computing a single forward rate. That is exact for
any curve of any shape, and it is how `Swap` values its floating leg.

It stopped describing the market in 2008. A three-month deposit rate and an
overnight-indexed rate had sat within a basis point or two of each other for a
decade; they separated and stayed separated, because the index a swap forecasts
carries term credit and funding risk and the rate a collateralised swap
discounts at does not. Once the two differ, nothing cancels and the leg has to
be valued coupon by coupon.

```
$ tenor multicurve examples/ois.txt --reference 2026-01-15 --basis ACT_365F \
      --forecast examples/forecast.txt --maturity 2036-01-15
reference: 2026-01-15
index: 3m index
forecast_curve:
  solving_for: 3m index
  quotes_used: 7
  short_leg_projected_on: itself
  sweeps: 1
  worst_repricing_error: 2.775557562e-16
  reprices: True
  ...
forward_basis:
  start=2026-04-15  end=2026-07-15  projected=0.03309  discounting=0.03063  basis_points=24.65
  start=2027-01-15  end=2027-04-15  projected=0.03540  discounting=0.03315  basis_points=22.43
  ...
swap:
  maturity: 2036-01-15
  par_rate: 0.0389
  par_rate_on_one_curve: 0.03648486931
  basis_points_from_separating: 24.15130694
  risk:
    from_the_discount_curve: -1.14551659e-06
    from_the_forecast_curve: 0.0008480843596
    from_both_together: 0.0008469389715
    cross_term: 1.285412055e-10
```

Two files go in. The first is the ordinary quote screen and builds the
discount curve; the second is quotes on the *index*, and a projection curve is
solved out of them with the discount curve held fixed. That is the order the
market works in, and it is the reason the bootstrap takes the discount curve as
an argument rather than returning a pair.

### The identity, kept as a test

Point both curves at the same object and the coupon-by-coupon leg has to
reproduce the subtraction. On the ten-year quarterly leg it does, to **7.2e-16**
of the leg's value — not bit for bit, since the two expressions accumulate
rounding in different orders, but to the last few bits, which is what says the
new path carries no error of its own.

What the identity needs turned out to be narrower than it looked, and the first
draft of this got it wrong. The cancellation is between one period's `P(end)`
and the next period's `P(start)`, so it needs the payment to land on the accrual
end. The rolling convention was the suspect and it is innocent: under modified
following a period ending on a Saturday rolls to the Monday, and the payment
rolls with it, so the dates still meet. A **payment lag** is the culprit, since
it moves the payment without moving the accrual it pays for. At the two business
days an overnight-indexed leg settles on, the gap is 0.095 basis points of par
rate; at five days, 0.262. The lag is therefore a field on the index, it
defaults to zero, and that zero is load-bearing.

### The discount curve is worth almost nothing on a par rate

Separating the curves is usually introduced as the thing that repriced swaps.
For a par rate it does not. Shifting the projection curve by 100 basis points
moves the ten-year par rate by 101.62 basis points. Shifting the *discount*
curve by the same amount moves it by **-0.12 basis points** — the other way, and
smaller by a factor of 843.

That is not a surprise once the par rate is written down as what it is: a
discount-weighted average of the forwards the leg projects. The projection curve
moves every term in the average; the discount curve only reweights them. On an
upward-sloping curve, heavier discounting tilts the weights towards the earlier
and lower forwards, so the average falls a little.

The split in the output above is the same statement in money: a par swap's
response to the discount curve is `-1.1e-06` against the forecast curve's
`8.5e-04`. The two do not add to the joint response, because a swap's value is
bilinear in the two curves; the cross term is `1.3e-10`, which is 1.4e-07 of the
joint response and grows with the square of the shift.

Where the discount curve *is* worth something is a swap that is not at par. A
ten-year swap struck 100 basis points away from the market is the rate
difference times the annuity, and the annuity is all discount curve: that same
shift moves its mark by **4.67%** of itself, against the 0.031% it moves the par
rate's level. Third-order for a new trade and first-order for a book of old
ones, which is the reverse of the order the choice is usually introduced in.

### The basis passes through at 1.0146, and that factorises exactly

Raise every projected forward by a flat 20 basis points and the ten-year par
swap rate rises by 20.29. The pass-through is not one, for two reasons that pull
opposite ways: the two legs' annuities stand in the ratio **1.0191** — actual/360
floating against 30/360 fixed — and a flat shift to a continuously compounded
curve lifts a simply compounded quarterly forward by **0.9956** of itself. Their
product is 1.014643 and the measured pass-through is 1.014643. They agree to
5.8e-15, which is what distinguishes a decomposition from an explanation that
happens to come out near the right size.

### A curve that reprices every quote is not the curve that produced them

The projection bootstrap fits its inputs to 1e-16 and settles in one sweep under
log-linear discounts, five under monotone convex. Feed it par rates generated
off a known curve and ask how close the result is to that curve: beyond a year,
within 0.03 basis points of zero rate. Between the three-month forward quote and
the one-year swap, **5.30 basis points** out at 0.49 years. Nothing in that gap
is pinned by an input, so the interpolation decides it, and an exact fit is a
statement about the quotes rather than about the curve.

The same hole has a sharper cost in a basis curve. Bootstrapping a six-month
projection curve from tenor basis spreads of 5 to 9 basis points, with nothing
quoted inside the first year, leaves an implied six-month-over-three-month
forward basis of **16.87 basis points** over the first period — the one-year
quote spread backwards by the interpolation, and three times the number it came
from. One forward quote on the long index puts the first period at exactly the
5.00 it says.

```
$ tenor multicurve examples/ois.txt --reference 2026-01-15 --basis ACT_365F \
      --forecast examples/basis.txt --flat-frequency SEMI_ANNUAL
```

A basis file solves the *longer* tenor's curve with the shorter one taken as
known, so a file mixing par swap rates with basis spreads is asking one solve
for two curves and is refused rather than resolved. And since the only curve
this command has been handed is the discount curve, the short leg is projected
off that — an assumption rather than a quote, so it is named in the output
rather than left to be inferred from a number that is partly a proxy.

## Swaptions, and where the discount curve finally matters

The section above measured a factor of 843 between what the projection curve
and the discount curve are worth to a par swap rate. The obvious next question
is where the discount curve *does* matter, and it has a structural answer:
anything whose value is an annuity times something. A swaption is the first such
instrument.

```
$ tenor swaption examples/ois-long.txt --reference 2026-01-15 --basis ACT_365F \
      --forecast examples/forecast-long.txt --expiry 2036-01-15 \
      --swap-maturity 2046-01-15 --volatility 0.20
forward_swap_rate: 0.04037356175
strike: 0.04037356175
annuity: 5.739934379
value: 0.05752663246
discount_shift:
  value: 0.04971483016
  relative_change_in_value: -0.1357945348
  relative_change_in_forward: 0.001240501324
  relative_change_in_annuity: -0.1384831056
strip:
  value: 0.06696852659
  over_the_option: 0.1641308335
  periods: 40
implied_volatility_from_its_own_premium: 0.2
```

Read the three numbers under `discount_shift`. A hundred basis points on the
discount curve moves the forward swap rate by 0.124% of its level and the
option by **-13.58%** — a hundred and nine times as much — and the decomposition
says why: the annuity falls 13.85% and the rate's rise claws 0.12% back. Under
the annuity measure the forward swap rate is a martingale and the swaption is a
European call on it, so the discount curve has been factored into the numeraire.
That is precisely why it comes back multiplicatively.

### Both conventions, and how far apart they are

A negative forward swap rate has no lognormal volatility, so `--normal` reads
the volatility as a Bachelier one in absolute units of the rate. The two are not
conversions of each other. At the money both give `numeraire * vol * sqrt(2T/pi)`
to leading order, so the product of a lognormal volatility and the forward is
the right first guess for the normal one — and it is 39.78 basis points against
the product's 39.80 at a 10% volatility over one year, 4.2e-04 apart; 4.2e-03
apart over ten years; and **1.6e-02** apart at a 20% volatility over ten years.
The gap grows with `σ√T`, which is worth knowing before converting a quote at a
long expiry.

### A strip of options is not an option on the strip

The `strip` block prices the cap of the same strike and tenor beside the
swaption. The cap's holder chooses period by period, so it is worth more: 16.4%
more on the structure above, and **78.1% more** on the five-year quarterly
structure in the tests at a 20% volatility. The premium *shrinks* as volatility
rises — 82.8% at 10% against 74.8% at 40% — which is the reverse of the natural
guess.

That inequality is Jensen's, and it is the sharpest check available that both
prices are built on the same curve: an error in either breaks it. Over
twenty-five strikes and volatilities it holds twenty-four times and fails once,
by 6.98e-05, and the failure turned out to be the most informative part of the
phase.

It is not a pricing error. The fixed leg's annuity accrues on unadjusted
boundaries under 30/360; the caplet weights accrue on adjusted ones under
actual/360. So the two numeraires differ by 4.655e-05 relative, and the forward
swap rate and the weight-average forward differ by 4.654e-05 — the same number.
Deep in the money at a low volatility both prices collapse to their intrinsics,
the ratio collapses to those two mismatches, and `(1 + 4.655e-05)(1 - 1.1635e-04)`
is 0.99993019 against a measured 0.99993019. The inequality is exact in the
mathematics and violated in the conventions, by precisely the amount the
conventions differ.

## One parameter set, and the identities that check it

`tenor.options` prices a swaption on the annuity measure and takes the
volatility as an input. `tenor.lattice` prices a callable bond on a tree and
says nothing about a swaption. `tenor.hullwhite` is the smallest thing that
prices both from one specification:

```python
from tenor import HullWhite, swaption_price, reprices_curve

model = HullWhite(a=0.08, sigma=0.009)
reprices_curve(curve, model)        # 0.0 -- an identity, not a fit
trade = swaption_price(curve, model, option, reference)
trade.jamshidian, trade.quadrature  # two routes
trade.gap                           # -8.9e-16 between them
```

The short rate is Gaussian with its drift fitted to the initial curve through
`A(t,T)`, so the model reprices every discount factor it was given *by
construction*. Measured across a seventeen-pillar humped curve at every mean
reversion from 0.01 to 0.5 and every volatility to 2%, the worst relative error
is exactly **0.0**.

### The forward-measure mean is implied, and comes out exact

Pricing by quadrature needs the law of `r(T)` under the `T`-forward measure.
Rather than copy a drift adjustment, `forward_measure` implies it from the fact
that every forward bond price is a martingale under that measure — which
*over-determines* it, since each probe maturity gives its own answer. The
answers agree to **0.0**, because the terms cancel identically: the `B²v/2`
from the lognormal expectation is exactly the convexity term inside `A`,
leaving `E^T[r(T)] = f(0,T)`. The forward-measure expectation of the future
short rate is the instantaneous forward, with nothing left over.

### The obvious quadrature is wrong by twelve per cent

A Gaussian weight invites Gauss-Hermite, and a 64-point rule got the
near-the-money prices to five digits and a deep out-of-the-money one wrong by
**1.25e-01 relative**. A Gauss rule converges at that rate on a kink and an
option payoff is nothing but a kink. The symptom was misleading: both routes
agreed on the payer-less-receiver *difference* to 1e-12 while both were wrong,
because the error is in the straddle and cancels in parity, so every identity
still held and only the level was out. Putting the exercise boundary at a panel
edge of a composite Gauss-Legendre rule turned five digits into fifteen, and
the two routes now agree to **3.1e-14** over eighteen combinations of expiry,
coupon and side.

### Mean reversion is an argument, because one quote cannot choose

A ten-into-ten payer at its forward is repriced exactly at every reversion from
0.01 to 0.30: the refitted volatility runs 0.492% to 3.226%, a factor of
**6.55**, with a refit error of at most 2.2e-16 at every point. A routine
fitting both parameters to one quote would return whichever point on that ridge
its guess fell nearest and report a residual of zero either way. What separates
them is the term structure — the pair fitted to the ten-year expiry prices a
one-year-into-ten-year from **23.4% below to 57.2% above** the `a = 0.08`
answer:

```bash
tenor hullwhite ois.txt --reference 2026-01-15 --basis ACT_365F \
  --expiry 2036-01-15 --swap-maturity 2046-01-15 \
  --mean-reversion 0.08 --volatility 0.009 --ridge 0.02 0.08 0.3
```

And the model's native convention is the normal one, measurably: its own prices
imply a normal volatility moving 1.06 basis points on a level of 45.55 across
strikes from 60% to 140% of the forward — 2.33% wide, and monotone rather than
curved — against the lognormal convention's 41.91% on the same prices.

## A second factor, because one cannot decorrelate two rates

The model above is a real one and has a limitation calibration cannot touch.
With a single factor every bond price is affine in the same scalar, so two
maturities are correlated **exactly one** at every parameter setting: the short
end and the long end move together, always. `r = x + y + phi(t)` with two
speeds and a correlation between the drivers removes that and keeps every
closed form.

The exact fit is better than the one-factor model's, for the same reason. Both
factors start at zero and the three variance terms telescope at `t = 0`, so the
curve reprices with an error of identically `0.0`.

**Equal mean reversions collapse it to Hull-White exactly**, which is why they
are allowed rather than refused: with both speeds equal the bond price depends
on `x + y`, itself an Ornstein-Uhlenbeck process with volatility
`sqrt(sigma^2 + eta^2 + 2 rho sigma eta)`. Measured against `tenor.hullwhite`,
the option volatility agrees to 1e-16, the option price to 1e-15, the bond
price to 1e-14 and the implied correlation to 1.0 within 1e-12. That check
crosses modules, and it earned its keep immediately by catching the bond
option's call/put convention pointing the opposite way to the one-factor
module's — it disagreed by exactly the parity amount at every parameter
setting.

What the second factor buys: at a 0.50 fast speed, a 0.05 slow speed and a
driver correlation of -0.9, the one-year and ten-year zero rates come out
correlated **0.3592** at a one-year horizon. The work is done by the gap
between the speeds, and it closes as they meet — 0.3592, 0.3387, 0.4255,
0.8862, 0.9999, then exactly 1.0000 — with a shallow minimum on the way that is
worth knowing before fitting the pair.

**A cap cannot identify any of it.** A cap is a strip of options on single
bonds, so its price depends only on each bond's own volatility and never on the
joint law of two maturities. Refitting the fast volatility to hold a one-year
to ten-year cap at 3.8% fixed, seven driver correlations from -0.9 to 0.9 give
cap prices equal to within 2.4e-15 relative while the one-year-to-ten-year
correlation runs from 0.0674 to 0.9965 — seven calibrations a cap market cannot
tell apart.

That refit is also where this phase's bug was. A cap price is not monotone in
the fast volatility: at a negative driver correlation the bond option variance
is a parabola in `sigma` with its minimum inside the domain, so raising
`sigma` from nothing makes the cap cheaper. A bisection from a floor to a
ceiling returned a price a third away from the target while reporting success.
`fit_fast_volatility` searches for its bracket on a grid, takes the crossing
from below, and raises `NoRoot` naming the cheapest attainable price when there
is none.

```bash
tenor g2 ois.txt --reference 2026-01-15 --basis ACT_365F \
  --start 2027-01-15 --end 2036-01-15 --frequency SEMI_ANNUAL --strike 0.035 \
  --mean-reversion 0.50 --volatility 0.011 \
  --slow-mean-reversion 0.05 --slow-volatility 0.007 --correlation 0.0 --identify
```

## A swap rate paid outside the measure it is a martingale under

Every option above is priced by making the forward swap rate a martingale under
the fixed leg's annuity, which is the measure a swaption settles against. A
constant maturity swap leaves it: the rate is observed once and paid *once*, on
a single date, so the expectation wanted is under that date's forward measure,
and the rate is not a martingale there. The forward read off the curve is a
biased forecast of what the contract pays.

`tenor.cms` values one by static replication. The change of measure carries
`alpha(S) = P(T, Tp) / A(T)` — a function of the rate, not a constant — and
everything above that is the Carr-Madan expansion of `g alpha` around the
forward and a quadrature over the swaptions already in the package. So a CMS is
valued off the surface a swaption book is marked on rather than off a separate
model.

```python
from datetime import date
from tenor import ConstantMaturity, Frequency

fixing = ConstantMaturity(
    expiry=date(2031, 1, 15),      # when the rate fixes
    effective=date(2031, 1, 15),
    maturity=date(2041, 1, 15),    # a ten-year rate
    payment=date(2031, 7, 15),     # paid six months after it fixes
    frequency=Frequency.ANNUAL,
)
swaplet = fixing.swaplet(discount, projection, 0.24)
swaplet.forward         # 0.041053  the forward swap rate
swaplet.rate            # 0.043659  what it pays, in expectation
swaplet.basis_points    # 26.06     the convexity adjustment
```

### A single period reduces exactly, which is the check that carries the file

If the underlying swap has one fixed period and the CMS pays at the end of it,
then `A(T) = delta P(T, Tp)` identically, so `alpha` is `1 / delta` — flat in
the rate, every derivative of it zero, and the adjustment **identically 0.0**.
It has to be: that contract is a forward rate paid in arrears and needs no
adjustment, as anyone can say without a model. The replication is then the
point mass at the kink and nothing else, and it matches `tenor.options` to
**5e-16** across four strikes and both payoffs. It is the only check here that
would catch the point mass carrying the wrong sign — the error that cost the
two-factor model a call and a put.

The quadrature itself is checked against arithmetic from outside the package.
Against that same flat mapping, replicating `S**2` must return
`F**2 exp(sigma**2 T)` under Black and `F**2 + sigma**2 T` under Bachelier. Both
come back at rounding.

### Widening the ceiling made it worse, by eight orders of magnitude

The second moment is the one functional here sensitive to the far tail, and it
is how the panel layout was found out. The lognormal ceiling is
`F exp(w sigma sqrt(T))`, so it moves out exponentially in the width while the
panel count stays put: every extra standard deviation coarsens the mesh *at the
forward*, where the integrand lives. With uniform panels the relative error ran
1.4e-09, 3.2e-08, 7.9e-04 and 1.8e-01 as the ceiling went from six standard
deviations to twenty — monotonically worse for covering more of the
distribution. Spacing the edges evenly in `log K` holds every ceiling from
eight to thirty at rounding, and sixteen panels then reach 5.1e-13.

### The flat-curve model is arbitrageable twice, and both are measured

The mapping is evaluated by assuming the curve is flat at whatever level the
rate fixes at, which makes `P(T, Tp)` and `A(T)` explicit powers of `1 + S/q`.
Raw, it returns the flat curve's ratio at the forward rather than the real one,
and prices a zero-coupon bond **0.88% wrong** on the upward-sloping curve in the
tests — a quarter of the adjustment it exists to compute. `annuity_map` scales
that away at the forward, and the constant payoff is then *still* worth 0.217%
more than the bond it is, because matching at a point does not match an
expectation. The replication divides by the replicated unit payoff, which
enforces the one no-arbitrage condition available and moves the adjustment by
**3.6%**. It is a no-op wherever the mapping is constant or the volatility is
zero, so no reduction above is disturbed, and `measure_error` reports the gap
rather than hiding it.

### Paying later unwinds the convexity rather than compounding it

This came out against the guess. Moving the payment of that fixing from the
fixing date out to ten years past it takes the adjustment **29.28, 26.06,
22.92, 16.83, 5.43, 0.07 and -23.86 basis points** — monotonically down, and
through zero. The sign is the sign of `alpha'(F)`, and `alpha'(F)` vanishes
where the payment date meets the annuity's own annuity-weighted mean payment
time: 5.1872 years here, against a slope that crosses zero at 62 months, which
is 5.1667. Structural rather than coincidental — `alpha` is one discount factor
over a weighted sum of them, so putting the payment at the weighted mean makes
numerator and denominator respond to the rate alike.

### It is a smile instrument, and an upward skew has no answer at all

The adjustment is an integral of swaption prices across every strike, so it
reads the whole surface and not the at-the-money point. Against surfaces that
leave every at-the-money swaption worth exactly what the flat one does, it comes
to 100.0%, 96.9%, 94.3%, 92.1% and 88.5% of the flat number at skews of 0,
-0.10, -0.20, -0.30 and -0.50 per unit of rate. An 8% error from a marking
choice no at-the-money quote can see.

A surface sloping *up* in strike is worse than inaccurate. A volatility rising
without bound stops the swaption price decaying, so the integral diverges and
whatever comes back is a function of the ceiling: the same fixing against a
+0.30 skew reads 35.6 basis points at a six standard deviation ceiling and
36565 at fourteen. Every one of those is wrong and none looks it, so
`Replication.truncation` measures the share of the value beyond the ceiling —
5.5e-24 flat, exactly 0.0 against a downward skew, 120% here — and the
quadrature's tolerance refuses it rather than returning a number whose only
real input was the ceiling.

```bash
tenor cms ois.txt --reference 2026-01-15 --basis ACT_365F --forecast forward.txt \
  --expiry 2031-01-15 --tenor-years 10 --delay-months 6 \
  --fixed-frequency ANNUAL --volatility 0.24 \
  --delays 0 6 24 60 120 --skews 0.0 -0.3 0.3
```

## Several exercise dates, and no formula for any of them

Every option above has one exercise date, which is exactly what lets
`tenor.hullwhite` price it twice in closed form: the payoff is a function of the
short rate at that date, and both routes integrate it against that rate's
density. A Bermudan swaption breaks the structure rather than complicating it.
The value of not exercising today is the value of still holding the right
tomorrow, so the quantity being integrated is the answer to the same problem one
date later, and there is nothing to write down.

`tenor.bermudan` runs the recursion on a Hull-White trinomial tree built from the
same `(a, sigma)` the Europeans use, fitted to the curve by forward induction.
Exercise dates may be given as the unadjusted anniversary or as the business day
it rolls to.

```python
from datetime import date
from tenor import Basis, HullWhite, flat_curve
from tenor.bermudan import BermudanSwaption, bermudan_swaption

reference = date(2026, 1, 15)
curve = flat_curve(reference, date(2046, 1, 15), 0.03, basis=Basis.ACT_365F)

option = BermudanSwaption(
    expiries=tuple(date(year, 1, 15) for year in range(2029, 2036)),
    maturity=date(2036, 1, 15),
    strike=0.03,
)
value = bermudan_swaption(curve, HullWhite(a=0.05, sigma=0.01), option, reference)

value.price           # 0.041895
value.best_european   # 0.032385 -- the best single date, priced in closed form
value.switch_value    # 0.009510 -- 29.4% of it, which is what the dates buy
```

### The switch value is the number, not the price

A Bermudan is worth at least the best of its co-terminal Europeans, because
exercising only on that one date is a strategy available to it. So the price on
its own does not say whether a desk is paying for optionality or for a European
with extra steps; the excess over that best single date does. Here the seven
annual dates are worth **29.4%** on top of the best of them.

### Rounding to the nearest node removes the cap on the tree

The textbook Hull-White tree branches up-mid-down from a fixed offset and needs a
cap on its width, because far from the centre the mean reversion pulls the
conditional mean past the neighbouring node and a probability turns negative.
Branching instead to the three nodes around each node's *own* conditional mean
bounds the residual drift by half a spacing, which keeps every probability
positive everywhere. Nothing is truncated, so total state-price mass at each step
is the discount factor to it rather than a share of it, and the width stops
growing on its own once the drift pulls back a full node.

### Exercise dates land on nodes, which equal steps cannot deliver

An exercise decision is a kink in the value function, and a time step that
straddles one smears it. Exercise dates are swap anniversaries whose day counts
differ by a day or three, so a single spacing is shared across steps of slightly
different length and each gap is split into whole sub-steps. Positivity then needs
the step variance between a quarter and three quarters of `dx**2` — the uniform
case sits at a third — and the bound is checked against the grid produced rather
than assumed from the request.

### Forward induction recovers the convexity term

Fitting one rate per step to reproduce the next zero looks like a statement about
the step's average instantaneous forward, which is a curve quantity with no model
in it. The fitted shift sits above that average by exactly the second term of
`phi(t)`: **8.9e-04** against an analytic 8.9e-04 at `a = 0.05, sigma = 0.01` and
five years, agreeing to 1.3e-05 relative at four steps a year and 9.0e-08 at
forty-eight, and identically on a flat, a sloped and a humped curve. The
discount-weighted spread of the state across a step *is* that convexity, so
fitting a discount factor finds it.

### What is exact, and what is only first order

Two things are identities and are tested as such. The grid's zeros reprice to
**2.2e-16**. And a model with no volatility has no decision to make, so the price
collapses to the best discounted intrinsic value computed off the curve alone — to
**5.9e-17** across twelve curve, payoff and mesh combinations.

The rest is first order in the step, and it is visible without going through a
price. State prices at a node, weighted by analytic prices of a bond that
outlives the grid, fall short of today's discount factor by **9.7e-05** at twelve
steps a year and halve at every doubling: a ratio of **2.00** each time, the same
on all three curves, always negative. `forward_bond_error` reports it, which is
why it is the diagnostic to read when choosing a step density.

### A single mesh does not measure convergence

The exercise boundary does not fall on a node and moves between nodes as the mesh
changes, so the error against the analytic European oscillates in sign and size.
At 48 steps a year this instrument came within **1.9e-06** of the analytic price
while the worst case over neighbouring meshes was **1.8e-05**, and a convergence
ratio read off that point came out at **-865**. Over bands the worst case runs
4.8e-05, 1.8e-05 and 6.5e-06 across 24-40, 48-64 and 96-112 steps a year.

### The boundary falls on a flat curve, and need not elsewhere

The exercise boundary — the short rate at which taking the swap stops being worse
than waiting — runs **3.92%, 3.80%, 3.68%, 3.54%, 3.40%, 3.21%, 2.95%** across the
seven dates above. A shorter remaining swap gives up less by being taken, so the
bar falls. That reasoning leaves out where the forwards are going: on a curve
rising 15 basis points a year the boundary reverses once, from 3.128% to 3.212%,
and on a humped curve it rises over four consecutive dates. Only the last date is
reliably the lowest bar.

### Payer-receiver parity is the identity to not reach for

A European payer less a European receiver is the forward swap exactly, since the
two payoffs are the positive and negative parts of one number. Two Bermudans are
maxima over different stopping rules, and a difference of maxima is not the
maximum of a difference: measured at 0.00212 against a swap worth 0.00136. Using
it as a check would report a 56% pricing error on correct prices.

```bash
tenor bermudan ois.txt --reference 2026-01-15 --basis ACT_365F \
  --expiry 2031-01-15 2032-01-15 2033-01-15 2034-01-15 \
  --swap-maturity 2036-01-15 --mean-reversion 0.05 --volatility 0.01
```

## Reading a quote convention back into a model input

`Cap.value` takes one volatility for the whole strip, which is how a cap is
*quoted* and not how anything is priced. `tenor.stripping` is the inverse: a
bootstrap in maturity where each longer cap's new periods get one volatility
from a root solve, while the earlier periods stay where previous steps put them.

```bash
tenor caplets ois.txt --reference 2026-01-15 --basis ACT_365F \
    --caps caps.txt --effective 2026-01-15 --strike 0.035 --basis-spread 0.0020
```

```
reference: 2026-01-15
strike: 0.035
buckets:
  maturity=2027-01-15  periods=3  quoted=0.18  repriced=0.18  caplet=0.18        sensitivity=1.000  indeterminacy=9.6e-14
  maturity=2028-01-15  periods=4  quoted=0.20  repriced=0.20  caplet=0.2075780   sensitivity=1.385  indeterminacy=2.5e-13
  maturity=2029-01-15  periods=4  quoted=0.22  repriced=0.22  caplet=0.2419188   sensitivity=2.095  indeterminacy=5.3e-13
  maturity=2030-01-15  periods=4  quoted=0.24  repriced=0.24  caplet=0.2764117   sensitivity=2.830  indeterminacy=8.1e-13
  maturity=2031-01-15  periods=4  quoted=0.26  repriced=0.26  caplet=0.3120089   sensitivity=3.631  indeterminacy=1.1e-12
worst_repricing_error: 1.942890293e-16
```

The repriced column is the only check on the bootstrap that does not go through
another approximation, and it costs one solve per quote. The other two columns
are the reason this command exists.

### The stripped curve outruns the quotes

A flat volatility is a premium-weighted average of the caplet volatilities under
it, not an average of the volatilities themselves, so a rising quote curve needs
a caplet curve that rises faster. Quotes of 18% to 26% strip to 18.0% to
**31.2%**: the far bucket sits 5.2 volatility points above the quote that set it,
and the slope between the last two buckets is 1.77 times the slope between the
last two quotes. The first bucket's volatility is its quote exactly — a
one-bucket strip has nothing to average — which is the one row that can be
checked by hand.

A hump is worse than a slope. Quotes of 18, 26, 30, 26 and 22 per cent strip to
18.0, 29.3, **34.5**, 18.6 and 11.5, so a four-point fall in the quotes is a
sixteen-point fall in the buckets: the later periods have to undo an average the
earlier ones are holding up.

### Monotone is not steep, and that is the trap at both ends

The premium rises with the volatility everywhere, so a bracket on `[0, ceiling]`
always contains a root and a solver always returns something.

At the far maturities it returns something badly conditioned, which the
sensitivity column reports: a basis point on the one-year quote moves its bucket
by 1.00 basis points, and a basis point on the five-year quote moves the last
bucket by **3.63**, because each bucket is a smaller share of its cap's premium
than the last.

Deep in the money it returns nonsense, which is what the indeterminacy column
is for. An option eleven standard deviations into the money is worth its
intrinsic value to *the same double*, so every volatility explains the quote
equally well; a bucket whose repricing set is wider than one basis point of
volatility is refused as unidentified rather than reported. Measured at an 18%
quote against forwards of 3.23% to 3.39%, the answer is pinned to 1.2e-13 at a
strike of 3.5%, 1.3e-09 at 2%, 6.2e-06 at 1.5%, and the lowest strike the quote
says anything about at all is **1.3901%**.

That guard is also what stops a falling quote curve, which was the surprise.
Holding the one-year quote at 18% and walking the two-year quote down, the strip
survives a fall of **1202 basis points** and not 1203 — and at the last quote it
accepts, the bucket's indeterminacy has come within 0.3% of the 1.0e-04
limit, with the premium unattainable one step below. The two refusals coincide
because they are one statement measured two ways. The usual worry about a
sequential volatility bootstrap is a negative variance, and that is not what
binds.

The least obvious consequence: **a zero quote identifies a zero volatility only
exactly at the money.** Vega vanishes with the volatility away from the strike,
so an interval of volatilities all produce the intrinsic value. At the money the
premium is linear in the volatility and zero is pinned to 2.8e-16.

## Development

```bash
pip install -e ".[dev]"
pytest          # 1475 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks,
lints, and installs the wheel *and* the sdist into separate clean environments to
confirm each can produce a number — a distribution that imports but cannot compute
is not a working library, and the two artefacts are built by different code paths,
so a file missing from one can be present in the other.

### Releasing

The version lives in `pyproject.toml` and is mirrored by `tenor.__version__`; a
test asserts the two agree, and the release refuses to run if the tag disagrees
with either.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag builds both artefacts, has `twine` read the metadata the way the index
will, installs each into a clean environment and bootstraps the bundled quote
screen from it, and then publishes through the index's trusted-publishing flow — so
there is no API token in this repository, in the workflow, or in the repository's
secrets. Registering the publisher is a one-time step done on the index, naming
this repository, `release.yml` and the `pypi` environment.

## Licence

MIT.
