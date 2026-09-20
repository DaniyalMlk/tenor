# tenor

A fixed-income curve toolkit, in pure Python with no dependencies.

Fixed income is a field where the arithmetic is easy and the conventions decide
the answer. A bond priced on the wrong day count basis is wrong by a few basis
points, which is the size of the spread anybody is trying to measure — so it is
not approximately right, it is answering a different question. This library
treats every convention as an argument with a name, never a default, and checks
each one against the published rules rather than against itself.

`ROADMAP.md` says what is built and what is not. Phase 1 — dates, day counts,
holiday calendars and payment schedules — is done, and phase 2 — discount
curves, compounding conventions and interpolation — is done but for the
monotone convex scheme.

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

## Design

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

## Development

```bash
pip install -e ".[dev]"
pytest          # 240 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks,
lints, and installs the built wheel into a clean environment to confirm it can
produce a number — a wheel that imports but cannot compute is not a working
library.

## Licence

MIT.
