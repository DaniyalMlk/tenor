# tenor

A fixed-income curve toolkit, in pure Python with no dependencies.

Fixed income is a field where the arithmetic is easy and the conventions decide
the answer. A bond priced on the wrong day count basis is wrong by a few basis
points, which is the size of the spread anybody is trying to measure — so it is
not approximately right, it is answering a different question. This library
treats every convention as an argument with a name, never a default, and checks
each one against the published rules rather than against itself.

`ROADMAP.md` says what is built and what is not. Phase 1 — dates, day counts,
holiday calendars and payment schedules — is done.

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

### Negative periods are refused

A year fraction with the end before the start is a mistake in the caller —
almost always a schedule generated backwards — not a negative quantity. Returned
as a number it travels a long way before anything notices, so it is refused at
the point it is asked for.

## Development

```bash
pip install -e ".[dev]"
pytest          # 122 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks,
lints, and installs the built wheel into a clean environment to confirm it can
produce a number — a wheel that imports but cannot compute is not a working
library.

## Licence

MIT.
