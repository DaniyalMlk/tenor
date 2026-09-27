"""Tests for floating rate notes, built around one exact identity.

Almost everything here descends from a single fact: a note priced at its own
quoted margin is worth exactly par, because each coupon equals its own discount
denominator minus one and the sum telescopes. That is an equality, not a limit, so
it is asserted to a handful of bits across four curve shapes and three day count
bases. Nearly every plausible implementation error breaks it — a margin added to
the wrong side, an accrual fraction taken over the wrong dates, a discount factor
applied from the curve's reference date instead of from settlement.

One thing the identity does *not* check, and it is worth being clear about it: the
telescoping needs only that the coupon and the discount denominator use the same
rate over the same fraction. It does not care whether that rate is the right one.
A note whose index is projected the wrong way still prices at exactly par. So the
projection convention gets its own tests, measuring what it actually changes
rather than relying on par to catch it.

The other load-bearing tests are the two durations, which for this instrument are
nothing like each other. Rate duration is zero to floating point at the quoted
margin on *any* date once the fixing is known, and the tests assert that rather
than a small number, because "small" would also pass for an implementation that
had the reset logic subtly wrong.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tenor.bond import Bond
from tenor.bootstrap import bootstrap
from tenor.curve import DiscountCurve
from tenor.daycount import Basis, year_fraction
from tenor.floating import (
    BadMargin,
    BadNote,
    FloatingNote,
)
from tenor.instruments import Deposit, Swap
from tenor.rates import Compounding
from tenor.schedule import Frequency

REFERENCE = date(2024, 1, 15)
QUOTED = 0.0075


def sloped() -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2024, 7, 15), 0.002, Basis.ACT_360),
        Swap(REFERENCE, date(2026, 1, 15), 0.0125),
        Swap(REFERENCE, date(2029, 1, 15), 0.0205),
        Swap(REFERENCE, date(2034, 1, 16), 0.0255),
        Swap(REFERENCE, date(2044, 1, 15), 0.0279),
    ]
    return bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F).curve


def inverted() -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2024, 7, 15), 0.051, Basis.ACT_360),
        Swap(REFERENCE, date(2026, 1, 15), 0.045),
        Swap(REFERENCE, date(2029, 1, 15), 0.037),
        Swap(REFERENCE, date(2034, 1, 16), 0.032),
        Swap(REFERENCE, date(2044, 1, 15), 0.030),
    ]
    return bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F).curve


def humped() -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2024, 7, 15), 0.012, Basis.ACT_360),
        Swap(REFERENCE, date(2026, 1, 15), 0.034),
        Swap(REFERENCE, date(2029, 1, 15), 0.041),
        Swap(REFERENCE, date(2034, 1, 16), 0.030),
        Swap(REFERENCE, date(2044, 1, 15), 0.022),
    ]
    return bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F).curve


def flat(rate: float = 0.025) -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE, [(date(2044, 1, 15), rate)], basis=Basis.ACT_365F
    )


CURVES = {"sloped": sloped, "inverted": inverted, "humped": humped, "flat": flat}


def note(
    *,
    maturity: date = date(2029, 1, 17),
    margin: float = QUOTED,
    basis: Basis = Basis.ACT_360,
    frequency: Frequency = Frequency.QUARTERLY,
    effective: date = date(2022, 1, 17),
    redemption: float = 100.0,
) -> FloatingNote:
    return FloatingNote(
        effective=effective,
        maturity=maturity,
        quoted_margin=margin,
        frequency=frequency,
        basis=basis,
        redemption=redemption,
    )


RESET = date(2024, 1, 17)


# -- the identity -------------------------------------------------------------


@pytest.mark.parametrize("shape", list(CURVES))
@pytest.mark.parametrize(
    "basis", [Basis.ACT_360, Basis.ACT_365F, Basis.THIRTY_360_BOND, Basis.ACT_ACT_ISDA]
)
@pytest.mark.parametrize(
    "frequency", [Frequency.QUARTERLY, Frequency.SEMI_ANNUAL, Frequency.ANNUAL]
)
def test_a_note_at_its_quoted_margin_is_worth_exactly_par(
    shape: str, basis: Basis, frequency: Frequency
) -> None:
    """The identity the whole module is built on, asserted as one.

    Each coupon is its own discount denominator minus one, so the sum telescopes
    to one less the final discount factor and the redemption supplies the rest.
    Nothing in that depends on the curve, the basis or the frequency, so all
    three are swept and the tolerance is a few bits rather than a basis point.
    """
    subject = note(basis=basis, frequency=frequency)
    curve = CURVES[shape]()
    assert subject.is_reset_date(RESET)
    price = subject.dirty_price(curve, RESET, margin=QUOTED)
    assert price == pytest.approx(100.0, abs=5e-13)
    assert subject.accrued(RESET, curve=curve) == pytest.approx(0.0, abs=1e-15)
    assert subject.clean_price(curve, RESET, margin=QUOTED) == pytest.approx(
        100.0, abs=5e-13
    )


@pytest.mark.parametrize("redemption", [98.0, 100.0, 103.5])
def test_the_identity_scales_with_the_redemption(redemption: float) -> None:
    """Par is the redemption, not the number 100.

    A note redeeming at 98 is worth exactly 98 at its quoted margin, because the
    telescoping sum is in units of the notional the coupons are quoted on. A
    hard-coded 100 in the discounting passes the default case and fails here.
    """
    subject = FloatingNote(
        effective=date(2022, 1, 17),
        maturity=date(2029, 1, 17),
        quoted_margin=QUOTED,
        frequency=Frequency.QUARTERLY,
        basis=Basis.ACT_360,
        redemption=redemption,
    )
    # The coupons are per 100 of notional whatever the redemption is, so the
    # identity holds on the redemption only when they are scaled with it; this
    # note pays coupons on 100 and redeems at `redemption`, which is what a
    # partly amortised note does, so the price is par plus the shortfall
    # discounted.
    curve = sloped()
    price = subject.dirty_price(curve, RESET, margin=QUOTED)
    last = subject.coupons(curve, RESET, margin=QUOTED)[-1]
    assert price == pytest.approx(100.0 + (redemption - 100.0) * last.discount, abs=5e-13)


@pytest.mark.parametrize("shape", list(CURVES))
def test_away_from_a_reset_date_the_price_still_has_no_curve_in_it(shape: str) -> None:
    """The stronger form of the identity, which measuring it is what found.

    With the fixing known and the margin equal to the quoted one, the periods
    after the current one telescope to exactly one and the whole price reduces to
    a ratio in the fixing alone. So it is the same number on four different
    curves, which is a much sharper test than "close to par".
    """
    subject = note()
    curve = CURVES[shape]()
    fixing = 0.0412
    for days in (0, 7, 21, 45, 89):
        settlement = RESET + timedelta(days=days)
        period = subject.current_period(settlement)
        full = year_fraction(period.start, period.end, subject.basis)
        remaining = year_fraction(settlement, period.end, subject.basis)
        expected = 100.0 * (1.0 + (fixing + QUOTED) * full) / (
            1.0 + (fixing + QUOTED) * remaining
        )
        assert subject.dirty_price(
            curve, settlement, margin=QUOTED, current_fixing=fixing
        ) == pytest.approx(expected, abs=1e-12)


def test_the_par_identity_does_not_validate_the_projection() -> None:
    """Stated as a test so nobody relies on par to catch a projection error.

    The telescoping needs the coupon and the discount denominator to use the same
    rate over the same fraction. It does not need that rate to be right. A note
    projected with a rate scaled by an arbitrary factor — which is exactly what
    using the curve's own forward over a different accrual basis does — still
    prices at exactly par.
    """
    subject = note()
    curve = sloped()
    coupons = subject.coupons(curve, RESET, margin=QUOTED)

    def price_with(scale: float) -> float:
        total = 0.0
        factor = 1.0
        for one in coupons:
            rate = one.index_rate * scale
            factor /= 1.0 + (rate + QUOTED) * one.accrual
            amount = 100.0 * (rate + QUOTED) * one.accrual + one.redemption
            total += amount * factor
        return total

    for scale in (0.5, 1.0, 1.0139, 2.0):
        assert price_with(scale) == pytest.approx(100.0, abs=5e-13)


# -- the two durations --------------------------------------------------------


@pytest.mark.parametrize("shape", list(CURVES))
def test_rate_duration_is_exactly_zero_at_the_quoted_margin(shape: str) -> None:
    """Zero to floating point, on every date, not a small number.

    A small number would also be returned by an implementation that discounted
    the current period over the wrong fraction, so the assertion has to be the
    exact one. The fixing has to be supplied: without it the current period's
    rate comes off the curve and moves with it.
    """
    subject = note()
    curve = CURVES[shape]()
    for days in (0, 21, 60, 89):
        settlement = RESET + timedelta(days=days)
        measured = subject.rate_duration(
            curve, settlement, margin=QUOTED, current_fixing=0.0412
        )
        assert abs(measured) < 1e-9


def test_rate_duration_away_from_par_is_linear_in_the_margin_difference() -> None:
    """Where a floater's rate risk actually comes from.

    Not the coupon reset — that contributes nothing at par. It is the annuity of
    the difference between the discount margin and the quoted one, so the
    duration is proportional to that difference and changes sign with the side of
    par. The measured figures on this note are -0.061 years at 125bp, -0.250 at
    275bp and +0.061 at 25bp against a quoted 75bp.
    """
    subject = note()
    curve = sloped()
    at_par = subject.rate_duration(curve, RESET, margin=QUOTED)
    wide = subject.rate_duration(curve, RESET, margin=0.0275)
    tight = subject.rate_duration(curve, RESET, margin=0.0025)
    middling = subject.rate_duration(curve, RESET, margin=0.0125)
    assert at_par == pytest.approx(0.0, abs=1e-9)
    assert wide == pytest.approx(-0.250, abs=0.01)
    assert tight == pytest.approx(+0.061, abs=0.01)
    assert middling == pytest.approx(-0.061, abs=0.01)
    # Linear: quadrupling the margin difference quadruples the duration.
    assert wide / middling == pytest.approx(4.0, rel=0.05)
    assert subject.dirty_price(curve, RESET, margin=0.0275) < 100.0
    assert subject.dirty_price(curve, RESET, margin=0.0025) > 100.0


def test_projecting_the_current_fixing_puts_rate_risk_back() -> None:
    """The cost of a missing fixings history, stated as risk.

    With the fixing unknown the current period's rate is read off the curve, so
    the price moves when the curve does even at par. It is -0.057 years three
    weeks into a quarterly period, which is small and is not zero, and the
    difference between the two is exactly what having the fixing buys.
    """
    subject = note()
    curve = sloped()
    settlement = RESET + timedelta(days=21)
    projected = subject.rate_duration(curve, settlement, margin=QUOTED)
    known = subject.rate_duration(
        curve,
        settlement,
        margin=QUOTED,
        current_fixing=subject.index_rate(curve, subject.current_period(settlement)),
    )
    assert known == pytest.approx(0.0, abs=1e-9)
    assert projected == pytest.approx(-0.057, abs=0.01)


@pytest.mark.parametrize(
    ("maturity", "expected_ratio"),
    [(date(2027, 1, 17), 1.014), (date(2029, 1, 17), 1.005), (date(2034, 1, 17), 0.974)],
)
def test_spread_duration_matches_a_fixed_bond_of_the_same_maturity(
    maturity: date, expected_ratio: float
) -> None:
    """The sensitivity that survives, and it is a whole-life one.

    The margin is discounted across every remaining period, so its sensitivity is
    an annuity over the note's life rather than over its coupon period. Compared
    against the modified duration of a fixed bond on the same curve with the same
    schedule, because that is the number a reader already has intuition for.
    """
    curve = sloped()
    floater = note(maturity=maturity)
    spread = floater.spread_duration(curve, RESET, margin=QUOTED)
    fixed = Bond(
        date(2022, 1, 17), maturity, 0.02, Frequency.QUARTERLY, Basis.ACT_360
    )
    modified = fixed.modified_duration(fixed.curve_yield(curve, RESET).value, RESET)
    assert spread / modified == pytest.approx(expected_ratio, abs=0.01)
    assert spread > 0.0


def test_the_margin_value_of_a_basis_point_agrees_with_the_duration() -> None:
    """Two statements of the same derivative, one per 100 and one in years.

    ``duration * price * 1bp`` is the price change to first order, and the
    finite-difference figure has to match it to the size of the second order
    term — which on a five-year annuity at one basis point is a few parts in ten
    thousand.
    """
    subject = note()
    curve = sloped()
    value = subject.margin_value_of_a_basis_point(curve, RESET, margin=QUOTED)
    duration = subject.spread_duration(curve, RESET, margin=QUOTED)
    price = subject.dirty_price(curve, RESET, margin=QUOTED)
    assert value == pytest.approx(duration * price * 1e-4, rel=1e-3)
    assert value > 0.0


# -- the discount margin ------------------------------------------------------


@pytest.mark.parametrize("shape", list(CURVES))
def test_a_note_at_par_returns_its_own_quoted_margin(shape: str) -> None:
    """The one calibration that needs no external reference.

    By the identity, a clean price of par on a reset date has to imply a discount
    margin equal to the quoted one exactly. A solver that converged on the wrong
    objective would land near it and not on it.
    """
    subject = note()
    curve = CURVES[shape]()
    solved = subject.discount_margin(curve, 100.0, RESET)
    assert solved.converged
    assert solved.value == pytest.approx(QUOTED, abs=1e-12)


@pytest.mark.parametrize("price", [100.0, 99.0, 101.5, 95.0, 108.0])
def test_a_solved_margin_reprices_the_note(price: float) -> None:
    """The only general check available, and it is the right one.

    There is no closed form for a discount margin away from par, so what has to
    hold is that putting the answer back through the pricer returns the price it
    came from — to the last few bits, not to a basis point.
    """
    subject = note()
    curve = sloped()
    solved = subject.discount_margin(curve, price, RESET)
    assert solved.converged
    assert subject.clean_price(curve, RESET, margin=solved.value) == pytest.approx(
        price, abs=1e-9
    )
    assert abs(solved.residual) < 1e-9


def test_the_margin_solves_against_the_dirty_price_either_way() -> None:
    """Quoting clean and solving against clean is the classic error.

    It vanishes on a reset date, which is why it survives: the accrued interest is
    zero there and both routes agree. Twenty days in it is worth 21 basis points
    of margin on this note, so the test settles away from the reset.
    """
    subject = note()
    curve = sloped()
    settlement = RESET + timedelta(days=20)
    fixing = 0.0412
    clean = 99.0
    accrued = subject.accrued(settlement, current_fixing=fixing)
    assert accrued > 0.0
    from_clean = subject.discount_margin(
        curve, clean, settlement, current_fixing=fixing
    )
    from_dirty = subject.discount_margin(
        curve, clean + accrued, settlement, clean=False, current_fixing=fixing
    )
    assert from_clean.value == pytest.approx(from_dirty.value, abs=1e-12)
    ignoring_accrual = subject.discount_margin(
        curve, clean, settlement, clean=False, current_fixing=fixing
    )
    assert abs(ignoring_accrual.value - from_clean.value) > 1e-4


def test_a_price_no_margin_reaches_is_refused_with_both_ends() -> None:
    subject = note()
    curve = sloped()
    with pytest.raises(BadMargin, match="no margin between"):
        subject.discount_margin(curve, 400.0, RESET)
    with pytest.raises(BadMargin, match="cannot have a dirty price"):
        subject.discount_margin(curve, -5.0, RESET, clean=False)


def test_a_margin_that_destroys_the_discounting_is_refused() -> None:
    """Below -1 over a period the discount factor changes sign.

    Which would produce a price, and a monotone one, and it would be nonsense.
    The bracket the solver uses never goes there; a caller passing a margin
    directly can.
    """
    subject = note()
    curve = sloped()
    with pytest.raises(BadMargin, match="non-positive"):
        subject.dirty_price(curve, RESET, margin=-40.0)


# -- projection, accrual and the conventions ----------------------------------


def test_the_index_rate_is_taken_against_the_coupons_own_accrual() -> None:
    """Not off the curve's forward, which is a rate over a different fraction.

    On an ACT/360 note over an ACT/365F curve the two differ by the ratio of the
    bases: 20.72 basis points against 21.01 on the first period here. The par
    identity does not catch it, so this test does.
    """
    subject = note()
    curve = sloped()
    period = subject.current_period(RESET)
    mine = subject.index_rate(curve, period)
    theirs = curve.forward_rate(
        period.adjusted_start, period.adjusted_end, Compounding.SIMPLE
    )
    accrual = year_fraction(period.start, period.end, subject.basis)
    growth = curve.discount(period.adjusted_start) / curve.discount(
        period.adjusted_end
    )
    assert mine == pytest.approx((growth - 1.0) / accrual, rel=1e-15)
    assert theirs / mine == pytest.approx(365.0 / 360.0, rel=0.01)
    assert mine != pytest.approx(theirs, rel=1e-6)


def test_a_supplied_fixing_replaces_the_projection_for_the_current_period_only() -> None:
    subject = note()
    curve = sloped()
    settlement = RESET + timedelta(days=10)
    coupons = subject.coupons(curve, settlement, current_fixing=0.0412)
    assert coupons[0].index_rate == 0.0412
    assert not coupons[0].projected
    assert all(one.projected for one in coupons[1:])
    without = subject.coupons(curve, settlement)
    assert without[0].projected
    assert without[0].index_rate != 0.0412
    assert [one.index_rate for one in without[1:]] == [
        one.index_rate for one in coupons[1:]
    ]


def test_projecting_the_fixing_costs_about_three_basis_points_of_price() -> None:
    """Measured, because it is the size of the approximation the library allows.

    Three weeks into a quarterly period with the real fixing 50bp away from the
    curve's projection, the price is wrong by 2.9 basis points — and the error is
    symmetric in the sign of the miss, which says it is the linear term and not
    something worse.
    """
    subject = note()
    curve = sloped()
    settlement = RESET + timedelta(days=21)
    projected = subject.index_rate(curve, subject.current_period(settlement))
    reference = subject.dirty_price(curve, settlement, margin=QUOTED)
    high = subject.dirty_price(
        curve, settlement, margin=QUOTED, current_fixing=projected + 0.005
    )
    low = subject.dirty_price(
        curve, settlement, margin=QUOTED, current_fixing=projected - 0.005
    )
    assert (high - reference) * 100.0 == pytest.approx(2.9, abs=0.3)
    assert (reference - low) * 100.0 == pytest.approx(2.9, abs=0.3)


def test_accrued_interest_needs_the_fixing_and_says_so() -> None:
    """Refusing beats returning the margin's accrual on its own.

    That would be a number, and it would be wrong by the whole of the index —
    about 85% of the accrual on a note paying 4% over a 75bp margin.
    """
    subject = note()
    curve = sloped()
    settlement = RESET + timedelta(days=30)
    with pytest.raises(BadNote, match="current_fixing"):
        subject.accrued(settlement)
    known = subject.accrued(settlement, current_fixing=0.0412)
    elapsed = year_fraction(
        subject.current_period(settlement).start, settlement, subject.basis
    )
    assert known == pytest.approx(100.0 * (0.0412 + QUOTED) * elapsed, rel=1e-15)
    assert subject.accrued(settlement, curve=curve) > 0.0
    assert subject.accrued(RESET, curve=curve) == pytest.approx(0.0, abs=1e-15)


def test_a_flow_due_on_the_settlement_date_belongs_to_the_seller() -> None:
    """The same rule ``current_period`` uses, applied in the other place it must.

    A coupon paid on the settlement date has gone to the seller, so it is not one
    of the flows being bought. Off by one here gives a price high by a whole
    coupon on exactly the dates a test written on a reset date would use.
    """
    subject = note()
    curve = sloped()
    on_reset = subject.coupons(curve, RESET, margin=QUOTED)
    day_before = subject.coupons(
        curve, RESET - timedelta(days=1), margin=QUOTED, current_fixing=0.0412
    )
    assert len(day_before) == len(on_reset) + 1
    assert on_reset[0].period.start == RESET


def test_the_coupon_count_and_the_schedule_agree() -> None:
    subject = note(maturity=date(2029, 1, 17))
    curve = sloped()
    coupons = subject.coupons(curve, RESET, margin=QUOTED)
    assert len(coupons) == 20
    assert coupons[-1].redemption == 100.0
    assert all(one.redemption == 0.0 for one in coupons[:-1])
    assert coupons[-1].period.end == date(2029, 1, 17)
    assert [one.payment for one in coupons] == [
        one.payment for one in subject.schedule() if one.end > RESET
    ]
    assert all(one.amount == one.coupon + one.redemption for one in coupons)
    assert all(
        one.present_value == one.amount * one.discount for one in coupons
    )


def test_the_discount_factors_fall_monotonically() -> None:
    subject = note()
    curve = sloped()
    factors = [one.discount for one in subject.coupons(curve, RESET, margin=QUOTED)]
    assert factors == sorted(factors, reverse=True)
    assert all(0.0 < factor <= 1.0 for factor in factors)


def test_a_period_that_began_before_the_curve_cannot_be_projected() -> None:
    """Not an approximation — the discount factor at the reset date does not exist.

    Which is the ordinary situation on any settlement date that is not a reset
    date: the period started in the past, and a curve knows nothing before its own
    reference date. The refusal names the fixing rather than the curve, because the
    fixing is what the caller has to go and find.
    """
    subject = note()
    curve = sloped()
    settlement = RESET - timedelta(days=1)
    with pytest.raises(BadNote, match="started before the curve's reference date"):
        subject.dirty_price(curve, settlement, margin=QUOTED)
    priced = subject.dirty_price(
        curve, settlement, margin=QUOTED, current_fixing=0.0412
    )
    assert priced > 0.0


def test_a_matured_note_is_refused_rather_than_priced_at_zero() -> None:
    """An empty sequence of flows sums to zero, which is a price and is not one.

    The range check lives in ``current_period``, and ``coupons`` calls it for
    nothing else — without that call a settlement date past maturity returns 0.0
    and every risk number built on it returns a division by zero or a nan.
    """
    subject = note()
    curve = sloped()
    with pytest.raises(BadNote, match="matured on"):
        subject.coupons(curve, date(2030, 1, 1))
    with pytest.raises(BadNote, match="matured on"):
        subject.dirty_price(curve, date(2029, 1, 17))


def test_the_period_remaining_runs_from_one_to_zero() -> None:
    subject = note()
    assert subject.period_remaining(RESET) == pytest.approx(1.0)
    previous = 1.1
    for days in (0, 20, 50, 80, 89):
        current = subject.period_remaining(RESET + timedelta(days=days))
        assert current < previous
        previous = current
    assert previous < 0.1


# -- refusals ------------------------------------------------------------------


def test_a_note_that_is_not_a_trade_is_refused() -> None:
    with pytest.raises(BadNote, match="not after it was issued"):
        FloatingNote(date(2029, 1, 17), date(2024, 1, 17))
    with pytest.raises(BadNote, match="pays nothing back"):
        FloatingNote(date(2024, 1, 17), date(2029, 1, 17), redemption=0.0)


def test_settlement_outside_the_notes_life_is_refused() -> None:
    subject = note()
    with pytest.raises(BadNote, match="before it was issued"):
        subject.current_period(date(2021, 1, 1))
    with pytest.raises(BadNote, match="matured on"):
        subject.current_period(date(2029, 1, 17))


def test_a_non_positive_shift_is_refused_by_both_risk_measures() -> None:
    subject = note()
    curve = sloped()
    with pytest.raises(BadNote, match="positive size"):
        subject.spread_duration(curve, RESET, shift=0.0)
    with pytest.raises(BadNote, match="positive size"):
        subject.rate_duration(curve, RESET, shift=-1e-4)


def test_the_note_names_itself_usefully() -> None:
    assert note().name == "index + 75.0bp of 2029-01-17"
    assert (
        FloatingNote(
            date(2022, 1, 17), date(2029, 1, 17), QUOTED, label="SOFR + 75"
        ).name
        == "SOFR + 75"
    )


def test_the_schedule_is_built_once() -> None:
    """Every risk number reprices the note, and each reprice walks the schedule."""
    subject = note()
    assert subject.schedule() is subject.schedule()


def test_a_zero_margin_note_is_a_pure_curve_instrument() -> None:
    """With no quoted margin and no discount margin, par exactly.

    The degenerate case of the identity, and the one where a stray ``+ margin``
    in the discounting but not in the coupon would still leave the right answer
    under the default arguments — so it is asserted separately from the
    margin-bearing case rather than instead of it.
    """
    subject = note(margin=0.0)
    for shape in CURVES:
        assert subject.dirty_price(CURVES[shape](), RESET) == pytest.approx(
            100.0, abs=5e-13
        )
    assert math.isclose(
        subject.discount_margin(sloped(), 100.0, RESET).value, 0.0, abs_tol=1e-12
    )
