"""Covered interest parity, the curve it implies, and the basis left over.

Three kinds of check, and the first two are stronger than anything in the
library's other modules because the mathematics here is an identity rather than
a model.

**The exact ones are exact.** Parity is one division, so the round trip through
it returns the same float rather than a close one, and the assertions are on
equality. A curve rebuilt from forwards generated off a known curve reproduces
it to round-off. A par spread between a curve and itself is the telescoping of
the notional exchange and comes out at 1e-16.

**The conventions are measured, because that is the only way to know which one
a number is in.** The spot settlement lag, the accrual day count and the
difference between a continuously compounded basis and a spread paid on an
accrual all move the answer, and the tests record by how much rather than
asserting the move is small. Two of them pull in opposite directions and
partly cancel, which is exactly the situation where a single net tolerance
hides a mistake.

**The refusals cover the pairs of arguments that are individually fine and
jointly wrong**: two curves with different reference dates, a forward settling
before the spot date, a repeated tenor.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor.calendar import WEEKENDS_ONLY
from tenor.curve import DiscountCurve, OffCurve
from tenor.daycount import Basis, year_fraction
from tenor.fx import (
    BadFx,
    FxForward,
    basis_curve,
    forward_outright,
    forward_points,
    implied_basis,
    implied_curve,
    implied_foreign_discount,
    par_basis_spread,
)
from tenor.schedule import Frequency, Schedule, generate

REFERENCE = date(2026, 1, 15)
#: Two business days out, which over a Thursday reference is four calendar days.
SPOT_DATE = date(2026, 1, 19)
SPOT = 1.10
DOMESTIC_RATE = 0.045
FOREIGN_RATE = 0.030


def flat(rate: float, years: int = 11) -> DiscountCurve:
    """A continuously compounded flat curve on ACT/365F year fractions."""
    points = []
    for step in range(1, years):
        day = date(2026 + step, 1, 15)
        time = year_fraction(REFERENCE, day, Basis.ACT_365F)
        points.append((day, math.exp(-rate * time)))
    points.append(
        (
            date(2026, 4, 15),
            math.exp(-rate * year_fraction(REFERENCE, date(2026, 4, 15), Basis.ACT_365F)),
        )
    )
    return DiscountCurve.from_discounts(REFERENCE, points, basis=Basis.ACT_365F)


def shifted(curve: DiscountCurve, spread: float) -> DiscountCurve:
    """``curve`` with a continuously compounded spread subtracted from its zeros."""
    points = [
        (pillar.day, pillar.discount * math.exp(-spread * pillar.time))
        for pillar in curve.pillars
        if pillar.time > 0.0
    ]
    return DiscountCurve.from_discounts(REFERENCE, points, basis=curve.basis)


def term_shifted(curve: DiscountCurve) -> DiscountCurve:
    """``curve`` with a zero basis rising linearly from 10 to 40 basis points.

    The *forward* basis is then twice as steep at the long end, because
    differentiating ``b(t) t`` adds ``t b'(t)``. That is the point of the
    fixture.
    """
    points = [
        (
            pillar.day,
            pillar.discount
            * math.exp(
                -(0.0010 + 0.0030 * min(pillar.time, 5.0) / 5.0) * pillar.time
            ),
        )
        for pillar in curve.pillars
        if pillar.time > 0.0
    ]
    return DiscountCurve.from_discounts(REFERENCE, points, basis=curve.basis)


def quarterly() -> Schedule:
    return generate(
        SPOT_DATE, date(2031, 1, 19), Frequency.QUARTERLY, calendar=WEEKENDS_ONLY
    )


# -- parity --------------------------------------------------------------------


def test_the_parity_round_trip_returns_the_same_float() -> None:
    """One division out and one back, so equality is the right assertion.

    A tolerance here would be hiding something: there is nowhere for an error
    to come from except the two divisions, and if those do not invert then the
    arguments are not what the function thinks they are.
    """
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    for step in (1, 2, 5, 10):
        day = date(2026 + step, 1, 15)
        outright = forward_outright(SPOT, domestic, foreign, day, spot_day=SPOT_DATE)
        recovered = implied_foreign_discount(
            SPOT,
            outright,
            domestic,
            day,
            spot_day=SPOT_DATE,
            foreign_spot=foreign.discount(SPOT_DATE),
        )
        assert recovered == foreign.discount(day)


def test_the_higher_rate_currency_trades_at_a_forward_discount() -> None:
    """Which is the direction people get backwards, so it is asserted directly."""
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    day = date(2027, 1, 15)
    # Domestic rate above foreign: the foreign currency buys more domestic
    # forward than spot, so the outright is above the spot.
    assert forward_outright(SPOT, domestic, foreign, day, spot_day=SPOT_DATE) > SPOT
    # Reverse the two and the forward is below.
    assert forward_outright(SPOT, foreign, domestic, day, spot_day=SPOT_DATE) < SPOT
    # Equal rates leave the forward at the spot, exactly.
    same = flat(DOMESTIC_RATE)
    assert forward_outright(
        SPOT, domestic, same, day, spot_day=SPOT_DATE
    ) == pytest.approx(SPOT, rel=1e-15)


def test_forward_points_carry_the_pip_and_the_sign() -> None:
    assert forward_points(SPOT, SPOT + 0.0100) == pytest.approx(100.0, rel=1e-12)
    assert forward_points(SPOT, SPOT - 0.0100) == pytest.approx(-100.0, rel=1e-12)
    # The yen convention, two decimal places rather than four.
    assert forward_points(150.0, 150.50, pip=1e-2) == pytest.approx(50.0, rel=1e-12)
    with pytest.raises(BadFx, match="a pip is a positive size"):
        forward_points(SPOT, SPOT, pip=0.0)
    with pytest.raises(BadFx, match="an exchange rate is positive"):
        forward_points(0.0, 1.0)


def test_ignoring_the_spot_lag_costs_nearly_the_same_at_every_tenor() -> None:
    """Which is why it is dangerous: it is a rounding error at five years and
    several per cent of the price at three months.

    The absolute error is the rate differential over the lag applied to the
    spot, and nothing in it grows with maturity. The forward points do grow, so
    the *relative* error collapses with tenor and the convention matters most
    where the forward is cheapest.
    """
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    shares = []
    for day in (date(2026, 4, 15), date(2027, 1, 15), date(2031, 1, 15)):
        correct = forward_outright(SPOT, domestic, foreign, day, spot_day=SPOT_DATE)
        naive = forward_outright(SPOT, domestic, foreign, day)
        error = forward_points(SPOT, naive) - forward_points(SPOT, correct)
        # Between one and two pips at every tenor, and always the same sign.
        assert 1.5 < error < 2.5
        shares.append(error / forward_points(SPOT, correct))
    assert shares[0] > 0.04
    assert shares[2] < 0.005
    assert shares == sorted(shares, reverse=True)


def test_parity_refuses_arguments_that_are_jointly_wrong() -> None:
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    other = DiscountCurve.from_discounts(
        date(2026, 2, 1), [(date(2027, 2, 1), 0.96)], basis=Basis.ACT_365F
    )
    with pytest.raises(BadFx, match="two different todays"):
        forward_outright(SPOT, domestic, other, date(2027, 1, 15))
    with pytest.raises(BadFx, match="not a price"):
        forward_outright(-1.0, domestic, foreign, date(2027, 1, 15))
    with pytest.raises(BadFx, match="before the spot date"):
        forward_outright(
            SPOT, domestic, foreign, date(2026, 1, 16), spot_day=SPOT_DATE
        )
    with pytest.raises(BadFx, match="before the curves' reference date"):
        forward_outright(
            SPOT, domestic, foreign, date(2027, 1, 15), spot_day=date(2025, 1, 1)
        )
    with pytest.raises(OffCurve):
        forward_outright(SPOT, domestic, foreign, date(2099, 1, 15))
    with pytest.raises(BadFx, match="must be finite"):
        FxForward(date(2027, 1, 15), float("nan"))
    with pytest.raises(BadFx, match="not a price"):
        FxForward(date(2027, 1, 15), 0.0)


# -- the implied curve ---------------------------------------------------------


def test_the_implied_curve_reproduces_the_curve_the_forwards_came_from() -> None:
    """Generated off a known curve and inverted back, so the answer is known.

    The strongest check available: there is no model in parity, so a curve
    recovered from its own forwards has to be the curve, and the only thing
    standing between them is double precision.
    """
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    days = [date(2026 + step, 1, 15) for step in (1, 2, 3, 4, 5)]
    forwards = [
        FxForward(
            day, forward_outright(SPOT, domestic, foreign, day, spot_day=SPOT_DATE)
        )
        for day in days
    ]
    implied = implied_curve(
        SPOT,
        forwards,
        domestic,
        spot_day=SPOT_DATE,
        foreign_spot=foreign.discount(SPOT_DATE),
    )
    worst = max(
        abs(implied.discount(day) / foreign.discount(day) - 1.0) for day in days
    )
    assert worst < 1e-15, worst
    # And the basis against the curve it came from is zero at every pillar.
    for point in basis_curve(implied, foreign, days):
        assert abs(point.spread) < 1e-15


def test_the_implied_curve_recovers_a_basis_that_was_put_into_it() -> None:
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    synthetic = shifted(foreign, 0.0025)
    days = [date(2026 + step, 1, 15) for step in (1, 2, 5)]
    forwards = [
        FxForward(
            day, forward_outright(SPOT, domestic, synthetic, day, spot_day=SPOT_DATE)
        )
        for day in days
    ]
    implied = implied_curve(
        SPOT,
        forwards,
        domestic,
        spot_day=SPOT_DATE,
        foreign_spot=synthetic.discount(SPOT_DATE),
    )
    for point in basis_curve(implied, foreign, days):
        assert point.spread == pytest.approx(0.0025, rel=1e-12)


def test_defaulting_the_spot_discount_factor_costs_basis_points() -> None:
    """The one quantity a strip of forwards cannot determine, and what it is worth.

    Parity fixes the ratio of foreign discount factors from the spot date, not
    their level, so the level has to come from the foreign curve. Taking the
    domestic one instead assumes the two currencies discount alike over the lag,
    and the error enters the basis divided by the maturity — worst at the short
    end, which is where the basis is actually quoted.
    """
    domestic, foreign = flat(DOMESTIC_RATE), flat(FOREIGN_RATE)
    synthetic = shifted(foreign, 0.0025)
    days = [date(2027, 1, 15), date(2031, 1, 15)]
    forwards = [
        FxForward(
            day, forward_outright(SPOT, domestic, synthetic, day, spot_day=SPOT_DATE)
        )
        for day in days
    ]
    careless = implied_curve(SPOT, forwards, domestic, spot_day=SPOT_DATE)
    errors = [
        implied_basis(careless, foreign, day).spread - 0.0025 for day in days
    ]
    # Over a basis point at one year, a fifth of that at five, same sign.
    assert 1.0e-4 < errors[0] < 2.0e-4
    assert 0.0 < errors[1] < 0.5e-4
    assert errors[0] > 4.0 * errors[1]


def test_the_implied_curve_refuses_a_strip_it_cannot_use() -> None:
    domestic = flat(DOMESTIC_RATE)
    day = date(2027, 1, 15)
    with pytest.raises(BadFx, match="at least one forward"):
        implied_curve(SPOT, [], domestic)
    with pytest.raises(BadFx, match="appears twice"):
        implied_curve(
            SPOT, [FxForward(day, 1.12), FxForward(day, 1.13)], domestic
        )
    with pytest.raises(BadFx, match="before the spot date"):
        implied_curve(
            SPOT, [FxForward(date(2026, 1, 16), 1.10)], domestic, spot_day=SPOT_DATE
        )
    with pytest.raises(BadFx, match="before the curve's reference date"):
        implied_curve(
            SPOT, [FxForward(day, 1.12)], domestic, spot_day=date(2025, 1, 1)
        )
    with pytest.raises(BadFx, match="discount factors are strictly positive"):
        implied_curve(
            SPOT, [FxForward(day, 1.12)], domestic, foreign_spot=0.0
        )


def test_there_is_no_basis_at_the_reference_date() -> None:
    foreign = flat(FOREIGN_RATE)
    synthetic = shifted(foreign, 0.0025)
    with pytest.raises(OffCurve, match="any spread fits"):
        implied_basis(synthetic, foreign, REFERENCE)
    other = DiscountCurve.from_discounts(
        date(2026, 2, 1), [(date(2027, 2, 1), 0.96)], basis=Basis.ACT_365F
    )
    with pytest.raises(BadFx, match="references"):
        implied_basis(other, foreign, date(2027, 1, 15))


# -- the par spread ------------------------------------------------------------


def test_one_curve_telescopes_to_nothing() -> None:
    """The identity the par spread is built on, asserted as round-off.

    A floating leg projected and discounted on the same curve, with notional
    received at the start and repaid at the end, is worth exactly nothing: each
    coupon is the difference of two neighbouring discount factors. So a
    cross-currency leg is a single-currency leg whose projection and discounting
    have come apart, and the spread measures how far.
    """
    foreign = flat(FOREIGN_RATE)
    schedule = quarterly()
    for accrual in (Basis.ACT_360, Basis.ACT_365F):
        spread = par_basis_spread(schedule, foreign, foreign, basis=accrual)
        assert abs(spread) < 1e-15, spread


def test_the_accrual_convention_and_the_compounding_pull_opposite_ways() -> None:
    """Two corrections, measured separately, because the net hides both.

    A spread paid quarterly on an accrual is worth more than the same number
    compounded continuously, which pushes the par spread above the curve basis.
    An ACT/360 accrual is 365/360 larger than an ACT/365F one, which pushes it
    back down by more. Reporting only the ACT/360 figure would leave a reader
    thinking the correction had one sign.
    """
    foreign = flat(FOREIGN_RATE)
    synthetic = shifted(foreign, 0.0025)
    schedule = quarterly()
    on_365 = par_basis_spread(schedule, foreign, synthetic, basis=Basis.ACT_365F)
    on_360 = par_basis_spread(schedule, foreign, synthetic, basis=Basis.ACT_360)
    assert on_365 > 0.0025 > on_360
    assert (on_365 / 0.0025 - 1.0) == pytest.approx(0.00785, abs=5e-5)
    assert (on_360 / 0.0025 - 1.0) == pytest.approx(-0.00595, abs=5e-5)
    # And the two differ by exactly the ratio of the day counts.
    assert on_360 == pytest.approx(on_365 * 360.0 / 365.0, rel=2e-4)


def test_the_gap_is_proportional_to_the_basis_rather_than_quadratic() -> None:
    """Measured across a decade of basis sizes, because the shape says what it is.

    A convention error scales with the quantity; an approximation error does
    not. The ratio of the gap to the basis is flat to within a sixth over a
    tenfold range, which is the signature of the former.
    """
    foreign = flat(FOREIGN_RATE)
    schedule = quarterly()
    ratios = []
    for basis in (0.0010, 0.0025, 0.0100):
        spread = par_basis_spread(
            schedule, foreign, shifted(foreign, basis), basis=Basis.ACT_360
        )
        ratios.append((spread - basis) / basis)
    assert all(-0.007 < ratio < -0.004 for ratio in ratios)
    assert max(ratios) - min(ratios) < 0.0015


def test_a_term_structure_is_averaged_over_the_forward_basis() -> None:
    """Not over the zero basis, and the difference is most of the answer.

    With a zero basis rising from 10 to 40 basis points over five years, the
    forward basis reaches 70 — differentiating ``b(t) t`` adds ``t b'(t)`` — and
    the par spread is the annuity-weighted average of *that*. A reader who
    averages the zero basis over time, or reads its short end, is not making a
    small error.
    """
    foreign = flat(FOREIGN_RATE)
    synthetic = term_shifted(foreign)
    schedule = quarterly()
    spread = par_basis_spread(schedule, foreign, synthetic, basis=Basis.ACT_360)
    assert spread == pytest.approx(0.003902, abs=2e-5)
    # The zero basis at the far end, the time average, and the short end.
    far = implied_basis(synthetic, foreign, date(2031, 1, 15)).spread
    near = implied_basis(synthetic, foreign, date(2026, 4, 15)).spread
    assert far == pytest.approx(0.0040, rel=1e-3)
    assert spread / 0.0025 - 1.0 > 0.35
    assert spread / near - 1.0 > 2.0


def test_the_par_spread_refuses_what_it_cannot_price() -> None:
    foreign = flat(FOREIGN_RATE)
    schedule = quarterly()
    other = DiscountCurve.from_discounts(
        date(2026, 2, 1), [(date(2032, 2, 1), 0.85)], basis=Basis.ACT_365F
    )
    with pytest.raises(BadFx, match="references"):
        par_basis_spread(schedule, foreign, other, basis=Basis.ACT_360)
    with pytest.raises(OffCurve):
        par_basis_spread(
            generate(
                SPOT_DATE,
                date(2099, 1, 19),
                Frequency.ANNUAL,
                calendar=WEEKENDS_ONLY,
            ),
            foreign,
            foreign,
            basis=Basis.ACT_360,
        )
