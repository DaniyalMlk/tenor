"""Tests for bond pricing, yield and risk.

Everything with a closed form is checked against one written out here from the
textbook formula rather than against the library's own arithmetic. That matters
more than usual for duration: differencing the price to check the derivative
only proves the derivative was differentiated consistently, and the standard
mistake - an exponent off by one period - is consistent with itself.

The closed forms used, for a bond settling on a coupon date with ``n`` whole
periods remaining, periodic coupon ``c`` and periodic yield ``r``, per 100:

    price    = c * (1 - (1 + r)^-n) / r + 100 * (1 + r)^-n
    Macaulay = (1 + r)/r - (1 + r + n(c - r)) / (c((1 + r)^n - 1) + r)

both in periods, both independent of anything in the library.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tenor.bond import Accrual, BadBond, Bond
from tenor.bootstrap import bootstrap
from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis, year_fraction
from tenor.instruments import Deposit, Swap
from tenor.rates import Compounding
from tenor.schedule import Frequency

ISSUE = date(2021, 1, 15)
MATURITY = date(2031, 1, 15)


def bond(
    coupon: float = 0.05,
    *,
    basis: Basis = Basis.THIRTY_360_BOND,
    frequency: Frequency = Frequency.SEMI_ANNUAL,
    accrual: Accrual = Accrual.DAY_COUNT,
) -> Bond:
    return Bond(ISSUE, MATURITY, coupon, frequency, basis, accrual=accrual)


def annuity_price(periodic_coupon: float, periodic_yield: float, periods: int) -> float:
    """The textbook closed form, written out rather than called."""
    discounted = (1.0 + periodic_yield) ** -periods
    return periodic_coupon * (1.0 - discounted) / periodic_yield + 100.0 * discounted


def closed_form_macaulay(
    periodic_coupon: float, periodic_yield: float, periods: int
) -> float:
    """Macaulay duration in periods, from the closed form."""
    coupon = periodic_coupon / 100.0
    growth = 1.0 + periodic_yield
    return (growth) / periodic_yield - (
        growth + periods * (coupon - periodic_yield)
    ) / (coupon * (growth**periods - 1.0) + periodic_yield)


# -- price --------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.001, 0.02, 0.05, 0.06, 0.12])
def test_the_price_matches_the_annuity_closed_form(rate: float) -> None:
    """On a coupon date, where the closed form applies exactly."""
    subject = bond(0.05)
    expected = annuity_price(2.5, rate / 2.0, 20)
    # Relative rather than absolute, because at a 0.1% yield the closed form is
    # the less accurate of the two: its ``(1 - (1 + r)**-n) / r`` cancels to
    # about fourteen significant figures, where summing the flows does not.
    assert subject.dirty_price(rate, ISSUE) == pytest.approx(expected, rel=1e-13)


def test_a_bond_at_its_coupon_is_worth_par() -> None:
    """The oldest check there is, and it catches an exponent off by one."""
    subject = bond(0.05)
    assert subject.dirty_price(0.05, ISSUE) == pytest.approx(100.0, abs=1e-12)
    assert subject.clean_price(0.05, ISSUE) == pytest.approx(100.0, abs=1e-12)


def test_price_falls_as_yield_rises() -> None:
    subject = bond(0.05)
    prices = [subject.dirty_price(rate / 1000.0, ISSUE) for rate in range(1, 120)]
    assert prices == sorted(prices, reverse=True)


def test_a_yield_below_minus_the_frequency_is_refused() -> None:
    """Past that point the per-period growth factor is not positive."""
    with pytest.raises(BadBond, match="not positive"):
        bond(0.05).dirty_price(-2.5, ISSUE)


# -- accrued interest ---------------------------------------------------------


def test_accrued_is_zero_on_a_coupon_date() -> None:
    for convention in Accrual:
        assert bond(0.05, accrual=convention).accrued(ISSUE) == 0.0
        assert bond(0.05, accrual=convention).accrued(date(2021, 7, 15)) == 0.0


def test_the_two_accrual_conventions_agree_under_thirty_360() -> None:
    """A semi-annual thirty-360 period is exactly half a year by construction."""
    settlement = date(2021, 4, 15)
    direct = bond(0.05, basis=Basis.THIRTY_360_BOND, accrual=Accrual.DAY_COUNT)
    fraction = bond(0.05, basis=Basis.THIRTY_360_BOND, accrual=Accrual.PERIOD_FRACTION)
    assert direct.accrued(settlement) == pytest.approx(1.25, abs=1e-15)
    assert fraction.accrued(settlement) == pytest.approx(1.25, abs=1e-15)
    assert direct.accrued(settlement) == fraction.accrued(settlement)


def test_the_two_accrual_conventions_disagree_under_an_actual_basis() -> None:
    """And by enough to matter, which is why they have to be named.

    Three months into a 181-day period on a 5% coupon, the two differ by just
    over a cent per 100 - about 0.8% of the accrued interest. The ratio is
    exactly twice the period's actual length over 365, which is the whole
    content of the disagreement.
    """
    settlement = date(2021, 4, 15)
    direct = bond(0.05, basis=Basis.ACT_365F, accrual=Accrual.DAY_COUNT)
    fraction = bond(0.05, basis=Basis.ACT_365F, accrual=Accrual.PERIOD_FRACTION)
    assert direct.accrued(settlement) == pytest.approx(90.0 / 365.0 * 5.0, abs=1e-12)
    assert fraction.accrued(settlement) == pytest.approx(2.5 * 90.0 / 181.0, abs=1e-12)
    difference = fraction.accrued(settlement) - direct.accrued(settlement)
    assert difference == pytest.approx(0.01022, abs=1e-5)
    assert direct.accrued(settlement) / fraction.accrued(settlement) == pytest.approx(
        2.0 * 181.0 / 365.0, abs=1e-12
    )


def test_accrued_grows_through_the_period_and_resets() -> None:
    subject = bond(0.05)
    through = [
        subject.accrued(date(2021, 1, 15) + _days(step)) for step in range(0, 180, 15)
    ]
    assert through == sorted(through)
    assert through[0] == 0.0
    assert subject.accrued(date(2021, 7, 15)) == 0.0


def _days(count: int) -> timedelta:
    return timedelta(days=count)


def test_dirty_is_clean_plus_accrued() -> None:
    settlement = date(2021, 4, 15)
    subject = bond(0.05)
    assert subject.dirty_price(0.06, settlement) - subject.clean_price(
        0.06, settlement
    ) == pytest.approx(subject.accrued(settlement), abs=1e-14)


# -- yield --------------------------------------------------------------------


@pytest.mark.parametrize("settlement", [ISSUE, date(2021, 4, 15), date(2025, 9, 30)])
@pytest.mark.parametrize("rate", [0.005, 0.043, 0.09])
def test_yield_and_price_round_trip(settlement: date, rate: float) -> None:
    subject = bond(0.05)
    solved = subject.yield_from_clean(subject.clean_price(rate, settlement), settlement)
    assert solved.converged
    assert solved.value == pytest.approx(rate, abs=1e-12)


def test_the_yield_is_solved_against_the_dirty_price() -> None:
    """Mid-period, solving against the clean price is 18 basis points out.

    On a coupon date the two are identical, because accrued is zero - which is
    exactly why a yield solver written against the clean price passes every
    test anybody writes and is wrong on every day but one in six months.
    """
    subject = bond(0.05)
    settlement = date(2021, 4, 15)
    correct = subject.yield_from_clean(
        subject.clean_price(0.06, settlement), settlement
    )
    assert correct.value == pytest.approx(0.06, abs=1e-12)

    against_clean = subject.yield_from_dirty(
        subject.clean_price(0.06, settlement), settlement
    )
    assert against_clean.value - 0.06 == pytest.approx(0.0018, abs=1e-4)

    # And the mistake disappears entirely on a coupon date.
    on_coupon = subject.yield_from_dirty(subject.clean_price(0.06, ISSUE), ISSUE)
    assert on_coupon.value == pytest.approx(0.06, abs=1e-12)


def test_the_yield_solve_reports_its_convergence() -> None:
    subject = bond(0.05)
    solved = subject.yield_from_clean(96.5, ISSUE)
    assert solved.converged
    assert solved.iterations > 0
    assert abs(solved.residual) < 1e-10
    assert "converged" in str(solved)


def test_a_dirty_price_no_yield_can_produce_is_refused() -> None:
    with pytest.raises(BadBond, match="no yield discounts them"):
        bond(0.05).yield_from_dirty(0.0, ISSUE)


def test_the_yield_solve_tries_an_ordinary_bracket_before_the_guaranteed_one() -> None:
    """Fast for real quotes, and still correct for absurd ones.

    The bracket that must contain the yield runs to the pole at ``y = -f``,
    where the price is near-vertical and interpolation is useless - so Brent
    bisects most of a range no bond is quoted in. Trying an ordinary range
    first cuts a fifteen-year bond at 96.5 from 54 iterations to 15. The wide
    bracket is still there and still works, and this pins both halves.
    """
    subject = Bond(date(2021, 1, 4), date(2036, 1, 4), 0.05)
    settlement = date(2021, 4, 15)

    ordinary = subject.yield_from_dirty(96.5, settlement)
    assert ordinary.converged
    assert ordinary.iterations < 25
    assert ordinary.bracket == (-0.5, 1.0)

    # Past a dirty price of about 500,000 the narrow bracket runs out and the
    # guaranteed one takes over. Nothing real gets here; correctness does.
    extreme = subject.yield_from_dirty(1e6, settlement)
    assert extreme.converged
    assert extreme.value < -0.5
    assert extreme.bracket[0] < -1.9
    assert subject.dirty_price(extreme.value, settlement) == pytest.approx(
        1e6, rel=1e-9
    )


# -- duration and convexity ---------------------------------------------------


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.06, 0.11])
def test_macaulay_duration_matches_its_closed_form(rate: float) -> None:
    """In periods from the textbook formula, converted to years here."""
    subject = bond(0.05)
    expected = closed_form_macaulay(2.5, rate / 2.0, 20) / 2.0
    assert subject.macaulay_duration(rate, ISSUE) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("rate", [0.001, 0.03, 0.08])
def test_a_zero_coupon_bonds_duration_is_its_maturity(rate: float) -> None:
    """Exactly, at any yield. The test that catches an exponent off by one."""
    zero = Bond(ISSUE, MATURITY, 0.0, Frequency.SEMI_ANNUAL, Basis.THIRTY_360_BOND)
    assert zero.macaulay_duration(rate, ISSUE) == pytest.approx(10.0, abs=1e-12)
    assert zero.modified_duration(rate, ISSUE) == pytest.approx(
        10.0 / (1.0 + rate / 2.0), abs=1e-12
    )


def test_modified_duration_is_macaulay_over_one_plus_the_periodic_yield() -> None:
    subject = bond(0.05)
    for rate in (0.02, 0.05, 0.09):
        assert subject.modified_duration(rate, ISSUE) == pytest.approx(
            subject.macaulay_duration(rate, ISSUE) / (1.0 + rate / 2.0), abs=1e-14
        )


@pytest.mark.parametrize("settlement", [ISSUE, date(2021, 4, 15), date(2028, 2, 1)])
def test_the_analytic_derivatives_agree_with_differencing_the_price(
    settlement: date,
) -> None:
    """The cross-check, in the direction that is evidence rather than tautology.

    Modified duration and convexity are computed in closed form from the flow
    weights; here they are compared against a central difference of the price
    function, which shares only the price with them.
    """
    subject = bond(0.05)
    rate = 0.043
    price = subject.dirty_price(rate, settlement)

    # Different bumps for the two derivatives, which is not fussiness. A second
    # difference divides the rounding in the price by the square of the bump,
    # so at 1e-6 the noise alone is about one part in ten thousand - larger
    # than the discrepancy the test is meant to detect. At 1e-4 the rounding
    # and the truncation are both around 1e-8.
    first = 1e-6
    up = subject.dirty_price(rate + first, settlement)
    down = subject.dirty_price(rate - first, settlement)
    assert subject.modified_duration(rate, settlement) == pytest.approx(
        -(up - down) / (2.0 * first * price), abs=1e-7
    )

    second = 1e-4
    up = subject.dirty_price(rate + second, settlement)
    down = subject.dirty_price(rate - second, settlement)
    assert subject.convexity(rate, settlement) == pytest.approx(
        (up + down - 2.0 * price) / (second * second * price), rel=1e-6
    )


def test_a_longer_bond_has_more_duration_and_more_convexity() -> None:
    short = Bond(ISSUE, date(2024, 1, 15), 0.05)
    long = Bond(ISSUE, date(2051, 1, 15), 0.05)
    assert short.modified_duration(0.05, ISSUE) < long.modified_duration(0.05, ISSUE)
    assert short.convexity(0.05, ISSUE) < long.convexity(0.05, ISSUE)


def test_a_lower_coupon_lengthens_duration_at_the_same_maturity() -> None:
    """Weight moves towards the redemption, which is the whole mechanism."""
    low = bond(0.01).macaulay_duration(0.05, ISSUE)
    high = bond(0.09).macaulay_duration(0.05, ISSUE)
    assert low > high


# -- basis point values -------------------------------------------------------


def test_pv01_is_close_to_but_not_equal_to_the_duration_approximation() -> None:
    """The convexity term is the difference, and it is the reason to quote pv01.

    Modified duration times price times a basis point is the linear estimate.
    The repriced number includes the curvature, and on a ten-year 5% bond at
    par the two differ by 4.7 parts in ten thousand - small, and exactly the
    sort of small a hedge accumulates.
    """
    subject = bond(0.05)
    rate = 0.05
    linear = (
        subject.modified_duration(rate, ISSUE) * subject.dirty_price(rate, ISSUE) * 1e-4
    )
    repriced = subject.pv01(rate, ISSUE)
    assert abs(repriced - linear) / linear == pytest.approx(4.72e-4, rel=1e-2)
    assert repriced < linear  # convexity works in the holder's favour


def test_dv01_and_pv01_differ_mostly_because_of_compounding() -> None:
    """The finding that a flat curve makes visible.

    The usual explanation for the gap is the curve's shape. On a *flat* curve
    the shape explanation predicts no gap at all, and there is one: 1.5%,
    which is ``exp(z/2)`` - the factor between a basis point of continuously
    compounded rate and a basis point of the semi-annual rate that discounts
    identically. Applying the shift in the bond's own compounding closes it to
    the convexity term.
    """
    settlement = date(2021, 1, 4)
    subject = Bond(settlement, date(2036, 1, 4), 0.05)
    flat = DiscountCurve.from_zeros(
        settlement, [(date(2041, 1, 7), 0.03)], basis=Basis.ACT_365F
    )
    rate = subject.curve_yield(flat, settlement).value

    continuous = subject.pv01(rate, settlement) / subject.dv01(flat, settlement)
    semi_annual = subject.pv01(rate, settlement) / subject.dv01(
        flat, settlement, compounding=Compounding.SEMI_ANNUAL
    )
    # Measured: 1.540% apart under a continuous shift, 0.050% under a matched
    # one. The prediction from compounding alone accounts for all but the last
    # five hundredths of a percent, which is the convexity.
    assert continuous == pytest.approx(0.984602, abs=1e-5)
    assert semi_annual == pytest.approx(0.999497, abs=1e-5)
    assert continuous == pytest.approx(math.exp(-0.03 / 2.0), abs=1e-3)
    assert abs(1.0 - continuous) > 25.0 * abs(1.0 - semi_annual)


def test_dv01_and_pv01_also_differ_because_of_the_curves_shape() -> None:
    """The smaller effect, isolated by matching the compounding conventions.

    With the shift applied in the bond's own compounding, what is left is the
    shape: a yield is a weighted average of the curve over the bond's flows,
    and shifting every zero rate by a basis point moves that average by a basis
    point only if the curve is flat. On this curve it is worth +1.008% on a
    fifteen-year bullet.

    And then the sting. The compounding effect on this same bond is -1.540%.
    The two have opposite signs and very nearly cancel: leave the shift
    continuous, make *both* mistakes at once, and the two numbers come out
    0.226% apart - a quarter of the smaller error and a seventh of the larger.
    The discrepancy looks most negligible exactly where it is least
    understood, which is a reasonable description of how the shortcut survives.
    """
    settlement = date(2021, 1, 4)
    subject = Bond(settlement, date(2036, 1, 4), 0.05)
    sloped = sloped_curve(settlement)
    rate = subject.curve_yield(sloped, settlement).value
    matched = subject.pv01(rate, settlement) / subject.dv01(
        sloped, settlement, compounding=Compounding.SEMI_ANNUAL
    )
    continuous = subject.pv01(rate, settlement) / subject.dv01(sloped, settlement)
    assert matched == pytest.approx(1.010080, abs=1e-5)
    assert continuous == pytest.approx(0.997739, abs=1e-5)
    # Opposite signs, and the combined error is smaller than either alone.
    assert (matched - 1.0) > 0.0 > (continuous - 1.0)
    assert abs(continuous - 1.0) < abs(matched - 1.0)


def test_effective_duration_tracks_modified_duration_on_a_flat_curve() -> None:
    """Different definitions, the same number where the definitions coincide."""
    settlement = date(2021, 1, 4)
    subject = Bond(settlement, date(2036, 1, 4), 0.05)
    flat = DiscountCurve.from_zeros(
        settlement, [(date(2041, 1, 7), 0.03)], basis=Basis.ACT_365F
    )
    rate = subject.curve_yield(flat, settlement).value
    effective = subject.effective_duration(
        flat, settlement, compounding=Compounding.SEMI_ANNUAL
    )
    assert effective == pytest.approx(subject.modified_duration(rate, settlement), rel=1e-3)


def test_effective_convexity_is_positive_and_of_the_right_size() -> None:
    settlement = date(2021, 1, 4)
    subject = Bond(settlement, date(2036, 1, 4), 0.05)
    flat = DiscountCurve.from_zeros(
        settlement, [(date(2041, 1, 7), 0.03)], basis=Basis.ACT_365F
    )
    rate = subject.curve_yield(flat, settlement).value
    effective = subject.effective_convexity(
        flat, settlement, compounding=Compounding.SEMI_ANNUAL
    )
    assert effective > 0.0
    assert effective == pytest.approx(subject.convexity(rate, settlement), rel=5e-2)


# -- pricing on a curve -------------------------------------------------------


def sloped_curve(reference: date) -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(reference, date(2021, 7, 5), 0.002, Basis.ACT_360),
        Swap(reference, date(2023, 1, 4), 0.006),
        Swap(reference, date(2026, 1, 5), 0.015),
        Swap(reference, date(2031, 1, 6), 0.022),
        Swap(reference, date(2041, 1, 7), 0.027),
    ]
    return bootstrap(reference, quotes, basis=Basis.ACT_365F).curve


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_a_bond_on_a_flat_curve_prices_at_its_equivalent_yield(
    interpolation: Interpolation,
) -> None:
    """Curve pricing and yield pricing are two routes to one number.

    On a flat continuously compounded curve the equivalent yield is known:
    it is the semi-annual rate that discounts the same way. Both routes have to
    agree, and they share no code past the cash flows.
    """
    settlement = date(2021, 1, 4)
    # An exact ten years, so the coupon schedule has no stub. With one, the
    # street convention discounts the stub's own coupon by a *full* period -
    # which is what it means and not what this test is measuring.
    subject = Bond(settlement, date(2031, 1, 4), 0.04)
    flat = DiscountCurve.from_zeros(
        settlement,
        [(date(2041, 1, 7), 0.025)],
        basis=Basis.ACT_365F,
        interpolation=interpolation,
    )
    from_curve = subject.price_from_curve(flat, settlement)
    equivalent = subject.curve_yield(flat, settlement)
    assert equivalent.converged
    assert subject.dirty_price(equivalent.value, settlement) == pytest.approx(
        from_curve, abs=1e-10
    )
    # The equivalent yield is near the continuous rate, converted.
    assert equivalent.value == pytest.approx(
        2.0 * (math.exp(0.025 / 2.0) - 1.0), abs=5e-4
    )


def test_pricing_off_a_curve_is_forward_to_settlement() -> None:
    """Not a present value today, which is a different and smaller number."""
    settlement = date(2022, 1, 4)
    subject = Bond(date(2021, 1, 4), date(2031, 1, 6), 0.04)
    curve = sloped_curve(date(2021, 1, 4))
    forward = subject.price_from_curve(curve, settlement)
    today = sum(
        flow.amount * curve.discount(flow.day) for flow in subject.cashflows(settlement)
    )
    assert forward > today
    assert forward == pytest.approx(today / curve.discount(settlement), rel=1e-14)


def test_a_bond_on_a_rising_curve_yields_between_its_ends() -> None:
    settlement = date(2021, 1, 4)
    curve = sloped_curve(settlement)
    subject = Bond(settlement, date(2031, 1, 6), 0.04)
    solved = subject.curve_yield(curve, settlement)
    short = curve.zero_rate(date(2023, 1, 4))
    long = curve.zero_rate(date(2031, 1, 6))
    assert short < solved.value < long


# -- refusals -----------------------------------------------------------------


def test_a_bond_that_is_not_a_bond_is_refused() -> None:
    with pytest.raises(BadBond, match="not after"):
        Bond(MATURITY, ISSUE, 0.05)
    with pytest.raises(BadBond, match="pays nothing back"):
        Bond(ISSUE, MATURITY, 0.05, redemption=0.0)


def test_settlement_outside_the_bonds_life_is_refused() -> None:
    subject = bond(0.05)
    with pytest.raises(BadBond, match="before it was issued"):
        subject.accrued(date(2020, 1, 1))
    with pytest.raises(BadBond, match="matured"):
        subject.dirty_price(0.05, MATURITY)
    with pytest.raises(BadBond, match="matured"):
        subject.dirty_price(0.05, date(2032, 1, 1))


def test_the_flow_due_on_the_settlement_date_belongs_to_the_seller() -> None:
    """Settling on a coupon date: that coupon is gone, and a full period runs."""
    subject = bond(0.05)
    flows = subject.cashflows(date(2021, 7, 15))
    assert len(flows) == 19
    assert flows[0].periods == pytest.approx(1.0)
    assert subject.accrued(date(2021, 7, 15)) == 0.0
    assert flows[-1].redemption
    assert flows[-1].amount == pytest.approx(102.5)


def test_the_schedule_is_generated_once() -> None:
    subject = bond(0.05)
    assert subject.schedule() is subject.schedule()


@pytest.mark.parametrize(
    "frequency", [Frequency.ANNUAL, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY]
)
def test_every_frequency_prices_to_par_at_its_own_coupon(frequency: Frequency) -> None:
    subject = Bond(ISSUE, MATURITY, 0.05, frequency, Basis.THIRTY_360_BOND)
    assert subject.dirty_price(0.05, ISSUE) == pytest.approx(100.0, abs=1e-12)
    periods = 10 * frequency.value
    assert subject.dirty_price(0.07, ISSUE) == pytest.approx(
        annuity_price(5.0 / frequency.value, 0.07 / frequency.value, periods), abs=1e-12
    )


def test_a_period_fraction_bond_still_prices_to_par_on_a_coupon_date() -> None:
    """The accrual convention moves accrued interest, not the dirty price."""
    direct = bond(0.05, basis=Basis.ACT_365F, accrual=Accrual.DAY_COUNT)
    fraction = bond(0.05, basis=Basis.ACT_365F, accrual=Accrual.PERIOD_FRACTION)
    settlement = date(2021, 4, 15)
    assert direct.dirty_price(0.06, settlement) == pytest.approx(
        fraction.dirty_price(0.06, settlement), abs=1e-15
    )
    assert direct.clean_price(0.06, settlement) != pytest.approx(
        fraction.clean_price(0.06, settlement), abs=1e-6
    )


def test_the_period_remaining_runs_from_one_down_to_zero() -> None:
    subject = bond(0.05)
    assert subject.period_remaining(ISSUE) == pytest.approx(1.0)
    assert subject.period_remaining(date(2021, 4, 15)) == pytest.approx(0.5)
    assert subject.period_remaining(date(2021, 7, 14)) < 0.02
    assert subject.period_remaining(date(2021, 7, 15)) == pytest.approx(1.0)


def test_the_exponent_counts_coupon_periods_not_year_fractions() -> None:
    """A short first period does not change what the later exponents are.

    Both bonds pay on the same dates from the second coupon onwards, so their
    second and subsequent flows sit at the same period counts. A year fraction
    would put them at different places, because one bond's first period is
    shorter than the other's.
    """
    regular = Bond(date(2021, 1, 15), date(2026, 1, 15), 0.05)
    stubbed = Bond(date(2021, 3, 1), date(2026, 1, 15), 0.05)
    assert stubbed.schedule().has_stub
    settlement = date(2021, 7, 15)
    regular_flows = regular.cashflows(settlement)
    stubbed_flows = stubbed.cashflows(settlement)
    assert len(regular_flows) == len(stubbed_flows)
    for left, right in zip(regular_flows, stubbed_flows, strict=True):
        assert left.periods == pytest.approx(right.periods)
        assert left.day == right.day
    # The year fractions to those same dates are equal here too, because both
    # bonds are past their stubs; the exponent is what is being pinned.
    assert regular_flows[0].periods == pytest.approx(1.0)


def test_the_curve_and_the_yield_agree_on_a_bootstrapped_curve() -> None:
    """End to end: quotes to curve to price to yield and back to price."""
    settlement = date(2021, 1, 4)
    curve = sloped_curve(settlement)
    subject = Bond(settlement, date(2036, 1, 4), 0.045)
    price = subject.price_from_curve(curve, settlement)
    solved = subject.curve_yield(curve, settlement)
    assert solved.converged
    assert subject.dirty_price(solved.value, settlement) == pytest.approx(
        price, abs=1e-10
    )
    assert year_fraction(settlement, date(2036, 1, 4), Basis.ACT_365F) > 14.0
