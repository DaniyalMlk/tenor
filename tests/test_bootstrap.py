"""Tests for the instruments and the curve built from them.

The identity under test is that every quote used to build a curve prices to par
off the finished curve. It is asserted for all three interpolations, which is
the whole point: a bootstrapper written against a local interpolation passes
that test and quietly fails it for a smooth one, and the failure is a basis
point in the middle of each interval rather than anything that raises.

There is also a round trip through independent arithmetic. A flat curve at a
known rate is used to compute what each instrument would quote at, the quotes
are handed to the bootstrapper with the original curve thrown away, and the
rebuilt curve has to be the one we started with. That check does not share code
with the bootstrapper in either direction.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor.bootstrap import BootstrapFailed, bootstrap
from tenor.calendar import WEEKENDS_ONLY
from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis, year_fraction
from tenor.instruments import (
    BadInstrument,
    Deposit,
    Future,
    Swap,
    ho_lee_convexity,
    maturity_of,
)
from tenor.rates import Compounding

REFERENCE = date(2021, 1, 4)


def strip() -> list[Deposit | Future | Swap]:
    """A plausible screen: deposits at the front, futures, then par swaps."""
    return [
        Deposit(REFERENCE, date(2021, 4, 5), 0.0030, Basis.ACT_360, "3m deposit"),
        Deposit(REFERENCE, date(2021, 7, 5), 0.0035, Basis.ACT_360, "6m deposit"),
        Future(date(2021, 9, 15), date(2021, 12, 15), 99.55, Basis.ACT_360, label="Sep21"),
        Future(date(2021, 12, 15), date(2022, 3, 16), 99.45, Basis.ACT_360, label="Dec21"),
        Swap(REFERENCE, date(2023, 1, 4), 0.0075, label="2y swap"),
        Swap(REFERENCE, date(2026, 1, 5), 0.0140, label="5y swap"),
        Swap(REFERENCE, date(2031, 1, 6), 0.0195, label="10y swap"),
    ]


# -- the instruments ----------------------------------------------------------


def test_a_deposit_prices_to_par_on_the_curve_that_implies_it() -> None:
    """Built by hand from the definition, with no bootstrapper involved."""
    deposit = Deposit(REFERENCE, date(2021, 7, 5), 0.0035, Basis.ACT_360)
    accrual = year_fraction(REFERENCE, date(2021, 7, 5), Basis.ACT_360)
    assert accrual == pytest.approx(182.0 / 360.0)
    factor = 1.0 / (1.0 + 0.0035 * accrual)
    curve = DiscountCurve.from_discounts(
        REFERENCE, [(date(2021, 7, 5), factor)], basis=Basis.ACT_365F
    )
    assert deposit.par_error(curve) == pytest.approx(0.0, abs=1e-16)
    # And a curve that is a basis point away does not price it to par.
    off = DiscountCurve.from_discounts(
        REFERENCE,
        [(date(2021, 7, 5), 1.0 / (1.0 + 0.0036 * accrual))],
        basis=Basis.ACT_365F,
    )
    assert abs(deposit.par_error(off)) > 1e-7


def test_a_deposit_uses_simple_interest_not_compounded() -> None:
    """Six months at 3.5%: the two conventions differ inside the bid-offer."""
    deposit = Deposit(REFERENCE, date(2021, 7, 5), 0.035, Basis.ACT_360)
    accrual = deposit.accrual
    simple = 1.0 / (1.0 + 0.035 * accrual)
    compounded = (1.0 + 0.035) ** -accrual
    assert simple != pytest.approx(compounded, abs=1e-6)
    curve = DiscountCurve.from_discounts(
        REFERENCE, [(date(2021, 7, 5), simple)], basis=Basis.ACT_365F
    )
    assert deposit.par_error(curve) == pytest.approx(0.0, abs=1e-16)


def test_the_futures_convexity_adjustment_reproduces_hulls_example() -> None:
    """1.2% volatility at eight years: 47.5 basis points, as published."""
    assert ho_lee_convexity(0.012, 8.0, 8.25) == pytest.approx(0.004752, abs=1e-9)


def test_the_convexity_adjustment_is_negligible_near_and_large_far() -> None:
    """The number people expect to be small, and is not, past a few years.

    Under a basis point on a contract starting in a year; fifty-one basis
    points on the same contract starting in ten. It grows with the product of
    the two maturities, so roughly with the square.
    """
    near = ho_lee_convexity(0.01, 1.0, 1.25)
    far = ho_lee_convexity(0.01, 10.0, 10.25)
    assert near == pytest.approx(0.0000625, abs=1e-12)
    assert far == pytest.approx(0.005125, abs=1e-12)
    assert near < 0.0001 < 0.005 < far


def test_the_adjustment_is_subtracted_from_the_futures_rate() -> None:
    """The sign that matters: a future implies a *lower* forward rate."""
    quoted = Future(date(2031, 1, 6), date(2031, 4, 6), 98.05)
    adjusted = Future(
        date(2031, 1, 6),
        date(2031, 4, 6),
        98.05,
        convexity=ho_lee_convexity(0.01, 10.0, 10.25),
    )
    assert quoted.futures_rate == pytest.approx(0.0195)
    assert adjusted.futures_rate == pytest.approx(0.0195)
    assert adjusted.forward_rate < quoted.forward_rate
    assert quoted.forward_rate - adjusted.forward_rate == pytest.approx(0.005125)


def test_a_negative_convexity_adjustment_is_refused() -> None:
    with pytest.raises(BadInstrument, match="never negative"):
        Future(date(2021, 3, 15), date(2021, 6, 15), 99.5, convexity=-0.0001)


def test_a_swaps_annuity_accrues_on_unadjusted_dates() -> None:
    """Accrual on the term sheet's dates, discounting on the payment dates.

    Accruing on the rolled dates instead moves the fixed leg by a day of
    interest whenever a period end lands on a weekend, which is most of them.
    """
    swap = Swap(REFERENCE, date(2026, 1, 5), 0.0140, calendar=WEEKENDS_ONLY)
    curve = flat(0.02)
    by_hand = sum(
        year_fraction(period.start, period.end, swap.basis)
        * curve.discount(period.payment)
        for period in swap.schedule()
    )
    assert swap.annuity(curve) == pytest.approx(by_hand, rel=1e-15)
    rolled = sum(
        year_fraction(period.adjusted_start, period.adjusted_end, swap.basis)
        * curve.discount(period.payment)
        for period in swap.schedule()
    )
    assert swap.annuity(curve) != pytest.approx(rolled, rel=1e-12)


def test_a_swap_at_its_own_par_rate_is_worth_nothing() -> None:
    curve = flat(0.02)
    quoted = Swap(REFERENCE, date(2031, 1, 6), 0.0, calendar=WEEKENDS_ONLY)
    par = quoted.par_rate(curve)
    at_par = Swap(REFERENCE, date(2031, 1, 6), par, calendar=WEEKENDS_ONLY)
    assert at_par.par_error(curve) == pytest.approx(0.0, abs=1e-15)
    # Under a flat 2% continuous curve the par swap rate is close to 2% but not
    # equal to it: the fixed leg is thirty-360 and pays semi-annually.
    assert 0.019 < par < 0.021
    assert par != pytest.approx(0.02, abs=1e-5)


def test_instruments_that_do_not_describe_a_trade_are_refused() -> None:
    with pytest.raises(BadInstrument, match="not after"):
        Deposit(date(2021, 7, 5), REFERENCE, 0.003)
    with pytest.raises(BadInstrument, match="not after"):
        Future(date(2021, 7, 5), REFERENCE, 99.5)
    with pytest.raises(BadInstrument, match="not after"):
        Swap(date(2031, 1, 6), REFERENCE, 0.02)
    with pytest.raises(BadInstrument, match="not negative"):
        ho_lee_convexity(-0.01, 1.0, 2.0)
    with pytest.raises(BadInstrument, match="forwards in time"):
        ho_lee_convexity(0.01, 2.0, 1.0)


def flat(rate: float) -> DiscountCurve:
    """A flat continuously compounded curve, built without the bootstrapper."""
    days = (
        date(2021, 7, 5),
        date(2023, 1, 4),
        date(2026, 1, 5),
        date(2031, 1, 6),
        date(2041, 1, 7),
    )
    return DiscountCurve.from_discounts(
        REFERENCE,
        [
            (day, math.exp(-rate * year_fraction(REFERENCE, day, Basis.ACT_365F)))
            for day in days
        ],
        basis=Basis.ACT_365F,
    )


# -- the identity -------------------------------------------------------------


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_every_instrument_reprices_off_the_finished_curve(
    interpolation: Interpolation,
) -> None:
    """The identity the exercise exists to satisfy, under every scheme.

    Parametrised over the whole enum rather than over the local schemes,
    because passing for those and failing for a smooth one is precisely the
    bug this bootstrapper is built to avoid.
    """
    built = bootstrap(REFERENCE, strip(), basis=Basis.ACT_365F, interpolation=interpolation)
    assert built.reprices(1e-12)
    assert built.worst_error < 1e-13
    assert len(built) == 7
    for error in built.par_errors():
        assert abs(error) < 1e-13


def test_a_single_sequential_pass_is_not_enough_for_a_smooth_interpolation() -> None:
    """The failure the sweeping exists to prevent, measured rather than asserted.

    One sequential pass leaves the September future mispriced by 2.6e-05 of
    present value on unit notional. Over its quarterly accrual that is about
    ten basis points of rate - not a rounding error, and not anything that
    raises on its own.
    """
    with pytest.raises(BootstrapFailed, match="did not settle") as failure:
        bootstrap(
            REFERENCE,
            strip(),
            basis=Basis.ACT_365F,
            interpolation=Interpolation.MONOTONE_CONVEX,
            max_sweeps=1,
        )
    message = str(failure.value)
    assert "Sep21" in message
    assert failure.value.instrument is not None
    assert failure.value.instrument.name == "Sep21"

    # The local schemes are already consistent after that same single pass,
    # which is why a bootstrapper written against them looks correct.
    for scheme in (Interpolation.LOG_LINEAR_DISCOUNT, Interpolation.LINEAR_ZERO):
        assert bootstrap(
            REFERENCE, strip(), basis=Basis.ACT_365F, interpolation=scheme, max_sweeps=1
        ).sweeps == 1


def test_the_sweep_count_is_reported_and_means_something() -> None:
    local = bootstrap(
        REFERENCE,
        strip(),
        basis=Basis.ACT_365F,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )
    smooth = bootstrap(
        REFERENCE,
        strip(),
        basis=Basis.ACT_365F,
        interpolation=Interpolation.MONOTONE_CONVEX,
    )
    assert local.sweeps == 1
    assert smooth.sweeps > local.sweeps
    assert all(one.converged for one in smooth.solutions)


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_the_quotes_come_back_out_of_the_curve(interpolation: Interpolation) -> None:
    """Each instrument re-quoted off the curve gives back what went in.

    A stronger statement than the present value being zero, and in the units
    people actually check: a swap's par rate, a deposit's simple rate, a
    future's forward rate.
    """
    built = bootstrap(REFERENCE, strip(), basis=Basis.ACT_365F, interpolation=interpolation)
    curve = built.curve
    for instrument in built.instruments:
        if isinstance(instrument, Swap):
            assert instrument.par_rate(curve) == pytest.approx(instrument.rate, abs=1e-12)
        elif isinstance(instrument, Deposit):
            assert _simple_rate(
                curve, instrument.start, instrument.maturity, instrument.basis
            ) == pytest.approx(instrument.rate, abs=1e-12)
        else:
            assert _simple_rate(
                curve, instrument.start, instrument.end, instrument.basis
            ) == pytest.approx(instrument.forward_rate, abs=1e-12)


def _simple_rate(curve: DiscountCurve, start: date, end: date, basis: Basis) -> float:
    """The simple rate the curve implies, quoted on the instrument's own basis.

    The curve returns a forward over *its* day count, and money market quotes
    are on theirs. The growth factor is what the two agree on, so the
    conversion goes through it: ``r * tau`` is invariant, not ``r``. Scaling
    the rate by the ratio the other way up is an error of ``(365/360)**2`` here,
    about three tenths of a basis point on a 30bp deposit, which is exactly the
    size of thing a repricing test is supposed to catch.
    """
    curve_rate = curve.forward_rate(start, end, Compounding.SIMPLE)
    curve_accrual = year_fraction(start, end, curve.basis)
    return curve_rate * curve_accrual / year_fraction(start, end, basis)


def test_a_flat_curve_rebuilds_itself_from_its_own_quotes() -> None:
    """A round trip through arithmetic the bootstrapper does not share.

    The quotes are computed from a known flat curve by the instrument code,
    the curve is discarded, and the bootstrapper has to find it again from the
    quotes alone. Both directions have to be right for this to close.
    """
    original = flat(0.025)
    quotes: list[Deposit | Future | Swap] = []
    for day in (date(2021, 7, 5), date(2022, 1, 4)):
        accrual = year_fraction(REFERENCE, day, Basis.ACT_360)
        factor = original.discount(day)
        quotes.append(Deposit(REFERENCE, day, (1.0 / factor - 1.0) / accrual, Basis.ACT_360))
    for maturity in (date(2023, 1, 4), date(2026, 1, 5), date(2031, 1, 6)):
        probe = Swap(REFERENCE, maturity, 0.0, calendar=WEEKENDS_ONLY)
        quotes.append(
            Swap(REFERENCE, maturity, probe.par_rate(original), calendar=WEEKENDS_ONLY)
        )

    rebuilt = bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F)
    assert rebuilt.reprices(1e-12)
    for quote in quotes:
        day = maturity_of(quote)
        assert rebuilt.curve.discount(day) == pytest.approx(
            original.discount(day), rel=1e-12
        )
        assert rebuilt.curve.zero_rate(day) == pytest.approx(0.025, abs=1e-12)


def test_the_convexity_adjustment_moves_the_curve_the_way_it_should() -> None:
    """Adjusting a future down lowers the forward, so the far zero rate falls."""
    plain = strip()
    adjusted = strip()
    for index, instrument in enumerate(adjusted):
        if isinstance(instrument, Future):
            start = year_fraction(REFERENCE, instrument.start, Basis.ACT_365F)
            end = year_fraction(REFERENCE, instrument.end, Basis.ACT_365F)
            adjusted[index] = Future(
                instrument.start,
                instrument.end,
                instrument.price,
                instrument.basis,
                convexity=ho_lee_convexity(0.01, start, end),
                label=instrument.label,
            )
    without = bootstrap(REFERENCE, plain, basis=Basis.ACT_365F)
    with_adjustment = bootstrap(REFERENCE, adjusted, basis=Basis.ACT_365F)
    day = date(2022, 3, 16)
    assert with_adjustment.curve.zero_rate(day) < without.curve.zero_rate(day)
    # Small at the front, as the adjustment is under a basis point there.
    difference = without.curve.zero_rate(day) - with_adjustment.curve.zero_rate(day)
    assert 0.0 < difference < 0.0001


# -- refusals -----------------------------------------------------------------


def test_a_curve_that_cannot_be_built_names_the_instrument() -> None:
    quotes = strip()
    quotes.append(Deposit(REFERENCE, date(2026, 1, 5), 0.0035, Basis.ACT_360, "clash"))
    with pytest.raises(BootstrapFailed, match="both mature on") as failure:
        bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F)
    assert failure.value.instrument is not None
    assert "clash" in str(failure.value) or "5y swap" in str(failure.value)


def test_an_instrument_in_the_past_is_refused() -> None:
    with pytest.raises(BootstrapFailed, match="says nothing about the future"):
        bootstrap(
            REFERENCE,
            [Deposit(date(2020, 1, 4), date(2020, 7, 4), 0.003)],
            basis=Basis.ACT_365F,
        )


def test_an_instrument_starting_before_the_reference_date_is_refused() -> None:
    with pytest.raises(BootstrapFailed, match="before the curve's reference date"):
        bootstrap(
            REFERENCE,
            [Deposit(date(2020, 7, 4), date(2021, 7, 5), 0.003)],
            basis=Basis.ACT_365F,
        )


def test_an_empty_strip_is_refused() -> None:
    with pytest.raises(BootstrapFailed, match="at least one instrument"):
        bootstrap(REFERENCE, [], basis=Basis.ACT_365F)


def test_a_quote_no_discount_factor_can_reach_is_refused() -> None:
    """A deposit rate so negative that the redemption exceeds the principal.

    There is no positive discount factor that prices it, and saying so beats
    returning the edge of the search range.
    """
    with pytest.raises(BootstrapFailed, match="prices"):
        bootstrap(
            REFERENCE,
            [Deposit(REFERENCE, date(2021, 7, 5), -4.0, Basis.ACT_360, "absurd")],
            basis=Basis.ACT_365F,
        )


def test_a_single_instrument_is_a_curve() -> None:
    built = bootstrap(
        REFERENCE,
        [Deposit(REFERENCE, date(2021, 7, 5), 0.0035, Basis.ACT_360)],
        basis=Basis.ACT_365F,
    )
    assert len(built) == 1
    assert built.reprices()
    assert built.sweeps == 1


def test_zero_sweeps_is_refused() -> None:
    with pytest.raises(BootstrapFailed, match="at least one sweep"):
        bootstrap(REFERENCE, strip(), basis=Basis.ACT_365F, max_sweeps=0)


def test_the_curve_is_ordered_however_the_quotes_arrive() -> None:
    """Sorting by maturity is the bootstrapper's job, not the caller's."""
    shuffled = list(reversed(strip()))
    built = bootstrap(REFERENCE, shuffled, basis=Basis.ACT_365F)
    maturities = [maturity_of(one) for one in built.instruments]
    assert maturities == sorted(maturities)
    assert built.reprices()
