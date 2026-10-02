"""Swaptions, caps and floors.

Two identities carry most of the weight here, because both are exact for any
volatility and so leave nowhere for an error to hide. A payer less a receiver at
one strike is the forward-starting swap; a cap less a floor is the same
statement for the strip. If the swaption and the caplets were reading the curve
even slightly differently — one accruing on adjusted dates and the other not,
one discounting at the period end and the other at the payment — the difference
would show up here as a residual proportional to the mistake rather than to
double precision.

The third check is Jensen's inequality: a strip of options is worth at least the
option on the strip. That one is *violated* in one of twenty-five cases, and
tracking down why was the useful part of writing this file. It is not a pricing
error; it is the day-count mismatch between the fixed leg's annuity and the
floating periods' weights, and the violation's size is predictable in closed
form from that mismatch to eight digits.

The rest is degenerate cases with known answers — zero volatility is intrinsic,
the at-the-money straddle is symmetric, a normal and a lognormal volatility
agree to leading order at the money and the gap grows with ``sigma sqrt(T)`` —
and refusals for the states the two models genuinely have no value at.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis, year_fraction
from tenor.multicurve import ForecastIndex, shifted_forecast
from tenor.options import (
    BadOption,
    Cap,
    Convention,
    Payoff,
    Swaption,
    bachelier,
    black,
    implied_volatility,
)
from tenor.schedule import Frequency

REFERENCE = date(2026, 1, 15)
EXPIRY = date(2036, 1, 15)
SWAP_END = date(2046, 1, 15)

OIS_QUOTES = [
    (date(2026, 4, 15), 0.0290),
    (date(2026, 7, 15), 0.0298),
    (date(2027, 1, 15), 0.0310),
    (date(2028, 1, 17), 0.0325),
    (date(2029, 1, 15), 0.0338),
    (date(2031, 1, 15), 0.0352),
    (date(2033, 1, 17), 0.0361),
    (date(2036, 1, 15), 0.0368),
    (date(2041, 1, 15), 0.0372),
    (date(2046, 1, 15), 0.0374),
]

CAP_START = date(2027, 1, 15)
CAP_END = date(2032, 1, 15)


@pytest.fixture
def discount() -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE,
        OIS_QUOTES,
        basis=Basis.ACT_365F,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )


@pytest.fixture
def projection(discount: DiscountCurve) -> DiscountCurve:
    return shifted_forecast(discount, 0.0020)


@pytest.fixture
def index() -> ForecastIndex:
    return ForecastIndex(tenor=Frequency.QUARTERLY, basis=Basis.ACT_360)


def swaption(
    strike: float,
    index: ForecastIndex,
    payoff: Payoff = Payoff.PAYER,
    *,
    expiry: date = EXPIRY,
    effective: date = EXPIRY,
    maturity: date = SWAP_END,
    frequency: Frequency = Frequency.SEMI_ANNUAL,
    basis: Basis = Basis.THIRTY_360_BOND,
) -> Swaption:
    return Swaption(
        expiry=expiry,
        effective=effective,
        maturity=maturity,
        strike=strike,
        payoff=payoff,
        index=index,
        frequency=frequency,
        basis=basis,
    )


def quarterly(
    strike: float, index: ForecastIndex, payoff: Payoff = Payoff.PAYER
) -> Swaption:
    """The five-year quarterly structure the cap is compared against."""
    return swaption(
        strike,
        index,
        payoff,
        expiry=CAP_START,
        effective=CAP_START,
        maturity=CAP_END,
        frequency=Frequency.QUARTERLY,
        basis=Basis.ACT_360,
    )


# -- the two formulas ---------------------------------------------------------


@pytest.mark.parametrize("payoff", list(Payoff))
def test_zero_volatility_is_the_intrinsic_value(payoff: Payoff) -> None:
    for forward, strike in ((0.04, 0.03), (0.04, 0.05), (0.04, 0.04)):
        expected = max(payoff.sign * (forward - strike), 0.0)
        assert black(forward, strike, 0.0, payoff) == expected
        assert bachelier(forward, strike, 0.0, payoff) == expected


def test_black_has_no_value_at_a_non_positive_rate() -> None:
    """Which is the reason the normal convention exists."""
    with pytest.raises(BadOption, match="cannot reach a non-positive rate"):
        black(-0.001, 0.01, 0.2, Payoff.PAYER)
    with pytest.raises(BadOption, match="cannot reach a non-positive rate"):
        black(0.01, -0.001, 0.2, Payoff.PAYER)


def test_bachelier_takes_the_rates_black_cannot() -> None:
    value = bachelier(-0.001, 0.002, 0.004, Payoff.PAYER)
    assert value > 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("total_vol", [-0.1, math.nan, math.inf])
def test_a_volatility_that_is_not_one_is_refused(total_vol: float) -> None:
    with pytest.raises(BadOption, match="not a number of one"):
        black(0.04, 0.04, total_vol, Payoff.PAYER)
    with pytest.raises(BadOption, match="not a number of one"):
        bachelier(0.04, 0.04, total_vol, Payoff.PAYER)


@pytest.mark.parametrize("convention", list(Convention))
def test_the_at_the_money_straddle_is_symmetric(convention: Convention) -> None:
    """Both halves of an at-the-money straddle are worth the same thing.

    Not a property of the market but of the formulas at ``F == K``, where the
    two cumulative normals are reflections of each other.
    """
    maker = black if convention is Convention.LOGNORMAL else bachelier
    total = 0.2 if convention is Convention.LOGNORMAL else 0.008
    payer = maker(0.04, 0.04, total, Payoff.PAYER)
    receiver = maker(0.04, 0.04, total, Payoff.RECEIVER)
    assert payer == pytest.approx(receiver, rel=1e-15)


@pytest.mark.parametrize("convention", list(Convention))
def test_the_at_the_money_straddle_is_the_leading_closed_form(
    convention: Convention,
) -> None:
    """``numeraire * vol * sqrt(2T/pi)``, which both conventions share.

    Checked at a small total volatility, where the leading term is the whole
    answer: this is what makes a normal volatility about a lognormal one times
    the forward at the money, and the test below measures where that stops.
    """
    total = 1e-4
    if convention is Convention.LOGNORMAL:
        value = black(0.04, 0.04, total, Payoff.PAYER)
        leading = 0.04 * total / math.sqrt(2.0 * math.pi)
    else:
        value = bachelier(0.04, 0.04, total, Payoff.PAYER)
        leading = total / math.sqrt(2.0 * math.pi)
    assert value == pytest.approx(leading, rel=1e-7)


# -- the swaption -------------------------------------------------------------


def test_the_swaption_is_the_annuity_times_a_call_on_the_forward_rate(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    probe = swaption(0.0, index)
    forward = probe.forward_rate(discount, projection)
    at_the_money = swaption(forward, index)
    annuity = at_the_money.annuity(discount)
    years = at_the_money.time_to_expiry(REFERENCE)

    assert annuity == pytest.approx(5.701580, rel=1e-5)
    assert forward == pytest.approx(0.04042513, rel=1e-6)
    assert at_the_money.value(discount, projection, 0.20) == pytest.approx(
        annuity * black(forward, forward, 0.20 * math.sqrt(years), Payoff.PAYER),
        rel=1e-14,
    )
    assert at_the_money.value(discount, projection, 0.20) == pytest.approx(
        0.05721523, rel=1e-6
    )


def test_a_swaption_at_no_volatility_is_its_intrinsic(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    for strike in (0.02, 0.04042513, 0.06):
        for payoff in Payoff:
            one = swaption(strike, index, payoff)
            assert one.value(discount, projection, 0.0) == pytest.approx(
                one.intrinsic(discount, projection), rel=1e-15
            )


def test_the_discount_curve_moves_a_swaption_a_hundred_times_the_rate(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The measurement the module exists to make.

    A par swap rate barely notices the discount curve. A swaption is the
    annuity times a call on that rate, and the annuity is nothing but discount
    factors, so the same shift moves the option by a hundred times as much —
    and the move decomposes into the annuity's own and the rate's.
    """
    probe = swaption(0.0, index)
    forward = probe.forward_rate(discount, projection)
    one = swaption(forward, index)

    shifted = discount.shifted(0.0100)
    base = one.value(discount, projection, 0.20)
    moved = one.value(shifted, projection, 0.20)
    rate_move = (
        probe.forward_rate(shifted, projection) - forward
    ) / forward
    annuity_move = (
        one.annuity(shifted) - one.annuity(discount)
    ) / one.annuity(discount)

    assert (moved - base) / base == pytest.approx(-0.13573, abs=1e-4)
    assert rate_move == pytest.approx(0.001254, abs=1e-5)
    assert annuity_move == pytest.approx(-0.13845, abs=1e-4)
    assert abs((moved - base) / base) / abs(rate_move) == pytest.approx(108.0, abs=2.0)
    # The option's move is the annuity's, less what the rate's rise claws back.
    assert (moved - base) / base == pytest.approx(annuity_move + rate_move, abs=2e-3)


@pytest.mark.parametrize(
    ("convention", "volatilities"),
    [
        (Convention.LOGNORMAL, (0.05, 0.20, 0.60)),
        (Convention.NORMAL, (0.002, 0.010, 0.030)),
    ],
)
def test_a_payer_less_a_receiver_is_the_forward_swap(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    convention: Convention,
    volatilities: tuple[float, ...],
) -> None:
    """Exact for any volatility, which is why it finds a misread curve.

    If the two legs disagreed about an accrual or a payment date by a day, the
    residual here would be proportional to that day rather than to the last
    bit of the annuity.
    """
    forward = swaption(0.0, index).forward_rate(discount, projection)
    worst = 0.0
    for strike in (0.02, 0.03, forward, 0.05, 0.07):
        payer = swaption(strike, index, Payoff.PAYER)
        receiver = swaption(strike, index, Payoff.RECEIVER)
        annuity = payer.annuity(discount)
        for volatility in volatilities:
            gap = payer.value(
                discount, projection, volatility, convention=convention
            ) - receiver.value(
                discount, projection, volatility, convention=convention
            )
            worst = max(worst, abs(gap - annuity * (forward - strike)) / annuity)
    assert worst < 1e-16


@pytest.mark.parametrize(
    ("years", "expiry", "effective", "maturity", "volatility", "agreement"),
    [
        (1.0, date(2027, 1, 15), date(2027, 1, 15), date(2037, 1, 15), 0.10, 4.2e-4),
        (10.0, EXPIRY, EXPIRY, SWAP_END, 0.10, 4.2e-3),
        (10.0, EXPIRY, EXPIRY, SWAP_END, 0.20, 1.6e-2),
    ],
)
def test_a_normal_volatility_is_the_lognormal_one_times_the_forward_to_leading_order(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    years: float,
    expiry: date,
    effective: date,
    maturity: date,
    volatility: float,
    agreement: float,
) -> None:
    """And only to leading order: the gap grows with ``sigma sqrt(T)``.

    Which is the thing to know before converting a quote. Both conventions give
    ``numeraire * vol * sqrt(2T/pi)`` at the money to first order, so the
    product is the right first guess and is 1.6% wrong by ten years at a 20%
    volatility.
    """
    probe = swaption(0.0, index, expiry=expiry, effective=effective, maturity=maturity)
    forward = probe.forward_rate(discount, projection)
    one = swaption(
        forward, index, expiry=expiry, effective=effective, maturity=maturity
    )
    premium = one.value(discount, projection, volatility) / one.annuity(discount)
    normal = implied_volatility(
        premium,
        forward,
        forward,
        one.time_to_expiry(REFERENCE),
        Payoff.PAYER,
        convention=Convention.NORMAL,
    )
    assert abs(normal / (volatility * forward) - 1.0) == pytest.approx(
        agreement, rel=0.1
    )
    assert normal < volatility * forward


def test_a_swaption_refuses_a_swap_that_ends_before_it_starts(
    index: ForecastIndex,
) -> None:
    with pytest.raises(BadOption, match="not after it starts"):
        swaption(0.04, index, maturity=date(2030, 1, 15), effective=date(2036, 1, 15))


def test_a_swaption_refuses_an_expiry_after_its_swap_begins(
    index: ForecastIndex,
) -> None:
    """A mid-curve swaption expires early, which is fine; this is the other way."""
    with pytest.raises(BadOption, match="already begun accruing"):
        swaption(0.04, index, expiry=date(2037, 1, 15), effective=EXPIRY)
    # And the legitimate direction is accepted.
    early = swaption(0.04, index, expiry=date(2030, 1, 15), effective=EXPIRY)
    assert early.expiry < early.effective


def test_a_swaption_refuses_a_valuation_date_at_or_after_its_expiry(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    expired = swaption(
        0.04, index, expiry=date(2026, 1, 15), effective=date(2026, 1, 15),
        maturity=date(2031, 1, 15),
    )
    with pytest.raises(BadOption, match="exercise decision rather than a value"):
        expired.value(discount, projection, 0.2)


def test_a_swaption_refuses_a_volatility_that_is_not_one(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    with pytest.raises(BadOption, match="is not one"):
        swaption(0.04, index).value(discount, projection, -0.1)


def test_the_volatility_basis_is_separate_from_the_coupon_basis(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Because the day count a volatility is quoted on is a different convention.

    Using the fixed leg's 30/360 instead of actual/365 moves the time to expiry
    and so the option price, by about the ratio of the two day counts.
    """
    one = swaption(0.04, index)
    on_act = one.time_to_expiry(REFERENCE, Basis.ACT_365F)
    on_thirty = one.time_to_expiry(REFERENCE, Basis.THIRTY_360_BOND)
    assert on_act != pytest.approx(on_thirty, rel=1e-6)
    assert one.value(discount, projection, 0.2) != pytest.approx(
        one.value(discount, projection, 0.2, vol_basis=Basis.THIRTY_360_BOND),
        rel=1e-6,
    )


# -- caps and floors ----------------------------------------------------------


def test_a_cap_prices_every_period_on_its_own_forward(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    cap = Cap(effective=CAP_START, maturity=CAP_END, strike=0.035, index=index)
    caplets = cap.caplets(discount, projection, 0.20, include_first=True)
    assert len(caplets) == 20
    for one in caplets:
        total = 0.20 * math.sqrt(
            year_fraction(REFERENCE, one.start, Basis.ACT_365F)
        )
        assert one.value == pytest.approx(
            one.weight * black(one.forward, 0.035, total, Payoff.PAYER), rel=1e-14
        )
        assert one.weight == pytest.approx(one.accrual * one.discount)
    assert cap.value(discount, projection, 0.20, include_first=True) == pytest.approx(
        sum(one.value for one in caplets)
    )


def test_the_first_caplet_is_excluded_by_default_and_is_worth_its_intrinsic(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Its rate has already fixed, so there is no optionality left in it.

    Worth 7.8% of the whole cap at a strike it is in the money at, which is not
    something to lose silently either way.
    """
    cap = Cap(effective=REFERENCE, maturity=date(2028, 1, 15), strike=0.025, index=index)
    with_first = cap.caplets(discount, projection, 0.20, include_first=True)
    without = cap.caplets(discount, projection, 0.20)
    assert len(with_first) == len(without) + 1

    first = with_first[0]
    assert first.value == pytest.approx(
        first.weight * (first.forward - 0.025), rel=1e-15
    )
    whole = cap.value(discount, projection, 0.20, include_first=True)
    assert first.value / whole == pytest.approx(0.0778, abs=5e-4)
    assert whole - cap.value(discount, projection, 0.20) == pytest.approx(first.value)


def test_a_cap_with_nothing_left_to_exercise_says_so(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    quarter = Cap(
        effective=REFERENCE, maturity=date(2026, 4, 15), strike=0.03, index=index
    )
    with pytest.raises(BadOption, match="no periods left with any optionality"):
        quarter.caplets(discount, projection, 0.20)


@pytest.mark.parametrize("convention", list(Convention))
def test_a_cap_less_a_floor_is_the_swap(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    convention: Convention,
) -> None:
    volatility = 0.20 if convention is Convention.LOGNORMAL else 0.008
    forward = quarterly(0.0, index).forward_rate(discount, projection)
    worst = 0.0
    for strike in (0.02, 0.03, forward, 0.05):
        cap = Cap(effective=CAP_START, maturity=CAP_END, strike=strike, index=index)
        floor = Cap(
            effective=CAP_START,
            maturity=CAP_END,
            strike=strike,
            payoff=Payoff.RECEIVER,
            index=index,
        )
        gap = cap.value(
            discount, projection, volatility, convention=convention, include_first=True
        ) - floor.value(
            discount, projection, volatility, convention=convention, include_first=True
        )
        caplets = cap.caplets(
            discount, projection, volatility, convention=convention, include_first=True
        )
        swap = sum(one.weight * (one.forward - strike) for one in caplets)
        weight = sum(one.weight for one in caplets)
        worst = max(worst, abs(gap - swap) / weight)
    assert worst < 1e-17


def test_a_cap_refuses_a_maturity_before_its_start(index: ForecastIndex) -> None:
    with pytest.raises(BadOption, match="not after it starts"):
        Cap(effective=CAP_END, maturity=CAP_START, strike=0.03, index=index)


def test_a_cap_refuses_a_volatility_that_is_not_one(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    cap = Cap(effective=CAP_START, maturity=CAP_END, strike=0.03, index=index)
    with pytest.raises(BadOption, match="is not one"):
        cap.value(discount, projection, math.nan)


def test_a_floor_is_named_for_what_it_is(index: ForecastIndex) -> None:
    assert "floor" in Cap(
        effective=CAP_START, maturity=CAP_END, strike=0.03,
        payoff=Payoff.RECEIVER, index=index,
    ).name
    assert "cap" in Cap(
        effective=CAP_START, maturity=CAP_END, strike=0.03, index=index
    ).name


@pytest.mark.parametrize(
    ("volatility", "premium"), [(0.10, 0.8277), (0.20, 0.7813), (0.40, 0.7482)]
)
def test_the_strip_is_worth_far_more_than_the_option_on_it(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    volatility: float,
    premium: float,
) -> None:
    """And the premium shrinks as volatility rises, which is the surprise.

    The natural guess is that more volatility means more to choose between and
    so a larger premium for the right to choose period by period. Measured, it
    goes the other way.
    """
    forward = quarterly(0.0, index).forward_rate(discount, projection)
    option = quarterly(forward, index)
    cap = Cap(effective=CAP_START, maturity=CAP_END, strike=forward, index=index)
    ratio = cap.value(
        discount, projection, volatility, include_first=True
    ) / option.value(discount, projection, volatility)
    assert ratio - 1.0 == pytest.approx(premium, abs=5e-4)


def test_jensens_inequality_fails_by_exactly_the_day_count_mismatch(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The check that found something, and it was not a pricing error.

    A strip of options is worth at least the option on the strip. That holds in
    twenty-four of twenty-five cases here and fails in the twenty-fifth by
    6.98e-05 — deep in the money at a low volatility, where both prices are
    their intrinsics and the ratio collapses to two conventions disagreeing.

    The fixed leg's annuity accrues on unadjusted boundaries under 30/360 and
    the caplet weights on adjusted ones under actual/360, so the annuity and the
    weight sum differ by 4.655e-05 relative and the forward swap rate and the
    weight-average forward by 4.654e-05. At a strike of 0.6 times the forward
    the ratio is then ``(1 + 4.655e-05)(1 - 1.1635e-04)``, which is 0.99993019
    against a measured 0.99993019. The inequality is exact in the mathematics
    and violated in the conventions, by precisely the amount they differ.
    """
    probe = quarterly(0.0, index)
    forward = probe.forward_rate(discount, projection)
    annuity = probe.annuity(discount)
    cap = Cap(effective=CAP_START, maturity=CAP_END, strike=forward, index=index)
    caplets = cap.caplets(discount, projection, 0.20, include_first=True)
    weight = sum(one.weight for one in caplets)
    average = sum(one.weight * one.forward for one in caplets) / weight

    weight_gap = (weight - annuity) / annuity
    rate_gap = (average - forward) / forward
    assert weight_gap == pytest.approx(4.655e-5, rel=0.01)
    assert rate_gap == pytest.approx(-4.654e-5, rel=0.01)

    worst = (10.0, 0.0, 0.0)
    for volatility in (0.05, 0.10, 0.20, 0.40, 0.80):
        for multiple in (0.6, 0.85, 1.0, 1.2, 1.6):
            strike = forward * multiple
            option = quarterly(strike, index)
            strip = Cap(
                effective=CAP_START, maturity=CAP_END, strike=strike, index=index
            )
            ratio = strip.value(
                discount, projection, volatility, include_first=True
            ) / option.value(discount, projection, volatility)
            if ratio < worst[0]:
                worst = (ratio, volatility, multiple)

    assert worst[0] == pytest.approx(0.99993019, abs=1e-7)
    assert (worst[1], worst[2]) == (0.05, 0.6)
    # And that number is the two mismatches, not a residual to be shrugged at.
    predicted = (1.0 + weight_gap) * (1.0 + rate_gap / (1.0 - 0.6))
    assert worst[0] == pytest.approx(predicted, rel=1e-6)


# -- reading a premium back ---------------------------------------------------


@pytest.mark.parametrize(
    ("convention", "volatility"),
    [(Convention.LOGNORMAL, 0.25), (Convention.NORMAL, 0.009)],
)
@pytest.mark.parametrize("payoff", list(Payoff))
def test_a_premium_inverts_to_the_volatility_that_made_it(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    convention: Convention,
    volatility: float,
    payoff: Payoff,
) -> None:
    forward = swaption(0.0, index).forward_rate(discount, projection)
    for strike in (0.025, 0.035, forward, 0.05, 0.06):
        one = swaption(strike, index, payoff)
        premium = one.value(
            discount, projection, volatility, convention=convention
        ) / one.annuity(discount)
        recovered = implied_volatility(
            premium,
            forward,
            strike,
            one.time_to_expiry(REFERENCE),
            payoff,
            convention=convention,
        )
        assert recovered == pytest.approx(volatility, rel=1e-12)


def test_an_inversion_refuses_a_premium_below_intrinsic() -> None:
    with pytest.raises(BadOption, match="below the intrinsic"):
        implied_volatility(0.001, 0.05, 0.03, 1.0, Payoff.PAYER)


def test_a_premium_at_intrinsic_inverts_to_no_volatility() -> None:
    assert implied_volatility(0.02, 0.05, 0.03, 1.0, Payoff.PAYER) == 0.0


def test_an_inversion_refuses_a_maturity_that_cannot_scale_a_volatility() -> None:
    with pytest.raises(BadOption, match="cannot scale a volatility"):
        implied_volatility(0.01, 0.04, 0.04, 0.0, Payoff.PAYER)


def test_the_inversion_widens_its_bracket_rather_than_assuming_one() -> None:
    """A very expensive option needs a volatility above any fixed bound."""
    premium = black(0.04, 0.04, 4.0, Payoff.PAYER)
    assert implied_volatility(premium, 0.04, 0.04, 1.0, Payoff.PAYER) == pytest.approx(
        4.0, rel=1e-10
    )


def test_the_payoff_names_the_fixed_leg_and_not_the_option() -> None:
    """Which is the market's convention and the usual source of a sign error."""
    assert Payoff.PAYER.rate_option == "call"
    assert Payoff.RECEIVER.rate_option == "put"
    assert Payoff.PAYER.sign == 1.0
    assert Payoff.RECEIVER.sign == -1.0
