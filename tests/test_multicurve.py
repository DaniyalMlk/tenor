"""Forecasting and discounting on separate curves.

The load-bearing test here is the degenerate one. Point both curves at the same
object and the coupon-by-coupon leg has to reproduce the single-curve leg's one
subtraction, because that subtraction is a theorem about the same arithmetic.
Everything else in this module — the bootstrap, the basis swap, the risk split
— is built on top of that leg, so if it is wrong by a day of discounting
nothing downstream will show it and every number will look plausible.

The second kind of test is the measurement. Several claims in the module
docstring were written from an argument, measured, and found to be about a
different thing than the argument said: the rolling convention does not break
the telescoping identity and a payment lag does; the discount curve moves a par
rate the opposite way from the guess and by a factor of 843 less than the
projection curve does. Those numbers are asserted here rather than in prose
alone, so that a change which moves them fails.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tenor.calendar import Rolling
from tenor.curve import BadCurve, DiscountCurve, Interpolation
from tenor.daycount import Basis
from tenor.instruments import Swap
from tenor.multicurve import (
    BadForecast,
    BasisSwap,
    DualSwap,
    FlatLegOf,
    FloatingLeg,
    ForecastIndex,
    IndexForward,
    SpreadLegOf,
    bootstrap_forecast,
    discount_key_rates,
    forecast_key_rates,
    forward_spread,
    projected_forward,
    shifted_forecast,
    split_buckets,
    split_risk,
)
from tenor.risk import Bucket
from tenor.schedule import Frequency

REFERENCE = date(2026, 1, 15)
TEN_YEAR = date(2036, 1, 15)

#: A plausible upward-sloping overnight-indexed curve, in continuous zero rates.
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
]

SWAP_PILLARS = [
    date(2027, 1, 15),
    date(2028, 1, 17),
    date(2029, 1, 15),
    date(2031, 1, 15),
    date(2033, 1, 17),
    date(2036, 1, 15),
]


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
    """The same curve with every forward 20 basis points higher."""
    return shifted_forecast(discount, 0.0020)


@pytest.fixture
def index() -> ForecastIndex:
    return ForecastIndex(tenor=Frequency.QUARTERLY, basis=Basis.ACT_360)


def par_swap(maturity: date, index: ForecastIndex, rate: float = 0.0) -> DualSwap:
    return DualSwap(effective=REFERENCE, maturity=maturity, rate=rate, index=index)


# -- the forward rate ---------------------------------------------------------


def test_forward_rate_grows_one_unit_into_the_curves_own_redemption(
    projection: DiscountCurve,
) -> None:
    start, end = date(2027, 1, 15), date(2027, 4, 15)
    rate = projected_forward(projection, start, end, Basis.ACT_360)
    accrual = (end - start).days / 360.0
    # One unit at ``end`` is worth ``1 / (1 + r tau)`` at ``start``, so the
    # redemption discounted back is the principal: P(end) * (1 + r tau) = P(start).
    grown = projection.discount(end) * (1.0 + rate * accrual)
    assert grown == pytest.approx(projection.discount(start), abs=1e-15)


def test_forward_rate_refuses_a_period_that_runs_backwards(
    projection: DiscountCurve,
) -> None:
    with pytest.raises(BadForecast, match="runs backwards"):
        projected_forward(projection, date(2027, 4, 15), date(2027, 1, 15), Basis.ACT_360)


def test_forward_rate_is_negative_where_the_curve_rises(discount: DiscountCurve) -> None:
    inverted = DiscountCurve.from_discounts(
        REFERENCE,
        [(date(2027, 1, 15), 1.02), (date(2028, 1, 15), 1.04)],
        basis=Basis.ACT_365F,
    )
    rate = projected_forward(inverted, date(2027, 1, 15), date(2028, 1, 15), Basis.ACT_360)
    assert rate < 0.0


def test_shifted_forecast_lifts_the_instantaneous_forward_by_the_spread(
    discount: DiscountCurve,
) -> None:
    moved = shifted_forecast(discount, 0.0020)
    for day in (date(2027, 1, 15), date(2031, 1, 15), date(2036, 1, 15)):
        time = (day - REFERENCE).days / 365.0
        bare = -math.log(discount.discount(day)) / time
        lifted = -math.log(moved.discount(day)) / time
        assert lifted - bare == pytest.approx(0.0020, abs=1e-12)


def test_forward_spread_reads_the_basis_back_off_the_two_curves(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    start, end = date(2030, 1, 15), date(2030, 4, 15)
    spread = forward_spread(projection, discount, start, end, Basis.ACT_360)
    # Simple compounding over a quarter, so slightly under the continuous 20bp.
    assert spread == pytest.approx(0.0020, rel=0.01)
    assert spread < 0.0020


# -- the identity the single-curve swap rests on ------------------------------


def test_projected_leg_reproduces_the_telescoping_subtraction(
    discount: DiscountCurve, index: ForecastIndex
) -> None:
    """Forty projected coupons against one subtraction, on the same curve."""
    leg = FloatingLeg(effective=REFERENCE, maturity=TEN_YEAR, index=index)
    single = Swap(effective=REFERENCE, maturity=TEN_YEAR, rate=0.0)
    projected = leg.value(discount, discount)
    telescoped = single.floating_value(discount)
    assert len(leg.coupons(discount, discount)) == 40
    assert abs(projected - telescoped) / telescoped < 1e-15


@pytest.mark.parametrize(
    "rolling",
    [Rolling.NONE, Rolling.FOLLOWING, Rolling.MODIFIED_FOLLOWING, Rolling.PRECEDING],
)
def test_the_rolling_convention_does_not_break_the_identity(
    discount: DiscountCurve, rolling: Rolling
) -> None:
    """The suspect that turned out to be innocent.

    A rolled period end moves the payment with it, so the dates still meet and
    nothing escapes the cancellation. This was asserted the other way round
    before it was measured.
    """
    index = ForecastIndex(
        tenor=Frequency.QUARTERLY, basis=Basis.ACT_360, rolling=rolling
    )
    leg = FloatingLeg(effective=REFERENCE, maturity=TEN_YEAR, index=index)
    single = Swap(effective=REFERENCE, maturity=TEN_YEAR, rate=0.0, rolling=rolling)
    assert leg.value(discount, discount) == pytest.approx(
        single.floating_value(discount), rel=1e-15
    )


@pytest.mark.parametrize(
    ("lag", "expected_basis_points"),
    [(1, -0.0466), (2, -0.0952), (5, -0.2620)],
)
def test_a_payment_lag_does_break_the_identity_by_a_measured_amount(
    discount: DiscountCurve, lag: int, expected_basis_points: float
) -> None:
    """The culprit, and how much it is worth in par rate terms."""
    index = ForecastIndex(
        tenor=Frequency.QUARTERLY, basis=Basis.ACT_360, payment_lag=lag
    )
    swap = par_swap(TEN_YEAR, index)
    leg = swap.floating
    single = Swap(effective=REFERENCE, maturity=TEN_YEAR, rate=0.0)
    gap = leg.value(discount, discount) - single.floating_value(discount)
    assert gap < 0.0
    assert gap / swap.annuity(discount) * 1e4 == pytest.approx(
        expected_basis_points, abs=5e-4
    )


def test_a_negative_payment_lag_is_refused() -> None:
    with pytest.raises(BadForecast, match="before it has run"):
        ForecastIndex(payment_lag=-1)


def test_a_leg_starting_before_the_projection_curve_is_refused(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    leg = FloatingLeg(
        effective=REFERENCE - timedelta(days=40), maturity=TEN_YEAR, index=index
    )
    with pytest.raises(BadForecast, match="first fixing is then history"):
        leg.value(discount, projection)


def test_a_leg_that_matures_before_it_starts_is_refused(index: ForecastIndex) -> None:
    with pytest.raises(BadForecast, match="not after it starts"):
        FloatingLeg(effective=TEN_YEAR, maturity=REFERENCE, index=index)


# -- the leg's own arithmetic -------------------------------------------------


def test_coupons_report_every_number_that_went_into_them(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    leg = FloatingLeg(
        effective=REFERENCE, maturity=date(2027, 1, 15), index=index, spread=0.0025
    )
    coupons = leg.coupons(discount, projection)
    assert len(coupons) == 4
    for one in coupons:
        assert one.rate == pytest.approx(one.forward + 0.0025)
        assert one.cashflow == pytest.approx(one.rate * one.accrual)
        assert one.value == pytest.approx(one.cashflow * one.discount)
        assert one.discount == pytest.approx(discount.discount(one.payment))
        assert one.forward == pytest.approx(
            projected_forward(projection, one.start, one.end, Basis.ACT_360)
        )
    assert leg.value(discount, projection) == pytest.approx(
        sum(one.value for one in coupons)
    )


def test_the_spread_is_worth_its_annuity_and_the_projection_curve_does_not_enter(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    bare = FloatingLeg(effective=REFERENCE, maturity=TEN_YEAR, index=index)
    spread = FloatingLeg(
        effective=REFERENCE, maturity=TEN_YEAR, index=index, spread=0.0030
    )
    lifted = spread.value(discount, projection) - bare.value(discount, projection)
    assert lifted == pytest.approx(0.0030 * bare.annuity(discount), rel=1e-13)
    # The annuity is a discounting question only.
    assert spread.annuity(discount) == pytest.approx(bare.annuity(discount))


def test_the_leg_endpoints_are_the_adjusted_ones(
    index: ForecastIndex,
) -> None:
    leg = FloatingLeg(effective=date(2026, 1, 17), maturity=TEN_YEAR, index=index)
    assert leg.first_accrual.weekday() < 5
    assert leg.final_payment.weekday() < 5


# -- the swap -----------------------------------------------------------------


def test_par_rate_makes_the_swap_worth_nothing(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    probe = par_swap(TEN_YEAR, index)
    rate = probe.par_rate(discount, projection)
    struck = par_swap(TEN_YEAR, index, rate)
    assert struck.par_error(discount, projection) == pytest.approx(0.0, abs=1e-16)
    assert struck.forecast_error(discount, projection) == pytest.approx(0.0, abs=1e-16)


def test_a_dual_swap_on_one_curve_is_the_single_curve_swap(
    discount: DiscountCurve, index: ForecastIndex
) -> None:
    dual = par_swap(TEN_YEAR, index, 0.0350)
    single = Swap(effective=REFERENCE, maturity=TEN_YEAR, rate=0.0350)
    assert dual.annuity(discount) == pytest.approx(single.annuity(discount), rel=1e-15)
    assert dual.par_error(discount, discount) == pytest.approx(
        single.par_error(discount), rel=1e-14
    )
    assert dual.par_rate(discount, discount) == pytest.approx(
        single.par_rate(discount), rel=1e-14
    )


def test_a_swap_that_matures_before_it_starts_is_refused(index: ForecastIndex) -> None:
    with pytest.raises(BadForecast, match="not after it starts"):
        DualSwap(effective=TEN_YEAR, maturity=REFERENCE, rate=0.03, index=index)


def underflowed(swap: DualSwap) -> DiscountCurve:
    """A curve whose discount factors are the smallest number there is.

    An annuity is a sum of strictly positive terms, so the only way it reaches
    zero is underflow: at 5e-324 an accrual times a discount factor rounds to
    nothing and the sum is exactly 0.0. At 1e-300 it does not — the annuity
    comes back as 1e-300 and the division is fine. That is the whole domain of
    the guard, and it is worth pinning rather than deleting, because the
    alternative to a refusal here is a division by zero reported as a rate.
    """
    days = sorted(
        {one.payment for one in swap.fixed_schedule()}
        | {one.payment for one in swap.floating.schedule()}
    )
    return DiscountCurve.from_discounts(
        REFERENCE, [(day, 5e-324) for day in days], basis=Basis.ACT_365F
    )


def test_par_rate_refuses_an_annuity_of_nothing(
    projection: DiscountCurve, index: ForecastIndex
) -> None:
    swap = par_swap(date(2027, 1, 15), index)
    dead = underflowed(swap)
    assert swap.annuity(dead) == 0.0
    with pytest.raises(BadForecast, match="makes it worth nothing"):
        swap.par_rate(dead, projection)


def test_an_annuity_that_merely_underflows_the_terms_still_divides(
    projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The other side of the same boundary, so the guard is not over-wide."""
    swap = par_swap(date(2027, 1, 15), index)
    days = sorted(
        {one.payment for one in swap.fixed_schedule()}
        | {one.payment for one in swap.floating.schedule()}
    )
    faint = DiscountCurve.from_discounts(
        REFERENCE, [(day, 1e-300) for day in days], basis=Basis.ACT_365F
    )
    assert swap.annuity(faint) > 0.0
    assert math.isfinite(swap.par_rate(faint, projection))


def test_the_swaps_final_date_covers_both_legs(index: ForecastIndex) -> None:
    annual = DualSwap(
        effective=REFERENCE,
        maturity=TEN_YEAR,
        rate=0.03,
        index=index,
        frequency=Frequency.ANNUAL,
    )
    assert annual.final_date >= annual.floating.final_payment
    assert annual.final_date >= annual.fixed_schedule()[-1].payment


# -- what each curve is actually worth ----------------------------------------


def test_the_projection_curve_moves_a_par_rate_and_the_discount_curve_barely_does(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The measurement the module was built to make.

    The guess was that separating the curves repriced par swaps. It does not:
    the discount curve enters a par rate only as the weights in an average of
    forwards, so reweighting a nearly flat set of forwards moves almost
    nothing — and it moves it *downwards* on an upward-sloping curve, because
    heavier discounting tilts the weights towards the earlier, lower forwards.
    """
    swap = par_swap(TEN_YEAR, index)
    base = swap.par_rate(discount, projection)
    from_forecast = swap.par_rate(discount, projection.shifted(0.0100)) - base
    from_discount = swap.par_rate(discount.shifted(0.0100), projection) - base

    assert from_forecast * 1e4 == pytest.approx(101.62, abs=0.01)
    assert from_discount * 1e4 == pytest.approx(-0.12, abs=0.01)
    assert from_discount < 0.0
    assert abs(from_forecast / from_discount) == pytest.approx(843.0, abs=2.0)


def test_the_discount_curve_is_first_order_on_a_seasoned_swap(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """And third-order on a new one, which is the reverse of the usual telling."""
    probe = par_swap(TEN_YEAR, index)
    market = probe.par_rate(discount, projection)
    seasoned = par_swap(TEN_YEAR, index, market + 0.0100)

    mark = seasoned.par_error(discount, projection)
    moved = seasoned.par_error(discount.shifted(0.0100), projection)
    assert (moved - mark) / abs(mark) == pytest.approx(0.04668, abs=1e-4)

    par_move = probe.par_rate(discount.shifted(0.0100), projection) - market
    assert abs(par_move / market) == pytest.approx(0.000309, abs=1e-5)


@pytest.mark.parametrize(
    ("basis_points", "expected"),
    [(10.0, 10.1452), (20.0, 20.2929), (50.0, 50.7512)],
)
def test_a_flat_basis_passes_through_to_the_par_rate(
    discount: DiscountCurve,
    index: ForecastIndex,
    basis_points: float,
    expected: float,
) -> None:
    swap = par_swap(TEN_YEAR, index)
    bare = swap.par_rate(discount, discount)
    lifted = swap.par_rate(discount, shifted_forecast(discount, basis_points / 1e4))
    assert (lifted - bare) * 1e4 == pytest.approx(expected, abs=5e-4)


def test_the_pass_through_factorises_into_two_measured_ratios(
    discount: DiscountCurve, index: ForecastIndex
) -> None:
    """The decomposition as an identity rather than as an explanation.

    The pass-through is the ratio of the two legs' annuities — actual/360
    against 30/360 — times the amount a flat continuous shift lifts a simply
    compounded quarterly forward by. Neither factor is one and they pull in
    opposite directions; their product is the measured number to 6e-15, which
    is what distinguishes a decomposition from a plausible story.
    """
    spread = 0.0020
    swap = par_swap(TEN_YEAR, index)
    lifted = shifted_forecast(discount, spread)
    passthrough = (
        swap.par_rate(discount, lifted) - swap.par_rate(discount, discount)
    ) / spread

    annuity_ratio = swap.floating.annuity(discount) / swap.annuity(discount)
    before = swap.floating.coupons(discount, discount)
    after = swap.floating.coupons(discount, lifted)
    weighted = sum(
        (new.forward - old.forward) * old.accrual * old.discount
        for old, new in zip(before, after, strict=True)
    )
    weights = sum(old.accrual * old.discount for old in before)
    forward_lift = (weighted / weights) / spread

    assert annuity_ratio == pytest.approx(1.019107, abs=1e-6)
    assert forward_lift == pytest.approx(0.995619, abs=1e-6)
    assert annuity_ratio * forward_lift == pytest.approx(passthrough, abs=1e-13)


# -- the projection bootstrap -------------------------------------------------


def generated_quotes(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> list[object]:
    """Quotes priced off a known projection curve, for the bootstrap to recover."""
    quotes: list[object] = [
        IndexForward(
            REFERENCE,
            date(2026, 4, 15),
            projected_forward(projection, REFERENCE, date(2026, 4, 15), Basis.ACT_360),
            label="3m forward",
        )
    ]
    for maturity in SWAP_PILLARS:
        probe = par_swap(maturity, index)
        quotes.append(
            DualSwap(
                effective=REFERENCE,
                maturity=maturity,
                rate=probe.par_rate(discount, projection),
                index=index,
                label=f"swap to {maturity.isoformat()}",
            )
        )
    return quotes


def test_the_bootstrap_reprices_every_quote(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    quotes = generated_quotes(discount, projection, index)
    built = bootstrap_forecast(
        REFERENCE, quotes, discount=discount, basis=Basis.ACT_365F  # type: ignore[arg-type]
    )
    assert len(built) == 7
    assert built.worst_error < 1e-15
    assert built.reprices()
    assert built.sweeps == 1
    assert all(one.converged for one in built.solutions)
    assert built.discount is discount


def test_an_interpolation_that_is_not_local_needs_more_than_one_sweep(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The sweep loop, earning its place.

    Under log-linear discounts a pillar is independent of the ones after it and
    the sequential pass is already the answer. Under monotone convex it is not,
    and the fit has to be iterated to a fixed point — five times here.
    """
    quotes = generated_quotes(discount, projection, index)
    built = bootstrap_forecast(
        REFERENCE,
        quotes,  # type: ignore[arg-type]
        discount=discount,
        basis=Basis.ACT_365F,
        interpolation=Interpolation.MONOTONE_CONVEX,
    )
    assert built.sweeps == 5
    assert built.reprices()


def test_an_exact_fit_says_nothing_about_the_curve_between_quotes(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Reprices to 1e-16 and 5.3 basis points out where no quote reaches.

    The quotes here were generated off ``projection``, so the right answer is
    known exactly. Beyond a year the recovered curve is within 0.04 basis
    points of it. Between the three-month forward and the one-year swap there
    is no quote at all, and the interpolation fills the gap with something 5.3
    basis points wrong — an exact fit to the inputs being a statement about the
    inputs and not about the curve.
    """
    quotes = generated_quotes(discount, projection, index)
    built = bootstrap_forecast(
        REFERENCE, quotes, discount=discount, basis=Basis.ACT_365F  # type: ignore[arg-type]
    )
    assert built.worst_error < 1e-15

    def worst_gap(first: int, last: int) -> float:
        worst = 0.0
        for days in range(first, last, 5):
            day = REFERENCE + timedelta(days=days)
            time = days / 365.0
            recovered = -math.log(built.curve.discount(day)) / time
            truth = -math.log(projection.discount(day)) / time
            worst = max(worst, abs(recovered - truth) * 1e4)
        return worst

    assert worst_gap(30, 365) == pytest.approx(5.30, abs=0.05)
    assert worst_gap(400, 3600) < 0.04


def test_the_bootstrap_refuses_an_empty_quote_set(discount: DiscountCurve) -> None:
    with pytest.raises(BadForecast, match="at least one quote"):
        bootstrap_forecast(REFERENCE, [], discount=discount, basis=Basis.ACT_365F)


def test_the_bootstrap_refuses_two_quotes_on_one_pillar(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    probe = par_swap(date(2028, 1, 17), index)
    rate = probe.par_rate(discount, projection)
    pair = [
        DualSwap(
            effective=REFERENCE,
            maturity=date(2028, 1, 17),
            rate=rate,
            index=index,
            label="one",
        ),
        DualSwap(
            effective=REFERENCE,
            maturity=date(2028, 1, 17),
            rate=rate + 0.0001,
            index=index,
            label="two",
        ),
    ]
    with pytest.raises(BadForecast, match="no curve satisfies both"):
        bootstrap_forecast(
            REFERENCE, pair, discount=discount, basis=Basis.ACT_365F
        )


def test_the_bootstrap_refuses_a_quote_that_has_already_settled(
    discount: DiscountCurve,
) -> None:
    stale = IndexForward(
        REFERENCE - timedelta(days=120), REFERENCE - timedelta(days=30), 0.03
    )
    with pytest.raises(BadForecast, match="already settled"):
        bootstrap_forecast(
            REFERENCE, [stale], discount=discount, basis=Basis.ACT_365F
        )


def test_the_bootstrap_refuses_a_discount_curve_on_another_date(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    quotes = generated_quotes(discount, projection, index)
    with pytest.raises(BadForecast, match="cannot discount each other"):
        bootstrap_forecast(
            REFERENCE + timedelta(days=1),
            quotes,  # type: ignore[arg-type]
            discount=discount,
            basis=Basis.ACT_365F,
        )


def test_the_bootstrap_refuses_a_sweep_budget_of_nothing(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    quotes = generated_quotes(discount, projection, index)
    with pytest.raises(BadForecast, match="at least one sweep"):
        bootstrap_forecast(
            REFERENCE,
            quotes,  # type: ignore[arg-type]
            discount=discount,
            basis=Basis.ACT_365F,
            max_sweeps=0,
        )


def test_an_index_forward_does_not_touch_the_discount_curve(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    quote = IndexForward(REFERENCE, date(2026, 7, 15), 0.0315)
    other = shifted_forecast(discount, 0.05)
    assert quote.forecast_error(discount, projection) == pytest.approx(
        quote.forecast_error(other, projection)
    )


def test_an_index_forward_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(BadForecast, match="not after it starts"):
        IndexForward(date(2027, 1, 15), date(2026, 1, 15), 0.03)


def test_a_bootstrapped_projection_curve_reads_its_quotes_back_as_forwards(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    quote = IndexForward(REFERENCE, date(2026, 7, 15), 0.0330, label="6m")
    built = bootstrap_forecast(
        REFERENCE, [quote], discount=discount, basis=Basis.ACT_365F
    )
    recovered = projected_forward(built.curve, REFERENCE, date(2026, 7, 15), Basis.ACT_360)
    assert recovered == pytest.approx(0.0330, abs=1e-14)


# -- tenor basis --------------------------------------------------------------


THREE_MONTH = ForecastIndex(
    tenor=Frequency.QUARTERLY, basis=Basis.ACT_360, label="3m index"
)
SIX_MONTH = ForecastIndex(
    tenor=Frequency.SEMI_ANNUAL, basis=Basis.ACT_360, label="6m index"
)

BASIS_QUOTES = [
    (date(2027, 1, 15), 0.0005),
    (date(2028, 1, 17), 0.0006),
    (date(2029, 1, 15), 0.0007),
    (date(2031, 1, 15), 0.0008),
    (date(2036, 1, 15), 0.0009),
]


def test_a_basis_swap_refuses_two_legs_on_one_tenor() -> None:
    with pytest.raises(BadForecast, match="between an index and itself"):
        BasisSwap(
            effective=REFERENCE,
            maturity=TEN_YEAR,
            spread=0.0005,
            spread_index=THREE_MONTH,
            flat_index=ForecastIndex(tenor=Frequency.QUARTERLY, basis=Basis.ACT_365F),
        )


def test_a_basis_swap_that_matures_before_it_starts_is_refused() -> None:
    with pytest.raises(BadForecast, match="not after it starts"):
        BasisSwap(effective=TEN_YEAR, maturity=REFERENCE, spread=0.0005)


def test_the_par_spread_is_the_one_that_prices_the_legs_the_same(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    six = shifted_forecast(projection, 0.0008)
    swap = BasisSwap(
        effective=REFERENCE,
        maturity=TEN_YEAR,
        spread=0.0,
        spread_index=THREE_MONTH,
        flat_index=SIX_MONTH,
    )
    spread = swap.par_spread(discount, projection, six)
    struck = BasisSwap(
        effective=REFERENCE,
        maturity=TEN_YEAR,
        spread=spread,
        spread_index=THREE_MONTH,
        flat_index=SIX_MONTH,
    )
    # Three ulps of a leg worth about 0.3, which is as close as a difference
    # of two sums of twenty and forty terms gets.
    assert struck.par_error(discount, projection, six) == pytest.approx(
        0.0, abs=1e-15
    )
    # The six-month curve is the richer one, so the three-month leg is paid up.
    assert spread > 0.0


def test_the_par_spread_refuses_an_annuity_of_nothing(
    projection: DiscountCurve,
) -> None:
    swap = BasisSwap(
        effective=REFERENCE,
        maturity=date(2027, 1, 15),
        spread=0.0,
        spread_index=THREE_MONTH,
        flat_index=SIX_MONTH,
    )
    days = sorted(
        {one.payment for one in swap.spread_leg.schedule()}
        | {one.payment for one in swap.flat_leg.schedule()}
    )
    dead = DiscountCurve.from_discounts(
        REFERENCE, [(day, 5e-324) for day in days], basis=Basis.ACT_365F
    )
    assert swap.spread_leg.annuity(dead) == 0.0
    with pytest.raises(BadForecast, match="spread annuity"):
        swap.par_spread(dead, projection, projection)


def test_the_long_tenor_curve_is_solved_out_of_the_quoted_basis(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    quotes = [
        FlatLegOf(
            BasisSwap(
                effective=REFERENCE,
                maturity=maturity,
                spread=spread,
                spread_index=THREE_MONTH,
                flat_index=SIX_MONTH,
            ),
            projection,
        )
        for maturity, spread in BASIS_QUOTES
    ]
    built = bootstrap_forecast(
        REFERENCE, quotes, discount=discount, basis=Basis.ACT_365F
    )
    assert built.reprices()
    assert "flat leg" in built.quotes[0].name
    # A six-month index richer than the three-month one, as the quotes say.
    for start in (date(2030, 1, 15), date(2035, 1, 15)):
        end = date(start.year, 7, 15)
        assert forward_spread(built.curve, projection, start, end, Basis.ACT_360) > 0.0


def test_the_short_tenor_curve_can_be_the_unknown_instead(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    six = shifted_forecast(projection, 0.0008)
    quotes = [
        SpreadLegOf(
            BasisSwap(
                effective=REFERENCE,
                maturity=maturity,
                spread=spread,
                spread_index=THREE_MONTH,
                flat_index=SIX_MONTH,
            ),
            six,
        )
        for maturity, spread in BASIS_QUOTES
    ]
    built = bootstrap_forecast(
        REFERENCE, quotes, discount=discount, basis=Basis.ACT_365F
    )
    assert built.reprices()
    assert "spread leg" in built.quotes[0].name


def test_an_unpinned_short_end_puts_the_whole_front_basis_in_one_period(
    discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Why :class:`IndexForward` exists, measured.

    With nothing quoted inside the first year, the one-year basis swap is the
    earliest statement about the six-month curve, and the interpolation spreads
    it backwards: the implied forward basis over the first period comes out at
    16.87 basis points, three times the five basis points the quote carries.
    One forward quote on the long index fixes the first period at exactly what
    it says.
    """
    swaps = [
        BasisSwap(
            effective=REFERENCE,
            maturity=maturity,
            spread=spread,
            spread_index=THREE_MONTH,
            flat_index=SIX_MONTH,
        )
        for maturity, spread in BASIS_QUOTES
    ]
    first = (REFERENCE, date(2026, 7, 15))

    loose = bootstrap_forecast(
        REFERENCE,
        [FlatLegOf(one, projection) for one in swaps],
        discount=discount,
        basis=Basis.ACT_365F,
    )
    assert (
        forward_spread(loose.curve, projection, *first, Basis.ACT_360) * 1e4
    ) == pytest.approx(16.87, abs=0.01)

    pin = IndexForward(
        REFERENCE,
        date(2026, 7, 15),
        projected_forward(projection, *first, Basis.ACT_360) + 0.0005,
        label="6m forward",
    )
    pinned = bootstrap_forecast(
        REFERENCE,
        [pin, *[FlatLegOf(one, projection) for one in swaps]],
        discount=discount,
        basis=Basis.ACT_365F,
    )
    assert pinned.reprices()
    assert (
        forward_spread(pinned.curve, projection, *first, Basis.ACT_360) * 1e4
    ) == pytest.approx(5.00, abs=1e-6)


# -- risk, split between the curves -------------------------------------------


def seasoned(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> DualSwap:
    market = par_swap(TEN_YEAR, index).par_rate(discount, projection)
    return par_swap(TEN_YEAR, index, market + 0.0100)


def test_split_risk_separates_the_two_curves(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    swap = seasoned(discount, projection, index)
    split = split_risk(swap.par_error, discount, projection)
    assert split.forecast > split.discount > 0.0
    # The projection curve carries the great majority of the risk.
    assert split.forecast / split.discount == pytest.approx(21.0, abs=1.0)


@pytest.mark.parametrize(
    ("shift", "relative"),
    [(1e-4, 1.45e-7), (1e-3, 1.45e-5), (1e-2, 1.44e-3)],
)
def test_the_cross_term_is_second_order_in_the_shift(
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    shift: float,
    relative: float,
) -> None:
    """Shifting both curves is not the sum of shifting each, and by how much.

    A swap's value is bilinear in the two curves, so the joint response carries
    a cross term the separate ones do not. It is 1.4e-07 of the joint response
    at a basis point — small enough to ignore and not zero — and it grows with
    the square of the shift, which is the test that identifies it as the cross
    term rather than as a numerical artefact.
    """
    swap = seasoned(discount, projection, index)
    split = split_risk(swap.par_error, discount, projection, shift=shift)
    assert split.crossed / split.joint == pytest.approx(relative, rel=0.05)


def test_split_risk_refuses_a_shift_of_nothing(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    swap = seasoned(discount, projection, index)
    with pytest.raises(BadForecast, match="needs a step to take"):
        split_risk(swap.par_error, discount, projection, shift=0.0)


def test_the_bucket_responses_add_back_to_the_parallel_ones(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    swap = seasoned(discount, projection, index)
    buckets = [Bucket(time) for time in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)]
    rows = split_buckets(swap.par_error, discount, projection, buckets)
    whole = split_risk(swap.par_error, discount, projection)
    assert sum(one.discount for one in rows) == pytest.approx(whole.discount, rel=1e-7)
    assert sum(one.forecast for one in rows) == pytest.approx(whole.forecast, rel=1e-7)
    # A ten-year swap's forecast risk is concentrated at its own maturity,
    # because the final period's forward is the one nothing else offsets.
    assert max(rows, key=lambda one: one.forecast).bucket.time == 10.0


def test_key_rates_against_each_curve_hold_the_other_still(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    swap = seasoned(discount, projection, index)
    buckets = [Bucket(time) for time in (1.0, 3.0, 5.0, 10.0)]
    on_discount = discount_key_rates(swap.par_error, discount, projection, buckets)
    on_forecast = forecast_key_rates(swap.par_error, discount, projection, buckets)
    assert len(on_discount) == len(on_forecast) == 4
    # Opposite signs: the mark is negative, so a duration computed against it
    # flips, and the two curves move the mark in the same direction.
    assert all(
        one.duration * other.duration > 0.0
        for one, other in zip(on_discount, on_forecast, strict=True)
    )
    assert sum(one.value for one in on_discount) != pytest.approx(
        sum(one.value for one in on_forecast)
    )


def test_a_par_swap_has_no_key_rate_duration_to_report(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """A duration is relative, and a par swap is worth nothing.

    The money responses in :func:`split_buckets` exist for exactly this case,
    which is most of the positions anybody wants this for on the day they are
    struck.
    """
    probe = par_swap(TEN_YEAR, index)
    market = probe.par_rate(discount, projection)
    at_par = par_swap(TEN_YEAR, index, market)
    buckets = [Bucket(time) for time in (1.0, 10.0)]
    with pytest.raises(Exception, match="worth nothing"):
        discount_key_rates(at_par.par_error, discount, projection, buckets)
    rows = split_buckets(at_par.par_error, discount, projection, buckets)
    assert any(abs(one.forecast) > 0.0 for one in rows)


def test_a_curve_the_bracket_cannot_reach_is_reported_not_crashed(
    discount: DiscountCurve, index: ForecastIndex
) -> None:
    """A quote implying a forward outside the admissible range is refused.

    Under monotone convex the bracket is hard — a factor outside its
    neighbours' range is a negative forward the scheme will not represent — so
    an absurd quote has nowhere to solve and says so rather than returning a
    curve that prices nothing.
    """
    absurd = [
        IndexForward(REFERENCE, date(2027, 1, 15), 0.03, label="sane"),
        IndexForward(
            date(2027, 1, 15), date(2028, 1, 17), -0.95, label="impossible forward"
        ),
    ]
    with pytest.raises((BadForecast, BadCurve)):
        bootstrap_forecast(
            REFERENCE,
            absurd,
            discount=discount,
            basis=Basis.ACT_365F,
            interpolation=Interpolation.MONOTONE_CONVEX,
        )
