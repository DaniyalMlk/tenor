"""Caplet volatilities stripped from flat cap quotes.

Two identities carry the weight. Reading a flat volatility back out of a
stripped curve returns the quote it came from, which is the only check that
does not go through another approximation. And a flat caplet curve has to quote
flat at every maturity — the degenerate case, and the one that catches a
volatility-basis mismatch between the bootstrap and the inverse, since each of
those would still be self-consistent on its own.

Everything else is a measurement. How much steeper the stripped curve is than
the quotes, how fast the conditioning degrades with maturity, how far a quote
can fall before no non-negative volatility explains it, and what a hump in the
quotes does to the buckets under it.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis
from tenor.multicurve import ForecastIndex, shifted_forecast
from tenor.options import Cap, Convention, Payoff
from tenor.schedule import Frequency
from tenor.stripping import (
    INDETERMINACY_LIMIT,
    PREMIUM_TOLERANCE,
    VOLATILITY_CEILING,
    CapletBucket,
    CapletVolatility,
    CapQuote,
    StrippingError,
    flat_volatility,
    strip_caplets,
)

REFERENCE = date(2026, 1, 15)

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

MATURITIES = (
    date(2027, 1, 15),
    date(2028, 1, 15),
    date(2029, 1, 15),
    date(2030, 1, 15),
    date(2031, 1, 15),
)

STRIKE = 0.035
RISING = (0.18, 0.20, 0.22, 0.24, 0.26)


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


def strip(
    volatilities: tuple[float, ...],
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    *,
    maturities: tuple[date, ...] = MATURITIES,
    strike: float = STRIKE,
    payoff: Payoff = Payoff.PAYER,
    convention: Convention = Convention.LOGNORMAL,
) -> CapletVolatility:
    quotes = [
        CapQuote(maturity, volatility)
        for maturity, volatility in zip(maturities, volatilities, strict=True)
    ]
    return strip_caplets(
        quotes,
        REFERENCE,
        strike,
        discount,
        projection,
        index=index,
        payoff=payoff,
        convention=convention,
    )


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("volatility", [-0.01, math.nan, math.inf])
def test_a_quote_that_is_not_a_volatility_is_rejected(volatility: float) -> None:
    with pytest.raises(StrippingError, match="is not one"):
        CapQuote(date(2028, 1, 15), volatility)


def test_an_empty_quote_set_is_rejected(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    with pytest.raises(StrippingError, match="at least one"):
        strip_caplets([], REFERENCE, STRIKE, discount, projection, index=index)


def test_a_maturity_off_the_roll_grid_is_refused_by_name(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """generate runs backward from maturity, so these schedules interleave.

    A cap to March does not contain a cap to January's periods as a prefix,
    and there is then nothing for the bootstrap to hold fixed. The error says
    which maturities and shows the dates that disagree.
    """
    quotes = [CapQuote(date(2027, 1, 15), 0.18), CapQuote(date(2028, 3, 15), 0.20)]
    with pytest.raises(StrippingError, match="do not nest"):
        strip_caplets(quotes, REFERENCE, STRIKE, discount, projection, index=index)


def test_a_maturity_adding_no_period_is_refused(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Two maturities inside one quarterly period leave the second nothing to say."""
    quotes = [CapQuote(date(2028, 1, 15), 0.18), CapQuote(date(2028, 1, 15), 0.20)]
    with pytest.raises(StrippingError, match="adds no period"):
        strip_caplets(quotes, REFERENCE, STRIKE, discount, projection, index=index)


def test_quotes_are_sorted_rather_than_required_sorted(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    forward = strip(RISING, discount, projection, index)
    shuffled = strip_caplets(
        [CapQuote(MATURITIES[position], RISING[position]) for position in (3, 0, 4, 2, 1)],
        REFERENCE,
        STRIKE,
        discount,
        projection,
        index=index,
    )
    assert [bucket.volatility for bucket in shuffled.buckets] == [
        bucket.volatility for bucket in forward.buckets
    ]


# -- the two identities -------------------------------------------------------


def test_every_quote_reprices_from_the_stripped_curve(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The only check that does not go through another approximation."""
    curve = strip(RISING, discount, projection, index)
    for maturity, quoted in zip(MATURITIES, RISING, strict=True):
        back = flat_volatility(curve, maturity, REFERENCE, discount, projection, index=index)
        assert back == pytest.approx(quoted, rel=1e-14)
        assert abs(back - quoted) / quoted < 1e-15


@pytest.mark.parametrize("level", [0.05, 0.20, 0.45])
def test_a_flat_caplet_curve_quotes_flat_at_every_maturity(
    level: float,
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
) -> None:
    """The degenerate case, and the one that catches a basis mismatch.

    The bootstrap and the inverse would each stay self-consistent while
    measuring the time to a fixing on different day-count bases. Only this
    identity, where every bucket has to come back as the number fed in, puts
    them against each other.
    """
    curve = strip((level,) * 5, discount, projection, index)
    for bucket in curve.buckets:
        assert bucket.volatility == pytest.approx(level, rel=1e-14)
    for maturity in MATURITIES:
        assert flat_volatility(
            curve, maturity, REFERENCE, discount, projection, index=index
        ) == pytest.approx(level, rel=1e-14)


def test_the_first_bucket_is_its_own_quote(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """A one-bucket strip has nothing to average, so there is nothing to solve.

    This is the row of the table that can be checked by hand, and its
    sensitivity is exactly one for the same reason.
    """
    curve = strip(RISING, discount, projection, index)
    assert curve.buckets[0].volatility == pytest.approx(0.18, rel=1e-14)
    assert curve.buckets[0].sensitivity == pytest.approx(1.0, abs=1e-6)


def test_the_curve_matches_a_cap_priced_at_one_volatility(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """A flat strip has to agree with Cap.value, which knows nothing of this module."""
    curve = strip((0.22,) * 5, discount, projection, index)
    for maturity in MATURITIES:
        cap = Cap(effective=REFERENCE, maturity=maturity, strike=STRIKE, index=index)
        direct = cap.value(discount, projection, 0.22)
        implied = flat_volatility(curve, maturity, REFERENCE, discount, projection, index=index)
        assert implied == pytest.approx(0.22, rel=1e-13)
        assert cap.value(discount, projection, implied) == pytest.approx(direct, rel=1e-13)


# -- what the measurements said -----------------------------------------------


def test_the_stripped_curve_is_steeper_than_the_quotes(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """A flat volatility averages premiums, not volatilities.

    So a rising quote curve has to be explained by a caplet curve that rises
    faster, and the gap compounds along the strip rather than staying level.
    """
    curve = strip(RISING, discount, projection, index)
    levels = [bucket.volatility for bucket in curve.buckets]
    assert curve.quoted_span == (0.18, 0.26)
    low, high = curve.stripped_span
    assert low == pytest.approx(0.18, rel=1e-12)
    assert high == pytest.approx(0.3121, rel=1e-3)
    assert levels[-1] - 0.26 == pytest.approx(0.0521, abs=5e-4)
    quote_slope = RISING[-1] - RISING[-2]
    assert (levels[-1] - levels[-2]) / quote_slope == pytest.approx(1.770, rel=1e-3)
    gaps = [level - quoted for level, quoted in zip(levels, RISING, strict=True)]
    assert gaps == sorted(gaps)


def test_the_conditioning_degrades_with_maturity(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Each bucket is a smaller share of its cap's premium than the last.

    Which is the number that says how much of the curve's far end to believe:
    a basis point of quoting error is 3.6 basis points on the five-year
    bucket.
    """
    curve = strip(RISING, discount, projection, index)
    sensitivities = [bucket.sensitivity for bucket in curve.buckets]
    assert sensitivities == sorted(sensitivities)
    expected = [1.0000, 1.4077, 2.1154, 2.8403, 3.6328]
    for measured, target in zip(sensitivities, expected, strict=True):
        assert measured == pytest.approx(target, rel=5e-3)
    assert sensitivities[-1] / sensitivities[0] == pytest.approx(3.63, rel=0.02)


def test_the_sensitivity_is_the_derivative_it_claims_to_be(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Checked against a one-basis-point re-strip rather than its own bump."""
    base = strip(RISING, discount, projection, index)
    for position in range(len(RISING)):
        bumped_quotes = list(RISING)
        bumped_quotes[position] += 1e-4
        bumped = strip(tuple(bumped_quotes), discount, projection, index)
        moved = (bumped.buckets[position].volatility - base.buckets[position].volatility) / 1e-4
        assert moved == pytest.approx(base.buckets[position].sensitivity, rel=1e-3)
        # and an earlier bucket does not move at all
        if position > 0:
            assert bumped.buckets[position - 1].volatility == pytest.approx(
                base.buckets[position - 1].volatility, rel=1e-14
            )


def test_a_hump_in_the_quotes_is_an_overshoot_in_the_buckets(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The later periods have to undo an average the earlier ones hold up.

    A four-point fall in the quotes is a sixteen-point fall in the buckets,
    which is the ill-posedness made visible rather than inferred from a
    condition number.
    """
    humped = (0.18, 0.26, 0.30, 0.26, 0.22)
    curve = strip(humped, discount, projection, index)
    levels = [bucket.volatility for bucket in curve.buckets]
    for measured, target in zip(levels, [0.1800, 0.2931, 0.3450, 0.1862, 0.1151], strict=True):
        assert measured == pytest.approx(target, rel=5e-3)
    quote_fall = humped[2] - humped[3]
    bucket_fall = levels[2] - levels[3]
    assert quote_fall == pytest.approx(0.04)
    assert bucket_fall == pytest.approx(0.159, rel=0.02)
    assert bucket_fall / quote_fall > 3.9


def test_monotone_quotes_give_monotone_buckets_here_and_not_in_general(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Recorded as a measurement on this structure, not as a property.

    The humped case above shows the mapping is not order-preserving in
    general, so the rising case is a fact about these quotes rather than
    something the construction guarantees.
    """
    rising = [bucket.volatility for bucket in strip(RISING, discount, projection, index).buckets]
    assert rising == sorted(rising)
    falling = [
        bucket.volatility
        for bucket in strip((0.26, 0.24, 0.22, 0.20, 0.18), discount, projection, index).buckets
    ]
    assert falling == sorted(falling, reverse=True)
    humped = [
        bucket.volatility
        for bucket in strip((0.18, 0.26, 0.30, 0.26, 0.22), discount, projection, index).buckets
    ]
    assert humped != sorted(humped)
    assert humped != sorted(humped, reverse=True)


# -- the feasibility boundary -------------------------------------------------


def test_the_quote_curve_can_fall_only_so_far(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Measured by bisection, and sharp to a basis point of the quote.

    Below the boundary the two-year cap is worth less than its first three
    periods plus the intrinsic value of its four new ones, and no non-negative
    volatility on those four closes the gap.
    """
    pair = (date(2027, 1, 15), date(2028, 1, 15))

    def feasible(volatility: float) -> bool:
        try:
            strip_caplets(
                [CapQuote(pair[0], 0.18), CapQuote(pair[1], volatility)],
                REFERENCE,
                STRIKE,
                discount,
                projection,
                index=index,
            )
        except StrippingError:
            return False
        return True

    low, high = 0.0, 0.18
    for _ in range(50):
        middle = 0.5 * (low + high)
        if feasible(middle):
            high = middle
        else:
            low = middle
    assert high == pytest.approx(0.059832, abs=2e-5)
    assert not feasible(high - 1e-5)
    assert feasible(high + 1e-5)
    assert 0.18 - high == pytest.approx(0.1202, abs=2e-4)


def test_the_boundary_is_where_the_premium_stops_responding(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Not where a variance would go negative, which is the usual worry.

    At the last quote the strip accepts, the bucket is still a positive 0.23%
    and its indeterminacy has risen to within 0.3% of the 1.0e-04 limit -- so
    identification is what binds. One step below, the premium is also
    unattainable, and the two coincide rather than happening to be close: both
    say the premium has stopped responding to the volatility, one measured as a
    gap against the intrinsic value and the other as a width in the answer.
    """
    pair = (date(2027, 1, 15), date(2028, 1, 15))

    def strip_pair(volatility: float) -> CapletVolatility:
        return strip_caplets(
            [CapQuote(pair[0], 0.18), CapQuote(pair[1], volatility)],
            REFERENCE,
            STRIKE,
            discount,
            projection,
            index=index,
        )

    low, high = 0.0, 0.18
    for _ in range(50):
        middle = 0.5 * (low + high)
        try:
            strip_pair(middle)
        except StrippingError:
            low = middle
        else:
            high = middle

    bucket = strip_pair(high).buckets[1]
    assert 0.0 < bucket.volatility < 0.005
    assert bucket.volatility == pytest.approx(0.00231, rel=0.05)
    # Just under the limit, which is the claim. Where exactly a fifty-step
    # bisection stops depends on the interpreter's rounding -- 9.974e-05 on
    # 3.13 against 9.999e-05 on 3.11 -- so the assertion is the band and not
    # the digit.
    assert 0.99 * INDETERMINACY_LIMIT < bucket.indeterminacy < INDETERMINACY_LIMIT
    assert bucket.indeterminacy < INDETERMINACY_LIMIT
    assert 0.18 - high == pytest.approx(0.1202, abs=2e-4)
    assert strip_pair(high).buckets[0].volatility == pytest.approx(0.18, rel=1e-13)


def test_a_zero_quote_identifies_zero_only_exactly_at_the_money(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Because vega vanishes as the volatility does, away from the strike.

    At the money the premium is linear in the volatility with a slope of about
    ``0.4 F sqrt(T)``, so zero is pinned to 2.8e-16. Off the money ``d1`` and
    ``d2`` run off to infinity as the volatility falls, the density term goes
    with them, and a whole interval of volatilities produces the intrinsic
    value -- so the same quote is refused as unidentified. This is also the one
    case that reaches the branch returning zero without a root solve, which
    exists because Brent sees no sign change where the answer is at the edge of
    its bracket.
    """
    maturity = date(2026, 7, 15)
    cap = Cap(effective=REFERENCE, maturity=maturity, strike=0.03, index=index)
    periods = cap.caplets(discount, projection, 0.2)
    assert len(periods) == 1
    forward = periods[0].forward

    curve = strip_caplets(
        [CapQuote(maturity, 0.0)],
        REFERENCE,
        forward,
        discount,
        projection,
        index=index,
    )
    assert curve.buckets[0].volatility == 0.0
    assert curve.buckets[0].indeterminacy < 1e-15

    for strike in (0.030, 0.035):
        with pytest.raises(StrippingError, match="does not identify"):
            strip_caplets(
                [CapQuote(maturity, 0.0)],
                REFERENCE,
                strike,
                discount,
                projection,
                index=index,
            )


def test_an_unattainable_quote_says_what_the_attainable_range_is(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    with pytest.raises(StrippingError, match="only be worth between"):
        strip_caplets(
            [CapQuote(date(2027, 1, 15), 0.18), CapQuote(date(2028, 1, 15), 0.01)],
            REFERENCE,
            STRIKE,
            discount,
            projection,
            index=index,
        )


def test_a_quote_above_the_ceiling_is_refused(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """The premium is increasing in volatility without limit, so the ceiling
    is what makes "no solution" a refusal rather than a widening search."""
    with pytest.raises(StrippingError, match="only be worth between"):
        strip_caplets(
            [
                CapQuote(date(2027, 1, 15), 0.18),
                CapQuote(date(2028, 1, 15), VOLATILITY_CEILING * 2.0),
            ],
            REFERENCE,
            STRIKE,
            discount,
            projection,
            index=index,
        )


def test_flat_volatility_refuses_a_premium_outside_its_range(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    curve = CapletVolatility(
        strike=STRIKE,
        buckets=(
            CapletBucket(
                maturity=date(2031, 1, 15),
                first_fixing=date(2026, 4, 15),
                last_fixing=date(2030, 10, 15),
                periods=19,
                volatility=VOLATILITY_CEILING * 4.0,
                quoted=0.2,
                sensitivity=1.0,
            ),
        ),
    )
    with pytest.raises(StrippingError, match="outside the range"):
        flat_volatility(curve, date(2031, 1, 15), REFERENCE, discount, projection, index=index)


def test_the_premium_tolerance_is_relative_to_the_premium(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """A premium is a sum of option values, so its own scale is the right one.

    An absolute tolerance would mean a different number of significant figures
    on a cap struck near the money and one struck far from it. Checked at and
    above the money, where the premium is not degenerate.
    """
    assert PREMIUM_TOLERANCE == 1.0e-12
    for strike in (0.035, 0.08, 0.35):
        curve = strip(RISING, discount, projection, index, strike=strike)
        for maturity, quoted in zip(MATURITIES, RISING, strict=True):
            assert flat_volatility(
                curve, maturity, REFERENCE, discount, projection, index=index
            ) == pytest.approx(quoted, rel=1e-12)


def test_the_answer_is_pinned_to_a_tiny_fraction_of_a_basis_point_at_the_money(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    curve = strip(RISING, discount, projection, index)
    for bucket in curve.buckets:
        assert bucket.indeterminacy < 2e-12
    assert curve.buckets[0].indeterminacy == pytest.approx(1.24e-13, rel=0.15)


def test_a_deep_strike_is_refused_as_unidentified_rather_than_answered(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Monotone is not steep. The solver would return a number regardless.

    Eleven standard deviations into the money the option is worth its
    intrinsic value to machine precision and every volatility explains the
    quote, so a single answer is a lie about what the quote said.
    """
    for strike in (0.001, 0.005, 0.010):
        with pytest.raises(StrippingError, match="does not identify"):
            strip(RISING, discount, projection, index, strike=strike)


def test_the_identification_threshold_is_measured_not_assumed(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Sharp to a basis point of strike, against forwards of 3.23% to 3.39%."""
    one_year = (date(2027, 1, 15),)

    def identified(strike: float) -> bool:
        try:
            strip((0.18,), discount, projection, index, maturities=one_year, strike=strike)
        except StrippingError:
            return False
        return True

    low, high = 0.005, 0.02
    for _ in range(40):
        middle = 0.5 * (low + high)
        if identified(middle):
            high = middle
        else:
            low = middle
    assert high == pytest.approx(0.013901, abs=2e-5)
    assert not identified(high - 1e-5)
    assert identified(high + 1e-5)
    assert INDETERMINACY_LIMIT == 1.0e-4


def test_the_indeterminacy_narrows_as_the_strike_rises(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Six orders of magnitude between a 1.5% strike and an 8% one."""
    widths = []
    for strike in (0.015, 0.020, 0.025, 0.035, 0.050, 0.080):
        curve = strip(RISING, discount, projection, index, strike=strike)
        widths.append(curve.buckets[0].indeterminacy)
    assert widths == sorted(widths, reverse=True)
    assert widths[0] / widths[-1] > 1e8
    assert widths[0] == pytest.approx(6.19e-06, rel=0.1)


# -- floors, the normal convention, and the curve object ----------------------


def test_floors_strip_the_same_way(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    curve = strip(RISING, discount, projection, index, payoff=Payoff.RECEIVER)
    for maturity, quoted in zip(MATURITIES, RISING, strict=True):
        assert flat_volatility(
            curve,
            maturity,
            REFERENCE,
            discount,
            projection,
            index=index,
            payoff=Payoff.RECEIVER,
        ) == pytest.approx(quoted, rel=1e-13)


def test_the_normal_convention_strips_and_reprices(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Normal volatilities are a different quantity, so the levels differ."""
    normal = (0.006, 0.0065, 0.007, 0.0075, 0.008)
    curve = strip(normal, discount, projection, index, convention=Convention.NORMAL)
    for maturity, quoted in zip(MATURITIES, normal, strict=True):
        assert flat_volatility(
            curve,
            maturity,
            REFERENCE,
            discount,
            projection,
            index=index,
            convention=Convention.NORMAL,
        ) == pytest.approx(quoted, rel=1e-13)
    assert curve.buckets[0].volatility == pytest.approx(0.006, rel=1e-13)


def test_the_curve_is_piecewise_constant_and_flat_outside_its_range(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Extrapolating a slope past the last quote would invent one."""
    curve = strip(RISING, discount, projection, index)
    assert curve.volatility(date(2026, 1, 20)) == pytest.approx(curve.buckets[0].volatility)
    assert curve.volatility(date(2060, 1, 1)) == pytest.approx(curve.buckets[-1].volatility)
    for bucket in curve.buckets:
        assert curve.volatility(bucket.first_fixing) == pytest.approx(bucket.volatility)
        assert curve.volatility(bucket.last_fixing) == pytest.approx(bucket.volatility)


def test_the_buckets_tile_the_schedule_without_a_gap(
    discount: DiscountCurve, projection: DiscountCurve, index: ForecastIndex
) -> None:
    """Every period with optionality in it belongs to exactly one bucket."""
    curve = strip(RISING, discount, projection, index)
    cap = Cap(effective=REFERENCE, maturity=MATURITIES[-1], strike=STRIKE, index=index)
    periods = cap.caplets(discount, projection, 0.2, include_first=False)
    assert sum(bucket.periods for bucket in curve.buckets) == len(periods)
    fixings = [one.start for one in periods]
    assert curve.buckets[0].first_fixing == fixings[0]
    assert curve.buckets[-1].last_fixing == fixings[-1]
    for earlier, later in zip(curve.buckets[:-1], curve.buckets[1:], strict=True):
        assert earlier.last_fixing < later.first_fixing
