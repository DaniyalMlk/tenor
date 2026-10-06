"""Constant maturity swaps.

Four things here are exact rather than approximate, and between them they pin
down everything the module does.

**A single period reduces.** If the underlying swap has one fixed period and
the CMS pays at the end of it, the annuity is ``delta`` times the discount
factor to that date, so the mapping is ``1 / delta``, flat in the rate. Every
derivative of it is zero, the convexity adjustment is identically zero, and the
contract is a forward rate paid in arrears — which needs no adjustment, as
everyone knows without a model. So the CMS caplet has to equal the swaption
from :mod:`tenor.options` divided by the accrual, for every strike and both
payoffs, to rounding. That is the test that would catch the mapping being
inverted, the point mass at the kink having the wrong sign, or the payer and
receiver legs being swapped — the class of error that cost the previous phase a
sign.

**The second moment has a closed form.** Against that same flat mapping the
replication of ``S**2`` must return ``F**2 exp(sigma**2 T)`` under Black and
``F**2 + sigma**2 T`` under Bachelier. There is no model in either statement,
so it checks the Carr-Madan expansion, the Gauss-Legendre rule, the panel split
at the forward and the grading, against arithmetic this file did not produce.
It is also the only check here that is sensitive to the far tail, which is how
the uniform panel spacing was found to make a wider ceiling worse.

**Caplet less floorlet is swaplet less strike.** Exact for any volatility,
any mapping and any surface, so it leaves nowhere for a quadrature error to
hide, and it holds through the normalisation because both sides are divided by
the same number.

**Zero volatility is the forward.** No optionality, nothing to integrate, so
the adjustment is exactly 0.0 and not nearly.

The rest is the measured behaviour — how the adjustment moves with volatility,
with the payment delay and with the skew — and the refusals, of which the
interesting one is a surface the integral does not converge against at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date

import pytest

from tenor.cms import (
    AnnuityMap,
    BadConvexity,
    ConstantMaturity,
    ConstantMaturityLeg,
    OptionPayoff,
    PowerPayoff,
    Quadrature,
    RatePayoff,
    Strike,
    annuity_map,
    flat_smile,
    measure_error,
)
from tenor.curve import DiscountCurve, Interpolation
from tenor.daycount import Basis, year_fraction
from tenor.multicurve import ForecastIndex, shifted_forecast
from tenor.options import Convention, Payoff, Swaption
from tenor.schedule import Frequency, add_months

REFERENCE = date(2026, 1, 15)
FIXING = date(2031, 1, 15)
TEN_YEAR = date(2041, 1, 15)
SIX_MONTH = date(2031, 7, 15)

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
    return ForecastIndex(tenor=Frequency.SEMI_ANNUAL, basis=Basis.ACT_360)


@pytest.fixture
def ten_year(index: ForecastIndex) -> ConstantMaturity:
    """A ten-year rate fixing in five years, paid six months after it fixes."""
    return ConstantMaturity(
        expiry=FIXING,
        effective=FIXING,
        maturity=TEN_YEAR,
        payment=SIX_MONTH,
        index=index,
        frequency=Frequency.ANNUAL,
    )


@pytest.fixture
def single(index: ForecastIndex) -> ConstantMaturity:
    """One semi-annual period, paid at the end of it: the degenerate case."""
    return ConstantMaturity(
        expiry=FIXING,
        effective=FIXING,
        maturity=SIX_MONTH,
        payment=SIX_MONTH,
        index=index,
        frequency=Frequency.SEMI_ANNUAL,
    )


ACCRUAL = year_fraction(FIXING, SIX_MONTH, Basis.THIRTY_360_BOND)


# -- the mapping --------------------------------------------------------------


def test_one_period_maps_to_the_reciprocal_accrual(
    single: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """``A(T) = delta P(T, Tp)`` identically, so ``alpha`` is ``1 / delta`` flat.

    Checked across three decades of rate rather than at the forward, because
    the whole content of the claim is that the rate does not enter.
    """
    mapping = single.mapping(discount, projection)
    for rate in (0.0005, 0.01, 0.04, 0.20, 0.75):
        assert mapping.value(rate) == pytest.approx(1.0 / ACCRUAL, abs=1e-15)
        assert mapping.slope(rate) == pytest.approx(0.0, abs=1e-15)
        assert mapping.curvature(rate) == pytest.approx(0.0, abs=1e-15)


def test_one_period_needs_no_rescaling(
    single: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The curve's own ratio is also ``1 / delta`` there, so the scale is one.

    Exactly one, not nearly: both sides are the same two discount factors.
    """
    assert single.mapping(discount, projection).scale == pytest.approx(1.0, abs=5e-16)


def test_the_mapping_derivatives_agree_with_differences(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The analytic first and second derivatives, against central differences.

    A step of 1e-4 in the rate, which is a basis point: large enough that the
    second difference is not cancellation and small enough that the fourth
    derivative does not show. Both are checked, because the second derivative
    is the one the whole quadrature is weighted by and an error in it would
    otherwise look like a modelling choice.
    """
    mapping = ten_year.mapping(discount, projection)
    step = 1e-4
    for rate in (0.01, 0.0411, 0.08):
        up = mapping.value(rate + step)
        down = mapping.value(rate - step)
        here = mapping.value(rate)
        assert mapping.slope(rate) == pytest.approx((up - down) / (2 * step), rel=1e-7)
        assert mapping.curvature(rate) == pytest.approx(
            (up - 2 * here + down) / (step * step), rel=1e-5
        )


def test_the_flat_curve_mapping_is_arbitrageable_before_rescaling(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The raw model disagrees with the curve about a zero-coupon bond.

    It is the reason :func:`annuity_map` scales at all. The gap is 0.88% of the
    ratio on this upward-sloping curve — a quarter of the convexity adjustment
    the model exists to compute — so it is not a rounding concern.
    """
    scaled = annuity_map(ten_year, discount, projection)
    raw = AnnuityMap(
        accruals=scaled.accruals,
        offsets=scaled.offsets,
        delay=scaled.delay,
        frequency=scaled.frequency,
    )
    forward = ten_year.forward_rate(discount, projection)
    target = discount.discount(ten_year.payment) / ten_year.annuity(discount)
    assert scaled.value(forward) == pytest.approx(target, rel=1e-15)
    assert raw.value(forward) / target - 1.0 == pytest.approx(0.00884, abs=5e-5)


def test_a_mapping_refuses_a_rate_below_its_domain(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    mapping = ten_year.mapping(discount, projection)
    assert mapping.floor == -1.0
    with pytest.raises(BadConvexity, match="stops being positive"):
        mapping.value(mapping.floor)


@pytest.mark.parametrize(
    ("accruals", "offsets", "delay", "frequency", "message"),
    [
        ((), (), 0.5, 2, "at least one fixed period"),
        ((0.5, 0.5), (1.0,), 0.5, 2, "cannot be paired"),
        ((0.0,), (1.0,), 0.5, 2, "non-positive accrual"),
        ((0.5,), (1.0,), -0.5, 2, "before it"),
        ((0.5,), (1.0,), 0.5, 0, "times a year is not one"),
    ],
)
def test_a_mapping_refuses_what_it_cannot_be(
    accruals: tuple[float, ...],
    offsets: tuple[float, ...],
    delay: float,
    frequency: int,
    message: str,
) -> None:
    with pytest.raises(BadConvexity, match=message):
        AnnuityMap(
            accruals=accruals, offsets=offsets, delay=delay, frequency=frequency
        )


# -- the reduction to a vanilla swaption --------------------------------------


@pytest.mark.parametrize("strike", [0.030, 0.0400, 0.0411, 0.055])
@pytest.mark.parametrize("side", list(Strike))
def test_a_single_period_cms_option_is_a_swaption(
    single: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    index: ForecastIndex,
    strike: float,
    side: Strike,
) -> None:
    """The degenerate case, against :mod:`tenor.options` rather than against itself.

    With the mapping flat the replication is the point mass at the kink and
    nothing else, so this asserts the mass is ``alpha(K)`` with the right sign
    and that the payer and receiver legs are not crossed. It is the check that
    the previous phase's sign error would have failed.
    """
    replicated = single.option(strike, discount, projection, 0.24, side=side)
    payoff = Payoff.PAYER if side is Strike.ABOVE else Payoff.RECEIVER
    vanilla = Swaption(
        expiry=FIXING,
        effective=FIXING,
        maturity=SIX_MONTH,
        strike=strike,
        payoff=payoff,
        index=index,
        frequency=Frequency.SEMI_ANNUAL,
    ).value(discount, projection, 0.24)
    assert replicated.value == pytest.approx(vanilla / ACCRUAL, rel=5e-16)
    assert replicated.above == pytest.approx(0.0, abs=1e-18)
    assert replicated.below == pytest.approx(0.0, abs=1e-18)


def test_a_single_period_cms_has_no_convexity_at_all(
    single: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """A forward rate paid in arrears needs no adjustment, and gets exactly none."""
    for volatility in (0.0, 0.08, 0.24, 0.60):
        swaplet = single.swaplet(discount, projection, volatility)
        assert swaplet.adjustment == 0.0
        assert swaplet.rate == pytest.approx(swaplet.forward, rel=1e-15)
        assert swaplet.replication.measure == pytest.approx(1.0, abs=5e-16)


# -- the second moment, which has a closed form -------------------------------


@pytest.mark.parametrize(
    ("convention", "volatility"),
    [(Convention.LOGNORMAL, 0.24), (Convention.NORMAL, 0.0080)],
)
def test_the_square_replicates_to_its_closed_form(
    single: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    convention: Convention,
    volatility: float,
) -> None:
    """``E[S**2]`` against Black's and Bachelier's own second moments.

    Against the flat mapping, so no model enters: this is the quadrature, the
    expansion and the panel grading checked against arithmetic from outside
    this package.
    """
    forward = single.forward_rate(discount, projection)
    time = single.time_to_expiry(REFERENCE)
    replication = single.replicate(
        PowerPayoff(2),
        discount,
        projection,
        volatility,
        quadrature=Quadrature(convention=convention),
    )
    moment = replication.value / discount.discount(single.payment) * ACCRUAL / ACCRUAL
    expected = (
        forward**2 * math.exp(volatility**2 * time)
        if convention is Convention.LOGNORMAL
        else forward**2 + volatility**2 * time
    )
    assert moment == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("width", [8.0, 10.0, 14.0, 20.0, 30.0])
def test_the_square_is_stable_as_the_ceiling_widens(
    single: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    width: float,
) -> None:
    """Graded panels make a wider ceiling harmless; uniform ones did not.

    The second moment is the one functional here that is sensitive to the far
    tail, so it is where uniform spacing showed up: at a fixed panel count it
    went 3.2e-08, 7.9e-04 and 1.8e-01 relative as the ceiling went from ten
    standard deviations to twenty, monotonically *worse* for covering more of
    the distribution. Spacing the edges evenly in ``log K`` holds every one of
    these at rounding.
    """
    forward = single.forward_rate(discount, projection)
    time = single.time_to_expiry(REFERENCE)
    replication = single.replicate(
        PowerPayoff(2), discount, projection, 0.24, quadrature=Quadrature(width=width)
    )
    moment = replication.value / discount.discount(single.payment)
    assert moment == pytest.approx(forward**2 * math.exp(0.24**2 * time), rel=1e-12)


@pytest.mark.parametrize("panels", [16, 32, 64, 128])
def test_the_square_converges_in_the_panel_count(
    single: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    panels: int,
) -> None:
    """Sixteen graded panels already reach 5.1e-13; thirty-two reach rounding."""
    forward = single.forward_rate(discount, projection)
    time = single.time_to_expiry(REFERENCE)
    replication = single.replicate(
        PowerPayoff(2), discount, projection, 0.24, quadrature=Quadrature(panels=panels)
    )
    moment = replication.value / discount.discount(single.payment)
    assert moment == pytest.approx(forward**2 * math.exp(0.24**2 * time), rel=1e-11)


# -- parity -------------------------------------------------------------------


@pytest.mark.parametrize("strike", [0.020, 0.030, 0.0411, 0.060, 0.090])
def test_a_cms_caplet_less_a_floorlet_is_the_swaplet(
    ten_year: ConstantMaturity,
    discount: DiscountCurve,
    projection: DiscountCurve,
    strike: float,
) -> None:
    """Exact for any volatility and any mapping, so it hides nothing.

    It holds through the normalisation because both sides are divided by the
    same replicated unit payoff, and it reaches into the wings — a strike at
    0.020 and one at 0.090 against a forward near 0.041 — because a kink badly
    placed against the panel edges shows up there first and at the money not at
    all.
    """
    cap = ten_year.option(strike, discount, projection, 0.24, side=Strike.ABOVE)
    floor = ten_year.option(strike, discount, projection, 0.24, side=Strike.BELOW)
    swaplet = ten_year.swaplet(discount, projection, 0.24)
    unit = discount.discount(ten_year.payment)
    assert cap.value - floor.value == pytest.approx(
        swaplet.value - strike * unit, rel=1e-13
    )


def test_an_at_the_money_cms_straddle_is_symmetric(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Struck at the *adjusted* rate, not at the forward.

    The CMS rate is what the contract pays in expectation, so it is the strike
    the two legs are worth the same at — and striking at the forward instead
    leaves them 2.6 basis points of rate apart, which is the adjustment.
    """
    swaplet = ten_year.swaplet(discount, projection, 0.24)
    cap = ten_year.option(swaplet.rate, discount, projection, 0.24, side=Strike.ABOVE)
    floor = ten_year.option(swaplet.rate, discount, projection, 0.24, side=Strike.BELOW)
    assert cap.value == pytest.approx(floor.value, rel=1e-12)
    at_forward_cap = ten_year.option(
        swaplet.forward, discount, projection, 0.24, side=Strike.ABOVE
    )
    at_forward_floor = ten_year.option(
        swaplet.forward, discount, projection, 0.24, side=Strike.BELOW
    )
    assert at_forward_cap.value > at_forward_floor.value


# -- zero volatility ----------------------------------------------------------


def test_zero_volatility_is_the_forward_exactly(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Nothing to integrate, so the adjustment is 0.0 and the model is consistent."""
    swaplet = ten_year.swaplet(discount, projection, 0.0)
    assert swaplet.adjustment == 0.0
    assert swaplet.basis_points == 0.0
    assert swaplet.replication.arbitrage == pytest.approx(0.0, abs=5e-16)


def test_zero_volatility_leaves_an_option_at_its_intrinsic(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    forward = ten_year.forward_rate(discount, projection)
    unit = discount.discount(ten_year.payment)
    for strike in (0.030, 0.060):
        cap = ten_year.option(strike, discount, projection, 0.0, side=Strike.ABOVE)
        assert cap.value == pytest.approx(unit * max(forward - strike, 0.0), abs=1e-15)


# -- what the adjustment is worth ---------------------------------------------


def test_the_adjustment_is_positive_and_grows_with_volatility(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Measured, and not quadratic — which is the usual rule of thumb.

    The expansion says the adjustment is leading-order in the variance, and at
    low volatility it is: the ratio to ``sigma**2`` is 0.0382 at 6%. By 48% it
    is 0.0773, double. So ``sigma**2`` scaling is a low-volatility statement
    and reading a 48% adjustment off a 24% one underestimates it by a third.
    """
    ratios = []
    previous = -1.0
    for volatility in (0.06, 0.12, 0.24, 0.36, 0.48):
        adjustment = ten_year.swaplet(discount, projection, volatility).adjustment
        assert adjustment > previous
        previous = adjustment
        ratios.append(adjustment / volatility**2)
    assert ratios[0] == pytest.approx(0.0382, abs=5e-4)
    assert ratios[-1] == pytest.approx(0.0773, abs=5e-4)
    assert sorted(ratios) == ratios


def test_the_adjustment_at_the_reference_point(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The one figure the README and the docstring quote, asserted here.

    Twenty-six basis points on a 4.105% forward is 0.63% of the rate, which is
    far larger than anything being quoted as a spread on it.
    """
    swaplet = ten_year.swaplet(discount, projection, 0.24)
    assert swaplet.forward == pytest.approx(0.04105294, abs=5e-8)
    assert swaplet.basis_points == pytest.approx(26.062, abs=5e-3)


def test_normalising_moves_the_adjustment_by_a_measurable_amount(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The model's residual arbitrage is 3.6% of the number it produces.

    Which is the argument for enforcing the one no-arbitrage condition there is
    rather than noting it: 0.22% of a zero-coupon bond does not sound like much
    until it is a basis point of a twenty-six basis point adjustment.
    """
    normalised = ten_year.swaplet(discount, projection, 0.24)
    raw = ten_year.swaplet(discount, projection, 0.24, normalise=False)
    assert measure_error(ten_year, discount, projection, 0.24) == pytest.approx(
        2.169e-3, rel=5e-3
    )
    assert raw.adjustment / normalised.adjustment - 1.0 == pytest.approx(
        0.0363, abs=5e-4
    )


def test_the_adjustment_falls_with_the_payment_delay(
    index: ForecastIndex, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Monotone down, and through zero — the opposite of the obvious guess.

    Paying later does not compound the convexity; it unwinds it. The adjustment
    on this fixing runs 29.3, 26.1, 22.9, 16.8, 5.4, 0.1 and -23.9 basis points
    as the payment moves from the fixing date out to ten years after it.
    """
    adjustments = []
    for months in (0, 6, 12, 24, 48, 60, 120):
        fixing = ConstantMaturity(
            expiry=FIXING,
            effective=FIXING,
            maturity=TEN_YEAR,
            payment=add_months(FIXING, months, keep_end_of_month=True),
            index=index,
            frequency=Frequency.ANNUAL,
        )
        adjustments.append(
            1e4 * fixing.swaplet(discount, projection, 0.24).adjustment
        )
    assert adjustments == sorted(adjustments, reverse=True)
    assert adjustments[0] == pytest.approx(29.284, abs=5e-3)
    assert adjustments[-1] == pytest.approx(-23.862, abs=5e-3)
    assert adjustments[-2] > 0.0 > adjustments[-1]


def test_the_sign_flips_at_the_annuity_centre_of_mass(
    index: ForecastIndex, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """``alpha'(F)`` vanishes where the payment date meets the annuity's own mean.

    Structural rather than coincidental: ``alpha`` is one discount factor over a
    weighted sum of them, so moving the payment date to the weighted mean of
    the annuity's payment times makes the numerator and the denominator respond
    to the rate alike and the ratio stop depending on it to first order.
    Measured both ways on this fixing: the annuity-weighted mean payment time
    is 5.1872 years, and ``alpha'(F)`` crosses zero at 62 months, which is
    5.1667. The adjustment's own zero is about two months earlier, because
    ``alpha''`` contributes to it as well.
    """
    forward = -1.0
    slopes: dict[int, float] = {}
    for months in (58, 62, 66):
        fixing = ConstantMaturity(
            expiry=FIXING,
            effective=FIXING,
            maturity=TEN_YEAR,
            payment=add_months(FIXING, months, keep_end_of_month=True),
            index=index,
            frequency=Frequency.ANNUAL,
        )
        forward = fixing.forward_rate(discount, projection)
        slopes[months] = fixing.mapping(discount, projection).slope(forward)
    assert slopes[58] > 0.0 > slopes[66]
    assert slopes[62] == pytest.approx(0.0, abs=1e-3)

    swaption = ConstantMaturity(
        expiry=FIXING,
        effective=FIXING,
        maturity=TEN_YEAR,
        payment=SIX_MONTH,
        index=index,
        frequency=Frequency.ANNUAL,
    ).swaption(0.0)
    weighted = 0.0
    total = 0.0
    for period in swaption.swap.fixed_schedule():
        weight = year_fraction(
            period.start, period.end, Basis.THIRTY_360_BOND
        ) * discount.discount(period.payment)
        weighted += weight * year_fraction(
            FIXING, period.payment, Basis.THIRTY_360_BOND
        )
        total += weight
    assert weighted / total == pytest.approx(5.1872, abs=5e-4)


def test_a_downward_skew_reduces_the_adjustment(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """A CMS reads the whole surface, so the at-the-money point does not fix it.

    Every surface here leaves the at-the-money swaption worth exactly what the
    flat one does, and the adjustment still moves: 100.0%, 96.9%, 94.3%, 92.1%
    and 88.5% of the flat number at skews of 0, -0.10, -0.20, -0.30 and -0.50
    per unit of rate. An 8% error from a marking choice that no at-the-money
    quote can see.
    """
    forward = ten_year.forward_rate(discount, projection)
    flat = ten_year.swaplet(discount, projection, 0.24).adjustment

    def skewed(slope: float) -> Callable[[float], float]:
        def at(strike: float) -> float:
            return min(max(0.24 + slope * (strike - forward), 0.05), 0.60)

        return at

    shares = [
        ten_year.swaplet(discount, projection, skewed(slope)).adjustment / flat
        for slope in (0.0, -0.10, -0.20, -0.30, -0.50)
    ]
    assert shares[0] == pytest.approx(1.0, abs=1e-12)
    assert shares == sorted(shares, reverse=True)
    assert shares[3] == pytest.approx(0.921, abs=2e-3)
    assert shares[-1] == pytest.approx(0.885, abs=2e-3)


def test_an_upward_skew_does_not_converge_and_is_refused(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The integral has no value, and the ceiling would have decided the answer.

    Uncapped, the adjustment reads 35.6 basis points at a six standard
    deviation ceiling and 36565 at fourteen. The tolerance on the measured tail
    catches it where nothing in the number itself would have.
    """
    forward = ten_year.forward_rate(discount, projection)

    def rising(strike: float) -> float:
        return max(0.24 + 0.30 * (strike - forward), 0.05)

    with pytest.raises(BadConvexity, match="has not converged"):
        ten_year.swaplet(discount, projection, rising)

    loose = Quadrature(tolerance=10.0)
    spread = [
        1e4
        * ten_year.swaplet(
            discount,
            projection,
            rising,
            quadrature=Quadrature(width=width, tolerance=10.0),
        ).adjustment
        for width in (6.0, 14.0)
    ]
    assert spread[0] == pytest.approx(35.578, rel=1e-3)
    assert spread[1] > 1000.0 * spread[0]
    assert ten_year.swaplet(
        discount, projection, rising, quadrature=loose
    ).replication.truncation > 1.0


def test_a_converging_replication_leaves_nothing_beyond_the_ceiling(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The diagnostic has to be quiet on the cases that are fine, or it is noise."""
    swaplet = ten_year.swaplet(discount, projection, 0.24)
    assert swaplet.replication.truncation < 1e-20
    assert swaplet.replication.upper > 20.0 * swaplet.forward
    assert swaplet.replication.lower < 0.05 * swaplet.forward


# -- the leg ------------------------------------------------------------------


def test_a_cms_leg_is_the_sum_of_its_fixings(
    index: ForecastIndex, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Each period a fresh fixing on its own ten-year swap, and nothing shared.

    The accruals are the *paying* leg's, on its own basis, which is why a
    ten-year rate paid semi-annually is not a scaled ten-year swaption.
    """
    leg = ConstantMaturityLeg(
        effective=date(2027, 1, 15),
        maturity=date(2031, 1, 15),
        tenor_years=10,
        index=index,
        frequency=Frequency.ANNUAL,
    )
    fixings = leg.fixings(REFERENCE)
    assert len(fixings) == 8
    assert fixings[0].expiry == date(2027, 1, 15)
    assert fixings[0].maturity == date(2037, 1, 15)
    assert fixings[-1].maturity == date(2040, 7, 15)

    periods = {period.adjusted_start: period for period in leg.schedule()}
    total = sum(
        leg.accrual(periods[fixing.expiry])
        * fixing.swaplet(discount, projection, 0.24).value
        for fixing in fixings
    )
    assert leg.value(discount, projection, 0.24) == pytest.approx(total, rel=1e-15)


def test_a_cms_cap_and_floor_strip_satisfy_parity(
    index: ForecastIndex, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """The strip inherits the identity, period by period, so the leg cannot drift."""
    leg = ConstantMaturityLeg(
        effective=date(2027, 1, 15),
        maturity=date(2030, 1, 15),
        tenor_years=5,
        index=index,
        frequency=Frequency.ANNUAL,
    )
    strike = 0.040
    cap = leg.value(discount, projection, 0.24, strike=strike, side=Strike.ABOVE)
    floor = leg.value(discount, projection, 0.24, strike=strike, side=Strike.BELOW)
    swaplets = leg.value(discount, projection, 0.24)
    periods = {period.adjusted_start: period for period in leg.schedule()}
    fixed = sum(
        leg.accrual(periods[fixing.expiry]) * discount.discount(fixing.payment)
        for fixing in leg.fixings(REFERENCE)
    )
    assert cap - floor == pytest.approx(swaplets - strike * fixed, rel=1e-13)


def test_a_leg_drops_the_periods_that_have_already_fixed(
    index: ForecastIndex, discount: DiscountCurve
) -> None:
    leg = ConstantMaturityLeg(
        effective=date(2025, 1, 15),
        maturity=date(2027, 1, 15),
        tenor_years=5,
        index=index,
    )
    assert all(one.expiry > REFERENCE for one in leg.fixings(REFERENCE))
    stale = ConstantMaturityLeg(
        effective=date(2024, 1, 15),
        maturity=date(2025, 1, 15),
        tenor_years=5,
        index=index,
    )
    with pytest.raises(BadConvexity, match="has not already"):
        stale.fixings(REFERENCE)


# -- refusals -----------------------------------------------------------------


def test_a_fixing_cannot_pay_before_it_fixes(index: ForecastIndex) -> None:
    with pytest.raises(BadConvexity, match="before the rate fixes"):
        ConstantMaturity(
            expiry=FIXING,
            effective=FIXING,
            maturity=TEN_YEAR,
            payment=date(2030, 1, 15),
            index=index,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0.0}, "not one to integrate to"),
        ({"panels": 0}, "cannot cover a strike range"),
        ({"nodes": 1}, "cannot integrate a curvature"),
        ({"tolerance": 0.0}, "can never be met"),
    ],
)
def test_the_quadrature_refuses_settings_that_cannot_work(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(BadConvexity, match=message):
        Quadrature(**kwargs)  # type: ignore[arg-type]


def test_a_power_below_two_is_refused() -> None:
    with pytest.raises(BadConvexity, match="linear or constant"):
        PowerPayoff(1)


def test_a_smooth_payoff_has_no_point_mass(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """Asking for one is a programming error rather than a numerical edge."""
    mapping = ten_year.mapping(discount, projection)
    for payoff in (RatePayoff(), PowerPayoff(2)):
        assert payoff.nodes == ()
        with pytest.raises(BadConvexity, match="no kink"):
            payoff.mass(0.04, mapping)


def test_a_negative_volatility_is_refused() -> None:
    with pytest.raises(BadConvexity, match="is not one"):
        flat_smile(-0.01)


def test_a_surface_returning_nonsense_is_refused(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    def broken(strike: float) -> float:
        return -1.0 if strike > 0.05 else 0.24

    with pytest.raises(BadConvexity, match="returned a volatility"):
        ten_year.swaplet(discount, projection, broken)


def test_a_lognormal_surface_cannot_reach_a_negative_strike(
    index: ForecastIndex, discount: DiscountCurve
) -> None:
    """The refusal names the convention that can, rather than the arithmetic."""
    flat = DiscountCurve.from_zeros(
        REFERENCE,
        [(day, -0.004) for day, _ in OIS_QUOTES],
        basis=Basis.ACT_365F,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )
    fixing = ConstantMaturity(
        expiry=FIXING,
        effective=FIXING,
        maturity=TEN_YEAR,
        payment=SIX_MONTH,
        index=index,
        frequency=Frequency.ANNUAL,
    )
    with pytest.raises(BadConvexity, match="normal convention"):
        fixing.swaplet(flat, flat, 0.24)


def test_a_negative_rate_prices_under_the_normal_convention(
    index: ForecastIndex,
) -> None:
    """And the adjustment is still there, which is the point of the convention."""
    flat = DiscountCurve.from_zeros(
        REFERENCE,
        [(day, -0.004) for day, _ in OIS_QUOTES],
        basis=Basis.ACT_365F,
        interpolation=Interpolation.LOG_LINEAR_DISCOUNT,
    )
    fixing = ConstantMaturity(
        expiry=FIXING,
        effective=FIXING,
        maturity=TEN_YEAR,
        payment=SIX_MONTH,
        index=index,
        frequency=Frequency.ANNUAL,
    )
    swaplet = fixing.swaplet(
        flat, flat, 0.0060, quadrature=Quadrature(convention=Convention.NORMAL)
    )
    assert swaplet.forward < 0.0
    assert swaplet.adjustment != 0.0
    assert abs(swaplet.replication.arbitrage) < 1e-2


def test_a_leg_refuses_a_tenor_or_a_term_that_is_not_one(
    index: ForecastIndex,
) -> None:
    with pytest.raises(BadConvexity, match="is not a tenor"):
        ConstantMaturityLeg(
            effective=date(2027, 1, 15),
            maturity=date(2030, 1, 15),
            tenor_years=0,
            index=index,
        )
    with pytest.raises(BadConvexity, match="not after it starts"):
        ConstantMaturityLeg(
            effective=date(2030, 1, 15),
            maturity=date(2027, 1, 15),
            tenor_years=5,
            index=index,
        )


def test_an_option_struck_outside_the_quadrature_is_refused(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    """A kink the panels do not reach would silently lose the whole point mass."""
    with pytest.raises(BadConvexity, match="outside the"):
        ten_year.option(
            5.0, discount, projection, 0.24, quadrature=Quadrature(width=1.0)
        )


def test_the_payoffs_report_what_they_are(
    ten_year: ConstantMaturity, discount: DiscountCurve, projection: DiscountCurve
) -> None:
    assert RatePayoff().name == "the swap rate"
    assert RatePayoff(scale=0.0, offset=1.0).name == "a flat 1"
    assert RatePayoff(scale=2.0).name == "a linear payoff"
    assert PowerPayoff(3).name == "the swap rate to the power 3"
    assert OptionPayoff(0.04).name == "the rate above 4.0000%"
    assert OptionPayoff(0.04, side=Strike.BELOW).name == "the rate below 4.0000%"
    assert Strike.of(Payoff.PAYER) is Strike.ABOVE
    assert Strike.of(Payoff.RECEIVER) is Strike.BELOW
    assert "2041-01-15" in ten_year.name
