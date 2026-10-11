"""The value of the short's option to choose, not merely its geometry."""

from __future__ import annotations

import math
import random
from datetime import date
from itertools import pairwise

import pytest

from tenor import Bond, BondFuture, delivery_option, delivery_switch
from tenor.bond import Accrual
from tenor.daycount import Basis
from tenor.futures import BadDelivery, _legendre, _normal_density
from tenor.schedule import Frequency

SETTLEMENT = date(2026, 11, 20)
CONTRACT = BondFuture(
    first_delivery=date(2026, 12, 1), delivery=date(2026, 12, 31), label="USZ6"
)
UNROUNDED = BondFuture(
    first_delivery=date(2026, 12, 1),
    delivery=date(2026, 12, 31),
    label="USZ6",
    digits=None,
)
#: Near the level at which this basket's cheapest bond changes, so that the
#: option is actually worth something. Away from it the basket is one bond.
CROSSING = 0.0590


def deliverable(maturity: date, coupon: float, label: str) -> Bond:
    return Bond(
        effective=date(2010, 1, 1),
        maturity=maturity,
        coupon=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        basis=Basis.ACT_ACT_ISDA,
        accrual=Accrual.PERIOD_FRACTION,
        label=label,
    )


def basket() -> list[Bond]:
    """Four long bonds with similar maturities and very different coupons."""
    return [
        deliverable(maturity, coupon, label)
        for maturity, coupon, label in (
            (date(2046, 8, 15), 0.0175, "1.750% of 2046"),
            (date(2046, 11, 15), 0.0300, "3.000% of 2046"),
            (date(2047, 2, 15), 0.0450, "4.500% of 2047"),
            (date(2047, 5, 15), 0.0625, "6.250% of 2047"),
        )
    ]


# --------------------------------------------------------------------------- #
# the limits


def test_no_volatility_leaves_the_intrinsic_exactly() -> None:
    got = delivery_option(
        basket(), CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.0
    )
    assert got.futures_price == got.intrinsic
    assert got.convexity == 0.0
    assert got.switch == 0.0
    assert got.net == 0.0
    assert got.kinks == ()
    assert [share for _, share in got.probabilities].count(1.0) == 1
    assert sum(share for _, share in got.probabilities) == 1.0
    # And it is the bond the switch walk names at that level.
    walked = delivery_switch(
        basket(), CONTRACT, settlement=SETTLEMENT, repo=0.042, yields=[CROSSING]
    )
    assert got.intrinsic_cheapest == walked[0].cheapest


def test_a_basket_of_one_has_no_choice_and_the_option_is_still_not_zero() -> None:
    """The sign trap. Convexity does not vanish when the choice does."""
    one = basket()[:1]
    got = delivery_option(
        one, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    assert got.switch == 0.0
    assert got.convexity > 0.0
    assert got.net == pytest.approx(got.convexity, rel=1e-14)
    assert got.futures_price > got.intrinsic
    assert len(got.probabilities) == 1
    assert got.probabilities[0][0] == one[0].name
    assert got.probabilities[0][1] == pytest.approx(1.0, abs=1e-15)


def test_the_two_effects_have_fixed_signs_and_add_to_the_net() -> None:
    for level in (0.03, 0.045, CROSSING, 0.07, 0.09):
        for volatility in (0.001, 0.005, 0.009, 0.03):
            got = delivery_option(
                basket(),
                CONTRACT,
                settlement=SETTLEMENT,
                forward_level=level,
                volatility=volatility,
            )
            assert got.convexity >= 0.0, (level, volatility, got.convexity)
            # Non-positive up to the quadrature's round-off: when the choice is
            # never exercised the two integrals differ only in their panel
            # layout, and cancel to the last few bits rather than exactly.
            assert got.switch <= 1e-12, (level, volatility, got.switch)
            assert got.net == pytest.approx(got.convexity + got.switch, abs=1e-11)
            assert got.futures_price == pytest.approx(got.intrinsic + got.net, abs=1e-11)


def test_the_net_changes_sign_with_the_level_so_one_number_will_not_do() -> None:
    """At the crossing the choice dominates; away from it the convexity does."""
    at_crossing = delivery_option(
        basket(), CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    away = delivery_option(
        basket(), CONTRACT, settlement=SETTLEMENT, forward_level=0.03, volatility=0.009
    )
    assert at_crossing.net < 0.0
    assert away.net > 0.0
    # Away from the crossing one bond is cheapest with near certainty.
    assert max(share for _, share in away.probabilities) > 0.999


def test_the_switch_term_grows_with_volatility_and_the_convexity_faster() -> None:
    values = [
        delivery_option(
            basket(),
            CONTRACT,
            settlement=SETTLEMENT,
            forward_level=CROSSING,
            volatility=volatility,
        )
        for volatility in (0.002, 0.005, 0.009, 0.02, 0.04)
    ]
    switches = [-got.switch for got in values]
    convexities = [got.convexity for got in values]
    assert all(b > a for a, b in pairwise(switches))
    assert all(b > a for a, b in pairwise(convexities))
    # The switch is roughly linear in the deviation and the convexity quadratic,
    # so the convexity overtakes it: the net is negative at low volatility and
    # turns positive at high.
    assert values[0].net < 0.0
    assert values[-1].net > 0.0


# --------------------------------------------------------------------------- #
# against references


def test_the_convexity_term_is_half_the_second_derivative_times_the_variance() -> None:
    """And the remainder shrinks like the variance, which is the fourth moment."""
    bond = basket()[0]
    factor = CONTRACT.conversion_factor(bond)
    step = 1e-5
    second = (
        bond.clean_price(CROSSING + step, CONTRACT.delivery)
        - 2.0 * bond.clean_price(CROSSING, CONTRACT.delivery)
        + bond.clean_price(CROSSING - step, CONTRACT.delivery)
    ) / (step * step * factor)
    errors = []
    for volatility in (0.02, 0.01, 0.005, 0.0025):
        got = delivery_option(
            [bond],
            CONTRACT,
            settlement=SETTLEMENT,
            forward_level=CROSSING,
            volatility=volatility,
        )
        taylor = 0.5 * second * got.yield_deviation**2
        errors.append(abs(got.convexity / taylor - 1.0))
    ratios = [a / b for a, b in pairwise(errors)]
    assert all(3.7 < ratio < 4.3 for ratio in ratios), ratios
    assert errors[-1] < 1e-4


def test_the_futures_price_against_a_simulation_that_shares_no_derivation() -> None:
    bonds = basket()
    factors = [CONTRACT.conversion_factor(bond) for bond in bonds]
    got = delivery_option(
        bonds, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    rng = random.Random(7)
    draws = 60_000
    total = 0.0
    square = 0.0
    for _ in range(draws):
        level = CROSSING + got.yield_deviation * rng.gauss(0.0, 1.0)
        value = min(
            bond.clean_price(level, CONTRACT.delivery) / factor
            for bond, factor in zip(bonds, factors, strict=True)
        )
        total += value
        square += value * value
    mean = total / draws
    error = math.sqrt(max(square / draws - mean * mean, 0.0) / draws)
    assert abs(got.futures_price - mean) < 4.0 * error


def test_the_delivery_probabilities_are_a_distribution() -> None:
    for level in (0.045, CROSSING, 0.07):
        got = delivery_option(
            basket(), CONTRACT, settlement=SETTLEMENT, forward_level=level, volatility=0.009
        )
        shares = [share for _, share in got.probabilities]
        assert all(share >= 0.0 for share in shares)
        # The tail beyond nine standard deviations is 2.3e-19 and is not
        # assigned to anyone.
        assert sum(shares) == pytest.approx(1.0, abs=1e-15)
        assert len(got.kinks) == len({name for name, share in got.probabilities if share > 0.0}) - 1


def test_the_crossing_moves_between_settlement_and_delivery_by_ten_basis_points() -> None:
    """The neighbouring module's convention is not this one's, and it matters.

    :func:`delivery_switch` prices the basket at *settlement*, because it is
    describing the geometry of the basis a desk is looking at today. The option
    has to price it at *delivery*, because that is when the choice is made and
    what the invoice is computed on. On this basket the two crossings are 9 to
    10 basis points apart -- against a window in which the intermediate bond is
    cheapest that is a tenth of a basis point wide, so using one date's
    crossings with the other date's measure would put every region in the wrong
    place.
    """
    bonds = basket()
    factors = [CONTRACT.conversion_factor(bond) for bond in bonds]

    def crossing(day: date, first: int, second: int) -> float:
        def gap(level: float) -> float:
            return (
                bonds[first].clean_price(level, day) / factors[first]
                - bonds[second].clean_price(level, day) / factors[second]
            )

        low, high = 0.03, 0.09
        sign = gap(low)
        for _ in range(200):
            middle = 0.5 * (low + high)
            if middle <= low or middle >= high:
                break
            if gap(middle) * sign > 0.0:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    got = delivery_option(
        bonds, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    # The option's kinks are the delivery-date crossings, exactly.
    assert got.kinks[0] == pytest.approx(crossing(CONTRACT.delivery, 3, 2), abs=1e-10)
    assert got.kinks[1] == pytest.approx(crossing(CONTRACT.delivery, 2, 0), abs=1e-10)
    for first, second in ((3, 2), (2, 0), (3, 0)):
        gap = (
            crossing(CONTRACT.delivery, first, second)
            - crossing(SETTLEMENT, first, second)
        ) * 1e4
        assert 9.0 < gap < 10.5, (first, second, gap)
    # And the switch walk, which prices at settlement, names a different
    # cheapest bond at the option's own kink.
    walked = delivery_switch(
        bonds, CONTRACT, settlement=SETTLEMENT, repo=0.042, yields=[got.kinks[0]]
    )
    assert walked[0].cheapest != got.intrinsic_cheapest
    assert walked[0].margin > 1e-3


# --------------------------------------------------------------------------- #
# the numerical point, and how little it is worth here


def test_the_kinks_do_not_move_with_the_volatility() -> None:
    """They are solved for, not scanned, so the grid cannot leak into them.

    This is the test that caught the first implementation: a scan fine enough
    to resolve a tenth-of-a-basis-point window at one volatility is not fine
    enough at another, so the reported kinks -- and with them the delivery
    probabilities -- moved with the volatility for no reason but the grid.
    """
    kinks = [
        delivery_option(
            basket(),
            CONTRACT,
            settlement=SETTLEMENT,
            forward_level=CROSSING,
            volatility=volatility,
        ).kinks
        for volatility in (0.002, 0.009, 0.02, 0.05)
    ]
    for later in kinks[1:]:
        assert later == pytest.approx(kinks[0], abs=1e-12)


def test_aligning_the_panels_on_the_kinks_is_nearly_free_and_still_right() -> None:
    """Measured, because here it buys far less than it usually does.

    The discontinuity is in the first derivative only, and the two bonds that
    cross have nearly the same slope where they do, so a panel straddling a
    kink costs about 2e-6 of a price point rather than the orders of magnitude
    a jump in the value would cost. What it does cost is monotone convergence:
    the uniform grid's error is non-monotone in the panel count, so a two-point
    convergence check on it reports the method diverging.
    """
    bonds = basket()
    factors = [CONTRACT.conversion_factor(bond) for bond in bonds]
    got = delivery_option(
        bonds, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    deviation = got.yield_deviation
    nodes, weights = _legendre(16)

    def uniform(panels: int) -> float:
        low, high = CROSSING - 9.0 * deviation, CROSSING + 9.0 * deviation
        total = 0.0
        for panel in range(panels):
            left = low + (high - low) * panel / panels
            right = low + (high - low) * (panel + 1) / panels
            half, centre = 0.5 * (right - left), 0.5 * (right + left)
            for node, weight in zip(nodes, weights, strict=True):
                level = centre + half * node
                value = min(
                    bond.clean_price(level, CONTRACT.delivery) / factor
                    for bond, factor in zip(bonds, factors, strict=True)
                )
                total += (
                    half
                    * weight
                    * value
                    * _normal_density((level - CROSSING) / deviation)
                    / deviation
                )
        return total

    errors = [uniform(panels) - got.futures_price for panels in (24, 48, 96, 192, 384)]
    assert max(abs(error) for error in errors) < 1e-5
    assert max(abs(error) for error in errors) > 1e-7
    # Non-monotone: at least one refinement makes it worse.
    assert any(
        abs(later) > abs(earlier)
        for earlier, later in pairwise(errors)
    )


def test_the_published_rounding_of_the_factor_narrows_the_middle_of_the_basket() -> None:
    """A tenth of a basis point against one and a fifth, measured both ways.

    The conversion factors are published to four decimal places, and that
    rounding is not cosmetic: it shrinks the window in which the basket's
    intermediate bond is the cheapest by a factor of nine, and its chance of
    being delivered with it.
    """
    bonds = basket()
    rounded = delivery_option(
        bonds, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    exact = delivery_option(
        bonds, UNROUNDED, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    assert len(rounded.kinks) == 2
    assert len(exact.kinks) == 2
    rounded_window = (rounded.kinks[1] - rounded.kinks[0]) * 1e4
    exact_window = (exact.kinks[1] - exact.kinks[0]) * 1e4
    assert rounded_window == pytest.approx(0.131, abs=0.005)
    assert exact_window == pytest.approx(1.195, abs=0.01)
    assert exact_window / rounded_window == pytest.approx(9.1, abs=0.2)
    middle = dict(rounded.probabilities)["4.500% of 2047"]
    middle_exact = dict(exact.probabilities)["4.500% of 2047"]
    assert middle == pytest.approx(0.0017, abs=0.0004)
    assert middle_exact == pytest.approx(0.0157, abs=0.001)
    # The other intermediate bond is never cheapest at either rounding, which
    # is the honest description of a four-bond basket with two live members.
    assert dict(rounded.probabilities)["3.000% of 2046"] == 0.0
    assert dict(exact.probabilities)["3.000% of 2046"] == 0.0


def test_the_gauss_legendre_nodes_are_right() -> None:
    """They are computed here rather than imported, so they are checked here."""
    nodes, weights = _legendre(16)
    assert len(nodes) == len(weights) == 16
    assert sum(weights) == pytest.approx(2.0, rel=1e-15)
    assert all(a < b for a, b in pairwise(nodes))
    assert nodes[0] == pytest.approx(-nodes[-1], rel=1e-14)
    # A sixteen-point rule is exact for polynomials up to degree 31.
    for degree in range(0, 32):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        got = sum(weight * node**degree for node, weight in zip(nodes, weights, strict=True))
        assert got == pytest.approx(exact, abs=1e-13)
    # And not for degree 32.
    got = sum(weight * node**32 for node, weight in zip(nodes, weights, strict=True))
    assert abs(got - 2.0 / 33.0) > 1e-13


# --------------------------------------------------------------------------- #
# spreads and refusals


def test_a_spread_moves_the_crossing_and_can_remove_it() -> None:
    bonds = basket()
    flat = delivery_option(
        bonds, CONTRACT, settlement=SETTLEMENT, forward_level=CROSSING, volatility=0.009
    )
    tilted = delivery_option(
        bonds,
        CONTRACT,
        settlement=SETTLEMENT,
        forward_level=CROSSING,
        volatility=0.009,
        spreads=[-0.0015, -0.0005, 0.0005, 0.0015],
    )
    assert tilted.kinks != flat.kinks
    # The tilt makes the high-coupon bond cheapest well past the old crossing,
    # so the choice is gone and only the convexity is left.
    assert tilted.switch > flat.switch
    assert tilted.net > 0.0


def test_bad_arguments_are_refused() -> None:
    bonds = basket()
    with pytest.raises(BadDelivery, match="nothing to choose"):
        delivery_option([], CONTRACT, settlement=SETTLEMENT, forward_level=0.05, volatility=0.01)
    with pytest.raises(BadDelivery, match="line up"):
        delivery_option(
            bonds,
            CONTRACT,
            settlement=SETTLEMENT,
            forward_level=0.05,
            volatility=0.01,
            spreads=[0.0],
        )
    with pytest.raises(BadDelivery, match="non-negative"):
        delivery_option(
            bonds, CONTRACT, settlement=SETTLEMENT, forward_level=0.05, volatility=-0.01
        )
    with pytest.raises(BadDelivery, match="no time for the level to move"):
        delivery_option(
            bonds,
            CONTRACT,
            settlement=date(2027, 1, 1),
            forward_level=0.05,
            volatility=0.01,
        )
