"""Tests for spreads, the short rate lattice, and option-adjusted spread.

Two things carry the weight. The lattice has to reprice the curve exactly
rather than closely, because everything measured on it is quoted in basis
points and a calibration error is indistinguishable from a spread. And the
option cost - Z-spread less OAS - has to come out as an identity, including
the case where there is no option and the two must be equal.

The exercise logic is checked in both directions and at both extremes: a call
struck far above any reachable value must give back the bullet exactly, and so
must a put struck far below it. Those two are the cheapest tests that catch a
comparison written the wrong way round, which otherwise produces a number that
still looks like a spread.
"""

from __future__ import annotations

from datetime import date

import pytest

from tenor.bond import Bond
from tenor.bootstrap import bootstrap
from tenor.curve import DiscountCurve
from tenor.daycount import Basis
from tenor.instruments import Deposit, Swap
from tenor.lattice import (
    BadLattice,
    Exercise,
    Lattice,
    lattice_price,
    option_adjusted_spread,
    option_cost,
    steps_between,
)
from tenor.spread import (
    BadSpread,
    discounted_with_spread,
    i_spread,
    par_rate_at,
    z_spread,
)

REFERENCE = date(2021, 1, 5)
STEP = 0.5
STEPS = 20


def sloped() -> DiscountCurve:
    quotes: list[Deposit | Swap] = [
        Deposit(REFERENCE, date(2021, 7, 5), 0.002, Basis.ACT_360),
        Swap(REFERENCE, date(2023, 1, 5), 0.006),
        Swap(REFERENCE, date(2026, 1, 5), 0.015),
        Swap(REFERENCE, date(2031, 1, 6), 0.022),
        Swap(REFERENCE, date(2041, 1, 7), 0.027),
    ]
    return bootstrap(REFERENCE, quotes, basis=Basis.ACT_365F).curve


def flat(rate: float = 0.025) -> DiscountCurve:
    return DiscountCurve.from_zeros(
        REFERENCE, [(date(2041, 1, 7), rate)], basis=Basis.ACT_365F
    )


# -- Z-spread and I-spread ----------------------------------------------------


def test_a_zero_spread_reproduces_the_curve_price() -> None:
    curve = sloped()
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.03)
    assert discounted_with_spread(bond, curve, REFERENCE, 0.0) == pytest.approx(
        bond.price_from_curve(curve, REFERENCE), abs=1e-13
    )


def test_a_bond_priced_on_the_curve_has_no_z_spread() -> None:
    """Exactly zero, which is the definition rather than a coincidence."""
    curve = sloped()
    for maturity, coupon in (
        (date(2023, 1, 5), 0.02),
        (date(2026, 1, 5), 0.02),
        (date(2036, 1, 5), 0.05),
    ):
        bond = Bond(REFERENCE, maturity, coupon)
        fair = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
        solved = z_spread(bond, fair, curve, REFERENCE)
        assert solved.converged
        assert solved.value == pytest.approx(0.0, abs=1e-12)


def test_a_bond_priced_on_the_curve_still_shows_an_i_spread() -> None:
    """The reason Z-spread is the honest one, in numbers.

    None of these bonds has any spread at all - each is priced exactly on the
    curve. The I-spread reports one anyway, growing with maturity and with
    distance from par, because it compares a whole bond against a single par
    rate at one point. On a flat curve every figure collapses to zero, which is
    why a flat curve tests nothing here.
    """
    curve = sloped()
    expected = {
        date(2023, 1, 5): -0.14,
        date(2026, 1, 5): -0.46,
        date(2031, 1, 5): -1.62,
        date(2036, 1, 5): -6.03,
    }
    coupons = {
        date(2023, 1, 5): 0.02,
        date(2026, 1, 5): 0.02,
        date(2031, 1, 5): 0.03,
        date(2036, 1, 5): 0.05,
    }
    measured = []
    for maturity, basis_points in expected.items():
        bond = Bond(REFERENCE, maturity, coupons[maturity])
        assert not bond.schedule().has_stub
        fair = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
        spread = 1e4 * i_spread(bond, fair, curve, REFERENCE)
        assert spread == pytest.approx(basis_points, abs=0.02)
        measured.append(spread)
    # Monotone in maturity, and all of it an artefact.
    assert measured == sorted(measured, reverse=True)

    # The same bonds on a flat curve show nothing.
    level = flat()
    for maturity, coupon in coupons.items():
        bond = Bond(REFERENCE, maturity, coupon)
        fair = bond.price_from_curve(level, REFERENCE) - bond.accrued(REFERENCE)
        assert 1e4 * i_spread(bond, fair, level, REFERENCE) == pytest.approx(
            0.0, abs=0.02
        )


def test_the_z_spread_moves_the_right_way_with_the_price() -> None:
    curve = sloped()
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.03)
    fair = bond.price_from_curve(curve, REFERENCE) - bond.accrued(REFERENCE)
    cheaper = z_spread(bond, fair - 2.0, curve, REFERENCE).value
    dearer = z_spread(bond, fair + 2.0, curve, REFERENCE).value
    assert cheaper > 0.0 > dearer


def test_the_par_benchmark_is_a_swap_rate_not_a_zero_rate() -> None:
    """They differ by the coupon effect, by more than spreads are quoted to."""
    curve = sloped()
    par = par_rate_at(curve, REFERENCE, date(2031, 1, 6))
    zero = curve.zero_rate(date(2031, 1, 6))
    assert par != pytest.approx(zero, abs=1e-5)
    assert abs(par - zero) > 1e-4


def test_a_price_no_spread_can_produce_is_refused() -> None:
    curve = sloped()
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.03)
    with pytest.raises(BadSpread, match="no spread discounts them"):
        z_spread(bond, -100.0, curve, REFERENCE, clean=False)


def test_the_dirty_and_clean_routes_agree() -> None:
    curve = sloped()
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.03)
    settlement = date(2021, 4, 5)
    clean = 98.0
    dirty = clean + bond.accrued(settlement)
    assert z_spread(bond, clean, curve, settlement).value == pytest.approx(
        z_spread(bond, dirty, curve, settlement, clean=False).value, abs=1e-14
    )


# -- the lattice --------------------------------------------------------------


def lattice(volatility: float = 0.15, curve: DiscountCurve | None = None) -> Lattice:
    return Lattice.calibrated(
        curve if curve is not None else sloped(),
        REFERENCE,
        steps=STEPS,
        step=STEP,
        volatility=volatility,
    )


def discount_at_step(curve: DiscountCurve, index: int) -> float:
    base = curve.discount(REFERENCE)
    return curve.discount_at(curve.time_to(REFERENCE) + index * STEP) / base


@pytest.mark.parametrize("volatility", [0.0, 0.05, 0.15, 0.40])
def test_the_lattice_reprices_the_curve_exactly(volatility: float) -> None:
    """Exactly, not closely. A calibration error is indistinguishable from a spread."""
    curve = sloped()
    built = lattice(volatility, curve)
    assert built.reprices(curve, REFERENCE, tolerance=1e-14)
    worst = max(
        abs(built.zero_price(index) - discount_at_step(curve, index))
        for index in range(1, STEPS + 1)
    )
    assert worst < 1e-15


def test_the_rates_fan_out_by_exactly_the_volatility_spacing() -> None:
    """The shape is fixed and only the level is solved, so this is checkable."""
    import math

    built = lattice(0.15)
    spacing = math.exp(2.0 * 0.15 * math.sqrt(STEP))
    for row in built.rates:
        for position in range(1, len(row)):
            assert row[position] / row[position - 1] == pytest.approx(spacing, rel=1e-12)


def test_every_rate_on_the_tree_is_positive() -> None:
    """The reason for a lognormal short rate rather than a normal one.

    A negative rate in a low branch would still produce a plausible OAS, while
    the exercise decisions in that branch were being taken about rates that do
    not exist.
    """
    for volatility in (0.0, 0.15, 0.50):
        for row in lattice(volatility).rates:
            assert all(rate > 0.0 for rate in row)


def test_a_zero_volatility_tree_has_one_rate_per_step() -> None:
    built = lattice(0.0)
    for row in built.rates:
        assert len(set(row)) == 1


def test_a_lattice_past_the_curve_is_refused() -> None:
    curve = sloped()
    with pytest.raises(BadLattice, match="past the curve's last pillar"):
        Lattice.calibrated(curve, REFERENCE, steps=100, step=1.0, volatility=0.15)


def test_a_lattice_that_is_not_a_lattice_is_refused() -> None:
    curve = sloped()
    with pytest.raises(BadLattice, match="at least one step"):
        Lattice.calibrated(curve, REFERENCE, steps=0, step=0.5, volatility=0.1)
    with pytest.raises(BadLattice, match="positive length of time"):
        Lattice.calibrated(curve, REFERENCE, steps=4, step=0.0, volatility=0.1)
    with pytest.raises(BadLattice, match="not negative"):
        Lattice.calibrated(curve, REFERENCE, steps=4, step=0.5, volatility=-0.1)
    with pytest.raises(BadLattice, match="outside a lattice"):
        lattice().zero_price(STEPS + 5)


def test_a_maturity_between_nodes_is_refused_rather_than_rounded() -> None:
    curve = sloped()
    assert steps_between(curve, REFERENCE, date(2031, 1, 5), STEP) == 20
    with pytest.raises(BadLattice, match="not a whole number"):
        steps_between(curve, REFERENCE, date(2031, 4, 20), STEP)
    with pytest.raises(BadLattice, match="less than one step"):
        steps_between(curve, REFERENCE, date(2021, 2, 1), STEP)


# -- valuing bonds on the tree ------------------------------------------------


def test_a_bullet_on_the_tree_matches_a_direct_sum_of_its_flows() -> None:
    """The tree's own discount factors, summed by hand. No options involved."""
    curve = sloped()
    built = lattice(0.15, curve)
    direct = sum(2.5 * discount_at_step(curve, index) for index in range(1, STEPS + 1))
    direct += 100.0 * discount_at_step(curve, STEPS)
    assert lattice_price(built, coupon=2.5) == pytest.approx(direct, abs=1e-12)


@pytest.mark.parametrize("volatility", [0.0, 0.15, 0.40])
def test_a_bullets_value_does_not_depend_on_the_volatility(volatility: float) -> None:
    """Nothing to exercise, so the tree can only give back the curve."""
    curve = sloped()
    reference = lattice_price(lattice(0.0, curve), coupon=2.5)
    assert lattice_price(lattice(volatility, curve), coupon=2.5) == pytest.approx(
        reference, abs=1e-12
    )


def test_an_option_that_never_binds_gives_back_the_bullet_exactly() -> None:
    """Both directions, which is the cheapest catch for an inverted comparison."""
    built = lattice(0.15)
    bullet = lattice_price(built, coupon=2.5)
    never_called = Exercise(
        tuple((index, 500.0) for index in range(10, STEPS)), issuer=True
    )
    never_put = Exercise(
        tuple((index, 1.0) for index in range(10, STEPS)), issuer=False
    )
    assert lattice_price(built, coupon=2.5, exercise=never_called) == bullet
    assert lattice_price(built, coupon=2.5, exercise=never_put) == bullet


def test_a_call_caps_the_value_and_a_put_floors_it() -> None:
    """Opposite sides of the trade, opposite directions."""
    built = lattice(0.15)
    bullet = lattice_price(built, coupon=2.5)
    schedule = tuple((index, 105.0) for index in range(10, STEPS))
    callable_price = lattice_price(
        built, coupon=2.5, exercise=Exercise(schedule, issuer=True)
    )
    putable_price = lattice_price(
        built, coupon=2.5, exercise=Exercise(schedule, issuer=False)
    )
    assert callable_price < bullet < putable_price


@pytest.mark.parametrize("issuer", [True, False])
def test_more_volatility_is_worth_more_to_whoever_holds_the_option(
    issuer: bool,
) -> None:
    curve = sloped()
    schedule = tuple((index, 105.0) for index in range(10, STEPS))
    prices = [
        lattice_price(
            lattice(volatility, curve),
            coupon=2.5,
            exercise=Exercise(schedule, issuer=issuer),
        )
        for volatility in (0.0, 0.10, 0.20, 0.40)
    ]
    if issuer:
        # The issuer holds it, so the holder's bond is worth less as it rises.
        assert prices == sorted(prices, reverse=True)
    else:
        assert prices == sorted(prices)
    assert max(prices) - min(prices) > 0.5


def test_an_option_exercised_in_every_state_is_volatility_free() -> None:
    """A pleasant consequence of the calibration being exact.

    A 5% coupon against a curve around 2% is so far in the money that the
    issuer calls at the first opportunity in every state. The exercise decision
    is then the same everywhere, so the price depends only on the state prices
    - and those sum to the curve's discount factor whatever the volatility.
    """
    curve = sloped()
    schedule = tuple((index, 100.0) for index in range(10, STEPS))
    prices = [
        lattice_price(
            lattice(volatility, curve),
            coupon=2.5,
            exercise=Exercise(schedule, issuer=True),
        )
        for volatility in (0.0, 0.02, 0.05)
    ]
    assert max(prices) - min(prices) < 1e-12
    # And it equals a bond that simply redeems at the first call date.
    redeemed = sum(2.5 * discount_at_step(curve, index) for index in range(1, 11))
    redeemed += 100.0 * discount_at_step(curve, 10)
    assert prices[0] == pytest.approx(redeemed, abs=1e-12)


# -- option-adjusted spread ---------------------------------------------------


def test_with_no_option_the_oas_is_the_z_spread() -> None:
    """The identity that has to hold before the option cost means anything."""
    built = lattice(0.15)
    fair = lattice_price(built, coupon=2.5)
    for offset in (-3.0, 0.0, 2.0):
        solved = option_adjusted_spread(built, fair + offset, coupon=2.5)
        without = option_adjusted_spread(built, fair + offset, coupon=2.5, exercise=None)
        assert solved.value == pytest.approx(without.value, abs=1e-15)
        assert option_cost(without.value, solved.value) == pytest.approx(0.0, abs=1e-15)


def test_the_tree_and_the_curve_agree_on_a_spread_to_within_discretisation() -> None:
    """They are not identical, and the gap is a stateable quantity.

    The tree works in uniform half-year steps and the bond in actual dates, so
    a flow the curve places at 0.4986 years the tree places at 0.5. On this
    curve the two spreads for the same bond at the same price differ by 0.23
    basis points - small, real, and not a bug to be tolerated silently.
    """
    curve = sloped()
    built = lattice(0.15, curve)
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.05)
    assert not bond.schedule().has_stub
    price = lattice_price(built, coupon=2.5) - 3.0

    on_tree = option_adjusted_spread(built, price, coupon=2.5).value
    on_curve = z_spread(bond, price - bond.accrued(REFERENCE), curve, REFERENCE).value
    assert 1e4 * abs(on_tree - on_curve) == pytest.approx(0.23, abs=0.05)


def test_a_callable_bonds_z_spread_overstates_what_the_holder_earns() -> None:
    """The reason OAS exists, as an identity rather than a description.

    Price the callable on the tree, then ask two questions of that one price:
    what spread reproduces it if the option is ignored, and what spread
    reproduces it if the option is modelled. The first is the Z-spread, the
    second the OAS, and the difference is what the option is worth.
    """
    built = lattice(0.15)
    schedule = tuple((index, 100.0) for index in range(10, STEPS))
    calls = Exercise(schedule, issuer=True)
    price = lattice_price(built, coupon=2.5, exercise=calls)

    ignoring = option_adjusted_spread(built, price, coupon=2.5).value
    modelling = option_adjusted_spread(built, price, coupon=2.5, exercise=calls).value
    cost = option_cost(ignoring, modelling)

    assert modelling == pytest.approx(0.0, abs=1e-12)
    assert 1e4 * ignoring == pytest.approx(89.0, abs=1.0)
    assert 1e4 * cost == pytest.approx(89.0, abs=1.0)
    assert cost > 0.0, "the holder is short the call, so it costs them"


def test_a_putable_bonds_option_cost_is_negative() -> None:
    """The holder is long the put, so the Z-spread understates what they earn."""
    built = lattice(0.15)
    puts = Exercise(tuple((index, 100.0) for index in range(10, STEPS)), issuer=False)
    price = lattice_price(built, coupon=2.5, exercise=puts)
    ignoring = option_adjusted_spread(built, price, coupon=2.5).value
    modelling = option_adjusted_spread(built, price, coupon=2.5, exercise=puts).value
    assert option_cost(ignoring, modelling) < 0.0


def test_the_option_cost_grows_with_volatility() -> None:
    """A more valuable option costs the holder more spread."""
    curve = sloped()
    schedule = tuple((index, 105.0) for index in range(10, STEPS))
    calls = Exercise(schedule, issuer=True)
    costs = []
    for volatility in (0.05, 0.15, 0.30):
        built = lattice(volatility, curve)
        price = lattice_price(built, coupon=2.5, exercise=calls)
        ignoring = option_adjusted_spread(built, price, coupon=2.5).value
        modelling = option_adjusted_spread(
            built, price, coupon=2.5, exercise=calls
        ).value
        costs.append(option_cost(ignoring, modelling))
    assert costs == sorted(costs)
    assert all(one > 0.0 for one in costs)


def test_an_oas_reprices_the_bond_it_came_from() -> None:
    built = lattice(0.15)
    calls = Exercise(tuple((index, 103.0) for index in range(8, STEPS)), issuer=True)
    price = lattice_price(built, coupon=2.5, exercise=calls) - 2.0
    solved = option_adjusted_spread(built, price, coupon=2.5, exercise=calls)
    assert solved.converged
    assert lattice_price(
        built, coupon=2.5, spread=solved.value, exercise=calls
    ) == pytest.approx(price, abs=1e-10)


def test_a_price_no_oas_can_reach_is_refused() -> None:
    built = lattice(0.15)
    with pytest.raises(BadLattice, match="no spread between"):
        option_adjusted_spread(built, 1e6, coupon=2.5)
