"""Bermudan swaptions: two exact identities, one independent route, one measured floor.

There is no closed form for the price, so the question is what can be checked
against something that is not the thing being checked.

Two things are **exact**, and are asserted to the last bit rather than to a
tolerance. Forward induction makes the tree reprice every zero on its own grid,
which is an identity in the construction and not a convergence property. And a
model with no volatility has no decision to make, so the price collapses to the
best discounted intrinsic value — computed here from the curve alone, with no
tree in it.

One route is **independent**. Each exercise right is an option on the fixed leg
it would enter, and :mod:`tenor.hullwhite` prices that two ways, by Jamshidian's
decomposition and by a Gauss-Legendre integral. A Bermudan with one exercise date
is that European, so backward induction can be measured against analysis that
shares no code with it.

What remains is **measured**: the discretisation error against that European has
an oscillating component, because the exercise boundary does not sit on a node
and moves between nodes as the mesh changes. One step count tells you nothing
about the error at another, so the error is reported as the worst case over a
*band* of step counts, and the convergence claim is about how the band shrinks.
"""

from __future__ import annotations

import math
from datetime import date
from itertools import pairwise

import pytest

from tenor.bermudan import (
    BadBermudan,
    BermudanSwaption,
    TrinomialTree,
    bermudan_swaption,
    build_tree,
    coterminal_bonds,
    coterminal_europeans,
    short_rate_level,
)
from tenor.curve import DiscountCurve, flat_curve
from tenor.daycount import Basis
from tenor.hullwhite import HullWhite, Method, coupon_bond_option, swaption_price
from tenor.options import Payoff
from tenor.schedule import Frequency

REFERENCE = date(2026, 1, 15)
MATURITY = date(2036, 1, 15)
ANNUAL_DATES = tuple(date(year, 1, 15) for year in range(2029, 2036))


def flat() -> DiscountCurve:
    """Three per cent everywhere, where the step shift and ``phi`` coincide."""
    return flat_curve(REFERENCE, date(2046, 1, 15), 0.03, basis=Basis.ACT_365F)


def sloped() -> DiscountCurve:
    """Rising 15 basis points a year, which is what separates ``phi`` from a shift."""
    return DiscountCurve.from_zeros(
        REFERENCE,
        tuple((date(2026 + k, 1, 15), 0.015 + 0.0015 * k) for k in range(1, 21)),
        basis=Basis.ACT_365F,
    )


def humped() -> DiscountCurve:
    """A curve that rises then falls, so the forward changes sign of slope."""
    return DiscountCurve.from_zeros(
        REFERENCE,
        tuple(
            (date(2026 + k, 1, 15), 0.02 + 0.02 * math.sin(math.pi * k / 24.0))
            for k in range(1, 21)
        ),
        basis=Basis.ACT_365F,
    )


CURVES = {"flat": flat, "sloped": sloped, "humped": humped}


def payer(
    expiries: tuple[date, ...] = ANNUAL_DATES,
    strike: float = 0.03,
    payoff: Payoff = Payoff.PAYER,
) -> BermudanSwaption:
    return BermudanSwaption(
        expiries=expiries, maturity=MATURITY, strike=strike, payoff=payoff
    )


def forward_swap_value(
    curve: DiscountCurve, option: BermudanSwaption, position: int
) -> float:
    """Today's value of the swap one exercise date would enter, from the curve alone.

    No model and no tree: the fixed leg is discounted off the curve and the
    floating leg is par at the start, so this is the intrinsic value the
    zero-volatility case has to reproduce.
    """
    bond = coterminal_bonds(option, REFERENCE)[position]
    legs = math.fsum(
        flow.amount * curve.discount_at(flow.time) for flow in bond.flows
    )
    return option.payoff.sign * (curve.discount_at(bond.expiry) - legs)


# -- the tree's own construction ----------------------------------------------


@pytest.mark.parametrize("name", sorted(CURVES))
@pytest.mark.parametrize("steps_per_year", [7, 16, 33])
def test_forward_induction_reprices_every_grid_zero(name: str, steps_per_year: int) -> None:
    """Exact, not approximate. The shift is chosen to make it so."""
    curve = CURVES[name]()
    tree = build_tree(
        curve, HullWhite(a=0.05, sigma=0.012), (3.0, 5.0, 8.5), steps_per_year
    )
    assert tree.curve_error(curve) < 1e-14


def test_state_price_mass_is_the_discount_factor() -> None:
    """Nothing is discarded, so the mass is the discount factor and not a share of it.

    This is the payoff of branching to the nodes around each node's own
    conditional mean: there is no truncated tail whose probability has to be
    pushed back onto an edge.
    """
    curve = sloped()
    tree = build_tree(curve, HullWhite(a=0.08, sigma=0.015), (4.0, 7.0), 24)
    for time, prices in zip(tree.times, tree.state_prices, strict=True):
        assert math.fsum(prices) == pytest.approx(curve.discount_at(time), rel=1e-14)
        assert all(price >= 0.0 for price in prices)


@pytest.mark.parametrize("a,sigma", [(0.02, 0.004), (0.05, 0.01), (0.2, 0.02), (0.6, 0.03)])
def test_every_branching_probability_is_strictly_positive(a: float, sigma: float) -> None:
    """The reason no cap on the tree width is needed.

    Rounding to the nearest node bounds the residual drift by half a spacing, so
    ``v + u**2 <= 1`` and ``v + u**2 >= |u|`` hold everywhere with ``v`` near a
    third. Reconstructed here from the tree's own geometry rather than trusted.
    """
    model = HullWhite(a=a, sigma=sigma)
    tree = build_tree(sloped(), model, (2.0, 6.0, 9.0), 20)
    worst_middle = 1.0
    worst_wing = 1.0
    for index, ratio in enumerate(tree.variance_ratio):
        gap = tree.times[index + 1] - tree.times[index]
        decay = math.exp(-model.a * gap)
        for node in range(tree.lower[index], tree.upper[index] + 1):
            residual = node * decay - round(node * decay)
            assert abs(residual) <= 0.5 + 1e-12
            square = residual * residual
            up = 0.5 * (ratio + square + residual)
            middle = 1.0 - ratio - square
            down = 0.5 * (ratio + square - residual)
            assert up > 0.0 and down > 0.0
            assert middle > 0.0
            assert up + middle + down == pytest.approx(1.0, abs=1e-15)
            worst_middle = min(worst_middle, middle)
            worst_wing = min(worst_wing, up, down)
    assert worst_middle > 0.0
    assert worst_wing > 0.0


def test_node_width_saturates_under_mean_reversion() -> None:
    """The width stops growing on its own, which is why nothing has to stop it.

    A node branches to the nodes around ``j e^{-a dt}``, so once ``j a dt``
    exceeds half a spacing the centre comes back one node and the top stops
    moving. No truncation rule imposes this.
    """
    tree = build_tree(flat(), HullWhite(a=0.5, sigma=0.01), (10.0,), 24)
    widths = [hi - lo for lo, hi in zip(tree.lower, tree.upper, strict=True)]
    assert widths[-1] == max(widths)
    assert widths[-1] < tree.steps  # strictly narrower than one node per step
    # and it really has stopped, rather than merely slowed
    assert widths[-1] == widths[len(widths) // 2]


def test_uneven_grid_is_refused_rather_than_approximated() -> None:
    """One spacing cannot serve step variances that differ by more than threefold."""
    with pytest.raises(BadBermudan, match="too uneven for one state spacing"):
        build_tree(flat(), HullWhite(a=0.05, sigma=0.01), (0.02, 10.0), 1)


@pytest.mark.parametrize(
    "times,steps,match",
    [
        ((1.0, 1.0), 12, "ascending"),
        ((2.0, 1.0), 12, "ascending"),
        ((-1.0,), 12, "ascending"),
        ((1.0,), 0, "steps per year must be positive"),
        ((), 12, "at least one required time"),
    ],
)
def test_build_tree_refuses_a_grid_it_cannot_make(
    times: tuple[float, ...], steps: int, match: str
) -> None:
    with pytest.raises(BadBermudan, match=match):
        build_tree(flat(), HullWhite(a=0.05, sigma=0.01), times, steps)


def test_required_times_land_exactly_on_nodes() -> None:
    """Not nearly on nodes. An exercise date off a node smears the decision's kink."""
    required = (1.0, 2.5, 7.25)
    tree = build_tree(sloped(), HullWhite(a=0.05, sigma=0.01), required, 19)
    for time in required:
        assert time in tree.times


def test_short_rate_step_is_outside_the_tree() -> None:
    tree = build_tree(flat(), HullWhite(a=0.05, sigma=0.01), (1.0,), 8)
    with pytest.raises(BadBermudan, match="outside the tree"):
        tree.short_rate(len(tree.shift), 0)


def average_forward(curve: DiscountCurve, left: float, right: float) -> float:
    """The step's average instantaneous forward, which needs no model to compute."""
    return -math.log(curve.discount_at(right) / curve.discount_at(left)) / (right - left)


@pytest.mark.parametrize("name", sorted(CURVES))
@pytest.mark.parametrize("a,sigma", [(0.05, 0.01), (0.1, 0.02)])
def test_forward_induction_recovers_the_convexity_term(
    name: str, a: float, sigma: float
) -> None:
    """An identity hiding inside the fit, and the reason the shift is not naive.

    Reproducing a discount factor with one rate per step looks like a statement
    about the average instantaneous forward, which is a curve quantity with no
    model in it. The fitted shift sits above that average by exactly the second
    term of ``phi``, the convexity the volatility adds, because the
    discount-weighted spread of the state across the step *is* that convexity.

    Measured at 8.9e-04 against an analytic 8.9e-04 for ``a = 0.05, sigma = 0.01``
    at five years, agreeing to 1.3e-05 relative at four steps a year and 9.0e-08
    at forty-eight, and identically on all three curves.
    """
    curve = CURVES[name]()
    model = HullWhite(a=a, sigma=sigma)
    errors = []
    for steps_per_year in (4, 12, 48):
        tree = build_tree(curve, model, (5.0,), steps_per_year)
        step = tree.times.index(5.0) - 1
        left = tree.times[step]
        recovered = tree.shift[step] - average_forward(curve, left, tree.times[step + 1])
        analytic = short_rate_level(curve, model, left) - curve.instantaneous_forward(left)
        assert analytic > 1e-4  # there is a convexity term to find
        errors.append(abs(recovered / analytic - 1.0))
    assert errors[0] < 1e-4
    assert errors[0] > errors[1] > errors[2]


def test_without_volatility_the_shift_the_average_and_phi_all_coincide() -> None:
    """No volatility, no convexity, no spread — so the three quantities collapse.

    Which is why this is the case that can be checked against arithmetic off the
    curve alone, and why ``phi`` rather than the shift is still what a payoff is
    read at: the separation is a property of the model, not of the mesh.
    """
    curve = sloped()
    model = HullWhite(a=0.05, sigma=0.0)
    tree = build_tree(curve, model, (5.0,), 4)
    step = tree.times.index(5.0) - 1
    left = tree.times[step]
    average = average_forward(curve, left, tree.times[step + 1])
    assert tree.shift[step] == pytest.approx(average, rel=1e-14)
    assert short_rate_level(curve, model, left) == pytest.approx(average, abs=1e-9)


@pytest.mark.parametrize("name", sorted(CURVES))
def test_a_bond_outliving_the_grid_reprices_to_first_order_in_the_step(name: str) -> None:
    """The scheme's dominant error, measured rather than inferred from the price.

    ``curve_error`` is exact by construction, but it only sees bonds maturing on
    the grid. An exercise value pays one that outlives it, read analytically at
    the node, so the quantity that matters is whether the state prices weighted
    by those analytic prices add back to today's discount factor.

    They are short by 9.7e-05 at twelve steps a year and halve at every doubling:
    a ratio of 2.00 at each step, the same on all three curves, and always
    negative. First order in the step, which is the rectangle rule the discounting
    uses and the reason the price's own convergence is first order too.
    """
    curve = CURVES[name]()
    model = HullWhite(a=0.05, sigma=0.01)
    errors = []
    for steps_per_year in (12, 24, 48, 96):
        tree = build_tree(curve, model, (5.0,), steps_per_year)
        error = tree.forward_bond_error(curve, model, tree.times.index(5.0), 12.0)
        assert error < 0.0
        errors.append(abs(error))
    for coarse, fine in pairwise(errors):
        assert coarse / fine == pytest.approx(2.0, abs=0.1)


# -- the two exact identities --------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURVES))
@pytest.mark.parametrize("payoff", list(Payoff))
@pytest.mark.parametrize("steps_per_year", [8, 32])
def test_zero_volatility_is_the_best_discounted_intrinsic(
    name: str, payoff: Payoff, steps_per_year: int
) -> None:
    """With nothing to learn there is nothing to wait for: pick the best date.

    The benchmark is built from the curve alone, so this tests the exercise
    values, the discounting and the recursion against arithmetic with no model
    in it. It is exact rather than approximate, and it was not: evaluating the
    payoff at the tree's step shift instead of at ``phi`` put it 5.3e-03 out on
    the sloped curve.
    """
    curve = CURVES[name]()
    option = payer(payoff=payoff)
    value = bermudan_swaption(
        curve, HullWhite(a=0.05, sigma=0.0), option, REFERENCE,
        steps_per_year=steps_per_year,
    )
    best = max(
        [0.0]
        + [
            forward_swap_value(curve, option, position)
            for position in range(len(option.expiries))
        ]
    )
    assert math.isfinite(value.price)
    assert value.price == pytest.approx(best, abs=1e-15)


def test_zero_volatility_price_is_finite_at_every_step_count() -> None:
    """A degenerate model gets a degenerate tree.

    Widening a tree with no volatility in it produced nodes of exactly zero
    probability whose rates were a placeholder spacing apart. A bond price at one
    of those overflowed to infinity and the leg subtraction returned NaN, which
    is the worst available outcome: a number-shaped non-number in a payload.
    """
    option = payer()
    for steps_per_year in (8, 16, 32, 64, 96):
        value = bermudan_swaption(
            sloped(), HullWhite(a=0.05, sigma=0.0), option, REFERENCE,
            steps_per_year=steps_per_year,
        )
        assert math.isfinite(value.price)
        assert value.tree.spacing == 0.0
        assert value.tree.lower == value.tree.upper


def test_jamshidian_and_the_integral_agree_on_the_coterminal_bonds() -> None:
    """The independent route is only independent if both of its halves work.

    Measured at 3.2e-15 relative, which is the decomposition confirming the
    quadrature and not a tolerance either of them was tuned to.
    """
    curve = sloped()
    model = HullWhite(a=0.05, sigma=0.01)
    option = payer()
    decomposed = coterminal_europeans(curve, model, option, REFERENCE)
    integrated = coterminal_europeans(
        curve, model, option, REFERENCE, method=Method.QUADRATURE
    )
    assert len(decomposed) == len(option.expiries)
    for one, other in zip(decomposed, integrated, strict=True):
        assert one == pytest.approx(other, rel=1e-12)


# -- against the independent European -----------------------------------------


def band_error(
    curve: DiscountCurve,
    model: HullWhite,
    option: BermudanSwaption,
    densities: range,
) -> float:
    """Worst absolute error against the analytic European over a band of meshes.

    One mesh is a coin flip: the exercise boundary's position between nodes sets
    the sign and much of the size of the error, so a single step count can land
    an order of magnitude below the band it sits in. At 48 steps a year this
    instrument happened to come within 1.9e-06 of the analytic price while its
    band's worst case was 1.8e-05, and reading convergence off such a point gave
    a ratio of -865.
    """
    exact = coupon_bond_option(
        curve, model, coterminal_bonds(option, REFERENCE)[0], Method.JAMSHIDIAN
    )
    return max(
        abs(
            bermudan_swaption(
                curve, model, option, REFERENCE, steps_per_year=density
            ).price
            - exact
        )
        for density in densities
    )


@pytest.mark.parametrize("name", sorted(CURVES))
@pytest.mark.parametrize("payoff", list(Payoff))
def test_one_exercise_date_is_the_european(name: str, payoff: Payoff) -> None:
    """Measured to an absolute tolerance in units of notional, stated as such.

    Three hundredths of a basis point of notional, which is two orders inside any
    bid-offer on the instrument.
    """
    curve = CURVES[name]()
    option = payer(expiries=(date(2031, 1, 15),), payoff=payoff)
    assert band_error(curve, HullWhite(a=0.05, sigma=0.01), option, range(32, 45, 4)) < 5e-5


def test_the_error_band_shrinks_with_the_mesh() -> None:
    """Convergence asserted on the band, because the point errors oscillate.

    Measured on the flat curve: 4.8e-05 over 24-40 steps a year, 1.8e-05 over
    48-64 and 6.5e-06 over 96-112 — about 2.7 per doubling. The assertion is
    deliberately weaker than the measurement, since the band's worst case is
    itself a sample of a few meshes.
    """
    curve = flat()
    model = HullWhite(a=0.05, sigma=0.01)
    option = payer(expiries=(date(2031, 1, 15),))
    coarse = band_error(curve, model, option, range(24, 41, 4))
    medium = band_error(curve, model, option, range(48, 65, 4))
    fine = band_error(curve, model, option, range(96, 113, 8))
    assert coarse > medium > fine
    assert coarse / medium > 1.5
    assert medium / fine > 1.5


def test_a_deep_in_the_money_payer_still_holds_optionality() -> None:
    """A payoff positive everywhere loses its kink, not its choice.

    This test first asserted the opposite — that a payer struck at a basis point
    is worth exactly the forward swap it would be exercised into, because there
    is nothing left to wait for. That is wrong, and the number it was wrong by,
    1.33e-03 on a swap worth 0.202, is stable to three figures across meshes from
    24 to 96 steps a year, which is how it was identified as value rather than
    discretisation.

    The reason: the co-terminal swap shortens with every date, so its value falls
    along the forward curve and date one dominates *in expectation*. Per path it
    does not. Where rates have fallen by date one the swap is worth less than
    expected and holding on is worth more than taking it, and 7 to 9 per cent of
    the state-price mass does exactly that.
    """
    curve = sloped()
    model = HullWhite(a=0.05, sigma=0.01)
    option = payer(strike=0.0001)
    immediate = forward_swap_value(curve, option, 0)
    # every later date enters a shorter swap worth less, so date one leads
    assert immediate > forward_swap_value(curve, option, 1)
    prices = [
        bermudan_swaption(
            curve, model, option, REFERENCE, steps_per_year=density
        )
        for density in (24, 48, 96)
    ]
    for value in prices:
        assert value.price > immediate
        assert value.price - immediate == pytest.approx(1.33e-3, abs=5e-5)
        assert 0.85 < value.steps[0].exercised < 1.0


def test_the_reference_swaption_agrees_where_no_roll_moved_the_boundary() -> None:
    """A transcription check on the flows, against the package's public swaption.

    ``reference_swaption`` regenerates its own schedule, so it describes the same
    swap only where the exercise date needed no business-day roll. Where it does,
    the two differ by the days between, which is why the Europeans this module
    prices are built from the tail flows instead.
    """
    curve = sloped()
    model = HullWhite(a=0.05, sigma=0.01)
    expiry = date(2031, 1, 15)  # a Wednesday, so no roll
    option = payer(expiries=(expiry,))
    schedule = option.schedule()
    assert schedule[0].adjusted_start == expiry
    through_swaption = swaption_price(
        curve, model, option.reference_swaption(expiry), REFERENCE
    ).price
    through_flows = coterminal_europeans(curve, model, option, REFERENCE)[0]
    assert through_flows == pytest.approx(through_swaption, rel=1e-12)


# -- what the extra dates buy --------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURVES))
@pytest.mark.parametrize("payoff", list(Payoff))
@pytest.mark.parametrize("strike", [0.02, 0.03, 0.04])
def test_a_bermudan_beats_its_best_european(
    name: str, payoff: Payoff, strike: float
) -> None:
    """Exercising only on the best single date is a strategy available to it.

    So the switch value is non-negative as a matter of the instrument, not of the
    numerics, and a negative one would mean the discretisation error had grown
    past the optionality.
    """
    value = bermudan_swaption(
        CURVES[name](),
        HullWhite(a=0.05, sigma=0.01),
        payer(strike=strike, payoff=payoff),
        REFERENCE,
        steps_per_year=24,
    )
    assert value.switch_value > 0.0
    assert value.price > value.best_european
    assert value.best_european == max(value.europeans)


@pytest.mark.parametrize("payoff", list(Payoff))
def test_adding_an_exercise_date_never_costs_anything(payoff: Payoff) -> None:
    """A superset of rights, so a value that cannot fall. Checked from both ends."""
    curve = sloped()
    model = HullWhite(a=0.05, sigma=0.01)
    previous = -1.0
    for count in range(1, len(ANNUAL_DATES) + 1):
        value = bermudan_swaption(
            curve, model, payer(expiries=ANNUAL_DATES[-count:], payoff=payoff),
            REFERENCE, steps_per_year=24,
        ).price
        assert value >= previous - 1e-12
        previous = value
    previous = -1.0
    for count in range(1, len(ANNUAL_DATES) + 1):
        value = bermudan_swaption(
            curve, model, payer(expiries=ANNUAL_DATES[:count], payoff=payoff),
            REFERENCE, steps_per_year=24,
        ).price
        assert value >= previous - 1e-12
        previous = value


def test_payer_receiver_parity_holds_for_the_europeans_and_fails_for_the_bermudans() -> None:
    """The identity every European obeys, and the reason a Bermudan cannot.

    A European payer less a European receiver is the forward swap, exactly: the
    two payoffs are the positive and negative parts of one number. Two Bermudans
    are maxima over different stopping rules, and a difference of maxima is not
    the maximum of a difference. Measured here at 0.00076 on a forward swap worth
    0.00136 — the gap is 56% of the swap, so treating Bermudan parity as a check
    would report a 56% pricing error on correct prices.
    """
    curve = flat()
    model = HullWhite(a=0.05, sigma=0.01)
    as_payer = payer()
    as_receiver = payer(payoff=Payoff.RECEIVER)
    swap = forward_swap_value(curve, as_payer, 0)
    europeans = (
        coterminal_europeans(curve, model, as_payer, REFERENCE)[0]
        - coterminal_europeans(curve, model, as_receiver, REFERENCE)[0]
    )
    assert europeans == pytest.approx(swap, rel=1e-10)
    bermudans = (
        bermudan_swaption(curve, model, as_payer, REFERENCE, steps_per_year=24).price
        - bermudan_swaption(
            curve, model, as_receiver, REFERENCE, steps_per_year=24
        ).price
    )
    assert bermudans > swap
    assert abs(bermudans - swap) / abs(swap) > 0.4


# -- the exercise boundary -----------------------------------------------------


@pytest.mark.parametrize("a,sigma", [(0.02, 0.005), (0.05, 0.01), (0.15, 0.02), (0.4, 0.012)])
def test_the_exercise_decision_has_one_boundary_and_it_falls(a: float, sigma: float) -> None:
    """Measured, not assumed, and over a spread of parameters.

    A single sign change across the nodes means one rate describes the decision
    at that date. That it falls over time is the economics: with less swap left,
    the continuation value is smaller and less is given up by exercising, so a
    payer exercises for less. Measured at 3.92% down to 2.95% over seven annual
    dates at ``a = 0.05``.
    """
    value = bermudan_swaption(
        flat(), HullWhite(a=a, sigma=sigma), payer(), REFERENCE, steps_per_year=24
    )
    assert value.crossings == 0
    boundaries = [step.boundary for step in value.steps]
    assert all(bound is not None for bound in boundaries)
    for earlier, later in pairwise(boundaries):
        assert earlier is not None and later is not None
        assert later < earlier


def test_the_boundary_separates_exercising_from_waiting() -> None:
    """The reported rate is where the decision turns, checked against the decision.

    Below the boundary a payer's immediate value is under its continuation value
    and above it is over, which is recovered here by repricing the two
    sub-instruments the decision is between rather than by trusting the
    interpolation.
    """
    curve = flat()
    model = HullWhite(a=0.05, sigma=0.01)
    option = payer()
    value = bermudan_swaption(curve, model, option, REFERENCE, steps_per_year=24)
    first = value.steps[0]
    assert first.boundary is not None
    # at the boundary the swap is in the money, since waiting is worth something
    level = short_rate_level(curve, model, first.time)
    assert first.boundary > level
    # and every later date exercises at a lower rate, so more of its mass does
    assert value.steps[-1].exercised > first.exercised


def test_exercised_shares_are_probabilities_and_rise_towards_the_end() -> None:
    value = bermudan_swaption(
        sloped(), HullWhite(a=0.05, sigma=0.01), payer(), REFERENCE, steps_per_year=24
    )
    for step in value.steps:
        assert 0.0 <= step.exercised <= 1.0
    assert value.steps[-1].exercised > value.steps[0].exercised


# -- the instrument ------------------------------------------------------------


@pytest.mark.parametrize(
    "expiries,maturity,strike,match",
    [
        ((), MATURITY, 0.03, "at least one exercise date"),
        ((date(2030, 1, 15), date(2029, 1, 15)), MATURITY, 0.03, "ascending"),
        ((date(2030, 1, 15), date(2030, 1, 15)), MATURITY, 0.03, "ascending"),
        ((date(2036, 1, 15),), MATURITY, 0.03, "not before the swap's maturity"),
        ((date(2037, 1, 15),), MATURITY, 0.03, "not before the swap's maturity"),
        ((date(2030, 1, 15),), MATURITY, float("nan"), "strike must be finite"),
    ],
)
def test_the_instrument_refuses_what_it_cannot_describe(
    expiries: tuple[date, ...], maturity: date, strike: float, match: str
) -> None:
    with pytest.raises(BadBermudan, match=match):
        BermudanSwaption(expiries=expiries, maturity=maturity, strike=strike)


def test_an_exercise_date_off_a_boundary_is_refused_with_the_boundaries() -> None:
    """The message names what would have worked, because the caller cannot guess it."""
    option = BermudanSwaption(
        expiries=(date(2029, 1, 15), date(2029, 4, 15)),
        maturity=MATURITY,
        strike=0.03,
    )
    with pytest.raises(BadBermudan, match="2029-07-16"):
        bermudan_swaption(
            flat(), HullWhite(a=0.05, sigma=0.01), option, REFERENCE, steps_per_year=12
        )


def test_an_unadjusted_anniversary_resolves_to_its_business_day() -> None:
    """15 January 2033 is a Saturday, and a term sheet still says 15 January.

    The resolved date is reported, so the caller can see which day the right
    actually falls on rather than having to reproduce the calendar to find out.
    """
    weekend = date(2033, 1, 15)
    assert weekend.weekday() == 5
    option = payer(expiries=(weekend,))
    value = bermudan_swaption(
        flat(), HullWhite(a=0.05, sigma=0.01), option, REFERENCE, steps_per_year=12
    )
    assert value.steps[0].expiry == date(2033, 1, 17)


def test_either_form_of_a_later_boundary_is_the_same_instrument() -> None:
    """Resolution normalises the dates the schedule is read against.

    The *first* exercise date is the exception, and not by oversight: the fixed
    leg is generated from it, so giving the adjusted day there anchors the swap
    two days later and leaves a short front stub. That is a different swap, with a
    different first accrual, and it would be wrong to quietly treat the two as
    one. Every later date is matched against a schedule already anchored, so both
    forms of it name the same period and the same price.
    """
    curve = flat()
    model = HullWhite(a=0.05, sigma=0.01)
    first = date(2029, 1, 15)
    unadjusted = payer(expiries=(first, date(2033, 1, 15)))
    adjusted = payer(expiries=(first, date(2033, 1, 17)))
    one = bermudan_swaption(curve, model, unadjusted, REFERENCE, steps_per_year=12)
    other = bermudan_swaption(curve, model, adjusted, REFERENCE, steps_per_year=12)
    assert one.steps[1].expiry == other.steps[1].expiry == date(2033, 1, 17)
    assert one.price == pytest.approx(other.price, rel=1e-14)


def test_the_two_forms_of_one_boundary_are_not_two_rights() -> None:
    option = payer(expiries=(date(2033, 1, 15), date(2033, 1, 17)))
    with pytest.raises(BadBermudan, match="same fixed-leg period"):
        bermudan_swaption(
            flat(), HullWhite(a=0.05, sigma=0.01), option, REFERENCE, steps_per_year=12
        )


def test_an_expiry_on_or_before_the_valuation_date_is_a_decision() -> None:
    option = payer(expiries=(REFERENCE, date(2030, 1, 15)))
    with pytest.raises(BadBermudan, match="that is a decision, not a value"):
        bermudan_swaption(
            flat(), HullWhite(a=0.05, sigma=0.01), option, REFERENCE, steps_per_year=12
        )


def test_quarterly_exercise_on_a_quarterly_leg() -> None:
    """Denser rights on a denser leg, which is where the grid design earns itself."""
    expiries = tuple(
        date(year, month, 15) for year in (2029, 2030) for month in (1, 4, 7, 10)
    )
    option = BermudanSwaption(
        expiries=expiries,
        maturity=date(2033, 1, 15),
        strike=0.03,
        frequency=Frequency.QUARTERLY,
    )
    value = bermudan_swaption(
        sloped(), HullWhite(a=0.05, sigma=0.01), option, REFERENCE, steps_per_year=24
    )
    assert len(value.steps) == len(expiries)
    assert value.switch_value > 0.0
    assert value.tree.curve_error(sloped()) < 1e-14
    for time in (step.time for step in value.steps):
        assert time in value.tree.times


def test_a_name_is_readable_and_a_label_wins() -> None:
    assert "payer" in payer().name
    assert BermudanSwaption(
        expiries=ANNUAL_DATES, maturity=MATURITY, strike=0.03, label="7y into 10y"
    ).name == "7y into 10y"


def test_the_tree_reports_its_own_size() -> None:
    tree: TrinomialTree = build_tree(flat(), HullWhite(a=0.05, sigma=0.01), (5.0,), 12)
    assert tree.steps == len(tree.times) - 1
    assert tree.nodes == sum(
        hi - lo + 1 for lo, hi in zip(tree.lower, tree.upper, strict=True)
    )
    assert tree.state(0) == 0.0
    assert tree.state(3) == pytest.approx(3.0 * tree.spacing)
