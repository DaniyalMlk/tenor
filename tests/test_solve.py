"""Tests for the root finder.

Checked against functions whose roots are known in closed form, and against the
shapes that break naive solvers: a flat multiple root, a near-discontinuity, a
high-order polynomial, and a bracket that has to be found before it can be
used. The first implementation of Brent's method here reported convergence at
points that were not roots and did so on every one of these, which is why the
tests assert the error against the known root rather than that the solver
returned.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from tenor.solve import NoRoot, Root, bracket_root, brent, solve


def test_it_solves_the_classic_cases_to_machine_precision() -> None:
    """Known roots, so the error is measurable rather than merely small."""
    cases = [
        (lambda x: x**3 - 2.0 * x - 5.0, 2.0945514815423265, 1.0, 3.0),
        (lambda x: math.cos(x) - x, 0.7390851332151607, 0.0, 2.0),
        (lambda x: math.exp(x) - 4.0, math.log(4.0), -1.0, 5.0),
        (lambda x: x**10 - 1e-3, 1e-3 ** (1.0 / 10.0), 0.0, 2.0),
    ]
    for function, root, low, high in cases:
        found = brent(function, low, high)
        assert found.converged
        assert found.value == pytest.approx(root, abs=1e-15)
        assert abs(found.residual) < 1e-14


def test_it_converges_fast_on_a_well_behaved_function() -> None:
    """Superlinear, which is the entire reason for the interpolation steps.

    Bisection would need about fifty iterations to reach machine precision on
    these brackets. Asserting a low count is what distinguishes this from a
    bisection implementation that happens to also return the right answer.
    """
    found = brent(lambda x: math.cos(x) - x, 0.0, 2.0)
    assert found.iterations <= 12
    found = brent(lambda x: x**3 - 2.0 * x - 5.0, 1.0, 3.0)
    assert found.iterations <= 15


def test_a_steep_function_does_not_throw_it_out_of_the_bracket() -> None:
    """Near-vertical at the root, where a secant step overshoots wildly."""
    found = brent(lambda x: math.atan(1000.0 * (x - 0.7)), 0.0, 1.0)
    assert found.value == pytest.approx(0.7, abs=1e-12)
    assert 0.0 <= found.value <= 1.0


def test_a_flat_multiple_root_still_converges() -> None:
    """A triple root, where the function is flat and interpolation is useless.

    The interval conditions force bisection here, so it is slow - and it does
    arrive, which is the point. A solver that gave up interpolation control
    would sit near the root reporting progress forever.
    """
    found = brent(lambda x: (x - 0.3) ** 3, -1.0, 2.0)
    assert found.converged
    assert found.value == pytest.approx(0.3, abs=1e-12)


def test_an_exact_root_at_a_bracket_end_is_returned_immediately() -> None:
    found = brent(lambda x: x - 1.0, 1.0, 2.0)
    assert found.value == 1.0
    assert found.iterations == 0
    assert found.residual == 0.0


def test_a_bracket_without_a_sign_change_is_refused() -> None:
    with pytest.raises(NoRoot, match="does not change sign"):
        brent(lambda x: x * x + 1.0, -1.0, 1.0)


def test_running_out_of_iterations_is_reported_not_hidden() -> None:
    """The reason the solver returns a record rather than a float."""
    found = brent(lambda x: math.cos(x) - x, 0.0, 2.0, max_iterations=2)
    assert not found.converged
    assert found.iterations == 2
    assert abs(found.residual) > 0.0
    assert "did not converge" in str(found)


def test_the_bracket_widens_until_it_finds_a_sign_change() -> None:
    low, high = bracket_root(lambda x: x * x - 2.0, 0.0, 0.1)
    assert low <= math.sqrt(2.0) <= high
    found = solve(lambda x: x * x - 2.0, low=0.0, high=0.1)
    assert found.value == pytest.approx(math.sqrt(2.0), abs=1e-14)


def test_a_function_that_never_crosses_says_so() -> None:
    with pytest.raises(NoRoot, match="no sign change found"):
        solve(math.exp, low=0.0, high=1.0)


def test_a_root_sitting_exactly_on_a_bracket_end_short_circuits() -> None:
    low, high = bracket_root(lambda x: x, 0.0, 1.0)
    assert low == high == 0.0
    assert solve(lambda x: x, low=0.0, high=1.0).value == 0.0


def test_a_backwards_bracket_is_refused() -> None:
    with pytest.raises(NoRoot, match="runs from low to high"):
        bracket_root(lambda x: x, 2.0, 1.0)


def test_the_record_reports_where_it_started() -> None:
    found = brent(lambda x: x - 0.5, 0.0, 1.0)
    assert found.bracket == (0.0, 1.0)
    assert isinstance(found, Root)
    assert "converged" in str(found)


class TestTheWideningBudget:
    """``attempts`` and ``growth``: the two numbers the widening is made of.

    Neither had been chosen from outside the module, so neither the docstring's
    claim about reach nor the behaviour at the edge of the budget had ever been
    looked at. Both turned out to be wrong.
    """

    @staticmethod
    def far(target: float) -> Callable[[float], float]:
        return lambda x: x - target

    def test_the_span_is_multiplied_by_one_plus_growth_each_attempt(self) -> None:
        # The step the old docstring got wrong: one end moves by growth times
        # the span, so the span is multiplied by 1 + growth, not by growth.
        assert pytest.approx(7.914e24, rel=1e-3) == (1.0 + 1.6) ** 60
        assert math.log10((1.0 + 1.6) ** 60) == pytest.approx(24.9, abs=0.1)

    def test_the_default_budget_reaches_what_the_docstring_now_claims(self) -> None:
        # Measured against roots on the powers of ten rather than argued from
        # the arithmetic above, so the two are independent readings.
        low, high = bracket_root(self.far(1.0e24), 0.0, 1.0)
        assert low < 1.0e24 < high
        with pytest.raises(NoRoot, match="no sign change found after 60 widenings"):
            bracket_root(self.far(1.0e25), 0.0, 1.0)

    def test_a_small_budget_runs_out_and_a_sufficient_one_does_not(self) -> None:
        # The option has to change the outcome, not merely be accepted.
        with pytest.raises(NoRoot, match="no sign change found after 4 widenings"):
            bracket_root(self.far(1.0e6), 0.0, 1.0, attempts=4)
        low, high = bracket_root(self.far(1.0e6), 0.0, 1.0, attempts=15)
        assert low < 1.0e6 < high
        # Fifteen is the smallest budget that gets there at this growth, so the
        # failure above is the budget and not the function.
        with pytest.raises(NoRoot):
            bracket_root(self.far(1.0e6), 0.0, 1.0, attempts=14)

    def test_the_budget_walks_downward_too(self) -> None:
        # Every other case here has its root above the bracket, so only the
        # upper end ever moves and the lower branch of the widening is never
        # taken. A break sweep found that: changing the low-side step left the
        # whole class green. The two branches are mirror images, so the budgets
        # have to match.
        down = bracket_root(lambda x: x + 1.0e6, 0.0, 1.0, attempts=15)
        assert down[0] < -1.0e6 < down[1]
        with pytest.raises(NoRoot, match="after 14 widenings"):
            bracket_root(lambda x: x + 1.0e6, 0.0, 1.0, attempts=14)
        # And the end that moved is the low one, with the span multiplied by
        # one plus the growth per attempt, exactly as upward.
        assert down[1] == pytest.approx(1.0, abs=1e-12)
        assert down[1] - down[0] == pytest.approx((1.0 + 1.6) ** 15, rel=1e-9)

    def test_the_two_options_trade_against_each_other(self) -> None:
        # Which is the reason both are options: a slower growth needs more
        # attempts for the same reach, and the exchange rate is measurable.
        smallest = {}
        for growth in (1.05, 1.6, 3.0):
            for attempts in range(0, 60):
                try:
                    bracket_root(self.far(1.0e6), 0.0, 1.0, growth=growth, attempts=attempts)
                except NoRoot:
                    continue
                smallest[growth] = attempts
                break
        assert smallest == {1.05: 20, 1.6: 15, 3.0: 10}

    def test_a_bracket_that_already_changes_sign_needs_no_widenings(self) -> None:
        # The sign-change test used to live inside the widening loop, so a
        # budget of zero refused an interval that already contained the root --
        # and said `no sign change found` while printing values of -0.5 and
        # +0.5, which plainly change sign. A budget is a number of widenings;
        # checking whether any are needed is not one of them.
        assert bracket_root(self.far(0.5), 0.0, 1.0, attempts=0) == (0.0, 1.0)
        # And zero still refuses a bracket that genuinely needs widening.
        with pytest.raises(NoRoot, match="after 0 widenings"):
            bracket_root(self.far(1.0e6), 0.0, 1.0, attempts=0)

    def test_hoisting_the_check_did_not_cost_an_attempt(self) -> None:
        # The loop now tests after widening rather than before, so the budget
        # still buys exactly `attempts` widenings and no more. If the check had
        # simply been hoisted, every budget would have been one short.
        # A target of 2.0 lies strictly inside the once-widened [0, 2.6], so
        # one attempt finds it. (2.6 itself would land exactly on the new end,
        # where the product of the two values is zero rather than negative --
        # the one place this search does not call a touch a crossing.)
        assert bracket_root(self.far(2.0), 0.0, 1.0, attempts=1) == (0.0, 2.6)
        with pytest.raises(NoRoot, match="after 1 widenings"):
            bracket_root(self.far(100.0), 0.0, 1.0, attempts=1)
