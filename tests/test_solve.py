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
