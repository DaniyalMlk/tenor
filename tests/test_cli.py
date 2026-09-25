"""Tests for the command line.

Mostly its refusals. A command line that silently picked a day count would
undo six phases of treating every convention as an argument with a name, so
what is tested hardest is that it does not: a missing basis, an unknown
convention, an instrument kind it does not recognise, a quote file line with a
field missing.

The numbers it prints are checked against the library rather than pinned as
literals, since they are the library's answers and the library's own tests
already pin those. What is worth pinning here is that the command line reaches
them at all, and reports the identities alongside them rather than only the
numbers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tenor.bond import Bond
from tenor.bootstrap import bootstrap
from tenor.cli import BadInput, main, parse_quotes
from tenor.daycount import Basis
from tenor.horizon import horizon_return
from tenor.instruments import Deposit, Future, Swap

REFERENCE = date(2021, 1, 5)
QUOTES = """
# a comment, and a blank line follow

deposit  2021-07-05  0.0020  ACT_360
swap     2023-01-05  0.0060  SEMI_ANNUAL  THIRTY_360_BOND
swap     2031-01-05  0.0220  SEMI_ANNUAL  THIRTY_360_BOND
"""


@pytest.fixture
def quote_file(tmp_path: Path) -> str:
    path = tmp_path / "quotes.txt"
    path.write_text(QUOTES)
    return str(path)


def shared(quote_file: str) -> list[str]:
    return [quote_file, "--reference", "2021-01-05", "--basis", "ACT_365F"]


# -- parsing ------------------------------------------------------------------


def test_the_quote_file_parses_all_three_instruments() -> None:
    text = """
    deposit 2021-07-05 0.002 ACT_360
    future  2021-09-15 2021-12-15 99.55 ACT_360 0.0001
    swap    2031-01-05 0.022 SEMI_ANNUAL THIRTY_360_BOND
    """
    quotes = parse_quotes(text, REFERENCE)
    assert isinstance(quotes[0], Deposit)
    assert isinstance(quotes[1], Future)
    assert isinstance(quotes[2], Swap)
    assert quotes[0].basis is Basis.ACT_360
    assert quotes[1].convexity == pytest.approx(0.0001)
    assert quotes[2].rate == pytest.approx(0.022)


def test_a_convention_may_be_named_or_spelled_as_the_market_does() -> None:
    """Both resolve to the same thing, and neither is a default."""
    by_name = parse_quotes("deposit 2021-07-05 0.002 ACT_360", REFERENCE)
    by_value = parse_quotes("deposit 2021-07-05 0.002 ACT/360", REFERENCE)
    assert by_name[0].basis is by_value[0].basis is Basis.ACT_360


def test_comments_and_blank_lines_are_ignored() -> None:
    assert len(parse_quotes(QUOTES, REFERENCE)) == 3


def test_an_unusable_quote_file_says_which_line() -> None:
    with pytest.raises(BadInput, match="line 1"):
        parse_quotes("wombat 2021-07-05 0.002 ACT_360", REFERENCE)
    with pytest.raises(BadInput, match="line 2"):
        parse_quotes("deposit 2021-07-05 0.002 ACT_360\ndeposit 2022-01-05 0.003", REFERENCE)
    with pytest.raises(BadInput, match="not a day count"):
        parse_quotes("deposit 2021-07-05 0.002 WEEKS", REFERENCE)
    with pytest.raises(BadInput, match="no instruments in it"):
        parse_quotes("# nothing but a comment", REFERENCE)


# -- the subcommands ----------------------------------------------------------


def test_the_curve_command_reports_the_repricing_identity(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "curve", *shared(quote_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reprices"] is True
    assert payload["worst_repricing_error"] < 1e-12
    assert payload["sweeps"] == 1
    assert len(payload["pillars"]) == 3
    assert payload["basis"] == "ACT/365F"


def test_a_smooth_interpolation_takes_more_sweeps(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--json", "curve", *shared(quote_file), "--interpolation", "MONOTONE_CONVEX"]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sweeps"] > 1
    assert payload["reprices"] is True


def test_the_price_command_agrees_with_the_library(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "price", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    curve = bootstrap(
        REFERENCE, parse_quotes(QUOTES, REFERENCE), basis=Basis.ACT_365F
    ).curve
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.05)
    assert payload["dirty"] == pytest.approx(bond.price_from_curve(curve, REFERENCE))
    assert payload["clean"] == pytest.approx(payload["dirty"] - payload["accrued"])
    assert payload["modified_duration"] < payload["macaulay_duration"]
    assert payload["convexity"] > 0.0
    assert "z_spread" not in payload


def test_a_quoted_price_adds_both_spreads(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "price", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05", "--quote", "120.0",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quoted_clean"] == 120.0
    assert payload["z_spread"] > 0.0  # quoted below its curve price
    assert payload["i_spread"] != pytest.approx(payload["z_spread"], rel=1e-3)


def test_the_risk_command_reports_the_sum_identity(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "risk", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05",
        "--buckets", "0.5", "2", "10",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["key_rate_sum"] == pytest.approx(payload["total_duration"], rel=1e-5)
    assert len(payload["key_rates"]) == 3
    assert len(payload["instruments"]) == 3


def test_the_option_command_reports_the_option_cost(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "option", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05",
        "--volatility", "0.15", "--first-call", "2026-01-05",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lattice_reprices_curve"] is True
    assert payload["with_option"] < payload["bullet"]
    assert payload["option_value"] > 0.0
    assert payload["option_cost"] > 0.0
    assert payload["oas"] == pytest.approx(0.0, abs=1e-9)
    assert payload["option_cost"] == pytest.approx(payload["z_spread"], abs=1e-12)


def test_a_put_goes_the_other_way(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "option", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05",
        "--volatility", "0.15", "--first-call", "2026-01-05", "--put",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["with_option"] > payload["bullet"]
    assert payload["option_cost"] < 0.0


# -- refusals -----------------------------------------------------------------


def test_the_basis_is_required(quote_file: str) -> None:
    """The whole point of the library, enforced at its front door."""
    with pytest.raises(SystemExit):
        main(["curve", quote_file, "--reference", "2021-01-05"])


def test_an_unknown_convention_is_refused(quote_file: str) -> None:
    with pytest.raises(SystemExit):
        main(["curve", quote_file, "--reference", "2021-01-05", "--basis", "WEEKS"])


def test_a_missing_file_is_reported_rather_than_raised(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["curve", "nowhere.txt", "--reference", "2021-01-05", "--basis", "ACT_365F"]) == 2
    assert "FileNotFoundError" in capsys.readouterr().err


def test_a_maturity_between_lattice_nodes_is_reported(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "option", *shared(quote_file),
        "--maturity", "2030-04-17", "--coupon", "0.05",
        "--volatility", "0.15", "--first-call", "2026-01-05",
    ]
    assert main(arguments) == 2
    assert "whole number" in capsys.readouterr().err


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_the_plain_output_is_readable(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["curve", *shared(quote_file)]) == 0
    out = capsys.readouterr().out
    assert "reprices: True" in out
    assert "pillars:" in out
    assert "date=2021-07-05" in out


def test_the_horizon_command_reports_the_carry_identity(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "horizon", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.03",
        "--horizon", "2022-01-05",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arbitrage_free"] is True
    assert payload["carry"] == pytest.approx(payload["financing_cost"], abs=1e-11)
    assert payload["total_return"] == pytest.approx(
        payload["carry"] + payload["roll_down"], abs=1e-11
    )
    assert payload["excess_over_financing"] == pytest.approx(payload["roll_down"], abs=1e-11)


def test_the_horizon_command_shows_where_the_return_comes_from(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """On this screen the curve runs from 20bp to 220bp, so a ten-year bond
    held for a year earns far more from rolling down it than from holding it."""
    arguments = [
        "--json", "horizon", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.03",
        "--horizon", "2022-01-05",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_return_bps"] > 200.0
    assert payload["roll_down"] > 4.0 * payload["financing_cost"]
    assert len(payload["coupons"]) == 2


def test_the_horizon_command_agrees_with_the_library(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "horizon", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.05",
        "--horizon", "2023-01-05",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    curve = bootstrap(
        REFERENCE, parse_quotes(QUOTES, REFERENCE), basis=Basis.ACT_365F
    ).curve
    bond = Bond(REFERENCE, date(2031, 1, 5), 0.05)
    expected = horizon_return(bond, curve, REFERENCE, date(2023, 1, 5))
    assert payload["roll_down"] == pytest.approx(expected.roll_down)
    assert payload["start_price"] == pytest.approx(expected.start_price)


def test_a_horizon_past_maturity_is_reported_rather_than_raised(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "horizon", *shared(quote_file),
        "--maturity", "2031-01-05", "--coupon", "0.03",
        "--horizon", "2032-01-05",
    ]
    assert main(arguments) == 2
    assert "no price at the" in capsys.readouterr().err
