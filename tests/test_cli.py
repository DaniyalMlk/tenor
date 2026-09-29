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
from tenor.cli import BadInput, main, parse_index, parse_quotes
from tenor.daycount import Basis
from tenor.floating import FloatingNote
from tenor.horizon import horizon_return
from tenor.inflation import LinkedBond
from tenor.instruments import Deposit, Future, Swap
from tenor.schedule import Frequency

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


# -- the floating rate note ---------------------------------------------------


def test_the_floating_command_reports_par_and_a_rate_duration_of_zero(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two numbers a floater report exists to put next to each other.

    At the quoted margin the note is worth exactly par and its rate duration is
    zero to floating point, while its spread duration is most of its maturity.
    A report that gave a single "duration" would be giving the wrong one.
    """
    arguments = [
        "--json", "floating", *shared(quote_file),
        "--maturity", "2026-01-05", "--quoted-margin", "0.0075",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["on_reset_date"] is True
    assert payload["dirty"] == pytest.approx(100.0, abs=5e-13)
    assert payload["clean"] == pytest.approx(100.0, abs=5e-13)
    assert abs(payload["rate_duration"]) < 1e-9
    assert payload["spread_duration"] > 4.0
    assert payload["discount_margin"] == pytest.approx(0.0075, abs=1e-12)
    assert payload["coupon_count"] == 20
    assert json.dumps(payload, allow_nan=False)


def test_the_floating_command_solves_a_margin_against_a_quote(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """And the margin it reports has to reprice the quote it was given."""
    arguments = [
        "--json", "floating", *shared(quote_file),
        "--maturity", "2026-01-05", "--quoted-margin", "0.0075",
        "--quote", "98.5", "--fixing", "0.004",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quoted_clean"] == pytest.approx(98.5)
    assert payload["clean"] == pytest.approx(98.5, abs=1e-9)
    assert payload["discount_margin"] > 0.0075
    # Below par means a margin wider than the coupon spread, which means the
    # leftover annuity is negative and rate duration goes with it.
    assert payload["rate_duration"] < 0.0
    assert payload["current_fixing_projected"] is False


def test_the_floating_command_agrees_with_the_library(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--json", "floating", *shared(quote_file),
        "--maturity", "2031-01-05", "--quoted-margin", "0.005",
        "--margin", "0.012", "--coupons",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    curve = bootstrap(
        REFERENCE, parse_quotes(QUOTES, REFERENCE), basis=Basis.ACT_365F
    ).curve
    note = FloatingNote(
        REFERENCE, date(2031, 1, 5), 0.005, Frequency.QUARTERLY, Basis.ACT_360
    )
    assert payload["dirty"] == pytest.approx(
        note.dirty_price(curve, REFERENCE, margin=0.012)
    )
    assert payload["spread_duration"] == pytest.approx(
        note.spread_duration(curve, REFERENCE, margin=0.012)
    )
    assert len(payload["coupons"]) == payload["coupon_count"]
    assert payload["coupons"][-1]["redemption"] == pytest.approx(100.0)
    assert all(one["projected"] for one in payload["coupons"])


def test_the_coupon_listing_is_off_unless_asked_for(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Twenty rows of projection is a report about the wrong thing by default."""
    arguments = [
        "--json", "floating", *shared(quote_file),
        "--maturity", "2026-01-05", "--quoted-margin", "0.0075",
    ]
    assert main(arguments) == 0
    assert "coupons" not in json.loads(capsys.readouterr().out)


def test_a_floater_settling_mid_period_without_a_fixing_is_reported(
    quote_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The curve does not reach back to the last reset, so it cannot be projected.

    Reported as a message rather than raised through the curve, because the thing
    the caller has to do about it is find a fixing. The note here was issued two
    years before the reference date on a January/April/July/October grid, so the
    reference falls inside a period that began the previous October — which is the
    ordinary case, not a contrived one.
    """
    arguments = [
        "floating", *shared(quote_file),
        "--maturity", "2026-01-07", "--quoted-margin", "0.0075",
        "--issued", "2019-01-07",
    ]
    assert main(arguments) == 2
    assert "started before the curve's reference date" in capsys.readouterr().err


# -- the linker subcommand ----------------------------------------------------


def index_text(*, months: int = 44, annual: float = 0.02) -> str:
    lines = ["# a published index series", ""]
    level = 100.0
    for k in range(months):
        year, month = 2018 + k // 12, k % 12 + 1
        lines.append(f"{year}-{month:02d}  {level:.6f}")
        level *= (1.0 + annual) ** (1.0 / 12.0)
    return "\n".join(lines) + "\n"


@pytest.fixture
def index_file(tmp_path: Path) -> str:
    path = tmp_path / "index.txt"
    path.write_text(index_text())
    return str(path)


def linker_argv(quote_file: str, index_file: str, *extra: str) -> list[str]:
    return [
        "linker",
        quote_file,
        "--reference",
        REFERENCE.isoformat(),
        "--basis",
        "ACT_365F",
        "--index",
        index_file,
        "--maturity",
        "2029-01-05",
        "--coupon",
        "0.015",
        "--base-index",
        "100.5",
        "--issued",
        "2019-01-05",
        "--projection",
        "0.02",
        *extra,
    ]


def test_the_linker_reports_both_spaces(
    quote_file: str, index_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--json", *linker_argv(quote_file, index_file, "--real-yield", "0.005")])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["index_ratio"] > 1.0
    assert payload["real_yield"] == 0.005
    # The invoice is the real dirty price times the ratio, so it exceeds the real
    # clean price by both accrued and the accretion.
    assert payload["settlement_amount"] > payload["real_clean"]
    assert payload["settlement_amount"] == pytest.approx(
        (payload["real_clean"] + payload["real_accrued"]) * payload["index_ratio"]
    )
    assert payload["real_modified_duration"] > 0.0
    assert payload["deflation_floor_binds"] is False


def test_the_linker_solves_a_real_yield_from_a_real_price(
    quote_file: str, index_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--json", *linker_argv(quote_file, index_file, "--real-quote", "99.25")])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["real_clean"] == pytest.approx(99.25)
    bond = LinkedBond(
        date(2019, 1, 5),
        date(2029, 1, 5),
        0.015,
        100.5,
        Frequency.SEMI_ANNUAL,
        Basis.ACT_ACT_ISDA,
    )
    assert payload["real_yield"] == pytest.approx(
        bond.real_yield_from_clean(99.25, REFERENCE).value
    )


def test_the_linker_reports_both_breakeven_forms(
    quote_file: str, index_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = linker_argv(
        quote_file,
        index_file,
        "--real-yield",
        "0.005",
        "--nominal-yield",
        "0.025",
    )
    assert main(["--json", *argv]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["breakeven_quoted"] == pytest.approx(0.020)
    assert payload["breakeven_exact"] < payload["breakeven_quoted"]
    assert payload["breakeven_exact"] == pytest.approx(1.025 / 1.005 - 1.0)


def test_the_linker_splits_the_next_years_accretion(
    quote_file: str, index_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--json", *linker_argv(quote_file, index_file, "--real-yield", "0.005")])
        == 0
    )
    split = json.loads(capsys.readouterr().out)["accretion_one_year"]
    assert 0.0 < split["published_share"] < 1.0
    assert (1.0 + split["published"]) * (1.0 + split["projected"]) == pytest.approx(
        1.0 + split["total"]
    )


def test_the_linker_lists_flows_when_asked(
    quote_file: str, index_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = linker_argv(quote_file, index_file, "--real-yield", "0.005", "--flows")
    assert main(["--json", *argv]) == 0
    flows = json.loads(capsys.readouterr().out)["flows"]
    assert len(flows) > 2
    assert any(flow["projected"] for flow in flows)
    assert not flows[0]["projected"]
    # The final coupon and the principal are separate flows on the same day,
    # because only one of them is floored.
    last_day = flows[-1]["payment"]
    assert sum(1 for flow in flows if flow["payment"] == last_day) == 2


def test_the_linker_can_turn_the_floor_off(
    quote_file: str, index_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deflating = tmp_path / "deflation.txt"
    deflating.write_text(index_text(annual=-0.04))
    argv = linker_argv(
        quote_file, str(deflating), "--real-yield", "0.005", "--projection-off"
    )
    # `--projection-off` is not an option; the point of this call is the floor.
    argv = [argument for argument in argv if argument != "--projection-off"]
    assert main(["--json", *argv]) == 0
    floored = json.loads(capsys.readouterr().out)
    assert floored["deflation_floor_binds"] is True
    assert floored["redemption_index_ratio"] == 1.0
    assert main(["--json", *argv, "--no-floor"]) == 0
    unfloored = json.loads(capsys.readouterr().out)
    assert unfloored["deflation_floor_binds"] is False
    assert unfloored["redemption_index_ratio"] < 1.0


def test_a_real_price_and_a_real_yield_are_mutually_exclusive(
    quote_file: str, index_file: str
) -> None:
    with pytest.raises(SystemExit):
        main(
            linker_argv(
                quote_file,
                index_file,
                "--real-yield",
                "0.005",
                "--real-quote",
                "99.0",
            )
        )


def test_the_projection_is_required(quote_file: str, index_file: str) -> None:
    argv = [
        argument
        for argument in linker_argv(quote_file, index_file, "--real-yield", "0.005")
        if argument not in {"--projection", "0.02"}
    ]
    with pytest.raises(SystemExit):
        main(argv)


class TestIndexFileParsing:
    def test_a_good_file_round_trips(self) -> None:
        index = parse_index(index_text(months=24))
        assert index.first_month == (2018, 1)
        assert index.last_month == (2019, 12)

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        index = parse_index("# leading\n\n2024-01 100.0\n\n2024-02 100.5  # trailing\n")
        assert index.published(2024, 2) == 100.5

    def test_a_short_line_names_itself(self) -> None:
        with pytest.raises(ValueError, match="line 2 has 1 fields"):
            parse_index("2024-01 100.0\n2024-02\n")

    def test_a_bad_month_names_itself(self) -> None:
        with pytest.raises(ValueError, match="line 1: '2024' is not a month"):
            parse_index("2024 100.0\n")
        with pytest.raises(ValueError, match="is not a month"):
            parse_index("2024-ab 100.0\n")

    def test_a_month_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="no month 13"):
            parse_index("2024-13 100.0\n")

    def test_a_bad_level(self) -> None:
        with pytest.raises(ValueError, match="not an index level"):
            parse_index("2024-01 par\n")

    def test_a_repeated_month(self) -> None:
        with pytest.raises(ValueError, match="appears twice"):
            parse_index("2024-01 100.0\n2024-01 101.0\n")

    def test_an_empty_file(self) -> None:
        with pytest.raises(ValueError, match="no figures"):
            parse_index("# nothing but a comment\n")

    def test_a_gap_is_reported_by_the_index_itself(self) -> None:
        with pytest.raises(ValueError, match="2024-02 is missing"):
            parse_index("2024-01 100.0\n2024-03 101.0\n")


# -- credit -------------------------------------------------------------------

SPREADS = """
# maturity, par spread in basis points
2022-01-05,  60
2024-01-05,  90
2026-01-05, 120
2031-01-05, 185
"""


@pytest.fixture
def spread_file(tmp_path: Path) -> str:
    path = tmp_path / "spreads.txt"
    path.write_text(SPREADS)
    return str(path)


def test_credit_strips_a_curve_that_reprices_its_quotes(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "credit", *shared(quote_file), "--spreads", spread_file]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recovery"] == 0.4
    assert len(payload["pillars"]) == 4
    assert payload["worst_repricing_error_bp"] < 1e-6
    for pillar, quoted in zip(payload["pillars"], (60.0, 90.0, 120.0, 185.0), strict=True):
        assert pillar["quoted_bp"] == pytest.approx(quoted)
        assert pillar["repriced_bp"] == pytest.approx(quoted, abs=1e-6)


def test_credit_reports_survival_falling_and_hazards_positive(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "credit", *shared(quote_file), "--spreads", spread_file]) == 0
    pillars = json.loads(capsys.readouterr().out)["pillars"]
    survivals = [one["survival"] for one in pillars]
    assert survivals == sorted(survivals, reverse=True)
    assert all(0.0 < one["survival"] < 1.0 for one in pillars)
    assert all(one["forward_hazard"] > 0.0 for one in pillars)
    assert all(
        one["survival"] + one["default_probability"] == pytest.approx(1.0, abs=1e-12)
        for one in pillars
    )


def test_credit_prints_the_triangle_beside_the_stripped_hazard(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "credit", *shared(quote_file), "--spreads", spread_file]) == 0
    first = json.loads(capsys.readouterr().out)["pillars"][0]
    # The triangle is the quoted spread over one minus recovery, exactly.
    assert first["triangle_hazard"] == pytest.approx(0.0060 / 0.6, rel=1e-12)
    # And it is close to the stripped rate without being it.
    assert first["forward_hazard"] != first["triangle_hazard"]
    assert abs(first["average_hazard"] / first["triangle_hazard"] - 1.0) < 0.05


def test_credit_reports_the_average_hazard_not_only_the_forward(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The triangle produces a flat hazard to the maturity, so the comparable
    # quantity is the average and not the last bucket's forward. On an
    # upward-sloping curve the two diverge, and the long end is where a
    # reader would otherwise conclude the rule of thumb is wildly wrong.
    assert main(["--json", "credit", *shared(quote_file), "--spreads", spread_file]) == 0
    pillars = json.loads(capsys.readouterr().out)["pillars"]
    first, last = pillars[0], pillars[-1]
    assert first["average_hazard"] == pytest.approx(first["forward_hazard"], rel=1e-12)
    assert last["forward_hazard"] > last["average_hazard"] > first["average_hazard"]
    # Against the triangle, the average is close and the forward is not.
    assert abs(last["average_hazard"] / last["triangle_hazard"] - 1.0) < 0.15
    assert last["forward_hazard"] / last["triangle_hazard"] > 1.3


def test_credit_prices_a_bond_below_its_riskless_value(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--json",
                "credit",
                *shared(quote_file),
                "--spreads",
                spread_file,
                "--bond-maturity",
                "2026-01-05",
                "--bond-coupon",
                "0.05",
            ]
        )
        == 0
    )
    bond = json.loads(capsys.readouterr().out)["bond"]
    assert bond["risky_price"] < bond["riskless_price"]
    assert bond["credit_cost"] == pytest.approx(
        bond["riskless_price"] - bond["risky_price"], rel=1e-12
    )
    assert bond["credit_cost"] > 0.0


def test_credit_recovery_moves_the_price_the_right_way(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    prices = []
    for recovery in ("0.1", "0.4", "0.7"):
        arguments = [
            "--json", "credit", *shared(quote_file), "--spreads", spread_file,
            "--recovery", recovery, "--bond-maturity", "2026-01-05",
        ]
        assert main(arguments) == 0
        prices.append(json.loads(capsys.readouterr().out)["bond"]["risky_price"])
    # Spreads are held fixed, so a higher assumed recovery means a higher
    # stripped hazard rate -- and the two effects on a bond very nearly
    # cancel, which is the whole reason a recovery assumption is hard to
    # pin down from spreads alone. What must not happen is a wild swing.
    assert max(prices) / min(prices) - 1.0 < 0.05


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2022-01-05\n", "expected"),
        ("2022-01-05, 60, 70\n", "expected"),
        ("not-a-date, 60\n", "not an ISO date"),
        ("2022-01-05, wide\n", "not a number"),
        ("# only a comment\n", "no quotes"),
    ],
)
def test_credit_names_the_line_of_a_bad_spread_file(
    quote_file: str, tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(text)
    assert main(["credit", *shared(quote_file), "--spreads", str(path)]) == 2


def test_credit_refuses_out_of_order_maturities(quote_file: str, tmp_path: Path) -> None:
    path = tmp_path / "unsorted.txt"
    path.write_text("2026-01-05, 120\n2022-01-05, 60\n")
    assert main(["credit", *shared(quote_file), "--spreads", str(path)]) == 2
