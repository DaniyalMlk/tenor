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
from tenor.cli import (
    BadInput,
    main,
    parse_forecast_quotes,
    parse_index,
    parse_quotes,
)
from tenor.daycount import Basis
from tenor.floating import FloatingNote
from tenor.horizon import horizon_return
from tenor.inflation import LinkedBond
from tenor.instruments import Deposit, Future, Swap
from tenor.multicurve import DualSwap, ForecastIndex, IndexForward
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


# -- futures -----------------------------------------------------------------

BASKET = """
# maturity, coupon (decimal), clean price, label
2046-08-15,  0.0175,   63.782,  1.750% of 2046
2046-11-15,  0.0300,   79.795,  3.000% of 2046
2047-02-15,  0.0450,   99.337,  4.500% of 2047
2047-05-15,  0.0625,  122.497,  6.250% of 2047
"""


@pytest.fixture
def basket_file(tmp_path: Path) -> str:
    path = tmp_path / "basket.txt"
    path.write_text(BASKET)
    return str(path)


def futures_arguments(basket: str) -> list[str]:
    return [
        "futures",
        basket,
        "--settlement",
        "2026-11-20",
        "--first-delivery",
        "2026-12-01",
        "--delivery",
        "2026-12-31",
        "--price",
        "118.75",
        "--repo",
        "0.042",
    ]


def test_futures_needs_no_curve(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one command with no quote file, because it reads no curve.

    Worth a test of its own: every other subcommand requires ``--reference`` and
    ``--basis``, and the temptation when adding this one was to take them for
    consistency and ignore them.
    """
    assert main(["--json", *futures_arguments(basket_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settlement"] == "2026-11-20"
    assert payload["notional_coupon"] == 0.06
    assert len(payload["deliverables"]) == 4


def test_futures_reports_every_identity_beside_the_numbers(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *futures_arguments(basket_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    for entry in payload["deliverables"]:
        assert entry["net_basis"] == pytest.approx(
            entry["gross_basis"] - entry["carry"], rel=1e-10, abs=1e-12
        )
        assert entry["net_basis"] == pytest.approx(
            (0.042 - entry["implied_repo"]) * entry["financed_balance"], rel=1e-9
        )
        assert entry["net_basis"] == pytest.approx(
            (entry["breakeven_futures_price"] - 118.75) * entry["conversion_factor"],
            rel=1e-9,
        )


def test_futures_names_the_cheapest_bond_and_the_price_the_basket_justifies(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *futures_arguments(basket_file)]) == 0
    verdict = json.loads(capsys.readouterr().out)["cheapest_to_deliver"]
    assert verdict["unanimous"] is True
    assert verdict["by_net_basis"] == "6.250% of 2047"
    assert verdict["by_implied_repo"] == verdict["by_futures_price"] == "6.250% of 2047"
    # The quote is below the price the basket justifies, which is the delivery
    # option being paid for rather than an arbitrage.
    assert verdict["quote_less_implied"] < 0.0
    assert verdict["implied_futures_price"] == pytest.approx(118.93, abs=0.01)


def test_futures_walks_the_switch_when_asked(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--json",
                *futures_arguments(basket_file),
                "--switch",
                "0.03",
                "0.045",
                "0.075",
                "0.09",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    levels = payload["switch"]["levels"]
    assert [one["cheapest"] for one in levels] == [
        "6.250% of 2047",
        "6.250% of 2047",
        "1.750% of 2046",
        "1.750% of 2046",
    ]
    # And the output says the switch was priced off a flat yield, not off the
    # quotes in the basket file, because those two sets of prices are both in
    # the same payload.
    assert "flat yield" in payload["switch"]["priced_at"]
    assert all(one["margin"] > 0.0 for one in levels)


def test_futures_omits_the_switch_by_default(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *futures_arguments(basket_file)]) == 0
    assert "switch" not in json.loads(capsys.readouterr().out)


def test_futures_refuses_a_basket_of_percentages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error that does not raise on its own.

    A basket quoting coupons as 4.75 rather than 0.0475 parses perfectly and
    produces conversion factors a hundred times too large, so the refusal has to
    be explicit and has to say which form is wanted.
    """
    path = tmp_path / "percent.txt"
    path.write_text("2047-02-15, 4.50, 99.337, wrong units\n")
    assert main(futures_arguments(str(path))) == 2
    assert "percentage, not a decimal" in capsys.readouterr().err


def test_futures_names_the_line_of_a_malformed_basket(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("2046-08-15, 0.0175, 63.782\n2047-02-15, 0.0450\n")
    assert main(futures_arguments(str(path))) == 2
    assert "line 2" in capsys.readouterr().err

    path.write_text("not-a-date, 0.0175, 63.782\n")
    assert main(futures_arguments(str(path))) == 2
    assert "is not an ISO date" in capsys.readouterr().err

    path.write_text("2046-08-15, x, y\n")
    assert main(futures_arguments(str(path))) == 2
    assert "are not both numbers" in capsys.readouterr().err

    path.write_text("# nothing but a comment\n")
    assert main(futures_arguments(str(path))) == 2
    assert "no deliverable bonds" in capsys.readouterr().err


def test_futures_takes_a_whitespace_basket_and_an_unlabelled_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("2046-08-15  0.0175  63.782\n2047-05-15  0.0625  122.497\n")
    assert main(["--json", *futures_arguments(str(path))]) == 0
    payload = json.loads(capsys.readouterr().out)
    # With no label the bond names itself from its coupon and maturity.
    assert payload["deliverables"][0]["bond"] == "1.750% of 2046-08-15"


def test_futures_refuses_a_bond_that_has_redeemed_by_delivery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "expiring.txt"
    path.write_text("2026-12-15, 0.05, 100.0, redeems mid-month\n")
    assert main(futures_arguments(str(path))) == 2
    assert "has redeemed by then" in capsys.readouterr().err


def test_the_text_report_renders_a_nested_block_rather_than_a_dict(
    basket_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The formatter used to fall through to ``str`` on a nested mapping.

    Which printed the whole block on one line with braces and quotes in it. The
    ``credit`` command's bond block had the same shape and the same problem.
    """
    assert main(futures_arguments(basket_file)) == 0
    text = capsys.readouterr().out
    assert "cheapest_to_deliver:\n  by_implied_repo: 6.250% of 2047" in text
    assert "{" not in text
    assert "'by_net_basis'" not in text


def test_the_text_report_renders_the_credit_bond_block_too(
    quote_file: str, spread_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "credit",
                *shared(quote_file),
                "--spreads",
                spread_file,
                "--bond-maturity",
                "2026-01-05",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "bond:\n  maturity: 2026-01-05" in text
    assert "'risky_price'" not in text


# -- multicurve ---------------------------------------------------------------

OIS_QUOTES = """
deposit  2026-04-15  0.0290  ACT_360
swap     2027-01-15  0.0310  ANNUAL  ACT_365F
swap     2028-01-17  0.0325  ANNUAL  ACT_365F
swap     2031-01-15  0.0352  ANNUAL  ACT_365F
swap     2036-01-15  0.0368  ANNUAL  ACT_365F
swap     2037-01-15  0.0371  ANNUAL  ACT_365F
"""

FORECAST_QUOTES = """
# three-month index: one forward at the front, par swaps beyond
forward  2026-01-15  2026-04-15  0.0312  ACT_360
swap     2027-01-15  0.0332
swap     2028-01-17  0.0346
swap     2031-01-15  0.0373
swap     2036-01-15  0.0389
"""

BASIS_QUOTES_FILE = """
forward  2026-01-15  2026-07-15  0.0315  ACT_360
basis    2027-01-15  0.0005
basis    2031-01-15  0.0008
basis    2036-01-15  0.0009
"""


@pytest.fixture
def ois_file(tmp_path: Path) -> str:
    path = tmp_path / "ois.txt"
    path.write_text(OIS_QUOTES)
    return str(path)


def multicurve_arguments(ois: str, forecast: str) -> list[str]:
    return [
        "multicurve",
        ois,
        "--reference",
        "2026-01-15",
        "--basis",
        "ACT_365F",
        "--forecast",
        forecast,
    ]


def test_the_forecast_file_parses_all_three_kinds() -> None:
    index = ForecastIndex(tenor=Frequency.QUARTERLY, basis=Basis.ACT_360)
    quotes, spreads = parse_forecast_quotes(
        """
        forward 2026-01-15 2026-04-15 0.0312 ACT_360
        swap    2036-01-15 0.0389
        """,
        date(2026, 1, 15),
        index,
    )
    assert isinstance(quotes[0], IndexForward)
    assert isinstance(quotes[1], DualSwap)
    assert quotes[0].basis is Basis.ACT_360
    assert quotes[0].rate == pytest.approx(0.0312)
    assert spreads == []

    quotes, spreads = parse_forecast_quotes(
        "basis 2031-01-15 0.0008", date(2026, 1, 15), index
    )
    assert quotes == []
    assert spreads[0].spread == pytest.approx(0.0008)
    assert spreads[0].flat_index.tenor is Frequency.SEMI_ANNUAL


def test_the_forecast_file_refuses_mixing_swaps_with_basis_spreads() -> None:
    with pytest.raises(BadInput, match="two curves"):
        parse_forecast_quotes(
            """
            swap  2036-01-15 0.0389
            basis 2031-01-15 0.0008
            """,
            date(2026, 1, 15),
            ForecastIndex(),
        )


def test_the_forecast_file_refuses_an_unknown_kind() -> None:
    with pytest.raises(BadInput, match="not a forecast quote"):
        parse_forecast_quotes("wibble 2027-01-15 0.03", date(2026, 1, 15), ForecastIndex())


def test_the_forecast_file_refuses_an_empty_file() -> None:
    with pytest.raises(BadInput, match="no quotes in it"):
        parse_forecast_quotes("# nothing but a comment\n", date(2026, 1, 15), ForecastIndex())


def test_the_forecast_file_names_the_line_of_a_short_quote() -> None:
    with pytest.raises(BadInput, match="line 2"):
        parse_forecast_quotes(
            "swap 2036-01-15 0.0389\nforward 2026-01-15\n",
            date(2026, 1, 15),
            ForecastIndex(),
        )


def test_the_forecast_file_refuses_a_basis_against_its_own_tenor() -> None:
    with pytest.raises(BadInput, match="says nothing"):
        parse_forecast_quotes(
            "basis 2031-01-15 0.0008 QUARTERLY",
            date(2026, 1, 15),
            ForecastIndex(tenor=Frequency.QUARTERLY),
        )


def test_multicurve_recovers_the_quoted_par_rate_and_splits_the_risk(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forecast = tmp_path / "forecast.txt"
    forecast.write_text(FORECAST_QUOTES)
    arguments = multicurve_arguments(ois_file, str(forecast))
    assert main(["--json", *arguments, "--maturity", "2036-01-15"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["forecast_curve"]["reprices"] is True
    assert payload["forecast_curve"]["quotes_used"] == 5
    assert payload["discount_curve"]["reprices"] is True
    # The ten-year quote goes in and comes back out.
    assert payload["swap"]["par_rate"] == pytest.approx(0.0389, abs=1e-12)
    assert payload["swap"]["par_rate_on_one_curve"] < payload["swap"]["par_rate"]
    assert payload["swap"]["basis_points_from_separating"] == pytest.approx(
        24.15, abs=0.05
    )
    # Every quoted forward basis is in the twenty-something basis point band the
    # quotes imply, which is the check that the two curves were not swapped.
    assert all(20.0 < one["basis_points"] < 30.0 for one in payload["forward_basis"])

    risk = payload["swap"]["risk"]
    assert abs(risk["from_the_forecast_curve"]) > 100 * abs(
        risk["from_the_discount_curve"]
    )
    assert risk["cross_term"] == pytest.approx(
        risk["from_both_together"]
        - risk["from_the_discount_curve"]
        - risk["from_the_forecast_curve"],
        rel=1e-9,
    )


def test_multicurve_omits_the_swap_block_without_a_maturity(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forecast = tmp_path / "forecast.txt"
    forecast.write_text(FORECAST_QUOTES)
    assert main(["--json", *multicurve_arguments(ois_file, str(forecast))]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "swap" not in payload
    assert payload["forward_basis"]


def test_multicurve_uses_the_forwards_in_a_basis_file_too(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dropped quote is the failure a repricing check cannot see.

    The curve reprices whatever it was handed, so leaving a quote on the floor
    shows up as a clean fit to a smaller set. The count is the only thing that
    catches it.
    """
    forecast = tmp_path / "basis.txt"
    forecast.write_text(BASIS_QUOTES_FILE)
    assert main(["--json", *multicurve_arguments(ois_file, str(forecast))]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["forecast_curve"]["quotes_used"] == 4
    assert payload["forecast_curve"]["solving_for"] == "6m index"
    assert payload["forecast_curve"]["reprices"] is True
    assert "for want of a quote" in payload["forecast_curve"]["short_leg_projected_on"]


def test_multicurve_reports_the_assumption_it_makes_on_a_plain_index_curve(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forecast = tmp_path / "forecast.txt"
    forecast.write_text(FORECAST_QUOTES)
    assert main(["--json", *multicurve_arguments(ois_file, str(forecast))]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["forecast_curve"]["short_leg_projected_on"] == "itself"


def test_multicurve_refuses_a_mixed_forecast_file(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forecast = tmp_path / "mixed.txt"
    forecast.write_text(FORECAST_QUOTES + BASIS_QUOTES_FILE)
    assert main(multicurve_arguments(ois_file, str(forecast))) == 2
    assert "two curves" in capsys.readouterr().err


def test_multicurve_refuses_a_lag_that_pays_past_the_curve(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The curve is quoted to the leg's maturity, not to its final payment.

    Found by writing the lag test against the ten-year point, which is the last
    pillar: two business days later there is no curve, and the refusal that
    came back named the missing pillar without naming the lag that caused it.
    It names both now.
    """
    forecast = tmp_path / "forecast.txt"
    forecast.write_text(FORECAST_QUOTES)
    arguments = multicurve_arguments(ois_file, str(forecast))
    assert (
        main([*arguments, "--maturity", "2036-01-15", "--payment-lag", "300"]) == 2
    )
    message = capsys.readouterr().err
    assert "settle past the discount curve" in message
    assert "pays with a lag" in message


def test_multicurve_carries_a_payment_lag_through_to_the_par_rate(
    ois_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lag shows in the single-curve rate, not in the dual-curve one.

    This test was first written against ``par_rate`` and asserted nothing: the
    forecast curve is bootstrapped from the same quote under the same lag, so
    the quote round-trips to the last bit whatever the lag is. The number the
    lag can be seen in is the rate the leg would have on the discount curve
    alone, which no quote has been fitted to.
    """
    forecast = tmp_path / "forecast.txt"
    forecast.write_text(FORECAST_QUOTES)
    arguments = multicurve_arguments(ois_file, str(forecast))
    assert main(["--json", *arguments, "--maturity", "2036-01-15"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["swap"]["par_rate"] == pytest.approx(0.0389, abs=1e-12)
    prompt = payload["swap"]["par_rate_on_one_curve"]

    assert (
        main(["--json", *arguments, "--maturity", "2036-01-15", "--payment-lag", "2"])
        == 0
    )
    lagged = json.loads(capsys.readouterr().out)
    # The quote still round-trips, because the curve was fitted under the lag.
    assert lagged["swap"]["par_rate"] == pytest.approx(0.0389, abs=1e-12)
    # Paying two days late is worth less, so the fair fixed rate is lower.
    assert lagged["swap"]["par_rate_on_one_curve"] < prompt
    assert (prompt - lagged["swap"]["par_rate_on_one_curve"]) * 1e4 == pytest.approx(
        0.095, abs=0.03
    )


# -- swaption -----------------------------------------------------------------

LONG_OIS = OIS_QUOTES + """
swap     2041-01-15  0.0372  ANNUAL  ACT_365F
swap     2046-01-15  0.0374  ANNUAL  ACT_365F
"""

LONG_FORECAST = FORECAST_QUOTES + """
swap     2041-01-15  0.0393
swap     2046-01-15  0.0395
"""


@pytest.fixture
def long_curves(tmp_path: Path) -> tuple[str, str]:
    ois = tmp_path / "ois-long.txt"
    ois.write_text(LONG_OIS)
    forecast = tmp_path / "forecast-long.txt"
    forecast.write_text(LONG_FORECAST)
    return str(ois), str(forecast)


def swaption_arguments(curves: tuple[str, str]) -> list[str]:
    ois, forecast = curves
    return [
        "swaption",
        ois,
        "--reference",
        "2026-01-15",
        "--basis",
        "ACT_365F",
        "--forecast",
        forecast,
        "--expiry",
        "2036-01-15",
        "--swap-maturity",
        "2046-01-15",
        "--volatility",
        "0.20",
    ]


def cms_arguments(curves: tuple[str, str]) -> list[str]:
    ois, forecast = curves
    return [
        "cms",
        ois,
        "--reference",
        "2026-01-15",
        "--basis",
        "ACT_365F",
        "--forecast",
        forecast,
        "--expiry",
        "2031-01-15",
        "--tenor-years",
        "10",
        "--delay-months",
        "6",
        "--fixed-frequency",
        "ANNUAL",
        "--volatility",
        "0.24",
    ]


def test_cms_reports_the_adjustment_and_the_mapping_behind_it(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *cms_arguments(long_curves)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["cms_rate"] > payload["forward_swap_rate"]
    assert payload["adjustment_basis_points"] == pytest.approx(
        1e4 * (payload["cms_rate"] - payload["forward_swap_rate"]), rel=1e-12
    )
    # Tens of basis points, not a rounding concern on the rate it adjusts.
    assert 10.0 < payload["adjustment_basis_points"] < 60.0

    mapping = payload["mapping"]
    # The scale is the rescaling onto the curve; near one, and not one.
    assert 0.95 < mapping["scale"] < 1.0
    assert mapping["scale"] != 1.0
    # The payment falls well before the annuity's own weighted mean, so the
    # slope is positive and so is the adjustment.
    assert mapping["slope"] > 0.0
    assert mapping["annuity_mean_payment_years"] == pytest.approx(5.19, abs=0.05)

    replication = payload["replication"]
    # A swaplet has no kink, so the whole of the convexity is in the integrals.
    assert replication["point_mass"] == 0.0
    assert replication["payer_leg"] > 0.0
    assert replication["receiver_leg"] > 0.0
    assert replication["truncation"] < 1e-20
    # The model's residual arbitrage is reported rather than hidden.
    assert 0.0 < replication["model_arbitrage"] < 0.01


def test_cms_walks_the_payment_date_down_through_zero(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The table that says the adjustment is not simply growing with time."""
    assert (
        main(["--json", *cms_arguments(long_curves), "--delays", "0", "6", "60", "120"])
        == 0
    )
    walk = json.loads(capsys.readouterr().out)["payment_delay"]
    adjustments = [row["adjustment_basis_points"] for row in walk]
    slopes = [row["alpha_slope"] for row in walk]
    assert adjustments == sorted(adjustments, reverse=True)
    assert adjustments[0] > 0.0 > adjustments[-1]
    # The sign of the adjustment is the sign of the mapping's slope throughout.
    assert [one > 0.0 for one in adjustments] == [one > 0.0 for one in slopes]


def test_cms_says_which_skews_have_no_answer(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """An upward slope is reported as unreachable, not as a number.

    The rows either side of it still price, so this is the surface being
    refused rather than the command giving up.
    """
    assert (
        main(["--json", *cms_arguments(long_curves), "--skews", "0.0", "-0.3", "0.3"])
        == 0
    )
    rows = json.loads(capsys.readouterr().out)["skew"]
    assert rows[0]["share_of_flat"] == pytest.approx(1.0, rel=1e-12)
    assert 0.85 < rows[1]["share_of_flat"] < 1.0
    assert "unreachable" in rows[2]
    assert "has not converged" in rows[2]["unreachable"]
    assert "adjustment_basis_points" not in rows[2]


def test_cms_prices_the_strip_beside_the_single_fixing(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--json", *cms_arguments(long_curves), "--leg-maturity", "2034-01-15"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    leg = payload["leg"]
    assert leg["fixings"] == 12
    assert leg["swaplets"] > 0.0
    # Struck at the forward of the single fixing, so the cap is worth more than
    # the floor: every fixing's own adjusted rate sits above that strike.
    assert leg["cap"] > leg["floor"] > 0.0


def test_cms_refuses_a_basis_spread_file(
    tmp_path: Path, long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    ois, _ = long_curves
    spreads = tmp_path / "spreads.txt"
    spreads.write_text("basis  2031-01-15  0.0008  SEMI_ANNUAL\n")
    arguments = cms_arguments((ois, str(spreads)))
    assert main(arguments) == 2
    assert "different index" in capsys.readouterr().err


def test_cms_takes_the_normal_convention(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = cms_arguments(long_curves)
    arguments[arguments.index("0.24")] = "0.0080"
    assert main(["--json", *arguments, "--normal"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["convention"] == "normal"
    assert payload["adjustment_basis_points"] > 0.0


def test_swaption_reports_the_annuity_the_rate_and_both_sensitivities(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *swaption_arguments(long_curves)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["payoff"] == "payer"
    assert payload["rate_option"] == "call"
    # Struck at the forward by default, so there is no intrinsic.
    assert payload["strike"] == pytest.approx(payload["forward_swap_rate"])
    assert payload["intrinsic"] == 0.0
    assert payload["annuity"] > 5.0
    assert payload["value"] > 0.0
    # Its own premium inverts back to the volatility it was priced at.
    assert payload["implied_volatility_from_its_own_premium"] == pytest.approx(
        0.20, rel=1e-9
    )

    shift = payload["discount_shift"]
    # The option moves two orders of magnitude more than the rate it is on.
    assert abs(shift["relative_change_in_value"]) > 100.0 * abs(
        shift["relative_change_in_forward"]
    )
    assert shift["relative_change_in_value"] < 0.0
    assert shift["relative_change_in_annuity"] < 0.0


def test_swaption_prices_the_strip_beside_the_option(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *swaption_arguments(long_curves)]) == 0
    strip = json.loads(capsys.readouterr().out)["strip"]
    assert strip["periods"] == 40
    assert strip["over_the_option"] > 0.0


def test_swaption_takes_the_normal_convention_and_the_receiver(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = swaption_arguments(long_curves)
    arguments[arguments.index("0.20")] = "0.008"
    assert main(["--json", *arguments, "--normal", "--receiver"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["convention"] == "normal"
    assert payload["payoff"] == "receiver"
    assert payload["rate_option"] == "put"
    assert payload["value"] > 0.0


def test_swaption_refuses_a_basis_file_for_a_curve_it_does_not_want(
    long_curves: tuple[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ois, _ = long_curves
    basis = tmp_path / "basis.txt"
    basis.write_text(BASIS_QUOTES_FILE)
    assert (
        main(
            [
                "swaption",
                ois,
                "--reference",
                "2026-01-15",
                "--basis",
                "ACT_365F",
                "--forecast",
                str(basis),
                "--expiry",
                "2036-01-15",
                "--swap-maturity",
                "2046-01-15",
                "--volatility",
                "0.2",
            ]
        )
        == 2
    )
    assert "solves a different index's curve" in capsys.readouterr().err


def test_swaption_refuses_an_expiry_after_its_swap_starts(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = swaption_arguments(long_curves)
    assert main([*arguments, "--effective", "2030-01-15"]) == 2
    assert "already begun accruing" in capsys.readouterr().err


# -- the short rate model -----------------------------------------------------


def hullwhite_arguments(curves: tuple[str, str]) -> list[str]:
    ois, _ = curves
    return [
        "hullwhite",
        ois,
        "--reference",
        "2026-01-15",
        "--basis",
        "ACT_365F",
        "--expiry",
        "2036-01-15",
        "--swap-maturity",
        "2046-01-15",
        "--mean-reversion",
        "0.08",
        "--volatility",
        "0.009",
    ]


def test_hullwhite_reports_the_two_identities_and_the_two_routes(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *hullwhite_arguments(long_curves)]) == 0
    payload = json.loads(capsys.readouterr().out)

    # The repricing is an identity rather than a fit, so it is exactly zero.
    assert payload["curve_repricing_error"] == 0.0
    measure = payload["forward_measure"]
    assert measure["probe_spread"] < 1e-15
    assert measure["mean"] == pytest.approx(measure["instantaneous_forward"], abs=1e-15)

    routes = payload["routes"]
    assert routes["monotone"] is True
    assert abs(routes["relative_gap"]) < 1e-11
    assert routes["jamshidian"] == pytest.approx(routes["quadrature"], rel=1e-11)
    assert routes["reference_rate"] is not None
    assert payload["value"] > 0.0
    assert payload["strike"] == pytest.approx(payload["forward_swap_rate"])


def test_hullwhite_fits_the_volatility_to_a_premium(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *hullwhite_arguments(long_curves)]) == 0
    priced = json.loads(capsys.readouterr().out)["value"]

    arguments = hullwhite_arguments(long_curves)
    arguments[arguments.index("--volatility")] = "--premium"
    arguments[arguments.index("0.009")] = str(priced)
    assert main(["--json", *arguments]) == 0
    payload = json.loads(capsys.readouterr().out)
    # The volatility the price came from comes back out of it.
    assert payload["volatility"] == pytest.approx(0.009, rel=1e-9)
    assert payload["calibration"]["worst"] < 1e-12
    assert payload["calibration"]["iterations"] > 0


def test_hullwhite_reports_the_ridge(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [*hullwhite_arguments(long_curves), "--ridge", "0.02", "0.08", "0.3"]
    assert main(["--json", *arguments]) == 0
    ridge = json.loads(capsys.readouterr().out)["ridge"]
    assert [row["mean_reversion"] for row in ridge] == [0.02, 0.08, 0.3]
    volatilities = [row["volatility"] for row in ridge]
    # A steeper reversion needs a larger volatility to reach the same price,
    # and every point on the ridge reprices it.
    assert volatilities == sorted(volatilities)
    assert volatilities[-1] / volatilities[0] > 3.0
    assert all(row["refit_error"] < 1e-12 for row in ridge)
    assert ridge[1]["volatility"] == pytest.approx(0.009, rel=1e-9)


def test_hullwhite_reports_the_annuity_measure_volatilities(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [*hullwhite_arguments(long_curves), "--normal-vol"]
    assert main(["--json", *arguments]) == 0
    implied = json.loads(capsys.readouterr().out)["implied"]
    # A Gaussian short rate at a 0.9% volatility is tens of basis points of
    # normal volatility and low tens of per cent of lognormal.
    assert 0.002 < implied["normal"] < 0.010
    assert 0.05 < implied["lognormal"] < 0.30


def test_hullwhite_prices_a_mid_curve_by_quadrature_alone(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [*hullwhite_arguments(long_curves), "--effective", "2038-01-15"]
    assert main(["--json", *arguments]) == 0
    payload = json.loads(capsys.readouterr().out)
    routes = payload["routes"]
    assert routes["monotone"] is False
    assert routes["jamshidian"] is None
    assert routes["reference_rate"] is None
    assert routes["quadrature"] > 0.0


def test_hullwhite_takes_the_receiver(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *hullwhite_arguments(long_curves)]) == 0
    payer = json.loads(capsys.readouterr().out)["value"]
    assert main(["--json", *hullwhite_arguments(long_curves), "--receiver"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["payoff"] == "receiver"
    # Struck at the forward, so the two are worth the same.
    assert payload["value"] == pytest.approx(payer, rel=1e-12)


def test_hullwhite_needs_exactly_one_of_a_volatility_and_a_premium(
    long_curves: tuple[str, str]
) -> None:
    arguments = hullwhite_arguments(long_curves)
    both = [*arguments, "--premium", "0.03"]
    assert main(both) == 2
    without = [
        part
        for index, part in enumerate(arguments)
        if part != "--volatility" and arguments[index - 1] != "--volatility"
    ]
    assert main(without) == 2


def test_hullwhite_refuses_a_zero_mean_reversion(long_curves: tuple[str, str]) -> None:
    arguments = hullwhite_arguments(long_curves)
    arguments[arguments.index("0.08")] = "0.0"
    assert main(arguments) == 2


def test_hullwhite_prints_a_human_report(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(hullwhite_arguments(long_curves)) == 0
    out = capsys.readouterr().out
    assert "mean_reversion" in out
    assert "jamshidian" in out


def g2_arguments(curves: tuple[str, str]) -> list[str]:
    ois, _ = curves
    return [
        "g2",
        ois,
        "--reference",
        "2026-01-15",
        "--basis",
        "ACT_365F",
        "--start",
        "2027-01-15",
        "--end",
        "2036-01-15",
        "--frequency",
        "SEMI_ANNUAL",
        "--strike",
        "0.035",
        "--mean-reversion",
        "0.50",
        "--volatility",
        "0.011",
        "--slow-mean-reversion",
        "0.05",
        "--slow-volatility",
        "0.007",
        "--correlation",
        "0.0",
    ]


def test_g2_prices_a_cap_and_reports_the_identities(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *g2_arguments(long_curves)]) == 0
    payload = json.loads(capsys.readouterr().out)

    # Exact rather than fitted: both factors start at zero and the three
    # variance terms cancel at t = 0.
    assert payload["curve_repricing_error"] == 0.0
    assert payload["value"] > 0.0
    assert payload["instrument"] == "cap"
    # The strip sums to the cap, which is the only thing a cap is.
    assert sum(one["value"] for one in payload["caplets"]) == pytest.approx(
        payload["value"]
    )
    assert len(payload["caplets"]) == 18
    # phi exceeds the curve's own forward by the convexity term.
    shift = payload["short_rate_shift"]
    assert shift["phi"] > shift["instantaneous_forward"]
    # Two maturities are no longer perfectly correlated, which one factor
    # cannot manage at any parameter setting.
    assert payload["correlations"]
    for row in payload["correlations"]:
        assert 0.0 < row["correlation"] < 1.0
    # And the integrated factors are less correlated than their drivers.
    assert abs(payload["factor_correlation"]) <= abs(payload["model"]["driver_correlation"])


def test_g2_shows_that_the_cap_does_not_pin_the_correlation(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", *g2_arguments(long_curves), "--identify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    rows = [row for row in payload["identification"] if row["reachable"]]
    assert len(rows) >= 5
    # Every row prices the same cap to rounding.
    for row in rows:
        assert abs(row["relative_gap"]) < 1e-10
    correlations = [row["zero_rate_correlation"] for row in rows]
    assert max(correlations) - min(correlations) > 0.05
    # And the fast volatility has to move a long way to hold the price.
    volatilities = [row["fast_volatility"] for row in rows]
    assert max(volatilities) / min(volatilities) > 2.0


def test_g2_reports_an_unreachable_target_rather_than_a_wrong_fit(
    long_curves: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A strongly negative base correlation puts the target below the floor.

    A cap price is not monotone in the fast volatility at a negative driver
    correlation, so a target set from one such model is often below what any
    volatility reaches at another. The honest answer is to say so; the earlier
    bisection returned the search floor and a price a third of the way off
    while reporting success.
    """
    arguments = g2_arguments(long_curves)
    arguments[arguments.index("--correlation") + 1] = "-0.9"
    assert main(["--json", *arguments, "--identify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    refused = [row for row in payload["identification"] if not row["reachable"]]
    assert refused
    assert "cheapest price on the grid" in refused[0]["reason"]


def test_g2_rejects_a_cap_that_ends_before_it_starts(
    long_curves: tuple[str, str],
) -> None:
    arguments = g2_arguments(long_curves)
    arguments[arguments.index("--end") + 1] = "2026-06-15"
    assert main(arguments) == 2
