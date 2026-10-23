"""Help text is only as good as the last time anybody rendered it.

Two holes this closes, both invisible to the rest of the suite.

The first is that ``--help`` can raise. argparse expands every help string
through ``%``-formatting so that ``%(default)s`` works, which makes a literal
per cent sign a conversion specifier: ``20% in cash`` is read as ``%i``,
handed a dict of parameters, and raises ``TypeError``. Nothing else notices,
because the commands themselves never render their help. Only ``--help`` is
broken, which is the first thing anybody types.

The second is that help text can simply be absent. An option that prints its
flag name and its choices and nothing else tells the reader what values are
accepted and not what any of them mean.

Both sweeps enumerate commands and arguments from the parser rather than from
a list written here. A hand-written list is why this class of bug survives:
the subcommand added next week is not in it.
"""

from __future__ import annotations

import argparse
import inspect
import io
import re
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from tenor.cli import main, parser


def command_paths(
    command: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Every command path the parser can reach, the root included.

    Recursive, so a subcommand that grows subcommands of its own is swept
    without anybody remembering to come back here.
    """
    yield prefix
    for action in command._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in sorted(action.choices.items()):
                yield from command_paths(child, (*prefix, name))


def arguments_of(command: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in command._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        # --help and --version describe themselves.
        if isinstance(action, argparse._HelpAction | argparse._VersionAction):
            continue
        yield action


PATHS = tuple(command_paths(parser()))
IDS = tuple(" ".join(path) or "tenor" for path in PATHS)


def test_the_sweep_found_the_commands() -> None:
    """Guards the guard.

    If introspection stops finding subcommands the sweeps below would pass by
    testing nothing at all, so the count and a known name are asserted here
    rather than left implicit.
    """
    assert len(PATHS) >= 22  # the root plus twenty-one subcommands
    assert () in PATHS
    assert ("curve",) in PATHS
    assert ("assetswap",) in PATHS


@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_help_renders(path: tuple[str, ...]) -> None:
    printed = io.StringIO()
    with redirect_stdout(printed), pytest.raises(SystemExit) as exited:
        main([*path, "--help"])
    assert exited.value.code == 0
    text = printed.getvalue()
    assert text.startswith("usage:")
    assert " ".join(("tenor", *path)) in " ".join(text.split())


@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_every_argument_has_help(path: tuple[str, ...]) -> None:
    command = parser()
    for name in path:
        for action in command._actions:
            if isinstance(action, argparse._SubParsersAction):
                command = action.choices[name]
                break
    missing = [
        action.dest
        for action in arguments_of(command)
        if not (action.help or "").strip()
    ]
    assert not missing, f"{' '.join(('tenor', *path))}: {missing}"


def test_a_literal_per_cent_sign_would_be_caught() -> None:
    """The sweep above is only a guard if a bad string really fails it.

    Built here rather than left to trust: the same expansion argparse performs
    at display time, on the string that breaks it.
    """
    broken = argparse.ArgumentParser(prog="broken", add_help=True)
    broken.add_argument("--cash", help="20% in cash")
    with pytest.raises(TypeError):
        broken.format_help()

    # And the escaped form renders, with the sign still in it.
    fixed = argparse.ArgumentParser(prog="fixed", add_help=True)
    fixed.add_argument("--cash", help="20%% in cash")
    assert "20% in cash" in " ".join(fixed.format_help().split())


# -- a bad date names the option it came from ----------------------------------

CLI_SOURCE = Path(inspect.getsourcefile(main) or "").read_text()

#: Every option that `_date` is asked to read, taken from the source rather
#: than written out here, so a new date option is swept without anybody
#: remembering to come back.
DATE_OPTIONS = tuple(sorted(set(re.findall(r'_date\(arguments, "([a-z-]+)"\)', CLI_SOURCE))))


def long_options(command: argparse.ArgumentParser) -> set[str]:
    """Every long option the parser accepts anywhere, without its dashes."""
    found: set[str] = set()
    for path in command_paths(command):
        node = command
        for name in path:
            for action in node._actions:
                if isinstance(action, argparse._SubParsersAction):
                    node = action.choices[name]
                    break
        for action in arguments_of(node):
            found.update(
                flag.lstrip("-") for flag in action.option_strings if flag.startswith("--")
            )
    return found


def test_the_sweep_found_the_date_options() -> None:
    # If the pattern stops matching, every test below passes by sweeping
    # nothing. Nineteen distinct options read a date at forty-two call sites.
    assert len(DATE_OPTIONS) >= 15


@pytest.mark.parametrize("option", DATE_OPTIONS)
def test_the_option_a_date_error_names_is_one_the_parser_has(option: str) -> None:
    """The string in the message is the part that can go stale.

    `_date` is handed the option name as text, so renaming an option leaves a
    message naming one that no longer exists and nothing else would notice.
    Checked against the parser rather than against a list here, for the reason
    in this module's docstring.
    """
    assert option in long_options(parser()), (
        f"a date error names --{option}, which the parser does not accept"
    )


def test_nothing_reads_a_date_without_going_through_the_helper() -> None:
    # The claim is that the behaviour is uniform, not that it is better in the
    # places somebody happened to touch. Forty-two sites read a date straight
    # out of the namespace before this; none do now.
    assert "date.fromisoformat(arguments." not in CLI_SOURCE


@pytest.mark.parametrize(
    ("argv", "option"),
    [
        (["curve", "QUOTES", "--reference", "yesterday", "--basis", "ACT_365F"], "reference"),
        (
            ["price", "QUOTES", "--reference", "2021-01-05", "--basis", "ACT_365F",
             "--maturity", "soon", "--coupon", "0.05"],
            "maturity",
        ),
        (
            ["horizon", "QUOTES", "--reference", "2021-01-05", "--basis", "ACT_365F",
             "--maturity", "2031-01-05", "--coupon", "0.05", "--horizon", "a while"],
            "horizon",
        ),
    ],
)
def test_a_bad_date_names_the_option_and_exits_two(
    argv: list[str],
    option: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quotes = tmp_path / "quotes.txt"
    quotes.write_text("swap 2031-01-05 0.022 SEMI_ANNUAL THIRTY_360_BOND\n")
    assert main([one.replace("QUOTES", str(quotes)) for one in argv]) == 2
    printed = capsys.readouterr().err
    assert f"--{option} is " in printed
    assert "not a date" in printed
    # And still says what was wrong with it, which the standard library's
    # message did get right.
    assert "Invalid isoformat" not in printed


def test_a_good_date_is_not_mentioned_at_all(tmp_path: Path) -> None:
    # The other half: the helper has to be transparent when the date is fine,
    # or the sweep above would pass on a parser that rejects everything.
    quotes = tmp_path / "quotes.txt"
    quotes.write_text("swap 2031-01-05 0.022 SEMI_ANNUAL THIRTY_360_BOND\n")
    assert main(["curve", str(quotes), "--reference", "2021-01-05", "--basis", "ACT_365F"]) == 0
