"""The figures on the front page are the figures the code produces.

The README opens with a table of results checked against things outside this
repository. A table like that is worth exactly as much as the guarantee that
it still matches, and a number copied into a document once is a number that
stops being true quietly: the code changes, the suite stays green because
nothing in it reads the document, and the front page goes on claiming a
precision the library no longer has.

So the claim is pinned here. Every figure the README quotes has to appear in
the output of ``examples/worked.py``, which is itself a gate — it reproduces
published results and exits non-zero if any of them moves. The two together
mean a reader can trust the front page without running anything, and a change
that moves a figure fails the suite rather than the reader.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
WORKED = ROOT / "examples" / "worked.py"

#: Figures the README quotes, as they appear in the worked example's output.
#: Each entry is the substring that must be present. They are written out
#: rather than parsed from the README so that deleting a row from the table
#: does not silently delete the check that defends it.
QUOTED = (
    "5.551115123e-17",  # log-linear bootstrap, worst repricing error
    "3.191891196e-16",  # monotone convex, worst repricing error
    "47.52",  # Hull's futures convexity adjustment
    "7.89499734",  # Macaulay duration against its closed form
    "4166.666667",  # accrued premium on 10mm at 100bp, a day count
)

#: Figures that are floating-point noise and whose digits are therefore not
#: reproducible: the Z-spread of a bond priced on its own curve comes back at
#: -8.3e-14 on CPython 3.12 and +1.3e-13 on 3.11, because the last bits a root
#: finder lands on are not portable. The README quotes them as bounds, so the
#: bound is what is checked. Each entry is the label the worked example prints
#: and the magnitude the README promises it stays under.
BOUNDED = (
    ("Z-spread of a bond priced on the curve, bp", 1e-12),
    ("OAS of the callable, bp", 1e-12),
    ("premium leg of a running contract, relative to a midpoint rule", 1e-10),
    # The same quantity by two routes, but not the same arithmetic: one divides
    # each flow by the settlement discount factor and sums, the other sums and
    # divides once. The residue is 1.4e-14 on 3.11 and exactly zero on 3.12 and
    # 3.13, which is why this is a bound and not one of the EXACT rows below.
    ("risky less riskless forward price, settling 2026-01-05", 1e-13),
    ("risky less riskless forward price, settling 2028-01-05", 1e-13),
)

#: Rows whose reference is an identity rather than a published figure, so the
#: answer is zero and not a small number. These are checked as exact zeros:
#: accepting "near enough" from an identity is accepting the one failure the
#: row exists to catch, since a subtly wrong implementation returns a
#: plausibly small number and a correct one returns nothing at all.
EXACT = (
    "protection before the mark, running less struck-today",
    "settlement less its own principal and accrued",
)


@pytest.fixture(scope="module")
def worked_output() -> str:
    """Run the worked example once and share its output across the tests."""
    finished = subprocess.run(
        [sys.executable, str(WORKED)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert finished.returncode == 0, (
        f"examples/worked.py exited {finished.returncode}:\n{finished.stdout}"
        f"\n{finished.stderr}"
    )
    return finished.stdout


@pytest.mark.parametrize("figure", QUOTED)
def test_a_quoted_figure_is_one_the_worked_example_prints(
    figure: str, worked_output: str
) -> None:
    assert figure in worked_output, (
        f"the README quotes {figure}, which examples/worked.py no longer prints"
    )


@pytest.mark.parametrize(("label", "bound"), BOUNDED)
def test_a_noise_figure_stays_inside_the_bound_the_readme_promises(
    label: str, bound: float, worked_output: str
) -> None:
    """Quoting the digits of a rounding error is quoting the platform.

    These two are spreads measured against the curve the instrument was priced
    on, so the true answer is zero and what prints is whatever the root finder
    had left over. That value is not the same on every interpreter, which is
    why the README states a bound and this reads the printed figure back and
    checks it against that bound rather than against a remembered string.
    """
    match = re.search(
        rf"{re.escape(label)}\s+(-?[\d.]+e?-?\d*)", worked_output
    )
    assert match is not None, f"the worked example no longer prints {label!r}"
    value = float(match.group(1))
    assert abs(value) < bound, f"{label} printed {value}, outside {bound}"
    # And that the README actually promises a bound of this size. The two
    # halves of this used to be the same string or-ed together, which cannot
    # fail: once any row quoted 1e-12 the check passed for every bound.
    assert f"{bound:.0e}" in README.read_text(), (
        f"the README does not promise {bound:.0e} anywhere"
    )


@pytest.mark.parametrize("label", EXACT)
def test_an_identity_row_reproduces_as_an_exact_zero(label: str, worked_output: str) -> None:
    """Not "small": zero.

    Both of these are differences between two routes to one quantity that
    agree by construction rather than by cancellation - a protection integral
    whose lower limit is the valuation date under either contract, and a
    settlement split into two halves derived from it. Nothing is approximated
    on the way, so a non-zero answer is a bug and not a rounding error, and a
    tolerance here would accept exactly the failure the row is for.
    """
    match = re.search(rf"{re.escape(label)}\s+(\S+)", worked_output)
    assert match is not None, f"the worked example no longer prints {label!r}"
    assert float(match.group(1)) == 0.0, f"{label} printed {match.group(1)}"


#: The row labels of the README's evidence table. Presence is what is checked
#: rather than the values beside them, for the reason in the test below.
ROWS = (
    "A bootstrapped curve reprices every instrument it was built from",
    "under monotone convex interpolation",
    "Hull's futures convexity adjustment",
    "The three ISDA 4.16 conventions",
    "Z-spread of a bond priced on its own curve",
    "Option-adjusted spread of a callable priced on its own curve",
    "Macaulay duration at a 6% yield",
    "Premium leg of a contract already running, against a midpoint rule",
    "Protection before the valuation date, priced from inside the schedule",
    "A settlement amount against its own split into principal and accrued",
    "A defaultable bond's forward price, against the riskless one at no hazard",
)


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str) -> None:
    """A row removed from the table takes its claim with it, so it is pinned.

    Only the label, not the figure beside it. A check that the README still
    quotes the right *value* cannot be done by scanning the document: the
    table deliberately prints the library's result next to the published
    number it is being compared against — "47.52 bp against his 47.5" — and
    those two agree to four parts in ten thousand, which is far inside any
    tolerance that would still accept the rounding elsewhere in the same
    table. A scan cannot tell which number it found.

    The direction that catches drift is the other one, above: every figure the
    table quotes has to be one the worked example prints. That fails the
    moment the code's output moves, which is the failure worth catching. This
    test covers the other way a claim disappears, which is somebody deleting
    it.
    """
    assert row in README.read_text(), f"the evidence table no longer has: {row}"


def test_the_sections_are_numbered_without_a_gap(worked_output: str) -> None:
    """The count the README gives has to be the count the example prints.

    The last section was labelled 9 while being the eighth, which is the kind
    of thing nobody notices until they are counting.
    """
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", worked_output, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1))
    # The README writes the count as a word. Reading it back and comparing it
    # to the sections actually printed is the whole point; the previous version
    # accepted the literal "eight" as an alternative to the real count, and
    # "eight" appears elsewhere in the README, so it could no longer fail.
    words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten",
    }
    expected = f"all {words[len(numbers)]} sections"
    assert expected in README.read_text(), (
        f"the worked example prints {len(numbers)} sections and the README does "
        f"not say {expected!r}"
    )


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately, and applied to every figure rather than one.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. An order-of-
    magnitude check still catches the README claiming hundreds when there are
    thousands, which is the error that matters.

    This used to take the *first* count on the page and check that one. The
    README states the size twice -- once in the evidence section and once in
    the development instructions -- so the second figure was never looked at.
    It had drifted to 1,535 against a real 2,014: 23.8% low, which is to say
    *just* inside this band and one commit from falling out of it, with nothing
    reading it either way. A guard that stops at the first match on a long page
    is a guard over the first match, so every figure the page states is checked
    here.
    """
    text = README.read_text()
    stated = [int(found.replace(",", "")) for found in re.findall(r"([\d,]+) tests", text)]
    assert stated, "the README no longer states a suite size"
    assert len(stated) == 2, (
        f"the README states {len(stated)} suite sizes; this test expects the two it "
        "has always had, and a new one needs looking at rather than averaging in"
    )
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-o",
         "addopts="],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    found = re.search(r"(\d+) tests collected", collected.stdout)
    assert found is not None, collected.stdout[-500:]
    actual = int(found.group(1))
    for figure in stated:
        assert 0.75 * actual <= figure <= 1.25 * actual, (
            f"the README says {figure} tests somewhere and there are {actual}"
        )
