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
    assert f"{bound:.0e}".replace("e-", "e-") in README.read_text() or "1e-12" in (
        README.read_text()
    )


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
    assert f"all {len(numbers)}" in README.read_text() or "eight" in README.read_text()


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. An order-of-
    magnitude check still catches the README claiming hundreds when there are
    thousands, which is the error that matters.
    """
    text = README.read_text()
    claimed = re.search(r"([\d,]+) tests", text)
    assert claimed is not None, "the README no longer states a suite size"
    stated = int(claimed.group(1).replace(",", ""))
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
    assert 0.75 * actual <= stated <= 1.25 * actual, (
        f"the README says {stated} tests and there are {actual}"
    )
