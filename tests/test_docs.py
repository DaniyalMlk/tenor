"""The two documents agree with each other, and the roadmap agrees with itself.

A count written into prose is a claim that stops being true quietly. The front
page said "All seven phases in `ROADMAP.md` are built" for twenty-three phases
after it stopped being seven, because the suite was green: nothing in it read
either document. The roadmap meanwhile had two different phases both numbered
8, so every phase after the second one read one lower than its position and
there was no phase 31 in a document with thirty-one phases.

Both are the same failure. A number nothing recomputes is a number nobody is
checking, so the numbers are recomputed here from the documents themselves.

Six things are asserted, each of which has been demonstrated to fail by being
broken the way the bug would break it:

* the phase count the README states equals the number of sections the roadmap
  has;
* no two phases share a number;
* the numbering has no gap;
* the headings ascend, so a phase is not filed out of order;
* every phase a cross-reference names exists;
* every cross-reference points strictly backwards, at a phase earlier than the
  one it is written in — which is the invariant a half-finished renumber
  breaks, since moving the headings without moving the prose leaves a
  reference naming its own phase or one still to come.

The last of those is the only one that needs defending. A roadmap phase can
refer to work already done and cannot refer to work it precedes, so a forward
reference is not a style question, it is a number that slipped.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"

#: A phase heading: ``## Phase 12 — Deliverable bond futures``. The dash is an
#: em dash in every heading, and matching it rather than any dash keeps a
#: hyphen inside a title from being read as the separator.
HEADING = re.compile(r"^## Phase (\d+) — (.+)$", re.MULTILINE)

#: A reference to a phase from inside the prose, in the three forms the
#: document uses: ``Phase 16``, ``phase 6`` and ``Phases 14 and 15``. The
#: second number is optional and only the compound form has one.
REFERENCE = re.compile(r"[Pp]hases? (\d+)(?: and (\d+))?")

#: The count on the front page is written as a word, so the words that could
#: plausibly appear are mapped rather than a numeral being required. A phase
#: count outside this range would need a line added here, which is the point at
#: which somebody notices the document is being counted.
WORDS = {
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty": 30,
    "thirty-one": 31,
    "thirty-two": 32,
    "thirty-three": 33,
    "thirty-four": 34,
    "thirty-five": 35,
}

#: The sentence the count is written in, on the front page.
STATED_COUNT = re.compile(
    r"All ([a-z]+(?:-[a-z]+)?) phases in `ROADMAP\.md` are built"
)


def phase_headings(roadmap: str) -> list[tuple[int, str]]:
    """Every phase heading in the roadmap, in the order it appears."""
    return [(int(number), title) for number, title in HEADING.findall(roadmap)]


def stated_phase_count(readme: str) -> int:
    """The phase count the front page claims, read off its own sentence."""
    found = STATED_COUNT.search(readme)
    assert found is not None, (
        "the README no longer states a phase count in the form this gate reads. "
        "Either the sentence moved, in which case fix the pattern, or the claim "
        "was deleted, in which case delete this check with it rather than "
        "leaving a pattern that matches nothing and passes."
    )
    word = found.group(1)
    assert word in WORDS, f"unmapped number word {word!r}; add it to WORDS"
    return WORDS[word]


def cross_references(roadmap: str) -> list[tuple[int, int]]:
    """Every prose reference to a phase, as (phase it is written in, named).

    Headings are excluded: a heading is the phase, not a reference to one.
    """
    headings = {match.start() for match in HEADING.finditer(roadmap)}
    bounds = sorted(
        (match.start(), int(match.group(1))) for match in HEADING.finditer(roadmap)
    )

    def enclosing(position: int) -> int:
        current = 0
        for start, number in bounds:
            if start <= position:
                current = number
            else:
                break
        return current

    found: list[tuple[int, int]] = []
    for match in REFERENCE.finditer(roadmap):
        line_start = roadmap.rfind("\n", 0, match.start()) + 1
        if line_start in headings:
            continue
        for group in (1, 2):
            if match.group(group) is not None:
                found.append((enclosing(match.start()), int(match.group(group))))
    return found


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_the_front_page_count_is_the_number_of_phases(readme: str, roadmap: str) -> None:
    """The README's stated count equals the sections the roadmap has.

    This is the check the original defect needed. The sentence said seven and
    the roadmap had thirty-one.
    """
    headings = phase_headings(roadmap)
    assert stated_phase_count(readme) == len(headings), (
        f"the README claims {stated_phase_count(readme)} phases and the roadmap "
        f"has {len(headings)} sections"
    )


def test_no_two_phases_share_a_number(roadmap: str) -> None:
    """Each phase number appears once.

    Two phases were both 8, which is how the numbering came to run one short of
    the count for twenty-three phases.
    """
    numbers = [number for number, _ in phase_headings(roadmap)]
    repeated = {number: count for number, count in Counter(numbers).items() if count > 1}
    assert not repeated, f"phase numbers used more than once: {repeated}"


def test_the_numbering_has_no_gap(roadmap: str) -> None:
    """The numbers run 1 to N with nothing missing."""
    numbers = [number for number, _ in phase_headings(roadmap)]
    expected = set(range(1, len(numbers) + 1))
    assert set(numbers) == expected, (
        f"missing: {sorted(expected - set(numbers))}; "
        f"unexpected: {sorted(set(numbers) - expected)}"
    )


def test_the_headings_ascend(roadmap: str) -> None:
    """A phase is filed in the order its number says."""
    numbers = [number for number, _ in phase_headings(roadmap)]
    out_of_order = [
        (numbers[i], numbers[i + 1])
        for i in range(len(numbers) - 1)
        if numbers[i + 1] <= numbers[i]
    ]
    assert not out_of_order, f"headings out of order at: {out_of_order}"


def test_every_phase_a_reference_names_exists(roadmap: str) -> None:
    """No cross-reference names a phase the document does not have."""
    numbers = {number for number, _ in phase_headings(roadmap)}
    dangling = sorted(
        {named for _, named in cross_references(roadmap) if named not in numbers}
    )
    assert not dangling, f"references to phases that do not exist: {dangling}"


def test_every_reference_points_backwards(roadmap: str) -> None:
    """A phase refers to earlier work, never to itself or to what follows.

    This is the invariant that catches a renumber finished halfway. Shifting the
    headings and leaving the prose behind turns a backward reference into one
    that names its own phase or a later one, and nothing else in this file can
    see that.
    """
    forward = [
        (written_in, named)
        for written_in, named in cross_references(roadmap)
        if named >= written_in
    ]
    assert not forward, (
        "cross-references pointing at their own phase or a later one "
        f"(written in, named): {forward}"
    )


def test_the_reference_probe_finds_the_references_that_are_there(roadmap: str) -> None:
    """The probe above is not matching nothing.

    Both reference tests pass vacuously if the pattern stops matching, and a
    gate that passes because it found nothing to check is worse than no gate.
    The roadmap has eight prose references, two of them in one compound
    sentence, so nine (written in, named) pairs come out.
    """
    found = cross_references(roadmap)
    assert len(found) == 9, f"expected 9 reference pairs, found {len(found)}: {found}"
    assert (17, 16) in found, (
        "the 'Phase 16 is a real model' reference, written in phase 17, is missing"
    )
    assert {(16, 14), (16, 15)} <= set(found), (
        "the compound 'Phases 14 and 15' reference is not being read as two"
    )
    assert all(written_in > 0 for written_in, _ in found), (
        "a reference was found outside any phase, so the enclosing-phase lookup "
        "returned its zero default and the backwards check cannot see it"
    )
