"""The README's known-gaps list defends itself.

A list of shortcomings is the first thing in a document to go stale, because
closing a gap is satisfying and deleting the line that described it is not. So
each line carries a predicate, and the predicate holds only while the gap is
real: implement the feature and its line fails, forcing the line out.

Two rules about what belongs on the list:

**Absences, not refusals.** Nearly everything this package declines to do it
declines on purpose -- a survival curve will not be read before today, a
zero-reversion Hull-White is refused rather than approximated as Ho-Lee, an
uneven Bermudan grid is refused rather than interpolated -- and those are
documented as behaviour where they belong. A list of deliberate refusals
relabelled as gaps would read as candour and be worth less than no list.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under a different one. Each predicate below calls an entry point and
observes what happens, scans for instances rather than names, or pins an exact
declared surface, so that an addition under any name breaks it.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
import re
from collections.abc import Callable
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

import tenor
from tenor import (
    G2,
    WEEKENDS_ONLY,
    BadModel,
    Basis,
    BermudanSwaption,
    Calendar,
    HullWhite,
    Payoff,
    ReferenceIndex,
    bermudan_swaption,
    flat_curve,
    swaption_price,
)
from tenor import g2 as g2_module

README = Path(__file__).resolve().parent.parent / "README.md"

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Development"

REFERENCE = date(2026, 1, 15)
CURVE = flat_curve(REFERENCE, date(2046, 1, 15), 0.03, basis=Basis.ACT_365F)
TWO_FACTOR = G2(a=0.05, sigma=0.01, b=0.5, eta=0.008, rho=-0.7)


def parameters(function: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


# --- one predicate per line of the list ----------------------------------- #


def the_bermudan_tree_is_one_factor() -> None:
    """The same option prices under Hull-White and is refused under G2++."""
    option = BermudanSwaption(
        expiries=tuple(date(year, 1, 15) for year in range(2029, 2036)),
        maturity=date(2036, 1, 15),
        strike=0.03,
        payoff=Payoff.PAYER,
    )
    value = bermudan_swaption(CURVE, HullWhite(a=0.05, sigma=0.01), option, REFERENCE)
    assert value.price > 0.0
    with pytest.raises(BadModel, match="one-factor"):
        bermudan_swaption(CURVE, TWO_FACTOR, option, REFERENCE)  # type: ignore[arg-type]
    assert parameters(bermudan_swaption) == (
        "curve",
        "model",
        "option",
        "reference",
        "basis",
        "steps_per_year",
    )


def g2_has_no_swaption() -> None:
    """G2++ declares exactly these; the one swaption pricer is Hull-White's."""
    assert set(g2_module.__all__) == {
        "G2",
        "Correlation",
        "FactorMoments",
        "bond_option",
        "cap_price",
        "caplet",
        "factor_moments",
        "fit_fast_volatility",
        "simulate_factors",
        "zero_rate_correlation",
    }
    with pytest.raises(BadModel, match="one-factor"):
        swaption_price(CURVE, TWO_FACTOR, None, REFERENCE)  # type: ignore[arg-type]


def no_holiday_calendar_is_bundled() -> None:
    """Every calendar object anywhere in the package is the weekends-only one.

    Instances are scanned rather than names, so a calendar shipped under any
    name, in any module, public or not, breaks this.
    """
    found = []
    for info in pkgutil.iter_modules(tenor.__path__):
        module = importlib.import_module(f"tenor.{info.name}")
        for name, value in vars(module).items():
            if isinstance(value, Calendar):
                found.append((info.name, name, value))
    assert found, "the scan found no calendar at all, so it is not looking"
    for where, name, value in found:
        assert value == WEEKENDS_ONLY, f"tenor.{where}.{name} is a bundled calendar"
    assert WEEKENDS_ONLY.holidays == frozenset()


def inflation_projects_at_one_rate() -> None:
    """Past the history, every month grows at the same annualised rate."""
    values: dict[tuple[int, int], float] = {}
    level = 100.0
    for offset in range(36):
        values[(2022 + offset // 12, offset % 12 + 1)] = level
        level *= 1.002
    index = ReferenceIndex(values)
    start = index.last_known_date() + timedelta(days=40)
    days = [start + timedelta(days=30 * k) for k in range(13)]
    projected = [index.projected(day, rate=0.025) for day in days]
    for (left, right), (low, high) in zip(pairwise(days), pairwise(projected), strict=True):
        assert not index.is_known(left)
        annual = math.exp(math.log(high / low) * 365.0 / (right - left).days) - 1.0
        assert annual == pytest.approx(0.025, rel=1e-9, abs=0.0)
    assert parameters(ReferenceIndex.projected) == ("self", "day", "rate", "anchor")


GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("Bermudan swaptions are priced on a one-factor tree.", the_bermudan_tree_is_one_factor),
    ("G2++ prices caps and bond options, and no swaption.", g2_has_no_swaption),
    ("No holiday calendar is bundled.", no_holiday_calendar_is_bundled),
    ("Inflation is projected at one flat rate.", inflation_projects_at_one_rate),
)


def gaps_section() -> str:
    """The text from the list's heading up to the heading that follows it."""
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index(NEXT_SECTION, start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The bold lead of each bullet under the heading, in page order."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.MULTILINE)


@pytest.mark.parametrize("line, predicate", GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    predicate()


@pytest.mark.parametrize("line", [line for line, _ in GAPS])
def test_the_line_is_still_on_the_page(line: str) -> None:
    assert line in rendered_bullets(), f"not a bullet under {SECTION!r}: {line!r}"


def test_the_page_lists_nothing_that_has_no_predicate() -> None:
    """Same lines, same order: a bullet added to the page needs a predicate too."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    predicates = [predicate for _, predicate in GAPS]
    assert len({p.__name__ for p in predicates}) == len(predicates)


def test_a_bold_lead_does_not_wrap() -> None:
    """A lead broken across two lines is invisible to the bullet reader.

    The reader matches within a line, so a wrapped lead is simply not seen and
    the page appears to list one gap fewer. Every bullet in the section must
    therefore close its bold on the line it opened it.
    """
    for line in gaps_section().splitlines():
        if line.startswith("- **"):
            assert line.count("**") >= 2, f"the bold lead wraps: {line!r}"


def test_the_section_ends_at_the_heading_that_follows_it() -> None:
    """The list is bounded by the next ``## `` heading, whatever it is called."""
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.MULTILINE)
    assert SECTION in headings
    following = headings[headings.index(SECTION) + 1]
    assert following == NEXT_SECTION, (
        f"the list is bounded by {NEXT_SECTION!r} and is followed by {following!r}"
    )
