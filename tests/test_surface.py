"""The package's public surface is the union of what its modules declare.

``src/tenor/__init__.py`` re-exports the package's public names, and it is a
hand-maintained list. A hand-maintained list of three hundred names is a list
with something missing from it, and what was missing was a whole phase:
``bermudan`` — a trinomial tree, coterminal Europeans, ``bermudan_swaption``,
ten names in the module's own ``__all__`` — did not appear in ``__init__.py``
at all, so ``tenor.bermudan_swaption`` raised ``AttributeError`` while the
module worked perfectly.

Nothing noticed because nothing looked. Every test file in this suite imports
submodules directly, ``from tenor.bermudan import ...``, which is the right
style for a test and exactly the style that cannot see this: the module is
exercised in full and the surface it is missing from is never touched. The CLI
reached it too, through its own subcommand. Only a caller following the
package's own convention — that public names live at the root — was stopped.

So the surface is recomputed here from the modules rather than compared against
a second list that would need the same remembering.

**A module's own ``__all__`` is the authority.** The alternative is to decide
here which names are public, which puts the judgement in the gate instead of
beside the code it is about, and makes the gate wrong whenever a module changes
its mind. A module that declares a name public and a root that does not carry
it is a disagreement inside the package, and that is what is asserted.

**The reverse direction is not asserted, and that is a gap rather than an
oversight.** Twenty-four names reach the root without appearing in any module's
``__all__``: ``floating`` and ``inflation`` declare none at all, and a handful
of constants elsewhere were never added to theirs. Asserting root equals union
would mean deciding, here, that each of those is public — the judgement this
file just refused to make. Closing it properly means giving those two modules
an ``__all__`` and completing the others, in the modules, after which the
equality can replace the containment. Until then the containment is the claim.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

import tenor

ROOT = Path(__file__).resolve().parent.parent

#: ``from tenor.something import ...`` in a front-page example: an import that
#: reaches past the package into one of its modules.
SUBMODULE_IMPORT = re.compile(r"^from (tenor\.[a-z0-9_]+) import", re.MULTILINE)

#: ``from tenor import ...``, the house style every other example uses.
ROOT_IMPORT = re.compile(r"^from tenor import", re.MULTILINE)

#: The command line is not part of the importable surface: it is an entry point
#: and its names are wired to ``argparse``, not to callers.
NOT_SURFACE = frozenset({"cli"})

#: The modules that declare no ``__all__``. Pinned rather than discovered, so
#: that a new module arriving without one is a decision somebody makes here
#: instead of a silent exemption from every check in this file.
NO_DUNDER_ALL = frozenset({"floating", "inflation"})


def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry point."""
    found = {}
    for info in pkgutil.iter_modules(tenor.__path__):
        if info.name in NOT_SURFACE:
            continue
        found[info.name] = importlib.import_module(f"tenor.{info.name}")
    return found


def declared() -> dict[str, frozenset[str]]:
    """Each module's own declaration of what it makes public."""
    return {
        name: frozenset(module.__all__)
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is not None
    }


@pytest.fixture(scope="module")
def root() -> frozenset[str]:
    return frozenset(tenor.__all__)


def test_every_declared_name_reaches_the_root(root: frozenset[str]) -> None:
    """A name a module declares public is importable from the package.

    This is the check the defect needed. Seventeen names failed it, ten of them
    one module's entire surface.
    """
    missing: dict[str, list[str]] = {}
    for module, names in declared().items():
        absent = sorted(names - root)
        if absent:
            missing[module] = absent
    assert not missing, (
        "declared public but not re-exported from tenor/__init__.py: "
        f"{missing}"
    )


def test_no_module_is_absent_from_the_root_entirely(root: frozenset[str]) -> None:
    """A module whose whole surface is missing is reported as the module.

    The check above would catch this as well, as seventeen loose names across
    six modules, which reads as six small omissions rather than as the one
    large one it was. A module nobody wired in at all is a different mistake
    from a name somebody forgot, and it is worth saying so.
    """
    absent = sorted(
        module for module, names in declared().items() if names and not (names & root)
    )
    assert not absent, (
        f"these modules declare public names and none of them is at the root, "
        f"so the module was never wired into tenor/__init__.py: {absent}"
    )


def test_every_exported_name_is_actually_there(root: frozenset[str]) -> None:
    """Nothing in ``__all__`` is a name the package does not have.

    ``__all__`` is a list of strings and nothing checks it against the module,
    so a rename that updates the import and not the list, or a typo in either,
    leaves an entry that only fails for somebody doing ``from tenor import *``.
    """
    phantom = sorted(name for name in root if not hasattr(tenor, name))
    assert not phantom, f"in tenor.__all__ but not attributes of tenor: {phantom}"


def test_the_root_exports_each_name_once() -> None:
    """``__all__`` does not repeat itself.

    A duplicate is harmless at import time and is the visible residue of a
    merge that went wrong, so it is worth failing on while it is still one
    line.
    """
    repeated = {
        name: count for name, count in Counter(tenor.__all__).items() if count > 1
    }
    assert not repeated, f"repeated in tenor.__all__: {repeated}"


def test_the_modules_without_a_declaration_are_the_ones_pinned_here() -> None:
    """Only the two known modules carry no ``__all__``.

    Every check above is silent about a module that declares nothing, so the
    set of modules exempt from them is written down. A new module without an
    ``__all__`` fails here, which is the point at which somebody decides
    whether that is what was wanted.
    """
    silent = {
        name
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is None
    }
    assert silent == set(NO_DUNDER_ALL), (
        f"modules carrying no __all__: {sorted(silent)}; "
        f"pinned: {sorted(NO_DUNDER_ALL)}"
    )


def test_the_front_page_imports_from_the_package_and_not_from_inside_it() -> None:
    """No example on the front page reaches into a submodule.

    This is the check that would have caught the original defect through the
    documentation rather than through the surface, and it is worth having
    because it caught it *in the document*: thirteen examples wrote
    ``from tenor import ...`` and the Bermudan one wrote
    ``from tenor.bermudan import BermudanSwaption, bermudan_swaption`` on the
    line directly below a ``from tenor import Basis, HullWhite, flat_curve``.
    The example had been written around the missing export, in adjacent lines,
    and the workaround read as a style choice.

    So the convention is asserted rather than left to whoever writes the next
    example: if a module is not re-exported, its example cannot be written in
    the house style, and that shows up here instead of being worked around.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reaching_in = sorted(set(SUBMODULE_IMPORT.findall(readme)))
    assert not reaching_in, (
        "front-page examples importing from inside the package rather than from "
        f"it: {reaching_in}. Either re-export the names or say here why this one "
        "is different."
    )
    assert ROOT_IMPORT.search(readme) is not None, (
        "no `from tenor import` line found on the front page at all, so the "
        "check above passed by reading nothing"
    )


def test_the_probe_reads_a_surface_worth_checking(root: frozenset[str]) -> None:
    """The checks above are not passing because they found nothing.

    Every assertion in this file is satisfied by an empty set, so a probe that
    stops importing modules or stops finding ``__all__`` turns the whole file
    green. The sizes are therefore asserted with loose bounds — the claim is
    that a surface was read at all, not that it is exactly this big — and
    ``bermudan`` is named, because it is the module that was missing and the
    one a future rewrite is most likely to drop again.
    """
    modules = declared()
    assert len(modules) >= 20, f"only {len(modules)} modules declared an __all__"
    assert len(root) >= 250, f"the root surface came back as {len(root)} names"
    assert "bermudan" in modules, "the bermudan module was not read"
    assert {"bermudan_swaption", "BermudanSwaption", "TrinomialTree"} <= root
