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

**The reverse direction is asserted too, now that the modules can carry it.**
It could not before: twenty-four names reached the root without appearing in
any module's ``__all__``, because ``floating`` and ``inflation`` declared none
at all and ``cms``, ``futures`` and ``parametric`` had each forgotten names of
their own. Asserting root equals union would have meant deciding, here, which
of those twenty-four were public — the judgement this file refuses to make.

Measuring them showed the judgement was not needed: in every module concerned
the public top-level definitions were already exactly what ``__init__``
imports. So each ``__all__`` records the shipped surface, and the equality
replaces the containment.

That makes a third claim available, and it is the one that catches the next
omission rather than this one: **a module's ``__all__`` contains every public
name the module defines.** A function added to a module and not declared fails
here, at the module, instead of travelling as far as the root before anybody
notices. The probe for it reads the source with :mod:`ast` rather than the
imported module, because a name's ``__module__`` is unreliable — a plain
``float`` constant has none at all, which is how the two in ``futures`` and the
two in ``parametric`` stayed invisible to an attribute-based sweep.

**A re-export is permitted and pinned.** ``curve.__all__`` carries
``BadRate``, which ``rates`` defines; ``curve`` imports it and never raises it,
so the entry's only effect is a second import path to a name already at the
root. That is a re-export, not a forgotten definition, and removing it would
change what ``from tenor.curve import *`` yields, which is somebody's decision
and not this file's. It is the one case in the package, so it is named here: a
second one fails until somebody adds it deliberately.
"""

from __future__ import annotations

import ast
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

#: Names in the root's ``__all__`` that no module declares. ``__version__`` is
#: assigned in ``__init__.py`` itself, so it is the one legitimate entry.
ROOT_ONLY = frozenset({"__version__"})

#: ``(module, name)`` pairs where a module declares a name another module
#: defines. One case exists: ``curve`` re-exports ``rates.BadRate``. Pinned so
#: that a second one is a decision rather than a drift.
RE_EXPORTS = frozenset({("curve", "BadRate")})


def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry point."""
    found = {}
    for info in pkgutil.iter_modules(tenor.__path__):
        if info.name in NOT_SURFACE:
            continue
        found[info.name] = importlib.import_module(f"tenor.{info.name}")
    return found


def public_definitions(module: str) -> frozenset[str]:
    """Every public name a module defines at its top level, read from source.

    Read with :mod:`ast` rather than off the imported module. An attribute
    sweep has to decide which attributes a module *defines* as opposed to
    imports, and the usual test — ``obj.__module__`` — does not exist on a
    plain ``int``, ``float`` or ``str``. Four of the five omissions this check
    was written for were exactly that: ``DEFAULT_DIGITS``,
    ``DEFAULT_NOTIONAL_COUPON``, ``GRID_POINTS`` and ``SERIES_LIMIT``, all of
    them bare numbers an attribute-based probe cannot attribute to anybody.
    """
    source = ROOT / "src" / "tenor" / f"{module}.py"
    return public_names_in(source.read_text(encoding="utf-8"))


def public_names_in(source: str) -> frozenset[str]:
    """The public top-level names a piece of module source defines.

    Separated from the file it usually reads so that it can be tested against
    source written for the purpose. Four of its branches are not exercised by
    this package as it stands — no module declares an ``async def`` or an
    annotated assignment at the top level — and a rule the source does not
    exercise is a rule nobody has checked.
    """
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        found.update(
            name for name in names
            if not name.startswith("_") and name != "__all__"
        )
    return frozenset(found)


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


def test_every_module_declares_something() -> None:
    """No module carries no ``__all__`` at all.

    Every other check in this file is silent about a module that declares
    nothing, so a module without one is exempt from all of them. ``floating``
    and ``inflation`` were in exactly that position, which is why the
    equality below could not be asserted until they were given one.

    There is no exemption list. A module with nothing public to say can
    declare an empty ``__all__`` and say so.
    """
    silent = sorted(
        name
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is None
    )
    assert not silent, (
        f"modules carrying no __all__, and so exempt from every check here: "
        f"{silent}"
    )


def test_a_module_declares_every_public_name_it_defines() -> None:
    """``__all__`` is not a curated subset of what the module defines.

    This is the assertion that catches the *next* omission rather than the
    last one: a function added to a module and left out of its ``__all__``
    fails here, at the module, instead of reaching the root check as a name
    nobody wired in.

    Five names failed it. ``cms.RatePayoff`` is the telling one — two of the
    three payoff types were declared and the third was not, which is what a
    forgotten line looks like rather than a decision.
    """
    undeclared: dict[str, list[str]] = {}
    for module, names in declared().items():
        missing = sorted(public_definitions(module) - names)
        if missing:
            undeclared[module] = missing
    assert not undeclared, (
        f"defined at a module's top level and absent from its own __all__: "
        f"{undeclared}"
    )


def test_a_name_a_module_declares_but_does_not_define_is_a_pinned_re_export() -> None:
    """A module may re-export, but only the one case anybody decided on.

    ``curve.__all__`` carries ``BadRate``, which ``rates`` defines. That is a
    re-export and is left alone; what is not left alone is a second one
    appearing without anybody saying so, because a declaration of a name from
    elsewhere is also what a bad merge looks like.
    """
    found = {
        (module, name)
        for module, names in declared().items()
        for name in names - public_definitions(module)
    }
    assert found == set(RE_EXPORTS), (
        f"re-exports found: {sorted(found)}; pinned: {sorted(RE_EXPORTS)}"
    )


def test_the_root_is_exactly_the_union_of_the_declarations(
    root: frozenset[str],
) -> None:
    """Equality, not containment.

    The containment above catches a name the root is missing. This catches the
    other direction — a name the root offers that no module claims — which is
    what a hand-kept list of three hundred entries accumulates: a name left
    behind by a rename, or one somebody exported from the wrong place.
    """
    union = frozenset().union(*declared().values())
    assert root == union | ROOT_ONLY, (
        f"only at the root: {sorted(root - union - ROOT_ONLY)}; "
        f"only in a module: {sorted(union - root)}"
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


#: Source written to exercise the probe rather than to be a module. Each line
#: is one rule, and the comment says which.
PROBE_SOURCE = '''
"""A docstring, which defines nothing."""

from __future__ import annotations

import math                      # an import is not a definition
from pathlib import Path         # nor is this

PUBLIC_CONSTANT = 1.0            # a bare float: no __module__ to sweep by
ANNOTATED: int = 2               # an annotated assignment
PAIR_ONE = PAIR_TWO = 3          # one statement, two names
_PRIVATE = 4                     # underscore: not public
__all__ = ["PUBLIC_CONSTANT"]    # the declaration itself is never a definition


def public_function() -> None:
    """A function."""
    local_name = 5               # a name inside a body is not top level
    return None


async def public_coroutine() -> None:
    """An async function, which this package has none of."""


def _private_function() -> None:
    """Underscore again."""


class PublicClass:
    """A class."""

    attribute = 6                # a class attribute is not a module name


class _PrivateClass:
    """And a private one."""
'''


def test_the_probe_reads_each_kind_of_definition_and_skips_each_non_definition() -> None:
    """The probe, against source written for it.

    The package exercises only three of this function's branches: plain
    functions, classes and plain assignments. Nothing in ``src/tenor`` declares
    an ``async def`` or an annotated assignment at the top level, and nothing
    assigns two names in one statement, so those paths were carried by reading
    rather than by running. Here they run.

    The exclusions matter as much as the inclusions. A probe that counted
    imports would report every module as failing to declare ``math``; one that
    counted names inside function bodies would report ``local_name``; one that
    missed ``__all__`` itself would have every module failing to declare its own
    declaration.
    """
    found = public_names_in(PROBE_SOURCE)
    assert found == {
        "PUBLIC_CONSTANT",
        "ANNOTATED",
        "PAIR_ONE",
        "PAIR_TWO",
        "public_function",
        "public_coroutine",
        "PublicClass",
    }, f"the probe read {sorted(found)}"

    # Said the other way round, so that a future widening has to break one of
    # these deliberately rather than by accident.
    for absent in ("math", "Path", "_PRIVATE", "__all__", "local_name",
                   "_private_function", "_PrivateClass", "attribute"):
        assert absent not in found, f"the probe counted {absent!r} as a definition"
