"""A one-factor calculation refuses a two-factor model, and the reverse.

``HullWhite`` and ``G2`` are both exported from the root and both carry an
``a`` and a ``sigma``. Before these checks, handing a ``G2`` to a Hull-White
function did one of three things depending on which attributes the function
happened to read first: answered as Hull-White with the second factor dropped
(``risk_neutral_drift``, ``short_rate_level``, ``build_tree``), failed on an
attribute the ``G2`` lacks, or failed with ``'float' object is not callable``
because ``G2.b`` is a field where ``HullWhite.b`` is a method.

The functions are found by signature, not listed: every public function in the
package whose ``model`` parameter is annotated with one of the two classes. A
new one is checked as soon as it exists, and one that forgets the guard fails
here rather than in somebody's pricing.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from datetime import date
from types import ModuleType

import pytest

import tenor
from tenor import G2, BadModel, Basis, HullWhite, flat_curve
from tenor.bermudan import build_tree, short_rate_level
from tenor.convexity import risk_neutral_drift

ONE_FACTOR = HullWhite(a=0.05, sigma=0.01)
TWO_FACTOR = G2(a=0.05, sigma=0.01, b=0.5, eta=0.008, rho=-0.7)

#: The annotation each kind of function carries, and the model it must refuse.
WRONG = {"HullWhite": TWO_FACTOR, "G2": ONE_FACTOR}


def modules() -> list[ModuleType]:
    return [
        importlib.import_module(f"tenor.{info.name}")
        for info in pkgutil.iter_modules(tenor.__path__)
    ]


def model_annotation(function: Callable[..., object]) -> str | None:
    """The class named on ``model``, read as text since annotations are deferred."""
    parameter = inspect.signature(function).parameters.get("model")
    if parameter is None:
        return None
    annotation = parameter.annotation
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)
    # An unannotated parameter reads as ``inspect.Parameter.empty``, whose
    # ``__name__`` is ``"_empty"``; only the two model classes count.
    return text if text in WRONG else None


def model_functions() -> list[tuple[str, Callable[..., object], str]]:
    found = []
    for module in modules():
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                kind = model_annotation(obj)
                if kind is not None:
                    found.append((f"{module.__name__}.{name}", obj, kind))
    return sorted(found, key=lambda item: item[0])


FUNCTIONS = model_functions()


def call_with(function: Callable[..., object], model: object) -> object:
    """``model`` in its place and ``None`` for every other required argument.

    ``None`` is deliberate. The guard must run before anything else is read, and
    any argument read first fails on ``None`` with an error that is not
    :class:`BadModel`, so this call also proves the ordering.
    """
    arguments: dict[str, object] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "model":
            arguments[name] = model
        elif parameter.default is inspect.Parameter.empty:
            arguments[name] = None
    return function(**arguments)


def test_the_discovery_finds_both_kinds() -> None:
    """Twenty-two today; a probe that found nothing would pass everything below."""
    kinds = [kind for _, _, kind in FUNCTIONS]
    assert set(kinds) == {"HullWhite", "G2"}
    assert kinds.count("HullWhite") >= 15
    assert kinds.count("G2") >= 7


def test_the_discovery_reads_unknown_annotations_as_absent() -> None:
    def untyped(model, time):  # type: ignore[no-untyped-def]
        return None

    def no_model(curve: object) -> None:
        return None

    assert model_annotation(untyped) is None
    assert model_annotation(no_model) is None
    assert model_annotation(call_with) is None


@pytest.mark.parametrize(
    "name, function, kind", FUNCTIONS, ids=[name for name, _, _ in FUNCTIONS]
)
def test_the_other_model_is_refused_before_anything_is_read(
    name: str, function: Callable[..., object], kind: str
) -> None:
    wrong = WRONG[kind]
    short = name.rsplit(".", 1)[1]
    with pytest.raises(BadModel, match=rf"^{short} is a .* calculation and was given a "
                       rf"{type(wrong).__name__};"):
        call_with(function, wrong)


@pytest.mark.parametrize("model", [None, 0.05, (0.05, 0.01)], ids=["none", "float", "tuple"])
def test_something_that_is_not_a_model_at_all(model: object) -> None:
    with pytest.raises(BadModel, match=f"was given a {type(model).__name__};"):
        risk_neutral_drift(model, 1.0)  # type: ignore[arg-type]


REFERENCE = date(2026, 1, 15)
CURVE = flat_curve(REFERENCE, date(2046, 1, 15), 0.03, basis=Basis.ACT_365F)


@pytest.mark.parametrize(
    "calculation",
    [
        lambda model: risk_neutral_drift(model, 2.0),
        lambda model: short_rate_level(CURVE, model, 2.0),
        lambda model: build_tree(CURVE, model, (1.0, 2.0), 12),
    ],
    ids=["risk_neutral_drift", "short_rate_level", "build_tree"],
)
def test_the_three_that_used_to_answer(calculation: Callable[[object], object]) -> None:
    """With real arguments: these returned the Hull-White figure for a G2."""
    calculation(ONE_FACTOR)
    with pytest.raises(BadModel, match="was given a G2"):
        calculation(TWO_FACTOR)
