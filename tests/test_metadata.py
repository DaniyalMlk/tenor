"""Checks on the distribution metadata rather than on the mathematics.

A library nobody can install is not useful, and the failures that stop an
install are not the kind unit tests catch: a version written in two files that
drift apart, a licence claimed in metadata but absent from the tree, a typing
marker that exists in the source and never reaches the wheel. Each of these is
invisible until someone tries to depend on the package, which is the worst
moment to find out.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import metadata, version
from pathlib import Path

import pytest

import tenor

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_exported() -> None:
    """The package says what it is without an installed-metadata lookup."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", tenor.__version__)


def test_exported_version_matches_the_installed_distribution() -> None:
    """The attribute and the installed metadata are the same string.

    These are written in two different files. Bumping one and forgetting the
    other is the ordinary mistake, and it produces a release whose reported
    version is a lie.
    """
    assert tenor.__version__ == version("tenor")


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_exported_version_matches_the_source_of_truth() -> None:
    """The attribute agrees with pyproject.toml in the checkout.

    The check above compares against what is *installed*, which in an editable
    install is a snapshot taken at install time. That snapshot can be stale, so
    it would pass on a tree where pyproject.toml has already moved on. Reading
    the file is what makes the check bite without a reinstall.
    """
    import tomllib

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():  # installed without the source tree alongside
        pytest.skip("no checkout to compare against")

    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert tenor.__version__ == declared


def test_licence_text_is_present() -> None:
    """The licence the metadata claims exists as a file, and says MIT."""
    licence = ROOT / "LICENSE"
    if not licence.exists():
        pytest.skip("no checkout to compare against")
    assert "MIT License" in licence.read_text()


def test_licence_reaches_the_built_distribution() -> None:
    """The installed distribution carries the licence, not just the repository.

    `license-files` in pyproject.toml is what does this. Without it — and the
    legacy `license = {text = "MIT"}` table it replaced had no equivalent — the
    metadata goes on claiming MIT while the artefact ships no terms at all. That
    is the state every wheel this project built before now was in.
    """
    fields = metadata("tenor")
    assert fields["License-Expression"] == "MIT"
    assert "LICENSE" in fields.get_all("License-File", [])


def test_the_typing_marker_ships() -> None:
    """`Typing :: Typed` is only true if py.typed is inside the package.

    The classifier is a promise to type checkers. It is made in pyproject.toml
    and kept by a file in the package directory, and nothing connects the two
    except this test.

    The failure is quiet: in a source checkout a type checker reads the
    annotations from the files and everything looks fine, while a consumer who
    installed the package gets none of them.
    """
    assert (Path(tenor.__file__).parent / "py.typed").is_file()
    assert "Typing :: Typed" in metadata("tenor").get_all("Classifier", [])


def test_the_entry_point_reports_the_version() -> None:
    """`tenor --version` prints it, and exits zero doing so.

    argparse's version action raises SystemExit, so the assertion is on the
    exit code as much as on the text.
    """
    from tenor.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
