# SPDX-License-Identifier: GPL-3.0-only
"""The engine pin in `pyproject.toml`'s git-URL dependency must name the same
commit as the `hexset/` submodule checked out alongside it, in the private
development repository this package is cut from.

Two independent copies of "which HexSet commit does hexn train against"
exist on purpose -- the submodule gitlink is what the development checkout
resolves locally, the git URL is what a consumer building from the published
package installs -- and nothing enforces they stay in sync except this test.
Reads the submodule's recorded commit straight from git (`git ls-tree HEAD --
hexset`) rather than the submodule's own checkout, so it still catches a
drift even if the submodule working tree hasn't been `git submodule
update`d to match the gitlink yet.

Locates the enclosing git repository from this file's own location rather
than assuming a fixed number of parent directories, because this file lives
at a different depth depending on which tree it is running in: under
`src/tests/hexn/` in the private development repository, and under
`tests/hexn/` in the published package, where `src/` is projected to the
repository root. Skipped cleanly whenever that repository has no `hexset`
gitlink at its HEAD -- the published package never has one, and neither
does an installed distribution or a source tarball with no `.git` at all;
the dependency URL is the only pin that matters in any of those cases.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - requires-python is >=3.11
    tomllib = None

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

DEPENDENCY_SHA_RE = re.compile(
    r"hexset @ git\+https://github\.com/0xBrsm/HexSet\.git@([0-9a-f]{40})(?:#subdirectory=\S+)?"
)


def _pyproject_pin() -> str:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    for dep in deps:
        match = DEPENDENCY_SHA_RE.search(dep)
        if match:
            return match.group(1)
    pytest.fail(f"no hexset git-URL dependency found in {PYPROJECT}")


def _repo_root() -> Path | None:
    """The git repository containing this file, or None where there isn't one
    (an installed distribution, a source tarball)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out) if out else None


def _submodule_commit() -> str | None:
    """The commit HEAD's tree records for a `hexset/` gitlink, or None if
    this file isn't inside a git checkout, or that checkout's HEAD has no
    `hexset` gitlink (the published package, which never has one)."""
    repo_root = _repo_root()
    if repo_root is None:
        return None
    try:
        out = subprocess.run(
            ["git", "ls-tree", "HEAD", "--", "hexset"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out:
        return None
    # "160000 commit <sha>\thexset"
    fields = out.split()
    if len(fields) < 3 or fields[1] != "commit":
        return None
    return fields[2]


def test_pyproject_pin_matches_submodule_commit() -> None:
    submodule_commit = _submodule_commit()
    if submodule_commit is None:
        pytest.skip("no .git / no hexset gitlink at HEAD -- not a development checkout with the submodule")

    pyproject_sha = _pyproject_pin()
    assert pyproject_sha == submodule_commit, (
        f"src/pyproject.toml pins hexset @ {pyproject_sha}, but the hexset/ "
        f"submodule gitlink at HEAD is {submodule_commit} -- repin both to "
        "the same commit."
    )
