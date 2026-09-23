# SPDX-License-Identifier: GPL-3.0-only
"""One distribution, one `pyproject.toml`, one version. Read with `tomllib`
rather than by import, so stale dist-info cannot mask a drift.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import hexn

ROOT = Path(__file__).resolve().parents[2]


def test_hexn_dunder_version_matches_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        assert hexn.__version__ == tomllib.load(f)["project"]["version"]
