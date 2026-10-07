# SPDX-License-Identifier: GPL-3.0-only
"""HexNet must never touch the engine's private state field.

`Game.state` is a method now (`game.state(seat, *, hidden=True)`), not a
field: reads go through it and the only sanctioned write is
`game.set_state(...)`. This guards against a regression back to the raw
`_state` field, which would bypass the engine's information-set boundary
silently. (An attribute-style `game.state = ...` shadows the method, so the
next `game.state(seat)` fails loudly on its own.)
"""
from __future__ import annotations

import re
from pathlib import Path

HEXN_SRC = Path(__file__).resolve().parents[2] / "hexn"

RAW_STATE = re.compile(r"\._state\b")


def _offenders(pattern: re.Pattern[str]) -> list[str]:
    hits = []
    for path in sorted(HEXN_SRC.rglob("*.py")):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(HEXN_SRC)}:{lineno}: {line.strip()}")
    return hits


def test_no_file_under_hexn_touches_the_raw_state_field():
    assert HEXN_SRC.is_dir()
    assert _offenders(RAW_STATE) == []
