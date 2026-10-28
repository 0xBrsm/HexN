# SPDX-License-Identifier: GPL-3.0-only
"""HexN's benchmark instruments. Every game one plays is kept: `add_games_dir`
gives its command line `--games-dir`, and `games_kept` is where a collector
writes each game as it finishes (`hexn.durable.Partial`: one game file per
game, its `hexset.record.Record` inside)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


def add_games_dir(parser) -> None:
    """`--games-dir` on a benchmark's command line."""
    parser.add_argument(
        "--games-dir", default=None,
        help="where every game played is kept (default: runs/benchmarks/<benchmark>/<UTC time>)",
    )


def games_kept(directory: str | None, benchmark: str, part: str = "games"):
    """The collection `part` of a benchmark's games go to: `<directory>/<part>`,
    `directory` defaulting to `runs/benchmarks/<benchmark>/<UTC time>`."""
    from ..durable import Partial

    root = Path(directory) if directory else (
        Path("runs/benchmarks") / benchmark / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    where = root / part
    print(f"keeping every game in {where}", file=sys.stderr)
    return Partial(where)
