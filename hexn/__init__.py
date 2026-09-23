# SPDX-License-Identifier: GPL-3.0-only
"""PPO/training research on top of the `hexset` engine.

Split out of `hexset` so the engine, bots, ledger and search stay
installable (and torch-free) on their own; `hexn` depends on `hexset`
and never the other way around. Importing `hexn.netbot` (directly, via
`hexn.ppo`/`hexn.league`/`hexn.collect`, or by naming it to a HexSet
CLI as `--runtime hexn.netbot`) registers the network-backed entrant
kinds with `hexset.arena`'s registry -- see
`hexset.clients.netbot.register_entrants`.
"""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

# `hexn` is one distribution (`../pyproject.toml`), and that file is the only
# place its version lives. Read it from the source tree first: an editable
# install's dist-info metadata is only regenerated on reinstall and goes stale
# silently. Fall back to installed metadata only for a wheel, which ships no
# `pyproject.toml`. This was a literal "0.1.0" through 0.23.4, which every
# release since 0.1.0 contradicted.
try:
    with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]
except (OSError, KeyError, tomllib.TOMLDecodeError):
    try:
        __version__ = metadata.version("hexn")
    except metadata.PackageNotFoundError:
        __version__ = "0+unknown"
