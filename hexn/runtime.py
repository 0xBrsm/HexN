# SPDX-License-Identifier: GPL-3.0-only
"""The bots a run names, loaded the way HexSet loads them.

HexSet ships no playing bots. A bot -- `heximax` or `rehex` as a ladder
rung, a `--mix` opponent or a `--trader` -- is registered with
`hexset.arena` by the runtime module that carries it, and a process can name
it only once that module is imported (`hexset.arena.load_runtime`). hexn
names no runtime of its own: every entry point that resolves a bot by name
takes a repeatable `--runtime <module>`, as HexSet's own commands do, loads
it before it resolves anything, and hands it to each collector worker
(`hexn.collect.WorkerSpec.runtime`), since a spawned process inherits no
registrations.

A trainer's `--runtime` is frozen into its config like every other
parameter, so the run's manifest records which runtime its names resolved
through; `hexn.run.engine_provenance` records the commit of each.
"""

from __future__ import annotations

import argparse
from typing import Sequence

# `hexset.arena` is imported inside each function, as everywhere hexn reaches
# it: `hexn.collect` imports this module, and the collector is imported where
# the arena's heavier entrant kinds are not wanted.


def add_argument(parser: argparse.ArgumentParser) -> None:
    """`--runtime <module>`, repeatable, onto `parser`; `args.runtime` is
    the list of modules, empty when none is named."""
    from hexset.arena import RUNTIME_HELP

    parser.add_argument(
        "--runtime", action="append", default=[], metavar="MODULE", help=RUNTIME_HELP
    )


def load(modules: Sequence[str]) -> tuple[str, ...]:
    """Import every module in `modules` for its registrations; the modules,
    as a tuple a picklable spec can carry. Idempotent: a module imports
    once."""
    from hexset.arena import load_runtime

    modules = tuple(modules or ())
    load_runtime(*modules)
    return modules
