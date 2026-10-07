# SPDX-License-Identifier: GPL-3.0-only
"""A bot runtime for the tests: `random-too`, a second name for HexSet's
own `random` bot.

The tests that need two scripted entrants by name -- a table pool, two
ladder rungs -- name `random` and this. Loading it the way a run loads its
bots (`--runtime`, `WorkerSpec.runtime`) also exercises the path: a spawned
collector worker resolves `random-too` only if it loaded this module itself.
"""

from __future__ import annotations

from hexset.arena import Entrant, register_preset

NAME = "random-too"

register_preset(NAME, Entrant(NAME, kind="random"))
