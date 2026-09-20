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

__version__ = "0.1.0"
