# SPDX-License-Identifier: GPL-3.0-only
"""`pytest --runtime <module>`: load a bot runtime before the suite.

HexSet ships no playing bots, and hexn names no runtime of its own
(`hexn.runtime`). The tests that seat a bot by name seat `mybot`. A runtime
named here, as a run names one with `--runtime`, may provide it; without
one the suite registers its own (`_mybot`), a bot that moves at random and
trades by card scarcity, so those tests run on the engine alone.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--runtime",
        action="append",
        default=[],
        metavar="MODULE",
        help="a bot runtime to import before collecting the tests "
        "(hexset.arena.load_runtime), repeatable; a runtime that registers "
        "mybot replaces the suite's own",
    )


def pytest_configure(config):
    from hexset import seeds
    from hexset.arena import load_runtime

    load_runtime(*config.getoption("runtime"))
    # The tests that seat `mybot` play the suite's own when no runtime
    # registers one (`_mybot`), so they run without a runtime too.
    from hexset.arena import registered_presets

    if "mybot" not in registered_presets():
        import _mybot  # noqa: F401 -- registers `mybot` with hexset.arena
    # A suite's games are fixtures, not runs: no seed policy holds them
    # (`hexset.seeds`), whatever a runtime requires.
    seeds.suspend()
