# SPDX-License-Identifier: GPL-3.0-only
"""`pytest --runtime <module>`: load a bot runtime before the suite.

HexSet ships no playing bots, and hexn names no runtime of its own
(`hexn.runtime`), so the tests that seat `heximax` or `rehex` need the
module that registers them named here, as a run names it with
`--runtime`. Without one those tests skip (`_bots.needs`); the rest of the
suite runs on the engine alone.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--runtime",
        action="append",
        default=[],
        metavar="MODULE",
        help="a bot runtime to import before collecting the tests "
        "(hexset.arena.load_runtime), repeatable; tests that seat a bot "
        "it registers skip without it",
    )


def pytest_configure(config):
    from hexset.arena import load_runtime

    load_runtime(*config.getoption("runtime"))
