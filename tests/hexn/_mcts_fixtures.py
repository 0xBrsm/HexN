# SPDX-License-Identifier: GPL-3.0-only
"""Search-test fixtures `test_expert.py` needs: `a_game`, `Stub`.

Copied from HexSet's `tests/test_mcts.py` (now a different repo) rather than
imported across the boundary -- a test file is not an installable package
`hexn`'s dependency on `hexset` can reach, the way `test_expert.py` used to
when both suites shared one `tests/` directory (`from test_mcts import ...`).
Keep this in sync with `test_mcts.py` by hand if either changes.

`test_expert.py` is the sole consumer and needs only the two fixtures below;
keep it that way rather than growing this module for one more caller.
"""

from __future__ import annotations

import random

import numpy as np

from hexset.board.board import random_base_board
from hexset.game import start
from hexset.mcts import Leaf


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


class Stub:
    """Uniform prior and a fixed value, remembering every wave it was handed."""

    def __init__(self, value=(0.0, 0.0, 0.0, 0.0), favour: int | None = None) -> None:
        self.value = value
        self.favour = favour
        self.waves: list[list[Leaf]] = []

    @property
    def leaves(self) -> int:
        return sum(len(wave) for wave in self.waves)

    def evaluate(self, leaves):
        self.waves.append(list(leaves))
        out = []
        for leaf in leaves:
            n = len(leaf.options)
            prior = np.full(n, 1.0 / n)
            if self.favour is not None and n > 1:
                prior = np.full(n, 0.01 / (n - 1))
                prior[self.favour % n] = 0.99
            out.append((prior, self.value))
        return out
