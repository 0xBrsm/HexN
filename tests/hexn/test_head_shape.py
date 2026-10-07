# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from hexn.benchmarks.head_shape import split


def test_the_holdout_is_taken_by_game_so_no_game_reaches_both_sides():
    corpus = list(range(128))
    fit, held = split(corpus, 0.2, random.Random(1))
    assert len(held) == 26
    assert len(fit) + len(held) == len(corpus)
    assert not set(fit) & set(held)
