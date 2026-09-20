# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import pytest

from hexn.benchmarks.head_shape import load_corpus, split


def test_the_holdout_is_taken_by_game_so_no_game_reaches_both_sides():
    corpus = list(range(128))
    fit, held = split(corpus, 0.2, random.Random(1))
    assert len(held) == 26
    assert len(fit) + len(held) == len(corpus)
    assert not set(fit) & set(held)


def test_a_holdout_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        split(list(range(10)), 1.0, random.Random(0))


def test_a_corpus_that_is_not_episodes_is_refused(tmp_path):
    path = tmp_path / "not.corpus"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_corpus(str(path))
