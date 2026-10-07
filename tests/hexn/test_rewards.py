# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexn.rewards import relative_points, reward, win_loss
from hexn.selfplay import Outcome
from hexset.victory import WINNING_POINTS


def an_outcome(points, winner=None, truncated=False):
    return Outcome(
        winner=winner,
        points=tuple(points),
        turns=50,
        actions=500,
        truncated=truncated,
    )


@pytest.mark.parametrize("seats", [2, 4])
def test_the_reward_is_zero_sum(seats):
    rng = random.Random(seats)
    for _ in range(50):
        points = [rng.randrange(0, 11) for _ in range(seats)]
        assert sum(relative_points(points)) == pytest.approx(0.0, abs=1e-9)


def test_lifting_every_seat_equally_earns_nothing():
    """The whole reason for reading points relatively rather than absolutely."""
    before = relative_points((7, 4, 3, 2))
    after = relative_points((9, 6, 5, 4))
    assert after == pytest.approx(before)


def test_winning_pays_more_than_losing_closely():
    won, second, *_ = relative_points((10, 9, 3, 2))
    assert won > second
    close = relative_points((10, 9, 8, 7))[0]
    runaway = relative_points((10, 2, 1, 0))[0]
    assert runaway > close, "a wider margin should be worth more"


def test_a_winner_lands_inside_the_unit_range():
    """The scale exists so a value head does not have to learn the units."""
    best = relative_points((WINNING_POINTS, 0, 0, 0))
    assert 0.0 < max(best) <= 1.0
    assert -1.0 <= min(best) < 0.0


def test_win_loss_pays_only_the_winner():
    assert win_loss(an_outcome((10, 5, 4, 3), winner=0)) == (1.0, 0.0, 0.0, 0.0)
    assert win_loss(an_outcome((8, 5, 4, 3))) == (0.0, 0.0, 0.0, 0.0)


def test_a_truncated_game_is_still_scored():
    """Zeroing it would teach a policy that stalling escapes a loss."""
    losing = an_outcome((2, 9, 8, 7), truncated=True)
    assert reward(losing)[0] < 0.0
