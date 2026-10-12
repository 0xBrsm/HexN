# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

import numpy as np

from hexn.rewards import points_lost_next, points_next_round, relative_points, reward, win_loss
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


def _lost(steps, points, final):
    got = points_lost_next([np.asarray(s) for s in steps], [np.asarray(p) for p in points], final)
    return [list(map(float, g)) for g in got]


def test_a_points_drop_while_others_act_labels_the_whole_run_before_it():
    # seat 0 acts at steps 0 and 1 holding 4, seat 1 at 2 and takes an award, seat 0 next decides with 2
    steps = [[0, 1, 3], [2]]
    points = [[4, 4, 2], [5]]
    assert _lost(steps, points, (2, 7)) == [[1.0, 1.0, 0.0], [0.0]]


def test_a_gain_is_never_a_loss_and_the_last_run_reads_the_outcome():
    steps = [[0, 2], [1, 3]]
    points = [[3, 4], [5, 5]]
    # seat 0 gains between its runs; seat 1's last run is compared with the final points, where it lost 2
    assert _lost(steps, points, (4, 3)) == [[0.0, 0.0], [0.0, 1.0]]


def test_runs_are_cut_by_any_other_seats_decision_and_an_empty_seat_is_skipped():
    steps = [[0, 1, 4], [], [2, 3]]
    points = [[6, 6, 6], [], [2, 2]]
    assert _lost(steps, points, (6, 0, 2)) == [[0.0, 0.0, 0.0], [], [0.0, 0.0]]


def test_the_points_target_is_every_seats_change_until_the_decider_acts_again():
    # seat 0 decides at steps 0 and 1, seat 1 at 2 (and takes an award: +2 for it, -2 for seat 0), seat 0 at 3
    steps = [np.asarray([0, 1, 3]), np.asarray([2])]
    seat_points = [np.asarray([[4, 3], [5, 3], [3, 5]]), np.asarray([[5, 3]])]
    got = points_next_round(steps, seat_points, (3, 6))
    # seat 0's first run (steps 0-1) ends at its decision at step 3: from (4,3) and (5,3) to (3,5)
    assert got[0].tolist() == [[-1.0, 2.0], [-2.0, 2.0], [0.0, 1.0]]
    # seat 1's only run ends at the outcome (3, 6)
    assert got[1].tolist() == [[-2.0, 3.0]]
    assert got[0].dtype == np.float32
