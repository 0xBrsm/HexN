# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from hexset.board.terrain import NUM_RESOURCES
from hexset.encoding import encode
from hexset.game import to_move
from hexset.record import replay_to
from hexn.replay import replay_to_ply
from hexn.selfplay import Collector, RandomPolicy


def a_trade_free_episode(seed: int = 11, action_cap: int = 400):
    """A collector with no `trader` hook seats every gate `None`
    (`LaneEnv` reads it as "this seat never trades"), so nothing here ever
    trades -- the same premise
    `test_selfplay.py`'s own `replay()` helper rests on."""
    collector = Collector(
        RandomPolicy(random.Random(seed)), lanes=1, seed=seed, action_cap=action_cap
    )
    return collector.collect(1)[0]


def test_replay_to_ply_matches_the_stored_observation_at_every_step():
    """The bit-exactness check this module exists for: rebuild the game at
    each recorded ply and re-encode it, and it must equal what was actually
    stored -- not merely equivalent to it."""
    episode = a_trade_free_episode()
    stream = episode.stream()
    checked = 0
    for step, transition in enumerate(stream):
        game = replay_to_ply(episode, step)
        assert to_move(game) == transition.seat
        expected = encode(game, transition.seat)
        assert np.array_equal(expected.hexes, transition.observation.hexes)
        assert np.array_equal(expected.vertices, transition.observation.vertices)
        assert np.array_equal(expected.edges, transition.observation.edges)
        assert np.array_equal(expected.globals, transition.observation.globals)
        checked += 1
    assert checked > 0


def test_replay_to_ply_ply_zero_is_the_untouched_start():
    episode = a_trade_free_episode(seed=17)
    game = replay_to_ply(episode, 0)
    assert game.turns == 0
    assert to_move(game) == episode.stream()[0].seat


def test_replay_to_ply_is_the_engines_own_replay_and_nothing_added():
    """The whole claim of this module now: `hexn.replay.replay_to_ply` is
    `hexset.record.replay_to` called on a collected episode's own `record`,
    so at any ply the two cannot disagree -- a second derivation of one
    replay path is exactly what used to drift."""
    episode = a_trade_free_episode(seed=23)
    stream = episode.stream()
    for ply in (0, len(stream) // 2, len(stream)):
        # Not `==`: `Game` carries a `random.Random`/`Chance` that two
        # independently replayed games never share by identity, whatever
        # position they hold. The encoding is the position.
        mine = encode(replay_to_ply(episode, ply), 0)
        theirs = encode(replay_to(episode.record, ply), 0)
        assert np.array_equal(mine.hexes, theirs.hexes)
        assert np.array_equal(mine.vertices, theirs.vertices)
        assert np.array_equal(mine.edges, theirs.edges)
        assert np.array_equal(mine.globals, theirs.globals)


def test_replay_to_ply_refuses_a_ply_past_the_recorded_stream():
    episode = a_trade_free_episode(seed=5, action_cap=20)
    with pytest.raises(ValueError, match="more"):
        replay_to_ply(episode, len(episode.stream()) + 1)


def test_replay_to_ply_refuses_an_episode_with_an_opponent_seat():
    """`Episode.stream`'s own docstring: a cast episode "has gaps and cannot
    replay the game" once an opponent seat's trajectory was never recorded.
    Reconstructing from a stream with missing actions would silently build
    the wrong position, so this must be a loud refusal, not a guess."""
    collector = Collector(
        RandomPolicy(random.Random(1)),
        lanes=1,
        seed=1,
        action_cap=200,
        opponents=[RandomPolicy(random.Random(2))],
        caster=lambda index: (0, 1, 0, 1),
    )
    episode = collector.collect(1)[0]
    with pytest.raises(ValueError, match="opponent"):
        replay_to_ply(episode, 0)


def test_replay_to_ply_applies_a_hand_built_trade():
    """Hand-built rather than bot-driven: contract 6 raised the trade floor
    enough that self-play games trade rarely (`CHANGELOG.md`), so waiting for
    a scripted bot to produce a real one is slow and unreliable where a
    manually composed `Trade` is neither -- the convention's own rule:
    the cheapest position with the property under test, never a whole
    game played out by a real bot.
    """
    episode = a_trade_free_episode(seed=3)
    stream = episode.stream()
    step = len(stream) // 2

    hands_before = replay_to_ply(episode, step).state(0, hidden=False).hands
    a = 0
    b = next(
        seat
        for seat in range(episode.players)
        if seat != a and sum(hands_before[seat]) > 0
    )
    resource = next(i for i in range(NUM_RESOURCES) if hands_before[b][i] > 0)
    received = tuple(1 if i == resource else 0 for i in range(NUM_RESOURCES))

    traded = dataclasses.replace(
        episode,
        record=dataclasses.replace(
            episode.record, trades=((step, a, b, received),)
        ),
    )

    plain_after = replay_to_ply(episode, step + 1).state(0, hidden=False)
    traded_after = replay_to_ply(traded, step + 1).state(0, hidden=False)

    expected_a = list(plain_after.hands[a])
    expected_a[resource] += 1
    expected_b = list(plain_after.hands[b])
    expected_b[resource] -= 1

    assert list(traded_after.hands[a]) == expected_a
    assert list(traded_after.hands[b]) == expected_b
    # Nothing but the two hands moved: the trade rides on top of the
    # ordinary action replay rather than displacing it.
    assert to_move(replay_to_ply(traded, step + 1)) == to_move(
        replay_to_ply(episode, step + 1)
    )
