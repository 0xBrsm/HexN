# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

from hexn.expert import SearchPolicy, Target
from hexn.selfplay import Collector, Request
from hexset.mcts import Search

from _mcts_fixtures import Stub, a_game


def a_policy(stub=None, *, simulations=16, wave=4, temperature=1.0, seed=0):
    search = Search(
        stub or Stub(), simulations=simulations, wave=wave, rng=random.Random(seed)
    )
    return SearchPolicy(search, temperature=temperature, rng=random.Random(seed))


def a_collector(policy, **kwargs):
    return Collector(policy, lanes=1, seed=7, players=4, **kwargs)


def a_request(policy: SearchPolicy, game, options=None) -> Request:
    """What a collector would ask, without running one."""
    return Request(
        lane=0,
        seat=0,
        observation=None,
        mask=np.zeros(0, dtype=bool),
        options=policy.search._options(game) if options is None else options,
        game=game,
    )


def test_searches_from_several_lanes_share_one_root_evaluation():
    stub = Stub()
    collector = Collector(a_policy(stub), lanes=4, seed=7, players=4)
    collector.tick()
    assert collector.steps == 4
    assert len(stub.waves[0]) == 4


def test_the_visit_counts_ride_along_on_the_transition():
    stub = Stub()
    collector = a_collector(a_policy(stub, simulations=24), action_cap=12)
    episode = collector.collect(1)[0]
    filed = episode.stream()
    assert filed
    for transition in filed:
        target = transition.aux
        assert isinstance(target, Target)
        assert len(target.options) == len(target.visits)
        # A forced move is not searched, so its counts are a single 1 rather
        # than the budget; anything with a real choice spends the whole budget.
        assert target.visits.sum() in (1.0, 24.0)


def test_the_recorded_value_is_what_the_search_concluded_not_what_it_started_from():
    class First(Stub):
        """Only the very first leaf — the root — is worth anything."""

        def evaluate(self, leaves):
            out = super().evaluate(leaves)
            if len(self.waves) == 1:
                return [(prior, (1.0, 0.0, 0.0, 0.0)) for prior, _ in out]
            return out

    policy = a_policy(First(), simulations=16)
    game = a_game()
    request = a_request(policy, game)
    choice = policy.act([request])[0]
    # `root.value` is (1, 0, 0, 0) and every descendant is zero, so reporting
    # the root's own estimate rather than the backed-up mean is visible here.
    assert choice.value == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_every_option_carries_the_mean_the_search_ranked_it_on():
    policy = a_policy(Stub(value=(0.4, -0.1, -0.1, -0.2)), simulations=32)
    game = a_game()
    root, options, visits = policy.search.run(game)
    target = policy._choice(a_request(policy, game, options), (root, options, visits)).aux
    assert target.values is not None
    # The arithmetic `_select` does: an unvisited edge scores 0, a visited one
    # its stance-ranked total over its count.
    expected = np.where(visits > 0, root.ranked / np.maximum(visits, 1.0), 0.0)
    assert np.allclose(target.values, expected)
    assert target.values[int(np.argmax(visits))] != 0.0


def test_a_forced_move_reports_the_evaluators_value_and_every_visit():
    """HexSet evaluates a root with one legal move rather than skipping it:
    one evaluation, its only edge credited with the whole budget at that
    value. So a forced move records a real estimate, not the empty "none"
    a scripted policy records."""
    class OneWay(Search):
        def _options(self, game):
            return super()._options(game)[:1]

    stub = Stub(value=(0.4, 0.3, 0.2, 0.1))
    policy = SearchPolicy(
        OneWay(stub, simulations=16, rng=random.Random(0)), rng=random.Random(0)
    )
    game = a_game()
    choice = policy.act([a_request(policy, game)])[0]
    assert choice.value == pytest.approx(stub.value)
    assert choice.log_prob == 0.0
    assert stub.leaves == 1
    assert list(choice.aux.visits) == [16]
    assert list(choice.aux.prior) == [1.0]


def test_a_search_that_roots_on_different_options_is_refused():
    # The collector's mask is built from its own enumeration, so a search that
    # disagrees would file actions the mask calls illegal.
    policy = a_policy()
    game = a_game()
    short = policy.search._options(game)[:-1]
    with pytest.raises(ValueError, match="where the collector offered"):
        policy.act([a_request(policy, game, options=short)])


def test_temperature_zero_plays_the_most_visited_action():
    policy = a_policy(Stub(favour=0), simulations=32, temperature=0.0)
    game = a_game()
    choice = policy.act([a_request(policy, game)])[0]
    target = choice.aux
    assert choice.action == target.options[int(np.argmax(target.visits))]
    assert choice.log_prob == pytest.approx(0.0)


def test_temperature_one_does_not_always_play_the_most_visited_action():
    # The corpus is the point: four searches that all take the argmax replay
    # one game, and the target then only ever covers the best line.
    game = a_game()
    played = set()
    for seed in range(12):
        policy = a_policy(simulations=32, seed=seed)
        played.add(policy.act([a_request(policy, game)])[0].action)
    assert len(played) > 1
