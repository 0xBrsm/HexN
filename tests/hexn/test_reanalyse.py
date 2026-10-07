# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.game import start  # noqa: E402
from hexset.mcts import Search  # noqa: E402
from hexn.expert import SearchPolicy, Target  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing  # noqa: E402
from hexn.netbot import LeafEvaluator  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.reanalyse import reanalyse  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402
from hexn.store import EpisodeStore  # noqa: E402


def a_policy(seed: int = 0, width: int = 16) -> NetworkPolicy:
    """A real, tiny `NetworkPolicy` -- what a `LeafEvaluator` actually wraps
    and what `hexn.exit` searches over."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    space = space_for(game)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space, graph, 4, ModelConfig(width=width, rounds=1))
    return NetworkPolicy(net, space, packing(graph, 4))


def a_search(policy: NetworkPolicy, seed: int = 0, simulations: int = 8) -> Search:
    return Search(
        LeafEvaluator(policy=policy),
        simulations=simulations,
        wave=4,
        rng=random.Random(seed),
    )


def searched_episodes(policy, search, seed: int = 0):
    """One short searched game.

    Trade-free (`max_offers=0`): none of this file is about trading, and a
    network's gate pricing every coverable candidate after every MAIN action
    is the dominant cost of a searched game left unset. Same trade
    `test_exit.searched_episodes` makes, for the same reason. Capped short,
    because reanalysis replays and re-searches every position it samples.
    """
    expert = SearchPolicy(search, rng=random.Random(seed))
    return Collector(
        expert, lanes=1, seed=seed, action_cap=60, max_offers=0
    ).collect(1)


@pytest.fixture(scope="module")
def corpus():
    """Three iterations' worth of searched games, collected once.

    Episodes are frozen and a store writes its own copy, so every test fills
    a fresh store from these and reanalyses that.
    """
    policy = a_policy()
    search = a_search(policy)
    return [searched_episodes(policy, search, seed=iteration) for iteration in range(3)]


def a_filled_store(tmp_path, corpus, iterations: int = 2):
    store = EpisodeStore(tmp_path / "replay", capacity=10**9)
    for iteration in range(iterations):
        store.append(iteration, corpus[iteration])
    return store


def all_transitions(store):
    return [
        transition
        for iteration in store.iterations()
        for episode in store.shard_episodes(iteration)
        for trajectory in episode.trajectories
        for transition in trajectory
    ]


def test_reanalyse_replaces_every_sampled_targets_visits(tmp_path, corpus):
    """No `--reanalyse-samples` cap here (`samples=len(before)`): the point
    is that a sampled position's stale `Target` is gone, replaced by a fresh
    one out of a fresh root search over the current net."""
    search = a_search(a_policy(seed=1), seed=1)
    store = a_filled_store(tmp_path, corpus, iterations=2)
    before = [t.aux for t in all_transitions(store)]

    stats = reanalyse(store, search, samples=len(before), rng=random.Random(0))

    after = [t.aux for t in all_transitions(store)]
    assert stats.reanalysed == len(before)
    assert all(isinstance(target, Target) for target in after)
    # Every sampled row got a fresh search, not the one recorded at
    # collection time -- a distinct array object, whatever its numbers.
    assert all(a.visits is not b.visits for a, b in zip(before, after))


def test_reanalyse_draws_from_every_retained_iteration_not_only_the_newest(
    tmp_path, corpus
):
    search = a_search(a_policy(seed=2), seed=2)
    store = a_filled_store(tmp_path, corpus, iterations=3)
    before_by_iteration = {
        iteration: [t.aux for e in store.shard_episodes(iteration)
                    for traj in e.trajectories for t in traj]
        for iteration in store.iterations()
    }

    reanalyse(store, search, samples=10**6, rng=random.Random(1))

    for iteration in store.iterations():
        after = [
            t.aux
            for e in store.shard_episodes(iteration)
            for traj in e.trajectories
            for t in traj
        ]
        before = before_by_iteration[iteration]
        assert any(a.visits is not b.visits for a, b in zip(before, after)), (
            f"iteration {iteration} was never sampled"
        )


def test_reanalyse_reports_how_many_positions_and_a_finite_mean_kl(tmp_path, corpus):
    search = a_search(a_policy(seed=3), seed=3)
    store = a_filled_store(tmp_path, corpus, iterations=1)
    total = len(all_transitions(store))

    stats = reanalyse(store, search, samples=total // 2, rng=random.Random(2))

    assert stats.reanalysed == total // 2
    assert stats.mean_kl >= 0.0
    assert stats.mean_kl < float("inf")


def test_reanalyse_never_touches_the_recorded_value_estimate(tmp_path, corpus):
    """The value target is the terminal outcome
    (`hexn.exit._value_targets`), not the search's own backed-up
    estimate -- reanalysis rewrites `Transition.aux`, never `.value`."""
    search = a_search(a_policy(seed=4), seed=4)
    store = a_filled_store(tmp_path, corpus, iterations=1)
    before = [t.value for t in all_transitions(store)]

    reanalyse(store, search, samples=10**6, rng=random.Random(3))

    after = [t.value for t in all_transitions(store)]
    assert after == before


def test_reanalyse_with_zero_samples_or_an_empty_store_is_a_no_op(tmp_path, corpus):
    search = a_search(a_policy(seed=5), seed=5)
    store = a_filled_store(tmp_path, corpus, iterations=1)

    stats = reanalyse(store, search, samples=0, rng=random.Random(0))
    assert stats == type(stats)(reanalysed=0, mean_kl=0.0)

    empty = EpisodeStore(tmp_path / "empty", capacity=10**9)
    stats = reanalyse(empty, search, samples=10, rng=random.Random(0))
    assert stats.reanalysed == 0
