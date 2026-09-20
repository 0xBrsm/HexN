# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexn.selfplay import Collector, RandomPolicy
from hexn.store import EpisodeStore, shard_path


def collected(seed: int, games: int = 2, action_cap: int = 300):
    """Real episodes (real `Observation`s, real actions) rather than
    fabricated ones -- `Collector` with a torch-free policy is already the
    cheapest way to get them, and the store must not care what produced its
    positions."""
    return Collector(
        RandomPolicy(random.Random(seed)), lanes=2, seed=seed, action_cap=action_cap
    ).collect(games)


def positions(episodes) -> int:
    return sum(len(episode) for episode in episodes)


def fingerprint(episodes):
    """A cheap stand-in for episode equality: `Episode`/`Transition` carry
    numpy arrays, and the dataclass-generated `==` on those raises rather
    than compares (ambiguous truth value), so pin the scalar facts instead.

    `record` is included: it is a plain dataclass of tuples (no numpy), so
    it compares and pickles fine, and a shard round trip that dropped it
    silently is exactly the kind of thing this fingerprint exists to catch.
    """
    return [
        (e.index, e.seed, e.players, len(e), e.outcome, e.cast, e.trades, e.record)
        for e in episodes
    ]


def test_a_store_needs_room_for_at_least_one_position(tmp_path):
    with pytest.raises(ValueError):
        EpisodeStore(tmp_path / "replay", capacity=0)


def test_the_freshest_iteration_is_retained_even_past_the_budget(tmp_path):
    """An iteration that just cost real search time to collect must be
    trainable on its own -- capacity 1 is smaller than any real iteration."""
    episodes = collected(seed=1, games=2)
    store = EpisodeStore(tmp_path / "replay", capacity=1)
    store.append(0, episodes)

    assert store.iterations() == [0]
    assert store.positions() == positions(episodes)
    assert fingerprint(store.episodes()) == fingerprint(episodes)


def test_older_iterations_are_evicted_once_the_budget_is_covered(tmp_path):
    directory = tmp_path / "replay"
    first = collected(seed=2, games=2)
    store = EpisodeStore(directory, capacity=positions(first))
    store.append(0, first)
    assert store.iterations() == [0]

    second = collected(seed=3, games=2)
    store.append(1, second)

    # The newest iteration is never evicted on its own arrival; the eviction
    # loop stops the moment only one shard is left, so with two appends the
    # oldest is exactly what goes.
    assert store.iterations() == [1]
    assert store.positions() == positions(second)
    assert fingerprint(store.episodes()) == fingerprint(second)


def test_evicting_an_iteration_deletes_its_shard_file(tmp_path):
    directory = tmp_path / "replay"
    first = collected(seed=4, games=2)
    store = EpisodeStore(directory, capacity=positions(first))
    store.append(0, first)
    assert shard_path(directory, 0).exists()

    store.append(1, collected(seed=5, games=2))

    assert not shard_path(directory, 0).exists()
    assert shard_path(directory, 1).exists()


def test_a_budget_covering_several_iterations_keeps_all_of_them(tmp_path):
    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    episodes_by_iteration = {}
    for iteration, seed in enumerate((10, 11, 12)):
        episodes = collected(seed=seed, games=2)
        episodes_by_iteration[iteration] = episodes
        store.append(iteration, episodes)

    assert store.iterations() == [0, 1, 2]
    assert store.positions() == sum(
        positions(e) for e in episodes_by_iteration.values()
    )
    # Every position drawn for an update is drawn from this same set with no
    # further filtering -- "uniform over stored positions" is this: nothing
    # is weighted or dropped, and the freshest iteration (2) is in it.
    stored = fingerprint(store.episodes())
    for episodes in episodes_by_iteration.values():
        for row in fingerprint(episodes):
            assert row in stored


def test_resume_reloads_exactly_the_retained_window(tmp_path):
    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    for iteration, seed in enumerate((20, 21, 22)):
        store.append(iteration, collected(seed=seed, games=2))

    resumed = EpisodeStore.resume(directory, capacity=1_000_000)

    assert resumed.iterations() == store.iterations()
    assert resumed.positions() == store.positions()
    assert fingerprint(resumed.episodes()) == fingerprint(store.episodes())


def test_resume_respects_whatever_window_eviction_already_left_on_disk(tmp_path):
    directory = tmp_path / "replay"
    first = collected(seed=30, games=2)
    store = EpisodeStore(directory, capacity=positions(first))
    store.append(0, first)
    store.append(1, collected(seed=31, games=2))
    assert store.iterations() == [1]

    resumed = EpisodeStore.resume(directory, capacity=positions(first))

    assert resumed.iterations() == [1]


def test_resuming_an_empty_directory_is_an_empty_store(tmp_path):
    store = EpisodeStore.resume(tmp_path / "nothing-yet", capacity=100)

    assert store.iterations() == []
    assert store.positions() == 0
    assert store.episodes() == []


def test_replace_swaps_a_shards_episodes_and_rewrites_its_file(tmp_path):
    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    store.append(0, collected(seed=40, games=1))
    replacement = collected(seed=41, games=1)

    store.replace(0, replacement)

    assert fingerprint(store.shard_episodes(0)) == fingerprint(replacement)
    reloaded = EpisodeStore.resume(directory, capacity=1_000_000)
    assert fingerprint(reloaded.shard_episodes(0)) == fingerprint(replacement)


def test_replace_refuses_an_iteration_that_was_never_appended(tmp_path):
    store = EpisodeStore(tmp_path / "replay", capacity=100)
    with pytest.raises(KeyError):
        store.replace(0, [])
