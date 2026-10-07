# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexn.selfplay import Collector, RandomPolicy
from hexn.store import EpisodeStore, shard_path


def collected(seed: int, games: int = 2, action_cap: int = 80):
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
    # Eviction deletes the shard, not just the entry.
    assert not shard_path(directory, 0).exists()

    resumed = EpisodeStore.resume(directory, capacity=positions(first))

    assert resumed.iterations() == [1]


def test_replace_swaps_a_shards_episodes_and_rewrites_its_file(tmp_path):
    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    store.append(0, collected(seed=40, games=1))
    replacement = collected(seed=41, games=1)

    store.replace(0, replacement)

    assert fingerprint(store.shard_episodes(0)) == fingerprint(replacement)
    reloaded = EpisodeStore.resume(directory, capacity=1_000_000)
    assert fingerprint(reloaded.shard_episodes(0)) == fingerprint(replacement)


def test_a_write_killed_before_its_rename_leaves_the_old_shard(tmp_path, monkeypatch):
    """Reanalysis rewrites old shards every iteration; a kill mid-rewrite must
    leave the shard it was replacing, not a truncated one the next resume
    cannot load."""
    from hexn import durable

    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    original = collected(seed=50, games=1)
    store.append(0, original)

    def killed(*args, **kwargs):
        raise KeyboardInterrupt("killed between the write and the rename")

    monkeypatch.setattr(durable.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        store.replace(0, collected(seed=51, games=1))
    monkeypatch.undo()

    reloaded = EpisodeStore.resume(directory, capacity=1_000_000)
    assert fingerprint(reloaded.shard_episodes(0)) == fingerprint(original)


def test_a_shard_written_ahead_of_its_checkpoint_is_adopted_not_recollected(tmp_path):
    """Collected, shard written, killed before the checkpoint: resuming at that
    iteration takes the shard as the iteration's collection, once."""
    directory = tmp_path / "replay"
    store = EpisodeStore(directory, capacity=1_000_000)
    store.append(0, collected(seed=60, games=1))
    ahead = collected(seed=61, games=1)
    store.append(1, ahead)
    written = shard_path(directory, 1).stat().st_mtime_ns

    resumed = EpisodeStore.resume(directory, capacity=1_000_000, before=1)
    assert resumed.iterations() == [0], "not trained on before its iteration"
    assert sorted(resumed.ahead) == [1]

    assert fingerprint(resumed.adopt(1)) == fingerprint(ahead)
    assert resumed.iterations() == [0, 1] and not resumed.ahead
    assert shard_path(directory, 1).stat().st_mtime_ns == written, "not rewritten"


def test_appending_an_iteration_twice_replaces_its_shard(tmp_path):
    store = EpisodeStore(tmp_path / "replay", capacity=1_000_000)
    store.append(0, collected(seed=70, games=1))
    again = collected(seed=71, games=1)
    store.append(0, again)

    assert store.iterations() == [0]
    assert fingerprint(store.episodes()) == fingerprint(again)
