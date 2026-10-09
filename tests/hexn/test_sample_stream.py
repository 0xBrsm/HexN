# SPDX-License-Identifier: GPL-3.0-only
"""Per-game sample streams: a game is a pure function of its index and the
weights, however the games are sharded, batched or interrupted."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.collect import ParallelCollector, WorkerSpec, _build  # noqa: E402
from hexn.policy import keyed_gumbel  # noqa: E402
from hexn.selfplay import Collector, draw_key  # noqa: E402


def spec(worker: int, workers: int, **overrides) -> WorkerSpec:
    # Trade-free and capped short, as in test_collect: nothing here is about
    # trading or about games finishing, and a truncated game is a game.
    base = dict(
        seed=5,
        players=4,
        lanes=2,
        action_cap=200,
        max_offers=0,
        first_game=worker,
        stride=workers,
        width=8,
        rounds=1,
        torch_seed=1000 + worker,
        mix=(),
        parent="",
        sample_stream="game",
    )
    base.update(overrides)
    return WorkerSpec(**base)


@pytest.fixture(scope="module")
def weights():
    """One set of weights every shard below loads, as a sync would ship it:
    each worker otherwise initialises its net from its own torch seed."""
    policies, _ = _build(spec(0, 1, torch_seed=77))
    return {k: v.clone() for k, v in policies[0].net.state_dict().items()}


def shard(weights, worker: int, workers: int, **overrides) -> Collector:
    policies, collector = _build(spec(worker, workers, **overrides))
    policies[0].net.load_state_dict(weights)
    return collector


def played(weights, workers: int, lanes: int, games: int, **overrides) -> dict:
    """Index -> episode for `games` games dealt across `workers` in-process
    shards of `lanes` lanes each, built exactly as a worker builds them."""
    out = {}
    share, extra = divmod(games, workers)
    for worker in range(workers):
        collector = shard(weights, worker, workers, lanes=lanes, **overrides)
        for episode in collector.cohort(share + (1 if worker < extra else 0)):
            out[episode.index] = episode
    return out


def assert_same_games(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for index in a:
        assert a[index].record == b[index].record, f"game {index} played differently"
        mine = [t for trajectory in a[index].trajectories for t in trajectory]
        theirs = [t for trajectory in b[index].trajectories for t in trajectory]
        assert [t.index for t in mine] == [t.index for t in theirs]
        np.testing.assert_allclose(
            [t.log_prob for t in mine], [t.log_prob for t in theirs], rtol=0, atol=1e-5
        )


def test_a_draw_key_is_pure_in_each_part_and_never_shared():
    assert draw_key(5, 10, 2, 33) == draw_key(5, 10, 2, 33)
    keys = {
        draw_key(seed, index, seat, step)
        for seed in (0, 5)
        for index in range(20)
        for seat in range(4)
        for step in range(50)
    }
    assert len(keys) == 2 * 20 * 4 * 50
    assert all(0 <= key < 2**64 for key in keys)
    assert draw_key(-1, 0, 0, 0) != draw_key(0, 0, 0, 0)


def test_keyed_gumbel_draws_follow_the_distribution():
    log_p = np.log(np.array([0.5, 0.3, 0.15, 0.05, 0.0 + 1e-300]))
    keys = [draw_key(1, i, 0, 0) for i in range(40_000)]
    picks = (log_p[None, :] + keyed_gumbel(keys, len(log_p))).argmax(1)
    freq = np.bincount(picks, minlength=len(log_p)) / len(keys)
    np.testing.assert_allclose(freq, np.exp(log_p), atol=0.01)
    # A key is its own draw: the same key twice, the same noise.
    np.testing.assert_array_equal(keyed_gumbel(keys[:3], 7), keyed_gumbel(keys[:3], 7))


def test_a_keyed_act_answers_each_row_from_its_own_key(weights):
    collector = shard(weights, 0, 1, lanes=6, cohort=False)
    requests = collector.requests()
    assert len(requests) == 6 and all(r.draw is not None for r in requests)
    policy = collector.policy
    whole = policy.act(requests)
    # Reversed, and one row alone: the batch around a row changes nothing.
    backwards = policy.act(requests[::-1])[::-1]
    alone = [policy.act([request])[0] for request in requests]
    for a, b, c in zip(whole, backwards, alone):
        assert a.action == b.action == c.action
        assert a.log_prob == pytest.approx(c.log_prob, abs=1e-6)
    with pytest.raises(ValueError, match="mixes keyed and unkeyed"):
        from dataclasses import replace

        policy.act([requests[0], replace(requests[1], draw=None)])


def test_games_do_not_depend_on_the_worker_count_or_the_lanes(weights):
    one_lane = played(weights, workers=1, lanes=1, games=6)
    assert_same_games(one_lane, played(weights, workers=1, lanes=4, games=6))
    assert_same_games(one_lane, played(weights, workers=3, lanes=2, games=6))


def test_worker_streams_do_depend_on_the_lanes(weights):
    """The control: under the worker's own generator the same deals play out
    differently once the batching changes, which is what keyed draws fix."""
    a = played(weights, workers=1, lanes=1, games=6, sample_stream="worker")
    b = played(weights, workers=1, lanes=4, games=6, sample_stream="worker")
    assert any(a[i].record != b[i].record for i in a)


def test_a_searched_worker_refuses_keyed_streams():
    with pytest.raises(ValueError, match="plain policy only"):
        _build(spec(0, 1, simulations=4))


def test_a_collection_killed_midway_replays_exactly_on_another_worker_count(
    tmp_path, monkeypatch, weights
):
    """Kill a worker mid-cohort, resume on a different worker count: every
    game -- kept before the kill or replayed after it -- is the game an
    uninterrupted single-shard collection plays."""
    import os
    import signal
    import time

    from hexn import collect, durable

    # Enough games that the kill lands mid-cohort: one lane a worker, twelve
    # games each, and the kill as soon as the first one is kept.
    games = 24
    reference = played(weights, workers=1, lanes=3, games=games)
    net = shard(weights, 0, 1).policy.net

    monkeypatch.setattr(collect, "POLL_SECONDS", 0.2)
    partial = durable.Partial(tmp_path / "partial" / "iter-00000")
    first = ParallelCollector([spec(0, 2, lanes=1), spec(1, 2, lanes=1)])
    try:
        first.sync(net)
        first.start_collect(games, partial)
        deadline = time.time() + 120
        while not partial.finished() and time.time() < deadline:
            time.sleep(0.05)
        assert partial.finished(), "no game was kept before the kill"
        for process in first._processes:
            os.kill(process.pid, signal.SIGKILL)
            process.join(10)
    finally:
        for process in first._processes:
            if process.is_alive():
                process.kill()
    kept = partial.finished()
    assert 0 < len(kept) < games, "the kill came after the cohort finished; nothing to replay"

    base = durable.resume_base(0, partial.indices())
    resumed = ParallelCollector([spec(w, 3, first_game=base + w, lanes=1) for w in range(3)])
    try:
        resumed.sync(net)
        episodes = {e.index: e for e in resumed.collect(games, partial)}
    finally:
        resumed.close()
    assert_same_games(reference, episodes)
