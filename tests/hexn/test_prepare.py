# SPDX-License-Identifier: GPL-3.0-only
"""The helper that assembles an async cohort off the trainer's critical path."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import durable, gamefile  # noqa: E402
from hexn.collect import WorkerSpec, _build  # noqa: E402
from hexn.ppo import PPOConfig  # noqa: E402
from hexn.ppo.prepare import Prepared, Preparer, games_digest, prepare  # noqa: E402
from hexn.steps import by_index  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    """Six encoded games from a real (tiny) shard, as a worker sends them."""
    policies, collector = _build(WorkerSpec(seed=5, players=4, lanes=3, action_cap=200, max_offers=0, first_game=0,
                                            stride=1, width=8, rounds=1, torch_seed=1, mix=(), parent=""))
    episodes = collector.cohort(6)
    return policies[0].layout, [gamefile.encode(e) for e in episodes]


CONFIG = PPOConfig(lam=0.95, decided_cut=0.05, setup_weight=2, decoupled=True)


def test_the_digest_is_the_cohorts_in_index_order(cohort):
    _, games = cohort
    assert games_digest(games) == games_digest(list(reversed(games)))
    assert games_digest(games) != games_digest(games[:-1])


def test_the_helper_prepares_exactly_what_the_trainer_would(cohort, tmp_path):
    layout, games = cohort
    partial = durable.Partial(tmp_path / "iter-00000")
    partial.write_plan(sorted(gamefile.index_of(g) for g in games))
    for game in games:
        partial.write(game)
    preparer = Preparer(layout, CONFIG, 4, None, "float32")
    try:
        preparer.submit(0, partial.directory)
        got = preparer.take(0, games_digest(games), timeout=300)
    finally:
        preparer.close()
    assert isinstance(got, Prepared), got
    local_batch, local_gauges, local_episodes = prepare(
        by_index(gamefile.decode_all(sorted(games, key=gamefile.index_of))), layout, CONFIG, 4)
    for name in local_batch.__dataclass_fields__:
        mine, theirs = getattr(local_batch, name), getattr(got.batch, name)
        assert (mine is None) == (theirs is None), name
        if mine is not None:
            assert mine.dtype == theirs.dtype and torch.equal(mine, theirs), name
    assert got.gauges == local_gauges
    assert [e.index for e in got.episodes] == [e.index for e in local_episodes]


def test_the_trainer_is_told_when_the_helper_read_other_games(cohort, tmp_path):
    layout, games = cohort
    partial = durable.Partial(tmp_path / "iter-00000")
    partial.write_plan(sorted(gamefile.index_of(g) for g in games))
    for game in games:
        partial.write(game)
    preparer = Preparer(layout, CONFIG, 4, None, "float32")
    try:
        preparer.submit(0, partial.directory)
        got = preparer.take(0, games_digest(games[:-1]), timeout=300)
        assert got == "the helper read different games from the ones received"
        assert preparer.take(1, "x") == "cohort 1 was not submitted"
    finally:
        preparer.close()


@pytest.mark.parametrize(
    "config,dtype",
    [
        (PPOConfig(lam=0.95, decided_cut=0.05, setup_weight=2, decoupled=True, amp="fp16"), "float16"),
        (PPOConfig(lam=0.95, decided_cut=0.05, setup_weight=2), "float32"),
        (PPOConfig(lam=0.9), "float32"),
        (PPOConfig(lam=0.95, setup_weight=3, amp="fp16"), "float16"),
        (PPOConfig(lam=0.95, aux_margin_weight=0.5, aux_target="loss", aux_pos_weight=68.0), "float32"),
        (PPOConfig(lam=0.95, decided_cut=0.05, setup_weight=2, aux_target="loss", amp="fp16"), "float16"),
        (PPOConfig(lam=0.95, aux_margin_weight=0.5, aux_target="points"), "float32"),
        (PPOConfig(lam=0.95, decided_cut=0.05, setup_weight=2, aux_target="points", amp="fp16"), "float16"),
    ],
)
def test_columnar_assembly_is_the_object_assembly_bit_for_bit(cohort, config, dtype):
    from hexn.ppo.prepare import prepare_columns

    layout, games = cohort
    ordered = sorted(games, key=gamefile.index_of)
    want_batch, want_gauges, want_episodes = prepare(by_index(gamefile.decode_all(ordered, dtype)), layout, config, 4)
    got = prepare_columns(gamefile.decode_columns(ordered, dtype), layout, config, 4)
    assert got is not None
    batch, gauges, episodes = got
    for name in want_batch.__dataclass_fields__:
        mine, theirs = getattr(want_batch, name), getattr(batch, name)
        assert (mine is None) == (theirs is None), name
        if mine is not None:
            assert mine.dtype == theirs.dtype and mine.shape == theirs.shape and torch.equal(mine, theirs), name
    assert gauges == want_gauges
    assert [(e.index, e.seed, e.outcome, e.trajectories) for e in episodes] == [
        (e.index, e.seed, e.outcome, e.trajectories) for e in want_episodes]


def test_columnar_assembly_declines_what_it_does_not_cover(cohort):
    from hexn.ppo.prepare import prepare_columns

    layout, games = cohort
    columns = gamefile.decode_columns(sorted(games, key=gamefile.index_of))
    assert prepare_columns(columns, layout, PPOConfig(critic="none"), 4) is None
    assert prepare_columns(columns, layout, PPOConfig(vp_reward=0.1, aux_margin_weight=0.1), 4) is None
