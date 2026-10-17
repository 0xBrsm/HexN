# SPDX-License-Identifier: GPL-3.0-only
"""Contract 9's data path: a game of whole decisions as bytes and back, its
columns, the batch with ragged Road Building and discard candidates, the
minibatch gathered by offsets, and the points head's zero-sum target."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.record import advance, moves, open_record  # noqa: E402
from hexset.victory import victory_points  # noqa: E402
from hexn import gamefile  # noqa: E402
from hexn.collect import WorkerSpec, _build  # noqa: E402
from hexn.model9 import collate  # noqa: E402
from hexn.ppo import PPOConfig, decision_rows, zero_sum_targets  # noqa: E402
from hexn.ppo.prepare import Preparer, games_digest, prepare, prepare_columns9, prepare_games  # noqa: E402
from hexn import durable  # noqa: E402

NET9 = (("heads", "linear"), ("discard", "mlp"), ("norm", True), ("messages", "pooled"), ("attention", False),
        ("points", True))
CONFIG = PPOConfig(contract="9", aux_target="zero_sum", aux_margin_weight=0.1, decoupled=True)


def spec(**overrides) -> WorkerSpec:
    base = dict(seed=5, players=4, lanes=3, action_cap=900, max_offers=0, first_game=0, stride=1, width=8,
                rounds=1, torch_seed=1, mix=(), parent="", contract="9", net9=NET9, discard_order="random",
                sample_stream="game", trunk="fused")
    base.update(overrides)
    return WorkerSpec(**base)


@pytest.fixture(scope="module")
def cohort():
    """Five contract-9 games from a tiny shard, as objects and as bytes."""
    policies, collector = _build(spec())
    episodes = collector.cohort(5)
    return policies[0], episodes, [gamefile.encode(e) for e in episodes]


def rows(episode):
    return [t for seat in episode.trajectories for t in seat]


def test_a_contract_9_game_comes_back_exactly(cohort):
    _, episodes, games = cohort
    for episode, back in zip(episodes, gamefile.decode_all(games)):
        assert (back.index, back.outcome, back.record, back.cast) == (episode.index, episode.outcome,
                                                                      episode.record, episode.cast)
        for a, b in zip(rows(episode), rows(back)):
            assert a.observation.tobytes() == b.observation.tobytes()
            for field in ("flat", "setup", "road_building", "discard"):
                assert np.array_equal(getattr(a.decision, field), getattr(b.decision, field)), field
                assert getattr(a.decision, field).dtype == getattr(b.decision, field).dtype
            assert a.decision.double == b.decision.double is False
            assert (a.seat, a.step, a.kind, a.index, a.log_prob, a.value, a.points, a.seat_points, a.turns,
                    a.decision_id, a.part) == (
                b.seat, b.step, b.kind, b.index, b.log_prob, b.value, b.points, b.seat_points, b.turns,
                b.decision_id, b.part)


def test_damage_is_detected_and_contracts_are_not_mixed(cohort):
    policy, _, games = cohort
    broken = games[0][:-9] + bytes(9)
    with pytest.raises(gamefile.Corrupt):
        gamefile.check(broken)
    policies, collector = _build(spec(contract="8", net9=()))
    eight = gamefile.encode(collector.cohort(1)[0])
    with pytest.raises(ValueError, match="mixes"):
        gamefile.decode_columns([games[0], eight])


def test_the_columns_hold_every_decision_and_the_batch_its_ragged_candidates(cohort):
    _, episodes, games = cohort
    columns = gamefile.decode_columns(games)
    transitions = [t for e in episodes for t in rows(e)]
    assert columns.buffer.shape[0] == len(transitions)
    assert any(len(t.decision.road_building) for t in transitions) or any(
        len(t.decision.discard) for t in transitions)
    batch, gauges, stripped = prepare_columns9(columns, CONFIG, 4)
    assert gauges == {} and all(e.trajectories == () for e in stripped)
    assert batch.chosen.tolist() == [t.index for t in transitions]
    assert batch.kind.tolist() == [t.kind for t in transitions]
    # A shuffled minibatch gathered by offsets is `collate` of those rows.
    order = torch.randperm(len(batch), generator=torch.Generator().manual_seed(3))[:300]
    got = decision_rows(batch, order, "cpu")
    want = collate([transitions[i].decision for i in order.tolist()], [transitions[i].kind for i in order.tolist()],
                   [0] * len(order))
    for key in ("setup", "rb", "rb_row", "rb_index", "disc", "disc_row", "disc_index", "tkind"):
        assert np.array_equal(got[key].numpy(), want[key]), key
    assert torch.equal(batch.mask[order], torch.from_numpy(want["mask"]))


def test_the_object_path_and_the_columns_assemble_the_same_batch(cohort):
    _, episodes, games = cohort
    by_objects, _, _ = prepare(sorted(episodes, key=lambda e: e.index), None, CONFIG, 4)
    by_bytes, _, _ = prepare_games(games, np.float32, None, CONFIG, 4)
    for field in by_objects.__dataclass_fields__:
        a, b = getattr(by_objects, field), getattr(by_bytes, field)
        assert (a is None and b is None) or torch.equal(a, b), field


def test_a_half_precision_batch_is_the_columns_rounded():
    """Under fp16 the observations, owner pointers and reach included, are
    held in half: the pointers exactly."""
    policies, collector = _build(spec(lanes=1, action_cap=200))
    games = [gamefile.encode(e) for e in collector.cohort(1)]
    half, _, _ = prepare_games(games, np.float16, None, PPOConfig(contract="9", amp="fp16"), 4)
    full, _, _ = prepare_games(games, np.float32, None, PPOConfig(contract="9"), 4)
    assert half.buffer.dtype == torch.float16
    assert torch.equal(half.buffer, full.buffer.half())
    layout = policies[0].layout
    for name in ("vertex_owner", "road_owner"):
        assert torch.equal(half.buffer[:, layout.slices[name]].float(), full.buffer[:, layout.slices[name]])


def bc_targets(record, players: int, gamma: float) -> np.ndarray:
    """The supervised points head's target, move by move off the record's
    replay: at each move, every later change in each seat's relative points
    weighted `gamma` per round from that move."""
    game = open_record(record)
    turns, points = [], []
    for actor, action, trades in moves(record):
        state = game.state(0, hidden=False)
        turns.append(game.turns)
        points.append([victory_points(state, s) for s in range(players)])
        advance(game, action, trades, actor)
    state = game.state(0, hidden=False)
    points.append([victory_points(state, s) for s in range(players)])
    points = np.asarray(points, dtype=np.float64)
    relative = points - (points.sum(1, keepdims=True) - points) / (players - 1)
    change = relative[1:] - relative[:-1]
    turns = np.asarray(turns)
    out = np.zeros((len(turns), players))
    for t in range(len(turns)):
        later = np.arange(t, len(turns))
        out[t] = (gamma ** ((turns[later] - turns[t]) // players)) @ change[later]
    return out


def test_the_zero_sum_target_is_the_one_each_moves_replay_gives(cohort):
    _, episodes, _ = cohort
    for episode in episodes:
        transitions = sorted(rows(episode), key=lambda t: t.step)
        target = zero_sum_targets(
            np.asarray([t.step for t in transitions]), np.asarray([t.turns for t in transitions]),
            np.asarray([t.seat_points for t in transitions], dtype=np.float64), episode.outcome.points, 0.6)
        reference = bc_targets(episode.record, 4, 0.6)
        np.testing.assert_allclose(target, reference[[t.step for t in transitions]], rtol=0, atol=1e-5)
        np.testing.assert_allclose(target.sum(1), 0.0, atol=1e-5)


def test_the_points_target_is_in_each_deciders_frame(cohort):
    _, episodes, games = cohort
    batch, _, _ = prepare_games(games, np.float32, None, CONFIG, 4)
    k = 0
    for episode in sorted(episodes, key=lambda e: e.index):
        ordered = sorted(rows(episode), key=lambda t: t.step)
        target = zero_sum_targets(np.asarray([t.step for t in ordered]), np.asarray([t.turns for t in ordered]),
                                  np.asarray([t.seat_points for t in ordered], dtype=np.float64),
                                  episode.outcome.points, 0.6)
        by_step = {t.step: target[i] for i, t in enumerate(ordered)}
        # One target per decision, at its first part.
        for t in rows(episode):
            if t.part == 0:
                np.testing.assert_array_equal(batch.margin_target[k].numpy(), np.roll(by_step[t.step], -t.seat))
                k += 1
    assert k == len(batch.margin_target) == batch.units()


def test_the_helper_prepares_a_contract_9_cohort_as_the_trainer_would(cohort, tmp_path):
    _, _, games = cohort
    partial = durable.partial(tmp_path / "partial", 0)
    partial.write_plan([gamefile.index_of(g) for g in games])
    for game in games:
        partial.write(game)
    helper = Preparer(None, CONFIG, 4, None, np.float32)
    try:
        helper.submit(0, partial.directory)
        got = helper.take(0, games_digest(games), timeout=120)
    finally:
        helper.close()
    want, _, _ = prepare_games(games, np.float32, None, CONFIG, 4)
    assert not isinstance(got, str), got
    for field in want.__dataclass_fields__:
        a, b = getattr(want, field), getattr(got.batch, field)
        assert (a is None and b is None) or torch.equal(a, b), field


@pytest.mark.parametrize("unbuilt", [dict(prior_kl=0.1), dict(decided_cut=0.1), dict(setup_weight=2),
                                     dict(critic="none"), dict(aux_target="relative", aux_margin_weight=0.1)])
def test_what_contract_9_is_not_built_for_is_refused(unbuilt):
    with pytest.raises(ValueError):
        PPOConfig(contract="9", **unbuilt)


def test_the_zero_sum_target_is_contract_9s():
    with pytest.raises(ValueError, match="contract 9"):
        PPOConfig(aux_target="zero_sum")
