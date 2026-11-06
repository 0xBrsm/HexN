# SPDX-License-Identifier: GPL-3.0-only
"""Paired rollouts (`hexset.rollout.PairedRollouts`) over a contract-9 net,
which chooses whole decisions. A seat that searches no root plays game for
game what the net plays alone, as the torch checkpoint and as its exported
file; a searched setup turn is one decision whose first candidate is the
net's own whole choice; and the exported file lists the same whole options
with the same probabilities as the torch net."""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
pytest.importorskip("onnx", reason="only needed to run hexn.export_onnx9")
pytest.importorskip("onnxruntime", reason="only needed to run hexn.export_onnx9")

from hexset import decisions as D  # noqa: E402
from hexset.actions import apply, legal_actions  # noqa: E402
from hexset.arena import compete, deal_game, lineup_from_names  # noqa: E402
from hexset.clients import onnxbot  # noqa: E402
from hexset.clients.netbot import bot_for, rollouts_for  # noqa: E402
from hexset.game import UNSTRUCTURED_TURN_CAP, is_over, to_move  # noqa: E402
from hexset.rollout import PairedRollouts  # noqa: E402
from hexset.rules import DUEL_VARIANT_GAME  # noqa: E402
from hexn.export_onnx9 import export9  # noqa: E402

from test_export_onnx9 import JOINT, PARTS, _deal, _torch_policy, a_checkpoint, positions  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_sessions():
    yield
    onnxbot._load_cached.cache_clear()


def _moves(bot, seed: int, count: int, players: int = 4) -> list:
    """The first `count` moves of a game every seat of which `bot` plays."""
    game, out = _deal(seed, 0, players), []
    while len(out) < count and not is_over(game):
        action = bot.choose(game)
        out.append(action)
        apply(game, action)
    return out


@pytest.mark.parametrize("config", [PARTS, JOINT], ids=["parts", "joint"])
def test_a_rollout_seat_that_searches_no_root_plays_as_the_file_alone(config, tmp_path):
    """The exported file, as one forward (`network`) and as paired rollouts
    whose every root is decisive: the same moves, setup turns, the double
    turn and discards included. The rollout seat lists whole options."""
    out = export9(str(a_checkpoint(tmp_path / "net.pt", config)), tmp_path / "net.onnx")
    board = _deal(0, 0)._state.board
    loaded = onnxbot.load(str(out), board.topology)
    assert loaded.policy.has_options
    for seed in (1, 2):
        alone = _moves(bot_for(loaded, rng=random.Random(seed)), seed, 220)
        rollouts = rollouts_for(loaded, worlds=2, top=3, decisive=1e-6, rng=random.Random(seed))
        assert _moves(rollouts, seed, 220) == alone
        assert rollouts.decision_log and all(d.worlds == 0 for d in rollouts.decision_log)


def test_a_rollout_seat_that_searches_no_root_plays_as_the_checkpoint_alone(tmp_path):
    """The torch checkpoint through the arena, `network:` against
    `rollout:...:decisive=` on the same duel deals: the game records match
    action for action."""
    import hexn.netbot  # noqa: F401  (registers the checkpoint entrant kinds)
    from hexset.arena import read_journal

    config = replace(PARTS, contract="9d", others=True, points=False)
    source = a_checkpoint(tmp_path / "duel.pt", config, players=2)
    records = {}
    for arm, spec in (("network", f"network:{source}!notrade"),
                      ("rollout", f"rollout:{source}@2:top=3:decisive=0.000001!notrade")):
        journal = tmp_path / f"{arm}.games.jsonl"
        compete(lineup_from_names([spec, "retired", "random", "retired"]), 4, seed=98000, action_cap=1500,
                turn_cap=UNSTRUCTURED_TURN_CAP, game_type=DUEL_VARIANT_GAME, journal=journal)
        records[arm] = {i: o.record for i, o in read_journal(journal)[1].items()}
    assert sorted(records["network"]) == sorted(records["rollout"]) == [0, 1, 2, 3]
    for index, record in records["network"].items():
        assert record.actions == records["rollout"][index].actions, f"game {index}"


@pytest.mark.parametrize("served", [False, True], ids=["torch", "file"])
def test_a_searched_setup_turn_is_one_decision_led_by_the_nets_own(served, tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", PARTS)
    if served:
        out = export9(str(source), tmp_path / "net.onnx")
        policy = onnxbot.load(str(out), _deal(0, 0)._state.board.topology).policy
    else:
        policy = _torch_policy(source, PARTS)
    game = _deal(3, 0)
    seat = to_move(game)
    own = policy._choose(game, seat)
    search = PairedRollouts(policy, worlds=2, top=3, horizon=1, marks=(), rng=random.Random(4))
    decision = search.decide(game)
    assert decision.whole and decision.worlds == 2 and len(decision.candidates) == 3
    assert decision.options[decision.candidates[0]] == own
    chosen = decision.options[decision.chosen]
    assert isinstance(chosen, D.Compound) and decision.move == chosen.parts[0]
    apply(game, decision.move)
    road = search.decide(game)
    assert road.worlds == 0 and road.move == chosen.parts[1]
    assert policy.plans.peek(game, seat) == ()


@pytest.mark.parametrize("config", [PARTS, JOINT], ids=["parts", "joint"])
def test_the_file_lists_the_whole_options_the_torch_net_lists(config, tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", config)
    out = export9(str(source), tmp_path / "net.onnx")
    served = onnxbot.load(str(out), _deal(0, 0)._state.board.topology).policy
    reference = _torch_policy(source, config)
    rows = positions(3, 120)
    kinds = set()
    for (listed, probs), (want, want_probs) in zip(served.option_rows(rows), reference.option_rows(rows)):
        assert listed == want
        np.testing.assert_allclose(probs, want_probs, rtol=1e-3, atol=1e-4)
        assert abs(probs.sum() - 1) < 1e-9
        kinds.update(o.kind for o in listed if isinstance(o, D.Compound))
    assert {D.CompoundKind.SETUP, D.CompoundKind.ROAD_BUILDING, D.CompoundKind.DISCARD} <= kinds
    assert (D.CompoundKind.SETUP_DOUBLE in kinds) == (config.double_turn == "joint")


def test_a_file_without_whole_options_still_plays_a_move_at_a_time(tmp_path):
    """A file exported before the whole-option outputs (here: the outputs
    stripped) loads, lists none, and is searched a move at a time with its
    own move first."""
    import onnx

    out = export9(str(a_checkpoint(tmp_path / "net.pt", PARTS)), tmp_path / "net.onnx")
    model = onnx.load(str(out))
    keep = [o for o in model.graph.output if o.name not in D.OPTION_OUTPUTS]
    del model.graph.output[:]
    model.graph.output.extend(keep)
    old = tmp_path / "old.onnx"
    onnx.save(model, str(old))
    policy = onnxbot.load(str(old), _deal(0, 0)._state.board.topology).policy
    assert not policy.has_options
    game = _deal(3, 0)
    seat = to_move(game)
    own = policy._choose(game, seat)
    decision = PairedRollouts(policy, worlds=2, top=3, horizon=1, marks=(), rng=random.Random(4)).decide(game)
    assert not decision.whole and decision.options[decision.candidates[0]] == own.parts[0]
    if decision.move != own.parts[0]:
        assert policy.plans.peek(game, seat) == ()
    else:
        assert policy.plans.peek(game, seat) == own.parts[1:]
    assert len(legal_actions(game)) > 1
