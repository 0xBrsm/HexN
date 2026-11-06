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


def _native_built() -> bool:
    from hexset.native import available

    return available()


#: The duel searches a two-seat net is run with: greedy, deterministic, mixed and both.
_SEARCHES = {"greedy": "", "det": ":det=1:salt=3", "mix": ":mix=0.05:anchor=0.5", "mix-det": ":mix=0.05:det=1"}


@pytest.mark.skipif(not _native_built(), reason="the native extension is not built")
@pytest.mark.parametrize("search", sorted(_SEARCHES))
def test_an_11d_checkpoint_searches_a_duel_dealt_four_natively_as_in_python(search, tmp_path, monkeypatch):
    """A contract-11d (architecture a10) checkpoint as `rollout:` at a duel
    table dealt four with seats 1 and 3 retired, every root searched and the
    setup in full, greedy, deterministic (`det`, `salt`) and mixed (`mix`,
    `anchor`): with playout copies on the native engine it plays and logs
    the games the Python engine's copies play, searched decisions included,
    and the native copies are used."""
    import hexn.netbot  # noqa: F401  (registers the checkpoint entrant kinds)
    from hexset import rollout
    from hexset.arena import _play_one
    from hexset.gamelog import read_lines
    from hexset.native.playout import SWITCH

    config = replace(PARTS, contract="11d", others=True, points=False)
    source = a_checkpoint(tmp_path / "a10.pt", config, players=2)
    spec = f"rollout:{source}@3:top=3:horizon=2:setup=full{_SEARCHES[search]}!notrade"
    real, held = rollout.copyable, []
    monkeypatch.setattr(rollout, "copyable", lambda *a: held.append(real(*a)) or held[-1])
    lines = {}
    for arm, switch in (("native", "1"), ("python", "0")):
        monkeypatch.setenv(SWITCH, switch)
        held.clear()
        journal = tmp_path / f"{arm}.games.jsonl"
        lineup = tuple(lineup_from_names([spec, "retired", f"network:{source}!notrade", "retired"]))
        for index in (0, 1):  # a board pair, each game stopped at 12 turns and kept as exhausted
            _play_one((lineup, index, 98000, 400, True, "round", DUEL_VARIANT_GAME, 12, "random"), journal)
        assert held and (all(held) if arm == "native" else not any(held))
        lines[arm] = [{k: v for k, v in e.items() if k not in ("cpu_seconds", "wall_seconds", "started", "ended")}
                      for e in read_lines(journal) if e.get("kind") in ("game", "exhausted")]
        for e in lines[arm]:
            for log in e.get("searches", ()):
                for d in log:
                    d.pop("cpu_seconds", None)
    assert len(lines["native"]) == 2 and lines["native"] == lines["python"]
    searched = [d for e in lines["native"] for log in e.get("searches", ()) for d in log if d["worlds"]]
    assert searched and any(d.get("whole") for d in searched)
    if "det" in search:
        assert all("det_key" in d for d in searched)


def _searcher(bot):
    """The `PairedRollouts` behind a served seat's wrappers."""
    queue = [bot]
    while queue:
        current = queue.pop(0)
        if isinstance(current, PairedRollouts):
            return current
        queue += [getattr(current, name) for name in ("mover", "proposer", "bot", "first", "second")
                  if hasattr(current, name)]
    raise LookupError("no searcher behind the seat")


@pytest.mark.skipif(not _native_built(), reason="the native extension is not built")
def test_an_11d_file_served_with_a_deterministic_search_plays_natively_as_in_python(tmp_path, monkeypatch):
    """A contract-11d file stamped with a whole-decision search (`setup=full`,
    `det=1`), seated as served (`hexset.clients.onnxbot.spawn`) at a duel
    table dealt four: native and Python playout copies log the same searched
    decisions, and each searched decision, decided again by a fresh seat on
    a copy of its position, logs the same outcomes and move."""
    import copy

    from hexset.clients.onnxmeta import with_metadata
    from hexset.game import Phase
    from hexset.native.playout import SWITCH

    config = replace(PARTS, contract="11d", others=True, points=False)
    plain = export9(str(a_checkpoint(tmp_path / "a10.pt", config, players=2)), tmp_path / "a10.onnx")
    keys = {"search": "rollout", "worlds": "3", "top": "3", "horizon": "2", "setup": "full", "det": "1", "salt": "5"}
    served = tmp_path / "a10-det.onnx"
    served.write_bytes(with_metadata(plain.read_bytes(), keys))

    def play(seed: int):
        game = deal_game(seed, 0, 4, game_type=DUEL_VARIANT_GAME, locked=(1, 3))
        board = game._state.board
        seats = {0: onnxbot.spawn(str(served), board, rng=random.Random(seed)),
                 2: onnxbot.spawn(str(plain), board, rng=random.Random(seed + 1))}
        before = []
        while game.turns < 6 and not is_over(game):
            seat = next(s for s, n in enumerate(game.discard_quota) if n) if game.phase is Phase.DISCARD \
                else to_move(game)
            if seat == 0:
                before.append(copy.deepcopy(game))
            action = seats[seat].choose(game)
            apply(game, action)
        log = [d.to_json() for d in _searcher(seats[0]).decision_log]
        return log, before

    strip = lambda d: {k: v for k, v in d.items() if k != "cpu_seconds"}  # noqa: E731
    monkeypatch.setenv(SWITCH, "1")
    native, positions_seen = play(11)
    monkeypatch.setenv(SWITCH, "0")
    python, _ = play(11)
    assert [strip(d) for d in native] == [strip(d) for d in python]
    searched = [i for i, d in enumerate(native) if d["worlds"]]
    assert searched and any(native[i].get("whole") for i in searched)
    monkeypatch.setenv(SWITCH, "1")
    by_key = {native[i]["det_key"]: native[i] for i in searched}
    replayed = 0
    for game in positions_seen:
        fresh = _searcher(onnxbot.spawn(str(served), game._state.board, rng=random.Random(20260910)))
        key = fresh.decision_key(game, 0).hex()
        if key not in by_key:
            continue
        fresh.choose(game)
        assert strip(fresh.decision_log[-1].to_json()) == strip(by_key[key])
        replayed += 1
    assert replayed >= 3


@pytest.mark.skipif(not _native_built(), reason="the native extension is not built")
def test_an_11d_file_lists_the_same_whole_options_on_native_and_python_copies(tmp_path):
    """At positions of duels dealt four with seats 1 and 3 retired, robber
    moves and discards among them, an 11d file lists the same whole options
    with the same probabilities, and the same values, on a native playout
    copy as on its Python twin."""
    from hexset.game import Phase
    from hexset.native.playout import NativeCopy, held
    from hexset.rollout import paired_copy

    from test_export_onnx9 import _decider

    config = replace(PARTS, contract="11d", others=True, points=False)
    out = export9(str(a_checkpoint(tmp_path / "a10.pt", config, players=2)), tmp_path / "a10.onnx")
    rng = random.Random(7)
    checked, phases = 0, set()
    for index in range(6):
        game = deal_game(98000, index, 4, game_type=DUEL_VARIANT_GAME, locked=(1, 3))
        policy = onnxbot.load(str(out), game._state.board.topology).policy
        for step in range(600):
            if is_over(game):
                break
            seat = _decider(game)
            if step % 5 == 0 or game.phase in (Phase.ROBBER, Phase.DISCARD):
                python = paired_copy(game, 11 + step, seat)
                native = held(paired_copy(game, 11 + step, seat))
                assert isinstance(native, NativeCopy)
                (listed, probs), = policy.option_rows([(native, seat)])
                (want, want_probs), = policy.option_rows([(python, seat)])
                assert listed == want and probs.tobytes() == want_probs.tobytes()
                for s in (0, 2):
                    assert policy.value_rows([(native, s)]) == policy.value_rows([(python, s)])
                checked += 1
                phases.add(game.phase)
            asking = seat if game.phase is Phase.DISCARD else None
            apply(game, rng.choice(legal_actions(game, asking)), asking)
    assert checked >= 60
    assert {Phase.SETUP_SETTLEMENT, Phase.MAIN, Phase.ROBBER} <= phases
