# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.export_onnx9`: a contract-9 checkpoint exported to a decision graph,
which `hexset.clients.onnxbot.DecisionPolicy` runs: HexSet's contract-9
observation and a decision's arrays in, the decision out. What is checked
here: the graph chooses the decision `hexn.policy9.Policy9` chooses on the
torch net, reads the same `value` and `prior` within float tolerance, the
`prior` is the whole decision's probability on its first move, plays
the same moves through HexSet's runner, declares the inputs and outputs the
contract names, and carries the metadata HexSet reads."""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from _bots import needs

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
onnx = pytest.importorskip("onnx", reason="only needed to run hexn.export_onnx9")
ort = pytest.importorskip("onnxruntime", reason="only needed to run hexn.export_onnx9")

from hexset import decisions as D  # noqa: E402
from hexset.actions import apply, legal_actions, options_for, space_for  # noqa: E402
from hexset.arena import deal_game  # noqa: E402
from hexset.cards import DevCard  # noqa: E402
from hexset.clients.onnxbot import DecisionPolicy, _load_cached, load  # noqa: E402
from hexset.clients.onnxmeta import read_metadata  # noqa: E402
from hexset.game import Phase, is_over, to_move  # noqa: E402
from hexn.export_onnx import main as export_main  # noqa: E402
from hexn.export_onnx9 import export9, input_names, output_names, serving_props  # noqa: E402
from hexn.model9 import HexNet, ModelConfig  # noqa: E402
from hexn.policy9 import Policy9  # noqa: E402

PARTS = ModelConfig(width=16, rounds=2, messages="pooled", attention=False, points=True, double_turn="parts")
JOINT = ModelConfig(width=16, rounds=2, messages="separate", attention=True, points=False, double_turn="joint")


@pytest.fixture(autouse=True)
def _fresh_sessions():
    yield
    _load_cached.cache_clear()


def a_checkpoint(path: Path, config: ModelConfig, *, seed: int = 0, iteration: int = 7) -> Path:
    """A small contract-9 net saved as a trainer saves one, its option heads
    sharpened past an untrained net's near-ties so a choice tests something."""
    game = deal_game(0, 0, 4)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), game._state.board.topology, 4, config).eval()
    with torch.no_grad():
        for head in (net.vertex_head, net.road_head, net.global_head, net.setup_head, net.road_building,
                     net.robber_head, net.first_head, net.second_head):
            if head is not None:
                for p in head.parameters():
                    p.mul_(50)
    torch.save({"net": net.state_dict(), "model_config": {**asdict(config), "contract": "9"},
                "args": {"contract": "9", "players": 4, "max_offers": 0}, "iteration": iteration}, path)
    return path


def _torch_policy(source: Path, config: ModelConfig) -> Policy9:
    game = deal_game(0, 0, 4)
    net = HexNet(space_for(game), game._state.board.topology, 4, config)
    net.load_state_dict(torch.load(source, weights_only=False)["net"])
    return Policy9(net.eval(), space_for(game))


def _decider(game) -> int:
    return next(s for s, n in enumerate(game.discard_quota) if n) if game.phase is Phase.DISCARD else to_move(game)


def positions(seed: int, count: int):
    """Seeded positions with every decision kind among them: random playouts
    stopped anywhere (setup turns, the snake seat's double turn, a setup road
    owed, discards, robber moves, ordinary turns), and some ordinary turns
    with a matured Road Building card handed to the mover, so the card's road
    sets are asked too."""
    rng = random.Random(seed)
    out = []
    index = 0
    while len(out) < count:
        game = deal_game(seed, index, 4)
        index += 1
        # Six moves in, the snake seat places its first settlement: its double turn.
        for _ in range(rng.choice((0, 3, 6, 7, 12, 40, 120, 250))):
            if is_over(game):
                break
            seat = _decider(game)
            asking = seat if game.phase is Phase.DISCARD else None
            apply(game, rng.choice(legal_actions(game, asking)), asking)
        if is_over(game):
            continue
        if game.phase is Phase.MAIN and not game.dev_card_played and rng.random() < 0.5:
            game._state.dev_cards[to_move(game)][DevCard.ROAD_BUILDING] += 1
        out.append((game, _decider(game)))
    return out


def _kinds(rows, joint: bool, space) -> set:
    seen = set()
    for game, seat in rows:
        if game.phase is Phase.SETUP_ROAD or game.free_roads:
            seen.add("inside")
            continue
        decision = D.tensors(game, seat, space, double_turn=joint)
        seen.update(k for k, on in (("double", decision.double), ("setup", decision.setup.any() and not decision.double),
                                     ("road_building", len(decision.road_building)),
                                     ("discard", len(decision.discard)), ("flat", decision.flat.any())) if on)
    return seen


@pytest.mark.parametrize("config", [PARTS, JOINT], ids=["parts", "joint"])
def test_the_graph_chooses_the_decision_the_torch_net_chooses(config, tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", config)
    out = export9(str(source), tmp_path / "net.onnx")
    game = deal_game(0, 0, 4)
    topology = game._state.board.topology
    served = load(str(out), topology).policy
    reference = _torch_policy(source, config)
    joint = config.double_turn == "joint"
    assert isinstance(served, DecisionPolicy) and ("double_rows" in served.inputs) == joint
    rows = positions(3, 160)
    kinds = _kinds(rows, joint, served.space)
    assert {"setup", "road_building", "discard", "flat", "inside"} <= kinds and ("double" in kinds) == joint
    # The graph's answers on one batch of every row, against the torch net's.
    decisions = [reference.decision(g, s) for g, s in rows]
    feed = served._feed(rows, decisions)
    got = served._run(feed, ["decision", "value", "prior"])
    arrays = {k: v for k, v in feed.items() if k != "observation"}
    arrays.setdefault("double_rows", np.zeros(0, dtype=np.int64))
    want = reference.answers(feed["observation"], arrays, topology)
    np.testing.assert_array_equal(got["decision"], want["decision"])
    for name in ("value", "prior"):
        np.testing.assert_allclose(got[name], want[name], rtol=1e-3, atol=1e-4, err_msg=name)
    # The same moves, values and search priors through HexSet's runner.
    for game, seat in rows:
        options = tuple(options_for(game))
        assert served.act_rows([(game, seat, options)]) == reference.act_rows([(game, seat, options)])
        np.testing.assert_allclose(served.value_rows([(game, s) for s in range(4)]),
                                   reference.value_rows([(game, s) for s in range(4)]), atol=1e-4)
        ((p1, v1),), ((p2, v2),) = served.score_rows([(game, seat, options)]), reference.score_rows(
            [(game, seat, options)])
        np.testing.assert_allclose(p1, p2, atol=1e-4)
        np.testing.assert_allclose(v1, v2, atol=1e-4)


def _first_moves(option, seat, space) -> dict[int, float]:
    """Where a whole option's probability goes among the first atomic moves:
    its first part, or for a discard each kind of card it gives, by its share
    of the cards (the order the cards are given in drawn uniformly)."""
    if not isinstance(option, D.Compound):
        return {D.flat_index(space, option, seat, 4): 1.0}
    if option.kind is D.CompoundKind.DISCARD:
        out = {}
        for part in option.parts:
            out[space.index(part)] = out.get(space.index(part), 0.0) + 1.0 / len(option.parts)
        return out
    return {space.index(option.parts[0]): 1.0}


@pytest.mark.parametrize("config", [PARTS, JOINT], ids=["parts", "joint"])
def test_the_prior_is_the_whole_decision_marginalised_onto_its_first_move(config, tmp_path):
    """`prior` against `option_rows`, every whole option and its probability:
    at a setup turn, the double turn, Road Building and a discard the mass of
    every option goes to its first move, so a search a move at a time sees
    them all; a decision of flat actions alone keeps the flat softmax."""
    policy = _torch_policy(a_checkpoint(tmp_path / "net.pt", config), config)
    rows = positions(3, 160)
    _, _, heads = policy._scored(rows, prior=True)
    seen = set()
    for (game, seat), prior in zip(rows, heads.prior):
        ((options, probs),) = policy.option_rows([(game, seat)])
        want = np.zeros(policy.space.size)
        for option, p in zip(options, probs):
            for index, share in _first_moves(option, seat, policy.space).items():
                want[index] += p * share
        np.testing.assert_allclose(prior, want, rtol=1e-4, atol=1e-6)
        seen.update(o.kind for o in options if isinstance(o, D.Compound))
        if not any(isinstance(o, D.Compound) for o in options):
            legal = policy.decision(game, seat).flat
            scores = policy._scored([(game, seat)])[2].flat[0][legal].astype(np.float64)
            np.testing.assert_allclose(prior[legal], np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum(),
                                       rtol=1e-4, atol=1e-6)
        # What a search reads: every legal move with mass where the decision gives it any.
        atomic = tuple(legal_actions(game, seat))
        ((weights, _),) = policy.score_rows([(game, seat, atomic)])
        index = [D.flat_index(policy.space, a, seat, 4) for a in atomic]
        np.testing.assert_allclose(weights, want[index] / want[index].sum(), rtol=1e-4, atol=1e-6)
    assert {D.CompoundKind.SETUP, D.CompoundKind.ROAD_BUILDING, D.CompoundKind.DISCARD} <= seen
    assert (D.CompoundKind.SETUP_DOUBLE in seen) == (config.double_turn == "joint")


def test_the_graph_declares_the_inputs_it_reads_and_answers_in_the_action_contract(tmp_path):
    for config in (PARTS, JOINT):
        joint = config.double_turn == "joint"
        out = export9(str(a_checkpoint(tmp_path / f"{joint}.pt", config)), tmp_path / f"{joint}.onnx")
        model = onnx.load(str(out))
        assert [t.name for t in model.graph.input] == list(input_names(joint))
        assert [t.name for t in model.graph.output] == list(output_names(config.points))
        assert ("double_rows" in [t.name for t in model.graph.input]) == joint
        assert ("points" in [t.name for t in model.graph.output]) == config.points


def test_the_metadata_names_contract_9_and_no_offers_unless_asked(tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", PARTS, iteration=42)
    meta = read_metadata(export9(str(source), tmp_path / "plain.onnx").read_bytes())
    assert meta["contract"] == "9" and meta["double_turn"] == "parts" and meta["max_offers"] == "0"
    assert meta["iteration"] == "42" and meta["players"] == "4"
    assert (meta["num_hexes"], meta["num_vertices"], meta["num_edges"]) == ("19", "54", "72")
    assert meta["checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not {"trader", "trader_veto", "trade_floor", "gate_own_frame", "robber_tip", "search"} & set(meta)


@needs("heximax")
def test_the_served_shape_is_written_by_the_command_line_in_the_keys_hexset_reads(tmp_path):
    """A trader, its veto by the file's own gate at a floor, one offer a
    turn and the own-frame gate: the keys a served file carries."""
    source = a_checkpoint(tmp_path / "net.pt", PARTS)
    out = tmp_path / "served.onnx"
    export_main(["--checkpoint", str(source), "--out", str(out), "--trader", "heximax", "--trader-veto",
                 "--trade-floor", "0.03", "--max-offers", "1", "--gate-own-frame", "1"])
    meta = read_metadata(out.read_bytes())
    assert {k: meta[k] for k in ("contract", "trader", "trader_veto", "trade_floor", "max_offers", "gate_own_frame")} \
        == {"contract": "9", "trader": "heximax", "trader_veto": "1", "trade_floor": "0.03", "max_offers": "1",
            "gate_own_frame": "1"}
    from hexset.clients.netbot import declared_trader

    game = deal_game(0, 0, 4)
    loaded = load(str(out), game._state.board.topology)
    assert loaded.trader == "heximax" and loaded.trader_veto and loaded.trade_floor == pytest.approx(0.03)
    assert loaded.gate.max_offers == 1 and loaded.gate.gate_own_frame
    assert declared_trader(loaded, str(out)) == "heximax"


def test_a_search_is_written_in_the_keys_hexset_searches_by(tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", PARTS)
    out = tmp_path / "searched.onnx"
    export_main(["--checkpoint", str(source), "--out", str(out), "--search", "mcts", "--simulations", "8"])
    meta = read_metadata(out.read_bytes())
    assert (meta["search"], meta["simulations"]) == ("mcts", "8") and "wave" not in meta
    loaded = load(str(out), deal_game(0, 0, 4)._state.board.topology)
    assert loaded.search.searches and loaded.search.simulations == 8 and loaded.policy.has_prior


def test_a_veto_with_nothing_to_veto_is_refused_not_written(tmp_path):
    source = a_checkpoint(tmp_path / "net.pt", PARTS)
    with pytest.raises(SystemExit):
        export_main(["--checkpoint", str(source), "--out", str(tmp_path / "x.onnx"), "--trader-veto"])
    with pytest.raises(ValueError):
        serving_props(trader="heximax", trader_veto=True, max_offers=0)
    assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob(".x.onnx*"))


def _real_checkpoints():
    return [p for p in os.environ.get("HEXN_C9_CHECKPOINTS", "").split(os.pathsep) if p]


@pytest.mark.skipif(not _real_checkpoints(), reason="HEXN_C9_CHECKPOINTS names no contract-9 checkpoint")
@pytest.mark.parametrize("path", _real_checkpoints())
def test_a_real_contract_9_checkpoint_exports_and_plays_as_its_torch_net(path, tmp_path):
    """A real checkpoint (`HEXN_C9_CHECKPOINTS`, paths joined by the path
    separator): exported, then the file and the torch net choose alike on
    seeded positions of every kind."""
    out = export9(path, tmp_path / "real.onnx")
    game = deal_game(0, 0, 4)
    from hexn import netbot

    reference = netbot.load(path, game._state.board.topology).policy
    served = load(str(out), game._state.board.topology).policy
    for game, seat in positions(5, 300):
        options = tuple(options_for(game))
        assert served.act_rows([(game, seat, options)]) == reference.act_rows([(game, seat, options)])
