# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
onnx = pytest.importorskip("onnx", reason="only needed to run hexn.export_onnx")
ort = pytest.importorskip(
    "onnxruntime", reason="only needed to run hexn.export_onnx"
)

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import encode, encode_batch, static_graph, to_frame  # noqa: E402
from hexn.export_onnx import (  # noqa: E402
    _INPUT_NAMES,
    _OUTPUT_NAMES,
    RecordEncoder,
    _rotate_slot,
    _sample_inputs,
    export,
)
from hexset.game import is_over, start, to_move  # noqa: E402
from hexn.model import HexNet, ModelConfig  # noqa: E402
from hexset.onnx_record import RECORD_FIELDS, record_batch, record_from_game  # noqa: E402
from hexset.play import step_randomly  # noqa: E402


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        if is_over(game):
            break
        step_randomly(game, rng)
    return game


def as_tensors(record: dict[str, np.ndarray]) -> dict[str, "torch.Tensor"]:
    return {name: torch.from_numpy(value) for name, value in record.items()}


def run_encoder(encoder: RecordEncoder, record: dict[str, np.ndarray]):
    tensors = as_tensors(record)
    args = [tensors[name] for name in RECORD_FIELDS if name != "action_mask"]
    with torch.no_grad():
        return encoder(*args)


def assert_exact(got: tuple, want) -> None:
    names = ("hexes", "vertices", "edges", "globals")
    for name, g, w in zip(names, got, (want.hexes, want.vertices, want.edges, want.globals)):
        g_np = g.numpy()
        assert g_np.dtype == np.float32 == w.dtype, name
        assert g_np.shape == w.shape, name
        assert np.array_equal(g_np, w), f"{name} mismatch: {np.abs(g_np - w).max()}"


def test_the_traced_rotation_is_the_encoders_own():
    """`_rotate_slot` is the seat rotation written in tensor ops because a
    `Tensor` cannot walk a Python list -- the one place hexn restates a
    convention `hexset.encoding` owns, and it stays only because it has to be
    traceable into the graph. Its oracle is `hexset.encoding.to_frame`:
    slot `i` of a perspective frame is board seat `(perspective + i) %
    players`, so the slot an owner lands in is where that owner appears
    in the frame.
    `NO_OWNER` sits last, after every real seat.
    """
    from hexset.state import NO_OWNER

    for players in (2, 3, 4):
        frames = [list(to_frame(range(players), p)) for p in range(players)]
        owners = torch.tensor(
            [[o for o in range(players)] + [NO_OWNER] for _ in range(players)]
        )
        perspective = torch.arange(players)
        slots = _rotate_slot(owners, perspective, players)
        for p in range(players):
            for owner in range(players):
                assert int(slots[p, owner]) == frames[p].index(owner)
            assert int(slots[p, players]) == players


def test_record_encoder_matches_encode_exactly_on_a_seeded_playout():
    """Real positions from a random playout, not gaussians: `encode` is a
    construction from integers, so the torch mirror must match it bit for
    bit, with no tolerance. Moved here from `tests/test_onnx_record.py`:
    `RecordEncoder` itself now lives in `hexn.export_onnx`, not
    `hexset.onnx_record`, which stays torch-free."""
    rng = random.Random(7)
    games = [start(random_base_board(rng), 4, rng) for _ in range(6)]
    space = space_for(games[0])
    # public field
    graph = static_graph(games[0].state(0, hidden=False).board.topology)
    encoder = RecordEncoder(graph, players=4)
    encoder.eval()

    checked = 0
    for _ in range(60):
        for game in games:
            if not is_over(game):
                step_randomly(game, rng)
        for game in games:
            if is_over(game):
                continue
            seat = to_move(game)
            record = record_from_game(game, seat, space)
            got = run_encoder(encoder, {k: v[None] for k, v in record.items()})
            want = encode(game, seat)
            assert_exact(tuple(g[0] for g in got), want)
            checked += 1

    assert checked > 200


def test_record_encoder_matches_encode_batch():
    """The batched numpy path is the one the graph actually replaces, since a
    search encodes a whole wave of leaves at once -- so it needs its own
    check against the torch encoder, not just the single-position path."""
    rng = random.Random(11)
    games = [a_game(seed=seed, steps=40 + seed) for seed in range(10)]
    live = [g for g in games if not is_over(g)]
    space = space_for(live[0])
    # public field
    graph = static_graph(live[0].state(0, hidden=False).board.topology)
    encoder = RecordEncoder(graph, players=4)
    encoder.eval()

    perspectives = [to_move(g) for g in live]
    record = record_batch(list(zip(live, perspectives)), space)
    got = run_encoder(encoder, record)

    want = encode_batch(live, perspectives)
    for i, obs in enumerate(want):
        assert_exact(tuple(g[i] for g in got), obs)


def a_checkpoint(path, *, players: int = 4, max_trades: int | None = 3, seed: int = 0):
    """A checkpoint in the shape `hexn.loop.save` writes, tiny enough to
    export quickly. Mirrors `test_netbot.py::a_checkpoint` — kept as its own
    copy rather than a cross-file import, same as that file does for itself.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1))
    torch.save(
        {
            "iteration": 7,
            "net": net.state_dict(),
            "args": {
                "players": players,
                "width": 16,
                "rounds": 1,
                "max_trades": max_trades,
            },
        },
        path,
    )
    return board


def test_export_writes_a_parity_checked_onnx_file(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "latest.onnx"

    # export() runs its own numerical-parity check internally (see
    # hexn.export_onnx._verify_parity) and raises if the eager and
    # onnxruntime paths disagree — a clean return is itself the load-bearing
    # assertion here.
    result = export(str(checkpoint), out, topology=board.topology)

    assert result == out
    assert out.exists()


def test_the_graph_is_shaped_the_way_hexset_feeds_and_reads_it_v2(tmp_path):
    """The v2 consumer is `hexset.clients.onnxbot`'s contract-2 path: it feeds the
    information-set record and reads back one argmaxed index and two
    distributions -- nothing about masking or rotation left for the caller to
    do."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "latest.onnx"
    export(str(checkpoint), out, topology=board.topology)

    model = onnx.load(str(out))
    assert [t.name for t in model.graph.input] == list(RECORD_FIELDS)
    assert [t.name for t in model.graph.output] == list(_OUTPUT_NAMES)

    bool_inputs = {"action_mask"}
    # No float input: contract 6 deleted the public valuation vector
    # (`hexset.trading`), and every remaining record field is int64.
    float_inputs: set[str] = set()
    for tensor in model.graph.input:
        if tensor.name in bool_inputs:
            expected = onnx.TensorProto.BOOL
        elif tensor.name in float_inputs:
            expected = onnx.TensorProto.FLOAT
        else:
            expected = onnx.TensorProto.INT64
        assert tensor.type.tensor_type.elem_type == expected, tensor.name
        assert tensor.type.tensor_type.shape.dim[0].dim_param == "batch", tensor.name

    int_outputs = {"action_index"}
    for tensor in model.graph.output:
        expected = onnx.TensorProto.INT64 if tensor.name in int_outputs else onnx.TensorProto.FLOAT
        assert tensor.type.tensor_type.elem_type == expected, tensor.name
        assert tensor.type.tensor_type.shape.dim[0].dim_param == "batch", tensor.name

    game = start(board, 4, random.Random(1))
    space = space_for(game)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        inputs = _sample_inputs(space, 4, batch=batch, seed=batch)
        action_index, prior, value = session.run(list(_OUTPUT_NAMES), inputs)
        assert action_index.shape == (batch,)
        assert prior.shape == (batch, space.size)
        assert value.shape == (batch, 4)
        assert action_index.dtype == np.int64
        assert all(a.dtype == np.float32 for a in (prior, value))
        # Every row's prior is a probability distribution over the legal
        # actions the record's own `action_mask` names -- masking, softmax
        # and normalisation all happened inside the graph.
        for row in range(batch):
            legal = inputs["action_mask"][row]
            assert np.all(prior[row][~legal] == 0.0)
            assert prior[row].sum() == pytest.approx(1.0, abs=1e-4)


def test_the_exported_metadata_says_contract_6_and_matches_the_checkpoints_own_args(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, players=3, max_trades=5)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["contract"] == "6"
    assert meta["players"] == "3"
    assert meta["max_trades"] == "5"
    assert meta["iteration"] == "7"
    assert meta["num_hexes"] == str(board.topology.num_hexes)
    assert meta["num_vertices"] == str(board.topology.num_vertices)
    assert meta["num_edges"] == str(board.topology.num_edges)
    assert meta["source_checkpoint"] == str(checkpoint)
    # A plain export says nothing about search, so the UI plays one forward
    # pass — the cheap default every pre-`search` export already gets.
    assert "search" not in meta
    assert "simulations" not in meta
    assert "wave" not in meta
    # Nor about the trade gate's own continuation budget: the UI's gate
    # rolls no rollout unless a file asks for one.
    assert "gate_plies" not in meta


def test_a_checkpoint_that_omits_max_trades_exports_empty_metadata_not_none(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, max_trades=None)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map["max_trades"] == ""


def test_a_search_export_declares_itself_in_the_keys_hexset_reads(tmp_path):
    """`hexset.clients.modelmeta.search_config`: `search == "mcts"` turns the
    search on, `simulations`/`wave` are its budget, and the UI clamps them."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "mcts256.onnx"

    export(
        str(checkpoint), out, topology=board.topology,
        search="mcts", simulations=256, wave=32,
    )

    meta = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    assert meta["contract"] == "6"
    assert meta["search"] == "mcts"
    assert meta["simulations"] == "256"
    assert meta["wave"] == "32"


def test_a_gate_plies_export_declares_itself_in_the_keys_hexset_reads(tmp_path):
    """`hexset.clients.modelmeta.gate_config`: `gate_plies` is the trade gate's
    own continuation budget, independent of `search` -- a plain policy file
    (no `search` at all here) can still ask its gate to roll forward."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "gated8.onnx"

    export(str(checkpoint), out, topology=board.topology, gate_plies=8)

    meta = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    assert meta["contract"] == "6"
    assert meta["gate_plies"] == "8"
    assert "search" not in meta


def test_a_search_budget_without_a_search_is_refused_not_written(tmp_path):
    """The UI ignores `simulations` unless `search=mcts`, so a file carrying
    one without the other would silently be a plain policy."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="search='mcts'"):
        export(str(checkpoint), tmp_path / "x.onnx", topology=board.topology, simulations=64)
    with pytest.raises(ValueError, match="search must be one of"):
        export(str(checkpoint), tmp_path / "x.onnx", topology=board.topology, search="puct")


def test_input_names_are_exactly_the_record_fields():
    """Pins the exporter's input order to `onnx_record.RECORD_FIELDS` --
    hexset's phase-3 record builder has to produce a dict with exactly
    these keys, in a contract both repos can only agree on by name."""
    assert _INPUT_NAMES == RECORD_FIELDS
