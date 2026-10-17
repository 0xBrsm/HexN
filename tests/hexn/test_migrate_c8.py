# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.migrate_c8`: a contract-7 checkpoint onto contract 8's inputs."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import global_columns, static_graph  # noqa: E402
from hexset.game import start  # noqa: E402

from hexn.migrate_c8 import PROJECTIONS, c7_widths, main, migrate_checkpoint  # noqa: E402
from hexn.model import HexNet, ModelConfig  # noqa: E402


def a_contract_7_checkpoint(path, *, seed: int = 0) -> dict:
    """A small net with contract 7's input widths, as a trainer saved it."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), static_graph(board.topology), 4, ModelConfig(width=16, rounds=1))
    for name, width in c7_widths(4).items():
        setattr(net, name, torch.nn.Linear(width, 16))
    state = {"iteration": 50, "net": net.state_dict(), "args": {"players": 4, "width": 16, "rounds": 1},
             "optimiser": {"state": {}, "param_groups": []}}
    torch.save(state, path)
    return state


def test_contract_7_widths_are_the_retired_layout():
    assert c7_widths(4) == {"embed_hex": 11, "embed_vertex": 14, "embed_edge": 5, "embed_global": 75}


def test_a_contract_7_checkpoint_is_refused_by_name_where_contract_8_is_read(tmp_path):
    path = tmp_path / "c7.pt"
    state = a_contract_7_checkpoint(path)
    rng = random.Random(0)
    board = random_base_board(rng)
    net = HexNet(space_for(start(board, 4, rng)), static_graph(board.topology), 4,
                 ModelConfig(width=16, rounds=1))
    with pytest.raises(ValueError, match="hexn.migrate_c8"):
        net.load_state_dict(state["net"])


def test_the_exact_projections_are_exact_and_the_new_columns_are_zero(tmp_path):
    path = tmp_path / "c7.pt"
    a_contract_7_checkpoint(path)
    out, report = migrate_checkpoint(path, games=8, seed=3, stride=5)

    # Hexes, edges: contract 8 restates contract 7 linearly.
    assert report["fit_max_residual_embed_hex"] < 1e-5
    assert report["fit_max_residual_embed_edge"] < 1e-5
    assert report["held_out_max_abs_embed_hex"] < 1e-4
    assert report["held_out_max_abs_embed_edge"] < 1e-4

    net = out["net"]
    columns = global_columns(4)
    for block in ("own_dev_playable", "opponent_dev_ages", "dev_odds"):
        assert not net["embed_global.weight"][:, columns[block]].any(), block
    assert not net["embed_vertex.weight"][:, -4:].any()  # seat_reach

    assert "optimiser" not in out
    assert out["migrate_c8"]["contract_to"] == "8"
    assert out["iteration"] == 50
    for name in PROJECTIONS:
        assert report[f"held_out_rms_{name}"] < 1.0
    assert report["held_out_decisions"] > 0
    assert 0.0 <= report["held_out_same_move"] <= 1.0
    assert report["held_out_mean_kl"] >= 0.0


def test_the_migrated_checkpoint_loads_into_a_contract_8_net(tmp_path):
    path, out = tmp_path / "c7.pt", tmp_path / "c8.pt"
    a_contract_7_checkpoint(path)
    assert main(["--checkpoint", str(path), "--out", str(out), "--games", "4", "--stride", "9"]) == 0
    written = torch.load(out, map_location="cpu", weights_only=False)
    rng = random.Random(0)
    board = random_base_board(rng)
    net = HexNet(space_for(start(board, 4, rng)), static_graph(board.topology), 4,
                 ModelConfig(width=16, rounds=1))
    net.load_state_dict(written["net"], strict=True)
    with pytest.raises(SystemExit, match="exists"):
        main(["--checkpoint", str(path), "--out", str(out)])
