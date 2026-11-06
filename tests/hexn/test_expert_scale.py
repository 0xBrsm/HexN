# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.benchmarks.expert_scale` runs end to end: every worker gets its own
place to keep its games."""

import random
import sys

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")


def test_two_worker_counts_run_and_report_a_point_each(tmp_path, monkeypatch, capsys):
    import json

    from hexset.actions import build_space
    from hexset.board.board import random_base_board
    from hexset.encoding import static_graph

    from hexn.benchmarks import expert_scale
    from hexn.model import HexNet, ModelConfig

    topology = random_base_board(random.Random(0)).topology
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, 4)
    net = HexNet(space, static_graph(topology), 4, ModelConfig(width=8, rounds=1))
    checkpoint = tmp_path / "net.pt"
    torch.save({"net": net.state_dict(), "args": {"width": 8, "rounds": 1}}, checkpoint)

    monkeypatch.setattr(sys, "argv", [
        "expert_scale", "--checkpoint", str(checkpoint), "--workers", "1", "2",
        "--moves", "2", "--simulations", "4", "--wave", "2", "--compile", "none",
        "--games-dir", str(tmp_path / "games"), "--json",
    ])
    assert expert_scale.main() == 0
    points = json.loads(capsys.readouterr().out)["points"]
    assert [point["workers"] for point in points] == [1, 2]
    assert all(point["decisions"] > 0 for point in points)
