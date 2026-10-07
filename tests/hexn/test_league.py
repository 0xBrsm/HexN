# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.league import nudged, parse_learner, standings  # noqa: E402
from hexn.ppo import PPOConfig  # noqa: E402
from hexn.selfplay import Episode, Outcome  # noqa: E402


def test_learner_overrides_apply_and_unknown_keys_refuse():
    base = PPOConfig()
    varied, target, gain = parse_learner("entropy=0.05,lr=6e-4,epochs=2,eps=1e-8", base)
    assert target is None
    assert gain == 0.10, "default gain, even though this seat has no controller"
    assert varied.entropy_coefficient == 0.05
    assert varied.learning_rate == 6e-4
    assert varied.epochs == 2
    assert varied.adam_eps == 1e-8
    assert varied.clip == base.clip, "untouched fields keep the base"
    assert parse_learner("", base) == (base, None, 0.10)
    with pytest.raises(SystemExit):
        parse_learner("games=256", base)


def test_the_entropy_controller_nudges_toward_the_target_and_clamps():
    assert nudged(0.02, entropy=0.5, target=0.65) == pytest.approx(0.022)
    assert nudged(0.02, entropy=0.8, target=0.65) == pytest.approx(0.02 / 1.1)
    assert nudged(0.10, entropy=0.1, target=0.65) == 0.10, "clamped above"
    assert nudged(0.005, entropy=0.9, target=0.65) == 0.005, "clamped below"


def test_standings_count_wins_and_vp_by_cast():
    def game(winner, cast, points):
        return Episode(
            index=0,
            seed=0,
            players=4,
            trajectories=((), (), (), ()),
            outcome=Outcome(
                winner=winner, points=points, turns=10, actions=40, truncated=False
            ),
            cast=cast,
        )

    episodes = [
        game(0, (0, 1, 0, 1), (10, 2, 3, 4)),
        game(0, (1, 0, 1, 0), (10, 2, 3, 4)),
        game(None, (0, 1, 0, 1), (5, 5, 5, 5)),
    ]
    # Game one's winner seat 0 is learner 0's; game two's seat 0 is learner
    # 1's under the rotated cast — one win each, by cast and not by seat.
    wins, vp = standings(episodes, 2)
    assert wins == [1, 1]
    assert vp[0] == pytest.approx(vp[1]), "symmetric fixtures score level"



def test_a_relaunched_heat_resumes_each_seat_from_its_own_state(tmp_path, monkeypatch):
    """The heat's own manifest launched again over its checkpoints resumes,
    with no `--resume`: each seat from its own config -- the controller's
    moved coefficient, not the override re-parsed -- and the standings summed
    once over the iterations the checkpoints hold."""
    import json
    import pathlib
    import random

    from hexset.actions import build_space
    from hexset.board.board import random_base_board
    from hexset.encoding import static_graph

    from hexn import league
    from hexn.model import HexNet, ModelConfig
    from hexn.run import freeze

    board = random_base_board(random.Random(0))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, 4
    )
    net = HexNet(space, static_graph(topology), 4, ModelConfig(width=8, rounds=1))
    base = tmp_path / "base.pt"
    torch.save({"net": net.state_dict(), "args": {"width": 8, "rounds": 1}}, base)
    heat = tmp_path / "heat"

    def launch(iterations):
        argv = [
            "--base", str(base),
            "--learner", "epochs=1,minibatch=256",
            "--learner", "epochs=1,minibatch=256,target_entropy=5.0",
            "--iterations", str(iterations), "--games-per-iteration", "2",
            "--lanes", "2", "--collect-workers", "1", "--device", "cpu",
            "--action-cap", "100", "--max-offers", "0",
            "--checkpoint-dir", str(heat),
        ]
        freeze("league", "heat", heat, argv, repo=pathlib.Path(tmp_path),
               description="test heat")
        return league.main([str(heat)])

    assert launch(1) == 0
    moved = torch.load(heat / "learner1" / "latest.pt", weights_only=False)
    coefficient = moved["config"]["entropy_coefficient"]
    assert coefficient != PPOConfig().entropy_coefficient, "the controller moved it"
    assert "torch_rng" in moved

    seen = []
    real_update = league.update

    def update(policy, optimiser, batch, config):
        seen.append(config.entropy_coefficient)
        return real_update(policy, optimiser, batch, config)

    monkeypatch.setattr(league, "update", update)
    assert launch(2) == 0

    assert seen[1] == coefficient, "learner 1 carried on from its own coefficient"
    rows = [json.loads(line) for line in (heat / "log.jsonl").read_text().splitlines()]
    assert rows[1] == {"resumed_from": 1}
    first, second = rows[0], rows[2]
    assert second["iteration"] == 1
    assert second["standings"] == [
        a["wins"] + b["wins"] for a, b in zip(first["learners"], second["learners"])
    ]
