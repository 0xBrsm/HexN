"""`hexn.serve`: one process runs every net's forward for its clients.

A served read is the in-process read of the same net on the batch the server
ran, so a lone client asking one batch at a time gets the local answer bit for
bit; joined batches split back exactly into their parts.
"""
from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hexset import decisions as D  # noqa: E402
from hexset.actions import apply, options_for, space_for  # noqa: E402
from hexset.arena import deal_game  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.game import Phase, to_move  # noqa: E402
from hexn import netbot  # noqa: E402
from hexn.model import HexNet as HexNet8, ModelConfig as ModelConfig8  # noqa: E402
from hexn.model9 import HexNet as HexNet9, ModelConfig as ModelConfig9  # noqa: E402
from hexn.serve import Panel, Server, join_arrays, split_heads  # noqa: E402

SEED = 20260909


def checkpoints(tmp_path):
    """A small contract-9 net and a small contract-8 net, saved as trainers save them."""
    game = deal_game(SEED, 0, 4)
    topology = game._state.board.topology
    torch.manual_seed(0)
    config = ModelConfig9(width=16, rounds=1, contract="9", double_turn="parts", attention=False,
                          messages="pooled")
    net9 = HexNet9(space_for(game), topology, 4, config).eval()
    nine = tmp_path / "nine.pt"
    torch.save({"net": net9.state_dict(), "model_config": asdict(config),
                "args": {"contract": "9", "players": 4, "max_offers": 0}, "iteration": 1}, nine)
    net8 = HexNet8(space_for(game), static_graph(topology), 4, ModelConfig8(width=16, rounds=1))
    eight = tmp_path / "eight.pt"
    torch.save({"iteration": 1, "net": net8.state_dict(),
                "args": {"players": 4, "width": 16, "rounds": 1, "max_offers": 0}}, eight)
    return [str(nine), str(eight)], topology


def positions(count=40):
    rows = []
    for index in range(4):
        game = deal_game(SEED, index, 4)
        walk = random.Random(index)
        for step in range(400):
            if game.phase is Phase.GAME_OVER or len(rows) >= count * (index + 1) // 4:
                break
            options, seat = tuple(options_for(game)), to_move(game)
            if len(options) > 1 and step % 5 == 0:
                import copy
                rows.append((copy.deepcopy(game), seat, options))
            apply(game, options[walk.randrange(len(options))], seat)
    return rows


def same(a, b):
    assert len(a) == len(b)
    for (p, v), (q, w) in zip(a, b):
        assert np.array_equal(np.asarray(p), np.asarray(q)) and tuple(v) == tuple(w)


def test_joined_arrays_and_split_heads_are_their_parts(tmp_path):
    paths, topology = checkpoints(tmp_path)
    policy = netbot.load_torch(paths[0], topology).policy
    rows = positions()
    parts = [rows[:7], rows[7:8], rows[8:25], rows[25:]]
    feeds = [policy.score_feed(part) for part in parts]
    joined, counts = join_arrays([arrays for _, arrays in feeds])
    whole_obs, whole = policy.score_feed(rows)
    assert np.array_equal(np.concatenate([obs for obs, _ in feeds]), whole_obs)
    for key in whole:
        assert np.array_equal(joined[key], whole[key]), key
    heads = policy._heads(whole_obs, whole, prior=True)
    split = split_heads(heads, counts)
    for name in ("value", "flat", "setup", "prior", "road_building", "discard"):
        assert np.array_equal(np.concatenate([getattr(h, name) for h in split]), getattr(heads, name)), name
    assert [len(h.value) for h in split] == [len(p) for p in parts]


@pytest.mark.parametrize("gather", [0.0, 0.02])
def test_a_lone_client_reads_every_net_as_it_reads_in_process(tmp_path, gather):
    paths, topology = checkpoints(tmp_path)
    rows = positions()
    local = [netbot.load_torch(path, topology).policy for path in paths]
    server = Server.start(paths, topology, clients=2, device="cpu", gather=gather)
    try:
        panel, other = Panel(server.connection(0), paths, topology), Panel(server.connection(1), paths, topology)
        for chunk in (rows[:1], rows[:9], rows):
            reads = panel.score_rows_all(chunk)
            for policy, read in zip(local, reads):
                same(read, policy.score_rows(chunk))
        same(other.policy(1).score_rows(rows[:5]), local[1].score_rows(rows[:5]))
        assert panel.score_rows_all([]) == [[], []]
        same(panel.score_rows_all(rows[:3], [1])[0], local[1].score_rows(rows[:3]))
        stats = panel.stop()
        other.stop()
        assert stats["rounds"] >= 4 and stats["requests"] >= 7
        # One client asked at a time, so each net's ask was its own forward.
        assert stats["forwards"] == stats["requests"]
        assert stats["rows"] == 2 * (1 + 9 + len(rows)) + 5 + 3
    finally:
        server.stop()
