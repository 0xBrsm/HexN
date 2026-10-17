"""`hexn.model9`: the contract-9 network and the decision loss."""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hexset import contract9, decisions as D  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.game import Phase, is_over, players_owing_discards, start, to_move  # noqa: E402
from hexn.model9 import HexNet, decision_log_prob, unpack9  # noqa: E402


def _positions(count=60, seed=0):
    """Positions from random play on decisions, with the decision at each."""
    rng = random.Random(seed)
    out = []
    for g in range(4):
        game = start(random_base_board(random.Random(g)), 4, random.Random(g))
        space = space_for(game)
        while not is_over(game) and len(out) < count * (g + 1) // 4:
            seat = players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game)
            options = D.options(game, seat)
            decision = D.tensors(game, seat, space)
            choice = rng.choice(options)
            kind, index = D.locate(decision, choice, game, seat, space)
            out.append((contract9.flatten(contract9.encode9(game, seat)), decision, int(kind), index))
            D.apply_option(game, choice, seat if game.phase is Phase.DISCARD else None)
    return out, game


def _batch(rows):
    """A batch as a trainer assembles one: `decision_log_prob`'s inputs."""
    batch = {
        "x": torch.from_numpy(np.stack([r[0] for r in rows])),
        "mask": torch.from_numpy(np.stack([r[1].flat for r in rows])),
        "setup": torch.from_numpy(np.stack([r[1].setup for r in rows])),
        "double": torch.tensor([r[1].double for r in rows]),
        "tkind": torch.tensor([r[2] for r in rows]),
        "y": torch.tensor([r[3] for r in rows]),
    }
    for name, field, width in (("rb", "road_building", 2), ("disc", "discard", 10)):
        values = [getattr(r[1], field) for r in rows]
        batch[name] = torch.from_numpy(np.concatenate(values).reshape(-1, width).astype(np.int64))
        batch[f"{name}_row"] = torch.tensor([i for i, v in enumerate(values) for _ in range(len(v))], dtype=torch.long)
        batch[f"{name}_index"] = torch.tensor([j for v in values for j in range(len(v))], dtype=torch.long)
    return batch


@pytest.fixture(scope="module")
def setting():
    rows, game = _positions()
    torch.manual_seed(0)
    net = HexNet(space_for(game), game._state.board.topology, 4).eval()
    return net, rows, contract9.layout(4, 19, 54, 72)


def test_the_decision_loss_is_the_softmax_over_every_legal_option(setting):
    net, rows, layout = setting
    batch = _batch(rows)
    with torch.no_grad():
        out = net(**unpack9(layout, batch["x"]))
        log_prob, hit = decision_log_prob(net, out, batch)
        logits = net.full_logits(out)
        setup_s, first_s, second_s, pair_s = net.setup_heads(out, torch.arange(len(rows)))
    pairs = net.pairs_ok
    for i, (_, decision, kind, index) in enumerate(rows):
        scores, chosen = [], None
        for j in np.flatnonzero(decision.flat):
            scores.append(logits[i, j])
            chosen = logits[i, j] if kind == D.DecisionKind.FLAT and j == index else chosen
        legal = np.flatnonzero(decision.setup)
        if decision.double:
            for a in legal:
                for b in legal:
                    if pairs[a, b]:
                        s = first_s[i, a] + second_s[i, b] + pair_s[i, net.inc_vertex[a], net.inc_vertex[b]]
                        scores.append(s)
                        chosen = s if a * len(pairs) + b == index else chosen
        else:
            for a in legal:
                scores.append(setup_s[i, a])
                chosen = setup_s[i, a] if kind == D.DecisionKind.SETUP and a == index else chosen
        for name, method, k in (("rb", net.road_building_scores, D.DecisionKind.ROAD_BUILDING),
                                ("disc", net.discard_scores, D.DecisionKind.DISCARD)):
            sel = batch[f"{name}_row"] == i
            for c, s in enumerate(method(out, batch[f"{name}_row"][sel], batch[name][sel])):
                scores.append(s)
                chosen = s if kind == k and c == index else chosen
        want = chosen - torch.logsumexp(torch.stack(scores), 0)
        assert abs(float(want) - float(log_prob[i])) < 1e-4
        assert bool(hit[i]) == bool(chosen >= torch.stack(scores).max())
    assert torch.isfinite(log_prob).all() and (log_prob <= 0).all()


def test_the_seats_are_read_by_the_same_weights(setting):
    """Relabel the other three seats: the win head's reads move with them."""
    net, rows, layout = setting
    x = torch.from_numpy(np.stack([r[0] for r in rows[:16]]))
    inputs = unpack9(layout, x)
    order = torch.tensor([0, 2, 3, 1])                  # new row i reads old row order[i]
    inverse = torch.argsort(order)
    swapped = dict(inputs)
    swapped["seats"] = inputs["seats"][:, order]
    swapped["reach"] = inputs["reach"][:, order]
    for name in ("vertex_owner", "road_owner"):
        owner = inputs[name]
        swapped[name] = torch.where(owner >= 0, inverse[owner.clamp(min=0)], owner)
    with torch.no_grad():
        a = net(**inputs).value
        b = net(**swapped).value
        value_only = net(**inputs, policy=False).value
    torch.testing.assert_close(b, a[:, order], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(value_only, a)


def test_a_batch_trains(setting):
    _, rows, layout = setting
    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    torch.manual_seed(1)
    net = HexNet(space_for(game), game._state.board.topology, 4)
    batch = _batch(rows)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        out = net(**unpack9(layout, batch["x"]))
        loss = -decision_log_prob(net, out, batch)[0].mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.75 * losses[0]


def test_heads_are_linear_unless_a_checkpoint_says_mlp():
    from hexn.model9 import ModelConfig, config_of

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    linear = HexNet(space_for(game), game._state.board.topology, 4)
    assert isinstance(linear.road_head, torch.nn.Linear) and isinstance(linear.value_head, torch.nn.Linear)
    assert isinstance(linear.discard_head, torch.nn.Sequential)      # the discard keeps its hidden layer
    mlp = HexNet(space_for(game), game._state.board.topology, 4, ModelConfig(heads="mlp"))
    assert isinstance(mlp.road_head, torch.nn.Sequential)
    # c9-bc's first run wrote its model config before `heads` existed.
    assert config_of({"width": 64, "rounds": 3, "contract": "9"}).heads == "mlp"
    assert config_of({"width": 64, "rounds": 3, "heads": "linear"}).heads == "linear"
    assert sum(p.numel() for p in linear.parameters()) < sum(p.numel() for p in mlp.parameters())


def test_a_linear_discard_head_reads_the_position(setting):
    """The same candidates in two positions rank differently: the context sets
    each resource's weight rather than adding one constant to every score."""
    from hexn.model9 import ModelConfig

    _, rows, layout = setting
    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    torch.manual_seed(2)
    net = HexNet(space_for(game), game._state.board.topology, 4, ModelConfig(discard="linear"))
    torch.nn.init.normal_(net.discard_head.weight)
    candidates = torch.tensor([[5, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 5, 5, 0, 0, 0, 0]], dtype=torch.int64)
    x = torch.from_numpy(np.stack([rows[0][0], rows[-1][0]]))
    with torch.no_grad():
        out = net(**unpack9(layout, x))
        a = net.discard_scores(out, torch.tensor([0, 0]), candidates)
        b = net.discard_scores(out, torch.tensor([1, 1]), candidates)
    assert not torch.allclose(a[0] - a[1], b[0] - b[1])


def test_layernorm_is_on_unless_a_checkpoint_predates_it():
    from hexn.model9 import ModelConfig, config_of

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    net = HexNet(space_for(game), game._state.board.topology, 4)
    assert isinstance(net.rounds[0].norm_vertex, torch.nn.LayerNorm) and isinstance(net.out_seat, torch.nn.LayerNorm)
    assert config_of({"width": 64, "rounds": 3, "contract": "9"}).norm is False
    plain = HexNet(space_for(game), game._state.board.topology, 4, ModelConfig(norm=False))
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in plain.modules())
    # Tokens leave the trunk at unit scale, whatever their inputs' size.
    rows, _ = _positions(count=8)
    inputs = unpack9(contract9.layout(4, 19, 54, 72), torch.from_numpy(np.stack([r[0] for r in rows])))
    for name in ("seats", "table", "hexes", "vertices"):
        inputs[name] = inputs[name] * 50
    with torch.no_grad():
        out = net(**inputs)
    assert float(out.seats.std(-1).mean()) < 2.0


def test_pooled_messages_read_every_seat_with_the_same_weights_and_train():
    from hexn.model9 import ModelConfig

    rows, game = _positions(count=40, seed=3)
    torch.manual_seed(4)
    net = HexNet(space_for(game), game._state.board.topology, 4, ModelConfig(heads="linear", messages="pooled"))
    layout = contract9.layout(4, 19, 54, 72)
    inputs = unpack9(layout, torch.from_numpy(np.stack([r[0] for r in rows[:16]])))
    order = torch.tensor([0, 2, 3, 1])
    inverse = torch.argsort(order)
    swapped = dict(inputs, seats=inputs["seats"][:, order], reach=inputs["reach"][:, order])
    for name in ("vertex_owner", "road_owner"):
        owner = inputs[name]
        swapped[name] = torch.where(owner >= 0, inverse[owner.clamp(min=0)], owner)
    net.eval()
    with torch.no_grad():
        torch.testing.assert_close(net(**swapped).value, net(**inputs).value[:, order], atol=1e-5, rtol=1e-4)
    net.train()
    batch = _batch(rows)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        loss = -decision_log_prob(net, net(**unpack9(layout, batch["x"])), batch)[0].mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.75 * losses[0]


@pytest.mark.parametrize("messages,attention", [("separate", True), ("pooled", True), ("separate", False), ("pooled", False)])
def test_the_fused_trunk_computes_the_reference(messages, attention):
    from hexn.model9 import ModelConfig

    rows, game = _positions(count=24, seed=5)
    torch.manual_seed(6)
    config = ModelConfig(heads="linear", messages=messages, attention=attention)
    net = HexNet(space_for(game), game._state.board.topology, 4, config).eval()
    for p in net.parameters():                       # off its zero start, so every path carries signal
        p.data.add_(0.05 * torch.randn_like(p))
    inputs = unpack9(contract9.layout(4, 19, 54, 72), torch.from_numpy(np.stack([r[0] for r in rows])))
    with torch.no_grad():
        net.fused = False
        reference = net(**inputs)
        net.fused = True
        fused = net(**inputs)
    for name in ("value_logits", "logits", "road", "vertices", "seats", "hexes", "table"):
        torch.testing.assert_close(getattr(fused, name), getattr(reference, name), atol=2e-4, rtol=2e-4, msg=name)


def test_dropping_attention_removes_its_parameters():
    from hexn.model9 import ModelConfig, config_of

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    def size(**kw):
        return sum(p.numel() for p in HexNet(space_for(game), game._state.board.topology, 4,
                                             ModelConfig(heads="linear", messages="pooled", **kw)).parameters())
    assert size() - size(attention=False) == 3 * (3 * 64 * 64 + 3 * 64 + 64 * 64)   # query, key, value; one seat input
    assert config_of({"width": 64, "rounds": 3}).attention is True


def test_the_board_heads_read_the_trunk_with_nothing_added():
    """No projection of the perspective's token is added to the board's tokens
    before the heads read them: what the heads score is the trunk's own."""
    rows, game = _positions(count=8, seed=7)
    net = HexNet(space_for(game), game._state.board.topology, 4).eval()
    assert not any("me_" in name for name, _ in net.named_parameters())
    assert net.global_head.in_features == 64


def test_the_points_head_reads_each_seat_with_the_table_and_is_off_unless_asked():
    from hexn.model9 import ModelConfig, config_of

    rows, game = _positions(count=16, seed=8)
    topology, space = game._state.board.topology, space_for(game)
    assert config_of({"width": 64, "rounds": 3}).points is False
    torch.manual_seed(9)
    plain = HexNet(space, topology, 4, ModelConfig(messages="pooled"))
    torch.manual_seed(9)
    net = HexNet(space, topology, 4, ModelConfig(messages="pooled", points=True)).eval()
    assert plain.points_head is None
    assert (sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in plain.parameters())
            == 2 * 64 + 1)                                  # one linear head on seat token and table
    inputs = unpack9(contract9.layout(4, 19, 54, 72), torch.from_numpy(np.stack([r[0] for r in rows])))
    with torch.no_grad():
        out, value_only = net(**inputs), net(**inputs, policy=False)
        net.fused = False
        reference = net(**inputs)
    assert out.points.shape == (len(rows), 4) and out.points.dtype == torch.float32
    torch.testing.assert_close(value_only.points, out.points)
    torch.testing.assert_close(reference.points, out.points, atol=2e-4, rtol=2e-4)
    # Read seat by seat with the same weights: permuting the seats permutes it.
    order = torch.tensor([0, 2, 3, 1])
    inverse = torch.argsort(order)
    swapped = dict(inputs, seats=inputs["seats"][:, order], reach=inputs["reach"][:, order])
    for name in ("vertex_owner", "road_owner"):
        owner = inputs[name]
        swapped[name] = torch.where(owner >= 0, inverse[owner.clamp(min=0)], owner)
    net.fused = True
    with torch.no_grad():
        torch.testing.assert_close(net(**swapped).points, out.points[:, order], atol=1e-5, rtol=1e-4)
    # A net without the head loads into one built from its config; one with it does not lose it.
    state = plain.state_dict()
    again = HexNet(space, topology, 4, config_of({"width": 64, "rounds": 3, "norm": True, "heads": "linear",
                                                  "messages": "pooled"}))
    again.load_state_dict(state)
    assert again.points_head is None
