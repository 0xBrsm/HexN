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
from hexn import model9  # noqa: E402
from hexn.model9 import HexNet, _segment_logsumexp, decision_log_prob, unpack9  # noqa: E402


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


@pytest.mark.parametrize("messages,attention,others", [
    ("separate", True, False), ("pooled", True, False), ("separate", False, False), ("pooled", False, False),
    ("pooled", False, True), ("separate", False, True), ("pooled", True, True),
])
def test_the_fused_trunk_computes_the_reference(messages, attention, others):
    from hexn.model9 import ModelConfig

    rows, game = _positions(count=24, seed=5)
    torch.manual_seed(6)
    config = ModelConfig(heads="linear", messages=messages, attention=attention, others=others)
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
    # The others read is one more seat input a round and nothing else; off unless a checkpoint says so.
    assert size(attention=False, others=True) - size(attention=False) == 3 * 64 * 64
    assert config_of({"width": 64, "rounds": 3}).others is False


def _duel_positions(count=40, seed=0, contract="9d"):
    """Positions from random play at a two-seat duel-variant table, encoded in a two-seat contract."""
    from hexset.rules import DUEL_VARIANT_GAME

    rng = random.Random(seed)
    game = start(random_base_board(random.Random(seed)), 2, random.Random(seed), game_type=DUEL_VARIANT_GAME)
    out = []
    while not is_over(game) and len(out) < count:
        seat = players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game)
        choice = rng.choice(D.options(game, seat))
        out.append(contract9.flatten(contract9.encode9(game, seat, contract)))
        D.apply_option(game, choice, seat if game.phase is Phase.DISCARD else None)
    return out, game


def test_a_two_seat_net_reads_its_opponents_token_directly():
    """On contract 9d at two seats the others read is the one opponent's
    normalised token, and both trunks compute it alike."""
    from hexset.rules import DUEL_VARIANT_GAME
    from hexn.model9 import ModelConfig, _others

    rows, game = _duel_positions()
    torch.manual_seed(2)
    config = ModelConfig(heads="linear", messages="pooled", attention=False, others=True, contract="9d")
    net = HexNet(space_for(game), game._state.board.topology, 2, config).eval()
    for p in net.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    assert net.embed_hex.in_features == 8 and net.embed_seat.in_features == 22 and net.embed_table.in_features == 20
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "9d"), torch.from_numpy(np.stack(rows)))
    with torch.no_grad():
        sn = net.rounds[0].norm_seat(net.embed_seat(inputs["seats"]))
        read = _others(sn, 1, 2)
        torch.testing.assert_close(read[:, 0], sn[:, 1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(read[:, 1], sn[:, 0], atol=1e-5, rtol=1e-5)
        net.fused = False
        reference = net(**inputs)
        net.fused = True
        fused = net(**inputs)
    for name in ("value_logits", "logits", "road", "vertices", "seats", "hexes", "table"):
        torch.testing.assert_close(getattr(fused, name), getattr(reference, name), atol=2e-4, rtol=2e-4, msg=name)
    assert fused.value.shape == (len(rows), 2) and torch.allclose(fused.value.sum(-1), torch.ones(len(rows)))


def test_a_9d_net_loads_into_9d1_with_the_seat_read_global_head_and_plays_as_before():
    """A 9d checkpoint loads into a 9d.1 net whose global head also reads the
    perspective's own seat token: the economy columns' embedding and the
    global head's seat weights start at zero, so on the same positions every
    output is the 9d net's, fused and reference alike. Once those weights
    move, the global logits read the seat token and the economy columns."""
    from dataclasses import replace

    from hexn.model9 import ModelConfig, config_of

    rows9d, game = _duel_positions(contract="9d")
    rows, _ = _duel_positions(contract="9d.1")
    torch.manual_seed(3)
    old = ModelConfig(heads="linear", messages="pooled", attention=False, others=True, contract="9d")
    source = HexNet(space_for(game), game._state.board.topology, 2, old).eval()
    for p in source.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    config = replace(old, contract="9d.1", global_seat=True)
    net = HexNet(space_for(game), game._state.board.topology, 2, config).eval()
    net.load_state_dict(source.state_dict())
    width = config.width
    assert net.embed_seat.in_features == 32 and net.global_head.in_features == 2 * width
    assert config_of({"width": 64, "rounds": 3}).global_seat is False
    before = unpack9(contract9.layout(2, 19, 54, 72, "9d"), torch.from_numpy(np.stack(rows9d)))
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "9d.1"), torch.from_numpy(np.stack(rows)))
    assert (inputs["seats"][..., 22:] != 0).any()
    with torch.no_grad():
        want = source(**before)
        for fused in (False, True):
            net.fused = fused
            got = net(**inputs)
            for name in ("value_logits", "logits", "road", "vertices", "seats", "hexes", "table"):
                torch.testing.assert_close(getattr(got, name), getattr(want, name), atol=2e-4, rtol=2e-4,
                                           msg=f"{name} fused={fused}")
        slots = net.slot_global
        net.global_head.weight[:, width:].normal_(0, 0.5)
        moved = net(**inputs).logits[:, slots]
        assert not torch.allclose(moved, want.logits[:, slots], atol=1e-3)
        net.embed_seat.weight[:, 22:].normal_(0, 0.5)
        assert not torch.allclose(net(**inputs).logits[:, slots], moved, atol=1e-3)


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


def _old_chosen_scores(scores, batch):
    """`chosen_scores` as it read the chosen candidate by boolean masks: the
    reference the stall-free version is held to."""
    from hexn.model9 import DecisionKind
    neg = float("-inf")
    flat, setup = scores.flat, scores.setup
    tkind, y = batch["tkind"].long(), batch["y"].long()
    chosen = flat.new_full((flat.shape[0],), neg)
    chosen = torch.where(tkind == DecisionKind.FLAT, flat.gather(1, y.clamp(max=flat.shape[1] - 1)[:, None]).squeeze(1),
                         chosen)
    chosen = torch.where(tkind == DecisionKind.SETUP, setup.gather(1, y.clamp(max=setup.shape[1] - 1)[:, None]).squeeze(1),
                         chosen)
    if scores.rb.numel():
        taken = (tkind[scores.rb_row] == DecisionKind.ROAD_BUILDING) & (batch["rb_index"] == y[scores.rb_row])
        chosen = chosen.index_put((scores.rb_row[taken],), scores.rb[taken])
    if scores.disc.numel():
        taken = (tkind[scores.disc_row] == DecisionKind.DISCARD) & (batch["disc_index"] == y[scores.disc_row])
        chosen = chosen.index_put((scores.disc_row[taken],), scores.disc[taken])
    if len(scores.double_rows):
        d = scores.double_rows
        taken = tkind[d] == DecisionKind.DOUBLE
        picked = scores.double.gather(1, y[d].clamp(max=scores.double.shape[1] - 1)[:, None]).squeeze(1)
        chosen = chosen.index_put((d[taken],), picked[taken])
    return chosen


def _losses_and_grads(net, layout, batch, scorer=None):
    """The decision loss's log-probabilities and every parameter's gradient,
    on one thread: several make the backward's sums run in varying order, so
    the same code differs from itself in the last bits."""
    from hexn import model9
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _losses_and_grads_one_thread(net, layout, batch, scorer or model9.chosen_scores)
    finally:
        torch.set_num_threads(threads)


def _losses_and_grads_one_thread(net, layout, batch, scorer):
    from hexn import model9
    net.zero_grad(set_to_none=True)
    out = net(**unpack9(layout, batch["x"]))
    scores = model9.decision_scores(net, out, batch)
    chosen = scorer(scores, batch)
    log_prob = chosen - scores.total
    (-log_prob[torch.isfinite(log_prob)].mean()).backward()
    return log_prob.detach(), {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}


def test_rows_found_on_the_host_give_the_same_loss_and_gradients(setting):
    """`host_rows` names exactly the rows `decision_scores` finds itself, so a
    batch carrying them scores, and trains, identically."""
    from hexn.model9 import host_rows
    net, _, layout = setting
    rows, _ = _positions(count=400, seed=1)
    batch = _batch(rows)
    found = host_rows(net, batch)
    assert len(found["robber_rows"]) and len(found["setup_rows"])
    plain, grads = _losses_and_grads(net, layout, batch)
    hosted, grads_hosted = _losses_and_grads(net, layout, {**batch, **found})
    assert torch.equal(plain, hosted)
    assert grads.keys() == grads_hosted.keys()
    for name in grads:
        assert torch.equal(grads[name], grads_hosted[name]), name


def test_the_chosen_candidate_reads_without_masks_as_it_read_with_them(setting):
    """The stall-free `chosen_scores` gives the boolean-mask version's scores
    and gradients, Road Building, discard and double-turn rows included."""
    from hexn import model9
    from hexn.model9 import DecisionKind
    net, _, layout = setting
    rows, _ = _positions(count=400, seed=1)
    batch = _batch(rows)
    kinds = batch["tkind"].tolist()
    assert DecisionKind.DISCARD in kinds or DecisionKind.ROAD_BUILDING in kinds or DecisionKind.DOUBLE in kinds
    new, grads_new = _losses_and_grads(net, layout, batch)
    old, grads_old = _losses_and_grads(net, layout, batch, _old_chosen_scores)
    assert torch.equal(new, old)
    for name in grads_old:
        assert torch.equal(grads_new[name], grads_old[name]), name
    # Every kind, forced: each row's taken candidate moved to every position.
    torch.manual_seed(3)
    scores = model9.Scores(flat=torch.randn(6, 5), setup=torch.randn(6, 4), rb=torch.randn(7, requires_grad=True),
                           rb_row=torch.tensor([0, 0, 1, 1, 1, 4, 4]), disc=torch.randn(5, requires_grad=True),
                           disc_row=torch.tensor([2, 2, 3, 5, 5]), double_rows=torch.tensor([1, 4, 5]),
                           double=torch.randn(3, 9, requires_grad=True), total=torch.zeros(6))
    for case in range(12):
        tkind = torch.tensor([[DecisionKind.ROAD_BUILDING, DecisionKind.FLAT, DecisionKind.DISCARD, DecisionKind.SETUP,
                               DecisionKind.DOUBLE, DecisionKind.DISCARD][(i + case) % 6] for i in range(6)])
        b = {"tkind": tkind, "y": torch.tensor([(case + i) % 4 for i in range(6)]),
             "rb_index": torch.tensor([0, 1, 0, 1, 2, 0, 1]), "disc_index": torch.tensor([0, 1, 0, 0, 1])}
        for tensor in (scores.rb, scores.disc, scores.double):
            tensor.grad = None
        a = model9.chosen_scores(scores, b)
        a[torch.isfinite(a)].sum().backward()
        ga = [t.grad.clone() if t.grad is not None else None for t in (scores.rb, scores.disc, scores.double)]
        for tensor in (scores.rb, scores.disc, scores.double):
            tensor.grad = None
        r = _old_chosen_scores(scores, b)
        r[torch.isfinite(r)].sum().backward()
        gr = [t.grad.clone() if t.grad is not None else None for t in (scores.rb, scores.disc, scores.double)]
        assert torch.equal(a, r), case
        for x, z in zip(ga, gr):
            assert (x is None and (z is None or not z.any())) or (z is None and not x.any()) or torch.equal(x, z), case


# --- contract 11d: two seat tokens of their own widths, no table token ----------


def _duel_decisions(count=60, seed=0, contract="11d", every=6, double_turn=False):
    """Positions from random two-seat duel-variant play with the decision at
    each, every `every`th decision of whole games, so discards and Road
    Building come up beside the setup; with `double_turn` the snake seat's
    two setup turns are asked as one decision."""
    from hexset.rules import DUEL_VARIANT_GAME

    rng = random.Random(seed)
    out = []
    g = 0
    while len(out) < count:
        game = start(random_base_board(random.Random(seed + g)), 2, random.Random(seed + g),
                     game_type=DUEL_VARIANT_GAME)
        space = space_for(game)
        step = 0
        while not is_over(game) and len(out) < count:
            seat = players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game)
            options = D.options(game, seat, double_turn=double_turn)
            choice = rng.choice(options)
            if step % every == 0 or game.phase is Phase.DISCARD or any(isinstance(o, D.Compound) for o in options):
                decision = D.tensors(game, seat, space, double_turn=double_turn)
                kind, index = D.locate(decision, choice, game, seat, space)
                out.append((contract9.flatten(contract9.encode9(game, seat, contract)), decision, int(kind), index))
            D.apply_option(game, choice, seat if game.phase is Phase.DISCARD else None)
            step += 1
        g += 1
    return out, game


def _two_seat_net(game, seed, **kwargs):
    from hexn.model9 import ModelConfig

    torch.manual_seed(seed)
    config = ModelConfig(**{"heads": "linear", "messages": "pooled", "attention": False, "others": True,
                            "double_turn": "parts", "contract": "11d", **kwargs})
    net = HexNet(space_for(game), game._state.board.topology, 2, config).eval()
    for p in net.parameters():                       # off its zero start, so every path carries signal
        p.data.add_(0.05 * torch.randn_like(p))
    return net


def test_an_11d_net_has_two_seat_embeddings_a_board_token_and_query_key_heads():
    """A contract-11d net embeds my token (35 columns) and the opponent's
    (27) by layers of their own, the board block (3) as its board token, and
    reads 17-wide hexes. Its heads are a type score per move type and a
    query per role from my token, the opponent's and the board token, read
    against vertex, edge and robber keys; no global, setup, joint
    double-turn, discard, Road Building or points head; the move head has a
    hidden layer, a hand value scores the resource moves, and Road
    Building's two roads read a pair matrix. 332,281 parameters at width 64,
    3 rounds, pooled, the others read on and attention off; a "joint" net
    adds the 16 x 16 double-turn pair matrix, 332,537."""
    from hexn.model9 import QUERY_ROLES, MOVE_TYPES, ModelConfig

    rows, game = _duel_decisions(count=24)
    net = _two_seat_net(game, 4)
    width = net.config.width
    assert net.embed_me.in_features == 35 and net.embed_other.in_features == 27 and net.embed_hex.in_features == 17
    assert net.embed_board.in_features == 3 and not hasattr(net, "embed_table")
    assert all(r.board and r.table[0].in_features == 4 * width + 16 for r in net.rounds)
    for gone in ("vertex_head", "road_head", "robber_head", "nobody_head", "global_head", "setup_head", "first_head",
                 "second_head", "pair_head", "discard_head", "road_building", "points_head"):
        assert getattr(net, gone) is None, gone
    assert len(MOVE_TYPES) == 15 and len(QUERY_ROLES) == 8
    assert net.move_head[0].in_features == 3 * width and net.move_head[0].out_features == 64
    assert net.move_head[-1].out_features == 15
    assert net.query_head.in_features == 3 * width and net.query_head.out_features == 8 * 16
    assert net.hand_value[0].in_features == 3 * width + 10 and net.hand_value[-1].out_features == 1
    assert net.road_pair.shape == (16, 16) and net.query_gain.shape == (8, 16) and net.key_gain.shape == (3, 16)
    assert sum(p.numel() for p in net.parameters()) == 332_281 and net.pair_matrix is None
    joint = _two_seat_net(game, 4, double_turn="joint")
    assert sum(p.numel() for p in joint.parameters()) == 332_537 and joint.pair_matrix.shape == (16, 16)
    for refused in (dict(global_seat=True), dict(points=True)):
        with pytest.raises(ValueError):
            ModelConfig(contract="11d", **refused)
    with pytest.raises(ValueError, match="two-seat"):
        HexNet(space_for(game), game._state.board.topology, 4, net.config)
    layout = contract9.layout(2, 19, 54, 72, "11d")
    inputs = unpack9(layout, torch.from_numpy(np.stack([r[0] for r in rows])))
    assert inputs["seats"].shape == (len(rows), 62) and inputs["table"].shape == (len(rows), 3)
    with torch.no_grad():
        for fused in (False, True):
            net.fused = fused
            out = net(**inputs)
            assert out.table.shape == (len(rows), width) and out.points is None and out.seats.shape == (len(rows), 2, width)
            me, other = out.seats[:, 0], out.seats[:, 1]
            mine = net.value_head(torch.cat([me, me], -1)).squeeze(-1)
            theirs = net.value_head(torch.cat([other, me], -1)).squeeze(-1)
            torch.testing.assert_close(out.value_logits, torch.stack([mine, theirs], 1))
            # The opponent's columns reach every output through its own layer.
            changed = dict(inputs, seats=inputs["seats"].clone())
            changed["seats"][:, 35:] += 0.5
            assert not torch.allclose(net(**changed).logits, out.logits, atol=1e-4)
            # The board block reaches every output through the board token.
            changed = dict(inputs, table=inputs["table"].clone())
            changed["table"] += 0.5
            assert not torch.allclose(net(**changed).logits, out.logits, atol=1e-4)
            assert not torch.allclose(net(**changed).value_logits, out.value_logits, atol=1e-5)


def _unit_ref(x, gain):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5) * gain


def _query_reference(net, out, seats, game):
    """Every flat action's 11d score, one action at a time from the net's own
    layers (`hexn.model9._emit_queries`' formulas written out per option),
    and the formula each used: the check that each option kind is scored by
    exactly one formula. A query . key is the layer-normalised query (with
    its gain) and key (with its kind's gain), over sqrt(16), times the role's
    exp(scale); a resource move's hand value reads the context and the two
    hands it leaves."""
    from hexset.actions import YEAR_OF_PLENTY_PAIRS, ActionType as T
    from hexset.contract9 import MY_COLUMNS_11D as M, OTHER_COLUMNS_11D as O
    from hexn.model9 import QUERY_ROLES, KEY_WIDTH

    space = net.space
    topology = game._state.board.topology
    rows = out.seats.shape[0]
    me, other, h, v = out.seats[:, 0], out.seats[:, 1], out.hexes, out.vertices
    context = torch.cat([me, other, out.table], -1)
    types = net.move_head(context)
    raw = net.query_head(context).reshape(rows, len(QUERY_ROLES), KEY_WIDTH)
    q = {role: _unit_ref(raw[:, i], net.query_gain[i]) * net.role_scales()[i] / 4
         for i, role in enumerate(QUERY_ROLES)}

    def dot(role, key):
        return (q[role] * key).sum(-1)

    def k_vertex(x):
        return _unit_ref(net.vertex_key(v[:, x]), net.key_gain[0])

    def k_edge(e):
        a, b = topology.edges[e]
        both = net.edge_key(torch.cat([v[:, a], v[:, b]], -1)) + net.edge_key(torch.cat([v[:, b], v[:, a]], -1))
        return _unit_ref(both, net.key_gain[1])

    def k_robber(hex_, victim):
        if victim == 2:
            return _unit_ref(net.nobody_key(h[:, hex_]), net.key_gain[2])
        return _unit_ref(net.robber_key(torch.cat([h[:, hex_], out.seats[:, victim]], -1)), net.key_gain[2])

    scores, used = torch.zeros(rows, space.size), {}
    resource = (T.BANK_TRADE, T.PLAY_YEAR_OF_PLENTY, T.PLAY_MONOPOLY, T.DISCARD)
    for index in range(space.size):
        action = space.decode(index)
        t, a, b = action.type, action.a, action.b
        # A setup move has no type score: its position offers nothing else.
        base = torch.zeros(rows) if t in model9.UNTYPED else types[:, int(t)]
        if t in resource:
            continue
        if t in (T.ROLL, T.END_TURN, T.BUY_DEV_CARD, T.PLAY_KNIGHT, T.PLAY_ROAD_BUILDING):
            formula, score = "type", base
        elif t in (T.BUILD_SETTLEMENT, T.BUILD_CITY, T.SETUP_SETTLEMENT):
            if t is T.SETUP_SETTLEMENT:
                # The paying query where my token says a settlement is already placed.
                placed = (seats[:, M["pieces"].start] * 10).round() < 5
                score = base + torch.where(placed, dot("setup_settlement_paying", k_vertex(a)),
                                           dot("setup_settlement", k_vertex(a)))
                formula = "setup_settlement"
            else:
                role = {T.BUILD_SETTLEMENT: "settlement", T.BUILD_CITY: "city"}[t]
                formula, score = role, base + dot(role, k_vertex(a))
        elif t in (T.BUILD_ROAD, T.SETUP_ROAD):
            role = "road" if t is T.BUILD_ROAD else "setup_road"
            formula, score = role, base + dot(role, k_edge(a))
        elif t is T.MOVE_ROBBER:
            formula, score = "robber", base + dot("robber", k_robber(a, b))
        else:
            raise AssertionError(f"no 11d formula for {t}")
        used.setdefault(t, set()).add(formula)
        scores[:, index] = score

    mine = seats[:, M["hand"]].clone()                     # counts over 10
    theirs = seats[:, 35 + O["hand"].start:35 + O["hand"].stop].clone()
    rate = seats[:, M["rates"]] * 4                        # cards a bank trade gives

    def value(my_hand, their_hand, at=slice(None)):        # V(after | C), the whole MLP on the whole input
        return net.hand_value(torch.cat([context[at], my_hand, their_hand], -1)).squeeze(-1)

    def card(r, n=1.0):
        out = torch.zeros(5)
        out[r] = n / 10
        return out

    for index in range(space.size):
        action = space.decode(index)
        t, a, b = action.type, action.a, action.b
        if t not in resource:
            continue
        base = torch.zeros(rows) if t in model9.UNTYPED else types[:, int(t)]
        if t is T.BANK_TRADE:
            given = torch.zeros(rows, 5)
            given[:, a] = rate[:, a] / 10
            formula, score = "trade", base + value(mine - given + card(b), theirs)
        elif t is T.PLAY_YEAR_OF_PLENTY:
            first, second = YEAR_OF_PLENTY_PAIRS[a]
            formula, score = "year_of_plenty", base + value(mine + card(first) + card(second), theirs)
        elif t is T.PLAY_MONOPOLY:
            taken = torch.zeros(rows, 5)
            taken[:, a] = theirs[:, a]
            formula, score = "monopoly", base + value(mine + taken, theirs - taken)
        else:
            formula, score = "discard", base + value(mine - card(a), theirs)
        used.setdefault(t, set()).add(formula)
        scores[:, index] = score
    return scores, used, types, q, (k_vertex, k_edge, value, theirs)


def test_every_flat_option_of_the_two_seat_space_is_scored_by_one_11d_formula():
    """Written out one action at a time, each flat slot's score is its type's
    plus its roles' normalised queries against its keys, or its hand value,
    and the net's flat row, robber block included, is that score at every
    slot: every `ActionType` has exactly one formula. The setup turn, Road
    Building and discard candidates are their moves' scores summed, Road
    Building's two roads with their pair term."""
    from hexset.actions import ActionType as T

    rows, game = _duel_decisions(count=60, seed=7)
    net = _two_seat_net(game, 9)
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), torch.from_numpy(np.stack([r[0] for r in rows])))
    batch = _batch(rows)
    with torch.no_grad():
        out = net(**inputs)
        want, used, types, q, (_, k_edge, value, theirs) = _query_reference(net, out, inputs["seats"], game)
        flat = net.flat_scores(out)
        torch.testing.assert_close(flat, want, atol=1e-5, rtol=1e-5)
        assert set(used) == set(T) and all(len(f) == 1 for f in used.values())
        # A setup turn: its settlement's and its road's flat scores, summed.
        at = torch.arange(len(rows))
        pairs = contract9.incidences(game._state.board.topology)
        space = net.space
        setup = torch.stack([flat[:, space.index(net_action(T.SETUP_SETTLEMENT, v))]
                             + flat[:, space.index(net_action(T.SETUP_ROAD, e))] for v, e in pairs], 1)
        torch.testing.assert_close(net.setup_scores(out, at), setup, atol=1e-5, rtol=1e-5)
        # Road Building: the card's type, q_road_building . k_e per road placed,
        # and with two roads k_first' R k_second / 4.
        assert len(batch["rb"]) and len(batch["disc"])
        rb = net.road_building_scores(out, batch["rb_row"], batch["rb"])
        want_rb, single = [], 0
        for row, (e1, e2) in zip(batch["rb_row"].tolist(), batch["rb"].tolist()):
            score = types[row, int(T.PLAY_ROAD_BUILDING)]
            for e in (e1, e2):
                if e >= 0:
                    score = score + (q["road_building"][row] * k_edge(e)[row]).sum()
            if e1 >= 0 and e2 >= 0:
                score = score + k_edge(e1)[row] @ net.road_pair @ k_edge(e2)[row] / 4
            single += e2 < 0
            want_rb.append(score)
        torch.testing.assert_close(rb, torch.stack(want_rb), atol=1e-5, rtol=1e-5)
        row0, (e1, e2) = 0, (0, 5)
        lone = net.road_building_scores(out, torch.tensor([row0]), torch.tensor([[e1, -1]]))
        torch.testing.assert_close(lone, (types[row0, int(T.PLAY_ROAD_BUILDING)]
                                          + (q["road_building"][row0] * k_edge(e1)[row0]).sum()).reshape(1),
                                   atol=1e-5, rtol=1e-5)
        # A whole discard: the value of the hand it keeps, no type (UNTYPED).
        disc = net.discard_scores(out, batch["disc_row"], batch["disc"])
        row = batch["disc_row"]
        kept = batch["disc"][:, :5].float() / 10
        want_disc = value(kept, theirs[row], row)
        torch.testing.assert_close(disc, want_disc, atol=1e-5, rtol=1e-5)
        # A kept hand that is the hand before less what is given.
        np.testing.assert_allclose(kept.numpy(), (out.hands[row, 0] - batch["disc"][:, 5:].float() / 10).numpy(),
                                   atol=1e-6)


def test_11d_query_key_scores_are_bounded_by_the_role_scale():
    """However large the trunk's, the query head's and the keys' weights
    grow, a query . key score is at most 4 exp(scale) times the largest query
    and key gains (each normalised to length 4, their product over 4), so a
    score runs away only through the learned scale; the initial scale is
    `QUERY_SCALE`, and the scale stays under `ROLE_SCALE_MAX` however far its
    parameter grows."""
    from hexset.actions import ActionType as T
    from hexn.model9 import QUERY_SCALE, ROLE_SCALE_MAX

    rows, game = _duel_decisions(count=40, seed=7)
    net = _two_seat_net(game, 5)
    assert torch.allclose(HexNet(space_for(game), game._state.board.topology, 2, net.config).role_scales(),
                          torch.full((8,), QUERY_SCALE))
    grown = HexNet(space_for(game), game._state.board.topology, 2, net.config)
    with torch.no_grad():
        grown.role_scale.fill_(50.0)
    assert grown.role_scales().max() <= ROLE_SCALE_MAX and grown.role_scales().min() > 0.99 * ROLE_SCALE_MAX
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), torch.from_numpy(np.stack([r[0] for r in rows])))
    with torch.no_grad():
        for module in (net.query_head, net.vertex_key, net.edge_key, net.robber_key, net.nobody_key):
            for p in module.parameters():
                p.mul_(1000)
        out = net(**inputs)
        flat = net.flat_scores(out)
        space = net.space
        bound = 4 * net.role_scales().max() * net.query_gain.abs().max() * net.key_gain.abs().max()
        for t in (T.BUILD_SETTLEMENT, T.BUILD_CITY, T.BUILD_ROAD, T.SETUP_SETTLEMENT, T.SETUP_ROAD, T.MOVE_ROBBER):
            block = flat[:, space.offsets[t]:space.offsets[t] + space.sizes[t]]
            if t not in model9.UNTYPED:
                block = block - out.types[:, int(t), None]
            assert block.abs().max() <= bound * 1.001, t
            assert block.abs().max() > 0.1 * bound, t       # the bound is the scale, not a vanishing score
        assert out.rb_road.abs().max() <= bound * 1.001


def test_a_setup_moves_type_score_is_left_out_and_no_choice_changes():
    """A setup position offers setup moves alone, so their type score would
    be shared by every option: an 11d net leaves it out. Put back, at any
    size, it changes no decision's log-probability, setup turns asked whole
    or a move at a time included."""
    import copy

    from hexset.actions import ActionType as T

    rows = []
    for seed in range(3):                            # each game's setup, then some of its play
        rows += _duel_decisions(count=12, seed=seed, every=1)[0]
    more, game = _duel_decisions(count=40, seed=5, every=3)
    rows += more
    net = _two_seat_net(game, 6)
    space = net.space
    setup_slots = torch.zeros(space.size, dtype=torch.bool)
    for t in (T.SETUP_SETTLEMENT, T.SETUP_ROAD):
        setup_slots[space.offsets[t]:space.offsets[t] + space.sizes[t]] = True
    batch = _batch(rows)
    del batch["double"]                              # a "parts" net asks no double turn
    is_setup = batch["setup"].any(1) | (batch["mask"] & setup_slots).any(1)
    assert is_setup.sum() >= 12 and (~is_setup).sum() >= 12
    # The premise: a position with a setup move legal has no other kind legal.
    assert not (batch["mask"][is_setup] & ~setup_slots).any()
    with_type = copy.deepcopy(net)
    for t in (T.SETUP_SETTLEMENT, T.SETUP_ROAD):
        with_type.slot_type[space.offsets[t]:space.offsets[t] + space.sizes[t]] = int(t)
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), batch["x"])
    with torch.no_grad():
        with_type.move_head[2].bias[[int(T.SETUP_SETTLEMENT), int(T.SETUP_ROAD)]] += 50.0
        out, out_typed = net(**inputs), with_type(**inputs)
        assert (out.logits[:, setup_slots] - out_typed.logits[:, setup_slots]).abs().max() > 40
        torch.testing.assert_close(decision_log_prob(net, out, batch),
                                   decision_log_prob(with_type, out_typed, batch), atol=1e-4, rtol=0)


def test_the_robbers_and_a_discards_type_score_is_left_out_and_no_choice_changes():
    """Robber placement and a discard, like a setup turn, are offered alone,
    so an 11d net gives them no type score (`model9.UNTYPED`). Put back, at
    any size, the type changes no decision's log-probability."""
    import copy
    from unittest import mock

    from hexset.actions import ActionType as T
    from hexn import model9

    rows = []
    for seed in range(6):
        rows += _duel_decisions(count=60, seed=seed, every=40)[0]
    more, game = _duel_decisions(count=60, seed=11, every=40)
    rows += more
    net = _two_seat_net(game, 6)
    space = net.space
    slots = {t: torch.zeros(space.size, dtype=torch.bool) for t in (T.MOVE_ROBBER, T.DISCARD)}
    for t, mask in slots.items():
        mask[space.offsets[t]:space.offsets[t] + space.sizes[t]] = True
    batch = _batch(rows)
    del batch["double"]
    robber = (batch["mask"] & slots[T.MOVE_ROBBER]).any(1)
    discard = torch.zeros(len(robber), dtype=torch.bool)
    discard[batch["disc_row"]] = True
    discard |= (batch["mask"] & slots[T.DISCARD]).any(1)
    assert robber.sum() >= 3 and discard.sum() >= 3 and (~(robber | discard)).sum() >= 12
    # The premise: where either is legal, nothing else is.
    assert not (batch["mask"][robber] & ~slots[T.MOVE_ROBBER]).any()
    assert not (batch["mask"][discard] & ~slots[T.DISCARD]).any() and not batch["setup"][robber | discard].any()
    with_type = copy.deepcopy(net)
    with_type.slot_type[slots[T.DISCARD]] = int(T.DISCARD)
    robber_logits, discard_scores = model9.HexNet.robber_logits, model9.HexNet.discard_scores

    def typed_robber(self, out, rows):
        got = robber_logits(self, out, rows)
        return got + out.types[rows, int(T.MOVE_ROBBER)].to(got.dtype)[:, None] if self is with_type else got

    def typed_discard(self, out, rows, candidates):
        got = discard_scores(self, out, rows, candidates)
        return got + out.types[rows, int(T.DISCARD)].to(got.dtype) if self is with_type else got

    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), batch["x"])
    with torch.no_grad(), mock.patch.object(model9.HexNet, "robber_logits", typed_robber), \
            mock.patch.object(model9.HexNet, "discard_scores", typed_discard):
        with_type.move_head[2].bias[[int(T.MOVE_ROBBER), int(T.DISCARD)]] += 50.0
        out, out_typed = net(**inputs), with_type(**inputs)
        rows_all = torch.arange(len(robber))
        assert (net.robber_logits(out, rows_all) - with_type.robber_logits(out_typed, rows_all)).abs().max() > 40
        torch.testing.assert_close(decision_log_prob(net, out, batch)[0],
                                   decision_log_prob(with_type, out_typed, batch)[0], atol=1e-4, rtol=0)


def test_untyped_is_every_move_type_a_position_offers_alone():
    """`model9.UNTYPED` against random duel-variant play, a move at a time:
    every type in it is the only type legal wherever it is legal, and every
    other type is legal beside another type somewhere, so its score trains.
    A type missing from UNTYPED would drift as the discard type did."""
    from hexset.actions import options_for
    from hexset.play import step_randomly
    from hexset.rules import DUEL_VARIANT_GAME
    from hexn import model9

    seen, mixed = set(), set()
    for seed in range(40):
        rng = random.Random(seed)
        game = start(random_base_board(random.Random(seed)), 2, random.Random(seed), game_type=DUEL_VARIANT_GAME)
        moves = 0
        while not is_over(game) and moves < 3000:
            legal = options_for(game)
            if len(legal) > 1:
                types = {a.type for a in legal}
                seen |= types
                if len(types) > 1:
                    mixed |= types
            step_randomly(game, rng)
            moves += 1
    assert model9.UNTYPED <= seen
    assert not (model9.UNTYPED & mixed), model9.UNTYPED & mixed
    assert seen - mixed <= model9.UNTYPED, seen - mixed


def test_the_paying_setup_settlement_is_the_seats_second_by_its_settlements_left():
    """The second setup settlement pays the starting hand
    (`hexset.game._in_second_setup_round`), and in a two-seat snake, dealt
    two or four with two retired, it is exactly the one placed with one of
    the seat's own already down: my token's settlements left read 4, not 5."""
    from hexset.contract9 import MY_COLUMNS_11D as M
    from hexset.game import _in_second_setup_round
    from hexset.play import step_randomly
    from hexset.rules import DUEL_VARIANT_GAME

    seen = {True: 0, False: 0}
    for seed in range(12):
        rng = random.Random(seed)
        locked = (1, 3) if seed % 2 else ()
        game = start(random_base_board(rng), 4 if locked else 2, rng, game_type=DUEL_VARIANT_GAME, locked=locked)
        while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            if game.phase is Phase.SETUP_SETTLEMENT:
                seat = game.current_player
                left = contract9.encode9(game, seat, "11d").seats[M["pieces"].start] * 10
                paying = _in_second_setup_round(game)
                assert round(float(left)) == (4 if paying else 5)
                seen[paying] += 1
            step_randomly(game, rng)
    assert seen[True] == seen[False] == 24


def test_an_11d_joint_net_scores_the_double_turn_with_a_paying_second_and_an_ordered_pair():
    """A "joint" 11d net's double turn, first by second, is the first turn's
    plain query and the second's paying one against their vertex keys, the
    setup-road query against both edges, and the pair
    k_first' P k_second, which reads the order. Asked in parts the same net
    scores each setup turn by its own query and no pair term."""
    from hexset.actions import ActionType as T
    from hexn.model9 import decision_scores

    rows, game = _duel_decisions(count=40, seed=3, every=50, double_turn=True)
    double = [i for i, r in enumerate(rows) if r[1].double]
    assert double
    net = _two_seat_net(game, 11, double_turn="joint")
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), torch.from_numpy(np.stack([r[0] for r in rows])))
    pairs = contract9.incidences(game._state.board.topology)
    with torch.no_grad():
        out = net(**inputs)
        _, _, types, q, (k_vertex, k_edge, _, _) = _query_reference(net, out, inputs["seats"], game)
        at = torch.tensor(double)
        got = net.double_scores(out, at)
        P = net.pair_matrix
        for n, row in enumerate(double):
            kv = torch.stack([k_vertex(v)[row] for v, _ in pairs])
            ke = torch.stack([k_edge(e)[row] for _, e in pairs])
            first = kv @ q["setup_settlement"][row] + ke @ q["setup_road"][row]
            second = kv @ q["setup_settlement_paying"][row] + ke @ q["setup_road"][row]
            want = first[:, None] + second[None, :] + kv @ P @ kv.T / 4
            torch.testing.assert_close(got[n], want, atol=1e-4, rtol=1e-4)
        assert not torch.allclose(P, P.T)
        # Its losses train both ways: whole (the double rows) and in parts.
        batch = _batch(rows)
        scores = decision_scores(net, out, batch)
        assert len(scores.double_rows) == len(double) and torch.isfinite(scores.total).all()
        del batch["double"]
        assert torch.isfinite(decision_scores(net, out, batch).total).all()


def test_an_11d_joint_batch_trains_and_the_fused_trunk_agrees():
    rows, game = _duel_decisions(count=60, seed=7, double_turn=True)
    net = _two_seat_net(game, 1, double_turn="joint")
    layout = contract9.layout(2, 19, 54, 72, "11d")
    batch = _batch(rows)
    assert batch["double"].any()
    with torch.no_grad():
        inputs = unpack9(layout, batch["x"])
        net.fused = False
        reference = net(**inputs)
        net.fused = True
        fused = net(**inputs)
        at = batch["double"].nonzero().squeeze(1)
        torch.testing.assert_close(net.double_scores(fused, at), net.double_scores(reference, at),
                                   atol=2e-4, rtol=2e-4)
    net.train()
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
    assert net.pair_matrix.grad is not None and net.pair_matrix.grad.abs().sum() > 0


def net_action(kind, a, b=0):
    from hexset.actions import Action

    return Action(kind, a, b)


@pytest.mark.parametrize("messages,attention,others", [
    ("separate", True, False), ("pooled", True, False), ("separate", False, False), ("pooled", False, False),
    ("pooled", False, True), ("separate", False, True), ("pooled", True, True),
])
def test_the_fused_trunk_computes_the_reference_under_11d(messages, attention, others):
    """Under 11d the node-major trunk broadcasts my seat token where the
    table's was, as the reference loop does: every output and every
    decision's scores agree, the query heads' candidates included."""
    from hexn.model9 import decision_scores

    rows, game = _duel_decisions(count=60, seed=7)
    net = _two_seat_net(game, 6, messages=messages, attention=attention, others=others)
    inputs = unpack9(contract9.layout(2, 19, 54, 72, "11d"), torch.from_numpy(np.stack([r[0] for r in rows])))
    batch = _batch(rows)
    del batch["double"]
    assert len(batch["disc"]) and len(batch["rb"]) and batch["setup"].any()
    with torch.no_grad():
        net.fused = False
        reference = net(**inputs)
        want = decision_scores(net, reference, batch)
        net.fused = True
        fused = net(**inputs)
        got = decision_scores(net, fused, batch)
    for name in ("value_logits", "logits", "road", "vertices", "seats", "hexes", "me", "table", "types",
                 "robber_query", "rb_road", "hand_context", "hands", "edge_keys"):
        torch.testing.assert_close(getattr(fused, name), getattr(reference, name), atol=2e-4, rtol=2e-4, msg=name)
    for name in ("flat", "setup", "rb", "disc", "total"):
        torch.testing.assert_close(getattr(got, name), getattr(want, name), atol=2e-4, rtol=2e-4, msg=name)


def test_a_11d_batch_trains():
    rows, game = _duel_decisions(count=60, seed=7)
    net = _two_seat_net(game, 1)
    net.train()
    layout = contract9.layout(2, 19, 54, 72, "11d")
    batch = _batch(rows)
    del batch["double"]
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


def test_segment_logsumexp_passes_a_finite_gradient_through_a_row_with_no_finite_score():
    import torch

    scores = torch.tensor([1.0, 2.0, float("-inf"), float("-inf"), 0.5], requires_grad=True)
    rows = torch.tensor([0, 0, 1, 1, 3])
    got = _segment_logsumexp(scores, rows, 4)
    assert torch.equal(torch.isinf(got), torch.tensor([False, True, True, False]))
    assert torch.allclose(got[[0, 3]], torch.stack([torch.logsumexp(scores[:2], 0), scores[4]]))
    # Downstream, as decision_scores combines it, a -inf part takes no weight.
    torch.logsumexp(torch.stack([got, torch.zeros(4)]), 0).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad[2:4], torch.zeros(2))
    want = torch.softmax(torch.stack([got[0], torch.tensor(0.0)]), 0)[0] * torch.softmax(scores[:2].detach(), 0)
    assert torch.allclose(scores.grad[:2], want)


def test_decision_entropy_passes_a_finite_gradient_when_a_rows_total_is_far_below_zero():
    # A decision of discard candidates alone whose log-sum-exp is below
    # log(float32 max) negated: exp(0 - total) at the illegal flat and setup
    # entries overflows, and 0 * inf in the dropped branch would be NaN.
    import torch
    from hexn import model9

    disc = torch.tensor([-94.0, -100.0, -106.0], requires_grad=True)
    flat, setup = torch.full((1, 5), float("-inf")), torch.full((1, 4), float("-inf"))
    total = torch.logsumexp(disc, 0).reshape(1)
    scores = model9.Scores(flat=flat, setup=setup, rb=torch.zeros(0), rb_row=torch.zeros(0, dtype=torch.long),
                           disc=disc, disc_row=torch.zeros(3, dtype=torch.long),
                           double_rows=torch.zeros(0, dtype=torch.long), double=torch.zeros(0, 16), total=total)
    entropy = model9.decision_entropy(scores)
    p = torch.softmax(disc.detach(), 0)
    assert torch.allclose(entropy, -(p * p.log()).sum().reshape(1), atol=1e-3)  # total - E[s] at -100: fp32 cancellation
    entropy.sum().backward()
    assert torch.isfinite(disc.grad).all()
