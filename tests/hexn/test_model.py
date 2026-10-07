# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import encode, encode_batch, static_graph  # noqa: E402
from hexset.game import start  # noqa: E402
from hexn.model import (  # noqa: E402
    VALUE_HEADS,
    HexNet,
    ModelConfig,
    collate,
    config_from_args,
    pack,
    packing,
    unpack,
)
from hexset.play import step_randomly  # noqa: E402
from hexn.readout import scatter_logits  # noqa: E402


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def a_net(players: int = 4, seed: int = 0, **kwargs):
    torch.manual_seed(seed)
    game = a_game(players=players)
    space = space_for(game)
    # public field
    net = HexNet(
        space,
        static_graph(game.state(0, hidden=False).board.topology),
        players,
        ModelConfig(**kwargs),
    )
    return game, space, net


def observations(count: int, players: int = 4):
    return [encode(a_game(players=players, seed=s, steps=60 + s)) for s in range(count)]


def test_output_shapes():
    # Two seats: every other test here runs at four, which is the default.
    players = 2
    game, space, net = a_net(players=players)
    batch = collate(observations(3, players=players))

    out = net(*batch)

    assert out.logits.shape == (3, space.size)
    assert out.value.shape == (3, players)
    assert out.value_logits.shape == (3, players)
    assert out.margin.shape == (3, players)
    # Nothing else: the trade heads were deleted with the trade actions
    # (`hexset.trading`), and the network's private gate is derived from the
    # value head instead (`hexset.clients.netbot.NetworkBot`).
    assert not hasattr(out, "give")
    assert not hasattr(out, "want")


def test_the_win_head_sums_to_one_and_reads_the_perspective_row():
    """`Prediction.value` is `softmax(value_logits)` over the seat axis: a
    proper win-probability distribution, nonnegative and summing to one
    across `players`, with the encoded perspective seat's own probability
    at column 0 -- the same
    convention the encoder and every other per-seat vector this network
    emits already use.
    """
    _, _, net = a_net(players=3)
    batch = collate(observations(5, players=3))

    out = net(*batch)

    assert torch.all(out.value >= 0.0)
    assert torch.allclose(
        out.value.sum(-1), torch.ones(out.value.shape[0]), atol=1e-5
    )
    assert torch.allclose(out.value, torch.softmax(out.value_logits, dim=-1))


def test_the_win_heads_perspective_row_moves_with_the_encoders_rotation():
    """Column 0 is always the encoded perspective seat's own row -- the
    win head reads whatever `_value_features` hands it and does nothing of
    its own to track *which* seat that is, so this has to come from the
    encoder's rotation alone. Pinned with a hand-set net (`rounds=0`, so the
    value head reads the embedded globals directly) whose weights read only
    the "own hand" block: encoding the same position from two different
    seats must then move column 0 to whichever seat's *own* hand differs,
    proving the column tracks the encoder's rotation rather than a fixed
    board seat.
    """
    from hexset.actions import build_space

    graph = static_graph(random_base_board(random.Random(0)).topology)
    space = build_space(graph.num_vertices, graph.num_edges, graph.num_hexes, 4)
    net = HexNet(space, graph, 4, ModelConfig(width=4, rounds=0))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        # Row 0 of `embed_global.weight` reads column 0 (the first entry of
        # the perspective seat's own hand) with weight 1; `value.weight`
        # reads only that one feature into seat 0's own logit.
        net.embed_global.weight[0, 0] = 1.0
        net.value.weight[0, 0] = 1.0

    game = a_game(players=4)
    # Force two seats' first-resource counts apart so their own-hand reading
    # actually differs -- real random play may not have left them so.
    state = game.state(0, hidden=False)
    state.hands[0][0] = 5
    state.hands[1][0] = 0

    batch = collate([encode(game, 0), encode(game, 1)])
    out = net(*batch)

    # Softmax is strictly increasing in its own logit, so seat 0's row (own
    # hand[0] = 5) must read a strictly higher column-0 probability than
    # seat 1's row (own hand[0] = 0) -- the column followed the rotation.
    assert out.value[0, 0] > out.value[1, 0]


def test_packing_round_trips_to_the_same_batch_as_collate():
    """`pack` writes through reshaped views, which numpy may silently copy.

    If a slice ever reshaped to a copy instead of a view, `pack` would stack
    into a temporary and hand back an uninitialised buffer — garbage inputs,
    no error. This compares it against `collate`, which builds the same batch
    the obvious way.
    """
    # Two seats; four is `test_pack_reuses_the_batch_encoders_contiguous_buffer`'s.
    players = 2
    obs = observations(3, players=players)
    # public field
    layout = packing(
        static_graph(a_game(players=players).state(0, hidden=False).board.topology),
        players,
    )
    packed = pack(layout, obs)

    assert packed.shape == (3, layout.width)
    for got, want in zip(unpack(layout, packed), collate(obs), strict=True):
        assert torch.equal(got, want)


def test_pack_reuses_the_batch_encoders_contiguous_buffer():
    games = [a_game(seed=seed, steps=60 + seed) for seed in range(8)]
    observations = encode_batch(games, [game.current_player for game in games])
    # public field
    layout = packing(static_graph(games[0].state(0, hidden=False).board.topology), players=4)

    packed = pack(layout, observations)

    assert observations[0]._packed is not None
    assert packed.data_ptr() == observations[0]._packed.ctypes.data
    canonical = [
        encode(game, game.current_player)
        for game in games
    ]
    for got, want in zip(unpack(layout, packed), collate(canonical), strict=True):
        assert torch.equal(got, want)


def test_the_flat_permutation_agrees_with_the_numpy_scatter():
    """Ties the model's one-gather shortcut to the tested reference in readout.

    The model concatenates its heads and permutes once; `scatter_logits` writes
    each head into place. They must agree or the policy is trained on the wrong
    actions, which is the failure this whole mapping exists to prevent.
    """
    _, _, net = a_net()
    rng = np.random.default_rng(0)

    emitted = {
        head.source: rng.standard_normal((head.num_nodes, head.width)).astype(np.float32)
        for head in net.readout.heads
    }
    expected = scatter_logits(net.readout, emitted)

    stacked = torch.cat(
        [torch.from_numpy(emitted[head.source]).reshape(1, -1) for head in net.readout.heads],
        dim=1,
    )
    actual = stacked.index_select(1, net.flat_gather)

    assert np.allclose(actual.numpy()[0], expected)


def test_each_row_of_a_batch_is_independent():
    """Catches message passing that leaks across the batch dimension."""
    _, _, net = a_net()
    net.eval()
    obs = observations(4)

    with torch.no_grad():
        together = net(*collate(obs))
        apart = [net(*collate([one])) for one in obs]

    for i, single in enumerate(apart):
        assert torch.allclose(together.logits[i], single.logits[0], atol=1e-5)
        assert torch.allclose(together.value[i], single.value[0], atol=1e-5)


# Written out rather than derived from a freshly built net, because a net is
# exactly the thing under test: generating the expectation from the code would
# make this pass no matter what the code did. Every checkpoint under `runs/`
# holds these names and no others, so this list is what makes a shape flag safe
# to add.
DEFAULT_KEYS = frozenset(
    {
        "embed_hex.weight",
        "embed_hex.bias",
        "embed_vertex.weight",
        "embed_vertex.bias",
        "embed_edge.weight",
        "embed_edge.bias",
        "embed_global.weight",
        "embed_global.bias",
        "value.weight",
        "value.bias",
        # The auxiliary VP-margin head (contract 6's win-head registration):
        # always built, so every checkpoint carries it whatever
        # `--aux-margin-weight` trains under.
        "aux_margin.weight",
        "aux_margin.bias",
        # Buffers are in the `state_dict` too, and a checkpoint carries them.
        "hv_hex",
        "hv_vertex",
        "ve_vertex",
        "ve_edge",
        "n_vertex_from_hex",
        "n_hex_from_vertex",
        "n_edge_from_vertex",
        "n_vertex_from_edge",
        "flat_gather",
    }
    | {
        f"heads.{source}.{parameter}"
        for source in ("hexes", "vertices", "edges", "globals")
        for parameter in ("weight", "bias")
    }
    | {
        f"rounds.{round_}.{block}.{layer}.{parameter}"
        for round_ in range(2)
        for block in ("hex", "vertex", "edge", "globe")
        for layer in (0, 2)
        for parameter in ("weight", "bias")
    }
)


def test_the_default_config_writes_exactly_the_state_dict_keys_it_always_has():
    """The one test that protects every checkpoint already on disk.

    `ModelConfig` gained `value_head` and `policy_head`, and the obvious way to
    implement either — wrap the head in a module and dispatch on its type —
    renames `value.weight` to `value.out.weight` and makes every run recorded so
    far unloadable. The default has to *be* the old network, not a new network
    that happens to compute the same function.

    `aux_margin.{weight,bias}` is the one deliberate exception: contract 6's
    win-head registration adds a genuinely new head to every checkpoint from
    here on, which is exactly what `hexn.migrate` exists to carry an
    existing contract-5 checkpoint across (it warm-starts the new head from
    the old value head's own weights) -- this list says what the shape *is*
    now, not that nothing may ever be added to it again.
    """
    _, _, net = a_net()

    assert set(net.state_dict()) == DEFAULT_KEYS


def test_a_deep_policy_head_fills_the_same_flat_action_space():
    """The scatter is the contract: a deeper head may not move a single slot."""
    _, space, net = a_net(policy_head="mlp")

    out = net(*collate(observations(3)))

    assert out.logits.shape == (3, space.size)
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("shape", ["linear", "mlp_pooled", "attn"])
def test_a_backward_pass_reaches_every_parameter_of_every_value_head(shape):
    """Catches an attention query, or a pooled input, wired in but never read.

    `linear` is the default net; `mlp_pooled` is both the pooled input and the
    hidden layer, so it covers `pooled` and `mlp`; `attn` adds the query.
    """
    _, _, net = a_net(value_head=shape)

    out = net(*collate(observations(2)))
    # `out.value` is a softmax, and a softmax row's sum is the constant 1 --
    # summing it backward would carry exactly zero gradient into the value
    # head and the trunk beneath it, not a weak signal but a mathematical
    # zero. `value_logits` (pre-softmax) and `margin` (the auxiliary head)
    # are what actually carry gradient from the value side.
    (out.logits.sum() + out.value_logits.sum() + out.margin.sum()).backward()

    unused = [
        name for name, p in net.named_parameters() if p.grad is None or not p.grad.any()
    ]
    assert unused == []


def test_detaching_the_value_cuts_every_tensor_the_head_reads_not_only_the_global():
    """`detach_value` cuts the head off the trunk; it must not cut off the head.

    The flag detaches every tensor the head reads, not only `g`, or a pooled
    head would keep the exact gradient path the flag exists to remove. The head's
    own parameters still have to train, which is the whole point of detaching
    rather than freezing. `mlp_pooled` reads every node type besides `g`, so
    it covers `pooled` and the vertex read `attn` makes.
    """
    _, _, net = a_net(value_head="mlp_pooled")
    net.detach_value = True

    # `value_logits`, not `value`: a softmax's row-sum is the constant 1 and
    # carries no gradient at all, detached or not.
    net(*collate(observations(2))).value_logits.sum().backward()

    head = [p for name, p in net.named_parameters() if name.startswith("value")]
    assert any(p.grad is not None and p.grad.any() for p in head)
    trunk = net.embed_vertex.weight.grad
    assert trunk is None or not trunk.any()


def test_a_deeper_head_only_adds_keys_under_its_own_name():
    """A shape change may not disturb the trunk, or an ablation is confounded."""
    _, _, deep = a_net(value_head="mlp", policy_head="mlp")
    keys = set(deep.state_dict())

    # `aux_margin` shares `value_head`'s own depth (`HexNet.__init__`), so a
    # deep `value_head` deepens it exactly like `value` -- excluded from the
    # trunk set for the same reason "value" is, and checked explicitly below.
    trunk = {
        name for name in DEFAULT_KEYS if not name.startswith(("value", "heads.", "aux_margin"))
    }
    assert trunk <= keys
    assert "value.0.weight" in keys and "value.2.weight" in keys
    assert "value.weight" not in keys
    assert "aux_margin.0.weight" in keys and "aux_margin.2.weight" in keys
    assert "aux_margin.weight" not in keys
    assert "heads.vertices.0.weight" in keys and "heads.vertices.2.weight" in keys


def test_the_output_layer_of_a_deep_head_is_the_one_that_gets_the_small_gain():
    """`_initialise`'s reasoning, restated as a property.

    A hidden layer inside a head feeds a SiLU and is trunk by every property
    that method cares about, so it keeps gain √2. Only the layer emitting a
    logit is held near zero — giving the whole `Sequential` gain 0.01 would
    start a two-layer head weaker than the one-layer head it is being compared
    against, which would decide the ablation by initialisation.
    """
    _, _, net = a_net(value_head="mlp", policy_head="mlp")

    hidden = net.heads["vertices"][0].weight.std().item()
    output = net.heads["vertices"][2].weight.std().item()
    assert output < hidden / 10

    # The value head predicts at its target's scale, so its output stays at 1.0.
    assert net.value[2].weight.std().item() > net.value[0].weight.std().item() / 10


def test_stored_args_rebuild_the_shape_they_were_trained_with():
    """The bug this pins cost a launched probe.

    `hexn.benchmarks.minibatch_iso_kl` and `hexn.benchmarks.noise_scale` both rebuilt the
    net from width and rounds alone, so the first `--policy-head mlp` lineage
    would not load at all: `heads.hexes.weight` against `heads.hexes.0.weight`.
    One reader now, and this is its contract.
    """
    stored = {"width": 64, "rounds": 2, "value_head": "attn", "policy_head": "mlp"}

    assert config_from_args(stored) == ModelConfig(
        width=64, rounds=2, value_head="attn", policy_head="mlp"
    )


def test_a_namespace_that_predates_a_knob_means_the_default_shape():
    """Every checkpoint on disk from before the fields existed, and `{}` is the
    honest test of that: a run recorded nothing about a shape it did not have."""
    assert config_from_args({}) == ModelConfig()
    assert config_from_args({"width": 96}) == ModelConfig(width=96)


@pytest.mark.parametrize(
    "field,value",
    [("value_head", "attention"), ("policy_head", "pooled")],
)
def test_an_unknown_head_shape_is_refused_before_a_run_starts(field, value):
    """A typo that silently fell back to the default would be discovered as an
    ablation that read flat, which is the most expensive way to find it."""
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_a_shaped_head_costs_parameters_but_keeps_the_model_small():
    _, _, plain = a_net()
    _, _, shaped = a_net(value_head="mlp_pooled", policy_head="mlp")

    plain_total = sum(p.numel() for p in plain.parameters())
    shaped_total = sum(p.numel() for p in shaped.parameters())
    assert shaped_total > plain_total
    assert shaped_total < 500_000, shaped_total


def test_the_fused_forward_matches_the_reference_path():
    """Fused is a kernel change, not a math change.

    Same parameters, same inputs, outputs equal to float tolerance — the only
    licensed difference is summation order. And no new state_dict keys: the
    adjacency buffers are derived from the graph, so a checkpoint from before
    they existed must still load strictly. The head shape is not a parameter
    here: fusing changes only the message-passing rounds, which every head
    reads the same way.
    """
    _, _, net = a_net()
    assert not net.fused, "reference is the default; fused is the GPU opt-in"
    batch = collate(observations(3))

    net.fused = True
    fused = net(*batch)
    net.fused = False
    reference = net(*batch)

    for name in ("logits", "value"):
        assert torch.allclose(
            getattr(fused, name), getattr(reference, name), rtol=1e-4, atol=1e-5
        ), name
    assert not any(key.startswith("A_") for key in net.state_dict())


def test_the_fused_forward_trains_the_same_parameters():
    """Weight views must not orphan any parameter from the backward pass."""
    _, _, net = a_net()
    net.fused = True
    out = net(*collate(observations(2)))
    # `value_logits`/`margin`, not `value`: a softmax row-sum is the
    # constant 1 and carries no gradient.
    (out.logits.sum() + out.value_logits.sum() + out.margin.sum()).backward()
    unused = [
        name for name, p in net.named_parameters() if p.grad is None or not p.grad.any()
    ]
    assert unused == []


# ---------------------------------------------------------------------------
# The quantile value head.
# ---------------------------------------------------------------------------


def test_the_quantile_head_s_forward_is_exactly_the_mean_of_its_quantiles():
    """`value_logits` is the mean-of-quantiles and nothing downstream sees
    the spread except through it.

    Exact equality, not `allclose`: `hexn.ppo`'s quantile loss and the win
    head's own softmax both read `value_logits`, so if the head's forward
    were the mean of anything other than the tensor the loss trains, the loss
    and the wire would be optimising two different numbers. `Prediction.
    value` is `softmax(value_logits)`, checked separately (`test_the_win_
    head_sums_to_one_and_reads_the_perspective_row` covers every shape).
    """
    _, _, net = a_net(value_head="quantile")

    out = net(*collate(observations(3)))

    assert out.quantiles.shape == (3, 4, 32)
    assert torch.equal(out.value_logits, out.quantiles.mean(-1))
    assert torch.allclose(out.value, torch.softmax(out.value_logits, dim=-1))


def test_only_the_quantile_head_carries_a_spread():
    """The new field is inert for every shape that predates it."""
    for shape in VALUE_HEADS:
        _, _, net = a_net(value_head=shape)
        out = net(*collate(observations(2)))
        assert (out.quantiles is None) == (shape != "quantile"), shape
        assert out.value.shape == (2, 4)


def test_a_quantile_checkpoint_reloads_and_reproduces_its_own_predictions(tmp_path):
    """End to end: the keys a quantile net writes are the keys its stored args
    rebuild, and the rebuilt net predicts the same numbers bit for bit."""
    _, space, net = a_net(value_head="quantile", quantiles=8)
    batch = collate(observations(3))
    with torch.no_grad():
        before = net(*batch)

    stored = {"width": 64, "rounds": 2, "value_head": "quantile", "quantiles": 8}
    path = tmp_path / "quantile.pt"
    torch.save({"net": net.state_dict(), "args": stored}, path)

    state = torch.load(path, map_location="cpu", weights_only=False)
    _, _, rebuilt = a_net(**{
        field: getattr(config_from_args(state["args"]), field)
        for field in ("width", "rounds", "value_head", "policy_head", "quantiles")
    })
    rebuilt.load_state_dict(state["net"])
    with torch.no_grad():
        after = rebuilt(*batch)

    assert torch.equal(before.value, after.value)
    assert torch.equal(before.quantiles, after.quantiles)
    assert torch.equal(before.logits, after.logits)


def test_the_midpoint_levels_are_the_registered_ones_and_pair_about_a_half():
    """`(i + 0.5) / Q`, symmetric about 0.5 — which is what makes the mean of
    the quantiles the mean of a symmetric law rather than an approximation."""
    from hexn.model import quantile_levels

    levels = quantile_levels(4)

    assert torch.allclose(levels, torch.tensor([0.125, 0.375, 0.625, 0.875]))
    assert torch.allclose(levels + levels.flip(0), torch.ones(4))
    with pytest.raises(ValueError):
        quantile_levels(0)


def test_the_pinball_loss_is_minimised_exactly_at_the_true_quantiles():
    """The property the whole head rests on, checked where it is exact.

    With `kappa=0` the loss is the exact pinball loss, whose minimiser over a
    finite sample is the sample quantile. On nine sorted samples and levels
    that land between order statistics, the minimiser is the corresponding
    order statistic exactly — so a grid search over the samples themselves must
    pick it, and perturbing away from it must cost.
    """
    from hexn.model import quantile_huber_loss, quantile_levels

    sample = torch.tensor([-1.0, -0.6, -0.3, -0.1, 0.0, 0.2, 0.45, 0.7, 1.3])
    target = sample.view(-1, 1)  # nine rows, one seat
    levels = quantile_levels(3)  # 1/6, 1/2, 5/6

    def loss_at(values):
        predicted = torch.tensor(values).view(1, 1, -1).expand(9, 1, 3)
        return quantile_huber_loss(predicted, target, levels, 0.0).item()

    best = None
    for a in sample:
        for b in sample:
            for c in sample:
                score = loss_at([a, b, c])
                if best is None or score < best[0]:
                    best = (score, [a.item(), b.item(), c.item()])

    # Levels 1/6, 1/2, 5/6 of nine samples: order statistics 2, 5 and 8.
    assert best[1] == pytest.approx([-0.6, 0.0, 0.7])
    for nudge in (-0.05, 0.05):
        assert loss_at([-0.6 + nudge, 0.0, 0.7]) > best[0]
        assert loss_at([-0.6, 0.0 + nudge, 0.7]) > best[0]
        assert loss_at([-0.6, 0.0, 0.7 + nudge]) > best[0]


def test_a_warm_started_quantile_head_predicts_what_the_scalar_head_predicted():
    """A warm-started quantile head opens on the scalar head's own critic,
    not on noise.

    Every level starts at the scalar head's own output, so `V` is unchanged to
    a few float32 roundings of a 32-term mean — four orders of magnitude below
    the 1/30 lattice the label itself lives on.

    The bound is *relative*, and has to be: the error is a handful of ulps of
    the value being summed, so an absolute tolerance would be a claim about
    how large the initialised head's outputs happen to be rather than about
    the warm start.
    """
    from hexn.model import quantile_warm_start

    _, _, scalar = a_net(seed=3)
    _, _, quantile = a_net(seed=4, value_head="quantile")
    batch = collate(observations(4))

    quantile.load_state_dict(quantile_warm_start(scalar.state_dict(), 4, 32))
    with torch.no_grad():
        before = scalar(*batch)
        after = quantile(*batch)

    assert torch.allclose(before.value, after.value, rtol=1e-5, atol=1e-7)
    # Anti-vacuity: the head it started from was a different net entirely.
    assert not torch.equal(before.value, torch.zeros_like(before.value))
    # Every level is the same number, so the spread opens at exactly zero.
    assert torch.equal(after.quantiles.std(-1), torch.zeros(4, 4))
    # The trunk is copied through untouched.
    assert torch.equal(before.logits, after.logits)


def test_a_warm_start_off_a_head_that_is_not_per_seat_is_refused():
    from hexn.model import quantile_warm_start

    _, _, deep = a_net(value_head="mlp")
    # `value.0.*` is the hidden layer and `value.2.*` emits the seats; a head
    # whose emitting layer is not per-seat is not a scalar value head.
    weights = dict(deep.state_dict())
    weights["value.2.weight"] = torch.zeros(7, 64)
    with pytest.raises(ValueError):
        quantile_warm_start(weights, 4, 8)


def test_the_value_read_is_the_last_thing_the_forward_builds():
    """Op order in `_emit` is load-bearing, and this is the anchor for it.

    `g` feeds the global policy head and the value head, so its gradient is a
    sum and autograd accumulates the terms in the order the forward created
    the nodes. Hoisting the value read above the logits reassociates that sum:
    the loss stays bit-identical and the *gradient* moves in the last couple
    of bits -- enough, under floating-point chaos, to make an update diverge
    after a single optimiser step. A loss-equality test does not catch it;
    the creation order does. `_emit` orders the two reads the same way for
    every head shape, so the default net is the anchor.
    """
    _, _, net = a_net()

    out = net(*collate(observations(2)))

    order = [out.logits, out.value]
    sequence = [tensor.grad_fn._sequence_nr() for tensor in order]
    assert sequence == sorted(sequence), sequence
