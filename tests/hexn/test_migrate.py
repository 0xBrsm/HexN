# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.migrate`: a contract-5 checkpoint onto contract 6.

A real contract-5 checkpoint's exact shapes are reconstructed by hand here --
the old (per-hex, per-victim) `PLAY_KNIGHT` block and the twenty-float public
valuation columns both left with the engine, so the tests manufacture a
contract-5-shaped state dict by widening a fresh contract-6 net's tensors
with *junk*, not zeros: a migration that merely reordered columns would pass
a zero-filled fixture, and only a fixture whose dropped columns and merged
rows carry real weights can show that dropping/averaging them is what the
migration actually does.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hexset.actions import ActionType, space_for
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.encoding import encode, global_features, static_graph
from hexset.game import is_over, start
from hexn.migrate import (
    NEW_KEYS,
    REBUILT_KEYS,
    _fit_temperature,
    _global_columns,
    compare,
    migrate_state_dict,
    source_globals,
)
from hexn.model import HexNet, ModelConfig
from hexn.readout import GLOBALS, plan
from hexset.play import step_randomly

PLAYERS = 4
SPAN = PLAYERS + 1  # num_players + 1: MOVE_ROBBER's, and contract 5's PLAY_KNIGHT's, per-hex stride
CONTRACT_6_GLOBALS = global_features(PLAYERS)
CONTRACT_5_GLOBALS = source_globals(PLAYERS)


def _net(seed: int = 0) -> HexNet:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, PLAYERS, rng)
    return HexNet(space_for(game), static_graph(board.topology), PLAYERS, ModelConfig())


def _observations(count: int = 32, seed: int = 1000) -> list:
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        game = start(random_base_board(rng), PLAYERS, rng)
        for _ in range(40 + (i % 80)):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.append(encode(game))
    return out


def _contract_5_state(net: HexNet, seed: int = 7) -> dict:
    """The contract-5 checkpoint this net would have been: `embed_global`
    widened by twenty junk valuation columns at their old offset, and
    `heads.hexes`/`heads.globals` reshaped for contract 5's per-hex
    `PLAY_KNIGHT` block instead of contract 6's single spend-only slot.
    """
    generator = torch.Generator().manual_seed(seed)
    state = {k: v.clone() for k, v in net.state_dict().items()}
    # A contract-5 checkpoint has no auxiliary margin head at all; the
    # migration is what warm-starts one (`NEW_KEYS`).
    for key in NEW_KEYS:
        del state[key]
    columns = _global_columns(PLAYERS)

    weight = state["embed_global.weight"]
    rows = weight.shape[0]
    old = torch.randn(rows, CONTRACT_5_GLOBALS, generator=generator)
    for new, source in enumerate(columns):
        old[:, source] = weight[:, new]
    state["embed_global.weight"] = old

    # `heads.hexes`: contract 6 has one block (MOVE_ROBBER, `SPAN` rows).
    # Contract 5 has that block *plus* an equally-wide PLAY_KNIGHT block,
    # appended -- junk, since the migration only ever reads its mean.
    hexes_weight = state["heads.hexes.weight"]
    hexes_bias = state["heads.hexes.bias"]
    assert hexes_weight.shape[0] == SPAN
    knight_weight = torch.randn(SPAN, hexes_weight.shape[1], generator=generator)
    knight_bias = torch.randn(SPAN, generator=generator)
    state["heads.hexes.weight"] = torch.cat([hexes_weight, knight_weight], dim=0)
    state["heads.hexes.bias"] = torch.cat([hexes_bias, knight_bias], dim=0)

    # `heads.globals`: contract 6 has one row for `PLAY_KNIGHT`; contract 5
    # has none (it lived on `heads.hexes` instead) -- so contract 5's globals
    # head is exactly one row narrower, with every other row in the same
    # relative order.
    space = space_for(_dummy_game())
    globals_head = plan(space).head(GLOBALS)
    knight_index = None
    col = 0
    for kind in globals_head.kinds:
        if kind is ActionType.PLAY_KNIGHT:
            knight_index = col
            break
        col += space.sizes[kind]
    assert knight_index is not None
    for key in ("heads.globals.weight", "heads.globals.bias"):
        tensor = state[key]
        state[key] = torch.cat([tensor[:knight_index], tensor[knight_index + 1 :]], dim=0)

    # `flat_gather` is one index per flat slot; contract 5's total is 94 more
    # (a `SPAN`-wide extra hexes block minus the one globals row) -- its exact
    # content is never read (`REBUILT_KEYS`), only its rebuild from the
    # template matters, so any placeholder of the right length does.
    old_total = net.readout.size + (SPAN - 1)
    state["flat_gather"] = torch.arange(old_total, dtype=state["flat_gather"].dtype)

    return state


def _dummy_game(seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), PLAYERS, rng)


def test_the_column_map_is_a_pure_drop_of_the_valuation_block():
    columns = _global_columns(PLAYERS)
    assert len(columns) == CONTRACT_6_GLOBALS
    # A selection, not a pad: every contract-6 column has a contract-5 source.
    assert all(c is not None for c in columns)
    assert len(set(columns)) == len(columns)
    # A map, not a shuffle: what survives keeps its relative order.
    assert columns == sorted(columns)
    # Exactly twenty (at 4 players) columns of contract 5 never appear.
    dropped = set(range(CONTRACT_5_GLOBALS)) - set(columns)
    assert len(dropped) == PLAYERS * NUM_RESOURCES == 20
    # Contiguous, and where the module docstring says: right after the three
    # scalars (free roads, deck size, turn) and before the ledger tail.
    assert dropped == set(range(min(dropped), min(dropped) + 20))


def test_migration_drops_exactly_the_valuation_columns():
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    columns = _global_columns(PLAYERS)

    migrated = migrate_state_dict(old, template, PLAYERS)

    assert set(migrated) == set(template)
    weight = migrated["embed_global.weight"]
    assert weight.shape == template["embed_global.weight"].shape
    for new, source in enumerate(columns):
        assert torch.equal(weight[:, new], old["embed_global.weight"][:, source])


def test_an_unrelated_globals_slot_keeps_its_own_row():
    """A kind untouched by the knight collapse (`BANK_TRADE`, say) keeps
    exactly its own weight row, only shifted by the one row the new
    `PLAY_KNIGHT` slot inserts ahead of it."""
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    migrated = migrate_state_dict(old, template, PLAYERS)

    new_space = space_for(_dummy_game())
    new_head = plan(new_space).head(GLOBALS)
    old_col = 0
    new_col = 0
    for kind in new_head.kinds:
        size = new_space.sizes[kind]
        if kind is not ActionType.PLAY_KNIGHT:
            assert torch.equal(
                migrated["heads.globals.weight"][new_col : new_col + size],
                old["heads.globals.weight"][old_col : old_col + size],
            )
            assert torch.equal(
                migrated["heads.globals.bias"][new_col : new_col + size],
                old["heads.globals.bias"][old_col : old_col + size],
            )
            old_col += size
        new_col += size


def test_the_knight_slot_is_the_mean_of_its_old_block():
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    migrated = migrate_state_dict(old, template, PLAYERS)

    # `heads.hexes`: contract 5's rows 0:SPAN (MOVE_ROBBER) survive
    # unchanged; rows SPAN:2*SPAN (the old per-hex PLAY_KNIGHT block) are
    # gone from `heads.hexes` and become the mean that seeds the new
    # `heads.globals` row instead.
    assert torch.equal(migrated["heads.hexes.weight"], old["heads.hexes.weight"][:SPAN])
    assert torch.equal(migrated["heads.hexes.bias"], old["heads.hexes.bias"][:SPAN])

    new_space = space_for(_dummy_game())
    new_head = plan(new_space).head(GLOBALS)
    knight_row = 0
    for kind in new_head.kinds:
        if kind is ActionType.PLAY_KNIGHT:
            break
        knight_row += new_space.sizes[kind]

    expected_weight = old["heads.hexes.weight"][SPAN:].mean(dim=0)
    expected_bias = old["heads.hexes.bias"][SPAN:].mean()
    assert torch.equal(migrated["heads.globals.weight"][knight_row], expected_weight)
    assert torch.equal(migrated["heads.globals.bias"][knight_row], expected_bias)


def test_flat_gather_is_rebuilt_not_migrated():
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    migrated = migrate_state_dict(old, template, PLAYERS)

    assert "flat_gather" in REBUILT_KEYS
    assert torch.equal(migrated["flat_gather"], template["flat_gather"])
    assert len(migrated["flat_gather"]) == net.readout.size


def test_the_aux_margin_head_is_a_copy_of_the_old_value_head():
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    migrated = migrate_state_dict(old, template, PLAYERS)

    assert set(NEW_KEYS) == {"aux_margin.weight", "aux_margin.bias"}
    assert torch.equal(migrated["aux_margin.weight"], old["value.weight"])
    assert torch.equal(migrated["aux_margin.bias"], old["value.bias"])
    # The value head itself passes through unscaled here -- `migrate_
    # checkpoint` is what applies `1 / T` once a temperature is fit.
    assert torch.equal(migrated["value.weight"], old["value.weight"])


def test_everything_else_passes_through_untouched():
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    migrated = migrate_state_dict(old, template, PLAYERS)

    moved = {
        "embed_global.weight",
        "heads.hexes.weight",
        "heads.hexes.bias",
        "heads.globals.weight",
        "heads.globals.bias",
        *REBUILT_KEYS,
        *NEW_KEYS,
    }
    for key, tensor in migrated.items():
        if key not in moved:
            assert torch.equal(tensor, old[key]), key


def test_a_shape_the_migration_does_not_know_about_is_refused():
    net = _net()
    template = net.state_dict()
    bad = _contract_5_state(net)
    bad["embed_hex.weight"] = bad["embed_hex.weight"][:, :-1].clone()
    with pytest.raises(ValueError, match="not part of the contract-6 migration"):
        migrate_state_dict(bad, template, PLAYERS)


def test_the_embed_global_projection_is_exact_on_zero_valuation_positions():
    """The one function-preserving claim: pad a real contract-6 encoding back
    out to contract 5's width with the dropped columns held at zero, and the
    two widths' `embed_global` must agree bit for bit -- linear algebra, not
    an empirical coincidence, and `compare` is the numerical witness of it.
    """
    net = _net()
    template = net.state_dict()
    old = _contract_5_state(net)
    columns = _global_columns(PLAYERS)
    migrated = migrate_state_dict(old, template, PLAYERS)
    net.load_state_dict(migrated, strict=True)
    net.eval()

    report = compare(net, columns, _observations(), PLAYERS)
    assert report["max_abs_embed_delta_on_zero_valuation_positions"] == 0.0


def test_fit_temperature_recovers_a_known_scale():
    """A synthetic check of the MLE fit against data actually generated by
    the model it assumes: random logits, and a winner *sampled* from
    `softmax(margin / T_true)` -- not simply assigned to the largest logit,
    which would reward driving the fitted temperature to zero (arbitrarily
    more confident is arbitrarily better when the labels are noiseless).
    Maximum likelihood on enough samples from the true model recovers the
    parameter that generated them.
    """
    rng = np.random.default_rng(0)
    true_t = 3.0
    n = 20000
    margins = rng.normal(scale=2.0, size=(n, 4))
    scaled = margins / true_t
    probabilities = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    winners = np.array(
        [rng.choice(4, p=row) for row in probabilities], dtype=np.int64
    )

    fitted_t, loss = _fit_temperature(margins, winners)
    assert fitted_t == pytest.approx(true_t, rel=0.15)
    assert loss < math.log(4)


def test_a_matching_checkpoint_passes_through_untouched():
    """A checkpoint that is already contract 6 -- every shape matches the
    template, `aux_margin.*` included -- is untouched end to end: the
    contract-5-specific remaps assume the wider shapes and are never
    reached."""
    net = _net()
    template = net.state_dict()
    out = migrate_state_dict(
        {k: v.clone() for k, v in template.items()}, template, PLAYERS
    )
    for key in template:
        assert torch.equal(out[key], template[key]), key
