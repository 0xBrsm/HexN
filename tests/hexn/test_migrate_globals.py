# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.migrate_globals`: a contract-6 checkpoint onto contract 7's globals.

The claim is exact: zero columns for the appended blocks, nothing else
changed, so the migrated net on today's encoding computes what the old net
computed on the contract-6 encoding (today's minus its last `2 * players`
columns), on real positions from every seat.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import encode, global_columns, global_features, static_graph  # noqa: E402
from hexset.game import start  # noqa: E402
from hexn.migrate_globals import (  # noqa: E402
    WEIGHT,
    _positions,
    appended,
    compare,
    main,
    migrate_checkpoint,
    source_globals,
    widen_optimiser,
    widen_state_dict,
)
from hexn.model import HexNet, ModelConfig  # noqa: E402

PLAYERS = 4
CONFIG = ModelConfig(width=16, rounds=2)


def _net(width: int | None = None, seed: int = 0) -> HexNet:
    """A net at today's globals width, or, given `width`, one whose global
    projection reads that many columns: a contract-6 net."""
    torch.manual_seed(seed)
    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, PLAYERS, rng)
    net = HexNet(space_for(game), static_graph(board.topology), PLAYERS, CONFIG)
    if width is not None:
        net.embed_global = torch.nn.Linear(width, CONFIG.width)
    return net.eval()


def _contract_6_checkpoint(path, seed: int = 3) -> HexNet:
    old = _net(source_globals(PLAYERS), seed=seed)
    optimiser = torch.optim.Adam(old.parameters(), lr=1e-3)
    # One real step, so the Adam moments exist and are not zero.
    globals_ = torch.randn(2, source_globals(PLAYERS))
    old.embed_global(globals_).sum().backward()
    optimiser.step()
    torch.save(
        {
            "iteration": 200,
            "games_started": 1234,
            "net": old.state_dict(),
            "optimiser": optimiser.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "args": {"players": PLAYERS, "width": CONFIG.width, "rounds": CONFIG.rounds},
        },
        path,
    )
    return old


def test_the_appended_blocks_are_the_globals_tail():
    columns = global_columns(PLAYERS)
    assert appended(PLAYERS) == 2 * PLAYERS == 8
    assert source_globals(PLAYERS) == columns["ledger"].stop == 67
    assert columns["roads_to_take"].start == source_globals(PLAYERS)
    assert columns["to_move"].stop == global_features(PLAYERS)


def test_only_the_global_projection_widens_and_only_with_zeros():
    old = _net(source_globals(PLAYERS)).state_dict()
    new = widen_state_dict(old, PLAYERS)
    assert set(new) == set(old)
    for key in old:
        if key != WEIGHT:
            assert new[key] is old[key], key
    assert new[WEIGHT].shape == (CONFIG.width, global_features(PLAYERS))
    assert torch.equal(new[WEIGHT][:, : source_globals(PLAYERS)], old[WEIGHT])
    assert not new[WEIGHT][:, source_globals(PLAYERS) :].any()
    # Loads into a net built at today's width.
    _net().load_state_dict(new, strict=True)


def test_a_net_not_at_contract_6_width_is_refused():
    with pytest.raises(ValueError, match="contract 6"):
        widen_state_dict(_net().state_dict(), PLAYERS)


def test_the_migrated_net_computes_what_the_old_one_did():
    old = _net(source_globals(PLAYERS), seed=5)
    new = _net()
    new.load_state_dict(widen_state_dict(old.state_dict(), PLAYERS), strict=True)
    observations = [encode(game, seat) for game, seat in _positions(PLAYERS, 24, seed=11)]
    # Off-turn perspectives and a nonzero roads block are in the sample.
    columns = global_columns(PLAYERS)
    assert any(o.globals[columns["to_move"]][0] == 0.0 for o in observations)
    assert any((o.globals[columns["roads_to_take"]] < 1.0).any() for o in observations)

    report = compare(old, new, observations, PLAYERS)
    assert report["positions"] == len(observations)
    for name in ("logits", "value_logits", "value", "margin"):
        assert report[f"max_abs_{name}"] <= 1e-6, (name, report)


def test_the_optimiser_moments_widen_with_the_weight():
    old = _net(source_globals(PLAYERS))
    optimiser = torch.optim.Adam(old.parameters(), lr=1e-3)
    old.embed_global(torch.randn(2, source_globals(PLAYERS))).sum().backward()
    optimiser.step()
    state = optimiser.state_dict()
    index = [name for name, _ in old.named_parameters()].index(WEIGHT)

    widened = widen_optimiser(state, index, PLAYERS)
    for key in ("exp_avg", "exp_avg_sq"):
        moment = widened["state"][index][key]
        assert moment.shape == (CONFIG.width, global_features(PLAYERS))
        assert torch.equal(moment[:, : source_globals(PLAYERS)], state["state"][index][key])
        assert not moment[:, source_globals(PLAYERS) :].any()
    # Resumable: a fresh optimiser over the migrated net takes the state.
    new = _net()
    new.load_state_dict(widen_state_dict(old.state_dict(), PLAYERS))
    resumed = torch.optim.Adam(new.parameters(), lr=1e-3)
    resumed.load_state_dict(widened)
    new.embed_global(torch.randn(2, global_features(PLAYERS))).sum().backward()
    resumed.step()


def test_the_command_writes_a_checkpoint_that_matches_the_old_net(tmp_path):
    source = tmp_path / "iter-00200.pt"
    old = _contract_6_checkpoint(source)
    out = tmp_path / "iter-00200-c7.pt"
    assert main(["--checkpoint", str(source), "--out", str(out), "--observations", "8"]) == 0

    written = torch.load(out, map_location="cpu", weights_only=False)
    assert written["iteration"] == 200 and written["games_started"] == 1234
    assert written["migrate_globals"]["contract_to"] == "7"
    report = written["migrate_globals"]["report"]
    assert report["max_abs_logits"] <= 1e-6 and report["max_abs_value"] <= 1e-6
    new = _net()
    new.load_state_dict(written["net"], strict=True)
    observations = [encode(game, seat) for game, seat in _positions(PLAYERS, 6, seed=2)]
    assert compare(old, new, observations, PLAYERS)["max_abs_logits"] <= 1e-6

    # Already migrated: nothing to do, and an existing file is not overwritten.
    with pytest.raises(SystemExit, match="nothing to migrate"):
        migrate_checkpoint(out)
    with pytest.raises(SystemExit, match="exists"):
        main(["--checkpoint", str(source), "--out", str(out)])


def test_a_contract_6_state_dict_loads_straight_into_todays_net():
    """`HexNet` pads a narrower `embed_global.weight` on load, so every loader
    (`netbot`, `collect.frozen`, the export) reads an old checkpoint as the
    migration would have written it."""
    old = _net(source_globals(PLAYERS), seed=7)
    padded = _net()
    padded.load_state_dict(old.state_dict(), strict=True)
    migrated = _net()
    migrated.load_state_dict(widen_state_dict(old.state_dict(), PLAYERS), strict=True)
    assert torch.equal(padded.embed_global.weight, migrated.embed_global.weight)
    observations = [encode(game, seat) for game, seat in _positions(PLAYERS, 12, seed=13)]
    report = compare(old, padded, observations, PLAYERS)
    for name in ("logits", "value_logits", "value", "margin"):
        assert report[f"max_abs_{name}"] <= 1e-6, (name, report)


def test_a_width_that_is_neither_is_still_refused_on_load():
    odd = _net(source_globals(PLAYERS) - 3)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _net().load_state_dict(odd.state_dict(), strict=True)
