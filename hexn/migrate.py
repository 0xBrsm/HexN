# SPDX-License-Identifier: GPL-3.0-only
"""Migrate a contract-5 checkpoint onto contract 6, without changing what it
computes on a position whose dropped columns read zero.

Contract 6 carries two independent changes that happened to land together
(`hexset.onnx_record.
CONTRACT_VERSION`'s own docstring is the record of that): the trading
redesign deletes the public layer outright, and the knight two-step fix
changes the flat action space's width. This tool handles both, plus the
value head's reinterpretation as a win head:

* **The twenty valuation columns are dropped**, not zeroed. There is no
  public layer left (`hexset.trading`) for a seat to publish a vector to, so
  `hexset.encoding.global_features` is twenty floats narrower, at the same
  offset the old block sat: right after the three scalars (free roads, deck
  size, turn) and before the public-knowledge ledger tail. `_global_columns`
  is the map from each of contract 6's global columns to its contract-5
  source; every one of them has a source (a drop is a pure column
  *selection*, not a pad), so there is no fresh, zero-initialised weight
  anywhere in `embed_global.weight`.

* **The knight's per-hex-and-victim block collapses to one slot.** Contract 5
  routed `PLAY_KNIGHT` through the HEXES head exactly like `MOVE_ROBBER` — the
  same `num_players + 1` victim slots per hex, a second full block alongside
  the real robber move — because playing a knight still named a target and a
  victim directly. Contract 6 splits that: `PLAY_KNIGHT` only spends the card
  now (one GLOBALS slot) and the actual placement runs through the same
  `MOVE_ROBBER` decision a rolled seven already uses. So `heads.hexes`
  shrinks by one whole `MOVE_ROBBER`-shaped block (kept only for the real
  robber move) and `heads.globals` gains the one new slot, initialised as the
  **mean** of the block it replaces — a warm start, not a claim of
  equivalence, since a single slot cannot reproduce a 5-way (at 4 players)
  targeting decision. Every other slot on both heads keeps its own row: nei-
  ther head's other kinds moved, only their absolute offset shifted by
  exactly the knight block's width.

* **The value head is warm-started as a win head.** Its weights are kept
  exactly — they are still `softmax(margin / T)`'s logits, just linearly
  rescaled by `1 / T` — and `T` is fitted by maximum likelihood against the
  eventual winner of real games played on the target engine (`fit_
  temperature`, mirroring the protocol used to fit
  `hexset.bots.search2.win`'s own `WIN_TEMPERATURE`: one sample per turn
  at the mover's first decision, labelling the eventual
  winner). The **auxiliary VP-margin head is new** and starts as an exact
  copy of the old (unscaled) value head's weights — the most direct warm
  start available for a head meant to keep predicting what the old one did.

## In what sense this is function-preserving

Only in the `embed_global` projection, and only on the positions where the
dropped valuation columns would have read zero — which was every position
before contract 5's public layer had anything live to publish, since
`_recolumn`'s selection reproduces the old weight columns exactly (`compare`
below verifies this by construction, padding a real contract-6 encoding back
out to contract 5's width with the dropped columns held at zero and
comparing `embed_global`'s two outputs bit for bit). It is **not** claimed
for any position where a contract-5 checkpoint's seats had actually
published something — those columns carried a real, trained signal, and
dropping them changes what the trunk reads. It is **not** claimed at all for
the knight-collapse (an explicit warm start) or for the value head's
rescaling (a calibration, not an identity).

    python -m hexn.migrate --checkpoint runs/example/iter-02400-c5.pt \
        --out runs/example/iter-02400-c6.pt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from hexset.actions import ActionSpace, ActionType, space_for
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.encoding import encode, global_columns, global_features, static_graph
from hexset.game import is_over, start
from hexset.play import step_randomly
from . import runtime
from .model import HexNet, ModelConfig, collate, config_from_args
from .readout import GLOBALS, plan

ADAM_EPS = 1e-5

# Buffers the net derives from the action space rather than learns, so the
# migrated net's own is right by construction and the checkpoint's is stale.
REBUILT_KEYS = ("flat_gather",)

# The new auxiliary head has no source in a contract-5 checkpoint at all --
# it is initialised, not migrated, from the old (unscaled) value head.
NEW_KEYS = ("aux_margin.weight", "aux_margin.bias")


# Contract 5's public valuation block, at 4 players twenty floats wide: one
# per resource per seat. The only block this hop drops, and the only width
# `hexset.encoding` no longer knows, so it is the only one named here.
def _valuation(players: int) -> int:
    return players * NUM_RESOURCES


def source_globals(players: int) -> int:
    """Contract 5's own globals width: contract 6's plus the twenty-float
    (at 4 players) public valuation block `hexset.trading` deleted."""
    return global_features(players) + _valuation(players)


def _global_columns(players: int) -> list[int]:
    """For each of contract 6's global columns, its contract-5 source column.

    A pure selection -- every entry is defined, none is `None` -- because
    this hop *drops* a live block rather than padding in a fresh one: the
    prefix (head stats through the three scalars) is copied unchanged, the
    valuation block is skipped outright, and the ledger tail is copied from
    wherever it actually sits in the wider source.

    The two offsets it needs are read by name off
    `hexset.encoding.global_columns`, never counted out from the feature
    constants here: the valuation block sat exactly where contract 6's
    ledger tail now starts, so the prefix is that tail's own start and the
    tail is its own slice. A hand-copied offset table is the drift this
    module exists to migrate across, and it has no business containing a
    second one.
    """
    ledger_block = global_columns(players)["ledger"]
    prefix = ledger_block.start
    valuation = _valuation(players)
    ledger = ledger_block.stop - ledger_block.start
    old_total = prefix + valuation + ledger
    columns = list(range(prefix))
    columns.extend(range(prefix + valuation, old_total))
    new_total = global_features(players)
    if len(columns) != new_total:
        raise AssertionError(
            f"built {len(columns)} contract-6 global columns, expected {new_total}"
        )
    return columns


def _recolumn(tensor: Tensor, target: torch.Size, columns: Sequence[int]) -> Tensor:
    if tensor.shape[0] != target[0] or target[1] != len(columns):
        raise ValueError(
            f"embed_global.weight: {tuple(tensor.shape)} -> {tuple(target)} is "
            f"not a map onto contract 6's {len(columns)} global columns"
        )
    if any(c >= tensor.shape[1] for c in columns):
        raise ValueError(
            f"embed_global.weight: {tensor.shape[1]} global features is not "
            "wide enough for the columns this migration reads"
        )
    out = torch.empty(target, dtype=tensor.dtype)
    for new, old in enumerate(columns):
        out[:, new] = tensor[:, old]
    return out


def _migrate_heads(
    state: dict, template: dict, players: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The four head tensors the knight collapse touches.

    Built generically off the *new* (contract-6) action space's own readout,
    never off a hand-rolled contract-5 `ActionSpace`: only `PLAY_KNIGHT`'s
    per-node stride differs between the two contracts (`num_players + 1`,
    the same as `MOVE_ROBBER`'s, versus the single spend-only slot contract
    6 gives it), and every other kind's width, order and node count is read
    straight off the space this migration is building the checkpoint to run
    on. `span` is that stride.
    """
    span = players + 1
    old_hexes_weight = state["heads.hexes.weight"]
    old_hexes_bias = state["heads.hexes.bias"]
    if old_hexes_weight.shape[0] != 2 * span:
        raise ValueError(
            f"heads.hexes.weight has {old_hexes_weight.shape[0]} rows, expected "
            f"{2 * span} (MOVE_ROBBER plus contract 5's per-hex PLAY_KNIGHT "
            "block, both num_players + 1 wide) -- this checkpoint does not "
            "look like contract 5"
        )
    new_hexes_weight = old_hexes_weight[:span].clone()
    new_hexes_bias = old_hexes_bias[:span].clone()
    knight_weight = old_hexes_weight[span:].mean(dim=0)
    knight_bias = old_hexes_bias[span:].mean()

    old_globals_weight = state["heads.globals.weight"]
    old_globals_bias = state["heads.globals.bias"]
    new_globals_weight = torch.zeros_like(template["heads.globals.weight"])
    new_globals_bias = torch.zeros_like(template["heads.globals.bias"])

    space = space_for(_dummy_game(players))
    globals_head = plan(space).head(GLOBALS)
    new_col = 0
    old_col = 0
    for kind in globals_head.kinds:
        size = space.sizes[kind]
        if kind is ActionType.PLAY_KNIGHT:
            new_globals_weight[new_col] = knight_weight
            new_globals_bias[new_col] = knight_bias
            new_col += size
            continue
        new_globals_weight[new_col : new_col + size] = old_globals_weight[
            old_col : old_col + size
        ]
        new_globals_bias[new_col : new_col + size] = old_globals_bias[
            old_col : old_col + size
        ]
        new_col += size
        old_col += size
    if old_col != old_globals_weight.shape[0]:
        raise AssertionError(
            f"consumed {old_col} of {old_globals_weight.shape[0]} contract-5 "
            "GLOBALS rows -- the kind accounting above missed one"
        )
    if new_col != new_globals_weight.shape[0]:
        raise AssertionError(
            f"wrote {new_col} of {new_globals_weight.shape[0]} contract-6 "
            "GLOBALS rows -- the kind accounting above missed one"
        )
    return new_hexes_weight, new_hexes_bias, new_globals_weight, new_globals_bias


def _dummy_game(players: int, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    return start(board, players, rng)


def migrate_state_dict(
    state: dict[str, Tensor], template: dict[str, Tensor], players: int
) -> dict[str, Tensor]:
    """`state` (a contract-5 checkpoint's `net`) reshaped to `template`'s
    shapes -- everything but `embed_global.weight`, the two head tensors the
    knight collapse touches, `flat_gather`, and the new `aux_margin.*` keys
    passes through unchanged. The value head is returned **unscaled**: the
    caller applies `1 / T` once `fit_temperature` has read it.

    A key already at its template shape passes straight through before any
    of the contract-5-specific remaps run, so a checkpoint that is already
    contract 6 (every shape matches) is untouched end to end -- the remaps
    themselves assume contract 5's wider shapes and would refuse it.
    """
    columns = _global_columns(players)
    hexes_w = hexes_b = globals_w = globals_b = None

    def heads() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        nonlocal hexes_w, hexes_b, globals_w, globals_b
        if hexes_w is None:
            hexes_w, hexes_b, globals_w, globals_b = _migrate_heads(state, template, players)
        return hexes_w, hexes_b, globals_w, globals_b

    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key in REBUILT_KEYS:
            out[key] = template[key].clone()
            continue
        if key not in template:
            raise KeyError(f"{key} is in the checkpoint but not in the migrated net")
        target = template[key].shape
        if tuple(tensor.shape) == tuple(target):
            out[key] = tensor.clone()
            continue
        if key == "embed_global.weight":
            out[key] = _recolumn(tensor, target, columns)
            continue
        if key == "heads.hexes.weight":
            out[key] = heads()[0]
            continue
        if key == "heads.hexes.bias":
            out[key] = heads()[1]
            continue
        if key == "heads.globals.weight":
            out[key] = heads()[2]
            continue
        if key == "heads.globals.bias":
            out[key] = heads()[3]
            continue
        raise ValueError(
            f"{key}: {tuple(tensor.shape)} -> {tuple(target)} is not part of the "
            "contract-6 migration (only embed_global.weight and the hexes/"
            "globals policy heads change shape)"
        )
    if "aux_margin.weight" not in state:
        # A genuine contract-5 checkpoint has no aux head at all: warm-start
        # it as an exact copy of the (still-unscaled) old value head. A
        # checkpoint that already has one (already contract 6) already
        # passed its own through the shape-match branch above.
        out["aux_margin.weight"] = state["value.weight"].clone()
        out["aux_margin.bias"] = state["value.bias"].clone()
    missing = set(template) - set(out)
    if missing:
        raise KeyError(f"the migrated net has keys the checkpoint lacks: {sorted(missing)}")
    return out


def _net(players: int, config: ModelConfig, board_seed: int) -> HexNet:
    game = _dummy_game(players, board_seed)
    return HexNet(space_for(game), static_graph(game.state(0, hidden=False).board.topology), players, config)


def _observations(players: int, count: int, seed: int) -> list:
    """Real contract-6 positions from seeded random play, for the
    `embed_global` zero-column check. Trade-free (nobody is seated with a
    gate) is irrelevant here -- this check is about the *input* columns
    contract 5 no longer has, not about what any seat published."""
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        board = random_base_board(rng)
        game = start(board, players, rng)
        for _ in range(40 + (i % 80)):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.append(encode(game))
    return out


@torch.no_grad()
def compare(migrated: HexNet, columns: Sequence[int], observations: list, players: int) -> dict[str, float]:
    """The one thing a single process can check without the removed engine
    code: that `embed_global`'s output on a real contract-6 position equals
    what contract 5's own (wider) layer would have computed on the same
    position padded back out to contract 5's width with the dropped columns
    held at zero -- the "nothing was ever published" case. See the module
    docstring for exactly what this does and does not claim.
    """
    batch = collate(observations)
    globals_ = batch[3]
    old_width = source_globals(players)
    padded = torch.zeros(globals_.shape[0], old_width, dtype=globals_.dtype)
    padded[:, columns] = globals_

    weight = migrated.embed_global.weight
    bias = migrated.embed_global.bias
    old_weight = torch.zeros(weight.shape[0], old_width, dtype=weight.dtype)
    old_weight[:, columns] = weight
    restricted = torch.nn.functional.linear(padded, old_weight, bias)
    actual = migrated.embed_global(globals_)
    return {
        "max_abs_embed_delta_on_zero_valuation_positions": float(
            (restricted - actual).abs().max()
        ),
    }


def _win_temperature_samples(
    players: int, games: int, seed: int, action_cap: int = 4000
) -> tuple[list, np.ndarray]:
    """`(observations, winner_slot)` from `games` heximax x `players` games
    on the target engine, one sample per turn at the mover's first decision
    -- the same protocol used to fit `hexset.bots.search2.win`'s
    `WIN_TEMPERATURE`. `winner_slot` is the
    eventual winner's seat-relative index from that sample's mover
    (`0` if the mover itself goes on to win), matching the frame the value
    head's own rows are read in.

    An unfinished game (the action cap, not a winner) contributes no
    samples: there is no "eventual winner" to label them with.
    """
    from hexset.actions import apply
    from hexset.arena import entrant_from_name, spawn
    from hexset.game import to_move

    # By name, through the arena: HexSet ships no bots, and `heximax` is
    # whatever bot the loaded runtime (`--runtime`) put behind the name.
    heximax = entrant_from_name("heximax")

    observations: list = []
    winner_slots: list[int] = []
    for g in range(games):
        board = random_base_board(random.Random(f"{seed}:{g}:board"))
        game = start(board, players, random.Random(f"{seed}:{g}:game"))
        bots = [
            spawn(heximax, board, random.Random(f"{seed}:{g}:{s}")) for s in range(players)
        ]
        game.gates = tuple(bots)
        # Reproduce the historical protocol: contract 5's default was the
        # exhaustive automatic clearing house, uncapped. The engine's default
        # is now the trade round, so the mechanism is set explicitly. There
        # is no table cap left to lift: since HexSet 0.60 each seat's bot
        # declares its own budget.
        game.trade_mode = "auto"
        pending: list[tuple[object, int]] = []
        last_turns = -1
        actions = 0
        while not is_over(game) and actions < action_cap:
            seat = to_move(game)
            if game.turns != last_turns:
                pending.append((encode(game, seat), seat))
                last_turns = game.turns
            apply(game, bots[seat].choose(game))
            actions += 1
        if game.won_by is None:
            continue
        for observation, seat in pending:
            observations.append(observation)
            winner_slots.append((game.won_by - seat) % players)
    return observations, np.asarray(winner_slots, dtype=np.int64)


def _cross_entropy(margins: np.ndarray, winner_slots: np.ndarray, temperature: float) -> float:
    scaled = margins / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=1, keepdims=True)
    chosen = probs[np.arange(len(margins)), winner_slots]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, None))))


def _fit_temperature(margins: np.ndarray, winner_slots: np.ndarray) -> tuple[float, float]:
    """The temperature minimising `_cross_entropy`, by golden-section search
    over `log(T)` in a wide bracket -- no scipy dependency, and log-space
    because a temperature is a positive scale, not an additive offset."""
    phi = (5 ** 0.5 - 1) / 2
    lo, hi = math.log(1e-2), math.log(1e2)

    def loss_at(log_t: float) -> float:
        return _cross_entropy(margins, winner_slots, math.exp(log_t))

    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = loss_at(c), loss_at(d)
    for _ in range(200):
        if b - a < 1e-6:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss_at(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss_at(d)
    best_log_t = (a + b) / 2
    return math.exp(best_log_t), loss_at(best_log_t)


def fit_win_temperature(
    net: HexNet, players: int, *, games: int = 24, seed: int = 90000
) -> dict[str, float]:
    """`T` and its log loss, fitted on `games` heximax x `players` games
    played fresh on the target engine (contract 6 has no bank of recorded
    games yet). `net`'s value head must still be the **unscaled** copy of
    the source checkpoint's margin head -- its raw output on these positions
    is exactly the quantity `T` calibrates.
    """
    observations, winner_slots = _win_temperature_samples(players, games, seed)
    if not observations:
        raise RuntimeError(
            f"none of {games} heximax games (seed {seed}) reached a winner; "
            "cannot fit a win temperature without labels"
        )
    buffer = collate(observations)
    with torch.no_grad():
        prediction = net(*buffer)
    margins = prediction.value_logits.numpy()
    temperature, log_loss = _fit_temperature(margins, winner_slots)
    return {
        "temperature": temperature,
        "log_loss": log_loss,
        "log_loss_uniform": math.log(players),
        "games": float(games),
        "samples": float(len(observations)),
    }


def migrate_checkpoint(
    source: Path,
    *,
    players: int = 4,
    board_seed: int = 0,
    observations: int = 256,
    win_games: int = 24,
    win_seed: int = 90000,
) -> tuple[dict, dict[str, float]]:
    """The migrated checkpoint and the migration report, nothing written."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    args = dict(state["args"])
    config = config_from_args(args)
    net = _net(players, config, board_seed)
    template = net.state_dict()

    new_width = global_features(players)
    old_width = state["net"]["embed_global.weight"].shape[1]
    if old_width == new_width:
        raise SystemExit(f"{source} already reads {new_width} global features; nothing to migrate")
    expected_old = source_globals(players)
    if old_width != expected_old:
        raise SystemExit(
            f"{source} reads {old_width} global features, not contract 5's "
            f"{expected_old} at {players} players -- this migration reads "
            "only contract 5"
        )

    columns = _global_columns(players)
    migrated = migrate_state_dict(state["net"], template, players)
    net.load_state_dict(migrated, strict=True)
    net.eval()

    report = compare(net, columns, _observations(players, observations, seed=1000), players)
    report["global_features_before"] = float(old_width)
    report["global_features_after"] = float(new_width)
    report["action_slots_after"] = float(net.readout.size)

    temperature_report = fit_win_temperature(net, players, games=win_games, seed=win_seed)
    report.update(temperature_report)

    T = temperature_report["temperature"]
    migrated["value.weight"] = migrated["value.weight"] / T
    migrated["value.bias"] = migrated["value.bias"] / T
    net.load_state_dict(migrated, strict=True)
    net.eval()

    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=float(args.get("learning_rate", 3e-4)),
        eps=float(args.get("adam_eps", ADAM_EPS)),
    )
    out = {
        "iteration": state["iteration"],
        "games_started": state["games_started"],
        "net": net.state_dict(),
        "optimiser": optimiser.state_dict(),
        "torch_rng": state["torch_rng"],
        "args": args,
        "config": asdict(config),
        "migrate": {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_iteration": state["iteration"],
            "global_features_from": old_width,
            "global_features_to": new_width,
            "contract_from": "5",
            "contract_to": "6",
            "report": report,
        },
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--observations", type=int, default=256,
                        help="real positions the embed_global exactness report is computed on")
    parser.add_argument("--win-games", type=int, default=24,
                        help="heximax x players games played fresh to fit the win temperature")
    parser.add_argument("--win-seed", type=int, default=90000)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    runtime.add_argument(parser)
    args = parser.parse_args(argv)
    runtime.load(args.runtime)

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    checkpoint, report = migrate_checkpoint(
        args.checkpoint,
        players=args.players,
        observations=args.observations,
        win_games=args.win_games,
        win_seed=args.win_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    for key, value in report.items():
        print(f"{key}: {value:.6e}" if isinstance(value, float) else f"{key}: {value}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
