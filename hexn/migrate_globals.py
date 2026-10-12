# SPDX-License-Identifier: GPL-3.0-only
"""Migrate a contract-6 checkpoint onto contract 7's wider globals, computing
exactly what it computed before.

Contract 7 appends two blocks to the end of `hexset.encoding`'s globals, after
the ledger tail: each seat's roads to take Longest Road (`roads_to_take`) and
a one-hot of the seat to move (`to_move`), `2 * players` floats in all. Every
earlier column keeps its offset, and the only parameter that reads the
globals is the global input projection, `embed_global` (`hexn.model.HexNet`).
So the migration appends that many **zero** columns to `embed_global.weight`
and changes nothing else in the net: on any position, the new columns are
multiplied by zero, and the migrated net's logits, value and margin are the
old net's. Training then learns what the new columns are worth.

The optimiser's Adam moments for that weight are widened the same way (zero
first and second moments for the new columns, what a fresh parameter starts
with), so `--resume` continues from the migrated file. Everything else in the
checkpoint (iteration, game count, RNG, args) passes through.

    python -m hexn.migrate_globals --checkpoint runs/example/iter-00200.pt \
        --out runs/example/iter-00200-c7.pt

`--observations N` (default 64) checks the claim before anything is written:
the old net on the contract-6 encoding (the new encoding minus its last
`2 * players` columns) against the migrated net on the full encoding, on `N`
positions from seeded random play, from every seat. The largest absolute
difference of each output is printed and stored in the checkpoint's
`migrate_globals` block.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
from pathlib import Path

import torch
from torch import Tensor

from hexset.actions import space_for
from hexset.board.board import random_base_board
from hexset.encoding import encode, global_columns, global_features, static_graph
from hexset.game import is_over, start
from hexset.play import step_randomly

from .model import HexNet, collate, config_from_args

__all__ = [
    "WEIGHT",
    "appended",
    "compare",
    "migrate_checkpoint",
    "source_globals",
    "widen_optimiser",
    "widen_state_dict",
]

WEIGHT = "embed_global.weight"


def source_globals(players: int) -> int:
    """Contract 6's globals width: today's up to the end of the ledger tail."""
    return global_columns(players)["ledger"].stop


def appended(players: int) -> int:
    """The columns contract 7 appends: `roads_to_take` and `to_move`."""
    columns = global_columns(players)
    added = (columns["roads_to_take"].stop - columns["roads_to_take"].start) + (
        columns["to_move"].stop - columns["to_move"].start
    )
    if source_globals(players) + added != global_features(players):
        raise AssertionError("the contract-7 blocks are not the globals' tail")
    return added


def _widen(tensor: Tensor, players: int, what: str) -> Tensor:
    old, new = source_globals(players), global_features(players)
    if tensor.dim() != 2 or tensor.shape[1] != old:
        raise ValueError(
            f"{what}: shape {tuple(tensor.shape)} does not read contract 6's "
            f"{old} global features at {players} players"
        )
    return torch.cat([tensor, tensor.new_zeros(tensor.shape[0], new - old)], dim=1)


def widen_state_dict(state: dict[str, Tensor], players: int) -> dict[str, Tensor]:
    """`state` (a contract-6 net's `state_dict`) with zero columns appended to
    `embed_global.weight`; every other entry is the same tensor."""
    out = dict(state)
    out[WEIGHT] = _widen(state[WEIGHT], players, WEIGHT)
    return out


def widen_optimiser(optimiser: dict, index: int, players: int) -> dict:
    """An Adam `state_dict` whose parameter `index` is `embed_global.weight`,
    with that parameter's moments widened by zero columns. Other parameters'
    state is shared, not copied."""
    out = {"state": dict(optimiser["state"]), "param_groups": copy.deepcopy(optimiser["param_groups"])}
    entry = optimiser["state"].get(index)
    if entry is not None:
        entry = dict(entry)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in entry:
                entry[key] = _widen(entry[key], players, f"optimiser state {index}.{key}")
        out["state"][index] = entry
    return out


def _net(players: int, args: dict, board_seed: int = 0) -> HexNet:
    rng = random.Random(board_seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    return HexNet(space_for(game), static_graph(board.topology), players, config_from_args(args))


def _positions(players: int, count: int, seed: int) -> list[tuple]:
    """`count` positions from seeded random play, each with every seat as
    the perspective: `(game, seat)` pairs."""
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        game = start(random_base_board(rng), players, rng)
        for _ in range(rng.randrange(0, 400)):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.extend((game, seat) for seat in range(players))
    return out


@torch.no_grad()
def compare(old: HexNet, new: HexNet, observations: list, players: int) -> dict[str, float]:
    """Largest absolute difference of each output between `old` on the
    contract-6 globals (the encoding minus its appended columns) and `new` on
    the full encoding."""
    hexes, vertices, edges, globals_ = collate(observations)
    width = source_globals(players)
    before = old(hexes, vertices, edges, globals_[:, :width].contiguous())
    after = new(hexes, vertices, edges, globals_)
    report = {"positions": float(len(observations))}
    for name in ("logits", "value_logits", "value", "margin", "fast"):
        a, b = getattr(before, name, None), getattr(after, name, None)
        if a is None or b is None:
            continue
        report[f"max_abs_{name}"] = float((a.float() - b.float()).abs().max())
    return report


def migrate_checkpoint(
    source: Path, *, observations: int = 64, seed: int = 0
) -> tuple[dict, dict[str, float]]:
    """The migrated checkpoint and its report, nothing written."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    args = dict(state.get("args", {}))
    players = int(args.get("players", 4))
    width = state["net"][WEIGHT].shape[1]
    if width == global_features(players):
        raise SystemExit(f"{source} already reads {width} global features; nothing to migrate")

    net = _net(players, args)
    names = [name for name, _ in net.named_parameters()]
    widened = widen_state_dict(state["net"], players)
    net.load_state_dict(widened, strict=True)
    net.eval()

    report: dict[str, float] = {
        "global_features_before": float(width),
        "global_features_after": float(global_features(players)),
    }
    if observations:
        old = _net(players, args)
        old.embed_global = torch.nn.Linear(width, old.embed_global.out_features)
        old.load_state_dict(state["net"], strict=True)
        old.eval()
        encoded = [encode(game, seat) for game, seat in _positions(players, observations, seed)]
        report.update(compare(old, net, encoded, players))

    out = dict(state)
    out["net"] = widened
    if "optimiser" in state:
        out["optimiser"] = widen_optimiser(state["optimiser"], names.index(WEIGHT), players)
    out["migrate_globals"] = {
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "global_features_from": width,
        "global_features_to": global_features(players),
        "contract_from": "6",
        "contract_to": "7",
        "report": report,
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--observations", type=int, default=64,
                        help="positions (each from every seat) the exactness check runs on; 0 skips it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    checkpoint, report = migrate_checkpoint(
        args.checkpoint, observations=args.observations, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(args.out)
    for key, value in report.items():
        print(f"{key}: {value:.6e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
