# SPDX-License-Identifier: GPL-3.0-only
"""Migrate a contract-7 checkpoint onto contract 8's inputs.

Contract 8 (`hexset.encoding`) re-encodes the hex, vertex, edge and global
inputs, so the four input projections (`embed_hex`, `embed_vertex`,
`embed_edge`, `embed_global` in `hexn.model.HexNet`) change width; every other
parameter is copied as it is. Each projection is refitted by least squares so
that, on the same positions, the migrated net's embedding matches the old
net's pre-activation: rows `(x8, W7 @ x7 + b7)`, from every seat, solved for
`(W8, b8)`. Where contract 8 restates contract 7 linearly -- the five
producing terrains with has-token and the desert, the edges' and ports'
columns, the summed hand and card counts -- the fit is exact (its residual is
reported). Where it does not -- a building's points against the building and
owner one-hots, the award reads against the holder one-hots, the expected
hands against the ledger's known and unknown -- the fit is the best linear
one on the positions given. Contract 8's new information (`seat_reach`,
`own_dev_playable`, `opponent_dev_ages`, `dev_odds`) starts
at zero weight: the migrated net ignores it until training reads it.

The contract-7 inputs come from the record, which still carries every field
contract 7 read (`hexset.onnx_record`), through `_Contract7Inputs`: contract
7's own record reader, frozen here at its last layout for this migration
alone.

    python -m hexn.migrate_c8 --checkpoint runs/example/iter-00050.pt \
        --out runs/example/iter-00050-c8.pt --runtime <module> --entrant <bot>

Positions come from seeded games among four `--entrant` seats
(`hexset.arena`; `random` plays uniformly over the legal moves), every
`--stride` decisions from every seat. The report -- each projection's largest
residual, and each output's largest difference between the old net on
contract 7 and the migrated net on contract 8, on positions held out from the
fit -- is printed and stored in the checkpoint's `migrate_c8` block. The
optimiser state is not carried: a migrated net starts a run (`--base`), it
does not resume one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hexset.actions import apply, options_for, space_for
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.encoding import RULE_BLOCKS, encode, global_columns, static_graph, vertex_features
from hexset.game import Game, is_over, start, to_move
from hexset.onnx_record import record_batch
from hexset.state import NO_OWNER

from . import runtime
from .export_onnx import _onehot, _pips_table, _rotate_slot
from .model import HexNet, collate, config_from_args
from .policy import masked_log_softmax

__all__ = ["migrate_checkpoint", "positions", "PROJECTIONS"]

PROJECTIONS = ("embed_hex", "embed_vertex", "embed_edge", "embed_global")

# Contract 7's layout, frozen: what `_Contract7Inputs` reads.
NUM_TERRAIN = 8
NUM_BUILDINGS = 3
NUM_PHASES = 7
HAND_SCALE = 10.0
BANK_SCALE = 19.0
TURN_SCALE = 200.0
DECK_SIZE = 25
ROADS_TO_TAKE_CAP = 4
C7_HEX, C7_EDGE = NUM_TERRAIN + 3, 5
_C7_FIELDS = (
    "terrain", "token", "port_code", "robber", "vertex_owner", "vertex_building",
    "edge_owner", "bank", "knights_played", "award_points", "longest_road_holder",
    "largest_army_holder", "phase", "free_roads", "deck_size", "turns",
    "roads_to_take", "to_move", "perspective", "own_hand", "hand_totals", "own_dev",
    "dev_totals", "ledger_known", "ledger_unknown",
)


def c7_widths(players: int) -> dict[str, int]:
    """Contract 7's input widths at `players` seats."""
    return {
        "embed_hex": C7_HEX,
        "embed_vertex": NUM_BUILDINGS + players + 1 + 1 + NUM_RESOURCES,
        "embed_edge": players + 1,
        "embed_global": (
            NUM_RESOURCES + (players - 1) + NUM_RESOURCES + NUM_RESOURCES + (players - 1)
            + players + players + 2 * (players + 1) + NUM_PHASES + 3
            + (players - 1) * (NUM_RESOURCES + 1) + 2 * players
        ),
    }


class _Contract7Inputs(nn.Module):
    """Contract 7's record reader, as `hexn.export_onnx.RecordEncoder` was at
    contract 7: `record -> (hexes, vertices, edges, globals)`."""

    def __init__(self, num_hexes: int, players: int) -> None:
        super().__init__()
        self.players = players
        self.num_hexes = num_hexes
        self.register_buffer(
            "pips_table", torch.tensor(_pips_table(), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        terrain: Tensor,
        token: Tensor,
        port_code: Tensor,
        robber: Tensor,
        vertex_owner: Tensor,
        vertex_building: Tensor,
        edge_owner: Tensor,
        bank: Tensor,
        knights_played: Tensor,
        award_points_: Tensor,
        longest_road_holder: Tensor,
        largest_army_holder: Tensor,
        phase: Tensor,
        free_roads: Tensor,
        deck_size: Tensor,
        turns: Tensor,
        roads_to_take: Tensor,
        to_move: Tensor,
        perspective: Tensor,
        own_hand: Tensor,
        hand_totals: Tensor,
        own_dev: Tensor,
        dev_totals: Tensor,
        ledger_known: Tensor,
        ledger_unknown: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        players = self.players

        # --- hexes: terrain one-hot, has-token, pips/5, is-robber ---
        terrain_onehot = _onehot(terrain, NUM_TERRAIN)
        has_token = (token != 0).to(torch.float32)
        pips_scaled = self.pips_table[token]
        is_robber = _onehot(robber, self.num_hexes)
        hexes = torch.cat(
            [
                terrain_onehot,
                has_token.unsqueeze(-1),
                pips_scaled.unsqueeze(-1),
                is_robber.unsqueeze(-1),
            ],
            dim=-1,
        )

        # --- vertices: building one-hot, owner slot one-hot, port block ---
        building_onehot = _onehot(vertex_building, NUM_BUILDINGS)
        owner_slot = _rotate_slot(vertex_owner, perspective, players)
        owner_onehot = _onehot(owner_slot, players + 1)
        is_generic_port = (port_code == 0).to(torch.float32)
        specific_resource = torch.clamp(port_code - 1, min=0)
        has_specific_port = (port_code >= 1).to(torch.float32)
        specific_port_onehot = _onehot(specific_resource, NUM_RESOURCES) * has_specific_port.unsqueeze(-1)
        vertices = torch.cat(
            [
                building_onehot,
                owner_onehot,
                is_generic_port.unsqueeze(-1),
                specific_port_onehot,
            ],
            dim=-1,
        )

        # --- edges: owner slot one-hot ---
        edge_slot = _rotate_slot(edge_owner, perspective, players)
        edges = _onehot(edge_slot, players + 1)

        # --- building points, already in seat-relative slot order because
        # `owner_onehot`'s columns already are: mirrors
        # `encoding._building_points`, a contraction of the vertex block
        # rather than a second board walk.
        building_value = torch.arange(NUM_BUILDINGS, dtype=torch.float32, device=vertices.device)
        per_vertex_value = building_onehot @ building_value
        building_points = torch.einsum("bv,bvp->bp", per_vertex_value, owner_onehot[..., :players])

        # --- globals, in exactly `encoding._encode_globals`'s part order ---
        seats = [torch.remainder(perspective + i, players) for i in range(players)]

        def gather_seat(values: Tensor, seat: Tensor) -> Tensor:
            return values.gather(1, seat.unsqueeze(1)).squeeze(1)

        def gather_seat_vec(values: Tensor, seat: Tensor) -> Tensor:
            """`gather_seat` for a `(B, players, width)` tensor -- the
            per-seat `width`-wide row instead of a scalar."""
            width = values.shape[-1]
            index = seat.view(-1, 1, 1).expand(-1, 1, width)
            return values.gather(1, index).squeeze(1)

        parts: list[Tensor] = [own_hand.to(torch.float32) / HAND_SCALE]
        parts.append(
            torch.stack(
                [gather_seat(hand_totals, seats[i]).to(torch.float32) / HAND_SCALE for i in range(1, players)],
                dim=1,
            )
        )
        parts.append(bank.to(torch.float32) / BANK_SCALE)
        parts.append(own_dev.to(torch.float32) / 5.0)
        parts.append(
            torch.stack(
                [gather_seat(dev_totals, seats[i]).to(torch.float32) / 5.0 for i in range(1, players)],
                dim=1,
            )
        )
        parts.append(
            torch.stack(
                [gather_seat(knights_played, seats[i]).to(torch.float32) / 5.0 for i in range(players)],
                dim=1,
            )
        )
        award_seat = torch.stack(
            [gather_seat(award_points_, seats[i]).to(torch.float32) for i in range(players)], dim=1
        )
        parts.append((building_points + award_seat) / 10.0)

        for holder in (longest_road_holder, largest_army_holder):
            parts.append(_onehot(_rotate_slot(holder, perspective, players), players + 1))

        parts.append(_onehot(phase, NUM_PHASES))
        parts.append((free_roads.to(torch.float32) / 2.0).unsqueeze(-1))
        parts.append((deck_size.to(torch.float32) / DECK_SIZE).unsqueeze(-1))
        parts.append(torch.clamp(turns.to(torch.float32) / TURN_SCALE, max=1.0).unsqueeze(-1))

        # There is no public valuation block here any more: contract 6
        # deleted trading's whole public layer (`hexset.trading`), so
        # `encoding._encode_globals` goes straight from the three scalars
        # above to the ledger tail below, and so does this trace.

        # --- the public-knowledge ledger, in exactly `encoding._ledger_parts`'s
        # order: each opponent's known[5] then unknown, seat-relative, own
        # seat excluded (own hand is already exact via `own_hand` above).
        for i in range(1, players):
            parts.append(gather_seat_vec(ledger_known, seats[i]).to(torch.float32) / HAND_SCALE)
            parts.append(
                (gather_seat(ledger_unknown, seats[i]).to(torch.float32) / HAND_SCALE).unsqueeze(-1)
            )

        # --- appended after the ledger, `encoding._roads_to_take_parts`
        # then the mover one-hot: each seat's roads to take Longest Road over
        # the cap, seat-relative, and whose decision it is.
        parts.append(
            torch.stack(
                [
                    gather_seat(roads_to_take, seats[i]).to(torch.float32) / ROADS_TO_TAKE_CAP
                    for i in range(players)
                ],
                dim=1,
            )
        )
        parts.append(_onehot(_rotate_slot(to_move, perspective, players), players))

        globals_ = torch.cat(parts, dim=-1)

        return hexes, vertices, edges, globals_



# The global blocks a contract-7 net has nothing to say about, contract 8's
# and contract 8.1's rules: fitted at zero weight. Every other block is fitted.
_NEW_GLOBALS = ("own_dev_playable", "opponent_dev_ages", "dev_odds", *RULE_BLOCKS)


def _fitted_columns(name: str, players: int) -> np.ndarray:
    """The contract-8 input columns of projection `name` the fit may weight."""
    if name == "embed_global":
        columns = global_columns(players)
        keep = np.ones(max(s.stop for s in columns.values()), dtype=bool)
        for block in _NEW_GLOBALS:
            keep[columns[block]] = False
        return np.flatnonzero(keep)
    if name == "embed_vertex":
        # `seat_building` and the ports; `seat_reach`, the last `players`, is new.
        return np.arange(vertex_features(players) - players)
    return None  # every column


def positions(
    players: int, games: int, seed: int, entrant: str = "random", stride: int = 7
) -> list[tuple[Game, int]]:
    """`(game, seat)` pairs from `games` seeded games among `players` seats of
    `entrant`: every `stride` decisions, the position from every seat. A
    position is a copy, so later play does not move it."""
    from hexset.arena import entrant_from_name, spawn

    out: list[tuple[Game, int]] = []
    for g in range(games):
        rng = random.Random(seed + g)
        board = random_base_board(rng)
        game = start(board, players, rng)
        bots = None
        if entrant != "random":
            spec = entrant_from_name(entrant)
            bots = [spawn(spec, board, random.Random(rng.random())) for _ in range(players)]
        step = 0
        while not is_over(game) and step < 4000:
            if step % stride == 0:
                frozen = copy.deepcopy(game)
                out.extend((frozen, seat) for seat in range(players))
            if bots is None:
                action = rng.choice(options_for(game))
            else:
                action = bots[to_move(game)].choose(game)
            apply(game, action)
            step += 1
    return out


def _inputs(pairs, space, players: int, num_hexes: int):
    """Contract 7's and contract 8's inputs for `pairs`, as four
    `(rows, width)` pairs of float tensors, in `PROJECTIONS` order."""
    record = record_batch(pairs, space)
    reader = _Contract7Inputs(num_hexes, players)
    with torch.no_grad():
        old = reader(*(torch.from_numpy(record[name]) for name in _C7_FIELDS))
    new = collate([encode(game, seat) for game, seat in pairs])
    return [
        (o.reshape(-1, o.shape[-1]).double(), n.reshape(-1, n.shape[-1]).double())
        for o, n in zip(old, new)
    ]


def _net(players: int, args: dict, widths: dict[str, int] | None = None) -> HexNet:
    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, players, rng)
    net = HexNet(space_for(game), static_graph(board.topology), players, config_from_args(args))
    for name, width in (widths or {}).items():
        setattr(net, name, nn.Linear(width, getattr(net, name).out_features))
    return net


def _fit(old: Tensor, new: Tensor, weight: Tensor, bias: Tensor, columns: np.ndarray):
    """`(W8, b8, residual)`: least squares of `new[:, columns]` plus a bias
    onto the old pre-activation `old @ weight.T + bias`."""
    target = old @ weight.double().T + bias.double()
    design = torch.cat([new[:, columns], torch.ones(new.shape[0], 1, dtype=new.dtype)], dim=1)
    solution = torch.linalg.lstsq(design, target, driver="gelsd").solution
    fitted = torch.zeros(weight.shape[0], new.shape[1], dtype=torch.float64)
    fitted[:, columns] = solution[:-1].T
    residual = float((design @ solution - target).abs().max())
    return fitted.float(), solution[-1].float(), residual


def migrate_checkpoint(
    source: Path, *, games: int = 64, seed: int = 0, entrant: str = "random",
    stride: int = 7, held_out: float = 0.25,
) -> tuple[dict, dict[str, float]]:
    """The migrated checkpoint and its report, nothing written."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    args = dict(state.get("args", {}))
    players = int(args.get("players", 4))
    widths = c7_widths(players)
    found = {name: state["net"][f"{name}.weight"].shape[1] for name in PROJECTIONS}
    if found != widths:
        raise SystemExit(f"{source} reads inputs {found}, not contract 7's {widths}")

    old_net = _net(players, args, widths)
    old_net.load_state_dict(state["net"], strict=True)
    old_net.eval()

    # Whole games held out: the report is on games the fit never saw.
    fit_games = max(1, round(games * (1 - held_out)))
    fit_pairs = positions(players, fit_games, seed, entrant, stride)
    test_pairs = positions(players, games - fit_games, seed + fit_games, entrant, stride)
    space = space_for(fit_pairs[0][0])
    num_hexes = fit_pairs[0][0].state(0, hidden=False).board.num_hexes
    fit_inputs = _inputs(fit_pairs, space, players, num_hexes)

    new_state = {k: v for k, v in state["net"].items() if not k.startswith(PROJECTIONS)}
    report: dict[str, float] = {"positions_fit": float(len(fit_pairs)), "positions_held_out": float(len(test_pairs))}
    for (old, new), name in zip(fit_inputs, PROJECTIONS):
        columns = _fitted_columns(name, players)
        if columns is None:
            columns = np.arange(new.shape[1])
        weight, bias, residual = _fit(
            old, new, state["net"][f"{name}.weight"], state["net"][f"{name}.bias"], columns
        )
        new_state[f"{name}.weight"] = weight
        new_state[f"{name}.bias"] = bias
        report[f"fit_max_residual_{name}"] = residual

    net = _net(players, args)
    net.load_state_dict(new_state, strict=True)
    net.eval()
    if test_pairs:
        report.update(_held_out(old_net, net, test_pairs, space, players, num_hexes))

    out = {k: v for k, v in state.items() if k not in ("optimiser", "net")}
    out["net"] = new_state
    out["migrate_c8"] = {
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "contract_from": "7",
        "contract_to": "8",
        "entrant": entrant,
        "games": games,
        "seed": seed,
        "stride": stride,
        "report": report,
    }
    return out, report


@torch.no_grad()
def _held_out(old_net: HexNet, net: HexNet, pairs, space, players: int, num_hexes: int) -> dict[str, float]:
    """On `pairs`: each projection's largest and root-mean-square difference
    between the old net on contract 7 and the migrated net on contract 8;
    and, where the seat is the one to decide, how often the two pick the same
    legal move, the mean KL of the old net's legal-move distribution from the
    new one's, and the mean and largest change in the seat's own win
    probability."""
    record = record_batch(pairs, space)
    reader = _Contract7Inputs(num_hexes, players)
    old_inputs = reader(*(torch.from_numpy(record[name]) for name in _C7_FIELDS))
    new_inputs = collate([encode(game, seat) for game, seat in pairs])
    report: dict[str, float] = {}
    for name, o, n in zip(PROJECTIONS, old_inputs, new_inputs):
        delta = getattr(old_net, name)(o.float()) - getattr(net, name)(n.float())
        report[f"held_out_max_abs_{name}"] = float(delta.abs().max())
        report[f"held_out_rms_{name}"] = float(delta.pow(2).mean().sqrt())
    before = old_net(*(t.float() for t in old_inputs))
    after = net(*new_inputs)
    deciding = torch.tensor([to_move(game) == seat for game, seat in pairs])
    mask = torch.from_numpy(record["action_mask"])[deciding]
    old_lp = masked_log_softmax(before.logits[deciding], mask)
    new_lp = masked_log_softmax(after.logits[deciding], mask)
    kl = (old_lp.exp() * (old_lp - new_lp)).masked_fill(~mask, 0.0).sum(-1)
    report["held_out_decisions"] = float(int(deciding.sum()))
    report["held_out_same_move"] = float((old_lp.argmax(-1) == new_lp.argmax(-1)).float().mean())
    report["held_out_mean_kl"] = float(kl.mean())
    win = (before.value[:, 0] - after.value[:, 0]).abs()
    report["held_out_mean_abs_win"] = float(win.mean())
    report["held_out_max_abs_win"] = float(win.max())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--entrant", default="random",
                        help="the bot all four seats play, as hexset.arena names it; random plays uniformly")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--stride", type=int, default=7, help="decisions between sampled positions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    runtime.add_argument(parser)
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    runtime.load(args.runtime)
    checkpoint, report = migrate_checkpoint(
        args.checkpoint, games=args.games, seed=args.seed, entrant=args.entrant, stride=args.stride
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
