# SPDX-License-Identifier: GPL-3.0-only
"""Export a trained checkpoint to ONNX, for hexset's onnxruntime backend.

`hexn.netbot` loads a checkpoint straight into PyTorch: fine for the training
box, but heavier than a browser game on borrowed hardware needs. This script
runs the exact reconstruction path `netbot.load()` already trusts (same
shapes, same `state_dict` keys), traces it once with `torch.onnx.export`, and
embeds the handful of facts hexset's `hexset.clients.onnxbot.load()` needs at
runtime as ONNX metadata props.

**Contract v2.** The serving side is an *interface*: it emits state, it
accepts an action, and nothing in that repo should know how a network reads a
position.
Contract 1 broke that — the serving side carried its own `encoding`, a
340-line reimplementation of `hexset.encoding`, and `onnxbot.py` carried the
masking, log-softmax,
give/want factorisation and seat un-rotation in numpy, both of which had to
stay bit-identical with this repo forever, in a language it does not run.
Contract 2 moves all of that into the graph: `record -> encoder -> HexNet
-> heads`, where `encoder` is this module's own `RecordEncoder` (the traced
mirror of `hexset.encoding`, see `onnx_record.py`'s own docstring) and `heads`
is masking, log-softmax, argmax and value un-rotation — lifted here out of
`onnxbot.OnnxPolicy`, the contract-1 path that ran them in numpy. That class
was deleted when contract 1 was dropped, so nothing on the serving side
mirrors this math any more; `hexn.policy.masked_log_softmax` is the same
formula, reused here rather than written a third time.

The line contract 2 draws is **the rules, not the network**: hexset still
computes the position and what a seat may legally know (encoded as the
*information-set record*, `hexset.onnx_record.RECORD_FIELDS`), still enumer-
ates legal moves and builds `action_mask`, and still runs
`mcts.py`/`search2.py` over the rules. Everything downstream of "what is true
and visible" — how to read it — is now the graph's job, so `NetworkBot` can
read `action_index` straight off one forward pass and a search can read
`prior`/`value` off the same one.

**Contract 6** carries two changes that happened to land together
(`hexset.onnx_record.CONTRACT_VERSION`'s own docstring is the record):
the trading redesign deletes the public layer outright, so the contract-5
record's `valuations` field is gone with nothing replacing it -- there is no
seat vector left to encode -- and the knight two-step fix shrinks the flat
action space (`PLAY_KNIGHT` only spends the card now; the robber move it
used to name directly runs through the same `MOVE_ROBBER` decision a rolled
seven already uses), 550 -> 456. `globals` is 67.

* **Inputs**, a dynamic leading batch axis, named exactly
  `hexset.onnx_record.RECORD_FIELDS` (24 names: board, position, information
  set, legality — see that module's docstring). All int64 except
  `action_mask`, which is the only bool one. `B` is
  1 for a single decision and a whole wave of leaves for the UI's MCTS
  `LeafEvaluator`, so the axis must really be dynamic.
* **Outputs**, in this order: `action_index` (`(B,)` int64, argmax over the
  masked distribution), `prior` `(B, space.size)` (float32, normalised over
  legal entries, zero elsewhere), and `value` `(B, players)` (float32,
  **board-seat order**, already un-rotated). Contract 6's value head is a win
  head (`hexn.model.Prediction`): `value` is each seat's own win
  probability, nonnegative and summing to one across the row, not a points
  margin. One forward serves both callers:
  `NetworkBot` reads the index, a search reads the two distributions.
* **Metadata props** (`hexset.clients.modelmeta` and `onnxbot._load_cached`):
  `contract` is `hexset.onnx_record.CONTRACT_VERSION` (`"6"`); `players` and the
  `num_hexes`/`num_vertices`/
  `num_edges` fingerprint are required; `max_trades` (`""` for none) and
  `iteration` are read with defaults; and only when asked for on the command
  line, `search=mcts` with `simulations`/`wave`, and `gate_plies` (the trade
  gate's own continuation budget, independent of `search`). Inference device
  is deliberately not metadata: the UI takes it from its own `--device`.
* **Provenance props**, which nothing reads at runtime and everything reads
  afterwards: `source_checkpoint` (repo-relative), `checkpoint_sha256`,
  `exported_at`, and `exporter_commit` when it can be determined. A deployed
  filename does not identify a file: it is chosen for the in-game picker, not
  for provenance, and `source_checkpoint` is normally `runs/<run>/latest.pt`,
  a moving pointer. The digest pins the weights and the commit pins the
  code, so an artifact can be reproduced from itself. See `_provenance`.
* **Scope cut, deliberate.** The stochastic/Gumbel sampling path is gone —
  every checkpoint served here plays argmax, so there is nothing to sample.
  A temperature input would be the way back in, if that is ever wanted.
* **Runtime**: onnxruntime >= 1.18 on the CPU provider, inside the UI's
  Python server — not onnxruntime-web; the browser only ever talks HTTP.
  Opset 18 is well inside that runtime's range.

A numerical parity check runs automatically after every export and raises
(non-zero exit) rather than warns: an export bug that silently degrades a
policy's quality is much more expensive to catch later, on a board, than
here, on real seeded positions.

Needs `torch` (not a declared dependency of this package -- see
`hexn.netbot`'s own docstring for why) plus `onnx`/`onnxruntime`
(`pip install onnx onnxruntime`; this package declares no extras)::

    python -m hexn.export_onnx --checkpoint runs/first/latest.pt --out model.onnx
    python -m hexn.export_onnx --checkpoint runs/first/latest.pt \\
        --out mcts256.onnx --search mcts --simulations 256

Then drop the file into hexset's `models/`; its stem is the name in the
in-game picker.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hexset.actions import ActionSpace, build_space
from hexset.board.board import pips, random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.board.topology import Topology
from hexset.encoding import (
    BANK_SCALE,
    HAND_SCALE,
    NUM_BUILDINGS,
    NUM_PHASES,
    NUM_TERRAIN,
    TURN_SCALE,
    StaticGraph,
    static_graph,
)
from hexset.game import Game, is_over, start, to_move
from .model import HexNet, config_from_args
from hexset.cards import DECK_SIZE
from hexset.onnx_record import RECORD_FIELDS, record_batch, record_shapes
from hexset.play import step_randomly
from hexset.state import NO_OWNER
from .policy import masked_log_softmax

_INPUT_NAMES = RECORD_FIELDS
_OUTPUT_NAMES = ("action_index", "prior", "value")
_BATCH = "batch"

# What the graph stamps as `contract`: the engine's own constant. The record
# layout and this export's tuple move together again now that
# `hexset.onnx_record.CONTRACT_VERSION` is "6" (the knight is one slot; the
# trading redesign shares the same contract), so importing it is what keeps a
# deployed file's number equal to the record it speaks.
from hexset.onnx_record import CONTRACT_VERSION as _CONTRACT_VERSION

# `action_mask` is the only bool input; every other input is int64 -- there
# is no float input left now that contract 6 deleted the public valuation
# vector (`hexset.onnx_record.RECORD_FIELDS` has no field for it). Every
# output is float32 except `action_index`.
_BOOL_INPUTS = frozenset({"action_mask"})
_FLOAT_INPUTS: frozenset[str] = frozenset()
_INT_OUTPUTS = frozenset({"action_index"})

# The only value `hexset.clients.modelmeta.search_config` acts on; anything else in
# the `search` key is read as "no search", so writing anything else would be
# a silent no-op rather than a setting.
_SEARCHES = ("none", "mcts")


def _base_topology() -> Topology:
    """The served table only ever plays the base map, and its topology is
    seed-invariant — only terrain, tokens and ports are randomised, never the
    hex/vertex/edge graph (`random_base_board` always calls
    `build_topology(BASE_LAYOUT)`) — so any seed gives the same topology a
    checkpoint will actually see."""
    return random_base_board(random.Random(0)).topology


def _shapes(graph: StaticGraph, players: int, space: ActionSpace) -> dict[str, tuple]:
    """Every tensor's per-row shape, i.e. the contract minus the batch axis:
    `hexset.onnx_record.record_shapes`'s input half plus this export's own
    four outputs. One table for the sample inputs, the read-back check and
    the tests."""
    return {
        **record_shapes(graph, players, space),
        "action_index": (),
        "prior": (space.size,),
        "value": (players,),
    }


def _sample_games(players: int, count: int, seed: int = 0) -> list[tuple[Game, int]]:
    """`count` real `(game, perspective)` pairs from independent seeded
    playouts on the base topology.

    Not random gaussians, and not even random integers: the record is
    integers *with meaning* — `vertex_owner` in `[-1, players)`, a mask with
    at least one legal action per row — so anything else would trace and run
    the graph without ever exercising a mask that looks like one, which is
    exactly the case `action_index`'s exact-match gate exists to
    stress. `perspective` is always `to_move(game)`, same as
    the seat `hexset.clients.onnxbot.V2Policy` passes to `record_from_game`
    (contract 1 carried it as `Request.seat`, removed with that path).
    """
    rng = random.Random(seed)
    pairs: list[tuple[Game, int]] = []
    while len(pairs) < count:
        game = start(random_base_board(rng), players, rng)
        for _ in range(rng.randrange(0, 150)):
            if is_over(game):
                break
            step_randomly(game, rng)
        if is_over(game):
            continue
        pairs.append((game, to_move(game)))
    return pairs


def _sample_inputs(space: ActionSpace, players: int, batch: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Real information-set records, for tracing and for the parity check
    alike — `_sample_games` plus `hexset.onnx_record.record_batch`."""
    return record_batch(_sample_games(players, batch, seed), space)


def _onehot(index: Tensor, num_classes: int) -> Tensor:
    """`(..., num_classes)` one-hot from an equality broadcast rather than
    `F.one_hot`, so it traces to a plain `Equal` the exporter has never
    wobbled on, whatever the batch shape."""
    classes = torch.arange(num_classes, device=index.device, dtype=index.dtype)
    return (index.unsqueeze(-1) == classes).to(torch.float32)


def _rotate_slot(owner: Tensor, perspective: Tensor, players: int) -> Tensor:
    """Owner-as-stored -> seat-relative column, `players` for `NO_OWNER`.

    Mirrors `hexset.encoding._seat` (`(seat - perspective) % players`) plus the
    "`NO_OWNER` sits last" convention `_slots` encodes as a lookup table --
    written here as `where`/`remainder` because a `Tensor` cannot walk a
    Python list.
    """
    if perspective.dim() < owner.dim():
        perspective = perspective.unsqueeze(-1)
    slot = torch.remainder(owner - perspective, players)
    return torch.where(owner == NO_OWNER, torch.full_like(slot, players), slot)


# Ways to roll each token with two dice, scaled and cast to float32 exactly
# once -- as a plain Python computation, not a tensor op -- so this table is
# bit-identical to `hexset.encoding._template_by_value`'s
# `pips(token) / MAX_TOKEN_PIPS` for every entry. Reusing `hexset.board.board.pips`
# rather than re-deriving `6 - abs(7 - token)` here keeps the formula in one
# place, per the standing rule against a second definition of a rules number.
_MAX_TOKEN = 12
MAX_TOKEN_PIPS = 5


def _pips_table() -> list[float]:
    return [pips(token) / MAX_TOKEN_PIPS for token in range(_MAX_TOKEN + 1)]


class RecordEncoder(nn.Module):
    """`record -> (hexes, vertices, edges, globals)`: the traceable half of
    `hexset.encoding.encode`/`encode_batch`, reading `hexset.onnx_record`'s
    torch-free record.

    No data-dependent control flow -- every branch below is a `where` or a
    comparison evaluated on every row, never a Python `if` on a tensor value
    -- which is exactly what tracing requires and what the original numpy
    encoder already had none of.

    Board adjacency (`StaticGraph`) plays no part here: that is baked into
    `HexNet`'s own buffers, unchanged by this work, and only `num_hexes` is
    needed, to size the robber one-hot.
    """

    def __init__(self, graph: StaticGraph, players: int) -> None:
        super().__init__()
        self.players = players
        self.num_hexes = graph.num_hexes
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

        globals_ = torch.cat(parts, dim=-1)

        return hexes, vertices, edges, globals_


class _ExportWrapper(nn.Module):
    """`record -> encoder -> HexNet -> heads`.

    `torch.onnx.export` wants a tuple of tensors back, not `HexNet`'s own
    `Prediction` dataclass or a dict, so this both assembles the pipeline and
    flattens its output.
    """

    def __init__(self, encoder: RecordEncoder, net: HexNet, players: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.net = net
        self.players = players

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
        award_points: Tensor,
        longest_road_holder: Tensor,
        largest_army_holder: Tensor,
        phase: Tensor,
        free_roads: Tensor,
        deck_size: Tensor,
        turns: Tensor,
        perspective: Tensor,
        own_hand: Tensor,
        hand_totals: Tensor,
        own_dev: Tensor,
        dev_totals: Tensor,
        ledger_known: Tensor,
        ledger_unknown: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hexes, vertices, edges, globals_ = self.encoder(
            terrain,
            token,
            port_code,
            robber,
            vertex_owner,
            vertex_building,
            edge_owner,
            bank,
            knights_played,
            award_points,
            longest_road_holder,
            largest_army_holder,
            phase,
            free_roads,
            deck_size,
            turns,
            perspective,
            own_hand,
            hand_totals,
            own_dev,
            dev_totals,
            ledger_known,
            ledger_unknown,
        )
        pred = self.net(hexes, vertices, edges, globals_)

        # `masked_log_softmax` is written once in `hexn.policy` for
        # training-time sampling and reused here. Contract 1 ran the same
        # formula in numpy on the serving side; that helper went with
        # `OnnxPolicy` when contract 1 was dropped, so the graph is now the
        # only place it lives.
        slot_log_probs = masked_log_softmax(pred.logits, action_mask)
        action_index = slot_log_probs.argmax(dim=-1)
        # Illegal entries sit at `NEG` before the softmax, so `exp` of the
        # shifted, normalised log-probs is already exactly zero there and
        # already sums to one over the legal entries -- nothing left to mask
        # or renormalise, unlike `LeafEvaluator._prior`'s option-subset gather.
        prior = slot_log_probs.exp()

        # The un-rotation contract 1 did in numpy (`onnxbot._board_order`,
        # removed with it), as a gather: board seat `j`'s value is the
        # seat-relative value at `(j - perspective) % players`. Contract 2
        # exports `value` already in board-seat order, so the server does not
        # repeat this.
        board_seat = torch.arange(self.players, device=perspective.device)
        rotate = torch.remainder(
            board_seat.unsqueeze(0) - perspective.unsqueeze(1), self.players
        )
        value = pred.value.gather(1, rotate)

        return action_index, prior, value


def _load_checkpoint(
    checkpoint: str, topology: Topology
) -> tuple[HexNet, ActionSpace, dict]:
    """The same reconstruction path `netbot.load` already trusts — so export
    uses the exact model-building logic the torch bot relies on, not a second
    implementation that could drift from it."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
    config = config_from_args(args)
    graph = static_graph(topology)
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    net = HexNet(space, graph, players, config)
    net.load_state_dict(state["net"])
    net.eval()
    # The default (non-fused) forward path aggregates neighbours with
    # `index_add_`, whose index tensor has duplicates by construction (many
    # vertices share a hex, many edges share a vertex) — the legacy
    # TorchScript ONNX exporter mishandles that ("does not support duplicated
    # values in 'index' field... will cause the ONNX model to be incorrect").
    # `fused=True` swaps in the dense adjacency-matmul path instead: "the
    # same numbers up to float reassociation" per its own docstring, with a
    # dedicated equivalence test already pinning that in torch — just
    # architecturally the wrong default to optimise for a training GPU
    # (slower in CPU eager mode), which is irrelevant for exporting a graph
    # that runs once per move.
    net.fused = True
    return net, space, {
        "players": players,
        "max_trades": args.get("max_trades"),
        "iteration": int(state.get("iteration", 0)),
    }


def _repo_root() -> Path:
    """The checkout this package is being run from — `src/hexset/x.py` upward."""
    return Path(__file__).resolve().parents[2]


def _exporter_commit() -> str | None:
    """The commit that produced a file, or None rather than a guess.

    `git` is asked first but usually fails here: an export runs in a container
    that bind-mounts the worktree, and a worktree's `.git` is a file holding an
    absolute gitdir that does not exist inside it (the same trap the dev
    status journal records). `HEXSET_EXPORT_COMMIT` is the way in, set by
    whoever launches the container from a shell that *can* read the repo.

    Returns None when neither works. A missing key is honest; `"unknown"`
    stamped into an artifact is a lie that survives longer than the session.
    """
    env = os.environ.get("HEXSET_EXPORT_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _provenance(checkpoint: str) -> dict[str, str]:
    """What the artifact needs to describe itself once it has left this box.

    A filename cannot be trusted to do it — it is chosen for the in-game
    picker, not for provenance — so everything that identifies a file has to
    be inside it.

    `source_checkpoint` alone is not enough either: it is usually
    `runs/<run>/latest.pt`, a *moving* pointer whose bytes change the next time
    that run is resumed. `checkpoint_sha256` is what actually pins the weights,
    and `exporter_commit` pins the code that read them, so an artifact can be
    reproduced from itself without consulting anything external.
    """
    source = Path(checkpoint)
    try:
        relative = source.resolve().relative_to(_repo_root())
    except ValueError:
        # Outside the checkout entirely; the absolute path is all there is.
        relative = source
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    props = {
        "source_checkpoint": str(relative),
        "checkpoint_sha256": digest,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    commit = _exporter_commit()
    if commit:
        props["exporter_commit"] = commit
    return props


def _embed_metadata(
    path: Path,
    *,
    topology: Topology,
    players: int,
    max_trades: int | None,
    iteration: int,
    source_checkpoint: str,
    search: str,
    simulations: int | None,
    wave: int | None,
    gate_plies: int | None,
) -> None:
    """The facts `hexset.clients.onnxbot.load()` needs back at runtime: which
    contract the graph speaks, enough to build an `ActionSpace`, enough to
    refuse a topology mismatch loudly (the way `net.load_state_dict` already
    fails loudly on a shape mismatch today), and how the file wants to be
    played. Width/rounds/head-shape aren't included — those only matter for
    reconstructing the torch module, and tracing already baked them in.

    `search`/`simulations`/`wave` are written only when a search was asked
    for. `hexset.clients.modelmeta` would ignore a stray budget anyway ("a stale
    `simulations` left behind by an export cannot quietly turn a policy
    checkpoint into a search"), so leaving the keys out keeps the file's
    metadata saying exactly what it does. `gate_plies` -- the trade gate's
    own continuation budget, unrelated to `search` -- follows the same rule
    for the same reason: written only when this export was asked for one,
    so a file that never asked reads at `hexset.clients.modelmeta`'s unmeasured
    default (no rollout) rather than whatever the exporter's own default
    happens to be."""
    import onnx

    props = {
        "contract": _CONTRACT_VERSION,
        "players": str(players),
        "num_hexes": str(topology.num_hexes),
        "num_vertices": str(topology.num_vertices),
        "num_edges": str(topology.num_edges),
        "max_trades": "" if max_trades is None else str(max_trades),
        "iteration": str(iteration),
        **_provenance(source_checkpoint),
    }
    if search == "mcts":
        props["search"] = "mcts"
        if simulations is not None:
            props["simulations"] = str(simulations)
        if wave is not None:
            props["wave"] = str(wave)
    if gate_plies is not None:
        props["gate_plies"] = str(gate_plies)

    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(path))


def _check_tensor(tensor, shapes: dict[str, tuple], expected_dtype) -> None:
    import onnx

    kind = tensor.type.tensor_type
    if kind.elem_type != expected_dtype:
        name = onnx.TensorProto.DataType.Name(expected_dtype)
        raise ValueError(f"{tensor.name} is not {name}")
    dims = [d.dim_param or d.dim_value for d in kind.shape.dim]
    expected = [_BATCH, *shapes[tensor.name]]
    if dims != expected:
        raise ValueError(f"{tensor.name} is shaped {dims}, not {expected}")


def _verify_contract(onnx_path: Path, shapes: dict[str, tuple]) -> None:
    """Read the graph back and hold it to the names, dtypes and shapes
    `hexset.clients.onnxbot`'s v2 path hard-codes — the parity check below would
    pass a graph whose outputs were merely *renamed*, and the UI would then
    fail at the first move with an onnxruntime error rather than here."""
    import onnx

    model = onnx.load(str(onnx_path))
    found = {
        "input": [t.name for t in model.graph.input],
        "output": [t.name for t in model.graph.output],
    }
    expected = {"input": list(_INPUT_NAMES), "output": list(_OUTPUT_NAMES)}
    if found != expected:
        raise ValueError(f"graph signature {found} is not hexset's {expected}")
    for tensor in model.graph.input:
        if tensor.name in _BOOL_INPUTS:
            dtype = onnx.TensorProto.BOOL
        elif tensor.name in _FLOAT_INPUTS:
            dtype = onnx.TensorProto.FLOAT
        else:
            dtype = onnx.TensorProto.INT64
        _check_tensor(tensor, shapes, dtype)
    for tensor in model.graph.output:
        dtype = onnx.TensorProto.INT64 if tensor.name in _INT_OUTPUTS else onnx.TensorProto.FLOAT
        _check_tensor(tensor, shapes, dtype)


def _verify_parity(
    net: HexNet,
    encoder: RecordEncoder,
    space: ActionSpace,
    players: int,
    onnx_path: Path,
    samples: int = 8,
    seed: int = 1,
) -> None:
    """Fails loudly (raises) rather than warns — there's no other test
    coverage of the ONNX path, so this is the one thing standing between a
    silent export bug and a bot that just plays worse for no logged reason.

    The reference path is `hexset.encoding.encode` (numpy) feeding the same
    `net` and `masked_log_softmax` the wrapper's `forward`
    uses — the exact math `hexset.clients.onnxbot`'s numpy port mirrors, computed
    here in eager torch rather than a fourth reimplementation. What this
    check actually exercises is tracing and onnxruntime execution: does the
    exported graph, run through the runtime hexset actually uses, agree
    with the eager computation on real positions.

    `action_index` must match exactly — argmax on a near-tie is
    the one failure this whole check exists to catch, and a tolerance would
    hide exactly that. `prior`/`value` keep the wider tolerance
    calibrated against a real checkpoint for the v1 contract (a
    width=64/rounds=2 net tripped a 1e-4 bound on ordinary fp32 accumulation
    drift, not a bug): `rtol=1e-3, atol=1e-4`.
    """
    import onnxruntime as ort

    pairs = _sample_games(players, samples, seed)
    record = record_batch(pairs, space)

    from hexset.encoding import encode

    observations = [encode(game, seat) for game, seat in pairs]
    hexes = torch.from_numpy(np.stack([o.hexes for o in observations]))
    vertices = torch.from_numpy(np.stack([o.vertices for o in observations]))
    edges = torch.from_numpy(np.stack([o.edges for o in observations]))
    globals_ = torch.from_numpy(np.stack([o.globals for o in observations]))
    mask = torch.from_numpy(record["action_mask"])
    perspective = torch.from_numpy(record["perspective"])

    with torch.no_grad():
        pred = net(hexes, vertices, edges, globals_)
        slots = masked_log_softmax(pred.logits, mask)
        ref_action_index = slots.argmax(dim=-1).numpy()
        ref_prior = slots.exp().numpy()
        board_seat = torch.arange(players)
        rotate = torch.remainder(board_seat.unsqueeze(0) - perspective.unsqueeze(1), players)
        ref_value = pred.value.gather(1, rotate).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = dict(zip(_OUTPUT_NAMES, session.run(list(_OUTPUT_NAMES), record)))

    np.testing.assert_array_equal(onnx_out["action_index"], ref_action_index, err_msg="action_index")
    for name, ref in (("prior", ref_prior), ("value", ref_value)):
        np.testing.assert_allclose(onnx_out[name], ref, rtol=1e-3, atol=1e-4, err_msg=name)


def export(
    checkpoint: str,
    out: Path,
    *,
    topology: Topology | None = None,
    opset: int = 18,
    search: str = "none",
    simulations: int | None = None,
    wave: int | None = None,
    gate_plies: int | None = None,
) -> Path:
    """Load `checkpoint`, trace it to `out`, verify contract and parity, return `out`.

    `search="mcts"` (with an optional `simulations`/`wave` budget) makes the
    file ask hexset to search over its own priors; the default plays one
    forward pass per move. A budget without a search is refused rather than
    written, since the UI would read the file as a plain policy regardless.

    `gate_plies` asks the file's trade gate to roll its own continuation
    forward that many plies before valuing an exchange -- conceptually a
    search on the trade decision, same footing as `search`/`simulations`.
    Independent of `search` -- a plain policy file
    can still ask its gate to roll forward. `None`, the default, writes
    nothing, so training collection's own exports (which never pass this
    flag) leave the served file to `hexset.clients.modelmeta`'s unmeasured
    default of no rollout.
    """
    if search not in _SEARCHES:
        raise ValueError(f"search must be one of {_SEARCHES}, not {search!r}")
    if search == "none" and (simulations is not None or wave is not None):
        raise ValueError("simulations/wave only mean something with search='mcts'")

    topology = topology or _base_topology()
    net, space, meta = _load_checkpoint(checkpoint, topology)
    players = meta["players"]
    graph = static_graph(topology)
    shapes = _shapes(graph, players, space)

    encoder = RecordEncoder(graph, players)
    encoder.eval()
    wrapper = _ExportWrapper(encoder, net, players)
    wrapper.eval()

    dummy = _sample_inputs(space, players, batch=1, seed=0)
    dummy_tensors = tuple(torch.from_numpy(dummy[name]) for name in _INPUT_NAMES)

    torch.onnx.export(
        wrapper,
        dummy_tensors,
        str(out),
        # The classic TorchScript-based exporter, not torch's newer
        # dynamo-based default (`dynamo=True` since ~2.7, needing the extra
        # `onnxscript` dependency) — nothing in the encoder or `HexNet`'s
        # forward has data-dependent control flow, so tracing is safe and
        # there's nothing the dynamo path would buy here.
        dynamo=False,
        input_names=list(_INPUT_NAMES),
        output_names=list(_OUTPUT_NAMES),
        dynamic_axes={n: {0: _BATCH} for n in (*_INPUT_NAMES, *_OUTPUT_NAMES)},
        opset_version=opset,
    )
    _embed_metadata(
        out,
        topology=topology,
        source_checkpoint=checkpoint,
        search=search,
        simulations=simulations,
        wave=wave,
        gate_plies=gate_plies,
        **meta,
    )
    _verify_contract(out, shapes)
    _verify_parity(net, encoder, space, players, out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt checkpoint (as hexn.ppo writes)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: the checkpoint's own path with a .onnx suffix).",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--search",
        choices=_SEARCHES,
        default="none",
        help="How hexset should play the file: one forward pass (none) or a "
        "PUCT search over its own priors (mcts). Written as metadata.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="MCTS descents per decision (hexset defaults to 128, caps at 4096).",
    )
    parser.add_argument(
        "--wave",
        type=int,
        default=None,
        help="MCTS leaves batched per expansion (hexset defaults to 16, caps at 256).",
    )
    parser.add_argument(
        "--gate-plies",
        type=int,
        default=None,
        help="The served trade gate's own continuation budget: how many plies the "
        "mover's greedy policy rolls forward before an exchange is valued "
        "(hexset defaults to 0, no rollout, caps at 16). Independent of --search.",
    )
    args = parser.parse_args(argv)
    if args.search == "none" and (args.simulations is not None or args.wave is not None):
        parser.error("--simulations/--wave need --search mcts")

    checkpoint = Path(args.checkpoint)
    out = args.out or checkpoint.with_suffix(".onnx")
    path = export(
        str(checkpoint),
        out,
        opset=args.opset,
        search=args.search,
        simulations=args.simulations,
        wave=args.wave,
        gate_plies=args.gate_plies,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
