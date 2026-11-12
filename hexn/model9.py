# SPDX-License-Identifier: GPL-3.0-only
"""`HexNet`: the contract-9 network, every seat read by the same weights.

It reads `hexset.contract9`'s observation: four seat tokens (the perspective
first), a table token, 19 hex tokens and 54 vertex tokens, with a building's
and a road's owner as a pointer to a seat token and each seat's reach as a
weighted seat-vertex link. Roads are vertex-vertex links: there are no edge
tokens. Each round passes, with one set of weights for every seat:

- hex <-> vertex (the mean over each one's neighbours);
- vertex <- its neighbour vertices, and the roads it touches, each read as
  its owner's seat token (plus a learned road vector) where one stands;
- vertex <- its building's owner's seat token, and every seat token weighted
  by that seat's reach to it; seat <- the vertices it builds on, the roads it
  owns and the vertices it reaches, the same links the other way;
- seat <-> seat by one attention over the four, or by each seat reading the
  others' mean token (`ModelConfig.others`: at two seats, the opponent's);
- the table token to every token, and back from the means of each kind.

Under contract 11d my token (seat row 0) and the opponent's are embedded by
layers of their own, since they hold different columns, and the table token
is a board token, embedded from the board block: each round it is broadcast
as the table is, and its update reads the hexes' and vertices' means, my
seat token and one attention read over every hex and vertex
(`_board_step`). Its heads are query heads (`_emit_queries`): from my
token, the opponent's and the board token, a score per
move type and a query per targeted role; a board move scored as its type's
score plus its roles' queries against the keys of what it names (vertices,
edges, a robber's hex with its victim), a resource move as its type's score
plus the value of the two hands it leaves. A move of a type in `UNTYPED`
(setup, robber placement, discard) has no type score: its position offers
moves of that type alone, so the type would be one number shared by every
option, which no loss trains and no choice reads. Each seat's win logit
reads its own token with mine. The decisions are asked as for every other
contract-9 net (`flat_scores`, `setup_scores`, `road_building_scores`,
`discard_scores`, `robber_logits`).

Heads, all shared across seats: the win head reads one logit off each seat's
token (with the table), softmaxed over the four, the perspective first; the
policy acts for the perspective, whose view every token the trunk built is
already in (seat row 0, owners and reach relative to it), so the board heads
read the trunk's vertex and hex tokens as they are and the off-board actions
the table token, as `hexn.model.HexNet`'s heads do. A road is scored from its two vertices by one function, symmetric in
the ends; a robber move from the hex and the victim's seat token. The
compound decisions (`hexset.decisions`) are scored as wholes: a setup turn
from its settlement and road, the snake seat's double turn as a first turn,
a second (paying) turn and a term for the two settlements together, Road
Building as its roads' scores, a discard from the hand it keeps and the cards
it gives. `decision_log_prob` is the softmax over every legal option of a
decision, compound and ordinary together.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
import torch
from torch import Tensor, nn

from hexset.actions import YEAR_OF_PLENTY_PAIRS, Action, ActionSpace, ActionType
from hexset.board.terrain import NUM_RESOURCES
from hexset.contract9 import (
    CARD_SCALE, CONTRACTS, HEX_FEATURES, MY_COLUMNS_11D, MY_FEATURES_11D, OTHER_COLUMNS_11D, OTHER_FEATURES_11D,
    PIECE_SCALE, SEAT_FEATURES, SEAT_FEATURES_9D, TABLE_FEATURES, VERTEX_FEATURES, Layout9, features, hex_features,
    incidences,
)
from hexset.decisions import DecisionKind, double_pairs
from hexset.board.ports import BASE_TRADE_RATIO
from hexset.state import MAX_SETTLEMENTS

__all__ = ["DOUBLE_TURNS", "ModelConfig", "config_of", "HexNet", "Prediction", "Scores", "unpack9", "collate", "decision_scores",
           "chosen_scores", "decision_entropy", "decision_log_prob", "first_move_prior", "option_probs"]

# The move types an 11d net scores with no type score: a position that offers
# one of them offers moves of that type alone (setup turns, the robber's
# placement after a seven or a knight, a discard), so the type would be one
# number added to every option, cancelled by the softmax, trained by no loss
# and free to drift (a drifted discard type sent a decision's log-sum-exp below
# float32's exp range). `test_model9.py` replays random games against this set.
UNTYPED = frozenset({ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD, ActionType.MOVE_ROBBER,
                     ActionType.DISCARD})


# `ModelConfig.double_turn`'s choices.
DOUBLE_TURNS = ("joint", "parts")

# The contract-11d heads: a score per move type, one per `ActionType` in its
# order; a query per targeted role, each `KEY_WIDTH` wide, read against keys
# of vertices, edges and robber targets; and a hand value, the worth of the
# two hands a resource move leaves (`HexNet._emit_queries`).
MOVE_TYPES = tuple(ActionType)
# A setup settlement reads `setup_settlement` for a seat's first and
# `setup_settlement_paying` for its second, the one that pays the starting
# hand (`HexNet._emit_queries`).
QUERY_ROLES = ("settlement", "city", "setup_settlement", "setup_settlement_paying", "road", "setup_road",
               "road_building", "robber")
KEY_WIDTH = 16
# Contract 11d's board token reads the hexes and vertices through one
# attention, this wide (`HexNet._board_step`).
BOARD_READ = 16
# Every query and key is layer-normalised (with a gain) and their product
# divided by sqrt(KEY_WIDTH), so a score's size is the role's learned scale,
# exp(t), times a bounded product; the scale starts at QUERY_SCALE, about
# the spread an unnormalised query . key score starts at, and is soft-capped
# at ROLE_SCALE_MAX (`HexNet.role_scales`): with the gains at one a score is
# then within 4 ROLE_SCALE_MAX of its type's, which still separates the
# clearest choice from the rest, and the scale cannot run away as training
# sharpens.
QUERY_SCALE = 0.04
ROLE_SCALE_MAX = 4.0
# The keys' gains: a vertex's, an edge's, a robber target's.
KEY_KINDS = ("vertex", "edge", "robber")
# The hand value reads, beside the context, the two hands it values.
HAND_READS = NUM_RESOURCES
_ROLE = {name: i for i, name in enumerate(QUERY_ROLES)}
# The roles read against the vertex keys and against the edge keys, each a
# contiguous run of `QUERY_ROLES`, so each kind is read in one product.
VERTEX_ROLES = ("settlement", "city", "setup_settlement", "setup_settlement_paying")
EDGE_ROLES = ("road", "setup_road", "road_building")
assert QUERY_ROLES[:len(VERTEX_ROLES) + len(EDGE_ROLES)] == VERTEX_ROLES + EDGE_ROLES


def _as(x: Tensor, dtype: torch.dtype) -> Tensor:
    """`x` in `dtype`, without a cast where it already is (a cast is a call
    each, and a forward at batch 1 is mostly calls)."""
    return x if x.dtype == dtype else x.to(dtype)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    rounds: int = 3
    # "linear" (c8's): every head one linear layer. "mlp": each head with a
    # hidden layer of `width`, as c9-bc's first run trained (its model config
    # predates this field and loads as "mlp", `config_of`).
    heads: str = "linear"
    # The discard head's shape apart from the others: "mlp" scores a kept hand
    # with a hidden layer, so cards can count together (brick with wood), which
    # a linear score of the counts cannot.
    discard: str = "mlp"
    # LayerNorm on every token before each round's update (pre-norm) and once
    # more before the heads. Checkpoints written before the field had none.
    norm: bool = True
    # "separate": each message a token reads is an input of its own, with its
    # own first-layer weights (a vertex reads 7, a seat 6). "pooled": the
    # messages of a kind are summed into one input (a vertex reads itself, its
    # hexes, its edges -- neighbour plus road -- and its seats -- building owner
    # plus reach; a seat itself, the other seats and its board), and the table
    # is added once per position.
    messages: str = "separate"
    # One attention over the four seat tokens each round. Without it seats meet
    # only through the table and the board, and the seat update loses that input.
    attention: bool = True
    # The other seats' mean token as an input of each seat's update: what
    # attention tends to at two seats, where the mean is the opponent's token,
    # without query, key or value. Checkpoints written before the field had none.
    others: bool = False
    # The global head (bank trades, buying a card, the development plays,
    # ending the turn) reads the perspective's own seat token beside the
    # table's, where the table alone carries the seats only as their mean.
    # Checkpoints written before the field had the table alone: loading one
    # pads the head's weights for the seat token with zeros. Contract 11d has
    # no table, and its global head always reads both seat tokens.
    global_seat: bool = False
    # A points head beside the win head: one number per seat token (with the
    # table), its near-term points against the others', trained on a cache's
    # zero-sum targets. Checkpoints written before the field had none.
    points: bool = False
    # How the snake seat's double setup is chosen, not a shape: "joint", one
    # decision over every ordered pair of setup turns (the first, second and
    # pair heads, as contract-9 BC trains it); "parts", the setup head twice,
    # the second forward on the board after the first placement (as PPO
    # trains it, `hexn.policy9`). Only a "joint" net has the first, second
    # and pair heads: a "parts" net carries no parameter it never reads, and a
    # "joint" checkpoint initialises one through `without_joint_head`.
    double_turn: str = "joint"
    contract: str = "9"

    def __post_init__(self) -> None:
        if self.contract not in CONTRACTS:
            raise ValueError(f"unknown decision contract {self.contract!r}")
        for shape in (self.heads, self.discard):
            if shape not in ("linear", "mlp"):
                raise ValueError(f"unknown head shape {shape!r}")
        if self.messages not in ("separate", "pooled"):
            raise ValueError(f"unknown message layout {self.messages!r}")
        if self.double_turn not in DOUBLE_TURNS:
            raise ValueError(f"unknown double turn {self.double_turn!r}: {DOUBLE_TURNS}")
        if self.contract == "11d":
            # The contract-11d heads (`HexNet._query_heads`): no global head
            # and no points head.
            if self.global_seat:
                raise ValueError("contract 11d has no table and no global head: global_seat does not apply")
            if self.points:
                raise ValueError("a contract-11d net has no points head")


def config_of(model_config: dict) -> ModelConfig:
    """The `ModelConfig` a checkpoint's `model_config` names; one written
    before `heads` existed trained MLP heads."""
    return ModelConfig(contract=str(model_config.get("contract", "9")),
                         width=model_config["width"], rounds=model_config["rounds"],
                         heads=model_config.get("heads", "mlp"), discard=model_config.get("discard", "mlp"),
                         norm=model_config.get("norm", False), messages=model_config.get("messages", "separate"),
                         attention=model_config.get("attention", True), others=model_config.get("others", False),
                         global_seat=model_config.get("global_seat", False), points=model_config.get("points", False),
                         double_turn=model_config.get("double_turn", "joint"))


JOINT_HEADS = ("first_head.", "second_head.", "pair_head.", "pair_matrix")


def without_joint_head(state: dict) -> tuple[dict, int]:
    """`state` (a net's state dict) without the joint double-turn heads, and
    the number of parameters dropped: how a "joint" checkpoint (contract-9
    BC) initialises a "parts" net (PPO), which has no such heads."""
    kept = {k: v for k, v in state.items() if not k.startswith(JOINT_HEADS)}
    return kept, sum(int(v.numel()) for k, v in state.items() if k not in kept)


@dataclass
class Prediction:
    """`value` the win head, softmaxed over seats, the perspective first.

    The policy's outputs, `None` from `forward(..., policy=False)`: `logits`
    over the flat `ActionSpace` but its robber block, which `robber_logits`
    scores for the rows asked (robber victims seat-relative); `road` each
    edge's road score (Road Building reads it); `me`, `table` the features
    the candidate heads read; `vertices`, `hexes`, `seats` the trunk's tokens
    the heads scored on demand read (`robber_logits`, `setup_heads`). Heads
    only some decisions need run only on the rows that need them."""

    value: Tensor
    value_logits: Tensor
    points: Tensor | None = None                # the points head, per seat, the perspective first
    logits: Tensor | None = None
    road: Tensor | None = None
    me: Tensor | None = None
    table: Tensor | None = None
    vertices: Tensor | None = None
    hexes: Tensor | None = None
    seats: Tensor | None = None
    # The contract-11d heads (`HexNet._emit_queries`): each move type's score, the
    # robber's query, each edge's Road Building read, the hand value's context
    # term `(B, hidden)` and the two hands `(B, 2, resources)`, mine first.
    types: Tensor | None = None
    robber_query: Tensor | None = None
    rb_road: Tensor | None = None
    hand_context: Tensor | None = None
    hands: Tensor | None = None
    # ... and for the double turn asked whole: each vertex's key, each
    # vertex's first and paying setup-settlement reads `(B, 2, V)`, and each
    # edge's setup-road read.
    vertex_keys: Tensor | None = None
    edge_keys: Tensor | None = None
    setup_reads: Tensor | None = None
    setup_road_read: Tensor | None = None


def _mlp(in_dim: int, width: int, out_dim: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(), nn.Linear(width, out_dim or width))


def _mean_adjacency(dest: np.ndarray, src: np.ndarray, rows: int, cols: int) -> Tensor:
    out = torch.zeros(rows, cols)
    out[torch.as_tensor(dest), torch.as_tensor(src)] = 1.0
    return out / out.sum(-1, keepdim=True).clamp(min=1.0)


class _LayerNorm(nn.LayerNorm):
    """`nn.LayerNorm` in its input's dtype. Autocast runs layer norm in fp32,
    which casts every token set up and back down around each norm; in bf16
    the norm is one kernel."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype == self.weight.dtype:
            return super().forward(x)
        with torch.autocast(x.device.type, enabled=False):
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype),
                                            self.bias.to(x.dtype), self.eps)


def _norm(width: int, on: bool) -> nn.Module:
    return _LayerNorm(width) if on else nn.Identity()


class _Round(nn.Module):
    def __init__(self, width: int, norm: bool = True, pooled: bool = False, attention: bool = True,
                 others: bool = False, board: bool = False) -> None:
        super().__init__()
        # Pre-norm: each kind of token is normalised before its messages are
        # built and its update read, and the update is added to the token as
        # it stood. Identity where the net is built without (`ModelConfig.norm`).
        # The table token is contract 11d's board token (`board`): its update
        # also reads my seat token and an attention read over the hexes and
        # vertices (`HexNet._board_step`).
        self.norm_hex = _norm(width, norm)
        self.norm_vertex = _norm(width, norm)
        self.norm_seat = _norm(width, norm)
        self.norm_table = _norm(width, norm)
        self.pooled = pooled
        # The seat update's inputs beside the seat itself and its board: the
        # attention read and the others' mean, each where asked for.
        reads = int(attention) + int(others)
        if pooled:
            # `ModelConfig.messages == "pooled"`: each kind of message pooled
            # into one input (below), and the table projected once per position
            # and added to every token's first layer rather than copied into
            # each token's input.
            self.hex = _mlp(2 * width, width)
            self.vertex = _mlp(4 * width, width)
            self.seat = _mlp((2 + reads) * width, width)
            self.hex_table = nn.Linear(width, width, bias=False)
            self.vertex_table = nn.Linear(width, width, bias=False)
            self.seat_table = nn.Linear(width, width, bias=False)
            # What a building's owner's token carries beside a reach-weighted one.
            self.owned = nn.Parameter(torch.zeros(width))
        else:
            self.hex = _mlp(3 * width, width)
            self.vertex = _mlp(7 * width, width)
            self.seat = _mlp((5 + reads) * width, width)
        self.board = board
        if board:
            self.table = _mlp(4 * width + BOARD_READ, width)
            # The board token's read of the hexes and vertices: one query, and
            # each kind of node its own key and value, one layer emitting both
            # ([key, value], `BOARD_READ` each).
            self.board_query = nn.Linear(width, BOARD_READ)
            self.board_hex = nn.Linear(width, 2 * BOARD_READ)
            self.board_vertex = nn.Linear(width, 2 * BOARD_READ)
        else:
            self.table = _mlp(4 * width, width)
        self.attention = attention
        self.others = others
        if attention:
            self.query = nn.Linear(width, width)
            self.key = nn.Linear(width, width)
            self.value = nn.Linear(width, width)
        self.road = nn.Parameter(torch.zeros(width))

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # A board token written with a key layer and a value layer per kind of
        # node loads into the one layer emitting both: the same numbers.
        for kind in ("hex", "vertex"):
            for part in ("weight", "bias"):
                key, value = (f"{prefix}board_{kind}_{x}.{part}" for x in ("key", "value"))
                if key in state_dict and value in state_dict:
                    state_dict[f"{prefix}board_{kind}.{part}"] = torch.cat([state_dict.pop(key),
                                                                            state_dict.pop(value)])
        super()._load_from_state_dict(state_dict, prefix, *args)


def _others(sn: Tensor, dim: int, players: int) -> Tensor:
    """Each seat's mean of the other seats' tokens, the `players` seats along
    `dim`: at two seats, the one opponent's token. `players` is the net's
    own count, a Python number, so a traced graph carries no shape read."""
    return (sn.sum(dim, keepdim=True) - sn) / max(players - 1, 1)


def _pick(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` indexed by `index` along `dim`, as `x[..., index]` there. Without
    grad (collection) by `index_select`, the same copy at a fraction of
    advanced indexing's cost on CPU; with grad by indexing, whose backward
    (an accumulating `index_put`) the update has always run, deterministic on
    CUDA where `index_select`'s is not."""
    if torch.is_grad_enabled():
        return x[(slice(None),) * dim + (index,)]
    return x.index_select(dim, index.reshape(-1)).reshape(*x.shape[:dim], *index.shape, *x.shape[dim + 1:])


def _unit(x: Tensor, gain: Tensor) -> Tensor:
    """`x` `(..., KEY_WIDTH)` layer-normalised over its last axis (no bias),
    times `gain`: a query or key whose length is bounded by the gain alone."""
    # A constant normalised shape, not `x.shape[-1:]`, which an ONNX trace reads as dynamic.
    return nn.functional.layer_norm(x, (KEY_WIDTH,)) * gain.to(x.dtype)


def unpack9(layout: Layout9, x: Tensor) -> dict[str, Tensor]:
    """A `(B, layout.width)` batch of `contract9.flatten`ed rows as the
    tensors `HexNet.forward` reads."""
    out = {}
    for name, piece in layout.slices.items():
        out[name] = x[:, piece].reshape(x.shape[0], *layout.shapes[name])
    out["vertex_owner"] = out["vertex_owner"].round().long()
    out["road_owner"] = out["road_owner"].round().long()
    return out


class HexNet(nn.Module):
    CONTRACT = "9"

    def __init__(self, space: ActionSpace, topology, players: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.CONTRACT = self.config.contract
        # Contract 11d: two seat tokens of their own widths and a board token.
        self.two = self.CONTRACT == "11d"
        if self.two and players != 2:
            raise ValueError(f"contract 11d is the two-seat observation, not {players} seats")
        self.players = players
        self.space = space
        width = self.config.width
        hexes, vertices, edges = topology.num_hexes, topology.num_vertices, topology.num_edges
        self.num_hexes, self.num_vertices, self.num_edges = hexes, vertices, edges

        seat_features, table_features = features(self.CONTRACT)
        if self.two:
            self.embed_me = nn.Linear(MY_FEATURES_11D, width)
            self.embed_other = nn.Linear(OTHER_FEATURES_11D, width)
            self.embed_board = nn.Linear(table_features, width)
        else:
            self.embed_seat = nn.Linear(seat_features, width)
            self.embed_table = nn.Linear(table_features, width)
        self.embed_hex = nn.Linear(hex_features(self.CONTRACT), width)
        self.embed_vertex = nn.Linear(VERTEX_FEATURES, width)
        self.rounds = nn.ModuleList(_Round(width, self.config.norm, self.config.messages == "pooled",
                                           self.config.attention, self.config.others, board=self.two)
                                    for _ in range(self.config.rounds))
        self.out_hex = _norm(width, self.config.norm)
        self.out_vertex = _norm(width, self.config.norm)
        self.out_seat = _norm(width, self.config.norm)
        self.out_table = _norm(width, self.config.norm)

        # Heads. `me_*` put the perspective's token into what a board head reads.
        def head(in_dim: int, out_dim: int = 1) -> nn.Module:
            return nn.Linear(in_dim, out_dim) if self.config.heads == "linear" else _mlp(in_dim, width, out_dim)
        self.value_head = head(2 * width)
        self.points_head = head(2 * width) if self.config.points else None   # the value head's shape
        joint = self.config.double_turn == "joint"
        if self.two:
            self._query_heads(head, width)
        else:
            self.vertex_head = nn.Linear(width, 2)          # settlement, city
            self.road_head = head(2 * width)                # one ordered end pair
            self.robber_head = head(2 * width)              # hex with a victim's token
            self.nobody_head = nn.Linear(width, 1)          # hex, robbing nobody
            globals_size = sum(space.sizes[k] for k in _GLOBAL_KINDS)
            self.global_head = head(2 * width if self.config.global_seat else width, globals_size)  # off the table
            # (and with `global_seat` the perspective's own seat token), as HexNet's global head
            self.setup_head = head(2 * width)               # settlement with its road's far end
            self.first_head = head(2 * width) if joint else None
            self.second_head = head(2 * width) if joint else None
            self.pair_head = nn.Linear(width, width, bias=False) if joint else None
            # Linear: my token and the table give a weight per resource kept and per
            # resource given, and a candidate scores their sum over its counts (a
            # linear layer over all three would add the same context term to every
            # candidate, and the choice would not read the position at all).
            self.discard_head = (nn.Linear(2 * width, 10) if self.config.discard == "linear"
                                 else _mlp(2 * width + 10, width, 1))
            self.road_building = nn.Linear(2 * width, 1)    # playing the card at all
        self._initialise()

        hex_vertex = np.asarray([(h, v) for h, vs in enumerate(topology.hex_vertices) for v in vs]).T
        ends = np.asarray(topology.edges, dtype=np.int64)           # (edges, 2)
        vertex_edge = np.asarray([(v, e) for v, es in enumerate(topology.vertex_edges) for e in es]).T
        self.register_buffer("A_vh", _mean_adjacency(hex_vertex[1], hex_vertex[0], vertices, hexes), persistent=False)
        self.register_buffer("A_hv", _mean_adjacency(hex_vertex[0], hex_vertex[1], hexes, vertices), persistent=False)
        both = np.concatenate([ends, ends[:, ::-1]])
        self.register_buffer("A_vv", _mean_adjacency(both[:, 0], both[:, 1], vertices, vertices), persistent=False)
        self.register_buffer("A_ve", _mean_adjacency(vertex_edge[0], vertex_edge[1], vertices, edges), persistent=False)
        self.register_buffer("ends", torch.from_numpy(ends), persistent=False)
        # An edge's midpoint (the mean of its two ends) as one more adjacency.
        self.register_buffer("A_mid", _mean_adjacency(np.repeat(np.arange(edges), 2), ends.ravel(), edges, vertices),
                             persistent=False)
        # Wiring, not architecture (as `hexn.model.HexNet.fused`): the
        # node-major trunk (`_fused_trunk`), or the reference loop below. The
        # same numbers up to float reassociation, the same parameters.
        self.fused = True
        pairs = np.asarray(incidences(topology), dtype=np.int64)     # (incidences, 2): vertex, edge
        far = np.where(ends[pairs[:, 1], 0] == pairs[:, 0], ends[pairs[:, 1], 1], ends[pairs[:, 1], 0])
        self.register_buffer("inc_vertex", torch.from_numpy(pairs[:, 0]), persistent=False)
        self.register_buffer("inc_far", torch.from_numpy(far), persistent=False)
        if joint:
            self.register_buffer("pairs_ok", torch.from_numpy(double_pairs(topology)), persistent=False)

        # Where each block lands in the flat space.
        def block(kind: ActionType) -> Tensor:
            start = space.offsets[kind]
            return torch.arange(start, start + space.sizes[kind])
        self.register_buffer("slot_settlement", block(ActionType.BUILD_SETTLEMENT), persistent=False)
        self.register_buffer("slot_city", block(ActionType.BUILD_CITY), persistent=False)
        self.register_buffer("slot_road", block(ActionType.BUILD_ROAD), persistent=False)
        self.register_buffer("slot_robber", block(ActionType.MOVE_ROBBER), persistent=False)
        # The same slots kept on the host, never moved with the module: `host_rows`
        # reads them where the batch still is, so finding rows costs no device read.
        self.slot_robber_host = block(ActionType.MOVE_ROBBER)
        self.register_buffer("slot_setup_road", block(ActionType.SETUP_ROAD), persistent=False)
        self.register_buffer("slot_global", torch.cat([block(k) for k in _GLOBAL_KINDS]), persistent=False)
        if self.two:
            # 11d: a setup turn's two moves in the flat row, per incidence; the
            # hands a resource move leaves (`_resource_values`).
            inc_edge = torch.from_numpy(pairs[:, 1])
            self.register_buffer("inc_setup_settlement", space.offsets[ActionType.SETUP_SETTLEMENT]
                                 + torch.from_numpy(pairs[:, 0]), persistent=False)
            self.register_buffer("inc_setup_road", space.offsets[ActionType.SETUP_ROAD] + inc_edge, persistent=False)
            self.register_buffer("yop_pairs", torch.tensor(YEAR_OF_PLENTY_PAIRS, dtype=torch.long), persistent=False)
            self.register_buffer("inc_edge", inc_edge, persistent=False)
            # My token's settlements left: whether a setup settlement is the paying one.
            self.settlements_left = MY_COLUMNS_11D["pieces"].start
            # One card of each resource, as a hand column reads it (a count over
            # `CARD_SCALE`), and each Year of Plenty slot's two cards.
            card = torch.eye(NUM_RESOURCES) / CARD_SCALE
            self.register_buffer("card", card, persistent=False)
            self.register_buffer("plenty_cards", card[self.yop_pairs[:, 0]] + card[self.yop_pairs[:, 1]],
                                 persistent=False)
            self.my_hand = MY_COLUMNS_11D["hand"]
            self.my_rates = MY_COLUMNS_11D["rates"]
            self.their_hand = slice(MY_FEATURES_11D + OTHER_COLUMNS_11D["hand"].start,
                                    MY_FEATURES_11D + OTHER_COLUMNS_11D["hand"].stop)
            # The flat row is gathered, not concatenated block by block: each
            # slot's type score (the robber block a zero column: `robber_logits`
            # scores it on demand), plus each slot's term in `_emit_queries`'
            # vector of query reads and hand values (`ADDED`, in that order),
            # or a zero column where a move adds nothing to its type. A move in
            # `UNTYPED` reads the zero column too: its position offers no other
            # kind of move, so the type would be shared by every option, untrained
            # by any loss (it cancels), and free to drift.
            T = ActionType
            sizes = {T.SETUP_SETTLEMENT: vertices, T.SETUP_ROAD: edges, T.BUILD_SETTLEMENT: vertices,
                     T.BUILD_CITY: vertices, T.BUILD_ROAD: edges, T.PLAY_MONOPOLY: NUM_RESOURCES,
                     T.PLAY_YEAR_OF_PLENTY: len(YEAR_OF_PLENTY_PAIRS), T.BANK_TRADE: NUM_RESOURCES ** 2,
                     T.DISCARD: NUM_RESOURCES}
            if any(space.sizes[t] != n for t, n in sizes.items()):
                raise ValueError(f"contract 11d's flat blocks do not match the space: {space.sizes}")
            added = sum(sizes.values())
            slot_type = torch.full((space.size,), len(MOVE_TYPES), dtype=torch.long)
            slot_added = torch.full((space.size,), added, dtype=torch.long)
            for t in MOVE_TYPES:
                if t not in UNTYPED:
                    slot_type[block(t)] = int(t)
            at = 0
            for t, n in sizes.items():
                slot_added[block(t)] = torch.arange(at, at + n)
                at += n
            self.register_buffer("slot_type", slot_type, persistent=False)
            self.register_buffer("slot_added", slot_added, persistent=False)

    def _query_heads(self, head, width: int) -> None:
        """The contract-11d heads: every move is scored as its type's score
        (a hidden layer, so a move with no target still weighs the position)
        plus, for a board move, a query per role against the keys of what it
        names, and for a resource move the value of the hands it leaves
        (`_emit_queries`). They read a context of my token, the opponent's,
        and the board token."""
        context = 3 * width
        self.move_head = _mlp(context, width, len(MOVE_TYPES))               # a score per move type
        self.query_head = head(context, len(QUERY_ROLES) * KEY_WIDTH)          # a query per role
        self.vertex_key = nn.Linear(width, KEY_WIDTH)
        self.edge_key = nn.Linear(2 * width, KEY_WIDTH)                      # f(a, b) + f(b, a)
        self.robber_key = nn.Linear(2 * width, KEY_WIDTH)                    # a hex with a victim's token
        self.nobody_key = nn.Linear(width, KEY_WIDTH)                        # a hex, robbing nobody
        # The normalised query . key (`QUERY_SCALE`): each query's and each
        # kind of key's gain, and each role's log scale.
        self.query_gain = nn.Parameter(torch.ones(len(QUERY_ROLES), KEY_WIDTH))
        self.key_gain = nn.Parameter(torch.ones(len(KEY_KINDS), KEY_WIDTH))
        # t such that `role_scales` starts at exactly QUERY_SCALE.
        cap = math.log(ROLE_SCALE_MAX)
        start = cap - math.log(math.expm1(cap - math.log(QUERY_SCALE)))
        self.role_scale = nn.Parameter(torch.full((len(QUERY_ROLES),), start))
        # The hands a move leaves, mine and the opponent's (each a count over
        # `CARD_SCALE` by resource), valued in the context: the discard MLP's shape.
        self.hand_value = _mlp(context + 2 * HAND_READS, width, 1)
        # Road Building's two roads read together, k_first' R k_second on the
        # normalised edge keys, and a "joint" net's double turn its two
        # settlements, k_first' P k_second on the vertex keys. Neither is
        # symmetric: the order (which road first, which settlement pays) matters.
        self.road_pair = nn.Parameter(torch.zeros(KEY_WIDTH, KEY_WIDTH))
        joint = self.config.double_turn == "joint"
        self.pair_matrix = nn.Parameter(torch.zeros(KEY_WIDTH, KEY_WIDTH)) if joint else None
        for name in ("vertex_head", "road_head", "robber_head", "nobody_head", "global_head", "setup_head",
                     "first_head", "second_head", "pair_head", "discard_head", "road_building"):
            setattr(self, name, None)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # A 9.1 net can start from 9: only the appended input blocks are
        # new. Every other key and shape is still checked by torch's loader.
        if self.CONTRACT == "9.1":
            for name, old_width in (("embed_seat.weight", SEAT_FEATURES), ("embed_table.weight", TABLE_FEATURES),
                                    ("embed_hex.weight", HEX_FEATURES)):
                key = prefix + name
                weight = state_dict.get(key)
                target = self.state_dict()[name]
                if weight is not None and tuple(weight.shape) == (target.shape[0], old_width):
                    state_dict[key] = nn.functional.pad(weight, (0, target.shape[1] - old_width))
        # A 9d.1 net can start from 9d: the seat token's economy columns are
        # appended, so their embedding weights start at zero. A `global_seat`
        # net can start from one without: the global head's weights for the
        # seat token start at zero. Either way the loaded net plays as before.
        if self.CONTRACT == "9d.1":
            key = prefix + "embed_seat.weight"
            weight = state_dict.get(key)
            target = self.state_dict()["embed_seat.weight"]
            if weight is not None and tuple(weight.shape) == (target.shape[0], SEAT_FEATURES_9D):
                state_dict[key] = nn.functional.pad(weight, (0, target.shape[1] - SEAT_FEATURES_9D))
        if self.config.global_seat and isinstance(self.global_head, nn.Linear):
            key = prefix + "global_head.weight"
            weight = state_dict.get(key)
            if weight is not None and weight.shape[1] == self.config.width:
                state_dict[key] = nn.functional.pad(weight, (0, self.config.width))
        super()._load_from_state_dict(state_dict, prefix, *args)

    def _initialise(self) -> None:
        """`hexn.model.HexNet._initialise`'s scheme: orthogonal √2 through the
        trunk, 0.01 on every layer that emits a logit, 1.0 on the win head."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        def output(head: nn.Module) -> nn.Linear:     # the layer that emits the number
            return head if isinstance(head, nn.Linear) else head[-1]
        for head in (self.vertex_head, self.road_head, self.robber_head, self.nobody_head, self.global_head,
                     self.setup_head, self.first_head, self.second_head, self.pair_head, self.discard_head,
                     self.road_building):
            if head is not None:
                nn.init.orthogonal_(output(head).weight, gain=0.01)
        if self.two:                                  # 11d: the type scores and the queries make every logit
            # The queries are normalised: their scale is `role_scales`'.
            for head in (self.move_head, self.hand_value):
                nn.init.orthogonal_(output(head).weight, gain=0.01)
            for pair in (self.road_pair, self.pair_matrix):
                if pair is not None:
                    nn.init.orthogonal_(pair, gain=0.01)
        nn.init.orthogonal_(output(self.value_head).weight, gain=1.0)
        if self.points_head is not None:
            nn.init.orthogonal_(output(self.points_head).weight, gain=1.0)

    def forward(self, seats: Tensor, table: Tensor, hexes: Tensor, vertices: Tensor,
                vertex_owner: Tensor, road_owner: Tensor, reach: Tensor, *, policy: bool = True) -> Prediction:
        """The win head, and with `policy` the policy's outputs: a value-only
        row (another seat's view of a decision) skips every policy head."""
        if self.fused:
            s, t, h, v = self._fused_trunk(seats, table, hexes, vertices, vertex_owner, road_owner, reach)
            h, v, s, t = self.out_hex(h), self.out_vertex(v), self.out_seat(s), self._out_table(t)
            value_logits, value, points = self._value(s, t, seats)
            if not policy:
                return Prediction(value=value, value_logits=value_logits, points=points)
            return self._emit(s, t, h, v, value, value_logits, points, seats)
        # Ownership as one comparison each: a pointer of -1 matches no seat.
        rows = torch.arange(self.players, device=seats.device)
        owns_v = (vertex_owner.unsqueeze(-1) == rows).to(seats.dtype)            # (B, V, P)
        owns_e = (road_owner.unsqueeze(-1) == rows).to(seats.dtype)              # (B, E, P)
        road = owns_e.sum(-1)                                                     # (B, E)
        built = owns_v.sum(1).clamp(min=1.0).unsqueeze(-1)                        # (B, P, 1)
        roads_held = owns_e.sum(1).clamp(min=1.0).unsqueeze(-1)
        reached = reach.sum(-1).clamp(min=1e-3).unsqueeze(-1)

        s, t = self._embed_seats(seats, table, 1)
        h = self.embed_hex(hexes)
        v = self.embed_vertex(vertices)
        scale = s.shape[-1] ** 0.5
        for r in self.rounds:
            if r.pooled:
                h, v, s, t = self._pooled_round(r, h, v, s, t, owns_v, owns_e, road, built, roads_held, reach,
                                                reached, scale)
                continue
            hn, vn, sn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s)
            tn = r.norm_table(t)
            tb = tn.unsqueeze(1)
            v_from_h = self.A_vh @ hn
            h_from_v = self.A_hv @ vn
            v_from_v = self.A_vv @ vn
            road_token = owns_e @ sn + road.unsqueeze(-1) * r.road                 # (B, E, W)
            v_from_road = self.A_ve @ road_token
            v_from_owner = owns_v @ sn
            v_from_reach = reach.transpose(1, 2) @ sn
            middle = 0.5 * (vn[:, self.ends[:, 0]] + vn[:, self.ends[:, 1]])         # (B, E, W)
            s_from_built = owns_v.transpose(1, 2) @ vn / built
            s_from_roads = owns_e.transpose(1, 2) @ middle / roads_held
            s_from_reach = reach @ vn / reached
            seats_in = []
            if r.attention:
                attention = torch.softmax(r.query(sn) @ r.key(sn).transpose(1, 2) / scale, -1)
                seats_in.append(attention @ r.value(sn))
            if r.others:
                seats_in.append(_others(sn, 1, self.players))

            h = h + r.hex(torch.cat([hn, h_from_v, tb.expand_as(h)], -1))
            v = v + r.vertex(torch.cat([vn, v_from_h, v_from_v, v_from_road, v_from_owner, v_from_reach,
                                        tb.expand_as(v)], -1))
            s = s + r.seat(torch.cat([sn, *seats_in, s_from_built, s_from_roads, s_from_reach,
                                      tb.expand_as(s)], -1))
            # The table reads each kind's mean token, normalised: the norm of
            # the mean, one row per kind, rather than every token normalised again.
            if r.board:
                t = t + self._board_step(r, tn, hn, vn, h.mean(1), v.mean(1), s[:, 0])
            else:
                t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(1)), r.norm_hex(h.mean(1)),
                                           r.norm_vertex(v.mean(1))], -1))
        h, v, s, t = self.out_hex(h), self.out_vertex(v), self.out_seat(s), self._out_table(t)
        value_logits, value, points = self._value(s, t, seats)
        if not policy:
            return Prediction(value=value, value_logits=value_logits, points=points)
        return self._emit(s, t, h, v, value, value_logits, points, seats)

    def _embed_seats(self, seats: Tensor, table: Tensor, dim: int) -> tuple[Tensor, Tensor | None]:
        """The seat tokens, stacked along `dim` (1 batch-first, 0 seat-major),
        and the table token: under 11d my token and the opponent's, each by
        its own layer from its own columns, and the board token from the
        board block."""
        if self.two:
            return (torch.stack([self.embed_me(seats[..., :MY_FEATURES_11D]),
                                 self.embed_other(seats[..., MY_FEATURES_11D:])], dim), self.embed_board(table))
        s = self.embed_seat(seats)
        return (s if dim == 1 else s.transpose(0, 1)), self.embed_table(table)

    def _board_step(self, r: _Round, bn: Tensor, hn: Tensor, vn: Tensor, h_mean: Tensor, v_mean: Tensor,
                    me: Tensor) -> Tensor:
        """Contract 11d's board token's update in round `r`, from its own
        normalised token `bn` `(B, W)`: one attention read of the round's
        normalised hexes `hn` and vertices `vn` (batch-first), its query from
        `bn` and each kind of node's own keys and values, `BOARD_READ` wide;
        then an MLP over [bn, the hexes' and vertices' means and my seat
        token, each normalised, and the read], the table update's shape."""
        query = r.board_query(bn)                                                  # (B, K)
        pairs = torch.cat([r.board_hex(hn), r.board_vertex(vn)], 1)               # (B, H + V, 2K)
        keys, values = pairs.split(BOARD_READ, -1)
        weights = torch.softmax((keys @ _as(query, keys.dtype).unsqueeze(-1)).squeeze(-1) / BOARD_READ ** 0.5, -1)
        read = (_as(weights, values.dtype).unsqueeze(1) @ values).squeeze(1)
        return r.table(torch.cat([bn, r.norm_hex(h_mean), r.norm_vertex(v_mean), r.norm_seat(me),
                                  _as(read, bn.dtype)], -1))

    def _out_table(self, t: Tensor | None) -> Tensor | None:
        return None if t is None else self.out_table(t)

    def _fused_trunk(self, seats, table, hexes, vertices, vertex_owner, road_owner, reach):
        """The rounds node-major, as `hexn.model.HexNet._fused_round` runs
        HexNet's: tokens held `(nodes, batch, width)`, so each neighbour mean
        is one `(nodes, nodes) @ (nodes, batch * width)` product; each update's
        first linear applied block by block, the table's block once per
        position; and every message that reads a seat token projected on the
        four seat tokens before it is gathered (an owner) or mixed (reach) per
        vertex. The same numbers as the reference loop up to float
        reassociation. Returns batch-first tokens, as the reference does."""
        players, width = self.players, self.config.width
        batch = seats.shape[0]
        column = torch.arange(batch, device=seats.device).unsqueeze(0)            # (1, B)
        nobody = torch.full_like(vertex_owner, players)
        own_v = torch.where(vertex_owner >= 0, vertex_owner, nobody).t()          # (V, B), none -> P
        own_e = torch.where(road_owner >= 0, road_owner, torch.full_like(road_owner, players)).t()
        # A padded seat projection `(P + 1, B, W)` read at each node's owner,
        # `x[own, column]`, as rows of its `(P + 1) * B` flattening (`_pick`).
        at_v, at_e = own_v * batch + column, own_e * batch + column

        def owner_rows(x: Tensor, own: Tensor, at: Tensor) -> Tensor:
            if torch.is_grad_enabled():
                return x[own, column]
            return _pick(x.reshape(-1, x.shape[-1]), 0, at)
        rows = torch.arange(players, device=seats.device)
        owns_v = (vertex_owner.unsqueeze(-1) == rows)                            # (B, V, P) bool
        owns_e = (road_owner.unsqueeze(-1) == rows)                              # (B, E, P)
        road = (road_owner >= 0).t().unsqueeze(-1)                                # (E, B, 1)
        has_owner = (vertex_owner >= 0).t().unsqueeze(-1)                         # (V, B, 1)
        built = owns_v.sum(1).clamp(min=1).unsqueeze(-1)                          # (B, P, 1)
        roads_held = owns_e.sum(1).clamp(min=1).unsqueeze(-1)
        reached = reach.sum(-1).clamp(min=1e-3).unsqueeze(-1)                     # (B, P, 1)
        # Seat <- board weights, one (B, 2P, V) matrix: buildings (a mean) then reach (a weighted mean).
        to_seat = torch.cat([owns_v.transpose(1, 2) / built, reach / reached], 1)
        roads_to_seat = owns_e.transpose(1, 2) / roads_held                      # (B, P, E)

        def mean(adjacency: Tensor, x: Tensor) -> Tensor:
            nodes, b, w = x.shape
            return (adjacency.to(x.dtype) @ x.reshape(nodes, b * w)).reshape(adjacency.shape[0], b, w)

        def pad(x: Tensor) -> Tensor:                                             # a zero row: no owner
            return torch.cat([x, x.new_zeros(1, *x.shape[1:])], 0)

        def by_reach(projected: Tensor) -> Tensor:                                # (P, B, W) -> (V, B, W)
            return (reach.transpose(1, 2).to(projected.dtype) @ projected.transpose(0, 1)).transpose(0, 1)

        s, t = self._embed_seats(seats, table, 0)                                 # (P, B, W), (B, W)
        h = self.embed_hex(hexes).transpose(0, 1)                                 # (H, B, W)
        v = self.embed_vertex(vertices).transpose(0, 1)                           # (V, B, W)
        scale = width ** 0.5
        for r in self.rounds:
            hn, vn, sn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s)
            tn = r.norm_table(t)                                                  # what every token reads
            dtype = vn.dtype
            # Seat <- board: buildings and reach over vertices, roads over edge midpoints.
            vb = vn.transpose(0, 1)                                               # (B, V, W) view
            built_reach = (to_seat.to(dtype) @ vb).transpose(0, 1)                # (2P, B, W)
            s_built, s_reach = built_reach[:players], built_reach[players:]
            s_roads = (roads_to_seat.to(dtype) @ mean(self.A_mid, vn).transpose(0, 1)).transpose(0, 1)
            seats_in = []
            if r.attention:
                q, k, val = r.query(sn), r.key(sn), r.value(sn)                   # (P, B, W)
                attention = torch.softmax((q.unsqueeze(1) * k.unsqueeze(0)).sum(-1) / scale, 1)   # (P, P, B)
                seats_in.append((attention.unsqueeze(-1) * val.unsqueeze(0)).sum(1))  # (P, B, W)
            if r.others:
                seats_in.append(_others(sn, 0, players))
            if r.pooled:
                hw, vw, sw = (x[0].weight.split(width, 1) for x in (r.hex, r.vertex, r.seat))
                road_seat = pad(sn @ vw[2].T)
                road_term = owner_rows(road_seat, own_e, at_e) + road.to(dtype) * (r.road.to(dtype) @ vw[2].T.to(dtype))
                seat_proj = sn @ vw[3].T
                v_seats = (owner_rows(pad(seat_proj), own_v, at_v) + has_owner.to(dtype) * (r.owned.to(dtype) @ vw[3].T.to(dtype))
                           + by_reach(seat_proj))
                pre_h = hn @ hw[0].T + mean(self.A_hv, vn) @ hw[1].T + (r.hex_table(tn) + r.hex[0].bias).unsqueeze(0)
                pre_v = (vn @ vw[0].T + mean(self.A_vh, hn) @ vw[1].T + mean(self.A_vv, vn) @ vw[2].T
                         + mean(self.A_ve, road_term) + v_seats + (r.vertex_table(tn) + r.vertex[0].bias).unsqueeze(0))
                pre_s = sum(x @ w.T for x, w in zip([sn, *seats_in, s_built + s_roads + s_reach], sw))
                pre_s = pre_s + (r.seat_table(tn) + r.seat[0].bias).unsqueeze(0)
            else:
                hw, vw, sw = (x[0].weight.split(width, 1) for x in (r.hex, r.vertex, r.seat))
                road_term = owner_rows(pad(sn @ vw[3].T), own_e, at_e) + road.to(dtype) * (r.road.to(dtype) @ vw[3].T.to(dtype))
                pre_h = hn @ hw[0].T + mean(self.A_hv, vn) @ hw[1].T + (tn @ hw[2].T + r.hex[0].bias).unsqueeze(0)
                pre_v = (vn @ vw[0].T + mean(self.A_vh, hn) @ vw[1].T + mean(self.A_vv, vn) @ vw[2].T
                         + mean(self.A_ve, road_term) + owner_rows(pad(sn @ vw[4].T), own_v, at_v) + by_reach(sn @ vw[5].T)
                         + (tn @ vw[6].T + r.vertex[0].bias).unsqueeze(0))
                pre_s = sum(x @ w.T for x, w in zip([sn, *seats_in, s_built, s_roads, s_reach], sw))
                pre_s = pre_s + (tn @ sw[-1].T + r.seat[0].bias).unsqueeze(0)
            h = h + r.hex[2](r.hex[1](pre_h))
            v = v + r.vertex[2](r.vertex[1](pre_v))
            s = s + r.seat[2](r.seat[1](pre_s))
            if r.board:
                t = t + self._board_step(r, tn, hn.transpose(0, 1), vn.transpose(0, 1), h.mean(0), v.mean(0), s[0])
            else:
                t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(0)), r.norm_hex(h.mean(0)), r.norm_vertex(v.mean(0))],
                                          -1))
        return (s.transpose(0, 1).contiguous(), t, h.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous())

    def _pooled_round(self, r: _Round, h, v, s, t, owns_v, owns_e, road, built, roads_held, reach, reached, scale):
        """One round with each kind of message pooled into one input."""
        hn, vn, sn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s)
        tn = r.norm_table(t)
        h_from_v = self.A_hv @ vn
        v_from_h = self.A_vh @ hn
        # Per incident edge, the neighbour's token and the road's owner's (a
        # road vector where one stands), averaged over the vertex's edges.
        road_token = owns_e @ sn + road.unsqueeze(-1) * r.road
        v_edges = self.A_vv @ vn + self.A_ve @ road_token
        # The seats that reach this vertex, and its building's owner marked.
        v_seats = owns_v @ sn + owns_v.sum(-1, keepdim=True) * r.owned + reach.transpose(1, 2) @ sn
        middle = 0.5 * (vn[:, self.ends[:, 0]] + vn[:, self.ends[:, 1]])
        s_board = (owns_v.transpose(1, 2) @ vn / built + owns_e.transpose(1, 2) @ middle / roads_held
                   + reach @ vn / reached)
        seats_in = []
        if r.attention:
            attention = torch.softmax(r.query(sn) @ r.key(sn).transpose(1, 2) / scale, -1)
            seats_in.append(attention @ r.value(sn))
        if r.others:
            seats_in.append(_others(sn, 1, self.players))

        def update(mlp: nn.Sequential, table: nn.Linear, x: Tensor) -> Tensor:
            return mlp[2](mlp[1](mlp[0](x) + table(tn).unsqueeze(1)))

        h = h + update(r.hex, r.hex_table, torch.cat([hn, h_from_v], -1))
        v = v + update(r.vertex, r.vertex_table, torch.cat([vn, v_from_h, v_edges, v_seats], -1))
        s = s + update(r.seat, r.seat_table, torch.cat([sn, *seats_in, s_board], -1))
        if r.board:
            t = t + self._board_step(r, tn, hn, vn, h.mean(1), v.mean(1), s[:, 0])
        else:
            t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(1)), r.norm_hex(h.mean(1)), r.norm_vertex(v.mean(1))], -1))
        return h, v, s, t

    def _value(self, s: Tensor, t: Tensor | None, seats: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """The win head's logits and softmax over seats, and the points head's
        number per seat where the net has one: each seat token read with the
        table, or under 11d with my token."""
        context = s[:, :1] if self.two else t.unsqueeze(1)
        read = torch.cat([s, context.expand_as(s)], -1)
        value_logits = self.value_head(read).squeeze(-1)
        points = None if self.points_head is None else self.points_head(read).squeeze(-1).float()
        if self.CONTRACT == "9.1":
            live = seats[..., SEAT_FEATURES] > 0.5
            # Finite logits keep categorical cross-entropy's zero targets
            # finite even in fp16. Softmax gives retired slots exactly zero.
            value_logits = value_logits.masked_fill(~live, -1e4)
            if points is not None:
                points = points.masked_fill(~live, 0.0)
        return value_logits, torch.softmax(value_logits, -1), points

    def _emit(self, s: Tensor, t: Tensor, h: Tensor, v: Tensor, value: Tensor, value_logits: Tensor,
              points: Tensor | None = None, seats: Tensor | None = None) -> Prediction:
        if self.two:
            return self._emit_queries(s, t, h, v, value, value_logits, seats)
        batch = s.shape[0]
        me = s[:, 0]
        a, b = _pick(v, 1, self.ends[:, 0]), _pick(v, 1, self.ends[:, 1])
        road = (self.road_head(torch.cat([a, b], -1)) + self.road_head(torch.cat([b, a], -1))).squeeze(-1)
        vertex = self.vertex_head(v)                                              # (B, V, 2)
        # One dtype for the flat row: under autocast the heads emit bf16 while
        # LayerNorm'd tokens are fp32.
        logits = vertex.new_zeros(batch, self.space.size)
        logits = logits.index_copy(1, self.slot_settlement, vertex[..., 0])
        logits = logits.index_copy(1, self.slot_city, vertex[..., 1])
        logits = logits.index_copy(1, self.slot_road, road.to(vertex.dtype))
        globals_in = torch.cat([t, me.to(t.dtype)], -1) if self.config.global_seat else t
        logits = logits.index_copy(1, self.slot_global, self.global_head(globals_in).to(vertex.dtype))
        return Prediction(value=value, value_logits=value_logits, points=points, logits=logits, road=road, me=me,
                          table=t, vertices=v, hexes=h, seats=s)

    def _emit_queries(self, s: Tensor, t: Tensor, h: Tensor, v: Tensor, value: Tensor, value_logits: Tensor,
                      seats: Tensor) -> Prediction:
        """The contract-11d flat row: each move its type's score plus its
        roles' queries against the keys of what it names, block by block in
        the flat space's `ActionType` order:

        - roll, end turn, buy a card, play a knight, play Road Building (the
          card alone, as a table asks it a move at a time): the type alone;
        Each query . key is normalised (`QUERY_SCALE`): both layer-normalised
        with a gain, their product over sqrt(KEY_WIDTH), times the role's
        learned scale.

        - a settlement, a city, a setup settlement at v: + q . k_v, a setup
          settlement's query the paying one where my token's settlements left
          say one is already placed (the snake's second, which pays);
        - a road, a setup road at e: + q . k_e (a free road owed is a road);
        - a resource move: + V(the two hands it leaves), the hand value
          (`_resource_values`): Year of Plenty (a, b) my hand + a + b; Monopoly r
          my hand + the opponent's r, its r none; a bank trade giving g for t
          my hand less my rate of g, plus a t; a discard of one card of r my
          hand less it;
        - the robber block is `robber_logits`', scored on demand.

        `road` is each edge's road score; the setup turn, Road Building and
        discard candidates read `types`, `rb_road`, `edge_keys`,
        `hand_context` and `hands`."""
        batch = s.shape[0]
        me, other = s[:, 0], s[:, 1]
        context = torch.cat([me, other, t], -1)
        types = self.move_head(context)                                           # (B, types)
        # Each role's normalised query with its scale and 1/sqrt(KEY_WIDTH) folded in.
        queries = _unit(self.query_head(context).reshape(batch, len(QUERY_ROLES), KEY_WIDTH), self.query_gain)
        queries = queries * _as(self.role_scales() / KEY_WIDTH ** 0.5, queries.dtype).unsqueeze(-1)
        a, b = _pick(v, 1, self.ends[:, 0]), _pick(v, 1, self.ends[:, 1])
        k_vertex = _unit(self.vertex_key(v), self.key_gain[0])                   # (B, V, K)
        k_edge = _unit(self.edge_key(torch.cat([a, b], -1)) + self.edge_key(torch.cat([b, a], -1)), self.key_gain[1])
        # Every role's read of its kind's keys, one product a kind: (B, V, 4), (B, E, 3).
        roles = len(VERTEX_ROLES)
        on_vertex = k_vertex @ _as(queries[:, :roles], k_vertex.dtype).transpose(1, 2)
        on_edge = k_edge @ _as(queries[:, roles:roles + len(EDGE_ROLES)], k_edge.dtype).transpose(1, 2)
        settle, city, setup, paying_read = on_vertex.unbind(-1)
        road_read, setup_road, rb_road = on_edge.unbind(-1)
        mine, theirs = seats[:, self.my_hand], seats[:, self.their_hand]          # (B, resources)
        # A setup settlement pays the starting hand when it is the seat's
        # second: with a settlement already placed, fewer than all are left.
        paying = seats[:, self.settlements_left] * PIECE_SCALE < MAX_SETTLEMENTS - 0.5
        setup_settlement = torch.where(paying.unsqueeze(1), paying_read, setup)
        hand_context = self._hand_context(context)
        worth = self._resource_values(hand_context, mine, theirs, seats[:, self.my_rates])
        # Each slot its type's score plus its read or hand value (`slot_type`, `slot_added`).
        added = torch.cat([setup_settlement, setup_road, settle, city, road_read, _as(worth, on_vertex.dtype),
                           on_vertex.new_zeros(batch, 1)], 1)
        logits = (torch.cat([types, types.new_zeros(batch, 1)], 1).index_select(1, self.slot_type)
                  + _as(added, types.dtype).index_select(1, self.slot_added))
        return Prediction(value=value, value_logits=value_logits, logits=logits, road=logits[:, self.slot_road],
                          me=me, table=t, vertices=v, hexes=h, seats=s, types=types,
                          robber_query=queries[:, _ROLE["robber"]], rb_road=rb_road, hand_context=hand_context,
                          hands=torch.stack([mine, theirs], 1), vertex_keys=k_vertex, edge_keys=k_edge,
                          setup_reads=on_vertex[..., 2:4].transpose(1, 2), setup_road_read=setup_road)

    def direct_weights(self) -> tuple[nn.Linear, ...]:
        """The linear layers whose weight the forward reads outside the layer's
        own call: 11d's hand value's first layer, applied block by block and
        mixed with its hands in elementwise products. A cache of their weights
        cast to another precision (`hexn.inference9`) would change the forward,
        so such a cache leaves them out."""
        return (self.hand_value[0],) if self.two else ()

    def role_scales(self) -> Tensor:
        """Each role's scale on its normalised query . key: exp of the learned
        `role_scale` t, soft-capped at ROLE_SCALE_MAX, exp(c - softplus(c - t))
        with c = log(ROLE_SCALE_MAX): exp(t) while t is well below c, and never
        above the cap however far t grows."""
        cap = math.log(ROLE_SCALE_MAX)
        return (cap - nn.functional.softplus(cap - self.role_scale)).exp()

    def _hand_context(self, context: Tensor) -> Tensor:
        """The hand value's first layer on the context alone, once per row. Its
        input reads [context, my hand, the opponent's hand]."""
        first = self.hand_value[0]
        return nn.functional.linear(context, first.weight[:, :context.shape[-1]], first.bias)

    def _hand_values(self, hand_context: Tensor, mine: Tensor, theirs: Tensor) -> Tensor:
        """V over hands `(..., resources)` mine and the opponent's, beside the
        row's first-layer term (`_hand_context`, broadcast over them): the
        hand value MLP, its first layer applied block by block."""
        weight = self.hand_value[0].weight
        at = weight.shape[1] - 2 * HAND_READS
        hidden = hand_context
        for hand in (mine, theirs):
            hidden = hidden + nn.functional.linear(hand, weight[:, at:at + HAND_READS])
            at += HAND_READS
        return self.hand_value[2](self.hand_value[1](hidden)).squeeze(-1)

    def _resource_values(self, hand_context: Tensor, mine: Tensor, theirs: Tensor, rates: Tensor) -> Tensor:
        """`(B, 50)`: V of the two hands each resource move leaves, in the flat
        space's slot order, Monopoly (5), Year of Plenty (15), bank trade
        (`give * 5 + take`, 25), discard (5): Year of Plenty (a, b) adds a and
        b to mine; Monopoly r adds the opponent's r to mine and leaves it none;
        a bank trade giving g for t takes my rate of g (`rates` the column, the
        rate over 4) and adds a t; a discard of r takes one r.

        V's first layer is linear in the hands, so each move's is the row's
        (the context, my hand and the opponent's, once) plus the layer applied
        to what the move changes: a term per slot, not a hand per slot."""
        weight = self.hand_value[0].weight
        at = weight.shape[1] - 2 * HAND_READS
        first_mine, first_theirs = weight[:, at:at + HAND_READS], weight[:, at + HAND_READS:]  # (hidden, resources)
        row = (hand_context + nn.functional.linear(mine, first_mine)
               + nn.functional.linear(theirs, first_theirs)).unsqueeze(1)                # (B, 1, hidden)
        w_mine, w_theirs = first_mine.T, first_theirs.T                                   # (resources, hidden)
        card = _as(self.card, w_mine.dtype) @ w_mine                                      # one card of r
        rate = rates * (BASE_TRADE_RATIO / CARD_SCALE)                                    # (B, give)
        batch, hidden = mine.shape[0], w_mine.shape[1]
        change = torch.cat([
            theirs.unsqueeze(-1) * (w_mine - w_theirs),                                   # Monopoly r
            (_as(self.plenty_cards, w_mine.dtype) @ w_mine).expand(batch, -1, -1),        # Year of Plenty
            (card.unsqueeze(0) - (rate.unsqueeze(-1) * w_mine).unsqueeze(2)).reshape(batch, -1, hidden),  # trade
            (-card).expand(batch, -1, -1),                                                # discard r
        ], 1)
        return self.hand_value[2](self.hand_value[1](row + _as(change, row.dtype))).squeeze(-1)

    def robber_logits(self, out: Prediction, rows: Tensor) -> Tensor:
        """`(len(rows), hexes * (players + 1))`: the robber block of `rows`'
        flat logits, each hex with each victim's seat token, then nobody."""
        uh = out.hexes[rows]
        s = out.seats[rows]
        if self.two:                                  # 11d: q_robber . k_(hex, victim), nobody from the hex; no type (UNTYPED)
            keys = _unit(torch.cat([self.robber_key(torch.cat([
                uh.unsqueeze(2).expand(-1, -1, self.players, -1),
                s.unsqueeze(1).expand(-1, uh.shape[1], -1, -1)], -1)), self.nobody_key(uh).unsqueeze(2)], 2),
                self.key_gain[2])
            query = out.robber_query[rows].to(keys.dtype)
            scores = (keys @ query[:, None, :, None]).squeeze(-1)                 # (R, H, P + 1)
            return scores.reshape(rows.shape[0], -1)
        victims = self.robber_head(torch.cat([
            uh.unsqueeze(2).expand(-1, -1, self.players, -1),
            s.unsqueeze(1).expand(-1, uh.shape[1], -1, -1)], -1)).squeeze(-1)     # (R, H, P)
        # `rows.shape[0]`, not `len(rows)`: the same number, and one an ONNX
        # trace keeps as the batch's own size (`hexn.export_onnx9`).
        return torch.cat([victims, self.nobody_head(uh)], -1).reshape(rows.shape[0], -1)

    def full_logits(self, out: Prediction) -> Tensor:
        """The whole flat logits, the robber block included, for every row."""
        rows = torch.arange(out.logits.shape[0], device=out.logits.device)
        return out.logits.index_copy(1, self.slot_robber, self.robber_logits(out, rows).to(out.logits.dtype))

    def setup_heads(self, out: Prediction, rows: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """For `rows`, per (settlement, road) incidence the setup turn's score,
        the double turn's first and second (paying) turn's, and per settlement
        pair `(V, V)` the double turn's pair term. A "parts" net has none of the three.

        A contract-11d "joint" net: the first turn is
        q_setup_settlement . k_v + q_setup_road . k_e (it does not pay), the
        second q_setup_settlement_paying . k_v + q_setup_road . k_e, and the
        pair k_first' P k_second over sqrt(KEY_WIDTH), on the normalised vertex keys."""
        if self.two:
            if self.pair_matrix is None:
                raise ValueError(f"a {self.config.double_turn!r} net has no joint double-turn head")
            reads, road = out.setup_reads[rows], out.setup_road_read[rows][:, self.inc_edge]
            first = reads[:, 0][:, self.inc_vertex] + road
            second = reads[:, 1][:, self.inc_vertex] + road
            keys = out.vertex_keys[rows]
            pair = keys @ self.pair_matrix.to(keys.dtype) @ keys.transpose(1, 2) / KEY_WIDTH ** 0.5
            return self.setup_scores(out, rows), first, second, pair
        if self.first_head is None:
            raise ValueError(f"a {self.config.double_turn!r} net has no joint double-turn head")
        u = out.vertices[rows]
        ends = torch.cat([u[:, self.inc_vertex], u[:, self.inc_far]], -1)
        pair = self.pair_head(u) @ u.transpose(1, 2) / u.shape[-1] ** 0.5
        return (self.setup_head(ends).squeeze(-1), self.first_head(ends).squeeze(-1),
                self.second_head(ends).squeeze(-1), pair)

    def flat_scores(self, out: Prediction) -> Tensor:
        """Every flat action's score for every row: `full_logits`, with a
        setup road scored as the road head scores its edge, so a position
        inside a setup turn, asked a move at a time, reads the road head. A
        setup turn asked whole is scored by `setup_scores`. A contract-11d net's flat
        row already scores each setup road by its own query (`_emit_queries`)."""
        logits = self.full_logits(out)
        if self.two:
            return logits
        return logits.index_copy(1, self.slot_setup_road, out.road.to(logits.dtype))

    def double_scores(self, out: Prediction, rows: Tensor) -> Tensor:
        """For `rows`, the double turn's score of every ordered pair of setup
        turns, `(R, incidences, incidences)`, first by second: the first turn's,
        the second (paying) turn's and the pair term of their settlements,
        unmasked (`hexset.decisions.Scored.double`). A "parts" net has no such
        heads."""
        _, first, second, pair = self.setup_heads(out, rows)
        v = self.inc_vertex
        return first.unsqueeze(2) + second.unsqueeze(1) + pair[:, v][:, :, v]

    def setup_scores(self, out: Prediction, rows: Tensor) -> Tensor:
        """For `rows`, per (settlement, road) incidence the setup turn's score
        alone: `setup_heads`' first output, without the double turn's heads.
        A contract-11d net scores the turn as its two moves' flat scores summed,
        q_setup_settlement . k_v and q_setup_road . k_e, with no type score."""
        if self.two:
            logits = out.logits[rows]
            return logits[:, self.inc_setup_settlement] + logits[:, self.inc_setup_road]
        u = out.vertices[rows]
        return self.setup_head(torch.cat([u[:, self.inc_vertex], u[:, self.inc_far]], -1)).squeeze(-1)

    def discard_scores(self, out: Prediction, rows: Tensor, candidates: Tensor) -> Tensor:
        """One score per discard candidate: `candidates` `(K, 10)` (kept, then
        given, card counts), `rows` `(K,)` the batch row each belongs to. A
        contract-11d net scores the value of the hand it keeps, beside the
        opponent's (`_hand_values`), with no type score (`UNTYPED`)."""
        if self.two:
            hands = out.hands[rows]
            kept = candidates[:, :NUM_RESOURCES].to(hands.dtype) / CARD_SCALE
            return self._hand_values(out.hand_context[rows], kept, hands[:, 1]).to(out.types.dtype)
        counts = candidates.to(out.me.dtype) / 10.0
        if self.config.discard == "linear":
            weights = self.discard_head(torch.cat([out.me, out.table], -1))[rows]       # (K, 10)
            return (weights * counts).sum(-1)
        return self.discard_head(torch.cat([out.me[rows], out.table[rows], counts], -1)).squeeze(-1)

    def road_building_scores(self, out: Prediction, rows: Tensor, candidates: Tensor) -> Tensor:
        """One score per Road Building candidate: `candidates` `(K, 2)` edges,
        `-1` for a road not placed; the card's own term plus each road's. A
        contract-11d net's card term is the Road Building type, each road's
        is q_road_building . k_e, and two roads add k_first' R k_second over
        sqrt(KEY_WIDTH) on the normalised edge keys (`road_pair`)."""
        if self.two:
            played = out.types[:, int(ActionType.PLAY_ROAD_BUILDING)][rows]
            road = out.rb_road[rows].to(played.dtype)
            keys = out.edge_keys[rows]
            first, second = candidates[:, 0].long(), candidates[:, 1].long()

            def key(edge: Tensor) -> Tensor:                                        # (K, KEY_WIDTH)
                return keys.gather(1, edge.clamp(min=0)[:, None, None].expand(-1, 1, keys.shape[-1])).squeeze(1)
            # A batched product, not a sum over the key: no ReduceSum over no candidates.
            both = ((key(first) @ self.road_pair.to(keys.dtype)).unsqueeze(1) @ key(second).unsqueeze(2)).reshape(-1)
            played = played + torch.where((first >= 0) & (second >= 0), both.to(played.dtype) / KEY_WIDTH ** 0.5,
                                          torch.zeros_like(played))
        else:
            played = self.road_building(torch.cat([out.me, out.table], -1)).squeeze(-1)[rows]
            road = out.road[rows]
        total = played
        for column in range(2):
            edge = candidates[:, column].long()
            total = total + torch.where(edge >= 0, road.gather(1, edge.clamp(min=0)[:, None]).squeeze(1),
                                        torch.zeros_like(played))
        return total


# Every action type not scored off a board node, in `ActionType` order.
_GLOBAL_KINDS = tuple(
    kind for kind in ActionType
    if kind not in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD, ActionType.BUILD_ROAD,
                    ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.MOVE_ROBBER)
)


def _segment_logsumexp(scores: Tensor, rows: Tensor, batch: int) -> Tensor:
    """Per batch row, the log-sum-exp of the `scores` whose `rows` it is
    (`-inf` for a row with none, or with none finite). The log is taken of
    1 where the sum is 0, so such a row passes back a zero gradient: the log
    of 0 would pass back 0/0, which `index_add`'s backward gathers into the
    row's scores and the whole backward."""
    out = scores.new_full((batch,), float("-inf"))
    if scores.numel() == 0:
        return out
    top = out.scatter_reduce(0, rows, scores, reduce="amax", include_self=True)
    safe = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    total = scores.new_zeros(batch).index_add(0, rows, torch.exp(scores - safe[rows]))
    filled = total > 0
    return torch.where(filled, safe + torch.log(torch.where(filled, total, torch.ones_like(total))), out)


@dataclass
class Scores:
    """Every legal option of each row's decision, scored, in fp32, in the
    blocks `hexset.decisions.Decision` states it in: illegal entries are
    `-inf`. `double` holds the double-turn rows only (`double_rows`), each
    pair at `first * incidences + second`; `rb`/`disc` the candidates end to
    end beside their batch rows. `total` is each row's log-sum-exp over all of
    them: the softmax every option of a decision is drawn from."""

    flat: Tensor          # (B, space.size)
    setup: Tensor         # (B, incidences), single setup turns
    rb: Tensor            # (Kr,)
    rb_row: Tensor
    disc: Tensor          # (Kd,)
    disc_row: Tensor
    double_rows: Tensor   # (D,)
    double: Tensor        # (D, incidences ** 2)
    total: Tensor         # (B,)


def host_rows(net: HexNet, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """The rows `decision_scores` selects, found from a batch still on the
    host: `robber_rows` (rows with robber options), `setup_rows` (rows with
    setup options) and, with `double`, `double_within` (the double-turn rows
    among `setup_rows`, as positions in it). Exactly the indices it would find
    on the device. Passed in the batch, they spare the step a device-to-host
    read for each, which stalls the queue every time."""
    rows = {"robber_rows": batch["mask"][:, net.slot_robber_host].any(1).nonzero().squeeze(1),
            "setup_rows": batch["setup"].any(1).nonzero().squeeze(1)}
    if batch.get("double") is not None:
        rows["double_within"] = batch["double"][rows["setup_rows"]].nonzero().squeeze(1)
    return rows


def decision_scores(net: HexNet, out: Prediction, batch: dict[str, Tensor]) -> Scores:
    """`Scores` for a batch as `decision_log_prob` takes one. The robber and
    setup heads run only on the rows whose decision has such options. A batch
    without `double` has no double turn (self-play asks it in two setup
    turns), and the first, second and pair heads are not read. Those rows are
    the batch's `host_rows` where it carries them, else found here."""
    neg = float("-inf")
    rows = out.logits.shape[0]
    logits = out.logits.float()
    robbing = batch.get("robber_rows")
    if robbing is None:
        robbing = batch["mask"][:, net.slot_robber].any(1).nonzero().squeeze(1)
    if len(robbing):
        logits = logits.index_put((robbing[:, None], net.slot_robber[None, :]),
                                  net.robber_logits(out, robbing).float())
    setting = batch.get("setup_rows")
    if setting is None:
        setting = batch["setup"].any(1).nonzero().squeeze(1)
    count = batch["setup"].shape[1]
    setup_raw = logits.new_zeros(rows, count)
    double = batch.get("double")
    if len(setting) and double is None:
        setup_raw = setup_raw.index_put((setting,), net.setup_scores(out, setting).float())
    elif len(setting):
        setup_s, first_s, second_s, pair_s = net.setup_heads(out, setting)
        setup_raw = setup_raw.index_put((setting,), setup_s.float())
    single = batch["setup"] if double is None else batch["setup"] & ~double.unsqueeze(1)
    flat = logits.masked_fill(~batch["mask"], neg)
    setup = setup_raw.masked_fill(~single, neg)
    # A batch with no candidates of a kind has none to score. Without grad
    # (collection, most ticks) its head is not run; with grad it is, so its
    # parameters get the zero gradient they always had, not none.
    some = torch.is_grad_enabled()
    rb = (net.road_building_scores(out, batch["rb_row"], batch["rb"]).float() if some or len(batch["rb"])
          else logits.new_zeros(0))
    disc = (net.discard_scores(out, batch["disc_row"], batch["disc"]).float() if some or len(batch["disc"])
            else logits.new_zeros(0))
    parts = [torch.logsumexp(flat, 1), torch.logsumexp(setup, 1),
             _segment_logsumexp(rb, batch["rb_row"], rows), _segment_logsumexp(disc, batch["disc_row"], rows)]
    d = setting[:0]
    pairs = logits.new_zeros(0, count * count)
    if len(setting) and double is not None:
        within = batch.get("double_within")                          # double rows, as positions in `setting`
        if within is None:
            within = double[setting].nonzero().squeeze(1)
        if len(within):
            d = setting[within]
            legal = batch["setup"][d]
            ok = legal.unsqueeze(2) & legal.unsqueeze(1) & net.pairs_ok.unsqueeze(0)
            pair = pair_s.float()[within][:, net.inc_vertex][:, :, net.inc_vertex]
            pairs = first_s.float()[within].unsqueeze(2) + second_s.float()[within].unsqueeze(1) + pair
            pairs = pairs.masked_fill(~ok, neg).reshape(len(d), -1)
            parts.append(logits.new_full((rows,), neg).index_put((d,), torch.logsumexp(pairs, 1)))
    total = torch.logsumexp(torch.stack(parts, 1), 1)
    return Scores(flat, setup, rb, batch["rb_row"], disc, batch["disc_row"], d, pairs, total)


def chosen_scores(scores: Scores, batch: dict[str, Tensor]) -> Tensor:
    """Per row, the score of the decision taken (`batch["tkind"]`,
    `batch["y"]`, `batch["rb_index"]`/`batch["disc_index"]` each candidate's
    index within its row); `-inf` for one that is not legal."""
    neg = float("-inf")
    flat, setup = scores.flat, scores.setup
    tkind, y = batch["tkind"].long(), batch["y"].long()
    chosen = flat.new_full((flat.shape[0],), neg)
    chosen = torch.where(tkind == DecisionKind.FLAT, flat.gather(1, y.clamp(max=flat.shape[1] - 1)[:, None]).squeeze(1),
                         chosen)
    chosen = torch.where(tkind == DecisionKind.SETUP, setup.gather(1, y.clamp(max=setup.shape[1] - 1)[:, None]).squeeze(1),
                         chosen)
    # No boolean-mask reads, whose size the host would have to wait for: a row
    # takes at most one candidate, so the row's maximum over its taken
    # candidates (the rest -inf) is that candidate's score, and the gradient
    # reaches it alone.
    for kind, score, row, index in ((DecisionKind.ROAD_BUILDING, scores.rb, scores.rb_row, batch["rb_index"]),
                                    (DecisionKind.DISCARD, scores.disc, scores.disc_row, batch["disc_index"])):
        if score.numel():
            taken = (tkind[row] == kind) & (index == y[row])
            picked = chosen.new_full(chosen.shape, neg).scatter_reduce(
                0, row, torch.where(taken, score, torch.full_like(score, neg)), reduce="amax", include_self=True)
            chosen = torch.where(tkind == kind, picked, chosen)
    if len(scores.double_rows):
        d = scores.double_rows
        taken = tkind[d] == DecisionKind.DOUBLE
        picked = scores.double.gather(1, y[d].clamp(max=scores.double.shape[1] - 1)[:, None]).squeeze(1)
        chosen = chosen.index_put((d,), torch.where(taken, picked, chosen[d]))
    return chosen


def decision_entropy(scores: Scores) -> Tensor:
    """Per row, the entropy of the softmax over every legal option of its
    decision, `total - E[score]`, summed over the same blocks and segments
    `total` is taken over."""
    total = scores.total

    def expected(s: Tensor, at: Tensor) -> Tensor:              # p * s, 0 where illegal
        # Illegal entries are zeroed before anything reads them, exponent
        # included: a -inf in the branch `where` drops still sends NaN back
        # through its gradient, and so does an exponent that overflows there
        # (0 - at for a row whose log-sum-exp is below about -88.7).
        legal = torch.isfinite(s)
        safe = torch.where(legal, s, torch.zeros_like(s))
        gap = torch.where(legal, s - at, torch.zeros_like(s))
        return torch.where(legal, torch.exp(gap), torch.zeros_like(s)) * safe

    mean = expected(scores.flat, total[:, None]).sum(1) + expected(scores.setup, total[:, None]).sum(1)
    for s, row in ((scores.rb, scores.rb_row), (scores.disc, scores.disc_row)):
        if s.numel():
            mean = mean.index_add(0, row, expected(s, total[row]))
    if len(scores.double_rows):
        d = scores.double_rows
        mean = mean.index_add(0, d, expected(scores.double, total[d][:, None]).sum(1))
    return total - mean


def decision_log_prob(net: HexNet, out: Prediction, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """`(log_prob, hit)` per row: the log-probability of the decision taken
    (`batch["tkind"]`, `batch["y"]`) under the softmax over every legal option
    of the row's decision, and whether it is the highest-scored one.

    `batch` holds `mask` (B, flat) the ordinary actions, `setup` (B,
    incidences), `double` (B,), and the Road Building and discard candidates
    end to end, `rb` (Kr, 2) / `rb_row` (Kr,) and `disc` (Kd, 10) / `disc_row`
    (Kd,), each candidate's batch row beside it; `rb_index`/`disc_index` the
    candidate's index within its own row (`collate`). The robber and setup
    heads run only on the rows whose decision has such options."""
    scores = decision_scores(net, out, batch)
    chosen = chosen_scores(scores, batch)
    return chosen - scores.total, chosen >= _best(scores)


def first_move_prior(net: HexNet, flat: Tensor, setup: Tensor, rb: Tensor, rb_row: Tensor, disc: Tensor,
                     disc_row: Tensor, discards: Tensor, double: Tensor | None = None,
                     double_rows: Tensor | None = None) -> Tensor:
    """`(B, space.size)`: the softmax over every legal option of each row's
    decision, ordinary and compound together, marginalised onto the option's
    first atomic move, as a table asks the moves one at a time
    (`hexset.decisions.Plans`): a flat action keeps its own probability, a
    setup turn and the double turn go to their first settlement, every Road
    Building set to playing the card, and a discard, its cards given in an
    order drawn uniformly, to each kind it gives by the share of its cards
    that kind is. Zeros where nothing has a first move. A decision of flat
    actions alone (a position inside a decision among them) reads the softmax
    of its flat scores. The scores are each block's as `decision_scores`
    states them, `-inf` where illegal: `flat` `(B, space.size)`, `setup`
    `(B, incidences)`, `rb`/`disc` the candidates' beside their rows,
    `discards` `(Kd, 10)` the candidates themselves, and `double` `(D,
    incidences, incidences)` on `double_rows`. Traceable: an exported file
    answers it as `prior` (`hexn.export_onnx9`)."""
    space = net.space
    batch, size = flat.shape
    neg = flat.new_full((batch,), float("-inf"))
    tops = [flat.amax(1), setup.amax(1), neg.scatter_reduce(0, rb_row, rb, reduce="amax", include_self=True),
            neg.scatter_reduce(0, disc_row, disc, reduce="amax", include_self=True)]
    if double is not None:
        tops.append(neg.index_put((double_rows,), double.flatten(1).amax(1)))
    top = torch.stack(tops, 1).amax(1)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    settle = space.offsets[ActionType.SETUP_SETTLEMENT] + net.inc_vertex             # each incidence's settlement
    out = torch.exp(flat - top[:, None]).scatter_add(1, settle.expand(batch, -1), torch.exp(setup - top[:, None]))
    out = out.reshape(-1)
    if double is not None:
        first = torch.exp(double - top[double_rows][:, None, None]).sum(2)                   # (D, incidences)
        out = out.scatter_add(0, (double_rows[:, None] * size + settle[None, :]).reshape(-1), first.reshape(-1))
    card = space.index(Action(ActionType.PLAY_ROAD_BUILDING))
    out = out.scatter_add(0, rb_row * size + card, torch.exp(rb - top[rb_row]))
    given = discards[:, NUM_RESOURCES:].to(out.dtype)
    share = given / given.sum(1, keepdim=True).clamp(min=1)
    kinds = space.index(Action(ActionType.DISCARD)) + torch.arange(NUM_RESOURCES, device=out.device)
    out = out.scatter_add(0, (disc_row[:, None] * size + kinds[None, :]).reshape(-1),
                          (torch.exp(disc - top[disc_row])[:, None] * share).reshape(-1))
    out = out.reshape(batch, size)
    total = out.sum(1, keepdim=True)
    return torch.where(total > 0, out / total.clamp(min=1e-30), torch.zeros_like(out))


def option_probs(flat: Tensor, setup: Tensor, rb: Tensor, rb_row: Tensor, disc: Tensor, disc_row: Tensor,
                 double: Tensor | None = None, double_rows: Tensor | None = None) -> tuple[Tensor, ...]:
    """Every whole option's probability in its row's decision, block by
    block, as `hexset.decisions.OPTION_OUTPUTS` states them: the softmax over
    every legal option of the row, ordinary and compound together, the same
    distribution `first_move_prior` marginalises. The scores are as
    `first_move_prior` takes them, `-inf` where illegal. Returns `(flat (B,
    space.size), setup (B, incidences), rb (Kr,), disc (Kd,))`, and `double
    (D, incidences, incidences)` after them where `double` is given; zero where
    illegal, and on a row with nothing legal. Traceable: an exported file
    answers it (`hexn.export_onnx9`)."""
    batch = flat.shape[0]
    neg = flat.new_full((batch,), float("-inf"))
    tops = [flat.amax(1), setup.amax(1), neg.scatter_reduce(0, rb_row, rb, reduce="amax", include_self=True),
            neg.scatter_reduce(0, disc_row, disc, reduce="amax", include_self=True)]
    if double is not None:
        tops.append(neg.index_put((double_rows,), double.flatten(1).amax(1)))
    top = torch.stack(tops, 1).amax(1)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    e_flat = torch.exp(flat - top[:, None])
    e_setup = torch.exp(setup - top[:, None])
    e_rb = torch.exp(rb - top[rb_row])
    e_disc = torch.exp(disc - top[disc_row])
    zero = flat.new_zeros(batch)
    total = e_flat.sum(1) + e_setup.sum(1) + zero.scatter_add(0, rb_row, e_rb) + zero.scatter_add(0, disc_row, e_disc)
    e_double = None
    if double is not None:
        e_double = torch.exp(double - top[double_rows][:, None, None])
        total = total + zero.scatter_add(0, double_rows, e_double.flatten(1).sum(1))
    scale = torch.where(total > 0, 1.0 / total.clamp(min=1e-30), torch.zeros_like(total))
    out = (e_flat * scale[:, None], e_setup * scale[:, None], e_rb * scale[rb_row], e_disc * scale[disc_row])
    if e_double is not None:
        out += (e_double * scale[double_rows][:, None, None],)
    return out


def _best(scores: Scores) -> Tensor:
    rows = scores.flat.shape[0]
    best = [scores.flat.max(1).values, scores.setup.max(1).values,
            _segment_max(scores.rb, scores.rb_row, rows), _segment_max(scores.disc, scores.disc_row, rows)]
    if len(scores.double_rows):
        best.append(scores.flat.new_full((rows,), float("-inf")).index_put((scores.double_rows,),
                                                                          scores.double.max(1).values))
    return torch.stack(best, 1).max(1).values


def collate(decisions, kinds=None, indices=None) -> dict[str, np.ndarray]:
    """`hexset.decisions.Decision`s as one batch of `decision_log_prob`'s
    inputs, numpy: `mask`, `setup`, `double`, and the ragged candidates end to
    end with their rows and indices within the row. With `kinds` and
    `indices`, the decision each row took as `tkind` and `y`."""
    decisions = list(decisions)
    out = {
        "mask": np.stack([d.flat for d in decisions]),
        "setup": np.stack([d.setup for d in decisions]),
        "double": np.fromiter((d.double for d in decisions), dtype=np.bool_, count=len(decisions)),
    }
    for name, field, width in (("rb", "road_building", 2), ("disc", "discard", 10)):
        values = [getattr(d, field) for d in decisions]
        lengths = np.fromiter((len(v) for v in values), dtype=np.int64, count=len(values))
        out[name] = np.concatenate(values).reshape(-1, width).astype(np.int64)
        out[f"{name}_row"] = np.repeat(np.arange(len(values), dtype=np.int64), lengths)
        out[f"{name}_index"] = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths,
                                                                                        lengths)
    if kinds is not None:
        out["tkind"] = np.asarray(kinds, dtype=np.int64)
        out["y"] = np.asarray(indices, dtype=np.int64)
    return out


def _segment_max(scores: Tensor, rows: Tensor, batch: int) -> Tensor:
    out = scores.new_full((batch,), float("-inf"))
    if scores.numel() == 0:
        return out
    return out.scatter_reduce(0, rows, scores, reduce="amax", include_self=True)
