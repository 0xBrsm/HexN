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
- seat <-> seat by one attention over the four;
- the table token to every token, and back from the means of each kind.

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

import numpy as np
import torch
from torch import Tensor, nn

from hexset.actions import Action, ActionSpace, ActionType
from hexset.board.terrain import NUM_RESOURCES
from hexset.contract9 import (
    HEX_FEATURES, SEAT_FEATURES, TABLE_FEATURES, VERTEX_FEATURES, Layout9, incidences,
)
from hexset.decisions import DecisionKind, double_pairs

__all__ = ["DOUBLE_TURNS", "ModelConfig", "config_of", "HexNet", "Prediction", "Scores", "unpack9", "collate", "decision_scores",
           "chosen_scores", "decision_entropy", "decision_log_prob", "first_move_prior"]


# `ModelConfig.double_turn`'s choices.
DOUBLE_TURNS = ("joint", "parts")


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

    def __post_init__(self) -> None:
        for shape in (self.heads, self.discard):
            if shape not in ("linear", "mlp"):
                raise ValueError(f"unknown head shape {shape!r}")
        if self.messages not in ("separate", "pooled"):
            raise ValueError(f"unknown message layout {self.messages!r}")
        if self.double_turn not in DOUBLE_TURNS:
            raise ValueError(f"unknown double turn {self.double_turn!r}: {DOUBLE_TURNS}")


def config_of(model_config: dict) -> ModelConfig:
    """The `ModelConfig` a checkpoint's `model_config` names; one written
    before `heads` existed trained MLP heads."""
    return ModelConfig(width=model_config["width"], rounds=model_config["rounds"],
                         heads=model_config.get("heads", "mlp"), discard=model_config.get("discard", "mlp"),
                         norm=model_config.get("norm", False), messages=model_config.get("messages", "separate"),
                         attention=model_config.get("attention", True), points=model_config.get("points", False),
                         double_turn=model_config.get("double_turn", "joint"))


JOINT_HEADS = ("first_head.", "second_head.", "pair_head.")


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
    def __init__(self, width: int, norm: bool = True, pooled: bool = False, attention: bool = True) -> None:
        super().__init__()
        # Pre-norm: each kind of token is normalised before its messages are
        # built and its update read, and the update is added to the token as
        # it stood. Identity where the net is built without (`ModelConfig.norm`).
        self.norm_hex = _norm(width, norm)
        self.norm_vertex = _norm(width, norm)
        self.norm_seat = _norm(width, norm)
        self.norm_table = _norm(width, norm)
        self.pooled = pooled
        if pooled:
            # `ModelConfig.messages == "pooled"`: each kind of message pooled
            # into one input (below), and the table projected once per position
            # and added to every token's first layer rather than copied into
            # each token's input.
            self.hex = _mlp(2 * width, width)
            self.vertex = _mlp(4 * width, width)
            self.seat = _mlp((3 if attention else 2) * width, width)
            self.hex_table = nn.Linear(width, width, bias=False)
            self.vertex_table = nn.Linear(width, width, bias=False)
            self.seat_table = nn.Linear(width, width, bias=False)
            # What a building's owner's token carries beside a reach-weighted one.
            self.owned = nn.Parameter(torch.zeros(width))
        else:
            self.hex = _mlp(3 * width, width)
            self.vertex = _mlp(7 * width, width)
            self.seat = _mlp((6 if attention else 5) * width, width)
        self.table = _mlp(4 * width, width)
        self.attention = attention
        if attention:
            self.query = nn.Linear(width, width)
            self.key = nn.Linear(width, width)
            self.value = nn.Linear(width, width)
        self.road = nn.Parameter(torch.zeros(width))


def _pick(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` indexed by `index` along `dim`, as `x[..., index]` there. Without
    grad (collection) by `index_select`, the same copy at a fraction of
    advanced indexing's cost on CPU; with grad by indexing, whose backward
    (an accumulating `index_put`) the update has always run, deterministic on
    CUDA where `index_select`'s is not."""
    if torch.is_grad_enabled():
        return x[(slice(None),) * dim + (index,)]
    return x.index_select(dim, index.reshape(-1)).reshape(*x.shape[:dim], *index.shape, *x.shape[dim + 1:])


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
        self.players = players
        self.space = space
        width = self.config.width
        hexes, vertices, edges = topology.num_hexes, topology.num_vertices, topology.num_edges
        self.num_hexes, self.num_vertices, self.num_edges = hexes, vertices, edges

        self.embed_seat = nn.Linear(SEAT_FEATURES, width)
        self.embed_table = nn.Linear(TABLE_FEATURES, width)
        self.embed_hex = nn.Linear(HEX_FEATURES, width)
        self.embed_vertex = nn.Linear(VERTEX_FEATURES, width)
        self.rounds = nn.ModuleList(_Round(width, self.config.norm, self.config.messages == "pooled",
                                           self.config.attention)
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
        self.vertex_head = nn.Linear(width, 2)          # settlement, city
        self.road_head = head(2 * width)                # one ordered end pair
        self.robber_head = head(2 * width)              # hex with a victim's token
        self.nobody_head = nn.Linear(width, 1)          # hex, robbing nobody
        globals_size = sum(space.sizes[k] for k in _GLOBAL_KINDS)
        self.global_head = head(width, globals_size)           # off the table, as HexNet's global head
        self.setup_head = head(2 * width)               # settlement with its road's far end
        joint = self.config.double_turn == "joint"
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
        self.register_buffer("slot_setup_road", block(ActionType.SETUP_ROAD), persistent=False)
        self.register_buffer("slot_global", torch.cat([block(k) for k in _GLOBAL_KINDS]), persistent=False)

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
        nn.init.orthogonal_(output(self.value_head).weight, gain=1.0)
        if self.points_head is not None:
            nn.init.orthogonal_(output(self.points_head).weight, gain=1.0)

    def forward(self, seats: Tensor, table: Tensor, hexes: Tensor, vertices: Tensor,
                vertex_owner: Tensor, road_owner: Tensor, reach: Tensor, *, policy: bool = True) -> Prediction:
        """The win head, and with `policy` the policy's outputs: a value-only
        row (another seat's view of a decision) skips every policy head."""
        # Ownership as one comparison each: a pointer of -1 matches no seat.
        rows = torch.arange(self.players, device=seats.device)
        owns_v = (vertex_owner.unsqueeze(-1) == rows).to(seats.dtype)            # (B, V, P)
        owns_e = (road_owner.unsqueeze(-1) == rows).to(seats.dtype)              # (B, E, P)
        road = owns_e.sum(-1)                                                     # (B, E)
        built = owns_v.sum(1).clamp(min=1.0).unsqueeze(-1)                        # (B, P, 1)
        roads_held = owns_e.sum(1).clamp(min=1.0).unsqueeze(-1)
        reached = reach.sum(-1).clamp(min=1e-3).unsqueeze(-1)

        if self.fused:
            s, t, h, v = self._fused_trunk(seats, table, hexes, vertices, vertex_owner, road_owner, reach)
            h, v, s, t = self.out_hex(h), self.out_vertex(v), self.out_seat(s), self.out_table(t)
            value_logits, value, points = self._value(s, t)
            if not policy:
                return Prediction(value=value, value_logits=value_logits, points=points)
            return self._emit(s, t, h, v, value, value_logits, points)
        s = self.embed_seat(seats)
        t = self.embed_table(table)
        h = self.embed_hex(hexes)
        v = self.embed_vertex(vertices)
        scale = s.shape[-1] ** 0.5
        for r in self.rounds:
            if r.pooled:
                h, v, s, t = self._pooled_round(r, h, v, s, t, owns_v, owns_e, road, built, roads_held, reach,
                                                reached, scale)
                continue
            hn, vn, sn, tn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s), r.norm_table(t)
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
            if r.attention:
                attention = torch.softmax(r.query(sn) @ r.key(sn).transpose(1, 2) / scale, -1)
                seats_in = [attention @ r.value(sn)]
            else:
                seats_in = []

            h = h + r.hex(torch.cat([hn, h_from_v, tb.expand_as(h)], -1))
            v = v + r.vertex(torch.cat([vn, v_from_h, v_from_v, v_from_road, v_from_owner, v_from_reach,
                                        tb.expand_as(v)], -1))
            s = s + r.seat(torch.cat([sn, *seats_in, s_from_built, s_from_roads, s_from_reach,
                                      tb.expand_as(s)], -1))
            # The table reads each kind's mean token, normalised: the norm of
            # the mean, one row per kind, rather than every token normalised again.
            t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(1)), r.norm_hex(h.mean(1)),
                                       r.norm_vertex(v.mean(1))], -1))
        h, v, s, t = self.out_hex(h), self.out_vertex(v), self.out_seat(s), self.out_table(t)
        value_logits, value, points = self._value(s, t)
        if not policy:
            return Prediction(value=value, value_logits=value_logits, points=points)
        return self._emit(s, t, h, v, value, value_logits, points)

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

        s = self.embed_seat(seats).transpose(0, 1)                                # (P, B, W)
        t = self.embed_table(table)                                               # (B, W)
        h = self.embed_hex(hexes).transpose(0, 1)                                 # (H, B, W)
        v = self.embed_vertex(vertices).transpose(0, 1)                           # (V, B, W)
        scale = width ** 0.5
        for r in self.rounds:
            hn, vn, sn, tn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s), r.norm_table(t)
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
                seats_in = [(attention.unsqueeze(-1) * val.unsqueeze(0)).sum(1)]  # (P, B, W)
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
            t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(0)), r.norm_hex(h.mean(0)), r.norm_vertex(v.mean(0))], -1))
        return (s.transpose(0, 1).contiguous(), t, h.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous())

    def _pooled_round(self, r: _Round, h, v, s, t, owns_v, owns_e, road, built, roads_held, reach, reached, scale):
        """One round with each kind of message pooled into one input."""
        hn, vn, sn, tn = r.norm_hex(h), r.norm_vertex(v), r.norm_seat(s), r.norm_table(t)
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
            seats_in = [attention @ r.value(sn)]

        def update(mlp: nn.Sequential, table: nn.Linear, x: Tensor) -> Tensor:
            return mlp[2](mlp[1](mlp[0](x) + table(tn).unsqueeze(1)))

        h = h + update(r.hex, r.hex_table, torch.cat([hn, h_from_v], -1))
        v = v + update(r.vertex, r.vertex_table, torch.cat([vn, v_from_h, v_edges, v_seats], -1))
        s = s + update(r.seat, r.seat_table, torch.cat([sn, *seats_in, s_board], -1))
        t = t + r.table(torch.cat([tn, r.norm_seat(s.mean(1)), r.norm_hex(h.mean(1)), r.norm_vertex(v.mean(1))], -1))
        return h, v, s, t

    def _value(self, s: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """The win head's logits and softmax over seats, and the points head's
        number per seat where the net has one: each seat token read with the table."""
        read = torch.cat([s, t.unsqueeze(1).expand_as(s)], -1)
        value_logits = self.value_head(read).squeeze(-1)
        points = None if self.points_head is None else self.points_head(read).squeeze(-1).float()
        return value_logits, torch.softmax(value_logits, -1), points

    def _emit(self, s: Tensor, t: Tensor, h: Tensor, v: Tensor, value: Tensor, value_logits: Tensor,
              points: Tensor | None = None) -> Prediction:
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
        logits = logits.index_copy(1, self.slot_global, self.global_head(t).to(vertex.dtype))
        return Prediction(value=value, value_logits=value_logits, points=points, logits=logits, road=road, me=me,
                          table=t, vertices=v, hexes=h, seats=s)

    def robber_logits(self, out: Prediction, rows: Tensor) -> Tensor:
        """`(len(rows), hexes * (players + 1))`: the robber block of `rows`'
        flat logits, each hex with each victim's seat token, then nobody."""
        uh = out.hexes[rows]
        s = out.seats[rows]
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
        pair `(V, V)` the double turn's pair term. A "parts" net has none of the three."""
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
        setup turn asked whole is scored by `setup_scores`."""
        logits = self.full_logits(out)
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
        alone: `setup_heads`' first output, without the double turn's heads."""
        u = out.vertices[rows]
        return self.setup_head(torch.cat([u[:, self.inc_vertex], u[:, self.inc_far]], -1)).squeeze(-1)

    def discard_scores(self, out: Prediction, rows: Tensor, candidates: Tensor) -> Tensor:
        """One score per discard candidate: `candidates` `(K, 10)` (kept, then
        given, card counts), `rows` `(K,)` the batch row each belongs to."""
        counts = candidates.to(out.me.dtype) / 10.0
        if self.config.discard == "linear":
            weights = self.discard_head(torch.cat([out.me, out.table], -1))[rows]       # (K, 10)
            return (weights * counts).sum(-1)
        return self.discard_head(torch.cat([out.me[rows], out.table[rows], counts], -1)).squeeze(-1)

    def road_building_scores(self, out: Prediction, rows: Tensor, candidates: Tensor) -> Tensor:
        """One score per Road Building candidate: `candidates` `(K, 2)` edges,
        `-1` for a road not placed; the card's own term plus each road's."""
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
    (`-inf` for a row with none)."""
    out = scores.new_full((batch,), float("-inf"))
    if scores.numel() == 0:
        return out
    top = out.scatter_reduce(0, rows, scores, reduce="amax", include_self=True)
    safe = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    total = scores.new_zeros(batch).index_add(0, rows, torch.exp(scores - safe[rows]))
    return torch.where(total > 0, safe + torch.log(total), out)


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


def decision_scores(net: HexNet, out: Prediction, batch: dict[str, Tensor]) -> Scores:
    """`Scores` for a batch as `decision_log_prob` takes one. The robber and
    setup heads run only on the rows whose decision has such options. A batch
    without `double` has no double turn (self-play asks it in two setup
    turns), and the first, second and pair heads are not read."""
    neg = float("-inf")
    rows = out.logits.shape[0]
    logits = out.logits.float()
    robbing = batch["mask"][:, net.slot_robber].any(1).nonzero().squeeze(1)
    if len(robbing):
        logits = logits.index_put((robbing[:, None], net.slot_robber[None, :]),
                                  net.robber_logits(out, robbing).float())
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
        within = double[setting].nonzero().squeeze(1)               # double rows, as positions in `setting`
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


def decision_entropy(scores: Scores) -> Tensor:
    """Per row, the entropy of the softmax over every legal option of its
    decision, `total - E[score]`, summed over the same blocks and segments
    `total` is taken over."""
    total = scores.total

    def expected(s: Tensor, at: Tensor) -> Tensor:              # p * s, 0 where illegal
        # Illegal entries are zeroed before anything reads them: a -inf in
        # the branch `where` drops still sends NaN back through its gradient.
        legal = torch.isfinite(s)
        safe = torch.where(legal, s, torch.zeros_like(s))
        return torch.where(legal, torch.exp(safe - at), torch.zeros_like(s)) * safe

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
