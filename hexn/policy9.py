# SPDX-License-Identifier: GPL-3.0-only
"""`Policy9`: a contract-9 `hexn.model9.HexNet` as a
`hexset.clients.policy.Policy`, so `hexset.clients.netbot` plays it, gates
its trades and seats it in the arena like any other checkpoint.

The bot asks for one atomic move at a time (`act_rows` over
`hexset.actions.options_for`). Where a move opens a compound decision (a setup
turn, the snake seat's double turn, a Road Building card, a discard,
`hexset.decisions`), the policy chooses the whole decision there, at the
position before its first move, returns that first move, and keeps the rest as
the seat's plan (`hexset.decisions.Plans`): each later ask inside the decision
gets the plan's next move. A position inside a decision with no plan of its
own (one a gate imagined, or a game resumed part-way) is asked as a decision
of its legal moves alone (`hexset.decisions.atomic`): its best-scored road.

Greedy: the highest-scored option of the decision (`best` over its blocks,
`Scored`), a tie to the earlier block and then the lower index. An exported
file chooses with the same rule inside its graph (`hexn.export_onnx9`), so a
file and the net it was exported from play alike. A net whose
`ModelConfig.double_turn` is "parts" (one PPO trained) chooses the snake
seat's double turn as two setup turns, the setup head read once on the board
before the first and once after it; "joint" (contract-9 BC's) chooses the pair
at once with the double turn's heads. `score_rows` reads the decision's
probability marginalised onto its first atomic move, as a file's `prior`
does (`hexn.model9.first_move_prior`), so a search a move at a time sees
every whole option; `option_rows` lists every whole option of a decision with its
probability, for a caller that searches over whole decisions.

`NetworkPolicy9` is the same net behind `hexn.selfplay.BatchPolicy` for
self-play: one forward per tick over every learner decision the lanes ask for
whole (`hexset.gym.lanes.LaneEnv`'s `decisions`), a draw over every legal
option of each decision, ordinary and compound together, and the update's
recomputation of the same log-probability (`evaluate`). Each tick is one
part of a run of options (`hexset.decisions.closes`), which the update
credits as one decision; the lanes ask the double turn as two setup turns, so
self-play never reads the double turn's heads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from hexset import contract9, decisions as D
from torch import Tensor

from hexset.actions import Action, ActionSpace, ActionType
from hexset.decisions import option_of
from hexset.encoding import from_frame
from hexset.game import Game, Phase, pending_free_roads
from hexset.clients.batching import coalesce

from .model9 import (HexNet, Scores, chosen_scores, collate, decision_entropy, decision_scores, first_move_prior,
                     unpack9)
from .policy import Evaluation, keyed_gumbel, keyed_gumbel_at, PRECISIONS
from .selfplay import Choice, Request

__all__ = ["Policy9", "NetworkPolicy9", "Heads9", "Scored", "ranked", "best", "option_of"]


# --- choosing a whole decision ----------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """A score for every option of one `hexset.decisions.Decision`, in its
    blocks: higher is better, and only a legal entry is read. `flat` is one
    per action of the flat space (robber victims seat-relative), read where
    `decision.flat`; `setup` one per incidence, read where `decision.setup`
    and the decision is not `double`; `double` the double turn's ordered
    pairs, first by second `(incidences, incidences)`, read where both turns
    are legal and `hexset.decisions.double_pairs` allows them, `None` for a
    decision that is not `double`; `road_building` and `discard` one per
    candidate row of the decision's."""

    flat: np.ndarray
    setup: np.ndarray
    road_building: np.ndarray
    discard: np.ndarray
    double: np.ndarray | None = None


_DOUBLE_OK: dict[int, tuple[object, np.ndarray]] = {}


def _double_ok(topology) -> np.ndarray:
    """`hexset.decisions.double_pairs(topology)`, worked out once per topology
    (by identity, as `hexset.contract9.incidences` keeps its own)."""
    held = _DOUBLE_OK.get(id(topology))
    if held is not None and held[0] is topology:
        return held[1]
    out = D.double_pairs(topology)
    out.flags.writeable = False
    if len(_DOUBLE_OK) >= 64:
        _DOUBLE_OK.pop(next(iter(_DOUBLE_OK)))
    _DOUBLE_OK[id(topology)] = (topology, out)
    return out


def _blocks(decision: D.Decision, scored: Scored, topology) -> list[tuple[D.DecisionKind, np.ndarray]]:
    """Each block's scores with every illegal entry at `-inf`, in block
    order: flat, then the setup turn or the double turn, Road Building, the
    discard."""
    neg = -np.inf
    out = [(D.DecisionKind.FLAT, np.where(decision.flat, scored.flat, neg))]
    if decision.double:
        if scored.double is None:
            raise ValueError("a double turn's decision needs its pairs scored")
        legal = decision.setup[:, None] & decision.setup[None, :] & _double_ok(topology)
        out.append((D.DecisionKind.DOUBLE, np.where(legal, scored.double, neg).reshape(-1)))
    else:
        out.append((D.DecisionKind.SETUP, np.where(decision.setup, scored.setup, neg)))
    out.append((D.DecisionKind.ROAD_BUILDING, np.asarray(scored.road_building).reshape(-1)))
    out.append((D.DecisionKind.DISCARD, np.asarray(scored.discard).reshape(-1)))
    for (kind, scores), rows in zip(out[2:], (decision.road_building, decision.discard)):
        if len(scores) != len(rows):
            raise ValueError(f"{kind.name}: {len(scores)} scores for {len(rows)} candidates")
    return out


def ranked(decision: D.Decision, scored: Scored, topology) -> list[tuple[float, D.DecisionKind, int]]:
    """Every legal option of `decision` as `(score, kind, index)`, in block
    order (flat, the setup or double turn, Road Building, the discard) and
    index order within a block. `topology` is the board's."""
    out = []
    for kind, scores in _blocks(decision, scored, topology):
        for index in np.flatnonzero(np.isfinite(scores)):
            out.append((float(scores[index]), kind, int(index)))
    return out


def best(decision: D.Decision, scored: Scored, topology) -> tuple[D.DecisionKind, int]:
    """The highest-scored legal option of `decision`, as `(kind, index)`. A
    tie goes to the earlier block (flat, the setup or double turn, Road
    Building, the discard), then to the lower index. Raises where the
    decision has no legal option."""
    top, chosen = -np.inf, None
    for kind, scores in _blocks(decision, scored, topology):
        if not len(scores):
            continue
        index = int(np.argmax(scores))
        if scores[index] > top:
            top, chosen = float(scores[index]), (kind, index)
    if chosen is None:
        raise RuntimeError("a decision with no legal option")
    return chosen


@dataclass(frozen=True)
class Heads9:
    """One forward's numbers, numpy, each row in its own frame (row 0 the seat
    the observation is from): `value` the win head `(B, players)`, `points`
    the points head (`None` for a net without one), and, asked for a
    decision's arrays, `flat` `(B, space.size)` (`HexNet.flat_scores`),
    `setup` `(B, incidences)`, one score per Road Building and discard
    candidate in the order asked, `double` `(D, incidences, incidences)`
    for the double-turn rows asked (`None` for a "parts" net), and `prior`
    `(B, space.size)` where asked (`hexn.model9.first_move_prior`)."""

    value: np.ndarray
    points: np.ndarray | None = None
    flat: np.ndarray | None = None
    setup: np.ndarray | None = None
    road_building: np.ndarray | None = None
    discard: np.ndarray | None = None
    double: np.ndarray | None = None
    prior: np.ndarray | None = None


def _inside(game: Game) -> bool:
    """Whether `game` stands inside a decision begun a move at a time (a setup
    road or a free road owed), which `hexset.decisions.tensors` refuses."""
    return game.phase is Phase.SETUP_ROAD or bool(pending_free_roads(game))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


class Policy9:
    """A contract-9 net as a `hexset.clients.policy.Policy` (the module
    docstring)."""

    def __init__(self, net: HexNet, space: ActionSpace, *, device: str = "cpu") -> None:
        self.net = net.eval()
        self.space = space
        self.device = device
        self.players = net.players
        self.layout = contract9.layout(net.players, net.num_hexes, net.num_vertices, net.num_edges, net.CONTRACT)
        self.joint = net.config.double_turn == "joint"
        self._plans = D.Plans()

    # --- the forward -------------------------------------------------------------

    def observations(self, rows: Sequence[tuple[Game, int]]) -> np.ndarray:
        """Each `(game, seat)` as its contract-9 observation row, stacked."""
        return np.stack([contract9.flatten(contract9.encode9(game, seat, self.net.CONTRACT)) for game, seat in rows])

    def _forward(self, games: Sequence[Game], seats: Sequence[int], *, policy: bool):
        """The net's `Prediction` on these positions' observations."""
        x = self.observations(list(zip(games, seats)))
        with torch.no_grad():
            return self.net(**unpack9(self.layout, torch.from_numpy(x).to(self.device)), policy=policy)

    def _heads(self, observations: np.ndarray, arrays: dict[str, np.ndarray] | None, *,
               prior: bool = False) -> Heads9:
        """The heads on `observations`: the win and points heads alone where
        `arrays` is `None`; with a batch of decisions' arrays
        (`hexset.decisions.arrays`), every score `Heads9` holds for them, and
        the `prior` if asked."""
        net = self.net
        grouped = coalesce({"observation": observations, **(arrays or {})}) if str(self.device) == "cpu" else None
        if grouped is not None:
            observations = grouped.feed["observation"]
            if arrays is not None:
                arrays = {name: grouped.feed[name] for name in arrays}

        def restore(heads: Heads9) -> Heads9:
            if grouped is None:
                return heads
            parts = {}
            for name in Heads9.__dataclass_fields__:
                values = getattr(heads, name)
                indices = grouped.candidates.get("double_rows" if name == "double" else name, grouped.inverse)
                parts[name] = None if values is None else values[indices]
            return Heads9(**parts)

        def numpy(t: Tensor) -> np.ndarray:
            return t.float().cpu().numpy()

        def tensor(a: np.ndarray) -> Tensor:
            return torch.from_numpy(a).to(self.device)

        with torch.no_grad():
            out = net(**unpack9(self.layout, tensor(observations)), policy=arrays is not None)
            value, points = numpy(out.value), None if out.points is None else numpy(out.points)
            if arrays is None:
                return restore(Heads9(value=value, points=points))
            rows = torch.arange(len(observations), device=self.device)
            rb_row, disc_row = tensor(arrays["road_building_row"]), tensor(arrays["discard_row"])
            rb, disc, double = tensor(arrays["road_building"]), tensor(arrays["discard"]), tensor(arrays["double_rows"])
            flat, setup = net.flat_scores(out).float(), net.setup_scores(out, rows).float()
            rb_s = net.road_building_scores(out, rb_row, rb).float() if len(rb) else flat.new_zeros(0)
            disc_s = net.discard_scores(out, disc_row, disc).float() if len(disc) else flat.new_zeros(0)
            pairs = None if not self.joint else (
                net.double_scores(out, double).float() if len(double)
                else flat.new_zeros((0, len(net.inc_vertex), len(net.inc_vertex))))
            marginal = None
            if prior:
                # Each block's legal scores, `-inf` elsewhere, as `hexn.export_onnx9`'s graph masks them.
                legal, neg = tensor(arrays["setup"]), float("-inf")
                single = legal.index_put((double,), torch.tensor(False, device=self.device))
                masked = None
                if pairs is not None:
                    both = legal[double].unsqueeze(2) & legal[double].unsqueeze(1) & net.pairs_ok.unsqueeze(0)
                    masked = pairs.masked_fill(~both, neg)
                marginal = numpy(first_move_prior(
                    net, flat.masked_fill(~tensor(arrays["flat"]), neg), setup.masked_fill(~single, neg),
                    rb_s, rb_row, disc_s, disc_row, disc, masked, double))
            return restore(Heads9(value=value, points=points, flat=numpy(flat), setup=numpy(setup),
                          road_building=numpy(rb_s), discard=numpy(disc_s),
                          double=None if pairs is None else numpy(pairs), prior=marginal))

    def decision(self, game: Game, seat: int) -> D.Decision:
        """`seat`'s decision at `game` as this net is asked it: a position
        inside a decision as its legal moves alone."""
        if _inside(game):
            return D.atomic(game, seat, self.space)
        return D.tensors(game, seat, self.space, double_turn=self.joint)

    def _scored(self, rows: Sequence[tuple[Game, int]], decisions=None, *,
                prior: bool = False) -> tuple[list, list, Heads9]:
        """One forward over `rows`: each row's `Decision` (`decision`, unless
        given) and its `Scored`, and the heads (with the `prior` if asked)."""
        decisions = [self.decision(game, seat) for game, seat in rows] if decisions is None else list(decisions)
        arrays = D.arrays(decisions, self.space, rows[0][0].state(rows[0][1], hidden=False).board.topology)
        heads = self._heads(self.observations(rows), arrays, prior=prior)
        rb_row, disc_row = arrays["road_building_row"], arrays["discard_row"]
        double_at = {int(row): i for i, row in enumerate(arrays["double_rows"])}
        scored = []
        for row, decision in enumerate(decisions):
            scored.append(None if decision is None else Scored(
                flat=heads.flat[row], setup=heads.setup[row],
                road_building=heads.road_building[rb_row == row], discard=heads.discard[disc_row == row],
                double=heads.double[double_at[row]] if row in double_at else None,
            ))
        return decisions, scored, heads

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """The win head, one vector per row, in board-seat order."""
        if not rows:
            return []
        heads = self._heads(self.observations(rows), None)
        return [tuple(float(v) for v in from_frame(heads.value[i], seat)) for i, (_, seat) in enumerate(rows)]

    def points_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """The points head, one vector per row, in board-seat order. Raises
        for a net without one."""
        if not rows:
            return []
        heads = self._heads(self.observations(rows), None)
        if heads.points is None:
            raise ValueError("this contract-9 net has no points head")
        return [tuple(float(v) for v in from_frame(heads.points[i], seat)) for i, (_, seat) in enumerate(rows)]

    # --- choosing ------------------------------------------------------------------

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        """Ask the net once for rows needing a decision, then consume plans
        in row order. A plan evicted during the batch is recomputed where
        needed, just as when asking one row at a time."""
        ask = [i for i, (game, seat, options) in enumerate(rows)
               if len(options) > 1 and not self._plans.planned(game, seat, options)]
        chosen = dict(zip(ask, self._choose_many([rows[i][:2] for i in ask])))
        out = []
        for i, (game, seat, options) in enumerate(rows):
            choose = (lambda g, s, i=i: chosen.pop(i)) if i in chosen else self._choose
            out.append(self._plans.act(game, seat, options, choose, partial(self._finish, choose=choose)))
        return out

    def _choose(self, game: Game, seat: int):
        (choice,) = self._choose_many([(game, seat)])
        return choice

    def _choose_many(self, rows: Sequence[tuple[Game, int]]) -> list[D.Option]:
        """The highest-scored decision per row, from one network forward."""
        if not rows:
            return []
        decisions, scored, _ = self._scored(rows)
        out = []
        for (game, seat), decision, scores in zip(rows, decisions, scored):
            topology = game.state(seat, hidden=False).board.topology
            kind, index = best(decision, scores, topology)
            out.append(option_of(decision, kind, index, contract9.incidences(topology),
                                 seat, self.space, self.players))
        return out

    def _decide(self, game: Game, seat: int):
        """The highest-scored decision of `seat`'s at this position."""
        return self._choose(game, seat)

    def _finish(self, game: Game, seat: int, options: tuple[Action, ...], *, choose=None) -> Action:
        """A move inside a decision begun without this policy's plan: the
        best-scored of its legal moves (a road, by the road head)."""
        choice = (self._choose if choose is None else choose)(game, seat)
        if choice not in options:
            raise RuntimeError(f"contract-9 policy chose {choice}, not among {options}")
        return choice

    def score_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]):
        """`(priors, values)` per row: the decision's probability marginalised
        onto its first atomic move (`hexn.model9.first_move_prior`), at each
        option's flat action, renormalised over the options (uniform where
        none has mass), and the win head in board-seat order -- what an
        exported file's `prior` gives."""
        if not rows:
            return []
        _, _, heads = self._scored([(game, seat) for game, seat, _ in rows], prior=True)
        out = []
        for i, (game, seat, options) in enumerate(rows):
            weights = heads.prior[i][[D.flat_index(self.space, o, seat, self.players) for o in options]].astype(
                np.float64)
            total = weights.sum()
            out.append((weights / total if total > 0 else np.full(len(options), 1.0 / len(options)),
                        tuple(float(v) for v in from_frame(heads.value[i], seat))))
        return out

    def option_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[tuple[D.Option, ...], np.ndarray]]:
        """Every whole option of each row's decision and its probability, the
        softmax over all of them, in block order (flat by index, the setup or
        double turn, Road Building, the discard): what `act_rows` chooses the
        most probable of. A position inside a decision lists its legal moves."""
        if not rows:
            return []
        decisions, scored, _ = self._scored(rows)
        out = []
        for i, (game, seat) in enumerate(rows):
            topology = game.state(seat, hidden=False).board.topology
            pairs = contract9.incidences(topology)
            items = ranked(decisions[i], scored[i], topology)
            options = tuple(option_of(decisions[i], kind, index, pairs, seat, self.space, self.players)
                            for _, kind, index in items)
            out.append((options, _softmax(np.array([score for score, _, _ in items]))))
        return out

    def answers(self, observations: np.ndarray, arrays: dict[str, np.ndarray],
                topology) -> dict[str, np.ndarray]:
        """What an exported graph answers for this feed (`hexn.export_onnx9`),
        computed on the torch net: `decision` `(B, 2)` (`best` per row, `(0,
        0)` for a row with nothing legal), `value`, `prior`
        (`hexn.model9.first_move_prior`) and, for a net with one, `points`.
        For checking a file against the net it was exported from."""
        decisions = _decisions_of(arrays, self.space)
        heads = self._heads(observations, arrays, prior=True)
        rb_row, disc_row = arrays["road_building_row"], arrays["discard_row"]
        double_at = {int(row): i for i, row in enumerate(arrays["double_rows"])}
        chosen = np.zeros((len(observations), 2), dtype=np.int64)
        for row, decision in enumerate(decisions):
            scored = Scored(flat=heads.flat[row], setup=heads.setup[row],
                            road_building=heads.road_building[rb_row == row], discard=heads.discard[disc_row == row],
                            double=heads.double[double_at[row]] if row in double_at else None)
            try:
                chosen[row] = best(decision, scored, topology)
            except RuntimeError:
                pass
        out = {"decision": chosen, "value": heads.value, "prior": heads.prior}
        if heads.points is not None:
            out["points"] = heads.points
        return out


def _decisions_of(arrays: dict[str, np.ndarray], space: ActionSpace) -> list[D.Decision]:
    """The `Decision`s a batch of `hexset.decisions.arrays` was stated from."""
    rows = len(arrays["flat"])
    double = set(int(r) for r in arrays["double_rows"])
    out = []
    for row in range(rows):
        rb = arrays["road_building"][arrays["road_building_row"] == row].astype(np.int16).reshape(-1, 2)
        disc = arrays["discard"][arrays["discard_row"] == row].astype(np.int8).reshape(-1, 10)
        out.append(D.Decision(arrays["flat"][row], arrays["setup"][row], row in double, rb, disc))
    return out


# --- self-play -------------------------------------------------------------------------


def _block_argmax(scores: np.ndarray, rows: np.ndarray, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Per batch row, the best of the `scores` whose `rows` it is and that
    entry's index among them (`-inf`, -1 for a row with none)."""
    best = np.full(batch, -np.inf)
    at = np.full(batch, -1, dtype=np.int64)
    if len(scores):
        order = np.lexsort((-scores, rows))                       # by row, the best first
        first = np.flatnonzero(np.r_[True, rows[order][1:] != rows[order][:-1]])
        top = order[first]
        best[rows[top]] = scores[top]
        at[rows[top]] = top
    return best, at


class NetworkPolicy9:
    """A contract-9 `hexn.model9.HexNet` behind `hexn.selfplay.BatchPolicy`.

    `act` makes one forward for the whole tick and scores every legal option
    of every decision with `hexn.model9.decision_scores`, the function the
    update's `evaluate` calls, so the recorded log-probability is the one the
    update recomputes. The draw is Gumbel-max over those scores: an option's
    noise is keyed on the request's `draw` key and the option's slot, the
    flat action's index, then the setup turns and the Road Building or
    discard candidates, each block after the last
    (`slots`). A request without a key takes one from `generator`. `greedy`
    takes the best-scored option. Every seat plays at temperature 1.
    `precision` (`hexn.policy.PRECISIONS`) is the autocast the forward runs
    under: "fp32" none; "bf16" a bf16 trunk and heads, whose scores
    `decision_scores` hands back in fp32, so the recorded log-probability is
    the fp32 log-softmax of the scores the bf16 forward produced -- the
    behaviour policy, exactly, as a contract-8 worker's (`hexn.policy`). A
    double-turn decision (`Decision.double`) is refused: the lanes ask it as
    two setup turns."""

    def __init__(self, net: HexNet, space: ActionSpace, *, device: torch.device | str = "cpu", greedy: bool = False,
                 generator: torch.Generator | None = None) -> None:
        self.net = net
        self.space = space
        self.device = torch.device(device)
        self.greedy = greedy
        self.generator = generator
        self.precision = "fp32"
        self.layout = contract9.layout(net.players, net.num_hexes, net.num_vertices, net.num_edges, net.CONTRACT)
        self.incidences = int(net.inc_vertex.shape[0])
        self._pairs = None

    @property
    def players(self) -> int:
        return int(self.space.num_players)

    def slots(self) -> tuple[int, int]:
        """Where the setup and candidate blocks' slots start."""
        flat, count = self.space.size, self.incidences
        return flat, flat + count

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        if not requests:
            return []
        if any(r.temperature != 1.0 for r in requests):
            raise ValueError("a contract-9 policy samples at temperature 1")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {tuple(PRECISIONS)}, got {self.precision!r}")
        dtype = PRECISIONS[self.precision]
        draws = [r.draw for r in requests]
        if any((d is None) != (draws[0] is None) for d in draws):
            raise ValueError("a batch mixes keyed and unkeyed requests")
        if draws[0] is None and not self.greedy:
            draws = torch.randint(0, 2**62, (len(requests),), generator=self.generator).tolist()
        if any(r.decision.double for r in requests):
            raise ValueError("a double turn is two setup turns in self-play (hexset.gym.lanes)")
        x = torch.from_numpy(np.stack([r.observation for r in requests])).to(self.device)
        stated = collate([r.decision for r in requests])
        del stated["double"]
        batch = {k: torch.from_numpy(v).to(self.device) for k, v in stated.items()}
        # Inference mode, not just no grad: nothing here is kept for autograd,
        # and the dispatcher's bookkeeping is a share of a 30-row forward.
        # The forward and the on-demand heads `decision_scores` runs under one
        # autocast; the scores come back in fp32 and the draw and the recorded
        # log-probability are taken on those.
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=dtype, enabled=dtype is not None):
            out = self.net(**unpack9(self.layout, x))
            scores = decision_scores(self.net, out, batch)
            kinds, indices = self._draw(scores, batch, None if self.greedy else draws)
            batch["tkind"] = torch.from_numpy(kinds).to(self.device)
            batch["y"] = torch.from_numpy(indices).to(self.device)
            log_prob = (chosen_scores(scores, batch) - scores.total).cpu().numpy()
            value = out.value.float().cpu().numpy()
            points = None if out.points is None else out.points.float().cpu().numpy()
        if self._pairs is None:
            # The net's buffers are one topology's, so every board it plays has it.
            self._pairs = contract9.incidences(requests[0].game.state(requests[0].seat, hidden=False).board.topology)
        out_choices = []
        for row, request in enumerate(requests):
            kind, index = int(kinds[row]), int(indices[row])
            out_choices.append(Choice(
                action=option_of(request.decision, kind, index, self._pairs, request.seat, self.space,
                                 self.players),
                log_prob=float(log_prob[row]),
                value=tuple(value[row].tolist()),
                decision=(kind, index),
                points=() if points is None else tuple(points[row].tolist()),
            ))
        return out_choices

    def _draw(self, scores: Scores, batch: dict, keys: Sequence[int] | None) -> tuple[np.ndarray, np.ndarray]:
        """Each row's option: `(DecisionKind, index)` per row, as numpy."""
        rows = scores.flat.shape[0]
        setup_at, candidate_at = self.slots()
        key = None if keys is None else np.asarray(keys, dtype=np.uint64)
        flat = scores.flat.double().cpu().numpy()
        setup = scores.setup.double().cpu().numpy()
        if key is not None:
            # Noise only where an option is legal: an illegal entry is -inf
            # whatever is added, and each draw is a pure function of its key
            # and slot, so the legal entries get the noise they always had.
            legal = np.nonzero(np.isfinite(flat))
            flat[legal] += keyed_gumbel_at(key[legal[0]], legal[1].astype(np.uint64))
            setting = np.flatnonzero(np.isfinite(setup).any(1))
            if len(setting):
                setup[setting] += keyed_gumbel(key[setting], setup.shape[1], setup_at)
        best = [(flat.max(1), flat.argmax(1)), (setup.max(1), setup.argmax(1))]
        kinds = [D.DecisionKind.FLAT, D.DecisionKind.SETUP]
        for name, kind in (("rb", D.DecisionKind.ROAD_BUILDING), ("disc", D.DecisionKind.DISCARD)):
            values = getattr(scores, name).double().cpu().numpy()
            row_of = batch[f"{name}_row"].cpu().numpy()
            within = batch[f"{name}_index"].cpu().numpy()
            if key is not None and len(values):
                values = values + keyed_gumbel_at(key[row_of], candidate_at + within)
            top, at = _block_argmax(values, row_of, rows)
            best.append((top, np.where(at >= 0, within[np.maximum(at, 0)], -1) if len(values) else at))
            kinds.append(kind)
        tops = np.stack([b[0] for b in best], 1)
        if not np.isfinite(tops.max(1)).all():
            raise RuntimeError("a contract-9 decision with no legal option")
        block = tops.argmax(1)
        chosen_kind = np.asarray(kinds, dtype=np.int64)[block]
        chosen_index = np.stack([b[1] for b in best], 1)[np.arange(rows), block]
        return chosen_kind, chosen_index.astype(np.int64)

    def evaluate(self, buffer: Tensor, mask: Tensor, chosen: Tensor, decisions: dict[str, Tensor]) -> Evaluation:
        """A stored batch's log-probabilities, entropies and values, with grad:
        `act`'s own arithmetic on the same rows. `decisions` is the rest of
        `hexn.model9.collate`'s batch (`hexn.ppo.decision_rows`), `chosen`
        the index each row took within its kind. Every number is back in fp32
        before it is normalised (`hexn.model9.decision_scores`)."""
        out = self.net(**unpack9(self.layout, buffer))
        batch = {**decisions, "mask": mask, "y": chosen}
        scores = decision_scores(self.net, out, batch)
        return Evaluation(
            log_prob=chosen_scores(scores, batch) - scores.total,
            entropy=decision_entropy(scores),
            value=out.value.float(),
            value_logits=out.value_logits.float(),
            margin=None,
            points=None if out.points is None else out.points.float(),
        )
