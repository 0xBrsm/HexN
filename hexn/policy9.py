# SPDX-License-Identifier: GPL-3.0-only
"""`Policy9`: a contract-9 `hexn.model9.HexNet` as a
`hexset.clients.policy.Policy`, so `hexset.clients.netbot` plays it, gates
its trades and seats it in the arena like any other checkpoint.

The bot asks for one atomic move at a time (`act_rows` over
`hexset.actions.options_for`). Where a move opens a compound decision (a setup
turn, the snake seat's double turn, a Road Building card, a discard,
`hexset.decisions`), the policy chooses the whole decision there, at the
position before its first move, returns that first move, and keeps the rest as
the seat's plan: each later ask inside the decision gets the plan's next move.
Nobody else acts inside a decision and nothing is revealed, so the plan cannot
go stale; it is checked against the options all the same, and a position
inside a decision with no plan of its own (one a gate imagined, or a game
resumed part-way) is finished greedily from the position as it stands.

Greedy: the highest-scored decision. `score_rows` (search priors) is not
built: the contract-9 reads play one forward. A net whose
`ModelConfig.double_turn` is "parts" (one PPO trained) chooses the snake
seat's double turn as two setup turns, the setup head read once on the board
before the first and once after it; "joint" (contract-9 BC's) chooses the
pair at once with the double turn's heads.

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

import numpy as np
import torch

from hexset import contract9, decisions as D
from torch import Tensor

from hexset.actions import Action, ActionSpace, ActionType
from hexset.board.terrain import NUM_RESOURCES
from hexset.encoding import from_frame
from hexset.game import Game, Phase, pending_free_roads

from .model9 import HexNet, Scores, chosen_scores, collate, decision_entropy, decision_scores, unpack9
from .policy import Evaluation, keyed_gumbel, keyed_gumbel_at
from .selfplay import Choice, Request

__all__ = ["Policy9", "NetworkPolicy9", "option_of"]


class Policy9:
    def __init__(self, net: HexNet, space: ActionSpace, *, device: str = "cpu") -> None:
        self.net = net.eval()
        self.space = space
        self.device = device
        self.players = net.players
        topology_sizes = (net.num_hexes, net.num_vertices, net.num_edges)
        self.layout = contract9.layout(net.players, *topology_sizes)
        # Per game (by identity), the seat whose decision is under way and the
        # moves of it still to make.
        self._plans: dict[int, tuple[Game, int, list[Action]]] = {}

    # --- the forward -------------------------------------------------------------

    def _forward(self, games: Sequence[Game], seats: Sequence[int], *, policy: bool):
        x = np.stack([contract9.flatten(contract9.encode9(g, s)) for g, s in zip(games, seats)])
        with torch.no_grad():
            return self.net(**unpack9(self.layout, torch.from_numpy(x).to(self.device)), policy=policy)

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """The win head, one vector per row, in board-seat order."""
        if not rows:
            return []
        out = self._forward([g for g, _ in rows], [s for _, s in rows], policy=False)
        values = out.value.float().cpu().numpy()
        return [tuple(float(v) for v in from_frame(values[i], seat)) for i, (_, seat) in enumerate(rows)]

    def score_rows(self, rows):  # pragma: no cover - not built
        raise NotImplementedError("contract-9 search priors are not built; play one forward")

    # --- choosing ------------------------------------------------------------------

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        return [self._act(game, seat, options) for game, seat, options in rows]

    def _act(self, game: Game, seat: int, options: tuple[Action, ...]) -> Action:
        plan = self._plans.pop(id(game), None)
        if plan is not None and plan[0] is game and plan[1] == seat and plan[2] and plan[2][0] in options:
            rest = plan[2][1:]
            if rest:
                self._plans[id(game)] = (game, seat, rest)
            return plan[2][0]
        if len(options) == 1:
            return options[0]
        if game.phase is Phase.SETUP_ROAD or pending_free_roads(game):
            return self._finish(game, seat, options)
        choice = self._decide(game, seat)
        if len(self._plans) > 256:          # a game that ended inside a decision left its plan
            self._plans.clear()
        if isinstance(choice, D.Compound):
            first, rest = choice.parts[0], list(choice.parts[1:])
            if rest:
                self._plans[id(game)] = (game, seat, rest)
            choice = first
        if choice not in options:
            raise RuntimeError(f"contract-9 policy chose {choice}, not among {options}")
        return choice

    def _decide(self, game: Game, seat: int):
        """The highest-scored decision of `seat`'s at this position."""
        decision = D.tensors(game, seat, self.space, double_turn=self.net.config.double_turn == "joint")
        out = self._forward([game], [seat], policy=True)
        net, players = self.net, self.players
        neg = float("-inf")
        best: tuple[float, object] = (neg, None)

        def offer(score: float, option) -> None:
            nonlocal best
            if score > best[0]:
                best = (score, option)

        rows = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            if decision.flat.any():
                logits = net.full_logits(out)[0].float().cpu().numpy()
                legal = np.flatnonzero(decision.flat)
                index = int(legal[np.argmax(logits[legal])])
                action = self.space.decode(index)
                if action.type is ActionType.MOVE_ROBBER and action.b < players:
                    action = Action(action.type, action.a, (action.b + seat) % players)
                offer(float(logits[index]), action)
            if decision.setup.any():
                pairs = contract9.incidences(game._state.board.topology)
                legal = torch.from_numpy(decision.setup)
                if decision.double:
                    _, first, second, pair = (t[0].float() for t in net.setup_heads(out, rows))
                    v = net.inc_vertex
                    scores = first[:, None] + second[None, :] + pair[v][:, v]
                    ok = legal[:, None] & legal[None, :] & net.pairs_ok
                    flat = scores.masked_fill(~ok, neg).reshape(-1)
                    k = int(flat.argmax())
                    (v1, e1), (v2, e2) = pairs[k // len(pairs)], pairs[k % len(pairs)]
                    offer(float(flat[k]), D.Compound(D.CompoundKind.SETUP_DOUBLE, D.setup_parts(v1, e1) + D.setup_parts(v2, e2)))
                else:
                    masked = net.setup_scores(out, rows)[0].float().masked_fill(~legal, neg)
                    k = int(masked.argmax())
                    offer(float(masked[k]), D.Compound(D.CompoundKind.SETUP, D.setup_parts(*pairs[k])))
            if len(decision.road_building):
                candidates = torch.from_numpy(decision.road_building.astype(np.int64))
                scores = net.road_building_scores(out, torch.zeros(len(candidates), dtype=torch.long), candidates)
                k = int(scores.argmax())
                roads = tuple(int(e) for e in decision.road_building[k] if e >= 0)
                offer(float(scores[k]), D.Compound(D.CompoundKind.ROAD_BUILDING, D.road_building_parts(roads)))
            if len(decision.discard):
                candidates = torch.from_numpy(decision.discard.astype(np.int64))
                scores = net.discard_scores(out, torch.zeros(len(candidates), dtype=torch.long), candidates)
                k = int(scores.argmax())
                given = decision.discard[k][NUM_RESOURCES:]
                offer(float(scores[k]), D.Compound(D.CompoundKind.DISCARD, D.discard_parts(given)))
        if best[1] is None:
            raise RuntimeError("contract-9 policy found no decision to make")
        return best[1]

    def _finish(self, game: Game, seat: int, options: tuple[Action, ...]) -> Action:
        """A move inside a decision begun without this policy's plan: the
        best-scored road among the options, by the road head."""
        out = self._forward([game], [seat], policy=True)
        with torch.no_grad():
            road = out.road[0].float().cpu().numpy()
        return max(options, key=lambda a: road[a.a] if a.type in (ActionType.BUILD_ROAD, ActionType.SETUP_ROAD) else -np.inf)


# --- self-play -------------------------------------------------------------------------


def option_of(decision: D.Decision, kind: int, index: int, pairs, seat: int, space: ActionSpace,
              players: int) -> D.Option:
    """The option `(kind, index)` names in `decision` (`hexset.decisions.locate`
    the other way): an action, its robber victim back in board seats, or a
    `Compound`. `pairs` is the board's `contract9.incidences`."""
    if kind == D.DecisionKind.FLAT:
        action = space.decode(index)
        if action.type is ActionType.MOVE_ROBBER and action.b < players:
            action = Action(action.type, action.a, (action.b + seat) % players)
        return action
    if kind == D.DecisionKind.SETUP:
        return D.Compound(D.CompoundKind.SETUP, D.setup_parts(*pairs[index]))
    if kind == D.DecisionKind.DOUBLE:
        first, second = divmod(index, len(pairs))
        return D.Compound(D.CompoundKind.SETUP_DOUBLE, D.setup_parts(*pairs[first]) + D.setup_parts(*pairs[second]))
    if kind == D.DecisionKind.ROAD_BUILDING:
        roads = tuple(int(e) for e in decision.road_building[index] if e >= 0)
        return D.Compound(D.CompoundKind.ROAD_BUILDING, D.road_building_parts(roads))
    return D.Compound(D.CompoundKind.DISCARD, D.discard_parts(decision.discard[index][NUM_RESOURCES:]))


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
    takes the best-scored option. Every seat plays at temperature 1, and only
    in fp32 (`precision`). A double-turn decision (`Decision.double`) is
    refused: the lanes ask it as two setup turns."""

    def __init__(self, net: HexNet, space: ActionSpace, *, device: torch.device | str = "cpu", greedy: bool = False,
                 generator: torch.Generator | None = None) -> None:
        self.net = net
        self.space = space
        self.device = torch.device(device)
        self.greedy = greedy
        self.generator = generator
        self.precision = "fp32"
        self.layout = contract9.layout(net.players, net.num_hexes, net.num_vertices, net.num_edges)
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
        if self.precision != "fp32" or any(r.temperature != 1.0 for r in requests):
            raise ValueError("a contract-9 policy samples in fp32 at temperature 1")
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
        with torch.inference_mode():
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
            self._pairs = contract9.incidences(requests[0].game._state.board.topology)
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
