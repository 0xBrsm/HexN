# SPDX-License-Identifier: GPL-3.0-only
"""Episode assembly for native, trade-free parallel collection.

HexSet owns positions, requests, legal actions, Chance and records. This
adapter keeps the original policy and training rows, deals through the
ordinary Collector, and writes each completion through its existing sink.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hexset import contract9
from hexset.actions import space_for
from hexset.gym.native import NativeBatch, NativeLane, available

from .durable import Partial
from .policy9 import NetworkPolicy9
from .selfplay import Episode, Request, Transition9, draw_key


def supported(collector) -> bool:
    """Whether an empty collector has the supported native worker semantics."""
    return bool(
        available()
        and not collector.running
        and collector.contract in contract9.CONTRACTS
        and collector.players in (2, 4)
        and collector.caster is None
        and not collector.opponents
        and collector.coalition is None
        and collector.temperatures is None
        and not collector.outlook
        and not collector.marks
        and collector.sample_stream == "game"
        and collector.max_offers == 0
        and collector.trader is None
        and bool(collector._batched)
        and all(type(policy) is NetworkPolicy9 and policy.net.config.double_turn == "parts"
                for policy in collector._batched.values())
    )


class NativeWorkerCollector:
    """The original worker's complete-cohort and listed-resume interface.

    The ordinary Collector remains the deal factory. Its original lane count,
    index stride, board/rule law, cast, Chance and refill order stay in use.
    Native state replaces each fresh Game after import, before any move.
    """
    def __init__(self, original):
        if not supported(original):
            raise ValueError("unsupported native worker configuration")
        self.original = original
        self._next = original.games_started()
        self._active = None
        self.sink = self.beat = None

    @property
    def running(self):
        return self._active is not None and self._active.running

    def upcoming(self, count):
        return [self._next + k * self.original.stride for k in range(count)]

    def games_started(self):
        return self._next

    def _lane(self, dealt):
        if dealt is None:
            return None
        if any(dealt.seat_bots) or dealt.watching:
            raise ValueError("native collection does not seat scripted or watching bots")
        return NativeLane.import_game(dealt.game, index=dealt.index, seed=self.original.seed,
            tape=dealt.tape, action_cap=self.original.action_cap, cast=dealt.cast)

    def _play(self, factory):
        if self.sink is None or not hasattr(self.sink, "partial"):
            raise ValueError("native workers require the durable whole-game sink")
        journal = self.sink.partial
        keep = journal if isinstance(journal, Partial) else Partial(Path(journal.directory))
        plan = keep.plan()
        if plan is None:
            raise ValueError("the parent must persist its complete cohort plan before play")
        dealt = [lane for lane in factory._lanes if lane is not None]
        if not dealt:
            return []
        first = dealt[0].game
        space = space_for(first)
        topology = first.state(0, hidden=False).board.topology
        pairs = tuple(tuple(map(int, pair)) for pair in contract9.incidences(topology))
        self._check_topology(space, pairs)
        lanes = [self._lane(lane) for lane in dealt]
        factory._lanes = []
        del first, dealt
        trajectories = {lane.index: [[] for _ in range(lane.players)] for lane in lanes}
        pending = {}

        def on_decision(request, answer):
            transition = pending.pop(request.ids, None)
            if transition is not None:
                trajectories[request.index][request.seat].append(transition)

        def completed(game):
            episode = Episode(game.index, game.seed, game.players,
                tuple(tuple(rows) for rows in trajectories[game.index]), game.outcome,
                cast=(), record=game.record)
            result = self.sink(episode)
            # The sink commits the complete file before the engine may retire
            # this lane. Drop its rows immediately, as the ordinary worker does.
            del trajectories[game.index]
            return episode if result is None else result

        batch = NativeBatch(lanes, space=space, pairs=pairs, contract=self.original.contract,
            planned_indices=plan, finished_indices=keep.finished(), keep=completed,
            on_decision=on_decision, discard_order=self.original.discard_order)
        self._active = batch
        output = []
        try:
            while batch.running:
                native_requests = batch.requests()
                targets = batch.targets()
                requests = [Request(lane=r.lane, seat=r.seat, observation=r.observation,
                    mask=r.mask, options=(), game=None, decision=r.decision,
                    draw=draw_key(self.original.seed, r.index, r.seat, r.step),
                    decision_id=r.decision_id, part=r.part) for r in native_requests]
                answers = [None] * len(requests)
                for pid, policy in self.original._batched.items():
                    rows = [i for i, request in enumerate(native_requests) if request.policy == pid]
                    if rows:
                        choices = list(policy.act([requests[i] for i in rows]))
                        if len(choices) != len(rows):
                            raise ValueError("one native answer is required per policy request")
                        for row, choice in zip(rows, choices):
                            answers[row] = choice
                for row, (request, choice) in enumerate(zip(native_requests, answers)):
                    if request.policy in self.original._learners and choice is not None:
                        if choice.decision is None:
                            raise ValueError("native answers must identify their decision candidate")
                        kind, at = choice.decision
                        pending[request.ids] = Transition9(request.seat, request.step,
                            request.observation, request.decision, int(kind), int(at), choice.log_prob,
                            tuple(choice.value), tuple(choice.points), tuple(map(int, targets.points[row])),
                            int(targets.turns[row]), request.decision_id, request.part)
                output.extend(batch.advance(answers))
                if pending:
                    raise RuntimeError("native decisions did not consume their pending training rows")
                for slot, lane in enumerate(batch.lanes):
                    if lane.finished:
                        replacement = self._lane(factory._fresh())
                        if replacement is not None:
                            trajectories[replacement.index] = [[] for _ in range(replacement.players)]
                            batch.replace_lane(slot, replacement)
                if self.beat is not None:
                    self.beat()
            return output
        finally:
            self._active = None

    def _check_topology(self, space, pairs):
        """Prime the original policy's immutable public incidence cache once."""
        pair_array = np.asarray(pairs)
        for policy in self.original._batched.values():
            if policy.space != space or policy.net.CONTRACT != self.original.contract:
                raise ValueError("native policy action space or contract differs")
            ends = policy.net.ends.detach().cpu().numpy()
            vertices = policy.net.inc_vertex.detach().cpu().numpy()
            if (pair_array.shape != (len(vertices), 2) or pair_array.dtype.kind not in "iu"
                    or np.any(pair_array[:, 1] < 0) or np.any(pair_array[:, 1] >= len(ends))):
                raise ValueError("native setup incidence shape or edge differs")
            selected = ends[pair_array[:, 1]]
            far = np.where(selected[:, 0] == pair_array[:, 0], selected[:, 1], selected[:, 0])
            if (not np.array_equal(vertices, pair_array[:, 0])
                    or not np.array_equal(policy.net.inc_far.detach().cpu().numpy(), far)):
                raise ValueError("native setup incidences differ from the model topology")
            if policy._pairs is not None and not np.array_equal(policy._pairs, pairs):
                raise ValueError("the policy already holds another setup topology")
            policy._pairs = pairs

    def cohort(self, count):
        if count < 1 or self.running:
            raise ValueError("native cohorts require a positive count and empty lanes")
        factory = self.original._new_env(deal=count, first_game=self._next)
        try:
            return self._play(factory)
        finally:
            self._next = factory.games_started()

    def listed(self, indices):
        indices = list(indices)
        if not indices:
            return []
        if self.running:
            raise ValueError("listed native resume requires empty lanes")
        factory = self.original._new_env(deal=len(indices), first_game=indices[0], indices=indices)
        return self._play(factory)

    def collect(self, count):
        raise ValueError("native workers collect complete cohorts, not first-finish streams")
