# SPDX-License-Identifier: GPL-3.0-only
"""Collect self-play experience from many games at once.

The lockstep environment itself is HexSet's (`hexset.gym.lanes.LaneEnv`): it
holds the games, deals them from `(seed, index)`, seats every trade gate, steps
every lane once per tick, caps the action stream, counts the exchanges and
harvests the outcome. **Nothing about a position is computed here** -- this
module is the trainer's side of that loop and only that: it batches a tick's
decisions to a `BatchPolicy`, answers the environment, and files the extras a
policy-gradient update needs against the decisions the environment reports.

Why a batch at all: a forward pass costs a fixed dispatch toll per call plus a
much smaller cost per position, so a driver that stepped one game and called
the net per move would spend essentially all of its time in dispatch. That is
why `LaneEnv` exists and why the seam here is `BatchPolicy` -- one call per
tick per policy, whatever the lane count.

**Reward is deliberately not here.** A collector emits per-seat trajectories and
the terminal `Outcome`; turning that into returns is the caller's to decide,
because the choice between terminal win/loss and terminal victory points is
still open. Nothing in this module assumes either.

Trajectories come out demultiplexed by seat, which is `LaneEnv`'s doing: a Catan
game interleaves four seats' decisions into one action stream, and a seat's next
state is not the one that immediately follows its action -- it is the next
position that seat was asked about. What this module adds is the join, by
`(seat, step)`, between the environment's `Decision`s and the encoding,
log-probability, value and `aux` the policy produced for the same decision.

The plumbing stays numpy-only and torch-free, because PyTorch cannot be
installed on the development phone: it is tested and timed here against a
trivial policy and only the torch-backed policy is deferred to the training box.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Sequence

import numpy as np

from hexset.actions import Action, ActionSpace, mask_of, space_for
from hexset.arena import MAX_ACTIONS, deal_board, deal_game
from hexset.board.board import Board
from hexset.encoding import Observation, encode_batch
from hexset.game import MAX_TURNS, Game
# `_Lane` is internal to HexSet (an underscore name, outside `__all__`), used
# knowingly: `RuledLaneEnv` and `_Listed` deal their own games through
# `LaneEnv`'s `_fresh`, `_next`, `_stop`, `_cast` and `_seat_gates`, which a
# HexSet release may change without notice. Check them at every repin.
from hexset.gym.lanes import BoardBots, LaneEnv, Outcome, _Lane
from hexset.gym.lanes import Episode as LaneEpisode
from hexset.victory import victory_points
from hexset.gym.lanes import Request as LaneRequest
from hexset.record import Record, Tape, recording
from hexset.rules import STANDARD_GAME, GameType

if TYPE_CHECKING:
    from .durable import Partial

# Re-exported rather than redefined: HexSet's terminal facts are already the
# ones a trainer wants, `winner` for terminal win/loss and `points` for terminal
# victory points, and a second copy here is exactly the drift this package's
# boundary rule exists to stop.
__all__ = [
    "BatchPolicy",
    "BoardBots",
    "Choice",
    "Collector",
    "Episode",
    "Outcome",
    "RandomPolicy",
    "Request",
    "Transition",
    "owned",
]


@dataclass(frozen=True)
class Request:
    """One lane's decision, put to the policy alongside every other lane's.

    The trainer-side view of `hexset.gym.lanes.Request`: the same decision, with
    the encoding and the mask a network needs added and the engine bookkeeping
    (`index`, `step`, `policy`) left behind, because the join back onto the
    environment's `Decision`s happens in `Collector` and not in a policy.

    `options` rides along with `mask` because a searching policy wants the
    `Action` objects themselves and rebuilding them from the mask would mean
    enumerating the position a second time. Every one of them is recoverable
    from its flat index under contract 5 — there is no trade action left whose
    operands do not fit in one (`hexset.trading`).

    `seat` is `to_move`, which is not always `current_player` — discarding on a
    seven belongs to somebody else — and it is both the perspective
    `observation` was encoded from and the seat the transition is filed under.

    `game` is the live lane state, and it rides along for a policy that searches:
    a tree needs positions to step, and an observation is a lossy encoding of
    one. A policy that only reads the encoding can ignore it. Handed out rather
    than copied because `hexset.mcts` copies at its own root; a policy that
    mutates this corrupts the lane.
    """

    lane: int
    seat: int
    observation: Observation
    mask: np.ndarray
    options: tuple[Action, ...]
    game: Game | None = None
    # The sampling temperature this seat plays at, 1.0 unless the cast tempered
    # it (`hexn.collect.mix_temperatures`). A policy that samples divides its
    # logits by it and records the log-probability under *that* distribution,
    # which is what keeps the PPO ratio exact; a greedy policy ignores it.
    temperature: float = 1.0
    # The seat's own victory points at this decision, hidden VP cards included
    # (its own, so known to it); read by a policy recording a per-VP reward's
    # value (`hexn.policy.NetworkPolicy.vp_reward`) and kept on the transition.
    points: int = 0


@dataclass(frozen=True)
class Choice:
    """What the policy did, and what PPO will want to have recorded with it.

    `value` is the per-seat vector the value head emits, in the mover's frame
    like the encoder. Empty means the policy has no estimate, which is the case
    for every scripted policy.

    `aux` is carried through onto the `Transition` untouched and never read
    here: a pocket for whatever a policy needs its own update to see and the
    transition does not otherwise keep — a search's visit target
    (`hexn.expert`), a value spread (`benchmarks.sibling`). Default empty, so
    no scripted policy notices.
    """

    action: Action
    log_prob: float = 0.0
    value: tuple[float, ...] = ()
    aux: object = None


class BatchPolicy(Protocol):
    """The batched analogue of `hexset.bots.Bot`.

    One call per tick, so a torch implementation collates, moves and runs the
    network once for the whole batch. Must return one `Choice` per `Request`,
    in order.
    """

    def act(self, requests: Sequence[Request]) -> Sequence[Choice]: ...


@dataclass
class RandomPolicy:
    """Uniform over the legal actions.

    Torch-free, so it measures the cost of the plumbing alone and gives the
    tests a policy whose behaviour they can predict. Its `log_prob` is the real
    one for a uniform choice rather than a placeholder.
    """

    rng: random.Random = field(default_factory=random.Random)

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        return [
            Choice(
                action=self.rng.choice(request.options),
                log_prob=-math.log(len(request.options)),
            )
            for request in requests
        ]


@dataclass(frozen=True)
class Transition:
    """One decision, filed under the seat that made it.

    `seat` and `step` are `hexset.gym.lanes.Decision`'s own, so the interleaved
    order is recoverable even though the storage is per seat, and so this row
    and the engine's row for the same decision are joinable by that pair.

    `aux` is whatever the policy attached to its `Choice`, carried verbatim.
    """

    seat: int
    step: int
    observation: Observation
    mask: np.ndarray
    action: Action
    index: int
    log_prob: float
    value: tuple[float, ...]
    aux: object = None
    # The seat's own victory points at this decision (`Request.points`).
    points: int = 0


@dataclass(frozen=True)
class Episode:
    """One finished game: per-seat trajectories plus how it ended.

    `hexset.gym.lanes.Episode` with the engine's `Decision`s replaced by the
    trainer's `Transition`s — everything else on it is carried through
    unchanged, because those fields are the engine's account of the game and
    there is nothing here to add to them.

    `index` and `seed` are enough to rebuild the game exactly
    (`hexset.arena.deal_game`), so an episode can be replayed against the engine
    rather than trusted.

    `cast` is who played each seat — 0 the learner, `k` the collector's
    `opponents[k - 1]` — and empty means everyone was the learner. Opponent
    seats have empty trajectories, so `stream()` on a cast episode has gaps and
    cannot replay the game; the outcome still describes the whole table.

    `trades` is every exchange the table's trade event cleared over the whole
    game, `(step, a, b, received)` — the same shape `hexset.record.Record`
    keeps, `step` indexing into `stream()` exactly as `Transition.step` does.
    The engine's own trade event depends on `game.gates`, which a replay never
    seats (`hexset.record.advance`'s reasoning: nobody is there to ask), so a
    reconstruction that only replayed `stream()`'s actions would play a
    trade-free game however the table actually traded — this is what
    `hexn.replay.replay_to_ply` applies explicitly instead. Empty for a
    trade-free run, and for every episode collected before this field existed.

    `record` is the engine's own `hexset.record.Record` of the whole game --
    every seat's actions, not just the trainer-recorded ones `stream()` walks
    -- when the environment was built with `records=True`
    (`Collector._new_env`, always, since carrying it costs one lane's own
    `hexset.record.Tape` and a few KB a game). `None` for an episode collected
    before this field existed, exactly like `trades` above; `replay_to_ply`
    is what reads it.
    """

    index: int
    seed: int
    players: int
    trajectories: tuple[tuple[Transition, ...], ...]
    outcome: Outcome
    cast: tuple[int, ...] = ()
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    record: Record | None = None

    def stream(self) -> list[Transition]:
        """Every transition back in the order it was taken."""
        return sorted(
            (t for seat in self.trajectories for t in seat), key=lambda t: t.step
        )

    def __len__(self) -> int:
        return sum(len(seat) for seat in self.trajectories)


def owned(episodes: Sequence[Episode], learner: int) -> list[Episode]:
    """Each episode with only `learner`'s seats kept — the table league's
    assemble-side counterpart to the collector's recording gate.

    An empty `cast` means the pre-league convention: every seat was learner 0.
    Seats belonging to other learners are emptied, not dropped, so `assemble`
    skips them exactly as it has always skipped opponent seats, and per-seat
    indexing (rewards, rotation) is untouched.
    """
    out = []
    for episode in episodes:
        cast = episode.cast or (0,) * episode.players
        trajectories = tuple(
            seat_transitions if cast[seat] == learner else ()
            for seat, seat_transitions in enumerate(episode.trajectories)
        )
        out.append(
            Episode(
                index=episode.index,
                seed=episode.seed,
                players=episode.players,
                trajectories=trajectories,
                outcome=episode.outcome,
                cast=episode.cast,
                trades=episode.trades,
                record=episode.record,
            )
        )
    return out


class RuledLaneEnv(LaneEnv):
    """A `LaneEnv` whose games are dealt by a law of the game index:
    `game_law(index) -> (GameType, retired seats)`. HexSet's own lanes deal
    the standard game at every seat; this deals, say, a duel-variant game on a
    full table with two seats retired, which the engine then never asks to
    move. Everything but the deal is `LaneEnv`'s."""

    def __init__(
        self,
        *args,
        game_law: Callable[[int], tuple[GameType, frozenset[int]]],
        coalition: Callable[[int], object] | None = None,
        **kwargs,
    ) -> None:
        self.game_law = game_law
        # Game index -> the game's `hexn.collect.CoalitionPlan` or None
        # (`hexn.collect.mix_coalitions`); a plan is hung on the game as
        # `Game.coalition`, where its `targeted:` seats read it
        # (`hexn.coalition.Targeted`).
        self.coalition = coalition
        super().__init__(*args, **kwargs)

    def _fresh(self) -> _Lane | None:
        if self._stop is not None and self._next >= self._stop:
            return None
        index = self._next
        self._next += self.stride
        cast = self._cast(index)
        game_type, retired = self.game_law(index)
        board = self.board(index) if callable(self.board) else self.board
        rules = game_type.rules
        game = deal_game(
            self.seed,
            index,
            self.players,
            board=board,
            chance=(lambda rng: recording(rng, rules)) if self.records else None,
            game_type=game_type,
            locked=retired,
            turn_cap=self.turn_cap,
        )
        game.gates = self._seat_gates(game, cast)
        plan = None if self.coalition is None else self.coalition(index)
        if plan is not None:
            game.coalition = plan
        return _Lane(
            index=index,
            game=game,
            cast=cast,
            by_seat=[[] for _ in range(self.players)],
            tape=Tape() if self.records else None,
        )


def standard_law(index: int) -> tuple[GameType, frozenset[int]]:
    """The game law every run before `duel(...)` played: the standard game
    at every seat, nobody retired."""
    return STANDARD_GAME, frozenset()


class _Listed(LaneEnv):
    """A `LaneEnv` that deals exactly `indices`, in order, and nothing else.

    What a resumed cohort needs: the games a crash interrupted, by index. The
    pinned engine's `LaneEnv` deals a strided range only, so this points its
    counter at each listed index in turn and lets `LaneEnv._fresh` deal it --
    the game, its cast, board, gates and record all come from the engine's
    own dealing, not a copy of it. `_fresh`, `_next` and `_stop` are the
    engine's private names; an index list on `LaneEnv` itself retires this.
    """

    def __init__(self, *args, indices: Sequence[int], **kwargs) -> None:
        self._listed = deque(indices)
        kwargs["deal"] = len(self._listed)
        super().__init__(*args, **kwargs)

    def _fresh(self):
        if not self._listed:
            return None
        self._next = self._listed.popleft()
        self._stop = self._next + 1
        return super()._fresh()


class _RuledListed(_Listed, RuledLaneEnv):
    """A resumed cohort of a ruled run: `_Listed` picks the indices,
    `RuledLaneEnv` deals each one by its law."""


class Collector:
    """A `hexset.gym.lanes.LaneEnv` driven by a `BatchPolicy`, one batch a tick.

    The environment owns the games: it deals `(seed, index)`, refills a finished
    lane on the spot so the batch stays full, caps the action stream, seats each
    seat's trade gate and reports the outcome. This class owns the trainer's
    half — encode the tick, call each policy once, answer the environment, and
    file a `Transition` for every recorded decision.

    `opponents` and `caster` put other policies on some seats. The caster maps
    a game index to one policy id per seat — 0 the learner, `k` for
    `opponents[k - 1]` — and must be a pure function of the index, so a resumed
    run casts the same games the same way. An opponent that is a
    `hexset.gym.lanes.BoardBots` is played by the environment itself (a scripted
    bot pays no dispatch toll and is its own trade gate); every other opponent
    is a `BatchPolicy` and gets its own `act` call per tick, so the learner's
    batch stays batched. **Opponent decisions are never recorded**: their seats'
    trajectories stay empty, which is exactly what `hexn.ppo.assemble` skips, so
    opponent play shapes the games the learner sees without ever entering an
    update.
    """

    def __init__(
        self,
        policy: BatchPolicy,
        *,
        lanes: int = 8,
        players: int = 4,
        seed: int = 0,
        action_cap: int = MAX_ACTIONS,
        turn_cap: int = MAX_TURNS,
        board: Board | None = None,
        max_offers: int | None = None,
        trader: str | None = None,
        first_game: int = 0,
        deal: int | None = None,
        fill: bool = True,
        opponents: Sequence[object] = (),
        caster: Callable[[int], Sequence[int]] | None = None,
        learners: Sequence[int] = (0,),
        stride: int = 1,
        pair_boards: bool = False,
        temperatures: Callable[[int], Sequence[float]] | None = None,
        game_law: Callable[[int], tuple[GameType, frozenset[int]]] | None = None,
        coalition: Callable[[int], object] | None = None,
    ) -> None:
        """`first_game` is where the game counter starts.

        A training run that crashes and resumes would otherwise replay the games
        it had already learned from, since a game is a pure function of the seed
        and its index. Checkpointing `games_started` and passing it back here is
        what makes a resumed run continue rather than repeat.

        `deal` bounds how many games are ever started. Left `None`, a lane is
        refilled the moment its game ends and the collector runs forever, which
        is what a training run wants. An *evaluation* wants a fixed cohort, and
        without a bound the only way to get one is to keep dealing replacements
        and throw them away after playing them in full — which is where a
        400-game duel went to spend ten minutes.

        `stride` deals every `stride`-th index instead of every one, which is
        how parallel collectors shard one run's games: worker `w` of `K` takes
        `first_game = base + w, stride = K` and the workers' index sets are
        disjoint by construction while every game stays the same pure function
        of `(seed, index)` it always was.

        `pair_boards` transplants the duel pairing to collection: games `2k`
        and `2k+1` share the board keyed `f"{seed}:{2k}:board"` while each
        keeps its own `f"{seed}:{index}:game"` rng — same geometry, independent
        dice and play (`_PairedBoards`).

        `fill=False` leaves the collector empty until `cohort` deals one, which
        is what a PPO iteration wants; anything else deals its lanes here.

        `sink`, when set, is handed every episode the tick it finishes -- what
        keeps a game on disk before the batch it belongs to ends
        (`hexn.durable.Partial.keep`). `beat`, when set, is called once a
        tick, so a process watching this one can tell slow from wedged.
        """
        if lanes < 1:
            raise ValueError("a collector needs at least one lane")
        if deal is not None and deal < 1:
            raise ValueError("a collector cannot be asked to deal nothing")
        if stride < 1:
            raise ValueError("a collector cannot deal backwards or stand still")
        if caster is not None and not opponents:
            raise ValueError("a caster without opponents has nobody to cast")
        if pair_boards and board is not None:
            raise ValueError(
                "pair_boards deals each pair its own shared board; a fixed "
                "board= would put every game on one board and the pairing "
                "would compare nothing"
            )
        self.policy = policy
        self.opponents = tuple(opponents)
        self.caster = caster
        # Game index -> per-seat sampling temperature, pure in the index like
        # the caster it accompanies; None means every seat plays at 1.0.
        self.temperatures = temperatures
        # Game index -> (game type, retired seats), pure in the index like the
        # caster (`hexn.collect.mix_deal`); None deals the standard game at
        # every seat, as every run before `duel(...)` did.
        self.game_law = game_law
        # Game index -> the game's coalition plan or None, pure in the index
        # like the caster (`hexn.collect.mix_coalitions`); None hangs nothing
        # on any game.
        self.coalition = coalition
        # Which policy ids record their seats. {0} is every run before gen3 —
        # one learner, opponents as scenery. The table league seats several
        # learners in one game: id 0 is `policy`, id k>0 is `opponents[k-1]`,
        # and a seat records iff its id is here.
        self._learners = frozenset(learners)
        if any(not 0 <= pid <= len(self.opponents) for pid in self._learners):
            raise ValueError("a learner id names a policy that is not seated")
        self.players = players
        self.seed = seed
        self.action_cap = action_cap
        # HexSet's per-run turn cap (0.65): past it a game ends unfinished.
        # The default is read off agents trying to win; unstructured play
        # passes `hexset.game.UNSTRUCTURED_TURN_CAP`.
        self.turn_cap = turn_cap
        self.max_offers = max_offers
        self.trader = trader
        self.board = board
        self.pair_boards = pair_boards
        self.stride = stride
        self.lanes = lanes
        self.ticks = 0
        self.steps = 0
        self.games = 0
        self._deal = deal
        self._next_game = first_game
        self._stop = None if deal is None else first_game + deal * stride
        self.sink: Callable[[Episode], None] | None = None
        self.beat: Callable[[], None] | None = None

        # A scripted opponent is played by the environment, which seats it as
        # its own gate too; everything else answers through `act`.
        self._benches = {
            pid + 1: opponent
            for pid, opponent in enumerate(self.opponents)
            if isinstance(opponent, BoardBots)
        }
        self._batched: dict[int, BatchPolicy] = {0: policy}
        self._batched.update(
            (pid + 1, opponent)  # type: ignore[misc]
            for pid, opponent in enumerate(self.opponents)
            if pid + 1 not in self._benches
        )

        self._pending: dict[int, dict[tuple[int, int], tuple[Request, Choice]]] = {}
        self._env: LaneEnv | None = (
            self._new_env(deal=deal, first_game=first_game) if fill else None
        )
        # A pure function of the rules, not of the game, so any position gives
        # it — and an empty collector has none of its own to ask.
        first = next(self.in_flight(), None)
        self.space: ActionSpace = space_for(
            first
            if first is not None
            else deal_game(seed, first_game, players, board=board)
        )

    # --- the environment ---------------------------------------------------

    def _cast(self, index: int) -> tuple[int, ...]:
        """The caster's verdict, checked against the seated policies.

        `LaneEnv` only knows a cast must be the right length and non-negative;
        it has no idea how many opponents this collector holds, and a cast that
        names one it does not would otherwise fail much later as a lookup.
        """
        cast = tuple(self.caster(index))  # type: ignore[misc]
        if len(cast) != self.players or any(
            not 0 <= pid <= len(self.opponents) for pid in cast
        ):
            raise ValueError(f"cast {cast} does not fit game {index}")
        return cast

    def _trader(self, pid: int) -> Callable[[Game, int], object]:
        """This policy's gate factory in the shape `LaneEnv` seats.

        `LaneEnv` asks `gates[pid](game, seat)`; a `BatchPolicy`'s own hook also
        takes the network's offer budget and trader (`hexn.trade`). The table takes none:
        since HexSet 0.60 every seat declares its own, and a scripted opponent
        bargains as its own bot does.
        """
        make = self._batched[pid].trader  # type: ignore[attr-defined]
        if self.trader is None:
            return lambda game, seat: make(game, seat, self.max_offers)
        return lambda game, seat: make(game, seat, self.max_offers, self.trader)

    def _paired_board(self, index: int) -> Board:
        """`LaneEnv`'s board law for `pair_boards`: games `2k` and `2k+1` are
        dealt the board keyed to `2k` while each keeps its own
        `(seed, index)` game rng — same geometry, independent dice and play.

        The even half's board is exactly the one unpaired dealing derives for
        that index, so this extends the game law rather than amending it, and a
        pair straddling two strided workers still shares its board because each
        derives it from the same key.
        """
        return deal_board(self.seed, 2 * (index // 2))

    def _new_env(
        self, *, deal: int | None, first_game: int, indices: Sequence[int] = ()
    ) -> LaneEnv:
        env, extra = LaneEnv, {}
        ruled = self.game_law is not None or self.coalition is not None
        if ruled:
            env, extra = RuledLaneEnv, {
                "game_law": self.game_law or standard_law, "coalition": self.coalition,
            }
        if indices:
            env = _RuledListed if ruled else _Listed
            extra = {**extra, "indices": indices}
        return env(
            self.players,
            self.seed,
            self.lanes,
            deal=deal,
            action_cap=self.action_cap,
            turn_cap=self.turn_cap,
            board=self._paired_board if self.pair_boards else self.board,
            caster=None if self.caster is None else self._cast,
            # `bench.bot` rather than `bench.spawn`: the bench a caller built
            # (`hexn.collect.named_opponent`) is the one that must hold the
            # bots, so the capacity it chose is the capacity that applies and a
            # bot survives a cohort boundary.
            bots={pid: bench.bot for pid, bench in self._benches.items()},
            gates={
                pid: self._trader(pid)
                for pid in self._batched
                if getattr(self._batched[pid], "trader", None) is not None
            },
            first_game=first_game,
            stride=self.stride,
            # Every episode carries the engine's own `hexset.record.Record`
            # (`hexn.replay.replay_to_ply` reads it) -- a few KB a game
            # against observation arrays already dwarfing it, so there is no
            # cohort this should be switched off for.
            records=True,
            **extra,
        )

    # --- the tick ----------------------------------------------------------

    def _requests(self, asked: Sequence[LaneRequest]) -> list[Request]:
        """One tick's observations, through the vectorized encoder.

        Only the seats a `BatchPolicy` answers are encoded: a bot-played seat is
        the environment's to answer and never sees an observation, so encoding
        it would pay for the opponents as well as the learner.
        """
        if not asked:
            # Every seat this tick is played by a bot inside the environment.
            return []
        observations = encode_batch(
            [request.game for request in asked], [request.seat for request in asked]
        )
        return [
            Request(
                lane=request.lane,
                seat=request.seat,
                observation=observation,
                mask=np.asarray(mask_of(self.space, request.options), dtype=bool),
                options=request.options,
                game=request.game,
                points=victory_points(request.game.state(request.seat, hidden=False), request.seat),
                temperature=(
                    1.0
                    if self.temperatures is None
                    else float(self.temperatures(request.index)[request.seat])
                ),
            )
            for request, observation in zip(asked, observations, strict=True)
        ]

    def _answers(
        self, asked: Sequence[LaneRequest], requests: Sequence[Request]
    ) -> list[Choice]:
        """One `act` call per policy, reassembled in request order."""
        if len(self._batched) == 1:
            choices = list(self.policy.act(requests))
            if len(choices) != len(requests):
                raise ValueError(
                    f"policy answered {len(choices)} of {len(requests)} requests"
                )
            return choices
        shares: dict[int, list[int]] = {pid: [] for pid in self._batched}
        for i, request in enumerate(asked):
            shares[request.policy].append(i)
        out: list[Choice | None] = [None] * len(requests)
        for pid, share in shares.items():
            if not share:
                continue
            answers = self._batched[pid].act([requests[i] for i in share])
            if len(answers) != len(share):
                raise ValueError(
                    f"policy answered {len(answers)} of {len(share)} requests"
                )
            for i, choice in zip(share, answers):
                out[i] = choice
        return out  # type: ignore[return-value]

    def _episode(self, played: LaneEpisode) -> Episode:
        """The engine's episode with the trainer's rows joined onto it.

        The join key is `(seat, step)`, which is `Decision`'s own identity, so a
        decision the collector did not record — an opponent's — simply has no
        row and its seat stays empty.
        """
        kept = self._pending.pop(played.index, {})
        trajectories = []
        for decisions in played.decisions:
            row = []
            for decision in decisions:
                held = kept.get((decision.seat, decision.step))
                if held is None:
                    continue
                request, choice = held
                row.append(
                    Transition(
                        seat=decision.seat,
                        step=decision.step,
                        observation=request.observation,
                        mask=request.mask,
                        action=decision.action,
                        index=self.space.index(decision.action),
                        log_prob=choice.log_prob,
                        value=tuple(choice.value),
                        aux=choice.aux,
                        points=request.points,
                    )
                )
            trajectories.append(tuple(row))
        return Episode(
            index=played.index,
            seed=played.seed,
            players=played.players,
            trajectories=tuple(trajectories),
            # An uncast collector reports no cast, the convention every reader
            # already has ("empty means everyone was the learner"); `LaneEnv`
            # always stamps one because it has no such convention.
            cast=played.cast if self.caster is not None else (),
            trades=played.trades,
            record=played.record,
            outcome=played.outcome,
        )

    def _mine(self, outstanding: Sequence[LaneRequest]) -> list[LaneRequest]:
        """The decisions a `BatchPolicy` answers: everything not played by a bot
        inside the environment."""
        return [r for r in outstanding if r.policy not in self._benches]

    def requests(self) -> list[Request]:
        """This tick's decisions for the policies this collector drives.

        The trainer-side view of `LaneEnv.requests()`, in lane order, with the
        bot-played seats left out because the environment answers those itself.
        Same decisions until the tick is stepped -- `LaneEnv.requests()` is
        idempotent within a tick -- though the encoding is redone each call, so
        this is for inspecting a batch, not a way to avoid `tick`.
        """
        if self._env is None:
            return []
        return self._requests(self._mine(self._env.requests()))

    def tick(self) -> list[Episode]:
        """Step every live lane once. Returns the games that ended on this tick."""
        if self._env is None:
            return []
        outstanding = self._env.requests()
        if not outstanding:
            return []
        asked = self._mine(outstanding)
        requests = self._requests(asked)
        choices = self._answers(asked, requests)

        answered: dict[int, Action] = {}
        for lane_request, request, choice in zip(asked, requests, choices):
            answered[lane_request.lane] = choice.action
            if lane_request.policy in self._learners:
                self._pending.setdefault(lane_request.index, {})[
                    (lane_request.seat, lane_request.step)
                ] = (request, choice)
        # A lane left out is played by its seat's own bot (`LaneEnv.step`).
        finished = self._env.step(answered)

        self.ticks += 1
        self.steps += len(outstanding)
        self.games += len(finished)
        episodes = [self._episode(played) for played in finished]
        if self.sink is not None:
            for episode in episodes:
                self.sink(episode)
        if self.beat is not None:
            self.beat()
        return episodes

    # --- driving -----------------------------------------------------------

    @property
    def running(self) -> bool:
        """False once a bounded collector has played out everything it dealt."""
        return self._env is not None and self._env.running

    def _play_out(self) -> list[Episode]:
        out: list[Episode] = []
        while self.running:
            out.extend(self.tick())
        return out

    def drain(self) -> list[Episode]:
        """Play every dealt game to completion. Requires a `deal` bound."""
        if self._stop is None:
            raise ValueError("an unbounded collector never drains; pass `deal`")
        return self._play_out()

    def run(self, ticks: int) -> list[Episode]:
        out: list[Episode] = []
        for _ in range(ticks):
            out.extend(self.tick())
        return out

    def collect(
        self, episodes: int, partial: "Partial | None" = None
    ) -> list[Episode]:
        """Tick until `episodes` games have finished.

        Terminates: the action cap ends every lane within `action_cap` ticks, so
        this cannot spin however badly the policy plays.

        With a `partial`, every game is kept there as it finishes, and the
        games already there count toward `episodes`: a resumed stream returns
        them and tops up the rest. The caller deals past them
        (`hexn.durable.resume_base`), or a top-up could repeat one.
        """
        if partial is None:
            return self._collect(episodes)
        kept = partial.done()
        if len(kept) >= episodes:
            return kept
        held, self.sink = self.sink, partial.keep
        try:
            return kept + self._collect(episodes - len(kept))
        finally:
            self.sink = held

    def _collect(self, episodes: int) -> list[Episode]:
        if self._stop is not None:
            # A bounded collector stops dealing, so asking for more than it has
            # left would spin on empty ticks rather than block on a slow game.
            left = (self._stop - self.games_started()) // self.stride + len(
                self.pending()
            )
            if left < episodes:
                raise ValueError(f"{episodes} games wanted, {left} left to finish")
        out: list[Episode] = []
        while len(out) < episodes:
            out.extend(self.tick())
        return out

    def cohort(self, games: int, partial: "Partial | None" = None) -> list[Episode]:
        """Deal `games` fresh games and play every one of them to completion.

        This is what a PPO iteration wants and `collect` is not. `collect`
        refills a lane the moment its game ends, so its batch is part
        replacement games and part whatever happened to be mid-game when the
        last one finished — and those unfinished lanes carry across the
        learner's weight sync, which is how a trajectory ends up stitched from
        several policy generations. A cohort re-arms the bounded environment
        from where its counter stands (`LaneEnv.cohort`), so every position it
        returns was played under one set of weights and nothing is left in
        flight.

        It also removes the length bias `deal` exists to prevent: taking the
        first `n` games to finish selects for short ones, and game length is
        not independent of who is winning.

        `lanes` stays free — it is the concurrency, not the cohort. Below
        `games` the lanes refill until the cohort is dealt out and only the
        last wave tails off; at `games` every game starts together and the
        longest one ticks alone at the end. That tail is what a cohort costs.

        With a `partial`, every game is kept there as it finishes, and the
        cohort's indices are written there before the first one is played. A
        partial that already holds a plan is a cohort a crash interrupted: its
        finished games are loaded and only its missing indices are dealt
        (`listed`), so the cohort is the same games it was always going to be.
        """
        if self._deal is not None:
            raise ValueError("a bounded collector deals its one cohort at build time")
        if games < 1:
            raise ValueError("a cohort needs at least one game")
        if self.running:
            raise RuntimeError(
                "the lanes still hold games; a cohort collector must be built "
                "with fill=False and is empty again after every cohort"
            )
        if partial is None:
            return self._cohort(games)
        kept = partial.done()
        plan = partial.plan()
        if plan is None and kept:
            raise ValueError(f"{partial.directory} holds games but no plan")
        held, self.sink = self.sink, partial.keep
        try:
            if plan is None:
                partial.write_plan(self.upcoming(games))
                return self._cohort(games)
            finished = {episode.index for episode in kept}
            return kept + self.listed([i for i in plan if i not in finished])
        finally:
            self.sink = held

    def _cohort(self, games: int) -> list[Episode]:
        if self._env is None:
            # Built on the first cohort rather than at construction, because a
            # `fill=False` collector holds nothing until one is asked for; every
            # cohort after this one re-arms the same environment and keeps its
            # counters.
            self._env = self._new_env(deal=games, first_game=self._next_game)
        else:
            self._env.cohort(games)
        try:
            return self._play_out()
        finally:
            self._next_game = self._env.games_started()

    def upcoming(self, games: int) -> list[int]:
        """The indices the next `cohort(games)` deals, in the order it deals
        them -- a cohort's plan, known before any of it is played."""
        start = self.games_started()
        return [start + k * self.stride for k in range(games)]

    def listed(self, indices: Sequence[int]) -> list[Episode]:
        """Play exactly the games `indices` to completion and return them.

        A resumed cohort's missing games. Each is the same pure function of
        `(seed, index)` it is under any other deal; this collector's own
        lanes and counter are left as they were, so the caller keeps the
        counter past `indices` (`hexn.durable.resume_base`).
        """
        if not indices:
            return []
        held = self._env
        self._env = self._new_env(
            deal=len(indices), first_game=indices[0], indices=indices
        )
        try:
            return self._play_out()
        finally:
            self._env = held

    def in_flight(self) -> Iterator[Game]:
        return iter(()) if self._env is None else self._env.in_flight()

    def pending(self) -> tuple[int, ...]:
        """Actions taken so far in each live lane's unfinished game."""
        return () if self._env is None else self._env.pending()

    def games_started(self) -> int:
        """How many games this collector has dealt out, finished or not.

        Pass it back as `first_game` to carry on where a run left off. The games
        still in flight are lost on a resume — they hold engine state, not data
        — which costs at most `lanes` partial games once per crash.
        """
        return self._next_game if self._env is None else self._env.games_started()
