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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Sequence

import numpy as np

from hexset import contract9
from hexset import decisions as D
from hexset.actions import Action, ActionSpace, mask_of, space_for
from hexset.arena import MAX_ACTIONS, deal_board, deal_game
from hexset.board.board import BOARD_MODES, Board
from hexset.encoding import Observation
from hexset.game import MAX_TURNS, Game, Phase
from hexset.gym.driver import Driver, draw_key, observations, seat_points
from hexset.gym.driver import Request as DriverRequest
# `_Lane` is internal to HexSet (an underscore name, outside `__all__`), used
# knowingly: `RuledLaneEnv` hangs each game's coalition plan through
# `LaneEnv`'s `_fresh`, which a HexSet release may change without notice.
from hexset.gym.lanes import BoardBots, LaneEnv, Outcome, _Lane
from hexset.gym.lanes import Episode as LaneEpisode
from hexset.outlook import outlook_of
from hexset.victory import victory_points
from hexset.record import Record, advance, moves, open_record, shown_by_step
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
    "CONTRACTS",
    "Choice",
    "Collector",
    "Episode",
    "Outcome",
    "RandomPolicy",
    "Request",
    "Transition",
    "Transition9",
    "SAMPLE_STREAMS",
    "draw_key",
    "owned",
]

# Where a sampling policy's randomness comes from. "worker": the policy's own
# generator, consumed in batch order, so a game's sampled actions depend on
# which other games shared its batches -- on the worker count, the lane count
# and the order lanes finish in. "game": every decision carries its own key,
# `draw_key(seed, index, seat, step)`, and a sampling policy draws from that
# key alone, so a game is a pure function of its `(seed, index)` and the
# weights, however the games are sharded or batched.
SAMPLE_STREAMS = ("worker", "game")

# The observation and decision contracts a collector plays (`Collector`'s
# `contract`): "8" encodes `hexset.encoding` and asks for one atomic action a
# decision; "9" encodes `hexset.contract9` and asks the learner for whole
# decisions (`hexset.decisions`, `hexset.gym.lanes.LaneEnv`'s `decisions`).
CONTRACTS = ("8", *contract9.CONTRACTS)

# Every seat's true victory points, board order (`Request.seat_points`).
_seat_points = seat_points


# The phases of the turn under way an outlook label is read in: before the
# roll, moving the robber, and the main phase. Setup has no round to read and
# a discard is another seat's decision mid-roll.
_OUTLOOK_PHASES = frozenset((Phase.ROLL, Phase.ROBBER, Phase.MAIN))
# The share of those decisions labelled. A label costs several times what the
# rest of a decision's collection does (batched across lanes), so labelling
# them all would slow collection several-fold; a tenth adds about a tenth and
# still labels far more rows an iteration than the head needs. Which ones is a
# pure function of the decision (`_outlook_kept`), so it does not depend on
# lanes or workers.
OUTLOOK_SHARE = 0.10
_OUTLOOK_SALT = 0x6F75746C6F6F6B21  # keeps the choice apart from `draw_key`'s draws


def _outlook_kept(seed: int, index: int, seat: int, step: int) -> bool:
    return (draw_key(seed ^ _OUTLOOK_SALT, index, seat, step) >> 11) < OUTLOOK_SHARE * (1 << 53)


def _outlook(game: Game, seat: int) -> tuple[float, ...]:
    """Every seat's `hexset.outlook.Outlook.can` on the true state, board order
    (`Request.outlook`), or empty where the decision is not the turn's own: a
    seat other than the one on turn, or a phase outside `_OUTLOOK_PHASES`."""
    if seat != game.current_player or game.phase not in _OUTLOOK_PHASES:
        return ()
    # A knight played before the roll moves the robber with the roll still to
    # come: `resume_phase` is ROLL then.
    rolled = not (game.phase is Phase.ROLL or (game.phase is Phase.ROBBER and game.resume_phase is Phase.ROLL))
    return outlook_of(game.state(0, hidden=False), seat, rolled=rolled, card_played=game.dev_card_played).can

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
    # Every seat's true victory points at this decision, board order, hidden VP
    # cards included: a training target only (`hexn.rewards.points_next_round`),
    # never an input, so reading the truth here leaks nothing to the policy.
    seat_points: tuple[int, ...] = ()
    # Every seat's `hexset.outlook.Outlook.can` at this decision, board order,
    # read on the true state: a training target only (`--aux-target outlook`),
    # never an input. Empty unless the collector reads it (`Collector.outlook`),
    # the decision is the turn's own (`_outlook`) and among the `OUTLOOK_SHARE`
    # kept.
    outlook: tuple[float, ...] = ()
    # This decision's sampling key under per-game streams (`draw_key`), None
    # under the policy's own generator. A sampling policy that is handed keys
    # draws from them alone; a greedy or scripted one ignores them.
    draw: int | None = None
    # Contract 9: the decision as `hexset.decisions.tensors` states it, where
    # `observation` is the `hexset.contract9.flatten`ed row, `mask` the
    # decision's flat block and `options` empty. None under contract 8.
    decision: D.Decision | None = None
    # `Game.turns` at this decision: a training target's clock only
    # (`hexn.ppo.zero_sum_targets`), read under contract 9.
    turns: int = 0
    # Contract 9: the run of options this request is a part of and its place
    # there (`hexset.gym.lanes.Request.decision_id`, `.part`).
    decision_id: int = 0
    part: int = 0


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

    action: Action | D.Compound
    log_prob: float = 0.0
    value: tuple[float, ...] = ()
    aux: object = None
    # Contract 9: where the decision taken sits in its `Request.decision`,
    # `(hexset.decisions.DecisionKind, index)` as `hexset.decisions.locate`
    # gives it, and the points head's reading, the mover first.
    decision: tuple[int, int] | None = None
    points: tuple[float, ...] = ()


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
    # Every seat's, board order (`Request.seat_points`); empty in games
    # collected before it was recorded.
    seat_points: tuple[int, ...] = ()
    # `Request.outlook`; empty where it was not read.
    outlook: tuple[float, ...] = ()


@dataclass(frozen=True)
class Transition9:
    """One contract-9 decision, filed under the seat that made it: the
    `Transition` of a whole decision (`hexset.decisions`).

    `step` is the decision's first move in the game's action stream (a
    compound's later parts are steps of their own in the record, and no row
    here). `observation` is the `hexset.contract9.flatten`ed row and
    `decision` the decision's tensors; `kind`/`index` locate the option taken
    in them (`hexset.decisions.locate`), which is all the update needs to
    rebuild its log-probability. `value` and `points` are the win and points
    heads as the behaviour policy read them, the mover first; `seat_points`
    every seat's true victory points, board order, and `turns` the game's turn
    count, at this decision: training targets only, never inputs.

    A row is one part of a run of options (`hexset.decisions.closes`):
    `decision_id` names the run within its game (`hexset.gym.lanes`), `part`
    the row's place in it, 0 the first. The update credits a run as one
    decision, its log-probability the sum of its parts'
    (`hexn.ppo.prepare.prepare_columns9`)."""

    seat: int
    step: int
    observation: np.ndarray
    decision: D.Decision
    kind: int
    index: int
    log_prob: float
    value: tuple[float, ...]
    points: tuple[float, ...] = ()
    seat_points: tuple[int, ...] = ()
    turns: int = 0
    decision_id: int = 0
    part: int = 0


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

    `marks` is `(step, turns, seat_points)` before every move of a seat the
    collector did not record, read off `record` (`unrecorded_points`), when
    the collector was asked for them (`Collector`'s `marks`): what lets the
    points head's zero-sum target see each change in points when it happened
    at a table the learner shares (`hexn.ppo.prepare.prepare_columns9`).
    Empty in self-play, where every move is a row, and unless asked for.
    """

    index: int
    seed: int
    players: int
    trajectories: tuple[tuple[Transition, ...], ...]
    outcome: Outcome
    cast: tuple[int, ...] = ()
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    record: Record | None = None
    marks: tuple[tuple[int, int, tuple[int, ...]], ...] = ()

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
                marks=episode.marks,
            )
        )
    return out


def unrecorded_points(record: Record, steps: Sequence[int]) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """`(step, turns, seat_points)` before each of `steps`, in step order,
    off one walk of `record` (`Episode.marks`): the same clock and true
    points a recorded row carries (`Request.turns`, `Request.seat_points`)."""
    wanted = set(steps)
    if not wanted:
        return ()
    last = max(wanted)
    game = open_record(record)
    shown = shown_by_step(record)
    out = []
    for step, (actor, action, trades) in enumerate(moves(record)):
        if step in wanted:
            out.append((step, game.turns, _seat_points(game)))
        if step == last:
            break
        advance(game, action, trades, actor, shown.get(step, ()))
    if len(out) != len(wanted):
        raise ValueError(f"the record holds {len(out)} of the {len(wanted)} steps asked for")
    return tuple(out)


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
        from hexset.gym.distribution import Deal

        mode = kwargs.get("board_mode", "random")
        if kwargs.get("deal_law") is not None:
            raise ValueError("a legacy game_law cannot be combined with deal_law")
        kwargs["deal_law"] = lambda index: Deal(*self.game_law(index), board_mode=mode)
        super().__init__(*args, **kwargs)

    def _fresh(self) -> _Lane | None:
        lane = super()._fresh()
        if lane is not None and self.coalition is not None:
            plan = self.coalition(lane.index)
            if plan is not None:
                lane.game.coalition = plan
        return lane


def standard_law(index: int) -> tuple[GameType, frozenset[int]]:
    """The game law every run before `duel(...)` played: the standard game
    at every seat, nobody retired."""
    return STANDARD_GAME, frozenset()


class _Seat:
    """One policy id of a `Collector` as HexSet's driver seats it
    (`hexset.gym.driver.Seat`): the collector builds the policy's requests,
    asks it once a tick and holds its learner rows."""

    def __init__(self, collector: "Collector", pid: int) -> None:
        self.collector, self.pid = collector, pid

    @property
    def native_contract(self) -> str | None:
        """The collector's contract where its policy reads only a native
        lane's buffers (`Collector._native_ready`), else None."""
        return self.collector.contract if self.collector._native_ready() else None

    def choose_many(self, requests: Sequence[DriverRequest]) -> list[Choice]:
        return self.collector._answer(self.pid, requests)


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
        deal_law: Callable | None = None,
        coalition: Callable[[int], object] | None = None,
        sample_stream: str = "worker",
        outlook: bool = False,
        marks: bool = False,
        contract: str = "8",
        discard_order: str = "seat",
        board_mode: str = "random",
        keep: "Partial | None" = None,
    ) -> None:
        """`keep` is where every game is written as it finishes
        (`hexn.durable.Partial`, each game's file holding its
        `hexset.record.Record`), unless a call names its own `partial`
        (`collect`, `cohort`). A collector plays no game with nowhere to keep
        it: `tick` refuses.

        `first_game` is where the game counter starts.

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

        `sample_stream` is `SAMPLE_STREAMS`' choice: "game" hands every request
        its `draw_key`, so a game's sampled actions no longer depend on
        `lanes`, `stride` or which games shared a batch.

        `outlook` reads `Request.outlook`, the `--aux-target outlook` label, on
        `OUTLOOK_SHARE` of the turn's own decisions; off, it costs nothing.

        `marks` fills `Episode.marks` for a game a scripted opponent sat in,
        one walk of its record as it finishes; off, it costs nothing.

        `contract` is `CONTRACTS`' choice. Under "9" the learner (policy 0) is
        asked for whole decisions and files a `Transition9` per decision;
        beside it sit scripted bots only, which act a move at a time.
        `discard_order` is `hexset.gym.lanes.DISCARD_ORDERS`' choice.
        `board_mode` names how each game's board is dealt
        (`hexset.board.board.BOARD_MODES`); a `deal_law` names its own per
        game, so the two do not combine.

        `sink`, when set, is handed every episode the tick it finishes -- what
        keeps a game on disk before the batch it belongs to ends
        (`hexn.durable.Partial.keep`). A sink that returns something other
        than None replaces the episode with it in everything this collector
        returns (`hexn.collect`'s workers hand back the encoded game). `beat`, when set, is called once a
        tick, so a process watching this one can tell slow from wedged.
        """
        if lanes < 1:
            raise ValueError("a collector needs at least one lane")
        if deal is not None and deal < 1:
            raise ValueError("a collector cannot be asked to deal nothing")
        if stride < 1:
            raise ValueError("a collector cannot deal backwards or stand still")
        if sample_stream not in SAMPLE_STREAMS:
            raise ValueError(f"sample_stream must be one of {SAMPLE_STREAMS}, got {sample_stream!r}")
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}, got {contract!r}")
        if contract in contract9.CONTRACTS:
            if any(not isinstance(o, BoardBots) for o in opponents) or tuple(learners) != (0,):
                raise ValueError("a contract-9 collector seats scripted bots beside its one learner; "
                                 "another network would be asked a move at a time")
            if outlook or temperatures is not None:
                raise ValueError("a contract-9 collector reads no outlook and plays every seat at 1.0")
        if caster is not None and not opponents:
            raise ValueError("a caster without opponents has nobody to cast")
        if board_mode not in BOARD_MODES:
            raise ValueError(f"unknown board mode {board_mode!r}: {sorted(BOARD_MODES)}")
        if board_mode != "random" and (deal_law is not None or pair_boards or board is not None):
            raise ValueError("board_mode deals each game's own board; a deal_law, pair_boards or a fixed "
                             "board= decides the board itself")
        if pair_boards and board is not None:
            raise ValueError(
                "pair_boards deals each pair its own shared board; a fixed "
                "board= would put every game on one board and the pairing "
                "would compare nothing"
            )
        self.policy = policy
        self.contract = contract
        self.discard_order = discard_order
        self.board_mode = board_mode
        self.sample_stream = sample_stream
        self.outlook = outlook
        self.marks = marks
        self.opponents = tuple(opponents)
        self.caster = caster
        # Game index -> per-seat sampling temperature, pure in the index like
        # the caster it accompanies; None means every seat plays at 1.0.
        self.temperatures = temperatures
        # Game index -> (game type, retired seats), pure in the index like the
        # caster (`hexn.collect.mix_deal`); None deals the standard game at
        # every seat, as every run before `duel(...)` did.
        self.game_law = game_law
        if deal_law is not None and (game_law is not None or pair_boards):
            raise ValueError("deal_law cannot be combined with game_law or pair_boards")
        self.deal_law = deal_law
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
        self.sink: Callable[[Episode], object] | None = None if keep is None else keep.keep
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

        # A learner's decision between its answer and its apply, keyed
        # `(index, seat, step)`, and each game's rows so far, per seat.
        self._pending: dict[tuple[int, int, int], tuple[Request, Choice]] = {}
        self._rows: dict[int, list[list[Transition | Transition9]]] = {}
        # Who answers each policy id at HexSet's driver (`hexset.gym.driver`).
        self._seats = {pid: _Seat(self, pid) for pid in self._batched}
        self._driver: Driver | None = None
        # The partial a `cohort` call named, while it plays.
        self._keeping: "Partial | None" = None
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
            extra = {**extra, "indices": indices}
        return env(
            self.players,
            self.seed,
            self.lanes,
            deal=deal,
            action_cap=self.action_cap,
            turn_cap=self.turn_cap,
            board=self._paired_board if self.pair_boards else self.board,
            deal_law=self.deal_law,
            caster=None if self.caster is None else self._cast,
            # The bench itself: the one a caller built
            # (`hexn.collect.named_opponent`) is the one that must hold the
            # bots, so the capacity it chose is the capacity that applies, a
            # bot survives a cohort boundary, and a bot that reads the table
            # is built per seat by its `seat_spawn` (`BoardBots.fresh`).
            bots=dict(self._benches),
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
            discard_order=self.discard_order,
            board_mode=self.board_mode,
            decisions=(0,) if self.contract in contract9.CONTRACTS else (),
            **extra,
        )

    # --- the tick ----------------------------------------------------------
    #
    # HexSet's driver (`hexset.gym.driver.Driver`) runs the loop: it deals,
    # asks each policy id's `_Seat` once a tick for its rows, applies the
    # answers on Python or native lanes and hands back each finished game.
    # What is left here is the trainer's half: the requests a `BatchPolicy`
    # reads, a row per recorded decision, and the episode they make.

    def _requests(self, asked: Sequence[DriverRequest]) -> list[Request]:
        """The policies' requests for the driver's rows, observations through
        the vectorized encoder (or a native lane's own buffer).

        Only the seats a `BatchPolicy` answers are encoded: a bot-played seat is
        the environment's to answer and never sees an observation, so encoding
        it would pay for the opponents as well as the learner.
        """
        if not asked:
            # Every seat this tick is played by a bot inside the environment.
            return []
        if self.contract in contract9.CONTRACTS:
            return self._requests9(asked)
        encoded = self._encode8(asked)
        targets = self._driver.targets()
        keyed = self.sample_stream == "game"
        return [
            Request(
                lane=request.lane,
                seat=request.seat,
                observation=observation,
                mask=np.asarray(mask_of(self.space, request.options), dtype=bool),
                options=request.options,
                game=request.game,
                points=victory_points(request.game.state(request.seat, hidden=False), request.seat),
                seat_points=targets.points[request.row],
                outlook=(
                    _outlook(request.game, request.seat)
                    if self.outlook and _outlook_kept(self.seed, request.index, request.seat, request.step)
                    else ()
                ),
                temperature=(
                    1.0
                    if self.temperatures is None
                    else float(self.temperatures(request.index)[request.seat])
                ),
                draw=request.draw if keyed else None,
            )
            for request, observation in zip(asked, encoded, strict=True)
        ]

    def _encode8(self, asked: Sequence[DriverRequest]) -> list:
        """A contract-8 tick's observations, each under the observation
        contract its seat's policy reads (`NetworkPolicy.observation_contract`:
        "8", or "8.1" for a net trained across rule sets), one batch a contract."""
        contracts = [getattr(self._batched.get(r.policy, self.policy), "observation_contract", "8") for r in asked]
        encoded: list = [None] * len(asked)
        for contract in sorted(set(contracts)):
            rows = [i for i, c in enumerate(contracts) if c == contract]
            for i, observation in zip(rows, observations([asked[i] for i in rows], contract)):
                encoded[i] = observation
        return encoded

    def _requests9(self, asked: Sequence[DriverRequest]) -> list[Request]:
        """Contract 9's tick: every learner decision as `hexset.contract9`
        encodes it, with the decision's tensors the environment stated."""
        keyed = self.sample_stream == "game"
        targets = self._driver.targets()
        encoded = observations(asked, self.contract)
        out = []
        for request, observation in zip(asked, encoded):
            # The true state's points, every seat's and so the mover's own.
            seat_points = targets.points[request.row]
            out.append(Request(
                lane=request.lane,
                seat=request.seat,
                observation=observation,
                mask=request.decision.flat,
                options=(),
                game=request.game,
                points=seat_points[request.seat],
                seat_points=seat_points,
                draw=request.draw if keyed else None,
                decision=request.decision,
                turns=targets.turns[request.row],
                decision_id=request.decision_id,
                part=request.part,
            ))
        return out

    def _answer(self, pid: int, asked: Sequence[DriverRequest]) -> list[Choice]:
        """Policy `pid`'s answers to its rows of a tick, each learner
        decision held until the driver applies it (`_decided`)."""
        requests = self._requests(asked)
        choices = list(self._batched[pid].act(requests))
        if len(choices) != len(requests):
            raise ValueError(f"policy answered {len(choices)} of {len(requests)} requests")
        if pid in self._learners:
            for row, request, choice in zip(asked, requests, choices):
                self._pending[(row.index, row.seat, row.step)] = (request, choice)
        return choices

    def _decided(self, asked: DriverRequest, choice: Choice) -> None:
        """A decision the driver applied: a learner's becomes its seat's row."""
        held = self._pending.pop((asked.index, asked.seat, asked.step), None)
        if held is None:
            return
        request, choice = held
        rows = self._rows.get(asked.index)
        if rows is None:
            rows = self._rows[asked.index] = [[] for _ in range(self.players)]
        if request.decision is not None:
            kind, index = choice.decision
            rows[asked.seat].append(Transition9(
                seat=asked.seat,
                step=asked.step,
                observation=request.observation,
                decision=request.decision,
                kind=int(kind),
                index=int(index),
                log_prob=choice.log_prob,
                value=tuple(choice.value),
                points=tuple(choice.points),
                seat_points=request.seat_points,
                turns=request.turns,
                decision_id=request.decision_id,
                part=request.part,
            ))
            return
        rows[asked.seat].append(Transition(
            seat=asked.seat,
            step=asked.step,
            observation=request.observation,
            mask=request.mask,
            action=choice.action,
            index=self.space.index(choice.action),
            log_prob=choice.log_prob,
            value=tuple(choice.value),
            aux=choice.aux,
            points=request.points,
            seat_points=request.seat_points,
            outlook=request.outlook,
        ))

    def _kept(self, played: LaneEpisode) -> object:
        """A finished game made the trainer's `Episode` and handed to the
        sink, which keeps it before the lane is dealt again."""
        episode = self._episode(played)
        # What the sink hands back, when it hands anything back, is what this
        # collector holds and returns in the episode's place: a collector
        # worker keeps each game encoded, not as objects.
        kept = self.sink(episode)
        return episode if kept is None else kept

    def _episode(self, played: LaneEpisode) -> Episode:
        """The engine's episode with the trainer's rows joined onto it: each
        seat's rows in the order its decisions were taken. A seat the
        collector did not record -- an opponent's -- stays empty."""
        rows = self._rows.pop(played.index, None)
        trajectories = tuple(() for _ in range(played.players)) if rows is None else tuple(map(tuple, rows))
        return Episode(
            index=played.index,
            seed=played.seed,
            players=played.players,
            trajectories=trajectories,
            # An uncast collector reports no cast, the convention every reader
            # already has ("empty means everyone was the learner"); `LaneEnv`
            # always stamps one because it has no such convention.
            cast=played.cast if self.caster is not None else (),
            trades=played.trades,
            record=played.record,
            outcome=played.outcome,
            marks=self._marks(played),
        )

    def _marks(self, played: LaneEpisode) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        """`Episode.marks`: every scripted seat's moves, if asked for."""
        if not self.marks:
            return ()
        steps = [d.step for decisions in played.decisions for d in decisions if d.policy in self._benches]
        if steps and played.record is None:
            raise ValueError(f"game {played.index} has no record to read its unrecorded moves' points off")
        return unrecorded_points(played.record, steps) if steps else ()

    def _native_ready(self) -> bool:
        """Whether this collector's games play on native lanes: the native
        extension, a whole-decision learner on every seat (a contract-9 net
        that asks the double turn in parts) and nothing a Python game is
        needed for -- no other policy, caster, coalition, temperature, outlook
        or marks, no trades, and per-game sample streams."""
        from hexset.gym.native import available

        if not (
            available()
            and self.contract in contract9.CONTRACTS
            and self.players in (2, 4)
            and self.caster is None
            and not self.opponents
            and self.coalition is None
            and self.temperatures is None
            and not self.outlook
            and not self.marks
            and self.sample_stream == "game"
            and self.max_offers == 0
            and self.trader is None
        ):
            return False
        # Imported only here: `hexn.policy9` needs torch, and this module does not.
        from .policy9 import NetworkPolicy9

        return all(type(policy) is NetworkPolicy9 and policy.net.config.double_turn == "parts"
                   for policy in self._batched.values())

    def _drive(self, env: LaneEnv, *, keeping: "Partial | None" = None) -> Driver:
        """HexSet's driver over `env`: native lanes where `_native_ready`, each
        game checked against `keeping`'s cohort plan when there is one."""
        if keeping is None:
            keeping = getattr(self.sink, "partial", None)
        native = env._stop is not None and self._native_ready()
        plan = None if keeping is None or not native else keeping.plan()
        finished = () if keeping is None or not native else keeping.finished()
        self._driver = Driver(env, self._seats, keep=self._kept, on_decision=self._decided,
                              lanes="native" if native else "python", planned=plan, finished=finished,
                              beat=self.beat)
        if native and getattr(self._driver, "space", None) is not None:
            self._native_topology(self._driver.space, self._driver.pairs)
        return self._driver

    def _native_topology(self, space: ActionSpace, pairs) -> None:
        """Prime each policy's setup incidences for native requests, which
        carry no game to read them off, checked against the net's own."""
        pair_array = np.asarray(pairs)
        for policy in self._batched.values():
            if policy.space != space or policy.net.CONTRACT != self.contract:
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

    def requests(self) -> list[Request]:
        """This tick's decisions for the policies this collector drives.

        The trainer-side view of the driver's requests, in lane order, with the
        bot-played seats left out because the environment answers those itself.
        Same decisions until the tick is stepped, though the encoding is redone
        each call, so this is for inspecting a batch, not a way to avoid `tick`.
        """
        if self._env is None:
            return []
        driver = self._driver if self._driver is not None and self._driver.env is self._env else self._drive(self._env)
        return self._requests(driver.requests())

    def _keeps(self) -> None:
        if self.sink is None:
            raise RuntimeError(
                "a collector keeps every game it plays: build it with keep=<hexn.durable.Partial> "
                "or name a partial for the call"
            )

    def tick(self) -> list[Episode]:
        """Step every live lane once. Returns the games that ended on this tick,
        each already written where this collector keeps its games."""
        self._keeps()
        driver = self._driver
        if driver is None or driver.env is not self._env:
            if self._env is None:
                return []
            driver = self._drive(self._env)
        steps = driver.steps
        # The sink and the beat as they stand this tick: a call may name its
        # own partial (`collect`, `cohort`).
        driver.beat = self.beat
        episodes = driver.tick()
        if self._pending:
            raise RuntimeError("the driver applied answers without their pending training rows")
        self.ticks += 1
        self.steps += driver.steps - steps
        self.games += len(episodes)
        return episodes

    # --- driving -----------------------------------------------------------

    @property
    def running(self) -> bool:
        """False once a bounded collector has played out everything it dealt."""
        if self._driver is not None and self._driver.env is self._env and self._driver.lanes == "native":
            return self._driver.running
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
        held, self.sink, self._keeping = self.sink, partial.keep, partial
        try:
            if plan is None:
                partial.write_plan(self.upcoming(games))
                return self._cohort(games)
            finished = {episode.index for episode in kept}
            return kept + self.listed([i for i in plan if i not in finished])
        finally:
            self.sink, self._keeping = held, None

    def _cohort(self, games: int) -> list[Episode]:
        self._keeps()
        if self._env is None:
            # Built on the first cohort rather than at construction, because a
            # `fill=False` collector holds nothing until one is asked for; every
            # cohort after this one re-arms the same environment and keeps its
            # counters.
            self._env = self._new_env(deal=games, first_game=self._next_game)
        else:
            self._env.cohort(games)
        try:
            self._drive(self._env, keeping=self._keeping)
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
        self._keeps()
        held, driver = self._env, self._driver
        self._env = self._new_env(
            deal=len(indices), first_game=indices[0], indices=indices
        )
        try:
            self._drive(self._env, keeping=self._keeping)
            return self._play_out()
        finally:
            self._env, self._driver = held, driver

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
