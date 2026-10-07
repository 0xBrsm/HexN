# SPDX-License-Identifier: GPL-3.0-only
"""Parallel self-play collection: the torch-free Collector, sharded.

A single collector process leaves the box's other cores idle, because
everything a tick does — `legal_actions`, `encode`, the engine step, a
scripted opponent's one-ply search — is Python compute holding the GIL. That
rules out the usual vectorised-env fan-out, a thread pool over native engine
subprocesses whose `step()` blocks on pipe I/O and releases the GIL; ours
never releases it, so the shard has to be a process.

Each worker owns a slice of the lanes and a CPU copy of the network, and
plays its games end to end: worker `w` of `K` deals game indices
`base + w, base + w + K, ...` (the Collector's `stride`), so the workers'
index sets are disjoint by construction, every game stays the same pure
function of `(seed, index)` it always was, and the caster — also pure in the
index — casts identically at any worker count. Inference stays per-worker on
CPU: at the batch sizes one worker's slice of lanes reaches, per-worker CPU
inference beats moving those lanes' observations to a shared GPU and back.
The GPU stays free for the update, and the learner re-syncs weights to every
worker once per iteration.

What crosses processes is one weights dict per iteration going out and
finished episodes coming back — never per-tick observations. The other shape
considered, SampleFactory-style central inference, was rejected for this
model: it reintroduces a global tick barrier plus per-tick traffic over the
pipes, to buy GPU batching that only pays above the batch sizes a shard sees.

On a resume, `games_started()` reports the *max* over the workers' counters.
Indices between the slowest and fastest worker's next deal are skipped rather
than replayed — unused seeds, not lost games — and the same rule makes a
resume safe across a change in worker count.

**A worker keeps each game on disk the moment it finishes.** Handed a
`hexn.durable.Partial`, it writes every finished episode there before the
cohort is done, so a crash mid-iteration loses only the games still in
flight; the resumed iteration loads the rest and deals only the missing
indices. The parent never waits on a pipe blindly: it polls, and a worker
that has died, or has gone `STALL_SECONDS` without a tick, fails the run
with a sentence rather than hanging it. While it waits it writes a
`heartbeat.json` saying what each worker is doing.
"""

from __future__ import annotations

import ctypes
import gc
import json
import multiprocessing as mp
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

import numpy as np

from hexset.actions import build_space
from hexset.board.board import Board, random_base_board
from hexset.encoding import Observation, static_graph
# Re-exported under its own name: `hexn.loop` and the run configs say
# `collect.alternating`, and the law itself is HexSet's.
from hexset.casting import alternating  # noqa: F401
from hexset.casting import league_rotation, paired
from hexset.gym.lanes import BoardBots
from . import runtime
from .durable import Partial, write_atomic
from .model import HexNet, ModelConfig, Packing, packing
from .policy import NetworkPolicy
from .selfplay import BatchPolicy, Collector, Episode, Transition

# How long a busy worker may go without a tick before it counts as wedged. A
# tick is one decision per lane, searched or not, so this is generous: what it
# catches is a game stuck inside one engine step, which no action cap ends.
STALL_SECONDS = 1800.0
# How often the parent looks up from a pipe to check on its workers.
POLL_SECONDS = 5.0


def frozen(
    path: str,
    device: str,
    board: Board,
    players: int,
    *,
    greedy: bool = True,
    generator: torch.Generator | None = None,
) -> NetworkPolicy:
    """A checkpoint as a fixed policy — a ladder rung or a lane opponent.

    Greedy argmax by default, so these numbers stay comparable with the
    arena's `network:` entrants; `greedy=False` is a `sampled:` pool member
    (`SAMPLED`), drawing from its own `generator`. Either way `.eval()` with
    no optimiser anywhere means nothing here can train. Width, rounds and the
    head shapes come from the checkpoint's own recorded args, not the run's,
    so a differently-sized or differently-shaped parent still loads.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    stored = state.get("args", {})
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    graph = static_graph(topology)
    net = HexNet(
        space,
        graph,
        players,
        ModelConfig(
            width=int(stored.get("width", 64)),
            rounds=int(stored.get("rounds", 2)),
            value_head=str(stored.get("value_head", "linear")),
            policy_head=str(stored.get("policy_head", "linear")),
            quantiles=int(stored.get("quantiles", 32)),
            fast_win=bool(stored.get("fast_win")),
        ),
    ).to(device)
    net.load_state_dict(state["net"])
    net.eval()
    return NetworkPolicy(
        net,
        space,
        packing(graph, players),
        device=device,
        greedy=greedy,
        generator=generator,
    )


def named_opponent(name: str, seed: int, lanes: int) -> BoardBots:
    """Any arena entrant by name as a lane or ladder opponent.

    Routed through `hexset.arena.spawn` rather than constructing the bot here, so
    a rung is *literally* the entrant the arena scores, not a fresh
    reimplementation that happens to share a name.

    **A bot resolves here only once its runtime is loaded.** HexSet ships
    no playing bots: `heximax` and `rehex` are registered with `hexset.arena`
    by the runtime module that carries them (`register_preset`), so a
    process that never loaded it fails `--mix heximax=...` with `unknown
    entrant`. The trainers load their `--runtime` at start, and every worker
    loads `WorkerSpec.runtime` before it builds a lane opponent
    (`hexn.runtime`); `test_heximax_resolves_as_a_mix_opponent` pins the
    resolution.

    The import is function-scoped because `hexset.arena` reaches for torch on some
    entrant kinds and `hexn.selfplay` imports this module; the collector is
    tested and timed where torch cannot be installed.
    """
    from hexset.arena import entrant_from_name, spawn

    try:
        entrant = entrant_from_name(name)
    except KeyError:
        # Matching `--mix`'s style: a mistyped rung should fail with a sentence
        # before the run starts, not a bare KeyError traceback out of a lambda.
        raise SystemExit(f"unknown entrant: {name}") from None
    return BoardBots(
        lambda board: spawn(entrant, board, random.Random(seed)),
        capacity=max(256, 2 * lanes),
    )


# `--mix` names that are not arena entrants, and must never become them.
#
# Both predate `named_opponent`. A lane opponent whose strength moved
# silently would invalidate every run that used it, without changing a line
# of that run's own configuration. So they are resolved here, by name,
# exactly as they were before an entrant spec was accepted at all.
#
# `greedy` was the third reserved name and is gone: HexSet 0.47.0 deleted
# `search2` and with it `hexset.bots.greedy`, so there is no bot to resolve it
# to. It is not re-pointed at another bot -- a mix name that silently changed
# which opponent a run trained against is exactly what this list exists to
# prevent -- so `--mix greedy=...` is now an unknown name, and a recipe that
# wants a handcrafted opponent says `heximax` and means it.
RESERVED_MIX = ("parent",)

# The learner itself as a *table pool* member, and only there. A table whose
# pool holds `self` seats the learner in its drawn seat and, independently, in
# every other seat the draw lands on it -- one to four learner seats a game,
# every one of them trained on (they are all cast id 0) -- so a fully-pooled
# recipe (`table(self|self|parent|...)=1.0`) keeps self-play's position volume
# while every game's table is a fresh mix of copies and foreign styles. Repeat
# it to weight it, as any pool member. It is refused as a plain entry: a plain
# `self=` would seat the learner against itself on alternating pairs, which is
# self-play under another name and would put a non-opponent in the id law.
SELF = "self"

# `self~0.3` is `self` played at a sampling temperature drawn per seat per game
# from `[1 - 0.3, 1 + 0.3]` -- the learner's pool-drawn copies made to explore,
# so the seat it trains from meets tablemates that build, hold and trade a
# little unlike itself, while the seat the table draw seated first stays at
# 1.0 as the anchor. Every tempered seat is still cast id 0 and still trained
# on: `NetworkPolicy.act` records the log-probability under the tempered
# distribution, so the PPO ratio against it is exact and the clip bounds how
# far a tempered seat can pull. Bare `self` is `self~0`. Off unless asked for;
# a pool without a band draws no extra randomness, so its casts are unchanged.
SELF_BAND = "~"

# `sampled:network:<path>` is a bare `network:` member that samples its
# policy at temperature 1.0 instead of taking the argmax -- the distribution
# it was trained on, which for an RL net is its own behaviour policy. A
# greedy checkpoint answers a position the same way every time, which the
# learner can memorise and exploit; a sampled one cannot be read that way. Its sampling stream is
# seeded per worker and per entry (`mix_opponents`). Only a bare
# `network:` spec qualifies: a scripted bot or an `@trades` override has no
# policy to sample.
SAMPLED = "sampled:"


def sampled_spec(name: str) -> str | None:
    """The `network:` spec a `sampled:` member plays, None for any other
    name."""
    return name[len(SAMPLED):] if name.startswith(SAMPLED) else None


def _bare_network(name: str):
    """The entrant a bare `network:<path>` names -- one `frozen` can play --
    or None for any other name."""
    from hexset.arena import entrant_from_name

    try:
        entrant = entrant_from_name(name)
    except (KeyError, ValueError):
        return None
    if entrant.kind == "network" and entrant.trade is None and isinstance(entrant.weights, str):
        return entrant
    return None


def self_band(member: str) -> float | None:
    """The temperature half-width of a `self` pool member, None for any other
    name. Bare `self` is 0.0."""
    if member == SELF:
        return 0.0
    if not member.startswith(SELF + SELF_BAND):
        return None
    try:
        band = float(member[len(SELF) + len(SELF_BAND):])
    except ValueError:
        raise ValueError(f"a self temperature band must be a number: {member!r}") from None
    if not 0.0 <= band < 1.0:
        raise ValueError(f"a self temperature band must lie in [0, 1): {member!r}")
    return band


def tempered(mix: Sequence[tuple[str, float]]) -> bool:
    """Whether any table pool in `mix` seats a `self~band` with a band above 0."""
    return any(
        (self_band(member) or 0.0) > 0.0
        for name, _ in mix
        for member in (table_pool(name) or ())
    )


def mix_opponents(
    mix: Sequence[tuple[str, float]],
    *,
    seed: int,
    lanes: int,
    parent: Callable[[], BatchPolicy] | None = None,
    board: Board | None = None,
    players: int = 4,
    device: str = "cpu",
    torch_seed: int | None = None,
) -> list[BatchPolicy | BoardBots]:
    """The `--mix` names as lane opponents, in caster id order.

    Id `k + 1` is `mix[k]` — the contract `mixed_caster` casts against and
    `Collector._answers` dispatches on — so this list's order is load-bearing.

    Reserved names resolve as they always have (see `RESERVED_MIX`). A bare
    `network:<path>` (no `@trades` override) is an RL-only pool member with
    no trading stance of its own to preserve, so it is routed through
    `frozen` — the same batched `NetworkPolicy` a ladder rung uses — rather
    than `named_opponent`'s `hexset.arena.spawn`, whose `NetworkBot` answers
    one position at a time: a batch-of-one pays the network's whole dispatch
    toll per lane per tick, exactly the per-iteration GPU cost batching
    through `frozen` exists to avoid. `network:<path>@0`, HexSet's no-trade
    spelling, carries its own `TradeParams` (`Entrant.trade`, applied at
    `hexset.arena.spawn` time) — `frozen`'s policy has no such override and
    would silently lose it, so only the bare form qualifies. Every other name is an arena entrant spec resolved by
    `named_opponent`, which is what makes `heximax` or `mcts:<ckpt>@64` a
    training opponent without a branch here per bot, and makes the opponent
    a run trained against *literally* the entrant the arena scores.

    `parent` arrives as a thunk rather than a policy because the trainer has
    already loaded that checkpoint for its ladder rung and would hand the same
    object back, while a worker has to build its own on its own board — and a
    `torch.load` of a checkpoint nothing casts is not free.

    `board`/`players`/`device` are only read for a bare `network:` entry —
    `frozen` needs a topology and a device to build its net on, neither of
    which `named_opponent`'s arena spawn requires (it takes the board at
    spawn time, one bot per lane's own board). A bare entry with no `board`
    falls back to `named_opponent` too, so a caller that never passes one
    (there are none left in this tree) keeps today's behaviour.

    `torch_seed`, a worker's own, seeds a `sampled:` member's stream, so two
    workers sharing the board seed do not sample in lockstep; without it the
    stream falls back to the entry's `seed`.
    """
    out: list[BatchPolicy | BoardBots] = []
    for k, name in enumerate(mix_names(mix)):
        if name == "parent":
            if parent is None:
                raise ValueError("the 'parent' mix opponent needs a parent checkpoint")
            out.append(parent())
        else:
            # A stream per entry, so two entrants in one mix do not break
            # their tie-breaks in lockstep. 77 stays where it was.
            out.append(
                rung_opponent(
                    name,
                    seed=seed + 700 + k,
                    lanes=lanes,
                    board=board,
                    players=players,
                    device=device,
                    sample_seed=None if torch_seed is None else torch_seed + 700 + k,
                )
            )
    return out


def rung_opponent(
    name: str,
    *,
    seed: int,
    lanes: int,
    board: Board | None,
    players: int,
    device: str,
    sample_seed: int | None = None,
) -> BatchPolicy | BoardBots:
    """One named opponent, batched when it can be: a bare `network:<path>`
    is `frozen` -- one forward per tick for every lane, and bargaining under
    the run's own offer budget like every other network seat -- and every
    other name is `named_opponent`'s arena entrant. A `sampled:` member
    (`SAMPLED`) is the same `frozen` policy sampling from a generator seeded
    by `sample_seed`, or `seed` without one. Shared by `mix_opponents` and
    the PPO ladder's `--search-rung`, so a network means the same thing as a
    lane opponent and as a rung."""
    inner = sampled_spec(name)
    if inner is not None:
        entrant = _bare_network(inner)
        if entrant is None or board is None:
            raise ValueError(f"{SAMPLED} needs a bare network:<path> and a board: {name}")
        return frozen(
            entrant.weights,
            device,
            board,
            players,
            greedy=False,
            generator=torch.Generator().manual_seed(
                seed if sample_seed is None else sample_seed
            ),
        )
    entrant = _bare_network(name)
    if entrant is not None and board is not None:
        return frozen(entrant.weights, device, board, players)
    return named_opponent(name, seed, lanes)


def check_mix(mix: Sequence[tuple[str, float]], *, have_parent: bool) -> None:
    """Refuse an unusable `--mix` before the run starts.

    Every failure here would otherwise surface as a traceback out of a collector
    subprocess, after the manifest was frozen and the box was committed. The
    checkpoint existence check earns its place: `mcts:<path>@64` resolves to an
    entrant whatever `<path>` says, and the `torch.load` that finds out is inside
    a worker.

    Function-scoped import for the same reason `named_opponent` has one:
    `hexset.arena` reaches for torch on some entrant kinds, and this module is
    tested and timed where torch cannot be installed.
    """
    from hexset.arena import entrant_from_name

    for name in mix_names(mix):
        if name == "parent":
            if not have_parent:
                raise SystemExit("--mix parent needs --parent <checkpoint>")
            continue
        inner = sampled_spec(name)
        if inner is not None and _bare_network(inner) is None:
            raise SystemExit(
                f"--mix {name}: {SAMPLED} takes a bare network:<path>, nothing else"
            )
        spec = name if inner is None else inner
        try:
            entrant = entrant_from_name(spec)
        except (KeyError, ValueError):
            hint = f" (a `{TARGETED}` member needs --runtime hexn.coalition)" if spec.startswith(TARGETED) else ""
            raise SystemExit(f"unknown mix opponent: {name}{hint}") from None
        if spec.startswith(TARGETED):
            entrant = entrant_from_name(spec[len(TARGETED):])
        if isinstance(entrant.weights, str) and not Path(entrant.weights).exists():
            raise SystemExit(
                f"--mix {name} names a checkpoint that is not there: {entrant.weights}"
            )


# Seating is HexSet's (`hexset.casting`): who holds which seat in game `k` is a
# pure function of the index, and it is engine work in the same sense the game
# law is -- a bot table needs it exactly as a learner table does. `alternating`
# is re-exported above under its own name; the two below only rename HexSet's,
# because the run configuration, the manifests and the tests all say
# `league_caster`/`paired_caster`.


def league_caster(learners: int, players: int, order: Sequence[int] | None = None):
    """`hexset.casting.league_rotation` under the name the run configs use."""
    return league_rotation(learners, players, order)


def paired_caster(caster):
    """`hexset.casting.paired`: games `2k` and `2k+1` get `caster(k)`.

    The board pairing's casting half. Within a pair the same policy holds the
    same seat on the same board, which is what makes the mate game's reward a
    baseline for a seat rather than a comparison of two different players --
    `hexn.ppo`'s `pair_baseline` conditions on (board, seat, policy), and this
    is the policy leg; `hexn.selfplay.Collector._paired_board` -- the board law
    it hands `LaneEnv` -- is the board leg.
    """
    return paired(caster)


# A mix entry whose name reads `table(a|b|c)` casts a *table*: the learner in
# exactly one seat and the other seats each drawn independently from the pool
# `a`, `b`, `c` — with replacement, so three copies of one bot is a possible
# table. A plain `name=fraction` entry, by contrast, gives its opponent 2 of
# 4 seats on alternating parity, so the learner always has a twin at the
# table; `table(...)` is what makes a fully heterogeneous table expressible.
TABLE = "table("

# A mix entry whose name reads `duel(a|b|c)` casts a *duel*: a two-seat game
# under HexSet's duel-variant rules (`DUEL_VARIANT_GAME`), dealt as a full
# table with two seats retired. The learner takes a drawn seat, its opponent
# one of the other three -- so the opponent sits one, two or three seats
# round from it, uniformly -- drawn from the pool with replacement (`self` is
# the learner again), and the two seats left over are retired and never move.
# Seats and pool members are drawn per game index like a table's.
DUEL = "duel("

# A mix entry whose name reads `coalition(a|b|c;size=1-3;lead=0-2;...)` casts
# a *table with a coalition in it*: the learner in one drawn seat, a target
# seat, one to three frozen seats that gang up on that target
# (`hexn.coalition.Targeted`: the robber on its best hex, no trade with it,
# from the plan's trigger on), and every other seat drawn from the pool as a
# `table(...)` draws them. Everything about the coalition is drawn per game
# off the same per-index stream as the cast, so a game stays a pure function
# of `(seed, index)`:
#
# - `targeted=0.5`: the share of these games whose target is the learner's
#   own seat; otherwise the target is another seat and the learner watches a
#   coalition work on someone else.
# - `size=1-3`: how many members, drawn uniformly, capped by the seats left
#   once the learner and the target are placed (two, when they differ).
# - `start=0.25`: the share of these games hostile from the first move;
#   the rest wait on `lead`.
# - `lead=0-2`: the margin, drawn uniformly, by which the target must lead
#   the table in public points -- and have built past its setup -- before
#   the coalition turns on it. Once on, it stays on (`CoalitionPlan`).
# - `members=<spec>|<spec>`: the frozen checkpoints a member seat draws
#   from, each seated as `targeted:<spec>`; absent, the pool's non-`self`
#   names. A member is never `self`: its seat is never trained on, and it
#   has a stance of its own to play.
#
# The learner's rows are the only ones trained on, as at every table: a
# member is a pool checkpoint, cast under its own id.
COALITION = "coalition("

#: `targeted:<entrant>`, the spec a coalition member is seated under
#: (`hexn.coalition`, loaded as a runtime).
TARGETED = "targeted:"


@dataclass(frozen=True)
class CoalitionTerm:
    """A `coalition(...)` entry, parsed: its pool, its member specs and the
    draw's parameters (see `COALITION`)."""

    pool: tuple[str, ...]
    members: tuple[str, ...]
    size: tuple[int, int] = (1, 3)
    lead: tuple[int, int] = (0, 2)
    start: float = 0.25
    targeted: float = 0.5


@dataclass
class CoalitionPlan:
    """One game's coalition: who it is against, who is in it, and when it
    turns -- `lead` None from the first move, else once the target leads
    the table's public points by that margin having built past its setup.
    `hostile` latches: a coalition that has turned does not turn back."""

    target: int
    members: frozenset[int]
    lead: int | None
    triggered: bool = False

    def hostile(self, game) -> bool:
        if self.lead is None or self.triggered:
            return True
        from hexset.game import to_move
        from hexset.victory import public_victory_points

        # A seat's own view: public points read nothing hidden.
        state = game.state(to_move(game)).state
        mine = public_victory_points(state, self.target)
        others = max(
            public_victory_points(state, s) for s in range(state.num_players) if s != self.target
        )
        if mine > SETUP_POINTS and mine - others >= self.lead:
            self.triggered = True
        return self.triggered


# Every seat holds two settlements after setup; a lead counts only once the
# target has built past them.
SETUP_POINTS = 2


def _range(text: str, what: str, low: int) -> tuple[int, int]:
    a, sep, b = text.partition("-")
    try:
        lo, hi = int(a), int(b if sep else a)
    except ValueError:
        raise ValueError(f"a coalition {what} is `n` or `lo-hi`: {text!r}") from None
    if lo < low or hi < lo:
        raise ValueError(f"a coalition {what} needs {low} <= lo <= hi: {text!r}")
    return lo, hi


def _share(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"a coalition {what} is a share in [0, 1]: {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"a coalition {what} is a share in [0, 1]: {text!r}")
    return value


def coalition_term(name: str) -> CoalitionTerm | None:
    """The `coalition(...)` entry `name` spells, or None for any other entry."""
    if not name.startswith(COALITION):
        return None
    if not name.endswith(")"):
        raise ValueError(f"unclosed coalition entry: {name}")
    parts = [part.strip() for part in name[len(COALITION) : -1].split(";")]
    pool = tuple(member.strip() for member in parts[0].split("|"))
    if not pool or any(not member for member in pool):
        raise ValueError(f"a coalition entry needs at least one pool member: {name}")
    options: dict[str, str | None] = {
        "members": None, "size": "1-3", "lead": "0-2", "start": "0.25", "targeted": "0.5",
    }
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep or key.strip() not in options:
            raise ValueError(
                f"a coalition option is one of {sorted(options)}, as `key=value`: {part!r} in {name}"
            )
        options[key.strip()] = value.strip()
    spelled = options["members"]
    members = (
        tuple(member.strip() for member in spelled.split("|"))
        if spelled
        else tuple(member for member in pool if self_band(member) is None)
    )
    if not members or any(not member for member in members) or any(
        self_band(member) is not None for member in members
    ):
        raise ValueError(
            f"a coalition needs a frozen member: name one with `members=`, or put a "
            f"non-self name in the pool: {name}"
        )
    return CoalitionTerm(
        pool=pool,
        members=members,
        size=_range(options["size"], "size", 1),
        lead=_range(options["lead"], "lead", 0),
        start=_share(options["start"], "start"),
        targeted=_share(options["targeted"], "targeted"),
    )


def _pool(name: str, prefix: str, kind: str) -> list[str] | None:
    if not name.startswith(prefix):
        return None
    if not name.endswith(")"):
        raise ValueError(f"unclosed {kind} entry: {name}")
    pool = [member.strip() for member in name[len(prefix) : -1].split("|")]
    if not pool or any(not member for member in pool):
        raise ValueError(f"a {kind} entry needs at least one opponent: {name}")
    return pool


def table_pool(name: str) -> list[str] | None:
    """The pool a `table(...)` or `duel(...)` entry draws from, or None for a
    plain entry."""
    pool = _pool(name, TABLE, "table")
    if pool is not None:
        return pool
    term = coalition_term(name)
    return list(term.pool) if term is not None else duel_pool(name)


def duel_pool(name: str) -> list[str] | None:
    """The pool a `duel(...)` entry draws from, or None for any other entry."""
    return _pool(name, DUEL, "duel")


def parse_mix(spec: str) -> list[tuple[str, float]]:
    """`"heximax=0.15,parent=0.1"` — the share of games each opponent plays.

    A name is a reserved one (`RESERVED_MIX`), any arena entrant spec, or a
    `table(a|b|c)` pool (see `TABLE`); only the shares and the table syntax are
    checked here, because resolving a name needs `hexset.arena` and this parser
    runs where torch may not be installed. `check_mix` is the resolution check,
    and the trainers call it.

    Table entries live in the spec string rather than behind a flag on purpose:
    `run.manifest.load` refuses a config whose keys are not exactly the mode's
    parameter set, so one new argparse dest would make every frozen config on
    disk unloadable. Everything new is inside `--mix`'s value.
    """
    if not spec:
        return []
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        name, _, value = part.rpartition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"a mix entry needs a name: {part!r}")
        if self_band(name) is not None:
            raise ValueError(
                f"{SELF!r} is a table pool member, not a plain entry: {part!r}"
            )
        for member in table_pool(name) or ():
            self_band(member)  # validates a band, and is None for any other name
        out.append((name, float(value)))
    if any(f <= 0 for _, f in out) or sum(f for _, f in out) > 1.0 + 1e-9:
        raise ValueError(f"mix fractions must be positive and sum to at most 1: {spec}")
    return out


def mix_names(mix: Sequence[tuple[str, float]]) -> list[str]:
    """The distinct opponents a mix seats, in caster id order: id `k + 1` is
    `mix_names(mix)[k]`.

    Plain entries contribute their own name and a table entry contributes its
    pool, each name once at its first appearance — so a mix with no table
    entries and no repeated names lists exactly `[name for name, _ in mix]`,
    which is the id law every recorded run was cast under. `mix_opponents`
    builds this list and `mix_caster` casts against it; keeping them on one
    function is what stops the two from drifting.
    """
    out: list[str] = []
    for name, _ in mix:
        for member in table_pool(name) or [name]:
            if self_band(member) is None and member not in out:
                out.append(member)
        term = coalition_term(name)
        for member in term.members if term is not None else ():
            if TARGETED + member not in out:
                out.append(TARGETED + member)
    return out


def mix_table(mix: Sequence[tuple[str, float]], players: int, seed: int):
    """Game index -> `(cast, temperatures)` for any `--mix`, pure in the index.

    `cast` is `mix_caster`'s verdict, one policy id per seat. `temperatures` is
    one sampling temperature per seat: 1.0 everywhere except a seat a table
    draw filled with a `self~band` member, which draws its own from
    `[1 - band, 1 + band]`. The temperature draws come off the same per-index
    stream *after* every seat is cast, so a pool with no band consumes nothing
    extra and every cast law on record is byte-identical.
    """
    draw = _mix_draw(mix, players, seed)

    def table(index: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        cast, temperatures, _, _ = draw(index)
        return cast, temperatures

    return table


def _mix_draw(mix: Sequence[tuple[str, float]], players: int, seed: int):
    """`mix_table`'s whole verdict per game index: `(cast, temperatures,
    retired, coalition)`, where `retired` is the seats a `duel(...)` draw
    leaves empty and None for every other game, and `coalition` is the
    `CoalitionPlan` a `coalition(...)` draw made, a fresh one each call, and
    None for every other game. `mix_deal` reads the third part,
    `mix_coalitions` the fourth."""
    names = mix_names(mix)
    plan: list[tuple[float, int | tuple[tuple[int, float], ...], bool, object]] = []
    for name, fraction in mix:
        pool = table_pool(name)
        if pool is None:
            plan.append((fraction, names.index(name) + 1, False, None))
        else:
            term = coalition_term(name)
            plan.append((
                fraction,
                tuple(
                    (0, band) if (band := self_band(member)) is not None
                    else (names.index(member) + 1, 0.0)
                    for member in pool
                ),
                duel_pool(name) is not None,
                None if term is None else (
                    term, tuple(names.index(TARGETED + member) + 1 for member in term.members)
                ),
            ))
    ones = (1.0,) * players

    def table(index: int) -> tuple[
        tuple[int, ...], tuple[float, ...], frozenset[int] | None, CoalitionPlan | None
    ]:
        rng = random.Random(f"{seed}:{index}:cast")
        draw = rng.random()
        cumulative = 0.0
        for fraction, who, duel, coalition in plan:
            cumulative += fraction
            if draw < cumulative:
                if isinstance(who, int):
                    cast = [0] * players
                    for seat in range(1 - index % 2, players, 2):
                        cast[seat] = who
                    return tuple(cast), ones, None, None
                if coalition is not None:
                    # The draw order is the contract: learner seat, target,
                    # member count, members, trigger, then the seats' pool
                    # draws in seat order, then the temperatures.
                    term, member_ids = coalition
                    learner = rng.randrange(players)
                    others = [s for s in range(players) if s != learner]
                    target = learner if rng.random() < term.targeted else rng.choice(others)
                    candidates = [s for s in others if s != target]
                    size = min(rng.randint(*term.size), len(candidates))
                    members = frozenset(rng.sample(candidates, size))
                    lead = None if rng.random() < term.start else rng.randint(*term.lead)
                    drawn = [
                        (0, 0.0) if seat == learner
                        else (rng.choice(member_ids), 0.0) if seat in members
                        else rng.choice(who)
                        for seat in range(players)
                    ]
                    temperatures = [
                        1.0 + rng.uniform(-band, band) if band > 0.0 else 1.0
                        for _, band in drawn
                    ]
                    return (
                        tuple(pid for pid, _ in drawn), tuple(temperatures), None,
                        CoalitionPlan(target, members, lead),
                    )
                if duel:
                    learner = rng.randrange(players)
                    partner = rng.choice([s for s in range(players) if s != learner])
                    pid, band = rng.choice(who)
                    temperature = 1.0 + rng.uniform(-band, band) if band > 0.0 else 1.0
                    retired = frozenset(range(players)) - {learner, partner}
                    # A retired seat never moves; it carries the opponent's id
                    # only so every seat names a policy that exists.
                    cast = tuple(
                        0 if seat == learner else pid for seat in range(players)
                    )
                    temps = tuple(
                        temperature if seat == partner else 1.0 for seat in range(players)
                    )
                    return cast, temps, retired, None
                learner = rng.randrange(players)
                drawn = [
                    (0, 0.0) if seat == learner else rng.choice(who)
                    for seat in range(players)
                ]
                temperatures = [
                    1.0 + rng.uniform(-band, band) if band > 0.0 else 1.0
                    for _, band in drawn
                ]
                return tuple(pid for pid, _ in drawn), tuple(temperatures), None, None
        return (0,) * players, ones, None, None

    return table


def mix_coalitions(mix: Sequence[tuple[str, float]], players: int, seed: int, *, pair_boards: bool = False):
    """Game index -> this game's `CoalitionPlan`, or None for a game cast
    from any other entry; None altogether when `mix` holds no
    `coalition(...)` entry, so a collector hangs nothing on its games unless
    asked. A fresh plan each call: its trigger latches, and the latch is the
    game's. The fourth part of `_mix_draw`'s verdict, off the same per-index
    stream as the cast. `pair_boards` mirrors `paired_caster`: games `2k` and
    `2k+1` share one draw."""
    if not any(coalition_term(name) is not None for name, _ in mix):
        return None
    draw = _mix_draw(mix, players, seed)

    def plan(index: int) -> CoalitionPlan | None:
        return draw(index // 2 if pair_boards else index)[3]

    return plan


def mix_deal(mix: Sequence[tuple[str, float]], players: int, seed: int, *, pair_boards: bool = False):
    """Game index -> `(game_type, retired seats)` for a mix holding a
    `duel(...)` entry, or None when it holds none -- so a collector deals the
    standard game exactly as before unless asked. A game the walk casts from
    any other entry is `(STANDARD_GAME, frozenset())`. The third part of
    `mix_table`'s verdict, off the same per-index stream as the cast.
    `pair_boards` mirrors `paired_caster`: games `2k` and `2k+1` share one draw."""
    if not any(duel_pool(name) is not None for name, _ in mix):
        return None
    from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME

    draw = _mix_draw(mix, players, seed)

    def deal(index: int):
        retired = draw(index // 2 if pair_boards else index)[2]
        return (STANDARD_GAME, frozenset()) if retired is None else (DUEL_VARIANT_GAME, retired)

    return deal


def mix_caster(mix: Sequence[tuple[str, float]], players: int, seed: int):
    """Game index -> per-seat policy ids for any `--mix`, pure in the index.

    Without a table entry this *is* `mixed_caster` — same rng key, same
    cumulative walk over the shares, same alternating seat pairs — and the
    identity is pinned by test, because a resumed run must still deal the
    games an earlier version of this function would have dealt.

    A table entry, when the walk lands on it, seats the learner once and fills
    every other seat with an independent draw from its pool -- `SELF` in the
    pool is the learner again (id 0), so such a table trains on every seat the
    draw gives it. The learner's seat and the draws come off the same per-index
    stream as the entry draw, so a cast is still a function of `(seed, index)`
    alone. The seat is drawn rather than taken as `index % players`: the board
    is keyed to the index too, and `alternating`'s history is what a
    parity-correlated cast costs. The cast half of `mix_table`.
    """
    table = mix_table(mix, players, seed)

    def caster(index: int) -> tuple[int, ...]:
        return table(index)[0]

    return caster


def mix_temperatures(mix: Sequence[tuple[str, float]], players: int, seed: int, *, pair_boards: bool = False):
    """Game index -> per-seat sampling temperatures, or None when no pool in
    `mix` tempers anything -- so a collector pays the law only when asked.
    `pair_boards` mirrors `paired_caster`: games `2k` and `2k+1` share one draw."""
    if not tempered(mix):
        return None
    table = mix_table(mix, players, seed)

    def temperatures(index: int) -> tuple[float, ...]:
        return table(index // 2 if pair_boards else index)[1]

    return temperatures


def mixed_caster(fractions: Sequence[float], players: int, seed: int):
    """Game index -> per-seat policy ids, a pure function of the index.

    Opponent `k + 1` takes every other seat — pairs alternating by parity, so
    neither side owns a seat — for a `fractions[k]` share of games; the rest
    stay pure self-play. Pure in the index so a resumed run casts the games it
    would have cast, and the cast of a logged episode is recomputable.

    **One draw a game, so one opponent a game.** Three opponents at a tenth each
    is three *kinds of game*, not a table holding three kinds of opponent: 30% of
    games are cast, and each cast game seats a single id on 2 of the 4 seats. Two
    consequences, and neither is a defect for what `--mix` is for.

    The marginal distribution of opponents the learner faces is fully
    expressible through `fractions`. What is not expressible is a
    *heterogeneous table* — every other seat drawn independently rather than
    filled by one cast opponent — which is a distinct question about who a
    learner shares a table with, not about how often it meets any one
    opponent; conflating the two into one flag would make a run's result
    ambiguous between the two questions.

    A fraction also buys less exposure than it reads like. In a cast game the
    learner holds 2 seats, so 2 of a learner seat's 3 opponents are the cast bot
    and the third is itself: exposure is `fraction * 2/3`, so e.g.
    `heximax=0.15` gives the learner 10% opponent-facing exposure to heximax.

    Adding heterogeneous tables needs no new flag and should not get one — a new
    argparse dest changes `run.manifest.parameters` and every frozen config then
    fails `load`. It belongs in the spec string, as a `+` between names sharing
    one share: `"heximax+parent=0.1"` keeps every existing spec
    byte-identical, keeps the shares exact because there is still one draw a
    game, and turns each entry's cast from an id into a pair of ids. The seat
    assignment within the pair is the part that needs care rather than taste —
    see `alternating`, where a parity-correlated cast can read as a phantom
    effect that has nothing to do with what is actually being compared.
    """

    def caster(index: int) -> tuple[int, ...]:
        draw = random.Random(f"{seed}:{index}:cast").random()
        cumulative = 0.0
        for k, fraction in enumerate(fractions):
            cumulative += fraction
            if draw < cumulative:
                cast = [0] * players
                for seat in range(1 - index % 2, players, 2):
                    cast[seat] = k + 1
                return tuple(cast)
        return (0,) * players

    return caster


@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs to rebuild its shard. Picklable by design,
    the same rule the arena's entrants follow: descriptions cross processes,
    never built objects."""

    seed: int
    players: int
    lanes: int
    action_cap: int
    max_offers: int | None
    first_game: int
    stride: int
    width: int
    rounds: int
    torch_seed: int
    # Shape as well as size: a worker builds its own net and then has the
    # learner's parameters pushed into it, so a shape mismatch here would fail
    # at the first sync rather than at construction.
    value_head: str = "linear"
    policy_head: str = "linear"
    # Inert unless `value_head == "quantile"`; see `ModelConfig.quantiles`.
    quantiles: int = 32
    # The learner's net carries the fast-win head and records its values
    # (`hexn.ppo.PPOConfig.fast_win`).
    fast_win: bool = False
    # The per-VP reward the learner's value is recorded under
    # (`hexn.ppo.PPOConfig.vp_reward`); 0 records the plain value.
    vp_reward: float = 0.0
    # A bot that answers the networks' trades (`hexn.trade.trader_gate`).
    trader: str | None = None
    # The runtime modules providing the bots this shard names -- `mix`,
    # `trader` -- loaded before anything is built (`hexn.runtime`): a spawned
    # worker inherits none of its parent's registrations.
    runtime: tuple[str, ...] = ()
    mix: tuple[tuple[str, float], ...] = ()
    parent: str = ""
    cohort: bool = True
    # The table league: ids 0..learners-1 are learner nets sharing every game,
    # each recording its own seats. 1 is the default: no table league, one
    # learner. Mutually exclusive with `mix`: both allocate the caster's id
    # space, and a per-learner mix share is incoherent when learners share a
    # table.
    learners: int = 1
    # A permutation of 0..learners-1 applied by `league_caster` before its
    # rotation, so table adjacency can be varied independently of seat share.
    learner_order: tuple[int, ...] | None = None
    # Board-paired collection: games 2k and 2k+1 share 2k's board and — via
    # `paired_caster` — its cast, so each game's mate differs in dice and
    # play only. One flag drives both wires because either alone breaks what
    # `ppo.pair_baseline` conditions on.
    pair_boards: bool = False
    # Searched collection. `simulations` of 0 keeps the plain policy, which is
    # every PPO run on record. Above 0 the worker wraps its policy in a
    # `SearchPolicy`, so each transition carries the search's `Target` and the
    # corpus is distillable.
    #
    # The searched path is the one that needed this: with search on, a single
    # process spends most of a tick in single-threaded Python -- the engine,
    # `encode` over every leaf a decision visits, and the tree descent itself
    # -- and batching only amortises the network call, none of that. So
    # throughput comes from running many worker processes, not from widening
    # the batch inside one.
    simulations: int = 0
    wave: int = 16
    exploration: float = 1.25
    stance: str = "relative"
    root_noise: float = 0.0
    noise_fraction: float = 0.25
    play_temperature: float = 1.0
    # Determinized worlds a searched decision is rooted in, each drawn from
    # the mover's own belief (`hexset.mcts.Search.worlds`); the tree never
    # roots on the true state here.
    k: int = 1


def _build(spec: WorkerSpec) -> tuple[list[NetworkPolicy], Collector]:
    # One thread per worker: the whole point is many workers, and torch's
    # default of one thread per core would have them fighting for the box.
    torch.set_num_threads(1)
    torch.manual_seed(spec.torch_seed)
    runtime.load(spec.runtime)

    board = random_base_board(random.Random(spec.seed))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, spec.players
    )
    graph = static_graph(topology)
    net = HexNet(
        space,
        graph,
        spec.players,
        ModelConfig(
            width=spec.width,
            rounds=spec.rounds,
            value_head=spec.value_head,
            policy_head=spec.policy_head,
            quantiles=spec.quantiles,
            fast_win=spec.fast_win,
        ),
    )
    policy = NetworkPolicy(
        net,
        space,
        packing(graph, spec.players),
        device="cpu",
        generator=torch.Generator().manual_seed(spec.torch_seed),
    )
    policy.record_fast = spec.fast_win
    policy.vp_reward = spec.vp_reward

    if spec.learners > 1 and spec.mix:
        # Two separate blockers, and the id space is only the first.
        #
        # `opponents` is one flat list indexed by cast id, learners first, so a
        # mix's ids would start at `learners` while `mixed_caster` emits `k + 1`
        # — a mix meaning `heximax` would silently seat *learner 1*. That much is
        # an offset away from being fixed.
        #
        # The second is not. `league_caster` fills **every** seat with a learner
        # at any learner count, so a mixed game has to displace one, and the
        # league's premise is that every learner is seated in every game: that
        # is what makes the arms paired, what licenses the seat split under the
        # noise scale, and what lets `standings` score every learner off every
        # game. A combined caster would have to rotate *which* learner is
        # displaced along with the seats, so that every learner's seat share and
        # every learner's exposure to each mix opponent both balance over an
        # index window — and `standings` and `owned` would have to stop assuming
        # a learner appears in every episode.
        raise ValueError("league workers and mix opponents share the caster's "
                         "id space; run one or the other")
    fellow_learners: list[NetworkPolicy] = []
    for k in range(1, spec.learners):
        # Same architecture, its own sampling stream: a learner explores with
        # its own dice, or two identical configs would play identical games.
        fellow = HexNet(
            space,
            graph,
            spec.players,
            ModelConfig(
                width=spec.width,
                rounds=spec.rounds,
                value_head=spec.value_head,
                policy_head=spec.policy_head,
                quantiles=spec.quantiles,
            ),
        )
        fellow_learners.append(
            NetworkPolicy(
                fellow,
                space,
                packing(graph, spec.players),
                device="cpu",
                generator=torch.Generator().manual_seed(spec.torch_seed + 7000 + k),
            )
        )

    opponents: list[BatchPolicy | BoardBots] = list(fellow_learners)
    opponents.extend(
        mix_opponents(
            spec.mix,
            seed=spec.seed,
            lanes=spec.lanes,
            parent=lambda: frozen(spec.parent, "cpu", board, spec.players),
            board=board,
            players=spec.players,
            device="cpu",
            torch_seed=spec.torch_seed,
        )
    )
    if spec.learners > 1:
        caster = league_caster(spec.learners, spec.players, spec.learner_order)
    elif spec.mix:
        caster = mix_caster(spec.mix, spec.players, spec.seed)
    else:
        caster = None
    if spec.pair_boards and caster is not None:
        caster = paired_caster(caster)

    acting: BatchPolicy = policy
    if spec.simulations > 0:
        # Imported here rather than at module scope: `hexn.expert` pulls in the
        # search, and a plain PPO worker has no use for it.
        from .expert import SearchPolicy
        from .netbot import LeafEvaluator
        from hexset.mcts import Search

        acting = SearchPolicy(
            Search(
                LeafEvaluator(policy=policy),
                simulations=spec.simulations,
                wave=spec.wave,
                exploration=spec.exploration,
                stance=spec.stance,
                root_noise=spec.root_noise,
                noise_fraction=spec.noise_fraction,
                k=spec.k,
                # Per worker (`torch_seed` is), so the workers' searches do
                # not draw the same worlds and rolls in lockstep.
                rng=random.Random(spec.torch_seed + 991),
            ),
            temperature=spec.play_temperature,
            rng=random.Random(spec.torch_seed + 992),
        )

    collector = Collector(
        acting,
        lanes=spec.lanes,
        fill=not spec.cohort,
        players=spec.players,
        seed=spec.seed,
        action_cap=spec.action_cap,
        max_offers=spec.max_offers,
        trader=spec.trader,
        first_game=spec.first_game,
        stride=spec.stride,
        opponents=opponents,
        caster=caster,
        temperatures=(
            mix_temperatures(
                spec.mix, spec.players, spec.seed, pair_boards=spec.pair_boards
            )
            if spec.learners <= 1 and spec.mix
            else None
        ),
        learners=tuple(range(spec.learners)),
        pair_boards=spec.pair_boards,
        game_law=(
            mix_deal(spec.mix, spec.players, spec.seed, pair_boards=spec.pair_boards)
            if spec.learners <= 1 and spec.mix
            else None
        ),
        coalition=(
            mix_coalitions(spec.mix, spec.players, spec.seed, pair_boards=spec.pair_boards)
            if spec.learners <= 1 and spec.mix
            else None
        ),
    )
    return [policy, *fellow_learners], collector


class Flattened:
    """A cohort of episodes in wire form: a few large arrays instead of tens of
    thousands of pickled objects.

    A cohort crosses the worker pipes as ~0.9 GB of per-transition ndarrays —
    one observation row, one mask, several scalars each — and the parent
    deserialises all sixteen workers' worth serially, inside `collect_seconds`.
    `Observation.__reduce__` strips the shared tick buffer on purpose, so every
    row also arrives as its own allocation and `pack` loses its gather path.

    This container flattens each worker's trajectories into contiguous blocks
    where the time is parallel, and `episodes()` rebuilds the identical
    `Episode` objects on the far side. Observations come back as views into one
    shared `(positions, width)` buffer with `_packed`/`_row` set, so `pack`
    gathers the whole cohort in one strided copy instead of stacking 145k rows.
    Same numbers, different container: the rebuilt cohort assembles to a
    byte-identical `Batch`, and `test_collect` pins that.

    `action`, `aux` and `value` stay object lists: actions carry structured
    trade bundles, `aux` is whatever the policy attached, and a search policy
    may record `()` where it evaluated nothing — a ragged column cannot be an
    array without changing what it holds.
    """

    def __init__(self, episodes: Sequence[Episode], layout: Packing) -> None:
        self.layout = layout
        self.meta = [
            (e.index, e.seed, e.players, e.outcome, e.cast, e.trades, e.record)
            for e in episodes
        ]
        self.counts = [[len(seat) for seat in e.trajectories] for e in episodes]
        transitions = [t for e in episodes for seat in e.trajectories for t in seat]
        n = len(transitions)

        graphs: list = []
        graph_ids: dict[int, int] = {}
        graph_index = np.empty(n, dtype=np.int32)
        for i, t in enumerate(transitions):
            graph = t.observation.graph
            slot = graph_ids.get(id(graph))
            if slot is None:
                slot = len(graphs)
                graph_ids[id(graph)] = slot
                graphs.append(graph)
            graph_index[i] = slot
        self.graphs = graphs
        self.graph_index = graph_index

        self.buffer = np.empty((n, layout.width), dtype=np.float32)
        for name, start, stop, shape in layout.blocks:
            destination = self.buffer[:, start:stop].reshape(n, *shape)
            for i, t in enumerate(transitions):
                destination[i] = getattr(t.observation, name)
        self.mask = (
            np.stack([t.mask for t in transitions])
            if transitions
            else np.empty((0, 0), dtype=bool)
        )
        self.seat = np.array([t.seat for t in transitions], dtype=np.int64)
        self.step = np.array([t.step for t in transitions], dtype=np.int64)
        self.chosen = np.array([t.index for t in transitions], dtype=np.int64)
        self.log_prob = np.array([t.log_prob for t in transitions], dtype=np.float64)
        self.action = [t.action for t in transitions]
        self.value = [t.value for t in transitions]
        self.aux = [t.aux for t in transitions]
        self.points = np.array([t.points for t in transitions], dtype=np.int64)

    def episodes(self) -> list[Episode]:
        """The identical episodes back, observations as views into one buffer."""
        out: list[Episode] = []
        cursor = 0
        for (index, seed, players, outcome, cast, trades, record), counts in zip(
            self.meta, self.counts
        ):
            trajectories = []
            for count in counts:
                seat_transitions = []
                for k in range(cursor, cursor + count):
                    row = self.buffer[k]
                    components = {
                        name: row[start:stop].reshape(shape)
                        for name, start, stop, shape in self.layout.blocks
                    }
                    seat_transitions.append(
                        Transition(
                            seat=int(self.seat[k]),
                            step=int(self.step[k]),
                            observation=Observation(
                                graph=self.graphs[self.graph_index[k]],
                                _packed=self.buffer,
                                _row=k,
                                **components,
                            ),
                            mask=self.mask[k],
                            action=self.action[k],
                            index=int(self.chosen[k]),
                            log_prob=float(self.log_prob[k]),
                            value=self.value[k],
                            aux=self.aux[k],
                            points=int(self.points[k]),
                        )
                    )
                trajectories.append(tuple(seat_transitions))
                cursor += count
            out.append(
                Episode(
                    index=index,
                    seed=seed,
                    players=players,
                    trajectories=tuple(trajectories),
                    outcome=outcome,
                    cast=cast,
                    trades=trades,
                    record=record,
                )
            )
        return out


def _flat(episodes: list[Episode], policy) -> object:
    """Wrap for the pipe when the worker's policy carries a layout."""
    layout = getattr(policy, "layout", None)
    if layout is None:
        return episodes
    return Flattened(episodes, layout)


def release_memory() -> None:
    """Hand freed memory back: collect cycles, then ask glibc to return free
    heap to the system (a cohort is millions of small allocations, which
    `free` alone leaves mapped). The trainer calls it between iterations and
    each collector worker after sending a cohort. A no-op off glibc."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _serve(spec: WorkerSpec, connection, beat=None) -> None:
    """The worker loop: build once, then answer commands until told to stop.

    `beat` is a shared double the worker stamps with the wall clock on every
    command and every tick, which is how the parent tells a slow worker from
    a wedged one. A collection command's payload is `(count, partial)`: a
    `Partial` to keep each finished game in, or None. A bare count is the
    same with no partial.
    """

    def stamp() -> None:
        beat.value = time.time()

    try:
        policies, collector = _build(spec)
        policy = policies[0]
        if beat is not None:
            collector.beat = stamp
        while True:
            kind, payload = connection.recv()
            if beat is not None:
                stamp()
            if kind == "weights":
                # `policy` is the `NetworkPolicy`, which is what a `SearchPolicy`
                # wraps and what its evaluator holds — so syncing here reaches
                # the searched path too, and `_build` returns it deliberately
                # rather than returning whatever is acting. A list is the
                # league: one dict per learner, in id order.
                states = payload if isinstance(payload, list) else [payload]
                if len(states) != len(policies):
                    raise ValueError(
                        f"{len(states)} weight dicts for {len(policies)} learners"
                    )
                for learner, state in zip(policies, states):
                    learner.net.load_state_dict(state)
                connection.send(("ok", None))
            elif kind in ("collect", "cohort", "listed"):
                count, partial = payload if isinstance(payload, tuple) else (payload, None)
                # The plain calls with the sink set by hand: the parent owns the
                # plan (`ParallelCollector.start_collect`), so a worker keeps
                # its games and never writes one of its own.
                play = {
                    "collect": collector.collect,
                    "cohort": collector.cohort,
                    "listed": collector.listed,
                }[kind]
                collector.sink = None if partial is None else partial.keep
                try:
                    episodes = play(count)
                finally:
                    collector.sink = None
                # The games are on disk (the partial) and, once sent, the
                # parent's: none of them is held between commands.
                flat = _flat(episodes, policy)
                del episodes
                connection.send(("episodes", flat))
                del flat
                release_memory()
            elif kind == "upcoming":
                connection.send(("upcoming", collector.upcoming(payload)))
            elif kind == "counter":
                connection.send(("counter", collector.games_started()))
            elif kind == "stop":
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command: {kind}")
    except EOFError:
        return
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass


class ParallelCollector:
    """`WorkerSpec`s made processes, behind the slice of `Collector` the
    trainer uses: `collect`, `games`, `games_started` — plus `sync`, which
    ships the learner's current weights out once per iteration.

    `collect` forwards to each worker's `Collector.cohort` unless the specs ask
    for streaming, so the on-policy guarantee holds per worker: a worker deals
    its own quota, plays all of it out, and ends with empty lanes. The cost is
    that the iteration waits on the slowest game of the slowest worker.

    `heartbeat`, when given, is a JSON file rewritten every `POLL_SECONDS`
    while the parent waits on its workers: each one's pid, whether it is
    alive, busy and how long since it last ticked, and how many games the
    collection in flight has kept so far."""

    def __init__(
        self,
        specs: Sequence[WorkerSpec],
        *,
        heartbeat: Path | None = None,
        stall_seconds: float = STALL_SECONDS,
    ) -> None:
        if not specs:
            raise ValueError("a parallel collector needs at least one worker")
        context = mp.get_context("spawn")
        self._command = "cohort" if all(spec.cohort for spec in specs) else "collect"
        self.games = 0
        self.last_collect_seconds = 0.0
        self.heartbeat = heartbeat
        self.stall_seconds = stall_seconds
        self._pending: list[bool] | None = None
        self._kept: list[Episode] = []
        self._partial: Partial | None = None
        self._collect_started = 0.0
        self._connections = []
        self._processes = []
        self._beats = []
        self._sent: list[float] = []
        self._busy: set[int] = set()
        for spec in specs:
            ours, theirs = context.Pipe()
            beat = context.RawValue("d", time.time())
            process = context.Process(
                target=_serve, args=(spec, theirs, beat), daemon=True
            )
            process.start()
            theirs.close()
            self._connections.append(ours)
            self._processes.append(process)
            self._beats.append(beat)
            self._sent.append(time.time())

    def _send(self, worker: int, message) -> None:
        self._sent[worker] = time.time()
        self._busy.add(worker)
        self._connections[worker].send(message)

    def _check(self) -> None:
        """Fail on any busy worker that died or stopped ticking.

        A live worker whose answer is already waiting is fine however long
        ago it last ticked. A dead one is fine only if it exited cleanly with
        something waiting: that is its traceback, which `_hear` raises with
        the text. Killed by a signal -- the OOM killer, a segfault, a kill --
        it sent nothing worth waiting for, and a closed pipe polls readable
        exactly like one with an answer in it, so the exit code decides.
        """
        now = time.time()
        for worker in sorted(self._busy):
            process = self._processes[worker]
            waiting = self._connections[worker].poll(0)
            if process.is_alive():
                idle = now - max(self._beats[worker].value, self._sent[worker])
                if not waiting and idle > self.stall_seconds:
                    raise RuntimeError(
                        f"collector worker {worker} (pid {process.pid}) has not "
                        f"ticked for {idle:.0f}s; treating it as wedged"
                    )
            elif process.exitcode != 0 or not waiting:
                raise RuntimeError(
                    f"collector worker {worker} (pid {process.pid}) died with "
                    f"exit code {process.exitcode}"
                )

    def _beat(self) -> None:
        if self.heartbeat is None:
            return
        now = time.time()
        state = {
            "time": now,
            "kept": len(self._partial.finished()) if self._partial else None,
            "partial": str(self._partial.directory) if self._partial else None,
            "workers": [
                {
                    "worker": worker,
                    "pid": process.pid,
                    "alive": process.is_alive(),
                    "busy": worker in self._busy,
                    "idle_seconds": round(
                        now - max(self._beats[worker].value, self._sent[worker]), 1
                    ),
                }
                for worker, process in enumerate(self._processes)
            ],
        }
        try:
            write_atomic(self.heartbeat, json.dumps(state).encode())
        except OSError:
            pass  # a status file must never be what stops a run

    def _hear(self, worker: int, wanted: str, timeout: float | None = None):
        connection = self._connections[worker]
        deadline = None if timeout is None else time.time() + timeout
        wait = POLL_SECONDS if timeout is None else min(POLL_SECONDS, timeout)
        while not connection.poll(wait):
            self._check()
            self._beat()
            if deadline is not None and time.time() > deadline:
                raise RuntimeError(f"collector worker {worker} did not answer in {timeout}s")
        try:
            kind, payload = connection.recv()
        except (EOFError, ConnectionResetError):
            process = self._processes[worker]
            raise RuntimeError(
                f"collector worker {worker} (pid {process.pid}) closed its pipe; "
                f"exit code {process.exitcode}"
            ) from None
        self._busy.discard(worker)
        if kind == "error":
            raise RuntimeError(f"a collector worker failed:\n{payload}")
        if kind != wanted:
            raise RuntimeError(f"expected {wanted}, a worker sent {kind}")
        return payload

    def sync(self, net: torch.nn.Module) -> None:
        self.sync_many([net])

    def sync_many(self, nets: Sequence[torch.nn.Module]) -> None:
        """Ship every learner's weights, in id order — the league's sync."""
        if self._pending is not None:
            raise RuntimeError("cannot sync while collection is in flight")
        states = [
            {k: v.detach().cpu() for k, v in net.state_dict().items()} for net in nets
        ]
        payload = states if len(states) > 1 else states[0]
        for worker in range(len(self._connections)):
            self._send(worker, ("weights", payload))
        for worker in range(len(self._connections)):
            self._hear(worker, "ok")

    def _quotas(self, episodes: int) -> list[int]:
        share, extra = divmod(episodes, len(self._connections))
        return [
            share + (1 if worker < extra else 0)
            for worker in range(len(self._connections))
        ]

    def start_collect(self, episodes: int, partial: Partial | None = None) -> None:
        """Dispatch a fixed cohort without waiting for workers to finish it.

        With a `partial`, every worker keeps each game there as it finishes.
        A cohort's plan -- every index its workers will deal, asked of them
        first -- is written before any game is played; a partial that already
        holds one is a cohort a crash interrupted, and only its missing
        indices are dealt, spread over whichever workers there are now. A
        stream only tops up the games already kept. Either way the workers'
        counters must already be past the partial's indices
        (`hexn.durable.resume_base`), or a later deal could repeat one.
        """
        if self._pending is not None:
            raise RuntimeError("collection is already in flight")
        workers = len(self._connections)
        kept: list[Episode] = []
        if partial is None:
            commands = [
                (self._command, quota) if quota else None
                for quota in self._quotas(episodes)
            ]
        elif self._command == "cohort":
            kept = partial.done()
            plan = partial.plan()
            if plan is None and kept:
                raise ValueError(f"{partial.directory} holds games but no plan")
            if plan is None:
                quotas = self._quotas(episodes)
                partial.write_plan(self._upcoming(quotas))
                commands = [("cohort", (q, partial)) if q else None for q in quotas]
            else:
                finished = {episode.index for episode in kept}
                missing = [index for index in plan if index not in finished]
                commands = [
                    ("listed", (share, partial)) if share else None
                    for share in (missing[w::workers] for w in range(workers))
                ]
        else:
            kept = partial.done()
            commands = [
                ("collect", (q, partial)) if q else None
                for q in self._quotas(max(0, episodes - len(kept)))
            ]
        for worker, command in enumerate(commands):
            if command is not None:
                self._send(worker, command)
        self._pending = [command is not None for command in commands]
        self._kept = kept
        self._partial = partial
        self._collect_started = time.perf_counter()

    def _upcoming(self, quotas: Sequence[int]) -> list[int]:
        """Every index the workers' next cohorts would deal, worker by worker."""
        for worker, quota in enumerate(quotas):
            if quota:
                self._send(worker, ("upcoming", quota))
        plan: list[int] = []
        for worker, quota in enumerate(quotas):
            if quota:
                plan.extend(self._hear(worker, "upcoming"))
        return plan

    def finish_collect(self) -> list[Episode]:
        """Wait for the cohort dispatched by `start_collect`."""
        if self._pending is None:
            raise RuntimeError("there is nothing in flight to finish")
        pending = self._pending
        out: list[Episode] = list(self._kept)
        try:
            for worker, busy in enumerate(pending):
                if busy:
                    payload = self._hear(worker, "episodes")
                    out.extend(
                        payload.episodes() if isinstance(payload, Flattened) else payload
                    )
        finally:
            self._pending = None
            self._kept = []
            self._partial = None
        self.last_collect_seconds = time.perf_counter() - self._collect_started
        self.games += len(out)
        return out

    def collect(self, episodes: int, partial: Partial | None = None) -> list[Episode]:
        """Each worker plays out an equal share of the quota.

        Fixed shares rather than first-`n`-across-workers, so the cohort is
        decided before anyone plays — taking whichever games finish first
        selects for short games, the same bias the in-process collector's
        `deal` bound exists to prevent.
        """
        self.start_collect(episodes, partial)
        return self.finish_collect()

    def games_started(self) -> int:
        """The next safe base index: above it, no worker has dealt anything."""
        if self._pending is not None:
            raise RuntimeError("cannot read counters while collection is in flight")
        for worker in range(len(self._connections)):
            self._send(worker, ("counter", None))
        return max(
            self._hear(worker, "counter") for worker in range(len(self._connections))
        )

    def close(self) -> None:
        if self._pending is not None:
            try:
                self.finish_collect()
            except Exception:
                pass
        for worker, connection in enumerate(self._connections):
            try:
                self._send(worker, ("stop", None))
                self._hear(worker, "ok", timeout=10)
            except Exception:
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
