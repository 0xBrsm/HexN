# SPDX-License-Identifier: GPL-3.0-only
"""Assembling a collection off the trainer's critical path.

Under async cohort collection the workers finish the next cohort while the
update is still running, and until now the trainer only then decoded those
games and assembled their batch, with the workers and the GPU waiting on it.
`Preparer` is a helper process that does that work as soon as the cohort's
last game is on disk (`hexn.durable.Partial`): it decodes the games
(`hexn.gamefile`), assembles the batch (`prepare`) and hands back the batch,
the per-game gauges and the games stripped of their transitions. The
trainer only receives the games from the workers' pipes, as before.

**Exact.** The helper runs the trainer's own `prepare` on the same games: the
files the workers wrote are the bytes they sent home (one `gamefile.encode`
per game, written and sent). The trainer checks this per cohort. It compares
a digest of the bytes it received with the digest of the bytes the helper
read, and decodes the cohort itself on any mismatch, failure or timeout. The
batch crosses processes as shared-memory tensors, which keep their bits.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import torch

from ..selfplay import Episode
from . import Batch, PPOConfig, assemble, fast_weight, stakes_gauges, win_round

__all__ = [
    "Prepared",
    "Preparer",
    "coalition_gauges",
    "fast_win_gauges",
    "games_digest",
    "learner_final_vp",
    "prepare",
    "prepare_columns",
]

# How often the helper looks for the cohort's last game file.
POLL_SECONDS = 0.5


@dataclass
class Prepared:
    """One cohort, ready for the update: what `prepare` returns."""

    cohort: int
    digest: str
    batch: Batch
    gauges: dict
    episodes: list[Episode]


def games_digest(games: Sequence[bytes]) -> str:
    """sha256 over a cohort's encoded games, in game-index order."""
    from .. import gamefile

    digest = hashlib.sha256()
    for game in sorted(games, key=gamefile.index_of):
        digest.update(len(game).to_bytes(8, "little"))
        digest.update(game)
    return digest.hexdigest()


def fast_win_gauges(episodes: Sequence[Episode], curve: Sequence[float], players: int) -> dict:
    """The speed columns a fast-win run logs: the median round games were
    won in, the median round of the learner's own wins, and the mean payoff
    per won game."""
    import statistics

    won = [e for e in episodes if e.outcome.winner is not None]
    mine = [e for e in won if e.trajectories[e.outcome.winner]]
    return {
        "win_round_median": statistics.median(win_round(e.outcome, players) for e in won) if won else None,
        "learner_win_round_median": (
            statistics.median(win_round(e.outcome, players) for e in mine) if mine else None
        ),
        "fast_payoff_mean": (
            sum(fast_weight(curve, e.outcome, players) for e in won) / len(won) if won else None
        ),
    }


def coalition_gauges(episodes: Sequence[Episode], plan_of) -> dict:
    """What a `coalition(...)` mix dealt and how the learner fared in it:
    the share of games with a coalition, the share of those whose target
    was the learner's seat, and the learner's win rate in each kind of
    game -- targeted (its target seat won), neutral (a learner seat won at
    a table whose coalition was on someone else) and free (no coalition).
    `plan_of` is `hexn.collect.mix_coalitions`' law, re-drawn here from the
    game index: a plan is a pure function of the index, so nothing rides on
    the episode."""
    counts = {"targeted": [0, 0], "neutral": [0, 0], "free": [0, 0]}
    for episode in episodes:
        plan = plan_of(episode.index)
        winner = episode.outcome.winner
        if plan is None:
            kind, won = "free", winner is not None and bool(episode.trajectories[winner])
        elif episode.cast[plan.target] == 0:
            kind, won = "targeted", winner == plan.target
        else:
            kind, won = "neutral", winner is not None and bool(episode.trajectories[winner])
        counts[kind][0] += won
        counts[kind][1] += 1
    played = len(episodes)
    with_coalition = counts["targeted"][1] + counts["neutral"][1]
    return {
        "coalition_share": with_coalition / played if played else None,
        "coalition_targeted_share": counts["targeted"][1] / with_coalition if with_coalition else None,
        **{
            f"coalition_{kind}_win_rate": (wins / games if games else None)
            for kind, (wins, games) in counts.items()
        },
    }


def learner_final_vp(episodes: Sequence[Episode]) -> float | None:
    """The mean final victory points of the seats the learner played: the
    number a per-VP reward moves, read beside the win rate so points bought
    without wins show up."""
    held = [e.outcome.points[s] for e in episodes for s, t in enumerate(e.trajectories) if t]
    return sum(held) / len(held) if held else None


def prepare(
    episodes: Sequence[Episode],
    layout,
    config: PPOConfig,
    players: int,
    coalition_plans=None,
) -> tuple[Batch, dict, list[Episode]]:
    """A cohort's batch, the gauges that read its transitions, and the games
    with their transitions dropped. The trainer's assembly, in one place, so
    the trainer and the helper cannot assemble differently."""
    batch = assemble(episodes, layout, config)
    gauges = {
        **(fast_win_gauges(episodes, config.fast_win, players) if config.fast_win else {}),
        **({"learner_final_vp_mean": learner_final_vp(episodes)} if config.vp_reward else {}),
        **(coalition_gauges(episodes, coalition_plans) if coalition_plans is not None else {}),
        **(
            stakes_gauges(episodes, config)
            if config.decided_cut or config.setup_weight != 1
            else {}
        ),
    }
    # Every transition goes: the observations are views into the decode
    # buffer, a second copy of the batch that nothing after this reads.
    return batch, gauges, [replace(episode, trajectories=()) for episode in episodes]


def prepare_columns(columns, layout, config: PPOConfig, players: int):
    """`prepare` from `hexn.gamefile.Columns`, without a `Transition` or an
    `Observation` per position. It runs the same per-trajectory arithmetic in
    the same order (`advantages`, the `copies` rule, the frames), so the batch
    and the gauges are the ones `prepare` builds, bit for bit; the tests hold
    it to that. Returns None for a configuration it does not cover (fast-win
    or per-VP rewards, the pair baseline, REINFORCE, a coalition mix), and the
    caller takes the object path then."""
    import numpy as np

    from hexset.encoding import global_columns, to_frame
    from hexset.victory import relative_points

    from ..rewards import points_lost_next, points_next_round, win_loss
    from . import SETUP_PHASES, advantages, outlook_row

    if config.critic != "gae" or config.pair_baseline or config.fast_win or config.vp_reward:
        return None
    weighed = bool(config.decided_cut) or config.setup_weight != 1
    buffer, bounds = columns.buffer, columns.bounds
    globals_start = bounds[3]
    setup_phases = np.asarray([int(p) for p in SETUP_PHASES])
    chosen: list[int] = []
    log_probs: list[float] = []
    advantage_blocks = []
    seat_targets, seat_margins, seat_lengths, id_blocks, count_blocks = [], [], [], [], []
    recorded = low = low_won = high = high_won = setup = 0
    stripped = []
    for game in columns.games:
        fields = game["fields"]
        outcome = fields["outcome"]
        at = {name: i for i, name in enumerate(game["names"])}
        i_index, i_log_prob, i_value = at["index"], at["log_prob"], at["value"]
        wins = win_loss(outcome)
        margins = relative_points(outcome.points)
        lost = moved = None
        if config.aux_target in ("loss", "points"):
            seat_rows, k = [], 0
            for count in game["counts"]:
                seat_rows.append(game["rows"][k : k + count])
                k += count
            seat_steps = [np.asarray([row[at["step"]] for row in rows], dtype=np.int64) for rows in seat_rows]
            if config.aux_target == "loss":
                lost = points_lost_next(
                    seat_steps,
                    [np.asarray([row[at["points"]] for row in rows], dtype=np.int64) for rows in seat_rows],
                    outcome.points,
                )
            else:
                if "seat_points" not in at or any(len(row[at["seat_points"]]) != len(wins) for row in game["rows"]):
                    raise ValueError(
                        "aux_target 'points' needs every transition's seat_points; this game "
                        "was collected without them"
                    )
                moved = points_next_round(
                    seat_steps,
                    [np.asarray([row[at["seat_points"]] for row in rows], dtype=np.float32).reshape(len(rows), len(wins))
                     for rows in seat_rows],
                    outcome.points,
                )
        phase = global_columns(len(wins))["phase"]
        phase_cols = slice(globals_start + phase.start, globals_start + phase.stop)
        winner = outcome.winner
        k = 0
        for seat, count in enumerate(game["counts"]):
            if not count:
                continue
            seat_rows = game["rows"][k : k + count]
            first = game["start"] + k
            k += count
            if not all(row[i_value] for row in seat_rows):
                raise ValueError(
                    "a recorded transition has no value estimate; GAE cannot "
                    "substitute one without silently corrupting the chain"
                )
            own_estimates = [row[i_value][0] for row in seat_rows]
            estimates = np.array(own_estimates, dtype=np.float32)
            advantage_blocks.append(advantages(estimates, np.float32(to_frame(wins, seat)[0]), config.lam, None))
            seat_targets.append(np.asarray(to_frame(wins, seat), dtype=np.float32))
            if moved is not None:
                seat_margins.append(np.roll(moved[seat], -seat, axis=1))
            elif config.aux_target == "outlook":
                if "outlook" not in at or not any(row[at["outlook"]] for row in game["rows"]):
                    raise ValueError("aux_target 'outlook' needs games collected with Collector(outlook=True)")
                seat_margins.append(
                    np.stack([outlook_row(row[at["outlook"]], seat, len(wins)) for row in seat_rows])
                )
            elif lost is None:
                seat_margins.append(np.repeat(np.asarray(to_frame(margins, seat), dtype=np.float32)[None], count, axis=0))
            else:
                block = np.zeros((count, len(wins)), dtype=np.float32)
                block[:, 0] = lost[seat]
                seat_margins.append(block)
            seat_lengths.append(count)
            id_blocks.append(np.arange(first, first + count, dtype=np.intp))
            chosen.extend(row[i_index] for row in seat_rows)
            log_probs.extend(row[i_log_prob] for row in seat_rows)
            if weighed:
                # `hexn.ppo.copies`, a seat at a time: decided first, then setup.
                own = np.asarray(own_estimates, dtype=np.float64)
                n = np.ones(count, dtype=np.int64)
                if config.setup_weight != 1:
                    in_setup = np.isin(np.argmax(buffer[first : first + count, phase_cols], axis=1), setup_phases)
                    n[in_setup] = config.setup_weight
                if config.decided_cut:
                    decided = (own < config.decided_cut) | (own > 1.0 - config.decided_cut)
                    n[decided] = 0
                    lows = decided & (own < 0.5)
                    highs = decided & ~(own < 0.5)
                    low += int(lows.sum())
                    high += int(highs.sum())
                    if winner == seat:
                        low_won += int(lows.sum())
                        high_won += int(highs.sum())
                recorded += count
                setup += int((n > 1).sum())
                count_blocks.append(n)
        stripped.append(Episode(trajectories=(), **fields))
    if not chosen:
        raise ValueError("no transitions to learn from")
    advantage = np.concatenate(advantage_blocks)
    ids = np.concatenate(id_blocks)
    targets = np.repeat(np.stack(seat_targets), seat_lengths, axis=0)
    margin_targets = np.concatenate(seat_margins)
    chosen = np.asarray(chosen, dtype=np.int64)
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if weighed:
        rows = np.repeat(np.arange(len(ids)), np.concatenate(count_blocks))
        ids, chosen, log_probs = ids[rows], chosen[rows], log_probs[rows]
        targets, margin_targets, advantage = targets[rows], margin_targets[rows], advantage[rows]
    from . import observation_dtype

    want = np.dtype(observation_dtype(config))
    if buffer.dtype == np.float16 and want != np.float16:
        raise ValueError("float16 observations cannot make a float32 batch")
    if len(ids) == buffer.shape[0] and np.array_equal(ids, np.arange(len(ids))):
        observations = torch.from_numpy(buffer.astype(want, copy=False))
    else:
        observations = torch.from_numpy(buffer[ids].astype(want, copy=False))
    batch = Batch(
        buffer=observations,
        mask=torch.from_numpy(columns.mask[ids]),
        chosen=torch.from_numpy(chosen),
        log_prob=torch.from_numpy(log_probs.astype(np.float32)),
        advantage=torch.from_numpy(advantage),
        value_target=torch.from_numpy(np.ascontiguousarray(targets)),
        margin_target=torch.from_numpy(np.ascontiguousarray(margin_targets)),
        fast_target=None,
    )
    gauges = (
        {
            "decided_share": (low + high) / recorded if recorded else 0.0,
            "decided_low_won": low_won / low if low else 0.0,
            "decided_high_won": high_won / high if high else 0.0,
            "setup_repeated": setup,
        }
        if config.decided_cut or config.setup_weight != 1
        else {}
    )
    return batch, gauges, stripped


def _serve(layout, config, players, coalition_args, dtype, commands, results) -> None:
    """The helper's loop: for each cohort it is told about, wait for the
    cohort's games on disk, then decode and prepare them."""
    from .. import durable, gamefile
    from ..collect import mix_coalitions
    from ..steps import by_index

    torch.set_num_threads(1)
    coalition_plans = mix_coalitions(*coalition_args) if coalition_args else None
    while True:
        command = commands.get()
        if command is None:
            return
        cohort, directory = command
        try:
            partial = durable.Partial(Path(directory))
            plan = None
            while plan is None or not set(plan) <= set(partial.finished()):
                time.sleep(POLL_SECONDS)
                plan = partial.plan()
            games = [game for game in partial.games() if gamefile.index_of(game) in set(plan)]
            digest = games_digest(games)
            ordered = sorted(games, key=gamefile.index_of)
            del games
            prepared = None
            if coalition_plans is None:
                prepared = prepare_columns(gamefile.decode_columns(ordered, dtype), layout, config, players)
            if prepared is None:
                episodes = by_index(gamefile.decode_all(ordered, dtype))
                prepared = prepare(episodes, layout, config, players, coalition_plans)
                del episodes
            del ordered
            batch, gauges, stripped = prepared
            # Into shared memory once, here, rather than by the queue's pickler.
            for name in batch.__dataclass_fields__:
                value = getattr(batch, name)
                if isinstance(value, torch.Tensor):
                    value.share_memory_()
            results.put(Prepared(cohort, digest, batch, gauges, stripped))
        except Exception as error:  # the trainer falls back to its own decode
            results.put((cohort, f"{type(error).__name__}: {error}"))


class Preparer:
    """The helper process: `submit` a cohort when it is dispatched, `take` it
    once its games are home. The helper works through cohorts in the order
    they were submitted; the next one may be submitted before the last is
    taken."""

    def __init__(self, layout, config: PPOConfig, players: int, coalition_args, dtype) -> None:
        context = torch.multiprocessing.get_context("spawn")
        self._commands = context.Queue()
        self._results = context.Queue()
        self._process = context.Process(
            target=_serve,
            args=(layout, config, players, coalition_args, dtype, self._commands, self._results),
            daemon=True,
        )
        self._process.start()
        self._submitted: set[int] = set()

    def submit(self, cohort: int, directory: Path) -> None:
        self._submitted.add(cohort)
        self._commands.put((cohort, str(directory)))

    def take(self, cohort: int, digest: str, timeout: float = 1800.0) -> Prepared | str:
        """The prepared cohort, or a sentence saying why not (the trainer then
        decodes the cohort itself)."""
        if cohort not in self._submitted:
            return f"cohort {cohort} was not submitted"
        self._submitted.discard(cohort)
        deadline = time.time() + timeout
        while True:
            if not self._process.is_alive():
                return f"the helper died (exit {self._process.exitcode})"
            try:
                result = self._results.get(timeout=min(5.0, max(0.1, deadline - time.time())))
            except Exception:
                if time.time() > deadline:
                    return "the helper did not answer in time"
                continue
            # An answer for an earlier cohort the trainer stopped waiting for.
            earlier = result.cohort if isinstance(result, Prepared) else result[0]
            if earlier < cohort:
                continue
            break
        if not isinstance(result, Prepared):
            return f"the helper failed on cohort {result[0]}: {result[1]}"
        if result.cohort != cohort:
            return f"the helper answered for cohort {result.cohort}, not {cohort}"
        if result.digest != digest:
            return "the helper read different games from the ones received"
        return result

    def close(self) -> None:
        try:
            self._commands.put(None)
            self._process.join(timeout=10)
        finally:
            if self._process.is_alive():
                self._process.terminate()
