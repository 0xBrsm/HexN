# SPDX-License-Identifier: GPL-3.0-only
"""Turning a finished game into a number per seat.

Deliberately not in `hexn.selfplay`: the collector emits an `Outcome` and
scalarises nothing, so what a run rewards is a choice made here and recorded,
not a default buried in the plumbing.

Two scalarisations live here. **`win_loss`** is what the value head trains on
today — one-hot to the winner, nothing to anyone else (`hexn.ppo` explains why
in its own module docstring: the value head is a win head, cross-entropy
against this target).

**`relative_points`/`reward` is the older answer.** It now feeds the optional
auxiliary margin head (`--aux-margin-weight`, off by default) and evaluation
readouts elsewhere (`hexn.loop`'s `mean_relative_points`). It carries more
signal per game than a win/loss bit alone, because a losing seat still says
how close it came — but it has to be read *relatively*, not absolutely: an
absolute per-seat point total is not a fixed target to aim for, since what
counts as a good total depends on the position, so a value trained on it has
to learn that context rather than the actual difference between the seats at
the table. Reading points relatively removes that ambiguity by construction.

Two things `relative_points` is not, still. It is not the metric: win rate is
what gets reported, so a policy that farmed points without winning would be
invisible to a reward built only from them. And it is not a licence to
discount — see `relative_points` on why γ < 1 is a trap here.

**`points_lost_next`** and **`points_next_round`** are per-decision labels
rather than scalarisations of the outcome: whether the deciding seat loses
victory points before it acts again after its current turn (`--aux-target
loss`), and every seat's change in points over that same span (`--aux-target
points`, the points head).
"""

from __future__ import annotations

from .selfplay import Outcome

# `relative_points` lives in `hexset.victory`: `hexset.mcts` needs it too, and
# hexset must never import this module (hexn depends on hexset, not the
# other way around). Re-exported here, not redefined, so callers that already
# read `hexn.rewards.relative_points` see no change and there is still only
# one definition of the quantity the value head is trained to predict.
from hexset.victory import relative_points

import numpy as np

__all__ = ["relative_points", "win_loss", "reward", "points_lost_next", "points_next_round"]


def win_loss(outcome: Outcome) -> tuple[float, ...]:
    """+1 to the winner, 0 to everyone else. The thing points stand in for.

    Kept so the two can be compared on the same run rather than argued about,
    and so the reported metric has a definition in the same place as the
    training signal. An unfinished game gives every seat zero.
    """
    seats = len(outcome.points)
    if outcome.winner is None:
        return (0.0,) * seats
    return tuple(1.0 if seat == outcome.winner else 0.0 for seat in range(seats))


def reward(outcome: Outcome) -> tuple[float, ...]:
    """The reward a run trains on. Relative terminal points.

    A truncated game is scored the same way. The action cap stopping a game is
    a fact about the position reached, and zeroing it would teach a policy that
    stalling escapes a loss — the one lesson this reward must not contain.
    """
    return relative_points(outcome.points)


def points_lost_next(
    steps: "list[np.ndarray]", points: "list[np.ndarray]", final: "tuple[int, ...]"
) -> "list[np.ndarray]":
    """Per seat, one float32 0/1 per recorded decision: 1 where the seat's
    victory points fall between the end of the run of decisions it is in and
    its next decision, i.e. something the other seats do before it acts again
    takes points from it. Points only fall when an award (Longest Road,
    Largest Army) changes hands, so this is "I lose an award before my next
    turn", given what I do for the rest of this one.

    `steps[s]` and `points[s]` are seat `s`'s recorded decisions in order:
    `Transition.step` (the interleaved decision counter) and
    `Transition.points` (the seat's true points as it decides, before its
    action). A run is a maximal stretch of one seat's decisions with no other
    seat's recorded decision between them; every decision in a run gets the
    run's label. It compares the points at the run's last decision with the
    points at the seat's next decision (`final[s]`, the outcome's points, when
    there is none). Points the run's last action itself gains are counted in
    that comparison, so a gain and a loss of the same size in one gap cancel
    and read 0; it never reads a gain as a loss.

    Needs every seat's decisions (self-play): a seat with an empty trajectory
    contributes nothing to the interleaving, so with opponents unrecorded the
    runs of the recorded seats merge across their turns."""
    seats = len(steps)
    order = sorted(
        (int(step), seat, i)
        for seat in range(seats)
        for i, step in enumerate(steps[seat])
    )
    labels = [np.zeros(len(steps[seat]), dtype=np.float32) for seat in range(seats)]
    # runs as (seat, first index, last index) into that seat's own decisions
    runs: list[list[int]] = []
    for _, seat, i in order:
        if runs and runs[-1][0] == seat:
            runs[-1][2] = i
        else:
            runs.append([seat, i, i])
    for seat, first, last in runs:
        later = points[seat][last + 1] if last + 1 < len(points[seat]) else final[seat]
        if later < points[seat][last]:
            labels[seat][first : last + 1] = 1.0
    return labels


def points_next_round(
    steps: "list[np.ndarray]", seat_points: "list[np.ndarray]", final: "tuple[int, ...]"
) -> "list[np.ndarray]":
    """Per seat, a float32 `(decisions, seats)` array, board order: every seat's
    change in true victory points from each of this seat's decisions to its
    next decision after the run it is in -- what the rest of its own turn and
    the other seats' turns until it acts again do to everybody's points. The
    points head's target (`--aux-target points`).

    `steps[s]` is seat `s`'s `Transition.step` per decision and
    `seat_points[s]` its `Transition.seat_points` (every seat's points as it
    decides, before its action), shaped `(decisions, seats)`. Runs are
    `points_lost_next`'s: a maximal stretch of one seat's decisions with no
    other seat's recorded decision between them. With no later decision the
    span ends at the outcome's points (`final`). Like `points_lost_next` it
    needs every seat's decisions (self-play) for the runs to be turns."""
    seats = len(steps)
    order = sorted(
        (int(step), seat, i)
        for seat in range(seats)
        for i, step in enumerate(steps[seat])
    )
    out = [np.zeros((len(steps[seat]), len(final)), dtype=np.float32) for seat in range(seats)]
    runs: list[list[int]] = []
    for _, seat, i in order:
        if runs and runs[-1][0] == seat:
            runs[-1][2] = i
        else:
            runs.append([seat, i, i])
    for seat, first, last in runs:
        points = np.asarray(seat_points[seat], dtype=np.float32).reshape(len(steps[seat]), len(final))
        later = points[last + 1] if last + 1 < len(points) else np.asarray(final, dtype=np.float32)
        out[seat][first : last + 1] = later[None, :] - points[first : last + 1]
    return out
