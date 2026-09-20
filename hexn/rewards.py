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
"""

from __future__ import annotations

from .selfplay import Outcome

# `relative_points` lives in `hexset.victory`: `hexset.mcts` needs it too, and
# hexset must never import this module (hexn depends on hexset, not the
# other way around). Re-exported here, not redefined, so callers that already
# read `hexn.rewards.relative_points` see no change and there is still only
# one definition of the quantity the value head is trained to predict.
from hexset.victory import relative_points

__all__ = ["relative_points", "win_loss", "reward"]


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
