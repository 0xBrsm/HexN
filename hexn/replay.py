# SPDX-License-Identifier: GPL-3.0-only
"""Rebuild a stored episode's live `Game` at any ply, from its own record.

`hexn.selfplay.Collector` deals every game with `records=True`, so an
`Episode` already carries the exact `hexset.record.Record` of the game it
played -- every seat's actions, the chance stream and the trades the table
actually cleared. `hexset.record.replay_to` is the one path that walks a
`Record` back to a live `Game` at some ply; re-deriving a position from
`(seed, index)` plus a hand-rolled trade replay, which this module used to
do, is a second derivation of that path and exactly the kind of drift this
package's boundary rule exists to stop.

This is what turns `hexn.reanalyse`'s "re-search a stored position" into
something cheaper than "re-collect it": no live game has to be kept around
between iterations, only the compact `Episode`, and any ply of any retained
game is one small replay away.
"""

from __future__ import annotations

from hexset.game import Game
from hexset.record import replay_to

from .selfplay import Episode


def replay_to_ply(episode: Episode, ply: int) -> Game:
    """The live `Game` after `ply` of `episode`'s recorded actions.

    `ply` counts actions applied, matching `Transition.step`: replaying to
    `transition.step` reproduces the exact position `transition.observation`
    was encoded from -- `test_replay.py` checks that by re-encoding and
    comparing, and against `hexset.record.replay_to` on the same record.

    Refuses an episode whose `cast` seats an opponent: `Episode.stream`'s own
    docstring is explicit that such an episode "has gaps and cannot replay
    the game" (opponent seats are never recorded). The engine's `record` has
    no such gap -- it is the whole table's, not the learner's -- but the
    refusal is kept for a caller relying on it.

    Refuses an episode with no `record` at all: one collected before
    `Collector` carried `records=True`, or built without one by hand. There
    is nothing left here to reconstruct a game from instead.

    A negative or out-of-range `ply` is `replay_to`'s own refusal to raise,
    not duplicated here.
    """
    if episode.cast and any(seat != 0 for seat in episode.cast):
        raise ValueError(
            "replay_to_ply needs every seat's actions recorded; this episode's "
            f"cast {episode.cast} seats an opponent whose trajectory was never "
            "filed (see Episode.stream)"
        )
    if episode.record is None:
        raise ValueError(
            f"episode {episode.index} carries no record; collect it with "
            "Collector(..., records=True) -- the default -- to replay it"
        )
    return replay_to(episode.record, ply)
