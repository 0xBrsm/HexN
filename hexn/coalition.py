# SPDX-License-Identifier: GPL-3.0-only
"""A coalition inside self-play: frozen seats that gang up on one target.

HexSet's `hexset.bots.Coalition` seats a bot against *every* seat that is
not a member, from the first move. A training table wants less than that
and more: one target, chosen per game, and hostility that may wait until
the target is ahead.
`Targeted` is that member. It reads its orders off the game it is seated
at: `Game.coalition`, a `hexn.collect.CoalitionPlan` the collector hangs on
each game it deals from a `coalition(...)` mix entry (`hexn.collect`), and
plays `Coalition`'s table layer -- the robber on the target's best hex, no
trade with the target -- against that one seat, from the plan's trigger on.
A game with no plan, or a plan not yet triggered, leaves the bot entirely
its own.

Seated by the arena spec `targeted:<entrant>` (`TARGETED`), registered when
this module is imported -- a run names it with `--runtime hexn.coalition`,
the way every bot a run names is loaded (`hexn.runtime`). `hexn.collect`
spells a coalition member `targeted:<spec>` from the entry's `members=`.

One `Targeted` serves every seat and lane of its board that casts the same
checkpoint (`hexset.gym.lanes.BoardBots`), so it keeps no per-game state:
the targets are re-read from the game at every `seat_at` and `choose`, and
the trigger's latch lives on the plan, which is the game's.
"""
from __future__ import annotations

import random

from hexset.arena import Entrant, entrant_from_name, register_entrant_kind, register_spec, spawn
from hexset.board.board import Board
from hexset.bots import Coalition
from hexset.bots.base import play_against, seat_at
from hexset.game import Game

from .collect import TARGETED, CoalitionPlan

__all__ = ["TARGETED", "CoalitionPlan", "Targeted"]

TARGETED_KIND = "targeted"


class Targeted(Coalition):
    """`bot` as a coalition member whose one target, and whether the
    coalition has turned on it yet, are the game's plan to say."""

    __slots__ = ()

    def __init__(self, bot) -> None:
        super().__init__(bot)
        # `Coalition` leaves this None until seated and refuses to trade
        # before then; a targeted seat has no targets until its plan says so.
        object.__setattr__(self, "targets", frozenset())

    def __reduce__(self):
        return (Targeted, (self.bot,))

    def seat_at(self, game: Game) -> None:
        seat_at(self.bot, game)
        self._read(game)

    def _seat(self, gates) -> None:
        # The targets come off the plan, never off who else is seated.
        object.__setattr__(self, "_gates", gates)

    def _read(self, game: Game) -> None:
        plan = getattr(game, "coalition", None)
        targets = (
            frozenset({plan.target}) if plan is not None and plan.hostile(game) else frozenset()
        )
        if targets != self.targets:
            object.__setattr__(self, "targets", targets)
            play_against(self.bot, targets)

    def choose(self, game: Game):
        self._read(game)
        return super().choose(game)


def _parse(name: str) -> Entrant:
    inner = name[len(TARGETED):]
    if not inner:
        raise ValueError(f"{name!r}: name the entrant to seat, as in `{TARGETED}<entrant>`")
    entrant_from_name(inner)  # refuse an unknown entrant here, not at the deal
    return Entrant(name, kind=TARGETED_KIND, options=(("inner", inner),))


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Targeted:
    return Targeted(spawn(entrant_from_name(entrant.option("inner")), board, rng))


register_spec(TARGETED, _parse)
register_entrant_kind(TARGETED_KIND, _spawn)
