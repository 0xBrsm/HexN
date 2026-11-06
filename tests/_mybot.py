# SPDX-License-Identifier: GPL-3.0-only
"""`mybot`: a bot registered the way a runtime registers one, for the tests
that seat a bot by name (`_bots.needs`) when no `--runtime` supplies it.

It moves at random; its gate values a card more the fewer of that kind the
seat holds, so it trades for real, with an offer budget of two.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from hexset.actions import options_for
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.trading import DeclaredTrade, TradeParams

NAME = "mybot"
TRADE = TradeParams(max_offers=2)


@dataclass
class MyBot(DeclaredTrade):
    rng: random.Random = field(default_factory=random.Random)
    trade: TradeParams = TRADE

    def choose(self, game):
        return self.rng.choice(options_for(game))

    def gains_many(self, view, received, counterparties):
        hand = view.state.hands[view.perspective]
        return [sum(n / (1 + hand[r]) for r, n in enumerate(bundle)) for bundle in received]


register_entrant_kind(NAME, lambda entrant, board, rng: MyBot(rng, entrant.trade_params(TRADE)))
register_preset(NAME, Entrant(NAME, kind=NAME))
