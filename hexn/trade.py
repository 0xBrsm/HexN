# SPDX-License-Identifier: GPL-3.0-only
"""How the networks hexn trains and plays trade, in HexSet's terms.

HexSet decides how trading works; a table no longer caps it. Each bot
declares its own `hexset.trading.TradeParams`, and a checkpoint declares what
its network trained with. hexn has two trade settings for its networks;
opponents bring their own.

`max_offers` is how many offers the network makes a turn, priced by its own
value head. `0` is the no-trade network. `None` is HexSet's own default, which
is unlimited (HexSet 0.60 onward). `-1` is hexn's old spelling of "uncapped"
and reads as `None`.

`trader` names a bot -- a preset or spec, as a HexSet lineup spells it --
that answers every trade for the network instead (`trader_gate`): the network
still plays every move, bank trades included. It is the network half of
HexSet's `<entrant>~<trader>` lineup spelling, for the seats hexn gates
itself. HexSet ships no playing bots, so a name like `heximax` resolves only
in a process that loaded the runtime registering it (`hexn.runtime`).
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Mapping

from hexset.game import Game
from hexset.trading import UNLIMITED, TradeParams


def offer_budget(value: int | str | None) -> int | None:
    """`value` as a HexSet `max_offers`: `None` (unlimited) for a missing,
    empty or `-1` value, else the non-negative count itself."""
    if value is None or value == "":
        return None
    count = int(value)
    if count == -1:
        return None
    if count < 0:
        raise ValueError(f"an offer budget is 0 or more, or -1 for unlimited; got {count}")
    return count


def recorded_budget(args: Mapping) -> int | None:
    """The budget a run recorded in its saved arguments: `max_offers`, or the
    pre-0.24 `max_trades` key a checkpoint written before then carries. Read
    the way HexSet reads the same key out of an exported file."""
    value = args.get("max_offers", args.get("max_trades"))
    return offer_budget(value)


def trade_params(max_offers: int | None) -> TradeParams:
    """HexSet's `TradeParams` for a network declaring `max_offers`, with
    every other setting at HexSet's default."""
    budget = offer_budget(max_offers)
    return UNLIMITED if budget is None else replace(UNLIMITED, max_offers=budget)


def check_trader(trader: str | None, max_offers: int | None) -> None:
    """Refuse a `trader` HexSet cannot build, or one named beside an offer
    budget: a seat traded for by another bot has no budget of its own, and
    `max_offers 0` already says it does not trade."""
    from hexset.arena import entrant_from_name

    if trader is None:
        return
    if max_offers is not None:
        raise ValueError(
            f"--trader {trader} answers the network's trades with its own limits; "
            f"drop --max-offers {max_offers}"
        )
    entrant_from_name(trader)


def trader_gate(name: str, game: Game, seat: int):
    """The HexSet bot `name` names, built on `game`'s board to answer one
    network seat's trades: the trader side of `hexset.bots.TradesBy`, the
    moves left to the network's batched policy. It prices off the public view
    and is never asked to move, so a search bot here never searches. Seeded
    by the seat, so a replayed game trades the same way."""
    # Function-scoped, as everywhere hexn reaches `hexset.arena`: a spawn can
    # pull a runtime this module's importers must not pay for.
    from hexset.arena import entrant_from_name, spawn

    return spawn(entrant_from_name(name), game.state(seat).state.board, random.Random(seat))
