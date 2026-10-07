# SPDX-License-Identifier: GPL-3.0-only
"""The two seams hexn keeps now that the checkpoint runtime is HexSet's, both
provable without torch.

`hexset.clients.netbot` owns the bot, the leaf evaluation, the search and the
trade gate, and tests them against its own stub `Policy`
(`hexset/tests/clients/test_netbot.py`). What is left on this side is the
wiring: a *searched* seat has to trade through the same gate its raw policy
would, and the entrant kinds a torch process registers have to come out of
`hexset.arena.spawn` in the right shape.

A stub `Policy` stands in for `hexn.policy.NetworkPolicy` so both can be
checked in a devcontainer with no torch in it; the torch policy itself is
pinned by `test_netbot.py` and `test_policy.py`, which need the box.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from hexset.actions import ActionSpace, build_space
from hexset.board.board import random_base_board
from hexset.economy import hand_size
from hexset.clients.netbot import (
    GatedSearch,
    LeafEvaluator,
    NetworkBot,
    register_entrants,
)
from hexn.trade import trade_params
from hexset.game import start
from hexset.mcts import Search
from hexset.trading import one_for_one
from hexn.expert import SearchPolicy

PLAYERS = 4


@dataclass
class StubPolicy:
    """`hexset.clients.policy.Policy` with a hand count for a value head, plus
    the one method the protocol does not ask for and `hexn` adds:
    `trader`, which seats this policy behind the engine's own gate exactly as
    `hexn.policy.NetworkPolicy.trader` does."""

    space: ActionSpace

    def act_rows(self, rows):
        return [min(options, key=self.space.index) for _, _, options in rows]

    def value_rows(self, rows):
        return [self._value(game) for game, _ in rows]

    def score_rows(self, rows):
        return [
            ([1.0 / len(options)] * len(options), self._value(game))
            for game, _, options in rows
        ]

    def _value(self, game):
        # By size: the gate values the position an exchange leaves from its
        # own seat's frame, where the other seats' hands are hidden piles
        # (HexSet 1.1), so their composition cannot be read.
        state = game.state(0, hidden=False)
        return tuple(float(hand_size(state, seat)) for seat in range(state.num_players))

    def trader(self, game, seat, max_offers=None):
        gate = NetworkBot(
            policy=self,
            players=PLAYERS,
            trade=trade_params(max_offers),
            seat=seat,
        )
        gate.seat_at(game)
        return gate


@dataclass(frozen=True)
class StubCheckpoint:
    """`hexset.clients.policy.Checkpoint` over `StubPolicy` -- the shape
    `hexn.netbot.Loaded` has."""

    policy: StubPolicy
    space: ActionSpace
    players: int = PLAYERS
    max_offers: int | None = None


@pytest.fixture
def board():
    return random_base_board(random.Random(0))


def a_checkpoint(board) -> StubCheckpoint:
    space = build_space(
        board.topology.num_vertices,
        board.topology.num_edges,
        board.topology.num_hexes,
        PLAYERS,
    )
    return StubCheckpoint(policy=StubPolicy(space=space), space=space)


def test_a_searched_seat_trades_through_the_same_gate_its_policy_would(board):
    """`hexn.expert.SearchPolicy.trader` is what `hexn.selfplay.Collector`
    seats on `game.gates` for an expert-iteration lane. A search decides
    moves and nothing else, so the gate has to come from the policy behind
    its leaves -- and it has to be the engine's `NetworkBot`, seated at this
    position, not a second implementation."""
    checkpoint = a_checkpoint(board)
    game = start(board, PLAYERS, random.Random(2))
    # Hands on both sides, so the candidate below is coverable: an
    # uncoverable one -- this seat's hand short, or the counterparty's public
    # hand with no room for its side -- is refused before the value head sees
    # it and would pass this vacuously.
    game.state(0, hidden=False).hands[1] = [3] * 5
    game.state(0, hidden=False).hands[2] = [3] * 5
    search = Search(
        LeafEvaluator(policy=checkpoint.policy),
        simulations=4,
        wave=2,
        rng=random.Random(0),
    )

    gate = SearchPolicy(search).trader(game, 1, None)

    assert isinstance(gate, NetworkBot)
    assert gate._seated is game
    assert gate.seat == 1
    # Anti-vacuity: a gate nobody seated refuses everything. Value is hand
    # size here, so a one-for-one is worth exactly nothing -- a *priced*
    # answer, not the -1.0 an unseated gate returns, which proves the
    # position came across as well as the type.
    assert gate.gains_many(game.state(1), [one_for_one(0, 1)], [2]) == [0.0]


def test_a_search_over_handcrafted_leaves_seats_no_gate(board):
    """No policy behind the leaves, no gate to derive -- the seat plays
    trade-free, matching every scripted bot rather than guessing."""

    class Handcrafted:
        def evaluate(self, leaves):
            return [([1.0 / len(leaf.options)] * len(leaf.options), (0.0,) * PLAYERS)
                    for leaf in leaves]

    game = start(board, PLAYERS, random.Random(2))
    search = Search(Handcrafted(), simulations=4, wave=2, rng=random.Random(0))

    assert SearchPolicy(search).trader(game, 1, None) is None


def test_the_entrant_kinds_a_runtime_registers_come_out_in_the_right_shape(
    board, arena_registry
):
    """The registration `hexn.netbot` makes at import, with a stub loader in
    place of `torch.load`: one call, and the whole runtime is spawnable.

    Both kinds resolve through the loader, and a searched seat carries a
    `NetworkBot` gate rather than none -- the difference between a checkpoint
    that trades in a duel and one that silently does not.
    """
    from hexset.arena import Entrant, spawn

    checkpoint = a_checkpoint(board)
    register_entrants(lambda path, topology: checkpoint)

    plain = spawn(Entrant("stub", kind="network", weights="a-path"), board, random.Random(0))
    searched = spawn(
        Entrant("stub", kind="mcts", weights="a-path", simulations=4, wave=2),
        board,
        random.Random(0),
    )

    assert isinstance(plain, NetworkBot)
    assert isinstance(searched, GatedSearch)
    assert isinstance(searched.gate, NetworkBot)


@pytest.fixture
def arena_registry():
    """`hexset.arena`'s entrant-kind registry is process-global, so a test
    that registers into it puts it back."""
    from hexset import arena

    kinds = dict(arena._ENTRANT_KIND_FACTORIES)
    yield
    arena._ENTRANT_KIND_FACTORIES.clear()
    arena._ENTRANT_KIND_FACTORIES.update(kinds)
