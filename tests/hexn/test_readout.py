# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import ActionType, build_space, legal_actions, space_for
from hexset.board.board import random_base_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.topology import build as build_topology
from hexset.game import start
from hexset.play import step_randomly
from hexn.readout import EDGES, GLOBALS, HEXES, VERTICES, _SOURCES, plan, scatter_logits


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def test_every_flat_slot_is_written_exactly_once():
    # Two seats: `test_the_plan_is_sized_from_the_board` covers four.
    space = space_for(a_game(players=2))
    readout = plan(space)

    written = np.concatenate([head.scatter.ravel() for head in readout.heads])
    assert written.size == space.size
    assert sorted(written.tolist()) == list(range(space.size))


@pytest.mark.parametrize("players", [2, 4])
def test_each_slot_decodes_to_the_action_its_head_claims(players):
    """The scatter against `decode`, which is the canonical index arithmetic.

    Checking it against `index` instead would just restate the same formula.
    """
    space = space_for(a_game(players=players))

    for head in plan(space).heads:
        for node in range(head.num_nodes):
            for column in range(head.width):
                action = space.decode(int(head.scatter[node, column]))
                assert action.type in head.kinds
                assert _SOURCES.get(action.type, GLOBALS) == head.source
                if head.source != GLOBALS:
                    assert action.a == node


def test_legal_actions_reach_the_head_that_owns_them():
    """Over played positions, so every phase's actions get exercised."""
    rng = random.Random(7)
    game = start(random_base_board(rng), 4, rng)
    space = space_for(game)
    readout = plan(space)
    seen: set[ActionType] = set()

    for _ in range(600):
        for action in legal_actions(game):
            head = readout.head(_SOURCES.get(action.type, GLOBALS))
            row = action.a if head.source != GLOBALS else 0
            assert space.index(action) in head.scatter[row]
            seen.add(action.type)
        step_randomly(game, rng)

    # A vacuous pass is the failure mode worth guarding: the assertions above
    # never run if no action of a given kind ever comes up.
    assert {ActionType.ROLL, ActionType.END_TURN, ActionType.BUILD_ROAD} <= seen
    assert {ActionType.MOVE_ROBBER, ActionType.SETUP_SETTLEMENT} <= seen


def test_head_widths_are_what_the_model_must_emit():
    players = 4
    readout = plan(space_for(a_game(players=players)))

    assert readout.head(VERTICES).width == 3  # setup, settlement, city
    assert readout.head(EDGES).width == 2  # setup, road
    assert readout.head(HEXES).width == players + 1  # robber only: hexset.actions sizes PLAY_KNIGHT as operand-less (1), so it has no per-hex slot here
    assert readout.head(GLOBALS).num_nodes == 1


def test_scatter_logits_round_trips():
    space = space_for(a_game())
    readout = plan(space)

    outputs = {
        head.source: head.scatter.astype(np.float32) for head in readout.heads
    }
    flat = scatter_logits(readout, outputs)

    assert np.array_equal(flat, np.arange(space.size, dtype=np.float32))


def test_the_plan_is_sized_from_the_board():
    """A different layout narrows the space without any change here."""
    mini = build_topology(MINI_LAYOUT)
    space = build_space(mini.num_vertices, mini.num_edges, mini.num_hexes, 4)
    readout = plan(space)

    assert readout.head(VERTICES).num_nodes == mini.num_vertices == 24
    assert readout.head(EDGES).num_nodes == mini.num_edges == 30
    assert readout.head(HEXES).num_nodes == mini.num_hexes == 7

    written = np.concatenate([head.scatter.ravel() for head in readout.heads])
    assert sorted(written.tolist()) == list(range(space.size))
