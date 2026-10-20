# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.policy9.best` and `ranked`: how a contract-9 net's scores choose a
whole decision over `hexset.decisions.Decision`'s blocks -- every legal
option listed, the highest-scored chosen, a tie to the earlier block and then
the lower index. An exported graph chooses by the same rule inside it."""
from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("torch", reason="hexn.policy9 imports torch")

from hexset import decisions as D  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.contract9 import incidences  # noqa: E402
from hexset.game import Phase, is_over, players_owing_discards, start, to_move  # noqa: E402
from hexn.policy9 import Scored, best, ranked  # noqa: E402


def _decider(game):
    return players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game)


def _play(seed, until):
    """A game played on random whole decisions until `until(game)` holds."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(3000):
        if is_over(game) or until(game):
            return game
        seat = _decider(game)
        D.apply_option(game, rng.choice(D.options(game, seat)), seat if game.phase is Phase.DISCARD else None)
    return None


def _scored_at_random(decision, rng):
    """A `Scored` with a distinct random number on every entry of every block."""
    incidences_ = len(decision.setup)
    return Scored(
        flat=np.asarray([rng.random() for _ in range(len(decision.flat))]),
        setup=np.asarray([rng.random() for _ in range(incidences_)]),
        road_building=np.asarray([rng.random() for _ in range(len(decision.road_building))]),
        discard=np.asarray([rng.random() for _ in range(len(decision.discard))]),
        double=np.asarray([[rng.random() for _ in range(incidences_)] for _ in range(incidences_)])
        if decision.double else None,
    )


def test_ranked_lists_every_legal_option_and_best_is_its_highest():
    rng = random.Random(11)
    for seed in range(4):
        game = start(random_base_board(random.Random(seed)), 4, random.Random(seed))
        space = space_for(game)
        topology = game._state.board.topology
        pairs = incidences(topology)
        for _ in range(300):
            if is_over(game):
                break
            seat = _decider(game)
            options = D.options(game, seat)
            decision = D.tensors(game, seat, space)
            scored = _scored_at_random(decision, rng)
            listed = ranked(decision, scored, topology)
            assert len(listed) == len(options)
            assert {D.key(D.option_of(decision, k, i, pairs, seat, space, 4)) for _, k, i in listed} \
                == {D.key(o) for o in options}
            top = max(listed, key=lambda item: item[0])
            assert best(decision, scored, topology) == (top[1], top[2])
            D.apply_option(game, rng.choice(options), seat if game.phase is Phase.DISCARD else None)


def test_best_is_the_highest_scored_legal_option_ties_to_the_earlier_block_then_the_lower_index():
    def two_road_sets(g):
        return (g.phase is Phase.MAIN and g.free_roads == 0
                and len(D.tensors(g, to_move(g), space_for(g)).road_building) >= 2)

    game = next(g for g in (_play(seed, two_road_sets) for seed in range(60)) if g is not None and two_road_sets(g))
    seat = to_move(game)
    space = space_for(game)
    topology = game._state.board.topology
    decision = D.tensors(game, seat, space)
    legal = np.flatnonzero(decision.flat)
    flat = np.full(space.size, 9.0)            # illegal entries are never read, however high
    flat[legal] = 0.0
    scored = Scored(flat=flat, setup=np.full(len(decision.setup), 9.0),
                    road_building=np.zeros(len(decision.road_building)), discard=np.zeros(0))
    assert best(decision, scored, topology) == (D.DecisionKind.FLAT, int(legal[0]))
    scored = Scored(flat=flat, setup=scored.setup, discard=np.zeros(0),
                    road_building=np.r_[0.5, 0.5, np.zeros(len(decision.road_building) - 2)])
    assert best(decision, scored, topology) == (D.DecisionKind.ROAD_BUILDING, 0)
    flat[legal[-1]] = 0.5
    assert best(decision, scored, topology) == (D.DecisionKind.FLAT, int(legal[-1]))
    with pytest.raises(ValueError, match="candidates"):
        best(decision, Scored(flat=flat, setup=scored.setup, discard=np.zeros(0), road_building=np.zeros(1)),
             topology)


def test_a_double_turn_is_chosen_among_the_pairs_the_board_allows():
    game = _play(2, lambda g: D.is_double_turn(g))
    seat, space, topology = to_move(game), space_for(game), game._state.board.topology
    decision = D.tensors(game, seat, space)
    pairs = incidences(topology)
    double = np.zeros((len(pairs), len(pairs)))
    ok = D.double_pairs(topology)
    first = int(np.flatnonzero(decision.setup)[0])
    illegal = next(j for j in range(len(pairs)) if decision.setup[j] and not ok[first, j])
    double[first, illegal] = 5.0                # a pair the board refuses (the same vertex or neighbours)
    legal = next(j for j in range(len(pairs)) if decision.setup[j] and ok[first, j])
    double[first, legal] = 1.0
    scored = Scored(flat=np.zeros(space.size), setup=np.full(len(pairs), 9.0), road_building=np.zeros(0),
                    discard=np.zeros(0), double=double)
    kind, index = best(decision, scored, topology)
    assert (kind, index) == (D.DecisionKind.DOUBLE, first * len(pairs) + legal)
    option = D.option_of(decision, kind, index, pairs, seat, space, 4)
    assert option.kind is D.CompoundKind.SETUP_DOUBLE and option.parts[0].a == pairs[first][0]
