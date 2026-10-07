# SPDX-License-Identifier: GPL-3.0-only
"""`coalition(...)` in `--mix`: a table with frozen seats ganging up on one
target, drawn per game, and `hexn.coalition.Targeted`, the member that
plays it off the game's plan."""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import ActionType as A, apply, legal_actions  # noqa: E402
from hexset.arena import deal_game, entrant_from_name, spawn  # noqa: E402
from hexset.bots import RandomBot  # noqa: E402
from hexset.game import is_over, to_move  # noqa: E402
from hexset.robber import occupants  # noqa: E402
from hexn.collect import (  # noqa: E402
    TARGETED,
    CoalitionPlan,
    _mix_draw,
    check_mix,
    coalition_term,
    mix_caster,
    mix_coalitions,
    mix_names,
    mix_opponents,
    parse_mix,
)

SPEC = "coalition(self|self|random;members=random;size=1-2;lead=0-1;start=0.5;targeted=0.5)"
MIX = [(SPEC, 1.0)]


def test_a_coalition_entry_parses_its_pool_members_and_draw():
    term = coalition_term(SPEC)
    assert term.pool == ("self", "self", "random")
    assert term.members == ("random",)
    assert (term.size, term.lead, term.start, term.targeted) == ((1, 2), (0, 1), 0.5, 0.5)
    # Defaults, and the members read off the pool's frozen names.
    plain = coalition_term("coalition(self|random)")
    assert plain.members == ("random",)
    assert (plain.size, plain.lead, plain.start, plain.targeted) == ((1, 3), (0, 2), 0.25, 0.5)
    assert coalition_term("table(self|random)") is None
    assert parse_mix(SPEC + "=0.5") == [(SPEC, 0.5)]


@pytest.mark.parametrize("name", [
    "coalition(self)",                      # nobody frozen to be a member
    "coalition(self|random;members=self)",  # a member is never the learner
    "coalition(self|random;foo=1)",
    "coalition(self|random;size=0-2)",
    "coalition(self|random;size=3-1)",
    "coalition(self|random;start=2)",
    "coalition(self|random;lead=x)",
    "coalition(self|random",
])
def test_a_malformed_coalition_entry_is_refused(name):
    with pytest.raises(ValueError):
        coalition_term(name)


def test_members_are_cast_under_their_targeted_spelling():
    assert mix_names(MIX) == ["random", TARGETED + "random"]
    assert mix_names([("table(self|random)", 0.5), (SPEC, 0.5)]) == ["random", TARGETED + "random"]
    # A plain or table mix draws no plan at all.
    assert mix_coalitions([("table(self|random)", 1.0)], 4, 1) is None


def test_the_draw_seats_a_learner_a_target_and_members_pure_in_the_index():
    draw = _mix_draw(MIX, 4, seed=3)
    targeted_id = mix_names(MIX).index(TARGETED + "random") + 1
    starts = targeted = 0
    for index in range(600):
        cast, temperatures, retired, plan = draw(index)
        assert retired is None and plan is not None
        assert temperatures == (1.0,) * 4
        assert plan.target in range(4)
        assert 1 <= len(plan.members) <= 2
        assert plan.target not in plan.members
        for seat, pid in enumerate(cast):
            assert (pid == targeted_id) == (seat in plan.members), (cast, plan)
            assert pid in {0, 1, targeted_id}
        assert any(pid == 0 and seat not in plan.members for seat, pid in enumerate(cast))
        assert plan.lead in (None, 0, 1)
        starts += plan.lead is None
        targeted += cast[plan.target] == 0
        again = draw(index)
        assert again[0] == cast and again[3] == plan and again[3] is not plan
    assert 240 < starts < 360, starts
    assert targeted > 300, targeted            # the learner's seat, plus self draws
    # Pairing shares one draw between mates.
    law = mix_coalitions(MIX, 4, 3, pair_boards=True)
    assert law(10) == law(11) and law(10) != law(12)
    assert mix_caster(MIX, 4, 3)(5) == draw(5)[0]


def test_a_plan_turns_hostile_once_the_target_leads_and_latches():
    game = deal_game(11, 0, 4)
    plan = CoalitionPlan(target=0, members=frozenset({1, 2}), lead=0)
    assert not plan.hostile(game), "nobody has built past setup yet"
    state = game.state(0, hidden=False)
    state.vertex_owner[0], state.vertex_building[0] = 0, 2
    assert not plan.hostile(game), "a city alone is setup's two points"
    state.vertex_owner[2], state.vertex_building[2] = 0, 1
    assert plan.hostile(game)
    state.vertex_owner[2], state.vertex_building[2] = 1, 2
    assert plan.hostile(game), "a coalition that has turned stays turned"
    assert CoalitionPlan(0, frozenset({1}), lead=None).hostile(game)
    assert not CoalitionPlan(0, frozenset({1}), lead=5).hostile(game)


def _play(game, bots, watch, cap=4000):
    for _ in range(cap):
        if is_over(game):
            break
        seat = to_move(game)
        action = bots[seat].choose(game)
        watch(game, seat, action)
        apply(game, action)


def test_a_targeted_seat_robs_the_target_alone_from_the_trigger_on():
    from hexn.coalition import Targeted

    game = deal_game(21, 0, 4)
    bots = [RandomBot(random.Random(s)) for s in range(4)]
    bots[1], bots[2] = Targeted(RandomBot(random.Random(5))), Targeted(RandomBot(random.Random(6)))
    game.gates = tuple(bots)
    game.coalition = CoalitionPlan(target=0, members=frozenset({1, 2}), lead=None)
    moves = []

    def watch(game, seat, action):
        if seat in (1, 2) and action.type is A.MOVE_ROBBER:
            state = game.state(seat).state
            possible = any(0 in occupants(state, a.a) for a in legal_actions(game, seat))
            moves.append((possible, 0 in occupants(state, action.a)))

    _play(game, bots, watch)
    assert moves and any(possible for possible, _ in moves)
    assert all(hit for possible, hit in moves if possible)
    assert bots[1].targets == bots[2].targets == frozenset({0})

    # An untriggered plan leaves the member its own bot.
    game = deal_game(22, 0, 4)
    bots = [RandomBot(random.Random(s)) for s in range(4)]
    bots[3] = Targeted(RandomBot(random.Random(9)))
    game.gates = tuple(bots)
    game.coalition = CoalitionPlan(target=0, members=frozenset({3}), lead=9)
    _play(game, bots, lambda *_: None)
    assert bots[3].targets == frozenset()


def test_the_targeted_spec_is_an_arena_entrant():
    import hexn.coalition  # noqa: F401  (registers the spec)
    from hexn.coalition import Targeted

    entrant = entrant_from_name(TARGETED + "random")
    assert entrant.kind == "targeted"
    board = deal_game(1, 0, 4).state(0, hidden=False).board
    assert isinstance(spawn(entrant, board, random.Random(0)), Targeted)
    with pytest.raises(ValueError):
        entrant_from_name(TARGETED)
    check_mix(MIX, have_parent=False)


def test_a_collector_hangs_the_plan_on_the_game_and_trains_on_no_member():
    import hexn.coalition  # noqa: F401
    from hexn.coalition import Targeted
    from hexn.selfplay import Collector, RandomPolicy

    seed = 5
    opponents = mix_opponents(MIX, seed=1, lanes=2)
    law = mix_coalitions(MIX, 4, seed)
    collector = Collector(
        RandomPolicy(random.Random(1)),
        lanes=2,
        seed=seed,
        action_cap=400,
        opponents=opponents,
        caster=mix_caster(MIX, 4, seed),
        coalition=law,
    )
    for game in collector.in_flight():
        plan = game.coalition
        assert plan is not None
        for seat, gate in enumerate(game.gates):
            assert isinstance(gate, Targeted) == (seat in plan.members)
    episodes = collector.collect(4)
    assert episodes
    for episode in episodes:
        plan = law(episode.index)
        assert plan.members and plan.target not in plan.members
        for seat, pid in enumerate(episode.cast):
            if seat in plan.members:
                assert pid != 0 and not episode.trajectories[seat]
            assert bool(episode.trajectories[seat]) == (pid == 0)
    pytest.importorskip("torch")
    from hexn.ppo.__main__ import coalition_gauges

    gauges = coalition_gauges(episodes, law)
    assert gauges["coalition_share"] == 1.0
    assert set(gauges) == {
        "coalition_share", "coalition_targeted_share", "coalition_targeted_win_rate",
        "coalition_neutral_win_rate", "coalition_free_win_rate",
    }
