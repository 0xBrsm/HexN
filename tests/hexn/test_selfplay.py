# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from typing import Iterator, Sequence

import numpy as np
import pytest
from _bots import needs

from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.actions import apply, legal_mask, space_for
from hexset.arena import deal_game
from hexset.encoding import encode
from hexset.game import Game, to_move
from hexset.gym.lanes import BoardBots
from hexn.selfplay import (
    Choice,
    Collector,
    Episode,
    RandomPolicy,
    Request,
    Transition,
)


def replay(episode: Episode) -> Iterator[tuple[Game, Transition]]:
    """Rebuild the game from its seed and walk it alongside the recorded stream.

    The whole point of the collector is that a lane's interleaved actions come
    back split by seat, so the check that matters is whether the engine agrees:
    at every recorded step, is the seat the transition is filed under actually
    the seat the engine was asking?
    """
    game = deal_game(episode.seed, episode.index, episode.players)
    for transition in episode.stream():
        yield game, transition
        apply(game, transition.action)


class First:
    """Takes the first legal action, and stamps the extras PPO will want kept."""

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        return [
            Choice(
                action=request.options[0],
                log_prob=-float(request.seat),
                value=tuple(float(request.seat + i) for i in range(4)),
            )
            for request in requests
        ]


def test_seats_are_demultiplexed_against_a_replay():
    collector = Collector(RandomPolicy(random.Random(3)), lanes=1, seed=11)
    episode = collector.collect(1)[0]
    stream = episode.stream()

    off_turn = 0
    for step, (game, transition) in enumerate(replay(episode)):
        assert transition.step == step
        assert to_move(game) == transition.seat
        if game.current_player != transition.seat:
            off_turn += 1
        expected = encode(game, transition.seat)
        assert np.array_equal(expected.hexes, transition.observation.hexes)
        assert np.array_equal(expected.vertices, transition.observation.vertices)
        assert np.array_equal(expected.edges, transition.observation.edges)
        assert np.array_equal(expected.globals, transition.observation.globals)

    # None of the above means anything unless the stream really was interleaved
    # and every seat really was demultiplexed out of it.
    assert len(stream) == len(episode) == episode.outcome.actions
    assert all(len(seat) > 1 for seat in episode.trajectories)
    changes = sum(1 for a, b in zip(stream, stream[1:]) if a.seat != b.seat)
    assert changes > episode.players
    # And the hard case actually happened: decisions taken by a seat whose turn
    # it is not, where filing by `current_player` would have looked fine.
    assert off_turn > 0
    for seat in episode.trajectories:
        assert max(b.step - a.step for a, b in zip(seat, seat[1:])) > 1


def test_the_mask_marks_exactly_the_legal_actions():
    collector = Collector(RandomPolicy(random.Random(8)), lanes=1, seed=4, action_cap=150)
    episode = collector.collect(1)[0]
    space = space_for(deal_game(episode.seed, episode.index, episode.players))

    wide = 0
    for game, transition in replay(episode):
        assert np.array_equal(transition.mask, np.array(legal_mask(game, space)))
        assert transition.mask[transition.index]
        wide += transition.mask.sum() > 1
    # A mask with one bit set is trivially right, so insist a good share were
    # real choices. A third rather than a half since contract 5: the offer
    # sample used to make almost every MAIN position a wide one, and without
    # it the forced ROLL/END_TURN steps are a larger share of a game.
    assert wide > len(episode) // 3


def test_a_finished_lane_is_replaced_without_stalling_the_others():
    ticks = 1500
    collector = Collector(RandomPolicy(random.Random(2)), lanes=3, seed=13)
    episodes = collector.run(ticks)

    assert collector.games == len(episodes) > 0
    # Every tick moved every lane, whatever any other lane was doing.
    recorded = sum(len(e) for e in episodes) + sum(collector.pending())
    assert recorded == collector.steps == 3 * ticks
    # Lanes really were at different points, so the replacement was not a
    # lockstep restart of the whole batch.
    assert len(set(collector.pending())) > 1
    assert len({e.index for e in episodes}) == len(episodes)


def test_the_action_cap_truncates_rather_than_running_forever():
    collector = Collector(RandomPolicy(random.Random(4)), lanes=2, seed=9, action_cap=25)
    episodes = collector.run(75)

    assert len(episodes) == 6
    for episode in episodes:
        assert episode.outcome.truncated
        assert episode.outcome.winner is None
        assert episode.outcome.actions == len(episode) == 25


def test_the_policys_extras_are_recorded_against_the_seat_that_acted():
    collector = Collector(First(), lanes=2, seed=17, action_cap=200)
    episodes = collector.run(200)

    assert episodes
    seen = set()
    for episode in episodes:
        for seat, trajectory in enumerate(episode.trajectories):
            for transition in trajectory:
                assert transition.seat == seat
                assert transition.log_prob == -float(seat)
                assert transition.value == tuple(float(seat + i) for i in range(4))
                seen.add(seat)
    assert len(seen) > 1


def test_the_same_seed_collects_the_same_games_at_any_lane_count():
    """The `(seed, index)` law, which is `hexset.arena.deal_game`'s and not
    this module's: game `i` of seed `s` is the same game whichever lane draws
    it and however many lanes are in flight.

    Driven by `First` rather than `RandomPolicy`: a policy sampling from one
    shared stream consumes it in batch order, so the lane count moves the
    *policy* even where the game is untouched, and the invariance under test
    would be swamped by it.
    """

    def once(lanes: int) -> list[tuple]:
        collector = Collector(First(), lanes=lanes, seed=19, deal=4, action_cap=300)
        return sorted(
            (e.index, e.outcome, tuple(t.index for t in e.stream()))
            for e in collector.drain()
        )

    assert once(1) == once(2) == once(4)


def test_a_collector_game_is_the_arenas_game_for_the_same_index():
    """One law, one call. A lane and a tournament playing index `i` of seed `s`
    with the same scripted bot must play the identical game, action for action
    -- which is only guaranteed because both deal through
    `hexset.arena.deal_game` rather than mirroring each other's derivation.
    """
    from hexset.actions import legal_actions
    from hexset.arena import play_game
    from hexset.victory import victory_points

    class Scripted:
        """First legal action, and a log of every one it took. No `gains_many`,
        so it never trades on either side of the comparison."""

        def __init__(self) -> None:
            self.taken: list = []

        def choose(self, game: Game):
            action = legal_actions(game)[0]
            self.taken.append(action)
            return action

    lane_bot = Scripted()
    collector = Collector(
        RandomPolicy(random.Random(0)),
        lanes=1,
        seed=31,
        first_game=7,
        deal=1,
        action_cap=1500,
        opponents=[BoardBots(lambda board: lane_bot)],
        caster=lambda index: (1, 1, 1, 1),
    )
    (played,) = collector.drain()

    alone = Scripted()
    game = play_game(deal_game(31, 7, 4), [alone] * 4, action_cap=1500)

    assert lane_bot.taken == alone.taken
    assert played.index == 7
    assert played.outcome.actions == len(alone.taken)
    assert played.outcome.winner == game.won_by
    assert played.outcome.turns == game.turns
    # true state: terminal VP includes hidden dev cards
    state = game.state(0, hidden=False)
    assert played.outcome.points == tuple(
        victory_points(state, seat) for seat in range(4)
    )


def test_a_bounded_collector_deals_exactly_the_cohort_and_stops():
    """The point is the count, not the filter.

    An evaluation wants a fixed set of game indices; the naive way to get one is
    to let the collector refill freed lanes and discard the replacements, which
    plays every one of them in full first. `deal` stops them being started.
    """
    collector = Collector(
        RandomPolicy(random.Random(3)), lanes=4, seed=5, deal=6, action_cap=200
    )
    episodes = collector.drain()

    assert sorted(e.index for e in episodes) == [0, 1, 2, 3, 4, 5]
    assert collector.games_started() == 6
    assert not collector.running
    assert collector.tick() == []


def test_a_collector_seats_its_policies_as_the_tables_gates():
    """The engine runs its trade events itself and asks `game.gates`
    (`hexset.trading`), so a lane that seats nobody plays a trade-free game
    however its policies would have traded.

    A scripted `BoardBots` opponent seats the bot itself -- `hexset.bots.Bot`
    already carries `gains_many` -- and a policy with no `trader` hook seats
    `None`, which the engine reads as "this seat never trades".
    """
    collector = Collector(RandomPolicy(random.Random(12)), lanes=2, seed=6)
    for game in collector.in_flight():
        assert game.gates == (None,) * 4
        # The table caps nothing (HexSet 0.60): every seat declares its own
        # budget, so a game carries no trade limit of its own.
        assert not hasattr(game, "max_offers") and not hasattr(game, "max_trades")

    # The collector keeps the budget for its own network's gate (`trader`).
    switched_off = Collector(
        RandomPolicy(random.Random(12)), lanes=2, seed=6, max_offers=0
    )
    assert (collector.max_offers, switched_off.max_offers) == (None, 0)


def test_a_collector_hands_its_trader_to_every_network_gate():
    """`trader` rides beside `max_offers` to each network policy's own
    `trader` hook, so the learner and every frozen network in the pool trade
    through the same named bot."""
    asked = []

    class Hooked(RandomPolicy):
        def trader(self, game, seat, max_offers=None, trader=None):
            asked.append((seat, max_offers, trader))
            return None

    collector = Collector(Hooked(random.Random(12)), lanes=1, seed=6, trader="rehex")
    next(collector.in_flight())
    assert asked and all(entry[1:] == (None, "rehex") for entry in asked)


@needs("heximax")
def test_a_trader_is_named_alone_and_must_resolve():
    from hexn.trade import check_trader

    check_trader(None, 0)
    check_trader("heximax", None)
    with pytest.raises(ValueError, match="drop --max-offers"):
        check_trader("heximax", 0)
    with pytest.raises(Exception):
        check_trader("nobody-by-that-name", None)


@needs("heximax")
def test_a_scripted_opponent_is_seated_as_its_own_trader():
    from hexset.arena import entrant_from_name, spawn

    entrant = entrant_from_name("heximax")
    bots = BoardBots(lambda board: spawn(entrant, board, random.Random(0)))
    collector = Collector(
        RandomPolicy(random.Random(1)),
        lanes=1,
        seed=7,
        opponents=[bots],
        caster=lambda index: (0, 1, 0, 1),
    )
    game = next(collector.in_flight())
    assert game.gates[0] is None
    assert game.gates[2] is None
    # public field
    seated = bots.bot(game.state(0, hidden=False).board)
    assert game.gates[1] is seated
    assert game.gates[3] is seated
    assert hasattr(seated, "gains_many")


def test_the_cast_is_taken_from_the_game_index_not_the_lane():
    def swaps(index: int) -> tuple[int, ...]:
        return (0, 1, 0, 1) if index % 2 == 0 else (1, 0, 1, 0)

    collector = Collector(
        RandomPolicy(random.Random(2)),
        lanes=3,
        seed=9,
        action_cap=200,
        opponents=[RandomPolicy(random.Random(3))],
        caster=swaps,
    )
    episodes = collector.collect(4)
    assert {e.index % 2 for e in episodes} == {0, 1}, "both parities must appear"
    for episode in episodes:
        assert episode.cast == swaps(episode.index)
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_a_strided_cohort_collector_keeps_its_shard_across_cohorts():
    """Worker `w` of `K` owns `first_game + nK`; a second cohort must stay on
    that shard, or two workers start colliding after one iteration."""
    collector = Collector(
        RandomPolicy(random.Random(1)),
        lanes=2,
        fill=False,
        seed=4,
        first_game=1,
        stride=3,
        action_cap=200,
    )
    assert {e.index for e in collector.cohort(3)} == {1, 4, 7}
    assert {e.index for e in collector.cohort(3)} == {10, 13, 16}


def test_two_learners_share_a_table_and_each_records_only_its_seats():
    """The table league's stage-1 contract, torch-free.

    Two learners seated by parity, both recording: every seat records, every
    trajectory belongs to the seat its caster assigned, and `owned` partitions
    the table with no overlap and no loss.
    """
    from hexn.selfplay import owned

    collector = Collector(
        RandomPolicy(random.Random(0)),
        lanes=4,
        seed=11,
        action_cap=300,
        opponents=(RandomPolicy(random.Random(99)),),
        caster=lambda index: (0, 1, 0, 1) if index % 2 == 0 else (1, 0, 1, 0),
        learners=(0, 1),
    )
    episodes = collector.collect(4)

    for episode in episodes:
        assert episode.cast, "a casting collector stamps the cast"
        for seat, trajectory in enumerate(episode.trajectories):
            assert trajectory, "both ids are learners, so every seat records"
            assert all(t.seat == seat for t in trajectory)

    def count(eps):
        return sum(len(t) for e in eps for t in e.trajectories)

    zero, one = owned(episodes, 0), owned(episodes, 1)
    for share, learner in ((zero, 0), (one, 1)):
        for episode, original in zip(share, episodes):
            for seat, trajectory in enumerate(episode.trajectories):
                if original.cast[seat] == learner:
                    assert trajectory == original.trajectories[seat]
                else:
                    assert trajectory == ()
    assert count(zero) + count(one) == count(episodes)
    assert count(zero) and count(one)


def test_a_board_pair_shares_its_geometry_and_not_its_dice():
    from hexset.board.board import random_base_board

    collector = Collector(
        RandomPolicy(random.Random(0)),
        lanes=4,
        seed=9,
        pair_boards=True,
        deal=4,
        action_cap=200,
    )
    # public field
    boards = [game.state(0, hidden=False).board for game in collector.in_flight()]

    assert boards[0] == boards[1]
    assert boards[2] == boards[3]
    assert boards[0] != boards[2], "distinct pairs still draw distinct boards"
    # The even half's board is the one unpaired dealing derives for the same
    # index — pairing extends the (seed, index) law rather than amending it.
    assert boards[0] == random_base_board(random.Random("9:0:board"))
    assert boards[2] == random_base_board(random.Random("9:2:board"))

    episodes = sorted(collector.drain(), key=lambda e: e.index)
    even, odd = episodes[0], episodes[1]
    # Same geometry, its own `{seed}:{index}:game` rng each: the halves must
    # play different games, or the mate's reward is no baseline at all.
    assert [t.action for t in even.stream()] != [t.action for t in odd.stream()]
