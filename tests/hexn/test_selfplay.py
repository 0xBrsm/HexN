# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from typing import Iterator, Sequence

import numpy as np
import pytest

from hexset.actions import apply, legal_mask, space_for
from hexset.arena import deal_game
from hexset.encoding import HAND_SCALE, encode
from hexset.board.terrain import NUM_RESOURCES
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


class Counting:
    """Wraps a policy and records the shape of every batch it was given."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.batches: list[int] = []

    def act(self, requests: Sequence[Request]) -> Sequence[Choice]:
        self.batches.append(len(requests))
        return self.inner.act(requests)


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


def test_one_batch_per_tick_covering_every_lane():
    """The reason this module exists: a forward costs 1.5 ms plus 25 µs a
    position, so a tick must be one call carrying every lane."""
    policy = Counting(RandomPolicy(random.Random(0)))
    collector = Collector(policy, lanes=5, seed=1, action_cap=60)
    collector.run(30)

    assert len(policy.batches) == collector.ticks == 30
    assert set(policy.batches) == {5}


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


def test_a_seat_asked_off_turn_is_encoded_from_its_own_perspective():
    """Discarding on a seven and answering an offer belong to somebody other
    than the player whose turn it is, and `encode` defaults to the turn holder.
    """
    collector = Collector(RandomPolicy(random.Random(6)), lanes=1, seed=2)
    episode = collector.collect(1)[0]

    checked = 0
    for game, transition in replay(episode):
        if game.current_player == transition.seat:
            continue
        own_hand = transition.observation.globals[:NUM_RESOURCES] * HAND_SCALE
        # true state: verifies own-hand encoding against the true hand
        assert own_hand == pytest.approx(
            game.state(transition.seat, hidden=False).hands[transition.seat], abs=1e-4
        )
        checked += 1
    assert checked > 0


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
    ticks = 3000
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


def test_an_outcome_carries_both_candidate_rewards():
    """Reward design is still open, so a collector reports the terminal facts
    and leaves the scalarisation to whoever consumes it."""
    collector = Collector(RandomPolicy(random.Random(5)), lanes=2, seed=3)
    episodes = collector.collect(2)

    for episode in episodes:
        outcome = episode.outcome
        assert not outcome.truncated
        assert outcome.winner is not None
        assert len(outcome.points) == episode.players
        assert outcome.points[outcome.winner] >= 10
        assert outcome.points[outcome.winner] == max(outcome.points)
        assert outcome.turns > 0


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
        collector = Collector(First(), lanes=lanes, seed=19, deal=4, action_cap=1500)
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
    collector = Collector(RandomPolicy(random.Random(3)), lanes=4, seed=5, deal=6)
    episodes = collector.drain()

    assert sorted(e.index for e in episodes) == [0, 1, 2, 3, 4, 5]
    assert collector.games_started() == 6
    assert not collector.running
    assert collector.tick() == []


def test_asking_a_bounded_collector_for_more_than_it_has_fails_loudly():
    """Otherwise `collect` spins on empty ticks instead of blocking on a game."""
    collector = Collector(RandomPolicy(random.Random(3)), lanes=2, seed=5, deal=3)
    with pytest.raises(ValueError, match="4 games wanted, 3 left"):
        collector.collect(4)


def test_a_policy_that_answers_the_wrong_number_of_requests_is_rejected():
    class Short:
        def act(self, requests):
            return [Choice(action=requests[0].options[0])]

    collector = Collector(Short(), lanes=2, seed=0)
    with pytest.raises(ValueError, match="answered 1 of 2"):
        collector.tick()


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
        # `Collector`'s own `max_trades=None` means "the engine's own
        # default" -- one broadcast round a turn (`LaneEnv` maps `None` to
        # `1`), not an unbounded table.
        assert game.max_trades == 1

    switched_off = Collector(
        RandomPolicy(random.Random(12)), lanes=2, seed=6, max_trades=0
    )
    for game in switched_off.in_flight():
        assert game.max_trades == 0


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


def test_a_cast_lane_routes_each_seat_and_records_only_the_learner():
    learner = Counting(RandomPolicy(random.Random(0)))
    opponent = Counting(RandomPolicy(random.Random(1)))
    collector = Collector(
        learner,
        lanes=2,
        seed=5,
        action_cap=900,
        opponents=[opponent],
        caster=lambda index: (0, 1, 0, 1),
    )
    episode = collector.collect(1)[0]

    assert episode.cast == (0, 1, 0, 1)
    # The learner's seats have trajectories; the opponent's are empty, which is
    # what keeps its decisions out of `hexn.ppo.assemble` without a filter.
    assert all(episode.trajectories[seat] for seat in (0, 2))
    assert all(not episode.trajectories[seat] for seat in (1, 3))
    # The game itself still ran through every seat.
    assert len(episode) < episode.outcome.actions
    # Each policy was asked at most once per tick — the batching survived.
    assert len(learner.batches) <= collector.ticks
    assert len(opponent.batches) <= collector.ticks
    assert opponent.batches, "the opponent was never consulted"


def test_the_cast_is_taken_from_the_game_index_not_the_lane():
    def swaps(index: int) -> tuple[int, ...]:
        return (0, 1, 0, 1) if index % 2 == 0 else (1, 0, 1, 0)

    collector = Collector(
        RandomPolicy(random.Random(2)),
        lanes=3,
        seed=9,
        action_cap=700,
        opponents=[RandomPolicy(random.Random(3))],
        caster=swaps,
    )
    episodes = collector.collect(4)
    assert {e.index % 2 for e in episodes} == {0, 1}, "both parities must appear"
    for episode in episodes:
        assert episode.cast == swaps(episode.index)
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_a_caster_without_opponents_is_rejected():
    with pytest.raises(ValueError):
        Collector(RandomPolicy(), lanes=1, caster=lambda index: (0, 0, 0, 0))


def test_a_cast_reaching_past_the_opponents_fails_loudly():
    with pytest.raises(ValueError):
        Collector(
            RandomPolicy(),
            lanes=1,
            opponents=[RandomPolicy()],
            caster=lambda index: (0, 2, 0, 2),
        )


def test_a_bench_spawns_one_bot_per_board_and_reuses_it():
    from hexset.bots import RandomBot

    spawned = []

    def spawn(board):
        spawned.append(board)
        return RandomBot(random.Random(4))

    collector = Collector(
        RandomPolicy(random.Random(5)),
        lanes=2,
        seed=13,
        opponents=[BoardBots(spawn)],
        caster=lambda index: (0, 1, 0, 1),
    )
    collector.run(40)

    # Two lanes, two boards, two bots — and no respawn on any later request.
    assert len(spawned) == 2
    assert len({id(board) for board in spawned}) == 2


def test_a_cohort_larger_than_the_lane_count_refills_until_it_is_dealt_out():
    """Lanes are the concurrency, not the cohort.

    Keeping them independent is what lets the inference batch be chosen for
    throughput without deciding how many positions an update trains on.
    """
    collector = Collector(
        RandomPolicy(random.Random(4)), lanes=4, fill=False, seed=0, max_trades=3
    )
    assert {episode.index for episode in collector.cohort(12)} == set(range(12))
    assert list(collector.in_flight()) == []


def test_successive_cohorts_carry_on_from_where_the_last_one_stopped():
    """A PPO iteration asks for a cohort every time weights change, so the
    second one must deal the *next* indices, not replay the first's, and the
    environment must survive being re-armed (`LaneEnv.cohort`) rather than
    being rebuilt underneath its counters."""
    collector = Collector(
        RandomPolicy(random.Random(7)), lanes=3, fill=False, seed=2, action_cap=900
    )
    assert collector.games_started() == 0

    first = {episode.index for episode in collector.cohort(4)}
    assert first == set(range(4))
    assert collector.games_started() == 4
    assert list(collector.in_flight()) == []

    second = {episode.index for episode in collector.cohort(4)}
    assert second == set(range(4, 8))
    assert collector.games_started() == 8
    assert list(collector.in_flight()) == []
    # The counters are the environment's own and were never reset.
    assert collector.games == 8


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
        action_cap=900,
    )
    assert {e.index for e in collector.cohort(3)} == {1, 4, 7}
    assert {e.index for e in collector.cohort(3)} == {10, 13, 16}


def test_streaming_collection_returns_replacements_while_older_games_run_on():
    """The defect `cohort` removes, pinned so `stream` cannot quietly become it.

    `collect` refills a lane the moment its game ends, so a replacement dealt
    late can finish and enter the batch while a longer game dealt earlier is
    still being played. The PPO batch inherited both consequences for five
    runs: it selects for short games, and the unfinished lanes carry across the
    learner's weight sync into the next iteration's data.

    Asserted over several policy seeds rather than one, because whether a
    replacement *happens* to overtake depends on the spread of game lengths and
    not on the defect. Seed 1 alone carried it until repeated offers stopped
    being enumerated, which shortened every game and made that seed stop -- a
    red test that said nothing about `collect`.
    """
    overtaken = False
    for policy_seed in range(4):
        collector = Collector(
            RandomPolicy(random.Random(policy_seed)), lanes=8, seed=0, max_trades=3
        )

        returned = {episode.index for episode in collector.collect(16)}

        assert len(returned) == 16
        assert len(list(collector.in_flight())) == 8
        overtaken |= returned != set(range(16))

    assert overtaken, "no replacement outran an older game at any seed"


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
        action_cap=3000,
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


def test_owned_treats_an_empty_cast_as_learner_zero():
    from hexn.selfplay import owned

    episodes = Collector(
        RandomPolicy(random.Random(3)), lanes=2, seed=3, action_cap=3000
    ).collect(2)
    assert owned(episodes, 0)[0].trajectories == episodes[0].trajectories
    assert all(t == () for t in owned(episodes, 1)[0].trajectories)


def test_a_learner_id_with_no_seated_policy_is_an_error():
    with pytest.raises(ValueError):
        Collector(RandomPolicy(random.Random(4)), lanes=2, seed=4, learners=(0, 1))


def test_a_board_pair_shares_its_geometry_and_not_its_dice():
    from hexset.board.board import random_base_board

    collector = Collector(
        RandomPolicy(random.Random(0)), lanes=4, seed=9, pair_boards=True, deal=4
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
