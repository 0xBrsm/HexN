# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.expert import SearchPolicy, Target  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing  # noqa: E402
from hexset.clients.netbot import LeafEvaluator, NetworkBot  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.selfplay import Collector, Request  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.board.terrain import NUM_RESOURCES  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.game import start  # noqa: E402
from hexset.mcts import Search  # noqa: E402
from hexset.play import step_randomly  # noqa: E402
from hexset.trading import one_for_one  # noqa: E402

from _mcts_fixtures import Stub, a_game  # noqa: E402


def a_policy(stub=None, *, simulations=16, wave=4, temperature=1.0, seed=0):
    search = Search(
        stub or Stub(), simulations=simulations, wave=wave, rng=random.Random(seed)
    )
    return SearchPolicy(search, temperature=temperature, rng=random.Random(seed))


def a_collector(policy, **kwargs):
    return Collector(policy, lanes=1, seed=7, players=4, **kwargs)


def a_network_policy(seed: int = 0) -> NetworkPolicy:
    """A real, tiny `NetworkPolicy` -- what a `LeafEvaluator` actually wraps."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    space = space_for(game)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space, graph, 4, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space, packing(graph, 4))


def a_search_over_network(policy: NetworkPolicy, **kwargs) -> Search:
    return Search(LeafEvaluator(policy=policy), **kwargs)


def a_played_game(seed: int, steps: int = 50):
    """A mid-game position with cards in every hand, for the trade seam."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    state = game.state(0, hidden=False)
    for seat in range(4):
        for resource in range(NUM_RESOURCES):
            state.hands[seat][resource] = max(state.hands[seat][resource], 1)
    return game


class Counting:
    """Wraps the net so a test can say how many forwards a call actually ran."""

    def __init__(self, net):
        self.net = net
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.net(*args)

    def parameters(self):
        return self.net.parameters()


def a_request(policy: SearchPolicy, game, options=None) -> Request:
    """What a collector would ask, without running one."""
    return Request(
        lane=0,
        seat=0,
        observation=None,
        mask=np.zeros(0, dtype=bool),
        options=policy.search._options(game) if options is None else options,
        game=game,
    )


def test_a_search_policy_answers_one_choice_per_request():
    collector = a_collector(a_policy())
    collector.tick()
    assert collector.steps == 1


def test_searches_from_several_lanes_share_one_root_evaluation():
    stub = Stub()
    collector = Collector(a_policy(stub), lanes=4, seed=7, players=4)
    collector.tick()
    assert collector.steps == 4
    assert len(stub.waves[0]) == 4


def test_the_visit_counts_ride_along_on_the_transition():
    stub = Stub()
    collector = a_collector(a_policy(stub, simulations=24), action_cap=12)
    episode = collector.collect(1)[0]
    filed = episode.stream()
    assert filed
    for transition in filed:
        target = transition.aux
        assert isinstance(target, Target)
        assert len(target.options) == len(target.visits)
        # A forced move is not searched, so its counts are a single 1 rather
        # than the budget; anything with a real choice spends the whole budget.
        assert target.visits.sum() in (1.0, 24.0)


def test_the_action_played_is_one_the_search_gave_counts_to():
    stub = Stub()
    collector = a_collector(a_policy(stub, simulations=16), action_cap=12)
    episode = collector.collect(1)[0]
    assert episode.trajectories
    for seat in episode.trajectories:
        for transition in seat:
            assert transition.action in transition.aux.options
            assert transition.mask[transition.index]


def test_the_recorded_value_is_what_the_search_concluded_not_what_it_started_from():
    class First(Stub):
        """Only the very first leaf — the root — is worth anything."""

        def evaluate(self, leaves):
            out = super().evaluate(leaves)
            if len(self.waves) == 1:
                return [(prior, (1.0, 0.0, 0.0, 0.0)) for prior, _ in out]
            return out

    policy = a_policy(First(), simulations=16)
    game = a_game()
    request = a_request(policy, game)
    choice = policy.act([request])[0]
    # `root.value` is (1, 0, 0, 0) and every descendant is zero, so reporting
    # the root's own estimate rather than the backed-up mean is visible here.
    assert choice.value == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_every_option_carries_the_mean_the_search_ranked_it_on():
    policy = a_policy(Stub(value=(0.4, -0.1, -0.1, -0.2)), simulations=32)
    game = a_game()
    root, options, visits = policy.search.run(game)
    target = policy._choice(a_request(policy, game, options), (root, options, visits)).aux
    assert target.values is not None
    # The arithmetic `_select` does: an unvisited edge scores 0, a visited one
    # its stance-ranked total over its count.
    expected = np.where(visits > 0, root.ranked / np.maximum(visits, 1.0), 0.0)
    assert np.allclose(target.values, expected)
    assert target.values[int(np.argmax(visits))] != 0.0


def test_a_forced_move_reports_no_value_estimate():
    class OneWay(Search):
        def _options(self, game):
            return super()._options(game)[:1]

    stub = Stub()
    policy = SearchPolicy(
        OneWay(stub, simulations=16, rng=random.Random(0)), rng=random.Random(0)
    )
    game = a_game()
    choice = policy.act([a_request(policy, game)])[0]
    assert choice.value == ()
    assert choice.log_prob == 0.0
    assert stub.waves == []


def test_a_search_that_roots_on_different_options_is_refused():
    # The collector's mask is built from its own enumeration, so a search that
    # disagrees would file actions the mask calls illegal.
    policy = a_policy()
    game = a_game()
    short = policy.search._options(game)[:-1]
    with pytest.raises(ValueError, match="where the collector offered"):
        policy.act([a_request(policy, game, options=short)])


def test_a_policy_without_the_position_is_refused():
    request = Request(
        lane=0, seat=0, observation=None, mask=np.zeros(0, dtype=bool), options=()
    )
    with pytest.raises(ValueError, match="needs a position"):
        a_policy().act([request])


def test_temperature_zero_plays_the_most_visited_action():
    policy = a_policy(Stub(favour=0), simulations=32, temperature=0.0)
    game = a_game()
    choice = policy.act([a_request(policy, game)])[0]
    target = choice.aux
    assert choice.action == target.options[int(np.argmax(target.visits))]
    assert choice.log_prob == pytest.approx(0.0)


def test_temperature_one_does_not_always_play_the_most_visited_action():
    # The corpus is the point: four searches that all take the argmax replay
    # one game, and the target then only ever covers the best line.
    game = a_game()
    played = set()
    for seed in range(12):
        policy = a_policy(simulations=32, seed=seed)
        played.add(policy.act([a_request(policy, game)])[0].action)
    assert len(played) > 1


def test_the_collector_hands_the_lane_state_to_the_policy():
    # Not an accident of the search: a policy that ignores the encoding has to
    # be able to reach the position, or nothing here works.
    seen: list = []

    class Peek:
        def act(self, requests):
            seen.extend(request.game for request in requests)
            return [
                type("C", (), {"action": r.options[0], "log_prob": 0.0, "value": (), "aux": None})()
                for r in requests
            ]

    collector = a_collector(Peek())
    collector.tick()
    assert seen and seen[0] is next(collector.in_flight())


def test_a_searched_lanes_gate_delegates_to_the_networks_own_trader():
    # The collector seats whatever `policy.trader` returns as the seat's gate
    # (`hexset.gym.lanes.LaneEnv`'s `gates`); before this,
    # `SearchPolicy` had no `trader` at all and every searched seat sat out
    # trading with a silent `None` gate.
    policy = a_network_policy(seed=40)
    search = a_search_over_network(policy, simulations=4, wave=1)
    collector = Collector(
        SearchPolicy(search, rng=random.Random(41)),
        lanes=1,
        seed=7,
        players=4,
        max_trades=2,
    )
    game = next(collector.in_flight())
    for seat, gate in enumerate(game.gates):
        assert gate is not None
        assert isinstance(gate, NetworkBot)
        assert gate._seated is game
        assert gate.seat == seat
        assert gate.max_trades == 2


def test_a_searched_seats_off_switch_refuses_everything_without_a_forward():
    policy = a_network_policy(seed=42)
    counter = Counting(policy.net)
    policy.net = counter
    game = a_played_game(43)
    search = a_search_over_network(policy, simulations=4, wave=1)
    trader = SearchPolicy(search).trader(game, 1, 0)
    view = game.state(1)

    assert trader.accepts(view, one_for_one(0, 1), 2) is False
    assert trader.gains_many(view, [one_for_one(0, 1)], [2]) == [-1.0]
    assert counter.calls == 0
