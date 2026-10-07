# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest
from _bots import needs

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import legal_actions, mask_of, space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.board.terrain import NUM_RESOURCES  # noqa: E402
from hexset.clients.netbot import NetworkBot  # noqa: E402
from hexset.encoding import encode, static_graph  # noqa: E402
from hexset.game import start, to_move  # noqa: E402
from hexset.trading import one_for_one  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing  # noqa: E402
from hexset.play import step_randomly  # noqa: E402
from hexn.policy import NetworkPolicy, masked_log_softmax  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402


def a_policy(players: int = 4, seed: int = 0, **kwargs):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    space = space_for(game)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space, graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space, packing(graph, players), **kwargs)


def a_played_game(seed: int, steps: int = 200):
    """A mid-game position with cards in every hand, for the trade seam."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    # public field
    state = game.state(0, hidden=False)
    for seat in range(4):
        for resource in range(NUM_RESOURCES):
            state.hands[seat][resource] = max(state.hands[seat][resource], 1)
    return game


class Counting:
    """Wraps the net so a test can say how many forwards a tick actually ran."""

    def __init__(self, net):
        self.net = net
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.net(*args)

    def parameters(self):
        return self.net.parameters()


def test_a_tick_runs_exactly_one_forward_however_many_lanes_there_are():
    policy = a_policy()
    counter = Counting(policy.net)
    policy.net = counter
    lanes = 12
    collector = Collector(policy, lanes=lanes, seed=1)

    collector.tick()
    collector.tick()

    # Anti-vacuity: one forward per tick is trivially true at one lane, and the
    # whole point of the collector's shape is that it stays true at many.
    assert lanes > 1
    assert collector.steps == 2 * lanes
    assert counter.calls == 2, f"{counter.calls} forwards for 2 ticks"


def test_every_action_it_picks_is_one_the_engine_offered():
    policy = a_policy(seed=2)
    # Trade-free: the claim is about the move a policy picks, and a gate
    # pricing candidates after every MAIN action is most of a tick's cost.
    collector = Collector(policy, lanes=8, seed=2, max_offers=0)

    kinds = set()
    for _ in range(60):
        requests = collector.requests()
        for request, choice in zip(requests, policy.act(requests)):
            assert choice.action in request.options, choice.action
            kinds.add(choice.action.type)
        collector.tick()

    # Anti-vacuity: a run that only ever saw ROLL and END_TURN would pass this
    # while exercising none of the index arithmetic it is meant to check.
    assert len(kinds) >= 5, f"only saw {sorted(k.name for k in kinds)}"


def _as_batch(policy, requests, choices):
    from hexn.model import pack

    buffer = pack(policy.layout, [r.observation for r in requests])
    mask = torch.from_numpy(np.stack([r.mask for r in requests]))
    chosen = torch.tensor(
        [policy.space.index(c.action) for c in choices], dtype=torch.int64
    )
    return buffer, mask, chosen


def test_evaluate_reproduces_exactly_what_act_recorded():
    # PPO's ratio is `exp(evaluate - act)`, so if these two disagree the very
    # first update is already taking ratios against the wrong distribution.
    policy = a_policy(seed=8)
    collector = Collector(policy, lanes=16, seed=8, max_offers=0)

    for _ in range(20):
        requests = collector.requests()
        choices = policy.act(requests)
        buffer, mask, chosen = _as_batch(policy, requests, choices)
        with torch.no_grad():
            evaluation = policy.evaluate(buffer, mask, chosen)

        recorded = torch.tensor([c.log_prob for c in choices])
        assert torch.allclose(evaluation.log_prob, recorded, atol=1e-4)
        values = torch.tensor([c.value for c in choices])
        assert torch.allclose(evaluation.value, values, atol=1e-4)
        collector.tick()


# --- the row seam: this policy as `hexset.clients.policy.Policy` ----------


def test_value_rows_are_un_rotated_into_board_seat_order():
    """The encoder rotates the seat it was asked about to slot 0 and the value
    head keeps that frame; `hexset.clients.policy.Policy` is stated in board
    seats. Getting the direction backwards type-checks, trains and plays
    nonsense, so the rotation is written out here rather than checked against
    the helper that performs it."""
    policy = a_policy(seed=60)
    game = a_played_game(61)

    for seat in range(4):
        framed = policy.values([encode(game, seat)])[0]
        vector = policy.value_rows([(game, seat)])[0]
        assert len(vector) == 4
        assert vector[seat] == pytest.approx(float(framed[0]))
        for slot, score in enumerate(framed.tolist()):
            assert vector[(seat + slot) % 4] == pytest.approx(score)


def test_act_rows_answers_exactly_what_the_batched_act_answers():
    """One net, two seams: the collector hands over rows it has already
    encoded, the engine's bot hands over live positions. They must not be two
    policies."""
    policy = a_policy(seed=62, greedy=True)
    game = a_played_game(63)
    seat = to_move(game)
    options = tuple(legal_actions(game))

    from hexn.selfplay import Request

    request = Request(
        lane=0,
        seat=seat,
        observation=encode(game, seat),
        mask=np.asarray(mask_of(policy.space, options), dtype=bool),
        options=options,
        game=game,
    )
    assert policy.act_rows([(game, seat, options)]) == [policy.act([request])[0].action]
    assert policy.act_rows([]) == []


def test_score_rows_priors_are_a_distribution_over_that_row_s_own_options():
    """What `hexset.mcts` wants a leaf scored as: a weight per option of that
    row, aligned with the row's options and summing to one, with no mass
    parked where a search cannot reach it."""
    policy = a_policy(seed=64)
    game = a_played_game(65)
    seat = to_move(game)
    options = tuple(legal_actions(game))

    (prior, value), = policy.score_rows([(game, seat, options)])

    assert len(prior) == len(options)
    assert all(weight > 0 for weight in prior)
    assert sum(prior) == pytest.approx(1.0, abs=1e-5)
    assert value == pytest.approx(policy.value_rows([(game, seat)])[0], abs=1e-6)


def test_the_gate_a_policy_seats_is_the_engine_s():
    """There is no torch trade gate any more. `trader` seats this policy
    behind `hexset.clients.netbot.NetworkBot`, which builds the post-trade
    position itself and asks only for a value -- one gate for the served
    `.onnx` file, the duelled checkpoint and the self-play lane."""
    policy = a_policy(seed=66)
    game = a_played_game(67)
    gate = policy.trader(game, 2, max_offers=5)

    assert isinstance(gate, NetworkBot)
    assert gate._seated is game
    assert gate.seat == 2
    assert gate.players == 4
    assert gate.max_offers == 5
    # The seat guard is the engine's: a gate wired to seat 2 must refuse to
    # answer for anybody else rather than price somebody else's hand.
    with pytest.raises(ValueError, match="seated at 2"):
        gate.accepts(game.state(3), one_for_one(0, 1), 1)
    assert isinstance(gate.accepts(game.state(2), one_for_one(0, 1), 1), bool)


@needs("heximax")
def test_a_named_trader_answers_the_seats_trades_instead_of_the_value_head():
    """`trader` seats that bot's own gate (`hexn.trade.trader_gate`): the
    network keeps every move and its value head prices nothing."""
    from hexset.arena import entrant_from_name, spawn

    policy = a_policy(seed=66)
    game = a_played_game(67)
    gate = policy.trader(game, 2, trader="heximax")
    board = game.state(2).state.board

    assert type(gate) is type(spawn(entrant_from_name("heximax"), board, random.Random(2)))
    assert gate.trade_offer_budget == 2
    assert isinstance(gate.accepts(game.state(2), one_for_one(0, 1), 1), bool)
    # Seeded by the seat: the same seat gets the same trader's draws.
    assert policy.trader(game, 2, trader="heximax")._trade_seed == gate._trade_seed


def test_asking_for_a_gate_leaves_the_position_exactly_as_it_was():
    # A gate prices candidates on copies; a leak into the live position would
    # corrupt the lane it was asked about rather than fail.
    policy = a_policy(seed=14)
    game = a_played_game(15)
    # public field
    before = [hand[:] for hand in game.state(0, hidden=False).hands]
    identity = id(game.state(0, hidden=False))
    ledger_identity = id(game.ledger)

    policy.trader(game, 2).accepts(game.state(2), one_for_one(0, 1), 1)

    assert [hand[:] for hand in game.state(0, hidden=False).hands] == before
    assert id(game.state(0, hidden=False)) == identity
    assert id(game.ledger) == ledger_identity


def test_a_tempered_request_records_the_tempered_log_prob():
    """`Request.temperature` divides the logits before the masked softmax and
    the recorded `log_prob` is that distribution's -- the exact behaviour
    log-probability PPO's ratio needs. Greedy sampling ignores it."""
    from dataclasses import replace

    from hexn.model import pack, unpack
    from hexn.selfplay import Request

    policy = a_policy(seed=3, generator=torch.Generator().manual_seed(1))
    rng = random.Random(3)
    game = start(random_base_board(rng), 4, rng)
    seat = to_move(game)
    options = tuple(legal_actions(game))
    request = Request(
        lane=0,
        seat=seat,
        observation=encode(game, seat),
        mask=np.asarray(mask_of(policy.space, options), dtype=bool),
        options=options,
        game=game,
    )
    hot = replace(request, temperature=2.0)
    choice = policy.act([hot])[0]

    packed = pack(policy.layout, [hot.observation])
    with torch.no_grad():
        prediction = policy.net(*unpack(policy.layout, packed))
    mask = torch.from_numpy(np.stack([hot.mask]))
    tempered = masked_log_softmax(prediction.logits / 2.0, mask)
    plain = masked_log_softmax(prediction.logits, mask)
    index = policy.space.index(choice.action)
    assert choice.log_prob == pytest.approx(float(tempered[0, index]), abs=1e-5)
    assert float(tempered[0, index]) != pytest.approx(float(plain[0, index]), abs=1e-6)

    greedy = a_policy(seed=3, greedy=True)
    assert greedy.act([hot])[0].action == greedy.act([request])[0].action
