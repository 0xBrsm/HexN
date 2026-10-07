# SPDX-License-Identifier: GPL-3.0-only
"""The fast-win reward: a win in round r pays `fast_win[r - 1]`, through a
fast-win head the advantage reads while the win head trains as before."""
from __future__ import annotations

import dataclasses
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph, to_frame  # noqa: E402
from hexset.game import start  # noqa: E402
from hexn.model import HexNet, ModelConfig, graft_fast_win, packing, unpack  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.ppo import PPOConfig, assemble, fast_weight, update, win_round  # noqa: E402
from hexn.ppo import __main__ as ppo_main  # noqa: E402
from hexn.rewards import win_loss  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402

CURVE = (1.0, 1.0, 0.8, 0.5)


def a_net(*, fast_win: bool, players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1, fast_win=fast_win))
    return net, space_for(game), packing(graph, players)


@pytest.fixture(scope="module")
def played():
    """A fast-win policy recording its fast head, and a few decided games."""
    net, space, layout = a_net(fast_win=True)
    policy = NetworkPolicy(net, space, layout)
    policy.record_fast = True
    episodes = Collector(policy, lanes=8, seed=0, action_cap=400, max_offers=0).collect(3)
    return policy, [_decided(e) for e in episodes]


def _decided(episode, winner=None, turns=None):
    outcome = episode.outcome
    if winner is None:
        winner = outcome.winner
        if winner is None:
            points = outcome.points
            winner = max(range(len(points)), key=lambda seat: (points[seat], -seat))
    return dataclasses.replace(
        episode,
        outcome=dataclasses.replace(
            outcome, winner=winner, turns=outcome.turns if turns is None else turns
        ),
    )


def test_without_the_flag_the_net_keeps_its_exact_keys():
    plain, _, _ = a_net(fast_win=False)
    fast, _, _ = a_net(fast_win=True)
    assert not any(key.startswith("fast_win.") for key in plain.state_dict())
    extra = set(fast.state_dict()) - set(plain.state_dict())
    assert extra == {"fast_win.weight", "fast_win.bias"}


def test_a_grafted_head_ranks_seats_as_the_win_head_does():
    net, space, layout = a_net(fast_win=True)
    graft_fast_win(net)
    policy = NetworkPolicy(net, space, layout)
    episodes = Collector(policy, lanes=2, seed=3, action_cap=60, max_offers=0).collect(2)
    rows = [t.observation for e in episodes for t in e.trajectories[0]][:32]
    from hexn.model import pack

    prediction = net(*unpack(layout, pack(layout, rows)))
    seats = prediction.fast[:, :4]
    assert torch.allclose(prediction.fast.sum(-1), torch.ones(len(rows)), atol=1e-5)
    assert torch.allclose(seats / seats.sum(-1, keepdim=True), prediction.value, atol=1e-5)


def test_the_round_is_turns_over_seats_and_the_weight_follows_the_curve(played):
    _, episodes = played
    base = episodes[0].outcome
    at = lambda turns: dataclasses.replace(base, turns=turns, winner=0)  # noqa: E731
    assert [win_round(at(t), 4) for t in (1, 4, 5, 8, 9)] == [1, 1, 2, 2, 3]
    assert fast_weight(CURVE, at(9), 4) == 0.8
    assert fast_weight(CURVE, at(200), 4) == 0.5  # past the curve's end: its last entry
    assert fast_weight(CURVE, dataclasses.replace(base, winner=None), 4) == 0.0


def test_the_winner_is_paid_its_rounds_weight_and_the_win_head_keeps_its_one_hot(played):
    _, episodes = played
    games = [_decided(e, winner=1, turns=11) for e in episodes]  # round 3: weight 0.8
    config = PPOConfig(fast_win=CURVE)
    batch = assemble(games, played[0].layout, config)
    start = 0
    for episode in games:
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            rows = slice(start, start + len(trajectory))
            start += len(trajectory)
            one_hot = np.asarray(to_frame(win_loss(episode.outcome), seat), dtype=np.float32)
            assert np.array_equal(batch.value_target[rows].numpy(), np.tile(one_hot, (len(trajectory), 1)))
            fast = batch.fast_target[rows].numpy()
            assert np.allclose(fast[:, :4], 0.8 * one_hot) and np.allclose(fast[:, 4], 0.2)
            # The terminal the GAE chain ends on is the weighted payoff.
            values = np.array([t.value[0] for t in trajectory], dtype=np.float32)
            expected = ppo_advantages(values, 0.8 * one_hot[0], config.lam)
            assert np.allclose(batch.advantage[rows].numpy(), expected, atol=1e-6)
    assert start == len(batch)


def ppo_advantages(values, terminal, lam):
    from hexn.ppo import advantages

    return advantages(values, np.float32(terminal), lam)


def test_an_unfinished_game_puts_all_its_mass_on_not_won_in_time(played):
    _, episodes = played
    game = dataclasses.replace(
        episodes[0], outcome=dataclasses.replace(episodes[0].outcome, winner=None)
    )
    batch = assemble([game], played[0].layout, PPOConfig(fast_win=CURVE))
    assert np.allclose(batch.fast_target[:, :4].numpy(), 0.0)
    assert np.allclose(batch.fast_target[:, 4].numpy(), 1.0)


def test_the_policy_records_the_fast_heads_seat_values(played):
    policy, episodes = played
    transition = next(t for e in episodes for t in e.trajectories[0])
    from hexn.model import pack

    prediction = policy.net(*unpack(policy.layout, pack(policy.layout, [transition.observation])))
    assert np.allclose(transition.value, prediction.fast[0, :4].detach().numpy(), atol=1e-5)


def test_an_update_trains_the_fast_head_and_logs_its_loss(played):
    policy, episodes = played
    student_net, space, layout = a_net(fast_win=True)
    student_net.load_state_dict(policy.net.state_dict())
    student = NetworkPolicy(student_net, space, layout)
    config = PPOConfig(fast_win=CURVE, epochs=1, minibatch=256)
    batch = assemble(episodes, layout, config)
    before = student_net.fast_win.weight.detach().clone()
    stats = update(student, torch.optim.Adam(student_net.parameters(), lr=1e-3), batch, config)
    assert stats.fast_loss > 0.0 and np.isfinite(stats.fast_loss)
    assert not torch.equal(before, student_net.fast_win.weight)
    # And a batch without the target, or a config without the curve, is refused.
    with pytest.raises(ValueError, match="come together"):
        update(student, torch.optim.Adam(student_net.parameters()), batch, PPOConfig())


def test_weights_outside_zero_one_and_a_non_gae_critic_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PPOConfig(fast_win=(1.0, 1.2))
    with pytest.raises(ValueError, match="critic"):
        PPOConfig(fast_win=CURVE, critic="none")


def test_the_curve_file_reads_as_a_list_or_under_weights(tmp_path):
    listed, wrapped = tmp_path / "a.json", tmp_path / "b.json"
    listed.write_text(json.dumps(list(CURVE)))
    wrapped.write_text(json.dumps({"weights": list(CURVE), "note": "fit"}))
    assert ppo_main.fast_win_curve(str(listed)) == CURVE
    assert ppo_main.fast_win_curve(str(wrapped)) == CURVE
    assert ppo_main.fast_win_curve(None) == ()
