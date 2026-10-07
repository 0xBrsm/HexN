# SPDX-License-Identifier: GPL-3.0-only
"""The per-VP reward: `vp_reward` per victory point a seat gains, paid at the
decision after it lands, with the margin head as the critic for the points
still to come."""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph, to_frame  # noqa: E402
from hexset.game import start  # noqa: E402
from hexn.model import HexNet, ModelConfig, pack, packing, unpack  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.ppo import PPOConfig, advantages, assemble, update  # noqa: E402
from hexn.ppo import __main__ as ppo_main  # noqa: E402
from hexn.rewards import win_loss  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402

C = 0.05


def a_policy(seed: int = 0) -> NetworkPolicy:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, 4, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space_for(game), packing(graph, 4))


@pytest.fixture(scope="module")
def played():
    policy = a_policy()
    policy.vp_reward = C
    episodes = Collector(policy, lanes=8, seed=0, action_cap=400, max_offers=0).collect(3)
    return policy, episodes


def test_every_transition_carries_its_seats_own_points(played):
    _, episodes = played
    for episode in episodes:
        for seat, trajectory in enumerate(episode.trajectories):
            held = [t.points for t in trajectory]
            assert held and all(p >= 0 for p in held)
            assert max(held) <= episode.outcome.points[seat] + 2  # an award can pass after the last decision
    assert any(t.points > 0 for e in episodes for tr in e.trajectories for t in tr)


def test_the_policy_records_the_win_value_plus_the_points_still_to_come(played):
    policy, episodes = played
    t = next(t for e in episodes for t in e.trajectories[0] if t.points > 0)
    prediction = policy.net(*unpack(policy.layout, pack(policy.layout, [t.observation])))
    expected = float(prediction.value[0, 0] + C * (10 * prediction.margin[0, 0] - t.points))
    assert t.value[0] == pytest.approx(expected, abs=1e-5)
    assert np.allclose(t.value[1:], prediction.value[0, 1:].detach().numpy(), atol=1e-5)


def test_each_point_is_paid_when_it_lands_and_the_margin_head_learns_final_points(played):
    policy, episodes = played
    config = PPOConfig(vp_reward=C, aux_margin_weight=1.0)
    batch = assemble(episodes, policy.layout, config)
    start = 0
    for episode in episodes:
        final = episode.outcome.points
        for seat, trajectory in enumerate(episode.trajectories):
            rows = slice(start, start + len(trajectory)); start += len(trajectory)
            assert np.allclose(batch.margin_target[rows].numpy(), to_frame([p / 10 for p in final], seat))
            held = [t.points for t in trajectory] + [final[seat]]
            steps = C * np.diff(np.asarray(held, dtype=np.float32))
            values = np.array([t.value[0] for t in trajectory], dtype=np.float32)
            terminal = np.float32(to_frame(win_loss(episode.outcome), seat)[0])
            assert np.allclose(batch.advantage[rows].numpy(), advantages(values, terminal, config.lam, steps), atol=1e-6)
    assert start == len(batch)


def test_intermediate_rewards_enter_gae_where_they_are_earned():
    values = np.array([0.2, 0.4, 0.5], dtype=np.float32)
    plain = advantages(values, 1.0, 1.0)
    paid = advantages(values, 1.0, 1.0, np.array([0.1, 0.0, 0.2], dtype=np.float32))
    # At lambda 1 each advantage is the return to go minus the estimate.
    assert np.allclose(paid - plain, [0.3, 0.2, 0.2])


def test_an_update_trains_the_margin_head_toward_final_points(played):
    policy, episodes = played
    student = a_policy(); student.net.load_state_dict(policy.net.state_dict())
    config = PPOConfig(vp_reward=C, aux_margin_weight=1.0, epochs=1, minibatch=256)
    batch = assemble(episodes, policy.layout, config)
    stats = update(student, torch.optim.Adam(student.net.parameters(), lr=1e-3), batch, config)
    assert np.isfinite(stats.margin_loss) and stats.margin_loss > 0


def test_a_vp_reward_needs_its_critic_and_a_non_negative_weight():
    with pytest.raises(ValueError, match="aux_margin_weight"):
        PPOConfig(vp_reward=C)
    with pytest.raises(ValueError, match="negative"):
        PPOConfig(vp_reward=-0.1, aux_margin_weight=1.0)
    with pytest.raises(ValueError, match="pair baseline"):
        PPOConfig(vp_reward=C, aux_margin_weight=1.0, pair_baseline=True)


def test_the_log_reads_the_learners_final_points(played):
    _, episodes = played
    mean = ppo_main.learner_final_vp(episodes)
    held = [e.outcome.points[s] for e in episodes for s in range(4)]
    assert mean == pytest.approx(sum(held) / len(held))
