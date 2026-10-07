# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.ddp import UpdateCrew, UpdateSpec  # noqa: E402
from hexn.ppo import PPOConfig, assemble, update  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402
from hexn.loop import build  # noqa: E402


def _decided(episode):
    if episode.outcome.winner is not None:
        return episode
    points = episode.outcome.points
    winner = max(range(len(points)), key=lambda seat: (points[seat], -seat))
    return dataclasses.replace(
        episode, outcome=dataclasses.replace(episode.outcome, winner=winner)
    )


ARGS = SimpleNamespace(
    seed=5, players=4, width=8, rounds=1, device="cpu", learning_rate=3e-4
)


@pytest.fixture(scope="module")
def batch():
    """One collected batch for the module; each test builds its own weights."""
    policy, _, _ = build(ARGS)
    # Trade-free and decided, for the same reasons `test_ppo.some_episodes`
    # gives: the sharded update's claim is about the arithmetic, and a game
    # the cap truncates has no winner for the win head to be seen learning.
    collector = Collector(
        policy, lanes=2, players=4, seed=5, action_cap=100, max_offers=0, deal=2
    )
    episodes = [_decided(e) for e in collector.drain()]
    return assemble(episodes, policy.layout, PPOConfig())


@pytest.fixture(scope="module")
def crew():
    """Three spawned workers, shared: spawning them is most of what either test
    costs, and a worker keeps nothing across updates beyond the shard each
    update hands it afresh."""
    crew = UpdateCrew([UpdateSpec(seed=5, players=4, width=8, rounds=1)] * 3)
    try:
        yield crew
    finally:
        crew.close()


def test_the_sharded_update_matches_the_single_device_update(batch, crew):
    """The whole module's claim: same schedule, same math, same result.

    Both paths start from identical weights and optimizer state and walk the
    same seeded minibatch order. Only float summation order differs — the crew
    sums shard gradients weighted by row count — so the bar is allclose, not
    equality. The tolerance is set by Adam, not by the gradients: dividing by
    sqrt(v) makes a near-zero-gradient parameter's step sign-sensitive, so a
    1e-8 reduction-order wobble can become a full lr-sized (3e-4) step on a
    few parameters within a couple of epochs. Measured divergence ~1e-4; a
    genuinely different update moves parameters by ~5e-3, an order above the
    bar, which is what keeps the test non-vacuous.
    """
    policy, optimiser, _ = build(ARGS)
    config = PPOConfig(minibatch=64, epochs=2)
    weights = copy.deepcopy(policy.net.state_dict())
    opt_state = copy.deepcopy(optimiser.state_dict())
    initial = (
        torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach().clone()
    )

    single = update(
        policy, optimiser, batch, config, generator=torch.Generator().manual_seed(3)
    )
    reference = torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach()

    policy.net.load_state_dict(weights)
    optimiser.load_state_dict(opt_state)
    sharded = crew.update(
        policy,
        optimiser,
        batch,
        config,
        generator=torch.Generator().manual_seed(3),
    )
    result = torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach()

    assert torch.allclose(reference, result, atol=3e-4, rtol=1e-3), (
        f"largest divergence {float((reference - result).abs().max())}"
    )
    assert sharded.positions == single.positions
    assert abs(sharded.policy_loss - single.policy_loss) < 1e-4
    assert abs(sharded.value_loss - single.value_loss) < 1e-4
    assert abs(sharded.entropy - single.entropy) < 1e-4
    assert abs(sharded.explained_variance - single.explained_variance) < 1e-3
    # Anti-vacuity: the update actually moved the weights, by far more than
    # the tolerance above, so "same update" and "no update" stay distinguishable.
    assert float((result - initial).abs().max()) > 1e-3


def test_a_worker_with_no_rows_in_a_minibatch_is_harmless(batch, crew):
    # Three contiguous shards and two- or three-row minibatches: every step
    # misses at least one shard entirely, which must contribute nothing rather
    # than hang or skew. The first thirty rows are plenty of steps for that.
    batch = type(batch)(
        **{
            field.name: None if (value := getattr(batch, field.name)) is None else value[:30]
            for field in dataclasses.fields(batch)
        }
    )
    policy, optimiser, _ = build(ARGS)
    config = PPOConfig(minibatch=2, epochs=1)
    stats = crew.update(
        policy,
        optimiser,
        batch,
        config,
        generator=torch.Generator().manual_seed(4),
    )
    assert stats.positions == len(batch)
    assert stats.value_loss > 0.0
