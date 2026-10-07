# SPDX-License-Identifier: GPL-3.0-only
"""An update killed after any optimiser step resumes to the uninterrupted one.

Each test runs an update straight through, then the same update killed right
after step n (a raise where the next step would begin), then a fresh process's
worth of state -- new student, new optimiser, another global RNG -- opened on
the step file and run to the end. The two finished updates must agree to the
bit: weights, optimiser, logged gauges and the RNG the next iteration draws
from.
"""

from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import exit as distill  # noqa: E402
from hexn import ppo, steps  # noqa: E402
from hexn.ppo import attach_prior  # noqa: E402

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.game import start  # noqa: E402
from hexset.mcts import Search  # noqa: E402
from hexn.expert import SearchPolicy  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing  # noqa: E402
from hexn.netbot import LeafEvaluator  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402


def a_policy(seed: int = 0):
    """`test_ppo.a_policy`'s width-16 four-seat net."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, 4, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space_for(game), packing(graph, 4))


def a_student(policy):
    return NetworkPolicy(copy.deepcopy(policy.net), policy.space, policy.layout)


def some_episodes(policy, games, searched=False):
    """Trade-free capped games (`test_ppo.some_episodes`, `test_exit.
    searched_episodes`); a winner is not needed to compare two updates."""
    player = policy
    if searched:
        search = Search(LeafEvaluator(policy=policy), simulations=8, wave=4, rng=random.Random(0))
        player = SearchPolicy(search, rng=random.Random(0))
    return Collector(player, lanes=2, seed=0, action_cap=300, max_offers=0).collect(games)


class Crash(Exception):
    """A killed trainer, from inside the update."""


@pytest.fixture(scope="module")
def played():
    policy = a_policy()
    return policy, some_episodes(policy, 2)


@pytest.fixture(scope="module")
def searched():
    policy = a_policy()
    return policy, policy.space, policy.layout, some_episodes(policy, 1, searched=True)


def killed_after(n):
    """An optimiser whose step n+1 never happens."""

    def install(optimiser):
        real = optimiser.step
        taken = []

        def step(*args, **kwargs):
            if len(taken) == n:
                raise Crash
            taken.append(1)
            return real(*args, **kwargs)

        optimiser.step = step

    return install


def same_tensors(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tensors(x, y) for x, y in zip(a, b))
    return a == b


def three_runs(tmp_path, make, run, batch, n, every, generator_seed=None):
    """Straight through; killed after step n (`None`: after the update
    returned, before anything else was kept); resumed. Returns the first and
    the last (net, optimiser, stats, rng) and the resumed `Steps`."""

    def generator():
        if generator_seed is None:
            return None
        return torch.Generator().manual_seed(generator_seed)

    identity = steps.fingerprint(batch)

    def rng(drawn):
        return torch.get_rng_state() if drawn is None else drawn.get_state()

    torch.manual_seed(5)
    policy, optimiser = make()
    drawn = generator()
    stats = run(policy, optimiser, batch, drawn, None)
    straight = (policy.net.state_dict(), optimiser.state_dict(), stats, rng(drawn))

    torch.manual_seed(5)
    policy, optimiser = make()
    held = steps.Steps.open(tmp_path, 0, identity, every, policy, optimiser)
    if n is None:
        run(policy, optimiser, batch, generator(), held)
    else:
        killed_after(n)(optimiser)
        with pytest.raises(Crash):
            run(policy, optimiser, batch, generator(), held)

    torch.manual_seed(123)  # another process: its RNG is not the killed one's
    policy, optimiser = make()
    held = steps.Steps.open(tmp_path, 0, identity, every, policy, optimiser)
    if n is not None:
        assert held.resume["step"] == n - n % every
    drawn = generator()
    stats = run(policy, optimiser, batch, drawn, held)
    resumed = (policy.net.state_dict(), optimiser.state_dict(), stats, rng(drawn))
    return straight, resumed, held


def check(straight, resumed):
    assert same_tensors(straight[0], resumed[0]), "weights"
    assert same_tensors(straight[1], resumed[1]), "optimiser"
    assert straight[2] == resumed[2], "logged gauges"
    assert torch.equal(straight[3], resumed[3]), "the RNG the next iteration draws from"


@pytest.mark.parametrize(
    "n, every, micro, prior",
    [
        (1, 1, 0, False),  # after the first step
        (3, 1, 0, True),  # mid-epoch, with the tether's prior on the batch
        ("epoch", 1, 0, False),  # on the epoch boundary
        (8, 3, 32, False),  # every third step, micro-batched: resumes from step 6
        (None, 1, 0, False),  # after the last step: only the iteration checkpoint was missing
    ],
)
def test_a_ppo_update_killed_after_a_step_resumes_to_the_uninterrupted_update(
    tmp_path, played, n, every, micro, prior
):
    collecting, episodes = played
    config = ppo.PPOConfig(epochs=2, minibatch=64, micro_batch=micro, prior_kl=0.1 if prior else 0.0)
    batch = ppo.assemble(episodes, collecting.layout, config)
    if prior:
        batch = attach_prior(batch, collecting, collecting.layout)
    per_epoch = len(ppo._bounds(len(batch), 64))
    assert per_epoch >= 5, f"the fixture's batch takes {per_epoch} steps an epoch"
    if n == "epoch":
        n = per_epoch

    def make():
        student = a_student(collecting)
        return student, torch.optim.Adam(student.net.parameters(), lr=1e-3, eps=1e-5)

    def run(policy, optimiser, batch, generator, held):
        return ppo.update(policy, optimiser, batch, config, generator=generator, steps=held)

    straight, resumed, held = three_runs(tmp_path, make, run, batch, n, every)
    check(straight, resumed)
    # The resumed update keeps writing, and its last file is the finished update.
    last = torch.load(tmp_path / steps.STEP, weights_only=False)
    assert last["progress"]["step"] == 2 * per_epoch
    assert held.written == (0 if n is None else -(-(2 * per_epoch - held.resume["step"]) // every))


@pytest.mark.parametrize("before_the_break", [True, False])
def test_a_ppo_update_with_a_kl_break_resumes_to_the_same_break(tmp_path, played, before_the_break):
    """Killed inside the epoch the brake stops, the resumed update stops there
    too; killed after it, the break is read off the restored gauges."""
    collecting, episodes = played
    config = ppo.PPOConfig(epochs=3, minibatch=64, kl_break=1e-12, learning_rate=1e-2)
    batch = ppo.assemble(episodes, collecting.layout, config)
    per_epoch = len(ppo._bounds(len(batch), 64))

    def make():
        student = a_student(collecting)
        return student, torch.optim.Adam(student.net.parameters(), lr=1e-2, eps=1e-5)

    def run(policy, optimiser, batch, generator, held):
        return ppo.update(policy, optimiser, batch, config, generator=generator, steps=held)

    n = per_epoch - 1 if before_the_break else None
    straight, resumed, _ = three_runs(tmp_path, make, run, batch, n, 1, generator_seed=4)
    assert straight[2].epochs_taken == 1
    check(straight, resumed)


@pytest.mark.parametrize("pack, n", [(False, 2), (True, 3), (True, 5)])
def test_a_distillation_update_killed_after_a_step_resumes_to_the_uninterrupted_update(
    tmp_path, searched, pack, n
):
    policy, space, layout, episodes = searched
    config = distill.DistillConfig(
        epochs=2, minibatch=48, prior_kl=0.1, contested_only=pack, pack_contested=pack
    )
    batch = distill.assemble(episodes, space, layout, config)
    batch = attach_prior(batch, policy, layout)
    assert int(batch.policy_weight.sum()) > 1 or not pack

    def make():
        student = a_student(policy)
        return student, torch.optim.Adam(student.net.parameters(), lr=1e-3, eps=1e-5)

    def run(student, optimiser, batch, generator, held):
        return distill.update(student, optimiser, batch, config, generator=generator, steps=held)

    straight, resumed, _ = three_runs(tmp_path, make, run, batch, n, 1)
    check(straight, resumed)


def test_a_step_file_for_another_batch_or_iteration_is_removed_not_resumed(tmp_path, played):
    collecting, episodes = played
    config = ppo.PPOConfig(epochs=1, minibatch=64)
    batch = ppo.assemble(episodes, collecting.layout, config)
    identity = steps.fingerprint(batch)
    student = a_student(collecting)
    optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-3)
    killed_after(2)(optimiser)
    with pytest.raises(Crash):
        ppo.update(student, optimiser, batch, config, steps=steps.Steps(tmp_path / steps.STEP, 3, identity, 1))
    before = {k: v.clone() for k, v in student.net.state_dict().items()}

    for iteration, fingerprint in ((4, identity), (3, "another batch")):
        (tmp_path / "keep.pt").write_bytes((tmp_path / steps.STEP).read_bytes())
        fresh = a_student(collecting)
        opened = steps.Steps.open(
            tmp_path, iteration, fingerprint, 1, fresh, torch.optim.Adam(fresh.net.parameters())
        )
        assert opened.resume is None and not (tmp_path / steps.STEP).exists()
        assert not same_tensors(before, fresh.net.state_dict())
        (tmp_path / "keep.pt").rename(tmp_path / steps.STEP)


def test_the_fingerprint_tells_a_reordered_batch_from_the_same_one(played):
    collecting, episodes = played
    config = ppo.PPOConfig()
    one = ppo.assemble(episodes, collecting.layout, config)
    again = ppo.assemble(list(episodes), collecting.layout, config)
    swapped = ppo.assemble(list(reversed(episodes)), collecting.layout, config)
    assert steps.fingerprint(one) == steps.fingerprint(again)
    assert steps.fingerprint(one) != steps.fingerprint(swapped)
