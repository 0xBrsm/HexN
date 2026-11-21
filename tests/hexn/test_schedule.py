# SPDX-License-Identifier: GPL-3.0-only
"""The LR controller, pinned without torch — see `hexn.schedule` on why."""

from __future__ import annotations

import pytest

from hexn.schedule import AdaptiveLR, current_lr, linear_anneal, set_lr, warmup


class FakeOptimiser:
    """`param_groups` is the whole surface `hexn.schedule` touches."""

    def __init__(self, *rates: float) -> None:
        self.param_groups = [{"lr": rate, "params": []} for rate in rates]


def test_set_lr_moves_every_group():
    optimiser = FakeOptimiser(3e-4, 1e-3)
    set_lr(optimiser, 6e-4)
    assert [group["lr"] for group in optimiser.param_groups] == [6e-4, 6e-4]
    assert current_lr(optimiser) == pytest.approx(6e-4)


def test_hot_kl_lowers_the_rate():
    controller = AdaptiveLR(target_kl=0.02, band=2.0, factor=1.5)
    # 0.05 is above 0.02 * 2, so the update travelled too far.
    assert controller.next_lr(6e-4, 0.05) == pytest.approx(6e-4 / 1.5)


def test_the_dead_zone_is_a_fixed_point():
    controller = AdaptiveLR(target_kl=0.02, band=2.0)
    for kl in (0.01, 0.02, 0.04):
        assert controller.next_lr(6e-4, kl) == pytest.approx(6e-4)


def test_the_clamps_hold():
    controller = AdaptiveLR(target_kl=0.02, min_lr=1e-5, max_lr=1e-2)
    assert controller.next_lr(1e-2, 0.0) == pytest.approx(1e-2)
    assert controller.next_lr(1e-5, 1.0) == pytest.approx(1e-5)
    # And approaching a clamp never overshoots it.
    assert controller.next_lr(9e-3, 0.0) == pytest.approx(1e-2)
    assert controller.next_lr(1.2e-5, 1.0) == pytest.approx(1e-5)


def test_a_cold_gauge_walks_to_the_ceiling_and_stops():
    """A cold-KL regime: KL ignores the rate, so the controller saturates.

    The point of the test is the *stopping*. Without the clamp this loop is
    unbounded, which is the failure `AdaptiveLR`'s docstring is about.
    """
    controller = AdaptiveLR(target_kl=0.02, max_lr=1e-2)
    lr = 3e-4
    for _ in range(50):
        lr = controller.next_lr(lr, 0.006)
    assert lr == pytest.approx(1e-2)
    assert controller.deaf(lr, 0.006)


def test_deaf_is_false_in_band_even_at_a_clamp():
    """A converged controller also holds the rate still; that is not deafness."""
    controller = AdaptiveLR(target_kl=0.02, max_lr=1e-2)
    assert not controller.deaf(1e-2, 0.02)


def test_linear_anneal_endpoints_and_floor():
    assert linear_anneal(6e-4, 0, 100) == pytest.approx(6e-4)
    assert linear_anneal(6e-4, 50, 100) == pytest.approx(3e-4)
    assert linear_anneal(6e-4, 100, 100) == pytest.approx(0.0)
    assert linear_anneal(6e-4, 100, 100, floor=0.1) == pytest.approx(6e-5)
    # Past the end it holds the floor rather than going negative.
    assert linear_anneal(6e-4, 250, 100, floor=0.1) == pytest.approx(6e-5)
    # A degenerate total is the constant schedule, not a division by zero.
    assert linear_anneal(6e-4, 3, 0) == pytest.approx(6e-4)


def test_warmup_ramps_in_and_then_holds():
    rates = [warmup(6e-4, i, 4) for i in range(6)]
    assert rates == pytest.approx([1.5e-4, 3e-4, 4.5e-4, 6e-4, 6e-4, 6e-4])


def test_no_warmup_is_the_rate():
    assert [warmup(6e-4, i, 0) for i in range(3)] == [6e-4] * 3
    assert warmup(6e-4, 0, 1) == 6e-4


def test_the_schedule_warms_the_constant_and_the_linear_rate():
    from argparse import Namespace

    pytest.importorskip("torch", reason="hexn.ppo needs PyTorch, which runs on the training box only")
    from hexn.ppo.__main__ import scheduled_lr

    def args(schedule, length):
        return Namespace(learning_rate=6e-4, lr_schedule=schedule, iterations=10, lr_floor=0.5, lr_warmup=length)

    assert [scheduled_lr(args("constant", 0), i) for i in (0, 9)] == [6e-4, 6e-4]
    assert [scheduled_lr(args("constant", 2), i) for i in (0, 1, 2)] == pytest.approx([3e-4, 6e-4, 6e-4])
    # Without a warm-up, linear is the plain anneal; with one, the anneal ramps in.
    assert [scheduled_lr(args("linear", 0), i) for i in range(10)] == pytest.approx(
        [linear_anneal(6e-4, i, 10, 0.5) for i in range(10)])
    assert scheduled_lr(args("linear", 2), 0) == pytest.approx(linear_anneal(6e-4, 0, 10, 0.5) / 2)
    assert scheduled_lr(args("linear", 2), 5) == pytest.approx(linear_anneal(6e-4, 5, 10, 0.5))
