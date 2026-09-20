# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import dataclasses
import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import ActionType, space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexn.exit import (  # noqa: E402
    DistillConfig,
    assemble,
    contested,
    losses,
    project,
    stake,
    update,
)
from hexset.encoding import static_graph, to_frame  # noqa: E402
from hexn.expert import SearchPolicy  # noqa: E402
from hexset.game import start  # noqa: E402
from hexset.mcts import Search, visit_policy  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing  # noqa: E402
from hexn.netbot import LeafEvaluator  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.rewards import win_loss  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402


def a_setup(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    space = space_for(game)
    layout = packing(graph, players)
    torch.manual_seed(seed)
    net = HexNet(space, graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space, layout), space, layout


def searched_episodes(policy, space, games: int = 2, seed: int = 0):
    """Games played by a search over the network, so every transition has a target.

    Trade-free (`max_trades=0`), for the reason `test_ppo.some_episodes` gives
    for the plain-policy path: nothing here is about trading, and a width-16
    net's gate prices every coverable candidate after every MAIN action
    (`hexset.trading.trade_event`) -- unset, that made a searched game cost
    tens of seconds where it now costs a handful. `decided_searched_episodes`
    below already made this trade, this just extends it to every caller.
    """
    search = Search(
        LeafEvaluator(policy=policy),
        simulations=8,
        wave=4,
        rng=random.Random(seed),
    )
    expert = SearchPolicy(search, rng=random.Random(seed))
    return Collector(
        expert, lanes=2, seed=seed, action_cap=3000, max_trades=0
    ).collect(games)


def decided(episode):
    """`episode` with a winner: the one it has, else the seat on the most
    points, ties to the lower seat. Mirrors `test_ppo.decided`."""
    if episode.outcome.winner is not None:
        return episode
    points = episode.outcome.points
    winner = max(range(len(points)), key=lambda seat: (points[seat], -seat))
    return dataclasses.replace(
        episode, outcome=dataclasses.replace(episode.outcome, winner=winner)
    )


def decided_searched_episodes(policy, space, games: int = 3, seed: int = 0):
    """Searched, trade-free, decided games -- `test_ppo.some_episodes`'s own
    shape (trade-free, capped at 400 actions), but through a search so every
    transition still carries a `Target`. The win-head properties below need a
    game with an actual winner to be anything but vacuous, and search is slow
    enough that the cap keeps the corpus small."""
    search = Search(
        LeafEvaluator(policy=policy),
        simulations=8,
        wave=4,
        rng=random.Random(seed),
    )
    expert = SearchPolicy(search, rng=random.Random(seed))
    episodes = Collector(
        expert, lanes=2, seed=seed, action_cap=400, max_trades=0
    ).collect(games)
    return [decided(e) for e in episodes]


def a_batch(policy, space, layout, config=None, games: int = 2):
    episodes = searched_episodes(policy, space, games=games)
    return assemble(episodes, space, layout, config or DistillConfig())


# --- the projection -------------------------------------------------------


def test_the_projection_is_a_distribution_over_slots():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout)
    totals = batch.slot_target.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_every_visited_option_carries_its_share_onto_its_own_slot():
    # Contract 5 gives every option its own flat slot again, so the projection
    # is a plain sum onto `space.index(option)` -- which is exactly what has to
    # be checked now that there is no offer row to split anything into.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    checked = 0
    for episode in episodes:
        for trajectory in episode.trajectories:
            for transition in trajectory:
                target = transition.aux
                slots = project(target, space, 1.0)
                weights = visit_policy(target.visits, 1.0)
                for option, share in zip(target.options, weights):
                    if share > 0:
                        assert slots[space.index(option)] >= share - 1e-6
                        checked += 1
    assert checked, "no option was ever visited; untested"


def test_cooling_sharpens_the_target():
    """Temperature acts on options, before they land on slots."""
    policy, space, layout = a_setup()
    warm = a_batch(policy, space, layout, DistillConfig(temperature=1.0))
    cold = a_batch(policy, space, layout, DistillConfig(temperature=0.25))

    assert cold.slot_target.max(-1).values.mean() > warm.slot_target.max(-1).values.mean()


# --- the loss -------------------------------------------------------------


def test_the_projected_loss_equals_the_cross_entropy_over_options():
    """The identity the module rests on, checked against a direct sum.

    The policy scores a slot, not an option, and several options can share one
    slot -- so this is the check that projecting the target onto slots loses
    nothing: the slot cross-entropy must equal the plain cross-entropy over
    concrete options.
    """
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    config = DistillConfig()
    batch = assemble(episodes, space, layout, config)

    targets = [
        transition.aux
        for episode in episodes
        for trajectory in episode.trajectories
        for transition in trajectory
    ]

    rows = torch.arange(len(batch))
    slots, _, _, _ = policy.distributions(batch.buffer, batch.mask)
    projected = float(losses(slots, batch, rows).detach())
    slots = slots.detach()

    direct = 0.0
    for row, target in enumerate(targets):
        weights = visit_policy(target.visits, config.temperature)
        for option, share in zip(target.options, weights):
            direct -= share * float(slots[row, space.index(option)])
    direct /= len(targets)

    assert projected == pytest.approx(direct, rel=1e-4)


# --- the update -----------------------------------------------------------


def test_a_batch_refuses_transitions_that_carry_no_search_target():
    policy, space, layout = a_setup()
    episodes = Collector(policy, lanes=2, seed=0, action_cap=3000).collect(1)
    with pytest.raises(ValueError, match="search targets"):
        assemble(episodes, space, layout, DistillConfig())


def test_distilling_a_fixed_batch_moves_the_policy_toward_the_search():
    """The end-to-end property: repeated updates raise agreement and cut loss."""
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, games=3)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-3)
    config = DistillConfig(epochs=1, minibatch=256)

    first = update(policy, optimiser, batch, config)
    for _ in range(100):
        last = update(policy, optimiser, batch, config)

    assert last.policy_loss < first.policy_loss
    # 100 updates rather than 20: argmax agreement is noisy early on, and a
    # near-uniform init (policy heads at gain 0.01) makes the starting argmax
    # arbitrary rather than a real baseline, so a short run can land in a
    # trough by chance. The trend over the whole run is the property that
    # matters; too few updates risks measuring the noise instead.
    assert last.agreement > first.agreement


def test_a_bootstrapped_target_reads_the_estimate_that_many_decisions_later():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    horizon = 3
    batch = assemble(
        episodes, space, layout, DistillConfig(value_horizon=horizon)
    )

    row = 0
    bootstrapped = 0
    for episode in episodes:
        wins = win_loss(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            terminal = torch.tensor(to_frame(wins, seat), dtype=torch.float32)
            for index in range(len(trajectory)):
                ahead = index + horizon
                estimate = (
                    trajectory[ahead].value if ahead < len(trajectory) else ()
                )
                if estimate:
                    # Rotated, because the search stores board order and the
                    # target is in the seat's own frame.
                    expected = torch.tensor(
                        to_frame(estimate, seat), dtype=torch.float32
                    )
                    bootstrapped += 1
                else:
                    expected = terminal
                assert torch.allclose(batch.value_target[row], expected)
                row += 1
    assert row == len(batch)
    assert bootstrapped > 0, "no transition actually bootstrapped"


def test_a_bootstrapped_target_is_not_the_terminal_one():
    """Otherwise the test above would pass on a horizon that did nothing."""
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    terminal = assemble(episodes, space, layout, DistillConfig())
    near = assemble(episodes, space, layout, DistillConfig(value_horizon=3))
    assert not torch.allclose(terminal.value_target, near.value_target)


def test_the_value_head_is_trained_on_the_terminal_outcome():
    # The default, and what AlphaZero does: the one-hot eventual winner
    # (`hexn.rewards.win_loss`), exactly `hexn.ppo`'s own target.
    # `--value-horizon` bootstraps instead.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    batch = assemble(episodes, space, layout, DistillConfig())

    row = 0
    for episode in episodes:
        wins = win_loss(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            expected = torch.tensor(to_frame(wins, seat), dtype=torch.float32)
            for _ in trajectory:
                assert torch.allclose(batch.value_target[row], expected)
                row += 1
    assert row == len(batch)


def test_the_value_targets_are_one_hot_or_all_zero():
    # `hexn.rewards.win_loss`: exactly one 1.0 per row for a decided game,
    # all zero for one that was not -- never anything in between and never
    # summing to more than one. Mirrors `test_ppo.
    # test_the_win_targets_are_one_hot_or_all_zero`.
    policy, space, layout = a_setup(seed=3)
    episodes = decided_searched_episodes(policy, space, games=3, seed=3)
    batch = assemble(episodes, space, layout, DistillConfig())

    sums = batch.value_target.sum(dim=1)
    assert torch.all((sums == 0.0) | (sums == 1.0))
    assert torch.all((batch.value_target == 0.0) | (batch.value_target == 1.0))
    # Anti-vacuity: at least one row actually names a winner -- `decided`
    # guarantees every episode has one.
    assert (sums == 1.0).any()


def test_every_position_of_a_game_carries_the_winner():
    # Every one of a seat's decisions gets the same one-hot target, first
    # move to last -- there is no bootstrap at the default horizon. Mirrors
    # `test_ppo.test_every_position_in_a_game_carries_that_game_s_terminal_outcome`.
    policy, space, layout = a_setup(seed=2)
    episodes = decided_searched_episodes(policy, space, games=2, seed=2)
    batch = assemble(episodes[:1], space, layout, DistillConfig())

    episode = episodes[0]
    expected = {
        seat: to_frame(win_loss(episode.outcome), seat)
        for seat, trajectory in enumerate(episode.trajectories)
        if trajectory
    }
    assert len(expected) >= 2

    rows = 0
    for seat, trajectory in enumerate(episode.trajectories):
        for _ in trajectory:
            target = tuple(batch.value_target[rows].tolist())
            assert target == pytest.approx(expected[seat], abs=1e-6)
            rows += 1
    assert rows == len(batch)


def test_a_linear_heads_value_loss_is_exactly_cross_entropy_against_the_one_hot_winner():
    """`update`'s value pass, restated as a property -- `test_ppo.
    test_a_linear_heads_loss_is_exactly_cross_entropy_against_the_one_hot_winner`'s
    counterpart for distillation.

    A single minibatch spanning the whole batch, so the update's internal
    shuffle cannot change which rows land together; the loss is read off the
    parameters *before* `update` steps them, since that is when the forward
    pass computing it ran.
    """
    policy, space, layout = a_setup(seed=12)
    episodes = decided_searched_episodes(policy, space, games=2, seed=12)
    batch = assemble(episodes, space, layout, DistillConfig())

    with torch.no_grad():
        _, _, value_logits, quantiles = policy.distributions(batch.buffer, batch.mask)
    assert quantiles is None
    log_probs_win = torch.log_softmax(value_logits, dim=-1)
    expected = -(batch.value_target * log_probs_win).sum(-1).mean()

    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-3)
    config = DistillConfig(epochs=1, minibatch=len(batch))
    stats = update(policy, optimiser, batch, config)

    assert stats.value_loss == pytest.approx(float(expected), rel=1e-5)
    # Cross-entropy and squared error are different expressions in general.
    assert stats.value_mse != stats.value_loss


def test_an_update_moves_the_value_head_toward_the_outcome_it_was_shown():
    """Mirrors `test_ppo.
    test_an_update_moves_the_value_head_towards_the_outcome_it_was_shown`."""
    policy, space, layout = a_setup(seed=6)
    episodes = decided_searched_episodes(policy, space, games=3, seed=6)
    batch = assemble(episodes, space, layout, DistillConfig())
    config = DistillConfig(epochs=1, minibatch=len(batch))
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    first = update(policy, optimiser, batch, config)
    for _ in range(20):
        last = update(policy, optimiser, batch, config)

    assert last.value_loss < first.value_loss
    assert last.value_mse < first.value_mse
    # Anti-vacuity: a value loss that started at zero would satisfy nothing.
    assert first.value_loss > 1e-4


def a_target(visits, prior=None, values=None):
    """A `Target` whose options are placeholders: `contested` reads only their
    count, so a real action space is not needed to pin the filter."""
    import numpy as np

    from hexn.expert import Target

    return Target(
        options=tuple(range(len(visits))),
        visits=np.asarray(visits, dtype=np.float64),
        prior=None if prior is None else np.asarray(prior, dtype=np.float64),
        values=None if values is None else np.asarray(values, dtype=np.float64),
    )


def test_a_search_that_agrees_with_the_prior_is_not_contested():
    assert not contested(a_target([70.0, 20.0, 10.0], [0.7, 0.2, 0.1]))


def test_a_search_that_overrules_the_prior_is_contested():
    assert contested(a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1]))


def test_the_stake_is_the_gap_between_what_the_search_picked_and_what_it_overruled():
    target = a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1], [0.10, 0.16, -0.30])
    assert stake(target) == pytest.approx(0.06)


def test_the_stake_scale_weights_a_contested_row_by_what_it_is_worth():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    filtered = assemble(
        episodes, space, layout, DistillConfig(contested_only=True)
    )
    staked = assemble(
        episodes,
        space,
        layout,
        DistillConfig(contested_only=True, stake_scale=0.05),
    )
    assert (staked.policy_weight <= filtered.policy_weight + 1e-6).all()
    assert (staked.policy_weight >= 0.0).all()
    # The anchor holds whatever the policy term let go, and a part-staked row is
    # part-held: the two weights have to keep summing to one where a prior was
    # recorded, or the row is trained on less than all of itself.
    both = staked.policy_weight + staked.anchor_weight
    recorded = filtered.anchor_weight + filtered.policy_weight > 0.0
    assert torch.allclose(both[recorded], torch.ones_like(both[recorded]), atol=1e-6)


def test_the_hard_target_puts_all_of_a_row_on_one_option():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, config=DistillConfig(hard_target=True))
    totals = batch.slot_target.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    # A one-hot over *options*, and under contract 5 every option has its own
    # slot, so the row's max is the whole of it.
    assert torch.allclose(
        batch.slot_target.max(-1).values,
        torch.ones_like(totals),
        atol=1e-5,
    )


def test_the_searched_corpus_records_a_prior_beside_its_visits():
    # Without this the filter has nothing to compare against, and it cannot be
    # recovered later: by training time the policy has moved.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    targets = [
        t.aux
        for e in episodes
        for traj in e.trajectories
        for t in traj
        if t.aux is not None
    ]
    assert targets
    # A forced position is never expanded, so it has no prior -- and it cannot
    # be contested either, which is why `contested` treats a missing prior as
    # agreement rather than as an error.
    searched = [t for t in targets if len(t.options) > 1]
    assert searched
    assert all(t.prior is not None for t in searched)
    assert all(len(t.prior) == len(t.options) for t in searched)
    assert all(t.prior is None for t in targets if len(t.options) == 1)


def test_filtering_zeroes_the_policy_weight_where_the_search_agreed():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    plain = assemble(episodes, space, layout, DistillConfig())
    filtered = assemble(
        episodes, space, layout, DistillConfig(contested_only=True)
    )
    assert torch.all(plain.policy_weight == 1.0)
    # A network's argmax already agrees with the search on most rows, so
    # most of the batch must be switched off here -- and the batch must keep
    # its length regardless, since the value head still has to see it all.
    assert len(filtered) == len(plain)
    assert filtered.policy_weight.sum() < plain.policy_weight.sum()


# --- the anchor -----------------------------------------------------------
#
# `contested_only` filters the policy loss but cannot filter its effect: the
# trunk is shared, the value loss is unweighted, and a network has no
# per-position parameters. Left unchecked, a shared trunk drifts anyway on
# every row the policy loss switched off -- degrading agreement with the
# search on exactly the positions the filter was meant to leave alone, and
# costing the checkpoint strength against its own unfiltered parent. The
# anchor is the restoring force that holds those rows in place.


def test_the_anchor_weight_is_the_complement_of_the_policy_weight():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    filtered = assemble(episodes, space, layout, DistillConfig(contested_only=True))
    # Only where a prior was recorded. A forced position has none, and both
    # weights are zero there -- it is neither trained nor anchored.
    anchorable = filtered.anchor_slots.sum(-1) > 0
    assert bool(anchorable.any())
    assert torch.allclose(
        filtered.anchor_weight[anchorable],
        1.0 - filtered.policy_weight[anchorable],
    )


def test_the_anchor_target_is_a_distribution_over_the_same_slots():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, DistillConfig(contested_only=True))
    anchored = batch.anchor_weight > 0
    assert bool(anchored.any())
    totals = batch.anchor_slots[anchored].sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_the_anchor_holds_the_settled_rows_the_filter_let_go():
    """The whole point: rows the policy loss zeroed must not drift freely."""
    import copy

    policy, space, layout = a_setup(seed=3)
    batch = a_batch(policy, space, layout, DistillConfig(contested_only=True), games=4)
    settled = batch.anchor_weight > 0
    assert bool(settled.any())

    def drift(anchor: float) -> float:
        student = NetworkPolicy(copy.deepcopy(policy.net), space, layout)
        config = DistillConfig(contested_only=True, anchor=anchor, epochs=2)
        update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=3e-3),
            batch,
            config,
            generator=torch.Generator().manual_seed(0),
        )
        with torch.no_grad():
            slots, _, _, _ = student.distributions(batch.buffer, batch.mask)
            # How far the settled rows ended up from the prior they were
            # collected under, which is exactly what the anchor penalises.
            per_row = -(batch.anchor_slots * slots).sum(-1)
            return float(per_row[settled].mean())

    assert drift(anchor=1.0) < drift(anchor=0.0)


# --- the buffer and the packing -------------------------------------------
#
# Collection is 92% of wall clock -- 707 s against 66 s of update on the live
# run -- and with `--contested-only` about 3% of what it buys carries policy
# gradient. Both of these attack that: reuse the corpus, and spend the policy
# term's steps on rows that are actually in it.


def test_concatenating_batches_keeps_every_row():
    policy, space, layout = a_setup()
    from hexn.exit import Batch

    first = a_batch(policy, space, layout, games=2)
    second = a_batch(policy, space, layout, games=2)
    joined = Batch.concat([first, second])
    assert len(joined) == len(first) + len(second)
    for name in Batch.FIELDS:
        assert torch.equal(
            getattr(joined, name)[: len(first)], getattr(first, name)
        )
        assert torch.equal(
            getattr(joined, name)[len(first) :], getattr(second, name)
        )


def test_concatenating_one_batch_is_that_batch():
    policy, space, layout = a_setup()
    from hexn.exit import Batch

    only = a_batch(policy, space, layout, games=2)
    assert Batch.concat([only]) is only


def test_refreshing_against_the_collecting_policy_reproduces_the_filter():
    """The recorded prior came from this policy, so the two must agree.

    Asserted where the max is *strict*. Tied options have no argmax, only a
    tie-break, and the two sides break ties in unrelated orders: `contested`
    takes `np.argmax` over the option tuple, `refresh` takes `argmax` over
    slots. Swept against the fixture's simulation count the mismatch is the tie
    rate and nothing else. The invariant is real; asserting past the ties would
    only assert that two arbitrary orderings coincide on one seed.

    Contract 4 needed a second restriction here -- to rows the search never
    proposed from -- because the trade slot was a sum over every offer on the
    row and could top the slot ranking while the argmax *option* was not a
    trade at all. With the trade actions gone the option-to-slot map is
    injective everywhere and that restriction has nothing left to exclude.
    """
    from hexn.exit import refresh

    policy, space, layout = a_setup(seed=5)
    config = DistillConfig(contested_only=True)
    batch = a_batch(policy, space, layout, config, games=3)
    refreshed = refresh(policy, batch, config)

    top = batch.slot_target.max(-1, keepdim=True).values
    decided = (batch.slot_target >= top).sum(-1) == 1

    assert int(decided.sum()) > 0
    assert torch.equal(batch.policy_weight[decided], refreshed.policy_weight[decided])


def test_packing_reports_the_policy_loss_over_contested_rows_only():
    """Unpacked, a minibatch holding no contested row reports a zero loss.

    Those zeros are averaged into `policy_loss`, so the reported number is
    diluted by the filter's selectivity rather than describing the rows the
    loss actually trained on. Packed, every reported step has contested rows in
    it.
    """
    import copy

    policy, space, layout = a_setup(seed=11)
    batch = a_batch(
        policy, space, layout, DistillConfig(contested_only=True), games=3
    )
    assert float(batch.policy_weight.sum()) > 0

    # The dilution needs a minibatch with nothing contested in it, and this
    # corpus does not supply one: `losses` divides by the weight rather than by
    # the row count, so a minibatch holding any contested row already reports
    # the mean over those rows, and at this batch's natural density every one of
    # its seven minibatches holds hundreds. Thinned to three contested rows, six
    # of the seven report the zero this test is about — and the assertion stops
    # being a comparison of two numbers that agree to a tenth of a percent.
    keep = torch.nonzero(batch.policy_weight > 0.0).squeeze(-1)[:3]
    thinned = torch.zeros_like(batch.policy_weight)
    thinned[keep] = batch.policy_weight[keep]
    batch = batch.__class__(
        **{
            name: thinned if name == "policy_weight" else getattr(batch, name)
            for name in batch.FIELDS
        }
    )

    def run(pack: bool) -> float:
        student = NetworkPolicy(copy.deepcopy(policy.net), space, layout)
        stats = update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=1e-4),
            batch,
            DistillConfig(
                contested_only=True, hard_target=True, epochs=1, pack_contested=pack
            ),
            generator=torch.Generator().manual_seed(0),
        )
        return stats.policy_loss

    assert run(pack=True) > run(pack=False)
