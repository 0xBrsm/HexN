# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import ppo  # noqa: E402
from hexn.ppo import __main__ as ppo_main  # noqa: E402
from hexset.actions import ActionType, space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph, to_frame  # noqa: E402
from hexset.game import start  # noqa: E402
from hexn.model import HexNet, ModelConfig, packing, unpack  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.ppo import (  # noqa: E402
    PPOConfig,
    advantages,
    assemble,
    update,
)
from hexn.rewards import relative_points, win_loss  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402


def a_policy(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space_for(game), packing(graph, players))


def some_episodes(policy, games: int = 4, seed: int = 0, lanes: int = 8):
    """Real transitions, trade-free and decided.

    Nothing here is about trading, and a width-16 net's gate prices every
    coverable candidate after every MAIN action (`hexset.trading.trade_
    event`) -- at a few dozen ms per event that made one capped game cost
    minutes. A game the cap truncates has no winner and `win_loss` gives every
    seat zero, which is a legitimate target but not one the update can be
    seen learning from, so `decided` names one: `win_loss` reads only
    `outcome.winner`.
    """
    episodes = Collector(
        policy, lanes=lanes, seed=seed, action_cap=400, max_trades=0
    ).collect(games)
    return [decided(e) for e in episodes]


def decided(episode):
    """`episode` with a winner: the one it has, else the seat on the most
    points, ties to the lower seat."""
    if episode.outcome.winner is not None:
        return episode
    points = episode.outcome.points
    winner = max(range(len(points)), key=lambda seat: (points[seat], -seat))
    return dataclasses.replace(
        episode, outcome=dataclasses.replace(episode.outcome, winner=winner)
    )


def test_the_discount_is_one_and_is_not_something_a_run_can_change():
    # The reward is zero-sum, so about half of all terminal values are negative
    # and a discount below 1 makes a late loss cheaper than an early one. That
    # pays a losing policy to stall. It is a constant, not a knob.
    assert ppo.GAMMA == 1.0

    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    assert "gamma" not in fields
    # Anti-vacuity: an empty or renamed config would satisfy the line above for
    # the wrong reason.
    assert {"lam", "clip", "epochs"} <= fields

    parser_flags = ppo_main.main.__doc__ or ""
    assert "--gamma" not in parser_flags
    with pytest.raises(SystemExit):
        ppo_main.main(["--gamma", "0.99", "--iterations", "0"])


def test_gae_at_lambda_one_is_just_the_outcome_minus_the_estimate():
    # With gamma 1 and a terminal-only reward, GAE(1) telescopes to the Monte
    # Carlo advantage. Anything else means the bootstrap terms are wrong.
    values = np.array([0.1, -0.2, 0.4, 0.0], dtype=np.float32)
    out = advantages(values, terminal=0.6, lam=1.0)
    assert out == pytest.approx(0.6 - values, abs=1e-6)


def test_gae_below_one_leans_on_the_value_head_instead_of_the_outcome():
    values = np.array([0.1, -0.2, 0.4, 0.0], dtype=np.float32)
    monte_carlo = advantages(values, terminal=0.6, lam=1.0)
    shrunk = advantages(values, terminal=0.6, lam=0.5)

    # The last step is identical either way: there is no future to discount.
    assert shrunk[-1] == pytest.approx(monte_carlo[-1], abs=1e-6)
    # Earlier steps are pulled towards the value head, so they must differ.
    assert not np.allclose(shrunk[:-1], monte_carlo[:-1])


def test_a_single_step_trajectory_is_the_whole_payoff():
    assert advantages(np.array([0.25], dtype=np.float32), 1.0, 0.95) == pytest.approx(
        [0.75], abs=1e-6
    )


def test_assemble_keeps_every_transition_of_every_episode():
    policy = a_policy(seed=1)
    episodes = some_episodes(policy, games=3, seed=1)
    batch = assemble(episodes, policy.layout, PPOConfig())

    assert len(batch) == sum(len(e) for e in episodes)
    # Anti-vacuity: one episode with one seat acting would not exercise the
    # per-seat demultiplexing this walks over.
    assert len(episodes) >= 3
    assert all(sum(1 for s in e.trajectories if s) >= 2 for e in episodes)


def test_every_position_in_a_game_carries_that_game_s_terminal_outcome():
    # The win head is trained on the one-hot eventual winner, never on a
    # bootstrap, so a seat's target is the same vector at its first decision
    # and its last. The auxiliary margin head's target is `relative_points`,
    # rotated the same way, checked alongside.
    policy = a_policy(seed=2)
    episodes = some_episodes(policy, games=2, seed=2)
    batch = assemble(episodes[:1], policy.layout, PPOConfig())

    episode = episodes[0]

    expected_win = {
        seat: to_frame(win_loss(episode.outcome), seat)
        for seat, trajectory in enumerate(episode.trajectories)
        if trajectory
    }
    expected_margin = {
        seat: to_frame(relative_points(episode.outcome.points), seat)
        for seat, trajectory in enumerate(episode.trajectories)
        if trajectory
    }
    assert len(expected_win) >= 2

    rows = 0
    for seat, trajectory in enumerate(episode.trajectories):
        for _ in trajectory:
            target = tuple(batch.value_target[rows].tolist())
            assert target == pytest.approx(expected_win[seat], abs=1e-6)
            margin_target = tuple(batch.margin_target[rows].tolist())
            assert margin_target == pytest.approx(expected_margin[seat], abs=1e-6)
            rows += 1
    assert rows == len(batch)


def test_the_win_targets_are_one_hot_or_all_zero():
    # `hexn.rewards.win_loss`: exactly one 1.0 per row for a decided game,
    # all zero for one that was not -- never anything in between and never
    # summing to more than one.
    policy = a_policy(seed=3)
    episodes = some_episodes(policy, games=4, seed=3)
    batch = assemble(episodes, policy.layout, PPOConfig())

    sums = batch.value_target.sum(dim=1)
    assert torch.all((sums == 0.0) | (sums == 1.0))
    assert torch.all((batch.value_target == 0.0) | (batch.value_target == 1.0))
    # Anti-vacuity: at least one row actually names a winner.
    assert (sums == 1.0).any()


def test_the_margin_targets_of_one_game_sum_to_zero_across_the_table():
    policy = a_policy(seed=3)
    episodes = some_episodes(policy, games=2, seed=3)
    batch = assemble(episodes, policy.layout, PPOConfig())
    # Each row is one seat's view of `relative_points`, zero-sum by
    # construction, so every row sums to zero on its own.
    assert batch.margin_target.sum(dim=1).abs().max() < 1e-5


def test_a_batch_carries_one_flat_index_per_transition_and_nothing_else():
    # Contract 5 leaves the update with a single categorical: no pair mask, no
    # offer slot, no joint log-prob. If any of that came back, `Batch` would
    # have to carry it and `minibatch_terms` would have to be handed it.
    policy = a_policy(seed=4)
    episodes = some_episodes(policy, games=4, seed=4, lanes=16)
    batch = assemble(episodes, policy.layout, PPOConfig())

    transitions = [t for e in episodes for seat in e.trajectories for t in seat]
    assert len(batch) == len(transitions)
    assert not hasattr(batch, "pair")
    assert not hasattr(batch, "offer")
    assert batch.chosen.tolist() == [t.index for t in transitions]


def test_an_update_returns_a_ratio_of_one_before_it_has_changed_anything():
    # At the first minibatch of the first epoch the parameters are the ones that
    # collected the data, so the KL must be zero. A non-zero reading here means
    # `evaluate` and `act` disagree, which is the failure that trains happily on
    # the wrong distribution.
    policy = a_policy(seed=5)
    episodes = some_episodes(policy, games=3, seed=5)
    batch = assemble(episodes, policy.layout, PPOConfig())
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=0.0)

    stats = update(policy, optimiser, batch, PPOConfig(epochs=1, minibatch=len(batch)))
    assert stats.positions == len(batch)
    assert stats.approx_kl == pytest.approx(0.0, abs=1e-5)
    assert stats.clip_fraction == pytest.approx(0.0, abs=1e-6)


def test_an_update_moves_the_value_head_towards_the_outcome_it_was_shown():
    policy = a_policy(seed=6)
    episodes = some_episodes(policy, games=3, seed=6)
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig(epochs=1, minibatch=len(batch), entropy_coefficient=0.0)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    first = update(policy, optimiser, batch, config)
    for _ in range(20):
        last = update(policy, optimiser, batch, config)

    assert last.value_loss < first.value_loss
    # Anti-vacuity: a value loss that started at zero would satisfy nothing.
    assert first.value_loss > 1e-4


def test_the_clipped_surrogate_stops_rewarding_a_ratio_past_the_clip():
    # Hand-built rather than collected, so the arithmetic is checkable: with a
    # positive advantage and a ratio well past 1 + clip, the loss must be the
    # clipped branch and so independent of how far past it went.
    advantage = torch.tensor([1.0])
    clip = 0.2
    for log_ratio in (1.0, 2.0, 5.0):
        ratio = torch.tensor([log_ratio]).exp()
        clamped = ratio.clamp(1 - clip, 1 + clip) * advantage
        loss = -torch.min(ratio * advantage, clamped)
        assert loss.item() == pytest.approx(-(1 + clip))


def test_learning_from_no_episodes_at_all_is_an_error_rather_than_a_silent_pass():
    policy = a_policy(seed=7)
    with pytest.raises(ValueError):
        assemble([], policy.layout, PPOConfig())


def test_minibatches_partition_the_batch_and_never_yield_a_single_row():
    """A one-row trailing minibatch is a nan bomb, so it must not be emitted.

    `advantage.std()` on one row divides by `n - 1 == 0` and returns nan, which
    the caller's `+ 1e-8` cannot rescue; the loss goes nan, `clip_grad_norm_`
    propagates it and `optimiser.step()` writes nan into every parameter while
    the run keeps logging happily. `positions` varies per iteration, so this is
    a ~1-in-`minibatch` chance every iteration -- rare enough per iteration to
    go a long stretch of training without tripping, and exactly the kind of
    bug a long run can carry silently until this test makes it certain.
    """
    for size in range(2, 40):
        for minibatch in range(2, 12):
            chunks = list(
                ppo._minibatches(size, minibatch, torch.Generator().manual_seed(0))
            )
            assert chunks, f"no minibatches for size={size} minibatch={minibatch}"
            for chunk in chunks:
                assert len(chunk) >= 2, (
                    f"size={size} minibatch={minibatch} yielded {len(chunk)} row(s)"
                )
                assert len(chunk) <= minibatch + 1, "a fold may add one row, not more"
            # Still an exact partition of every row, each appearing once.
            assert sorted(torch.cat(chunks).tolist()) == list(range(size))


def test_the_kl_gauge_cannot_report_a_negative_divergence():
    """`(r - 1) - log r` is non-negative in exact arithmetic and was not in
    float32: `exp(x)` rounds to 1.0 below the ~1.2e-7 epsilon, so the estimator
    collapsed to `-log_ratio`. On-policy batches sit exactly there — cohort
    collection leaves log-ratios around 1e-9 from reduction order alone — and
    the gauge reported -5.8e-10 for a divergence that is really ~1.7e-19.
    """
    # A batch the current weights actually played: the log-ratios are not zero,
    # they are reduction-order noise around zero with a tiny drift. Symmetric
    # values would let the naive form's errors cancel and hide this.
    log_ratio = torch.tensor([1e-9, 2e-9, 5e-10, 3e-9, 1e-9], dtype=torch.float32)

    naive = ((log_ratio.exp() - 1) - log_ratio).mean()
    stable = (torch.expm1(log_ratio) - log_ratio).mean()

    assert naive < 0, "the defect this guards no longer reproduces"
    assert stable >= 0


def test_the_kl_gauge_is_unchanged_where_it_was_already_right():
    """Every reading on record sits above a log-ratio of ~1e-3, where the two
    forms agree — so no recorded figure is affected by the fix."""
    log_ratio = torch.tensor([0.01, -0.02, 0.05, 0.2], dtype=torch.float32)

    naive = ((log_ratio.exp() - 1) - log_ratio).mean()
    stable = (torch.expm1(log_ratio) - log_ratio).mean()

    assert stable == pytest.approx(float(naive), rel=1e-3)


def test_the_default_config_still_trains_on_terminal_outcomes():
    assert PPOConfig().value_lam == 1.0


def test_value_lam_below_one_is_refused():
    # `value_lam` mixed a continuous margin head's own estimates into a
    # lambda-return target (`lambda_returns`, since retired) -- the win
    # head's target is the categorical one-hot winner, which has no such
    # continuous bootstrap to mix towards, so anything below 1.0 is refused
    # outright rather than silently reinterpreted. See the module docstring.
    with pytest.raises(ValueError, match="value_lam"):
        PPOConfig(value_lam=0.97)


def test_reward_has_no_other_registered_mode():
    with pytest.raises(ValueError, match="reward"):
        PPOConfig(reward="margin")


def test_critic_none_credits_every_decision_with_the_terminal_return():
    # REINFORCE: the advantage of every step of a seat's trajectory is that
    # seat's terminal return, whole — not lam**(T-t) times it, which is what
    # running GAE over a zeroed head would silently produce. The terminal
    # return is `win_loss`: the win head's own target, not the auxiliary
    # margin head's.
    policy = a_policy(seed=3)
    episodes = some_episodes(policy, games=2, seed=3)
    batch = assemble(episodes[:1], policy.layout, PPOConfig(critic="none"))

    episode = episodes[0]
    rows = 0
    for seat, trajectory in enumerate(episode.trajectories):
        own = to_frame(win_loss(episode.outcome), seat)[0]
        for _ in trajectory:
            assert float(batch.advantage[rows]) == pytest.approx(own, abs=1e-6)
            rows += 1
    assert rows == len(batch)


def test_critic_none_never_touches_the_value_head():
    # Both wires cut: no value term in the loss means no gradient reaches the
    # head, so its parameters are bit-identical after an update that visibly
    # moved the rest of the network.
    policy = a_policy(seed=7)
    episodes = some_episodes(policy, games=3, seed=7)
    config = PPOConfig(critic="none", epochs=2, minibatch=64)
    batch = assemble(episodes, policy.layout, config)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    head = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if name.startswith("value")
    }
    rest = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if not name.startswith("value")
    }
    assert head, "the head module is built in every mode; only its wires differ"

    update(policy, optimiser, batch, config)

    for name, p in policy.net.named_parameters():
        if name.startswith("value"):
            assert torch.equal(p, head[name]), name
    assert any(
        not torch.equal(p, rest[name])
        for name, p in policy.net.named_parameters()
        if not name.startswith("value")
    )


def test_critic_aux_trains_the_head_the_policy_never_reads():
    # "aux" keeps the trunk-shaping wire and cuts the pricing one: advantages
    # are identical to critic="none", and the head still moves.
    policy = a_policy(seed=8)
    episodes = some_episodes(policy, games=2, seed=8)
    config = PPOConfig(critic="aux", epochs=1, minibatch=128)
    batch = assemble(episodes, policy.layout, config)

    plain = assemble(episodes, policy.layout, PPOConfig(critic="none"))
    assert torch.equal(batch.advantage, plain.advantage)

    head = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if name.startswith("value")
    }
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)
    update(policy, optimiser, batch, config)
    assert any(
        not torch.equal(p, head[name])
        for name, p in policy.net.named_parameters()
        if name.startswith("value")
    )


def test_the_kl_break_is_a_ceiling_that_stops_further_epochs():
    policy = a_policy(seed=9)
    episodes = some_episodes(policy, games=2, seed=9)
    batch = assemble(episodes, policy.layout, PPOConfig())
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    # A threshold below any real post-step divergence stops the update after
    # the first finished epoch; the default of 0 takes every epoch, which is
    # every run on record.
    tripped = update(
        policy, optimiser, batch, PPOConfig(epochs=4, minibatch=64, kl_break=1e-9)
    )
    assert tripped.epochs_taken == 1

    untripped = update(policy, optimiser, batch, PPOConfig(epochs=2, minibatch=64))
    assert untripped.epochs_taken == 2


def test_one_critic_flag_is_one_wire():
    with pytest.raises(ValueError):
        PPOConfig(critic="none", value_lam=0.9)
    with pytest.raises(ValueError):
        PPOConfig(critic="bogus")


def paired_episodes(policy, games: int, seed: int = 0):
    """A bounded pair-dealt cohort, played out in full — what `pair_baseline`
    is owed by collection."""
    episodes = Collector(
        policy,
        lanes=games,
        seed=seed,
        action_cap=400,
        max_trades=0,
        deal=games,
        pair_boards=True,
    ).drain()
    return [decided(e) for e in episodes]


def episodes_with_outcomes(policy, winners_by_index, seed: int = 21):
    """Real transitions, hand-chosen winners.

    The collector donates valid observations and actions; the test owns the
    payoff arithmetic, which is what lets the pair-baseline identities be
    asserted exactly rather than to a tolerance. `win_loss` reads only
    `outcome.winner`, so that is the one field this hands out differently
    per index; `points` rides along unchanged since nothing here reads it.
    """
    donor = some_episodes(policy, games=1, seed=seed)[0]
    return [
        dataclasses.replace(
            donor,
            index=index,
            outcome=dataclasses.replace(donor.outcome, winner=winner),
        )
        for index, winner in winners_by_index.items()
    ]


def test_the_pair_baseline_never_touches_the_value_targets():
    # The baseline is a policy-gradient control variate: it changes what the
    # policy is paid, never what the head is trained toward. Bit-identical,
    # not approximately equal — the head's target is off-path here.
    policy = a_policy(seed=12)
    episodes = paired_episodes(policy, games=4, seed=12)

    plain = assemble(episodes, policy.layout, PPOConfig())
    paired = assemble(episodes, policy.layout, PPOConfig(pair_baseline=True))

    assert torch.equal(plain.value_target, paired.value_target)
    assert not torch.equal(plain.advantage, paired.advantage)


def test_the_gae_terminal_is_the_pair_adjusted_payoff():
    policy = a_policy(seed=13)
    episodes = paired_episodes(policy, games=2, seed=13)
    config = PPOConfig(pair_baseline=True)
    batch = assemble(episodes, policy.layout, config)

    payoffs = {e.index: win_loss(e.outcome) for e in episodes}
    rows = 0
    for episode in episodes:
        own_pay = payoffs[episode.index]
        mate_pay = payoffs[episode.index ^ 1]
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            estimates = np.array([t.value[0] for t in trajectory], dtype=np.float32)
            adjusted = np.float32((own_pay[seat] - mate_pay[seat]) / 2)
            expected = advantages(estimates, adjusted, config.lam)
            got = batch.advantage[rows : rows + len(trajectory)].numpy()
            assert np.array_equal(got, expected)
            rows += len(trajectory)
    assert rows == len(batch)


def test_pair_adjusted_payoffs_are_zero_sum_negated_and_zero_on_a_draw():
    policy = a_policy(seed=21)
    episodes = episodes_with_outcomes(
        policy,
        {
            0: 0,  # seat 0 wins
            1: 1,  # its mate (1 ^ 1 == 0): the mirror winner, seat 1
            2: 0,  # a pair whose halves end identically: seat 0 both times
            3: 0,
        },
    )
    # critic="none" credits every decision with the terminal payoff whole, so
    # each seat's rows read the adjusted payoff directly.
    config = PPOConfig(critic="none", pair_baseline=True)
    batch = assemble(episodes, policy.layout, config)

    adjusted: dict[tuple[int, int], float] = {}
    row = 0
    for episode in episodes:
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            block = batch.advantage[row : row + len(trajectory)]
            assert float(block.min()) == float(block.max())
            adjusted[(episode.index, seat)] = float(block[0])
            row += len(trajectory)
    assert row == len(batch)

    # Exactly zero-sum per game: both halves' raw vectors are, and halving a
    # difference cannot leave the plane.
    assert sum(adjusted[(0, seat)] for seat in range(4)) == 0.0
    assert sum(adjusted[(1, seat)] for seat in range(4)) == 0.0
    # The two halves are exact negatives of each other, bitwise.
    for seat in range(4):
        assert adjusted[(0, seat)] == -adjusted[(1, seat)]
    # A pair whose halves ended identically pays exactly nothing: whatever the
    # two games shared — geometry, seat order, everything — cancels whole.
    for index in (2, 3):
        for seat in range(4):
            assert adjusted[(index, seat)] == 0.0
    # Anti-vacuity: the non-draw pair moved somebody.
    assert any(adjusted[(0, seat)] != 0.0 for seat in range(4))


def test_a_missing_mate_or_an_odd_cohort_refuses_to_baseline():
    policy = a_policy(seed=14)
    episodes = paired_episodes(policy, games=3, seed=14)
    config = PPOConfig(pair_baseline=True)

    # Game 2's mate was never dealt: an odd cohort cannot be complete pairs.
    with pytest.raises(ValueError, match="even cohort"):
        assemble(episodes, policy.layout, config)
    # One half alone is the same defect at any cohort size.
    solo = [episode for episode in episodes if episode.index == 0]
    with pytest.raises(ValueError, match="mate"):
        assemble(solo, policy.layout, config)


def test_the_pair_baseline_defaults_off():
    assert PPOConfig().pair_baseline is False


# ---------------------------------------------------------------------------
# The quantile value loss: an alternative value-head shape under evaluation.
# ---------------------------------------------------------------------------


def a_quantile_policy(players: int = 4, seed: int = 0, quantiles: int = 8):
    """`a_policy`'s net with the value head widened, and nothing else moved."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(
        space_for(game),
        graph,
        players,
        ModelConfig(width=16, rounds=1, value_head="quantile", quantiles=quantiles),
    )
    return NetworkPolicy(net, space_for(game), packing(graph, players))


def _terms(policy, batch, config, rows=None):
    """One minibatch's terms over the whole batch, advantages normalised as
    `update` normalises them."""
    rows = torch.arange(len(batch)) if rows is None else rows
    advantage = batch.advantage[rows]
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return ppo.minibatch_terms(
        policy,
        batch.buffer[rows],
        batch.mask[rows],
        batch.chosen[rows],
        batch.log_prob[rows],
        advantage,
        batch.value_target[rows],
        batch.margin_target[rows],
        config,
    ), advantage


def test_a_linear_heads_loss_is_exactly_cross_entropy_against_the_one_hot_winner():
    """`minibatch_terms`'s whole win-head arithmetic, restated as a property.

    The loss this builds has to be the cross-entropy expression built by
    hand, on the same graph -- `torch.equal`, not `approx`, since a
    reassociated sum would be a different run. The auxiliary margin term is
    checked at its default weight (0.0), where it must contribute exactly
    nothing to the loss or its gradient.
    """
    policy = a_policy(seed=31)
    episodes = some_episodes(policy, games=3, seed=31)
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig()
    assert config.aux_margin_weight == 0.0

    terms, advantage = _terms(policy, batch, config)

    evaluation = policy.evaluate(
        batch.buffer, batch.mask, batch.chosen
    )
    assert evaluation.quantiles is None
    ratio = (evaluation.log_prob - batch.log_prob).exp()
    policy_loss = -torch.min(
        ratio * advantage,
        ratio.clamp(1 - config.clip, 1 + config.clip) * advantage,
    ).mean()
    log_probs_win = torch.log_softmax(evaluation.value_logits, dim=-1)
    value_loss = -(batch.value_target * log_probs_win).sum(-1).mean()
    margin_loss = (evaluation.margin - batch.margin_target).pow(2).mean()
    entropy = evaluation.entropy.mean()
    expected = (
        policy_loss
        + config.value_coefficient * value_loss
        + config.aux_margin_weight * margin_loss
        - config.entropy_coefficient * entropy
    )

    assert torch.equal(terms.loss, expected)
    assert torch.equal(terms.value_term, config.value_coefficient * value_loss)
    assert torch.equal(terms.margin_term, torch.zeros_like(terms.loss))
    # Cross-entropy and squared error are on different scales, so the two are
    # never equal in general -- unlike the pre-win-head arithmetic where
    # `value_loss` and `value_mse` were the same expression.
    assert terms.value_mse != terms.value_loss

    # And the gradient, which is what actually moves the weights.
    got = torch.autograd.grad(terms.loss, list(policy.net.parameters()))
    want = torch.autograd.grad(expected, list(policy.net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_the_aux_margin_head_is_absent_from_the_advantage_at_the_default_weight():
    """`aux_margin_weight=0.0` -- every run before this head existed -- must
    keep the head from reaching the loss (and, through it, the advantage and
    the trunk it shapes) at all, not merely shrink its contribution to
    something small: a Python float `0.0` times a tensor is an exact zero.

    Pinned two ways: the loss term itself is bit-identical to zero (already
    checked above), and an update at this weight leaves the head's own
    parameters exactly where they started while the rest of the network
    visibly moves -- the same shape `critic="none"` pins for the win head.
    """
    policy = a_policy(seed=39)
    episodes = some_episodes(policy, games=3, seed=39)
    config = PPOConfig(epochs=2, minibatch=64)
    assert config.aux_margin_weight == 0.0
    batch = assemble(episodes, policy.layout, config)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    aux_before = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if name.startswith("aux_margin")
    }
    rest_before = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if not name.startswith("aux_margin")
    }
    assert aux_before, "the aux head is built at every weight, including zero"

    update(policy, optimiser, batch, config)

    for name, p in policy.net.named_parameters():
        if name.startswith("aux_margin"):
            assert torch.equal(p, aux_before[name]), name
    assert any(
        not torch.equal(p, rest_before[name])
        for name, p in policy.net.named_parameters()
        if not name.startswith("aux_margin")
    )


def test_a_positive_aux_margin_weight_trains_the_head_the_advantage_never_reads():
    policy = a_policy(seed=40)
    episodes = some_episodes(policy, games=2, seed=40)
    config = PPOConfig(epochs=1, minibatch=128, aux_margin_weight=0.5)
    batch = assemble(episodes, policy.layout, config)

    plain = assemble(episodes, policy.layout, PPOConfig())
    # The advantage is a pure function of the win-head estimates and the
    # terminal outcome, neither of which the margin weight touches.
    assert torch.equal(batch.advantage, plain.advantage)

    aux_before = {
        name: p.detach().clone()
        for name, p in policy.net.named_parameters()
        if name.startswith("aux_margin")
    }
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)
    update(policy, optimiser, batch, config)
    assert any(
        not torch.equal(p, aux_before[name])
        for name, p in policy.net.named_parameters()
        if name.startswith("aux_margin")
    )


def test_the_value_head_shape_never_reaches_the_assembled_batch():
    """`assemble` reads the estimates the collector recorded, so the value
    target, the GAE advantages and the zero-sum projection are the same tensors
    whatever shape produced those estimates. Bit-identical, off-path."""
    policy = a_policy(seed=32)
    episodes = some_episodes(policy, games=3, seed=32)

    plain = assemble(episodes, policy.layout, PPOConfig())
    again = assemble(episodes, a_quantile_policy(seed=32).layout, PPOConfig())

    assert torch.equal(plain.advantage, again.advantage)
    assert torch.equal(plain.value_target, again.value_target)
    assert torch.equal(plain.buffer, again.buffer)


def test_a_warm_started_quantile_head_prices_the_same_decisions():
    """The advantage path is untouched: same features, same mean, same GAE.

    The warm start makes the two heads' `V` the same number to one float32
    rounding of a 32-term mean, so the estimates GAE consumes — and therefore
    the advantages — are the same to that same rounding, four orders below the
    1/30 lattice the label lives on.
    """
    from hexn.model import quantile_warm_start

    policy = a_policy(seed=33)
    quantile = a_quantile_policy(seed=34, quantiles=32)
    quantile.net.load_state_dict(
        quantile_warm_start(policy.net.state_dict(), 4, 32)
    )
    episodes = some_episodes(policy, games=2, seed=33)
    batch = assemble(episodes, policy.layout, PPOConfig())

    with torch.no_grad():
        scalar_value = policy.net(*unpack(policy.layout, batch.buffer)).value
        widened = quantile.net(*unpack(quantile.layout, batch.buffer)).value

    assert torch.allclose(scalar_value, widened, rtol=0, atol=1e-6)
    # Anti-vacuity: a head that predicted nothing would satisfy the line above.
    assert scalar_value.abs().max() > 1e-3

    scalar_gae = advantages(scalar_value[:, 0].numpy(), 0.4, 0.95)
    widened_gae = advantages(widened[:, 0].numpy(), 0.4, 0.95)
    assert np.allclose(scalar_gae, widened_gae, rtol=0, atol=1e-5)
    assert np.abs(scalar_gae).max() > 1e-3


def test_the_quantile_value_term_is_the_pinball_loss_on_the_same_target():
    """The one thing that changes, and the one thing that must not.

    The value term becomes `quantile_huber_loss` against the identical
    `value_target` one-hot vector `assemble` already builds; the policy term
    and the entropy term are untouched. This exact pairing -- a quantile head
    against a categorical one-hot target -- is untested territory (see
    `hexn.ppo`'s module docstring, "The value loss under a quantile head"),
    but the wiring still has to be exactly this whether or not the
    combination has ever been trained.
    """
    from hexn.model import QUANTILE_HUBER_KAPPA, quantile_huber_loss

    policy = a_quantile_policy(seed=35)
    episodes = some_episodes(policy, games=3, seed=35)
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig()

    terms, advantage = _terms(policy, batch, config)

    evaluation = policy.evaluate(
        batch.buffer, batch.mask, batch.chosen
    )
    assert evaluation.quantiles is not None
    expected = quantile_huber_loss(
        evaluation.quantiles,
        batch.value_target,
        policy.net.value.levels,
        QUANTILE_HUBER_KAPPA,
    )
    assert terms.value_loss == pytest.approx(float(expected.detach()), rel=1e-6)
    # Not the squared error, which is the other column — and the two are on
    # different scales, which is exactly why both are logged.
    assert terms.value_mse != terms.value_loss
    assert terms.value_mse == pytest.approx(
        float((evaluation.value - batch.value_target).pow(2).mean().detach()),
        rel=1e-6,
    )


def test_the_quantile_loss_reaches_the_shared_trunk():
    """The value loss has to shape the features the policy reads, not stop
    at the value head's own weights.

    The policy head and the value head share a trunk. If the quantile value
    loss's gradient never reached that shared trunk, an ablation comparing
    value-head shapes by the policy they produce would be measuring nothing:
    the trunk, and so the policy, would never see the value loss at all.
    """
    policy = a_quantile_policy(seed=36)
    episodes = some_episodes(policy, games=2, seed=36)
    batch = assemble(episodes, policy.layout, PPOConfig())

    terms, _ = _terms(policy, batch, PPOConfig())
    trunk = [
        p for name, p in policy.net.named_parameters() if not name.startswith("value")
    ]
    grads = torch.autograd.grad(terms.value_term, trunk, allow_unused=True)

    assert any(g is not None and g.any() for g in grads)


def test_an_update_moves_a_quantile_head_towards_the_outcome_it_was_shown():
    """The same claim `test_an_update_moves_the_value_head_towards_the_outcome`
    makes for the scalar head, read on the column the two arms share."""
    policy = a_quantile_policy(seed=37)
    episodes = some_episodes(policy, games=3, seed=37)
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig(epochs=1, minibatch=len(batch), entropy_coefficient=0.0)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    first = update(policy, optimiser, batch, config)
    for _ in range(20):
        last = update(policy, optimiser, batch, config)

    assert last.value_loss < first.value_loss
    assert last.value_mse < first.value_mse
    assert first.value_mse > 1e-4
    # The head spreads out: it opens on orthogonal noise and ends up ordered
    # enough that the levels no longer coincide.
    with torch.no_grad():
        spread = policy.net(*unpack(policy.layout, batch.buffer)).quantiles
    assert spread.std(-1).mean() > 0


def test_the_logged_value_mse_is_read_off_the_win_probability_under_every_head():
    """`value_mse` is always the plain squared error of the softmaxed mean
    against the one-hot target -- the one column comparable across a value-
    head ablation, whatever the loss actually differentiated. Under a scalar
    head it necessarily differs from `value_loss` (cross-entropy), which is
    exactly why both are logged separately rather than assumed equal."""
    policy = a_policy(seed=38)
    episodes = some_episodes(policy, games=2, seed=38)
    batch = assemble(episodes, policy.layout, PPOConfig())
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=0.0)

    stats = update(policy, optimiser, batch, PPOConfig(epochs=1, minibatch=len(batch)))

    assert stats.value_mse != stats.value_loss
    assert stats.value_mse > 0.0
