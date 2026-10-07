# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import copy
import dataclasses
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import ppo  # noqa: E402
from hexn.ppo import __main__ as ppo_main  # noqa: E402
from hexset.actions import space_for  # noqa: E402
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
        policy, lanes=lanes, seed=seed, action_cap=400, max_offers=0
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


@pytest.fixture(scope="module")
def collected():
    """One policy and the decided games it played, for the whole module.

    Episodes are frozen and shared; a test that steps an optimiser takes
    `a_student` so the collecting policy stays the one that recorded
    `log_prob` and `value`.
    """
    policy = a_policy()
    return policy, some_episodes(policy, games=3)


def a_student(policy):
    """A trainable copy of the collecting policy."""
    return NetworkPolicy(copy.deepcopy(policy.net), policy.space, policy.layout)


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


def test_every_position_in_a_game_carries_that_game_s_terminal_outcome(collected):
    # The win head is trained on the one-hot eventual winner, never on a
    # bootstrap, so a seat's target is the same vector at its first decision
    # and its last. The auxiliary margin head's target is `relative_points`,
    # rotated the same way, checked alongside.
    policy, episodes = collected
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


def test_a_batch_carries_one_flat_index_per_transition_and_nothing_else(collected):
    # Contract 5 leaves the update with a single categorical: no pair mask, no
    # offer slot, no joint log-prob. If any of that came back, `Batch` would
    # have to carry it and `minibatch_terms` would have to be handed it. Every
    # transition of every episode is kept, in the per-seat order it was filed.
    policy, episodes = collected
    assert all(sum(1 for s in e.trajectories if s) >= 2 for e in episodes)
    batch = assemble(episodes, policy.layout, PPOConfig())

    transitions = [t for e in episodes for seat in e.trajectories for t in seat]
    assert len(batch) == len(transitions)
    assert not hasattr(batch, "pair")
    assert not hasattr(batch, "offer")
    assert batch.chosen.tolist() == [t.index for t in transitions]


def test_an_update_returns_a_ratio_of_one_before_it_has_changed_anything(collected):
    # At the first minibatch of the first epoch the parameters are the ones that
    # collected the data, so the KL must be zero. A non-zero reading here means
    # `evaluate` and `act` disagree, which is the failure that trains happily on
    # the wrong distribution.
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    policy = a_student(policy)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=0.0)

    stats = update(policy, optimiser, batch, PPOConfig(epochs=1, minibatch=len(batch)))
    assert stats.positions == len(batch)
    assert stats.approx_kl == pytest.approx(0.0, abs=1e-5)
    assert stats.clip_fraction == pytest.approx(0.0, abs=1e-6)


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


def test_critic_none_credits_every_decision_with_the_terminal_return(collected):
    # REINFORCE: the advantage of every step of a seat's trajectory is that
    # seat's terminal return, whole — not lam**(T-t) times it, which is what
    # running GAE over a zeroed head would silently produce. The terminal
    # return is `win_loss`: the win head's own target, not the auxiliary
    # margin head's.
    policy, episodes = collected
    batch = assemble(episodes[:1], policy.layout, PPOConfig(critic="none"))

    episode = episodes[0]
    rows = 0
    for seat, trajectory in enumerate(episode.trajectories):
        own = to_frame(win_loss(episode.outcome), seat)[0]
        for _ in trajectory:
            assert float(batch.advantage[rows]) == pytest.approx(own, abs=1e-6)
            rows += 1
    assert rows == len(batch)


def test_critic_none_never_touches_the_value_head(collected):
    # Both wires cut: no value term in the loss means no gradient reaches the
    # head, so its parameters are bit-identical after an update that visibly
    # moved the rest of the network.
    policy, episodes = collected
    config = PPOConfig(critic="none", epochs=1)
    batch = assemble(episodes, policy.layout, config)
    policy = a_student(policy)
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


def test_critic_aux_trains_the_head_the_policy_never_reads(collected):
    # "aux" keeps the trunk-shaping wire and cuts the pricing one: advantages
    # are identical to critic="none", and the head still moves.
    policy, episodes = collected
    config = PPOConfig(critic="aux", epochs=1)
    batch = assemble(episodes, policy.layout, config)

    plain = assemble(episodes, policy.layout, PPOConfig(critic="none"))
    assert torch.equal(batch.advantage, plain.advantage)

    policy = a_student(policy)

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


def test_the_kl_break_is_a_ceiling_that_stops_further_epochs(collected):
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    # Two minibatches an epoch: the break is read at an epoch's end.
    minibatch = len(batch) // 2 + 1
    policy = a_student(policy)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-2)

    # A threshold below any real post-step divergence stops the update after
    # the first finished epoch; the default of 0 takes every epoch, which is
    # every run on record.
    tripped = update(
        policy, optimiser, batch, PPOConfig(epochs=4, minibatch=minibatch, kl_break=1e-9)
    )
    assert tripped.epochs_taken == 1

    untripped = update(
        policy, optimiser, batch, PPOConfig(epochs=2, minibatch=minibatch)
    )
    assert untripped.epochs_taken == 2


def paired_episodes(policy, games: int, seed: int = 0):
    """A bounded pair-dealt cohort, played out in full — what `pair_baseline`
    is owed by collection."""
    episodes = Collector(
        policy,
        lanes=games,
        seed=seed,
        action_cap=400,
        max_offers=0,
        deal=games,
        pair_boards=True,
    ).drain()
    return [decided(e) for e in episodes]


def episodes_with_outcomes(donor, winners_by_index):
    """Real transitions, hand-chosen winners.

    The collector donates valid observations and actions; the test owns the
    payoff arithmetic, which is what lets the pair-baseline identities be
    asserted exactly rather than to a tolerance. `win_loss` reads only
    `outcome.winner`, so that is the one field this hands out differently
    per index; `points` rides along unchanged since nothing here reads it.
    """
    return [
        dataclasses.replace(
            donor,
            index=index,
            outcome=dataclasses.replace(donor.outcome, winner=winner),
        )
        for index, winner in winners_by_index.items()
    ]


def test_the_gae_terminal_is_the_pair_adjusted_payoff(collected):
    policy, _ = collected
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


def test_pair_adjusted_payoffs_are_zero_sum_negated_and_zero_on_a_draw(collected):
    policy, donors = collected
    episodes = episodes_with_outcomes(
        donors[0],
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


def test_a_missing_mate_or_an_odd_cohort_refuses_to_baseline(collected):
    policy, _ = collected
    episodes = paired_episodes(policy, games=3, seed=14)
    config = PPOConfig(pair_baseline=True)

    # Game 2's mate was never dealt: an odd cohort cannot be complete pairs.
    with pytest.raises(ValueError, match="even cohort"):
        assemble(episodes, policy.layout, config)
    # One half alone is the same defect at any cohort size.
    solo = [episode for episode in episodes if episode.index == 0]
    with pytest.raises(ValueError, match="mate"):
        assemble(solo, policy.layout, config)


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


def test_a_linear_heads_loss_is_exactly_cross_entropy_against_the_one_hot_winner(
    collected,
):
    """`minibatch_terms`'s whole win-head arithmetic, restated as a property.

    The loss this builds has to be the cross-entropy expression built by
    hand, on the same graph -- `torch.equal`, not `approx`, since a
    reassociated sum would be a different run. The auxiliary margin term is
    checked at its default weight (0.0), where it must contribute exactly
    nothing to the loss or its gradient.
    """
    policy, episodes = collected
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


def test_a_positive_aux_margin_weight_trains_the_head_the_advantage_never_reads(
    collected,
):
    policy, episodes = collected
    config = PPOConfig(epochs=1, aux_margin_weight=0.5)
    batch = assemble(episodes, policy.layout, config)

    plain = assemble(episodes, policy.layout, PPOConfig())
    # The advantage is a pure function of the win-head estimates and the
    # terminal outcome, neither of which the margin weight touches.
    assert torch.equal(batch.advantage, plain.advantage)

    policy = a_student(policy)
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


def test_the_quantile_value_term_is_the_pinball_loss_on_the_same_target(collected):
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

    # Any recorded transitions will do: only the value term is compared, and
    # it reads the batch's targets, not its log-probs.
    _, episodes = collected
    policy = a_quantile_policy(seed=35)
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


def test_the_quantile_loss_reaches_the_shared_trunk(collected):
    """The value loss has to shape the features the policy reads, not stop
    at the value head's own weights.

    The policy head and the value head share a trunk. If the quantile value
    loss's gradient never reached that shared trunk, an ablation comparing
    value-head shapes by the policy they produce would be measuring nothing:
    the trunk, and so the policy, would never see the value loss at all.
    """
    _, episodes = collected
    policy = a_quantile_policy(seed=36)
    batch = assemble(episodes, policy.layout, PPOConfig())

    terms, _ = _terms(policy, batch, PPOConfig())
    trunk = [
        p for name, p in policy.net.named_parameters() if not name.startswith("value")
    ]
    grads = torch.autograd.grad(terms.value_term, trunk, allow_unused=True)

    assert any(g is not None and g.any() for g in grads)


# ---------------------------------------------------------------------------
# The prior term: KL(pi || prior) over every legal action.
# ---------------------------------------------------------------------------


def test_the_prior_rows_are_the_priors_own_masked_log_softmax():
    learner = a_policy(seed=41)
    prior = a_policy(seed=42)
    batch = assemble(some_episodes(learner, games=2, seed=41), learner.layout, PPOConfig())

    attached = ppo.attach_prior(batch, prior, learner.layout, chunk=7)

    with torch.no_grad():
        logits = prior.net(*unpack(prior.layout, batch.buffer)).logits
        expected = torch.log_softmax(logits.masked_fill(~batch.mask, -1e9), dim=-1)
    assert attached.prior_log_probs.shape == (len(batch), batch.mask.shape[1])
    # Chunking is a memory bound, not a different computation.
    assert torch.allclose(attached.prior_log_probs, expected, atol=1e-6)
    # The batch it came from is left alone: the prior is an addition.
    assert batch.prior_log_probs is None


def test_a_prior_equal_to_the_learner_adds_a_zero_divergence_and_nothing_else():
    policy = a_policy(seed=43)
    batch = assemble(some_episodes(policy, games=2, seed=43), policy.layout, PPOConfig())
    with_prior = ppo.attach_prior(batch, policy, policy.layout)
    config = PPOConfig(prior_kl=0.5)

    plain, _ = _terms(policy, batch, config)
    rows = torch.arange(len(batch))
    anchored, _ = _terms_with_prior(policy, with_prior, config, rows)

    assert float(anchored.prior_kl) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(anchored.loss, plain.loss, atol=1e-6)
    assert plain.prior_term is None and plain.prior_kl is None


def a_sharp_policy(seed: int):
    """`a_policy` with noise on its policy heads. A freshly built net's logits
    sit near zero, so any two of them are both near uniform and their
    divergence is ~1e-5 -- too close to zero to tell a term that works from
    one that does not."""
    policy = a_policy(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in policy.net.heads.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator))
    return policy


def test_the_prior_term_is_the_learners_kl_to_the_prior_by_hand():
    learner = a_policy(seed=44)
    prior = a_sharp_policy(seed=45)
    batch = ppo.attach_prior(
        assemble(some_episodes(learner, games=2, seed=44), learner.layout, PPOConfig()),
        prior,
        learner.layout,
    )
    config = PPOConfig(prior_kl=0.25)
    rows = torch.arange(len(batch))

    terms, _ = _terms_with_prior(learner, batch, config, rows)
    plain, _ = _terms(learner, batch, PPOConfig())

    log_p = learner.evaluate(batch.buffer, batch.mask, batch.chosen).log_probs
    log_q = batch.prior_log_probs
    by_hand = torch.stack(
        [
            (log_p[i][m].exp() * (log_p[i][m] - log_q[i][m])).sum()
            for i, m in enumerate(batch.mask)
        ]
    ).mean()
    by_hand_value = float(by_hand.detach())
    assert float(terms.prior_kl) == pytest.approx(by_hand_value, rel=1e-5)
    # A sharpened prior disagrees, so this is not the zero case again.
    assert by_hand_value > 1e-2
    assert torch.allclose(terms.loss, plain.loss + 0.25 * by_hand, atol=1e-5)
    # It is on the graph: the term alone moves the weights.
    learner.net.zero_grad()
    terms.prior_term.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in learner.net.parameters()
    )


def test_an_update_refuses_a_prior_weight_with_no_prior_attached():
    policy = a_policy(seed=46)
    batch = assemble(some_episodes(policy, games=1, seed=46), policy.layout, PPOConfig())
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="attach_prior"):
        update(policy, optimiser, batch, PPOConfig(prior_kl=0.1))


def test_an_update_logs_the_divergence_it_was_held_to():
    learner = a_policy(seed=47)
    prior = a_sharp_policy(seed=48)
    batch = ppo.attach_prior(
        assemble(some_episodes(learner, games=2, seed=47), learner.layout, PPOConfig()),
        prior,
        learner.layout,
    )
    optimiser = torch.optim.Adam(learner.net.parameters(), lr=1e-4)
    stats = update(learner, optimiser, batch, PPOConfig(prior_kl=0.1, epochs=1))
    assert stats.prior_kl > 1e-3
    # And a run with no prior logs exactly zero, not a stale number.
    plain = update(
        learner,
        torch.optim.Adam(learner.net.parameters(), lr=1e-4),
        dataclasses.replace(batch, prior_log_probs=None),
        PPOConfig(epochs=1),
    )
    assert plain.prior_kl == 0.0


def test_a_negative_prior_weight_is_refused():
    with pytest.raises(ValueError, match="prior_kl"):
        PPOConfig(prior_kl=-0.1)


def _terms_with_prior(policy, batch, config, rows):
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
        prior_log_probs=batch.prior_log_probs[rows],
    ), advantage


def _normalised(batch, rows):
    advantage = batch.advantage[rows]
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8)


def test_micro_batches_accumulate_the_whole_minibatch_s_gradient(collected):
    policy, episodes = collected
    batch = ppo.attach_prior(
        assemble(episodes, policy.layout, PPOConfig()), a_policy(seed=7), policy.layout
    )
    config = PPOConfig(prior_kl=0.1)
    rows = torch.randperm(len(batch), generator=torch.Generator().manual_seed(3))
    advantage = _normalised(batch, rows)

    whole = a_student(policy)
    one = ppo._terms_for(whole, batch, rows, advantage, config)
    one.loss.backward()
    # Three slices, the last one short: the weights are the row shares.
    sliced = a_student(policy)
    parts = ppo._accumulate(
        sliced, batch, rows, advantage, dataclasses.replace(config, micro_batch=len(rows) // 3 + 1)
    )

    for a, b in zip(whole.net.parameters(), sliced.net.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7)
    for field in ppo._ACCUMULATED:
        expected = getattr(one, field)
        if expected is None:  # no fast-win target here
            assert getattr(parts, field) is None
            continue
        torch.testing.assert_close(getattr(parts, field), expected.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(parts.value, one.value)


def test_a_micro_batched_update_takes_the_same_steps_and_logs_the_same_gauges(collected):
    policy, episodes = collected
    batch = ppo.attach_prior(
        assemble(episodes, policy.layout, PPOConfig()), a_policy(seed=7), policy.layout
    )
    # Two minibatches an epoch and two epochs, so later steps read moved weights.
    whole = PPOConfig(epochs=2, minibatch=len(batch) // 2 + 1, prior_kl=0.1)
    sliced = dataclasses.replace(whole, micro_batch=whole.minibatch // 3 + 1)

    runs = []
    for config in (whole, sliced):
        student = a_student(policy)
        # Plain SGD: the step is the clipped gradient itself, so the weights
        # afterwards say whether the gradients were the same.
        optimiser = torch.optim.SGD(student.net.parameters(), lr=0.1)
        stats = update(student, optimiser, batch, config, generator=torch.Generator().manual_seed(5))
        runs.append((student, stats))
    (a, got_a), (b, got_b) = runs

    for pa, pb in zip(a.net.parameters(), b.net.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-4, atol=1e-6)
    for field, value in dataclasses.asdict(got_a).items():
        assert getattr(got_b, field) == pytest.approx(value, rel=1e-4, abs=1e-6), field
    assert got_a.approx_kl > 0 and got_a.prior_kl > 0


def test_a_micro_batch_at_or_above_the_minibatch_is_the_one_pass_update(collected):
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig(epochs=1, minibatch=len(batch) // 2 + 1)
    weights = []
    for micro in (0, config.minibatch, 10 * config.minibatch):
        student = a_student(policy)
        optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-3)
        update(student, optimiser, batch, dataclasses.replace(config, micro_batch=micro),
               generator=torch.Generator().manual_seed(9))
        weights.append([p.detach().clone() for p in student.net.parameters()])
    for other in weights[1:]:
        assert all(torch.equal(x, y) for x, y in zip(weights[0], other))


def test_a_negative_micro_batch_is_refused():
    with pytest.raises(ValueError):
        PPOConfig(micro_batch=-1)


def with_estimates(episode, estimate):
    """`episode` with each transition's own win estimate set to
    `estimate(i)`, `i` counting the episode's transitions in filing order."""
    count = iter(range(10**9))
    trajectories = tuple(
        tuple(
            dataclasses.replace(t, value=(estimate(next(count)), *t.value[1:]))
            for t in trajectory
        )
        for trajectory in episode.trajectories
    )
    return dataclasses.replace(episode, trajectories=trajectories)


def a_setup_position(transition):
    from hexset.encoding import global_columns
    from hexset.game import Phase

    phase = global_columns(4)["phase"]
    return int(np.argmax(transition.observation.globals[phase])) in (
        Phase.SETUP_SETTLEMENT,
        Phase.SETUP_ROAD,
    )


def test_the_decided_cut_drops_settled_positions_after_pricing_them(collected):
    # Rows outside the cut leave every field together, and the rows kept carry
    # exactly the advantage they had in the full batch: GAE ran over the whole
    # trajectory, dropped positions included.
    policy, episodes = collected
    levels = (0.01, 0.3, 0.5, 0.97)
    games = [with_estimates(e, lambda i: levels[i % 4]) for e in episodes]
    full = assemble(games, policy.layout, PPOConfig())
    cut = assemble(games, policy.layout, PPOConfig(decided_cut=0.05))

    transitions = [t for e in games for seat in e.trajectories for t in seat]
    kept = [i for i, t in enumerate(transitions) if 0.05 <= t.value[0] <= 0.95]
    assert 0 < len(kept) < len(transitions)
    assert len(cut) == len(kept)
    assert cut.chosen.tolist() == full.chosen[kept].tolist()
    assert torch.equal(cut.advantage, full.advantage[kept])
    assert torch.equal(cut.value_target, full.value_target[kept])
    assert torch.equal(cut.buffer, full.buffer[kept])


def test_the_setup_weight_repeats_each_setup_position_in_place(collected):
    policy, episodes = collected
    full = assemble(episodes, policy.layout, PPOConfig())
    tripled = assemble(episodes, policy.layout, PPOConfig(setup_weight=3))

    transitions = [t for e in episodes for seat in e.trajectories for t in seat]
    rows = [
        i
        for i, t in enumerate(transitions)
        for _ in range(3 if a_setup_position(t) else 1)
    ]
    assert any(a_setup_position(t) for t in transitions)
    assert len(tripled) == len(rows)
    assert torch.equal(tripled.buffer, full.buffer[rows])
    assert torch.equal(tripled.advantage, full.advantage[rows])


def test_the_stakes_gauges_count_the_drops_and_how_they_ended(collected):
    policy, episodes = collected
    games = [with_estimates(e, lambda i: (0.01, 0.5, 0.99)[i % 3]) for e in episodes]
    gauges = ppo.stakes_gauges(games, PPOConfig(decided_cut=0.05, setup_weight=2))

    low = high = low_won = high_won = setup = recorded = 0
    for e in games:
        for seat, trajectory in enumerate(e.trajectories):
            for t in trajectory:
                recorded += 1
                won = e.outcome.winner == seat
                if t.value[0] < 0.05:
                    low, low_won = low + 1, low_won + won
                elif t.value[0] > 0.95:
                    high, high_won = high + 1, high_won + won
                elif a_setup_position(t):
                    setup += 1
    assert gauges["decided_share"] == pytest.approx((low + high) / recorded)
    assert gauges["decided_low_won"] == pytest.approx(low_won / low)
    assert gauges["decided_high_won"] == pytest.approx(high_won / high)
    assert gauges["setup_repeated"] == setup


@pytest.mark.parametrize(
    "fields",
    [
        {"decided_cut": -0.01},
        {"decided_cut": 0.5},
        {"setup_weight": 0},
        {"decided_cut": 0.05, "vp_reward": 0.01, "aux_margin_weight": 0.1},
        {"decided_cut": 0.05, "fast_win": (1.0,)},
    ],
)
def test_out_of_range_stakes_settings_are_refused(fields):
    with pytest.raises(ValueError):
        PPOConfig(**fields)


def test_only_fp16_is_an_amp_dtype():
    PPOConfig(amp="fp16")
    with pytest.raises(ValueError):
        PPOConfig(amp="bf16")


def test_amp_and_a_gradient_scaler_come_together(collected):
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    student = a_student(policy)
    optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-4)
    with pytest.raises(ValueError):
        update(student, optimiser, batch, PPOConfig(amp="fp16"))
    with pytest.raises(ValueError):
        update(student, optimiser, batch, PPOConfig(), scaler=torch.amp.GradScaler("cpu"))


def test_evaluate_hands_back_fp32_under_half_autocast(collected):
    # The heads run in fp16 under autocast; every loss term downstream reads
    # fp32, and the mask's fill does not fit in half at all.
    policy, episodes = collected
    batch = assemble(episodes[:1], policy.layout, PPOConfig())
    with torch.autocast("cpu", dtype=torch.float16):
        out = policy.evaluate(batch.buffer, batch.mask, batch.chosen)
    for tensor in (out.log_prob, out.log_probs, out.entropy, out.value, out.value_logits, out.margin):
        assert tensor.dtype == torch.float32
    assert torch.isfinite(out.log_prob).all()


def test_an_fp16_update_tracks_the_fp32_update_it_stands_in_for(collected):
    # On policy, so the recomputed log-probs should sit on the recorded ones:
    # the first minibatch's KL is the half-precision error alone.
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    config = PPOConfig(epochs=1, minibatch=256)
    runs = {}
    for amp in ("", "fp16"):
        student = a_student(policy)
        optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu") if amp else None
        runs[amp] = update(
            student,
            optimiser,
            batch,
            dataclasses.replace(config, amp=amp),
            generator=torch.Generator().manual_seed(0),
            scaler=scaler,
        )
    half, full = runs["fp16"], runs[""]
    assert half.approx_kl_first_minibatch < 1e-4
    assert half.policy_loss == pytest.approx(full.policy_loss, abs=1e-2)
    assert half.value_loss == pytest.approx(full.value_loss, rel=1e-2)
    assert half.amp_scale > 0
    assert full.amp_skipped == 0 and full.amp_scale == 0.0


def test_the_prior_under_fp16_matches_the_fp32_prior(collected):
    policy, episodes = collected
    batch = assemble(episodes, policy.layout, PPOConfig())
    full = ppo.attach_prior(batch, policy, policy.layout)
    half = ppo.attach_prior(batch, policy, policy.layout, chunk=64, amp="fp16")
    legal = batch.mask
    assert half.prior_log_probs.dtype == torch.float32
    assert torch.allclose(half.prior_log_probs[legal], full.prior_log_probs[legal], atol=2e-2)
