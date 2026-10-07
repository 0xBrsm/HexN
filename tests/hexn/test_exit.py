# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import copy
import dataclasses
import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexn.exit import (  # noqa: E402
    DistillConfig,
    _backward,
    assemble,
    contested,
    losses,
    measure,
    stake,
    update,
)
from hexn.ppo import attach_prior  # noqa: E402
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
    """Searched, trade-free, decided games, so every transition has a target.

    Trade-free (`max_offers=0`), for the reason `test_ppo.some_episodes` gives
    for the plain-policy path: nothing here is about trading, and a width-16
    net's gate prices every coverable candidate after every MAIN action
    (`hexset.trading.trade_event`) -- unset, that made a searched game cost
    tens of seconds. Capped at 400 actions like `test_ppo.some_episodes`, and
    `decided`, because the win-head properties below need a game with an
    actual winner to be anything but vacuous.
    """
    search = Search(
        LeafEvaluator(policy=policy),
        simulations=8,
        wave=4,
        rng=random.Random(seed),
    )
    expert = SearchPolicy(search, rng=random.Random(seed))
    episodes = Collector(
        expert, lanes=2, seed=seed, action_cap=400, max_offers=0
    ).collect(games)
    return [decided(e) for e in episodes]


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


@pytest.fixture(scope="module")
def corpus():
    """One searched corpus for the whole module: `(policy, space, layout,
    episodes)`.

    A searched game is the expensive thing here and every test used to play
    its own. Episodes are frozen, so sharing them is safe; the policy is not,
    so a test that trains takes `a_student` instead -- `refresh` needs the
    policy that recorded the priors to still be that policy.
    """
    policy, space, layout = a_setup()
    return policy, space, layout, searched_episodes(policy, space, games=2)


def a_student(policy, space, layout):
    """A trainable copy of the collecting policy."""
    return NetworkPolicy(copy.deepcopy(policy.net), space, layout)


def a_batch(corpus, config=None):
    policy, space, layout, episodes = corpus
    return assemble(episodes, space, layout, config or DistillConfig())


# --- the projection -------------------------------------------------------


def test_cooling_sharpens_the_target(corpus):
    """Temperature acts on options, before they land on slots."""
    warm = a_batch(corpus, DistillConfig(temperature=1.0))
    cold = a_batch(corpus, DistillConfig(temperature=0.25))

    assert cold.slot_target.max(-1).values.mean() > warm.slot_target.max(-1).values.mean()


# --- the loss -------------------------------------------------------------


def test_the_projected_loss_equals_the_cross_entropy_over_options(corpus):
    """The identity the module rests on, checked against a direct sum.

    The policy scores a slot, not an option, and several options can share one
    slot -- so this is the check that projecting the target onto slots loses
    nothing: the slot cross-entropy must equal the plain cross-entropy over
    concrete options. Contract 5 gives every option its own flat slot, so this
    also pins that each visited option lands its share on `space.index(option)`.
    """
    policy, space, layout, episodes = corpus
    config = DistillConfig()
    batch = assemble(episodes, space, layout, config)

    targets = [
        transition.aux
        for episode in episodes
        for trajectory in episode.trajectories
        for transition in trajectory
    ]

    rows = torch.arange(len(batch))
    with torch.no_grad():
        slots, _, _, _ = policy.distributions(batch.buffer, batch.mask)
    projected = float(losses(slots, batch, rows))

    direct = 0.0
    for row, target in enumerate(targets):
        weights = visit_policy(target.visits, config.temperature)
        for option, share in zip(target.options, weights):
            direct -= share * float(slots[row, space.index(option)])
    direct /= len(targets)

    assert projected == pytest.approx(direct, rel=1e-4)


# --- the update -----------------------------------------------------------


def test_a_batch_refuses_transitions_that_carry_no_search_target(corpus):
    policy, space, layout, _ = corpus
    # The plain policy path, a few dozen actions: all it has to supply is a
    # transition with no `Target` on it.
    episodes = Collector(
        policy, lanes=1, seed=0, action_cap=40, max_offers=0
    ).collect(1)
    with pytest.raises(ValueError, match="search targets"):
        assemble(episodes, space, layout, DistillConfig())


def test_distilling_a_fixed_batch_moves_the_policy_toward_the_search(corpus):
    """The end-to-end property: repeated updates cut the distillation loss.

    Full-batch steps on a fixed seed. Argmax agreement is not asserted: the
    loss is a cross-entropy to the search's soft visit distribution, which
    can fall while the argmax wanders, and on a two-game corpus it does --
    from a near-uniform init (policy heads at gain 0.01) the starting argmax
    is arbitrary rather than a baseline.
    """
    student = a_student(*corpus[:3])
    batch = a_batch(corpus)
    optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-2)
    config = DistillConfig(epochs=1, minibatch=len(batch))

    first = update(student, optimiser, batch, config)
    for _ in range(10):
        last = update(student, optimiser, batch, config)

    assert last.policy_loss < first.policy_loss


# --- micro-batches, the tether and the movement gauges -------------------


def test_a_micro_batched_minibatch_has_the_whole_minibatchs_gradient(corpus):
    """Slices of a minibatch, each normalised by the minibatch's own
    denominators, sum to the gradient of one backward over all of it -- under
    a weighted policy term and with the anchor and the tether on, the terms
    whose denominators are not a plain row count."""
    policy, space, layout, _ = corpus
    batch = attach_prior(
        a_batch(corpus, DistillConfig(contested_only=True)), policy, layout
    )
    rows = torch.arange(len(batch))

    def gradients(micro: int) -> list:
        student = a_student(policy, space, layout)
        config = DistillConfig(micro_batch=micro, anchor=0.5, prior_kl=0.3)
        student.net.zero_grad(set_to_none=True)
        sums = _backward(student, batch, rows, config, with_value=True, with_policy=True)
        return sums, [p.grad.clone() for p in student.net.parameters() if p.grad is not None]

    whole, whole_grads = gradients(0)
    sliced, sliced_grads = gradients(max(2, len(batch) // 5))
    assert len(whole_grads) == len(sliced_grads)
    for a, b in zip(whole_grads, sliced_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    for name in ("policy_loss", "value_loss", "anchor", "prior_kl", "hits"):
        assert float(sliced[name]) == pytest.approx(float(whole[name]), rel=1e-4, abs=1e-6)


def test_the_policy_term_on_the_whole_batch_is_the_projected_loss(corpus):
    """`_backward`'s policy term is `losses`, the cross-entropy the projection
    identity above is checked against."""
    policy, space, layout, _ = corpus
    batch = a_batch(corpus)
    rows = torch.arange(len(batch))
    with torch.no_grad():
        slots, _, _, _ = policy.distributions(batch.buffer, batch.mask)
    student = a_student(policy, space, layout)
    sums = _backward(student, batch, rows, DistillConfig(), with_value=False, with_policy=True)
    assert float(sums["policy_loss"]) == pytest.approx(float(losses(slots, batch, rows)), rel=1e-5)


def test_the_tether_needs_the_start_policy_on_the_batch(corpus):
    student = a_student(*corpus[:3])
    with pytest.raises(ValueError, match="attach_prior"):
        update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=1e-3),
            a_batch(corpus),
            DistillConfig(prior_kl=0.1, epochs=1),
        )


def test_measure_reads_no_movement_before_an_update_and_some_after(corpus):
    policy, space, layout, _ = corpus
    student = a_student(policy, space, layout)
    batch = attach_prior(a_batch(corpus), student, layout)

    still = measure(student, batch)
    assert still["kl_to_start"] == pytest.approx(0.0, abs=1e-6)
    assert still["agreement_end"] == still["agreement_start"]
    assert still["target_ce_end"] == pytest.approx(still["target_ce_start"], rel=1e-5)
    assert 0 < still["decision_positions"] <= len(batch)

    optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-2)
    update(student, optimiser, batch, DistillConfig(epochs=3, minibatch=len(batch)))
    moved = measure(student, batch)
    assert moved["kl_to_start"] > 1e-4
    assert moved["target_ce_end"] < moved["target_ce_start"]


def test_the_tether_holds_the_policy_nearer_its_start(corpus):
    policy, space, layout, _ = corpus

    def walked(weight: float) -> float:
        student = a_student(policy, space, layout)
        batch = attach_prior(a_batch(corpus), student, layout)
        optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-2)
        stats = update(
            student,
            optimiser,
            batch,
            DistillConfig(epochs=3, minibatch=len(batch), prior_kl=weight),
            generator=torch.Generator().manual_seed(0),
        )
        assert (stats.prior_kl > 0.0) == (weight > 0.0)
        return measure(student, batch)["kl_to_start"]

    assert walked(20.0) < walked(0.0)


def test_a_searched_worker_roots_on_the_movers_belief_in_k_worlds():
    """The collection workers' search is the honest one: rooted in `k` worlds
    drawn from the mover's own view, never on the true state."""
    from hexn.collect import WorkerSpec, _build
    from hexn.expert import SearchPolicy

    spec = WorkerSpec(
        seed=5, players=4, lanes=1, action_cap=50, max_offers=0, first_game=0,
        stride=1, width=8, rounds=1, torch_seed=1000, simulations=4, wave=2, k=3,
    )
    _, collector = _build(spec)
    acting = collector.policy
    assert isinstance(acting, SearchPolicy)
    assert acting.search.k == 3
    assert acting.search.hidden is True


def test_a_bootstrapped_target_reads_the_estimate_that_many_decisions_later(corpus):
    _, space, layout, episodes = corpus
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


def test_the_value_head_is_trained_on_the_terminal_outcome(corpus):
    # The default, and what AlphaZero does: the one-hot eventual winner
    # (`hexn.rewards.win_loss`), exactly `hexn.ppo`'s own target, the same
    # vector at a seat's first decision and its last. `--value-horizon`
    # bootstraps instead.
    _, space, layout, episodes = corpus
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
    # Anti-vacuity: `decided` gives every episode a winner.
    assert (batch.value_target.sum(dim=1) == 1.0).all()


def test_a_linear_heads_value_loss_is_exactly_cross_entropy_against_the_one_hot_winner(
    corpus,
):
    """`update`'s value pass, restated as a property -- `test_ppo.
    test_a_linear_heads_loss_is_exactly_cross_entropy_against_the_one_hot_winner`'s
    counterpart for distillation.

    A single minibatch spanning the whole batch, so the update's internal
    shuffle cannot change which rows land together; the loss is read off the
    parameters *before* `update` steps them, since that is when the forward
    pass computing it ran.
    """
    student = a_student(*corpus[:3])
    batch = a_batch(corpus)

    with torch.no_grad():
        _, _, value_logits, quantiles = student.distributions(batch.buffer, batch.mask)
    assert quantiles is None
    log_probs_win = torch.log_softmax(value_logits, dim=-1)
    expected = -(batch.value_target * log_probs_win).sum(-1).mean()

    optimiser = torch.optim.Adam(student.net.parameters(), lr=1e-3)
    config = DistillConfig(epochs=1, minibatch=len(batch))
    stats = update(student, optimiser, batch, config)

    assert stats.value_loss == pytest.approx(float(expected), rel=1e-5)
    # Cross-entropy and squared error are different expressions in general.
    assert stats.value_mse != stats.value_loss


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


def test_a_search_that_overrules_the_prior_is_contested():
    assert contested(a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1]))


def test_the_stake_is_the_gap_between_what_the_search_picked_and_what_it_overruled():
    target = a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1], [0.10, 0.16, -0.30])
    assert stake(target) == pytest.approx(0.06)


def test_the_stake_scale_weights_a_contested_row_by_what_it_is_worth(corpus):
    filtered = a_batch(corpus, DistillConfig(contested_only=True))
    staked = a_batch(corpus, DistillConfig(contested_only=True, stake_scale=0.05))
    assert (staked.policy_weight <= filtered.policy_weight + 1e-6).all()
    assert (staked.policy_weight >= 0.0).all()
    # The anchor holds whatever the policy term let go, and a part-staked row is
    # part-held: the two weights have to keep summing to one where a prior was
    # recorded, or the row is trained on less than all of itself.
    both = staked.policy_weight + staked.anchor_weight
    recorded = filtered.anchor_weight + filtered.policy_weight > 0.0
    assert bool(recorded.any())
    assert torch.allclose(both[recorded], torch.ones_like(both[recorded]), atol=1e-6)


def test_the_hard_target_puts_all_of_a_row_on_one_option(corpus):
    batch = a_batch(corpus, DistillConfig(hard_target=True))
    totals = batch.slot_target.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    # A one-hot over *options*, and under contract 5 every option has its own
    # slot, so the row's max is the whole of it.
    assert torch.allclose(
        batch.slot_target.max(-1).values,
        torch.ones_like(totals),
        atol=1e-5,
    )


def test_the_searched_corpus_records_a_prior_beside_its_visits(corpus):
    # Without this the filter has nothing to compare against, and it cannot be
    # recovered later: by training time the policy has moved.
    *_, episodes = corpus
    targets = [
        t.aux
        for e in episodes
        for traj in e.trajectories
        for t in traj
        if t.aux is not None
    ]
    assert targets
    # A forced position is evaluated too, so it records a prior, one-hot on
    # its only option -- and it cannot be contested, since the search's pick
    # and the prior's are the same option.
    searched = [t for t in targets if len(t.options) > 1]
    assert searched
    assert all(t.prior is not None for t in targets)
    assert all(len(t.prior) == len(t.options) for t in targets)
    forced = [t for t in targets if len(t.options) == 1]
    assert all(not contested(t) for t in forced)


def test_filtering_zeroes_the_policy_weight_where_the_search_agreed(corpus):
    plain = a_batch(corpus)
    filtered = a_batch(corpus, DistillConfig(contested_only=True))
    assert torch.all(plain.policy_weight == 1.0)
    # A network's argmax already agrees with the search on most rows, so
    # most of the batch must be switched off here -- and the batch must keep
    # its length regardless, since the value head still has to see it all.
    assert len(filtered) == len(plain)
    assert 0 < filtered.policy_weight.sum() < plain.policy_weight.sum()


# --- the anchor -----------------------------------------------------------
#
# `contested_only` filters the policy loss but cannot filter its effect: the
# trunk is shared, the value loss is unweighted, and a network has no
# per-position parameters. Left unchecked, a shared trunk drifts anyway on
# every row the policy loss switched off -- degrading agreement with the
# search on exactly the positions the filter was meant to leave alone, and
# costing the checkpoint strength against its own unfiltered parent. The
# anchor is the restoring force that holds those rows in place.


def test_the_anchor_holds_the_settled_rows_the_filter_let_go(corpus):
    """The whole point: rows the policy loss zeroed must not drift freely."""
    policy, space, layout, _ = corpus
    batch = a_batch(corpus, DistillConfig(contested_only=True))
    settled = batch.anchor_weight > 0
    assert bool(settled.any())
    # The anchor's target is a distribution over the same slots.
    totals = batch.anchor_slots[settled].sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)

    def drift(anchor: float) -> float:
        student = a_student(policy, space, layout)
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


def test_concatenating_batches_keeps_every_row(corpus):
    from hexn.exit import Batch

    policy, space, layout, episodes = corpus
    first = assemble(episodes[:1], space, layout, DistillConfig())
    second = assemble(episodes[1:], space, layout, DistillConfig())
    joined = Batch.concat([first, second])
    assert len(joined) == len(first) + len(second)
    for name in Batch.FIELDS:
        assert torch.equal(
            getattr(joined, name)[: len(first)], getattr(first, name)
        )
        assert torch.equal(
            getattr(joined, name)[len(first) :], getattr(second, name)
        )


def test_refreshing_against_the_collecting_policy_reproduces_the_filter(corpus):
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

    policy = corpus[0]
    config = DistillConfig(contested_only=True)
    batch = a_batch(corpus, config)
    refreshed = refresh(policy, batch, config)

    top = batch.slot_target.max(-1, keepdim=True).values
    decided = (batch.slot_target >= top).sum(-1) == 1

    assert int(decided.sum()) > 0
    assert torch.equal(batch.policy_weight[decided], refreshed.policy_weight[decided])


def test_packing_reports_the_policy_loss_over_contested_rows_only(corpus):
    """Unpacked, a minibatch holding no contested row reports a zero loss.

    Those zeros are averaged into `policy_loss`, so the reported number is
    diluted by the filter's selectivity rather than describing the rows the
    loss actually trained on. Packed, every reported step has contested rows in
    it.
    """
    policy, space, layout, _ = corpus
    batch = a_batch(corpus, DistillConfig(contested_only=True))
    assert float(batch.policy_weight.sum()) > 0

    # The dilution needs minibatches with nothing contested in them: `losses`
    # divides by the weight rather than by the row count, so a minibatch holding
    # any contested row already reports the mean over those rows. Thinned to
    # three contested rows over four minibatches, at least one of them reports
    # the zero this test is about.
    keep = torch.nonzero(batch.policy_weight > 0.0).squeeze(-1)[:3]
    thinned = torch.zeros_like(batch.policy_weight)
    thinned[keep] = batch.policy_weight[keep]
    batch = batch.__class__(
        **{
            name: thinned if name == "policy_weight" else getattr(batch, name)
            for name in batch.FIELDS
        }
    )
    minibatch = len(batch) // 4 + 1

    def run(pack: bool) -> float:
        student = a_student(policy, space, layout)
        stats = update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=1e-4),
            batch,
            DistillConfig(
                contested_only=True,
                hard_target=True,
                epochs=1,
                minibatch=minibatch,
                pack_contested=pack,
            ),
            generator=torch.Generator().manual_seed(0),
        )
        return stats.policy_loss

    assert run(pack=True) > run(pack=False)


# --- the run --------------------------------------------------------------

TINY_RUN = [
    "--device", "cpu", "--width", "8", "--rounds", "1",
    "--lanes", "1", "--games-per-iteration", "1", "--action-cap", "60",
    "--max-offers", "0", "--simulations", "2", "--wave", "2",
    "--epochs", "1", "--minibatch", "256", "--reanalyse-samples", "0",
]


def launch(directory, iterations, extra=()):
    """Freeze an exit run and launch it, as `test_ppo_main.run` does for PPO."""
    import pathlib

    from hexn.exit import __main__ as exit_main
    from hexn.run import freeze

    argv = TINY_RUN + [
        "--iterations", str(iterations), "--checkpoint-dir", str(directory),
    ] + list(extra)
    freeze("exit", directory.name, pathlib.Path(directory), argv,
           repo=pathlib.Path(directory), description="test run")
    return exit_main.main([str(directory)])


def test_exit_refuses_to_resume_nothing_or_to_start_over_a_run(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        launch(tmp_path / "empty", 1, ["--resume"])
    (tmp_path / "held").mkdir()
    (tmp_path / "held" / "latest.pt").write_bytes(b"")
    with pytest.raises(SystemExit, match="already holds a run"):
        launch(tmp_path / "held", 1)


def test_a_shard_written_before_a_crash_is_that_iterations_collection(
    tmp_path, monkeypatch
):
    """Collected and stored, killed in the update: the resumed iteration takes
    its shard rather than collecting again, and deals past it afterwards."""
    from hexn.exit import __main__ as exit_main
    from hexn.store import EpisodeStore

    class Crash(Exception):
        pass

    real_update = exit_main.update
    calls = []

    def update(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Crash
        return real_update(*args, **kwargs)

    monkeypatch.setattr(exit_main, "update", update)
    with pytest.raises(Crash):
        launch(tmp_path, 2)
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["iteration"] == 1
    orphan = EpisodeStore.resume(tmp_path / "replay", 10**9).shard_episodes(1)

    collections = []
    real_collect = Collector.collect

    def collect(self, *args, **kwargs):
        collections.append(1)
        return real_collect(self, *args, **kwargs)

    monkeypatch.setattr(Collector, "collect", collect)
    assert launch(tmp_path, 3, ["--resume"]) == 0

    assert len(collections) == 1, "iteration 1 adopted its shard; only 2 collected"
    store = EpisodeStore.resume(tmp_path / "replay", 10**9)
    assert store.iterations() == [0, 1, 2]
    assert [e.index for e in store.shard_episodes(1)] == [e.index for e in orphan]
    later = {e.index for e in store.shard_episodes(2)}
    assert min(later) > max(e.index for e in orphan)


def test_without_a_store_a_resumed_iteration_plays_only_its_missing_games(
    tmp_path, monkeypatch
):
    """`--replay-positions 0`: an iteration's games live in its partial until
    its checkpoint, so a crash in the update loses none of them, and the
    resumed iteration trains on them without dealing them again. `--init`
    with `--resume` on an empty directory is a fresh start from those weights,
    kept as `iter-00000.pt`; every iteration row carries the movement gauges."""
    from hexn import durable
    from hexn.exit import __main__ as exit_main

    seed_run = tmp_path / "seed"
    assert launch(seed_run, 1, ["--replay-positions", "0"]) == 0
    init = seed_run / "latest.pt"

    class Crash(Exception):
        pass

    real_update = exit_main.update
    calls = []

    def update(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Crash
        return real_update(*args, **kwargs)

    run = tmp_path / "run"
    extra = ["--replay-positions", "0", "--games-per-iteration", "2",
             "--init", str(init), "--resume"]
    monkeypatch.setattr(exit_main, "update", update)
    with pytest.raises(Crash):
        launch(run, 2, extra)
    assert (run / "iter-00000.pt").exists()
    assert torch.load(run / "latest.pt", weights_only=False)["iteration"] == 1
    held = sorted(durable.partial(run / "partial", 1).indices())
    assert len(held) == 2, "iteration 1's games were kept before the crash"
    assert not (run / "replay").exists()

    monkeypatch.setattr(exit_main, "update", real_update)
    kept = []
    real_keep = durable.Partial.keep

    def keep(self, episode):
        kept.append((self.directory.name, episode.index))
        return real_keep(self, episode)

    monkeypatch.setattr(durable.Partial, "keep", keep)
    assert launch(run, 3, extra) == 0

    assert [name for name, _ in kept] == ["iter-00002", "iter-00002"], (
        f"only iteration 2 dealt games on resume, got {kept}"
    )
    assert min(index for _, index in kept) > max(held)
    rows = [row for row in durable.read_lines(run / "log.jsonl") if "kl_to_start" in row]
    assert [row["iteration"] for row in rows] == [0, 1, 2]
    assert rows[1]["positions"] > 0 and "agreement_end" in rows[1]
    assert not durable.partials(run / "partial"), "every checkpointed iteration's games pruned"


class Killed(Exception):
    """A killed trainer, from inside it."""


def test_a_round_killed_mid_distillation_resumes_from_its_last_step(tmp_path, monkeypatch):
    """Killed after the third optimiser step of round 1's distillation: the
    resumed round reads its games back from the partial, plays none again,
    continues the update from the step file and ends on the uninterrupted
    run's weights and row, to the bit."""
    from hexn import durable, steps
    from hexn.exit import __main__ as exit_main

    shape = ["--replay-positions", "0", "--games-per-iteration", "2", "--lanes", "2",
             "--minibatch", "32", "--epochs", "2", "--prior-kl", "0.1"]
    straight = tmp_path / "straight"
    assert launch(straight, 2, shape) == 0

    killed = tmp_path / "killed"
    real_update = exit_main.update
    calls = []

    def update(policy, optimiser, batch, config, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            real_step = optimiser.step
            taken = []

            def step(*args, **kw):
                if len(taken) == 3:
                    raise Killed
                taken.append(1)
                return real_step(*args, **kw)

            optimiser.step = step
        return real_update(policy, optimiser, batch, config, **kwargs)

    monkeypatch.setattr(exit_main, "update", update)
    with pytest.raises(Killed):
        launch(killed, 2, shape)
    monkeypatch.setattr(exit_main, "update", real_update)
    assert torch.load(killed / steps.STEP, weights_only=False)["progress"]["step"] == 3
    assert len(durable.partial(killed / "partial", 1).finished()) == 2

    kept = []
    real_keep = durable.Partial.keep
    monkeypatch.setattr(durable.Partial, "keep", lambda self, e: kept.append(e.index) or real_keep(self, e))
    assert launch(killed, 2, shape + ["--resume"]) == 0
    assert kept == [], "the resumed round played none of its games again"

    final = torch.load(killed / "latest.pt", weights_only=False)
    reference = torch.load(straight / "latest.pt", weights_only=False)
    assert all(torch.equal(final["net"][k], reference["net"][k]) for k in reference["net"])
    assert not (killed / steps.STEP).exists() and not durable.partials(killed / "partial")
    rows = [r for r in durable.read_lines(killed / "log.jsonl") if "positions" in r]
    reference_rows = [r for r in durable.read_lines(straight / "log.jsonl") if "positions" in r]
    assert rows[1]["resumed_at_step"] == 3
    for name in ("policy_loss", "value_loss", "agreement", "prior_kl", "kl_to_start", "agreement_end"):
        assert rows[1][name] == reference_rows[1][name], name
    assert sorted(p.name for p in killed.glob("recent-*.pt")) == ["recent-00001.pt", "recent-00002.pt"]


def test_a_round_killed_mid_collection_plays_only_the_games_it_is_missing(tmp_path, monkeypatch):
    """Killed with round 1 part-collected: the resumed round keeps the games
    already on disk, tops them up with new ones only (the in-process searched
    collector is a stream, with no plan), and trains on exactly those."""
    from hexn import durable
    from hexn.exit import __main__ as exit_main

    shape = ["--replay-positions", "0", "--games-per-iteration", "3", "--lanes", "1"]
    interrupted = durable.partial(tmp_path / "partial", 1)
    real_tick = Collector.tick

    def tick(self):
        out = real_tick(self)
        if 0 < len(interrupted.finished()) < 3:
            raise Killed
        return out

    monkeypatch.setattr(Collector, "tick", tick)
    with pytest.raises(Killed):
        launch(tmp_path, 2, shape)
    monkeypatch.setattr(Collector, "tick", real_tick)
    had = interrupted.finished()
    assert 0 < len(had) < 3

    kept, used = [], []
    real_keep = durable.Partial.keep
    monkeypatch.setattr(durable.Partial, "keep", lambda self, e: kept.append(e.index) or real_keep(self, e))
    real_assemble = exit_main.assemble
    monkeypatch.setattr(
        exit_main, "assemble",
        lambda episodes, *a, **k: used.append([e.index for e in episodes]) or real_assemble(episodes, *a, **k),
    )
    assert launch(tmp_path, 2, shape + ["--resume"]) == 0
    assert len(kept) == 3 - len(had) and min(kept) > max(had), "only the missing games were played"
    assert used == [sorted(had + kept)], "the resumed round trained on its games, in game order"
