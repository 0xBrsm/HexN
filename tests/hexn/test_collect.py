# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.collect import ParallelCollector, WorkerSpec  # noqa: E402


def spec(worker: int, workers: int, **overrides) -> WorkerSpec:
    # Trade-free by default (`max_trades=0`): nothing below is about trading,
    # and a network's gate prices every coverable candidate after every MAIN
    # action (`hexset.trading.trade_event`, `hexn.policy.NetworkPolicy.trader`)
    # -- at a nonzero budget that made a worker's cohort cost tens of seconds
    # to a couple of minutes for plumbing tests that never look at a trade.
    # Same trade `some_episodes` (`test_ppo.py`) and `searched_episodes`
    # (`test_exit.py`) already make. A test that needs trades overrides it.
    base = dict(
        seed=5,
        players=4,
        lanes=2,
        action_cap=500,
        max_trades=0,
        first_game=worker,
        stride=workers,
        width=8,
        rounds=1,
        torch_seed=1000 + worker,
        mix=(),
        parent="",
    )
    base.update(overrides)
    return WorkerSpec(**base)


def test_workers_deal_disjoint_strided_indices_and_ship_valid_episodes():
    collector = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        episodes = collector.collect(4)

        assert len(episodes) == 4
        indices = sorted(e.index for e in episodes)
        assert len(set(indices)) == 4, f"an index was dealt twice: {indices}"
        # Both workers contributed: the two stride residues both appear.
        assert {i % 2 for i in indices} == {0, 1}
        for episode in episodes:
            assert len(episode) > 0
            assert episode.outcome.actions > 0
        assert collector.games == 4
    finally:
        collector.close()


def test_collection_can_be_started_then_finished_around_other_work():
    collector = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        collector.start_collect(4)
        with pytest.raises(RuntimeError, match="already in flight"):
            collector.start_collect(2)

        # Production uses this window for the preceding batch's GPU update.
        assert collector.games == 0
        episodes = collector.finish_collect()

        assert len(episodes) == 4
        assert collector.games == 4
        assert collector.last_collect_seconds > 0.0
        with pytest.raises(RuntimeError, match="nothing in flight"):
            collector.finish_collect()
    finally:
        collector.close()


def test_a_parallel_resume_never_redeals_a_seen_index():
    first = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        seen = {e.index for e in first.collect(4)}
        base = first.games_started()
    finally:
        first.close()

    # The resume rule: base is the max over worker counters, so indices the
    # slow worker never reached are skipped — unused seeds, never replays.
    assert base > max(seen)
    second = ParallelCollector([spec(base + 0, 2), spec(base + 1, 2)])
    try:
        fresh = {e.index for e in second.collect(2)}
    finally:
        second.close()
    assert not (seen & fresh)
    assert min(fresh) >= base


def test_workers_cast_mix_opponents_and_record_only_the_learner():
    collector = ParallelCollector(
        [
            spec(0, 2, mix=(("random", 1.0),)),
            spec(1, 2, mix=(("random", 1.0),)),
        ]
    )
    try:
        episodes = collector.collect(2)
    finally:
        collector.close()

    for episode in episodes:
        assert any(pid == 1 for pid in episode.cast)
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_sync_ships_weights_the_workers_actually_load():
    from hexset.actions import build_space
    from hexset.board.board import random_base_board
    from hexset.encoding import static_graph
    from hexn.model import HexNet, ModelConfig

    board = random_base_board(random.Random(5))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, 4
    )
    net = HexNet(space, static_graph(topology), 4, ModelConfig(width=8, rounds=1))

    collector = ParallelCollector([spec(0, 1, lanes=2)])
    try:
        collector.sync(net)
        episodes = collector.collect(1)
        assert episodes and len(episodes[0]) > 0
    finally:
        collector.close()


def test_a_searched_worker_returns_episodes_whose_transitions_carry_targets():
    """The whole point of sharding the searched path: the corpus must still be
    distillable, which means every transition needs its `Target` to survive the
    pipe as well as the search."""
    from hexn.collect import ParallelCollector, WorkerSpec
    from hexn.expert import Target

    collector = ParallelCollector(
        [
            WorkerSpec(
                seed=5,
                players=4,
                lanes=2,
                action_cap=600,
                max_trades=0,
                first_game=0,
                stride=1,
                width=16,
                rounds=1,
                torch_seed=5,
                simulations=8,
                wave=4,
            )
        ]
    )
    try:
        episodes = collector.collect(1)
    finally:
        collector.close()
    assert episodes
    targets = [
        t.aux
        for e in episodes
        for traj in e.trajectories
        for t in traj
        if t.aux is not None
    ]
    assert targets
    assert all(isinstance(t, Target) for t in targets)
    # The prior rides along too, or `contested_only` has nothing to filter on.
    searched = [t for t in targets if len(t.options) > 1]
    assert searched and all(t.prior is not None for t in searched)


def test_a_worker_with_no_simulations_is_the_plain_policy_path():
    from hexn.collect import ParallelCollector, WorkerSpec

    collector = ParallelCollector(
        [
            WorkerSpec(
                seed=6,
                players=4,
                lanes=2,
                action_cap=600,
                max_trades=0,
                first_game=0,
                stride=1,
                width=16,
                rounds=1,
                torch_seed=6,
            )
        ]
    )
    try:
        episodes = collector.collect(1)
    finally:
        collector.close()
    assert episodes
    # The invariant is that no search target appears -- `aux` is a general
    # pocket and the plain path is free to use it for other things.
    from hexn.expert import Target

    auxes = [t.aux for e in episodes for traj in e.trajectories for t in traj]
    assert auxes
    assert not any(isinstance(a, Target) for a in auxes)


def test_the_flat_wire_format_rebuilds_byte_identical_episodes():
    """`Flattened` is a container change, not a data change.

    The same cohort, shipped flat and rebuilt, must assemble to a Batch equal
    tensor-for-tensor to the one the original objects assemble to — that is
    the acceptance test the perf review set. The rebuilt observations must
    also share one packed buffer, because restoring `pack`'s gather path is
    half the point.
    """
    import pickle

    from hexset.actions import space_for
    from hexset.board.board import random_base_board
    from hexn.collect import Flattened
    from hexset.encoding import static_graph
    from hexset.game import start
    from hexn.model import HexNet, ModelConfig, packing
    from hexn.policy import NetworkPolicy
    from hexn.ppo import PPOConfig, assemble
    from hexn.selfplay import Collector

    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(0)
    net = HexNet(space_for(game), graph, 4, ModelConfig(width=16, rounds=1))
    policy = NetworkPolicy(net, space_for(game), packing(graph, 4))
    # Trade-free and short: the round trip is about the flattening, and a
    # network gate prices every candidate after every MAIN action.
    episodes = Collector(policy, lanes=4, seed=3, action_cap=400, max_trades=0).collect(3)

    flat = pickle.loads(pickle.dumps(Flattened(episodes, policy.layout)))
    rebuilt = flat.episodes()

    assert [e.index for e in rebuilt] == [e.index for e in episodes]
    assert [e.outcome for e in rebuilt] == [e.outcome for e in episodes]
    assert [len(e) for e in rebuilt] == [len(e) for e in episodes]

    config = PPOConfig()
    original = assemble(episodes, policy.layout, config)
    again = assemble(rebuilt, policy.layout, config)
    for field in (
        "buffer",
        "mask",
        "chosen",
        "log_prob",
        "advantage",
        "value_target",
    ):
        assert torch.equal(getattr(original, field), getattr(again, field)), field

    shared = {
        id(t.observation._packed)
        for e in rebuilt
        for seat in e.trajectories
        for t in seat
    }
    assert shared == {id(flat.buffer)}


def test_a_league_worker_records_every_seat_and_loads_a_weight_list():
    """Stage 2's worker contract: two learner nets share the table, every seat
    records under its caster's id, and sync ships one dict per learner."""
    collector = ParallelCollector([spec(w, 2, learners=2) for w in range(2)])
    try:
        from hexn.collect import _build  # the same construction the worker runs

        policies, _ = _build(spec(0, 2, learners=2))
        assert len(policies) == 2
        collector.sync_many([policies[0].net, policies[1].net])

        episodes = collector.collect(4)
        assert len(episodes) == 4
        for episode in episodes:
            assert set(episode.cast) == {0, 1}
            for seat, trajectory in enumerate(episode.trajectories):
                assert trajectory, "both ids are learners; every seat records"
                assert all(t.seat == seat for t in trajectory)
    finally:
        collector.close()


def test_a_league_spec_refuses_a_mix():
    from hexn.collect import _build

    with pytest.raises(ValueError):
        _build(spec(0, 1, learners=2, mix=(("random", 0.15),)))


def test_the_league_caster_balances_seats_and_fixes_adjacency():
    """The property that made a permutation option necessary.

    Rotation alone balances every learner over every board seat -- which is why
    board position was excluded as the cause of the noise heats' two tight pairs
    -- but it leaves the cyclic order round the table invariant, so learner 0's
    turn-order successor is learner 1 in every game ever played.
    """
    from hexn.collect import league_caster

    caster = league_caster(4, 4)
    casts = [caster(i) for i in range(4)]
    for learner in range(4):
        seats = [cast.index(learner) for cast in casts]
        assert sorted(seats) == [0, 1, 2, 3], "each learner takes each seat once"
    for cast in casts:
        # Successor of learner k round the table is always k+1 (mod 4).
        for seat, learner in enumerate(cast):
            assert cast[(seat + 1) % 4] == (learner + 1) % 4


def test_a_permuted_learner_order_reseats_the_cycle_without_unbalancing_seats():
    from hexn.collect import league_caster

    caster = league_caster(4, 4, order=(0, 2, 1, 3))
    casts = [caster(i) for i in range(4)]
    for learner in range(4):
        seats = [cast.index(learner) for cast in casts]
        assert sorted(seats) == [0, 1, 2, 3], "the share stays balanced"
    successor = {}
    for cast in casts:
        for seat, learner in enumerate(cast):
            successor.setdefault(learner, cast[(seat + 1) % 4])
            assert successor[learner] == cast[(seat + 1) % 4], "adjacency is fixed"
    # 0 now sits before 2 rather than before 1, which is the whole point.
    assert successor[0] == 2 and successor[2] == 1


def test_a_learner_order_that_is_not_a_permutation_is_refused():
    import pytest

    from hexn.collect import league_caster

    with pytest.raises(ValueError, match="permutation"):
        league_caster(4, 4, order=(0, 1, 1, 3))


def test_a_paired_caster_casts_both_halves_of_a_pair_identically():
    from hexn.collect import league_caster, paired_caster

    plain = league_caster(4, 4)
    caster = paired_caster(plain)
    for k in range(12):
        assert caster(2 * k) == caster(2 * k + 1) == plain(k)


def test_a_paired_league_balances_every_seat_over_a_doubled_window():
    from collections import Counter

    from hexn.collect import league_caster, paired_caster

    learners, players = 4, 4
    caster = paired_caster(league_caster(learners, players))
    # The documented cost of pairing: the exact balance `league_caster` gives
    # over any `learners`-game window now takes `2 * learners` games — and it
    # holds at every offset, not just on pair boundaries.
    for offset in range(2 * learners):
        window = [caster(offset + i) for i in range(2 * learners)]
        for seat in range(players):
            share = Counter(cast[seat] for cast in window)
            assert share == {k: 2 for k in range(learners)}


def test_a_paired_worker_wraps_its_caster_and_deals_paired_boards():
    from hexn.collect import _build, league_caster

    _, collector = _build(spec(0, 1, learners=2, pair_boards=True))
    assert collector.pair_boards
    plain = league_caster(2, 4)
    for k in range(6):
        assert collector.caster(2 * k) == collector.caster(2 * k + 1) == plain(k)


# --- `--mix` routed through `named_opponent` -------------------------------
#
# The compatibility claim these pin: `parent` must resolve to exactly what it
# resolved to before any entrant spec was accepted, because recorded runs were
# collected against it and its numbers cannot be allowed to change meaning
# without their configuration changing. `greedy` was the other reserved name;
# HexSet 0.47.0 deleted the bot behind it with `search2`, and it is now an
# unknown mix name rather than a name quietly pointing at a different bot.


def _cast_cohort(opponents, *, seed=11, games=4, lanes=2):
    """A cohort with every game cast, played by a torch-free learner.

    `RandomPolicy` rather than a network on purpose: two runs of this with
    identically seeded policies play identical games, so any difference in the
    episodes is a difference in the *opponent* and nothing else. A network
    learner would put a torch generator between the claim and the evidence.
    """
    from hexn.collect import mixed_caster
    from hexn.selfplay import Collector, RandomPolicy

    collector = Collector(
        RandomPolicy(random.Random(seed)),
        lanes=lanes,
        fill=False,
        players=4,
        seed=seed,
        action_cap=600,
        max_trades=3,
        opponents=opponents,
        caster=mixed_caster([1.0], 4, seed),
    )
    return collector.cohort(games)


def _a_checkpoint(path, *, players: int = 4):
    """A checkpoint tiny enough to load, in the shape `hexn.loop.save`
    writes -- same pattern as `test_netbot.a_checkpoint`, kept local so this
    file stays independent of it."""
    from hexn.model import HexNet, ModelConfig
    from hexset.actions import space_for
    from hexset.board.board import random_base_board
    from hexset.encoding import static_graph
    from hexset.game import start

    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(0)
    net = HexNet(space_for(game), graph, players, ModelConfig(width=8, rounds=1))
    torch.save(
        {
            "iteration": 1,
            "net": net.state_dict(),
            "args": {"players": players, "width": 8, "rounds": 1},
        },
        path,
    )
    return board


def test_a_bare_network_mix_entry_routes_through_the_batched_frozen_policy(tmp_path):
    """The RL-only pool: a bare `network:<path>` (no `@trades` override) has
    no trading stance of its own to preserve, so it costs the collector one
    batched forward a tick (`frozen`, a `NetworkPolicy`) rather than the
    batch-of-one `NetworkBot` `named_opponent`/`hexset.arena.spawn` would
    give it -- a batch-of-one pays the network's whole per-call dispatch
    cost once per lane per tick, which is exactly the overhead routing
    through `frozen` exists to avoid. `mix_opponents` needs a `board` to
    build `frozen`'s net on; without one it falls back to `named_opponent`,
    which the `network:<path>@0` case (an explicit override `frozen` cannot
    carry) still wants regardless.
    """
    from hexn.collect import check_mix, mix_opponents
    from hexn.policy import NetworkPolicy

    path = tmp_path / "latest.pt"
    board = _a_checkpoint(path)
    spec = f"network:{path}"

    check_mix([(spec, 1.0)], have_parent=False)

    bare = mix_opponents(
        [(spec, 1.0)], seed=1, lanes=2, board=board, players=4
    )[0]
    assert isinstance(bare, NetworkPolicy), (
        f"a bare network: entry must resolve to the batched frozen policy, "
        f"not {type(bare).__name__}"
    )

    # An explicit `@trades` override keeps its own fixed trade switch, which
    # `frozen`'s plain NetworkPolicy has no field for -- it must still go
    # through `named_opponent`/`hexset.arena.spawn`, whose `NetworkBot`
    # carries `max_trades` per entrant.
    overridden = mix_opponents(
        [(f"{spec}@0", 1.0)], seed=1, lanes=2, board=board, players=4
    )[0]
    assert not isinstance(overridden, NetworkPolicy)

    # And with no board at all (a caller that predates this routing), the
    # bare entry still falls back to `named_opponent` rather than raising.
    no_board = mix_opponents([(spec, 1.0)], seed=1, lanes=2)[0]
    assert not isinstance(no_board, NetworkPolicy)


def test_heximax_resolves_as_a_mix_opponent():
    """`"heximax"` is not in `hexset.arena.PRESETS`: `hexset.bots.heximax`
    registers it at import, and `hexn.collect` importing `hexset.bots` is
    what makes it resolvable in `--mix` from a collector process. Without that
    import the run dies at launch with `unknown entrant`."""
    from hexset.board.board import random_base_board
    from hexn.collect import check_mix, mix_opponents

    check_mix([("heximax", 0.25)], have_parent=False)
    board = random_base_board(random.Random(3))
    bot = mix_opponents([("heximax", 0.25)], seed=1, lanes=2)[0].spawn(board)
    assert hasattr(bot, "gains_many")
    # And it is the entrant, not a lookalike: `heximax`'s own preset carries
    # `depth=2, width=6, max_trades=None`. `--max-trades` is the run's switch
    # for the seats the *run* deals; a named entrant is literally the entrant
    # the arena scores, budget included.
    assert (bot.depth, bot.width, bot.max_trades) == (2, 6, None)


def test_a_mix_without_a_parent_entry_never_builds_one():
    """`parent` arrives as a thunk so a worker pays no `torch.load` for a
    checkpoint nothing in its mix casts."""
    from hexn.collect import mix_opponents

    def thunk():
        raise AssertionError("built a parent for a mix that never asks for one")

    opponents = mix_opponents(
        [("heximax", 0.15), ("random", 0.1)],
        seed=1,
        lanes=2,
        parent=thunk,
    )
    assert len(opponents) == 2


def test_the_parent_mix_opponent_is_whatever_the_thunk_returned():
    from hexn.collect import mix_opponents

    sentinel = object()
    opponents = mix_opponents(
        [("heximax", 0.1), ("parent", 0.1)],
        seed=1,
        lanes=2,
        parent=lambda: sentinel,
    )
    # Order is the caster's id space: id 2 is `mix[1]`.
    assert opponents[1] is sentinel
    with pytest.raises(ValueError, match="needs a parent"):
        mix_opponents([("parent", 0.1)], seed=1, lanes=2)


def test_a_worker_casts_an_arena_entrant_and_still_records_only_the_learner():
    from hexn.selfplay import owned

    collector = ParallelCollector(
        [
            spec(0, 2, mix=(("random", 1.0),)),
            spec(1, 2, mix=(("random", 1.0),)),
        ]
    )
    try:
        episodes = collector.collect(2)
    finally:
        collector.close()

    assert len(episodes) == 2
    for episode in episodes:
        assert any(pid == 1 for pid in episode.cast)
        assert episode.outcome.actions > 0
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    # `owned` is the assemble-side gate on the same fact, and the two must agree:
    # nothing an opponent did may reach an update.
    for kept, episode in zip(owned(episodes, 0), episodes):
        assert kept.trajectories == episode.trajectories


def test_a_worker_casts_a_table_and_records_only_the_learner_seat():
    """The seating-free geometry in the sharded collector: one learner seat,
    the rest drawn from the pool by id, opponent seats never recorded."""
    mix = (("table(random|heximax|random-placement)", 1.0),)
    collector = ParallelCollector([spec(0, 1, lanes=4, mix=mix)])
    try:
        episodes = collector.collect(24)
    finally:
        collector.close()

    seen: set[int] = set()
    for episode in episodes:
        assert episode.cast.count(0) == 1, episode.cast
        assert set(episode.cast) <= {0, 1, 2, 3}
        seen |= {pid for pid in episode.cast if pid}
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    assert seen == {1, 2, 3}


def test_a_worker_seats_self_from_the_pool_and_records_every_learner_seat():
    """`self` in a table pool is the learner again: one to four learner seats a
    game, all of them recorded, opponent seats still never."""
    mix = (("table(self|self|random|random-placement)", 1.0),)
    collector = ParallelCollector([spec(0, 1, lanes=4, mix=mix)])
    try:
        episodes = collector.collect(24)
    finally:
        collector.close()

    learner_seats = {episode.cast.count(0) for episode in episodes}
    assert learner_seats <= {1, 2, 3, 4} and max(learner_seats) > 1, learner_seats
    for episode in episodes:
        assert set(episode.cast) <= {0, 1, 2}
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_a_worker_hands_tempered_self_seats_their_temperature():
    """`self~band` in the pool reaches the policy as `Request.temperature`,
    on cast-id-0 seats only, and those seats are still recorded."""
    from hexn.collect import mix_table, mix_temperatures
    from hexn.selfplay import Collector, RandomPolicy

    class Recording(RandomPolicy):
        seen: list[tuple[int, float]] = []

        def act(self, requests):
            for request in requests:
                self.seen.append((request.seat, request.temperature))
            return super().act(requests)

    mix = (("table(self~0.4|self~0.4|random)", 1.0),)
    collector = Collector(
        Recording(random.Random(0)),
        lanes=4,
        players=4,
        seed=11,
        opponents=[RandomPolicy(random.Random(1))],
        caster=lambda index: mix_table(mix, 4, 11)(index)[0],
        temperatures=mix_temperatures(mix, 4, 11),
    )
    episodes = collector.collect(8)
    temperatures = [t for _, t in Recording.seen]
    assert temperatures and all(0.6 <= t <= 1.4 for t in temperatures)
    assert any(t != 1.0 for t in temperatures) and any(t == 1.0 for t in temperatures)
    for episode in episodes:
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
