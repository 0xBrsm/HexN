# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import pytest
from _bots import needs

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.collect import ParallelCollector, WorkerSpec, _build, _serve  # noqa: E402


def spec(worker: int, workers: int, **overrides) -> WorkerSpec:
    # Trade-free by default (`max_offers=0`): nothing below is about trading,
    # and a network's gate prices every coverable candidate after every MAIN
    # action (`hexset.trading.trade_event`, `hexn.policy.NetworkPolicy.trader`)
    # -- at a nonzero budget that made a worker's cohort cost tens of seconds
    # to a couple of minutes for plumbing tests that never look at a trade.
    # Same trade `some_episodes` (`test_ppo.py`) and `searched_episodes`
    # (`test_exit.py`) already make. A test that needs trades overrides it.
    # Truncated games are as good as finished ones for every claim here.
    base = dict(
        seed=5,
        players=4,
        lanes=2,
        action_cap=200,
        max_offers=0,
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


# A table of arena entrants, so the one spawned pool below also carries the
# cast-and-record-only-the-learner claim. Scripted bots only: `heximax` would
# cost more than every other game in this file. `random-too` is registered by
# a test runtime each worker loads itself (`WorkerSpec.runtime`).
POOL = (("table(random|random-too)", 1.0),)
RUNTIME = ("_scripted_runtime",)


@pytest.fixture(scope="module")
def pool():
    """The one `ParallelCollector` this file spawns, shared by every test that
    needs real worker processes: a spawned worker pays a fresh interpreter's
    torch import, which cost more than the games these tests play. The pool
    keeps its history across tests, so they read its counters as deltas."""
    collector = ParallelCollector(
        [spec(0, 2, mix=POOL, runtime=RUNTIME), spec(1, 2, mix=POOL, runtime=RUNTIME)]
    )
    yield collector
    collector.close()


class ScriptedPipe:
    """A worker pipe's parent end with the commands queued up front, so
    `_serve` -- the loop a worker process runs -- can run in this process."""

    def __init__(self, *commands) -> None:
        self.commands = list(commands)
        self.sent: list = []

    def recv(self):
        if not self.commands:
            raise EOFError
        return self.commands.pop(0)

    def send(self, message) -> None:
        # Through pickle, as a real pipe would carry it.
        self.sent.append(pickle.loads(pickle.dumps(message)))


def served(worker: WorkerSpec, *commands) -> list:
    pipe = ScriptedPipe(*commands)
    _serve(worker, pipe)
    return pipe.sent


def test_workers_deal_disjoint_strided_indices_and_ship_valid_episodes(pool):
    before = pool.games
    episodes = pool.collect(4)

    assert len(episodes) == 4
    indices = sorted(e.index for e in episodes)
    assert len(set(indices)) == 4, f"an index was dealt twice: {indices}"
    # Both workers contributed: the two stride residues both appear.
    assert {i % 2 for i in indices} == {0, 1}
    for episode in episodes:
        assert len(episode) > 0
        assert episode.outcome.actions > 0
    assert pool.games == before + 4


def test_collection_can_be_started_then_finished_around_other_work(pool):
    before = pool.games
    pool.start_collect(4)
    with pytest.raises(RuntimeError, match="already in flight"):
        pool.start_collect(2)

    # Production uses this window for the preceding batch's GPU update.
    assert pool.games == before
    episodes = pool.finish_collect()

    assert len(episodes) == 4
    assert pool.games == before + 4
    assert pool.last_collect_seconds > 0.0
    with pytest.raises(RuntimeError, match="nothing in flight"):
        pool.finish_collect()


def test_a_parallel_resume_never_redeals_a_seen_index(pool):
    seen = {e.index for e in pool.collect(4)}
    base = pool.games_started()

    # The resume rule: base is the max over worker counters, so indices the
    # slow worker never reached are skipped — unused seeds, never replays.
    assert base > max(seen)
    # The resumed shards, built exactly as a worker builds them and played
    # here rather than behind a second spawn.
    fresh: set[int] = set()
    for worker in range(2):
        _, resumed = _build(spec(base + worker, 2, mix=POOL, runtime=RUNTIME))
        fresh |= {e.index for e in resumed.cohort(1)}
    assert not (seen & fresh)
    assert min(fresh) >= base


def test_sync_ships_weights_the_workers_actually_load(pool):
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

    pool.sync(net)
    episodes = pool.collect(1)
    assert episodes and len(episodes[0]) > 0


def test_a_worker_casts_a_table_and_records_only_the_learner_seat(pool):
    """The seating-free geometry in the sharded collector: one learner seat,
    the rest drawn from the pool by id, opponent seats never recorded."""
    from hexn.selfplay import owned

    episodes = pool.collect(4)

    seen: set[int] = set()
    for episode in episodes:
        assert episode.cast.count(0) == 1, episode.cast
        assert set(episode.cast) <= {0, 1, 2}
        seen |= {pid for pid in episode.cast if pid}
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    assert seen == {1, 2}
    # `owned` is the assemble-side gate on the same fact, and the two must agree:
    # nothing an opponent did may reach an update.
    for kept, episode in zip(owned(episodes, 0), episodes):
        assert kept.trajectories == episode.trajectories


def test_a_searched_worker_returns_episodes_whose_transitions_carry_targets():
    """The whole point of sharding the searched path: the corpus must still be
    distillable, which means every transition needs its `Target` to survive the
    pipe as well as the search."""
    from hexn.expert import Target

    (kind, payload), = served(
        spec(0, 1, width=16, simulations=8, wave=4, action_cap=120), ("cohort", 1)
    )
    assert kind == "episodes", payload
    episodes = payload.episodes()
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
    episodes = Collector(policy, lanes=4, seed=3, action_cap=150, max_offers=0).collect(3)

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
    league = spec(0, 1, learners=2)
    policies, _ = _build(league)  # the same construction the worker runs
    assert len(policies) == 2
    states = [policy.net.state_dict() for policy in policies]

    loaded, (kind, payload) = served(league, ("weights", states), ("cohort", 2))

    assert loaded == ("ok", None)
    assert kind == "episodes", payload
    episodes = payload.episodes()
    assert len(episodes) == 2
    for episode in episodes:
        assert set(episode.cast) == {0, 1}
        for seat, trajectory in enumerate(episode.trajectories):
            assert trajectory, "both ids are learners; every seat records"
            assert all(t.seat == seat for t in trajectory)

    # One dict for a two-learner table is refused, not loaded into learner 0.
    (kind, message), = served(league, ("weights", states[0]))
    assert kind == "error" and "1 weight dicts for 2 learners" in message


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


def test_a_paired_worker_wraps_its_caster_and_deals_paired_boards():
    from hexn.collect import league_caster

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
    # carries `max_offers` per entrant.
    overridden = mix_opponents(
        [(f"{spec}@0", 1.0)], seed=1, lanes=2, board=board, players=4
    )[0]
    assert not isinstance(overridden, NetworkPolicy)

    # And with no board at all (a caller that predates this routing), the
    # bare entry still falls back to `named_opponent` rather than raising.
    no_board = mix_opponents([(spec, 1.0)], seed=1, lanes=2)[0]
    assert not isinstance(no_board, NetworkPolicy)


def test_a_sampled_network_member_samples_and_a_bare_one_stays_greedy(tmp_path):
    """`sampled:network:<path>` is the same batched `frozen` policy as the
    bare entry, sampling from its own seeded stream instead of taking the
    argmax; the bare entry is unchanged."""
    from hexn.collect import check_mix, mix_opponents
    from hexn.policy import NetworkPolicy

    path = tmp_path / "latest.pt"
    board = _a_checkpoint(path)
    bare, hot = f"network:{path}", f"sampled:network:{path}"
    mix = [(f"table(self|{bare}|{hot})", 1.0)]
    check_mix(mix, have_parent=False)

    greedy, sampled = mix_opponents(mix, seed=1, lanes=2, board=board, players=4, torch_seed=9)
    assert isinstance(greedy, NetworkPolicy) and greedy.greedy
    assert isinstance(sampled, NetworkPolicy) and not sampled.greedy

    # Near-uniform rows: the argmax never moves, the sampled draws do, and a
    # second build from the same worker seed draws the same stream.
    rows = torch.log_softmax(torch.zeros(64, 32) + 0.01 * torch.arange(32.0), dim=-1)
    assert greedy._sample(rows).unique().numel() == 1
    draws = sampled._sample(rows)
    assert draws.unique().numel() > 1
    again = mix_opponents(mix, seed=1, lanes=2, board=board, players=4, torch_seed=9)[1]
    assert torch.equal(again._sample(rows), draws)
    other = mix_opponents(mix, seed=1, lanes=2, board=board, players=4, torch_seed=10)[1]
    assert not torch.equal(other._sample(rows), draws)


def test_a_sampled_member_casts_exactly_as_its_bare_spelling(tmp_path):
    """Sampling draws nothing from the cast stream, so swapping a bare
    `network:` member for its `sampled:` spelling deals the same tables."""
    from hexn.collect import mix_table, parse_mix

    bare = f"table(self|self|network:{tmp_path}/a.pt|random)=1.0"
    hot = bare.replace("network:", "sampled:network:")

    law, hot_law = mix_table(parse_mix(bare), 4, 3), mix_table(parse_mix(hot), 4, 3)
    assert [law(i) for i in range(200)] == [hot_law(i) for i in range(200)]


def test_sampled_takes_only_a_bare_network(tmp_path):
    from hexn.collect import check_mix, rung_opponent

    path = tmp_path / "latest.pt"
    board = _a_checkpoint(path)
    for name in ("sampled:heximax", f"sampled:network:{path}@0", "sampled:random"):
        with pytest.raises(SystemExit, match="bare network"):
            check_mix([(name, 1.0)], have_parent=False)
    with pytest.raises(SystemExit, match="not there"):
        check_mix([(f"sampled:network:{tmp_path}/missing.pt", 1.0)], have_parent=False)
    with pytest.raises(ValueError, match="bare network"):
        rung_opponent("sampled:heximax", seed=1, lanes=2, board=board, players=4, device="cpu")


def test_a_worker_seeds_its_sampled_members_from_its_own_torch_seed(tmp_path):
    """Workers share the board seed; a sampled member's stream comes from
    the worker's `torch_seed`, so two workers do not sample in lockstep."""
    path = tmp_path / "latest.pt"
    _a_checkpoint(path)
    mix = ((f"table(self|sampled:network:{path})", 1.0),)
    rows = torch.log_softmax(torch.zeros(64, 32), dim=-1)
    first, second = (
        _build(spec(worker, 2, mix=mix))[1].opponents[0]._sample(rows) for worker in (0, 1)
    )
    assert not torch.equal(first, second)


@needs("heximax")
def test_heximax_resolves_as_a_mix_opponent():
    """`"heximax"` is not one of HexSet's own presets: a runtime registers
    it, and a loaded runtime is what makes it resolvable in `--mix` from a
    collector process. Without it the run dies at launch with `unknown
    entrant`."""
    from hexset.board.board import random_base_board
    from hexn.collect import check_mix, mix_opponents

    check_mix([("heximax", 0.25)], have_parent=False)
    board = random_base_board(random.Random(3))
    bot = mix_opponents([("heximax", 0.25)], seed=1, lanes=2)[0].spawn(board)
    assert hasattr(bot, "gains_many")
    # And it is the entrant, not a lookalike: the bot the arena itself
    # spawns for the name, search and bargaining alike. `--max-offers` is the
    # budget of the networks the run trains; a named entrant is literally the
    # entrant the arena scores, budget included.
    from hexset.arena import entrant_from_name, spawn

    arena = spawn(entrant_from_name("heximax"), board, random.Random(1))
    assert type(bot) is type(arena)
    assert (bot.depth, bot.width, bot.max_offers) == (arena.depth, arena.width, arena.max_offers)


@needs("heximax")
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
        action_cap=200,
        opponents=[RandomPolicy(random.Random(1))],
        caster=lambda index: mix_table(mix, 4, 11)(index)[0],
        temperatures=mix_temperatures(mix, 4, 11),
    )
    episodes = collector.collect(4)
    temperatures = [t for _, t in Recording.seen]
    assert temperatures and all(0.6 <= t <= 1.4 for t in temperatures)
    assert any(t != 1.0 for t in temperatures) and any(t == 1.0 for t in temperatures)
    for episode in episodes:
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_a_duel_entry_casts_two_live_seats_at_every_relative_slot():
    """`duel(...)`: the learner and one drawn opponent on two live seats, the
    other two retired and carrying the opponent's id; the opponent sits one,
    two or three seats round from the learner; the law is pure in the index."""
    from hexn.collect import mix_caster, mix_deal, parse_mix
    from hexset.rules import DUEL_VARIANT_GAME

    mix = parse_mix("duel(self|parent|network:/x/a.pt)=1.0")
    cast, deal = mix_caster(mix, 4, 7), mix_deal(mix, 4, 7)
    slots, members = set(), set()
    for index in range(600):
        game_type, retired = deal(index)
        assert game_type is DUEL_VARIANT_GAME and len(retired) == 2
        seats = cast(index)
        live = [s for s in range(4) if s not in retired]
        learner = next(s for s in live if seats[s] == 0)
        partner = next(s for s in live if s != learner)
        slots.add((partner - learner) % 4)
        members.add(seats[partner])
        assert all(seats[s] == seats[partner] for s in retired)
        assert (cast(index), deal(index)) == (seats, (game_type, retired))
    assert slots == {1, 2, 3} and members == {0, 1, 2}


def test_mix_deal_is_absent_without_a_duel_and_standard_for_other_entries():
    from hexn.collect import mix_deal, parse_mix
    from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME

    assert mix_deal(parse_mix("table(self|parent)=1.0"), 4, 7) is None
    deal = mix_deal(parse_mix("table(self|parent)=0.5,duel(self|parent)=0.5"), 4, 7)
    kinds = {deal(index)[0] for index in range(200)}
    assert kinds == {STANDARD_GAME, DUEL_VARIANT_GAME}
    for index in range(200):
        game_type, retired = deal(index)
        assert (game_type is STANDARD_GAME) == (not retired)


def test_a_worker_plays_duel_variant_games_on_the_live_seats_only():
    """A duel entry reaches the worker's lanes: every game is dealt under the
    duel-variant rules with its two retired seats never moving, and only the
    learner's live seats are recorded."""
    from hexn.collect import mix_deal
    from hexset.rules import DUEL_VARIANT

    mix = (("duel(self|random)", 1.0),)
    collector = ParallelCollector([spec(0, 1, lanes=4, mix=mix)])
    try:
        episodes = collector.collect(12)
    finally:
        collector.close()

    deal = mix_deal(mix, 4, 5)
    for episode in episodes:
        _, retired = deal(episode.index)
        assert episode.record is not None and episode.record.rules == DUEL_VARIANT
        for seat in range(4):
            if seat in retired:
                assert not episode.trajectories[seat]
                assert episode.outcome.points[seat] == 0
            else:
                assert bool(episode.trajectories[seat]) == (episode.cast[seat] == 0)
        if episode.outcome.winner is not None:
            assert episode.outcome.winner not in retired


def test_a_wedged_worker_fails_the_collection_and_a_resume_plays_only_the_rest(
    tmp_path, monkeypatch
):
    """A worker that stops ticking fails the run instead of hanging it, a dead
    one too, and the iteration's games survive both: the resumed collection
    returns exactly the planned games, the kept ones as they were kept."""
    import json
    import os
    import signal
    import time

    from hexn import collect, durable

    monkeypatch.setattr(collect, "POLL_SECONDS", 0.2)
    partial = durable.Partial(tmp_path / "partial" / "iter-00000")
    heartbeat = tmp_path / "heartbeat.json"
    first = ParallelCollector([spec(0, 2), spec(1, 2)], heartbeat=heartbeat)
    try:
        # Built and answering first: a fresh worker's interpreter start and
        # torch import are not a stall, and are not what this is timing.
        first.games_started()
        first.stall_seconds = 2.0
        first.start_collect(8, partial)
        deadline = time.time() + 120
        while not partial.finished() and time.time() < deadline:
            time.sleep(0.05)
        assert partial.finished(), "no game was kept as it finished"
        stuck = first._processes[0]
        os.kill(stuck.pid, signal.SIGSTOP)
        with pytest.raises(RuntimeError, match="wedged"):
            first.finish_collect()
        assert json.loads(heartbeat.read_text())["workers"][0]["busy"]
        os.kill(stuck.pid, signal.SIGKILL)
        stuck.join(10)
        first._busy.add(0)
        with pytest.raises(RuntimeError, match="died"):
            first._check()
    finally:
        for process in first._processes:
            if process.is_alive():
                process.kill()

    plan = partial.plan()
    assert sorted(plan) == list(range(8))
    kept = {episode.index: episode for episode in partial.done()}
    base = durable.resume_base(0, partial.indices())
    resumed = ParallelCollector([spec(w, 2, first_game=base + w) for w in range(2)])
    try:
        episodes = resumed.collect(8, partial)
        assert sorted(e.index for e in episodes) == list(range(8))
        for episode in episodes:
            if episode.index in kept:
                assert episode.record == kept[episode.index].record
        assert resumed.games_started() >= base
    finally:
        resumed.close()
