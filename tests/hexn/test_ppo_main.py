# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import pathlib
import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import loop  # noqa: E402
from hexn.collect import parse_mix  # noqa: E402
from hexn.ppo import __main__ as ppo_main  # noqa: E402


TINY = [
    "--device",
    "cpu",
    "--width",
    "8",
    "--rounds",
    "1",
    # One lane, one game: `--collect-mode cohort` plays its whole cohort out,
    # so lanes above the cohort size would only sit empty.
    "--lanes",
    "1",
    "--games-per-iteration",
    "1",
    # Short, trade-free games: nothing here is about how a game is played, and
    # a width-8 net's gate pricing every candidate after every MAIN action was
    # most of what a run here cost.
    "--action-cap",
    "200",
    "--max-offers",
    "0",
    "--minibatch",
    "256",
    "--epochs",
    "1",
]


def run(directory, iterations, extra=()):
    """Freeze a run, then launch it -- `ppo_main.main` takes nothing else.

    Going through `hexn.run.freeze` rather than calling the parser directly is
    deliberate: it means every test below exercises the manifest path the real
    trainer uses, so a config that cannot round-trip through a freeze fails
    here rather than on the box.
    """
    from hexn.run import freeze

    argv = (
        TINY
        + ["--iterations", str(iterations), "--checkpoint-dir", str(directory)]
        + list(extra)
    )
    freeze(
        "ppo",
        directory.name,
        pathlib.Path(directory),
        argv,
        repo=pathlib.Path(directory),
        description="test run",
    )
    return ppo_main.main([str(directory)])


def test_numbered_checkpoints_are_kept_alongside_the_one_that_gets_overwritten(
    tmp_path,
):
    assert run(tmp_path, 4, ["--checkpoint-every", "1", "--keep-every", "2"]) == 0

    kept = sorted(p.name for p in tmp_path.glob("iter-*.pt"))
    assert kept == ["iter-00002.pt", "iter-00004.pt"]
    assert (tmp_path / "latest.pt").exists()
    # The point of keeping them is that they differ; identical copies of the
    # final weights would answer nothing about when training stopped helping.
    early = torch.load(tmp_path / "iter-00002.pt", weights_only=False)
    late = torch.load(tmp_path / "iter-00004.pt", weights_only=False)
    assert early["iteration"] == 2 and late["iteration"] == 4
    assert any(
        not torch.equal(early["net"][k], late["net"][k]) for k in early["net"]
    )


def test_a_resumed_run_carries_on_from_the_iteration_it_reached(tmp_path):
    run(tmp_path, 1, ["--checkpoint-every", "1"])
    first = torch.load(tmp_path / "latest.pt", weights_only=False)

    run(tmp_path, 3, ["--checkpoint-every", "1", "--resume"])
    second = torch.load(tmp_path / "latest.pt", weights_only=False)

    assert first["iteration"] == 1
    # What a resume needs from a checkpoint: the weights and the optimiser.
    assert any(v.numel() for v in first["net"].values())
    assert "state" in first["optimiser"]
    assert second["iteration"] == 3
    # It continued rather than restarted: the game counter only moves forward.
    assert second["games_started"] > first["games_started"]

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert lines[1] == {"resumed_from": 1}
    assert [record["iteration"] for record in lines if "iteration" in record] == [0, 1, 2]


def test_resuming_with_nothing_to_resume_is_an_error_rather_than_a_fresh_start(tmp_path):
    """It used to fall through and silently begin at iteration 0.

    On a 150-iteration GPU block that is hours spent discarding the campaign,
    with a log that starts at 0 and looks perfectly healthy. A typo'd
    `--checkpoint-dir` or an unseeded directory is all it takes. Same shape as
    the learning rate `--resume` used to throw away: a flag that quietly does
    nothing.
    """
    with pytest.raises(SystemExit):
        run(tmp_path / "empty", 1, ["--checkpoint-every", "1", "--resume"])


class Crash(Exception):
    """What a killed trainer looks like from inside it."""


def test_a_run_killed_mid_collection_resumes_without_losing_or_repeating_a_game(
    tmp_path, monkeypatch
):
    """Killed with iteration 1 part-collected: the resumed iteration trains on
    exactly the games it planned, the ones already finished among them, and
    no game index is ever trained on twice."""
    from hexn import durable
    from hexn.selfplay import Collector

    used: list[list[int]] = []
    real_assemble = ppo_main.assemble

    def assemble(episodes, *args, **kwargs):
        used.append(sorted(e.index for e in episodes))
        return real_assemble(episodes, *args, **kwargs)

    monkeypatch.setattr(ppo_main, "assemble", assemble)
    real_tick = Collector.tick
    interrupted = durable.partial(tmp_path / "partial", 1)

    def tick(self):
        out = real_tick(self)
        if 0 < len(interrupted.finished()) < 4:
            raise Crash
        return out

    four = ["--lanes", "2", "--games-per-iteration", "4"]
    monkeypatch.setattr(Collector, "tick", tick)
    with pytest.raises(Crash):
        run(tmp_path, 3, four)
    monkeypatch.setattr(Collector, "tick", real_tick)
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["iteration"] == 1
    plan = sorted(interrupted.plan())
    kept = interrupted.finished()

    assert run(tmp_path, 3, four + ["--resume"]) == 0

    assert len(used) == 3
    assert used[1] == plan, "the resumed iteration is the iteration it planned"
    assert set(kept) <= set(used[1])
    every = [index for indices in used for index in indices]
    assert len(every) == len(set(every)), "no game trained on twice"
    assert min(used[2]) > max(plan)
    assert not durable.partials(tmp_path / "partial")
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [line.get("iteration", "marker") for line in lines] == [0, "marker", 1, 2]


def test_a_run_killed_in_its_evaluation_keeps_the_iteration_it_finished(
    tmp_path, monkeypatch
):
    def dies(*args, **kwargs):
        raise Crash

    monkeypatch.setattr(ppo_main, "ladder", dies)
    with pytest.raises(Crash):
        run(tmp_path, 2, ["--eval-every", "1"])
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["iteration"] == 1
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [line["iteration"] for line in lines] == [0]
    assert "ladder" not in lines[0]

    monkeypatch.setattr(ppo_main, "ladder", lambda *args, **kwargs: {})
    assert run(tmp_path, 2, ["--eval-every", "1", "--resume"]) == 0
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert lines[1:] == [
        {"resumed_from": 1},
        lines[2],
        {"iteration": 1, "ladder": {}},
    ]
    assert lines[2]["iteration"] == 1 and "positions" in lines[2]


def test_a_fresh_start_over_a_run_is_refused_rather_than_overwriting_it(tmp_path):
    run(tmp_path, 1)
    with pytest.raises(SystemExit, match="already holds a run"):
        run(tmp_path, 2)


def test_a_failed_save_leaves_the_last_good_checkpoint_readable(tmp_path):
    # The whole point of writing to a temporary file and renaming: a crash
    # during `torch.save` must not take the previous checkpoint with it.
    path = tmp_path / "latest.pt"
    loop.save(path, {"iteration": 1, "marker": "good"})

    real = torch.save

    def explode(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("crashed mid-save")

    torch.save = explode
    try:
        with pytest.raises(RuntimeError):
            loop.save(path, {"iteration": 2, "marker": "bad"})
    finally:
        torch.save = real

    survived = torch.load(path, weights_only=False)
    assert survived["marker"] == "good"
    assert survived["iteration"] == 1


@pytest.mark.parametrize("dump", [True, False])
def test_the_brake_keeps_the_batch_only_when_asked(tmp_path, dump):
    # A brake that fires on any movement at all: several minibatches an epoch,
    # so the first epoch's mean KL is read after real steps.
    extra = ["--epochs", "2", "--minibatch", "16", "--learning-rate", "1e-2", "--kl-break", "1e-12"]
    assert run(tmp_path, 1, extra + ([] if dump else ["--no-dump-blowout-batch"])) == 0

    (record,) = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert record["epochs_taken"] == 1
    assert (tmp_path / "blowout-00001-pre.pt").exists()
    assert (tmp_path / "blowout-00001-batch.pt").exists() == dump


def test_a_micro_batched_run_trains_and_logs(tmp_path):
    assert run(tmp_path, 1, ["--minibatch", "64", "--micro-batch", "16"]) == 0
    (record,) = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert record["positions"] > 64 and record["minibatch"] == 64


def test_every_trainer_can_turn_the_batch_dump_off():
    from hexn.league import build_parser as league_parser

    for build, base in (
        (ppo_main.build_parser, []),
        (league_parser, ["--checkpoint-dir", "x", "--learner", ""]),
    ):
        parser = build()
        assert parser.parse_args(base).dump_blowout_batch is True
        assert parser.parse_args(base + ["--no-dump-blowout-batch"]).dump_blowout_batch is False


def test_parse_mix_rejects_overcommitted_or_empty_shares():
    assert parse_mix("") == []
    assert parse_mix("heximax=0.25,parent=0.25") == [
        ("heximax", 0.25),
        ("parent", 0.25),
    ]
    with pytest.raises(ValueError):
        parse_mix("heximax=0.7,parent=0.7")
    with pytest.raises(ValueError):
        parse_mix("heximax=0")


def test_async_collection_prefetches_each_batch_once(tmp_path):
    assert (
        run(
            tmp_path,
            2,
            [
                "--checkpoint-every",
                "1",
                "--collect-workers",
                "2",
                # One lane per worker: a tick then finishes at most one game,
                # so the exact cohort counts below cannot be blurred by two
                # capped lanes ending on the same tick.
                "--lanes",
                "2",
                # Prefetching *is* the off-policy path, so it has to say so.
                "--collect-mode",
                "stream",
                "--async-collect",
            ],
        )
        == 0
    )

    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 2
    lines = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert [line["iteration"] for line in lines] == [0, 1]
    # One game per iteration, with no unused final prefetch.
    assert lines[-1]["games"] == 2
    assert all(line["positions"] > 0 for line in lines)


def test_prefetching_has_to_opt_out_of_the_on_policy_guarantee(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, 1, ["--collect-workers", "2", "--async-collect"])


class Streamless:
    """A policy with no random stream, so a self-duel has an exact answer.

    `RandomPolicy` cannot serve here: the antithetic pair reuses one policy
    object across both cohorts, so a mutable stream carries its position from
    the first into the second and the two halves stop playing the same game --
    which is the one thing the pairing needs. A checkpoint played at argmax has
    this property for real; this stub has it without torch.

    It is a `hexset.bench.versus.BatchPolicy` and nothing more: `versus` hands
    a policy that answers neither `act_rows` nor a bot bench straight to the
    harness, so this is also the shape a stub has to have.
    """

    def act(self, requests):
        from hexn.selfplay import Choice

        return [Choice(action=request.options[0]) for request in requests]


def test_an_antithetic_self_duel_is_exactly_zero():
    """The test that catches the dice-keying trap.

    One agent on both sides has no edge, so a duel of a checkpoint against
    itself must read exactly 0.00 paired VP -- with identical weights and no
    RNG in the path, nothing else can explain a nonzero reading. Without the
    antithetic pairing, `alternating` keys the cast to the game index and
    `hexset.arena.deal_game` keys the board to the index too, so a cohort
    samples each side on only one seat-pair per board and never the other:
    a board-dependent seat effect leaks into the margin and looks like real
    signal rather than the sampling artifact it is.
    """
    stub = Streamless()
    result = loop.versus(
        stub, stub, games=8, lanes=4, players=4, seed=21, max_offers=3
    )

    assert result["antithetic"] and result["boards"] == 4 and result["games"] == 8
    # Exact, not approximate: every board's two readings are one game's margin
    # and its negation, so the average is 0.0 in floating point too.
    assert result["paired_vp"] == 0.0
    assert result["paired_vp_low"] == result["paired_vp_high"] == 0.0


def test_versus_is_the_engines_own_verdict_and_nothing_added():
    """Two scripted rungs, and `versus` reports exactly what HexSet reports.

    The whole claim of the refactor: `hexn.loop.versus` is a lineup, a cast
    and a cohort size handed to `hexset.bench.versus.compete_batched`, so at
    one seed it cannot read differently from the same call made directly. A
    second derivation of the pairing, the interval or the margin would show up
    here as a number that no longer matches.
    """
    from hexset.bench.versus import BotPolicy, compete_batched
    from hexset.casting import alternating

    from hexn import runtime
    from hexn.collect import named_opponent

    runtime.load(["_scripted_runtime"])
    # Two cheap rungs: the claim is about the
    # bookkeeping around the duel, not about who wins it.
    mine = loop.versus(
        named_opponent("random-too", 3, 2),
        named_opponent("random", 4, 2),
        games=2,
        lanes=2,
        players=4,
        seed=17,
        max_offers=1,
    )
    theirs = compete_batched(
        {
            0: BotPolicy(named_opponent("random-too", 3, 2).spawn),
            1: BotPolicy(named_opponent("random", 4, 2).spawn),
        },
        2,
        caster=alternating(4),
        players=4,
        seed=17,
        lanes=2,
        action_cap=4000,
        antithetic=True,
        learner=0,
    ).metrics()

    assert {"wins", "boards", "paired_vp"} <= set(mine)
    # `seconds` is each call's own wall clock, not a derived reading -- two
    # separate runs of the same duel take different amounts of time by
    # construction, so it is the one key this equality cannot cover.
    assert mine.keys() == theirs.keys()
    assert {k: v for k, v in mine.items() if k != "seconds"} == {
        k: v for k, v in theirs.items() if k != "seconds"
    }


def test_duellist_seats_a_networks_trade_gate_before_use():
    """`duellist` wraps a network in HexSet's own `hexset.bench.versus.
    PolicyPolicy`, whose `.gate` builds the checkpoint's `NetworkBot`
    (`hexset.clients.netbot.bot_for`) but does not seat it at the game it was
    asked about. Left unseated, `NetworkBot._seated` stays `None` for the
    whole game and `gains_many`/`estimate_many` price every candidate at
    -1.0 -- the seat is asked about trades and never clears one, which is
    exactly what an unseated gate looks like from outside: a served
    checkpoint that answers policy and value correctly but silently never
    trades. Seating it is the contract `NetworkBot.seat_at`'s own docstring
    describes for exactly this caller -- a driver that installs a bot as a
    gate only, rather than asking it to move. `loop.duellist` seats it
    itself; this is the regression an unpatched `PolicyPolicy` would pass
    without a single test noticing, since nothing else asserts on a duel's
    trade count.
    """
    from hexset.actions import space_for
    from hexset.arena import deal_game
    from hexset.board.board import random_base_board

    class Stub:
        """A `hexset.clients.policy.Policy` with no net behind it at all --
        `duellist` only reads `act_rows`/`space`/`players`, never a weight."""

        def __init__(self, space) -> None:
            self.space = space
            self.players = 4

        def act_rows(self, rows):
            return [options[0] for (_, _, options) in rows]

        def value_rows(self, rows):
            return [(0.25, 0.25, 0.25, 0.25) for _ in rows]

        def score_rows(self, rows):
            return [
                ([1.0 / len(options)] * len(options), (0.25, 0.25, 0.25, 0.25))
                for _, _, options in rows
            ]

    board = random_base_board(random.Random(0))
    game = deal_game(0, 0, 4, board=board)
    stub = Stub(space_for(game))

    gate = loop.duellist(stub).gate(game, 0)
    assert gate is not None
    assert gate._seated is game


def test_the_same_self_duel_reads_off_zero_without_the_pairing():
    """The control, so the guarantee above cannot pass for a trivial reason."""
    stub = Streamless()
    result = loop.versus(
        stub, stub, games=8, lanes=4, players=4, seed=21, max_offers=3,
        antithetic=False,
    )

    # The verdict stamps the key either way now (`Verdict.metrics`), which is
    # how an old reading is told apart from a new one in a recorded verdict.
    assert result["antithetic"] is False
    assert result["games"] == 8
    assert result["paired_vp"] != 0.0


def test_self_in_a_table_pool_is_the_learner_on_every_seat_it_draws():
    from collections import Counter

    from hexn.collect import mix_caster, mix_names

    mix = parse_mix("table(self|self|parent|network:/x/a.pt)=1.0")
    # `self` is id 0, never an opponent id: the id law below is unchanged.
    assert mix_names(mix) == ["parent", "network:/x/a.pt"]
    caster = mix_caster(mix, 4, seed=7)
    casts = [caster(index) for index in range(4000)]
    learner_seats = Counter(cast.count(0) for cast in casts)
    # Never a game without the learner, and one to four learner seats a game:
    # the drawn seat plus a coin per other seat at this pool's weighting.
    assert set(learner_seats) == {1, 2, 3, 4}
    assert 2.3 < sum(cast.count(0) for cast in casts) / len(casts) < 2.7
    assert all(set(cast) <= {0, 1, 2} for cast in casts)
    # Pure in the index, as every cast law here must be.
    assert casts == [mix_caster(mix, 4, seed=7)(index) for index in range(4000)]


def test_a_self_band_tempers_only_the_pool_drawn_self_seats():
    from hexn.collect import mix_caster, mix_table, mix_temperatures, tempered

    plain = parse_mix("table(self|self|parent|network:/x/a.pt)=1.0")
    banded = parse_mix("table(self~0.3|self~0.3|parent|network:/x/a.pt)=1.0")
    assert not tempered(plain) and tempered(banded)
    assert mix_temperatures(plain, 4, 7) is None
    # The band draws after every seat is cast, so the cast law is untouched.
    assert [mix_caster(plain, 4, 7)(i) for i in range(3000)] == [
        mix_caster(banded, 4, 7)(i) for i in range(3000)
    ]
    table = mix_table(banded, 4, 7)
    saw_tempered = False
    for index in range(3000):
        cast, temperatures = table(index)
        for pid, temperature in zip(cast, temperatures):
            if pid != 0:
                assert temperature == 1.0
            else:
                assert 0.7 <= temperature <= 1.3
                saw_tempered |= temperature != 1.0
        # The seat the table draw seated first is the anchor at exactly 1.0.
        assert any(pid == 0 and t == 1.0 for pid, t in zip(cast, temperatures))
        assert table(index) == (cast, temperatures)
    assert saw_tempered
    law = mix_temperatures(banded, 4, 7, pair_boards=True)
    assert law(10) == law(11) == table(5)[1]


def test_a_self_band_is_validated_where_it_is_written():
    with pytest.raises(ValueError, match="table pool member"):
        parse_mix("self~0.2=0.5")
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        parse_mix("table(self~1.5|parent)=1.0")
    with pytest.raises(ValueError, match="must be a number"):
        parse_mix("table(self~hot|parent)=1.0")


# ---------------------------------------------------------------------------
# Warm starts: --init, --prior-kl, --lag-rung.
# ---------------------------------------------------------------------------


# Nothing below is about trading, and a tiny random net with no offer budget
# bargains every turn -- minutes a game. The runs below go through
# `warm_run`, which takes `run`'s arguments and turns the network's offers off.
NO_OFFERS = ["--max-offers", "0"]


def warm_run(directory, iterations, extra=()):
    return run(directory, iterations, list(extra) + NO_OFFERS)


def _weights(path):
    return torch.load(path, weights_only=False)["net"]


def test_init_starts_the_run_from_the_given_weights_at_iteration_zero(tmp_path):
    source = tmp_path / "source"
    assert warm_run(source, 1, ["--checkpoint-every", "1", "--seed", "5"]) == 0

    # A zero rate makes the update a no-op, so what the warm-started run saves
    # is exactly what it started from.
    warm = tmp_path / "warm"
    assert warm_run(
        warm,
        1,
        ["--checkpoint-every", "1", "--learning-rate", "0", "--init",
         str(source / "latest.pt")],
    ) == 0
    cold = tmp_path / "cold"
    assert warm_run(cold, 1, ["--checkpoint-every", "1", "--learning-rate", "0"]) == 0

    seeded, fresh, given = (
        _weights(warm / "latest.pt"),
        _weights(cold / "latest.pt"),
        _weights(source / "latest.pt"),
    )
    assert all(torch.equal(seeded[k], given[k]) for k in given)
    # Anti-vacuity: the same recipe without --init is a different net.
    assert any(not torch.equal(fresh[k], given[k]) for k in given)
    # Weights only: the counter is this run's own.
    state = torch.load(warm / "latest.pt", weights_only=False)
    assert state["iteration"] == 1
    # The starting point is kept as the run's own iteration 0, recording the
    # run's arguments -- its offer budget -- rather than the source's.
    start = torch.load(warm / "iter-00000.pt", weights_only=False)
    assert start["iteration"] == 0
    assert all(torch.equal(start["net"][k], given[k]) for k in given)
    assert start["args"]["max_offers"] == 0


def test_init_with_resume_initialises_once_and_then_resumes(tmp_path):
    source = tmp_path / "source"
    warm_run(source, 1, ["--checkpoint-every", "1", "--seed", "5"])
    warm = tmp_path / "warm"
    flags = ["--checkpoint-every", "1", "--init", str(source / "latest.pt"), "--resume"]

    # The same frozen config serves the launch (nothing to resume: init) ...
    assert warm_run(warm, 1, flags) == 0
    # ... and the restart, which carries on rather than re-initialising.
    assert warm_run(warm, 2, flags) == 0
    lines = [json.loads(l) for l in (warm / "log.jsonl").read_text().splitlines()]
    assert [record["iteration"] for record in lines if "iteration" in record] == [0, 1]


def test_init_refuses_a_checkpoint_of_another_shape(tmp_path):
    source = tmp_path / "source"
    warm_run(source, 1, ["--checkpoint-every", "1"])
    with pytest.raises(RuntimeError):
        # TINY is one round; the source has one round and this run asks for two.
        warm_run(
            tmp_path / "warm",
            1,
            ["--rounds", "2", "--init", str(source / "latest.pt")],
        )


def test_a_prior_weight_needs_a_parent_to_diverge_from(tmp_path):
    with pytest.raises(SystemExit):
        warm_run(tmp_path / "run", 1, ["--prior-kl", "0.1"])


def test_a_prior_weighted_run_logs_its_divergence_from_the_parent(tmp_path):
    parent = tmp_path / "parent"
    warm_run(parent, 1, ["--checkpoint-every", "1", "--seed", "9"])
    child = tmp_path / "child"
    assert warm_run(
        child,
        1,
        ["--parent", str(parent / "latest.pt"), "--prior-kl", "0.1"],
    ) == 0
    (record,) = [json.loads(l) for l in (child / "log.jsonl").read_text().splitlines()]
    # A different random net from the parent, so the divergence is positive.
    assert record["prior_kl"] > 0


def test_the_lag_rung_reads_the_init_weights_at_first_and_kept_ones_after(tmp_path):
    source = tmp_path / "source"
    warm_run(source, 1, ["--checkpoint-every", "1", "--seed", "5"])
    warm = tmp_path / "warm"
    assert warm_run(
        warm,
        2,
        [
            "--checkpoint-every", "1", "--keep-every", "1",
            "--eval-every", "1", "--eval-games", "2", "--lag-rung", "1",
            "--init", str(source / "latest.pt"),
        ],
    ) == 0
    lines = [json.loads(l) for l in (warm / "log.jsonl").read_text().splitlines()]
    evaluations = [record for record in lines if "ladder" in record]
    # Iteration 1's lag is iteration 0, the init weights; iteration 2's is the
    # kept iter-00001.
    assert len(evaluations) == 2
    assert all("lag" in record["ladder"] for record in evaluations)
    assert not any("lag_checkpoint_missing" in record for record in lines)


def test_a_lag_with_nothing_kept_is_logged_as_a_miss(tmp_path):
    directory = tmp_path / "run"
    assert warm_run(
        directory,
        1,
        ["--eval-every", "1", "--eval-games", "2", "--lag-rung", "5"],
    ) == 0
    lines = [json.loads(l) for l in (directory / "log.jsonl").read_text().splitlines()]
    (evaluation,) = [record for record in lines if "ladder" in record]
    assert evaluation["lag_checkpoint_missing"] == -4
    assert "lag" not in evaluation["ladder"]


def test_a_fast_win_run_grafts_its_head_onto_an_init_and_logs_its_pace(tmp_path):
    """`--fast-win` from a checkpoint without the head: every other weight
    comes over unchanged, the head is grafted, and the log carries the speed
    columns next to the fast-win loss."""
    import json

    source = tmp_path / "source"
    assert warm_run(source, 1, ["--checkpoint-every", "1", "--seed", "5"]) == 0
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"weights": [1.0] * 12 + [0.9, 0.8, 0.6, 0.4]}))
    fast = tmp_path / "fast"
    assert warm_run(
        fast,
        1,
        ["--checkpoint-every", "1", "--learning-rate", "0", "--fast-win", str(curve),
         "--init", str(source / "latest.pt")],
    ) == 0
    given, start = _weights(source / "latest.pt"), _weights(fast / "iter-00000.pt")
    assert all(torch.equal(start[k], given[k]) for k in given)
    assert set(start) - set(given) == {"fast_win.weight", "fast_win.bias"}
    row = [json.loads(line) for line in (fast / "log.jsonl").read_text().splitlines() if line.strip()][-1]
    assert row["fast_loss"] > 0.0
    assert {"win_round_median", "learner_win_round_median", "fast_payoff_mean"} <= set(row)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_run_killed_mid_update_resumes_from_its_last_step_to_the_uninterrupted_run(
    tmp_path, monkeypatch, workers
):
    """Killed after the fourth optimiser step of iteration 1's update: the
    resumed run reads iteration 1's games back from its partial, continues
    the update from the step file, and ends -- two iterations later -- on the
    uninterrupted run's weights, to the bit. The games stay on disk until the
    iteration checkpoint, and the step file goes once it is written."""
    from hexn import durable, steps

    shape = ["--lanes", "4", "--games-per-iteration", "4", "--minibatch", "48", "--epochs", "2"]
    # Two collection workers return their games worker by worker, not in game
    # order, and a resumed collection returns its kept games first.
    shape += ["--collect-workers", str(workers), "--keep-every", "1"]
    straight = tmp_path / "straight"
    assert run(straight, 3, shape) == 0

    killed = tmp_path / "killed"
    real_update = ppo_main.update
    calls = []

    def update(policy, optimiser, batch, config, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            real_step = optimiser.step
            taken = []

            def step(*args, **kw):
                if len(taken) == 4:
                    raise Crash
                taken.append(1)
                return real_step(*args, **kw)

            optimiser.step = step
        return real_update(policy, optimiser, batch, config, **kwargs)

    monkeypatch.setattr(ppo_main, "update", update)
    with pytest.raises(Crash):
        run(killed, 3, shape)
    monkeypatch.setattr(ppo_main, "update", real_update)
    assert torch.load(killed / "latest.pt", weights_only=False)["iteration"] == 1
    held = torch.load(killed / steps.STEP, weights_only=False)
    assert (held["iteration"], held["progress"]["step"]) == (1, 4)
    games = durable.partial(killed / "partial", 1).finished()
    assert len(games) == 4, "the interrupted iteration's games are still on disk"

    kept = []
    real_keep = durable.Partial.keep
    monkeypatch.setattr(
        durable.Partial, "keep", lambda self, e: kept.append(self.directory.name) or real_keep(self, e)
    )
    assert run(killed, 3, shape + ["--resume"]) == 0
    if not workers:  # a worker process keeps its games past this patch
        assert set(kept) == {"iter-00002"}, "iteration 1 played none of its games again"

    # The interrupted iteration's end, and -- with the in-process collector,
    # whose games are a function of the restored RNG -- the run's. A worker
    # restarted by the resume draws its own stream afresh, so iteration 2's
    # games are another draw there, as on any resume.
    for name in ("iter-00002.pt", "latest.pt")[: 1 if workers else 2]:
        final = torch.load(killed / name, weights_only=False)
        reference = torch.load(straight / name, weights_only=False)
        assert all(torch.equal(final["net"][k], reference["net"][k]) for k in reference["net"]), name
    assert not (killed / steps.STEP).exists()
    assert not durable.partials(killed / "partial")
    rows = [row for row in durable.read_lines(killed / "log.jsonl") if "positions" in row]
    assert [row["iteration"] for row in rows] == [0, 1, 2]
    assert rows[1]["resumed_at_step"] == 4 and rows[2]["resumed_at_step"] is None
    assert rows[1]["step_checkpoints"] > 0 and rows[1]["step_checkpoint_seconds"] > 0
    reference_rows = [row for row in durable.read_lines(straight / "log.jsonl") if "positions" in row]
    for name in ("policy_loss", "value_loss", "approx_kl", "grad_norm", "explained_variance"):
        assert rows[1][name] == reference_rows[1][name], name


def test_the_recent_ring_keeps_the_last_n_whatever_keep_every_keeps(tmp_path):
    assert run(tmp_path, 4, ["--keep-every", "0", "--keep-recent", "2"]) == 0
    assert sorted(p.name for p in tmp_path.glob("recent-*.pt")) == [
        "recent-00003.pt",
        "recent-00004.pt",
    ]
    assert not list(tmp_path.glob("iter-*.pt"))


def test_a_run_killed_in_its_first_update_resumes_from_its_last_step_not_its_init(
    tmp_path, monkeypatch
):
    """The launch shape every frozen run uses (`--init` with `--resume`),
    killed in iteration 0's update before there is any `latest.pt`: the
    resumed run starts from `--init` again, reads iteration 0's games back and
    carries the update on from its step file, not from the initial weights."""
    from hexn import steps

    assert run(tmp_path / "seed", 1) == 0
    shape = ["--lanes", "2", "--games-per-iteration", "2", "--minibatch", "48",
             "--init", str(tmp_path / "seed" / "latest.pt"), "--resume"]
    straight = tmp_path / "straight"
    assert run(straight, 1, shape) == 0

    killed = tmp_path / "killed"
    real_update = ppo_main.update

    def update(policy, optimiser, batch, config, **kwargs):
        real_step = optimiser.step
        taken = []

        def step(*args, **kw):
            if len(taken) == 2:
                raise Crash
            taken.append(1)
            return real_step(*args, **kw)

        optimiser.step = step
        return real_update(policy, optimiser, batch, config, **kwargs)

    monkeypatch.setattr(ppo_main, "update", update)
    with pytest.raises(Crash):
        run(killed, 1, shape)
    monkeypatch.setattr(ppo_main, "update", real_update)
    assert not (killed / "latest.pt").exists()
    assert torch.load(killed / steps.STEP, weights_only=False)["progress"]["step"] == 2

    assert run(killed, 1, shape) == 0
    final = torch.load(killed / "latest.pt", weights_only=False)
    reference = torch.load(straight / "latest.pt", weights_only=False)
    assert all(torch.equal(final["net"][k], reference["net"][k]) for k in reference["net"])
    row = [r for r in loop_rows(killed) if "positions" in r][-1]
    assert row["resumed_at_step"] == 2


def loop_rows(directory):
    from hexn import durable

    return durable.read_lines(directory / "log.jsonl")


def test_an_fp16_run_on_a_device_batch_trains_resumes_and_keeps_its_scale(tmp_path):
    parent = tmp_path / "parent"
    warm_run(parent, 1, ["--checkpoint-every", "1", "--seed", "9"])
    child = tmp_path / "child"
    flags = [
        "--parent", str(parent / "latest.pt"), "--prior-kl", "0.1",
        "--amp", "fp16", "--batch-on-device", "--fused", "--checkpoint-every", "1",
    ]
    assert warm_run(child, 1, flags) == 0
    first = torch.load(child / "latest.pt", weights_only=False)
    assert first["scaler"]["scale"] > 0
    assert warm_run(child, 2, flags + ["--resume"]) == 0
    records = [
        json.loads(l) for l in (child / "log.jsonl").read_text().splitlines()
        if '"iteration"' in l
    ]
    assert [r["iteration"] for r in records] == [0, 1]
    assert all(r["amp_scale"] > 0 and r["prior_kl"] > 0 for r in records)
