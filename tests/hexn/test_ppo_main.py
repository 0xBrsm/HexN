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
from hexn.selfplay import Collector, RandomPolicy  # noqa: E402


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
    "--action-cap",
    "600",
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


def test_a_run_writes_a_checkpoint_carrying_the_weights_and_the_game_counter(tmp_path):
    assert run(tmp_path, 1, ["--checkpoint-every", "1"]) == 0

    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 1
    assert state["games_started"] > 0
    assert state["net"], "no weights in the checkpoint"
    assert "state" in state["optimiser"]
    # Anti-vacuity: a checkpoint that saved nothing would still have the keys.
    assert any(v.numel() for v in state["net"].values())


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
    assert second["iteration"] == 3
    # It continued rather than restarted: the game counter only moves forward.
    assert second["games_started"] > first["games_started"]

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [record["iteration"] for record in lines] == [0, 1, 2]


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


def test_a_resumed_run_plays_new_games_rather_than_the_ones_it_learned_from(tmp_path):
    # A game is a pure function of the seed and its index, so restarting the
    # counter would replay the training set — and it would look like it worked.
    first = Collector(RandomPolicy(random.Random(0)), lanes=4, seed=3, action_cap=600)
    first.collect(2)
    reached = first.games_started()
    assert reached > 0

    resumed = Collector(
        RandomPolicy(random.Random(0)),
        lanes=4,
        seed=3,
        action_cap=600,
        first_game=reached,
    )
    played = {e.index for e in resumed.collect(2)}
    assert played, "the resumed collector finished nothing"
    assert not (played & set(range(reached))), f"replayed {played & set(range(reached))}"
    assert min(played) >= reached


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


def test_a_save_never_leaves_a_partial_file_where_the_checkpoint_belongs(tmp_path):
    path = tmp_path / "latest.pt"
    loop.save(path, {"iteration": 7})
    assert path.exists()
    assert not list(tmp_path.glob("*.partial"))


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


def _rows(directory):
    return [
        json.loads(line)
        for line in (directory / "log.jsonl").read_text().splitlines()
    ]


def test_streaming_collection_ignores_the_batch_size_it_was_asked_for(tmp_path):
    """The contrast, which is what makes the on-policy guarantee worth having.

    `--games-per-iteration` reads like a batch size and under `stream` it is
    not one: `collect` stops on the tick that finishes the *first* game, and
    every other lane that ended on the same tick comes along. Asking for one
    game on four lanes trains on four. A cohort delivers what it was asked for.

    The staleness half of the same defect is pinned in `test_selfplay`, where
    the spread of game lengths is the point.

    The assertion is on the run's total, not per iteration, and not a multiple
    of the lane count. It was the multiple until repeated offers stopped being
    enumerated: every game used to run past `TINY`'s 600-action cap and so all
    four lanes truncated on the same tick. Games are shorter again under
    contract 5 -- no offer actions at all -- so a lane can now finish alone and
    an individual iteration can come back with one game. Four-at-a-time, and
    then more-than-one-every-time, were artefacts of the cap; the claim is that
    the stream does not deliver the batch it was asked for.
    """
    assert run(tmp_path, 2, ["--collect-mode", "stream", "--lanes", "4"]) == 0
    streamed = [row for row in _rows(tmp_path) if "positions" in row]
    assert all(row["collect_mode"] == "stream" for row in streamed)
    asked = len(streamed)  # one game an iteration
    assert sum(row["games"] for row in streamed) > asked, "one game was asked for"

    fresh = tmp_path / "cohort"
    assert run(fresh, 2, ["--lanes", "1"]) == 0
    cohort = [row for row in _rows(fresh) if "positions" in row]
    assert [row["games"] for row in cohort] == [1, 2]
    # Totalled over the run, not paired per iteration: one lane finishing a
    # short game can put a single streamed iteration below a truncated cohort
    # one without the claim being wrong.
    assert sum(row["positions"] for row in cohort) < sum(
        row["positions"] for row in streamed
    ), "the cohort trained on no less data than the four-lane stream"


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
        stub, stub, games=8, lanes=4, players=4, seed=21, max_trades=3
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

    from hexn.collect import named_opponent

    mine = loop.versus(
        named_opponent("heximax", 3, 2),
        named_opponent("random", 4, 2),
        games=4,
        lanes=2,
        players=4,
        seed=17,
        max_trades=1,
    )
    theirs = compete_batched(
        {
            0: BotPolicy(named_opponent("heximax", 3, 2).spawn),
            1: BotPolicy(named_opponent("random", 4, 2).spawn),
        },
        4,
        caster=alternating(4),
        players=4,
        seed=17,
        lanes=2,
        action_cap=4000,
        max_trades=1,
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

    gate = loop.duellist(stub).gate(game, 0, None)
    assert gate is not None
    assert gate._seated is game


def test_the_same_self_duel_reads_off_zero_without_the_pairing():
    """The control, so the guarantee above cannot pass for a trivial reason."""
    stub = Streamless()
    result = loop.versus(
        stub, stub, games=8, lanes=4, players=4, seed=21, max_trades=3,
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


def test_self_is_refused_as_a_plain_mix_entry():
    with pytest.raises(ValueError, match="table pool member"):
        parse_mix("self=0.5")
    with pytest.raises(ValueError, match="table pool member"):
        parse_mix("heximax=0.1,self=0.2")


def test_a_pool_without_self_casts_exactly_as_before():
    from hexn.collect import mix_caster

    mix = parse_mix("table(parent|network:/x/a.pt)=0.5")
    caster = mix_caster(mix, 4, seed=3)
    casts = [caster(index) for index in range(2000)]
    assert all(cast.count(0) in (1, 4) for cast in casts)


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
