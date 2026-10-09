# SPDX-License-Identifier: GPL-3.0-only
"""A batch keeps every unit as it finishes (`hexn.durable`), and a batch
killed part-way resumes to the units it would have had: none lost, none
played twice."""
from __future__ import annotations

import ast
import json
import os
import random
from pathlib import Path

import pytest

import hexn
from hexn import durable
from hexn.selfplay import Choice, Collector, RandomPolicy

PACKAGE = Path(hexn.__file__).resolve().parent


class Crash(Exception):
    """What a killed process looks like from inside the batch."""


class First:
    """Always the first legal option: no stream of its own, so a game is a pure
    function of `(seed, index)` and a replayed game must equal the original."""

    def act(self, requests):
        return [Choice(action=request.options[0]) for request in requests]


class Dies:
    """`policy`, until its `after`-th tick, where the process "dies"."""

    def __init__(self, policy, after: int) -> None:
        self.policy = policy
        self.left = after

    def act(self, requests):
        self.left -= 1
        if self.left < 0:
            raise Crash
        return self.policy.act(requests)


def fingerprint(episodes):
    """What makes two episodes the same game (see `test_store.fingerprint`)."""
    return [
        (e.index, e.seed, e.players, len(e), e.outcome, e.cast, e.trades, e.record)
        for e in sorted(episodes, key=lambda e: e.index)
    ]


QUICK = dict(lanes=2, seed=3, action_cap=40, max_offers=0)


def test_a_killed_cohort_resumes_to_exactly_the_games_it_planned(tmp_path):
    whole = Collector(First(), fill=False, **QUICK).cohort(6)

    partial = durable.Partial(tmp_path / "iter-00000")
    with pytest.raises(Crash):
        # Two lanes of 40-action games, and the process dies at tick 90 with
        # the last of them still in flight.
        Collector(Dies(First(), 90), fill=False, **QUICK).cohort(6, partial)
    assert partial.plan() == [0, 1, 2, 3, 4, 5]
    assert 0 < len(partial.finished()) < 6

    resumed = Collector(
        First(),
        fill=False,
        first_game=durable.resume_base(0, partial.indices()),
        **QUICK,
    )
    episodes = resumed.cohort(6, partial)

    # Nothing lost, nothing twice, and the replayed games are the games the
    # uninterrupted cohort played: dealing by index is the engine's own deal.
    assert sorted(e.index for e in episodes) == [0, 1, 2, 3, 4, 5]
    assert fingerprint(episodes) == fingerprint(whole)
    # And the run carries on past the plan rather than into it.
    assert resumed.upcoming(2) == [6, 7]


def test_a_killed_stream_tops_up_past_the_games_it_kept(tmp_path):
    partial = durable.Partial(tmp_path / "iter-00003")
    with pytest.raises(Crash):
        Collector(Dies(RandomPolicy(random.Random(1)), 90), **QUICK).collect(8, partial)
    kept = partial.finished()
    assert 0 < len(kept) < 8

    base = durable.resume_base(0, partial.indices())
    episodes = Collector(
        RandomPolicy(random.Random(2)), first_game=base, **QUICK
    ).collect(8, partial)

    indices = [e.index for e in episodes]
    assert len(indices) >= 8 and len(set(indices)) == len(indices)
    assert set(kept) <= set(indices)
    assert all(index >= base for index in set(indices) - set(kept))


def test_a_torn_line_is_dropped_by_the_reader_and_cut_by_the_next_writer(tmp_path):
    log = tmp_path / "log.jsonl"
    durable.append_line(log, {"iteration": 0})
    with open(log, "a") as handle:
        handle.write('{"iteration": 1, "posi')  # the writer died here

    assert durable.read_lines(log) == [{"iteration": 0}]
    durable.append_line(log, {"resumed_from": 1})
    durable.append_line(log, {"iteration": 1})
    assert durable.read_lines(log) == [
        {"iteration": 0},
        {"resumed_from": 1},
        {"iteration": 1},
    ]
    log.write_text('{"iteration": 0}\nnot json\n{"iteration": 1}\n')
    with pytest.raises(ValueError):
        durable.read_lines(log)


def test_a_benchmark_journal_resumes_its_own_run_and_refuses_another(tmp_path):
    path = tmp_path / "rank.rows.jsonl"
    header = {"seed": 0, "positions": 3, "children": (1, 2)}
    journal = durable.Rows(path, header)
    journal.add(0, "aa", row={"spearman": 0.5})
    journal.add(1, "bb", skipped="narrow")

    again = durable.Rows(path, header)
    assert sorted(again.kept) == [0, 1]
    assert again.check(0, "aa")["row"] == {"spearman": 0.5}
    assert again.check(2, "cc") is None
    with pytest.raises(SystemExit, match="did not reproduce"):
        again.check(1, "zz")
    with pytest.raises(SystemExit, match="other settings"):
        durable.Rows(path, {**header, "seed": 1})
    assert durable.rows_path("rank", header) == durable.rows_path("rank", dict(header))
    assert durable.rows_path("rank", header) != durable.rows_path(
        "rank", {**header, "seed": 1}
    )


# The trainers, and every collection call in them.
TRAINERS = ("ppo/__main__.py", "exit/__main__.py", "league.py")
COLLECTS = {"collect", "cohort", "start_collect"}


def test_every_trainer_collection_keeps_its_games():
    """The rule in AGENTS.md, for the loops it matters most in: a trainer that
    collects without a `partial=` holds an iteration's games in memory until
    the iteration ends, and a crash loses every one of them."""
    bare = []
    for relative in TRAINERS:
        tree = ast.parse((PACKAGE / relative).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            named = (isinstance(func, ast.Attribute) and func.attr in COLLECTS) or (
                isinstance(func, ast.Name) and func.id == "draw"
            )
            if named and not any(k.arg == "partial" for k in node.keywords):
                bare.append(f"{relative}:{node.lineno}")
    assert not bare, f"collection that keeps nothing until it ends: {bare}"


def test_a_partial_ignores_what_is_not_a_finished_game(tmp_path):
    partial = durable.Partial(tmp_path / "iter-00001")
    partial.write_plan([4, 5])
    (partial.directory / "4.pkl").write_bytes(b"")  # a file that lost its data
    (partial.directory / "5.pkl.tmp").write_bytes(b"half a game")

    assert partial.finished() == [4]
    assert partial.done() == []
    assert partial.finished() == [], "an unreadable game is removed, so it is missing"
    assert partial.indices() == {4, 5}
    assert json.loads((partial.directory / durable.PLAN).read_text()) == [4, 5]


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux-only")
def test_a_durable_write_leaves_the_page_cache_after_its_fsync(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_fadvise = os.fsync, os.posix_fadvise
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "posix_fadvise",
        lambda fd, offset, length, advice: (calls.append(advice), real_fadvise(fd, offset, length, advice))[1],
    )
    durable.write_atomic(tmp_path / "game.pkl", b"x" * 4096)
    assert calls == ["fsync", os.POSIX_FADV_DONTNEED]
    assert (tmp_path / "game.pkl").read_bytes() == b"x" * 4096


def test_a_partial_keeps_each_game_encoded_and_still_reads_the_old_pickles(tmp_path):
    """A game is kept as `<index>.game` (`hexn.gamefile`), and a partial an
    older trainer left as `<index>.pkl` pickles still loads, alone or mixed,
    so a run stopped on older code resumes on newer."""
    import pickle

    from hexn import gamefile

    episodes = Collector(First(), fill=False, **QUICK).cohort(4)
    partial = durable.Partial(tmp_path / "iter-00002")
    for episode in episodes[:2]:
        partial.keep(episode)
    for episode in episodes[2:]:
        (partial.directory / f"{episode.index}.pkl").write_bytes(pickle.dumps(episode))

    assert sorted(p.suffix for p in partial.directory.iterdir()) == [".game", ".game", ".pkl", ".pkl"]
    assert partial.finished() == sorted(e.index for e in episodes)
    assert fingerprint(partial.done()) == fingerprint(episodes)
    assert [gamefile.index_of(game) for game in partial.games()] == partial.finished()
    for kept, original in zip(partial.done(), sorted(episodes, key=lambda e: e.index)):
        for a, b in zip(
            (t for s in kept.trajectories for t in s), (t for s in original.trajectories for t in s)
        ):
            assert (a.observation.globals.view("uint32") == b.observation.globals.view("uint32")).all()
            assert (a.mask == b.mask).all() and a.log_prob == b.log_prob


def test_a_damaged_game_file_is_removed_and_played_again(tmp_path):
    episodes = Collector(First(), fill=False, **QUICK).cohort(2)
    partial = durable.Partial(tmp_path / "iter-00003")
    partial.write_plan([e.index for e in episodes])
    for episode in episodes:
        partial.keep(episode)
    damaged = partial.directory / f"{episodes[0].index}.game"
    damaged.write_bytes(damaged.read_bytes()[:-5])

    assert len(partial.done()) == 1
    assert partial.finished() == [episodes[1].index], "the damaged game is missing now"
    assert partial.indices() == {e.index for e in episodes}
