# SPDX-License-Identifier: GPL-3.0-only
"""A frozen manifest either round-trips exactly or refuses to load.

The refusals are the point of these tests, not the happy path. A manifest that
loads with a parameter quietly supplied from today's defaults is worse than one
that fails, because it changes what a recorded run meant without saying so —
which is the failure mode `hexn.run.manifest`'s docstring exists to describe.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="the parsers import torch")

from hexn import run  # noqa: E402

LEAGUE = [
    "--base",
    "/nonexistent/iter-00450.pt",
    "--learner",
    "",
    "--learner",
    "lr=1.5e-4",
    "--iterations",
    "3",
    "--seed",
    "11",
    # Required by the league parser; the freeze resolves it like any other.
    "--checkpoint-dir",
    "runs/unit",
]


def freeze(tmp_path, argv=None, **kwargs):
    return run.freeze(
        "league",
        "unit",
        tmp_path / "unit",
        list(LEAGUE if argv is None else argv),
        repo=tmp_path,
        **kwargs,
    )


def test_a_freeze_records_every_parameter_not_only_the_ones_passed(tmp_path):
    manifest = freeze(tmp_path)

    # Anti-vacuity: the flags above name five parameters, the league defines 17.
    assert set(manifest.config) == run.parameters("league")
    assert len(manifest.config) > len(LEAGUE)
    # A default that was never typed is now explicit on disk.
    assert "games_per_iteration" in manifest.config


def test_a_frozen_config_round_trips_through_load(tmp_path):
    written = freeze(tmp_path, parent="runs/x/iter-00450.pt", plan="plans/heat.md")
    read = run.load(tmp_path / "unit")

    assert read.config == written.config
    assert read.mode == "league"
    assert read.parent == "runs/x/iter-00450.pt"
    assert read.meta["plan"] == "plans/heat.md"
    assert vars(read.namespace()) == written.config


def test_a_config_missing_a_parameter_is_refused_rather_than_defaulted(tmp_path):
    """The load-bearing refusal: a manifest frozen before a flag existed."""
    freeze(tmp_path)
    path = tmp_path / "unit" / "config" / "league.json"
    config = json.loads(path.read_text())
    del config["games_per_iteration"]
    path.write_text(json.dumps(config))

    with pytest.raises(SystemExit, match="games_per_iteration"):
        run.load(tmp_path / "unit")


def _frozen_ppo(tmp_path, *argv):
    return run.freeze("ppo", "unit", tmp_path / "unit", list(argv), repo=tmp_path)


def _drop(tmp_path, mode, *keys):
    path = tmp_path / "unit" / "config" / f"{mode}.json"
    config = json.loads(path.read_text())
    for key in keys:
        del config[key]
    path.write_text(json.dumps(config))
    return path


def test_a_parameter_added_after_the_freeze_loads_with_its_old_behaviour(tmp_path):
    """Not today's default: the value the code ran with before the flag. A new
    freeze samples per game; a run frozen before the flag keeps the worker
    stream it was collected with, and the load says what it filled."""
    written = _frozen_ppo(tmp_path, "--decided-cut", "0.1")
    assert written.config["sample_stream"] == "game"
    path = _drop(tmp_path, "ppo", "sample_stream", "setup_weight")
    on_disk = path.read_text()

    read = run.load(tmp_path / "unit")

    assert read.config["sample_stream"] == "worker"
    assert read.config["setup_weight"] == 1
    assert read.config["decided_cut"] == 0.1  # what was frozen stands
    assert read.meta["filled"] == {"sample_stream": "worker", "setup_weight": 1}
    assert path.read_text() == on_disk  # the frozen file is never rewritten
    assert set(read.config) == run.parameters("ppo")


def test_a_complete_freeze_fills_nothing(tmp_path):
    _frozen_ppo(tmp_path)
    assert "filled" not in run.load(tmp_path / "unit").meta


def test_two_seat_model_round_trips_without_changing_the_recipe(tmp_path):
    written = _frozen_ppo(tmp_path, "--contract", "9d", "--players", "2", "--others",
                          "--messages", "pooled", "--no-attention")
    read = run.load(tmp_path / "unit")
    assert read.config == written.config
    assert read.config["contract"] == "9d"
    assert read.config["others"] is True
    assert "filled" not in read.meta


def test_older_model_settings_fill_only_their_previous_values(tmp_path):
    _frozen_ppo(tmp_path)
    path = _drop(tmp_path, "ppo", "others")
    frozen = path.read_bytes()
    read = run.load(tmp_path / "unit")
    assert read.meta["filled"] == {"others": False}
    assert path.read_bytes() == frozen


RAMP = {"games_ramp": 0.0, "ramp_every": 5, "ramp_repeats": 1, "ramp_units": 0, "ramp_max_games": 0,
        "ramp_min_games": 0, "ramp_seed": 20260909}


def _carry(tmp_path, mode, **keys):
    path = tmp_path / "unit" / "config" / f"{mode}.json"
    config = json.loads(path.read_text())
    config.update(keys)
    path.write_text(json.dumps(config))
    return path


def test_a_config_frozen_with_the_games_ramp_off_loads_without_it(tmp_path):
    """The removed ramp did nothing at `games_ramp` 0: its keys are dropped,
    the run means what it meant, and the load says what it dropped."""
    written = _frozen_ppo(tmp_path, "--games-per-iteration", "6")
    path = _carry(tmp_path, "ppo", **RAMP)
    on_disk = path.read_text()
    read = run.load(tmp_path / "unit")
    assert read.config == written.config
    assert read.meta["retired"] == RAMP
    assert "filled" not in read.meta
    assert path.read_text() == on_disk  # the frozen file is never rewritten


def test_a_config_frozen_before_the_ramp_existed_loads_as_before(tmp_path):
    """No ramp key at all: nothing is dropped."""
    written = _frozen_ppo(tmp_path)
    read = run.load(tmp_path / "unit")
    assert read.config == written.config
    assert "retired" not in read.meta


def test_ramp_settings_without_games_ramp_count_as_the_ramp_off(tmp_path):
    _frozen_ppo(tmp_path)
    _carry(tmp_path, "ppo", ramp_min_games=3, ramp_max_games=8)
    assert run.load(tmp_path / "unit").meta["retired"] == {"ramp_min_games": 3, "ramp_max_games": 8}


def test_a_config_frozen_with_the_games_ramp_on_is_refused(tmp_path):
    """No setting of today's parser sizes games from a noise reading, so the
    run is refused, naming its config, rather than run at fixed games."""
    _frozen_ppo(tmp_path)
    path = _carry(tmp_path, "ppo", **{**RAMP, "games_ramp": 2.0})
    with pytest.raises(SystemExit, match="games_ramp=2.0") as refused:
        run.load(tmp_path / "unit")
    assert str(path) in str(refused.value)


@pytest.mark.parametrize("mode", sorted(run.manifest.RETIRED))
def test_no_retired_parameter_is_a_live_one(mode):
    live = run.parameters(mode)
    for retired in run.manifest.RETIRED[mode]:
        assert not set(retired.keys) & live, f"{mode} still defines {set(retired.keys) & live}"
        assert retired.guard in retired.keys or retired.guard in live


def test_a_parameter_without_an_old_behaviour_value_is_still_refused(tmp_path):
    """`init` has no fill: it predates the current warm start."""
    _frozen_ppo(tmp_path)
    _drop(tmp_path, "ppo", "init", "sample_stream")
    with pytest.raises(SystemExit, match="init"):
        run.load(tmp_path / "unit")


def test_a_config_without_runtime_loads_naming_no_runtime(tmp_path):
    """A run that names no bot plays as it did; one that does would fail at
    startup, since no runtime provides the bot."""
    _frozen_ppo(tmp_path)
    _drop(tmp_path, "ppo", "runtime")
    read = run.load(tmp_path / "unit")
    assert read.config["runtime"] == []
    assert read.meta["filled"] == {"runtime": []}


@pytest.mark.parametrize("mode", sorted(run.manifest.ADDED))
def test_every_fill_names_a_real_parameter_and_a_value_its_parser_accepts(mode):
    parser = run.manifest._parser_for(mode)
    actions = {a.dest: a for a in parser._actions}
    for key, value in run.manifest.ADDED[mode].items():
        assert key in actions, f"{mode} has no parameter {key}"
        choices = actions[key].choices
        assert choices is None or value in choices, f"{mode} {key}={value!r}"


def test_a_config_carrying_an_unknown_parameter_is_refused(tmp_path):
    freeze(tmp_path)
    path = tmp_path / "unit" / "config" / "league.json"
    config = json.loads(path.read_text())
    config["retired_flag"] = 7
    path.write_text(json.dumps(config))

    with pytest.raises(SystemExit, match="retired_flag"):
        run.load(tmp_path / "unit")


def test_a_manifest_from_another_schema_is_refused(tmp_path):
    freeze(tmp_path)
    path = tmp_path / "unit" / "run.json"
    meta = json.loads(path.read_text())
    meta["schema"] = run.manifest.SCHEMA + 1
    path.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="schema"):
        run.load(tmp_path / "unit")


@pytest.mark.parametrize("mode", ["ppo", "exit"])
def test_every_mode_can_build_its_parser_twice(mode):
    """The check that was missing, and the bug it would have caught.

    `hexn.exit` declared `--detach-value` itself while also calling
    `hexn.loop.add_head_flags`, which declares it too. argparse raises on the
    duplicate, so *every* invocation of that module failed -- including
    `--help` -- for as long as it went unnoticed, because nothing built the
    parser except the module's own `main`, and no test ran it.

    Building twice rather than once is deliberate: a parser that mutates shared
    module state passes the first call and fails the second, which is exactly
    the shape of the bug.
    """
    first = run.parameters(mode)
    second = run.parameters(mode)

    assert first == second
    assert len(first) > 10, f"{mode} resolved suspiciously few parameters"


def test_a_manifest_without_an_engine_field_still_loads(tmp_path):
    """A manifest frozen before this field existed must still load -- the
    field is additive, not a schema bump."""
    freeze(tmp_path)
    path = tmp_path / "unit" / "run.json"
    meta = json.loads(path.read_text())
    del meta["engine"]
    path.write_text(json.dumps(meta))

    read = run.load(tmp_path / "unit")
    assert "engine" not in read.meta


def test_freeze_reads_provenance_before_it_writes_the_run_directory(tmp_path, monkeypatch):
    """The run directory must not dirty the tree it is recording.

    `provenance` reads `git status --porcelain`, which counts untracked paths, so
    capturing it after `mkdir` made `git_dirty` true for every run once run
    records became tracked. A constant cannot carry the "cannot be cited" signal.
    """
    from hexn.run import manifest as m

    seen = {}

    def fake_provenance(repo):
        seen["existed"] = (tmp_path / "run-under-test").exists()
        return {"git_commit": "deadbeef", "git_dirty": False, "git_branch": "main"}

    monkeypatch.setattr(m, "provenance", fake_provenance)
    out = m.freeze(
        "league", "run-under-test", tmp_path / "run-under-test",
        ["--base", "runs/x.pt", "--learner", "", "--iterations", "1",
         "--checkpoint-dir", "runs/run-under-test"],
        repo=tmp_path,
    )
    assert seen["existed"] is False, "provenance ran after the directory appeared"
    assert out.meta["git_dirty"] is False


def test_a_contract_9_run_frozen_before_decisions_were_grouped_is_refused(tmp_path):
    """`c9_clip` fills a contract-8 manifest (contract 8 never reads it), but
    no value reproduces a contract-9 run that credited every option alone."""
    _frozen_ppo(tmp_path)
    _drop(tmp_path, "ppo", "c9_clip")
    assert run.load(tmp_path / "unit").meta["filled"] == {"c9_clip": "per-decision"}
    nine = tmp_path / "nine"
    nine.mkdir()
    _frozen_ppo(nine, "--contract", "9", "--messages", "pooled")
    _drop(nine, "ppo", "c9_clip")
    with pytest.raises(SystemExit, match="c9_clip"):
        run.load(nine / "unit")


@pytest.fixture
def seed_policy(tmp_path, monkeypatch):
    """A seed policy installed for one test (the suite runs with none)."""
    import copy

    from hexset import seeds

    saved = copy.deepcopy(seeds._state)
    monkeypatch.delenv(seeds.POLICY_ENV, raising=False)
    path = tmp_path / seeds.FILENAME
    path.write_text(json.dumps({
        "eval": {"default": {"2": 2000, "*": 1000}, "screen": [1000, 2000], "confirm": [3000]},
        "train": {"rl": [0, 1, 11], "bc": [10]},
    }))
    yield seeds.register_seed_policy(path)
    seeds._state.clear()
    seeds._state.update(saved)


def _ppo(tmp_path, *flags, **kwargs):
    return run.freeze("ppo", "unit", tmp_path / "unit", ["--checkpoint-dir", "runs/unit", *flags],
                      repo=tmp_path, **kwargs)


def test_a_freeze_holds_the_training_seed_to_the_policy(tmp_path, seed_policy):
    from hexset.seeds import SeedRefused

    assert freeze(tmp_path).meta["seeds"]["train"]["status"] == "train"
    refused = [*LEAGUE[:-3], "12", *LEAGUE[-2:]]
    with pytest.raises(SeedRefused, match="rl training seed"):
        run.freeze("league", "refused", tmp_path / "refused", refused, repo=tmp_path)
    assert not (tmp_path / "refused").exists()  # refused before anything is written
    bypassed = freeze(tmp_path, refused, seed_bypass="a probe")
    assert bypassed.meta["seeds"]["train"]["reason"] == "a probe"


def test_a_freeze_fills_the_eval_seed_from_the_policy(tmp_path, seed_policy):
    manifest = _ppo(tmp_path, "--seed", "1")
    assert manifest.config["eval_seed"] == 1000
    assert manifest.meta["seeds"]["eval"]["status"] == "screen"
    assert _ppo(tmp_path, "--seed", "1", "--players", "2").config["eval_seed"] == 2000


def test_a_confirmation_eval_seed_takes_the_plan_as_its_registration(tmp_path, seed_policy):
    from hexset.seeds import SeedRefused

    with pytest.raises(SeedRefused, match="registration"):
        _ppo(tmp_path, "--eval-seed", "3000")
    plan = tmp_path / "plan.md"
    plan.write_text("evaluations on seed 3000")
    manifest = _ppo(tmp_path, "--eval-seed", "3000", plan=str(plan))
    assert manifest.meta["seeds"]["eval"]["status"] == "confirm"


def test_a_manifest_frozen_before_eval_seed_keeps_seed_plus_10000(tmp_path):
    _ppo(tmp_path)
    path = tmp_path / "unit" / "config" / "ppo.json"
    config = json.loads(path.read_text())
    del config["eval_seed"]
    path.write_text(json.dumps(config))
    assert run.load(tmp_path / "unit").config["eval_seed"] is None
