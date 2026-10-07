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
