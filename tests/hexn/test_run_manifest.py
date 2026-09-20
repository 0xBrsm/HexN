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


def test_the_resolved_values_are_the_parsers_own_types(tmp_path):
    manifest = freeze(tmp_path)

    assert manifest.config["iterations"] == 3
    assert isinstance(manifest.config["iterations"], int)
    assert manifest.config["learner"] == ["", "lr=1.5e-4"]


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path):
    (tmp_path / "bare").mkdir()

    with pytest.raises(SystemExit, match="has no run.json"):
        run.load(tmp_path / "bare")


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


def test_provenance_reports_no_commit_outside_a_repository(tmp_path):
    """None rather than a guess -- `init` turns this into a refusal."""
    assert run.provenance(tmp_path)["git_commit"] is None


def test_the_parameter_set_comes_from_the_parser_not_a_copy(tmp_path):
    """If a flag is added to the league, this set grows with no change here."""
    from hexn.league import build_parser

    assert run.parameters("league") == {
        action.dest for action in build_parser()._actions if action.dest != "help"
    }


@pytest.mark.parametrize("mode", ["ppo", "league", "exit"])
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


def test_the_modes_map_to_modules_that_exist():
    """Every mode's entry names a module `python -m` can actually launch.

    `hexn.ppo` and `hexn.exit` are packages: `import hexn.ppo` reaches the
    package's own math (`__init__.py`), while `python -m hexn.ppo` runs its
    `__main__.py` instead -- the loop this map exists to point at. The error
    messages interpolate this map, so a wrong entry would send a reader to a
    module that does not exist.
    """
    import importlib

    for mode, module in run.manifest.MODULES.items():
        assert mode in ("ppo", "league", "exit")
        assert importlib.import_module(module) is not None


def test_engine_provenance_records_the_installed_hexset():
    """The happy path assumed by everything else -- `hexset` is a hard
    dependency of this package, so a working `hexn` venv has it installed
    from the HexSet repo (or a local checkout of it)."""
    info = run.manifest.engine_provenance()

    assert "hexset" in info
    if info["hexset"] is not None:
        assert "version" in info["hexset"]
        assert "git_commit" in info["hexset"]
        # `hexset.experiment.provenance` answers this too, and a commit alone
        # cannot reproduce a run trained against a modified engine tree.
        assert "dirty" in info["hexset"]


def test_engine_provenance_is_none_without_hexset(monkeypatch):
    """A hexset-less environment (tooling that only reads manifests, say)
    gets a null field instead of an ImportError out of `run.init`."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        # The provenance read is `from hexset.experiment import provenance`,
        # so the name reaching `__import__` is the submodule's, not the
        # package's -- match the whole subtree.
        if name == "hexset" or name.startswith("hexset."):
            raise ImportError("simulated: hexset not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert run.manifest.engine_provenance() == {"hexset": None}


def test_freeze_records_engine_provenance_in_the_manifest(tmp_path):
    """Additive to `provenance()`: which `hexset` a run trained against,
    not just whether this repo's own tree was clean."""
    manifest = freeze(tmp_path)

    assert "engine" in manifest.meta
    assert "hexset" in manifest.meta["engine"]


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
