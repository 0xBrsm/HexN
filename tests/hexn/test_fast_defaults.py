# SPDX-License-Identifier: GPL-3.0-only
"""Hardware choices freeze once; execution defaults reach every trainer."""
import json

import pytest

torch = pytest.importorskip("torch")

from hexn.run import freeze, load  # noqa: E402
from hexn.run import resources  # noqa: E402
from _kept import kept


def flags(mode, directory):
    shared = ["--checkpoint-dir", str(directory)]
    return shared + (["--learner", "", "--learner", ""] if mode == "league" else [])


@pytest.mark.parametrize("mode,workers", [("ppo", 30), ("exit", 16), ("league", 30)])
@pytest.mark.parametrize("gpu", [False, True])
def test_new_manifests_freeze_hardware_sized_fast_defaults(tmp_path, monkeypatch, mode, workers, gpu):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    directory = tmp_path / mode
    manifest = freeze(mode, "test", directory, flags(mode, directory), repo=tmp_path)
    assert manifest.config["device"] == ("cuda" if gpu else "cpu")
    assert manifest.config["collect_workers"] == workers
    assert manifest.config["collect_trunk"] == "fused"
    # A different launch host does not recalculate the frozen choices.
    monkeypatch.setattr(resources, "default_workers", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: not gpu)
    assert load(directory).config == manifest.config


@pytest.mark.parametrize("mode", ["ppo", "exit", "league"])
def test_explicit_resources_and_reference_configs_remain_explicit(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    directory = tmp_path / mode
    args = flags(mode, directory) + ["--device", "cpu", "--collect-workers", "1", "--collect-trunk", "reference"]
    manifest = freeze(mode, "test", directory, args, repo=tmp_path)
    assert (manifest.config["device"], manifest.config["collect_workers"], manifest.config["collect_trunk"]) == (
        "cpu", 1, "reference")


def test_paired_league_auto_workers_keep_pairs_within_the_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    directory = tmp_path / "league"
    manifest = freeze("league", "test", directory, flags("league", directory) + ["--pair-boards"], repo=tmp_path)
    assert manifest.config["collect_workers"] == 16
    assert manifest.config["games_per_iteration"] % manifest.config["collect_workers"] == 0


@pytest.mark.parametrize("mode", ["exit", "league"])
def test_older_manifests_load_with_the_recorded_collect_execution(tmp_path, mode):
    directory = tmp_path / mode
    freeze(mode, "test", directory, flags(mode, directory) + ["--device", "cpu", "--collect-workers", "1"], repo=tmp_path)
    path = directory / "config" / f"{mode}.json"
    old = json.loads(path.read_text())
    del old["collect_trunk"]
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    restored = load(directory)
    assert restored.config["collect_trunk"] == "reference"
    assert restored.meta["filled"] == {"collect_trunk": "reference"}
    assert path.read_bytes() == before


def test_minimal_cohorts_do_not_start_idle_collectors(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    directory = tmp_path / "ppo"
    manifest = freeze("ppo", "test", directory,
                      flags("ppo", directory) + ["--lanes", "1", "--games-per-iteration", "1"], repo=tmp_path)
    assert manifest.config["collect_workers"] == 1


def test_stream_collection_is_not_capped_by_an_unused_cohort_size(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    directory = tmp_path / "ppo"
    manifest = freeze("ppo", "test", directory, flags("ppo", directory) + [
        "--collect-mode", "stream", "--lanes", "64", "--games-per-iteration", "1"], repo=tmp_path)
    assert manifest.config["collect_workers"] == 30


def test_a_fixed_corpus_does_not_start_unused_collect_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "default_workers", lambda: 32)
    directory = tmp_path / "exit"
    manifest = freeze("exit", "test", directory,
                      flags("exit", directory) + ["--corpus", "fixed-episodes"], repo=tmp_path)
    assert manifest.config["collect_workers"] == 1


@pytest.mark.parametrize("searched", [False, True])
def test_fused_workers_keep_the_collected_moves_and_search_targets(searched):
    import numpy as np
    from hexn.collect import WorkerSpec, _build

    def collect(trunk):
        spec = WorkerSpec(seed=7, players=4, lanes=2, action_cap=50, max_offers=0,
            first_game=0, stride=1, width=8, rounds=1, torch_seed=123,
            learners=1 if searched else 2, simulations=4 if searched else 0, wave=4,
            **({} if trunk is None else {"trunk": trunk}))
        policies, collector = _build(spec)
        collector.sink = kept().keep  # a worker's command names its partial
        assert all(policy.net.fused == (trunk != "reference") for policy in policies)
        return collector.cohort(2)

    reference, fused = collect("reference"), collect(None)
    checked = 0
    for a, b in zip(reference, fused):
        assert a.outcome == b.outcome and a.cast == b.cast
        assert len(a.stream()) == len(b.stream())
        for x, y in zip(a.stream(), b.stream()):
            assert (x.seat, x.step, x.action, x.index) == (y.seat, y.step, y.action, y.index)
            np.testing.assert_array_equal(x.mask, y.mask)
            np.testing.assert_allclose(x.value, y.value, atol=1e-5, rtol=0)
            assert x.log_prob == pytest.approx(y.log_prob, abs=1e-5)
            if searched:
                assert x.aux.options == y.aux.options
                np.testing.assert_array_equal(x.aux.visits, y.aux.visits)
                for name in ("prior", "values"):
                    value, other = getattr(x.aux, name), getattr(y.aux, name)
                    if value is None:
                        assert other is None
                    else:
                        np.testing.assert_allclose(value, other, atol=1e-5, rtol=0)
            checked += 1
    assert len(reference) == len(fused) == 2 and checked > 20
