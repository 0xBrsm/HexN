# SPDX-License-Identifier: GPL-3.0-only
"""The collect workers' trunk and precision (`WorkerSpec.trunk`, `.precision`)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn.collect import WorkerSpec, _build  # noqa: E402
from hexn.model import unpack  # noqa: E402
from hexn.policy import masked_log_softmax  # noqa: E402


def spec(worker: int, workers: int, **overrides) -> WorkerSpec:
    base = dict(
        seed=5,
        players=4,
        lanes=2,
        action_cap=200,
        max_offers=0,
        first_game=worker,
        stride=workers,
        width=16,
        rounds=2,
        torch_seed=1000 + worker,
        mix=(),
        parent="",
        sample_stream="game",
    )
    base.update(overrides)
    return WorkerSpec(**base)


@pytest.fixture(scope="module")
def weights():
    policies, _ = _build(spec(0, 1, torch_seed=77))
    return {k: v.clone() for k, v in policies[0].net.state_dict().items()}


def shard(weights, worker, workers, **overrides):
    policies, collector = _build(spec(worker, workers, **overrides))
    policies[0].net.load_state_dict(weights)
    return collector


def real_requests(weights, ticks: int = 40, lanes: int = 8):
    """Positions from real games: a collector ticked forward, its requests
    gathered every few ticks."""
    collector = shard(weights, 0, 1, lanes=lanes, cohort=False)
    out = []
    for tick in range(ticks):
        if tick % 5 == 0:
            out.extend(collector.requests())
        collector.tick()
    return collector.policy, out


def test_the_fused_trunk_matches_the_reference_to_float_noise(weights):
    policy, requests = real_requests(weights)
    assert len(requests) >= 40
    policy.net.fused = False
    reference = policy.act(requests)
    policy.net.fused = True
    fused = policy.act(requests)
    np.testing.assert_allclose(
        [c.log_prob for c in fused], [c.log_prob for c in reference], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        [c.value for c in fused], [c.value for c in reference], rtol=0, atol=1e-5
    )


def test_a_bf16_policy_records_the_fp32_log_softmax_of_its_own_logits(weights):
    policy, requests = real_requests(weights)
    policy.net.fused = True
    policy.precision = "bf16"
    choices = policy.act(requests)

    from hexn.model import pack

    buffer = pack(policy.layout, [r.observation for r in requests])
    mask = torch.from_numpy(np.stack([r.mask for r in requests]))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = policy.net(*unpack(policy.layout, buffer)).logits
    expected = masked_log_softmax(logits.float(), mask)
    for row, (choice, request) in enumerate(zip(choices, requests)):
        index = policy.space.index(choice.action)
        assert request.mask[index]
        assert choice.log_prob == pytest.approx(float(expected[row, index]), abs=1e-6)
        assert np.isfinite(choice.value).all()
        assert sum(choice.value) == pytest.approx(1.0, abs=1e-5)


def played(weights, workers: int, lanes: int, games: int, **overrides) -> dict:
    out = {}
    share, extra = divmod(games, workers)
    for worker in range(workers):
        collector = shard(weights, worker, workers, lanes=lanes, **overrides)
        for episode in collector.cohort(share + (1 if worker < extra else 0)):
            out[episode.index] = episode
    return out


@pytest.mark.parametrize("trunk,precision", [("fused", "fp32"), ("fused", "bf16"), ("reference", "bf16")])
def test_games_stay_independent_of_the_worker_count(weights, trunk, precision):
    a = played(weights, 1, 1, 6, trunk=trunk, precision=precision)
    b = played(weights, 3, 2, 6, trunk=trunk, precision=precision)
    assert sorted(a) == sorted(b)
    for index in a:
        assert a[index].record == b[index].record, f"game {index} played differently"


def test_a_worker_refuses_an_unknown_trunk_or_precision():
    with pytest.raises(ValueError, match="trunk"):
        _build(spec(0, 1, trunk="dense"))
    with pytest.raises(ValueError, match="precision"):
        _build(spec(0, 1, precision="fp8"))


def test_a_new_freeze_runs_fused_fp32_and_an_old_one_keeps_the_reference(tmp_path):
    import json

    from hexn import run

    written = run.freeze("ppo", "unit", tmp_path / "unit", [], repo=tmp_path)
    assert (written.config["collect_trunk"], written.config["collect_precision"]) == ("fused", "fp32")
    path = tmp_path / "unit" / "config" / "ppo.json"
    config = json.loads(path.read_text())
    del config["collect_trunk"], config["collect_precision"]
    path.write_text(json.dumps(config))
    read = run.load(tmp_path / "unit")
    assert (read.config["collect_trunk"], read.config["collect_precision"]) == ("reference", "fp32")


def test_a_new_freeze_keeps_the_batch_on_the_device_and_an_old_one_on_the_host(tmp_path):
    import json

    from hexn import run

    written = run.freeze("ppo", "unit", tmp_path / "unit", [], repo=tmp_path)
    assert written.config["batch_on_device"] is True
    path = tmp_path / "unit" / "config" / "ppo.json"
    config = json.loads(path.read_text())
    del config["batch_on_device"]
    path.write_text(json.dumps(config))
    read = run.load(tmp_path / "unit")
    assert read.config["batch_on_device"] is False
    assert read.meta["filled"]["batch_on_device"] is False
    off = run.freeze("ppo", "off", tmp_path / "off", ["--no-batch-on-device"], repo=tmp_path)
    assert off.config["batch_on_device"] is False
