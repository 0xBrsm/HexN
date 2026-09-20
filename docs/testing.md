# Testing

```bash
python -m pytest tests -q
```

run from `src/`. `pyproject.toml` sets `pythonpath = ["."]` and
`testpaths = ["tests"]`, so this works against an editable install with no
extra configuration.

## Torch is optional, and the suite reflects it

`hexn` does not declare `torch` as a dependency (see
[checkpoints.md](checkpoints.md)); it is provisioned separately, matched to
the training hardware. Test files that exercise a torch-dependent module
call `pytest.importorskip("torch", reason="PyTorch runs on the training box
only")` at import time, so the whole file is skipped cleanly, not failed,
where torch is absent:

`test_collect.py`, `test_netbot.py`, `test_ddp.py`, `test_widen.py`,
`test_policy.py`, `test_export_onnx.py`, `test_ppo.py`, `test_ppo_main.py`,
`test_migrate.py`, `test_exit.py`, `test_league.py`, `test_model.py`,
`test_reanalyse.py`, `test_expert.py`, and `test_run_manifest.py` (its
parsers import torch even though the manifest format itself does not).

The rest of the suite needs only `numpy` and `hexset`, and runs on a machine
that cannot install torch at all:

`test_head_shape.py`, `test_no_raw_state.py`, `test_rewards.py`,
`test_schedule.py`, `test_engine_pin.py`, `test_readout.py`,
`test_replay.py`, `test_store.py`, `test_gate_wiring.py`, and
`test_selfplay.py`, plus `_mcts_fixtures.py`, a shared fixture module (not a
test file itself) borrowed from HexSet's own search tests and kept in sync
by hand rather than imported across the package boundary.

This split is deliberate: `hexn.selfplay`, `hexn.readout`, `hexn.rewards`,
`hexn.schedule`, and `hexn.run` are themselves torch-free modules, and their
tests are the proof that they stay that way. `hexn.collect` is not among
them — it imports torch at module scope to load a checkpoint, which is why
`test_collect.py` sits in the list above.
`hexn.benchmarks.rollout` and `hexn.benchmarks.return_shape` are the only
benchmarks that share this property.

## Extra dependencies for specific tests

`test_export_onnx.py` additionally calls `pytest.importorskip("onnx")` and
`pytest.importorskip("onnxruntime")` (or the matching version constraint),
since `hexn.export_onnx` itself needs both, and neither is declared in
`pyproject.toml` either. Exercising this file requires installing them
separately.

## What `test_engine_pin.py` checks

It confirms the `hexset` commit pinned in `pyproject.toml`'s git-URL
dependency matches a `hexset` git submodule at the enclosing repository's
`HEAD`, when a submodule exists. It skips cleanly otherwise: an installed
distribution, a source tarball, or any checkout with no `.git` at all. This
covers every way this published package is normally obtained.

## What `test_no_raw_state.py` checks

A static check over `hexn`'s own source — not an import, not a running
process. It scans for any access to the engine's private game-state field
outside the sanctioned `game.state(seat, *, hidden=True)` read and
`game.set_state(...)` write, guarding against a regression that would let a
network see hidden information through a raw field access.

## Markers

The suite uses no custom pytest markers beyond `pytest.importorskip` calls
and ordinary `@pytest.mark.parametrize`. No `-m` marker expression is
needed to select the torch-free subset: running without torch installed
already skips the right files automatically.
</content>
