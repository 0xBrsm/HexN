# Testing

```bash
python -m pytest tests -q
```

run from `src/`. `pyproject.toml` sets `pythonpath = ["."]` and
`testpaths = ["tests"]`, so this works against an editable install with no
extra configuration.

## Named bots

HexSet ships no playing bots. The tests that seat a bot by name seat
`mybot`. When no runtime registers one, the suite registers its own
(`tests/_mybot.py`): it moves at random and trades by card scarcity, so
those tests run on the engine alone. To load a bot runtime first, name its
module:

```bash
python -m pytest tests -q --runtime <module>
```

`tests/conftest.py` adds the option, imports each module before the tests
are collected (`hexset.arena.load_runtime`), and then registers the suite's
`mybot` if no runtime did. A runtime that registers `mybot` replaces the
suite's. `tests/hexn/_bots.py`'s `needs(...)` skips a test whose bot is
still missing. The suite's own scripted names (`random-too`) come from
`tests/hexn/_scripted_runtime.py`, loaded the same way a run loads its bots.

## Torch is optional

`hexn` does not declare `torch` as a dependency (see
[checkpoints.md](checkpoints.md)); it is provisioned separately, matched to
the training hardware. A test file that imports torch, directly or through
the module it tests, calls `pytest.importorskip("torch", reason="PyTorch
runs on the training box only")` at import time, so the whole file skips
where torch is absent. Among them are `test_collect.py`, `test_model.py`,
`test_model9.py`, `test_policy9.py`, `test_optim.py`, `test_export_onnx.py`,
`test_export_onnx9.py`, `test_ppo.py`, `test_migrate_c8.py` and
`test_run_manifest.py` (its parsers import torch; the manifest format
itself does not).

The rest of the suite needs only `numpy` and `hexset`, and runs on a machine
that cannot install torch at all:

`test_durable.py`, `test_expert.py`, `test_gate_wiring.py`,
`test_head_shape.py`, `test_no_raw_state.py`, `test_readout.py`,
`test_replay.py`, `test_rewards.py`, `test_schedule.py`, `test_selfplay.py`,
`test_store.py`, `test_trade.py` and `test_versions.py`, plus
`_mcts_fixtures.py`, a shared fixture module (not a test file itself)
copied from HexSet's own search tests and kept in sync by hand, since the
tests do not import across the package boundary. The two tests in
`test_selfplay.py` that build an opponent through `hexn.collect` skip
without torch.

`hexn.selfplay`, `hexn.readout`, `hexn.rewards`, `hexn.schedule`,
`hexn.durable` and `hexn.run` are torch-free modules, and their tests run
without it. `hexn.collect` imports torch at module scope to load a
checkpoint, so `test_collect.py` skips without it. Of the benchmarks, only
`hexn.benchmarks.rollout` and `hexn.benchmarks.return_shape` are
torch-free.

## Extra dependencies for specific tests

`test_export_onnx.py`, `test_export_onnx9.py` and `test_whole_search.py`
also call `pytest.importorskip("onnx")` and
`pytest.importorskip("onnxruntime")`, since `hexn.export_onnx` and
`hexn.export_onnx9` need both, and neither is declared in `pyproject.toml`.
Install them separately to run these files.

## What `test_no_raw_state.py` checks

A static scan of `hexn`'s own source: it imports nothing and runs nothing.
It fails on any access to the engine's private game-state field
(`._state`). Reads go through `game.state(seat, *, hidden=True)` and writes
through `game.set_state(...)`; a raw field access would let a network see
hidden information.

## Markers

The suite registers no custom pytest markers. Besides `importorskip` and
`@pytest.mark.parametrize`, a few tests carry `@pytest.mark.skipif` for
what they need: HexSet's native extension, a GPU, a contract-9 checkpoint
named in `HEXN_C9_CHECKPOINTS`, or `os.posix_fadvise`. No `-m` expression
is needed to select the torch-free subset: without torch installed, the
right files skip.
