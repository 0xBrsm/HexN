# HexN

PPO self-play, expert iteration, and a resumable training loop for a
graph-native network that plays the classic ruleset published as
*Settlers of Catan* (see [Trademarks](#trademarks) below). The board is a
heterogeneous graph of hexes, vertices, and edges; the network reads that
graph directly and emits its policy as a per-node readout, so action
legality is a node property rather than a mask bolted onto a flattened
board. Published agents for this game instead flatten the hex board onto a
rectangular grid so an ordinary CNN can approximate hex adjacency.

## Three repos, one package

This package (`hexn`) is one of three things a reader of this README may
be looking at:

- **`hexn`** — this package. Model, PPO self-play, expert iteration, the
  resumable training loop, and its benchmarks. Needs PyTorch. The search
  that expert iteration trains against is not here: `hexset.mcts` owns it,
  and this package supplies the batched network evaluation it runs against
  (`hexn.policy`). The rule is that `hexn` calls the engine and never
  reimplements it.
- **`hexset`** — the rules engine, handcrafted bots, the ledger of public
  knowledge an honest policy may read, the observation encoder, and the
  search (`hexset.mcts`), in its own public repository. Needs nothing but
  numpy, so it is installable (and testable) on a machine where `hexn`'s
  dependencies cannot be. `hexn` depends on `hexset`; `hexset` never
  imports `hexn`. The sample bot, `heximax`, and `hexset.bench`
  (throughput, baselines, duels) ship from that same repository.
- The development repository this package is cut from also carries a
  research journal, training runs, and a training source tree that this
  README does not describe. Nothing about it is needed to build or run
  `hexn` from what is published here.

`hexset.arena` exposes a small registry (`register_entrant_kind`) that
`hexn.netbot` populates at import, through `hexset.clients.netbot`'s own
`register_entrants`. The loader is written once and shared with HexSet's
onnxruntime bot, which is how a duel can seat a trained checkpoint without
the arena itself needing torch.

## Install

```bash
pip install git+https://github.com/0xBrsm/HexN.git@<commit>
```

`hexn` is not published to PyPI: `pyproject.toml`'s own dependency on
`hexset` is a pinned git URL, and PyPI refuses a package whose dependencies
include one, so `pip install hexn` will not work. `<commit>` above is a
commit hash or a release tag; `pyproject.toml`'s own version field
(`0.23.3`) is package metadata, so check this repository's tags for what is
actually released. `pyproject.toml` pins `hexset` to one specific commit of
the engine rather than to its latest release, so installing `hexn` this way
pulls the exact `hexset` it was built and tested against.
Torch is not declared as a dependency; provision it separately (a ROCm or
CUDA build, matched to the training hardware) before training.

## Running things

A run is a directory, and its frozen manifest is the only input. The
configuration is frozen first, then launched: the trainers take a run
directory and refuse every flag, so a recorded result is reproducible from
what is committed rather than from whichever invocation happened to run.

```bash
# 1. freeze the configuration. Everything after `--` is the mode's own flags,
#    parsed by that mode's parser and frozen with every parameter explicit.
python -m hexn.run.init --mode ppo --name first \
    -- --lanes 128 --iterations 100 --checkpoint-dir runs/first

# 2. launch it, or resume it -- the same command either way
python -m hexn.ppo runs/first

# duels naming a trained checkpoint (HexSet's CLI; --runtime registers the loader)
python -m hexset.bench.duel --runtime hexn.netbot \
    network:runs/first/latest.pt heximax --games 400

# export for the served table: search budget and the trade gate's own look-ahead are
# per-file metadata HexSet reads (`search`/`simulations`, `gate_plies`)
python -m hexn.export_onnx --checkpoint runs/first/latest.pt --out latest.onnx \
    --search mcts --simulations 256 --gate-plies 8

# the test suite; torch-dependent files skip cleanly where torch is absent
python -m pytest tests -q
```

Requires Python 3.11+. `hexn.readout` (the index map from per-node heads to
flat action slots) is numpy-only and torch-free; everything else that
trains or evaluates a network needs PyTorch.

## Documentation

Start with [training.md](docs/training.md); the rest are listed by name.

| Document | Contents |
| --- | --- |
| [benchmarks.md](docs/benchmarks.md) | The 16 standalone benchmarks: throughput/cost, value-head, and optimiser diagnostics — what each measures and how to run it |
| [checkpoints.md](docs/checkpoints.md) | The network and its heads, contract versions, migrating and widening a checkpoint, ONNX export |
| [runs.md](docs/runs.md) | The run-manifest system: creating, loading and resuming a run directory |
| [testing.md](docs/testing.md) | Running the test suite, and which parts need PyTorch |
| [training.md](docs/training.md) | The three training modes (PPO self-play, expert iteration, the learner league) and the harness they share |

## Source Layout

| Path | Purpose |
|------|---------|
| `hexn/model.py`, `readout.py` | Message passing over the graph observation, and the index map from heads to flat action slots |
| `hexn/selfplay.py`, `collect.py`, `policy.py`, `rewards.py` | Vectorised lockstep rollout collection behind a `BatchPolicy` protocol, its sharded multi-process parallel collector, the torch policy, and the per-seat terminal scalarisation |
| `hexn/ppo/` (`__init__.py` the GAE/clipped-surrogate/value-loss math, `__main__.py` the runnable `python -m hexn.ppo` loop) | GAE, clipped surrogate, and value loss; the runnable, resumable training loop |
| `hexn/loop.py` | The checkpointing (atomic write, saved game counter), duel/versus/ladder evaluation stack, and network builder that `hexn.ppo`, `hexn.league`, and `hexn.exit` all import rather than each reimplementing |
| `hexn/league.py` | The table league: several learners training on their own seats of every shared game, for hyperparameter heats |
| `hexn/schedule.py` | The learning-rate controller |
| `hexn/expert.py`, `hexn/exit/` (`__init__.py` the distillation math, `__main__.py` the runnable `python -m hexn.exit` loop) | Expert iteration through the existing collector, and distilling a search's target back into the policy |
| `hexn/store.py`, `reanalyse.py`, `replay.py` | The retained-position replay store expert iteration trains from; refreshing a stored position's search target against a newer net without recollecting it; rebuilding a stored episode's live game at any ply |
| `hexn/netbot.py`, `export_onnx.py`, `migrate.py`, `widen.py`, `ddp.py` | A checkpoint as an arena entrant (`hexset.bench.duel --runtime hexn.netbot` is what registers it); the traced encoder and ONNX export; checkpoint migration and function-preserving widening; the data-parallel PPO update across CPU worker processes |
| `hexn/run/` | `run.init`/`run.manifest`: freezing a run's full parameter set before launch, and reading it back |
| `hexn/benchmarks/` | Forward-pass and rollout cost, value-head diagnostics, ranking probes, training-loop profiling |
| `tests/` | The `hexn` test suite; the torch-dependent tests skip cleanly where PyTorch is absent |

## Design Notes

- **"Contract N" is this package's observation/encoding version.** The
  network's input and output shapes are versioned as a single integer,
  bumped whenever that shape changes: the record fields it reads, the
  action space it emits over, and the metadata `hexn.export_onnx` writes
  into an ONNX file. This is not internal shorthand: HexSet's own
  `docs/onnx.md` defines the currently-served contract (`CONTRACT_VERSION`)
  and rejects a file naming any other. This package's own module docstrings
  cite specific numbers (`contract 5`, `contract 6`) as shorthand for
  "before/after that shape changed," most often around the trading
  redesign that motivated the last bump. `hexn.migrate` is the tool that
  carries a checkpoint across one.
- **Vector value, not scalar.** This game has four players and one winner,
  so the value head returns one number per seat rather than a zero-sum
  scalar. Every output is trained from every position, which lets a search
  back it up with max^n instead of a minimax sign flip.
- **Reproducible runs.** A game is a pure function of its seed and index,
  and the game counter is checkpointed with the weights, so a resumed run
  continues its training set instead of replaying it. Checkpoints are
  written to a temporary file and renamed.
- **Checkpoints outlive the observation and the trunk width, separately.**
  `hexn.migrate` carries a checkpoint forward, function-preserving, across
  a change to the encoder's own shape (a contract bump), so a changed
  observation does not orphan every checkpoint trained under the version
  before it. `hexn.widen` is a different, unrelated function-preserving
  transform (Net2WiderNet): it grows the *model's* own hidden width,
  leaving the observation unchanged, so a wider net can warm-start from a
  narrower checkpoint's exact function.

## Trademarks

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio. This
project is not affiliated with, endorsed by, or sponsored by either, and it ships
no Catan artwork, text, or other content. Those names appear here only to identify
which game's rules this implements — nominative use, not a claim on the marks.
</content>
