# Benchmarks

`hexn/benchmarks/` holds standalone scripts, each importable as
`hexn.benchmarks.<name>` and runnable as `python -m hexn.benchmarks.<name>
[flags]`. None of them is part of a training run; each measures one property
of a checkpoint, a collector, or the training loop's own throughput.

Most modules take `--checkpoint <path>`; the five that measure plumbing or a
file (`rollout`, `model_forward`, `mix_cost`, `training_loop`,
`return_shape`) do not. Every module except `training_loop` accepts
`--json` (a flag that switches stdout to JSON, or a path to write results
to, depending on the module) for scripted comparison across runs.
Each module's own `--help` documents its full flag list.

Two modules run without `torch`. `rollout` measures the self-play plumbing
(engine step, `encode`, legal actions, masking) with a trivial policy
standing in for the network; `return_shape` reads a file `floor` wrote and
does statistics on it. Both run where torch cannot be installed at all.

## Cost and throughput

| Module | What it measures |
| --- | --- |
| `rollout` | The self-play collector's cost with no network involved: ticks/sec and actions/sec, sweepable by lane count. Torch-free; this is the floor for reading `model_forward` against. |
| `model_forward` | One self-play move's cost, split into engine step, `encode`, and a batched forward, plus the batch-collation and host/device transfer costs a search pays per expansion, unlike a forward on tensors already resident. |
| `expert_cost` | The cost of a search-played move on real hardware, split into engine time and network time by timing a real collector against a real checkpoint (`--simulations` sweepable). Extrapolates `games_per_hour` from the measured rate. |
| `expert_scale` | Scaling of searched collection across independent CPU worker processes. Each process warms up and waits at a barrier, so staggered compiler startup does not skew the reported steady-state rate. |
| `mix_cost` | Cost of an opponent mix (`--mix` on `hexn.ppo`) to a PPO iteration, per decision and per shard. Times the learner and each mix opponent in one process by default, or the real `ParallelCollector` under `--workers N` for wall-clock timing only. |
| `training_loop` | Production-shape sync-vs-`--async-collect` PPO iteration timing. Runs identical seeded jobs with and without `--async-collect` and reports the median interval between iteration records after the pipeline-fill iteration. Run it on an otherwise idle machine: a GPU or CPU cores shared with other work measure contention, not overlap. |

## Value-head diagnostics

| Module | What it measures |
| --- | --- |
| `value_head` | Explained variance of the value head's on-policy predictions against the terminal targets `hexn.ppo` trains on. Reported by game stage as well as pooled; a pooled figure mixes an undetermined opening with a nearly-decided endgame. |
| `floor` | The irreducible variance in any value head's error at a position. Snapshots a position, replays it forward many times under the same policy, and splits mean-squared error into that variance (the floor) plus squared bias. `--dump-returns` keeps the raw return samples for `return_shape` to read. |
| `return_shape` | Reads a `floor --dump-returns` file and tests whether the sampled terminal-return distribution is Gaussian: excess kurtosis, a bimodality coefficient, and Wasserstein-1 distance against a matched Gaussian null. This is the premise a distributional (quantile) value head rests on. |
| `sibling` | Whether the value head can distinguish two positions that differ by one legal action: compares the standard deviation of its value across a probed position's legal children against its own RMS error at those same positions. Spread below error means a search ranks children on noise. |
| `rank` | Rolls out every legal child of a probed position many times to get each child's true Monte Carlo value, then checks the head's ordering of those children by rank correlation. `sibling` can only bound this, since correlated bias across siblings can cancel in `sibling`'s comparison without cancelling here. |
| `head_shape` | Freezes a trained trunk, refits a value head of a different shape (`--value-head` choice) on the same value targets, and hands the result to `sibling`. This tests whether the head's shape or its training is the limiting factor, without a fresh PPO run. |
| `head_swap` | Offline supervised comparison of a quantile head's *mean* against an MSE head's, on an identical frozen trunk and identical cached features. Tests only whether the mean estimate feeding GAE improves, not whether the spread is used for anything; nothing downstream reads it. |
| `horizon` | Whether bootstrapping the value target off a shorter horizon (`--value-horizon` on `hexn.exit`) removes the share of squared error attributable to unrolled dice, per the law-of-total-variance split it is designed to exploit. |

## Optimiser and batch-size diagnostics

These two measure the update.

| Module | What it measures |
| --- | --- |
| `noise_scale` | The gradient noise scale (McCandlish, Kaplan, Amodei, & Brown 2018): the batch size at which sampling noise stops dominating the gradient, estimated from the same quantity measured at two batch sizes. |
| `minibatch_iso_kl` | What learning rate makes a bigger minibatch travel the same policy distance (end-to-end KL) as a reference arm, holding every other input identical (weights, warm Adam state, minibatch order). Separates the effect of `--minibatch` from that of `--learning-rate`. |

## Running one

```bash
python -m hexn.benchmarks.value_head --checkpoint runs/example/latest.pt --games 256
python -m hexn.benchmarks.rollout --players 4 --ticks 200
```

Every benchmark that plays games keeps each one, its record included, as it
finishes, under `--games-dir` (default
`runs/benchmarks/<benchmark>/<UTC time>`).

Each benchmark's seed comes from the seed policy (`hexset.seeds`), with
`--bypass-seed-policy REASON` and `--registration PATH`. The ones that read
a checkpoint's value head on its own games (`value_head`, `sibling`,
`rank`, `horizon`, `floor`, `head_swap`, `head_shape`) take an evaluation
seed, never a training one. The ones that time or measure training
(`rollout`, `model_forward`, `mix_cost`, `noise_scale`,
`minibatch_iso_kl`, `expert_scale`, `expert_cost`, `training_loop`) take an
`rl` training seed, and `return_shape --null-seed` an analysis seed. With no
policy, each keeps its own default.

Most modules accept `--players` and `--seed` alongside their own knobs, and
about half also accept `--action-cap`. The three that sample and replay real
positions (`floor`, `rank`, `horizon`) take
`--seed-games`/`--positions`/`--rollouts`, controlling how many probed
positions are sampled and how many times each is rolled out; `sibling` probes
positions too, but sizes that work with `--games` and `--probe`.

`floor`, `rank` and `horizon` also append each probed position to a JSONL
journal as it finishes (`--rows`; named from the settings when omitted), and
a rerun with the same settings resumes from it, probing only the positions it
does not hold. Each position draws from streams keyed by its own index, so
it reads the same whichever positions ran before it.
</content>
