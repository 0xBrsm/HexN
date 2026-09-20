# Training

Three run modes — `hexn.ppo`, `hexn.exit`, `hexn.league` — launch from a run
directory (see [runs.md](runs.md) for how those are created, loaded, and
resumed). All three share one checkpoint format, one evaluation harness and
one network builder, in `hexn.loop`. This page describes what each mode
does, and the mechanics they share.

## PPO self-play (`hexn.ppo`)

```bash
python -m hexn.run.init --mode ppo --name my-run \
    -- --device cuda --lanes 128 --iterations 400 --checkpoint-dir runs/my-run
python -m hexn.ppo runs/my-run
```

`hexn.selfplay.Collector` (or `hexn.collect.ParallelCollector` under
`--collect-workers`) plays lockstep batches of games through
`hexn.policy.NetworkPolicy`. `hexn.rewards` turns each finished game's
`Outcome` into a per-seat target. `hexn.ppo`'s own math (GAE, the clipped
surrogate, the value loss) builds the update from the resulting batch.

**The discount is fixed at 1 and is not a flag.** The reward is zero-sum, so
roughly half of all terminal values are negative. Discounting would make a
negative terminal cheaper the later it arrives, rewarding a losing policy
for stalling. `--lam` (GAE's own lambda) is the bias/variance knob instead;
it does not change which policy is optimal.

**The value head is a win head.** `Prediction.value` is a softmax over
seats — each seat's own win probability, summing to one across the row —
trained by cross-entropy against the one-hot eventual winner
(`hexn.rewards.win_loss`). `--value-lambda` must be `1.0`: a categorical
one-hot target has no continuous bootstrap to lambda-mix in. An optional
second head, `Prediction.margin`, is trained on relative terminal points
(`--aux-margin-weight`, `0.0` by default) but is never read by a gate, an
advantage, or a search.

**Collection has two modes** (`--collect-mode`):
- `cohort` (default) deals exactly `--games-per-iteration` games and plays
  every one to completion — on-policy and unbiased in game length.
- `stream` refills a lane the moment its game ends, carrying unfinished
  games across the weight sync.

`--collect-workers N` shards collection across `N` CPU processes, each with
its own copy of the network for inference. `--async-collect` additionally
prefetches iteration `k+1`'s games during iteration `k`'s GPU update; this
requires `--collect-workers > 1` and applies only in `stream` mode, since a
`cohort` deals a fixed batch with nothing to prefetch into. `--update-workers
N` shards the PPO *update* itself across `N` CPU processes with a central
Adam step (`hexn.ddp`) — data parallelism at the gradient level, for boxes
where a single-device GPU update is not the bottleneck.

**Opponents in the training lanes** (`--mix`, e.g.
`"heximax=0.15,parent=0.1"`): a share of games are played against a named
arena entrant instead of self-play, on alternating seat pairs whose seats
are never trained on. `parent` (the checkpoint named by `--parent`) and any
`hexset` arena entrant spec both work, so a held-out opponent can be
collected against as well as evaluated on.

**The learning rate follows a schedule** (`hexn.schedule`, `--lr-schedule`):
`constant` keeps `--learning-rate` fixed; `linear` anneals it to
`--lr-floor` of itself by the final iteration; `adaptive` (`AdaptiveLR`)
holds the update's finished-epoch `approx_kl` inside a band around
`--target-kl` by scaling the rate multiplicatively, clamped to
`[--lr-min, --lr-max]` so a KL reading that does not respond to the rate
cannot walk it to infinity.

## Expert iteration (`hexn.exit`)

```bash
python -m hexn.run.init --mode exit --name my-distill \
    -- --device cuda --init runs/my-run/latest.pt \
       --simulations 256 --lanes 16 --iterations 200 --checkpoint-dir runs/my-distill
python -m hexn.exit runs/my-distill
```

`hexn.expert.SearchPolicy` plays games under a tree search
(`hexset.mcts.Search`) rooted on the *current* student's own priors rather
than a frozen teacher, and records what the search decided on each
transition's `aux`. `hexn.exit`'s own math (`__init__.py`) trains the
policy toward that visit distribution by cross-entropy, and the value head
toward the eventual winner exactly as `hexn.ppo` does (`--value-horizon`
opts into bootstrapping off the search's own backed-up value instead, at
that many decisions' remove).

Each iteration searches over a slightly better base than the last, since
the search improves along with the policy it is rooted on; the mechanism is
closed-loop rather than a fixed target to converge toward. `--init` names a
starting checkpoint (ignored when `--resume`ing) rather than a teacher.

**The trade switch is one flag feeding two places.** `--max-trades` bounds
both the collector's lanes and the search's own tree root. Playing under one
switch and searching under another would measure a different game than the
one the run is meant to improve.

**The replay store** (`hexn.store.EpisodeStore`, `--replay-positions`)
retains searched episodes on disk under `<checkpoint-dir>/replay/`, one
shard per iteration. It evicts whole iterations oldest-first once newer
ones already cover the position budget. The budget is sized in positions
rather than iterations, so a change to `--games-per-iteration` does not
silently change how much data a run trains on. `--reanalyse-samples`
re-searches that many stored positions over the *current* net each
iteration (`hexn.reanalyse.reanalyse`), replacing their stale visit target
without recollecting the episode; `hexn.replay.replay_to_ply` rebuilds the
stored episode's live game at the needed ply from its own
`hexset.record.Record`. Value targets are never reanalysed: they are the
terminal winner by default, which does not go stale the way a search prior
does.

`--corpus <file>` collects once and replays that fixed file every iteration
instead of growing a store. This is useful for comparing two losses against
identical collected data, off-policy with respect to whichever net wrote
the file.

`--contested-only`, `--anchor`, `--contested-margin`, `--stake-scale` gate
which rows the policy term actually trains on — only where the search
overruled the current policy, optionally weighted by how confident or how
valuable the correction was. The value head always sees every position
regardless. Each flag's own `--help` text documents the exact semantics;
they interact, and several are refused in combination
(`--contested-margin`/`--stake-scale` with `--refresh-prior`, in
particular).

## The learner league (`hexn.league`)

```bash
python -m hexn.run.init --mode league --name my-heat \
    --parent runs/my-run/latest.pt \
    -- --base runs/my-run/latest.pt \
       --learner "" --learner "lr=1.5e-4" --learner "entropy=0.03" \
       --iterations 60 --checkpoint-dir runs/heats/my-heat
python -m hexn.league runs/heats/my-heat
```

N learners warm-start from one `--base` checkpoint and share every game, one
seat each. A four-arm comparison therefore costs collection once plus four
quarter-sized updates, rather than four independent runs. `--learner` is
repeated once per learner; each occurrence is `""` (the base config
unchanged) or a comma-separated list of `key=value` overrides (`lr`,
`entropy`, `clip`, `c_v`, `value_lam`, `epochs`, `minibatch`, `kl_break`,
`eps`, plus `target_entropy`/`gain` to arm a per-learner entropy
controller). Standings come from the games themselves, since every game
scores every learner. Checkpoints land one directory per learner:
`<checkpoint-dir>/learner<k>/latest.pt`.

Collection knobs (`--lanes`, `--games-per-iteration`, `--collect-workers`,
`--action-cap`, `--max-trades`) are table properties every seat shares.
There is no per-learner override for them: varying one would make the arms
a sequential comparison rather than a shared-table league.

## The shared harness (`hexn.loop`)

All three modes import their checkpointing, evaluation, and network-building
code from here rather than each carrying its own copy.

**Checkpointing.** A checkpoint (`hexn.loop.save`) is written to a temporary
file and renamed over the live one, so a crash mid-write cannot leave a
truncated `latest.pt` in place of the good one. Every checkpoint carries the
game counter (`games_started`) alongside the network and optimiser state: a
game is a pure function of `(seed, index)`, so a resumed run that lost the
counter would replay games it has already trained on while looking healthy.
`--keep-every N` additionally keeps `iter-NNNNN.pt` every N iterations, so
earlier iterations remain available for comparison after `latest.pt` has
moved on; `--keep-recent N` keeps a rolling ring of the last N periodic
saves.

**Evaluation.** `hexn.loop.duel` plays the current policy against uniform
random opponents (`hexset.bots.RandomBot`) and reports a win-rate interval.
This is the floor reading: a policy that cannot clear it has nothing
else worth measuring. `hexn.loop.versus` plays two policies against each
other, seats rotating, and reports paired terminal victory points (board
and dice cancel) plus a Wilson interval on the win rate. `hexn.loop.ladder`
runs the current weights, argmax'd, against every configured rung: the
run's own `--parent` checkpoint, an optional `--search-rung` (a named arena
entrant, e.g. `heximax`), and, if `--rival <directory>` names another run,
that run's checkpoint at the *matched* iteration. This gives the ladder a
recipe-vs-recipe column alongside a common-opponent one. Rungs are fixed
points: a duel against a fixed weak opponent saturates once the policy is
reliably stronger, whereas a flat ladder reading means the policy stopped
improving.

**The network builder** (`hexn.loop.build`, `hexn.loop.add_head_flags`)
constructs a `HexNet`/`NetworkPolicy` pair from a run's `ModelConfig` fields
(`--width`, `--rounds`, `--value-head`, `--policy-head`, `--quantiles`),
shared so the three trainers cannot build three subtly different networks
from the same flags.

## Where to look next

- [runs.md](runs.md) — creating, loading, and resuming a run directory.
- [checkpoints.md](checkpoints.md) — the network shape, the checkpoint
  contract across versions, ONNX export, and widening.
- [benchmarks.md](benchmarks.md) — throughput and value-head diagnostics to
  run against a checkpoint or a collector.
</content>
