# Run directories

A run is a directory, and its manifest is the only input a trainer reads.
This page covers the mechanics of creating, loading, and resuming one. What
each of the three trainers does (PPO, expert iteration, the learner league)
is in [training.md](training.md).

## The manifest holds the flags

`python -m hexn.ppo`, `python -m hexn.league`, and `python -m hexn.exit`
accept no training flags. Each one's `main()` takes one argument, a run
directory.

```
usage: python -m hexn.ppo <run-directory>
  create one with: python -m hexn.run.init --mode ppo --name NAME -- <flags>
  the flags this used to accept are frozen into the run's config/ instead.
```

The flags go to `python -m hexn.run.init`. It resolves them through the
trainer's own `argparse.ArgumentParser` (`build_parser()` in
`hexn.ppo.__main__`, `hexn.league`, `hexn.exit.__main__`) and writes every
resulting value to disk before training starts, defaults included. Nothing
downstream reads a default: every value a run trained under is named in its
manifest.

## Creating a run

New manifests select a GPU when available, otherwise CPU. Collection uses
the available CPU count minus two (for the learner and batch assembly),
capped by lanes and, for cohorts, games per iteration. Fixed-corpus fits do
not start collect workers; paired leagues also require equal worker quotas.
CPU availability respects process affinity and container
quotas. These automatic choices resolve when the manifest is created and
are written as concrete values. When preparing a run for another host,
pass that host's `--device` and `--collect-workers` explicitly.

All three trainers default their collect workers to the fused trunk.
Existing manifests retain their explicit execution settings, including the
reference trunk for a manifest created before the collect-trunk flag existed.
Checkpoint decisions batch all positions needing a forward automatically;
single-position callers use the same implementation with one row.

Every collector, in process or in a worker, plays its games on HexSet's
batched game loop (`hexset.gym.driver.Driver`). It plays a cohort on native
lanes wherever HexSet's optional native core is built and nothing needs a
Python game: two or four seats, every seat a contract-9 net that takes the
double turn in parts, per-game sample streams (`--sample-stream game`), and
no other policy, caster, coalition, temperature, outlook, marks or trades.
Everything else plays on Python lanes. Either way the collector deals each
game, keeps lane counts, refill order, seeds and resume indices, and writes
the same game files and training rows; every completion goes through the
same durable game sink before its lane is dealt again, and a started game
never switches engines.

CPU BF16 workers also cache weight casts and lazily trace their original
forward on its first input. The manifest's precision stays in force,
and regular weight synchronization refreshes the cache without changing
checkpoint parameters. This runs inside collect worker processes, leaving
the learner's execution settings alone. See [inference.md](inference.md) for
the inference lifecycle. HexSet documents extension installation and state
ownership in its `docs/native.md`.

```bash
python -m hexn.run.init --mode ppo --name my-run \
    --description "what this run is for" \
    -- --device cuda --lanes 128 --iterations 400 --checkpoint-dir runs/my-run
```

`--mode` is one of `ppo`, `league`, `exit`. Everything after `--` is parsed
by that mode's own parser; everything before it configures `run.init` itself:

| Flag | Purpose |
| --- | --- |
| `--name` | required; the run's directory name |
| `--category` | groups the run under `runs/<category>/<name>` instead of `runs/<name>` |
| `--runs-root` | defaults to `runs` |
| `--repo` | where to read the git commit from; defaults to the working directory |
| `--parent` | the checkpoint this run continues from, recorded as lineage |
| `--plan` | the document registering this run's design, recorded for reference; a confirmation seed takes it as its registration ([seeds](#seeds)) |
| `--bypass-seed-policy REASON` | freeze a seed the seed policy refuses; the reason is kept in `run.json` |
| `--description` | free text |
| `--force` | overwrite an existing run directory |
| `--git-commit` / `--git-dirty` | record these instead of asking `git` (for a checkout where `git` cannot resolve the repository) |
| `--allow-missing-sha` | permit a manifest with no commit recorded, for a throwaway smoke run only |

`run.init` refuses to write a manifest with no git commit unless
`--allow-missing-sha` is passed: a result with no SHA cannot be tied to the
code that produced it.

This writes:

```
runs/my-run/
  run.json           # metadata: mode, name, parent, provenance, argv, seeds
  config/ppo.json    # every parameter build_parser() defines, resolved
```

and prints the launch command:

```bash
python -m hexn.ppo runs/my-run
```

### Seeds

`run.init` holds the run's seeds to the seed policy (`hexset.seeds`) before
anything is written. `--seed` must be one of the policy's `rl` training
seeds. `--eval-seed`, the seed the run's evaluations play, is filled with
the policy's evaluation default for the table's seat count when left unset,
and must be an evaluation seed; a confirmation seed needs the run's `--plan`
as its registration, a file that names the seed. A refused seed stops the
freeze unless `--bypass-seed-policy REASON` is given. How each seed passed
is kept in `run.json` as `seeds`. With no policy nothing is checked or
filled.

## Loading and validation (`hexn.run.load`)

`hexn.run.load(directory)` reads `run.json` and the referenced
`config/<mode>.json`, then checks the config's keys in both directions
against `hexn.run.parameters(mode)`, the exact set of `dest` names the
mode's current parser defines:

- **Missing a parameter** the parser now defines: refused. A manifest
  written before a flag existed is not silently filled in from today's
  default, because that would change what the run meant without saying so.
  The exception is `hexn.run.manifest.ADDED`: a flag whose old-behaviour
  value is known is filled with *that* value, never today's default, and the
  fill is listed in the loaded manifest's `meta["filled"]`.
- **Carries a parameter** the parser no longer defines: dropped where it had
  no effect, and listed in `meta["retired"]`; refused, naming the config,
  where it acted (`hexn.run.manifest.RETIRED`). The file is never rewritten.

A refusal raises `SystemExit` with the parameter names, and points at
re-freezing the run; do not patch the JSON by hand.

`Manifest.namespace()` turns the loaded config straight into an
`argparse.Namespace`, the same object shape the trainer's own
`parser.parse_args()` would have produced, without a re-parse, so no
serialise/parse round trip can introduce a type mismatch.

## Resuming a run

`--resume` is itself a frozen parameter (default `False`). For `ppo` and
`exit`, resuming means re-freezing the manifest with `--resume` set, and a
higher `--iterations` to train further, pointing at the same
`--checkpoint-dir`; `league` resumes whenever its `learner*/latest.pt`
checkpoints exist, so relaunching a crashed heat's own manifest is enough:

```bash
python -m hexn.run.init --mode ppo --name my-run --force \
    -- --device cuda --lanes 128 --iterations 800 --checkpoint-dir runs/my-run --resume
python -m hexn.ppo runs/my-run
```

`--force` is required because `runs/my-run` already has a `run.json`.
Passing `--iterations` lower than or equal to what the checkpoint already
reached does nothing further, since the trainer loop runs
`range(start_iteration, args.iterations)`. Raising it continues training
past where the last run stopped.

On resume, `hexn.ppo` and `hexn.exit` load `<checkpoint-dir>/latest.pt`,
restore the network, optimiser, RNG state, iteration count, and game
counter, then re-assert the command line's learning rate and Adam epsilon
onto the restored optimiser (loading an optimiser's `state_dict` otherwise
keeps its old hyperparameters and only refreshes `params`). `--resume` with
no `latest.pt` present is refused, and so is a fresh start into a `--checkpoint-dir` that already
holds a `latest.pt`.

A resume loses at most the games that were in flight. Every game is kept
under `<checkpoint-dir>/partial/iter-NNNNN/` as it finishes, and a cohort
writes the list of indices it deals there first; the resumed iteration
loads those games and deals only the missing indices, and every later deal
starts past them. Once a collection is no longer needed, `hexn.durable.prune`
archives every game in it (index, seed, seats, outcome, trades and record)
to `<checkpoint-dir>/games/iter-NNNNN.games.jsonl.gz` before it removes the
partial directory, and every iteration's checkpoint stays as
`iter-NNNNN.pt`, so each kept game's behaviour policy is kept too. The
checkpoint carries `games_played`, every game through it across resumes,
which the log reports as `games_total` beside the iteration's own `games`;
a checkpoint written without it logs `None` there. `hexn.exit` also takes a
replay shard written just before the crash as its iteration's collection. The log row and the checkpoint are
written before an iteration's evaluation, which gets a row of its own, and a
resumed run appends `{"resumed_from": N}` to `log.jsonl` before its first
row: an iteration logged but not checkpointed is logged again, so a reader
keeps the last row per iteration. A `league` heat restores each seat's own
config (a controller's entropy coefficient included) and sums its standings
from the rows before the iteration it resumes at.

## Two kinds of manifest

`run.json` carries a `"kind"` field:

- **`"run"`**: a freeze written by `run.init`. Launchable.
- **`"record"`**: a reconstruction of a run that predates this machinery,
  built from whatever a checkpoint's own `args`/`config` blobs recorded
  about themselves (`hexn.run.record`/`hexn.run.read_record`). It does not
  carry the full parameter set a mode's parser defines today, since old
  runs were launched before some of today's flags existed, and filling
  those from current defaults would misstate the run's provenance. A record
  is read-only. `hexn.run.load` refuses to launch one
  and says so, pointing at `hexn.run.init --parent <that checkpoint>` to
  continue the line as a new, frozen run instead.

## Provenance

Every frozen run's `run.json` carries:

- `git_commit`, `git_dirty`, `git_branch`: this repository's state at
  freeze time (`hexn.run.manifest.provenance`), read before the run
  directory is created, since the new untracked directory would otherwise
  make `git_dirty` read `true` for every run.
- `engine`: under `hexset`, the installed engine's own version, commit and
  dirty flag (`hexn.run.manifest.engine_provenance`, reading
  `hexset.experiment.provenance()`), or `null` where `hexset` cannot be
  imported; under `runtimes`, the commit and dirty flag of the checkout
  each `--runtime` module was found in. A run's own clean tree says nothing
  about which engine or bots it trained against.
- `argv`: what followed `--` on the `run.init` command line.
- `parent`: the checkpoint this run continues from, if any, so a run's
  lineage does not have to be reconstructed later by matching iteration
  numbers between log files.

## Schema

`run.json["schema"]` is an integer (`hexn.run.manifest.SCHEMA`, currently
`1`). `load` refuses a manifest whose schema does not match the running
build's.
</content>
