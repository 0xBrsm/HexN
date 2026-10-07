# Run directories

A run is a directory, and its manifest is the only input a trainer reads.
This page covers the mechanics of creating, loading, and resuming one. What
each of the three trainers does — PPO, expert iteration, the learner
league — is in [training.md](training.md).

## Why a manifest, not flags

`python -m hexn.ppo`, `python -m hexn.league`, and `python -m hexn.exit` do
not accept training flags at all. Each one's `main()` takes exactly one
argument: a run directory.

```
usage: python -m hexn.ppo <run-directory>
  create one with: python -m hexn.run.init --mode ppo --name NAME -- <flags>
  the flags this used to accept are frozen into the run's config/ instead.
```

The flags live in `python -m hexn.run.init` instead. It resolves them
through the trainer's own `argparse.ArgumentParser` (`build_parser()` in
`hexn.ppo.__main__`, `hexn.league`, `hexn.exit.__main__`) and writes every
resulting value to disk before training starts, not only the values passed
on the command line. Nothing downstream ever sees a default: a value a run
trained under is a value that run's manifest names explicitly and
permanently, whether or not it was ever mentioned on a command line.

## Creating a run

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
| `--parent` | the checkpoint this run continues from, recorded as lineage |
| `--plan` | a document describing this run's design, recorded for reference |
| `--description` | free text |
| `--force` | overwrite an existing run directory |
| `--git-commit` / `--git-dirty` | record these instead of asking `git` (for a checkout where `git` cannot resolve the repository) |
| `--allow-missing-sha` | permit a manifest with no commit recorded — for a throwaway smoke run only |

`run.init` refuses to write a manifest with no git commit unless
`--allow-missing-sha` is passed: a result with no SHA cannot be tied to the
code that produced it.

This writes:

```
runs/my-run/
  run.json           # metadata: mode, name, parent, provenance, argv
  config/ppo.json    # every parameter build_parser() defines, resolved
```

and prints the launch command:

```bash
python -m hexn.ppo runs/my-run
```

## Loading and validation (`hexn.run.load`)

`hexn.run.load(directory)` reads `run.json` and the referenced
`config/<mode>.json`, then checks the config's keys in both directions
against `hexn.run.parameters(mode)`, the exact set of `dest` names the
mode's current parser defines:

- **Missing a parameter** the parser now defines: refused. A manifest
  written before a flag existed is not silently filled in from today's
  default, because that would change what the run meant without saying so.
- **Carries a parameter** the parser no longer defines: refused. The flag
  was renamed or removed and the manifest is stale.

Both failures raise `SystemExit` with the parameter names, and both point
at re-freezing the run rather than patching the JSON by hand.

`Manifest.namespace()` turns the loaded config straight into an
`argparse.Namespace`, the same object shape the trainer's own
`parser.parse_args()` would have produced, without a re-parse. There is no
serialise/parse round trip to introduce a type mismatch.

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
no `latest.pt` present is a refusal rather than a silent restart from
scratch, and so is a fresh start into a `--checkpoint-dir` that already
holds a `latest.pt`.

A resume loses at most the games that were in flight. Every game is kept
under `<checkpoint-dir>/partial/iter-NNNNN/` as it finishes, and a cohort
writes the list of indices it deals there first; the resumed iteration
loads those games and deals only the missing indices, and every later deal
starts past them. `hexn.exit` also takes a replay shard written just before
the crash as its iteration's collection. The log row and the checkpoint are
written before an iteration's evaluation, which gets a row of its own, and a
resumed run appends `{"resumed_from": N}` to `log.jsonl` before its first
row: an iteration logged but not checkpointed is logged again, so a reader
keeps the last row per iteration. A `league` heat restores each seat's own
config (a controller's entropy coefficient included) and sums its standings
from the rows before the iteration it resumes at.

## Two kinds of manifest

`run.json` carries a `"kind"` field:

- **`"run"`** — a freeze written by `run.init`. Launchable.
- **`"record"`** — a reconstruction of a run that predates this machinery,
  built from whatever a checkpoint's own `args`/`config` blobs recorded
  about themselves (`hexn.run.record`/`hexn.run.read_record`). It does not
  carry the full parameter set a mode's parser defines today, since old
  runs were launched before some of today's flags existed; inventing values
  for those from current defaults would misrepresent the run's actual
  provenance. A record is read-only. `hexn.run.load` refuses to launch one
  and says so, pointing at `hexn.run.init --parent <that checkpoint>` to
  continue the line as a new, frozen run instead.

## Provenance

Every frozen run's `run.json` carries:

- `git_commit`, `git_dirty`, `git_branch` — this repository's state at
  freeze time (`hexn.run.manifest.provenance`), read **before** the run
  directory is created. Reading it afterward would make an untracked new
  directory appear as an uncommitted change, and `git_dirty` would read
  `true` for every run regardless of the actual tree.
- `engine` — the installed `hexset` package's own version, commit and dirty
  flag (`hexn.run.manifest.engine_provenance`, reading
  `hexset.experiment.provenance()`), `None` where `hexset` cannot be
  imported. A run's own clean tree says nothing about which engine commit it
  trained against; this is that fact recorded separately.
- `argv` — exactly what followed `--` on the `run.init` command line.
- `parent` — the checkpoint this run continues from, if any, so a run's
  lineage does not have to be reconstructed later by matching iteration
  numbers between log files.

## Schema

`run.json["schema"]` is an integer (`hexn.run.manifest.SCHEMA`, currently
`1`). `load` refuses a manifest whose schema does not match the running
build's, rather than guessing how to interpret an older or newer shape.
</content>
