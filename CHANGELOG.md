# Changelog

All notable changes to the `hexn` package are recorded here.

## 0.38.0

### Added

- `--decided-cut` (`PPOConfig.decided_cut`): positions whose mover's own
  recorded win estimate is below the cut or above one minus it leave the
  batch after GAE has run over the whole trajectory. 0, the default, keeps
  every position.
- `--setup-weight` (`PPOConfig.setup_weight`): each `SETUP_SETTLEMENT` and
  `SETUP_ROAD` position enters the batch that many times. 1, the default,
  enters it once.
- `decided_share`, `decided_low_won`, `decided_high_won` and
  `setup_repeated` gauges in the PPO log when either is set
  (`hexn.ppo.stakes_gauges`).
- `--amp fp16` (`PPOConfig.amp`): the PPO update's forward and backward, and
  the prior's forward, run under fp16 autocast, with a run-long gradient
  scaler saved in each checkpoint. The network's outputs return to fp32
  before the log-softmax and every loss term. `amp_skipped` and `amp_scale`
  are logged per update. Refused with `--update-workers`.
- `--batch-on-device`: each assembled batch moves to `--device` whole, before
  the prior forward, instead of row by row inside the update.

### Changed

- With `--fused`, the parent loaded for `--prior-kl` runs the fused trunk too.

### Fixed

- `hexn.coalition` documents what `Targeted` does without a claim about
  how tables behave.
- The coalition-mix tests skip without PyTorch, like the rest of the
  torch-dependent suite, instead of failing collection.
- The distillation test asserts that repeated updates cut the loss, and no
  longer that argmax agreement rises: the loss is a cross-entropy to the
  search's soft visit distribution, which can fall while the argmax wanders,
  and on HexSet 1.10.0's deal of the test's two games it does.

## 0.37.0

### Added

- `coalition(...)` as a `--mix` entry: a table with a coalition in it,
  drawn per game off the cast's own stream -- the learner in a drawn seat, a
  target (the learner's seat for `targeted=` of the games, another seat
  otherwise), `size=` frozen members from `members=` ganging up on it, the
  rest drawn from the pool like a `table(...)`. The coalition is hostile
  from the first move for `start=` of the games and otherwise from the move
  the target leads the table's public points by a drawn `lead=` margin,
  having built past its setup; once on it stays on
  (`hexn.collect.coalition_term`, `CoalitionPlan`, `mix_coalitions`).
- `hexn.coalition.Targeted`, HexSet 1.2.0's `Coalition` with one target and
  a trigger, both read off the game's plan, seated by the arena spec
  `targeted:<entrant>`; a run names it with `--runtime hexn.coalition`.
  The members play the coalition's table layer -- the robber on the target's
  best hex, no trade with it -- and are never trained on: a member is a pool
  checkpoint under its own cast id.
- `coalition_*` gauges in the PPO log: the share of games with a coalition,
  the share targeting the learner, and the learner's win rate in targeted,
  neutral and coalition-free games.

### Changed

- `Collector` takes a `coalition` law beside `game_law`; a run with one
  deals through `RuledLaneEnv`, which hangs the plan on each game as
  `Game.coalition`.

## 0.36.1

Trains against HexSet 1.2.1.

### Changed

- The `hexset` submodule moves to HexSet 1.2.1 (a trade pick passes over an
  answer the actor cannot cover). Nothing in `hexn` changes.

## 0.36.0

Trains against HexSet 1.2.0.

### Changed

- The `hexset` submodule moves to HexSet 1.2.0, which ships the coalition
  test opponent: `hexset.bots.Coalition` around any bot, the
  `coalition:<entrant>` spec, the `PlaysAgainst` hook, and a
  `hexset.mcts.Search` that plays as one side against named seats. The bots
  submodule moves to the release where `heximax` and `rehex` implement
  `play_against` and the runtime registers the `coalition` preset
  (`coalition:rehex`). Nothing in `hexn` changes.

## 0.35.0

Trains against HexSet 1.1.2, which ships no playing bots.

### Added

- **`--runtime <module>`** (repeatable) on `hexn.ppo`, `hexn.league`,
  `hexn.export_onnx` and `hexn.migrate` (`hexn.runtime`): the module that
  registers the bots a run names -- a `--mix` opponent, a `--search-rung`, a
  `--trader` -- imported through `hexset.arena.load_runtime` before anything
  is resolved. The trainers freeze it into the config like every other
  parameter and hand it to each collector worker
  (`hexn.collect.WorkerSpec.runtime`), since a spawned process inherits no
  registrations. A manifest frozen before the flag has none and is refused
  by `hexn.run.load`; re-freeze it.
- The run manifest's `engine` field records `runtimes`: per loaded module,
  the commit and dirty state of the checkout it was imported from, since
  the engine's commit no longer says which bots a run's names resolved to.
- **`pytest --runtime <module>`** (`tests/conftest.py`): the tests that seat
  `heximax` or `rehex` skip unless a loaded runtime registers them.

### Changed

- `heximax` and `rehex` resolve only through a loaded runtime. `hexn.collect`
  no longer imports `hexset.bots` for them, and `hexn.migrate` spawns the
  `heximax` entrant by name instead of importing the bot.
- `hexn.trade.trader_gate` spawns the trader as its name resolves; the
  engine's entrants no longer carry a placement wrapper to take off.
- A searched forced move records the search's value. HexSet evaluates a
  root with one legal move and credits its only edge with every simulation
  at that value, where it used to return the root unexpanded: the move's
  `Transition.value` is the evaluator's estimate rather than `()`, its
  `Target` carries a one-hot prior, and an ExIt value horizon that lands on
  one bootstraps from that estimate instead of falling back to the terminal
  target. The distillation target at a forced move is unchanged.
- From HexSet: a virtual-loss descent in `hexset.mcts` now counts as a lost
  visit under the search's stance (`hexset.mcts.lost_value`); a network
  checkpoint's trade gate prices an exchange without reading hidden cards;
  `max_offers=0` signs nothing, so the no-trade network also accepts no
  offer; lanes sharing a pinned board seat every gate at the lane being
  stepped; a game that reaches the turn cap is a reading neither side won,
  counted in `exhausted`.

## 0.34.0

### Added

- **`hexn.ppo` / `hexn.exit --step-checkpoint-every <K>`** (`hexn.steps`):
  the update writes `<checkpoint-dir>/latest-step.pt` -- weights, optimiser,
  steps taken, the shuffle RNG state of the pass in progress and every gauge
  accumulated so far -- after every K-th optimiser step and after the last,
  fsynced and renamed. A resumed run rebuilds the interrupted iteration's
  batch from its kept games, checks it against the fingerprint the file was
  written over, loads the file and continues the update from the next
  minibatch; the finished update, its logged gauges and the RNG the next
  iteration draws from are the uninterrupted run's. A file of another
  iteration or another batch is removed and the update starts from the
  iteration's start. `1`, the default, is every step; `0` disables; a
  manifest frozen before the flag has none. `--update-workers` keeps none.
- Every iteration row carries `step_checkpoints`, `step_checkpoint_seconds`
  and `resumed_at_step`.
- **`hexn.exit --keep-recent <N>`**: the rolling `recent-XXXXX.pt` ring
  `hexn.ppo` has, the last N checkpoints whatever `--keep-every` keeps
  (default 5).
- `hexn.ppo.update` and `hexn.exit.update` take `steps=` (`hexn.steps.Steps`).

### Changed

- Both trainers order an iteration's games by index before assembling them,
  so a collection resumed from its partial assembles the batch the
  uninterrupted one did.
- `hexn.loop.save` fsyncs the file before renaming it over the checkpoint.
- The step file is removed once the iteration's checkpoint is written; the
  iteration's games stay in its partial until then, as before.

## 0.33.0

### Added

- **`hexn.exit --k <worlds>`** (`hexn.collect.WorkerSpec.k`): the number of
  determinized worlds, drawn from the mover's own belief, each searched
  decision is rooted in (`hexset.mcts.Search(k=...)`). `1`, the default, is
  one sampled world, as before.
- **`hexn.exit --micro-batch <rows>`** (`DistillConfig.micro_batch`): each
  minibatch in slices of this many rows with gradients accumulated into its
  one step, every term normalised by the minibatch's own denominators. A
  micro-batched update leaves the batch on the host. `0`, the default, is one
  backward per minibatch.
- **`hexn.exit --prior-kl <weight>`** (`DistillConfig.prior_kl`): KL(policy ||
  the iteration's starting policy) over every legal action, added to the
  loss. `0`, the default, is off.
- Every `hexn.exit` iteration row carries `kl_to_start`, `agreement_start` /
  `agreement_end`, `target_ce_start` / `target_ce_end`, `entropy_start` /
  `entropy_end` and `decision_positions` (`hexn.exit.measure`), plus
  `assemble_seconds`, `measure_seconds`, `prior_kl` and `grad_norm`.
- **`hexn.exit --replay-positions 0`**: no replay store; each iteration
  trains on its own games, which stay in `<checkpoint-dir>/partial/` until its
  checkpoint is written, so a resumed iteration plays only its missing games.

### Changed

- `hexn.exit --resume` with `--init` on a directory without `latest.pt` starts
  fresh from `--init`, as `hexn.ppo` does, and keeps the starting weights as
  `iter-00000.pt`; without `--init` it still refuses.
- Searched collection workers seed their search from their own `torch_seed`
  rather than the run's shared `seed`, so workers no longer draw identical
  search streams.
- `hexn.exit.Batch` carries an optional `prior_log_probs`; `refresh` keeps it.

## 0.32.0

### Added

- **`hexn.ppo --micro-batch <rows>`** (`PPOConfig.micro_batch`): step through
  each minibatch in slices of this many rows, accumulating their gradients
  (each weighted by its share of the minibatch's rows) into the minibatch's
  one optimiser step. Advantages are still normalised over the whole
  minibatch, and the logged gauges are the same row-weighted means. A
  micro-batched update leaves the batch on the host and moves each slice's
  rows to the device as it is used. `0`, the default, is the whole minibatch
  in one pass with the batch moved to the device, as before. Refused with
  `--update-workers`.

### Changed

- A collector worker keeps nothing of a cohort once it has sent it; the
  trainer lets go of the episodes' transitions once the batch is assembled,
  and of the batch once its iteration is logged and saved, before the next
  collection (`hexn.collect.release_memory` trims the heap after each).
  `hexn.durable.write_atomic` drops what it wrote from the page cache once it
  is fsynced (`posix_fadvise` DONTNEED, where the platform has it). Peak
  memory falls; nothing logged changes.

## 0.31.0

### Added

- **`--no-dump-blowout-batch`** on `hexn.ppo` and `hexn.league`: when the KL
  brake fires, keep only the pre-update weights (`blowout-XXXXX-pre.pt`) and
  not the batch. `--dump-blowout-batch` stays the default.

## 0.30.1

### Changed

- Trains against HexSet 0.101.1. No hexn code changes.

## 0.30.0

### Added

- **`hexn.ppo --vp-reward <c>`**: pay `c` for every victory point a seat
  gains, at the decision after it lands (and charge it for one lost), on top
  of the win payoff (`PPOConfig.vp_reward`). Over a game it sums to
  `c * (final - opening points)`. The critic for the extra return is the
  margin head, retargeted to final points / 10 and trained at
  `--aux-margin-weight` (required > 0); the policy records the win (or
  fast-win) value plus `c` times the points still to come
  (`NetworkPolicy.vp_reward`). The win head is untouched.
- `Request.points` and `Transition.points`: the seat's own victory points at
  each recorded decision, hidden VP cards included, carried through the
  worker pipe.
- `advantages(..., rewards=)`: rewards earned between decisions.
- vp-reward runs log `learner_final_vp_mean`.

## 0.29.1

### Changed

- Trains against HexSet 0.99.0. No hexn code changes.

## 0.29.0

### Added

- **`hexn.ppo --fast-win <curve.json>`**: pay for fast wins. The curve is a
  list of weights (or an object with a `"weights"` list); a win in round `r`
  pays entry `r - 1`, the last entry holds past the curve's end, and a loss or
  an unfinished game pays 0 whenever it ends. `PPOConfig.fast_win`,
  `hexn.ppo.win_round`, `hexn.ppo.fast_weight`.
- **The fast-win head** (`ModelConfig.fast_win`, `HexNet.fast_win`,
  `Prediction.fast` / `fast_logits`): a softmax over the seats plus a "not won
  in time" slot, trained by cross-entropy against the round weight on the
  winner and the rest on the last slot. A fast-win run takes its advantage
  from it (`NetworkPolicy.record_fast`); the win head, which every gate and
  search reads, keeps its one-hot target. Built only when asked for, so every
  existing checkpoint keeps its key set. `hexn.model.graft_fast_win` seeds it
  from the win head when `--init` names a checkpoint without one.
- Fast-win runs log `fast_loss`, `win_round_median`,
  `learner_win_round_median` and `fast_payoff_mean` every iteration.
- `--fast-win` refuses a `duel(...)` mix and `--update-workers`, which it is
  not wired into.

## 0.28.0

### Added

- **`sampled:network:<path>`**, a `--mix` pool member: the checkpoint's
  policy sampled at temperature 1.0 instead of its argmax, on the same
  batched `frozen` path as a bare `network:` member. Its stream is seeded
  per worker and per entry (`mix_opponents(torch_seed=...)`,
  `rung_opponent(sample_seed=...)`), and it draws nothing from the cast
  stream, so a table deals exactly as it does with the bare spelling. Only
  a bare `network:<path>` can be sampled; `check_mix` refuses anything
  else. `frozen` gains `greedy` and `generator` keywords.

## 0.27.0

### Added

- **`hexn.export_onnx --trader <bot>`** (and `export(trader=...)`) writes a
  `trader` metadata key: a HexSet bot, as a lineup names it, that answers
  the served file's trades while the network plays every move. It defaults
  to the trader the run trained with (`hexn.ppo --trader`), and a file with
  none gets no key.

## 0.26.0

Needs a HexSet with `hexset.bots.TradesBy` (0.91.0 or later) for `--trader`.

### Added

- **`hexn.ppo --trader <bot>`** and **`hexn.league --trader <bot>`**: a HexSet
  bot, named as a lineup names it (e.g. `heximax`), answers every trade for
  the run's networks in collection and the ladder; the networks still play
  every move, bank trades included. Not combinable with `--max-offers`.
  `WorkerSpec.trader`, `Collector(trader=...)`, and a `trader` argument on
  `NetworkPolicy.trader`, `SearchPolicy.trader`, `hexn.loop.duellist`,
  `duel` and `versus`.
- **`hexn.trade.trader_gate`** and **`hexn.trade.check_trader`**.

### Removed

- **`tests/hexn/test_engine_pin.py`.** The development engine may run ahead
  of the HexSet release `pyproject.toml` pins; the release script checks that
  pin against the latest HexSet release.

## 0.25.0

Breaks clients: `max_trades` is `max_offers` throughout, and it is now the
network's own offer budget rather than a table-wide trade switch. Default
runs trade under HexSet's current rules, so they do not reproduce numbers
from 0.24.0 and earlier.

### Added

- **`hexn.ppo --init <checkpoint>`** starts a run from another checkpoint's
  weights, at iteration 0 with a fresh optimiser, and keeps them as the run's
  own `iter-00000.pt`. With `--resume` as well, a directory that already holds
  `latest.pt` resumes from it instead.
- **`hexn.ppo --prior-kl <weight>`** adds the policy's KL divergence from
  `--parent`, over every legal action, to the loss; `PPOConfig.prior_kl`,
  `hexn.ppo.attach_prior`, and a `prior_kl` column in the log.
- **`hexn.ppo --lag-rung <n>`** adds a ladder rung: the run's own kept
  checkpoint from `n` iterations before each evaluation. A missing checkpoint
  is logged as `lag_checkpoint_missing`.
- **`hexn.collect.rung_opponent`**: a named opponent, batched when it is a
  bare `network:<path>`.
- **`hexn.trade`**: `offer_budget`, `recorded_budget` and `trade_params`,
  the network's budget as HexSet's `TradeParams`.

### Changed

- **`hexset` is pinned to the HexSet 0.68.0 release**, up from 0.58.1, through
  the `hexset @ git+https://github.com/0xBrsm/HexSet.git@...` dependency.
- **`--max-offers` replaces `--max-trades`** in `hexn.ppo`, `hexn.exit`,
  `hexn.league` and the benchmarks; `--max-trades` is still accepted. It is
  how many offers the network makes a turn: `0` does not trade, omitted is
  HexSet's default (unlimited), and `-1` also reads as unlimited.
- **Opponents bargain as their own bots do.** A named entrant such as
  `heximax` keeps its own trade settings; no-trade opponents are HexSet's own
  `heximax:trading=off` and `network:<path>@0`.
- **Checkpoints and exports record `max_offers`.** `hexn.netbot.load` and
  `hexn.export_onnx` still read a pre-0.25 checkpoint's `max_trades`, and
  `Loaded.trade_params` is the checkpoint's own budget as HexSet reads it.
- **A bare `network:<path>` `--search-rung` is a batched network** playing
  under the run's own `--max-offers`, like the same name in `--mix`, rather
  than an arena bot answering one position at a time.

## 0.24.0

### Added

- **`hexn.durable`**: `write_atomic`, `append_line` and `read_lines` (JSON
  lines flushed and fsynced as written; a torn last line is dropped),
  `Partial` (one file per finished game, plus a cohort's index plan),
  `resume_base`, and `Rows`, a benchmark's per-position journal.
- **Collection keeps each game on disk as it finishes.** `Collector.cohort`,
  `Collector.collect`, `ParallelCollector.collect` and `start_collect` take
  `partial=`; a partial that holds a plan deals only its missing indices.
  `Collector.sink`, `beat`, `upcoming` and `listed` are the parts underneath.
- **`ParallelCollector(heartbeat=..., stall_seconds=...)`.** A worker that
  dies, or goes `hexn.collect.STALL_SECONDS` without a tick, raises
  `RuntimeError` instead of hanging; `heartbeat.json` reports every worker.
- **`--rows`** on `hexn.benchmarks.rank`, `floor` and `horizon`: one JSONL
  line per probed position as it finishes, and a rerun with the same
  settings resumes from it.
- **`EpisodeStore.resume(before=...)`, `EpisodeStore.ahead` and
  `EpisodeStore.adopt`**: a shard written ahead of the checkpoint is taken
  as its iteration's collection.

### Changed

- **`hexn.ppo`, `hexn.exit` and `hexn.league` keep every game under
  `<checkpoint-dir>/partial/`** and resume an interrupted iteration from it.
- **`--checkpoint-every` defaults to 1** in `hexn.ppo` and `hexn.league`.
- **The log row and the checkpoint come before in-loop evaluation**, which
  is its own row: `{"iteration": N, "ladder": ...}` in `hexn.ppo`,
  `{"iteration": N, "duel": ...}` in `hexn.exit`. A resumed run first
  appends `{"resumed_from": N}`; keep the last row per iteration.
- **`hexn.league` resumes whenever its `learner*/latest.pt` exist**, and its
  checkpoints carry `torch_rng`.
- **`hexn.benchmarks.rank` draws each position's chance children and search
  from streams keyed by the position's index**, not one shared stream, so it
  reads differently from earlier runs at the same `--seed`. `floor` and
  `horizon` reseed each position's rollouts the same way.
- **`EpisodeStore` writes every shard atomically**, and `append` of an
  iteration it already holds replaces that shard.

### Fixed

- **`hexn.exit --resume` with no `latest.pt` refuses** instead of starting
  fresh, and `hexn.exit` and `hexn.ppo` refuse a fresh start into a
  `--checkpoint-dir` that holds a `latest.pt`.
- **A resumed `hexn.league` heat** restores each seat's own `PPOConfig` (an
  entropy controller's coefficient included) and the torch RNG, sums its
  standings over the rows before the iteration it resumes at, and steps
  seats whose saves a crash split back to the iteration all of them reached.
- **`hexn.exit --corpus`** keeps each game under `<corpus>.partial/` and
  writes the corpus atomically.

## 0.23.6

No behaviour change. The test suite is smaller and faster.

### Changed

- **`tests/hexn` is smaller and faster.** Duplicate tests are removed, and
  the collection, search and training tests share module-scoped games and
  worker pools. `test_expert.py` no longer needs `torch`.

## 0.23.5

### Fixed

- **`hexn.__version__` reports the package's version.** It was a literal
  `"0.1.0"`, so every release since then reported the wrong number to
  anything that asked. It is now read from `pyproject.toml`, the one place
  the version lives, falling back to installed metadata for a wheel.
  `tests/hexn/test_versions.py` checks the two agree.

## 0.23.4

No behaviour change. Three comments and docstrings read as notes to the
people who wrote them rather than to someone reading the published code.

### Changed

- **`export_onnx`'s contract-version comment** states the current contract
  ("6") instead of the change request that introduced it.
- **`league`'s usage example** names generic checkpoint and output paths
  instead of specific local runs.
- **`tests/hexn/_mcts_fixtures.py`'s docstring** says what the module is for
  instead of recounting when it was trimmed.

## 0.23.3

### Added

- **A documentation tree under `docs/`.** `training.md` covers the three
  training modes and the harness they share, `runs.md` the run-manifest
  system, `checkpoints.md` the network, the contract versions, `hexn.migrate`,
  `hexn.widen` and ONNX export, `benchmarks.md` the sixteen benchmark modules,
  and `testing.md` the test suite and which parts require `torch`. `README.md`
  carries the index.

### Changed

- **`README.md` describes the current interface.** The training example
  showed flags on `python -m hexn.ppo`; the trainers take a run directory and
  refuse every flag, so the example now freezes a configuration with
  `python -m hexn.run.init` first. The introduction no longer lists batched
  MCTS among this package's contents: `hexset.mcts` provides the search, and
  `hexn.policy` provides the batched evaluation it runs against.

### Fixed

- **`hexn.export_onnx` names the engine it serves.** Its `--help` text,
  docstrings and comments referred to `hexset-ui` and to modules under a
  `hexset_ui` package; both were retired when that stack was renamed. They
  now name `hexset`, `hexset.clients.onnxbot` and `hexset.clients.modelmeta`.
  Comments citing `onnxbot._masked_log_softmax`, `onnxbot._board_order`,
  `onnxbot.Request` and `onnxbot.OnnxPolicy` now say that those belonged to
  the contract-1 serving path and were removed with it.
- **Corrected module references in docstrings and comments.** `hexn.rewards`
  was cited as `hexset.rewards` in `hexn.ppo`, `hexset.encoding.to_frame` as
  `hexn.ppo.rotate` in `hexn.benchmarks.head_shape`, and
  `hexn.collect.named_opponent` as `hexn.collect.greedy_opponent` in
  `hexn.selfplay`.
- **Corrected the documentation tree's flag and dependency claims.**
  `benchmarks.md` overstated which modules accept `--checkpoint`,
  `--players`, `--action-cap` and `--seed`, and listed `sibling` among the
  modules taking `--seed-games`/`--positions`/`--rollouts`; `benchmarks.md`
  and `testing.md` named `hexn.benchmarks.rollout` as the only torch-free
  benchmark, omitting `hexn.benchmarks.return_shape`; `testing.md` and
  `checkpoints.md` listed `hexn.collect` as torch-free. `README.md`'s install
  note cited a stale version.

## 0.23.2

### Fixed

- **The benchmarks document their own module path correctly.** Thirteen
  modules under `hexn/benchmarks/` gave their runnable example as
  `python -m benchmarks.<name>`. The package installs `hexn*` only, so no
  top-level `benchmarks` package exists; the path is
  `python -m hexn.benchmarks.<name>`.
- **`hexn.export_onnx` no longer names an extra that does not exist.** Its
  docstring gave the install for `onnx`/`onnxruntime` as
  `pip install -e '.[export]'`. This package declares no optional
  dependencies, so the install is `pip install onnx onnxruntime`. The same
  docstring's examples no longer assume a working directory or a `../runs/`
  path outside the package.

## 0.23.1

A patch repin onto the first release of the public HexSet repository.

The engine now has one: `0xBrsm/HexSet` was created on 2026-09-15 as the
published half of HexSet's private/public split, and `0.58.1` is its first
release -- a single squashed commit whose tree is the development repository's
`src/`, carrying none of that repository's history, branch names or commit
messages.

### Changed

- **The engine pin names a commit that exists.** Both references -- the
  `hexset` submodule gitlink and the `hexset @ git+...` requirement in
  `pyproject.toml` -- move to `b8f63dc4`, the commit `0.58.1` tags in the
  public engine repository. They previously named a commit that resolved only
  in a repository since deleted, so a fresh clone could not fetch the engine
  at all.

  The requirement names the commit rather than the tag deliberately:
  `tests/hexn/test_engine_pin.py` keeps the two pins in step by comparing the
  requirement's SHA against the submodule gitlink, which is a SHA, and a tag
  cannot be resolved to one without the network. The tag is the readable
  handle; the SHA is the one a test can check offline.
- **The engine moves 0.57.0 -> 0.58.1.** 0.58.0 collapsed HexSet's three game
  loops into one and gave `record.record_game` `trade_mode`/`max_trades`
  keywords whose defaults are the behaviour it had before, so nothing here
  changes call sites. 0.58.1 is documentation and comments only, with its
  modules' code byte-identical to 0.58.0.

## 0.23.0

The table pool can now seat the learner itself, plain or tempered, so a run
can collect fully pooled without giving up self-play's position volume; the
Catanatron bridge scripts are gone with the bridge; and the engine pin moves
to HexSet 0.56.0, which removed that bridge and keeps the adapter.

### Added

- `self` as a `table(...)` pool member in `--mix`: the learner is drawn into
  the other seats like any pool member, so a fully-pooled recipe such as
  `table(self|self|parent|network:<ckpt>)=1.0` seats one to four learner
  seats a game, every one of them trained on, with the rest a fresh draw of
  foreign styles. Repeat it to weight it. Refused as a plain entry (that is
  self-play under another name). Pools without `self` cast exactly as
  before; the opponent id law is unchanged.
- `self~<band>` in a table pool: the learner's pool-drawn copies play at a
  sampling temperature drawn per seat per game from `[1 - band, 1 + band]`,
  so the seat the table draw seated first (kept at 1.0) meets tablemates
  that build, hold and trade a little unlike itself. Tempered seats are
  still cast id 0 and still trained on: `NetworkPolicy.act` records the
  log-probability under the tempered distribution (`Request.temperature`),
  so the PPO ratio is exact and the clip bounds the pull. Off unless asked
  for; the band's draws come after every seat is cast, so no existing
  cast changes. `approx_kl` will read a tempered cohort's temperature gap as
  divergence, which is what it is.

### Removed

- The Catanatron bridge readout scripts, which drove `DC:` entrants through
  `hexset.catanatron.duel` -- removed from HexSet itself. Read a Catanatron
  reference opponent through the adapter instead, e.g.
  `python -m hexset.bench.duel <entrant> catanatron`.

### Changed

- Engine pinned to HexSet 0.56.1: the Catanatron-hosted bridge
  (`hexset.catanatron.duel`/`.player`/`.register`) is removed and the
  adapter that seats a Catanatron player at a HexSet table is unchanged, as
  is everything hexn's collector, gate and exports touch.
  `update_longest_road`'s incremental paths now raise
  `hexset.victory.StaleRoadLengths` rather than silently mis-scoring a
  `GameState` assembled without `new_game`, whose `road_lengths` cache was
  never filled. The 15-VP / 9-discard 1v1 preset is `hexset.rules.DUEL_VARIANT`
  (`GAME_TYPES["duel-variant"]`). Version 0.22.5 → 0.23.0.
- `pyproject.toml` declares the licence (`GPL-3.0-only`) and Python
  classifiers, so the built distribution carries them in its metadata.
- The `hexn.benchmarks` modules take relative example paths as their
  `--checkpoint` and `--json` defaults; they previously defaulted to
  absolute paths from one machine.
- `PPOConfig` rejects an unknown `reward` with "no other supported mode".


## 0.22.5

A patch repin: HexSet 0.55.0 makes the network trade gate one value forward
over every coverable hand again, with the rollout an opt-in behind
`gate_plies`. The collector, the duel and a served `.onnx` file all trade
through the same forward now.

### Changed

- Engine pinned to HexSet 0.55.0: at `gate_plies == 0`, the
  default, `NetworkBot` prices each candidate exchange as the head's own
  reading of the exchanged hand alongside the live position, one batched
  forward, nothing filtered -- the 0.52.0 affordability filter is gone.
  `gate_plies > 0` rolls every coverable candidate. Nothing in
  hexn's `src/` changes; `hexn.export_onnx --gate-plies` still writes the
  opt-in. Version 0.22.4 → 0.22.5.

## 0.22.4

A patch repin: HexSet 0.53.0 puts the trade gate's rollout continuation
back as an opt-in, per-checkpoint `gate_plies` setting -- the metadata
key hexn's 0.22.3 exporter already writes.

### Changed

- Engine pinned to HexSet 0.53.0: a checkpoint's `gate_plies`
  (default `0`, the 0.52.0 affordability filter's single forward) rolls
  `NetworkBot`'s own greedy policy up to that many plies forward from
  each survivor before valuing it, in one belief world drawn per ask from
  the seat's own information set, and a duel (`hexset.bench.duel`) can run
  under the automatic clearing house.
  `hexn.export_onnx --gate-plies` already writes the key this pin reads;
  nothing else in hexn's `src/` changes. Version 0.22.3 → 0.22.4.

## 0.22.3

`hexn.export_onnx` writes the served trade gate's own continuation budget,
for HexSet 0.53.0's `gate_plies` (`NetworkBot`'s rollout, restored as a
per-checkpoint setting rather than a default).

### Added

- `--gate-plies N` writes the `gate_plies` metadata key only when passed
  (the same "written only when asked" rule `--search`/`--simulations`
  already follow), independent of `--search`. Version 0.22.2 → 0.22.3.

## 0.22.2

A patch repin: HexSet 0.52.0's network trade gate no longer rolls a
position forward to price a trade. Nothing in hexn's `src/` changes.

### Changed

- Engine pinned to HexSet 0.52.0: `NetworkBot` values the
  exchanged hand from the asking seat's own frame in one batched forward,
  behind an affordability filter, in place of the sampled belief worlds
  and paired continuations; `gate_rows` is gone from
  `hexset.clients.modelmeta`/`onnxbot`/`netbot` and `gate_seed` from
  `hexset.bench.versus.PolicyPolicy`, neither of which hexn read. Longest
  road is cached per seat. Version 0.22.1 → 0.22.2.

## 0.22.1

A patch repin: HexSet 0.51.0 renames an internal evaluator class and makes
the trade gate cheaper, both invisible from hexn's `src/`.

### Changed

- Engine pinned to HexSet 0.51.0: `HonestEvaluator` is
  `ViewEvaluator` now (a `heximax_mcts` adapter came with it), and
  `NetworkBot`'s trade gate answers a repeated ask from its last evaluation
  rather than re-running its imagined-world rollouts for the counterparty's
  side. Neither name nor behaviour hexn's `src/` touches changed.

## 0.22.0

The two trainers are named after what they are: `hexn.ppo` and `hexn.exit`.
`hexn.train` and `hexn.distill_train` are gone, and so is `hexn.distill` as a
bare module name -- both are subpackages now, `__init__.py` holding the math
and `__main__.py` the runnable loop.

### Changed

- `hexn.train` -> `hexn.ppo`: `hexn/ppo.py` (`PPOConfig`, `assemble`,
  `update`, ...) is `hexn/ppo/__init__.py`, and `hexn/train.py` (the loop) is
  `hexn/ppo/__main__.py`. Launch is `python -m hexn.ppo <run-dir>`, unchanged
  in shape.
- `hexn.distill_train` -> `hexn.exit`: `hexn/distill.py` is
  `hexn/exit/__init__.py`, and `hexn/distill_train.py` (the loop) is
  `hexn/exit/__main__.py`. Launch is `python -m hexn.exit <run-dir>`.
- **`hexn.loop`** carries the harness `hexn.train` used to hold and
  `hexn.league`/`hexn.exit` import: `build`, `add_head_flags`, `duel`,
  `versus`, `ladder`, `rival_rung`, `summarise`, `save`, `prune_recent`,
  `preserve_blowout` and `duellist`. Neither trainer's own loop file carries
  a copy any more.
- `hexn.run.manifest.MODULES` and `hexn.run.init --mode` follow: the modes
  are `ppo`/`league`/`exit`, `python -m hexn.run.init --mode train` and
  `--mode distill` are gone, and a frozen run's config file is named after
  its mode (`config/ppo.json`, `config/exit.json`) same as before.
- Every tracked manifest under `runs/` is rewritten: 46 `"mode": "train"` ->
  `"ppo"`, one `"mode": "distill"` -> `"exit"` (the only `exit` run on
  record), their `config/train.json`/`config/distill.json` renamed
  alongside. A script did the rewrite, kept for the next time a mode gets
  renamed.
- The launch scripts and the docs that named the old modules point at
  `hexn.ppo` and `hexn.exit` now.

### No behaviour change

Every renamed name keeps its signature; this release moves files and updates
what points at them, and changes nothing a run's arithmetic depends on.

## 0.21.0

This release moves the engine to HexSet 0.50.0. The game a run trains in
changes with it: a turn's trading is a broadcast round by default, not the
exhaustive clearing house.

### Changed

- Engine pinned to HexSet 0.50.0. Trading defaults to
  `Game.trade_mode="round"` with `max_trades=1`: the acting seat puts one
  offer to the table a turn, every other seat answers once, the actor picks.
  The exhaustive clearing house (`trade_mode="auto"`) remains available for
  reproducing studies recorded under it. `Game.max_trades` is an `int`:
  `0` disables trading, `-1` leaves a turn's rounds uncapped. Version
  0.20.0 → 0.21.0.
- `--max-trades` on `hexn.train`, `hexn.distill_train`, `hexn.league` and the
  benchmark drivers: omitted now means the engine's default of one round a
  turn, not an unbounded table; `-1` is the uncapped form. Help text says so.
- `hexn.migrate`'s win-temperature fit sets `trade_mode="auto"` and
  `max_trades=-1` explicitly, reproducing the contract-5 protocol it was
  measured under.
- Heximax is one policy on this engine: the `heximax-notrade`,
  `heximax-balanced` and adaptive presets are gone, `heximax` and
  `heximax:pin-weights=1` remain. Docstrings and the table-pool test name
  the surviving entrants.
- `GameState` hands may be hidden on this engine (`HiddenHand`); size reads
  go through `hexset.economy.hand_size`, which the traced ONNX encoder's
  parity test covers.

## 0.20.0

This release drops interfaces HexSet removed and one module HexSet absorbed.
It is a minor bump: the `greedy` ladder rung and mix name are gone, and
`python -m hexn.duel` is gone -- HexSet's own duel CLI runs a checkpoint now.

### Changed

- Engine pinned to HexSet 0.48.0: `GET /api/version` answers for
  the API, `hexset.clients.modelmeta` replaces `hexset.server.modelmeta`, a
  checkpoint declares its own `trade_floor`/`gate_rows`, `hexset.bench.duel`
  takes `--runtime`, and `build_info()`, `NETWORK_GATE_ROWS`, `search2`,
  `greedy` and the tiered evaluator are gone. Version 0.19.10 → 0.20.0.
- `hexn.run.manifest.engine_provenance` reads `hexset.experiment.provenance()`,
  where the rest of a result's fingerprint already lives, in place of the
  removed `hexset.build_info()`. `run.json` gains a third key beside the two
  it had, `dirty`: a commit alone cannot reproduce a run trained against a
  modified engine tree. `version` and `git_commit` keep their names, so every
  manifest already on disk stays readable by the same tooling.
- **A checkpoint duel is `hexset.bench.duel --runtime hexn.netbot`**, and the
  checkpoint is named as an entrant spec (`network:<path>`, `mcts:<path>@64`)
  rather than passed as a bare path. HexSet 0.47.0 scores every duel through
  `arena.compete`, which resolves entrants by name, and 0.48.0 takes the
  runtime by name too.
- `hexn.collect.mix_opponents` takes no `max_trades`: the only name that read
  it was `greedy`, and a named entrant carries its own budget.

### Removed

- **The `greedy` ladder rung and the `greedy` mix name.** HexSet 0.47.0
  deleted `search2`, and `hexset.bots.greedy` was its depth-1 bot. No name is
  re-pointed at another bot -- a mix name that silently changed which opponent
  a run trained against is exactly what `RESERVED_MIX` exists to prevent -- so
  `--mix greedy=...` is an unknown name now, and a recipe that wants a
  handcrafted opponent says `heximax` and means it. `parent` is the only rung
  a run still gets automatically; `--search-rung heximax` is the handcrafted
  one, opt-in. Every ladder reading before this release was against a bot that
  no longer exists, so the discontinuity is named rather than papered over.
- `hexn.collect.greedy_opponent`, with the bot it built.
- **`hexn.duel`, the module.** Its versus backend (`_via_versus`, `side`,
  `entrant_seed`) went with the removed
  `hexset.bench.duel.register_versus_backend` -- `hexset.bench.duel` has one
  path now, `arena.compete` -- and what that left was a 39-line module whose
  two lines of code imported `hexn.netbot` and called HexSet's `main`. HexSet
  0.48.0's `--runtime <module>` is that import, made where the entrant is
  resolved, so there is nothing left for a wrapper to do.

  It is also strictly better placed than the wrapper was. Entrants are spawned
  inside the pool, and `--runtime` reaches it as `arena.compete`'s
  `worker_initializer`; the wrapper only ever covered a worker because fork is
  the Linux default, and would not have covered one under spawn or forkserver
  at all.
- The `hexn.netbot` re-exports of `NetworkEvaluator` and `evaluator_for`, and
  the `evaluator_max_trades=0` it passed `register_entrants`. Both were the
  network *evaluator* adapting a value head to `search2`'s leaves; the
  entrant they served (`netsearch`/`netgreedy`) went with it.

## 0.19.10

### Changed

- Engine pinned to HexSet 0.45.1: `PolicyPolicy` seats the gate it builds, so `hexn.train._Gated` -- the one-method workaround it carried -- is deleted. Version 0.19.9 → 0.19.10.

## 0.19.9

### Changed

- A network duellist is `hexset.bench.versus.PolicyPolicy` now (`train._Gated` over a `train._Checkpoint`), gated by the run's own trade budget through `compete_batched`'s three-argument gate form rather than a value bound in `train.duellist` at construction. `train.duellist` takes no `max_trades` parameter any more. `_Gated` also seats the gate itself, working around a HexSet gap where `PolicyPolicy.gate` built its checkpoint's `NetworkBot` without seating it, pricing every candidate at -1.0; fixed upstream in 0.19.10.
- `hexn.replay.replay_to_ply` is `hexset.record.replay_to(episode.record, ply)`: `hexn.selfplay.Collector` deals every game with `records=True`, so an `Episode` already carries the engine's own `hexset.record.Record` of the whole game, and re-deriving a position from `(seed, index)` plus a hand-rolled walk of `episode.stream()` and `hexset.record.advance` was a second derivation of that one replay path. The refusal for an episode whose `cast` seats an opponent is unchanged.

### Added

- `hexn.selfplay.Episode.record`: the engine's `hexset.record.Record` of the whole game the episode played. `None` for an episode collected before this field existed, the same convention `trades` already set. `hexn.store.EpisodeStore` and `ParallelCollector`'s wire format (`hexn.collect.Flattened`) carry it through a pickle round trip unchanged.

### Removed

- `hexn.train._Rows`: the adapter that seated a network behind `hexset.bench.versus.BatchPolicy` for a duel. `hexset.bench.versus.PolicyPolicy` (HexSet 0.45.0) is the adapter now; `train._Gated` (above) is the one line it still needs.
- The hand-rolled game reconstruction in `hexn.replay.replay_to_ply`: dealing from `(seed, index)` and walking `hexset.record.advance` over `episode.stream()` and `episode.trades`. `hexset.record.replay_to` over `episode.record` is the whole of it now.

## 0.19.8

### Changed

- Engine pinned to HexSet 0.45.0: lane episodes carry a `Record`, `record.replay_to`, `bench.versus.PolicyPolicy`, a gate that may take the run's trade budget, `seconds` in `Verdict.metrics()`. Version 0.19.7 → 0.19.8; nothing in hexn changes yet.

## 0.19.7

### Changed

- `hexn.train.versus`, `ladder` and `duel` run on `hexset.bench.versus.compete_batched`: HexSet owns the cohorts, the board pairing, the seat complement, `arena.wilson` and `arena.mean_interval`, and hexn supplies the lineup, the cast and the cohort size. The reported dict is `Verdict.metrics()`, key for key what `versus` returned, plus `truncated` (a game the action cap stopped, which `exhausted` was never folded with) and, on the unpaired path, `boards`/`antithetic`.
- Duel casts are unchanged: `versus` still passes `hexset.casting.alternating`, whose antithetic complement `swapped(alternating(n))` is `alternating(n, flip=True)`, so the boards played and the games played match exactly what a duel produced before. The paired-VP interval moves in the fifth decimal, and only there: `arena.mean_interval` uses the arena's `Z_95 = 1.959964` where hexn had written `1.96`.
- `hexn.train.duel` plays `hexset.bots.RandomBot` behind a `BotPolicy` instead of `hexn.selfplay.RandomPolicy`. Same uniform choice and the same refusal to trade, and one bot per board rather than one stream shared across lanes, so the reading no longer depends on how the lanes were scheduled. It keeps `decided`, `expected_share` and `mean_relative_points` alongside the verdict's own keys.
- `hexn.duel`'s `--workers 1` backend and the in-loop ladder are now the same measurement made from the same code, since both are `train.versus`.
- `hexn.train.duellist` is the one adapter: a `hexset.gym.lanes.BoardBots` bench becomes HexSet's `BotPolicy` from the bench's own spawn law, and a network is asked for `hexset.clients.policy.Policy.act_rows` -- the seam the engine's bot, search and trade gate already share -- at one forward per tick.

### Removed

- `hexn.train._antithetic`, `_paired` and `MixedPolicy`, and with them hexn's own two-cohort harness, per-board averaging, seat-complement caster, win counting, Wilson call and normal interval. Every one of those was a second derivation of a law HexSet already states, which is how two evaluations of the same pair come to disagree about what they measured.

## 0.19.6

### Changed

- `hexn.selfplay.Collector` is a driver over `hexset.gym.lanes.LaneEnv` rather than a game loop of its own: the environment holds the lanes, deals every game through `hexset.arena.deal_game`, seats each seat's trade gate, caps the action stream, counts the exchanges and reports the outcome, and the collector batches a tick's decisions to its `BatchPolicy` and files a `Transition` against each of the environment's `Decision`s by `(seat, step)`. The training contract is unchanged -- same game for the same `(seed, index)`, per-seat trajectories with `step` indices, `Episode.cast`/`trades`, `Outcome.truncated`, `Transition.aux`, and the `first_game`/`stride` sharding `hexn.collect` deals from.
- `hexn.selfplay.Outcome` is `hexset.gym.lanes.Outcome`, re-exported rather than redefined.
- `hexn.collect.alternating`, `league_caster` and `paired_caster` are `hexset.casting.alternating`, `league_rotation` and `paired`; `greedy_opponent` and `named_opponent` build a `hexset.gym.lanes.BoardBots` bench, so a scripted opponent is played *and* gated inside the engine and never crosses the `BatchPolicy` seam.
- `hexn.replay.replay_to_ply` rebuilds through `hexset.arena.deal_game` and `hexset.record.advance`; no hexn code derives a game from a seed any more except by calling `deal_game`.
- `Collector.requests()` is public: this tick's decisions for the policies the collector drives, the trainer-side view of `LaneEnv.requests()`, with bot-played seats left out. It replaces reaching into the collector for a batch to inspect.
- `pair_boards` is a board law handed to `LaneEnv` (`board(index) -> Board`) and a cohort re-arms the environment in place (`LaneEnv.cohort`), both HexSet 0.42.0 seams; masks come from `hexset.actions.mask_of`.

### Removed

- `hexn.selfplay.new_game`, `_Lane`, `BotPolicy`, `action_mask`, the lane refill, the action cap, the `Stuck` guard, the per-turn trade census and the outcome construction -- all of them the engine's, and every one of them a copy that could drift from what it mirrored.

## 0.19.5

### Changed

- Engine pinned to HexSet 0.44.3: a trade event ends at a revisited position instead of asserting, which self-play against a network checkpoint tripped on the distillation test. Version 0.19.4 → 0.19.5; nothing in hexn changes.

## 0.19.4

### Changed

- Engine pinned to HexSet 0.44.2: a finished game can still be opened after a restart. Version 0.19.3 → 0.19.4; nothing in hexn changes.

## 0.19.3

### Changed

- Engine pinned to HexSet 0.44.1: the network gate draws one world per candidate, seeded by the ask. Version 0.19.2 → 0.19.3; nothing in hexn changes.

## 0.19.2

### Changed

- Engine pinned to HexSet 0.44.0: the network gate rolls out the mover's turn in a paired belief world, so a responder prices what the actor will do with the cards and never counters into a seat that wins next action. Version 0.19.1 → 0.19.2; nothing in hexn changes.

## 0.19.1

### Changed

- Engine pinned to HexSet 0.43.0: a network checkpoint's trade gate prices continuations -- the seat's own best play from each position -- not raw hands, so a won seat offers and accepts nothing. Version 0.19.0 → 0.19.1; nothing in hexn changes.

## 0.19.0

### Removed

- The torch twin of HexSet's checkpoint runtime. `hexn.netbot.NetworkBot`, `NetworkEvaluator`, `LeafEvaluator`, `searcher`, `network_bot`, `network_evaluator` and the arena spawn functions are deleted, and so are `hexn.policy.DerivedTrader`, `after_exchange` and `_is_small` -- the network trade gate. All of it is `hexset.clients.netbot` now, over the `hexset.clients.policy.Policy` protocol; the four class names are re-exported from `hexn.netbot` and mean exactly what they did. hexn no longer builds a post-trade position anywhere.
- `hexn.ppo.rotate`, which was `hexset.encoding.to_frame` written a second time.

### Changed

- `hexn.netbot` is the torch loader and nothing else: `load()` returns a `Checkpoint`, and one `register_entrants(load, evaluator_max_trades=0)` call makes the arena's `network` and `mcts` kinds spawnable, with `netsearch`/`netgreedy` keeping their trade-free default.
- `hexn.policy.NetworkPolicy` is the torch `Policy`: `act_rows`, `value_rows` and `score_rows` over live positions, alongside the batched `act` the collector already used. Masks come from `hexset.actions.mask_of`. `trader` seats it behind HexSet's gate -- `seat` set and `seat_at` called, so a gate asked for another seat raises -- and a self-play lane, a duelled checkpoint and a served `.onnx` file trade through one implementation. `hexn.expert.SearchPolicy.trader` delegates to it.
- `hexn.migrate` reads the contract-6 global block's offsets from `hexset.encoding.global_columns` by name instead of counting them out from feature constants.

## 0.18.5

### Changed

- Engine pinned to HexSet 0.42.2: `record.from_journal` keeps a round's executed trades, so served games replay. Version 0.18.4 → 0.18.5; nothing in hexn changes.

## 0.18.4

### Changed

- Engine pinned to HexSet 0.42.1: the win banner uses the table's seat numbering, a finished game's own seats see the full log, no bot swap after game over. Version 0.18.3 → 0.18.4; nothing in hexn changes.

## 0.18.3

### Changed

- Engine pinned to HexSet 0.42.0: `hexset.bench.versus.compete_batched`, the netbot seat guard / `seat_at` / evaluator trade switch, the `LaneEnv` board law and cohorts, `actions.mask_of`. Version 0.18.2 → 0.18.3; nothing in hexn changes yet.

## 0.18.2

### Changed

- Engine pinned to HexSet 0.41.0: `hexset.clients.policy.Policy` and the runtime-free `hexset.clients.netbot`, `hexset.gym.lanes` with the one game law `arena.deal_game`, `encoding.to_frame`/`from_frame`, `encoding.global_columns`, `hexset.casting` -- the HexSet half of the boundary audit. Version 0.18.1 → 0.18.2; nothing in hexn changes yet.

## 0.18.1

### Changed

- Engine pinned to HexSet 0.40.0: a served network checkpoint's trade gate scores both sides of an exchange -- both hands moved, own row for the gain, the counterparty's row for the estimate -- so it offers and counters with the candidate best for itself among those it believes the other seat gains from too. Version 0.18.0 → 0.18.1; nothing in hexn changes.

## 0.18.0

### Changed

- Engine pinned to HexSet 0.39.0: the clearing floor belongs to the gate, not the table -- `hexset.trading.TRADE_FLOOR` is gone and every gate declares its own `trade_floor`. `hexn.netbot.NetworkBot` and `hexn.policy.DerivedTrader` declare `0.0`: a strict value comparison read as +1/-1 has no resolution for a floor to express. Version 0.17.5 → 0.18.0.

## 0.17.5

### Changed

- Engine pinned to HexSet 0.38.1: one trade event and one bot broadcast a turn (a knight's robber move no longer reopens either); a manual seat with no cards passes at once; the round's log line names only the counterparty at the close.

## 0.17.4

### Changed

- Engine pinned to HexSet 0.38.0, the release of what the 0.37.0 pin carried
  unreleased; version 0.17.3 → 0.17.4.

## 0.17.3

### Changed

- Engine pinned to HexSet 0.37.0 plus its unreleased changes: heximax's `omniscient` mode and the `heximax-omni` preset are gone (`heximax.MODES` is `("honest", "notrade")`); `default_offer` requires both floors so a bot offers only what it would take; the trade round is one collapsed log line over discrete journal notes; passes are answers; closed seats are fixed at the first move (`POST /api/open`); a checkpoint served with `search: mcts` trades through its own value-head gate (`GatedSearch`). Here: the `heximax-omni` mode's readout scripts are retired, one of them (an omni readout) deleted outright; version 0.17.2 → 0.17.3.

## 0.17.2

### Changed

- Engine pinned to HexSet 0.37.0: MCP served over HTTP at `POST /mcp` with `wait_for_turn`; client identity (`client: {id, kind}` on join, `POST /api/reclaim`, default seat names `human`/`api`/`mcp`); `GET /api/version` and the optional `version` guard on acting routes; `onnxbot` `threads` cap and terminal leaves scored on the win-probability scale; default responders never accept an offer they cannot cover. `hexn` itself is unchanged.

## 0.17.1

### Changed

- Engine pinned to HexSet 0.36.0: the trade round end to end on the served table — multi-card offers, counters, held bot turns — with the redesigned trade modal (state-named titles, one 44px icon button size, the acceptance pane in seat order). `hexn` itself is unchanged.

## 0.17.0

### Changed

- The package is renamed `hexn` (was `hexnet`); version 0.16.1 → 0.17.0.
  `import hexnet` is gone — no alias, no shim.

## 0.16.1

### Changed

- Engine pinned to HexSet 0.35.1: the public repo's research readouts, dated design drafts and one-off bench instruments moved into this repository; the simultaneous discard round (records carry `actors`). `hexn` itself is unchanged.

## 0.16.0

### Changed

- The pinned `hexset` engine advances to 0.33.0: `hexset.bench.duel`'s
  `--geometry` no longer names a seating mode -- it takes an a/b lineup
  pattern directly (`aabb` where a duel used to say `blocked`, `abab` where
  it used to say `interleaved`, or an explicit comma-separated lineup), and
  `GEOMETRIES`/`arena_lineup`'s mode table are gone from its public surface.
  `hexset.tuning`, `hexset.bench.tune`, `hexset.bench.win_temperature` and
  `hexset.bench.learn_weights` (the hill-climb weight fitter) are gone,
  superseded by `hexset.fitting`'s position-level likelihood fit;
  `hexset.bench.aivat` and `hexset.bench.trade_lab` are gone too --
  `hexset.arena.compete` and `catanatron/duel.py` are the only game loops
  left. `hexset.arena.Tournament` gains `roads`/`settlements`/`cities`/
  `seating`/`cleared` per game, and `Entrant` gains `temperature` (the `win`
  stance's calibration, read alongside its fitted weights).

## 0.15.3

### Added

- `hexn.netbot.LeafEvaluator` gains `terminal(game)`, the new
  `hexset.mcts.Evaluator` method a finished game is scored with: the
  one-hot winner in board-seat order, matching the win-probability scale
  contract 6's value head already scores every other leaf on, rather than
  `hexset.mcts.terminal_relative_points`. Raises if `game` has not
  finished. `hexn.benchmarks.expert_cost.Timed` passes `terminal`
  straight through to the evaluator it wraps, unmeasured, so both remain
  complete `Evaluator`s under the new protocol.

## 0.15.2

### Changed

- The pinned `hexset` engine advances to 0.30.0: `hexset.mcts.Evaluator`
  gains `terminal(self, game) -> Sequence[float]`, called for a finished
  game in board-seat order, unrotated, so the evaluator scores terminal
  leaves directly instead of falling through to a value-head call;
  unimplemented stances are refused rather than silently scored.
- `hexn.ppo.update` and `hexn.distill.update` no longer convert a
  minibatch's policy/value/entropy/KL/clip/grad-norm scalars to Python floats
  inside the minibatch loop -- each `float(...)` call was a blocking
  device->host sync (`hipMemcpyWithStream`). `hexn.ppo.Terms`'s scalar fields
  are device tensors now, accumulated across an update and read back once;
  the update reads its statistics back once per epoch (or once per update)
  instead of once per minibatch. `hexn.ddp`'s workers convert `Terms` to
  floats the same way they always did -- a worker is a single CPU process,
  so that read was never the sync this was about.

## 0.15.1

### Fixed

- `hexn.distill` was not ported to contract 6's win head: its value
  target was still `hexn.rewards.reward` (relative points) and its loss
  a squared error against that margin, fitted directly against
  `Prediction.value`'s win-probability output. It now trains on
  `hexn.rewards.win_loss` (the one-hot eventual winner) by cross-entropy
  against `Prediction.value_logits`, matching `hexn.ppo` term for term,
  including its quantile-head branch. `hexn.distill.Stats` gains
  `value_mse`, the squared error of `Prediction.value` against the target,
  logged next to the cross-entropy loss for comparison across value-head
  shapes. `hexn.policy.NetworkPolicy.distributions` now also returns
  `value_logits` and `quantiles` alongside the win probability and the
  policy log-probs.

## 0.15.0

### Added

- `hexn.store.EpisodeStore`: a disk-backed replay store for
  `hexn.distill_train`, one shard (`<checkpoint-dir>/replay/iter-
  NNNNN.pkl`) per iteration, retaining the last `--replay-positions`
  positions (default `300000`) and evicting whole iterations oldest-first —
  the freshest iteration is never evicted on its own arrival. `--resume`
  reloads it from disk. Replaces the in-memory `deque[Batch]` window.
- `hexn.reanalyse.reanalyse`: re-searches a sample of stored positions
  (`--reanalyse-samples`, default `3656`) over the *current* net every
  iteration and replaces their stale visit target before assembly, so a row's label keeps
  improving instead of ageing with the net that first searched it. Logged
  per iteration as `reanalysed_positions` and `reanalyse_kl`.
- `hexn.replay.replay_to_ply(episode, ply)`: rebuilds an episode's live
  `Game` at any ply from its `(seed, index)` and its recorded action and
  trade streams — the reconstruction `hexn.reanalyse` searches over.
- `hexn.expert.target_for(search, root, options, visits)`: the `Target`
  construction factored out of `SearchPolicy._choice` so a freshly searched
  root (a collector's own decision, or `hexn.reanalyse`'s re-search of a
  stored one) builds the same object either way.
- `hexn.selfplay.Episode.trades`: every exchange the table's trade event
  cleared over the whole game, `(step, a, b, received)` — the shape
  `hexset.record.Record.trades` already uses. Needed to replay a game
  bit-exactly: the engine's own trade event depends on `game.gates`, which a
  replay never seats, so the recorded exchanges are re-applied explicitly
  (`hexn.replay.replay_to_ply`, `hexset.record.advance`'s same fix for the
  same problem).

### Removed

- `hexn.distill_train`'s `--buffer-iterations` and `hexn.distill.
  DistillConfig.buffer_iterations`: superseded by `hexn.store.
  EpisodeStore` and `--replay-positions`, sized in positions rather than in
  iterations.

## 0.14.4

Cut with an engine repin (a trade moves at most three cards a side); that
work's entry was reworded afterwards and is filed under the release that
carried the rewording.

## 0.14.3

### Fixed

- `hexn.expert.SearchPolicy` gains a `trader(game, seat, max_trades)` that
  delegates to the wrapped `NetworkPolicy.trader` when the search's leaves
  are scored by a `hexn.netbot.LeafEvaluator`, and refuses (`None`)
  otherwise. Without it, `hexn.selfplay.Collector._gates` had no `trader`
  to call on a searched seat and seated `None`, so expert-iteration
  self-play (`hexn.distill_train`) played every searched seat trade-free.

## 0.14.2

### Changed

- The pinned `hexset` engine advances to 0.29.0: a deal clears only when
  both private gains exceed `hexset.trading.TRADE_FLOOR`, now `0.0197` win
  probability; the trade event fires once a turn, on the transition into MAIN, no longer
  after every MAIN action; and a trade moves at most three cards a side
  (`hexset.trading.MAX_TRADE_CARDS`). Self-play games trade rarely and
  `trades_per_turn` is read on that cadence.

## 0.14.1

Cut with an engine repin (a clearing floor, one trade event per turn); that
work's entry was reworded afterwards and is filed under the release that
carried the rewording.

## 0.14.0

### Added

- Contract 6: `hexn.model.HexNet`'s value head is a softmax over the
  seat axis. Its raw output is `Prediction.value_logits`; `Prediction.value`
  is the softmax, so every reader (gates, `hexn.ppo`'s advantage, a search
  leaf) keeps reading `value`, now as the perspective seat's own win
  probability rather than a points margin. An auxiliary VP-margin head,
  `Prediction.margin`, is always built alongside it.
- `hexn.ppo.PPOConfig.aux_margin_weight` (default `0.0`)
  weights the auxiliary margin head's squared-error term against
  `relative_points`; at the default the head is built but reaches neither
  the loss, the advantage nor a gate. `PPOConfig.reward` records the head's
  target (`"win"`, the only legal value). `hexn.train` gains
  `--aux-margin-weight` and `--reward`. `hexn.ppo.Batch` gains
  `margin_target` next to `value_target`, which is now the one-hot eventual
  winner.
- `hexn.policy.DerivedTrader.gains_many` and `hexn.netbot.NetworkBot.
  gains_many` are the network's trading surface: `gains_many(view,
  receiveds, counterparties) -> list[float]`, one batched forward per call,
  each gain the value head's own-row delta between the live position and
  the position after the exchange. The after-state moves both hands and
  re-certifies the ledger the way `hexset.trading.exchange` and
  `PublicLedger.apply_hand_diff` do; `hexn.policy.after_exchange` builds
  its encoding by rewriting the three global blocks an exchange changes
  rather than encoding the whole position again. Every candidate with two
  cards or fewer a side is always scored exactly; the rest fill whatever is
  left of `NETWORK_GATE_ROWS` (re-exported from `hexset.trading`) and every
  candidate past the scored set is declined without a forward.
  `accepts`/`accepts_many` remain as `gains_many`-thresholded wrappers.
- `hexn.policy.NetworkPolicy.trader(game, seat, max_trades)` builds the
  seat's `DerivedTrader`; `hexn.selfplay.Collector` seats its policies on
  `game.gates`, so self-play games trade. A `BatchPolicy` may offer a
  `trader(game, seat, max_trades)` hook; a policy without one never trades.
- `hexn.selfplay.Outcome.trades` counts the exchanges a game cleared, and
  `hexn.train` logs `trades_per_turn`.
- `hexn.collect.mix_opponents` routes a bare `network:<path>` pool member
  (no `@trades` override) through `frozen` — the batched `NetworkPolicy` a
  ladder rung uses — instead of a batch-of-one `NetworkBot`; an explicit
  `@trades` override still goes through `named_opponent`.
- `hexn.migrate` migrates a contract-5 checkpoint onto contract 6: the
  twenty public-valuation input columns are dropped (a selection — every
  contract-6 column has a contract-5 source), the knight's per-hex block on
  `heads.hexes` collapses into one `heads.globals` row initialised as the
  block's mean, the value head's weights are rescaled by `1 / T` with `T`
  fitted by maximum likelihood against the eventual winner of recorded
  games on the target engine, and the auxiliary margin head starts as an
  exact copy of the old, unscaled value head.
- `hexn.run` manifests carry an `engine` block: the installed `hexset`
  engine's own version and commit, alongside the run's own provenance.
- `hexn.duel` (`python -m hexn.duel`) wraps `hexset.bench.duel` for
  network-vs-network and bare-checkpoint duels, the half that needs torch.
- `hexn.train.versus` reports `turns_mean`/`turns_median`/`turns_max` and
  `exhausted` (a game that hit `MAX_TURNS` with no winner).

### Changed

- The pinned `hexset` engine advances to contract 6: there is no public
  trading layer (no valuation vector, no `Game.publish`), trading is one
  engine-run event per MAIN action answered by each seat's private gate,
  and `PLAY_KNIGHT` only spends the card — the robber placement runs
  through `MOVE_ROBBER`, the same decision a rolled seven uses. The flat
  action space is 456 (was 550) and `hexset.encoding.global_features` is 67
  (was 87).
- `hexn.model.HexNet`'s value head trains on the one-hot eventual winner
  (`hexn.rewards.win_loss`) by cross-entropy, not on `relative_points` by
  squared error. `hexn.ppo.PPOConfig.value_lam` accepts only `1.0`: a
  categorical label has no continuous bootstrap to mix towards.
  `relative_points` survives as the auxiliary margin head's target.
- `hexn.export_onnx` writes the contract-6 record: `globals` is 67, the
  flat action space 456, the outputs are `action_index`/`prior`/`value`,
  and every input is int64 or bool.
- `max_offers` is `max_trades` everywhere, including `--max-trades` on
  `hexn.train`, `hexn.league`, `hexn.distill_train` and the
  benchmarks: `0` means no trading, `None` (the new default) unbounded. It
  is a switch, not a budget.
- `search2-offers3`/`search2-offers0` are `search2`/`search2-notrade`, and
  `greedy-offers3`/`greedy-offers0` are `greedy`/`greedy-notrade`.
- `hexn.collect` imports `hexset.bots`, so `heximax`, `heximax-omni` and
  `heximax-notrade` resolve in `--mix`. `heximax` is `hexset.bots.heximax`
  and `hexset.evaluate` is `hexset.bots.evaluate`.
- `hexn.policy.NetworkPolicy.score`, `.distributions` and `.evaluate`
  drop their pair arguments and their offer return values; `hexn.ppo.
  Batch` and `hexn.ddp`'s shard drop `pair` and `offer`, and `hexn.
  distill.Batch` drops `offer_target`, `trade_mass`, `anchor_offers` and
  `anchor_mass`.
- `hexn.netbot.NetworkEvaluator` does not trade (`max_trades=0`): a
  `netsearch`/`netgreedy` entrant's gate would need a position and it
  holds only a state.
- This package is `hexn` (renamed from `hexset`): the engine, bots and
  encoder moved to the HexSet repo and are a dependency, not source here.
  `hexset.<module>` imports become `hexn.<module>` for `collect`, `ddp`,
  `distill`, `distill_train`, `expert`, `export_onnx`, `league`,
  `migrate`, `model`, `netbot`, `policy`, `ppo`, `readout`, `rewards`,
  `schedule`, `selfplay`, `train`, `widen` and `run`.
- `hexn` depends on the root `hexset` distribution as a pinned git
  dependency (no `#subdirectory=`); its duel/throughput/tuning tools are
  `hexset.bench.*`, not the former top-level `benchmarks.*`.
- The GPU image is `docker/Dockerfile` at the repo root, and needs a
  second bind mount for a local `hexset` engine checkout.
- HexNet reads the engine's state through `game.state(seat, hidden=...)`
  instead of the old `game.state` field.

### Removed

- The `trade_give`/`trade_want` heads and `Prediction.give`/`.want`.
- `hexn.policy`'s `pair_logits`, `NUM_PAIRS`, `pair_index`, `pair_mask`
  and `trade_slot`, and `hexn.ppo`'s `_offer_slot` and `_pair_mask` — the
  network no longer samples an offer, so there is no joint log-prob, no
  pair mask and no masked-pair entropy term.
- `hexn.distill.losses`/`anchor_losses` lose their offer term, and
  `Stats.offer_loss` with it.
- `hexn.migrate`'s contract-3 and contract-4 hops: the tool migrates
  contract 5 to contract 6 only.

## 0.13.0

### Changed

- An offer with no `ask` is now put to the table in a random order from
  the game's own RNG, not clockwise from the proposer.
  `trading.responders` keeps clockwise only as the eligibility list.
  Seeded games no longer replay bit-for-bit across this change.
- The piece supply is now enforced: 15 roads, 5 settlements, 4 cities a
  player. `can_place_settlement`, `can_upgrade_to_city` and
  `can_place_road` refuse a piece not in the box; this engine had no cap
  before.
- `benchmarks.duel` imports `catan.collect` and `catan.train` only where
  used, so the module loads on a box without torch.

### Added

- `catan.widen`: function-preserving checkpoint widening (Net2WiderNet).
  `--noise σ` adds Gaussian noise to the copied weights; `catan.train
  --resume` continues from a widened checkpoint with no new flag.
- `--mix` accepts a table entry, `table(a|b|c)=f`: in a share `f` of
  games the learner seats once and every other seat draws independently
  from `a|b|c`.
- `benchmarks.duel` records each verdict's seat geometry — blocked
  (`[a, a, b, b]`) or interleaved (`[a, b, a, b]`) — previously chosen
  silently by worker count.
- `--geometry {blocked,interleaved}` on the arena duel path. Default
  `blocked` reproduces old verdicts bit for bit; the versus path can
  only seat interleaved.

## 0.12.0

### Added

- `benchmarks.aivat`: AIVAT's chance-correction term (Burch et al., AAAI
  2018) for duels already recorded, lowering variance without changing
  the estimator's mean.
- The dice roll, dev-card draw and robber steal are now enumerated
  exactly rather than sampled. `game.imagine(..., randomize_deck=True)`
  reshuffles the unseen deck so this stays safe.
- `instrumented`, a bit-identical twin of `arena._play_one` for
  verifying a recorded verdict; `--check` asserts against it.
- Reports both AIVAT's realized variance reduction and its theoretical
  ceiling, `1 - sqrt(1 - rho^2)`.

## 0.11.0

### Added

- `benchmarks.human_agreement`: scores the policy against recorded human
  decisions one at a time, reporting top-1 agreement and log-loss
  against a matched uniform-null baseline.
- Decisions with one legal action are excluded from the score; the
  excluded count and any unrecognized action are reported separately.
- Results break down by `ActionType`, `Phase`, and game progress.
- Intervals are clustered on the game rather than the position, since
  decisions within a game are correlated.

## 0.10.0

### Added

- `--mix` now accepts any arena entrant spec — `search2-offers3`,
  `mcts:<ckpt>@64`, `network:<ckpt>`, or any preset — via
  `collect.named_opponent`.
- `collect.RESERVED_MIX` reserves `greedy`/`parent`: `--mix greedy`
  still means `greedy-offers3`, not the arena's unrestricted `greedy`
  preset.
- `benchmarks.mix_cost`: measures a mix's per-decision, per-shard
  training cost from its two pure-endpoint costs.

### Fixed

- The in-process collector's `--mix` fell through to the parent
  checkpoint for any name but `greedy`; both collectors now share one
  builder.

## 0.9.2

### Fixed

- `benchmarks.rank` and `benchmarks.sibling` no longer freeze one chance
  outcome per sibling: a `BUY_DEV_CARD`, `PLAY_KNIGHT` or `MOVE_ROBBER`
  row now scores as the mean over `--chance-draws` draws (default 8),
  matching what the search itself sees.
- `--chance-draws 1` restores the old single-draw behavior exactly.

### Added

- `catan.mcts.draws_hidden` and `.sampled_children`, now public so the
  probes share the tree's own chance semantics.
- `benchmarks.rank.head_row`, `.Row`, `.share`, `.lane_plan` and
  `.chance`, and `benchmarks.sibling.Spread.chance_children`/
  `.chance_spread`.

## 0.9.1

### Fixed

- `catan.mcts` no longer freezes `MOVE_ROBBER`, `PLAY_KNIGHT` and
  `BUY_DEV_CARD` children: they now share `ROLL`'s `_Chance` slot, keyed
  by outcome, so repeated visits average into one subtree instead of
  one frozen draw.

### Added

- `catan.actions.victim_of` (was `_victim`), now public.
- `catan.mcts.HIDDEN_DRAW`, the three action types whose `apply`
  resolves a hidden card.

## 0.9.0

### Added

- A `quantile` value head: `ModelConfig(value_head="quantile")` widens
  the linear head to `players x quantiles` outputs; its mean still
  feeds `V` everywhere unchanged. The full tensor is exposed via
  `Prediction.quantiles`.
- The matching quantile Huber value loss in `catan.ppo.minibatch_terms`,
  at Huber width `1/30`.
- `Stats.value_mse`/`Terms.value_mse`: the plain squared error of the
  mean, for comparing runs across value-head shapes.
- `catan.league --value-head`/`--quantiles`, and `catan.train
  --quantiles`. Warm-starting from a scalar base seeds every level from
  its output.

### Fixed

- `CatanNet._emit` now builds its four heads in a fixed order, since the
  order affects float-level gradient accumulation.

## 0.8.0

### Added

- Board-paired advantage baselines: `Collector(pair_boards=True)` deals
  games `2k`/`2k+1` on the same board; `PPOConfig(pair_baseline=True)`
  subtracts the mate game's reward as a control variate. `catan.league
  --pair-boards` enables both; off, behavior is bit-identical to
  `0.7.2`.
- `benchmarks.noise_scale --paired`: runs the estimator on one
  board-paired cohort with both the raw and pair-adjusted advantage.

## 0.7.2

### Fixed

- `run.manifest.freeze` read git provenance after creating the run
  directory, so `git_dirty` was always true. Now read first.

## 0.7.1

### Added

- `collect.league_caster` takes an `order` permutation of learner ids,
  exposed as `catan.league --learner-order`, varying table adjacency
  independent of seat balance.

## 0.7.0

### Added

- `catan.league`: N learners share every game in one directory, each
  with its own `PPOConfig` overrides; `standings` rates the run by
  `learner0`.
- `catan.run`: a run is a directory with a frozen manifest (`run.json`);
  `load` reconstructs the invocation from it.
- `catan.export_onnx`: `.pt` to `.onnx`, behind a new `export` optional
  dependency.
- A rolling checkpoint ring (`prune_recent`) and blowout preservation
  (`preserve_blowout`, keeps the pre-update weights and offending
  batch).
- Per-seat PPO overrides: `adam_eps` and an entropy-controller `gain`.
- `selfplay.owned` and a `learners` gate on `Collector`, for several
  learners recording from one game.

### Fixed

- `catan.distill_train` was unbuildable: its manifest no longer matched
  its parser.

## 0.6.1

### Changed

- Duels are antithetically paired by default (`train.versus`,
  `arena.compete`): every board plays both seat assignments, averaged.
  `antithetic=False` restores the old behavior.
- Duels take a different seed per pair rather than sharing one
  `--duel-seed`.

### Fixed

- `benchmarks.duel` seeds a stochastic entrant from its spec, not its
  argument position, so a swapped duel measures the swap.
- `benchmarks.duel` writes a verdict by default, and no longer
  double-writes it under `--json`.
- `arena.wilson`'s upper bound at `p = 1` no longer excludes its own
  point estimate.
- A net now rebuilds from its checkpoint's own head shapes
  (`catan.model.config_from_args`), fixing `--policy-head mlp` loading
  in two benchmarks.

## 0.6.0

### Added

- `--critic {gae,none,aux}`: the value head's route into training as one
  flag.
- `--kl-break`: a one-sided ceiling on a finished epoch's mean KL.
- `--fused`: dense GEMM trunk gather/scatter, an opt-in for the GPU update.
- A flat wire format for worker episodes: byte-identical rebuild from a
  few large arrays.
- `--rival`: every eval also duels a rival run's checkpoint at the
  matched iteration.
- `--detach-value` on `catan.train`.
- `benchmarks.generate --bot network:<path>` records self-play for
  `benchmarks.behaviour`.

### Changed

- Evaluation now gates on the matched-rival duel; `greedy` is a
  mix-exploitation canary.

## 0.5.0

### Added

- `DistillConfig.contested_only`/`.hard_target`: train only on rows the
  search overruled, toward its argmax.
- `DistillConfig.anchor`: cross-entropy toward the recorded prior on
  rows `contested_only` zeroed.
- `DistillConfig.stake_scale`: weights a contested row by its Q-gap,
  `min(gap / stake_scale, 1)`.
- `DistillConfig.buffer_iterations`/`.refresh_prior`: train on several
  iterations, refreshing the filter against the live policy.
- `DistillConfig.pack_contested`: dense minibatches for the policy term
  alone.
- `ModelConfig.value_head`/`.policy_head`: pluggable readout shapes
  (`linear`, `mlp`, `pooled`, `mlp_pooled`, `attn`), exposed as
  `--value-head`/`--policy-head`. Both default to `linear`.
- `benchmarks.head_shape`: sweeps readout shapes on a frozen trunk.
- `catan.distill_train --collect-workers`: shards the searched collector
  across processes.
- Distillation stats split by contested/settled rows, plus entropy and
  anchor loss.

### Changed

- `catan.ppo`/`catan.train` can cut the value loss off the trunk in one
  flag.
- `benchmarks.rank` gained a head learning-rate sweep.

### Fixed

- `legal_actions` re-enumerated a trade the table had already declined
  this turn; `Game.offered` now tracks bundles offered this turn.
- `--learning-rate` is honored on resume in the distillation trainer.
- The attention head's pooling query is built off the head, not the
  trunk.
- The from-scratch benchmark configurations deal 128 lanes, not 512.
- The zero-sum check tolerates float32.
- Valued corpora are collected from the parent checkpoint, not the
  checkpoint being trained.

## 0.4.0

### Added

- `catan.collect`: self-play collection sharded across worker processes
  with CPU inference.
- `catan.ddp`: PPO update data-parallel across CPU workers, behind
  `--update-workers`.
- `catan.schedule`: learning rate driven by `approx_kl` rather than a
  fixed schedule.
- `catan.selfplay.Collector.cohort`/`catan.train --collect-mode`: deal a
  fixed block of games to completion, instead of refilling lanes on the
  fly. `--async-collect` now requires `stream` mode.
- TD(λ) value targets behind `--lam`.
- Opponent mixing in the collector, and a frozen evaluation ladder
  including `search2-offers3`.
- `catan.ppo.Terms` carries `policy_term`/`value_term`/`entropy_term`
  separately.
- `benchmarks.noise_scale`: gradient noise scale (McCandlish et al.
  2018) from one collected batch.
- `benchmarks.duel`: two checkpoints head to head, paired terminal VP.
- `benchmarks.minibatch_iso_kl`: the learning rate holding step length
  constant across batch sizes.
- `benchmarks.rank`: whether the value head orders siblings correctly.
- `benchmarks.training_loop`: production-shape sync/async PPO timing.
- `catan.arena` takes `network:<path>`, `netsearch:<path>` and
  `netgreedy:<path>`; `network:<path>@<offers>` sets an offer budget.
- `catan.train` prints its device/worker counts and warns if crippled.

### Changed

- `catan.selfplay.Collector` encodes a worker tick as one packed NumPy
  batch for CPU inference.
- Board-template and topology lookups are cached, cutting recursive
  hashing cost.
- `catan.encoding._template`'s cache is 4096 boards deep.
- `--async-collect` overlaps collection with the GPU update, training
  on a policy one iteration stale.
- `benchmarks.duel` defaults `--workers` to 26 unless both sides are
  bare networks.
- `lam` and `minibatch` are columns in `log.jsonl`.
- The devcontainer installs Python.

### Fixed

- `--resume` was silently discarding `--learning-rate`.
- `--resume` with no checkpoint started fresh in silence.
- The RNG state is restored as a CPU tensor on a cuda resume.
- The KL gauge reported a negative divergence on on-policy batches.
- Log scalars are now detached in `minibatch_terms`.
- `--workers` could not duel two checkpoints against each other.
- A duel needs an explicit torch thread count under `--cpus`.
- The lambda sweep oversubscribed update workers.
- `summarise` stays compatible with `distill_train`.
- `tests/test_duel.py` imported torch at module scope, failing the
  whole suite on a torch-free box.

## 0.3.0

### Added

- `catan.placement`: an opening prior over pip count, resources
  reached, and scarce-resource access, fit by conditional logit.
- `benchmarks.placement_policy`: compares an entrant's setup picks
  against the prior's ranking.
- `Weights.scarce`: scarce resources reached, the only weight not fit
  against the engine.
- `board.scarce_resources`: resources with fewer hexes than the
  commonest.

### Changed

- Arena entrants can be constructed by name.
- Every arena number recorded before this release used `scarce` at
  zero.

## 0.2.0

### Added

- `catan.distill`: distills search visit counts into the policy, with
  Dirichlet root noise and a bootstrapped value target.
- `catan.distill_train`: the expert-iteration training loop.
- `benchmarks.expert_scale`: how synchronized expert collection scales.
- `benchmarks.horizon`: what shortening the value horizon removes.
- `LeafEvaluator` supports fixed-shape leaf inference.

### Changed

- Search leaves are now batched across games rather than evaluated one
  at a time.
- Hidden deck shuffles are deferred until a draw actually needs them.
- Linear PUCT edge scores are now cached.
- The MCTS wave size is a named quantity, not an inline constant.

### Fixed

- A loaded network is placed on the requested device.
- Dirichlet root noise defaults off.

## 0.1.0

### Added

- `catan.board.coords`: cube hex coordinates, neighbours, distance, and
  hexagonal layout generation.
- `catan.board.topology`: vertices, edges and adjacency, canonically
  keyed so disconnected and touching islands are supported.
- `catan.board.terrain`: resource and terrain types, including sea and
  gold for Seafarers.
- `catan.board.board`: terrain and number tokens, the official setup
  bags, and the rule keeping 6 and 8 off adjacent hexes.
- `catan.board.maps`: base and mini layouts, plus multi-island layout
  construction.
- `catan.board.ports`: coastline-derived port placement.
- `catan.state`: occupancy, hands, bank stock, and layout-agnostic
  placement legality.
- `catan.economy`: build costs, port-rate bank trades, and the official
  production-shortage rule.
- `catan.roads`: longest road as a longest trail.
- `catan.cards`, `catan.devcards`: the 25-card deck and its four
  playable effects.
- `catan.robber`: robber movement, hand-weighted stealing, and seven
  discards.
- `catan.victory`: victory points, longest road and largest army.
- `catan.game`: the turn/phase machine — setup, rolling, discarding,
  the robber, and win detection.
- `catan.actions`: a flat, board-sized action space with legality
  masking.
- `catan.play`: a random player for full games.
- `catan.evaluate`: handcrafted per-seat position scoring across nine
  ablatable weighted terms.
- `catan.bots`: a `Bot` protocol, a random bot, and max^n search
  (`greedy` is the one-ply case).
- `catan.game.imagine`: a hypothetical-play copy with its own RNG and a
  reshuffled deck.
- `catan.game.to_move`: whose decision it is, not always the current
  player.
- `catan.arena`: rotated head-to-head play with Wilson intervals and
  paired per-game scoring.
- `catan.tuning`/`benchmarks.tune`: fits evaluation weights by hill
  climbing through the arena.
- `benchmarks.production_curve`: tests whether a weight is identifiable
  from self-play at all.
- `catan.evaluate_tiered`: a second, magnitude-tiered evaluation, kept
  as a comparison baseline (`*-tiered` presets).
- `catan.encoding`: the seat-rotated graph observation the model reads.
- `catan.selfplay`: a vectorised, seat-demultiplexed rollout collector
  behind a torch-free `BatchPolicy` protocol.
- `catan.rewards`: zero-sum terminal points scaled to the winning
  margin.
- `catan.policy`: the torch `BatchPolicy`; `PROPOSE_TRADE`'s `log_prob`
  is the joint over slot and offer.
- `catan.ppo`: GAE, clipped surrogate, value loss and entropy bonus; the
  value head trains unbootstrapped on terminal outcomes.
- `catan.train`: the resumable training loop, with atomic checkpoint
  writes and a fixed eval baseline.
- `catan.netbot`: a trained checkpoint as a `catan.bots.Bot`.
- `catan.mcts`: PUCT with batched leaf waves, a per-seat value backup,
  and sampled chance nodes.
- `catan.expert`: `SearchPolicy`, a searched `BatchPolicy` recording raw
  visit counts as a `Target`.
- `benchmarks.throughput`: games/sec measurement with environment
  recording.
- `benchmarks.baselines`: runs a lineup and records the commit and
  environment with the result.
- `benchmarks.rollout`: ticks/sec and actions/sec for the collector
  under a trivial policy.
- `benchmarks.value_head`: what the value head explains, split by stage
  of the game.

### Changed

- `catan.actions.Action` carries an `ask` order on `PROPOSE_TRADE`,
  naming a preferred acceptor (`SearchBot(partner_choice=True)`).
- `catan.trading.responders` orders an offer from the proposer, not
  ascending seat index.
- `relative` is now the default stance for `greedy` and `search2`
  (`*-own` presets restore the old default).
- `benchmarks.throughput.environment` reports a dirty tree and resolves
  git correctly inside the devcontainer.
- `catan.tuning` fits either evaluation and takes a stance.
- `benchmarks.ablate` takes an `--evaluator`.
- `catan.bots.SearchBot` takes a `stance` (`own`, `relative`,
  `paranoid`) for turning its per-seat vector into one number.
- `catan.arena` entrants carry which evaluation to score with.
- `catan.game.roll_dice` takes an optional explicit roll.
- `catan.arena` entrants are a frozen `Entrant` (`FACTORIES` →
  `PRESETS` and `spawn`), so `compete` can fan out over a process pool.
