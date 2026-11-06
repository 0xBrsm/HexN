# The networks, checkpoints, and the ONNX contract

## Contracts

"Contract N" names the observation a network reads, the action space it
answers over, and the shape of the ONNX graph `hexn.export_onnx` writes for
it. HexSet's `docs/onnx.md` defines every contract its server serves and is
the source of truth for them. This package trains two networks:

| Contract | Network | Reads | Answers |
| --- | --- | --- | --- |
| `8`, `8.1` | `hexn.model.HexNet` | `hexset.encoding`'s hex, vertex, edge and global inputs | One flat action a decision |
| `9`, `9.1`, `9d`, `9d.1` | `hexn.model9.HexNet` | `hexset.contract9`'s seat, table, hex and vertex tokens | A whole decision (`hexset.decisions`) |

A trainer's `--contract` picks one (`8`, `9`, `9.1`, `9d` or `9d.1`; `9d`
and `9d.1` with `--players 2`). A contract-8 net becomes an 8.1 net by
loading 8.1 weights ([loading across contracts](#loading-across-contracts)).
Module docstrings cite older numbers as shorthand for "before/after that
shape changed":

- **Contract 5** deleted the public trade layer
  (`hexset.trading`): trades clear from each seat's own published
  valuation, which is observation, never a network action. The flat
  categorical policy stopped carrying a `PROPOSE_TRADE` slot with give/want
  heads riding alongside it.
- **Contract 6** carries two further changes. The trading redesign deletes
  the public layer's valuation columns from the encoding, since there is
  no seat vector left to publish. A knight-move fix collapses
  `PLAY_KNIGHT`'s per-hex-and-victim block down to one slot: robber
  placement now runs through the same `MOVE_ROBBER` decision a rolled seven
  already uses. Contract 6 also reinterprets the value head, a points margin
  before, as a win head (a win-probability softmax over seats), and adds the
  auxiliary margin head as a warm-started copy of the old value
  head's weights.
- **Contract 7** appended each seat's roads to take Longest Road and a
  one-hot of the seat to move to the globals.
- **Contract 8** re-encodes every input: opponents' cards as expected
  counts, card ages and the odds of the unseen cards; awards and
  each vertex's reach as closeness; buildings by the points they score
  (`hexset.encoding`'s module docstring, HexSet's changelog).
- **Contract 8.1** reads every seat's points against the points the game is
  played to, and appends 15 globals (90 at four seats): the table's rules
  (the points the game is played to, the discard limit, the friendly robber,
  the balanced dice) and each sum's chance on the next roll. It is the
  contract of a net trained across rule sets.
- **Contract 9** encodes every seat the same way, as a seat token, and the
  board as hexes and vertices, with a building's or road's owner a pointer
  to its seat's token; the network answers a whole decision.
- **Contract 9.1** appends a live-seat flag to each seat token and the
  table's rules and next-roll odds to the table token.
- **Contract 9d** is the two-seat observation: each hand exact, only the
  development deck hidden. **Contract 9d.1** appends each seat's production
  and bank rates to its token.

## The contract-8 network (`hexn.model`)

`HexNet` runs message passing directly over the encoded graph (hexes,
vertices, edges, and a global token that reads and writes every type),
using a fixed adjacency (`hexset.encoding.StaticGraph`). A batch is dense
`(B, nodes, features)` against one shared index set, not a block-diagonal
union of graphs. Board layout is fixed at construction: a different layout
needs a different model.

Every policy head emits the widths `hexn.readout` declares, scattered into
the flat action space by its plan, so legality is a per-node property
carried through from the encoder.

`ModelConfig` fields:

| Field | Meaning |
| --- | --- |
| `width` | hidden width (64) |
| `rounds` | message-passing rounds (2) |
| `value_head` | `"linear"`, `"mlp"`, `"pooled"`, `"mlp_pooled"`, `"attn"`, or `"quantile"` |
| `policy_head` | `"linear"` or `"mlp"` |
| `quantiles` | width of the quantile head's per-seat distribution (inert under every other head shape) |
| `fast_win` | build the fast-win head (`Prediction.fast`); off |
| `contract` | `"8"`, or `"8.1"` for a net trained across rule sets; a checkpoint carries its own in its weights |

The defaults (`"linear"`/`"linear"`) rebuild every checkpoint written
before these fields existed with the same modules under the same
`state_dict` keys, which `test_model` pins.

`Prediction` (the network's output):

| Field | Meaning |
| --- | --- |
| `logits` | flat action-space logits; mask with `legal_mask` before sampling |
| `value` | `softmax(value_logits)` over the seat axis: each seat's own win probability, summing to one across `players` |
| `value_logits` | the same head's pre-softmax output, one number per seat, used for the cross-entropy loss |
| `margin` | the auxiliary VP-margin head: a second linear read of the same trunk, trained only when `--aux-margin-weight > 0`; never read by a gate, an advantage, or a search |
| `fast` | the fast-win head, `None` unless built: a softmax over `players + 1` slots, the seats and then "not won in time", so `fast[..., :players]` is each seat's expected time-weighted win |
| `quantiles` | `(B, players, Q)` per-seat quantile tensor, present only under the `"quantile"` value head; its mean over the last axis is `value_logits` |

There is no trade head. A seat's private gate is read off the value head
(`hexset.clients.netbot.NetworkBot`).

## The contract-9 network (`hexn.model9`)

`hexn.model9.HexNet` reads `hexset.contract9`'s observation: a seat token per
seat (the perspective first), a table token, a token per hex and per vertex,
with a building's and a road's owner as a pointer to a seat token and each
seat's reach as a weighted seat-vertex link. Roads are vertex-vertex links:
there are no edge tokens. Each round passes messages between hexes and
vertices, along roads and neighbours, between vertices and the seats that
build on or reach them, and, under `attention`, among the seats, with one
set of weights for every seat.

The win head reads each seat's token with one shared head, softmaxed over
the seats, the perspective first. The policy scores the flat actions (a road
from its two ends, a robber move from the hex and the victim's token) and
`hexset.decisions`' compound decisions as wholes: a setup settlement with
its road, the snake seat's double turn, a Road Building card with its roads,
a whole discard. A decision's probability is the softmax over every legal
option, compound and ordinary together.

`ModelConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `width` | 64 | hidden width |
| `rounds` | 3 | message-passing rounds |
| `heads` | `"linear"` | every head one linear layer, or `"mlp"`, each with a hidden layer |
| `discard` | `"mlp"` | the discard head's shape apart from the others: an MLP lets cards in a kept hand count together |
| `norm` | `True` | LayerNorm on every token before each round's update and before the heads |
| `messages` | `"separate"` | each message a token reads is an input of its own, or `"pooled"`, a kind's messages summed into one |
| `attention` | `True` | one attention over the seat tokens each round |
| `others` | `False` | each seat's update also reads the other seats' mean token |
| `global_seat` | `False` | the global head (bank trades, buying a card, development plays, ending the turn) reads the perspective's own seat token beside the table's |
| `points` | `False` | a points head beside the win head: one number per seat, the perspective first (`Prediction.points`) |
| `double_turn` | `"joint"` | how the double turn is chosen: `"joint"`, one decision over every ordered pair of setup turns, or `"parts"`, the setup head twice, the second on the board after the first |
| `contract` | `"9"` | `"9"`, `"9.1"`, `"9d"` or `"9d.1"` |

A checkpoint carries its `model_config`; `config_of` reads it, a field the
checkpoint predates at the value it was trained without. The trainers'
flags `--contract`, `--heads`, `--discard-head`, `--norm`, `--messages`,
`--attention`, `--others`, `--global-seat` and `--points-head` set these, and
`--width` and `--rounds` the first two ([training.md](training.md#contract-9-training)).

## Loading across contracts

A checkpoint loads into a net of a later contract of its own family, the new
input weights at zero, so the loaded net plays as it did:

- 8.1 weights loaded onto a contract-8 net make it an 8.1 net; contract-8
  weights loaded onto an 8.1 net are widened with zero columns, which change
  only how points are read against a game played to more than 10.
- A contract-9 checkpoint loads into a 9.1 net, and a 9d checkpoint into a
  9d.1 net. A checkpoint without `global_seat` loads into a net with it, its
  head's seat weights at zero.
- A `"joint"` checkpoint initialises a `"parts"` net without its first,
  second and pair heads (`model9.without_joint_head`).

A checkpoint whose input projections read another contract's widths is
refused when it is loaded, naming the widths.

## Migrating a contract-7 checkpoint onto contract 8 (`hexn.migrate_c8`)

```bash
python -m hexn.migrate_c8 --checkpoint runs/example/iter-00050.pt \
    --out runs/example/iter-00050-c8.pt --runtime <module> --entrant <bot>
```

Contract 8 re-encodes every input, so the four input projections
(`embed_hex`, `embed_vertex`, `embed_edge`, `embed_global`) change width and
every other parameter is copied. Each projection is refitted by least squares
so that its output on the contract-8 inputs matches the old projection's on
the contract-7 inputs of the same positions, from every seat. Where contract
8 restates contract 7 linearly (terrain with has-token and the desert, edges,
ports, the summed counts) the fit is exact; where it does not (building points
against the building and owner one-hots, the award reads against the holder
one-hots, expected hands against the ledger) it is the best linear fit on the
positions given. Contract 8's new information (`seat_reach`,
`own_dev_playable`, `opponent_dev_ages`, `dev_odds`) starts
at zero weight.

Positions come from `--games` seeded games (default 64) among four seats of
`--entrant` (an arena name, with the `--runtime` that registers it; `random`
plays uniformly), every `--stride` decisions (default 7), from every seat. A
quarter of the games are held out. The report, printed and stored under the
checkpoint's `migrate_c8` key, gives each fit's largest residual and, on the
held-out games, each projection's and each output's largest difference
between the old net on contract 7 and the migrated net on contract 8. The
optimiser state is not carried: a migrated net starts a run, it does not
resume one. A checkpoint not on contract 7's widths is refused; `--out` must
not already exist unless `--force`. Its games' seed is an `rl` training seed
of the seed policy (`--seed`, `--bypass-seed-policy`).

`hexn.migrate_c8` is the only migration. Contracts 8.1, 9.1 and 9d.1 load
their predecessors' weights directly (above), and contract 9 is a different
network, trained from its own start.

## Widening a checkpoint (`hexn.widen`)

```bash
python -m hexn.widen --checkpoint runs/base-run/latest.pt --width 128 \
    --noise 0.01 --seed 0 --out runs/wide-128/latest.pt
```

Net2WiderNet (Chen, Goodfellow, & Shlens, 2016): every hidden unit at the
old width `d` is copied to fill the new width `D` (unit `j` of the wide net
is a copy of unit `j % d`), and every layer that *reads* the hidden vector
divides its weight on each copy by that unit's copy count, so the sum over
the wide vector equals the sum over the narrow one. The result computes the
same function as the source, to float precision; `widen_state` checks this
on real observations before anything is written. It widens a contract-8
`HexNet`, on the contract its weights carry.

`hexn.migrate_c8` carries a checkpoint across a change to the
*observation's* shape; widening grows the *model's* own hidden width with
the observation unchanged, so a wider net can warm-start from a narrower
checkpoint's exact function.

`--noise σ` adds Gaussian noise (σ × each copied row's RMS) to the copies'
incoming weights only, so the copies are not perfectly identical units that
would receive identical gradients forever. A wide net built at `--noise 0`
computes the same function as the narrow net, at a larger compute cost. The
tool reports the resulting max output deviation and mean policy KL against
the source, so the noise's effect on the function is measured. The widened checkpoint carries the parent's `iteration`,
`games_started`, and `torch_rng` (so `--resume` continues the same game
count and RNG stream a same-width continuation would), a fresh Adam state
(the parent's moments have the old shapes), and a `widen` block naming the
source path, its sha256, both widths, the noise, and the seed.

## Loading a checkpoint at runtime (`hexn.netbot`)

`hexn.netbot.load(path, topology, device, compile_mode)` rebuilds the net
from the checkpoint. A contract-8 checkpoint is rebuilt from its recorded
`args` (width, rounds, and head shapes all default to what a checkpoint
predating those fields was trained with) and wrapped in a greedy
`NetworkPolicy` (`greedy=True`, taking the argmax). A contract-9 checkpoint
is rebuilt from its `model_config` and played by `hexn.policy9.Policy9`,
which chooses each compound decision whole at its first move and plays the
rest from its plan (`hexset.decisions.Plans`), lists a decision's whole
options (`option_rows`), and makes no offers of its own; its gate still
answers the table's. Sampling is the behaviour distribution PPO trains on;
a scored policy plays the argmax. The result is cached per
`(path, topology)` (`lru_cache(maxsize=4)`), not by file mtime: a `.pt`
under `runs/` is an immutable per-run artifact, unlike a served
`models/*.onnx`, which is replaced by name.

`register_entrants(load)` runs at import time, and is how a checkpoint
becomes playable through `hexset.bench.duel --runtime
hexn.netbot network:<path> ...` (or `mcts:<path>`, `rollout:<path>`) or any
other `hexset.arena.spawn` caller; `hexset.arena` itself never imports torch
or this module.

How a loaded checkpoint is played (the bot, leaf evaluation, search, and
trade gate) is runtime-agnostic and lives in `hexset.clients.netbot`, over
the `hexset.clients.policy.Policy` protocol. `hexn.netbot` only supplies
the torch-specific half: finding the file and rebuilding the net.

## Exporting to ONNX (`hexn.export_onnx`)

```bash
python -m hexn.export_onnx --checkpoint runs/example/latest.pt --out model.onnx
python -m hexn.export_onnx --checkpoint runs/example/latest.pt \
    --out mcts256.onnx --search mcts --simulations 256
python -m hexn.export_onnx --checkpoint runs/example/latest.pt --out served.onnx \
    --runtime <module> --trader <bot> --trader-veto --max-offers 1 --trade-floor 0.03
```

Needs `torch` (not a declared dependency; see below) plus `onnx` and
`onnxruntime`, which are also not declared in this package's
`pyproject.toml`. Install them separately (`pip install onnx onnxruntime`)
before running this module.

Traces the reconstruction path `hexn.netbot.load()` uses (same shapes,
same `state_dict` keys) once with `torch.onnx.export`, and
embeds the facts HexSet's `hexset.clients.onnxbot.load()` reads as ONNX
metadata. `python -m hexn.export_onnx` dispatches on the checkpoint's own
contract: a contract-8 or 8.1 checkpoint exports a record graph, a
contract-9 one a decision graph (`hexn.export_onnx9`).

### A contract-8 or 8.1 record graph

- **Inputs**: a dynamic leading batch axis, named as
  `hexset.onnx_record.RECORD_FIELDS` names them, in its order: the 27 a
  contract-8 graph reads (all but `knights_played`, `longest_road_holder`,
  `largest_army_holder`, `hand_totals`, `ledger_known` and
  `ledger_unknown`, which only contracts 6 and 7 read), or those and the
  five rule and dice fields, 32, for an 8.1 net. All `int64` except
  `expected_hand`, `dev_ages` and `roll_odds` (float32) and `action_mask`
  (bool).
- **Outputs**, in order: `action_index` (`(B,)` int64, the argmax over the
  masked distribution), `prior` (`(B, space.size)` float32, normalised over
  legal entries, zero elsewhere), and `value` (`(B, players)` float32,
  board-seat order, already un-rotated: each seat's own win probability).
- `contract` is `"8"`, or `"8.1"` (`hexset.onnx_record.CONTRACT_VERSION`)
  for an 8.1 net only.

### A contract-9 decision graph (`hexn.export_onnx9`)

- **Inputs**, each with a dynamic leading axis, named as HexSet supplies
  them: `observation` (`(B, width)` float32, `hexset.contract9.flatten`
  rows), the decision's masks `flat` and `setup`, its Road Building
  candidates `road_building` with `road_building_row`, and its discards
  `discard` with `discard_row` (`hexset.decisions.arrays`); and, for a
  `"joint"` net, `double_rows`. A `"parts"` net declares no `double_rows`,
  so HexSet asks it the double turn as two setup turns.
- **Outputs**: `decision` (`(B, 2)` int64, each row's `(kind, index)` over its
  decision, chosen in the graph by `hexn.policy9.best`); `value`
  (`(B, players)`, the win head in the row's own frame); `prior` (the softmax
  over every legal option marginalised onto its first atomic move,
  `hexn.model9.first_move_prior`, so a file can be searched); every whole
  option's probability (`option_flat`, `option_setup`, `option_double` for a
  `"joint"` net, `option_road_building`, `option_discard`;
  `hexn.model9.option_probs`), so paired rollouts can search whole
  decisions; and, for a net with a points head, `points`, which HexSet does
  not read.
- `contract` is the net's own (`9`, `9.1`, `9d` or `9d.1`), `double_turn`
  describes the net, and `max_offers` is `"0"` unless asked for: a contract-9
  net reads no trading inputs and makes no offers of its own.

HexSet's `docs/onnx.md` gives both graphs' inputs and outputs in full.

### What every export writes

- **Metadata props** HexSet reads: `contract`, `players`, the topology
  fingerprint (`num_hexes`/`num_vertices`/`num_edges`), `max_offers` (the
  run's recorded offer budget for a contract-8 net, empty for none; a file
  still carrying the older `max_trades` key is read as it), and `iteration`.
  Only when asked for on the command line: `search=mcts` with
  `simulations`/`wave`, `gate_plies` (the served trade gate's own look-ahead
  budget, independent of `--search`), and the serving keys below. A
  contract-8 export also writes `trader` when the run trained with one.
- **Provenance props**, read by tooling afterward, not at runtime:
  `source_checkpoint`, `checkpoint_sha256`, `exported_at`, and
  `exporter_commit` when determinable. The digest pins the weights and the
  commit the code, so an artifact can be reproduced from itself.
- **Not written**: `name`, `modes`, `version` and `commit`. HexSet's server
  lists a file only by the `name` it declares, so name a file for its
  picker afterwards with `python -m hexset.clients.onnxmeta stamp`, which
  copies the graph unchanged.

The serving flags, written as HexSet's metadata keys of the same names
(`hexn.export_onnx9.serving_props`), for either contract:

| Flag | Writes |
| --- | --- |
| `--search mcts`, `--simulations`, `--wave` | `search`, `simulations`, `wave` |
| `--gate-plies N` | `gate_plies` |
| `--trader <bot>` | `trader`: a bot, as a lineup names it, registered by a `--runtime` module, that answers the file's trades while the network plays every move. The server that loads the file needs the same runtime |
| `--max-offers N` | `max_offers`, over the recorded budget |
| `--trade-floor F` | `trade_floor`, in [0, 1] |
| `--trader-veto` | `trader_veto`: the file's own gate vetoes its trader. Needs `--trader` and `--max-offers` above 0 |
| `--gate-own-frame 0\|1` | `gate_own_frame` |
| `--robber-tip F` | `robber_tip`, in [0, 1] |

A numerical parity check runs after every export, on real seeded
positions, and a mismatch raises (a nonzero exit): an export bug that
degrades quality costs less to catch here than on a served board. A
contract-9 file is held to
`Policy9` on positions of every kind of decision: the same `decision`,
`value` and `prior` within tolerance, the same moves played through
`hexset.clients.onnxbot`, and the same whole options. The stochastic
sampling path is not exported: every file plays argmax. The runtime target
is onnxruntime ≥ 1.18 on the CPU execution provider, opset 18 (`--opset`).

`--out` defaults to the checkpoint's own path with a `.onnx` suffix.

## Why torch is not a dependency

`pyproject.toml` declares `hexset` and `numpy`, and nothing else. Torch is
provisioned separately (a ROCm or CUDA build matched to the training
hardware), so that everything that does not need it stays installable and
testable without it: `hexn.readout`, `hexn.selfplay`, `hexn.rewards`,
`hexn.schedule`, `hexn.run`, `hexn.durable`, and the torch-free half of
the test suite. Every module that trains, evaluates, exports, or migrates a
network imports torch lazily, or requires it in that module alone.
