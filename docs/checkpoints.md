# The network, checkpoints, and the ONNX contract

## The network (`hexn.model`)

`HexNet` runs message passing directly over the encoded graph — hexes,
vertices, edges, and a global token that reads and writes every type —
using a fixed adjacency (`hexset.encoding.StaticGraph`). A batch is dense
`(B, nodes, features)` against one shared index set, not a block-diagonal
union of graphs. Board layout is fixed at construction: a different layout
needs a different model.

Every policy head emits exactly the widths `hexn.readout` declares,
scattered into the flat action space by its plan. Legality is therefore a
per-node property carried through from the encoder, rather than a mask
bolted onto a flattened board afterward.

`ModelConfig` fields:

| Field | Meaning |
| --- | --- |
| `width` | hidden width |
| `rounds` | message-passing rounds |
| `value_head` | `"linear"`, `"mlp"`, `"pooled"`, `"mlp_pooled"`, `"attn"`, or `"quantile"` |
| `policy_head` | `"linear"` or `"mlp"` |
| `quantiles` | width of the quantile head's per-seat distribution (inert under every other head shape) |

The defaults (`"linear"`/`"linear"`) are load-bearing: every checkpoint
written before these fields existed has to rebuild the same modules under
the same `state_dict` keys, which `test_model` pins.

`Prediction` (the network's output):

| Field | Meaning |
| --- | --- |
| `logits` | flat action-space logits; mask with `legal_mask` before sampling |
| `value` | `softmax(value_logits)` over the seat axis — each seat's own win probability, summing to one across `players` |
| `value_logits` | the same head's pre-softmax output, one number per seat, used for the cross-entropy loss |
| `margin` | the auxiliary VP-margin head — a second linear read of the same trunk, trained only when `--aux-margin-weight > 0`; never read by a gate, an advantage, or a search |
| `quantiles` | `(B, players, Q)` per-seat quantile tensor, present only under the `"quantile"` value head; its mean over the last axis is `value_logits` |

There is no trade head: a seat's private gate is read straight off the value
head (`hexset.clients.netbot.NetworkBot`), never a head of its own.

## Contract versions

"Contract N" is this package's shorthand for the observation/encoding
version: the record fields a network reads, the action space it emits over,
and the metadata `hexn.export_onnx` writes into a served file. HexSet's own
`docs/onnx.md` defines the currently served contract
(`hexset.onnx_record.CONTRACT_VERSION`) and is the source of truth for it.
This package's module docstrings cite specific numbers as shorthand for
"before/after that shape changed." The most recent are:

- **Contract 5** deleted the public trade layer outright
  (`hexset.trading`): trades clear from each seat's own published
  valuation, which is observation, never a network action. The flat
  categorical policy stopped carrying a `PROPOSE_TRADE` slot with give/want
  heads riding alongside it.
- **Contract 6** carries two further changes. The trading redesign deletes
  the public layer's valuation columns from the encoding entirely, since
  there is no seat vector left to publish. A knight-move fix collapses
  `PLAY_KNIGHT`'s per-hex-and-victim block down to one slot: robber
  placement now runs through the same `MOVE_ROBBER` decision a rolled seven
  already uses. Contract 6 also reinterprets the value head as a win head
  (a win-probability softmax over seats) rather than a points margin, and
  adds the auxiliary margin head as a warm-started copy of the old value
  head's weights.
- **Contract 7** appended each seat's roads to take Longest Road and a
  one-hot of the seat to move to the globals.
- **Contract 8** re-encodes every input: opponents' cards as expected
  counts, card ages and the odds of the unseen cards; awards and
  each vertex's reach as closeness; buildings by the points they score
  (`hexset.encoding`'s module docstring, HexSet's changelog).

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
not already exist unless `--force`.

A checkpoint whose input projections read another contract's widths is
refused when it is loaded into a `HexNet`, naming the widths.

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
on real observations before anything is written.

This is a different, unrelated transform from `hexn.migrate`: migration
carries a checkpoint across a change to the *observation's* shape, while
widening grows the *model's* own hidden width with the observation
unchanged, so a wider net can warm-start from a narrower checkpoint's exact
function.

`--noise σ` adds Gaussian noise (σ × each copied row's RMS) to the copies'
incoming weights only, so the copies are not perfectly identical units that
would receive identical gradients forever. A wide net built at `--noise 0`
computes the same function as the narrow net, at a larger compute cost. The
tool reports the resulting max output deviation and mean policy KL against
the source, so the noise's effect on the function is visible rather than
assumed. The widened checkpoint carries the parent's `iteration`,
`games_started`, and `torch_rng` (so `--resume` continues the same game
count and RNG stream a same-width continuation would), a fresh Adam state
(the parent's moments have the old shapes), and a `widen` block naming the
source path, its sha256, both widths, the noise, and the seed.

## Loading a checkpoint at runtime (`hexn.netbot`)

`hexn.netbot.load(path, topology, device, compile_mode)` rebuilds a
`HexNet` from the checkpoint's own recorded `args` — width, rounds, and
head shapes all default to what a checkpoint predating those fields was
trained with — and wraps it in a greedy `NetworkPolicy` (`greedy=True`,
taking the argmax). Sampling is the behaviour *distribution* PPO needs
during training; it is not the policy worth scoring.
The result is cached per `(path, topology)` (`lru_cache(maxsize=4)`), not by
file mtime: a `.pt` under `runs/` is an immutable per-run artifact, unlike a
served `models/*.onnx`, which is replaced by name.

`register_entrants(load)` runs at import time. This is the entire mechanism
by which a checkpoint becomes playable through `hexset.bench.duel --runtime
hexn.netbot network:<path> ...` or any other `hexset.arena.spawn` caller;
`hexset.arena` itself never imports torch or this module.

How a loaded checkpoint is played — the bot, leaf evaluation, search, and
trade gate — is runtime-agnostic and lives in `hexset.clients.netbot`, over
the `hexset.clients.policy.Policy` protocol. `hexn.netbot` only supplies
the torch-specific half: finding the file and rebuilding the net.

## Exporting to ONNX (`hexn.export_onnx`)

```bash
python -m hexn.export_onnx --checkpoint runs/example/latest.pt --out model.onnx
python -m hexn.export_onnx --checkpoint runs/example/latest.pt \
    --out mcts256.onnx --search mcts --simulations 256
```

Needs `torch` (not a declared dependency — see below) plus `onnx` and
`onnxruntime`, which are also not declared in this package's
`pyproject.toml`. Install them separately (`pip install onnx onnxruntime`)
before running this module.

Traces the exact reconstruction path `hexn.netbot.load()` already trusts
(same shapes, same `state_dict` keys) once with `torch.onnx.export`, and
embeds a checkpoint's contract-relevant facts as ONNX metadata for an
onnxruntime-based server to read at load time.

- **Inputs**: a dynamic leading batch axis, named exactly
  `hexset.onnx_record.RECORD_FIELDS` — the information-set record a
  position's legal seat may see. All `int64` except `action_mask`, the only
  `bool` field.
- **Outputs**, in order: `action_index` (`(B,)` int64, the argmax over the
  masked distribution), `prior` (`(B, space.size)` float32, normalised over
  legal entries, zero elsewhere), and `value` (`(B, players)` float32,
  board-seat order, already un-rotated — each seat's own win probability).
- **Metadata props**: `contract` (`hexset.onnx_record.CONTRACT_VERSION`),
  `players`, the topology fingerprint (`num_hexes`/`num_vertices`/
  `num_edges`), `max_offers` (the network's offer budget; a file still
  carrying the pre-0.24 `max_trades` key is read as it), `iteration`, and
  `--gate-plies` (the served
  trade gate's own look-ahead budget, independent of `--search`).
  `search=mcts` with `--simulations`/`--wave` is included only when passed
  on the command line.
- **Provenance props** (read by tooling afterward, not at runtime):
  `source_checkpoint`, `checkpoint_sha256`, `exported_at`, and
  `exporter_commit` when determinable.
- The stochastic/Gumbel sampling path is gone: every exported file plays
  argmax, since nothing downstream of this export samples.
- Runtime target is onnxruntime ≥ 1.18 on the CPU execution provider, opset
  18.

A numerical parity check runs automatically after every export. It raises
(a nonzero exit) rather than warns, since an export bug that silently
degrades quality is far more expensive to catch later, on a served board,
than here, against real seeded positions.

`--out` defaults to the checkpoint's own path with a `.onnx` suffix.

## Why torch is not a dependency

`pyproject.toml` declares `hexset` and `numpy`, and nothing else. Torch is
provisioned separately (a ROCm or CUDA build matched to the training
hardware), so that everything that does not need it stays installable and
testable without it: `hexn.readout`, `hexn.selfplay`, `hexn.rewards`,
`hexn.schedule`, `hexn.run`, and the torch-free half of the test suite.
Every module that trains, evaluates, exports, or migrates a network imports
torch lazily, or declares it as a hard runtime requirement of that one
module rather than the package as a whole.
</content>
