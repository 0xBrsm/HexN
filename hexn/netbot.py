# SPDX-License-Identifier: GPL-3.0-only
"""The torch half of the model boundary: loading a `.pt` checkpoint, and
nothing else.

How a checkpoint is *played* — the bot, the leaf evaluation, the search, the
trade gate — is runtime-agnostic and lives in `hexset.clients.netbot`, over
the `hexset.clients.policy.Policy` protocol. This module used to carry a
torch copy of all four of those classes alongside onnxruntime's copy in
HexSet, and the two drifted three ways in the trade gate alone before anybody
noticed. What is left here is the part that genuinely differs between
runtimes: finding the file, rebuilding the net from the `args` the run
recorded, and handing back a
`Checkpoint`. `hexn.policy.NetworkPolicy` is the `Policy` itself.

Three things here are not decoration:

*The checkpoint is loaded once per process, not once per game.* `arena.spawn`
is called per game per worker, and a `torch.load` per game would dominate the
thing being measured. The cache is keyed on the topology as well as the path,
because a model's adjacency buffers are baked in at construction: a second
layout is a second network, not the same one on a different board. Not on the
file's mtime, unlike `hexset.clients.onnxbot.load`: a `.pt` under `runs/` is
an immutable per-run artifact, where a served `models/*.onnx` is replaced by
name.

*Intraop threading is turned off.* A duel is thirty worker processes each
running a batch of one, so torch's default of one thread per core would have
thirty processes fighting over thirty-two cores and the measurement would be
of the thrash.

*The policy is greedy.* Sampling is the behaviour distribution PPO needed, not
the policy worth scoring; `NetworkPolicy(greedy=True)` takes the argmax.

*`load` plays a net through ONNX.* A file ending `.onnx` goes to
`hexset.clients.onnxbot.load` (one onnxruntime thread, its own cache keyed on
the file's mtime), and so does a contract-9 `.pt` on the CPU, through its
export (`onnx_of`, made once per checkpoint): the same decisions as the torch
net, about three times faster at a batch of one on one CPU thread. Every
entrant the arena seats (`network:`, `mcts:`, `rollout:`) loads this way.
`load_torch` is the torch net itself, for training and for code that reads
inside the net; `load` gives it too for an older contract or another device.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from hexset.actions import ActionSpace, build_space
from hexset.board.topology import Topology
from hexset.clients.netbot import (
    GatedSearch,
    LeafEvaluator,
    NetworkBot,
    bot_for,
    register_entrants,
    searcher_for,
)
from hexset.encoding import static_graph
from hexset.contract9 import CONTRACTS as CONTRACTS9
from hexset.holding import HoldModel
from .hold import declared as declared_hold
from .model import HexNet, config_from_args, packing
from .policy import NetworkPolicy
from .trade import recorded_budget, trade_params as _trade_params

# What this module exports. The three classes are `hexset.clients.netbot`'s
# now -- re-exported by name so every existing `from hexn.netbot import ...`
# keeps working, and because "the torch checkpoint's bot" is still what a
# caller here means by them.
__all__ = [
    "GatedSearch",
    "LeafEvaluator",
    "Loaded",
    "NetworkBot",
    "bot_for",
    "load",
    "load_torch",
    "onnx_of",
    "searcher_for",
]


@dataclass(frozen=True)
class Loaded:
    """A checkpoint made playable, plus what the run it came from was doing.

    `hexset.clients.policy.Checkpoint`, plus the one fact only a trained file
    carries: which training iteration it is.
    """

    policy: NetworkPolicy
    space: ActionSpace
    players: int
    max_offers: int | None
    iteration: int
    #: The hold model the checkpoint declares (`hexn.hold`), which its search
    #: and gate worlds deal the other seats' held development cards by;
    #: `None` deals them alike.
    hold: HoldModel | None = None

    @property
    def trade_params(self):
        """How this checkpoint's network bargains, as HexSet reads it by name
        (`hexset.trading.params_of`): the offer budget its run recorded."""
        return _trade_params(self.max_offers)


def load(path: str, topology: Topology, device: str = "cpu"):
    """The network at `path` as a seat plays it, on boards of this topology:
    an `.onnx` file, or a contract-9 `.pt` on the CPU through its export
    (`onnx_of`), in `hexset.clients.onnxbot` on one thread; any other `.pt`
    as `load_torch` rebuilds it (module docstring). The export's parity check
    holds it to `hexn.policy9.Policy9`'s decisions on the torch net."""
    from hexset.clients import onnxbot

    if str(path).endswith(".onnx"):
        return onnxbot.load(str(path), topology, device, 1)
    if device == "cpu" and _contract(str(path)) in CONTRACTS9:
        return onnxbot.load(str(onnx_of(str(path))), topology, "cpu", 1)
    return _load_pt(path, topology, device, "none")


def load_torch(path: str, topology: Topology, device: str = "cpu", compile_mode: str = "none") -> "Loaded":
    """The `.pt` checkpoint at `path` rebuilt as its torch net (`_load_pt`),
    for training and for code that reads inside the net (`Loaded.policy.net`,
    `_heads`, hooks); `compile_mode` other than `"none"` runs it under
    `torch.compile`. An `.onnx` file has no torch net and is refused."""
    if str(path).endswith(".onnx"):
        raise ValueError(f"{path} is an ONNX file: it has no torch net (`load` plays it)")
    return _load_pt(path, topology, device, compile_mode)


@lru_cache(maxsize=4)
def _load_pt(
    path: str,
    topology: Topology,
    device: str = "cpu",
    compile_mode: str = "none",
) -> Loaded:
    """The `.pt` checkpoint at `path`, ready to act on boards of this topology.

    Cached per process: a worker plays hundreds of games and every one of them
    would otherwise pay for the same `torch.load`.
    """
    # One thread, because there are thirty of these processes. Set here rather
    # than at import so it is attached to the decision that needs it.
    torch.set_num_threads(1)

    state = torch.load(path, map_location=device, weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
    if str(args.get("contract", "")) in CONTRACTS9:
        return _load9(state, args, topology, players, device)
    hold = declared_hold(state)
    # The head shapes read the same way width and rounds do, and default to the
    # shape every checkpoint written before they existed was trained with. A
    # checkpoint that records a shape has to be rebuilt with it or
    # `load_state_dict` fails on the keys, which is the loud failure and the one
    # worth having.
    config = config_from_args(args)

    graph = static_graph(topology)
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    net = HexNet(space, graph, players, config)
    net.load_state_dict(state["net"])
    net.fused = torch.device(device).type == "cuda"
    net = net.to(device).eval()
    if compile_mode != "none":
        net = torch.compile(net, mode=compile_mode)

    return Loaded(
        policy=NetworkPolicy(
            net, space, packing(graph, players), device=device, greedy=True
        ),
        space=space,
        players=players,
        max_offers=recorded_budget(args),
        iteration=int(state.get("iteration", 0)),
        hold=hold,
    )


def _load9(state: dict, args: dict, topology: Topology, players: int, device: str) -> Loaded:
    """A contract-9 checkpoint (`hexn.model9.HexNet`), played by `hexn.policy9.Policy9`.
    It makes no offers of its own (contract 9 reads no trading inputs yet); its gate
    still answers the table's."""
    from .model9 import HexNet as HexNet9, config_of
    from .policy9 import Policy9

    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, players)
    net = HexNet9(space, topology, players, config_of(state["model_config"]))
    net.load_state_dict(state["net"])
    net = net.to(device).eval()
    return Loaded(policy=Policy9(net, space, device=device), space=space, players=players,
                  max_offers=0, iteration=int(state.get("iteration", 0)), hold=declared_hold(state))


load_torch.cache_clear = _load_pt.cache_clear  # the `.pt` cache; onnxbot keeps its own


@lru_cache(maxsize=16)
def _contract(path: str) -> str:
    """The contract a `.pt` checkpoint's run recorded, `""` for none."""
    return str(torch.load(path, map_location="cpu", weights_only=False).get("args", {}).get("contract", ""))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=16)
def onnx_of(path: str) -> Path:
    """The contract-9 checkpoint at `path` exported to ONNX
    (`hexn.export_onnx9.export9`, its parity check on), once per checkpoint
    and exporter: kept in `HEXN_ONNX_CACHE` (default
    `~/.cache/hexn/onnx`) under the checkpoint's sha256 and the exporter's
    and model's source, and written under a lock so the workers of one run
    export it once between them."""
    from . import export_onnx9, model9
    from .export_onnx9 import export9

    source = Path(path)
    code = hashlib.sha256(b"".join(Path(m.__file__).read_bytes() for m in (export_onnx9, model9))).hexdigest()
    cache = Path(os.environ.get("HEXN_ONNX_CACHE") or Path.home() / ".cache" / "hexn" / "onnx")
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_digest(source)[:24]}-{code[:12]}.onnx"
    if out.is_file():
        return out
    with open(cache / f"{out.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.is_file():
            export9(str(source), out, verify=True)
    return out


load.cache_clear = lambda: (_contract.cache_clear(), onnx_of.cache_clear(), _load_pt.cache_clear())


# Registered at import, so importing this module is the whole of naming the
# runtime: any process that does -- directly, via `hexn.ppo`/`hexn.league`/
# `hexn.collect`, or through `hexset.bench.duel --runtime hexn.netbot`, which
# hands this module's name to each worker as `arena.compete`'s initializer --
# can spawn a network-backed entrant through `hexset.arena.spawn` without
# `hexset.arena` ever importing torch or this module.
# `hexset.clients.netbot` owns every factory behind it; a seat plays its net
# as `load` gives it.
register_entrants(load)
