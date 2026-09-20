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
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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
from .model import HexNet, config_from_args, packing
from .policy import NetworkPolicy

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
    max_trades: int | None
    iteration: int


@lru_cache(maxsize=4)
def load(
    path: str,
    topology: Topology,
    device: str = "cpu",
    compile_mode: str = "none",
) -> Loaded:
    """The network at `path`, ready to act on boards of this topology.

    Cached per process: a worker plays hundreds of games and every one of them
    would otherwise pay for the same `torch.load`.
    """
    # One thread, because there are thirty of these processes. Set here rather
    # than at import so it is attached to the decision that needs it.
    torch.set_num_threads(1)

    state = torch.load(path, map_location=device, weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
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
    net = net.to(device).eval()
    if compile_mode != "none":
        net = torch.compile(net, mode=compile_mode)

    return Loaded(
        policy=NetworkPolicy(
            net, space, packing(graph, players), device=device, greedy=True
        ),
        space=space,
        players=players,
        max_trades=args.get("max_trades"),
        iteration=int(state.get("iteration", 0)),
    )


# Registered at import, so importing this module is the whole of naming the
# runtime: any process that does -- directly, via `hexn.ppo`/`hexn.league`/
# `hexn.collect`, or through `hexset.bench.duel --runtime hexn.netbot`, which
# hands this module's name to each worker as `arena.compete`'s initializer --
# can spawn a network-backed entrant through `hexset.arena.spawn` without
# `hexset.arena` ever importing torch or this module.
# `hexset.clients.netbot` owns every factory behind it.
register_entrants(load)
