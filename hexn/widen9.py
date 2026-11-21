# SPDX-License-Identifier: GPL-3.0-only
"""A contract-11d checkpoint as a contract-11 one, for a table of `players`.

Contract 11 keeps 11d's columns in 11d's places and appends its own
(`hexset.contract9`), so an 11d net's weights carry over whole: my token's
and the opponent's embeddings are copied, every opponent is read by the
11d opponent's layer, and the appended columns' weights start at zero
(`HexNet._load_from_state_dict`). At a table of two the widened net plays as
the 11d net, up to float reassociation; at four, its heads read the
opponents' mean token where they read the one opponent's.

    python -m hexn.widen9 net11d.pt net11.pt --players 4

The output is an ordinary contract-11 checkpoint: `model_config["contract"]`
and `args["contract"]` are "11", `args["players"]` the table's seats, and
`widened_from` the source's contract and sha256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path

import torch

from hexset.actions import build_space

from .model9 import HexNet, config_of

__all__ = ["widen", "main"]


def widen(state: dict, players: int, topology=None) -> dict:
    """The checkpoint dict `state` of a contract-11d net as a contract-11 one
    for `players` seats: the same weights, 11's appended columns at zero."""
    from .export_onnx import _base_topology

    config = dict(state["model_config"])
    if str(config.get("contract")) != "11d":
        raise ValueError(f"widen takes a contract-11d checkpoint, not contract {config.get('contract')!r}")
    if not 2 <= players <= 4:
        raise ValueError(f"contract 11 seats two to four, not {players}")
    topology = topology or _base_topology()
    config["contract"] = "11"
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, players)
    net = HexNet(space, topology, players, config_of(config))
    net.load_state_dict(state["net"])
    out = {key: copy.deepcopy(value) for key, value in state.items() if key not in ("net", "optimizer", "opt")}
    out["net"] = net.state_dict()
    out["model_config"] = {**state["model_config"], "contract": "11"}
    out["args"] = {**state.get("args", {}), "contract": "11", "players": players}
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(prog="python -m hexn.widen9", description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="a contract-11d checkpoint")
    parser.add_argument("out", type=Path, help="where to write the contract-11 checkpoint")
    parser.add_argument("--players", type=int, default=4, help="the table's seats, two to four (default 4)")
    args = parser.parse_args(argv)
    state = torch.load(args.source, map_location="cpu", weights_only=False)
    out = widen(state, args.players)
    out["widened_from"] = {"contract": "11d", "sha256": _sha256(args.source)}
    temporary = args.out.with_name(args.out.name + ".tmp")
    torch.save(out, temporary)
    os.replace(temporary, args.out)
    print(f"{args.out} sha256 {_sha256(args.out)}")
    return args.out


if __name__ == "__main__":
    main()
