# SPDX-License-Identifier: GPL-3.0-only
"""One process on the box runs every network forward for many workers.

`Server` loads each net once on its device (`hexn.netbot.load`) and answers
the forwards its clients send over `multiprocessing` connections. Each round
it takes every request waiting on any connection, joins the rows asked of one
net across all of them into one forward, and sends each client its own rows
back. A worker therefore plays with nets it never runs: it encodes its rows,
sends them, and reads the answer, while one process keeps the device busy
with batches as large as all the workers' lanes together.

`Panel` is a worker's side: a fixed list of policies, some served and some
local, scored together. `Panel.score_rows_all(rows)` sends one message with
every served net's feed for `rows` (`score_feed`), scores the local policies
(an exported file, which runs on the worker's CPU) while the server works,
then reads each served net's answer (`score_read`). It is the `scorer` a
`hexset.gym.seats.PolicySeat` takes, so a seat reading several nets makes
one round trip a tick for all of them. `Panel.policy(k)` is a served net as a
`hexset.clients.policy.Policy` of its own, one round trip per call.

A served net answers exactly what the same net loaded on the server's device
answers in-process, on the batch the server ran: a contract-9 net
(`hexn.policy9.Policy9`) its `_heads` with the prior, a contract-8 net
(`hexn.policy.NetworkPolicy`) its `score`. A forward is not batch invariant,
and the server's batches depend on which requests arrive together, so a
served read can differ from an in-process one in the last bits, as any
batched read can; on another device it differs by that device's arithmetic
too. `precision` "bf16" runs the server's forwards under autocast, a
different treatment from fp32, not an equal one. `gather` holds a round open
that many seconds after its first request, or until every live client has
asked, so the forwards are larger; 0 runs whatever has arrived at once.

    server = Server.start(paths, topology, clients=3, device="cuda")
    panel = Panel(server.connection(0), paths, topology)       # in worker 0
    reads = panel.score_rows_all(rows)                          # one list per net
    server.stop()
"""
from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import fields
from multiprocessing.connection import wait

import numpy as np

__all__ = ["Panel", "Server", "join_arrays", "split_heads"]

_ROWS9 = ("value", "points", "flat", "setup", "prior")


def _kind(path: str) -> str:
    return "onnx" if path.endswith(".onnx") else "torch"


def join_arrays(parts: Sequence[dict]) -> tuple[dict, list[dict]]:
    """Several batches of `hexset.decisions.arrays` as one: row arrays
    stacked, candidate arrays joined with their row indices moved past the
    batches before them. Returns the joined arrays and, per batch, the counts
    `split_heads` needs (`rows`, `road_building`, `discard`, `double`)."""
    out: dict[str, list] = {key: [] for key in parts[0]}
    counts, offset = [], 0
    for part in parts:
        rows = len(part["flat"])
        for key, value in part.items():
            out[key].append(value + offset if key in ("road_building_row", "discard_row", "double_rows") else value)
        counts.append({"rows": rows, "road_building": len(part["road_building"]), "discard": len(part["discard"]),
                       "double": len(part["double_rows"])})
        offset += rows
    return {key: np.concatenate(values) for key, values in out.items()}, counts


def split_heads(heads, counts: Sequence[dict]) -> list:
    """`hexn.policy9.Heads9` of a joined batch, cut back into one per batch."""
    from .policy9 import Heads9

    out, at = [], {"rows": 0, "road_building": 0, "discard": 0, "double": 0}
    for count in counts:
        parts = {}
        for field in fields(Heads9):
            values = getattr(heads, field.name)
            unit = ("rows" if field.name in _ROWS9 else field.name)
            parts[field.name] = None if values is None else values[at[unit]:at[unit] + count[unit]]
        for unit in at:
            at[unit] += count[unit]
        out.append(Heads9(**parts))
    return out


def _forward(policy, kind: str, payloads: list):
    """One forward over every payload asked of one net, answered per payload."""
    if kind == "heads9":
        arrays, counts = join_arrays([arrays for _, arrays in payloads])
        observations = np.concatenate([observations for observations, _ in payloads])
        return split_heads(policy._heads(observations, arrays, prior=True), counts)
    observations = [o for part, _ in payloads for o in part]
    slots, values = policy.score(observations, np.concatenate([masks for _, masks in payloads]))
    out, at = [], 0
    for part, _ in payloads:
        out.append((slots[at:at + len(part)], values[at:at + len(part)]))
        at += len(part)
    return out


def _serve(paths: Sequence[str], topology, connections, device: str, precision: str, ready,
           gather: float = 0.0) -> None:
    """The server loop (module docstring): until every client has said "stop"."""
    import contextlib

    import torch

    from . import netbot

    torch.set_num_threads(1)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    policies = [None if _kind(path) == "onnx" else netbot.load(path, topology, device).policy for path in paths]
    kinds = [None if policy is None else "heads9" if hasattr(policy, "score_feed") and hasattr(policy, "_heads")
             else "score8" for policy in policies]
    autocast = (torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if precision == "bf16"
                else contextlib.nullcontext())
    ready.send(("ready", kinds))
    live = list(connections)
    # rounds: forward rounds; requests: (client, net) asks; forwards: net
    # forwards run; rows: rows forwarded, over every net; seconds: busy time.
    stats = {"rounds": 0, "requests": 0, "forwards": 0, "rows": 0, "seconds": 0.0}
    while live:
        waiting = []
        asked: set = set()
        deadline = None
        while live:
            # The first request opens the round; with `gather` the round
            # stays open that long, or until every live client has asked.
            timeout = None if deadline is None else max(0.0, deadline - time.perf_counter())
            ready_now = [c for c in wait(live, timeout) if c not in asked]
            for connection in ready_now:
                try:
                    message = connection.recv()
                except EOFError:
                    live.remove(connection)
                    continue
                if message[0] == "stop":
                    connection.send(("stats", stats))
                    live.remove(connection)
                    continue
                waiting.append((connection, message[1]))
                asked.add(connection)
            if not waiting:
                continue
            if deadline is None:
                deadline = time.perf_counter() + gather
            if len(asked) >= len(live) or time.perf_counter() >= deadline:
                break
        if not waiting:
            continue
        started = time.perf_counter()
        by_net: dict[int, list] = {}
        for k, (_, asks) in enumerate(waiting):
            for j, (net, payload) in enumerate(asks):
                by_net.setdefault(net, []).append((k, j, payload))
        answers = [[None] * len(asks) for _, asks in waiting]
        with torch.inference_mode(), autocast:
            for net, asked in by_net.items():
                for (k, j, _), answer in zip(asked, _forward(policies[net], kinds[net], [p for _, _, p in asked])):
                    answers[k][j] = answer
        for (connection, _), answer in zip(waiting, answers):
            connection.send(("answer", answer))
        stats["rounds"] += 1
        stats["requests"] += sum(len(asks) for _, asks in waiting)
        stats["forwards"] += len(by_net)
        stats["rows"] += sum(len(payload[0]) for _, asks in waiting for _, payload in asks)
        stats["seconds"] += time.perf_counter() - started


class Server:
    """The process that owns the device (module docstring). `start` returns
    once every net is loaded there; `connection(k)` is client `k`'s end, to
    hand to the process that plays with it; `stop` waits for the server to
    exit once every client has stopped."""

    def __init__(self, process, connections, kinds) -> None:
        self.process, self.connections, self.kinds = process, connections, kinds

    @classmethod
    def start(cls, paths: Sequence[str], topology, *, clients: int, device: str = "cuda",
              precision: str = "fp32", gather: float = 0.0, context=None) -> "Server":
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"precision is fp32 or bf16, not {precision!r}")
        context = context or multiprocessing.get_context("spawn")
        pairs = [context.Pipe() for _ in range(clients)]
        ready_parent, ready_child = context.Pipe()
        process = context.Process(target=_serve, args=(list(paths), topology, [server for _, server in pairs],
                                                       device, precision, ready_child, gather), daemon=True)
        process.start()
        kind, kinds = ready_parent.recv()
        if kind != "ready":
            raise RuntimeError("the inference server did not start")
        return cls(process, [client for client, _ in pairs], kinds)

    def connection(self, k: int):
        return self.connections[k]

    def stop(self, timeout: float = 60.0) -> None:
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            raise RuntimeError("the inference server did not stop")


class Panel:
    """A worker's policies, read together (module docstring). `paths` are the
    server's nets in its order; a `.onnx` file is loaded here and scored on
    this worker's CPU, every other net is served."""

    def __init__(self, connection, paths: Sequence[str], topology) -> None:
        from . import netbot

        self.connection = connection
        self.local: dict[int, object] = {}
        self.served: dict[int, object] = {}
        for k, path in enumerate(paths):
            if _kind(path) == "onnx":
                from hexset.clients import onnxbot

                self.local[k] = onnxbot.load(path, topology).policy
            else:
                # Loaded on this CPU for its encoder and its reading of the
                # answer; the forward runs on the server.
                self.served[k] = netbot.load(path, topology, "cpu").policy
        self.size = len(paths)
        self.round_trips = 0

    def score_rows_all(self, rows, nets: Sequence[int] | None = None) -> list[list]:
        """Every policy's `score_rows(rows)`, in `paths` order; with `nets`,
        those policies' only, in that order."""
        nets = list(range(self.size)) if nets is None else list(nets)
        if not rows:
            return [[] for _ in nets]
        served = {k: self.served[k] for k in nets if k in self.served}
        feeds = {k: policy.score_feed(rows) for k, policy in served.items()}
        if feeds:
            self.connection.send(("score", [(k, feed) for k, feed in feeds.items()]))
        out: dict = {}
        for k in nets:
            if k in self.local:
                out[k] = self.local[k].score_rows(rows)
        if not feeds:
            return [out[k] for k in nets]
        kind, answers = self.connection.recv()
        if kind != "answer":
            raise RuntimeError(f"the inference server answered {kind!r}")
        self.round_trips += 1
        for (k, policy), answer in zip(served.items(), answers):
            out[k] = policy.score_read(rows, answer) if hasattr(policy, "_heads") else policy.score_read(rows, *answer)
        return [out[k] for k in nets]

    def policy(self, k: int) -> "_Served":
        """Policy `k` alone, as a `Policy` whose `score_rows` reads through
        this panel."""
        return _Served(self, k)

    def stop(self) -> dict:
        """Tell the server this client is done; its counters so far."""
        self.connection.send(("stop", None))
        kind, stats = self.connection.recv()
        return stats


class _Served:
    def __init__(self, panel: Panel, k: int) -> None:
        self.panel, self.k = panel, k
        held = panel.served.get(k) or panel.local[k]
        self.players, self.space = getattr(held, "players", None), getattr(held, "space", None)

    def score_rows(self, rows):
        if not rows:
            return []
        served = self.panel.served.get(self.k)
        if served is None:
            return self.panel.local[self.k].score_rows(rows)
        self.panel.connection.send(("score", [(self.k, served.score_feed(rows))]))
        kind, (answer,) = self.panel.connection.recv()
        return served.score_read(rows, answer) if hasattr(served, "_heads") else served.score_read(rows, *answer)
