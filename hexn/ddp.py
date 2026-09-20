# SPDX-License-Identifier: GPL-3.0-only
"""The PPO update, data-parallel across CPU worker processes.

Why CPU processes rather than a single-device GPU update: at this model's
size, the forward+backward pass is too small an op for the usual
single-device speedups (compile, larger minibatches, lower precision) to
move much, and CPU *threads* don't scale either, because the ops are too
small for intra-op parallelism to pay for itself. What that does not rule
out is *data* parallelism: many independent single-threaded CPU processes,
each running the whole fwd+bwd on its own shard of the batch, are worth more
together than one device — provided the gradients can be combined cheaply,
which at this model's parameter count they can.

The shape mirrors `hexn.collect`: spawned workers behind pipes, specs not
objects, torch's shared-memory tensor transport doing the heavy lifting. Each
worker holds a contiguous shard of the batch. Every optimizer step, the
learner sends the current parameters and each worker's slice of the
minibatch; workers run the *same* `hexn.ppo.minibatch_terms` the
single-device update runs, and return flat gradients. The learner combines
them weighted by row count — the gradient of a mean over 4096 rows is the
row-weighted mean of the shard gradients — clips the global norm, and steps
the one true Adam. Workers never hold optimizer state, so there is nothing to
drift.

Semantics match the single-device update exactly up to float summation order:
same minibatch schedule from the same generator, same per-minibatch advantage
normalisation (the learner computes the minibatch's mean and std, because a
worker only sees its slice), same loss arithmetic by construction. The
equivalence is pinned by test rather than argued.

Two deployment requirements, both learned by hitting them. The tensor
transport goes through /dev/shm, and docker's 64 MB default SIGBUSes on the
first shard — run the container with `--ipc host` (what the training recipe
already uses) or an `--shm-size` sized to the batch. And any script that
builds an `UpdateCrew` needs the standard `if __name__ == "__main__"` guard:
spawn re-imports `__main__`, and an unguarded script re-runs its whole
preamble once per worker.
"""

from __future__ import annotations

import random
import traceback
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.multiprocessing
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from hexset.actions import build_space
from hexset.board.board import random_base_board
from hexset.encoding import static_graph
from .model import HexNet, ModelConfig, packing
from .policy import NetworkPolicy
from .ppo import Batch, PPOConfig, Stats, _explained_variance, _minibatches, minibatch_terms

FIELDS = (
    "buffer",
    "mask",
    "chosen",
    "log_prob",
    "advantage",
    "value_target",
    "margin_target",
)


@dataclass(frozen=True)
class UpdateSpec:
    """Everything an update worker needs to rebuild the net. Picklable."""

    seed: int
    players: int
    width: int
    rounds: int
    # A replica takes the learner's parameters as one flat vector, so its shape
    # has to match the learner's exactly — the head shapes belong here for the
    # same reason width does.
    value_head: str = "linear"
    policy_head: str = "linear"
    # Inert unless `value_head == "quantile"`; see `ModelConfig.quantiles`.
    quantiles: int = 32


def _replica(spec: UpdateSpec) -> NetworkPolicy:
    torch.set_num_threads(1)
    board = random_base_board(random.Random(spec.seed))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, spec.players
    )
    graph = static_graph(topology)
    net = HexNet(
        space,
        graph,
        spec.players,
        ModelConfig(
            width=spec.width,
            rounds=spec.rounds,
            value_head=spec.value_head,
            policy_head=spec.policy_head,
            quantiles=spec.quantiles,
        ),
    )
    return NetworkPolicy(net, space, packing(graph, spec.players), device="cpu")


def _flat_grad(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in net.parameters()
        ]
    )


def _serve(spec: UpdateSpec, connection) -> None:
    try:
        policy = _replica(spec)
        shard: dict[str, torch.Tensor] = {}
        config = PPOConfig()
        while True:
            kind, payload = connection.recv()
            if kind == "shard":
                fields, config = payload
                shard = fields
                connection.send(("ok", None))
            elif kind == "step":
                params, rows, mean, std, want_values = payload
                vector_to_parameters(params, policy.net.parameters())
                if rows.numel() == 0:
                    connection.send(("done", (0, None, None, None)))
                    continue
                advantage = (shard["advantage"][rows] - mean) / (std + 1e-8)
                policy.net.zero_grad(set_to_none=True)
                terms = minibatch_terms(
                    policy,
                    shard["buffer"][rows],
                    shard["mask"][rows],
                    shard["chosen"][rows],
                    shard["log_prob"][rows],
                    advantage,
                    shard["value_target"][rows],
                    shard["margin_target"][rows],
                    config,
                )
                terms.loss.backward()
                # `minibatch_terms`'s scalar fields are device tensors now (see
                # `hexn.ppo.Terms`), read back here rather than left that way:
                # a worker is a single CPU process, so `float(...)` on one of
                # its own tensors is a plain read, not the cross-device sync
                # the GPU update was paying it 112 times a minibatch to avoid.
                # `pieces_sum` downstream is a plain numpy accumulation and
                # wants Python floats, not tensors, to add into it.
                pieces = (
                    float(terms.policy_loss),
                    float(terms.value_loss),
                    float(terms.value_mse),
                    float(terms.margin_loss),
                    float(terms.entropy),
                    float(terms.approx_kl),
                    float(terms.clip_fraction),
                )
                values = None
                if want_values:
                    values = (terms.value[:, 0], shard["value_target"][rows][:, 0])
                connection.send(
                    ("done", (int(rows.numel()), _flat_grad(policy.net), pieces, values))
                )
            elif kind == "stop":
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command: {kind}")
    except EOFError:
        return
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass


class UpdateCrew:
    """K update workers behind the same call shape as `hexn.ppo.update`."""

    def __init__(self, specs: Sequence[UpdateSpec]) -> None:
        if not specs:
            raise ValueError("an update crew needs at least one worker")
        # torch.multiprocessing, so tensors sent over the pipes travel through
        # shared memory instead of being pickled byte by byte.
        context = torch.multiprocessing.get_context("spawn")
        self._connections = []
        self._processes = []
        for spec in specs:
            ours, theirs = context.Pipe()
            process = context.Process(target=_serve, args=(spec, theirs), daemon=True)
            process.start()
            theirs.close()
            self._connections.append(ours)
            self._processes.append(process)

    def _hear(self, connection, wanted: str):
        kind, payload = connection.recv()
        if kind == "error":
            raise RuntimeError(f"an update worker failed:\n{payload}")
        if kind != wanted:
            raise RuntimeError(f"expected {wanted}, a worker sent {kind}")
        return payload

    def update(
        self,
        policy: NetworkPolicy,
        optimiser: torch.optim.Optimizer,
        batch: Batch,
        config: PPOConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> Stats:
        """`hexn.ppo.update`, with the fwd+bwd farmed out to the workers.

        The learner keeps the one true optimizer and parameters; workers see
        parameters every step and never keep state across steps beyond their
        batch shard.
        """
        size = len(batch)
        workers = len(self._connections)
        bounds = [(w * size) // workers for w in range(workers + 1)]

        for w, connection in enumerate(self._connections):
            fields = {
                name: getattr(batch, name)[bounds[w] : bounds[w + 1]]
                for name in FIELDS
            }
            connection.send(("shard", (fields, config)))
        for connection in self._connections:
            self._hear(connection, "ok")

        device = next(policy.net.parameters()).device
        steps = [
            rows
            for _ in range(config.epochs)
            for rows in _minibatches(size, config.minibatch, generator)
        ]

        policy_losses, value_losses, entropies, kls, clipped = [], [], [], [], []
        value_mses: list[float] = []
        margin_losses: list[float] = []
        grad_norms: list[float] = []
        variance_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        # `steps` is the epoch loop already flattened, so the final epoch is the
        # last `len(steps) // config.epochs` entries. Derived rather than tracked
        # so the two update paths report the same gauges off the same shape.
        per_epoch = max(1, len(steps) // max(1, config.epochs))
        for step, rows in enumerate(steps):
            advantage = batch.advantage[rows]
            mean = advantage.mean()
            std = advantage.std()
            params = parameters_to_vector(policy.net.parameters()).detach().cpu()
            last = step == len(steps) - 1

            for w, connection in enumerate(self._connections):
                inside = rows[(rows >= bounds[w]) & (rows < bounds[w + 1])]
                connection.send(
                    ("step", (params, inside - bounds[w], mean, std, last))
                )

            total = 0
            combined = torch.zeros_like(params)
            pieces_sum = np.zeros(7)
            for connection in self._connections:
                count, grad, pieces, values = self._hear(connection, "done")
                if not count:
                    continue
                total += count
                combined += count * grad
                pieces_sum += count * np.asarray(pieces)
                if values is not None:
                    variance_parts.append(values)
            if not total:
                continue
            combined /= total

            grads = combined.to(device)
            offset = 0
            for parameter in policy.net.parameters():
                parameter.grad = (
                    grads[offset : offset + parameter.numel()]
                    .view_as(parameter)
                    .clone()
                )
                offset += parameter.numel()
            grad_norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        policy.net.parameters(), config.max_grad_norm
                    )
                )
            )
            optimiser.step()

            averaged = pieces_sum / total
            policy_losses.append(averaged[0])
            value_losses.append(averaged[1])
            value_mses.append(averaged[2])
            margin_losses.append(averaged[3])
            entropies.append(averaged[4])
            kls.append(averaged[5])
            clipped.append(averaged[6])

        if variance_parts:
            predicted = torch.cat([p for p, _ in variance_parts])
            actual = torch.cat([a for _, a in variance_parts])
            variance = _explained_variance(predicted, actual)
        else:
            variance = 0.0

        return Stats(
            positions=size,
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(kls)),
            clip_fraction=float(np.mean(clipped)),
            explained_variance=variance,
            approx_kl_first_minibatch=float(kls[0]) if kls else 0.0,
            approx_kl_last_epoch=float(np.mean(kls[-per_epoch:])) if kls else 0.0,
            clip_fraction_last_epoch=(
                float(np.mean(clipped[-per_epoch:])) if clipped else 0.0
            ),
            grad_norm=float(np.median(grad_norms)) if grad_norms else 0.0,
            value_target_variance=float(batch.value_target[:, 0].var()),
            lr=float(optimiser.param_groups[0]["lr"]),
            value_mse=float(np.mean(value_mses)) if value_mses else 0.0,
            margin_loss=float(np.mean(margin_losses)) if margin_losses else 0.0,
        )

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("stop", None))
                self._hear(connection, "ok")
            except Exception:
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
