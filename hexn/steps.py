# SPDX-License-Identifier: GPL-3.0-only
"""Keep an update's progress after its optimiser steps, and resume from it.

An iteration checkpoint holds the weights an iteration ends on, so an update
killed part-way through loses every step it took. `Steps` closes the gap.
A trainer hands one to `hexn.ppo.update` or `hexn.exit.update`, which walks
its passes through a `Cursor`; after every `every`-th optimiser step, and
after the last, the cursor writes `latest-step.pt` to the run directory
(`hexn.loop.save`: a temporary name, fsynced, renamed). It holds the weights,
the optimiser, how many steps the update has taken, the shuffle RNG state at
the start of the pass the last of them came from, and every gauge the update
has accumulated, so the resumed update logs what the uninterrupted one would
have.

A resumed run restores the iteration's start as it always has (`latest.pt`),
rebuilds the iteration's batch from its games on disk (`hexn.durable.
Partial`), attaches the prior, and only then opens the step file. When the
file is this iteration's and the rebuilt batch has the fingerprint it was
written against, the weights and the optimiser are loaded from it; the
cursor skips the passes already walked, replays the interrupted pass's
shuffle from the saved RNG state, skips the minibatches already stepped and
carries on. The update that results is the uninterrupted one's, to the bit
on a deterministic device. A step file of another iteration, or of a batch
that did not rebuild the same, is removed, and the update starts from the
iteration's beginning, as it would have with no file at all.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .selfplay import Episode

STEP = "latest-step.pt"

# Rows of a wide batch field read into the fingerprint: a strided sample, so
# the digest costs the same at any batch size. The one-dimensional fields
# (log-probs, advantages, chosen slots) are read whole, and those alone
# differ between any two collections.
SAMPLE = 4096


def fingerprint(batch: object) -> str:
    """A short digest of every tensor field of the dataclass `batch`, to tell
    a rebuilt batch from the batch a step file was written against."""
    digest = hashlib.sha1()
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if value is None:
            digest.update(f"{field.name}:none;".encode())
            continue
        value = value.detach()
        digest.update(f"{field.name}:{tuple(value.shape)}:{value.dtype};".encode())
        if value.dim() > 1 and value.shape[0] > SAMPLE:
            value = value[:: -(-value.shape[0] // SAMPLE)]
        digest.update(value.contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def by_index(episodes: Sequence["Episode"]) -> list["Episode"]:
    """A collection in game order. A collector returns games in the order they
    finished, or worker by worker, and a resumed one returns its kept games
    first, so the batch, its shuffle and every step taken on it are the same
    either way only once the order is the games' own."""
    return sorted(episodes, key=lambda episode: episode.index)


def _pack(value: object) -> object:
    """Gauges to the host, a list of scalar tensors as one stacked tensor (one
    record in the file rather than one per step)."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        if value and all(torch.is_tensor(v) and v.dim() == 0 for v in value):
            return {"scalars": torch.stack([v.detach() for v in value]).cpu()}
        return [_pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_pack(v) for v in value)
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    return value


def _unpack(value: object, device: torch.device | str) -> object:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict) and set(value) == {"scalars"}:
        return list(value["scalars"].to(device).unbind())
    if isinstance(value, list):
        return [_unpack(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_unpack(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _unpack(v, device) for k, v in value.items()}
    return value


class Steps:
    """One update's step file: where it is written, how often, and what a
    resumed update continues from (`resume`, None for a fresh update)."""

    def __init__(
        self,
        path: Path,
        iteration: int,
        batch: str,
        every: int,
        resume: dict | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"a step checkpoint every {every} steps")
        self.path = Path(path)
        self.iteration = iteration
        self.batch = batch
        self.every = every
        self.resume = resume
        self.written = 0
        self.seconds = 0.0
        self.last = resume["step"] if resume else 0

    @classmethod
    def open(
        cls,
        directory: Path,
        iteration: int,
        batch: str,
        every: int,
        policy,
        optimiser: torch.optim.Optimizer,
    ) -> "Steps":
        """The step file for `iteration`'s update, loading the weights and the
        optimiser it holds when it is this iteration's and this batch's."""
        path = Path(directory) / STEP
        resume = None
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["iteration"] == iteration and state["batch"] == batch:
                policy.net.load_state_dict(state["net"])
                optimiser.load_state_dict(state["optimiser"])
                resume = state["progress"]
                print(
                    f"resumed iteration {iteration}'s update after step {resume['step']}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{path} holds iteration {state['iteration']}'s update over "
                    f"batch {state['batch']}, not iteration {iteration}'s over "
                    f"batch {batch}: removed, the update starts from the "
                    "iteration's start",
                    file=sys.stderr,
                )
                path.unlink()
        return cls(path, iteration, batch, every, resume)

    def write(self, policy, optimiser: torch.optim.Optimizer, progress: dict) -> None:
        from .loop import save

        started = time.perf_counter()
        save(
            self.path,
            {
                "iteration": self.iteration,
                "batch": self.batch,
                "net": policy.net.state_dict(),
                "optimiser": optimiser.state_dict(),
                "progress": _pack(progress),
            },
        )
        self.seconds += time.perf_counter() - started
        self.written += 1
        self.last = progress["step"]

    def log(self) -> dict:
        """The iteration row's columns: what the step file cost and where the
        update was resumed from."""
        return {
            "step_checkpoints": self.written,
            "step_checkpoint_seconds": round(self.seconds, 4),
            "resumed_at_step": self.resume["step"] if self.resume else None,
        }


def remove(directory: Path) -> None:
    """The run's step file, gone: its iteration is checkpointed."""
    (Path(directory) / STEP).unlink(missing_ok=True)


class Cursor:
    """An update's walk through its shuffled passes, kept resumable.

    `passes` stands in for `hexn.ppo._minibatches`: it yields the minibatches
    an update has still to step, in the order the uninterrupted update would
    have stepped them. `stepped` is called after each optimiser step and
    writes the step file when one is due; `finish` after the last.

    The shuffle is the one RNG draw an update makes, one `randperm` a pass
    from `generator` or, without one, the global torch RNG. A pass every step
    of which came before the resumed step is skipped without a draw; the
    pass the last stepped minibatch came from is redrawn from its saved
    state, which also leaves the RNG where the uninterrupted update left it.
    """

    def __init__(
        self,
        generator: torch.Generator | None,
        steps: Steps | None,
        policy,
        optimiser: torch.optim.Optimizer,
        gauges: Callable[[], dict],
    ) -> None:
        self.generator = generator
        self.steps = steps
        self.policy = policy
        self.optimiser = optimiser
        self.gauges = gauges
        self.taken = 0
        resume = steps.resume if steps is not None else None
        self.skip = resume["step"] if resume else 0
        self._saved = resume["shuffle_rng"] if resume else None
        self._pass_rng: Tensor | None = None

    def restored(self, device: torch.device | str) -> dict | None:
        """The gauges the interrupted update had accumulated, on `device`."""
        if self.steps is None or self.steps.resume is None:
            return None
        return _unpack(self.steps.resume["gauges"], device)

    def _rng(self) -> Tensor:
        if self.generator is not None:
            return self.generator.get_state()
        return torch.get_rng_state()

    def _set_rng(self, state: Tensor) -> None:
        if self.generator is not None:
            self.generator.set_state(state)
        else:
            torch.set_rng_state(state)

    def passes(self, size: int, minibatch: int) -> Iterator[Tensor]:
        from .ppo import _bounds, _minibatches

        count = len(_bounds(size, minibatch))
        start = self.taken
        if start + count < self.skip:
            self.taken += count
            return
        if start < self.skip:
            self._set_rng(self._saved)
        self._pass_rng = self._rng()
        for rows in _minibatches(size, minibatch, self.generator):
            if self.taken < self.skip:
                self.taken += 1
                continue
            yield rows

    def stepped(self) -> None:
        self.taken += 1
        if self.steps is not None and self.taken % self.steps.every == 0:
            self._write()

    def finish(self) -> None:
        """After the update's last step: the file holds the finished update,
        so a kill before the iteration checkpoint loses none of it."""
        if self.steps is not None and self.steps.last < self.taken:
            self._write()

    def _write(self) -> None:
        self.steps.write(
            self.policy,
            self.optimiser,
            {"step": self.taken, "shuffle_rng": self._pass_rng, "gauges": self.gauges()},
        )
