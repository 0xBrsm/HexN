# SPDX-License-Identifier: GPL-3.0-only
"""The expert-iteration replay store: searched episodes, kept and reused.

Replaces `hexn.exit`'s old `deque[Batch]` window
(`--buffer-iterations`, retired). That deque held *assembled* batches -- the projection
already paid, the episodes themselves discarded -- which was the right shape
for reusing a corpus across an update's epochs, but the wrong one for two
things this store now does: reload a run's buffer across `--resume` without
recollecting it, and hand `hexn.reanalyse` the raw `Episode`s it needs to
rebuild a position and re-search it.

**Retention is by position, not by iteration.** `--buffer-iterations N` kept
exactly the last N iterations' batches whatever their size, so a change to
`--games-per-iteration` silently changed how much data a run actually trained
on. `--replay-positions` bounds `sum(len(episode) for episode in ...)`
directly, evicting whole iterations oldest-first once the newer ones already
cover the budget -- the newest iteration is never evicted on its own arrival,
even if it alone exceeds the budget, because an iteration that just cost real
search time to collect must be trainable at all.

**On disk, one shard per iteration**: `<checkpoint-dir>/replay/iter-NNNNN.pkl`,
each `pickle.dumps(list[Episode])` -- the exact format `--corpus` already
writes, reused rather than reinvented. `append` writes the new shard and
deletes any evicted one in the same call, so the directory's contents are
always exactly the retained window; `resume` reloads by reading whatever
shards are still there, in iteration order, and trusts the budget already
applied when they were written.

**Every write is atomic** (`hexn.durable.write_atomic`), `replace`'s
reanalysed rewrites included, so a kill mid-write leaves the old shard and
never a truncated one that would break the next resume. **A shard is its
iteration's completed collection.** A crash after `append` and before the
checkpoint leaves a shard for an iteration the checkpoint has not reached;
`resume(before=...)` holds such shards back and `adopt` takes each into the
window when the resumed run reaches its iteration, so that collection is
neither lost nor collected again. `append` of an iteration already held
replaces its shard and never keeps two.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .durable import write_atomic
from .selfplay import Episode

if TYPE_CHECKING:
    from hexset.actions import ActionSpace

    from .exit import Batch, DistillConfig
    from .model import Packing

SHARD = "iter-{iteration:05d}.pkl"


def shard_path(directory: Path, iteration: int) -> Path:
    return directory / SHARD.format(iteration=iteration)


@dataclass
class _Shard:
    iteration: int
    path: Path
    episodes: tuple[Episode, ...]
    # Lazily assembled and cached: `EpisodeStore.batch` pays the projection
    # once a shard and reuses it every later update, the same saving the old
    # `deque[Batch]` banked -- `replace` (a reanalysed iteration) clears this
    # so the next `batch()` call re-pays only the shard that actually changed.
    batch: "Batch | None" = None

    @property
    def positions(self) -> int:
        return sum(len(episode) for episode in self.episodes)


class EpisodeStore:
    """The last `~capacity` positions of searched episodes, held in memory and
    mirrored to `directory` one shard per iteration."""

    def __init__(self, directory: Path, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a replay store needs room for at least one position")
        self.directory = Path(directory)
        self.capacity = capacity
        self._shards: deque[_Shard] = deque()
        # Shards on disk for iterations a resumed run has not reached yet, by
        # iteration: collections a crash left ahead of their checkpoint.
        self.ahead: dict[int, tuple[Episode, ...]] = {}

    def append(self, iteration: int, episodes: list[Episode]) -> None:
        """Add one iteration's episodes: write its shard, then evict the
        oldest shards until the window is back at or under `capacity` -- or
        down to one shard, whichever comes first. An iteration already in
        the window has its shard replaced, not joined by a second."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = shard_path(self.directory, iteration)
        write_atomic(path, pickle.dumps(list(episodes)))
        self.ahead.pop(iteration, None)
        self._insert(_Shard(iteration, path, tuple(episodes)))

    def adopt(self, iteration: int) -> list[Episode]:
        """Take a shard `resume` held back into the window as its iteration's
        collection, exactly as `append` would have put it there, and return
        its episodes. The file is already on disk and is not rewritten."""
        episodes = self.ahead.pop(iteration)
        self._insert(_Shard(iteration, shard_path(self.directory, iteration), episodes))
        return list(episodes)

    def _insert(self, shard: _Shard) -> None:
        for index, held in enumerate(self._shards):
            if held.iteration == shard.iteration:
                self._shards[index] = shard
                break
        else:
            self._shards.append(shard)
        self._evict()

    def _evict(self) -> None:
        total = sum(shard.positions for shard in self._shards)
        while total > self.capacity and len(self._shards) > 1:
            oldest = self._shards.popleft()
            total -= oldest.positions
            oldest.path.unlink(missing_ok=True)

    def replace(self, iteration: int, episodes: list[Episode]) -> None:
        """Swap one retained iteration's episodes for reanalysed ones,
        rewriting its shard in place and invalidating its assembled-batch
        cache. Raises if `iteration` has since been evicted."""
        for index, shard in enumerate(self._shards):
            if shard.iteration == iteration:
                write_atomic(shard.path, pickle.dumps(list(episodes)))
                self._shards[index] = _Shard(iteration, shard.path, tuple(episodes))
                return
        raise KeyError(f"iteration {iteration} is not in the store")

    def shard_episodes(self, iteration: int) -> tuple[Episode, ...]:
        for shard in self._shards:
            if shard.iteration == iteration:
                return shard.episodes
        raise KeyError(f"iteration {iteration} is not in the store")

    def episodes(self) -> list[Episode]:
        """Every retained position, oldest iteration first."""
        return [e for shard in self._shards for e in shard.episodes]

    def positions(self) -> int:
        return sum(shard.positions for shard in self._shards)

    def iterations(self) -> list[int]:
        """Retained iteration numbers, oldest first."""
        return [shard.iteration for shard in self._shards]

    def batch(
        self, space: "ActionSpace", layout: "Packing", config: "DistillConfig"
    ) -> "Batch":
        """Every retained position, assembled and concatenated: the update's
        training batch, drawn uniformly over the whole store because nothing
        here weights one shard over another -- and the freshest iteration is
        always in it, because eviction never removes the shard that was just
        appended (see `EpisodeStore.append`)."""
        from .exit import Batch, assemble

        if not self._shards:
            raise ValueError("the replay store is empty; nothing to assemble")
        parts = []
        for shard in self._shards:
            if shard.batch is None:
                shard.batch = assemble(shard.episodes, space, layout, config)
            parts.append(shard.batch)
        return Batch.concat(parts)

    @classmethod
    def resume(
        cls, directory: Path, capacity: int, *, before: int | None = None
    ) -> "EpisodeStore":
        """Reload every shard already on disk, in iteration order.

        The shards on disk are exactly the retained window as of the last
        `append` -- eviction deletes a shard's file in the same call that
        drops it from memory -- so this trusts what is there rather than
        re-applying the budget; a directory left inconsistent by a crash mid
        `append` (a shard written, its eviction never run) is read as-is.

        `before` is the iteration the run resumes at. A shard at or after it
        was written ahead of the checkpoint -- collected, then the crash --
        and is held in `ahead` for `adopt` rather than loaded into the window
        early, where it would train updates that came before it.
        """
        store = cls(directory, capacity)
        directory = Path(directory)
        if not directory.exists():
            return store
        for path in sorted(directory.glob("iter-*.pkl")):
            iteration = int(path.stem.split("-")[1])
            episodes = tuple(pickle.loads(path.read_bytes()))
            if before is not None and iteration >= before:
                store.ahead[iteration] = episodes
            else:
                store._shards.append(_Shard(iteration, path, episodes))
        return store
