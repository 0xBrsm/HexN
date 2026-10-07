# SPDX-License-Identifier: GPL-3.0-only
"""Keep every unit of a batch the moment it finishes, and resume from it.

A training iteration, a collection and a benchmark are all batches of units
-- games, iterations, probed positions -- and a batch that holds its units in
memory until it ends loses all of them to one crash, one kill or one dead
container, and cannot be read part-way. So each unit goes to disk as it
finishes, in one of three shapes:

- `write_atomic`: a binary file (a game, a replay shard), written under a
  temporary name, flushed to the disk and renamed over the target. A kill
  mid-write leaves the previous file or none, never a truncated one.
- `append_line` / `read_lines`: one JSON line per unit, flushed and fsynced
  before the call returns. A reader drops a torn last line -- the one a dying
  writer was part-way through -- and the next `append_line` cuts it first.
  HexSet 0.77.0's `hexset.gamelog` keeps the same line discipline; this copy
  goes when the engine pin reaches it.
- `Partial`: one collection in progress, a directory holding one file per
  finished game and, for a cohort, the plan of every index it deals. A
  resumed collection loads what is there and deals only the rest.

`Rows` is the line journal a benchmark keeps, one line per probed position.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .selfplay import Episode


def write_atomic(path: str | os.PathLike, data: bytes) -> None:
    """Write `data` to `path` so a crash leaves the old file or the new one."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        # On disk now, and nothing here is read back until a resume: leave
        # it out of the page cache, which otherwise holds every game an
        # iteration keeps (gigabytes of memory the box counts as used).
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.replace(temporary, path)


def cut_torn_tail(path: str | os.PathLike) -> None:
    """Drop an unfinished last line from `path`, so the next line starts on a
    line of its own rather than completing a dead writer's."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if not size:
            return
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        data = handle.read()
        handle.truncate(data.rfind(b"\n") + 1)
        handle.flush()
        os.fsync(handle.fileno())


def append_line(path: str | os.PathLike, entry: dict) -> str:
    """Append `entry` as one JSON line, durably, and return the line.

    Serialised with `json.dumps`' defaults, as the trainers' logs always
    were, so a row reads back exactly as it printed.
    """
    line = json.dumps(entry)
    cut_torn_tail(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return line


def read_lines(path: str | os.PathLike) -> list[dict]:
    """Every complete line of `path`, in file order; `[]` for no file.

    A torn last line is dropped. A line that does not parse anywhere before
    it raises `ValueError`: that is an edited or corrupted file, not a crash.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    torn = lines.pop()  # "" when the file ends in a newline
    out = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not a JSON line: {error}") from error
    if torn.strip():
        try:
            out.append(json.loads(torn))
        except json.JSONDecodeError:
            pass
    return out


PLAN = "plan.json"


@dataclass(frozen=True)
class Partial:
    """One collection in progress: `<index>.pkl` per finished game.

    A cohort also writes `plan.json`, every index it deals, before its first
    game can finish, so a resumed cohort knows which games are missing
    rather than guessing from the ones that are there. A stream has no plan:
    its games are whichever finish first, so a resumed stream only tops up
    the count.

    A path and nothing else, so it pickles: a collector worker is handed one
    and writes its own games into it (`hexn.collect.ParallelCollector`).
    """

    directory: Path

    def keep(self, episode: "Episode") -> None:
        """A collector's `sink`: one finished game, atomically."""
        self.directory.mkdir(parents=True, exist_ok=True)
        write_atomic(self.directory / f"{episode.index}.pkl", pickle.dumps(episode))

    def finished(self) -> list[int]:
        """The indices of the games on disk, from the file names alone."""
        if not self.directory.exists():
            return []
        return sorted(
            int(path.stem)
            for path in self.directory.glob("*.pkl")
            if path.stem.isdigit()
        )

    def done(self) -> list["Episode"]:
        """Every game on disk, by index. One that does not load is removed,
        which makes it missing, and a resumed collection plays it again."""
        out = []
        for index in self.finished():
            path = self.directory / f"{index}.pkl"
            try:
                out.append(pickle.loads(path.read_bytes()))
            except (EOFError, pickle.UnpicklingError):
                path.unlink(missing_ok=True)
        return out

    def plan(self) -> list[int] | None:
        path = self.directory / PLAN
        if not path.exists():
            return None
        return [int(index) for index in json.loads(path.read_text())]

    def write_plan(self, indices: Iterable[int]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_atomic(self.directory / PLAN, json.dumps(list(indices)).encode())

    def indices(self) -> set[int]:
        """Every index this collection has planned or finished."""
        return set(self.plan() or ()) | set(self.finished())

    def remove(self) -> None:
        """Called once what it holds is kept somewhere final."""
        shutil.rmtree(self.directory, ignore_errors=True)


def partial(root: Path, iteration: int) -> Partial:
    """Iteration `iteration`'s collection under a run's `root`."""
    return Partial(Path(root) / f"iter-{iteration:05d}")


def partials(root: Path) -> dict[int, Partial]:
    """Every collection under `root` a crash left behind, by iteration."""
    root = Path(root)
    if not root.exists():
        return {}
    return {
        int(path.name[len("iter-"):]): Partial(path)
        for path in root.glob("iter-*")
        if path.is_dir() and path.name[len("iter-"):].isdigit()
    }


def prune(root: Path, through: int) -> None:
    """Remove the collections of every iteration up to `through`: a
    checkpoint now holds what they were for."""
    for iteration, held in partials(root).items():
        if iteration <= through:
            held.remove()


class Rows:
    """A benchmark's journal: a header line, then one line per position.

    The header is everything the run's numbers depend on; a journal whose
    header differs is another run's, and is refused rather than extended.
    `kept` is every position already written, by its index, so a rerun
    skips them. Each line carries the position's `fingerprint` too, which
    the rerun checks against the position it rebuilt, so a seeding pass that
    did not reproduce cannot mix two sets of positions into one result.
    """

    def __init__(self, path: str | os.PathLike, header: dict) -> None:
        self.path = Path(path)
        # As it reads back, so a tuple in it compares equal to its own list.
        header = json.loads(json.dumps(header, default=str))
        entries = read_lines(self.path)
        if entries and entries[0].get("header") != header:
            raise SystemExit(
                f"{self.path} is the journal of a run with other settings; "
                "name another --rows, or remove it to start over"
            )
        if not entries:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            append_line(self.path, {"header": header})
        self.kept = {
            entry["position"]: entry for entry in entries[1:] if "position" in entry
        }

    def check(self, position: int, fingerprint: str) -> dict | None:
        """The kept entry for `position`, or None; refuses a mismatch."""
        entry = self.kept.get(position)
        if entry is not None and entry["fingerprint"] != fingerprint:
            raise SystemExit(
                f"{self.path}: position {position} is not the position this "
                "run rebuilt; the seeding pass did not reproduce"
            )
        return entry

    def add(self, position: int, fingerprint: str, **fields) -> None:
        entry = {"position": position, "fingerprint": fingerprint, **fields}
        append_line(self.path, entry)
        self.kept[position] = entry


def rows_path(name: str, header: dict) -> Path:
    """A journal name unique to these settings, so a rerun of the same run
    finds its own journal and a run with other settings never does."""
    digest = hashlib.sha1(
        json.dumps(header, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    return Path(f"{name}-{digest}.rows.jsonl")


def fingerprint(*parts: object) -> str:
    """A short stable digest of `repr(parts)`, for `Rows`."""
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


def resume_base(games_started: int, *kept: Iterable[int]) -> int:
    """The first game index a resumed collector may deal.

    Past the checkpoint's `games_started`, and past every index a collection
    the crash interrupted already planned or finished (`Partial.indices`,
    or a replay shard written ahead of its checkpoint), so no new deal
    repeats one of theirs. Skipped indices are unused seeds, not lost games.
    """
    top = max((index for indices in kept for index in indices), default=-1)
    return max(games_started, top + 1)
