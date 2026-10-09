# SPDX-License-Identifier: GPL-3.0-only
"""One finished game as compact bytes, exactly: what a collection keeps on
disk (`hexn.durable.Partial`) and what a collector worker sends home
(`hexn.collect.ParallelCollector`).

An `Episode` pickled as objects is almost all observations -- one float32 row
per decision, about fourteen hundred values wide -- and consecutive rows are
nearly the same row: the board's hexes do not change, a building stays where
it was built. `encode` stacks a game's rows into one array, replaces each row
with its bitwise XOR against the row before it (an unchanged feature becomes
zero bits), packs the legality masks to bits, and deflates that with the rest
of the episode pickled beside it. `decode_all` undoes each step.

**Exact by construction.** No observation passes through float arithmetic on
the way: rows are handled as their uint32 bit patterns, so every value comes
back bit for bit, signed zeros included, and every other field is the object
that was pickled. zlib's own checksum catches a damaged file.

**One buffer per collection.** `decode_all` writes every game's rows into one
shared `(positions, width)` buffer in the network's packing order
(`hexn.model.packing`), the observations views into it with `_packed`/`_row`
set, so `hexn.model.pack` gathers a batch in one strided copy -- or, when the
batch is the collection in order, takes the buffer itself. Every game a
trainer learns from passes through here, whether it arrived over a worker's
pipe or was read back from disk after a resume, so the two cannot assemble
differently.

**Half precision for a half-precision update.** `dtype=np.float16` holds that
buffer in fp16. It is for a trainer whose update runs under fp16 autocast
(`hexn.ppo.PPOConfig.amp`) and for nothing else: observations reach the
network only through its four input `nn.Linear` embeddings
(`hexn.model.HexNet.forward`), which autocast feeds in fp16 by rounding the
float32 input to nearest-even -- the same rounding this cast applies -- so the
update reads the same inputs at half the host memory. A reader that is not
that update (the behaviour policy, an fp32 trainer, a gauge) gets float32.
"""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
import struct
import zlib
from typing import Sequence

import numpy as np

from hexset.encoding import Observation

from .selfplay import Episode, Transition

MAGIC = b"HXNG"
VERSION = 1
ZLIB = 1
# Speed over ratio: a worker deflates every game it finishes, and level 1
# already leaves the file a small fraction of the pickle.
LEVEL = 1

# magic, version, compressor, game index, positions, observation width, mask
# width: everything a reader needs to place a game before inflating it.
HEADER = struct.Struct("<4sHHqIII")
# Inside the deflated body: the byte lengths of the packed mask, the pickled
# graph and the pickled rest, after the observation rows.
SECTIONS = struct.Struct("<QQQ")

# The observation's arrays in the network's packing order (`hexn.model.packing`).
BLOCKS = ("hexes", "vertices", "edges", "globals")
_EPISODE = tuple(f.name for f in dataclasses.fields(Episode) if f.name != "trajectories")
_TRANSITION = tuple(
    f.name for f in dataclasses.fields(Transition) if f.name not in ("observation", "mask")
)


class Corrupt(ValueError):
    """Bytes that are not a whole game this module wrote."""


@dataclasses.dataclass(frozen=True)
class Head:
    index: int
    positions: int
    width: int
    mask_width: int


def head(blob: bytes) -> Head:
    """The game's index and sizes, read without inflating anything."""
    if len(blob) < HEADER.size:
        raise Corrupt("shorter than a game header")
    magic, version, compressor, index, positions, width, mask_width = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise Corrupt("not a game file")
    if version != VERSION or compressor != ZLIB:
        raise Corrupt(f"game file version {version}, compressor {compressor}: not readable here")
    return Head(index, positions, width, mask_width)


def index_of(blob: bytes) -> int:
    return head(blob).index


def encode(episode: Episode) -> bytes:
    """`episode` as bytes `decode_all` turns back into an equal episode.

    Refuses observations that are not float32 -- a half-precision collection
    (`decode_all(dtype=np.float16)`) is the update's copy, not a record -- and
    a game whose rows do not share one graph.
    """
    rows = [t for seat in episode.trajectories for t in seat]
    n = len(rows)
    if n:
        first = rows[0].observation
        shapes = tuple(getattr(first, name).shape for name in BLOCKS)
        sizes = [int(np.prod(shape)) for shape in shapes]
        obs = np.empty((n, sum(sizes)), dtype=np.float32)
        start = 0
        for name, size in zip(BLOCKS, sizes):
            blocks = [getattr(t.observation, name) for t in rows]
            if any(block.dtype != np.float32 for block in blocks):
                raise ValueError(f"observation {name} is not float32; only a float32 game is kept")
            obs[:, start : start + size] = np.stack(blocks).reshape(n, size)
            start += size
        graph = first.graph
        for t in rows:
            if t.observation.graph is not graph and pickle.dumps(t.observation.graph) != pickle.dumps(graph):
                raise ValueError(f"game {episode.index} has more than one graph")
        masks = np.stack([t.mask for t in rows])
        if masks.dtype != np.bool_:
            raise ValueError("a legality mask is not boolean")
        mask_width = masks.shape[1]
    else:
        shapes, obs, graph = (), np.empty((0, 0), dtype=np.float32), None
        masks, mask_width = np.empty((0, 0), dtype=np.bool_), 0
    bits = obs.view(np.uint32)
    delta = bits.copy()
    if n > 1:
        delta[1:] ^= bits[:-1]
    packed = np.packbits(masks, axis=1).tobytes()
    graph_bytes = pickle.dumps(graph, protocol=5)
    meta = pickle.dumps(
        (
            shapes,
            _EPISODE,
            tuple(getattr(episode, name) for name in _EPISODE),
            tuple(len(seat) for seat in episode.trajectories),
            _TRANSITION,
            [tuple(getattr(t, name) for name in _TRANSITION) for t in rows],
        ),
        protocol=5,
    )
    body = b"".join(
        (SECTIONS.pack(len(packed), len(graph_bytes), len(meta)), delta.tobytes(), packed, graph_bytes, meta)
    )
    header = HEADER.pack(MAGIC, VERSION, ZLIB, episode.index, n, obs.shape[1], mask_width)
    return header + zlib.compress(body, LEVEL)


def _inflate(blob: bytes) -> tuple[Head, bytes]:
    found = head(blob)
    try:
        body = zlib.decompress(memoryview(blob)[HEADER.size :])
    except zlib.error as error:
        raise Corrupt(f"game {found.index}: {error}") from None
    if len(body) < SECTIONS.size:
        raise Corrupt(f"game {found.index}: body too short")
    masks, graph, meta = SECTIONS.unpack_from(body)
    want = SECTIONS.size + 4 * found.positions * found.width + masks + graph + meta
    if len(body) != want:
        raise Corrupt(f"game {found.index}: body is {len(body)} bytes, its header says {want}")
    return found, body


def check(blob: bytes) -> None:
    """Raise `Corrupt` unless `blob` is a whole, undamaged game."""
    _inflate(blob)


def decode(blob: bytes, dtype=np.float32) -> Episode:
    """One game, its observations views into a buffer of its own."""
    return decode_all([blob], dtype)[0]


def decode_all(blobs: Sequence[bytes], dtype=np.float32) -> list[Episode]:
    """Every game, in the order given, with every observation a view into one
    shared buffer and every mask a row of one shared array -- one of each per
    observation shape, which in a collection means one.

    `dtype` is the buffer's: float32, the observations as collected, or
    float16 for an fp16-autocast update (see the module docstring)."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float16):
        raise ValueError(f"observations are held as float32 or float16, not {dtype}")
    heads = [head(blob) for blob in blobs]
    totals: dict[tuple[int, int], int] = {}
    for found in heads:
        key = (found.width, found.mask_width)
        totals[key] = totals.get(key, 0) + found.positions
    buffers = {key: np.empty((rows, key[0]), dtype=dtype) for key, rows in totals.items()}
    masks = {key: np.empty((rows, key[1]), dtype=np.bool_) for key, rows in totals.items()}
    shapes_of: dict[tuple[int, int], tuple] = {}
    cursor = dict.fromkeys(totals, 0)
    graphs: dict[bytes, object] = {}
    out = []
    for blob in blobs:
        found, body = _inflate(blob)
        n, width, key = found.positions, found.width, (found.width, found.mask_width)
        mask_len, graph_len, meta_len = SECTIONS.unpack_from(body)
        offset = SECTIONS.size
        start = cursor[key]
        buffer, mask = buffers[key], masks[key]
        rows = np.frombuffer(body, np.uint32, n * width, offset).reshape(n, width)
        offset += 4 * n * width
        if dtype == np.float32:
            np.bitwise_xor.accumulate(rows, axis=0, out=buffer[start : start + n].view(np.uint32))
        else:
            buffer[start : start + n] = np.bitwise_xor.accumulate(rows, axis=0).view(np.float32)
        if n:
            unpacked = np.unpackbits(
                np.frombuffer(body, np.uint8, mask_len, offset).reshape(n, -1),
                axis=1,
                count=found.mask_width,
            )
            mask[start : start + n] = unpacked.view(np.bool_)
        offset += mask_len
        graph_bytes = bytes(body[offset : offset + graph_len])
        offset += graph_len
        graph = graphs.get(graph_bytes)
        if graph is None:
            graph = graphs[graph_bytes] = pickle.loads(graph_bytes)
        shapes, episode_names, episode_values, counts, names, values = pickle.loads(
            body[offset : offset + meta_len]
        )
        if n:
            if shapes_of.setdefault(key, shapes) != shapes:
                raise ValueError("games of one width disagree on their observation shapes")
            sizes = [int(np.prod(shape)) for shape in shapes]
            bounds = np.cumsum([0, *sizes])
        trajectories = []
        k = 0
        for count in counts:
            seat = []
            for _ in range(count):
                row = start + k
                fields = dict(zip(names, values[k]))
                line = buffer[row]
                seat.append(
                    Transition(
                        observation=Observation(
                            graph=graph,
                            _packed=buffer,
                            _row=row,
                            **{
                                name: line[bounds[i] : bounds[i + 1]].reshape(shapes[i])
                                for i, name in enumerate(BLOCKS)
                            },
                        ),
                        mask=mask[row],
                        **fields,
                    )
                )
                k += 1
            trajectories.append(tuple(seat))
        if k != n:
            raise Corrupt(f"game {found.index}: {k} transitions for {n} positions")
        cursor[key] = start + n
        out.append(Episode(trajectories=tuple(trajectories), **dict(zip(episode_names, episode_values))))
    return out


@dataclass
class Columns:
    """A collection decoded without its per-row objects: what `decode_all`
    builds, as arrays and plain rows rather than one `Transition` and one
    `Observation` per position (`hexn.ppo.prepare.prepare_columns` assembles
    a batch from it). Games are in the order given; within a game the rows
    are seat-major, in each seat's decision order, as `encode` stacked them.

    `buffer` and `mask` are every row's observation (in the network's packing
    order) and legality mask. `games[k]` holds game k's episode fields (all
    but `trajectories`), its rows per seat, the index of its first row, the
    transition field names, and one tuple of field values per row. `bounds`
    are the observation blocks' column offsets (`BLOCKS` order)."""

    buffer: np.ndarray
    mask: np.ndarray
    games: list[dict]
    bounds: tuple[int, ...]


def decode_columns(blobs: Sequence[bytes], dtype=np.float32) -> Columns:
    """Every game, in the order given, as `Columns`: the same buffer and mask
    `decode_all` fills, bit for bit, without the objects."""
    import torch  # only here: the rest of this module stays torch-free
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float16):
        raise ValueError(f"observations are held as float32 or float16, not {dtype}")
    heads = [head(blob) for blob in blobs]
    widths = {(found.width, found.mask_width) for found in heads if found.positions}
    if len(widths) > 1:
        raise ValueError("a columnar collection needs one observation width")
    width, mask_width = widths.pop() if widths else (0, 0)
    rows_total = sum(found.positions for found in heads)
    buffer = np.empty((rows_total, width), dtype=dtype)
    mask = np.empty((rows_total, mask_width), dtype=np.bool_)
    games, bounds, cursor = [], None, 0
    for blob in blobs:
        found, body = _inflate(blob)
        n = found.positions
        mask_len, graph_len, meta_len = SECTIONS.unpack_from(body)
        offset = SECTIONS.size
        rows = np.frombuffer(body, np.uint32, n * found.width, offset).reshape(n, found.width)
        offset += 4 * n * found.width
        if dtype == np.float32:
            np.bitwise_xor.accumulate(rows, axis=0, out=buffer[cursor : cursor + n].view(np.uint32))
        else:
            wide = np.bitwise_xor.accumulate(rows, axis=0).view(np.float32)
            # torch's float16 cast rounds to nearest-even exactly as numpy's
            # and is many times faster; only a NaN's payload could come out
            # different, so a game holding one takes numpy's cast.
            narrow = torch.from_numpy(wide)
            if bool(torch.isnan(narrow).any()):
                buffer[cursor : cursor + n] = wide
            else:
                torch.from_numpy(buffer[cursor : cursor + n]).copy_(narrow)
        if n:
            unpacked = np.unpackbits(
                np.frombuffer(body, np.uint8, mask_len, offset).reshape(n, -1), axis=1, count=found.mask_width
            )
            mask[cursor : cursor + n] = unpacked.view(np.bool_)
        offset += mask_len + graph_len
        shapes, episode_names, episode_values, counts, names, values = pickle.loads(body[offset : offset + meta_len])
        if n and bounds is None:
            bounds = tuple(int(b) for b in np.cumsum([0, *[int(np.prod(shape)) for shape in shapes]]))
        if sum(counts) != n:
            raise Corrupt(f"game {found.index}: {sum(counts)} transitions for {n} positions")
        games.append({"fields": dict(zip(episode_names, episode_values)), "counts": tuple(counts),
                      "start": cursor, "names": names, "rows": values})
        cursor += n
    return Columns(buffer=buffer, mask=mask, games=games, bounds=bounds or (0,))
