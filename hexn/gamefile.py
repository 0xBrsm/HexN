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

**Contract 9.** A game of `Transition9`s is its own format (`MAGIC9`): each
row is a `hexset.contract9.flatten`ed observation, delta-coded the same way,
and its decision's tensors (`hexset.decisions.Decision`) go beside it -- the
flat masks as bits, the setup masks as bits for the rows that have one, and
the Road Building and discard candidates end to end with each row's count.
Its header carries `VERSION9`: a row is one part of a run of options, and the
snake seat's double turn is two rows, never one.
`decode_all` and `decode_columns` read either format and refuse a collection
that mixes them; the columns of a contract-9 collection are `Columns9`.
Under fp16 the observation's every field is held in half, the seats' reach
and the owners' pointers included. The pointers are small integers, exact in
half; the reach is read by the network outside its input embeddings, so
rounding it here is not bit for bit what an fp32 buffer under autocast
gives, as it is for contract 8 -- the difference is of the order of
autocast's own rounding, and decoupled PPO's proximal pass reads the same
buffer as the update.
"""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
import struct
import zlib
from typing import Sequence

import numpy as np

from hexset.decisions import Decision
from hexset.encoding import Observation

from .selfplay import Episode, Transition, Transition9

MAGIC = b"HXNG"
MAGIC9 = b"HXN9"
VERSION = 1
# A contract-9 game's own version: 2 since a row is one part of a run of
# options (`Transition9.decision_id`, `.part`) and no row is a double turn.
# Version 1 is refused: its rows were credited one by one.
VERSION9 = 2
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
# Contract 9's: the flat masks, the setup masks, the Road Building candidates,
# the discard candidates and the pickled rest, after the observation rows.
SECTIONS9 = struct.Struct("<QQQQQ")
# Road Building candidates are (first, second) edges, int16; a discard is the
# hand kept and the cards given, five counts each, int8 (`Decision`).
RB_WIDTH, DISCARD_WIDTH = 2, 10

# The observation's arrays in the network's packing order (`hexn.model.packing`).
BLOCKS = ("hexes", "vertices", "edges", "globals")
_EPISODE = tuple(f.name for f in dataclasses.fields(Episode) if f.name != "trajectories")
_TRANSITION = tuple(
    f.name for f in dataclasses.fields(Transition) if f.name not in ("observation", "mask")
)
_TRANSITION9 = tuple(
    f.name for f in dataclasses.fields(Transition9) if f.name not in ("observation", "decision")
)


class Corrupt(ValueError):
    """Bytes that are not a whole game this module wrote."""


@dataclasses.dataclass(frozen=True)
class Head:
    index: int
    positions: int
    width: int
    mask_width: int
    contract: str = "8"


def head(blob: bytes) -> Head:
    """The game's index and sizes, read without inflating anything."""
    if len(blob) < HEADER.size:
        raise Corrupt("shorter than a game header")
    magic, version, compressor, index, positions, width, mask_width = HEADER.unpack_from(blob)
    if magic not in (MAGIC, MAGIC9):
        raise Corrupt("not a game file")
    if version != (VERSION9 if magic == MAGIC9 else VERSION) or compressor != ZLIB:
        raise Corrupt(f"game file version {version}, compressor {compressor}: not readable here")
    return Head(index, positions, width, mask_width, "9" if magic == MAGIC9 else "8")


def index_of(blob: bytes) -> int:
    return head(blob).index


def encode(episode: Episode) -> bytes:
    """`episode` as bytes `decode_all` turns back into an equal episode.

    Refuses observations that are not float32 -- a half-precision collection
    (`decode_all(dtype=np.float16)`) is the update's copy, not a record -- and
    a game whose rows do not share one graph.
    """
    rows = [t for seat in episode.trajectories for t in seat]
    if rows and isinstance(rows[0], Transition9):
        return _encode9(episode, rows)
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


def _encode9(episode: Episode, rows: Sequence[Transition9]) -> bytes:
    """`encode` for a game of contract-9 decisions (`MAGIC9`)."""
    n = len(rows)
    if any(not isinstance(t, Transition9) for t in rows):
        raise ValueError(f"game {episode.index} mixes contract-9 decisions with others")
    if any(t.decision.double for t in rows):
        raise ValueError(f"game {episode.index} holds a double turn; self-play asks it as two setup turns")
    obs = np.stack([t.observation for t in rows])
    if obs.dtype != np.float32:
        raise ValueError("a contract-9 observation is not float32; only a float32 game is kept")
    masks = np.stack([t.decision.flat for t in rows])
    if masks.dtype != np.bool_:
        raise ValueError("a legality mask is not boolean")
    setup_rows = [k for k, t in enumerate(rows) if t.decision.setup.any()]
    setups = (np.stack([rows[k].decision.setup for k in setup_rows]) if setup_rows
              else np.zeros((0, rows[0].decision.setup.shape[0]), dtype=np.bool_))
    rb = np.concatenate([t.decision.road_building.reshape(-1, RB_WIDTH) for t in rows]).astype(np.int16)
    discard = np.concatenate([t.decision.discard.reshape(-1, DISCARD_WIDTH) for t in rows]).astype(np.int8)
    bits = obs.view(np.uint32)
    delta = bits.copy()
    if n > 1:
        delta[1:] ^= bits[:-1]
    packed = np.packbits(masks, axis=1).tobytes()
    setup_bits = np.packbits(setups, axis=1).tobytes()
    meta = pickle.dumps(
        (
            setups.shape[1],
            _EPISODE,
            tuple(getattr(episode, name) for name in _EPISODE),
            tuple(len(seat) for seat in episode.trajectories),
            _TRANSITION9,
            [tuple(getattr(t, name) for name in _TRANSITION9) for t in rows],
            setup_rows,
            [len(t.decision.road_building) for t in rows],
            [len(t.decision.discard) for t in rows],
        ),
        protocol=5,
    )
    body = b"".join((
        SECTIONS9.pack(len(packed), len(setup_bits), rb.nbytes, discard.nbytes, len(meta)),
        delta.tobytes(), packed, setup_bits, rb.tobytes(), discard.tobytes(), meta,
    ))
    header = HEADER.pack(MAGIC9, VERSION9, ZLIB, episode.index, n, obs.shape[1], masks.shape[1])
    return header + zlib.compress(body, LEVEL)


def _inflate(blob: bytes) -> tuple[Head, bytes]:
    found = head(blob)
    try:
        body = zlib.decompress(memoryview(blob)[HEADER.size :])
    except zlib.error as error:
        raise Corrupt(f"game {found.index}: {error}") from None
    if found.contract == "9":
        if len(body) < SECTIONS9.size:
            raise Corrupt(f"game {found.index}: body too short")
        want = SECTIONS9.size + 4 * found.positions * found.width + sum(SECTIONS9.unpack_from(body))
        if len(body) != want:
            raise Corrupt(f"game {found.index}: body is {len(body)} bytes, its header says {want}")
        return found, body
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
    if _contract(heads) == "9":
        columns = decode_columns(blobs, dtype)
        return [
            Episode(trajectories=columns.trajectories(k), **game["fields"])
            for k, game in enumerate(columns.games)
        ]
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


def _contract(heads: Sequence[Head]) -> str:
    """The one contract a collection's games are in."""
    # A game nobody recorded a decision in has no rows to say which.
    contracts = {found.contract for found in heads if found.positions}
    if len(contracts) > 1:
        raise ValueError("a collection mixes contract-8 and contract-9 games")
    return contracts.pop() if contracts else "8"


def _rows_into(buffer: np.ndarray, at: int, rows: np.ndarray) -> None:
    """A game's delta-coded rows, undone into `buffer[at:]` in its dtype."""
    import torch  # only here: the rest of this module stays torch-free

    n = len(rows)
    if buffer.dtype == np.float32:
        np.bitwise_xor.accumulate(rows, axis=0, out=buffer[at : at + n].view(np.uint32))
        return
    wide = np.bitwise_xor.accumulate(rows, axis=0).view(np.float32)
    # torch's float16 cast rounds to nearest-even exactly as numpy's and is
    # many times faster; only a NaN's payload could come out different, so a
    # game holding one takes numpy's cast.
    narrow = torch.from_numpy(wide)
    if bool(torch.isnan(narrow).any()):
        buffer[at : at + n] = wide
    else:
        torch.from_numpy(buffer[at : at + n]).copy_(narrow)


@dataclass
class Columns9:
    """A contract-9 collection decoded without its per-row objects, as
    `Columns` is for contract 8. `buffer` holds every row's flattened
    observation and `mask` its decision's flat block; `setup` the setup masks
    of the rows that have one, `setup_row` each row's index into it (-1 for
    none); `rb` and `disc` the Road Building and discard candidates end to
    end, row `r`'s at `rb_off[r]:rb_off[r + 1]`. `games` as `Columns`' (the
    transition fields `Transition9` keeps beside its observation and decision,
    its run's `decision_id` and `part` among them). No row is a double turn."""

    buffer: np.ndarray
    mask: np.ndarray
    setup: np.ndarray
    setup_row: np.ndarray
    rb: np.ndarray
    rb_off: np.ndarray
    disc: np.ndarray
    disc_off: np.ndarray
    games: list[dict]

    def decision(self, row: int) -> Decision:
        """Row `row`'s decision, its arrays views into the columns."""
        at = self.setup_row[row]
        return Decision(
            flat=self.mask[row],
            setup=self.setup[at] if at >= 0 else self._no_setup(),
            double=False,
            road_building=self.rb[self.rb_off[row] : self.rb_off[row + 1]],
            discard=self.disc[self.disc_off[row] : self.disc_off[row + 1]],
        )

    def _no_setup(self) -> np.ndarray:
        empty = getattr(self, "_empty", None)
        if empty is None:
            empty = self._empty = np.zeros(self.setup.shape[1], dtype=np.bool_)
            empty.flags.writeable = False
        return empty

    def trajectories(self, k: int) -> tuple[tuple[Transition9, ...], ...]:
        """Game `k`'s rows as `Transition9`s, seat by seat."""
        game = self.games[k]
        out, row = [], game["start"]
        for count in game["counts"]:
            seat = []
            for values in game["rows"][row - game["start"] : row - game["start"] + count]:
                seat.append(Transition9(observation=self.buffer[row], decision=self.decision(row),
                                        **dict(zip(game["names"], values))))
                row += 1
            out.append(tuple(seat))
        return tuple(out)


def _decode_columns9(blobs: Sequence[bytes], heads: Sequence[Head], dtype) -> Columns9:
    widths = {(found.width, found.mask_width) for found in heads if found.positions}
    if len(widths) > 1:
        raise ValueError("a columnar collection needs one observation width")
    width, mask_width = widths.pop() if widths else (0, 0)
    total = sum(found.positions for found in heads)
    buffer = np.empty((total, width), dtype=dtype)
    mask = np.empty((total, mask_width), dtype=np.bool_)
    setup_row = np.full(total, -1, dtype=np.int64)
    setups, rbs, discs, rb_counts, disc_counts, games = [], [], [], [], [], []
    cursor = held = 0
    incidences = 0
    for blob in blobs:
        found, body = _inflate(blob)
        n = found.positions
        if found.contract == "8":                   # no rows: only the episode's own fields
            _, _, meta_len = SECTIONS.unpack_from(body)
            _, episode_names, episode_values, counts, names, _ = pickle.loads(body[len(body) - meta_len :])
            games.append({"fields": dict(zip(episode_names, episode_values)), "counts": tuple(counts),
                          "start": cursor, "names": names, "rows": []})
            continue
        mask_len, setup_len, rb_len, disc_len, meta_len = SECTIONS9.unpack_from(body)
        offset = SECTIONS9.size
        _rows_into(buffer, cursor, np.frombuffer(body, np.uint32, n * found.width, offset).reshape(n, found.width))
        offset += 4 * n * found.width
        if n:
            mask[cursor : cursor + n] = np.unpackbits(
                np.frombuffer(body, np.uint8, mask_len, offset).reshape(n, -1), axis=1, count=found.mask_width
            ).view(np.bool_)
        offset += mask_len
        setup_bits = np.frombuffer(body, np.uint8, setup_len, offset)
        offset += setup_len
        rb = np.frombuffer(body, np.int16, rb_len // 2, offset).reshape(-1, RB_WIDTH)
        offset += rb_len
        disc = np.frombuffer(body, np.int8, disc_len, offset).reshape(-1, DISCARD_WIDTH)
        offset += disc_len
        (incidences, episode_names, episode_values, counts, names, values, set_rows, rb_count,
         disc_count) = pickle.loads(body[offset : offset + meta_len])
        if sum(counts) != n or len(values) != n:
            raise Corrupt(f"game {found.index}: {sum(counts)} transitions for {n} positions")
        if set_rows:
            setups.append(np.unpackbits(setup_bits.reshape(len(set_rows), -1), axis=1, count=incidences)
                          .view(np.bool_))
            setup_row[cursor + np.asarray(set_rows, dtype=np.int64)] = held + np.arange(len(set_rows))
            held += len(set_rows)
        rbs.append(rb)
        discs.append(disc)
        rb_counts.extend(rb_count)
        disc_counts.extend(disc_count)
        games.append({"fields": dict(zip(episode_names, episode_values)), "counts": tuple(counts),
                      "start": cursor, "names": names, "rows": values})
        cursor += n

    def offsets(counts):
        return np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64)

    return Columns9(
        buffer=buffer, mask=mask,
        setup=np.concatenate(setups) if setups else np.zeros((0, incidences), dtype=np.bool_),
        setup_row=setup_row,
        rb=np.concatenate(rbs) if rbs else np.zeros((0, RB_WIDTH), dtype=np.int16), rb_off=offsets(rb_counts),
        disc=np.concatenate(discs) if discs else np.zeros((0, DISCARD_WIDTH), dtype=np.int8),
        disc_off=offsets(disc_counts), games=games,
    )


def decode_columns(blobs: Sequence[bytes], dtype=np.float32) -> Columns | Columns9:
    """Every game, in the order given, as `Columns` (`Columns9` for a
    contract-9 collection): the same buffer and mask `decode_all` fills, bit
    for bit, without the objects."""
    import torch  # only here: the rest of this module stays torch-free
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float16):
        raise ValueError(f"observations are held as float32 or float16, not {dtype}")
    heads = [head(blob) for blob in blobs]
    if _contract(heads) == "9":
        return _decode_columns9(blobs, heads, dtype)
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
