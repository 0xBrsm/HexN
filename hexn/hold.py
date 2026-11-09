# SPDX-License-Identifier: GPL-3.0-only
"""The hold model a torch checkpoint declares, carried from parent to child.

A checkpoint may declare how the other seats' held development cards are
kept (`hexset.holding.HoldModel`), as the metadata pairs an ONNX file
declares it by (`hold_shape`, `hold_scale`; `HoldModel.meta`), under the
checkpoint's `hold` key. The loader reads it into `hexn.netbot.Loaded.hold`,
and the network seats deal their search and gate worlds by it
(`hexset.clients.netbot`). A checkpoint without the key declares none, and
its worlds deal every unseen card alike.

The trainers carry the key from the checkpoint they start or resume from to
every checkpoint they write (`carried`), and the exporters copy it into the
ONNX file's metadata, so a declared model reaches every descendant. HexN
carries no parameters of its own for it: a root checkpoint declares one
through `stamp`, with the parameters its caller passes.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from hexset.clients._modelmeta import hold_config
from hexset.holding import HoldModel

KEY = "hold"


def declared(state: dict) -> HoldModel | None:
    """The hold model the checkpoint `state` declares, or `None`. A declared
    model that cannot be read raises `ValueError`."""
    return hold_config(dict(state.get(KEY) or {}))


def carried(state: dict | None) -> dict:
    """The entry to write into a checkpoint descended from `state`: its
    declared hold model, or nothing when it declares none."""
    if state is None or declared(state) is None:
        return {}
    return {KEY: dict(state[KEY])}


def stamp(source: Path, model: HoldModel, target: Path | None = None) -> str:
    """Write the torch checkpoint `source` to `target` (`source` itself when
    `None`) declaring `model`, through a temporary file and a rename; every
    other entry is written back as loaded. A checkpoint that already declares
    another model is refused rather than overwritten. Returns the new file's
    sha256."""
    import torch

    target = source if target is None else target
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{source} is not a checkpoint dictionary")
    current = declared(state)
    if current is not None and current != model:
        raise ValueError(f"{source} already declares another hold model: {current}")
    state[KEY] = model.meta()
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, source.stat().st_mode & 0o7777)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return hashlib.sha256(target.read_bytes()).hexdigest()


__all__ = ["KEY", "carried", "declared", "stamp"]
