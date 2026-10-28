"""Where a test's games go: every game a collector plays is kept
(`hexn.selfplay.Collector(keep=...)`, a `partial`), and every duel names a
journal. All of it under one temporary directory removed when the session
ends."""
import atexit
import itertools
import os
import shutil
import tempfile
from pathlib import Path

from hexn.durable import Partial

_DIR = tempfile.mkdtemp(prefix="hexn-test-games-")
atexit.register(shutil.rmtree, _DIR, True)
_COUNT = itertools.count()


def kept() -> Partial:
    """A collection no other call has used."""
    return Partial(Path(_DIR) / f"games-{os.getpid()}-{next(_COUNT)}")


def fresh(name: str = "games") -> str:
    """A journal path no other call has used."""
    return os.path.join(_DIR, f"{name}-{os.getpid()}-{next(_COUNT)}.jsonl")
