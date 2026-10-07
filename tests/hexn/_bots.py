# SPDX-License-Identifier: GPL-3.0-only
"""`needs(*names)`: skip a test unless a loaded runtime provides every bot it names.

The bots come from the runtime `pytest --runtime <module>` loaded
(`tests/conftest.py`); registration happens before collection, so the mark
reads it as each test module is imported.
"""

from __future__ import annotations

import pytest
from hexset.arena import registered_presets


def needs(*names: str):
    """A mark that skips unless each of `names` is a registered preset."""
    missing = sorted(set(names) - set(registered_presets()))
    return pytest.mark.skipif(
        bool(missing),
        reason=f"no runtime registers {', '.join(missing)}; pass pytest --runtime <module>",
    )
