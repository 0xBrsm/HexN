# SPDX-License-Identifier: GPL-3.0-only
"""Adam execution shared by the trainers, including checkpoint restores."""
from collections.abc import Iterable

import torch
from torch import Tensor


def _restore_execution(optimizer, state):
    # Loading uses these flags to place the step counters. Set them before
    # torch casts state, and keep every saved moment and hyperparameter.
    return state | {
        "param_groups": [
            group | {"fused": optimizer.defaults["fused"],
                     "foreach": optimizer.defaults["foreach"]}
            for group in state["param_groups"]
        ]
    }


def _build(kind, parameters: Iterable[Tensor], kwargs):
    parameters = list(parameters)
    fused = bool(parameters) and all(p.device.type == "cuda" for p in parameters)
    optimizer = kind(parameters, fused=fused, foreach=False if fused else None, **kwargs)
    optimizer.register_load_state_dict_pre_hook(_restore_execution)
    return optimizer


def adam(parameters: Iterable[Tensor], **kwargs) -> torch.optim.Adam:
    """Fused Adam on CUDA/HIP parameters, ordinary Adam on CPU."""
    return _build(torch.optim.Adam, parameters, kwargs)


def adamw(parameters: Iterable[Tensor], **kwargs) -> torch.optim.AdamW:
    """Fused AdamW on CUDA/HIP parameters, ordinary AdamW on CPU."""
    return _build(torch.optim.AdamW, parameters, kwargs)
