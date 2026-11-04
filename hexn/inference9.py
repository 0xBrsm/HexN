# SPDX-License-Identifier: GPL-3.0-only
"""Worker-owned CPU BF16 inference for the original contract-9 policy.

``WorkerInference`` caches Linear weight casts and lazily traces the original
forward on its first real input. It leaves decision heads, sampling, batch
shapes and configured precision with the policy. Canonical parameters remain
FP32; ordinary in-place weight loads update the cache and live trace aliases.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref

import torch
from torch import nn

from .model9 import HexNet, Prediction
from .policy9 import NetworkPolicy9

__all__ = ["WorkerInference"]

_INPUTS = ("seats", "table", "hexes", "vertices", "vertex_owner", "road_owner", "reach")
_BUFFER = "_inference9_weight"
_ENTRY = "_inference9_entry"
_ACTIVE = None
_JIT_METHODS = ("_jit_can_fuse_on_cpu", "_jit_texpr_fuser_enabled", "_jit_llga_enabled",
                "_get_graph_executor_optimize", "_jit_set_autocast_mode",
                "_jit_override_can_fuse_on_cpu", "_jit_set_texpr_fuser_enabled",
                "_jit_set_llga_enabled", "_set_graph_executor_optimize")


def _signature(tensor):
    return (tensor.dtype, tensor.device, tuple(tensor.shape), tensor.stride(),
            tensor.storage_offset(), tensor.data_ptr(), tensor.untyped_storage().data_ptr())


@dataclass
class _Entry:
    module: nn.Linear
    parameter: nn.Parameter
    buffer: torch.Tensor
    version: int
    signature: tuple
    owner: weakref.ReferenceType

    def refresh(self):
        owner = self.owner()
        if owner is None or not owner.valid:
            return False
        parameter = self.module._parameters.get("weight")
        if (parameter is not self.parameter or _signature(parameter) != self.signature
                or self.module._buffers.get(_BUFFER) is not self.buffer):
            owner.invalidate()
            return False
        if parameter._version != self.version:
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                self.buffer.copy_(parameter.detach())
            self.version = parameter._version
        return True

    def visible(self):
        owner = self.owner()
        if (owner is not None and owner.thread == threading.get_ident()
                and not self.module.training and not torch.is_grad_enabled()
                and torch.is_autocast_enabled("cpu")
                and torch.get_autocast_dtype("cpu") == torch.bfloat16 and self.refresh()):
            return self.buffer
        return self.module._parameters.get("weight")


class _CachedLinear(nn.Linear):
    def __getattribute__(self, name):
        if name == "weight":
            entry = object.__getattribute__(self, "__dict__").get(_ENTRY)
            if entry is not None:
                return entry.visible()
        return super().__getattribute__(name)


class _WeightCache:
    def __init__(self, net):
        self.entries = []
        self.valid, self.closed = True, False
        self.thread = threading.get_ident()
        modules = [module for module in net.modules() if isinstance(module, nn.Linear)]
        if not modules or any(type(module) is not nn.Linear for module in modules):
            raise ValueError("weight caching requires ordinary Linear modules")
        if any(_BUFFER in module._buffers or _ENTRY in module.__dict__ for module in modules):
            raise ValueError("a weight cache is already installed")
        try:
            for module in modules:
                parameter = module._parameters["weight"]
                # These remain writable by normal state_dict loads even when
                # installation happens from inside inference mode.
                with torch.inference_mode(False), torch.no_grad():
                    buffer = parameter.detach().to(torch.bfloat16)
                entry = _Entry(module, parameter, buffer, parameter._version,
                               _signature(parameter), weakref.ref(self))
                self.entries.append(entry)
                module.register_buffer(_BUFFER, buffer, persistent=False)
                object.__setattr__(module, _ENTRY, entry)
                module.__class__ = _CachedLinear
        except Exception:
            self.close()
            raise

    def invalidate(self):
        self.valid = False

    def refresh(self):
        return self.valid and not self.closed and all(entry.refresh() for entry in self.entries)

    def close(self):
        if self.closed:
            return
        self.invalidate()
        for entry in self.entries:
            module = entry.module
            if module.__dict__.get(_ENTRY) is entry or module._buffers.get(_BUFFER) is entry.buffer:
                if module.__dict__.get(_ENTRY) is entry:
                    object.__delattr__(module, _ENTRY)
                if type(module) is _CachedLinear:
                    module.__class__ = nn.Linear
                module._buffers.pop(_BUFFER, None)
                module._non_persistent_buffers_set.discard(_BUFFER)
        self.closed = True


class _Settings:
    """Preserve optimized execution without fusion or autocast rewrites."""
    def __init__(self):
        c = torch._C
        self.previous = (c._jit_can_fuse_on_cpu(), c._jit_texpr_fuser_enabled(),
                         c._jit_llga_enabled(), c._get_graph_executor_optimize())
        self.autocast = c._jit_set_autocast_mode(False)
        self.closed = False
        try:
            c._jit_override_can_fuse_on_cpu(False)
            c._jit_set_texpr_fuser_enabled(False)
            c._jit_set_llga_enabled(False)
            c._set_graph_executor_optimize(True)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        c = torch._C
        fuse, texpr, llga, optimize = self.previous
        c._jit_override_can_fuse_on_cpu(fuse)
        c._jit_set_texpr_fuser_enabled(texpr)
        c._jit_set_llga_enabled(llga)
        c._set_graph_executor_optimize(optimize)
        c._jit_set_autocast_mode(self.autocast)
        self.closed = True


class _Trunk(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, seats, table, hexes, vertices, vertex_owner, road_owner, reach):
        out = self.net(seats, table, hexes, vertices, vertex_owner, road_owner, reach)
        points = out.value.new_empty((seats.shape[0], 0)) if out.points is None else out.points
        return (out.value, out.value_logits, points, out.logits, out.road,
                out.me, out.table, out.vertices, out.hexes, out.seats)


class _Forward:
    def __init__(self, owner, example):
        self.owner = weakref.ref(owner)
        self.original = owner.net.forward
        self.has_points = owner.net.config.points
        self.inputs = tuple((tensor.dtype, tuple(tensor.shape[1:])) for tensor in example)
        # Keep parameters and buffers live. Freezing or inference optimization
        # would sever the aliases normal checkpoint refresh depends on.
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            self.traced = torch.jit.trace(_Trunk(owner.net).eval(), example, check_trace=False, strict=True)

    def __call__(self, seats, table, hexes, vertices, vertex_owner, road_owner, reach, *, policy=True):
        tensors = (seats, table, hexes, vertices, vertex_owner, road_owner, reach)
        owner = self.owner()
        if (owner is None or not policy or not owner._ready(tensors)
                or any(t.dtype != dtype or tuple(t.shape[1:]) != shape
                       for t, (dtype, shape) in zip(tensors, self.inputs))):
            return self.original(*tensors, policy=policy)
        out = Prediction(*self.traced(*tensors))
        if not self.has_points:
            out.points = None
        return out


class WorkerInference:
    """Reversible inference owner for one isolated collection worker.

    Construct after the policy and close on worker exit, or use as a context
    manager. FP32, training and unsupported models remain untouched. Eligible
    CPU BF16 calls keep their original batch shapes and decision code. The
    first real call finishes through the original forward; its exact inputs
    also build the trace used on later calls, with no synthetic warmup.

    Normal in-place ``load_state_dict`` refreshes preserve trace aliases.
    Parameter/buffer rebinding, storage or dtype/device changes permanently
    invalidate this owner and retain the canonical eager path. It never
    changes ``policy.precision``. JIT settings belong to this worker's process
    until close, so only one inference owner may be installed in a process.
    """
    def __init__(self, policy):
        global _ACTIVE
        self.policy = policy
        self.net = getattr(policy, "net", None)
        self.eligible, self.valid, self.closed = False, False, False
        self.traced = False
        self._cache = self._forward = self._settings = None
        self._pre_hook = self._load_hook = None
        self._thread = threading.get_ident()
        if not self._supported():
            return
        if _ACTIVE is not None and _ACTIVE() is not None and not _ACTIVE().closed:
            return
        try:
            self._cache = _WeightCache(self.net)
            self._tracked = dict(self.net.named_parameters()) | dict(self.net.named_buffers())
            self._signatures = {key: _signature(value) for key, value in self._tracked.items()}
            self._storages = [value.untyped_storage() for value in self._tracked.values()]
            self._config = self.net.config
            self._structure = self._structure_of()
            self._modules = tuple((module, type(module)) for module in self.net.modules())
            self._pre_hook = self.net.register_forward_pre_hook(self._first_forward, with_kwargs=True)
            self._load_hook = self.net.register_load_state_dict_post_hook(self._after_load)
            self.eligible = self.valid = True
            _ACTIVE = weakref.ref(self)
        except Exception:
            self.close()

    def _supported(self):
        try:
            # Older Torch versions can have the JIT switches but lack these
            # CPU-query signatures. Probe before changing any module state.
            torch.is_autocast_enabled("cpu")
            torch.get_autocast_dtype("cpu")
            return self._supported_model()
        except (AttributeError, TypeError, RuntimeError):
            return False

    def _supported_model(self):
        if (type(self.policy) is not NetworkPolicy9 or type(self.net) is not HexNet
                or self.policy.device.type != "cpu" or self.policy.precision != "bf16"
                or self.net.training or not self.net.fused or self.net.CONTRACT not in ("9", "9d", "9d.1")
                or self.net.players not in (2, 4) or self.net.config.double_turn != "parts"
                or self.net.config.messages != "pooled" or self.net.config.attention or not self.net.config.norm
                or self.net.config.heads != "linear" or self.net.config.discard != "mlp"
                or "forward" in self.net.__dict__
                or not all(callable(getattr(torch._C, name, None)) for name in _JIT_METHODS)):
            return False
        if any(module.training or module._forward_pre_hooks or module._forward_hooks
               or "forward" in module.__dict__ for module in self.net.modules()):
            return False
        return all(parameter.dtype == torch.float32 and parameter.device.type == "cpu"
                   and parameter.layout == torch.strided and not parameter.is_inference()
                   for parameter in self.net.parameters())

    def _structure_of(self):
        return (self.net.fused, self.net.CONTRACT, self.net.players, self.net.num_hexes,
                self.net.num_vertices, self.net.num_edges, tuple(vars(self.net.config).items()))

    def _invalidate(self):
        self.valid = False
        if self._cache is not None:
            self._cache.invalidate()

    def _intact(self):
        if not self.valid or self.closed:
            return False
        current = dict(self.net.named_parameters()) | dict(self.net.named_buffers())
        modules = tuple(self.net.modules())
        own_hook = None if self._pre_hook is None else self._pre_hook.id
        if (self.policy.net is not self.net or self.net.config is not self._config
                or self._structure_of() != self._structure
                or len(modules) != len(self._modules)
                or any(module is not expected or type(module) is not kind
                       or module._forward_hooks
                       or any(key != own_hook or module is not self.net for key in module._forward_pre_hooks)
                       or ("forward" in module.__dict__ and not (module is self.net
                           and module.__dict__["forward"] is self._forward))
                       for module, (expected, kind) in zip(modules, self._modules))
                or current.keys() != self._tracked.keys()
                or any(current[key] is not value or _signature(value) != self._signatures[key]
                       for key, value in self._tracked.items())):
            self._invalidate()
            return False
        return self._cache.refresh()

    def _ready(self, tensors):
        if not self.valid or self.closed:
            return False
        if threading.get_ident() != self._thread:
            self._invalidate()
            return False
        if (self.net.training or self.policy.precision != "bf16" or self.policy.device.type != "cpu"
                or torch.is_grad_enabled() or not torch.is_autocast_enabled("cpu")
                or torch.get_autocast_dtype("cpu") != torch.bfloat16):
            return False
        if (len(tensors) != len(_INPUTS) or any(not isinstance(t, torch.Tensor) or t.device.type != "cpu"
                or t.ndim == 0 for t in tensors) or not tensors[0].shape[0]
                or any(t.shape[0] != tensors[0].shape[0] for t in tensors)):
            return False
        return self._intact()

    def _first_forward(self, module, args, kwargs):
        if kwargs.get("policy", True) is not True:
            return
        values = dict(zip(_INPUTS, args))
        values.update(kwargs)
        if any(key not in values for key in _INPUTS):
            return
        tensors = tuple(values[key] for key in _INPUTS)
        if not self._ready(tensors):
            return
        # Module.__call__ already captured the original bound forward. Remove
        # this hook before tracing, then let that original outer call finish.
        self._pre_hook.remove()
        self._pre_hook = None
        try:
            self._settings = _Settings()
            self._forward = _Forward(self, tensors)
            self.net.forward = self._forward
            self.traced = True
        except Exception:
            self._invalidate()
            if self._settings is not None:
                self._settings.close()

    def _after_load(self, module, incompatible):
        self._intact()

    def close(self):
        global _ACTIVE
        if self.closed:
            return
        self._invalidate()
        for hook in (self._pre_hook, self._load_hook):
            if hook is not None:
                hook.remove()
        if self._forward is not None and self.net.__dict__.get("forward") is self._forward:
            del self.net.forward
        if self._settings is not None:
            self._settings.close()
        if self._cache is not None:
            self._cache.close()
        self._pre_hook = self._load_hook = None
        self._cache = self._forward = self._settings = None
        if hasattr(self, "_tracked"):
            self._tracked.clear()
            self._storages.clear()
        self.closed = True
        if _ACTIVE is not None and _ACTIVE() is self:
            _ACTIVE = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
