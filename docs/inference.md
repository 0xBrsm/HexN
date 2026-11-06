# Collection inference

An isolated CPU collection worker owns `hexn.inference9.WorkerInference` for
its contract-9 policy. Eligible two- or four-seat models use contracts 9, 9d or 9d.1,
pooled messages without attention, normalization, linear heads, an MLP
discard head and parts decisions. In eval mode with fused collection and the
configured BF16 precision they cache Linear weight casts and trace the original
network forward. FP32, other precisions, training models and unsupported
Torch interfaces keep the ordinary route. This does not change precision,
decision heads, sampling, log probabilities or the sizes of policy calls.

The owner traces the first actual input. That call finishes through the
original forward, and later calls use the trace at their original sizes,
including single-row tails. There is no synthetic warmup. Canonical model
parameters and biases remain FP32, and cache buffers are absent from the
state dictionary. Normal in-place `load_state_dict` refreshes copy weight
casts into the same buffers so the trace continues to see current weights.

Parameter or buffer replacement, storage or dtype/device changes, and
module changes invalidate the owner permanently. Subsequent calls use the
canonical eager model. Gradient-enabled, training and value-only calls use
the original forward. Close the owner when the worker exits; it removes its
hooks, cached buffers and forward override and restores JIT settings. Only
one active owner is supported per process because those settings are local
to the process.

```python
from hexn.inference9 import WorkerInference

with WorkerInference(policy):
    choices = policy.act(requests)
```

The policy still controls its precision and the request batch. The owner
does not merge calls, retain game state or alter checkpoint formats.
