# Collection inference

An isolated CPU collection worker owns `hexn.inference9.WorkerInference` for
its contract-9 policy. Eligible two- or four-seat models use contracts 9, 9d, 9d.1 or 11d,
pooled messages without attention, normalization, linear heads, an MLP
discard head and parts decisions. In eval mode with fused collection and the
configured BF16 precision they cache Linear weight casts and trace the original
network forward. A layer whose weight the forward reads outside the layer's
own call (`HexNet.direct_weights`: an 11d net's hand value's first layer)
keeps its FP32 weight, and the trace returns every field the forward fills,
so the traced answers are the original forward's bit for bit. FP32, other precisions, training models and unsupported
Torch interfaces keep the ordinary route. This does not change precision,
decision heads, sampling, log probabilities or the sizes of policy calls.

`WorkerInference` traces the forward on its first input. That call finishes
through the original forward, and later calls use the trace at their
original sizes, including single-row tails. There is no synthetic warmup.
Canonical model parameters and biases remain FP32, and cache buffers are
absent from the state dictionary. Normal in-place `load_state_dict`
refreshes copy weight casts into the same buffers so the trace continues to
see current weights.

Parameter or buffer replacement, storage or dtype/device changes, and
module changes invalidate the owner permanently. Subsequent calls use the
canonical eager model. Gradient-enabled, training and value-only calls use
the original forward. Close the owner when the worker exits; it removes its
hooks, cached buffers and forward override. A process holds one owner per
net, so a worker running several nets (each learner beside the others) gives
each its own. The JIT settings a trace needs are the process's: the first
owner to trace sets them, every traced owner shares them, and the last one
to close restores them.

```python
from hexn.inference9 import WorkerInference

with WorkerInference(policy):
    choices = policy.act(requests)
```

The policy still controls its precision and the request batch. The owner
does not merge calls, retain game state or alter checkpoint formats.

## One server for every net on the box

`hexn.serve.Server` is one process that owns a device and runs every
forward for many worker processes. `Server.start(paths, topology,
clients=n, device="cuda")` loads each `.pt` net once on the device and
returns once they are loaded; `server.connection(k)` is client `k`'s end of a
`multiprocessing` pipe, handed to the worker that plays with it. Each round
the server takes every request waiting on any connection, joins the rows
asked of one net across all of them into one forward, and sends each client
its own rows back.

A worker reads through `hexn.serve.Panel(connection, paths, topology)`.
`panel.score_rows_all(rows, nets=None)` encodes `rows` once per served net
(`score_feed`), sends one message for all of them, scores any exported
`.onnx` file on the worker's own CPU while the server works, and reads each
answer (`score_read`): one `score_rows` answer per net, in order.
`panel.policy(k)` is net `k` alone: its `space`, `players` and
`score_rows`, served the same way, with no `act_rows` or `value_rows`. A
`hexset.gym.seats.PolicySeat` takes `scorer=panel.score_rows_all`, so a seat
reading several nets makes one round trip a tick for all of them.
`panel.stop()` ends the client and returns the server's counters.

A served net answers what the same net loaded in-process on the server's
device answers for the batch the server ran: `Policy9._heads` with the prior
for a contract-9 net, `NetworkPolicy.score` for a contract-8 one. Forwards are
not batch invariant, and another device rounds differently, so a served read
can differ from an in-process CPU read in the last bits.
`precision="bf16"` runs the server's forwards under bf16 autocast, which is
a different policy from fp32, not the same one faster.

The server runs one net's forward at a time, and a contract-9 forward copies
each input to the device and reads each head back separately. On a device
where each copy and synchronisation has a high fixed latency, a forward's
cost barely depends on its rows, so the server pays off only when its
forwards are large; otherwise workers that run their own forwards on their
own cores are faster. `gather=` seconds holds a round open for more clients.
`panel.stop()` returns the server's counters (`rounds`, `requests`,
`forwards`, `rows`, busy `seconds`), which show the rows per forward and the
time per forward.

```python
from hexn.serve import Panel, Server

server = Server.start(paths, topology, clients=1, device="cuda")
panel = Panel(server.connection(0), paths, topology)
reads = panel.score_rows_all(rows)   # one list of (prior, value) per net
panel.stop()
server.stop()
```
