# SPDX-License-Identifier: GPL-3.0-only
"""Export a contract-9 checkpoint to ONNX, for `hexset.clients.onnxbot`.

`python -m hexn.export_onnx` dispatches here for a checkpoint whose saved
`model_config` (or `args`) names contract 9; `export9` is the call.

**Contract 9: the contract-9 observation and the decision's arrays in, the
decision out.** As with contract 8's record, the graph declares what it reads
and answers in HexSet's action contract, and HexSet only supplies the inputs
and reads the answer (`hexset.clients.onnxbot.DecisionPolicy`). What a
contract-9 graph reads are HexSet's own: the position as `hexset.contract9`
encodes it, and the decision at hand as `hexset.decisions.tensors` states it
(`hexset.decisions.arrays`). Everything about this net -- its heads, how each
block of a decision is scored, the points head, how the snake seat's double
turn is chosen -- stays inside the graph, and choosing is in it too: the legal
masks, the argmax over the decision's blocks (a tie to the earlier block, then
the lower index: `hexn.policy9.best`, which `hexn.policy9.Policy9` chooses
with), and `prior`.

* **Inputs**, each with a dynamic leading axis, named as HexSet supplies them:
  `observation` `(B, width)` float32, `contract9.flatten` rows
  (`hexset.contract9.layout(...).width`, 1029 on the standard four-seat
  board); the decision's masks `flat` `(B, space.size)` and `setup` `(B,
  incidences)` bool (robber victims seat-relative, as `tensors` states them);
  its Road Building candidates `road_building` `(Kr, 2)` (edges, `-1` for a
  road not placed) with `road_building_row` `(Kr,)`, and discards `discard`
  `(Kd, 10)` (cards kept, then given) with `discard_row` `(Kd,)`, each beside
  the row it belongs to; and, for a net that chooses the double turn as one
  decision ("joint"), `double_rows` `(D,)`. A "parts" net declares no
  `double_rows`, so HexSet asks it the double turn as two setup turns.
* **Outputs**: `decision` `(B, 2)` int64, each row's `(kind, index)` over its
  decision (`hexset.decisions.DecisionKind`; the index within its block: a
  flat action, an incidence, `first * incidences + second` for a double
  turn, a candidate within the row's own); `value` `(B, players)`, the win
  head in the row's own frame (row 0 the seat the observation is from, the
  others in turn order); `prior` `(B, space.size)`, the softmax over every
  legal option of the decision marginalised onto its first atomic move, as
  a table asks the moves one at a time (`hexn.model9.first_move_prior`: a
  setup turn's mass on its settlement, Road Building's on playing the card,
  a discard's on the kinds of card it gives; a decision of flat actions
  alone reads the softmax of its flat scores), zeros where nothing has a
  first move, so a file can score search leaves, a robber tip and robber
  deals;
  every whole option's probability in its row's decision, one output per
  block (`hexset.decisions.OPTION_OUTPUTS`, `hexn.model9.option_probs`):
  `option_flat` `(B, space.size)`, `option_setup` `(B, incidences)`, for a
  "joint" net `option_double` `(D, incidences, incidences)` (each double-turn
  row's ordered pairs, first by second, in `double_rows` order), and
  `option_road_building` `(Kr,)` and `option_discard` `(Kd,)` beside the
  candidates, the softmax over every legal option of the row, zeros where
  illegal, so a search can be made over whole decisions
  (`hexset.clients.onnxbot.DecisionPolicy.option_rows`);
  and, for a net that has one, `points` `(B, players)`, the points head in
  the same frame, which HexSet does not read. A setup road is scored as the
  road head scores its edge (`hexn.model9.HexNet.flat_scores`), which is how
  a position inside a setup turn, asked a move at a time, is finished.
* **Metadata props**: `contract` `"9"` (`hexset.contract9.CONTRACT`);
  `players` and the `num_hexes`/`num_vertices`/`num_edges` fingerprint,
  required at load; `double_turn` (`joint` or `parts`, a description: HexSet
  reads the graph's inputs, not this); `max_offers`, `"0"` unless asked for
  (a contract-9 net reads no trading inputs and makes no offers of its own;
  its gate still answers the table's, as `hexn.netbot` serves the
  checkpoint); `iteration`; the provenance props `hexn.export_onnx` writes
  (`source_checkpoint`, `checkpoint_sha256`, `exported_at`,
  `exporter_commit`); `search` with `simulations`/`wave` where a search is
  asked for (`hexn.export_onnx.export`), so HexSet searches over `prior`;
  and the serving keys asked for (`serving_props`).
  `python -m hexset.clients.onnxmeta stamp` names a file for the server's
  picker after.

A parity check runs after every export and raises: on seeded positions of
every kind of decision the graph and `hexn.policy9.Policy9` on the torch net
answer the same `decision` for every row, `value` and `prior` within
`rtol=1e-3, atol=1e-4`, the file played through `hexset.clients.onnxbot`
makes the moves the torch policy makes, and lists the same whole options as
`Policy9.option_rows`, their probabilities within the same tolerance.

    python -m hexn.export_onnx --checkpoint latest.pt --out net.onnx
    python -m hexn.export_onnx --checkpoint latest.pt --out served.onnx --runtime <module> \\
        --trader <bot> --trader-veto --trade-floor 0.03 --max-offers 1 --gate-own-frame 1

`--trader` names a bot as a lineup names it, registered by the `--runtime`
module; the server that loads the file needs the same runtime.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hexset import contract9, decisions as D
from hexset.actions import ActionSpace, build_space
from hexset.board.topology import Topology

from .model9 import HexNet, config_of, first_move_prior, option_probs, unpack9

__all__ = ["CONTRACT", "input_names", "output_names", "serving_props", "is_contract9", "export9"]

#: What a contract-9 file stamps as `contract`: HexSet's own constant.
CONTRACT = contract9.CONTRACT

_NEG = float("-inf")


def input_names(joint: bool) -> tuple[str, ...]:
    """The graph's inputs, in order: the observation, then the decision's
    arrays (`hexset.decisions.ARRAYS`), `double_rows` for a "joint" net
    alone."""
    return ("observation",) + tuple(n for n in D.ARRAYS if joint or n != "double_rows")


def output_names(points: bool, joint: bool = False) -> tuple[str, ...]:
    """The graph's outputs, in order (the module docstring): `option_double`
    for a "joint" net alone, `points` for a net that has one."""
    options = tuple(n for n in D.OPTION_OUTPUTS if joint or n != "option_double")
    return ("decision", "value", "prior") + options + (("points",) if points else ())


def _first_max(scores: Tensor, row: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
    """Per batch row, the best of the candidate `scores` whose `row` it is and
    that candidate's index among the row's own (the first of equal bests):
    `-inf` and 0 for a row with none. The candidates of a row are contiguous,
    in the order the decision lists them."""
    count = scores.shape[0]
    position = torch.arange(count, device=scores.device)
    batch = rows.shape[0]
    top = torch.full((batch,), _NEG, device=scores.device).scatter_reduce(
        0, row, scores, reduce="amax", include_self=True)
    past = count + 1
    start = torch.full((batch,), past, dtype=torch.long, device=scores.device).scatter_reduce(
        0, row, position, reduce="amin", include_self=True)
    at_top = torch.where(scores >= top[row], position, torch.full_like(position, past))
    first = torch.full((batch,), past, dtype=torch.long, device=scores.device).scatter_reduce(
        0, row, at_top, reduce="amin", include_self=True)
    return top, torch.where(first < past, first - start, torch.zeros_like(first))


class _Export9(nn.Module):
    """`observation, decision arrays -> decision, value, prior[, points]`:
    `hexn.model9.HexNet`, the scores `hexn.policy9.Policy9` reads, the legal
    masks and `hexn.policy9.best`'s argmax, traceable."""

    def __init__(self, net: HexNet, layout: contract9.Layout9, topology: Topology) -> None:
        super().__init__()
        self.net = net
        self.layout = layout
        self.joint = net.config.double_turn == "joint"
        self.points = net.points_head is not None
        if self.joint:
            self.register_buffer("pairs_ok", torch.from_numpy(D.double_pairs(topology)), persistent=False)

    def forward(self, observation: Tensor, flat: Tensor, setup: Tensor, road_building: Tensor,
                road_building_row: Tensor, discard: Tensor, discard_row: Tensor,
                double_rows: Tensor | None = None) -> tuple[Tensor, ...]:
        net = self.net
        out = net(**unpack9(self.layout, observation), policy=True)
        rows = torch.arange(observation.shape[0], device=observation.device)
        neg = torch.full_like(rows, 0, dtype=torch.float32) + _NEG
        # Flat: every flat action's score where the row's mask allows it.
        flat_scores = net.flat_scores(out).float().masked_fill(~flat, _NEG)
        flat_at = flat_scores.argmax(1)
        flat_top = flat_scores.gather(1, flat_at.unsqueeze(1)).squeeze(1)
        # A setup turn: the single one, on a row that is not a double turn.
        double = torch.zeros_like(rows)
        if self.joint:
            double = double.index_put((double_rows,), torch.ones_like(double_rows))
        single = setup & (double == 0).unsqueeze(1)
        setup_scores = net.setup_scores(out, rows).float().masked_fill(~single, _NEG)
        setup_at = setup_scores.argmax(1)
        setup_top = setup_scores.gather(1, setup_at.unsqueeze(1)).squeeze(1)
        # The double turn's ordered pairs, on its rows.
        double_top, double_at, pairs = neg, torch.zeros_like(rows), None
        if self.joint:
            legal = setup[double_rows]
            ok = legal.unsqueeze(2) & legal.unsqueeze(1) & self.pairs_ok.unsqueeze(0)
            pairs = net.double_scores(out, double_rows).float().masked_fill(~ok, _NEG)
            # A width, not -1: no double-turn row in the batch is the common case, and -1 cannot be read off 0 rows.
            flat_pairs = pairs.reshape(double_rows.shape[0], self.pairs_ok.shape[0] * self.pairs_ok.shape[1])
            pair_at = flat_pairs.argmax(1)
            double_top = neg.index_put((double_rows,), flat_pairs.gather(1, pair_at.unsqueeze(1)).squeeze(1))
            double_at = double_at.index_put((double_rows,), pair_at)
        # The candidates, each row's best among its own.
        road_scores = net.road_building_scores(out, road_building_row, road_building).float()
        discard_scores = net.discard_scores(out, discard_row, discard).float()
        road_top, road_at = _first_max(road_scores, road_building_row, rows)
        discard_top, discard_at = _first_max(discard_scores, discard_row, rows)
        # The blocks in `DecisionKind` order; `argmax` takes the first best, so a tie goes to the earlier block.
        tops = torch.stack([flat_top, setup_top, double_top, road_top, discard_top], 1)
        kind = tops.argmax(1)
        index = torch.stack([flat_at, setup_at, double_at, road_at, discard_at], 1).gather(1, kind.unsqueeze(1))
        decision = torch.cat([kind.unsqueeze(1), index], 1)
        prior = first_move_prior(net, flat_scores, setup_scores, road_scores, road_building_row, discard_scores,
                                 discard_row, discard, pairs, double_rows)
        # Every whole option's probability, block by block (`hexset.decisions.OPTION_OUTPUTS` order).
        whole = option_probs(flat_scores, setup_scores, road_scores, road_building_row, discard_scores, discard_row,
                             pairs, double_rows)
        answers = [decision, out.value.float(), prior, whole[0], whole[1]]
        if self.joint:
            answers.append(whole[4])
        answers += [whole[2], whole[3]]
        if self.points:
            answers.append(out.points.float())
        return tuple(answers)


def _load(checkpoint: str, topology: Topology) -> tuple[HexNet, ActionSpace, dict]:
    """The net as `hexn.netbot` rebuilds a contract-9 checkpoint, and what the
    file says about itself."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, players)
    net = HexNet(space, topology, players, config_of(state["model_config"]))
    net.load_state_dict(state["net"])
    net.eval()
    # The node-major trunk, as every contract-9 reader runs it: the same
    # numbers as the reference loop up to float reassociation.
    net.fused = True
    return net, space, {"players": players, "iteration": int(state.get("iteration") or 0)}


def serving_props(
    *,
    max_offers: int | None = None,
    trader: str | None = None,
    trader_veto: bool = False,
    trade_floor: float | None = None,
    gate_own_frame: bool | None = None,
    gate_plies: int | None = None,
    robber_tip: float | None = None,
) -> dict[str, str]:
    """The serving keys `hexset.clients._modelmeta` reads, for what is asked:
    nothing for an argument left at its default. A `trader_veto` needs a
    `trader` to veto and a seat that trades (`max_offers` not 0), as
    `hexset.arena.traded` does."""
    if trader_veto and (trader is None or not max_offers):
        raise ValueError("a trader veto needs a trader and max_offers above 0")
    if trade_floor is not None and not 0.0 <= trade_floor <= 1.0:
        raise ValueError(f"trade_floor is a win-probability gain in [0, 1], not {trade_floor}")
    if robber_tip is not None and not 0.0 <= robber_tip <= 1.0:
        raise ValueError(f"robber_tip is a fraction in [0, 1], not {robber_tip}")
    props: dict[str, str] = {}
    if max_offers is not None:
        props["max_offers"] = str(int(max_offers))
    if trader is not None:
        props["trader"] = trader
    if trader_veto:
        props["trader_veto"] = "1"
    if trade_floor is not None:
        props["trade_floor"] = repr(float(trade_floor))
    if gate_own_frame is not None:
        props["gate_own_frame"] = "1" if gate_own_frame else "0"
    if gate_plies is not None:
        props["gate_plies"] = str(int(gate_plies))
    if robber_tip is not None:
        props["robber_tip"] = repr(float(robber_tip))
    return props


def _sample_rows(players: int, count: int, seed: int) -> list:
    """`(game, seat)` pairs from seeded random playouts on the base board,
    every kind of decision among them: playouts stopped in the setup (the
    snake seat's double turn, a setup road owed), at a discard and in
    ordinary turns, some with a Road Building card handed to the mover so
    its road sets are asked."""
    import random

    from hexset.board.board import random_base_board
    from hexset.cards import DevCard
    from hexset.game import Phase, is_over, start, to_move
    from hexset.play import step_randomly

    rng = random.Random(seed)
    out = []
    while len(out) < count:
        game = start(random_base_board(rng), players, rng)
        for _ in range(rng.choice((0, 3, 6, 7, 12, 60, 150))):
            if is_over(game) or (game.phase is Phase.DISCARD and rng.random() < 0.5):
                break
            step_randomly(game, rng)
        if is_over(game):
            continue
        if game.phase is Phase.MAIN and not game.dev_card_played and rng.random() < 0.5:
            state = game.state(game.current_player, hidden=False)
            state.dev_cards[game.current_player][DevCard.ROAD_BUILDING] += 1
            game.set_state(state)
        out.append((game, to_move(game)))
    return out


def _feed(rows, space: ActionSpace, topology: Topology, joint: bool, policy) -> dict[str, np.ndarray]:
    """The inputs for `rows`, each asked its decision as HexSet would ask it."""
    decisions = [policy.decision(game, seat) for game, seat in rows]
    arrays = D.arrays(decisions, space, topology)
    feed = {"observation": np.stack([contract9.flatten(contract9.encode9(g, s, policy.net.CONTRACT)) for g, s in rows])}
    feed.update((name, arrays[name]) for name in input_names(joint) if name != "observation")
    return feed


def _dummy(net: HexNet, layout: contract9.Layout9, space: ActionSpace, topology: Topology,
           joint: bool) -> tuple[Tensor, ...]:
    """Trace inputs: real positions, their decisions as HexSet states them,
    with a Road Building candidate, a discard and (for a joint net) a double
    turn among the rows, so no axis is traced at a size that could be
    mistaken for fixed."""
    from .policy9 import Policy9

    policy = Policy9(net, space)
    rows = _sample_rows(layout.players, 3, seed=0)
    feed = _feed(rows, space, topology, joint, policy)
    feed["road_building"] = np.array([[0, 1], [2, -1]], dtype=np.int64)
    feed["road_building_row"] = np.array([0, 1], dtype=np.int64)
    feed["discard"] = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 2, 1, 0, 1, 0, 0, 1, 1, 1]], dtype=np.int64)
    feed["discard_row"] = np.array([1, 2], dtype=np.int64)
    if joint:
        feed["double_rows"] = np.array([0], dtype=np.int64)
    return tuple(torch.from_numpy(feed[name]) for name in input_names(joint))


def _shapes(topology: Topology, layout: contract9.Layout9, space: ActionSpace) -> dict[str, tuple]:
    """Every tensor's shape, axis names for the dynamic ones: the contract
    (the module docstring)."""
    incidences = len(contract9.incidences(topology))
    return {
        "observation": ("batch", layout.width), "flat": ("batch", space.size), "setup": ("batch", incidences),
        "road_building": ("road_building", 2), "road_building_row": ("road_building",),
        "discard": ("discards", 10), "discard_row": ("discards",), "double_rows": ("doubles",),
        "decision": ("batch", 2), "value": ("batch", layout.players), "prior": ("batch", space.size),
        "option_flat": ("batch", space.size), "option_setup": ("batch", incidences),
        "option_double": ("doubles", incidences, incidences), "option_road_building": ("road_building",),
        "option_discard": ("discards",), "points": ("batch", layout.players),
    }


_TYPES = {"observation": "FLOAT", "flat": "BOOL", "setup": "BOOL", "decision": "INT64", "value": "FLOAT",
          "prior": "FLOAT", "points": "FLOAT", **{name: "FLOAT" for name in D.OPTION_OUTPUTS}}


def _embed(path: Path, props: dict[str, str], shapes: dict[str, tuple]) -> None:
    """`props` as the file's metadata, and every output's shape as `shapes`
    states it: the trace leaves some widths it computes at run time unnamed,
    which the contract fixes."""
    import onnx

    model = onnx.load(str(path))
    for tensor in model.graph.output:
        dims = tensor.type.tensor_type.shape.dim
        for dim, want in zip(dims, shapes[tensor.name]):
            if isinstance(want, int):
                dim.Clear()
                dim.dim_value = want
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _verify_contract(path: Path, shapes: dict[str, tuple], joint: bool, points: bool) -> None:
    """The graph read back, held to the names, dtypes and shapes HexSet's
    `DecisionPolicy` feeds and reads."""
    import onnx

    model = onnx.load(str(path))
    found = ([t.name for t in model.graph.input], [t.name for t in model.graph.output])
    expected = (list(input_names(joint)), list(output_names(points, joint)))
    if found != expected:
        raise ValueError(f"graph signature {found} is not hexset's {expected}")
    for tensor in [*model.graph.input, *model.graph.output]:
        kind = tensor.type.tensor_type
        want = getattr(onnx.TensorProto, _TYPES.get(tensor.name, "INT64"))
        if kind.elem_type != want:
            raise ValueError(f"{tensor.name} is not {onnx.TensorProto.DataType.Name(want)}")
        dims = tuple(d.dim_param or d.dim_value for d in kind.shape.dim)
        if dims != shapes[tensor.name]:
            raise ValueError(f"{tensor.name} is shaped {dims}, not {shapes[tensor.name]}")


def _verify_parity(path: Path, net: HexNet, space: ActionSpace, topology: Topology, *, samples: int = 64,
                   seed: int = 1) -> dict[str, float]:
    """The graph against `hexn.policy9.Policy9` on `net`, on seeded positions
    of every kind of decision (`_sample_rows`), each asked its decision as
    HexSet asks it: the same `decision` on every row, `value` and
    `prior` within `rtol=1e-3, atol=1e-4`; then the file through
    `hexset.clients.onnxbot` makes the move the torch policy makes at each,
    and lists each row's whole options as `Policy9.option_rows` does, their
    probabilities within the same tolerance. Raises on a difference; returns
    the largest absolute difference of `value`, `prior` and the whole
    options' probabilities (`options`)."""
    import onnxruntime as ort

    from hexset.clients.onnxbot import load

    from .policy9 import Policy9

    torch_policy = Policy9(net, space)
    joint = net.config.double_turn == "joint"
    rows = _sample_rows(space.num_players, samples, seed)
    feed = _feed(rows, space, topology, joint, torch_policy)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = output_names(net.points_head is not None, joint)
    got = dict(zip(names, session.run(list(names), feed)))
    want = torch_policy.answers(feed["observation"], {k: v for k, v in feed.items() if k != "observation"}
                                | ({} if joint else {"double_rows": np.zeros(0, dtype=np.int64)}), topology)
    np.testing.assert_array_equal(got["decision"], want["decision"], err_msg="decision")
    worst = {}
    for name in ("value", "prior", "points"):
        if name in got:
            np.testing.assert_allclose(got[name], want[name], rtol=1e-3, atol=1e-4, err_msg=name)
            worst[name] = float(np.abs(got[name] - want[name]).max())
    served = load(str(path), topology, threads=None).policy
    for game, seat in rows:
        mine, theirs = served._choose(game, seat), torch_policy._choose(game, seat)
        if mine != theirs:
            raise ValueError(f"parity: the file chose {mine}, torch {theirs}")
    worst["options"] = 0.0
    for ((listed, probs), (want_listed, want_probs)) in zip(served.option_rows(rows), torch_policy.option_rows(rows)):
        if listed != want_listed:
            raise ValueError("parity: the file lists other whole options than torch")
        np.testing.assert_allclose(probs, want_probs, rtol=1e-3, atol=1e-4, err_msg="options")
        worst["options"] = max(worst["options"], float(np.abs(probs - want_probs).max(initial=0.0)))
    return worst


def export9(
    checkpoint: str,
    out: Path,
    *,
    topology: Topology | None = None,
    opset: int = 18,
    serving: dict[str, str] | None = None,
    verify: bool = True,
) -> Path:
    """Load the contract-9 `checkpoint`, trace it to `out`, stamp it, verify
    its contract and parity, return `out`. `serving` is the serving keys to
    write (`serving_props`); `max_offers` defaults to `"0"`. The file is
    written beside `out` and renamed into place once verified."""
    from .export_onnx import _base_topology, _provenance

    topology = topology or _base_topology()
    net, space, meta = _load(checkpoint, topology)
    players = meta["players"]
    layout = contract9.layout(players, topology.num_hexes, topology.num_vertices, topology.num_edges, net.CONTRACT)
    joint = net.config.double_turn == "joint"
    points = net.points_head is not None
    serving = dict(serving or {})
    if "trader" in serving:
        from hexset.arena import entrant_from_name

        entrant_from_name(serving["trader"])  # refuse a name HexSet cannot seat, here
    wrapper = _Export9(net, layout, topology).eval()
    names_in, names_out = input_names(joint), output_names(points, joint)
    shapes = _shapes(topology, layout, space)
    out = Path(out)
    handle, temporary = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".onnx")
    os.close(handle)
    temporary = Path(temporary)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, _dummy(net, layout, space, topology, joint), str(temporary), dynamo=False,
                input_names=list(names_in), output_names=list(names_out),
                dynamic_axes={name: {0: shapes[name][0]} for name in (*names_in, *names_out)},
                opset_version=opset,
            )
        props = {
            "contract": net.CONTRACT,
            "players": str(players),
            "num_hexes": str(topology.num_hexes),
            "num_vertices": str(topology.num_vertices),
            "num_edges": str(topology.num_edges),
            "double_turn": net.config.double_turn,
            "max_offers": "0",
            "iteration": str(meta["iteration"]),
            **_provenance(checkpoint),
            **serving,
        }
        _embed(temporary, props, shapes)
        _verify_contract(temporary, shapes, joint, points)
        if verify:
            _verify_parity(temporary, net, space, topology)
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return out


def is_contract9(checkpoint: str) -> bool:
    """Whether the checkpoint at `checkpoint` is a contract-9 net, by its
    saved `model_config` or `args`."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("model_config") or {}
    return str(config.get("contract", state.get("args", {}).get("contract", ""))) in contract9.CONTRACTS
