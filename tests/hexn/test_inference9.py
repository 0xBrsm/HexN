"""Original policy outputs, refresh and reversible worker inference ownership."""
import copy
import dataclasses
import pickle
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hexset import contract9, decisions as D
from hexset.actions import space_for
from hexset.board.board import base_board
from hexset.cards import DevCard
from hexset.chance import DiceDeck, Recording, Scripted
from hexset.game import Game, Phase
from hexset.ledger import PublicLedger
from hexset.rules import DUEL_VARIANT, STANDARD
from hexset.state import Building, GameState
from hexn import policy9
from hexn.inference9 import WorkerInference
from hexn.model9 import HexNet, ModelConfig, unpack9
from hexn.policy9 import NetworkPolicy9
from hexn.selfplay import Request, draw_key


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def position(players, phase):
    board = base_board()
    topology = board.topology
    owners = [-1] * topology.num_vertices
    buildings = [Building.NONE] * topology.num_vertices
    for seat, vertex in enumerate(topology.hex_vertices[0][::2][:players]):
        owners[vertex], buildings[vertex] = seat, Building.SETTLEMENT
    hands = [[3, 2, 2, 2, 3] for _ in range(players)]
    rules = DUEL_VARIANT if players == 2 else STANDARD
    state = GameState(board=board, num_players=players, vertex_owner=owners,
        vertex_building=buildings, edge_owner=[-1] * topology.num_edges, robber=-1,
        hands=hands, bank=[10] * 5, deck=[DevCard.KNIGHT, DevCard.VICTORY_POINT],
        dev_cards=[[1] * 5 for _ in range(players)], new_dev_cards=[[0] * 5 for _ in range(players)],
        knights_played=[0] * players, dev_cards_played=[0] * 5, dev_ages=[[3] * 5 for _ in range(players)],
        road_lengths=[0] * players, longest_road_holder=-1, largest_army_holder=-1,
        rules=rules, dice=DiceDeck() if rules.balanced_dice else None)
    ledger = PublicLedger.new(players)
    for seat, hand in enumerate(hands):
        for resource, count in enumerate(hand):
            ledger.receive(seat, resource, count)
    return Game(state, random.Random(0), Recording(Scripted(())), ledger, phase=phase,
                current_player=0, setup_queue=list(range(players))+list(reversed(range(players))),
                discard_quota=[2] * players if phase is Phase.DISCARD else [0] * players,
                gates=(None,) * players)


def built(players=2, *, points=True, precision="bf16", others=None):
    contract = "9d" if players == 2 else "9"
    game = position(players, Phase.MAIN)
    space = space_for(game)
    config = ModelConfig(width=16, rounds=2, points=points, double_turn="parts", contract=contract,
                         others=players == 2 if others is None else others, attention=False, messages="pooled")
    torch.manual_seed(0)
    reference = HexNet(space, game._state.board.topology, players, config).eval()
    policies = [NetworkPolicy9(net, space) for net in (reference, copy.deepcopy(reference))]
    for policy in policies:
        policy.precision = precision
        policy._pairs = contract9.incidences(game._state.board.topology)
    requests = []
    for index, phase in enumerate((Phase.SETUP_SETTLEMENT, Phase.ROLL, Phase.MAIN, Phase.ROBBER, Phase.DISCARD)):
        game = position(players, phase)
        seat = 1 if phase is Phase.DISCARD else 0
        decision = D.tensors(game, seat, space, double_turn=False)
        observation = contract9.flatten(contract9.encode9(game, seat, contract))
        requests.append(Request(index, seat, observation, decision.flat, (), game=game, decision=decision,
                                draw=draw_key(0, index, seat, 0)))
    return (*policies, requests)


def bits(value):
    if value is None:
        return None
    # Own the flattened storage: empty indexed tensors can have stride zero.
    flat = torch.empty(value.numel(), dtype=value.dtype)
    flat.copy_(value.detach().cpu().reshape(-1))
    return str(value.dtype), tuple(value.shape), flat.view(torch.uint8).numpy().tobytes()


def observe(monkeypatch, policy, requests):
    score = policy9.decision_scores
    calls = []
    def capture(net, prediction, batch):
        scores = score(net, prediction, batch)
        calls.append((tuple((field.name, bits(getattr(prediction, field.name))) for field in dataclasses.fields(prediction)),
                      tuple((field.name, bits(getattr(scores, field.name))) for field in dataclasses.fields(scores))))
        return scores
    with monkeypatch.context() as patch:
        patch.setattr(policy9, "decision_scores", capture)
        choices = policy.act(requests)
    assert len(calls) == 1
    return calls[0], choices, pickle.dumps(choices, protocol=4)


def jit_settings():
    c = torch._C
    autocast = c._jit_set_autocast_mode(False)
    c._jit_set_autocast_mode(autocast)
    return (c._jit_can_fuse_on_cpu(), c._jit_texpr_fuser_enabled(), c._jit_llga_enabled(),
            c._get_graph_executor_optimize(), autocast)


@pytest.mark.parametrize("players", (2, 4))
@pytest.mark.parametrize("points", (False, True))
@pytest.mark.parametrize("others", (False, True))
def test_original_complete_outputs_typed_choices_and_actual_first_group_and_tails(monkeypatch, players, points, others):
    reference, candidate, rows = built(players, points=points, others=others)
    trace = torch.jit.trace
    observed, forwards = [], []
    forward = HexNet.forward
    def original_call(net, *args, **kwargs):
        if net is candidate.net:
            forwards.append(torch.jit.is_tracing())
        return forward(net, *args, **kwargs)
    monkeypatch.setattr(HexNet, "forward", original_call)
    def traced(module, inputs, **kwargs):
        observed.append(tuple(t.shape[0] for t in inputs))
        return trace(module, inputs, **kwargs)
    monkeypatch.setattr(torch.jit, "trace", traced)
    original_act = candidate.act.__func__
    with WorkerInference(candidate) as owner:
        assert owner.eligible and not owner.traced
        assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        assert owner.traced and owner.valid and observed == [(5,) * 7]
        assert forwards[-1] is False and True in forwards
        for count in (1, 3, 2, 5):
            assert observe(monkeypatch, reference, rows[:count]) == observe(monkeypatch, candidate, rows[:count])
        assert candidate.act.__func__ is original_act is NetworkPolicy9.act
        assert candidate.precision == "bf16"
        assert len(observed) == 1
    assert "forward" not in candidate.net.__dict__


def test_normal_bias_and_all_parameter_refresh_preserve_state_keys_fp32_and_cast_aliases(monkeypatch):
    reference, candidate, rows = built()
    keys = tuple(candidate.net.state_dict())
    parameters = dict(candidate.net.named_parameters())
    settings = jit_settings()
    with WorkerInference(candidate) as owner:
        before = observe(monkeypatch, candidate, rows)
        buffers = tuple(entry.buffer for entry in owner._cache.entries)
        for mode in ("bias", "all"):
            state = {key:value.clone() for key, value in reference.net.state_dict().items()}
            for index, (name, value) in enumerate(state.items()):
                if value.is_floating_point() and (mode == "all" or name == "value_head.bias"):
                    value.mul_(0.875).add_((index+1) * 0.00007)
            reference.net.load_state_dict(state)
            candidate.net.load_state_dict(state)
            assert owner.valid and owner.traced
            assert all(entry.buffer is buffer for entry, buffer in zip(owner._cache.entries, buffers))
            after = observe(monkeypatch, candidate, rows)
            assert after == observe(monkeypatch, reference, rows)
            assert after[0][0] != before[0][0]
            before = after
        assert tuple(candidate.net.state_dict()) == keys
        assert all(value is parameters[name] and value.dtype == torch.float32
                   for name, value in candidate.net.named_parameters())
    assert tuple(candidate.net.state_dict()) == keys
    assert jit_settings() == settings
    assert not candidate.net._forward_pre_hooks and not candidate.net._load_state_dict_post_hooks
    assert all(type(module) is torch.nn.Linear for module in candidate.net.modules() if isinstance(module, torch.nn.Linear))


@pytest.mark.parametrize("change", ("linear-rebind", "linear-storage", "norm-storage", "buffer", "assign"))
def test_replacement_and_storage_changes_reject_stale_trace_and_use_current_eager_model(monkeypatch, change):
    reference, candidate, rows = built()
    with WorkerInference(candidate) as owner:
        assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        if change == "assign":
            state = {key:value.clone() for key, value in reference.net.state_dict().items()}
            for value in state.values():
                if value.is_floating_point():
                    value.mul_(0.75)
            reference.net.load_state_dict({key:value.clone() for key, value in state.items()}, assign=True)
            candidate.net.load_state_dict({key:value.clone() for key, value in state.items()}, assign=True)
        elif change == "buffer":
            for policy in (reference, candidate):
                policy.net.inc_vertex = policy.net.inc_vertex.clone()
        else:
            for policy in (reference, candidate):
                module = policy.net.out_seat if change == "norm-storage" else policy.net.embed_seat
                value = module._parameters["weight"].detach().clone().mul_(0.75)
                if change == "linear-rebind":
                    module.weight = torch.nn.Parameter(value)
                else:
                    module._parameters["weight"].data = value
        assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        assert not owner.valid
    assert not any(name.endswith("_inference9_weight") for name, _ in candidate.net.named_buffers())


def test_grad_training_value_only_and_fp32_calls_keep_original_math_and_gradients(monkeypatch):
    reference, candidate, rows = built(4)
    tensors = unpack9(candidate.layout, torch.from_numpy(np.stack([row.observation for row in rows])))
    with WorkerInference(candidate) as owner:
        assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        for training in (False, True):
            for policy in (reference, candidate):
                policy.net.train(training)
                policy.net.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = [policy.net(**tensors) for policy in (reference, candidate)]
                for out in outputs:
                    (out.value_logits.float().square().sum()+out.logits.float().square().sum()).backward()
            assert tuple(bits(getattr(outputs[0], f.name)) for f in dataclasses.fields(outputs[0])) == tuple(
                bits(getattr(outputs[1], f.name)) for f in dataclasses.fields(outputs[1]))
            assert {name:bits(value.grad) for name, value in reference.net.named_parameters()} == {
                name:bits(value.grad) for name, value in candidate.net.named_parameters()}
        for policy in (reference, candidate):
            policy.net.eval()
        for bf16 in (False, True):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                outputs = [policy.net(**tensors, policy=False) for policy in (reference, candidate)]
            assert tuple(bits(getattr(outputs[0], f.name)) for f in dataclasses.fields(outputs[0])) == tuple(
                bits(getattr(outputs[1], f.name)) for f in dataclasses.fields(outputs[1]))
        assert owner.valid


@pytest.mark.parametrize("mode", ("fp32", "fp16", "train", "unfused", "unsupported-jit", "attention", "separate", "mlp"))
def test_ineligible_owner_leaves_policy_module_tree_and_precision_untouched(monkeypatch, mode):
    _, policy, rows = built(precision=mode if mode in ("fp32", "fp16") else "bf16")
    if mode == "train":
        policy.net.train()
    if mode == "unfused":
        policy.net.fused = False
    if mode == "unsupported-jit":
        monkeypatch.setattr(torch._C, "_jit_llga_enabled", None)
    if mode in ("attention", "separate", "mlp"):
        policy.net.config = dataclasses.replace(policy.net.config, **{
            "attention": {"attention": True}, "separate": {"messages": "separate"}, "mlp": {"heads": "mlp"}}[mode])
    precision = policy.precision
    keys = tuple(policy.net.state_dict())
    with WorkerInference(policy) as owner:
        assert not owner.eligible and not owner.traced
        assert tuple(policy.net.state_dict()) == keys and policy.precision == precision
        assert "forward" not in policy.net.__dict__
        assert not policy.net._forward_pre_hooks and not policy.net._load_state_dict_post_hooks
        assert all(type(module) is torch.nn.Linear for module in policy.net.modules() if isinstance(module, torch.nn.Linear))


def test_trace_failure_is_eager_and_close_preserves_a_foreign_forward(monkeypatch):
    reference, candidate, rows = built()
    def unavailable(*args, **kwargs):
        raise RuntimeError("tracing unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(torch.jit, "trace", unavailable)
        with WorkerInference(candidate) as owner:
            assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
            assert not owner.valid and not owner.traced
            assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
    original = candidate.net.forward
    with WorkerInference(candidate) as owner:
        observe(monkeypatch, candidate, rows)
        def replacement(*args, **kwargs):
            return original(*args, **kwargs)
        candidate.net.forward = replacement
    assert candidate.net.forward is replacement
    owner.close()
    assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)


@pytest.mark.parametrize("mode", ("dtype", "device"))
def test_dtype_and_device_changes_never_read_the_old_trace(monkeypatch, mode):
    reference, candidate, rows = built()
    with WorkerInference(candidate) as owner:
        assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        def stale(*args):
            raise AssertionError("stale trace must not run")
        owner._forward.traced = stale
        if mode == "dtype":
            for policy in (reference, candidate):
                module = policy.net.embed_seat
                module._parameters["weight"].data = module._parameters["weight"].detach().half()
            assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
        else:
            for policy in (reference, candidate):
                policy.net.to("meta")
            with pytest.raises(RuntimeError):
                observe(monkeypatch, reference, rows)
            with pytest.raises(RuntimeError):
                observe(monkeypatch, candidate, rows)
        assert not owner.valid
    assert not any(name.endswith("_inference9_weight") for name, _ in candidate.net.named_buffers())


@pytest.mark.parametrize("api", ("cpu-enabled-signature", "cpu-dtype-missing", "cpu-query-runtime", "tensor-query-missing"))
def test_older_torch_query_apis_are_ineligible_before_any_model_mutation(monkeypatch, api):
    reference, candidate, rows = built()
    parameters = dict(candidate.net.named_parameters())
    buffers = dict(candidate.net.named_buffers())
    def unavailable(*args):
        raise RuntimeError("CPU query unavailable")
    with monkeypatch.context() as patch:
        if api == "cpu-enabled-signature":
            patch.setattr(torch, "is_autocast_enabled", lambda: False)
        elif api == "cpu-dtype-missing":
            patch.delattr(torch, "get_autocast_dtype")
        elif api == "cpu-query-runtime":
            patch.setattr(torch, "get_autocast_dtype", unavailable)
        else:
            patch.setattr(torch.Tensor, "is_inference", None)
        with WorkerInference(candidate) as owner:
            assert not owner.eligible
            assert "forward" not in candidate.net.__dict__
            assert not candidate.net._forward_pre_hooks and not candidate.net._load_state_dict_post_hooks
            assert dict(candidate.net.named_parameters()).keys() == parameters.keys()
            assert all(value is parameters[name] for name, value in candidate.net.named_parameters())
            assert dict(candidate.net.named_buffers()).keys() == buffers.keys()
            assert all(value is buffers[name] for name, value in candidate.net.named_buffers())
            assert all(type(module) is torch.nn.Linear for module in candidate.net.modules() if isinstance(module, torch.nn.Linear))
    assert observe(monkeypatch, reference, rows) == observe(monkeypatch, candidate, rows)
