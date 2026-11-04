"""Native worker selection, complete bytes, refill and listed resume."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from hexset.gym.native import available
from hexn.collect import WorkerSpec, _Keep, _build, _serve
from hexn.durable import Partial
from hexn.inference9 import WorkerInference
from hexn.native_collect import NativeWorkerCollector, supported


def specification(players=4, precision="fp32"):
    return WorkerSpec(seed=0, players=players, lanes=4, action_cap=24, max_offers=0,
        first_game=0, stride=1, width=8, rounds=1, torch_seed=0, contract="9d" if players == 2 else "9",
        precision=precision, sample_stream="game", discard_order="random",
        net9=(("heads", "linear"), ("messages", "pooled"), ("attention", False),
              ("others", players == 2), ("norm", True), ("double_turn", "parts")))


def prepare(collector, directory, plan):
    partial = Partial(directory)
    partial.write_plan(plan)
    collector.sink = _Keep(partial)
    return partial


def game_bytes(directory):
    return {int(path.stem): path.read_bytes() for path in directory.glob("*.game")}


def test_native_selection_preserves_unsupported_worker_modes(monkeypatch):
    import hexn.native_collect as module

    _, collector = _build(specification())
    monkeypatch.setattr(module, "available", lambda: False)
    assert not supported(collector)
    monkeypatch.setattr(module, "available", lambda: True)
    assert supported(collector)
    for name, value in (("sample_stream", "worker"), ("max_offers", 1), ("marks", True),
                        ("outlook", True), ("trader", "a trader"), ("caster", lambda i: (0,) * 4)):
        old = getattr(collector, name)
        setattr(collector, name, value)
        assert not supported(collector), name
        setattr(collector, name, old)


def test_native_selection_requires_the_complete_batch_extension(monkeypatch):
    import hexset.gym.native as gym

    _, collector = _build(specification())
    monkeypatch.setattr(gym, "_state_available", lambda: True)
    monkeypatch.setattr(gym, "_NativeRequests", None)
    assert not supported(collector)


@pytest.mark.skipif(not available(), reason="optional native extension")
@pytest.mark.parametrize("players", [2, 4])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_complete_cohorts_refill_refresh_and_keep_exact_files(tmp_path, players, precision):
    spec = replace(specification(players, precision), first_game=5, stride=3)
    original_policies, original = _build(spec)
    native_policies, factory = _build(spec)
    native = NativeWorkerCollector(factory)
    inference = WorkerInference(native_policies[0])
    try:
        compare_cohorts(tmp_path, original_policies, original, native_policies, native)
    finally:
        inference.close()


def compare_cohorts(tmp_path, original_policies, original, native_policies, native):
    for iteration, count in enumerate((5, 3)):
        if iteration:
            state = {name: value.clone() for name, value in original_policies[0].net.state_dict().items()}
            for value in state.values():
                if value.is_floating_point():
                    value.add_(0.03125)
            for policy in (*original_policies, *native_policies):
                policy.net.load_state_dict(state)
        plan = original.upcoming(count)
        assert native.upcoming(count) == plan
        a, b = tmp_path/f"original-{iteration}", tmp_path/f"native-{iteration}"
        prepare(original, a, plan)
        prepare(native, b, plan)
        original.cohort(count)
        native.cohort(count)
        assert game_bytes(a) == game_bytes(b)
        assert set(game_bytes(a)) == set(plan)
        assert native.games_started() == original.games_started()
        assert not native.running and not original.running


@pytest.mark.skipif(not available(), reason="optional native extension")
@pytest.mark.parametrize("players", [2, 4])
def test_native_listed_resume_keeps_prefix_and_original_group_shapes(tmp_path, players):
    spec = replace(specification(players, "bf16"), first_game=5, stride=3)
    _, prefix = _build(replace(spec, lanes=1))
    prefix_dir = tmp_path/"prefix"
    prepare(prefix, prefix_dir, [5])
    prefix.cohort(1)
    first = game_bytes(prefix_dir)[5]
    _, original = _build(spec)
    _, factory = _build(spec)
    native = NativeWorkerCollector(factory)
    a, b = tmp_path/"original", tmp_path/"native"
    inference = WorkerInference(factory._batched[0])
    try:
        for collector, directory in ((original, a), (native, b)):
            partial = prepare(collector, directory, [5, 8, 17, 29])
            partial.write(first)
            collector.listed([8, 17, 29])
    finally:
        inference.close()
    assert game_bytes(a) == game_bytes(b)
    assert game_bytes(a)[5] == first
    assert set(game_bytes(a)) == {5, 8, 17, 29}
    assert native.games_started() == original.games_started() == 5


@pytest.mark.skipif(not available(), reason="optional native extension")
def test_native_refuses_play_without_a_durable_parent_plan(tmp_path):
    _, factory = _build(specification())
    native = NativeWorkerCollector(factory)
    with pytest.raises(ValueError, match="durable"):
        native.cohort(1)
    native.sink = _Keep(Partial(tmp_path/"no-plan"))
    with pytest.raises(ValueError, match="plan"):
        native.cohort(1)


@pytest.mark.skipif(not available(), reason="optional native extension")
def test_worker_exits_after_a_sink_fails_following_its_durable_write(tmp_path, monkeypatch):
    import hexn.collect as collect
    import hexn.native_collect as module

    selected = []

    class SelectedNative(NativeWorkerCollector):
        def __init__(self, original):
            super().__init__(original)
            selected.append(self)

    class FailedRetention(_Keep):
        def __call__(self, episode):
            super().__call__(episode)
            raise OSError("failure after durable retention")

    class Connection:
        def __init__(self, partial):
            self.commands = [("cohort", (1, partial)), ("counter", None)]
            self.sent = []

        def recv(self):
            return self.commands.pop(0)

        def send(self, answer):
            self.sent.append(answer)

    monkeypatch.setattr(module, "NativeWorkerCollector", SelectedNative)
    monkeypatch.setattr(collect, "_Keep", FailedRetention)
    partial = Partial(tmp_path / "kept")
    partial.write_plan([5])
    connection = Connection(partial)
    _serve(replace(specification(), first_game=5, lanes=1), connection)
    assert len(selected) == 1
    assert connection.commands == [("counter", None)]
    assert len(connection.sent) == 1 and connection.sent[0][0] == "error"
    assert "failure after durable retention" in connection.sent[0][1]
    assert partial.finished() == [5]
    assert set(game_bytes(partial.directory)) == {5}
