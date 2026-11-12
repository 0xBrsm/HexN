"""A committee of a contract-9 and a contract-8 net, asked one atomic move at a time, plays the same games and
records the same labels, byte for byte, on native lanes as on Python lanes (`hexset.gym.driver`)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hexset.gym.driver import Driver  # noqa: E402
from hexset.gym.lanes import LaneEnv  # noqa: E402
from hexset.gym.native import available  # noqa: E402
from hexset.gym.seats import Labels, Play, PolicySeat  # noqa: E402
from hexset.record import to_json  # noqa: E402
from hexn import netbot  # noqa: E402

from test_serve import SEED, checkpoints  # noqa: E402

pytestmark = pytest.mark.skipif(not available(), reason="optional native extension")


def play(paths, topology, kind):
    nets = [netbot.load_torch(path, topology).policy for path in paths]
    seat = PolicySeat(nets, {0: Play(members=(0, 1), temperature=0.5), 1: Play(members=(1,), temperature=1.0)},
                      record=True)
    env = LaneEnv(4, SEED, 3, deal=5, action_cap=400, discard_order="random", caster=lambda i: (0, 1, 0, 1))
    labels, kept = Labels(), {}
    driver = Driver(env, {0: seat, 1: seat}, on_decision=labels, lanes=kind,
                    keep=lambda e: kept.update({e.index: (to_json(e.record), e.outcome, [
                        (x.seat, x.chosen, x.priors.tobytes(), x.values.tobytes()) for x in labels.take(e.index)])}))
    driver.play()
    assert driver.lanes == kind and seat.native_atomic == {"9", "8"}
    return kept, seat.rows


def test_a_mixed_contract_committee_plays_the_same_games_and_labels_on_native_lanes(tmp_path):
    paths, topology = checkpoints(tmp_path)
    python, rows_a = play(paths, topology, "python")
    native, rows_b = play(paths, topology, "native")
    assert len(python) == 5 and rows_a == rows_b > 100
    assert python == native
