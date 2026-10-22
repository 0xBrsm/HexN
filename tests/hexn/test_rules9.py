"""Rule-conditioned checkpoints, masked values, and the mixed-rule PPO path."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hexset import contract9
from hexset.actions import space_for
from hexset.arena import deal_game
from hexset.gym.distribution import Distribution
from hexn import gamefile
from hexn.collect import WorkerSpec, _build
from hexn.model9 import HexNet, ModelConfig, unpack9
from hexn.ppo import PPOConfig, update, zero_sum_targets
from hexn.ppo.prepare import prepare_games


def spec():
    return {"random": {"weight": 1, "players": [[2, 1], [3, 1], [4, 1]],
                       "winning_points": [[8, 1], [15, 1]], "discard_limit": [[6, 1], [9, 1]],
                       "friendly_robber": [[False, 1], [True, 1]], "balanced_dice": [[False, 1], [True, 1]],
                       "board_mode": [["random", 1], ["spiral", 1]]}}


@pytest.mark.parametrize("fused", [False, True])
def test_old_weights_start_with_the_same_standard_game_predictions(fused):
    torch.set_num_threads(1)
    game = deal_game(5, 0, 4)
    space, topology = space_for(game), game._state.board.topology
    config = ModelConfig(width=8, rounds=1, messages="pooled", attention=False, points=True, double_turn="parts")
    old = HexNet(space, topology, 4, config).eval()
    new = HexNet(space, topology, 4, replace(config, contract="9.1")).eval()
    new.load_state_dict(old.state_dict())
    old.fused = new.fused = fused
    outputs = []
    for net in (old, new):
        x = torch.from_numpy(np.stack([contract9.flatten(contract9.encode9(game, s, net.CONTRACT)) for s in range(4)]))
        with torch.no_grad():
            outputs.append(net(**unpack9(contract9.layout(4, 19, 54, 72, net.CONTRACT), x)))
    for field in ("value", "value_logits", "points", "logits", "road", "vertices"):
        torch.testing.assert_close(getattr(outputs[0], field), getattr(outputs[1], field), rtol=1e-5, atol=1e-6)
    assert not new.embed_table.weight[:, 20:].count_nonzero()
    assert not new.embed_seat.weight[:, 39:].count_nonzero()
    assert not new.embed_hex.weight[:, 7:].count_nonzero()


def test_retired_slots_have_no_win_mass_or_points_and_the_loss_is_finite():
    torch.set_num_threads(1)
    law = Distribution(spec(), 4, 3)
    drawn = next(law(i) for i in range(30) if len(law(i).locked) == 2)
    game = deal_game(5, 0, 4, game_type=drawn.game_type, locked=drawn.locked)
    net = HexNet(space_for(game), game._state.board.topology, 4,
                 ModelConfig(width=8, rounds=1, points=True, contract="9.1"))
    x = torch.from_numpy(contract9.flatten(contract9.encode9(game, 0, "9.1"))[None])
    out = net(**unpack9(contract9.layout(4, 19, 54, 72, "9.1"), x))
    locked = sorted(drawn.locked)
    assert out.value[0, locked].count_nonzero() == 0
    assert out.points[0, locked].count_nonzero() == 0
    torch.testing.assert_close(out.value.sum(1), torch.ones(1))
    target = torch.zeros_like(out.value)
    target[0, next(s for s in range(4) if s not in drawn.locked)] = 1
    loss = -(target * out.value_logits.log_softmax(-1)).sum()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_points_targets_ignore_retired_seats_and_discount_by_live_rounds():
    # The last gain is two live turns ahead: one round, even on four slots.
    points = np.array([[2, 0, 2, 0], [2, 0, 2, 0], [2, 0, 2, 0]])
    got = zero_sum_targets(np.arange(3), np.array([0, 1, 2]), points, [3, 0, 2, 0], 0.6,
                           live=[True, False, True, False])
    np.testing.assert_allclose(got, [[0.6, 0, -0.6, 0], [1, 0, -1, 0], [1, 0, -1, 0]])


def test_a_rule_mixed_cohort_is_durable_resumable_and_trains(tmp_path):
    torch.set_num_threads(1)
    worker = WorkerSpec(seed=5, players=4, lanes=3, action_cap=32, max_offers=0, first_game=0, stride=1,
                        width=8, rounds=1, torch_seed=1, contract="9.1", cohort=True,
                        sample_stream="game", trunk="fused", rules_mix=json.dumps(spec()),
                        net9=(("double_turn", "parts"), ("points", True)))
    policies, collector = _build(worker)
    episodes = collector.cohort(6)
    law = Distribution(worker.rules_mix, 4, 5)
    encoded = []
    for episode in episodes:
        assert frozenset(episode.record.locked) == law(episode.index).locked
        encoded.append(gamefile.encode(episode))
        restored = gamefile.decode(encoded[-1])
        assert restored.record == episode.record
        assert all(not episode.trajectories[s] for s in episode.record.locked)
    # A missing game is replayed from its index, with another lane count.
    _, replay = _build(replace(worker, lanes=1))
    repeated = replay.listed([episodes[2].index])[0]
    assert repeated.record == episodes[2].record
    config = PPOConfig(contract="9.1", aux_target="zero_sum", aux_margin_weight=0.1, epochs=1, minibatch=128)
    batch, _, _ = prepare_games(encoded, np.float32, None, config, 4)
    policy = policies[0]
    optimizer = torch.optim.Adam(policy.net.parameters(), lr=1e-3)
    before = policy.net.embed_table.weight.detach().clone()
    stats = update(policy, optimizer, batch, config)
    assert np.isfinite(stats.policy_loss) and np.isfinite(stats.value_loss)
    assert not torch.equal(before, policy.net.embed_table.weight)
