# SPDX-License-Identifier: GPL-3.0-only
"""Contract 9's unit of credit is a decision, a run of options
(`hexset.decisions.closes`), while each part is one forward's choice: the
batch's decisions and their parts, the log-probability as the sum of the
parts', the two clips, value rows only at a decision's first part, the double
turn as two setup turns, and the contract-9 BC checkpoints' shapes loading
into the self-play net."""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset import contract9, decisions as D  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.arena import deal_game  # noqa: E402
from hexset.game import UNSTRUCTURED_TURN_CAP, to_move  # noqa: E402
from hexn import gamefile  # noqa: E402
from hexn.model9 import HexNet, ModelConfig, chosen_scores, collate, config_of, decision_scores, unpack9  # noqa: E402
from hexn.policy9 import NetworkPolicy9  # noqa: E402
from hexn.ppo import (  # noqa: E402
    PPOConfig, attach_proximal, decision_parts, decision_rows, decision_surrogate, decision_terms, update,
)
from hexn.ppo.prepare import prepare_columns9  # noqa: E402
from hexn.selfplay import Collector  # noqa: E402

SMALL = ModelConfig(width=16, rounds=2, messages="pooled", attention=False, points=True, double_turn="parts")
CONFIG = PPOConfig(contract="9", aux_target="zero_sum", aux_margin_weight=0.1, minibatch=64, epochs=1)

# The `model_config`s contract-9 BC checkpoints record (pooled messages, with
# and without seat attention, with a points head): what the self-play net
# must load.
BC_SHAPES = {
    "lite": {"width": 64, "rounds": 3, "heads": "linear", "discard": "mlp", "norm": True, "messages": "pooled",
             "attention": True},
    "noattn": {"width": 64, "rounds": 3, "heads": "linear", "discard": "mlp", "norm": True, "messages": "pooled",
               "attention": False},
    "final": {"width": 64, "rounds": 3, "heads": "linear", "discard": "mlp", "norm": True, "messages": "pooled",
              "attention": False, "points": True},
}
# The double turn's own heads, which a "parts" net reads nowhere.
from hexn.model9 import JOINT_HEADS, without_joint_head
from _kept import kept


def net(config: ModelConfig = SMALL, seed: int = 0, scale: float = 1.0) -> HexNet:
    game = deal_game(0, 0, 4)
    torch.manual_seed(seed)
    out = HexNet(space_for(game), game._state.board.topology, 4, config).eval()
    with torch.no_grad():
        for head in (out.vertex_head, out.road_head, out.global_head, out.setup_head, out.road_building):
            head.weight.mul_(scale * 100)
    return out


def policy(scale: float = 0.05) -> NetworkPolicy9:
    model = net(scale=scale)
    return NetworkPolicy9(model, model.space)


@pytest.fixture(scope="module")
def played():
    """Three games of self-play, their rows and their batch."""
    pol = policy()
    episodes = Collector(pol, lanes=3, deal=3, contract="9", sample_stream="game", discard_order="random",
                         action_cap=700, turn_cap=UNSTRUCTURED_TURN_CAP, seed=7, keep=kept()).drain()
    episodes = sorted(episodes, key=lambda e: e.index)
    games = [gamefile.encode(e) for e in episodes]
    batch, _, _ = prepare_columns9(gamefile.decode_columns(games), CONFIG, 4)
    return pol, episodes, games, batch


def rows(episodes):
    return [t for e in episodes for seat in e.trajectories for t in seat]


def decisions_of(episodes):
    """Each decision's rows, in the batch's order (game, seat, step)."""
    out = []
    for t in rows(episodes):
        if t.part == 0:
            out.append([t])
        else:
            assert out[-1][-1].decision_id == t.decision_id and out[-1][-1].part == t.part - 1
            out[-1].append(t)
    return out


def test_a_batch_row_is_a_part_and_its_decisions_are_the_runs(played):
    _, episodes, _, batch = played
    grouped = decisions_of(episodes)
    lengths = (batch.dec_off[1:] - batch.dec_off[:-1]).tolist()
    assert lengths == [len(d) for d in grouped]
    assert len(batch) == len(rows(episodes)) and batch.units() == len(grouped) < len(batch)
    assert batch.value_target.shape[0] == batch.advantage.shape[0] == batch.margin_target.shape[0] == len(grouped)
    # Post-roll build runs and the double turn make decisions of several parts.
    assert max(lengths) >= 3
    snake = [d for d in grouped if [t.kind for t in d] == [D.DecisionKind.SETUP] * 2]
    assert len(snake) == 3


def test_a_decisions_log_prob_is_the_sum_of_its_parts_each_computed_alone(played):
    pol, episodes, _, batch = played
    every = torch.arange(batch.units())
    part_rows, owner, heads, lengths = decision_parts(batch.dec_off, every)
    assert torch.equal(part_rows, torch.arange(len(batch)))
    with torch.no_grad():
        evaluation = pol.evaluate(batch.buffer, batch.mask, batch.chosen, decision_rows(batch, part_rows, "cpu"))
    summed = torch.zeros(batch.units()).index_add(0, owner, evaluation.log_prob)
    recorded = [sum(t.log_prob for t in d) for d in decisions_of(episodes)]
    np.testing.assert_allclose(summed.numpy(), recorded, rtol=0, atol=2e-5)
    # Each part on its own: one forward on its position, the softmax over its options.
    model = pol.net
    for d in [d for d in decisions_of(episodes) if len(d) >= 3][:4] + decisions_of(episodes)[:3]:
        alone = 0.0
        for t in d:
            stated = {k: torch.from_numpy(v) for k, v in collate([t.decision], [t.kind], [t.index]).items()}
            del stated["double"]
            with torch.no_grad():
                out = model(**unpack9(pol.layout, torch.from_numpy(t.observation[None])))
                scores = decision_scores(model, out, stated)
            alone += float(chosen_scores(scores, stated)[0] - scores.total[0])
        assert math.isclose(alone, sum(t.log_prob for t in d), rel_tol=0, abs_tol=2e-5)


def test_the_two_clips_by_hand():
    """Two decisions: one of two parts (advantage 1), one of one (-1)."""
    log_prob = torch.tensor([0.15, 0.15, -0.5])
    zero = torch.zeros(3)
    owner = torch.tensor([0, 0, 1])
    advantage = torch.tensor([1.0, -1.0])
    whole = decision_surrogate(log_prob, zero, zero, owner, 2, advantage, PPOConfig(contract="9"), False)
    parts = decision_surrogate(log_prob, zero, zero, owner, 2, advantage,
                               PPOConfig(contract="9", c9_clip="per-part"), False)
    # The run's ratio exp(0.3) is past 1.2 and clipped once; each part's exp(0.15) is inside.
    np.testing.assert_allclose(whole.numpy(), [1.2, -0.8], rtol=1e-6)
    np.testing.assert_allclose(parts.numpy(), [2 * math.exp(0.15), -0.8], rtol=1e-6)


def test_a_runaway_decision_ratio_stays_finite():
    """A decision the update has moved 200 nats above its anchor, whose
    behaviour weight has underflowed to zero: `e**200` overflows float32,
    and the uncapped surrogate was `inf * 0`, NaN (the contract-8 path's cap,
    `RATIO_LOG_CAP`, did not reach `decision_surrogate`). Both clips."""
    log_prob = torch.tensor([100.0, 100.0, 200.0])
    anchor = torch.zeros(3)
    old = torch.full((3,), 200.0)
    owner = torch.tensor([0, 0, 1])
    advantage = torch.tensor([1.0, -1.0])
    for clip in ("per-decision", "per-part"):
        out = decision_surrogate(log_prob, anchor, old, owner, 2, advantage,
                                 PPOConfig(contract="9", c9_clip=clip), True)
        assert torch.isfinite(out).all(), (clip, out)


def test_the_two_clips_give_one_gradient_on_policy(played):
    """At the behaviour policy every ratio is 1: the per-part sum and the
    decision's ratio differ in value but not in gradient."""
    pol, _, _, batch = played
    decisions = torch.arange(min(batch.units(), 120))
    part_rows, owner, heads, lengths = decision_parts(batch.dec_off, decisions)
    advantage = torch.randn(len(decisions), generator=torch.Generator().manual_seed(1))
    grads = []
    for clip in ("per-decision", "per-part"):
        config = PPOConfig(contract="9", c9_clip=clip, value_coefficient=0.0, entropy_coefficient=0.0)
        pol.net.zero_grad(set_to_none=True)
        terms = decision_terms(pol, batch.buffer[part_rows], batch.mask[part_rows], batch.chosen[part_rows],
                               decision_rows(batch, part_rows, "cpu"), batch.log_prob[part_rows], None, owner,
                               heads, lengths, advantage, batch.value_target[decisions],
                               batch.margin_target[decisions], config)
        terms.policy_term.backward()
        grads.append(torch.cat([p.grad.flatten() for p in pol.net.parameters() if p.grad is not None]))
        assert float(terms.clip_fraction) == 0.0
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-7)
    pol.net.zero_grad(set_to_none=True)


def test_value_rows_are_taken_at_a_decisions_first_part_only(played):
    _, episodes, games, batch = played
    columns = gamefile.decode_columns(games)
    for game in columns.games:
        at = {name: i for i, name in enumerate(game["names"])}
        moved = []
        for row in game["rows"]:
            row = list(row)
            if row[at["part"]] > 0:      # a mid-run estimate: read by nothing
                row[at["value"]] = tuple(v + 0.3 for v in row[at["value"]])
            moved.append(tuple(row))
        game["rows"] = moved
    again, _, _ = prepare_columns9(columns, CONFIG, 4)
    for field in ("advantage", "value_target", "margin_target", "dec_off", "log_prob"):
        assert torch.equal(getattr(again, field), getattr(batch, field)), field
    # A first part's estimate moves the advantages.
    game = columns.games[0]
    at = {name: i for i, name in enumerate(game["names"])}
    first = list(game["rows"][0])
    first[at["value"]] = tuple(v + 0.3 for v in first[at["value"]])
    game["rows"] = [tuple(first)] + game["rows"][1:]
    moved, _, _ = prepare_columns9(columns, CONFIG, 4)
    assert not torch.equal(moved.advantage, batch.advantage)


def test_an_update_draws_whole_decisions_and_logs_the_clips_by_length(played):
    pol, _, _, batch = played
    for clip in ("per-decision", "per-part"):
        trained = NetworkPolicy9(net(scale=0.05), pol.space)
        trained.net.train()
        config = PPOConfig(contract="9", aux_target="zero_sum", aux_margin_weight=0.1, minibatch=48, epochs=2,
                           c9_clip=clip, decoupled=True, learning_rate=3e-3)
        optimiser = torch.optim.Adam(trained.net.parameters(), lr=3e-3)
        stats = update(trained, optimiser, attach_proximal(batch, trained), config,
                       generator=torch.Generator().manual_seed(0))
        assert stats.positions == len(batch) and stats.decisions == batch.units()
        lengths = (batch.dec_off[1:] - batch.dec_off[:-1])
        by_length = stats.clip_by_parts
        assert by_length["1"]["decisions"] == 2 * int((lengths == 1).sum())
        assert sum(v["parts"] for v in by_length.values()) == 2 * len(batch)
        assert all(0.0 <= v["decision"] <= 1.0 and 0.0 <= v["part"] <= 1.0 for v in by_length.values())
        assert math.isfinite(stats.policy_loss) and math.isfinite(stats.value_loss)
        # The double turn's joint heads are not there to step: a "parts" net has none.
        assert not any(k.startswith(JOINT_HEADS) for k in trained.net.state_dict())


def test_the_double_turn_in_two_parts_is_a_distribution_over_the_joint_heads_pairs():
    """The snake seat's double turn as two setup turns: every ordered pair the
    joint head scores is reachable, nothing else is, and the parts' product
    sums to one. It is not the joint head's distribution: that one scores
    both turns off the board before either (first + second + pair heads), the
    parts the setup head on the board before and on the board after the first
    placement, different heads on different positions."""
    model = net(scale=0.05)
    space = model.space
    game = deal_game(7, 0, 4)
    for _ in range(3):
        D.apply_option(game, D.options(game)[0])
    assert D.is_double_turn(game)
    joint = D.tensors(game, 3, space)
    first = D.tensors(game, 3, space, double_turn=False)
    assert joint.double and not first.double and np.array_equal(joint.setup, first.setup)
    pairs = contract9.incidences(game._state.board.topology)
    layout = contract9.layout(4, model.num_hexes, model.num_vertices, model.num_edges)

    def log_probs(positions):
        x = torch.from_numpy(np.stack([contract9.flatten(contract9.encode9(g, 3)) for g in positions]))
        stated = [D.tensors(g, 3, space, double_turn=False) for g in positions]
        batch = {k: torch.from_numpy(v) for k, v in collate(stated).items()}
        del batch["double"]
        with torch.no_grad():
            scores = decision_scores(model, model(**unpack9(layout, x)), batch)
        return (scores.setup - scores.total[:, None]).double(), stated

    lp_first, _ = log_probs([game])
    legal = np.flatnonzero(first.setup)
    after = []
    for i in legal:
        g = copy.deepcopy(game)
        D.apply_option(g, D.Compound(D.CompoundKind.SETUP, D.setup_parts(*pairs[i])))
        assert to_move(g) == 3 and not D.closes(g, D.Compound(D.CompoundKind.SETUP, D.setup_parts(*pairs[i])), 3)
        after.append(g)
    lp_second, stated = log_probs(after)
    ok = D.double_pairs(game._state.board.topology)
    total = []
    for row, i in enumerate(legal):
        reachable = stated[row].setup
        assert np.array_equal(reachable, joint.setup & ok[i]), pairs[i]
        total.append(lp_first[0, i] + lp_second[row][torch.from_numpy(reachable)])
    assert math.isclose(float(torch.logsumexp(torch.cat(total), 0)), 0.0, abs_tol=1e-5)   # fp32 scores


@pytest.mark.parametrize("name", sorted(BC_SHAPES))
def test_a_bc_checkpoints_shape_loads_into_the_self_play_net(name):
    """A BC net's parameters bar the double turn's joint heads load strictly
    into the "parts" net self-play builds, which has no such heads; the
    joint heads are what `without_joint_head` drops, and nothing else."""
    shape = BC_SHAPES[name]
    bc = net(config_of(shape))
    assert bc.config.double_turn == "joint" and bc.first_head is not None
    played = net(config_of({**shape, "double_turn": "parts"}), seed=1)
    assert played.first_head is None and played.pair_head is None and not hasattr(played, "pairs_ok")
    with pytest.raises(RuntimeError):
        played.load_state_dict(bc.state_dict())
    kept, dropped = without_joint_head(bc.state_dict())
    played.load_state_dict(kept)
    assert dropped == 2 * (2 * shape["width"] + 1) + shape["width"] ** 2
    assert set(kept) == set(played.state_dict())


def _checkpoints():
    return [p for p in os.environ.get("HEXN_C9_CHECKPOINTS", "").split(os.pathsep) if p]


@pytest.mark.skipif(not _checkpoints(), reason="HEXN_C9_CHECKPOINTS names no contract-9 checkpoint")
@pytest.mark.parametrize("path", _checkpoints())
def test_a_contract_9_checkpoint_loads_and_plays_in_self_play(path):
    """A real checkpoint (`HEXN_C9_CHECKPOINTS`, paths joined by the path
    separator): strictly into the self-play net, and one game played."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = config_of({**state["model_config"], "double_turn": "parts"})
    model = net(config)
    model.load_state_dict(without_joint_head(state["net"])[0])
    pol = NetworkPolicy9(model.eval(), model.space)
    episode = Collector(pol, lanes=1, deal=1, contract="9", sample_stream="game", discard_order="random",
                        action_cap=400, turn_cap=UNSTRUCTURED_TURN_CAP, seed=3, keep=kept()).drain()[0]
    assert rows([episode]) and all(t.kind != D.DecisionKind.DOUBLE for t in rows([episode]))


def test_a_parts_net_plays_the_double_turn_a_setup_turn_at_a_time():
    """`hexn.policy9.Policy9` (the arena's and the reads' bot) on a "parts"
    net: the snake seat's second settlement is chosen on the board after its
    first, by a forward of its own; the joint heads are never asked."""
    from hexset.actions import apply, options_for
    from hexset.game import Phase
    from hexn.policy9 import Policy9

    model = net(scale=0.05)
    bot = Policy9(model, model.space)
    asked = []
    original = bot._choose_many

    def decide(rows):
        for game, seat in rows:
            asked.append((seat, D.tensors(game, seat, model.space, double_turn=False).double, D.is_double_turn(game)))
        return original(rows)

    bot._choose_many = decide
    assert model.first_head is None      # a "parts" net has no joint head to read
    game = deal_game(5, 0, 4)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        seat = to_move(game)
        apply(game, bot.act_rows([(game, seat, tuple(options_for(game)))])[0])
    assert [s for s, _, _ in asked] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert [double for _, _, double in asked].count(True) == 1 and not any(d for _, d, _ in asked)
