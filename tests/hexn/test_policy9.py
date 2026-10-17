# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.policy9.NetworkPolicy9`: contract-9 self-play's sampling policy. What
it records is what the update recomputes, its draw follows its distribution
over every legal option, and a game is a pure function of its index."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset import contract9, decisions as D  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.arena import deal_game  # noqa: E402
from hexset.game import UNSTRUCTURED_TURN_CAP  # noqa: E402
from hexn.model9 import HexNet, ModelConfig, collate, decision_entropy, decision_scores, unpack9  # noqa: E402
from hexn.policy9 import NetworkPolicy9  # noqa: E402
from hexn.ppo import Batch, attach_proximal, decision_rows  # noqa: E402
from hexn.selfplay import Collector, Request  # noqa: E402

CONFIG = ModelConfig(width=16, rounds=2, messages="pooled", attention=False, points=True)


def net(seed: int = 0, scale: float = 1.0) -> HexNet:
    """A small contract-9 net; `scale` sharpens its option logits past an
    untrained net's near-uniform ones, so the draws below test something."""
    game = deal_game(0, 0, 4)
    torch.manual_seed(seed)
    out = HexNet(space_for(game), game._state.board.topology, 4, CONFIG).eval()
    with torch.no_grad():
        for head in (out.vertex_head, out.road_head, out.global_head, out.setup_head, out.first_head,
                     out.second_head, out.discard_head[-1], out.road_building):
            head.weight.mul_(scale * 100)
    return out


def policy(seed: int = 0, scale: float = 1.0) -> NetworkPolicy9:
    model = net(seed, scale)
    return NetworkPolicy9(model, model.space)


def collector(pol, lanes: int, games: int, first_game: int = 0, stride: int = 1, order: str = "random"):
    return Collector(pol, lanes=lanes, deal=games, first_game=first_game, stride=stride, contract="9",
                     sample_stream="game", discard_order=order, action_cap=600, turn_cap=UNSTRUCTURED_TURN_CAP,
                     seed=7)


@pytest.fixture(scope="module")
def played():
    pol = policy(scale=0.05)
    episodes = collector(pol, lanes=3, games=3).drain()
    return pol, episodes


def rows(episodes):
    return [t for e in episodes for seat in e.trajectories for t in seat]


def batch_of(transitions) -> Batch:
    """The update's batch of these rows, as `prepare_columns9` lays it out."""
    decisions = [t.decision for t in transitions]
    setups = [k for k, d in enumerate(decisions) if d.setup.any()]
    setup_row = np.full(len(decisions), -1, dtype=np.int64)
    setup_row[setups] = np.arange(len(setups))
    lengths = lambda field: np.concatenate([[0], np.cumsum([len(getattr(d, field)) for d in decisions])])
    return Batch(
        buffer=torch.from_numpy(np.stack([t.observation for t in transitions])),
        mask=torch.from_numpy(np.stack([d.flat for d in decisions])),
        chosen=torch.tensor([t.index for t in transitions]),
        log_prob=torch.tensor([t.log_prob for t in transitions], dtype=torch.float32),
        advantage=torch.zeros(len(transitions)),
        value_target=torch.zeros(len(transitions), 4),
        margin_target=torch.zeros(len(transitions), 4),
        kind=torch.tensor([t.kind for t in transitions], dtype=torch.int8),
        setup=torch.from_numpy(np.stack([decisions[k].setup for k in setups])),
        setup_row=torch.from_numpy(setup_row),
        rb=torch.from_numpy(np.concatenate([d.road_building for d in decisions])),
        rb_off=torch.from_numpy(lengths("road_building").astype(np.int64)),
        disc=torch.from_numpy(np.concatenate([d.discard for d in decisions])),
        disc_off=torch.from_numpy(lengths("discard").astype(np.int64)),
    )


def test_every_kind_of_decision_is_played_and_filed_once(played):
    _, episodes = played
    kinds = {t.kind for t in rows(episodes)}
    # The snake seat's double turn is two setup turns: no row is one.
    assert {D.DecisionKind.FLAT, D.DecisionKind.SETUP} <= kinds and D.DecisionKind.DOUBLE not in kinds
    for episode in episodes:
        steps = sorted(t.step for t in rows([episode]))
        assert len(steps) == len(set(steps))
        # Every move of the game is some decision's, and a compound's later
        # moves are no decision of their own.
        assert len(episode.record.actions) == episode.outcome.actions >= len(steps)


def test_the_update_recomputes_the_log_prob_the_policy_recorded(played):
    """fp32 on both sides: the same function on the same rows."""
    pol, episodes = played
    transitions = rows(episodes)
    batch = attach_proximal(batch_of(transitions), pol, chunk=97)
    gap = (batch.prox_log_prob - batch.log_prob).abs()
    for kind in D.DecisionKind:
        picked = batch.kind == int(kind)
        if picked.any():
            assert float(gap[picked].max()) < 1e-5, kind.name


def test_the_recorded_log_prob_is_the_softmax_over_every_option_of_the_decision(played):
    pol, episodes = played
    model = pol.net
    for t in [t for t in rows(episodes) if t.kind != D.DecisionKind.FLAT][:6] + rows(episodes)[100:104]:
        batch = {k: torch.from_numpy(v) for k, v in collate([t.decision]).items()}
        with torch.no_grad():
            out = model(**unpack9(pol.layout, torch.from_numpy(t.observation[None])))
            scores = decision_scores(model, out, batch)
        everything = torch.cat([scores.flat[0], scores.setup[0], scores.rb, scores.disc,
                                scores.double.reshape(-1)])
        everything = everything[torch.isfinite(everything)]
        assert math.isclose(float(torch.logsumexp(everything, 0)), float(scores.total[0]), rel_tol=1e-5)
        assert t.log_prob <= 0.0


def test_the_entropy_is_the_decisions_own_over_every_option(played):
    pol, episodes = played
    picked = [t for t in rows(episodes) if t.kind != D.DecisionKind.FLAT][:8] + rows(episodes)[50:58]
    batch = {k: torch.from_numpy(v) for k, v in collate([t.decision for t in picked]).items()}
    with torch.no_grad():
        out = pol.net(**unpack9(pol.layout, torch.from_numpy(np.stack([t.observation for t in picked]))))
        scores = decision_scores(pol.net, out, batch)
        entropy = decision_entropy(scores)
    for i in range(len(picked)):
        parts = [scores.flat[i], scores.setup[i], scores.rb[scores.rb_row == i], scores.disc[scores.disc_row == i]]
        if i in scores.double_rows.tolist():
            parts.append(scores.double[scores.double_rows.tolist().index(i)])
        s = torch.cat(parts)
        p = torch.softmax(s[torch.isfinite(s)].double(), 0)
        assert math.isclose(float(-(p * p.log()).sum()), float(entropy[i]), rel_tol=1e-4, abs_tol=1e-6)


def test_the_entropy_gradient_is_finite():
    pol = policy()
    episodes = collector(pol, lanes=2, games=2).drain()
    transitions = rows(episodes)
    batch = batch_of(transitions)
    decisions = decision_rows(batch, torch.arange(len(batch)), "cpu")
    evaluation = pol.evaluate(batch.buffer, batch.mask, batch.chosen, decisions)
    (evaluation.entropy.mean() + evaluation.log_prob.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in pol.net.parameters() if p.grad is not None)


def test_a_keyed_draw_follows_the_decisions_distribution():
    """The snake seat's first setup turn drawn under many keys: the empirical
    choice frequencies follow the softmax the policy records."""
    pol = policy(scale=0.02)
    game = deal_game(7, 0, 4)
    from hexset.decisions import apply_option, options
    # Seats 0-2 take their first setup turns; seat 3's is the double turn.
    for _ in range(3):
        apply_option(game, options(game)[0])
    assert D.is_double_turn(game) and D.tensors(game, 3, pol.space).double
    decision = D.tensors(game, 3, pol.space, double_turn=False)
    assert not decision.double
    request = lambda key: Request(lane=0, seat=3, observation=contract9.flatten(contract9.encode9(game, 3)),
                                  mask=decision.flat, options=(), game=game, decision=decision, draw=key)
    choices = pol.act([request(k * 7919 + 1) for k in range(4000)])
    counts: dict = {}
    for c in choices:
        counts[c.decision] = counts.get(c.decision, 0) + 1
    top = max(counts, key=counts.get)
    expected = math.exp(next(c.log_prob for c in choices if c.decision == top))
    assert abs(counts[top] / 4000 - expected) < 4 * math.sqrt(expected * (1 - expected) / 4000) + 1e-3
    again = pol.act([request(k * 7919 + 1) for k in range(20)])
    assert [c.decision for c in again] == [c.decision for c in choices[:20]]


def test_a_game_is_the_same_whatever_the_lanes_and_shards():
    pol = policy(scale=0.05)
    alone = {e.index: e.record for e in collector(pol, lanes=1, games=3).drain()}
    sharded = {}
    for worker in range(2):
        share = len(range(worker, 3, 2))
        sharded.update({e.index: e.record for e in collector(pol, lanes=2, games=share, first_game=worker,
                                                             stride=2).drain()})
    assert alone == sharded


def test_greedy_takes_the_best_scored_decision(played):
    pol, episodes = played
    greedy = NetworkPolicy9(pol.net, pol.space, greedy=True)
    episode = collector(greedy, lanes=1, games=1).drain()[0]
    for t in rows([episode])[:30]:
        batch = {k: torch.from_numpy(v) for k, v in collate([t.decision], [t.kind], [t.index]).items()}
        with torch.no_grad():
            out = pol.net(**unpack9(pol.layout, torch.from_numpy(t.observation[None])))
            from hexn.model9 import decision_log_prob
            _, hit = decision_log_prob(pol.net, out, batch)
        assert bool(hit[0])


def test_a_contract_9_collector_seats_only_bots_beside_its_learner():
    pol = policy()
    with pytest.raises(ValueError, match="scripted bots"):
        Collector(pol, contract="9", opponents=[pol], caster=lambda i: (0, 1, 0, 1))


def test_fp16_autocast_moves_the_log_prob_only_by_its_rounding(played):
    """The update's fp16 forward against the fp32 the games were played in:
    what decoupled PPO's behaviour weight absorbs. CPU autocast, where it runs."""
    pol, episodes = played
    batch = batch_of(rows(episodes))
    try:
        half = attach_proximal(Batch(**{**batch.__dict__, "buffer": batch.buffer.half()}), pol, amp="fp16")
    except RuntimeError as error:  # an fp16 kernel this CPU build lacks
        pytest.skip(f"no fp16 autocast here: {error}")
    gap = (half.prox_log_prob - batch.log_prob).abs()
    assert float(gap.max()) < 0.1
    print(f"fp16 log-prob gap: max {float(gap.max()):.2e}, mean {float(gap.mean()):.2e}")


def test_a_double_turn_decision_is_refused_in_self_play():
    pol = policy()
    game = deal_game(7, 0, 4)
    for _ in range(3):
        D.apply_option(game, D.options(game)[0])
    decision = D.tensors(game, 3, pol.space)
    request = Request(lane=0, seat=3, observation=contract9.flatten(contract9.encode9(game, 3)),
                      mask=decision.flat, options=(), game=game, decision=decision, draw=1)
    with pytest.raises(ValueError, match="two setup turns"):
        pol.act([request])


def _draw_with_noise_everywhere(pol, scores, batch, keys):
    """`NetworkPolicy9._draw` with Gumbel noise on every slot of every row,
    legal or not, as it once drew."""
    from hexn.policy import keyed_gumbel, keyed_gumbel_at
    from hexn.policy9 import _block_argmax

    rows = scores.flat.shape[0]
    setup_at, candidate_at = pol.slots()
    key = np.asarray(keys, dtype=np.uint64)
    flat = scores.flat.double().numpy() + keyed_gumbel(key, scores.flat.shape[1])
    setup = scores.setup.double().numpy() + keyed_gumbel(key, scores.setup.shape[1], setup_at)
    best = [(flat.max(1), flat.argmax(1)), (setup.max(1), setup.argmax(1))]
    for name in ("rb", "disc"):
        values, row_of = getattr(scores, name).double().numpy(), batch[f"{name}_row"].numpy()
        within = batch[f"{name}_index"].numpy()
        if len(values):
            values = values + keyed_gumbel_at(key[row_of], candidate_at + within)
        top, at = _block_argmax(values, row_of, rows)
        best.append((top, np.where(at >= 0, within[np.maximum(at, 0)], -1) if len(values) else at))
    block = np.stack([b[0] for b in best], 1).argmax(1)
    kinds = np.asarray([D.DecisionKind.FLAT, D.DecisionKind.SETUP, D.DecisionKind.ROAD_BUILDING,
                        D.DecisionKind.DISCARD])[block]
    return kinds, np.stack([b[1] for b in best], 1)[np.arange(rows), block]


def test_noise_drawn_on_the_legal_options_only_draws_what_noise_everywhere_did(played):
    """Every kind of decision, under many keys each: the same option."""
    pol, episodes = played
    transitions = rows(episodes)
    kinds = {t.kind for t in transitions}
    assert {D.DecisionKind.FLAT, D.DecisionKind.SETUP} <= kinds
    batch = {k: torch.from_numpy(v) for k, v in collate([t.decision for t in transitions]).items()}
    del batch["double"]
    x = torch.from_numpy(np.stack([t.observation for t in transitions]))
    with torch.inference_mode():
        scores = decision_scores(pol.net, pol.net(**unpack9(pol.layout, x)), batch)
    rng = np.random.default_rng(0)
    for _ in range(20):
        keys = rng.integers(0, 2**62, len(transitions)).tolist()
        got = pol._draw(scores, batch, keys)
        want = _draw_with_noise_everywhere(pol, scores, batch, keys)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_the_noise_at_a_slot_is_the_same_drawn_alone_or_in_a_row():
    from hexn.policy import keyed_gumbel, keyed_gumbel_at

    keys = np.random.default_rng(1).integers(0, 2**63, 512, dtype=np.uint64)
    whole = keyed_gumbel(keys, 777, 5)
    rows, slots = np.nonzero(np.random.default_rng(2).random(whole.shape) < 0.05)
    assert np.array_equal(keyed_gumbel_at(keys[rows], (slots + 5).astype(np.uint64)), whole[rows, slots])


@pytest.mark.parametrize("fused", [False, True])
def test_the_forward_without_grad_is_the_forward_with_it_bit_for_bit(played, fused):
    """Collection reads node owners and road ends by `index_select`
    (`hexn.model9._pick`), the update by indexing: the same numbers."""
    pol, episodes = played
    x = torch.from_numpy(np.stack([t.observation for t in rows(episodes)[:300]]))
    was, pol.net.fused = pol.net.fused, fused
    try:
        with torch.inference_mode():
            fast = pol.net(**unpack9(pol.layout, x))
        with torch.enable_grad():
            slow = pol.net(**unpack9(pol.layout, x))
    finally:
        pol.net.fused = was
    for name in ("logits", "road", "value", "value_logits", "points", "seats", "vertices", "hexes", "table"):
        assert torch.equal(getattr(fast, name), getattr(slow, name).detach()), name
