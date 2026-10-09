# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.gamefile`: a game encoded and decoded is the same game, to the bit,
and assembles to the same batch -- the property a resumed run stands on."""
from __future__ import annotations

import dataclasses
import pickle
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexn import gamefile  # noqa: E402
from hexn.model import HexNet, ModelConfig, pack, packing  # noqa: E402
from hexn.policy import NetworkPolicy  # noqa: E402
from hexn.ppo import PPOConfig, assemble  # noqa: E402
from hexn.selfplay import Choice, Collector  # noqa: E402
from hexn.steps import fingerprint  # noqa: E402
from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.game import start  # noqa: E402

BATCH_FIELDS = ("buffer", "mask", "chosen", "log_prob", "advantage", "value_target", "margin_target")


def a_policy(seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, 4, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space_for(game), packing(graph, 4))


@pytest.fixture(scope="module")
def collected():
    """A network's games: log-probs, values and every field a trainer reads."""
    policy = a_policy()
    episodes = Collector(policy, lanes=4, seed=3, action_cap=300, max_offers=0).collect(4)
    return policy, episodes


class First:
    def act(self, requests):
        return [Choice(action=request.options[0]) for request in requests]


def same(a, b) -> None:
    """Every field equal, every observation and mask equal bit for bit."""
    for f in dataclasses.fields(a):
        if f.name != "trajectories":
            assert getattr(a, f.name) == getattr(b, f.name), f.name
    assert [len(s) for s in a.trajectories] == [len(s) for s in b.trajectories]
    for x, y in (pair for s, r in zip(a.trajectories, b.trajectories) for pair in zip(s, r)):
        for f in dataclasses.fields(x):
            if f.name not in ("observation", "mask"):
                assert getattr(x, f.name) == getattr(y, f.name), f.name
                assert type(getattr(x, f.name)) is type(getattr(y, f.name)), f.name
        assert x.mask.dtype == y.mask.dtype and np.array_equal(x.mask, y.mask)
        for name in gamefile.BLOCKS:
            p, q = getattr(x.observation, name), getattr(y.observation, name)
            assert p.shape == q.shape and p.dtype == q.dtype == np.float32
            assert np.array_equal(p.view(np.uint32), q.view(np.uint32)), name
        assert pickle.dumps(x.observation.graph) == pickle.dumps(y.observation.graph)


def test_a_network_game_comes_back_exactly(collected):
    _, episodes = collected
    for episode in episodes:
        blob = gamefile.encode(episode)
        assert gamefile.index_of(blob) == episode.index
        same(episode, gamefile.decode(blob))
        # The point of the format: a small fraction of the game as objects.
        assert len(blob) * 10 < len(pickle.dumps(episode))


def test_any_bit_pattern_comes_back_exactly(collected):
    """No float arithmetic on the way: signed zeros, NaN payloads, infinities
    and subnormals return as the bits they were."""
    _, episodes = collected
    episode = episodes[0]
    seat = episode.trajectories[0]
    odd = np.array([0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000, 0x00000001, 0x3EAAAAAB], np.uint32)
    first = seat[0].observation
    globals_ = first.globals.copy()
    globals_.view(np.uint32)[: len(odd)] = odd
    changed = dataclasses.replace(seat[0], observation=dataclasses.replace(first, globals=globals_))
    episode = dataclasses.replace(
        episode, trajectories=((changed, *seat[1:]), *episode.trajectories[1:])
    )
    same(episode, gamefile.decode(gamefile.encode(episode)))


def test_empty_seats_and_empty_games_come_back(collected):
    _, episodes = collected
    episode = episodes[0]
    cast = dataclasses.replace(
        episode, cast=(0, 1, 0, 1), trajectories=(episode.trajectories[0], (), episode.trajectories[2], ())
    )
    same(cast, gamefile.decode(gamefile.encode(cast)))
    empty = dataclasses.replace(episode, trajectories=((), (), (), ()))
    same(empty, gamefile.decode(gamefile.encode(empty)))
    decoded = gamefile.decode_all([gamefile.encode(empty), gamefile.encode(cast)])
    same(empty, decoded[0])
    same(cast, decoded[1])


def test_a_policy_with_no_estimates_comes_back():
    episodes = Collector(First(), lanes=2, seed=3, action_cap=60, max_offers=0).collect(2)
    for episode in episodes:
        same(episode, gamefile.decode(gamefile.encode(episode)))


def test_a_collection_decodes_into_one_buffer_and_assembles_to_the_same_batch(collected):
    policy, episodes = collected
    decoded = gamefile.decode_all([gamefile.encode(e) for e in episodes])
    for a, b in zip(episodes, decoded):
        same(a, b)
    packed = {id(t.observation._packed) for e in decoded for s in e.trajectories for t in s}
    assert len(packed) == 1
    for config in (PPOConfig(), PPOConfig(decided_cut=0.05, setup_weight=2)):
        original = assemble(episodes, policy.layout, config)
        again = assemble(decoded, policy.layout, config)
        for field in BATCH_FIELDS:
            assert torch.equal(getattr(original, field), getattr(again, field)), field
        assert fingerprint(original) == fingerprint(again)


def test_a_collection_in_order_is_its_own_batch_buffer(collected):
    """The batch of a whole collection in game order is the decoded buffer
    itself: no second copy of the observations."""
    policy, episodes = collected
    decoded = gamefile.decode_all([gamefile.encode(e) for e in episodes])
    batch = assemble(decoded, policy.layout, PPOConfig())
    source = decoded[0].trajectories[0][0].observation._packed
    assert batch.buffer.data_ptr() == source.ctypes.data


def test_a_half_precision_collection_is_the_rounding_autocast_applies(collected):
    policy, episodes = collected
    blobs = [gamefile.encode(e) for e in episodes]
    full = gamefile.decode_all(blobs)
    half = gamefile.decode_all(blobs, np.float16)
    buffer = half[0].trajectories[0][0].observation._packed
    assert buffer.dtype == np.float16
    reference = full[0].trajectories[0][0].observation._packed
    # Round to nearest-even, as torch casts a float32 tensor to half.
    assert np.array_equal(
        buffer.view(np.uint16),
        torch.from_numpy(reference).half().numpy().view(np.uint16),
    )
    config = PPOConfig(amp="fp16")
    for a, b in ((full, half), (episodes, half)):
        one, other = assemble(a, policy.layout, config), assemble(b, policy.layout, config)
        assert one.buffer.dtype == other.buffer.dtype == torch.float16
        for field in BATCH_FIELDS:
            assert torch.equal(getattr(one, field), getattr(other, field)), field
    # A float32 reader is refused half-precision observations outright.
    with pytest.raises(ValueError, match="float16"):
        assemble(half, policy.layout, PPOConfig())
    with pytest.raises(ValueError, match="float16"):
        pack(policy.layout, [t.observation for t in half[0].trajectories[0]])
    with pytest.raises(ValueError, match="float32"):
        gamefile.encode(half[0])


def test_damage_is_detected_not_decoded(collected):
    _, episodes = collected
    blob = gamefile.encode(episodes[0])
    for bad in (blob[:-7], blob[:10], b"XXXX" + blob[4:], blob[:40] + bytes([blob[40] ^ 0xFF]) + blob[41:]):
        with pytest.raises(gamefile.Corrupt):
            gamefile.check(bad)


def test_the_buffer_is_in_the_network_packing_order(collected):
    policy, _ = collected
    assert tuple(name for name, *_ in policy.layout.blocks) == gamefile.BLOCKS
