# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import legal_actions, space_for  # noqa: E402
from hexset.arena import Entrant, compete, spawn  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import encode, static_graph  # noqa: E402
from hexset.game import start, to_move  # noqa: E402
from hexset.clients.netbot import bot_for, searcher_for  # noqa: E402
from hexn.model import HexNet, ModelConfig  # noqa: E402
from hexn.netbot import load  # noqa: E402
from hexset.play import step_randomly  # noqa: E402


def a_bot(path, board, **kwargs):
    return bot_for(load(path, board.topology), **kwargs)


def a_search(path, board, **kwargs):
    return searcher_for(load(path, board.topology), **kwargs)


def a_checkpoint(
    path,
    *,
    players: int = 4,
    max_trades: int | None = 3,
    seed: int = 0,
    shape: dict | None = None,
):
    """A checkpoint in the shape `hexn.loop.save` writes, tiny enough to load.

    `shape` is omitted by default, which is the case that matters: it is what a
    checkpoint written before the head-shape flags existed looks like, and the
    loader has to keep rebuilding those as `"linear"` on both.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(
        space_for(game), graph, players, ModelConfig(width=16, rounds=1, **(shape or {}))
    )
    torch.save(
        {
            "iteration": 7,
            "net": net.state_dict(),
            "args": {
                "players": players,
                "width": 16,
                "rounds": 1,
                "max_trades": max_trades,
                **(shape or {}),
            },
        },
        path,
    )
    return board


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    board = a_checkpoint(path)
    load.cache_clear()
    yield str(path), board
    load.cache_clear()


def test_a_checkpoint_plays_a_legal_action_from_every_phase(checkpoint):
    path, board = checkpoint
    bot = a_bot(path, board)
    rng = random.Random(3)
    game = start(board, 4, rng)

    seen = set()
    for _ in range(400):
        if game.won_by is not None:
            break
        action = bot.choose(game)
        assert action in legal_actions(game)
        seen.add(game.phase)
        from hexset.actions import apply

        apply(game, action)
    assert len(seen) > 3


def test_a_checkpoint_that_predates_the_head_flags_still_loads_as_the_old_shape(
    checkpoint,
):
    """Every checkpoint under `runs/` has no head shape in its `args`.

    The default the loader reads has to be the shape those runs trained with, or
    `load_state_dict` fails on the keys — which is a working checkpoint made
    unloadable by a flag nobody passed.
    """
    path, board = checkpoint
    config = load(path, board.topology, "cpu").policy.net.config

    assert (config.value_head, config.policy_head) == ("linear", "linear")


@pytest.mark.parametrize(
    "shape",
    [
        {"value_head": "mlp"},
        {"value_head": "pooled"},
        {"value_head": "attn"},
        {"policy_head": "mlp"},
        {"value_head": "mlp_pooled", "policy_head": "mlp"},
    ],
)
def test_a_head_shape_round_trips_through_the_checkpoints_args_dict(tmp_path, shape):
    """A run's shape lives in its `args`, so a duel rebuilds what it scored.

    Without this the shape is known only to the launching command line, and a
    checkpoint from a shaped run would be rebuilt as `"linear"` and fail to
    load — with a key error a week after the run, not at the point of the bug.
    """
    path = tmp_path / "shaped.pt"
    board = a_checkpoint(path, shape=shape)
    load.cache_clear()

    loaded = load(str(path), board.topology, "cpu")

    config = loaded.policy.net.config
    for field, value in shape.items():
        assert getattr(config, field) == value
    position = encode(start(board, 4, random.Random(5)), 0)
    assert loaded.policy.values([position])[0].shape == (4,)
    load.cache_clear()


def test_the_offer_budget_comes_from_the_checkpoint_unless_overridden(checkpoint):
    path, board = checkpoint
    assert a_bot(path, board).max_trades == 3
    assert a_bot(path, board, max_trades=8).max_trades == 8


def test_it_only_ever_plays_an_action_the_engine_offered(checkpoint):
    """There is no budget left to filter by: every legal action is a candidate.

    `max_trades` stopped being a filter on the option list at contract 5 --
    trading is an engine event, not an action (`hexset.trading`) -- so what
    this pins is that the bot's own choice is always in `legal_actions`, and
    that switching trading off changes nothing about which actions it may take.
    """
    path, board = checkpoint
    bot = a_bot(path, board)
    strict = a_bot(path, board, max_trades=0)
    rng = random.Random(11)
    game = start(board, 4, rng)

    from hexset.actions import apply

    steps = 0
    for _ in range(600):
        if game.won_by is not None:
            break
        allowed = legal_actions(game)
        action = bot.choose(game)
        assert action in allowed
        assert strict.choose(game) == action
        apply(game, action)
        steps += 1
    assert steps > 100


def test_a_seated_network_answers_a_gate(checkpoint):
    """`Bot.gains_many` is the trade seam the engine calls -- there is no
    public layer left for a seat to publish (`hexset.trading`)."""
    from hexset.actions import apply
    from hexset.trading import one_for_one

    path, board = checkpoint
    bot = a_bot(path, board)
    rng = random.Random(12)
    game = start(board, 4, rng)
    for _ in range(200):
        if game.won_by is not None:
            break
        apply(game, bot.choose(game))

    seat = to_move(game)
    view = game.state(seat)
    received = one_for_one(0, 1)
    counterparty = (seat + 1) % 4
    gains = bot.gains_many(view, [received], [counterparty])
    assert len(gains) == 1
    assert isinstance(bot.accepts(view, received, counterparty), bool)

    off = a_bot(path, board, max_trades=0)
    off.choose(game)
    assert off.gains_many(view, [received], [counterparty]) == [-1.0]
    assert off.accepts(view, received, counterparty) is False


def test_a_network_entrant_can_play_a_whole_tournament(checkpoint):
    path, board = checkpoint
    lineup = [
        Entrant("network", kind="network", weights=path),
        Entrant("network2", kind="network", weights=path),
        Entrant("random", kind="random"),
        Entrant("random2", kind="random"),
    ]
    result = compete(lineup, 4, seed=1, action_cap=2000)
    assert result.games == 4
    assert sum(s.wins for s in result.standings) + result.unfinished == 4


def test_a_network_entrant_needs_a_path_rather_than_weights():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="checkpoint path"):
        spawn(Entrant("bogus", kind="network", weights=[1.0]), board, random.Random(0))


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(tmp_path):
    path = tmp_path / "three.pt"
    board = a_checkpoint(path, players=3)
    load.cache_clear()
    bot = a_bot(str(path), board)
    game = start(board, 4, random.Random(0))
    with pytest.raises(ValueError, match="trained for 3 players"):
        bot.choose(game)
    load.cache_clear()


def test_scoring_is_greedy_so_a_position_answers_the_same_way_twice(checkpoint):
    """A sampled policy is the behaviour distribution PPO needed, not the
    policy worth scoring, and a duel of one is not a measurement."""
    path, board = checkpoint
    bot = a_bot(path, board)
    game = start(board, 4, random.Random(5))
    rng = random.Random(5)
    for _ in range(40):
        step_randomly(game, rng)
    assert to_move(game) is not None
    assert bot.choose(game) == bot.choose(game)


def test_the_prior_covers_every_option_and_sums_to_one(checkpoint):
    """Every option has its own slot at contract 5, so the leaf prior is a
    plain gather from the masked row -- normalised over the options and with
    nothing parked anywhere a search cannot reach."""
    from hexset.mcts import Leaf

    path, board = checkpoint
    search = a_search(path, board, rng=random.Random(0))
    rng = random.Random(11)
    game = start(board, 4, rng)
    for _ in range(400):
        options = legal_actions(game)
        if len(options) > 3:
            break
        step_randomly(game, rng)
    else:
        pytest.skip("no position offering more than three actions turned up")

    seat = to_move(game)
    (prior, _), = search.evaluator.evaluate([Leaf(game, seat, tuple(options))])
    assert len(prior) == len(options)
    assert all(w > 0 for w in prior)
    assert sum(prior) == pytest.approx(1.0, abs=1e-5)


def test_a_search_over_a_learned_prior_plays_a_legal_action(checkpoint):
    path, board = checkpoint
    search = a_search(path, board, simulations=16, wave=4, rng=random.Random(0))
    game = start(board, 4, random.Random(2))
    for _ in range(20):
        action = search.choose(game)
        assert action in set(legal_actions(game))
        from hexset.actions import apply

        apply(game, action)
