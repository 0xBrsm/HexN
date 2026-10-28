"""`board_mode`: a self-play collector deals each game's board by
`hexset.board.board.BOARD_MODES`, the same board the arena deals that index."""
import pytest

torch = pytest.importorskip("torch")

from hexset.arena import deal_board
from hexset.record import board_fields
from hexn.collect import WorkerSpec, _build
from hexn.selfplay import Collector
from _kept import kept


def worker(**changes):
    fields = dict(seed=7, players=4, lanes=2, action_cap=24, max_offers=0, first_game=0, stride=1,
                  width=8, rounds=1, torch_seed=1, contract="9", cohort=True, sample_stream="game",
                  trunk="fused", net9=(("double_turn", "parts"), ("points", True)))
    return WorkerSpec(**{**fields, **changes})


def boards(episodes):
    return {e.index: {k: getattr(e.record, k) for k in ("terrain", "tokens", "ports")} for e in episodes}


def dealt(seed, index, mode):
    fields = board_fields(deal_board(seed, index, mode))
    return {k: tuple(fields[k]) for k in ("terrain", "tokens", "ports")}


@pytest.mark.parametrize("mode", ["random", "spiral"])
def test_a_worker_deals_its_board_mode_and_resumes_the_same_game(mode):
    torch.set_num_threads(1)
    spec = worker(board_mode=mode)
    _, collector = _build(spec)
    collector.sink = kept().keep  # a worker's command names its partial
    episodes = collector.cohort(4)
    for index, board in boards(episodes).items():
        assert board == dealt(spec.seed, index, mode)
    # A missing game is replayed from its index on its own board.
    _, replay = _build(spec)
    replay.sink = kept().keep  # a worker's command names its partial
    again = replay.listed([episodes[1].index])[0]
    assert again.record == episodes[1].record


def test_the_default_is_the_random_deal():
    assert WorkerSpec(seed=0, players=4, lanes=1, action_cap=1, max_offers=0, first_game=0,
                      stride=1, width=8, rounds=1, torch_seed=0).board_mode == "random"


def test_board_mode_refuses_a_law_that_deals_its_own_board():
    with pytest.raises(ValueError, match="unknown board mode"):
        Collector(None, board_mode="hexagonal", keep=kept())
    with pytest.raises(ValueError, match="decides the board itself"):
        Collector(None, board_mode="spiral", pair_boards=True, keep=kept())
    with pytest.raises(ValueError, match="decides the board itself"):
        Collector(None, board_mode="spiral", deal_law=lambda index: None, keep=kept())
