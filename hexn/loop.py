# SPDX-License-Identifier: GPL-3.0-only
"""The harness `hexn.ppo` carries and `hexn.league`/`hexn.exit` import.

Checkpointing, the duel/versus/ladder evaluation stack and the network
builder are common to every trainer, so they live here rather than in
`hexn.ppo` (PPO's own loop) once a second and third trainer needed them too.

## Checkpointing is for crashes, so it is written to survive one

An overnight run that cannot resume is an overnight run that has to be watched.
Two properties make that work and both are easy to leave out:

*The write is atomic.* A checkpoint is written to a temporary file and renamed
over the live one. A crash during `torch.save` otherwise leaves a truncated file
where the good one used to be, which is the failure mode where you lose the run
*and* the checkpoint at the same moment.

*The game counter is saved.* A game is a pure function of the seed and its
index, so a resumed run that restarts the counter replays exactly the games it
has already learned from. It would look like it was working.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from hexset.actions import space_for
from hexset.bench.versus import BotPolicy, PolicyPolicy, compete_batched
from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from .collect import alternating, frozen
from hexset.encoding import static_graph
from hexset.game import start
from hexset.gym.lanes import BoardBots
from hexset.trading import UNLIMITED, TradeParams
from .model import POLICY_HEADS, VALUE_HEADS, HexNet, ModelConfig, packing
from .policy import NetworkPolicy
from .ppo import ADAM_EPS
from .rewards import reward
from .selfplay import BatchPolicy, Episode
from .trade import trade_params, trader_gate


@dataclass
class Progress:
    iteration: int
    games: int
    positions: int
    collect_seconds: float
    assemble_seconds: float
    update_seconds: float
    positions_per_second: float
    truncated: float
    mean_actions: float
    mean_turns: float
    # The trading canary, and the registration's readout (d): how many
    # exchanges the engine's one trade event a turn actually cleared
    # (`hexset.trading`). It replaces the accept/propose pair, which counted
    # actions that no longer exist. There is no cap, so over-trading from a
    # miscalibrated value head shows up here rather than being silently
    # clipped; the shipped heuristic lineup reads a mean of 0.151.
    trades_per_turn: float = 0.0


@dataclass(frozen=True)
class _Checkpoint:
    """`hexset.clients.policy.Checkpoint` for a policy hexn is holding live.

    `trade_params` is how this network bargains, read by name by
    `hexset.clients.netbot.bot_for` (`hexset.trading.params_of`): the run's own
    offer budget (`hexn.trade.trade_params`). A table no longer caps trading;
    each seat declares its own, and this is the network declaring its own.
    """

    policy: object
    space: object
    players: int
    trade_params: TradeParams = UNLIMITED


class _Traded(PolicyPolicy):
    """A network's moves with a HexSet trader's gate at every seat it holds
    (`hexn.trade.trader_gate`)."""

    def __init__(self, policy, trader: str) -> None:
        super().__init__(policy)
        self.trader = trader

    def gate(self, game, seat: int) -> object:
        return trader_gate(self.trader, game, seat)


def duellist(policy, max_offers: int | None = None, trader: str | None = None) -> BatchPolicy:
    """Whatever hexn seats in an evaluation, behind HexSet's `BatchPolicy`.

    A scripted rung is a `hexset.gym.lanes.BoardBots` bench
    (`hexn.collect.named_opponent`), and HexSet's `BotPolicy` is that bench
    plus the two protocol methods -- rebuilt from the bench's own spawn law,
    so a rung stays literally the entrant the arena scores and is gated by
    the same bot that plays it.
    A network answers `act_rows` and is `hexset.bench.versus.PolicyPolicy`
    over a `_Checkpoint` built from its own `space`/`players`; HexSet seats
    the gate it builds (0.45.1). Anything
    else already answers `act` over the lane driver's own requests and is
    passed through; a gate it wants seated it exposes as `BatchPolicy.gate`.

    `max_offers` is the network's own offer budget (`hexn.trade`): it goes on
    the `_Checkpoint`, which is where HexSet reads how a network bargains.
    `trader` gates a network's seats with that HexSet bot instead. A scripted
    rung brings its own gate and ignores both.
    """
    if isinstance(policy, BoardBots):
        return BotPolicy(policy.spawn, capacity=policy.capacity)
    if hasattr(policy, "act_rows") and trader is not None:
        return _Traded(policy, trader)
    if hasattr(policy, "act_rows"):
        return PolicyPolicy(policy, _Checkpoint(
            policy, policy.space, policy.players, trade_params(max_offers)))
    return policy


def duel(
    policy: NetworkPolicy,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    network_seats: Sequence[int],
    max_offers: int | None,
    trader: str | None = None,
) -> dict:
    """Play the network against uniform-random opponents and report an interval.

    The floor reading, kept because a run that cannot beat `hexset.bots.
    RandomBot` has nothing else worth measuring. The harness is HexSet's
    (`hexset.bench.versus.compete_batched`), so this function is now the
    lineup and the seating and nothing else.

    Seats are fixed rather than rotated, which is a real limitation,
    documented rather than hidden: whatever seat-correlated effect exists in
    this engine rides straight into the reading uncancelled, and a constant
    caster is what makes that visible instead of averaging it away.

    The games counted are fixed in advance -- indices `0..games-1`, each
    played to completion -- because taking the first `games` episodes a
    driver happens to finish would select for short games, and game length is
    not independent of who is winning.

    `max_offers` is the network's own offer budget, passed in rather than
    defaulted, because a network that trades where the run did not is
    measuring a different game. The random seats never trade at all
    (`hexset.bots.RandomBot`), exactly as the `RandomPolicy` this replaced
    never did.
    """
    seats = frozenset(network_seats)
    cast = tuple(0 if seat in seats else 1 for seat in range(players))
    verdict = compete_batched(
        {
            0: duellist(policy, max_offers, trader),
            1: BotPolicy(lambda board: RandomBot(random.Random(seed))),
        },
        games,
        caster=lambda index: cast,
        players=players,
        seed=seed,
        lanes=lanes,
        action_cap=4000,
        antithetic=False,
        learner=0,
        episodes=True,
    )
    # The three readings the paired verdict has no place for: how many games
    # reached a winner at all, the share of seats that would win by chance,
    # and the network seats' mean scalarised reward (`hexn.rewards`, which is
    # relative points and not the terminal victory points `paired_vp` differs).
    outcomes = [episode.outcome for episode in verdict.episodes]
    points = [
        sum(reward(outcome)[seat] for seat in seats) / len(seats)
        for outcome in outcomes
    ]
    return {
        **verdict.metrics(),
        "decided": sum(1 for outcome in outcomes if outcome.winner is not None),
        "expected_share": len(seats) / players,
        "mean_relative_points": sum(points) / len(points) if points else 0.0,
    }


def versus(
    policy: BatchPolicy,
    reference: BatchPolicy,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    max_offers: int | None,
    antithetic: bool = True,
    trader: str | None = None,
) -> dict:
    """The learner against one reference, two seats each, seats rotating.

    Reports the win rate with its Wilson interval, and paired terminal
    victory points — the finer instrument: learner seats' mean minus
    reference seats' mean within each game, so the board and the dice
    cancel.

    **The harness is HexSet's.** `hexset.bench.versus.compete_batched` owns the
    cohorts, the board pairing, the seat complement, `arena.wilson` and
    `arena.mean_interval`; this function chooses the lineup, the cast and the
    cohort size, and hands back `Verdict.metrics()` unaltered. A duel run here
    and the same duel run through `hexset.arena.compete` therefore cannot
    disagree about what they measured — one pairing law, not two independent
    derivations of it.

    The cast stays `hexset.casting.alternating` rather than `compete`'s own
    adjacent lineup, for continuity with every existing reading under it.
    `swapped(alternating(n))` is `alternating(n, flip=True)`, so the
    antithetic complement is the same lineup this loop has always played.

    The board and the dice cancel; **the seats do not**, unless `antithetic`.
    `alternating` keys the cast to the game index and the board is keyed to the
    index too, so a cohort samples each entrant on one seat-pair per board and
    never the other: the mean seat effect cancels across a cohort, but a
    parity-correlated residual — whichever seat pair an entrant happens to
    draw — does not, and can be large enough to be mistaken for ordinary
    noise by an interval that does not account for it.

    `antithetic` (on by default) runs the paired game with seats swapped too,
    so that residual cancels along with the mean. Pass `antithetic=False`
    only to reproduce an old reading exactly, since its interval does not
    account for the seat residual the same way.
    """
    verdict = compete_batched(
        {0: duellist(policy, max_offers, trader), 1: duellist(reference, max_offers, trader)},
        games,
        caster=alternating(players),
        players=players,
        seed=seed,
        lanes=lanes,
        action_cap=4000,
        antithetic=antithetic,
        learner=0,
    )
    return verdict.metrics()


def rival_rung(directory: str, iteration: int, device: str, board, players: int):
    """The rival run's checkpoint at this exact iteration, or None.

    Matched iteration only — a nearest-checkpoint fallback would quietly turn
    the column into an unmatched comparison, which is the common-opponent trap
    with a second opponent. The caller records a miss rather than hiding it;
    align `--eval-every` with the rival's `--keep-every` (both 25) so evals
    land on checkpoints that exist.
    """
    path = Path(directory) / f"iter-{iteration:05d}.pt"
    if not path.exists():
        return None
    return frozen(str(path), device, board, players)


def lag_rung(directory: Path, iteration: int, device: str, board, players: int):
    """This run's own kept checkpoint at `iteration`, or None.

    Exact iteration only, for the same reason `rival_rung` refuses a nearest
    match: a lag that silently varied would not be a lag. A run launched with
    `--init` keeps `iter-00000.pt`, so its first lag reads the weights it
    started from.
    """
    path = Path(directory) / f"iter-{iteration:05d}.pt"
    if iteration < 0 or not path.exists():
        return None
    return frozen(str(path), device, board, players)


def ladder(
    policy: NetworkPolicy, rungs: dict[str, BatchPolicy], args
) -> dict[str, dict]:
    """The current weights, argmax'd, against every frozen rung.

    Argmax because sampling is the behaviour distribution PPO needs, not the
    policy worth scoring. One *fixed* eval seed rather than one per iteration:
    every eval replays the same boards, so differences between checkpoints are
    paired rather than riding board luck.
    """
    scorer = NetworkPolicy(
        policy.net, policy.space, policy.layout, device=policy.device, greedy=True
    )
    return {
        name: versus(
            scorer,
            reference,
            games=args.eval_games,
            lanes=args.lanes,
            players=args.players,
            seed=args.seed + 10_000,
            max_offers=args.max_offers,
            trader=getattr(args, "trader", None),
        )
        for name, reference in rungs.items()
    }


def summarise(
    episodes: Sequence[Episode],
    iteration: int,
    positions: int,
    collect_seconds: float,
    update_seconds: float,
    games: int,
    # Last and defaulted, so `hexn.exit`'s older positional call still
    # binds: (episodes, iteration, positions, collect, update, games).
    assemble_seconds: float = 0.0,
) -> Progress:
    trades = sum(e.outcome.trades for e in episodes)
    turns = sum(e.outcome.turns for e in episodes)
    return Progress(
        iteration=iteration,
        games=games,
        positions=positions,
        collect_seconds=collect_seconds,
        assemble_seconds=assemble_seconds,
        update_seconds=update_seconds,
        positions_per_second=positions / collect_seconds if collect_seconds else 0.0,
        truncated=sum(e.outcome.truncated for e in episodes) / len(episodes),
        mean_actions=sum(e.outcome.actions for e in episodes) / len(episodes),
        mean_turns=sum(e.outcome.turns for e in episodes) / len(episodes),
        trades_per_turn=trades / turns if turns else 0.0,
    )


def save(path: Path, payload: dict) -> None:
    """Write, fsync, then rename, so a crash mid-save cannot destroy the last
    good one and a renamed file is on the disk, not only in the page cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def prune_recent(directory: Path, keep: int) -> None:
    """Trim the rolling recent-*.pt ring to the newest `keep`.

    The ring exists so a run can rewind to just before a bad update: `latest`
    is overwritten every save and `--keep-every` is too coarse to guarantee a
    checkpoint survives right before one, which is exactly the gap this ring
    closes."""
    ring = sorted(directory.glob("recent-*.pt"))
    for stale in ring[: max(0, len(ring) - keep)]:
        stale.unlink()


def preserve_blowout(
    directory: Path,
    iteration: int,
    net_before: dict,
    optimiser_before: dict,
    batch,
    dump_batch: bool,
) -> None:
    """The KL brake fired: write the exact pre-update weights and, optionally,
    the batch that caused it. The next blowout arrives with its evidence
    attached instead of a guess — nothing upstream warns, so preservation at
    the brake is the only place to catch it."""
    tag = f"blowout-{iteration:05d}"
    save(
        directory / f"{tag}-pre.pt",
        {"iteration": iteration - 1, "net": net_before, "optimiser": optimiser_before},
    )
    if dump_batch:
        save(directory / f"{tag}-batch.pt", {"batch": batch})


def add_head_flags(parser: argparse.ArgumentParser) -> None:
    """The readout shapes, on every trainer that writes a checkpoint.

    Shared rather than repeated because the shape has to reach the checkpoint's
    `args` under exactly these names: `hexn.netbot.load` rebuilds the config
    from that dict, and a run whose shape is not recorded there is a run whose
    checkpoint cannot be loaded back. Both default to the shape every run on
    record used, so adding these changes nothing until one is passed.
    """
    parser.add_argument(
        "--value-head",
        default="linear",
        choices=VALUE_HEADS,
        help="what the value head reads and how deeply. 'linear' is the "
        "default; the other shapes exist to test whether reading more of "
        "the trunk changes what the value head is good at, since being "
        "well-calibrated in expectation and being good at ranking siblings "
        "are different properties and a shape can have one without the "
        "other",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=32,
        help="how many quantiles per seat a --value-head quantile emits; "
        "inert under every other shape. Reaches the checkpoint's args under "
        "this name so `config_from_args` can rebuild the head",
    )
    parser.add_argument(
        "--policy-head",
        default="linear",
        choices=POLICY_HEADS,
        help="per-node-type policy readout depth; holds capacity comparable "
        "across a --value-head ablation",
    )
    parser.add_argument(
        "--detach-value",
        action="store_true",
        help="the value head trains on detached trunk features, so the value "
        "loss cannot reshape what the policy reads; see ModelConfig",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="the learner's forward runs the fused GEMM trunk — same math, "
        "fewer kernels; measured slower on CPU, so this is the GPU update's "
        "opt-in and the collect workers keep the reference path",
    )


def build(args) -> tuple[NetworkPolicy, torch.optim.Optimizer, object]:
    rng = random.Random(args.seed)
    board = random_base_board(rng)
    game = start(board, args.players, rng)
    space = space_for(game)
    graph = static_graph(board.topology)

    torch.manual_seed(args.seed)
    # `getattr` on the head shapes for the same reason `adam_eps` uses it below:
    # this helper is called with hand-built namespaces from the test fixtures,
    # and a namespace that predates a knob should mean "the default shape", not
    # an AttributeError.
    net = HexNet(
        space,
        graph,
        args.players,
        ModelConfig(
            width=args.width,
            rounds=args.rounds,
            value_head=getattr(args, "value_head", "linear"),
            policy_head=getattr(args, "policy_head", "linear"),
            quantiles=int(getattr(args, "quantiles", 32)),
            fast_win=bool(getattr(args, "fast_win", None)),
        ),
    ).to(args.device)
    # The same instance attribute `hexn.exit.__main__` uses: gradient wiring,
    # not architecture, so it lives on the net rather than in ModelConfig and
    # changes nothing about what a checkpoint rebuilds for play.
    net.detach_value = getattr(args, "detach_value", False)
    # Learner-side only: collect workers build their own nets and keep the
    # reference path, which measured faster on CPU.
    net.fused = getattr(args, "fused", False)
    policy = NetworkPolicy(net, space, packing(graph, args.players), device=args.device)
    policy.record_fast = net.config.fast_win
    policy.vp_reward = float(getattr(args, "vp_reward", 0.0))
    # eps 1e-5 rather than torch's 1e-8, which is the standard PPO value and
    # matters more here than usual. `masked_log_softmax` zeroes the gradient at
    # illegal positions, and a position offers ~6 legal actions out of 456, so a
    # given logit's row sees gradient from a small minority of a minibatch's
    # rows. At 1e-8 Adam normalises those few tiny, noisy gradients up to a
    # near-full +/-lr step, so rarely-legal logits random-walk at the full
    # learning rate; 1e-5 damps exactly that without touching the well-sampled
    # directions.
    # `getattr` rather than `args.adam_eps`: this helper is also called with
    # hand-built namespaces from the test fixtures and from `hexn.exit`,
    # whose parsers know nothing about a knob added for PPO. A missing attribute
    # should mean "the standard value", not an AttributeError three frames down.
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        eps=getattr(args, "adam_eps", ADAM_EPS),
    )
    return policy, optimiser, space
