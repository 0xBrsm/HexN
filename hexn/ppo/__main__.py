# SPDX-License-Identifier: GPL-3.0-only
"""The PPO training loop, runnable and resumable.

    python -m hexn.ppo --device cuda --lanes 256 --iterations 400

Everything this joins already exists: `hexn.selfplay` produces the episodes,
`hexn.policy` is the one forward per tick, `hexn.rewards` scalarises the
outcome and `hexn.ppo`'s own math (`__init__.py`) does the arithmetic. This
file is the run — the loop, the log and the checkpoints. The checkpointing,
the duel/versus/ladder evaluation stack and the network builder are shared
with `hexn.league` and `hexn.exit` and live in `hexn.loop`.

## A crash loses at most the games in flight, or one optimiser step

Every game is kept on disk as it finishes (`<checkpoint-dir>/partial/`,
`hexn.durable.Partial`), the update writes its progress after every
`--step-checkpoint-every` optimiser steps (`latest-step.pt`, `hexn.steps`),
the iteration's log row and checkpoint are written before any evaluation
starts, and the evaluation gets a row of its own afterwards. A resumed run
loads the interrupted iteration's games and deals only the rest; killed in
the update, it rebuilds the same batch from those games and carries the
update on from its last step file. It writes `{"resumed_from": N}` to the log
first: an iteration logged but not checkpointed before the crash is logged
again, so a reader keeps the last row per iteration. The games stay on disk
until the iteration's checkpoint is written, and the step file goes then.

## Throughput

The rollout side — the engine step, `encode`, `legal_actions` — runs in
Python per position, while the GPU forward pass is batched, so collection
tends to be the bottleneck rather than the update. The loop reports collect
and update time separately so that split is visible: when collect dominates,
the fix is more collector processes, not a faster forward. Nothing in this
file tries to make the GPU side quicker.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from hexset.board.board import random_base_board
from .. import durable, runtime
from ..collect import (
    ParallelCollector,
    WorkerSpec,
    check_mix,
    duel_pool,
    frozen,
    mix_caster,
    mix_coalitions,
    mix_deal,
    mix_opponents,
    mix_temperatures,
    parse_mix,
    release_memory,
    rung_opponent,
)
from ..loop import (
    add_head_flags,
    build,
    ladder,
    lag_rung,
    preserve_blowout,
    prune_recent,
    rival_rung,
    save,
    summarise,
)
from . import (
    ADAM_EPS,
    PPOConfig,
    assemble,
    attach_prior,
    fast_weight,
    stakes_gauges,
    update,
    win_round,
)
from ..model import graft_fast_win
from ..schedule import AdaptiveLR, current_lr, linear_anneal, set_lr
from ..selfplay import BatchPolicy, Collector, Episode
from ..steps import Steps, by_index, fingerprint, remove as remove_steps
from ..trade import check_trader


# Arbitrary but documented: a single in-process collector saturates around
# 2-3 cores of concurrency, so below this many cores a lone collector is not
# obviously wrong -- a 4-core laptop has nowhere to shard to.
MANY_CORES = 8


def _crippled(device: str, collect_workers: int, cores: int) -> bool:
    """CPU training, or a lone collector idling most of a many-core box.

    Both are the *defaults*, so this only fires for a launch nobody pointed
    anywhere -- the same failure shape as a flag that quietly does nothing
    because nothing checks that it stuck.
    """
    return device == "cpu" or (collect_workers == 0 and cores > MANY_CORES)


def build_parser() -> argparse.ArgumentParser:
    """Every knob a run has, in one place `hexn.run` can introspect.

    Extracted from `main` so a run manifest can be checked against the real
    parameter set rather than a hand-kept copy of it: `hexn.run.manifest`
    walks this parser's actions to know what a frozen config must contain, so
    adding a flag here cannot silently produce manifests that omit it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    # At or below --games-per-iteration: a cohort deals that many games, so
    # lanes above it never fill. See --collect-mode.
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=32)
    parser.add_argument("--action-cap", type=int, default=4000)
    # Caps the trade horizon below the engine's own ceiling. The duel below
    # inherits it, since a policy has to be measured on the horizon it
    # learned on.
    parser.add_argument(
        "--max-offers",
        "--max-trades",
        type=int,
        default=None,
        help="the network's own offer budget a turn (hexn.trade): 0 for the no-trade network, omitted for HexSet's default (unlimited); -1 also reads as unlimited. Opponents bargain as their own bots do",
    )
    parser.add_argument(
        "--trader",
        default=None,
        help="a bot, by lineup name (e.g. 'heximax', registered by a --runtime), that answers every network seat's trades in collection and the ladder (hexn.trade.trader_gate); the network still plays every move. Not combinable with --max-offers",
    )
    # HexSet ships no bots: `--trader`, `--mix` and a rung name one only once
    # the runtime registering it is loaded (`hexn.runtime`).
    runtime.add_argument(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    add_head_flags(parser)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="global grad-norm clip; only matters when it actually binds -- "
        "under Adam a constant rescale of the gradient does not change the "
        "step. Exposed so whether it binds can be checked rather than assumed",
    )
    # The learning rate is a controller, not a constant — see `hexn.schedule`.
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "adaptive"),
        default="constant",
        help="constant keeps --learning-rate; linear anneals it to "
        "--lr-floor of itself by the final iteration; adaptive holds "
        "approx_kl in a band around --target-kl by scaling the rate",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="the KL the adaptive schedule steers toward, read from the update's "
        "final epoch rather than its all-epoch mean",
    )
    parser.add_argument("--lr-band", type=float, default=2.0)
    parser.add_argument("--lr-factor", type=float, default=1.5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--lr-floor", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    # The win head's target: `hexn.rewards.win_loss`, the eventual winner's
    # one-hot. `"win"` is the only legal value; kept as an explicit flag rather
    # than hardcoded so the choice is visible in a run's own config
    # (`PPOConfig.reward`'s own docstring).
    parser.add_argument("--reward", choices=("win",), default="win")
    # The value head's own horizon, separate from the advantage estimator's.
    # 1.0 is the terminal outcome, and the only value the win head accepts:
    # its target is the categorical one-hot winner, which has no continuous
    # bootstrap to mix in (see `hexn.ppo`'s module docstring for the
    # margin-head lambda-return this flag used to drive, before the win head
    # superseded the margin head as the trained objective).
    parser.add_argument("--value-lambda", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    # The auxiliary VP-margin head's loss weight (`hexn.model.Prediction.
    # margin`, trained on `relative_points`). `0.0`, the default, builds the
    # head but excludes it from the loss entirely; it never reaches the
    # advantage or a trade gate at any weight. Exposed for experimentation,
    # not a recommended setting.
    parser.add_argument("--aux-margin-weight", type=float, default=0.0)
    # Pay for fast wins (`PPOConfig.fast_win`): a JSON file holding a list of
    # weights, entry r-1 for a win in round r, or an object with a "weights"
    # list. Builds the net with its fast-win head; `--init` from a checkpoint
    # without one grafts it from the win head (`hexn.model.graft_fast_win`).
    parser.add_argument("--fast-win", default=None, help="JSON round-weight curve for fast wins")
    # Pay this much per victory point a seat gains, when it lands, on top of
    # the win (`PPOConfig.vp_reward`); needs --aux-margin-weight > 0, whose
    # head becomes the critic for the points still to come. 0 disables.
    parser.add_argument("--vp-reward", type=float, default=0.0)
    # Drop positions whose mover's win estimate is below this or above one
    # minus it, and repeat setup positions this many times; see PPOConfig.
    parser.add_argument("--decided-cut", type=float, default=0.0)
    parser.add_argument("--setup-weight", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    # A ceiling on the finished epoch's mean approx_kl, 0 off. Pairs with a
    # raised --epochs as "reuse until the trust region objects"; see PPOConfig.
    parser.add_argument("--kl-break", type=float, default=0.0)
    # Which wires connect the value head to learning: both ("gae", the
    # default), neither ("none", REINFORCE on the zero-sum terminal return),
    # or only the trunk-shaping loss ("aux"). See PPOConfig for the full
    # account.
    parser.add_argument("--critic", choices=("gae", "none", "aux"), default="gae")
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="rows per forward/backward inside a minibatch; gradients accumulate "
        "into the minibatch's one optimiser step. 0 = the whole minibatch",
    )
    parser.add_argument(
        "--amp",
        choices=("", "fp16"),
        default="",
        help="run the update's (and the prior's) forward and backward under "
        "fp16 autocast with a gradient scaler; '' keeps fp32. See PPOConfig",
    )
    parser.add_argument(
        "--batch-on-device",
        action="store_true",
        help="move each assembled batch to --device whole, before the prior "
        "forward, instead of moving every micro-batch's rows as it is used",
    )
    parser.add_argument("--checkpoint-dir", default="runs/ppo")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--step-checkpoint-every",
        type=int,
        default=1,
        help="write the update's progress (latest-step.pt: weights, optimiser, "
        "position, gauges) after every K-th optimiser step and after the last, "
        "so a resumed run continues the interrupted update from its last such "
        "step rather than from the iteration's start (hexn.steps). Not with "
        "--update-workers. 0 disables",
    )
    # `latest.pt` is for resuming and is overwritten; these are for asking,
    # after the fact, when the policy stopped improving -- a question that is
    # unanswerable if only `latest.pt` is kept, since the earlier weights are
    # already gone by the time you would want them.
    parser.add_argument("--keep-every", type=int, default=25, help="0 disables")
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=5,
        help="rolling ring of the last N periodic saves (recent-XXXXX.pt), so "
        "a run can rewind past a bad update without waiting for --keep-every; "
        "0 disables",
    )
    parser.add_argument(
        "--dump-blowout-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when the KL brake fires, also write the offending batch next to "
        "the pre-update weights (the file is the size of the whole batch); "
        "--no-dump-blowout-batch keeps only the weights",
    )
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument(
        "--rival",
        default="",
        help="another run's checkpoint dir; each eval also duels the rival's "
        "checkpoint at the *matched* iteration, so the ladder carries a "
        "recipe-vs-recipe column and not only the common-opponent one. Align "
        "--eval-every with the rival's --keep-every (both 25) or evals land "
        "between its checkpoints and the column records misses",
    )
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="run the ladder before the first update, so the trend has a baseline",
    )
    # A duel against a weak fixed opponent (e.g. random play) saturates once
    # the policy is reliably stronger, and measures nothing further after
    # that. The ladder replaces it: fixed rungs whose strength never moves,
    # so a flat reading means the policy stopped improving rather than the
    # yardstick running out.
    parser.add_argument(
        "--parent",
        default="",
        help="checkpoint path: a ladder rung, and the 'parent' mix opponent",
    )
    parser.add_argument(
        "--search-rung",
        default="",
        help="an extra ladder rung by arena entrant name, e.g. 'heximax' "
        "— the handcrafted baseline, local, and unlike catanatron it plays the "
        "trading game. Costs search time per eval, so it is opt-in; adding it "
        "leaves the parent rung untouched, so the existing trend stays "
        "comparable. There is no automatic handcrafted rung: name one here or "
        "the ladder is the parent alone",
    )
    parser.add_argument(
        "--mix",
        default="",
        help="opponents in the training lanes, e.g. "
        "'heximax=0.15,parent=0.1' — the share of games each plays, on "
        "alternating seat pairs; their seats are never trained on. A name is "
        "'parent' or any arena entrant spec, so a held-out opponent "
        "can be collected against and not merely evaluated on",
    )
    # The single in-process collector only uses 2-3 cores regardless of how
    # many the box has, so on a many-core box it is the bottleneck. Sharding
    # it across processes is the largest speedup available; see `hexn.collect`
    # for the shape and what was rejected.
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=0,
        help="shard collection across N processes with CPU inference; "
        "0 keeps the single in-process collector",
    )
    parser.add_argument(
        "--collect-mode",
        choices=("cohort", "stream"),
        default="cohort",
        help="cohort deals exactly --games-per-iteration games and plays every "
        "one to completion, so the batch is on-policy and unbiased in game "
        "length; stream is the old behaviour, which refills a lane the moment "
        "its game ends and carries the unfinished ones across the weight sync",
    )
    parser.add_argument(
        "--async-collect",
        action="store_true",
        help="prefetch iteration k+1 during the GPU update of k; requires "
        "--collect-workers > 1 and trains on a policy one iteration stale",
    )

    # Compile, precision, minibatch size and CPU thread count do not move the
    # GPU update's per-position cost; data-parallel CPU processes are the
    # lever left. See `hexn.ddp`.
    parser.add_argument(
        "--update-workers",
        type=int,
        default=0,
        help="shard the PPO update across N CPU processes with a central Adam "
        "step; 0 keeps the single-device update",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init",
        default="",
        help="checkpoint whose weights the run starts from, at iteration 0 with "
        "a fresh optimiser -- any file carrying a 'net' state dict of this run's "
        "shape, such as a supervised checkpoint. Only the first launch reads it: "
        "with --resume too, a directory that already holds latest.pt resumes "
        "from that instead, so the one frozen config serves the launch and "
        "every restart",
    )
    parser.add_argument(
        "--prior-kl",
        type=float,
        default=0.0,
        help="weight on KL(policy || parent) over every legal action, added to "
        "the loss (PPOConfig.prior_kl); needs --parent. 0 disables",
    )
    parser.add_argument(
        "--lag-rung",
        type=int,
        default=0,
        help="an extra ladder rung: this run's own kept checkpoint from N "
        "iterations before each eval (a run launched with --init keeps its "
        "iteration 0). Rising against fixed rungs while falling against this "
        "one is the policy cycling rather than improving. Align N with "
        "--keep-every; a missing checkpoint is logged as a miss. 0 disables",
    )
    return parser


def fast_win_curve(path: str | None) -> tuple[float, ...]:
    """`--fast-win`'s file as `PPOConfig.fast_win`: a JSON list of weights,
    or an object carrying one under "weights". No path, no curve."""
    if not path:
        return ()
    import json

    data = json.loads(Path(path).read_text())
    weights = data["weights"] if isinstance(data, dict) else data
    if not weights:
        raise SystemExit(f"--fast-win {path} holds no weights")
    return tuple(float(w) for w in weights)


def fast_win_gauges(episodes: Sequence[Episode], curve: Sequence[float], players: int) -> dict:
    """The speed columns a fast-win run logs: the median round games were
    won in, the median round of the learner's own wins, and the mean payoff
    per won game."""
    import statistics

    won = [e for e in episodes if e.outcome.winner is not None]
    mine = [e for e in won if e.trajectories[e.outcome.winner]]
    return {
        "win_round_median": statistics.median(win_round(e.outcome, players) for e in won) if won else None,
        "learner_win_round_median": (
            statistics.median(win_round(e.outcome, players) for e in mine) if mine else None
        ),
        "fast_payoff_mean": (
            sum(fast_weight(curve, e.outcome, players) for e in won) / len(won) if won else None
        ),
    }


def coalition_gauges(episodes: Sequence[Episode], plan_of) -> dict:
    """What a `coalition(...)` mix dealt and how the learner fared in it:
    the share of games with a coalition, the share of those whose target
    was the learner's seat, and the learner's win rate in each kind of
    game -- targeted (its target seat won), neutral (a learner seat won at
    a table whose coalition was on someone else) and free (no coalition).
    `plan_of` is `hexn.collect.mix_coalitions`' law, re-drawn here from the
    game index: a plan is a pure function of the index, so nothing rides on
    the episode."""
    counts = {"targeted": [0, 0], "neutral": [0, 0], "free": [0, 0]}
    for episode in episodes:
        plan = plan_of(episode.index)
        winner = episode.outcome.winner
        if plan is None:
            kind, won = "free", winner is not None and bool(episode.trajectories[winner])
        elif episode.cast[plan.target] == 0:
            kind, won = "targeted", winner == plan.target
        else:
            kind, won = "neutral", winner is not None and bool(episode.trajectories[winner])
        counts[kind][0] += won
        counts[kind][1] += 1
    played = len(episodes)
    with_coalition = counts["targeted"][1] + counts["neutral"][1]
    return {
        "coalition_share": with_coalition / played if played else None,
        "coalition_targeted_share": counts["targeted"][1] / with_coalition if with_coalition else None,
        **{
            f"coalition_{kind}_win_rate": (wins / games if games else None)
            for kind, (wins, games) in counts.items()
        },
    }


def learner_final_vp(episodes: Sequence[Episode]) -> float | None:
    """The mean final victory points of the seats the learner played: the
    number a per-VP reward moves, read beside the win rate so points bought
    without wins show up."""
    held = [e.outcome.points[s] for e in episodes for s, t in enumerate(e.trajectories) if t]
    return sum(held) / len(held) if held else None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the directory named on the command line, and take nothing else.

    A run is `runs/<...>/` holding `run.json` and a frozen `config/`, created by
    `python -m hexn.run.init`. Flags are not accepted here: the manifest is the
    only input, so a recorded result is reproducible from the repository rather
    than from whichever script in gitignored `tmp/` happened to survive. See
    `hexn.run.manifest` for the three losses that motivated this.
    """
    from ..run import load
    from ..run.manifest import MODULES

    tokens = list(sys.argv[1:] if argv is None else argv)
    if len(tokens) != 1 or tokens[0].startswith("-"):
        raise SystemExit(
            "usage: python -m hexn.ppo <run-directory>\n"
            "  create one with: python -m hexn.run.init --mode ppo --name NAME -- <flags>\n"
            "  the flags this used to accept are frozen into the run's config/ instead."
        )
    manifest = load(tokens[0])
    if manifest.mode != "ppo":
        raise SystemExit(
            f"{tokens[0]} is a {manifest.mode} run; launch it with "
            f"python -m {MODULES[manifest.mode]}"
        )
    args = manifest.namespace()
    # Cross-parameter validation below still reports through the parser, so a
    # bad combination frozen into a manifest fails with the same sentence it
    # would have as a flag.
    parser = build_parser()
    runtime.load(getattr(args, "runtime", None))
    try:
        check_trader(getattr(args, "trader", None), args.max_offers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.async_collect and args.collect_workers <= 1:
        parser.error("--async-collect requires --collect-workers > 1")
    if args.async_collect and args.update_workers > 1:
        parser.error("--async-collect is for the GPU update, not CPU update workers")
    if args.kl_break > 0 and args.update_workers > 1:
        parser.error("--kl-break is implemented in hexn.ppo.update only; the "
                     "sharded crew runs its own epoch loop and would ignore it")
    if args.prior_kl > 0 and not args.parent:
        parser.error("--prior-kl is a divergence to --parent; name one")
    if args.prior_kl > 0 and args.update_workers > 1:
        parser.error("--prior-kl is implemented in hexn.ppo.update only; the "
                     "sharded crew runs its own epoch loop and would ignore it")
    if args.micro_batch and args.update_workers > 1:
        parser.error("--micro-batch is implemented in hexn.ppo.update only; the "
                     "sharded crew runs its own epoch loop and would ignore it")
    if args.micro_batch < 0:
        parser.error("--micro-batch cannot be negative")
    # A manifest frozen before the flag existed has none: no step file.
    step_every = getattr(args, "step_checkpoint_every", 0)
    if step_every < 0:
        parser.error("--step-checkpoint-every cannot be negative")
    if step_every and args.update_workers > 1:
        # The sharded crew runs its own epoch loop.
        print("--update-workers: no step checkpoints, the sharded update keeps none",
              file=sys.stderr)
        step_every = 0
    if step_every and args.checkpoint_every > 1:
        print(
            f"WARNING: --checkpoint-every {args.checkpoint_every}: a step "
            "checkpoint resumes only the iteration right after latest.pt, so a "
            "kill in any other iteration still goes back to latest.pt",
            file=sys.stderr,
        )
    if args.kl_break > 0 and args.lr_schedule == "adaptive":
        parser.error("--kl-break truncates the epochs the adaptive controller "
                     "reads its gauge from; run one governor at a time")
    if args.collect_mode == "cohort":
        if args.lanes > args.games_per_iteration:
            parser.error(
                f"--lanes {args.lanes} exceeds --games-per-iteration "
                f"{args.games_per_iteration}: a cohort deals that many games and "
                "no more, so the surplus lanes would sit empty. Lanes are the "
                "concurrency, not the cohort — lower them, or raise the cohort"
            )
        if args.async_collect:
            parser.error(
                "--async-collect dispatches the next cohort before the update "
                "lands, which is exactly the staleness --collect-mode cohort "
                "exists to remove; pass --collect-mode stream to accept it"
            )

    # Printed unconditionally rather than only on the crippled path: a launch
    # that overrides every one of these still benefits from the confirmation,
    # and a launch that does not gets it without having to ask.
    cores = os.cpu_count() or 1
    print(
        f"device={args.device} collect-workers={args.collect_workers} "
        f"update-workers={args.update_workers} ({cores} cores on this box)",
        file=sys.stderr,
    )
    if _crippled(args.device, args.collect_workers, cores):
        print(
            "WARNING: this is the crippled default -- cpu training and/or a "
            "single in-process collector, which uses only 2-3 cores "
            "regardless of how many are available. Pass --device cuda and "
            "--collect-workers <N> unless this is deliberate (e.g. a smoke "
            "test).",
            file=sys.stderr,
        )

    # Deliberately absent: --gamma. See `hexn.rewards`.
    config = PPOConfig(
        reward=args.reward,
        lam=args.lam,
        value_lam=args.value_lambda,
        clip=args.clip,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy,
        aux_margin_weight=args.aux_margin_weight,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        minibatch=args.minibatch,
        micro_batch=args.micro_batch,
        learning_rate=args.learning_rate,
        critic=args.critic,
        kl_break=args.kl_break,
        prior_kl=args.prior_kl,
        fast_win=fast_win_curve(getattr(args, "fast_win", None)),
        vp_reward=getattr(args, "vp_reward", 0.0),
        decided_cut=getattr(args, "decided_cut", 0.0),
        setup_weight=getattr(args, "setup_weight", 1),
        amp=getattr(args, "amp", ""),
    )
    if config.amp and args.update_workers > 1:
        raise SystemExit("--amp is not wired into the sharded update (--update-workers)")
    if config.fast_win:
        if any(duel_pool(name) is not None for name, _ in parse_mix(args.mix)):
            raise SystemExit("--fast-win counts rounds over four live seats; a duel(...) mix is not supported")
        if args.update_workers > 1:
            raise SystemExit("--fast-win is not wired into the sharded update (--update-workers)")

    policy, optimiser, _ = build(args)
    # One scaler for the whole run, so the scale it has found carries from
    # update to update (and across a resume, through the checkpoint).
    scaler = torch.amp.GradScaler(torch.device(args.device).type) if config.amp else None
    directory = Path(args.checkpoint_dir)
    latest = directory / "latest.pt"
    log = directory / "log.jsonl"
    collecting = directory / "partial"

    start_iteration = 0
    first_game = 0
    if args.resume and not latest.exists() and not args.init:
        # `--resume` with nothing to resume used to fall through and start from
        # iteration 0 on freshly initialised weights, silently. On a long GPU
        # run that is hours of compute spent discarding progress, and the log
        # gives no sign of it — it simply begins at 0 and looks healthy. A
        # typo'd `--checkpoint-dir`, or a new directory nobody remembered to
        # seed, is all it takes. Same failure shape as the learning rate that
        # `--resume` used to discard: a flag that quietly does nothing.
        raise SystemExit(
            f"--resume was given but {latest} does not exist; "
            "seed the directory with the checkpoint to continue from, "
            "or drop --resume to start a fresh run"
        )
    if not args.resume and latest.exists():
        # The mirror image: a fresh start here would overwrite that run's
        # checkpoint at its first save and append a second iteration 0 to its
        # log -- the completed work this directory holds, discarded.
        raise SystemExit(
            f"{latest} already holds a run; resume it with --resume, or start "
            "the fresh run in an empty --checkpoint-dir"
        )
    if args.init and not (args.resume and latest.exists()):
        # Weights only. The optimiser, the iteration and the game counter are
        # this run's own and start where a fresh run's do; a supervised
        # checkpoint's AdamW moments were fitted to a different loss.
        # `load_state_dict` is strict, so a checkpoint of another shape --
        # the rounds a supervised trunk was trained with, say -- fails here,
        # loudly, rather than training a half-initialised net.
        seed_state = torch.load(args.init, map_location=args.device, weights_only=False)
        grafted = config.fast_win and not any(
            key.startswith("fast_win.") for key in seed_state["net"]
        )
        if grafted:
            # A checkpoint from before the fast-win head: every other key
            # must still match, and the new head starts from the win head.
            missing, unexpected = policy.net.load_state_dict(seed_state["net"], strict=False)
            if unexpected or any(not key.startswith("fast_win.") for key in missing):
                raise SystemExit(
                    f"--init {args.init} does not match this net: missing {missing}, "
                    f"unexpected {unexpected}"
                )
            graft_fast_win(policy.net)
            print("grafted the fast-win head from the win head", file=sys.stderr)
        else:
            policy.net.load_state_dict(seed_state["net"])
        print(f"initialised from {args.init}", file=sys.stderr)
        # Kept as this run's own iteration 0: the same weights, but recording
        # this run's arguments -- its offer budget among them -- where the init
        # file records whatever its own trainer did. A comparison against the
        # starting point is then between two files that bargain alike, and the
        # lag rung's iteration 0 is an ordinary kept checkpoint.
        if args.keep_every:
            save(
                directory / "iter-00000.pt",
                {
                    "iteration": 0,
                    "games_started": 0,
                    "net": policy.net.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "args": vars(args),
                    "config": asdict(config),
                },
            )
    elif args.resume:
        state = torch.load(latest, map_location=args.device, weights_only=False)
        policy.net.load_state_dict(state["net"])
        optimiser.load_state_dict(state["optimiser"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        # `Optimizer.load_state_dict` rebuilds `param_groups` from the *saved*
        # groups, keeping only `params` from the live ones — so every
        # hyperparameter, `lr` and `eps` included, comes back from the
        # checkpoint and the command line is silently discarded. A run can
        # record the CLI's learning rate in both its `args` and `config`
        # blobs while Adam quietly steps at whatever rate was baked into the
        # checkpoint it resumed from, with every gauge matching the previous
        # run exactly because the configuration *is* the previous run's.
        # Re-assert the CLI values, and log the live rate every iteration so
        # a knob that does not move is visible in the run rather than in a
        # post-mortem.
        set_lr(optimiser, args.learning_rate)
        for group in optimiser.param_groups:
            group["eps"] = args.adam_eps
        start_iteration = state["iteration"]
        first_game = state["games_started"]
        # `map_location` moved everything in the checkpoint to the training
        # device, but the RNG state must be a *CPU* ByteTensor — on a cuda
        # resume this line is the difference between restoring and crashing.
        torch.set_rng_state(state["torch_rng"].cpu())
        print(
            f"resumed at iteration {start_iteration}, game {first_game}",
            file=sys.stderr,
        )
    # Past every game an interrupted collection already dealt, so nothing the
    # resumed run deals repeats one (`durable.resume_base`). A fresh start
    # counts too: a crash before the first checkpoint leaves iteration 0's
    # games, played by the same seeded weights, and they are kept.
    first_game = durable.resume_base(
        first_game,
        *(
            held.indices()
            for iteration, held in durable.partials(collecting).items()
            if iteration >= start_iteration
        ),
    )
    durable.prune(collecting, start_iteration - 1)

    # The same board `build` derived from the seed, for loading the parent:
    # base boards all share one topology, so any of them names the right shapes.
    board = random_base_board(random.Random(args.seed))
    parent = (
        frozen(args.parent, args.device, board, args.players) if args.parent else None
    )
    if parent is not None and getattr(args, "fused", False):
        # The parent's one job on the learner's device is the prior forward;
        # it runs the same fused trunk as the learner's update.
        parent.net.fused = True

    mix = parse_mix(args.mix)
    check_mix(mix, have_parent=parent is not None)

    rungs: dict[str, BatchPolicy] = {}
    if parent is not None:
        rungs["parent"] = parent
    if args.search_rung:
        rungs[args.search_rung] = rung_opponent(
            args.search_rung,
            seed=args.seed + 79,
            lanes=args.lanes,
            board=board,
            players=args.players,
            device=args.device,
        )

    if args.collect_workers > 1:
        shard = max(1, -(-args.lanes // args.collect_workers))
        collector = ParallelCollector(
            [
                WorkerSpec(
                    seed=args.seed,
                    players=args.players,
                    lanes=shard,
                    action_cap=args.action_cap,
                    max_offers=args.max_offers,
                    trader=getattr(args, "trader", None),
                    runtime=tuple(getattr(args, "runtime", None) or ()),
                    first_game=first_game + worker,
                    stride=args.collect_workers,
                    width=args.width,
                    rounds=args.rounds,
                    torch_seed=args.seed + 100_000 + worker,
                    value_head=args.value_head,
                    policy_head=args.policy_head,
                    quantiles=args.quantiles,
                    fast_win=bool(config.fast_win),
                    vp_reward=config.vp_reward,
                    mix=tuple(mix),
                    parent=args.parent,
                    cohort=args.collect_mode == "cohort",
                )
                for worker in range(args.collect_workers)
            ],
            heartbeat=directory / "heartbeat.json",
        )
    else:
        # Note what the comprehension this replaced did with a name that was
        # neither: it fell through to `parent`, so a third opponent would have
        # silently been the parent checkpoint. Only the validation above kept
        # that unreachable, and the validation now admits every entrant spec.
        opponents: list[BatchPolicy] = mix_opponents(
            mix,
            seed=args.seed,
            lanes=args.lanes,
            parent=(lambda: parent) if parent is not None else None,
            board=board,
            players=args.players,
            device=args.device,
        )
        collector = Collector(
            policy,
            lanes=args.lanes,
            fill=args.collect_mode != "cohort",
            players=args.players,
            seed=args.seed,
            action_cap=args.action_cap,
            first_game=first_game,
            max_offers=args.max_offers,
            trader=getattr(args, "trader", None),
            opponents=opponents,
            caster=mix_caster(mix, args.players, args.seed) if mix else None,
            temperatures=(
                mix_temperatures(mix, args.players, args.seed) if mix else None
            ),
            game_law=mix_deal(mix, args.players, args.seed) if mix else None,
            coalition=mix_coalitions(mix, args.players, args.seed) if mix else None,
        )

    crew = None
    if args.update_workers > 1:
        from ..ddp import UpdateCrew, UpdateSpec

        crew = UpdateCrew(
            [
                UpdateSpec(
                    seed=args.seed,
                    players=args.players,
                    width=args.width,
                    rounds=args.rounds,
                    value_head=args.value_head,
                    policy_head=args.policy_head,
                    quantiles=args.quantiles,
                )
                for _ in range(args.update_workers)
            ]
        )

    controller = AdaptiveLR(
        target_kl=args.target_kl,
        band=args.lr_band,
        factor=args.lr_factor,
        min_lr=args.lr_min,
        max_lr=args.lr_max,
    )

    directory.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    if args.resume or log.exists():
        # Rows past this marker supersede any earlier row for the same
        # iteration: the checkpoint resumed from predates them.
        durable.append_line(log, {"resumed_from": start_iteration})

    if args.eval_at_start:
        # A resumed run's first ladder reading doubles as a regression control:
        # against its own parent the starting weights must duel to a dead heat,
        # so a first reading away from 50% flags the harness, not the policy.
        baseline = ladder(policy, rungs, args)
        line = durable.append_line(
            log, {"iteration": start_iteration - 1, "ladder": baseline}
        )
        print(line, flush=True)

    # `ParallelCollector.collect` already forwards to each worker's `cohort`
    # when the specs ask for it, so only the in-process collector needs steering.
    if args.collect_mode == "cohort" and isinstance(collector, Collector):
        draw = collector.cohort
    else:
        draw = collector.collect

    prefetched: list[Episode] | None = None
    prefetched_seconds = 0.0
    if args.async_collect and start_iteration < args.iterations:
        # Prime the pipeline. Later cohorts are dispatched immediately before
        # the preceding update and therefore act with the pre-update policy.
        # That one-iteration staleness is why this path is explicit opt-in.
        started = time.perf_counter()
        collector.sync(policy.net)
        prefetched = by_index(draw(
            args.games_per_iteration,
            partial=durable.partial(collecting, start_iteration),
        ))
        prefetched_seconds = time.perf_counter() - started

    coalition_plans = mix_coalitions(mix, args.players, args.seed) if mix else None
    for iteration in range(start_iteration, args.iterations):
        if args.async_collect:
            assert prefetched is not None
            episodes, prefetched = prefetched, None
            collected = prefetched_seconds
        else:
            started = time.perf_counter()
            if isinstance(collector, ParallelCollector):
                # The workers act with last iteration's weights until this lands,
                # so the sync is inside the loop and before the collect, always.
                collector.sync(policy.net)
            episodes = by_index(draw(
                args.games_per_iteration,
                partial=durable.partial(collecting, iteration),
            ))
            collected = time.perf_counter() - started
        played = collector.games

        started = time.perf_counter()
        batch = assemble(episodes, policy.layout, config)
        # The per-game gauges that read trajectories, taken now; then every
        # transition goes. Their observations are views into the collectors'
        # buffers, a second copy of the batch that nothing after this reads
        # (each game is already on disk in the partial).
        gauges = {
            **(fast_win_gauges(episodes, config.fast_win, args.players) if config.fast_win else {}),
            **({"learner_final_vp_mean": learner_final_vp(episodes)} if config.vp_reward else {}),
            **(coalition_gauges(episodes, coalition_plans) if coalition_plans is not None else {}),
            **(
                stakes_gauges(episodes, config)
                if config.decided_cut or config.setup_weight != 1
                else {}
            ),
        }
        episodes = [replace(episode, trajectories=()) for episode in episodes]
        if getattr(args, "batch_on_device", False):
            # Rebinding drops the host copy: on a shared-memory device the
            # two would otherwise each hold the whole batch.
            batch = batch.to(policy.device)
        if config.prior_kl > 0:
            batch = attach_prior(
                batch,
                parent,
                policy.layout,
                **(
                    {"chunk": config.micro_batch or config.minibatch, "amp": config.amp}
                    if config.amp
                    else {}
                ),
            )
        assembled = time.perf_counter() - started
        next_collect_started: float | None = None
        if args.async_collect and iteration + 1 < args.iterations:
            # Ship the current policy before `update` mutates it, then let the
            # CPU workers play while the GPU owns the learner's critical path.
            next_collect_started = time.perf_counter()
            collector.sync(policy.net)
            collector.start_collect(
                args.games_per_iteration,
                partial=durable.partial(collecting, iteration + 1),
            )

        # ~3 MB a clone: cheap insurance. The brake's third firing lost its
        # evidence because nothing held the pre-update state.
        net_before = {k: v.detach().cpu().clone() for k, v in policy.net.state_dict().items()}
        optimiser_before = {
            "state": {
                k: {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in optimiser.state_dict()["state"].items()
            },
            "param_groups": optimiser.state_dict()["param_groups"],
        }

        # Opened after the pre-update clones and the next collection's weight
        # sync, both of which want the iteration's starting weights: a step
        # file of this iteration's update replaces them with its own.
        stepper = (
            Steps.open(directory, iteration, fingerprint(batch), step_every, policy, optimiser)
            if step_every
            else None
        )
        started = time.perf_counter()
        if crew is not None:
            stats = crew.update(policy, optimiser, batch, config)
        else:
            stats = update(policy, optimiser, batch, config, steps=stepper, scaler=scaler)
        updated = time.perf_counter() - started

        if config.kl_break and stats.epochs_taken < config.epochs:
            preserve_blowout(
                directory,
                iteration + 1,
                net_before,
                optimiser_before,
                batch,
                args.dump_blowout_batch,
            )

        if next_collect_started is not None:
            prefetched = by_index(collector.finish_collect())
            prefetched_seconds = time.perf_counter() - next_collect_started
        else:
            prefetched = None

        progress = summarise(
            episodes,
            iteration,
            len(batch),
            collected,
            updated,
            played,
            assemble_seconds=assembled,
        )
        record = {
            **asdict(progress),
            **asdict(stats),
            **gauges,
            # A column, not just a config field: `PPOConfig.learning_rate` can
            # read back a correct-looking value while nothing downstream
            # actually applies it (see the resume path above), so the
            # quantity a run is steered by has to appear in its own log to be
            # checked against what actually happened.
            "collect_mode": args.collect_mode,
            # Same rule, applied ahead of the blocks that sweep them rather than
            # after: `lam` is the only horizon control there is under gamma 1,
            # and `minibatch` sets how many steps an iteration takes.
            "lam": config.lam,
            "value_lam": config.value_lam,
            "minibatch": config.minibatch,
            # Same rule again: which critic wiring was in effect appears in its
            # own log, so a row can never be attributed to the wrong wiring.
            "critic": config.critic,
            **(stepper.log() if stepper is not None else {}),
            "elapsed": time.perf_counter() - began,
        }

        # The rate for the *next* update, chosen from the gauge this one just
        # measured. Read off the final epoch rather than `approx_kl`: the
        # all-epoch mean includes epoch 1, where nothing has stepped yet, so
        # it understates the finished update's actual divergence — steering a
        # controller by the mean would target the wrong number. Applied after
        # the record is built, so a row always reports the rate its own
        # update used (`stats.lr`), never the successor's.
        if args.lr_schedule == "adaptive":
            gauge = stats.approx_kl_last_epoch
            moved = controller.next_lr(current_lr(optimiser), gauge)
            if controller.deaf(moved, gauge):
                # Saturated at a clamp and still out of band. Distinct from a
                # converged controller, which also holds the rate still.
                record["lr_deaf"] = True
            set_lr(optimiser, moved)
        elif args.lr_schedule == "linear":
            set_lr(
                optimiser,
                linear_anneal(
                    args.learning_rate, iteration + 1, args.iterations, args.lr_floor
                ),
            )

        # The iteration's row and checkpoint first, the evaluation after: an
        # evaluation that crashes must not take the iteration with it.
        print(durable.append_line(log, record), flush=True)

        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            state = {
                "iteration": iteration + 1,
                "games_started": collector.games_started(),
                "net": policy.net.state_dict(),
                "optimiser": optimiser.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "args": vars(args),
                "config": asdict(config),
                **({"scaler": scaler.state_dict()} if scaler is not None else {}),
            }
            save(latest, state)
            # The step file was this iteration's update; latest.pt holds it now.
            remove_steps(directory)
            # The ring is the last N checkpoints whatever --keep-every keeps.
            if args.keep_recent:
                save(directory / f"recent-{iteration + 1:05d}.pt", state)
                prune_recent(directory, args.keep_recent)
            if args.keep_every and (iteration + 1) % args.keep_every == 0:
                save(directory / f"iter-{iteration + 1:05d}.pt", state)
            # Only now is this iteration's collection held elsewhere. A
            # prefetched next one stays: it is the next iteration's.
            durable.prune(collecting, iteration)

        if args.eval_every and (iteration + 1) % args.eval_every == 0:
            evaluation: dict = {"iteration": iteration}
            eval_rungs = dict(rungs)
            if args.rival:
                opponent = rival_rung(
                    args.rival, iteration + 1, args.device, board, args.players
                )
                if opponent is None:
                    # A miss is data, not silence: the alignment note on the
                    # flag is only checkable if misses appear in the log.
                    evaluation["rival_checkpoint_missing"] = iteration + 1
                else:
                    eval_rungs["rival"] = opponent
            if args.lag_rung:
                lagged = lag_rung(
                    directory,
                    iteration + 1 - args.lag_rung,
                    args.device,
                    board,
                    args.players,
                )
                if lagged is None:
                    evaluation["lag_checkpoint_missing"] = iteration + 1 - args.lag_rung
                else:
                    eval_rungs["lag"] = lagged
            evaluation["ladder"] = ladder(policy, eval_rungs, args)
            print(durable.append_line(log, evaluation), flush=True)

        # Nothing of this iteration is read past here. Let the batch (and
        # what is left of the episodes) go now, or the next collection runs
        # with this whole batch still held on the host.
        batch = episodes = net_before = optimiser_before = None
        release_memory()
        if torch.device(args.device).type == "cuda":
            # The caching allocator keeps a freed batch's blocks reserved; on
            # a shared-memory GPU that is host RAM the next collection needs.
            torch.cuda.empty_cache()

    if isinstance(collector, ParallelCollector):
        collector.close()
    if crew is not None:
        crew.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
