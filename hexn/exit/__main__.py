# SPDX-License-Identifier: GPL-3.0-only
"""The expert-iteration loop: play with the search, train toward what it decided.

    python -m hexn.exit --device cuda --init runs/ppo/latest.pt \
        --simulations 256 --lanes 16 --iterations 200

The pieces already exist — `hexn.expert.SearchPolicy` produces the searched
games, `hexn.exit`'s own math (`__init__.py`) consumes their visit targets,
`hexn.selfplay.Collector` runs the lanes. This file is the run.

## It is a separate file from `hexn.ppo`, and reuses its harness

PPO and distillation share the checkpoint format, the duel, the progress record
and the network builder (`hexn.loop`), so those are imported rather than
copied. What differs is the two lines in the middle: who chooses the action,
and what the update optimises. Threading that through `hexn.ppo` as a mode
would put a branch in that trainer's loop, to save about forty lines.

## The search is over the *current* student, not a frozen teacher

`LeafEvaluator` holds the same `NetworkPolicy` the optimiser is stepping, so
each iteration searches over what the last one learned. That is the whole
mechanism: search improves on the policy it searches over, the policy is trained
toward the improvement, and the next search starts from a better base. A frozen
teacher would instead converge to a fixed target and stop.

`--init` is therefore a starting point rather than a teacher. Starting from the
PPO checkpoint matters because the improvement is measured relative to what the
search has to work with: at random initialisation the priors are noise and 256
simulations of noise is a slow uniform search.

## The trade switch is set in two places and they must agree

`Collector` plays its lanes under `--max-trades`; `Search` roots its tree under
its own. One flag feeds both, which is why it is passed to `Search` explicitly
here.

## Scaling collection

Collection cost scales with process count, which this file does not manage
on its own; run more of them for more throughput.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from . import DistillConfig, assemble, refresh, update
from ..expert import SearchPolicy
from hexset.mcts import Search
from ..netbot import LeafEvaluator
from ..collect import ParallelCollector, WorkerSpec
from ..reanalyse import reanalyse
from ..selfplay import Collector
from ..store import EpisodeStore
from ..loop import add_head_flags, build, duel, save, summarise


def build_parser() -> argparse.ArgumentParser:
    """Distillation's knobs, introspectable by `hexn.run` -- see
    `train.build_parser` for why the parser is a function rather than inline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="the trade switch every seat plays under: 0 for the no-trade referent, omitted for the engine's own default of one broadcast round a turn (`Game.trade_mode='round'`), -1 to leave a turn's rounds uncapped (`hexset.trading`)",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--wave", type=int, default=16)
    parser.add_argument("--exploration", type=float, default=1.25)
    # Left at "relative", unchanged by contract 6's win head. `stance` is how
    # `hexset.mcts.Search` reduces a node's per-seat backed-up vector to the
    # scalar PUCT maximises, and it does not care whether that vector is a
    # margin or (as it is now, off `hexset.clients.netbot.LeafEvaluator`) a win
    # probability -- see `hexn.exit`'s module docstring for the full
    # argument. `hexset.bots.STANCES` also carries `"win"`, built to *convert*
    # a margin into a probability at a fitted temperature, which is now the
    # wrong direction to apply on top of a leaf that is already one; worse,
    # `hexset.mcts.Search` (a read-only engine here) accepts `"win"` without
    # actually implementing it, so passing it silently zeroes PUCT's value
    # term rather than erroring. Not changed to `"win"` for that reason.
    parser.add_argument("--stance", default="relative")
    # Off by default. `benchmarks.sibling` is how a candidate root-noise
    # configuration should be checked before being turned on.
    parser.add_argument("--root-noise", type=float, default=0.0)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    # Two temperatures, because they do different jobs. The first shapes which
    # positions end up in the corpus and wants breadth; the second shapes the
    # target at a position already collected. `Target` keeps raw counts so they
    # can be set independently.
    # 0 is the terminal outcome. Higher bootstraps off the search's own backed-up
    # value that many of the seat's decisions later; see `distill._value_targets`.
    parser.add_argument("--value-horizon", type=int, default=0)
    parser.add_argument("--play-temperature", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--collect-workers", type=int, default=0,
                        help="shard searched collection across processes; each "
                             "runs its own search on CPU and the learner keeps "
                             "the GPU. 0 or 1 keeps the single in-process "
                             "collector, which is ~10x slower")
    parser.add_argument("--contested-only", action="store_true",
                        help="train the policy only where the search overruled "
                             "it; the value head still sees every position")
    parser.add_argument("--hard-target", action="store_true",
                        help="distil the search's argmax rather than its visit "
                             "distribution, which is what play actually uses")

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    add_head_flags(parser)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument(
        "--replay-positions",
        type=int,
        default=300_000,
        help="positions kept in the disk-backed replay store under "
        "<checkpoint-dir>/replay/ (hexn.store.EpisodeStore), evicting whole "
        "iterations oldest-first once the newer ones already cover the budget; "
        "the freshest iteration is never evicted on its own arrival. Replaces "
        "--buffer-iterations: sized in positions, which stays meaningful across "
        "--games-per-iteration, where a fixed iteration count did not.",
    )
    parser.add_argument(
        "--reanalyse-samples",
        type=int,
        default=3_656,
        help="stored positions re-searched over the CURRENT net each "
        "iteration (hexn.reanalyse), replacing their stale visit target "
        "before assembly -- staleness is worst for the oldest rows in the "
        "store, collected under the earliest, weakest net still retained. "
        "0 disables; there is no measured optimal fraction yet, so the "
        "default is a stated starting point rather than a tuned constant. "
        "Incompatible with --corpus, which never grows a store to reanalyse",
    )
    parser.add_argument(
        "--refresh-prior",
        action="store_true",
        help="recompute the contested filter and the anchor against the live "
        "policy instead of the prior recorded at collection time. What makes a "
        "buffer safe: the visit counts are expensive to produce and stay "
        "valid, the prior is one forward and does not. Also lets a row cross "
        "into the contested set as the student drifts",
    )
    parser.add_argument(
        "--pack-contested",
        action="store_true",
        help="give the policy term its own dense minibatches, rather than "
        "riding the value term's sampling of the contested rows",
    )
    parser.add_argument(
        "--anchor",
        type=float,
        default=0.0,
        help="weight on the trust region: a cross-entropy toward the recorded "
        "prior on the rows --contested-only zeroed. 0 disables the anchor. "
        "Only meaningful with --contested-only",
    )
    parser.add_argument(
        "--contested-margin",
        type=float,
        default=0.0,
        help="visit share the search's pick must lead the policy's by before "
        "the row is trained on. 0 disables the margin. Only meaningful with "
        "--contested-only",
    )
    parser.add_argument(
        "--stake-scale",
        type=float,
        default=0.0,
        help="Q-gap at which a contested row earns full weight, in reward units "
        "-- 0.01 is 0.1 VP. 0 is a plain 0/1 filter. Above it a row's weight "
        "is min(gap / scale, 1) and a negative gap drops out. Where "
        "--contested-margin gates on how sure the counts were, this gates on "
        "what the correction is worth. Only meaningful with --contested-only",
    )
    # `--detach-value` is NOT declared here: `add_head_flags` above already
    # provides it, and declaring it twice makes the parser unbuildable --
    # `hexn.exit` would raise `ArgumentError` on any invocation, including
    # --help.
    parser.add_argument(
        "--corpus",
        default=None,
        help="collect once into this file, then replay it every iteration. "
        "Written if absent, loaded if present, so runs that differ only in "
        "the loss share one collection: every run after the first costs only "
        "its updates. The teacher is frozen at whichever net wrote the file, "
        "which makes this off-policy rather than a reproduction of an "
        "on-policy run. Compare against the same file or the comparison "
        "does not hold",
    )
    parser.add_argument(
        "--corpus-games",
        type=int,
        default=0,
        help="games to collect when writing --corpus. 0 uses "
        "--games-per-iteration",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)

    parser.add_argument("--checkpoint-dir", default="runs/distill")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--keep-every", type=int, default=10, help="0 disables")
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init",
        default=None,
        help="checkpoint to start the student from; ignored when resuming",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the directory named on the command line, and take nothing else.

    See `hexn.run.manifest`: a run is a frozen manifest, not a set of flags in
    a gitignored script.
    """
    from ..run import load
    from ..run.manifest import MODULES

    tokens = list(sys.argv[1:] if argv is None else argv)
    if len(tokens) != 1 or tokens[0].startswith("-"):
        raise SystemExit(
            "usage: python -m hexn.exit <run-directory>\n"
            "  create one with: python -m hexn.run.init --mode exit --name NAME -- <flags>\n"
            "  the flags this used to accept are frozen into the run's config/ instead."
        )
    manifest = load(tokens[0])
    if manifest.mode != "exit":
        raise SystemExit(
            f"{tokens[0]} is a {manifest.mode} run; launch it with "
            f"python -m {MODULES[manifest.mode]}"
        )
    args = manifest.namespace()
    # Cross-parameter validation below still reports through the parser.
    parser = build_parser()

    # `refresh` rebuilds the filter from the projected target, and under
    # --hard-target that projection is already a one-hot, so the visit lead the
    # margin tests no longer exists by the time it runs. Refusing is the only
    # honest option: the alternative is a flag that silently stops applying.
    if args.contested_margin > 0.0 and args.refresh_prior:
        parser.error(
            "--contested-margin cannot be combined with --refresh-prior: the "
            "refreshed filter is rebuilt from the projected target, which no "
            "longer carries the visit margin"
        )
    # Same shape, different missing quantity: the refreshed weight is rebuilt
    # from the batch tensors, which carry no per-option value, so the gap cannot
    # be re-read against the live policy's pick.
    if args.stake_scale > 0.0 and args.refresh_prior:
        parser.error(
            "--stake-scale cannot be combined with --refresh-prior: the "
            "refreshed filter is rebuilt from the projected target, which "
            "carries no per-option value to take a gap across"
        )
    # `--corpus` replays one fixed teacher's episodes forever and never
    # builds a store -- there is nothing growing for reanalysis to sample.
    if args.corpus and args.reanalyse_samples:
        parser.error(
            "--reanalyse-samples needs a growing replay store, which --corpus "
            "never builds: it replays one fixed collection every iteration"
        )

    config = DistillConfig(
        temperature=args.target_temperature,
        value_coefficient=args.value_coefficient,
        epochs=args.epochs,
        minibatch=args.minibatch,
        learning_rate=args.learning_rate,
        value_horizon=args.value_horizon,
        contested_only=args.contested_only,
        hard_target=args.hard_target,
        contested_margin=args.contested_margin,
        stake_scale=args.stake_scale,
        anchor=args.anchor,
        refresh_prior=args.refresh_prior,
        pack_contested=args.pack_contested,
    )

    policy, optimiser, space = build(args)
    policy.net.detach_value = args.detach_value
    directory = Path(args.checkpoint_dir)
    latest = directory / "latest.pt"
    log = directory / "log.jsonl"
    replay_dir = directory / "replay"

    start_iteration = 0
    first_game = 0
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        policy.net.load_state_dict(state["net"])
        optimiser.load_state_dict(state["optimiser"])
        # The loaded state carries the *checkpoint's* learning rate in its
        # param groups, which silently overrides the flag unless reset
        # explicitly here.
        for group in optimiser.param_groups:
            group["lr"] = args.learning_rate
        start_iteration = state["iteration"]
        first_game = state["games_started"]
        # A CPU ByteTensor is required whatever `map_location` moved it to.
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed at iteration {start_iteration}, game {first_game}", file=sys.stderr)
        store = EpisodeStore.resume(replay_dir, args.replay_positions)
        print(
            f"resumed the replay store: {len(store.iterations())} iterations, "
            f"{store.positions()} positions",
            file=sys.stderr,
        )
    else:
        store = EpisodeStore(replay_dir, args.replay_positions)
        if args.init:
            # Weights only. The optimiser state belongs to a different
            # objective — Adam's second moments were accumulated against a
            # PPO gradient, and carrying them into a cross-entropy would
            # scale the first steps by a history that no longer describes
            # the loss.
            state = torch.load(args.init, map_location=args.device, weights_only=False)
            policy.net.load_state_dict(state["net"])
            print(f"initialised from {args.init}", file=sys.stderr)

    search = Search(
        LeafEvaluator(policy=policy),
        simulations=args.simulations,
        wave=args.wave,
        exploration=args.exploration,
        stance=args.stance,
        max_trades=args.max_trades,
        root_noise=args.root_noise,
        noise_fraction=args.noise_fraction,
        rng=random.Random(args.seed),
    )
    expert = SearchPolicy(
        search,
        temperature=args.play_temperature,
        rng=random.Random(args.seed),
    )
    if args.collect_workers > 1:
        # Searched collection is the path that needs this: a single process
        # is slow because batching amortises the network call and nothing
        # else. Each worker builds its own search over its own CPU copy of
        # the net; the learner keeps the GPU.
        shard = max(1, -(-args.lanes // args.collect_workers))
        collector = ParallelCollector(
            [
                WorkerSpec(
                    seed=args.seed,
                    players=args.players,
                    lanes=shard,
                    action_cap=args.action_cap,
                    max_trades=args.max_trades,
                    first_game=first_game + worker,
                    stride=args.collect_workers,
                    width=args.width,
                    rounds=args.rounds,
                    torch_seed=args.seed + 100_000 + worker,
                    value_head=args.value_head,
                    policy_head=args.policy_head,
                    quantiles=args.quantiles,
                    simulations=args.simulations,
                    wave=args.wave,
                    exploration=args.exploration,
                    stance=args.stance,
                    root_noise=args.root_noise,
                    noise_fraction=args.noise_fraction,
                    play_temperature=args.play_temperature,
                )
                for worker in range(args.collect_workers)
            ]
        )
    else:
        collector = Collector(
            expert,
            lanes=args.lanes,
            players=args.players,
            seed=args.seed,
            action_cap=args.action_cap,
            first_game=first_game,
            max_trades=args.max_trades,
        )

    directory.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    if args.eval_at_start:
        baseline = duel(
            policy,
            games=args.eval_games,
            lanes=max(args.lanes, 64),
            players=args.players,
            seed=args.seed + 9_000,
            network_seats=(0, 2),
            max_trades=args.max_trades,
        )
        line = json.dumps({"iteration": -1, "duel": baseline})
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

    corpus: list | None = None
    if args.corpus:
        cache = Path(args.corpus)
        if not cache.exists():
            if isinstance(collector, ParallelCollector):
                collector.sync(policy.net)
            cache.parent.mkdir(parents=True, exist_ok=True)
            episodes = collector.collect(args.corpus_games or args.games_per_iteration)
            cache.write_bytes(pickle.dumps(episodes))
            print(f"wrote {len(episodes)} episodes to {cache}", file=sys.stderr)
        corpus = pickle.loads(cache.read_bytes())
        print(f"replaying {len(corpus)} episodes from {cache}", file=sys.stderr)

    # Assembled once, not once an iteration, for the fixed-teacher --corpus
    # path: the projection, the filter and the margin are all pure functions
    # of the episodes and this config, so re-running them would rebuild the
    # same tensors. There is no store on this path (validated above).
    fixed_batch = None

    for iteration in range(start_iteration, args.iterations):
        started = time.perf_counter()
        reanalysis = None
        if corpus is None:
            if isinstance(collector, ParallelCollector):
                # The single-process collector shares the learner's policy
                # object, so its search always sees the current weights. Workers
                # hold their own copies and do not: without this the whole run
                # would distil a frozen teacher into a moving student and look
                # perfectly healthy doing it.
                collector.sync(policy.net)
            episodes = collector.collect(args.games_per_iteration)
            store.append(iteration, episodes)
            if args.reanalyse_samples:
                # Over the current net -- the same `search` the collector
                # itself just used, so a reanalysed row is searched exactly
                # the way a freshly collected one would be.
                reanalysis = reanalyse(
                    store,
                    search,
                    args.reanalyse_samples,
                    random.Random(args.seed + 500_000 + iteration),
                )
            batch = store.batch(space, policy.layout, config)
        else:
            episodes = corpus
            if fixed_batch is None:
                fixed_batch = assemble(episodes, space, policy.layout, config)
            batch = fixed_batch
        collected = time.perf_counter() - started
        started = time.perf_counter()
        if config.refresh_prior:
            # Before the update, not per epoch: this is the prior the whole
            # update is filtered and anchored against, the way PPO snapshots
            # `pi_old` once.
            batch = refresh(policy, batch, config)
        stats = update(policy, optimiser, batch, config)
        updated = time.perf_counter() - started

        progress = summarise(
            episodes, iteration, len(batch), collected, updated, collector.games
        )
        record = {
            **asdict(progress),
            **asdict(stats),
            "replay_iterations": len(store.iterations()) if corpus is None else 0,
            "replay_positions": store.positions() if corpus is None else 0,
            "buffer_mib": round(batch.nbytes() / 1024 / 1024, 1),
            "elapsed": time.perf_counter() - began,
        }
        if reanalysis is not None:
            record["reanalysed_positions"] = reanalysis.reanalysed
            record["reanalyse_kl"] = round(reanalysis.mean_kl, 6)

        if args.eval_every and (iteration + 1) % args.eval_every == 0:
            record["duel"] = duel(
                policy,
                games=args.eval_games,
                lanes=max(args.lanes, 64),
                players=args.players,
                seed=args.seed + 10_000 + iteration,
                network_seats=(0, 2),
                max_trades=args.max_trades,
            )

        line = json.dumps(record)
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            state = {
                "iteration": iteration + 1,
                "games_started": collector.games_started(),
                "net": policy.net.state_dict(),
                "optimiser": optimiser.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "args": vars(args),
                "config": asdict(config),
            }
            save(latest, state)
            if args.keep_every and (iteration + 1) % args.keep_every == 0:
                save(directory / f"iter-{iteration + 1:05d}.pt", state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
