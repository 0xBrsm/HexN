# SPDX-License-Identifier: GPL-3.0-only
"""Reanalysis: keep a stored position's target fresh without recollecting it.

The AlphaGo-Zero-style trick this module exists for: a position collected
three iterations ago carries a `Target` searched over the net *three
iterations ago* -- stale exactly the way `hexn.exit.refresh` already
knows the recorded *prior* goes stale, except `refresh` only re-scores the
policy's own forward pass and never re-expands the tree, so it can correct the
contested filter but cannot correct the visit counts themselves. Reanalysis
does the more expensive thing on a sample instead of on everything: rebuild a
stored position's live `Game` (`hexn.replay.replay_to_ply`), root the
*current* net's `Search` over it, and replace the stored `Target` outright.

Value targets are left alone. They are the one-hot eventual winner by default
(`hexn.exit._value_targets`, `hexn.rewards.win_loss`), not the search's
own estimate, so nothing about them goes stale the way a policy prior does --
only `visits`/`prior` are recomputed.

Sampling is uniform over every retained position, not only the freshest
iteration's: staleness is worst for the *oldest* rows, collected under the
earliest, weakest net still in the window, so weighting toward them (rather
than away) is the point.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

import numpy as np

from hexset.mcts import Search

from .expert import Target, target_for
from .replay import replay_to_ply
from .selfplay import Transition
from .store import EpisodeStore

# One retained position: which shard, which episode in it, which seat, which
# transition in that seat's trajectory. Cheap to carry in bulk (four ints),
# so the whole store's addressable positions can be listed before sampling.
_Address = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReanalysisStats:
    reanalysed: int
    mean_kl: float


def _addresses(store: EpisodeStore) -> list[_Address]:
    out: list[_Address] = []
    for iteration in store.iterations():
        for e_idx, episode in enumerate(store.shard_episodes(iteration)):
            for seat, trajectory in enumerate(episode.trajectories):
                for t_idx, transition in enumerate(trajectory):
                    if isinstance(transition.aux, Target):
                        out.append((iteration, e_idx, seat, t_idx))
    return out


def _kl(old: Target, new: Target) -> float | None:
    """KL(old || new) over the visit distributions, or `None` when the two
    do not share an option order to compare (should not happen -- the
    replayed position is bit-identical to the one first searched, so
    `legal_actions` enumerates it the same way -- but a caller that skips
    the comparison on mismatch is safer than one that reports nonsense)."""
    if old.options != new.options:
        return None
    p = np.asarray(old.visits, dtype=np.float64)
    q = np.asarray(new.visits, dtype=np.float64)
    p_total, q_total = p.sum(), q.sum()
    if p_total <= 0 or q_total <= 0:
        return None
    p = p / p_total
    q = q / q_total
    eps = 1e-12
    return float(np.sum(np.where(p > 0, p * (np.log(p + eps) - np.log(q + eps)), 0.0)))


def reanalyse(
    store: EpisodeStore,
    search: Search,
    samples: int,
    rng: random.Random,
) -> ReanalysisStats:
    """Refresh up to `samples` stored positions' targets against `search`'s
    current net, writing the fresh `Target`s back onto their shards.

    `search` must be rooted on the same net the trainer is stepping --
    `hexn.exit` passes the collector's own `Search`, so a
    reanalysed row is searched exactly the way a freshly collected one would
    be. Batched through one `Search.run_many` call over the whole sample,
    the same batching a collector tick gets from `run_many` over its lanes.
    """
    addresses = _addresses(store)
    if not addresses or samples <= 0:
        return ReanalysisStats(reanalysed=0, mean_kl=0.0)
    chosen = (
        addresses if samples >= len(addresses) else rng.sample(addresses, samples)
    )

    games = []
    for iteration, e_idx, seat, t_idx in chosen:
        episode = store.shard_episodes(iteration)[e_idx]
        transition = episode.trajectories[seat][t_idx]
        games.append(replay_to_ply(episode, transition.step))

    results = search.run_many(games)

    kls: list[float] = []
    # iteration -> episode index -> (seat, transition index) -> fresh Transition
    edits: dict[int, dict[int, dict[tuple[int, int], Transition]]] = {}
    for (iteration, e_idx, seat, t_idx), (root, options, visits) in zip(
        chosen, results
    ):
        episode = store.shard_episodes(iteration)[e_idx]
        transition = episode.trajectories[seat][t_idx]
        old = transition.aux
        assert isinstance(old, Target)  # guaranteed by `_addresses`
        fresh = target_for(search, root, options, visits)
        kl = _kl(old, fresh)
        if kl is not None:
            kls.append(kl)
        edits.setdefault(iteration, {}).setdefault(e_idx, {})[(seat, t_idx)] = (
            dataclasses.replace(transition, aux=fresh)
        )

    for iteration, per_episode in edits.items():
        episodes = list(store.shard_episodes(iteration))
        for e_idx, changes in per_episode.items():
            episode = episodes[e_idx]
            trajectories = [list(seat_transitions) for seat_transitions in episode.trajectories]
            for (seat, t_idx), new_transition in changes.items():
                trajectories[seat][t_idx] = new_transition
            episodes[e_idx] = dataclasses.replace(
                episode, trajectories=tuple(tuple(s) for s in trajectories)
            )
        store.replace(iteration, episodes)

    return ReanalysisStats(
        reanalysed=len(chosen),
        mean_kl=float(np.mean(kls)) if kls else 0.0,
    )
