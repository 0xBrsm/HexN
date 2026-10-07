# SPDX-License-Identifier: GPL-3.0-only
"""Expert iteration: play with a search, train toward what the search decided.

A search rooted on a checkpoint's own priors plays better than the raw policy
does on its own — that gap is why this module exists: the search supplies a
policy target the raw policy does not already contain. Producing those
searched games is all this module does; consuming their visit targets belongs
to distillation.

**It reuses `hexn.selfplay.Collector` rather than growing a second game loop.**
The lane bookkeeping, the seat demultiplexing, the action cap, the outcome and
the seed/index replay contract are all already there and already tested, and
none of them care what decided the action. What changes is the policy, so that
is the only thing written here.

`Search.run_many` batches leaf evaluations across independent lanes while each
tree still contributes at most its own wave. That distinction matters: making
one tree's wave wide causes descents to collide on unexpanded leaves and
changes how much search the nominal simulation count buys; combining wave-16
leaves from separate roots changes only the network batch. Which of those
trades favours GPU inference depends on the hardware; serial Python tree work
can make a compiled single-lane CPU configuration faster regardless. Scale
comes from independent CPU collector processes either way.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hexset.actions import Action
from hexset.clients.netbot import LeafEvaluator
from hexset.game import Game
from hexset.mcts import Node, Search, visit_policy
from .selfplay import Choice, Request


@dataclass(frozen=True)
class Target:
    """What the search decided, kept beside the decision it produced.

    Rides on `Choice.aux` and so onto `Transition.aux`, which is the pocket
    `hexn.selfplay` leaves for exactly this. `options` is kept beside the
    counts because the visit distribution is over the concrete actions the
    search rooted on, and a training step wants them in that order.

    `visits` is raw counts rather than a normalised distribution, so a training
    step can pick its own temperature and can also weight a position by how much
    search went into it.
    """

    options: tuple[Action, ...]
    visits: np.ndarray
    # The root's prior over the same options. Kept because the only aligned
    # distillation target is the *disagreement* between the search and the
    # policy, and that cannot be recovered afterwards: re-running the forward
    # would read a policy that has since moved.
    prior: np.ndarray | None = None
    # The backed-up mean value of each option, from the mover's seat and under
    # the search's own stance -- exactly what `_select` ranks. `visits` says
    # which option the search preferred and this says by how much it mattered:
    # a five-vote lead between two options worth the same thing is a coin flip,
    # and a five-vote lead across half a victory point is not. Unvisited edges
    # score 0.0, matching `_select`.
    values: np.ndarray | None = None


@dataclass
class SearchPolicy:
    """A `hexn.selfplay.BatchPolicy` that runs one tree per decision.

    `temperature` shapes the *play*, not the target: `Target` keeps raw counts,
    so a distillation step is free to choose its own. Sampling rather than
    taking the argmax is what makes the corpus worth collecting — four identical
    greedy searches replay one game, and a policy target built from positions
    that only ever occur on the best line teaches nothing about the rest.
    """

    search: Search
    temperature: float = 1.0
    rng: random.Random = field(default_factory=random.Random)

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        for request in requests:
            if request.game is None:
                raise ValueError("a searching policy needs a position, not just its encoding")
        results = self.search.run_many([request.game for request in requests])
        return [
            self._choice(request, result)
            for request, result in zip(requests, results)
        ]

    def trader(
        self, game: Game, seat: int | None, max_offers: int | None = None,
        trader: str | None = None,
    ):
        """What this policy brings to `game`'s trade event at `seat`.

        A gate is a pure function of the position (`hexset.trading.valued_many`
        reads it off `game.gates`) and never touches the tree, so there is
        nothing here for the search itself to contribute. When the wave is
        scored by a `hexset.clients.netbot.LeafEvaluator` -- a network's own
        leaves -- this delegates to the wrapped policy's `trader`, which is a
        `hexset.clients.netbot.NetworkBot` over that same checkpoint
        (`hexn.policy.NetworkPolicy.trader`), so a searched seat trades
        exactly as the raw policy would and through the same one gate (a
        policy with no `trader` is seated as `None`, which
        `hexset.gym.lanes.LaneEnv` reads as silently trade-free). Any other
        evaluator has no policy to derive a gate from, so this seat never
        trades, matching every scripted bot the engine seats itself.
        """
        evaluator = self.search.evaluator
        if not isinstance(evaluator, LeafEvaluator):
            return None
        make = getattr(evaluator.policy, "trader", None)
        if make is None:
            return None
        # Only a run that names a trader asks for one, so a policy whose hook
        # predates it is still asked exactly as it always was.
        return make(game, seat, max_offers) if trader is None else make(game, seat, max_offers, trader)

    def _one(self, request: Request) -> Choice:
        if request.game is None:
            raise ValueError("a searching policy needs a position, not just its encoding")
        return self._choice(request, self.search.run(request.game))

    def _choice(
        self,
        request: Request,
        result: tuple[Node, tuple[Action, ...], np.ndarray],
    ) -> Choice:
        root, options, visits = result
        # The transition is filed with `request.mask`, which was built from
        # `request.options`. A search that enumerated its root differently
        # would return actions the recorded mask calls illegal, and the corpus
        # would be quietly wrong rather than loudly broken.
        if options != request.options:
            raise ValueError(
                f"the search rooted on {len(options)} options where the "
                f"collector offered {len(request.options)}"
            )

        weights = visit_policy(visits, self.temperature)
        pick = self.rng.choices(range(len(options)), weights=list(weights))[0]
        share = float(weights[pick])
        return Choice(
            action=options[pick],
            log_prob=math.log(share) if share > 0 else -math.inf,
            # The backed-up mean, not `root.value`. The network's own estimate of
            # the root is what the value head already believes; the point of the
            # exercise is to move it toward what searching over it concluded.
            value=_value(root, visits),
            # `root.prior` is the policy's own distribution over these same
            # options, read at the one moment it is the policy that produced
            # the visits. Recorded, not recomputed: by training time the policy
            # has moved and the disagreement would be against the wrong prior.
            aux=target_for(self.search, root, options, visits),
        )


def target_for(
    search: Search, root: Node, options: tuple[Action, ...], visits: np.ndarray
) -> Target:
    """The `Target` a root search concluded, shared between `SearchPolicy`
    (fresh from a collector's own decision) and `hexn.reanalyse` (fresh from
    a stored position re-searched over a newer net) -- the two places a
    `(root, options, visits)` triple out of `Search.run`/`run_many` becomes a
    training label.
    """
    return Target(
        options=options,
        visits=visits,
        prior=root.prior,
        values=_means(search, root, visits),
    )


def _means(search: Search, root: Node, visits: np.ndarray) -> np.ndarray | None:
    """Per-option Q, read the way `_select` reads it.

    Recomputed from `totals` rather than taken from `ranked` so the
    `paranoid` stance is handled: its max is not linear, so the mean of what
    the stance read at each visit is not the stance's read of the mean, and
    `_select` uses the latter.
    """
    if not root.expanded or root.totals.size == 0:
        return None
    ranked = (
        search.rank_rows(root.totals, root.mover)
        if search.stance == "paranoid"
        else root.ranked
    )
    counts = np.asarray(visits, dtype=np.float64)
    return np.where(counts > 0, ranked / np.maximum(counts, 1.0), 0.0)


def _value(root: Node, visits: np.ndarray) -> tuple[float, ...]:
    total = float(visits.sum())
    if not root.expanded or total <= 0:
        # Nothing was evaluated here. Empty means "no estimate", as it does
        # for every scripted policy. A forced move is not this case: since
        # HexSet 1.1 a root with one legal move is evaluated and its only
        # edge credited with every simulation at that value, so it reads
        # like any other root (before, it came back unexpanded and empty).
        return ()
    return tuple(root.totals.sum(axis=0) / total)
