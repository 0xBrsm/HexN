# SPDX-License-Identifier: GPL-3.0-only
"""Distillation: move the policy toward what the search decided.

`hexn.expert.SearchPolicy` plays games with a tree and files what the tree
concluded on `Transition.aux`. This module is the other half — the training step
that consumes those targets: the visit counts carry information a search
finds that the raw policy does not already have.

## The loss is one cross-entropy over the flat slots

The search's target is a distribution over *concrete options*; the policy's is
one row over flat action slots. Under contract 4 those were different shapes —
`PROPOSE_TRADE` was a single slot standing for many offers, so the target had
to be factored into a slot row and an offer row and the two cross-entropies
weighted back together. Contract 5 deletes the trade actions
(`hexset.trading`), so every option now has its own slot, several options can
still share one (`space.index` is many-to-one nowhere else), and the
projection is a plain sum of each option's mass onto its own slot.

## Temperature is applied over options, before the projection

`visit_policy` sharpens the distribution the search actually chose among, so it
belongs at option level rather than after aggregation.

## The value head is still trained on terminal outcomes

Not on the search's backed-up root value, which is the tempting alternative and
is deliberately not taken here by default (`--value-horizon` opts into it). So
the value half is exactly what `hexn.ppo` does, `to_frame` and `win_loss`
reused unchanged: the target is the one-hot eventual winner, and the loss is
cross-entropy against `Prediction.value_logits`, mirroring `ppo.minibatch_terms`
term for term (including its quantile-head branch, `test_ppo`'s coverage of
which this module's own tests reuse the shape of). This module used to train
the value head on `hexn.rewards.reward` — relative terminal points, a margin
— by squared error; contract 6 made `Prediction.value` a win-probability
softmax over seats, and `hexn.ppo` moved first, so this module was left
mismatched (a probability minus a margin, squared) until now.

## `--value-horizon`'s bootstrap target is not a proper probability vector

`_value_targets` reads `Transition.value`, which for a searched transition is
`hexn.expert._value`: the search's root-level backed-up mean, averaged over
every simulation that reached this seat's decision. Under contract 6 that mean
is close to a per-seat win-probability vector when every backed-up leaf was
network-evaluated — `hexn.policy.NetworkPolicy.score_rows` reads
`Prediction.value` off the same softmax `hexn.ppo` trains toward, which sums to one
across seats by construction. It is **not** one in general: `hexset.mcts.
Search._node` scores a *terminal* leaf with `hexset.victory.relative_points`
instead, a zero-sum margin on a different scale, and a shallow simulation can
bottom out there — `hexset.mcts` is this project's read-only engine, so that
mismatch is not this module's to fix. `_value_targets` is explicit about it
rather than silently renormalising a vector that need not sum to one: see its
own docstring.

## The search's `stance` is orthogonal to this module

`hexn.exit`'s `--stance` (default `"relative"`) feeds `hexset.mcts.
Search`, never `hexn.exit` itself, which has no stance of its own. Of the
three stances `Search._backup`/`_select` actually implement — `own`, `relative`,
`paranoid` — none converts a margin to a probability or back; each is a plain
reduction of whatever per-seat vector a leaf already produced (own value,
own-less-the-mean-of-others, own-less-the-best-other) into the scalar PUCT
maximises, and stays sound whether that vector is `relative_points`-scale or a
`Prediction.value` probability. `relative`'s reading is even harmless on a
probability vector specifically: for a vector that sums to exactly one, `(p *
seats - 1) / (seats - 1)` is a fixed increasing affine function of `p`, so it
reorders nothing `_select`'s argmax reads. `hexset.bots.STANCES` also carries a
fourth stance, `"win"` — `softmax(vector / WIN_TEMPERATURE)[seat]` — built to
turn a *margin* into a probability at a temperature fitted for that purpose
(`hexset.bots.search2.win`); passing it a vector that is already a probability
would double-convert it. Worse, `hexset.mcts.Search` accepts `"win"` (it is a
member of `STANCES`) but never implements it: `STANCE_ROWS` and `_backup`'s own
branches only know `own`/`relative`/`paranoid`, so under `stance="win"` `Node.
ranked` is never written and PUCT's value term is silently always zero — the
search would rank purely on the prior and the visit-count bonus. That is a
defect in the read-only engine, not in this trainer, and it means `"win"` is
not a safe default to move `--stance` to from here: `"relative"` stays the
default, unchanged, because it is the one option that is both implemented and
correct on a probability-valued leaf.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import Tensor

from hexset.actions import Action, ActionSpace
from ..expert import Target
from hexset.mcts import visit_policy
from ..model import Packing, pack
from ..policy import NetworkPolicy, _entropy
from ..rewards import win_loss
from ..selfplay import Episode, Transition

# Borrowed from `hexn.ppo` rather than lifted into a third module: minibatch
# shuffling, explained variance and the once-per-update stat reduction are
# short functions that mean the same thing in both trainers, and a
# `hexn.training` holding only those would be more indirection than it
# saves. The reward rotation is not among them any more: it is
# `hexset.encoding.to_frame`, the encoder's own.
from hexset.encoding import to_frame
from ..ppo import _explained_variance, _stack_mean, _stack_median
from ..steps import Cursor

if TYPE_CHECKING:
    from ..steps import Steps


@dataclass(frozen=True)
class DistillConfig:
    temperature: float = 1.0
    # Train the policy only where the search overruled it, and on a hard argmax
    # rather than the visit distribution. Both default off, so existing
    # behaviour is unchanged.
    #
    # Why. Handing back the visit *distribution* everywhere is worse than
    # handing back nothing where the search agreed with the policy: the
    # distribution's entropy is set by the search's own exploration settings
    # -- `exploration`, `simulations`, branching -- not by the position. Play
    # reads the argmax; training read the distribution. These two flags close
    # that gap.
    contested_only: bool = False
    hard_target: bool = False
    # Visit share the search's pick must lead the policy's by before the row
    # counts as contested. 0.0 disables the margin. See `contested` for what
    # this filters and why.
    contested_margin: float = 0.0
    # The Q-gap at which a contested row earns full weight, in reward units --
    # 0.01 is 0.1 victory points. 0.0 is a plain 0/1 filter. Above it the
    # weight is `min(gap / stake_scale, 1)`, and rows whose gap is negative --
    # the search's own value says its pick is the worse of the two -- fall
    # out entirely.
    #
    # Why not gate on `contested_margin` (the visit lead) instead. The visit
    # lead measures how sure the search is, and certainty is not what
    # separates a real correction from a coin flip. The Q-gap gates on what
    # the correction is *worth*, which is a different quantity and the one
    # `_select` actually ranks: a narrow vote lead between two options the
    # search values identically is noise however sure the count looks, and
    # the same narrow lead across a large value gap is not.
    stake_scale: float = 0.0
    # Weight on the anchor: a cross-entropy toward the *recorded prior* on the
    # rows `contested_only` zeroed. 0.0 disables the anchor.
    #
    # Why it is needed. `contested_only` filters the policy loss but cannot
    # filter its effect. The trunk is shared and `self.value` is one linear
    # layer on it, the value loss is an unweighted mean over every row, and a
    # network has no per-position parameters -- so fitting only the contested
    # rows moves the policy on the rest with nothing holding it there.
    #
    # Why the prior and not the visits. Anchoring on the visit distribution
    # would flatten the policy: its entropy belongs to the search operator --
    # `exploration`, `simulations`, branching -- not to the position. The
    # prior's entropy is the policy's own, so anchoring to it is a restoring
    # force toward where the policy already was and carries no flattening
    # pressure of its own. It is PPO's trust region, spelled as a
    # cross-entropy against the collection-time policy.
    anchor: float = 0.0
    # Recompute the filter and the anchor against the live policy instead of the
    # prior recorded at collection time. This is what makes a replay store worth
    # having: the search is expensive and its counts stay a valid label for
    # their position, while the prior they are compared against is one
    # forward pass and goes stale the moment the policy moves. Recomputing it
    # also turns rows that carry no policy gradient from waste into
    # not-yet-contested -- rows cross into the contested set as the student
    # drifts, so one corpus keeps yielding fresh targets.
    #
    # Retaining more than the newest iteration's rows is safe here in a way it
    # is not for PPO (`hexn.store.EpisodeStore`, `--replay-positions`): a
    # policy-gradient batch that runs several generations off-policy costs
    # strength and caps the learning rate -- but that is a *policy gradient*,
    # an estimator that needs an importance correction to stay valid
    # off-policy. Distillation is a supervised cross-entropy against a fixed
    # label: no ratio, no correction, nothing to break. And the label is
    # local -- a visit count is what the search measured at that position, so
    # it inherits none of the game-level terminal noise that makes the *game*
    # the independent unit for PPO.
    refresh_prior: bool = False
    # Give the policy term its own dense minibatches instead of letting it
    # ride the value term's, which trades a low-variance shared pass for a
    # policy loss estimated from more, smaller steps at the same total visit
    # count. The anchor rides the value pass on purpose: that is the pass
    # that moves the trunk, and holding the policy while it does is the whole
    # point of the anchor. Same shape PPG uses, for the same reason.
    pack_contested: bool = False
    value_coefficient: float = 0.5
    epochs: int = 4
    minibatch: int = 1024
    # Rows per backward within a minibatch, gradients accumulated: the step is
    # the whole minibatch's, the device holds one slice. 0 (or anything at or
    # above `minibatch`) is one backward per minibatch. The batch then stays
    # on the host and each slice moves to the device as it is used.
    micro_batch: int = 0
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    # Weight on KL(policy || start) over every legal action, where `start` is
    # the policy the update began from -- the net the batch was searched over
    # (`Batch.prior_log_probs`, `hexn.ppo.attach_prior`). 0 disables. The same
    # term as `hexn.ppo`'s `prior_kl`, tethered to the round's own start
    # rather than to a fixed parent.
    prior_kl: float = 0.0
    # 0 keeps the terminal one-hot target (the eventual winner). Anything
    # higher bootstraps off the search's own backed-up estimate that many of
    # the seat's decisions ahead instead -- see `_value_targets` for exactly
    # what that estimate is and is not.
    value_horizon: int = 0


@dataclass(frozen=True)
class Stats:
    """What one update did, for the run log rather than for the maths."""

    positions: int
    policy_loss: float
    value_loss: float
    agreement: float
    explained_variance: float
    # The plain squared error of the softmaxed mean (`Prediction.value`)
    # against the one-hot target -- `hexn.ppo.Stats`'s own field, reused
    # here for the same reason: `value_loss` above is cross-entropy (or, under
    # a quantile head, the pinball loss), and the two are on different scales,
    # so a run compared against a value-head ablation needs one column both
    # configurations compute identically. Equal to `value_loss` under no head
    # shape any more, now that the win head's loss is categorical.
    value_mse: float = 0.0
    anchor_loss: float = 0.0
    # Mean policy entropy over the batch, useful for diagnosing whether the
    # policy loss is flattening the distribution rather than sharpening it
    # toward the search's pick.
    entropy: float = 0.0
    # `agreement` pools two movements that point opposite ways -- the contested
    # rows the loss pulls toward the search, and the settled rows nothing holds.
    # One number cannot say which moved, so it is also reported split. Both are
    # 0.0 when the corresponding subset is empty, which `contested_only=False`
    # makes true of `agreement_settled` by construction.
    agreement_contested: float = 0.0
    agreement_settled: float = 0.0
    # How many rows actually carried policy gradient. Derived from the split
    # gauges before, which needed algebra and an assumption; logged now.
    contested_positions: int = 0
    # The tether's divergence, averaged over the update's minibatches (0.0
    # with the tether off), and the median pre-clip gradient norm.
    prior_kl: float = 0.0
    grad_norm: float = 0.0


@dataclass
class Batch:
    """One update's worth of searched positions, flattened out of episodes."""

    buffer: Tensor
    mask: Tensor
    slot_target: Tensor
    value_target: Tensor
    # 1 where the policy should learn from this row, 0 where it should not.
    # A weight rather than a filter so the *value* head still sees every
    # position: restricting it to contested rows would train it on a biased
    # slice of the state distribution, which is a different experiment.
    policy_weight: Tensor
    # The anchor's target and its weight: the recorded prior on the same two
    # rows, and `1 - policy_weight` wherever a prior was actually recorded. Zero
    # weight on a corpus collected before `Target.prior` existed, so an old
    # corpus stays readable and simply cannot be anchored.
    anchor_slots: Tensor
    anchor_weight: Tensor
    # 1 where the search actually expanded this row. `contested` reads it off
    # the raw visits, but a projected slot target cannot: an all-zero row
    # becomes a one-hot on whichever option `argmax` reaches first, and
    # `refresh` would call that a disagreement.
    searched: Tensor
    # The masked log-softmax of the policy the update starts from, over every
    # slot, one row per position: what `prior_kl` tethers to and what
    # `measure` reads the update's movement against. Filled by
    # `hexn.ppo.attach_prior`, after assembly; `None` until then.
    prior_log_probs: Tensor | None = None

    FIELDS = (
        "buffer",
        "mask",
        "slot_target",
        "value_target",
        "policy_weight",
        "anchor_slots",
        "anchor_weight",
        "searched",
    )

    @staticmethod
    def concat(batches: Sequence["Batch"]) -> "Batch":
        """One batch out of several iterations' worth, for the replay buffer.

        Assembled batches rather than raw episodes: the projection is already
        paid, the tensors are compact, and nothing in a `Batch` depends on which
        iteration produced it once `refresh` recomputes the prior.
        """
        if not batches:
            raise ValueError("nothing to concatenate")
        if len(batches) == 1:
            return batches[0]
        return Batch(
            **{
                name: torch.cat([getattr(b, name) for b in batches])
                for name in Batch.FIELDS
            }
        )

    def nbytes(self) -> int:
        held = [getattr(self, name) for name in self.FIELDS]
        if self.prior_log_probs is not None:
            held.append(self.prior_log_probs)
        return sum(t.element_size() * t.nelement() for t in held)

    def __len__(self) -> int:
        return self.buffer.shape[0]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            **{name: getattr(self, name).to(device) for name in self.FIELDS},
            prior_log_probs=(
                None
                if self.prior_log_probs is None
                else self.prior_log_probs.to(device)
            ),
        )


def contested(target: Target, margin: float = 0.0) -> bool:
    """Did the search overrule the policy at this position, and mean it?

    False when the prior was not recorded, which keeps an old corpus readable
    and makes `contested_only` a no-op on it rather than an empty batch.
    A row of all-zero visits is not contested however the argmaxes fall: it
    means the search never expanded here, and `argmax` on ties would invent a
    disagreement.

    `margin` is the visit share the search's pick must lead the policy's by
    before the disagreement counts, and it defaults to 0. It exists because
    the unguarded definition is dominated by decisions the search cannot
    actually call: two trade *bundles* sharing one flat slot can disagree by
    only a handful of votes, which is noise around a tie rather than a
    considered correction.
    """
    if target.prior is None or not len(target.options):
        return False
    if float(np.max(target.visits)) <= 0.0:
        return False
    best = int(np.argmax(target.visits))
    theirs = int(np.argmax(target.prior))
    if best == theirs:
        return False
    if margin <= 0.0:
        return True
    total = float(np.sum(target.visits))
    lead = float(target.visits[best] - target.visits[theirs]) / max(total, 1.0)
    return lead >= margin


def stake(target: Target) -> float:
    """What the search says its correction is worth, in reward units.

    The backed-up mean of the search's pick less that of the policy's, both read
    from the mover's seat exactly as `_select` ranks them. Positive means the
    search's own value agrees with its own counts; negative means it does not,
    which happens when the exploration bonus carried the visits somewhere the
    means never followed.

    0.0 when there is nothing to compare -- a corpus collected before `Target`
    recorded values, no prior, an unexpanded row -- so a caller weighting by this
    drops those rows rather than inventing a stake for them.
    """
    if target.values is None or target.prior is None or not len(target.options):
        return 0.0
    if float(np.max(target.visits, initial=0.0)) <= 0.0:
        return 0.0
    values = np.asarray(target.values, dtype=np.float64)
    if values.shape != target.visits.shape:
        return 0.0
    best = int(np.argmax(target.visits))
    theirs = int(np.argmax(target.prior))
    return float(values[best] - values[theirs])


def project(
    target: Target, space: ActionSpace, temperature: float, hard: bool = False
) -> np.ndarray:
    """A visit distribution over options, as the policy's slot row sees it."""
    if hard:
        # The object play actually uses. `visit_policy` with a temperature
        # approaching zero is the same limit, but an exact one-hot avoids
        # asking what 0 ** (1/eps) should be on a row of zero-visit edges.
        weights = np.zeros(len(target.options), dtype=np.float64)
        weights[int(np.argmax(target.visits))] = 1.0
    else:
        weights = visit_policy(target.visits, temperature)
    return _spread(target.options, weights, space)


def _spread(
    options: Sequence[Action], weights: np.ndarray, space: ActionSpace
) -> np.ndarray:
    """Lay a distribution over options onto the policy's slot row.

    Shared by the visit target and the anchor's prior so the two land on the
    slot space identically.
    """
    slots = np.zeros(space.size, dtype=np.float32)
    for option, share in zip(options, weights):
        slots[space.index(option)] += share
    return slots


def project_prior(target: Target, space: ActionSpace) -> np.ndarray | None:
    """The root prior over the same options, on the same slot row.

    The anchor's target. `None` when there is no prior to anchor to -- an old
    corpus that never recorded one, or a degenerate row -- and the caller pairs
    that with zero anchor weight rather than inventing a uniform target, which
    would pull the policy toward uniform on exactly the rows it knows least
    about.
    """
    if target.prior is None or not len(target.options):
        return None
    prior = np.asarray(target.prior, dtype=np.float64)
    total = float(prior.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return _spread(target.options, prior / total, space)


def assemble(
    episodes: Sequence[Episode],
    space: ActionSpace,
    layout: Packing,
    config: DistillConfig,
) -> Batch:
    """Flatten searched episodes into one update's tensors.

    Every transition must carry a `Target`. A batch that silently skipped the
    ones that did not would train on whichever positions happened to be searched
    and quietly weight the corpus by that, so a missing target is an error.
    """
    observations = []
    masks: list[np.ndarray] = []
    slot_targets: list[np.ndarray] = []
    weights: list[float] = []
    values: list[np.ndarray] = []
    anchor_slots: list[np.ndarray] = []
    anchor_weights: list[float] = []
    searched_flags: list[float] = []

    for episode in episodes:
        wins = win_loss(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            seen = np.asarray(to_frame(wins, seat), dtype=np.float32)
            wanted = _value_targets(trajectory, seen, seat, config.value_horizon)
            for transition, want in zip(trajectory, wanted):
                target = _target(transition)
                slots = project(
                    target, space, config.temperature, config.hard_target
                )
                weight = (
                    1.0
                    if not config.contested_only
                    or contested(target, config.contested_margin)
                    else 0.0
                )
                if weight > 0.0 and config.stake_scale > 0.0:
                    weight = min(
                        max(stake(target), 0.0) / config.stake_scale, 1.0
                    )
                observations.append(transition.observation)
                masks.append(transition.mask)
                slot_targets.append(slots)
                weights.append(weight)
                values.append(want)
                # The anchor holds the rows the policy loss let go of, so its
                # weight is the complement -- and zero where there is no prior
                # to hold them to.
                anchored = project_prior(target, space)
                if anchored is None:
                    anchor_slots.append(np.zeros(space.size, dtype=np.float32))
                    anchor_weights.append(0.0)
                else:
                    anchor_slots.append(anchored)
                    anchor_weights.append(1.0 - weight)
                searched_flags.append(
                    1.0 if float(np.max(target.visits, initial=0.0)) > 0.0 else 0.0
                )

    if not observations:
        raise ValueError("no searched transitions to learn from")

    return Batch(
        buffer=pack(layout, observations),
        mask=torch.from_numpy(np.stack(masks)),
        slot_target=torch.from_numpy(np.stack(slot_targets)),
        value_target=torch.from_numpy(np.stack(values)),
        policy_weight=torch.tensor(weights, dtype=torch.float32),
        anchor_slots=torch.from_numpy(np.stack(anchor_slots)),
        anchor_weight=torch.tensor(anchor_weights, dtype=torch.float32),
        searched=torch.tensor(searched_flags, dtype=torch.float32),
    )


def _value_targets(
    trajectory: Sequence[Transition],
    terminal: np.ndarray,
    seat: int,
    horizon: int,
) -> list[np.ndarray]:
    """What each of one seat's decisions is told the position was worth.

    `horizon` 0 gives every decision the terminal target — `terminal` itself,
    `win_loss(episode.outcome)` rotated into this seat's frame by the caller:
    the one-hot eventual winner, or all-zero for an unfinished game, exactly
    `hexn.ppo`'s own value target. That is what AlphaZero does and what this
    trainer has always done.

    A bootstrapped target refuses to look that far. Counting `horizon` of the
    seat's *own* decisions ahead — not game steps, since three other seats move
    in between and the gap in dice rolled is what the variance is about — the
    target becomes the estimate recorded there, and the dice between the two
    stop entering it.

    The estimate used is the search's backed-up root mean rather than the raw
    head, because `SearchPolicy` already stores that in `Transition.value` and
    an average over a tree is the better of the two for free.

    **This estimate is not guaranteed to be a per-seat probability vector, and
    this function does not pretend otherwise.** `Transition.value` is
    `hexn.expert._value`: `Node.totals.sum(axis=0) / visits.sum()`, the mean
    over every leaf the search actually backed up through this root. Under
    contract 6 that mean sums to one when every one of those leaves was
    network-evaluated — `hexn.policy.NetworkPolicy.score_rows` reads the same
    `Prediction.value` softmax `terminal`'s one-hot lives on — but
    `hexset.mcts.Search._node` scores a *terminal* leaf with
    `hexset.victory.relative_points` instead, a zero-sum margin on a different
    scale, and a shallow simulation can bottom out there before the position
    ends. `hexset.mcts` is this project's read-only engine, so that mismatch is
    not fixed here: the horizon target is exactly this seat's own backed-up
    component, with the rest of the vector carried through unrenormalised, on
    whatever scale the search actually produced it — the same pass-through
    this function has always done, now on a vector whose own-seat component
    approximates a win probability rather than being one by construction.

    **`Transition.value` is in board order here, and the target is in the
    seat's frame.** `SearchPolicy._value` returns `Node.totals`, whose columns
    are board seats, while `to_frame` puts the seat itself first to match what the
    encoder fed the network. The two policies that fill this field disagree on
    frame — `NetworkPolicy` emits the mover's frame — which nothing caught
    because until now nothing read it. Rotating is not optional; skipping it
    trains without complaint and plays nonsense.

    Falls back to the terminal target at the end of a trajectory, and wherever
    no estimate was recorded. A forced move records one: since HexSet 1.1 the
    search evaluates a root with one legal move and returns the evaluator's
    value for it, where it used to return none, so a horizon that lands on a
    forced move now bootstraps from that value instead of falling back.
    """
    if horizon <= 0:
        return [terminal] * len(trajectory)
    targets = []
    for index in range(len(trajectory)):
        ahead = index + horizon
        estimate = trajectory[ahead].value if ahead < len(trajectory) else ()
        if estimate:
            targets.append(np.asarray(to_frame(estimate, seat), dtype=np.float32))
        else:
            targets.append(terminal)
    return targets


def _target(transition: Transition) -> Target:
    if not isinstance(transition.aux, Target):
        raise ValueError(
            "a distillation batch needs search targets; this transition came "
            "from a policy that does not produce them"
        )
    return transition.aux


def losses(slot_log_probs: Tensor, batch: Batch, rows: Tensor) -> Tensor:
    """The cross-entropy toward the search's target.

    Split out of `update` so it can be tested without an optimiser in the way.
    """
    weight = batch.policy_weight[rows]
    # Normalise by the weight, not the row count: a minibatch where one row in
    # twenty is contested must produce the same gradient scale as a full one,
    # or the effective learning rate falls with the filter's selectivity.
    total = weight.sum().clamp(min=1e-8)
    return (
        weight * -(batch.slot_target[rows] * slot_log_probs).sum(-1)
    ).sum() / total


def refresh(
    policy: NetworkPolicy,
    batch: Batch,
    config: DistillConfig,
    *,
    chunk: int = 4096,
) -> Batch:
    """Recompute the filter and the anchor against the live policy.

    What a buffer needs, and it is cheap for a specific reason: the two halves
    of the target age at completely different rates. The visit counts are
    expensive to produce by search and stay a valid label for their
    position; the prior they are compared against is one forward pass and
    is stale as soon as the policy moves -- which is exactly why
    `Target.prior` had to be recorded in the first place. Recomputing the
    prior is far cheaper than re-searching, so a cached corpus is reusable
    for a small fraction of what it cost to collect.

    The anchor becomes the *current* policy's own distribution, which is the
    correct trust region rather than an approximation of one: it holds the
    settled rows where they are at the start of this update, the way PPO anchors
    to `pi_old`.

    Nothing here has a gradient. It runs once per iteration, not per epoch.

    **The comparison is over options, and under contract 5 that is the same
    thing as over slots.** It was not under contract 4, where the trade slot
    was a *sum* over every offer on the row and the two definitions disagreed
    on about a third of rows; the offer row and the option-space argmax it
    needed both retired with the trade actions (`hexset.trading`).
    """
    rows = len(batch)
    device = policy.device
    slot_rows: list[Tensor] = []

    with torch.no_grad():
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            slots, _, _, _ = policy.distributions(
                batch.buffer[start:stop].to(device),
                batch.mask[start:stop].to(device),
            )
            slot_rows.append(slots.exp().cpu())

    current = torch.cat(slot_rows)

    if config.contested_only:
        overruled = (
            batch.slot_target.argmax(-1) != current.argmax(-1)
        ).to(torch.float32)
        # An unexpanded row is not a disagreement however the argmaxes fall.
        weight = overruled * batch.searched
    else:
        weight = torch.ones(rows, dtype=torch.float32)

    return replace(
        batch,
        policy_weight=weight,
        anchor_slots=current,
        anchor_weight=1.0 - weight,
    )


def anchor_losses(slot_log_probs: Tensor, batch: Batch, rows: Tensor) -> Tensor:
    """The trust-region term: the same cross-entropy, toward the prior.

    Structurally identical to `losses` on the complementary weight, which is the
    point -- the anchor has to be the same object as the target so the two can
    be traded off by one scalar. Minimising H(prior, pi) minimises
    KL(prior || pi): they differ by H(prior), a constant of the corpus and not
    of the parameters, so the gradient is the same one a KL penalty would give.
    """
    weight = batch.anchor_weight[rows]
    total = weight.sum().clamp(min=1e-8)
    return (
        weight * -(batch.anchor_slots[rows] * slot_log_probs).sum(-1)
    ).sum() / total


def _backward(
    policy: NetworkPolicy,
    batch: Batch,
    rows: Tensor,
    config: DistillConfig,
    *,
    with_value: bool,
    with_policy: bool,
) -> dict[str, Tensor]:
    """One minibatch's loss, backpropagated in `config.micro_batch`-row slices.

    Every term is normalised by the *minibatch's* own denominator -- its row
    count for the value term and the tether, its total weight for the
    weighted policy and anchor terms -- so the slices' gradients sum to
    exactly the gradient one backward over the whole minibatch would give,
    and the gauges returned are the minibatch's own. The caller zeroes the
    gradients before and steps after. Everything returned is detached.

    `with_value` is the pass that moves the trunk: the value term, and the
    anchor and the tether, which ride it (see `update`). `with_policy` adds
    the cross-entropy toward the search.
    """
    device = policy.device
    total = len(rows)
    step = (
        config.micro_batch
        if config.micro_batch and config.micro_batch < total
        else total
    )
    weights = batch.policy_weight[rows]
    weight_total = weights.sum().clamp(min=1e-8).to(device)
    anchor_total = batch.anchor_weight[rows].sum().clamp(min=1e-8).to(device)
    tethered = with_value and config.prior_kl > 0.0
    anchored = with_value and config.anchor > 0.0

    sums: dict[str, Tensor] = {}

    def add(name: str, value: Tensor) -> None:
        value = value.detach()
        sums[name] = value if name not in sums else sums[name] + value

    values: list[Tensor] = []
    for start in range(0, total, step):
        piece = rows[start : start + step]
        share = len(piece) / total

        def take(field: Tensor) -> Tensor:
            return field[piece].to(device)

        slots, value, value_logits, quantiles = policy.distributions(
            take(batch.buffer), take(batch.mask)
        )
        loss = torch.zeros((), device=device)
        if with_value:
            target = take(batch.value_target)
            if quantiles is None:
                # Cross-entropy against the one-hot eventual winner, exactly
                # `hexn.ppo.minibatch_terms`'s win-head term: every seat is a
                # legal "winner", so there is no mask to apply and this reads
                # `value_logits` raw rather than through `masked_log_softmax`.
                value_loss = -(target * torch.log_softmax(value_logits, dim=-1)).sum(-1).mean()
            else:
                # `players x Q` predictions against the same one-hot target --
                # untested under distillation exactly as it is under PPO
                # (`hexn.ppo`'s module docstring): the quantile shape was
                # designed for a continuous margin target, not a categorical
                # one.
                value_loss = policy.net.value.loss(quantiles, target)
            loss = loss + config.value_coefficient * share * value_loss
            add("value_loss", share * value_loss)
            with torch.no_grad():
                # Logged, never differentiated: the column comparable across a
                # value-head ablation, whatever the loss actually differentiated.
                add("value_mse", share * (value - target).pow(2).mean())
            values.append(value.detach())
        if with_policy:
            # Normalised by the weight, not the row count: a minibatch where
            # one row in twenty is contested must produce the same gradient
            # scale as a full one, or the effective learning rate falls with
            # the filter's selectivity.
            policy_loss = (
                take(batch.policy_weight) * -(take(batch.slot_target) * slots).sum(-1)
            ).sum() / weight_total
            loss = loss + policy_loss
            add("policy_loss", policy_loss)
        if anchored:
            penalty = (
                take(batch.anchor_weight) * -(take(batch.anchor_slots) * slots).sum(-1)
            ).sum() / anchor_total
            loss = loss + config.anchor * penalty
            add("anchor", penalty)
        if tethered:
            if batch.prior_log_probs is None:
                raise ValueError("prior_kl is set but the batch carries no start policy (attach_prior)")
            # Over the legal entries only, as `hexn.ppo.minibatch_terms` does:
            # the masked ones hold ~NEG in both rows.
            divergence = torch.where(
                take(batch.mask), slots.exp() * (slots - take(batch.prior_log_probs)), 0.0
            ).sum(-1).mean()
            loss = loss + config.prior_kl * share * divergence
            add("prior_kl", share * divergence)
        loss.backward()

        if with_value:
            with torch.no_grad():
                add("entropy", share * _entropy(slots).mean())
                # Top-1 agreement with the search, which is the number to
                # watch: the cross-entropy falls whenever the policy sharpens
                # anywhere, and only this says it sharpened toward the right
                # option. Read off the value pass, the one that sees every row.
                hit = (slots.argmax(-1) == take(batch.slot_target).argmax(-1)).float()
                trained = take(batch.policy_weight)
                add("hits", hit.sum())
                add("hits_trained", (hit * trained).sum())
                add("hits_settled", (hit * (1.0 - trained)).sum())
    if values:
        sums["value"] = torch.cat(values)
    sums["rows"] = torch.tensor(float(total), device=device)
    sums["trained"] = weights.sum().to(device)
    return sums


def update(
    policy: NetworkPolicy,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    config: DistillConfig,
    *,
    generator: torch.Generator | None = None,
    steps: "Steps | None" = None,
) -> Stats:
    """One distillation update: `config.epochs` passes over `batch`.

    Two shapes, chosen by `pack_contested`.

    Unpacked is the original: one pass, every term on the same minibatch. The
    policy term then rides the value term's sampling, so it sees relatively
    few contested rows a step -- a small-sample gradient estimate taken
    often.

    Packed gives the two terms their own cadence, which is the shape PPG uses
    and for the same reason: their noise scales
    differ by orders of magnitude, so one minibatch cannot serve both. The value
    pass keeps every row at `minibatch`, and **carries the anchor**, because it
    is the pass that moves the trunk and holding the policy while it does is
    what the anchor is for. The policy pass then walks the contested rows
    densely -- the same rows, seen the same `epochs` times each, at a fraction
    of the per-step variance.

    Either shape steps once per minibatch, whatever `micro_batch` slices it
    into (`_backward`). The `prior_kl` tether rides the value pass, beside the
    anchor, for the anchor's reason.

    With `steps` (`hexn.steps.Steps`) the update's progress goes to disk after
    its optimiser steps, and an update `steps` was opened to resume carries on
    from the step it holds -- both passes count as steps.
    """
    if config.prior_kl > 0.0 and batch.prior_log_probs is None:
        # A weight with nothing to weigh would train as if it were zero and
        # log a zero divergence, which reads as "the policy has not moved".
        raise ValueError("prior_kl is set but the batch carries no start policy (attach_prior)")
    # A micro-batched update leaves the batch where it is (the host, as
    # `assemble` builds it) and moves each slice's rows to the device as it
    # is used, so the device holds one slice rather than the whole batch.
    if not (config.micro_batch and config.micro_batch < config.minibatch):
        batch = batch.to(policy.device)
    size = len(batch)
    home = batch.buffer.device

    # Every list below holds device tensors, not Python floats: converting one
    # with `float(...)` inside the minibatch loop is a device->host sync
    # (`hexn.ppo.Terms`'s docstring). `_stack_mean` and `_ratio` below do the
    # one read each needs, once per update rather than once per minibatch.
    # One dict, so a step file can carry them (`hexn.steps`).
    g: dict = {
        name: []
        for name in (
            "slot_losses", "value_losses", "value_mses", "anchor_reports",
            "prior_kls", "entropies", "agreements", "contested_hits",
            "contested_rows_seen", "settled_hits", "settled_rows_seen",
            "grad_norms",
        )
    }
    g["variance"] = None
    cursor = Cursor(generator, steps, policy, optimiser, lambda: g)
    held = cursor.restored(policy.device)
    if held is not None:
        g = held
        if g["variance"] is not None:
            value, rows = g["variance"]
            g["variance"] = (value, rows.to(home))

    contested_rows = torch.nonzero(batch.policy_weight > 0.0).squeeze(-1)
    packed = config.pack_contested and len(contested_rows) > 0

    def step(rows: Tensor, *, with_value: bool, with_policy: bool) -> dict[str, Tensor]:
        optimiser.zero_grad(set_to_none=True)
        sums = _backward(
            policy, batch, rows, config, with_value=with_value, with_policy=with_policy
        )
        g["grad_norms"].append(
            torch.nn.utils.clip_grad_norm_(policy.net.parameters(), config.max_grad_norm).detach()
        )
        optimiser.step()
        return sums

    for _ in range(config.epochs):
        # The value pass. Every row, unbiased -- restricting it to contested rows
        # would train the value head on a biased slice of the state distribution,
        # which is a different experiment.
        for rows in cursor.passes(size, config.minibatch):
            rows = rows.to(home)
            sums = step(rows, with_value=True, with_policy=not packed)
            if not packed:
                g["slot_losses"].append(sums["policy_loss"])
            g["value_losses"].append(sums["value_loss"])
            g["value_mses"].append(sums["value_mse"])
            g["anchor_reports"].append(sums.get("anchor", torch.zeros_like(sums["rows"])))
            if "prior_kl" in sums:
                g["prior_kls"].append(sums["prior_kl"])
            g["entropies"].append(sums["entropy"])
            g["agreements"].append(sums["hits"] / sums["rows"])
            # Split by which side of the filter the row fell on. Pooled, a
            # fall cannot be read: the contested rows are being pulled toward
            # the search while the settled ones are held by nothing, and those
            # move opposite ways. Weighted by rows, so an empty split simply
            # contributes nothing.
            g["contested_hits"].append(sums["hits_trained"])
            g["contested_rows_seen"].append(sums["trained"])
            g["settled_hits"].append(sums["hits_settled"])
            g["settled_rows_seen"].append(sums["rows"] - sums["trained"])
            g["variance"] = (sums["value"], rows)
            cursor.stepped()

        if not packed:
            continue

        # The policy pass. Dense contested rows, one visit each per epoch, so a
        # row is trained on exactly as often as it would be unpacked.
        for chosen in cursor.passes(len(contested_rows), config.minibatch):
            rows = contested_rows[chosen.to(contested_rows.device)]
            sums = step(rows, with_value=False, with_policy=True)
            g["slot_losses"].append(sums["policy_loss"])
            cursor.stepped()
    cursor.finish()
    predicted_for_variance = g["variance"]

    with torch.no_grad():
        if predicted_for_variance is None:
            variance = 0.0
        else:
            predicted, rows = predicted_for_variance
            variance = _explained_variance(
                predicted[:, 0], batch.value_target[rows][:, 0].to(predicted.device)
            )

    def _ratio(hits: list[Tensor], seen: list[Tensor]) -> float:
        """Hits over rows across the whole update, read back once; 0.0 when
        the split never held a row."""
        if not seen:
            return 0.0
        rows = float(torch.stack(seen).double().sum())
        return float(torch.stack(hits).double().sum()) / rows if rows > 0 else 0.0

    return Stats(
        positions=size,
        policy_loss=_stack_mean(g["slot_losses"]),
        value_loss=_stack_mean(g["value_losses"]),
        value_mse=_stack_mean(g["value_mses"]),
        agreement=_stack_mean(g["agreements"]),
        explained_variance=variance,
        anchor_loss=_stack_mean(g["anchor_reports"]),
        entropy=_stack_mean(g["entropies"]),
        agreement_contested=_ratio(g["contested_hits"], g["contested_rows_seen"]),
        agreement_settled=_ratio(g["settled_hits"], g["settled_rows_seen"]),
        contested_positions=int(len(contested_rows)),
        prior_kl=_stack_mean(g["prior_kls"]),
        grad_norm=_stack_median(g["grad_norms"]),
    )


def measure(policy: NetworkPolicy, batch: Batch, *, chunk: int = 4096) -> dict[str, float]:
    """How far an update moved the policy, read over the batch it trained on.

    Compares the live policy with the one the update started from
    (`Batch.prior_log_probs`, which `hexn.ppo.attach_prior` filled before the
    update) without a gradient, in chunks:

    - `kl_to_start`: the mean over rows of KL(live || start), over legal
      actions, the divergence `prior_kl` penalises;
    - `agreement_start` / `agreement_end`: the share of decision rows (more
      than one legal option, searched) where the policy's argmax is the
      search's -- before and after;
    - `target_ce_start` / `target_ce_end`: the cross-entropy of the search's
      target under each policy on the same rows;
    - `entropy_start` / `entropy_end`: the policies' mean entropy there.

    Forced moves are left out of the agreement, the cross-entropy and the
    entropy: a one-option row agrees with any search and would read as
    agreement that nothing earned.
    """
    if batch.prior_log_probs is None:
        raise ValueError("measure needs the start policy on the batch (attach_prior)")
    device = policy.device
    totals = {
        "kl": 0.0, "rows": 0.0, "decisions": 0.0,
        "hit_start": 0.0, "hit_end": 0.0, "ce_start": 0.0, "ce_end": 0.0,
        "entropy_start": 0.0, "entropy_end": 0.0,
    }
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            stop = min(start + chunk, len(batch))
            mask = batch.mask[start:stop].to(device)
            target = batch.slot_target[start:stop].to(device)
            before = batch.prior_log_probs[start:stop].to(device)
            after, _, _, _ = policy.distributions(batch.buffer[start:stop].to(device), mask)
            decision = (mask.sum(-1) > 1) & (batch.searched[start:stop].to(device) > 0)
            pick = target.argmax(-1)
            kl = torch.where(mask, after.exp() * (after - before), 0.0).sum(-1)
            totals["kl"] += float(kl.double().sum())
            totals["rows"] += stop - start
            totals["decisions"] += float(decision.sum())
            for name, rows in (("start", before), ("end", after)):
                totals[f"hit_{name}"] += float(((rows.argmax(-1) == pick) & decision).sum())
                totals[f"ce_{name}"] += float(
                    torch.where(decision, -(target * rows).sum(-1), 0.0).double().sum()
                )
                totals[f"entropy_{name}"] += float(
                    torch.where(decision, _entropy(rows), 0.0).double().sum()
                )
    decisions = max(totals["decisions"], 1.0)
    return {
        "kl_to_start": totals["kl"] / max(totals["rows"], 1.0),
        "decision_positions": int(totals["decisions"]),
        "agreement_start": totals["hit_start"] / decisions,
        "agreement_end": totals["hit_end"] / decisions,
        "target_ce_start": totals["ce_start"] / decisions,
        "target_ce_end": totals["ce_end"] / decisions,
        "entropy_start": totals["entropy_start"] / decisions,
        "entropy_end": totals["entropy_end"] / decisions,
    }
