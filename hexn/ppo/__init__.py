# SPDX-License-Identifier: GPL-3.0-only
"""PPO over self-play episodes: GAE, the clipped surrogate, and the value loss.

`hexn.selfplay` hands back episodes whose transitions are already demultiplexed
by seat, which is the property this module rests on. A seat's next state is not
the position that follows its action — three other seats act in between — so an
advantage computed over the interleaved stream would be nonsense. Per seat, the
trajectory is an ordinary MDP again and the usual arithmetic applies.

## gamma is 1, and it is not a parameter

`hexn.rewards` explains why at length; the short form is that the reward is
zero-sum, so about half of all terminal values are negative, and discounting
makes a negative terminal cheaper the later it arrives. That pays a losing
policy to stall, and trading in circles is that move available for free. So
there is no `gamma` field on `PPOConfig`, no flag on the trainer, and
`test_the_discount_is_one_and_is_not_configurable` pins it.

`lam` is the knob instead, and it is the safe one: it trades bias against
variance in the *advantage estimator* without changing which policy is optimal.

## The value head is a win head

`hexn.model.Prediction.value` is `softmax(value_logits)` over the seat
axis: a proper win-probability distribution, not a margin. Its target is
`hexn.rewards.win_loss` — the one-hot eventual winner, rotated into each
seat's frame like every other
per-seat vector — and the loss is cross-entropy against it (`minibatch_terms`
below), the natural loss for a categorical label. `advantages` still runs on
`Choice.value[0]`, the mover's own recorded estimate at each decision — now a
probability in `[0, 1]` rather than a points margin, and the terminal payoff
GAE bootstraps against is that seat's own win/loss component, on the same
scale.

The one-hot target has no continuous intermediate to bootstrap a
lambda-return from, so `value_lam` below 1.0 — the old margin head's
lambda-mixed target — is refused outright (`PPOConfig.__post_init__`) rather
than silently reinterpreted; `1.0` (the flat terminal one-hot, replicated
across every transition of the trajectory) is the only legal value.

## The auxiliary VP-margin head

`Prediction.margin` is a second, independent head reading the same trunk
features as the win head, trained (when `aux_margin_weight > 0`) on
`hexn.rewards.reward` — `relative_points`, the margin head's old target —
by plain squared error, added to the loss at that weight. It is never read
by a gate, by `advantages`, or by a search's leaf: `minibatch_terms` is its
only reader, and only for its own term. `aux_margin_weight = 0.0` is the
default: the head is always built, so a checkpoint stays loadable either
way, but a weight of zero multiplies its gradient contribution to exactly
zero rather than merely declining to log it.

## The value loss under a quantile head

`ModelConfig.value_head = "quantile"` changes the win head's loss and
nothing else — `advantages`, the policy-gradient wire and the auxiliary
margin head are all untouched by it. The head emits `players x Q` numbers
whose mean is `value_logits`, and `minibatch_terms` puts the per-seat
quantile Huber loss (`hexn.model.quantile_huber_loss`) on that spread
against the same one-hot winner a scalar head would take cross-entropy
against instead. This combination is untested -- the quantile shape was
designed for a continuous margin target, never a categorical one -- so
treat a quantile-shaped win head as an open ablation, not a result.

`Stats.value_mse` exists for the reading. Two configurations' `value_loss`
columns are on different scales (a pinball loss is roughly `E|u|/2`, a
cross-entropy is unbounded above), so a curve comparison across
configurations needs one number both compute the same way: the plain
squared error of the softmaxed mean against the one-hot target, logged
alongside and never differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from hexset.encoding import to_frame
from ..model import Packing, pack
from ..policy import NetworkPolicy
from ..rewards import relative_points, win_loss
from ..selfplay import Episode, Transition

# Not a configurable. See the module docstring and `hexn.rewards`.
GAMMA = 1.0

# torch's Adam defaults to 1e-8; 1e-5 is the standard PPO value. It matters more
# here than usual: `masked_log_softmax` zeroes the gradient at illegal positions
# and a position offers ~6 legal actions out of 456, so a given logit's row sees
# gradient from a small minority of a minibatch. At 1e-8 Adam normalises those
# few tiny, noisy gradients up to a near-full +/-lr step, so rarely-legal logits
# random-walk at the full learning rate. 1e-5 damps exactly that.
ADAM_EPS = 1e-5


@dataclass(frozen=True)
class PPOConfig:
    lam: float = 0.95
    # The one value the win head's target has ever been fit against
    # (`hexn.model.Prediction`'s win-head registration): `"win"` is the only
    # legal value. Recorded on every run's config anyway, the way `--gamma`
    # is deliberately *not* -- the codebase's convention is that a trained
    # choice is an explicit, checked flag rather than an inferred constant,
    # even where there is currently only one legal answer.
    reward: str = "win"
    # 1.0 keeps the terminal outcome as the value target. The win head's
    # target is the one-hot eventual winner, a categorical label with no
    # bootstrapped intermediate to mix in, so anything below 1.0 is refused
    # (`__post_init__`) rather than silently reinterpreted; see the module
    # docstring's account of the continuous-return lambda-return this flag
    # mixed in before the win head.
    value_lam: float = 1.0
    # The auxiliary VP-margin head's loss weight (`hexn.model.Prediction.
    # margin`, trained on `hexset.victory.relative_points`). `0.0` is the
    # default: the head is always built (checkpoints stay loadable either
    # way) but contributes nothing to the loss, and nothing it computes ever
    # reaches the advantage or a trade gate -- both read `Prediction.value`
    # alone.
    aux_margin_weight: float = 0.0
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch: int = 1024
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    # Shared across seats by default; carried on the config
    # (rather than passed straight to the optimiser, as `hexn.loop` does)
    # so `hexn.league` can vary it per learner like every other knob.
    adam_eps: float = ADAM_EPS
    # Which wires connect the value head to learning. The head touches training
    # through exactly two: it prices each decision (GAE turns its estimates
    # into advantages) and its loss gradient shapes the shared trunk. "gae" is
    # the default — both wires. "none" cuts both: the advantage is the
    # seat's terminal return and no value term enters the loss, which is
    # REINFORCE with the zero-sum reward as an opponent baseline. "aux" keeps
    # the trunk-shaping wire and cuts the pricing one: the head trains exactly
    # as under "gae" while the policy gradient never reads it. The head module
    # is built in every mode so checkpoints stay loadable and the duel tooling
    # is untouched; the modes ablate the wires, not the module.
    critic: str = "gae"
    # Stop taking epochs once a finished epoch's mean `approx_kl` exceeds this;
    # 0 disables. A ceiling, not a target — it can only remove epochs, never
    # raise a rate. A controller steering *toward* the threshold from below
    # is a failure mode a one-sided break cannot reproduce. The default is a
    # tuned constant, not inherited from elsewhere.
    kl_break: float = 0.0
    # The board-paired advantage baseline. Each seat's terminal payoff on
    # the policy-gradient wire becomes r - (r+r')/2,
    # where r' is the same seat's reward in the mate game `index ^ 1` of a
    # board pair. Given (board, seat, policy) — which is what paired dealing
    # plus the paired caster hold fixed — r' is independent of this game's
    # actions, so the subtraction is a valid control variate and the gradient
    # stays unbiased. The value target keeps the RAW terminal: the baseline
    # changes what the policy is paid, never what the head is trained toward.
    # Off by default, so existing batch assembly is unchanged.
    pair_baseline: bool = False

    def __post_init__(self) -> None:
        if self.reward != "win":
            raise ValueError(
                f"unknown reward {self.reward!r}: the value head trains on "
                "the one-hot eventual winner and has no other supported mode"
            )
        if self.critic not in ("gae", "none", "aux"):
            raise ValueError(f"unknown critic mode {self.critic!r}")
        if self.critic != "gae" and self.value_lam < 1.0:
            raise ValueError(
                "value_lam below 1 mixes the head's estimates into the value "
                "target; run it only with critic='gae' so one flag is one wire"
            )
        if self.value_lam < 1.0:
            raise ValueError(
                "value_lam below 1 lambda-mixes the head's own estimates into "
                "a continuous bootstrap target; the win head's target is the "
                "categorical one-hot winner and has no such target to mix"
            )
        if self.aux_margin_weight < 0.0:
            raise ValueError(
                f"aux_margin_weight cannot be negative, got {self.aux_margin_weight}"
            )


@dataclass(frozen=True)
class Stats:
    """What a single update did, for the run log rather than for the maths."""

    positions: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    # Everything below defaults, so `hexn.ddp` and `hexn.exit` keep
    # constructing this positionally without knowing about the new gauges.
    #
    # `approx_kl` above is the mean over every minibatch of every epoch, which
    # is not a step size and is not comparable to the conventional 0.01-0.02
    # band, since that band is quoted for the *end* of an update: an epoch
    # average understates a finished update's actual divergence, because KL
    # accumulates monotonically as each epoch moves further off the batch's
    # original policy. The three fields below are the readable versions.
    #
    # `approx_kl_first_minibatch` is the diagnostic that matters most: on a
    # genuinely on-policy batch it must be ~0, because nothing has stepped
    # yet. If a production run logs a materially larger number, the batch is
    # off-policy before the update starts — which is what
    # `lanes / games_per_iteration` predicts, that ratio being how many
    # iterations a game survives in flight.
    approx_kl_first_minibatch: float = 0.0
    approx_kl_last_epoch: float = 0.0
    clip_fraction_last_epoch: float = 0.0
    # The pre-clip global norm, median over the update's steps.
    # `clip_grad_norm_` has always returned it and all three call sites threw
    # it away, which is why "is max_grad_norm binding?" was unanswerable
    # from the logs. (Note that under Adam a *constant* rescale of the
    # gradient cannot change the step anyway, since m/sqrt(v) is invariant
    # to it.)
    grad_norm: float = 0.0
    # Logged beside `explained_variance` so a shrinking denominator can never
    # again be misread as an improving head: EV against a terminal return is
    # bounded well below 1 for this target, which a shrinking denominator
    # can make look arbitrarily good even as the numerator stays flat.
    value_target_variance: float = 0.0
    # The rate Adam actually stepped with. Without it, a resumed run's
    # `--learning-rate` can be silently overridden by state carried in the
    # checkpoint and nothing in the log would show it.
    lr: float = 0.0
    # Epochs the update actually took. Equal to `config.epochs` unless
    # `kl_break` stopped it early; under a break regime this column is the
    # knob's own telemetry, and the threshold gets re-derived from it.
    epochs_taken: int = 0
    # The squared error of the *mean* the rest of the system reads as `V`,
    # whatever the value head's shape and whatever loss was differentiated.
    # Equal to `value_loss` under every scalar head; under the quantile head
    # `value_loss` is the pinball loss and this is the only column on which
    # two configurations can be compared directly. Never differentiated.
    value_mse: float = 0.0
    # The auxiliary VP-margin head's own MSE against `relative_points`,
    # logged unconditionally (even at `aux_margin_weight=0.0`, where the head
    # trains on nothing but still reads as an untrained head's error) so a
    # run that later turns the weight on has a baseline to compare against.
    margin_loss: float = 0.0


def advantages(
    values: np.ndarray, terminal: float, lam: float
) -> np.ndarray:
    """GAE over one seat's trajectory, with gamma fixed at 1.

    `values` is the seat's own value estimate at each of its decisions, in
    order. `terminal` is the reward the game ended on, from this seat's point of
    view. The reward is zero at every step but the last, so the residual is just
    the change in the seat's own estimate, and only the final step sees a payoff.
    """
    steps = len(values)
    out = np.zeros(steps, dtype=np.float32)
    running = 0.0
    for t in reversed(range(steps)):
        # The bootstrap is the *next* estimate, except at the end of the game
        # where there is no next state and the payoff arrives instead.
        nxt = values[t + 1] if t + 1 < steps else 0.0
        payoff = terminal if t + 1 == steps else 0.0
        delta = payoff + GAMMA * nxt - values[t]
        running = delta + GAMMA * lam * running
        out[t] = running
    return out


@dataclass
class Batch:
    """One update's worth of positions, flattened out of many episodes."""

    buffer: Tensor
    mask: Tensor
    chosen: Tensor
    log_prob: Tensor
    advantage: Tensor
    # The win head's target: the one-hot eventual winner, rotated into each
    # transition's own seat frame, replicated across every decision of the
    # trajectory (there is no bootstrapped intermediate for a categorical
    # label to mix towards -- see the module docstring).
    value_target: Tensor
    # The auxiliary margin head's target: `relative_points`, rotated the same
    # way. Read only by `minibatch_terms`'s aux term; never by the advantage.
    margin_target: Tensor

    def __len__(self) -> int:
        return self.buffer.shape[0]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "buffer",
                    "mask",
                    "chosen",
                    "log_prob",
                    "advantage",
                    "value_target",
                    "margin_target",
                )
            }
        )


def assemble(
    episodes: Sequence[Episode], layout: Packing, config: PPOConfig
) -> Batch:
    """Flatten finished episodes into one update's tensors.

    Only finished episodes: a game still in flight has no terminal reward, and
    the value head is never bootstrapped, so there is nothing to learn from a
    partial trajectory. `Collector.collect` returns exactly the finished ones.
    """
    observations = []
    masks: list[np.ndarray] = []
    chosen: list[int] = []
    log_probs: list[float] = []
    advantage_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    margin_targets: list[np.ndarray] = []

    # The pair baseline needs every game's mate in the same batch: a game is a
    # pure function of (seed, index), so the pairing key is the same pair.
    # `owned` keeps every episode (it empties seats, never drops games), so a
    # learner's slice of a paired cohort still carries both halves.
    dealt = (
        {(e.seed, e.index): win_loss(e.outcome) for e in episodes}
        if config.pair_baseline
        else {}
    )

    for episode in episodes:
        wins = win_loss(episode.outcome)
        margins = relative_points(episode.outcome.points)
        mate = None
        if config.pair_baseline:
            mate = dealt.get((episode.seed, episode.index ^ 1))
            if mate is None:
                raise ValueError(
                    f"game {episode.index} (seed {episode.seed}) has no mate "
                    f"{episode.index ^ 1} in this batch; pair_baseline needs "
                    "paired dealing (pair_boards) and an even cohort of "
                    "complete pairs"
                )
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            seen = to_frame(wins, seat)
            # Component 0 is this seat's own payoff, because `to_frame` puts it
            # first — the same convention the encoder and the value head use.
            own = np.float32(seen[0])
            if mate is not None:
                # (r - r')/2 is algebraically r - (r + r')/2 in exact
                # arithmetic, computed in the form whose float rounding makes
                # the two halves' adjusted payoffs bit-exact negatives —
                # negating a difference is exact, subtracting a rounded
                # midpoint is not. The raw `seen` still feeds `target` below:
                # only the policy-gradient terminal moves.
                own = np.float32((seen[0] - mate[seat]) / 2)
            if config.critic == "gae":
                # Raise rather than substitute 0.0 for a missing estimate. A single
                # zero mid-chain corrupts two GAE residuals directly and ~20 more
                # through `running`, silently — and it is reachable: `expert`'s
                # search policy returns `()` on forced moves and its values are in
                # *board* order rather than the mover's frame, so putting a
                # search-collected episode through this loop would read seat 0's
                # estimate for every seat and never complain. Today only
                # `NetworkPolicy` fills recorded seats, which always emits all
                # `players` components in the mover's frame.
                for transition in trajectory:
                    if not transition.value:
                        raise ValueError(
                            "a recorded transition has no value estimate; GAE cannot "
                            "substitute one without silently corrupting the chain"
                        )
                estimates = np.array(
                    [t.value[0] for t in trajectory],
                    dtype=np.float32,
                )
                advantage_blocks.append(advantages(estimates, own, config.lam))
            else:
                # REINFORCE: every decision in the game is credited with the
                # seat's terminal return, whole. Not GAE-with-zeros — with V≡0
                # the recursion yields lam**(T-t) * G, a horizon discount no
                # one asked for — the return itself, at every step. The
                # zero-sum reward already subtracts the opponents' mean and the
                # per-minibatch normalisation in `update` is unchanged, so this
                # differs from "gae" in exactly one thing: what the head
                # contributes to the gradient.
                advantage_blocks.append(
                    np.full(len(trajectory), own, dtype=np.float32)
                )

            target = np.asarray(seen, dtype=np.float32)
            margin_target = np.asarray(to_frame(margins, seat), dtype=np.float32)
            for transition in trajectory:
                observations.append(transition.observation)
                masks.append(transition.mask)
                chosen.append(transition.index)
                log_probs.append(transition.log_prob)
                targets.append(target)
                margin_targets.append(margin_target)

    if not observations:
        raise ValueError("no transitions to learn from")

    return Batch(
        buffer=pack(layout, observations),
        mask=torch.from_numpy(np.stack(masks)),
        chosen=torch.tensor(chosen, dtype=torch.int64),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        advantage=torch.from_numpy(np.concatenate(advantage_blocks)),
        value_target=torch.from_numpy(np.stack(targets)),
        margin_target=torch.from_numpy(np.stack(margin_targets)),
    )


def _minibatches(
    size: int, minibatch: int, generator: torch.Generator | None
) -> Iterator[Tensor]:
    """A shuffled partition of `size` rows into chunks of at most `minibatch`.

    **A trailing chunk of exactly one row is folded into its predecessor**, and
    that is not tidiness. One row cannot be advantage-normalised: `Tensor.std()`
    defaults to `correction=1`, so a single row divides by zero and returns nan,
    the `+ 1e-8` in the caller does not rescue a nan, `clip_grad_norm_`
    propagates it, and `optimiser.step()` then writes nan into every parameter
    while the run carries on logging. Since `positions` varies per iteration,
    a remainder of exactly one row is possible on any given iteration and
    would silently corrupt the whole run from that point forward -- this
    guard exists so that whether it happens is not left to chance.

    All three update paths — this module's, `hexn.ddp`'s and `hexn.exit`'s
    — draw their minibatches here, so the guard lands once for all of them.
    """
    order = torch.randperm(size, generator=generator)
    bounds = [(start, min(start + minibatch, size)) for start in range(0, size, minibatch)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    for start, stop in bounds:
        yield order[start:stop]


@dataclass(frozen=True)
class Terms:
    """One minibatch's PPO arithmetic: the loss to differentiate plus the
    numbers the log wants. One function produces it (`minibatch_terms`) and
    both the single-device update below and the sharded one in `hexn.ddp`
    consume it, so the two cannot drift apart."""

    loss: Tensor
    value: Tensor
    # The four summands of `loss`, still attached to the graph. `loss` is what
    # the update differentiates; these exist so a caller can differentiate one
    # term at a time — which is the only way to ask whether the policy or the
    # value head dominates the shared trunk, an open question four separate
    # audits answered four different ways from the loss magnitudes alone.
    policy_term: Tensor
    value_term: Tensor
    entropy_term: Tensor
    # The auxiliary VP-margin head's own term, `aux_margin_weight * margin_loss`
    # -- zero exactly (not merely small) at the default weight, since a
    # multiply by the Python float `0.0` zeroes the gradient into the head and
    # the trunk alike.
    margin_term: Tensor
    # The scalar reads below all stay device tensors (detached, no graph)
    # rather than Python floats. Converting each one with `float(...)` here
    # would force a device->host sync per minibatch, one per scalar
    # (`hipMemcpyWithStream`), which adds up fast at this shape. `update`
    # (and `hexn.exit.update`, `hexn.ddp`'s workers) now accumulate these
    # as tensors across an epoch or a whole update and read them back with one
    # `.item()`/`.mean()` each, not one per minibatch.
    policy_loss: Tensor
    value_loss: Tensor
    # The mean's plain squared error, on one scale across head shapes. Equal to
    # `value_loss` under a scalar head, by the same expression.
    value_mse: Tensor
    margin_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def minibatch_terms(
    policy: NetworkPolicy,
    buffer: Tensor,
    mask: Tensor,
    chosen: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    value_target: Tensor,
    margin_target: Tensor,
    config: PPOConfig,
) -> Terms:
    """The clipped surrogate, win-head loss, auxiliary margin loss and
    entropy for one minibatch.

    `advantage` arrives already normalised — the caller owns that, because a
    sharded worker only holds a slice of the minibatch and cannot compute the
    minibatch's own mean and std.
    """
    evaluation = policy.evaluate(buffer, mask, chosen)
    ratio = (evaluation.log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clamped = ratio.clamp(1 - config.clip, 1 + config.clip) * advantage
    policy_loss = -torch.min(unclipped, clamped).mean()
    if evaluation.quantiles is None:
        # Cross-entropy against the one-hot eventual winner: the win head's
        # raw output is a set of logits, never read directly -- only through
        # `masked_log_softmax`'s unmasked cousin here, since every seat is a
        # legal "winner" and there is no mask to apply.
        log_probs_win = torch.log_softmax(evaluation.value_logits, dim=-1)
        value_loss = -(value_target * log_probs_win).sum(-1).mean()
    else:
        # `players x Q` predictions against the same `value_target` one-hot
        # vector. Untested: the quantile shape was designed for a
        # continuous margin target, never a categorical one -- see
        # the module docstring.
        value_loss = policy.net.value.loss(evaluation.quantiles, value_target)
    with torch.no_grad():
        # Logged, never differentiated: the comparable column across
        # configurations, whatever the head's loss actually was.
        value_mse = (evaluation.value - value_target).pow(2).mean()
    # The auxiliary VP-margin head: plain squared error against
    # `relative_points`, weighted separately from the win head's own term and
    # never read by the advantage or a gate. `aux_margin_weight = 0.0`
    # multiplies the gradient into the head (and, through it, the trunk) to
    # exactly zero -- not merely a small number -- since it is a Python float
    # zero times a tensor.
    margin_loss = (evaluation.margin - margin_target).pow(2).mean()
    margin_term = config.aux_margin_weight * margin_loss
    entropy = evaluation.entropy.mean()
    # Under critic="none" the value term is absent from the loss, so no
    # gradient reaches the head or the trunk through it; `value_loss` is still
    # computed and logged, and reads as the untrained head's error.
    value_term = (
        config.value_coefficient * value_loss
        if config.critic != "none"
        else torch.zeros_like(policy_loss)
    )
    loss = (
        policy_loss
        + value_term
        + margin_term
        - config.entropy_coefficient * entropy
    )
    with torch.no_grad():
        # Schulman's low-variance estimator, which unlike the plain log-ratio
        # mean is non-negative and so cannot hide a diverging update behind
        # cancellation.
        log_ratio = evaluation.log_prob - old_log_prob
        # `expm1`, not `ratio - 1`. The estimator is non-negative in exact
        # arithmetic, but in float32 `exp(x)` rounds to 1.0 for any |x| below
        # the ~1.2e-7 epsilon, so the difference collapses to `-log_ratio` and
        # the gauge reports a negative KL for a genuinely on-policy batch,
        # whose log-ratios sit near machine epsilon from reduction order
        # alone. `expm1` is accurate for small arguments, so the floor reads
        # 0 instead of a small impossible negative number, and the
        # correction only matters in that regime -- it leaves any log-ratio
        # above roughly 1e-3 unchanged.
        kl = (torch.expm1(log_ratio) - log_ratio).mean()
        clipped = (ratio - 1).abs().gt(config.clip).float().mean()
    return Terms(
        loss=loss,
        value=evaluation.value.detach(),
        policy_term=policy_loss,
        value_term=value_term,
        entropy_term=-config.entropy_coefficient * entropy,
        margin_term=margin_term,
        # Detached, not read back to Python here -- `kl` and `clipped` are
        # already no-grad tensors from the block above. `update` (and
        # `hexn.exit.update`, `hexn.ddp`'s workers) accumulate these
        # across minibatches and read them back once.
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        value_mse=value_mse.detach(),
        margin_loss=margin_loss.detach(),
        entropy=entropy.detach(),
        approx_kl=kl,
        clip_fraction=clipped,
    )


def _stack_mean(values: Sequence[Tensor]) -> float:
    """The mean of one update's per-minibatch device tensors, read back to a
    Python float once instead of once per minibatch.

    Accumulated in float64 before the single read, matching what `np.mean`
    over the old per-minibatch Python floats (each a float32 value promoted to
    double on conversion) would have produced -- the reduction order differs
    from numpy's pairwise summation, but every reader of these numbers already
    compares them to a tolerance (`tests/hexn/test_ppo.py`,
    `tests/hexn/test_ddp.py`), not bit-for-bit.
    """
    if not values:
        return 0.0
    return float(torch.stack(list(values)).double().mean())


def _stack_median(values: Sequence[Tensor]) -> float:
    """`np.median`'s linear-interpolation convention, over device tensors read
    back once. `torch.quantile(0.5)` uses the same interpolation numpy's
    default `median` does, unlike `torch.median` -- which for an even count
    returns the lower of the two middle values rather than their average."""
    if not values:
        return 0.0
    return float(torch.stack(list(values)).double().quantile(0.5))


def _explained_variance(predicted: Tensor, actual: Tensor) -> float:
    """1 - Var(residual)/Var(actual), the standard read on a value head.

    Zero means the head is no better than predicting the mean, which is the
    number to watch early: a value head stuck at zero explained variance is the
    single clearest sign a run is not learning.
    """
    variance = actual.var()
    if variance < 1e-8:
        return 0.0
    return float(1 - (actual - predicted).var() / variance)


def update(
    policy: NetworkPolicy,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    config: PPOConfig,
    *,
    generator: torch.Generator | None = None,
) -> Stats:
    """One PPO update: `config.epochs` passes over `batch` in minibatches."""
    batch = batch.to(policy.device)
    size = len(batch)

    policy_losses: list[Tensor] = []
    value_losses: list[Tensor] = []
    entropies: list[Tensor] = []
    kls: list[Tensor] = []
    clipped: list[Tensor] = []
    value_mses: list[Tensor] = []
    margin_losses: list[Tensor] = []
    grad_norms: list[Tensor] = []
    epoch_kls: list[list[Tensor]] = []
    epoch_clips: list[list[Tensor]] = []
    predicted_for_variance = None

    epochs_taken = 0
    for _ in range(config.epochs):
        epoch_kls.append([])
        epoch_clips.append([])
        for rows in _minibatches(size, config.minibatch, generator):
            rows = rows.to(policy.device)
            advantage = batch.advantage[rows]
            # Normalised per minibatch, which is what makes one clip range work
            # across a run whose reward scale is fixed but whose advantage
            # spread collapses as the value head improves. `_minibatches`
            # guarantees at least two rows, without which `std()` returns nan.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            terms = minibatch_terms(
                policy,
                batch.buffer[rows],
                batch.mask[rows],
                batch.chosen[rows],
                batch.log_prob[rows],
                advantage,
                batch.value_target[rows],
                batch.margin_target[rows],
                config,
            )

            optimiser.zero_grad(set_to_none=True)
            terms.loss.backward()
            # The return value is the norm *before* clipping, which is the only
            # way to know whether `max_grad_norm` is binding. Kept, not
            # dropped, and kept as a tensor -- read back once, with the rest of
            # the update's gauges, in `_stack_median` below.
            grad_norms.append(
                torch.nn.utils.clip_grad_norm_(
                    policy.net.parameters(), config.max_grad_norm
                ).detach()
            )
            optimiser.step()

            policy_losses.append(terms.policy_loss)
            value_losses.append(terms.value_loss)
            value_mses.append(terms.value_mse)
            margin_losses.append(terms.margin_loss)
            entropies.append(terms.entropy)
            kls.append(terms.approx_kl)
            clipped.append(terms.clip_fraction)
            epoch_kls[-1].append(terms.approx_kl)
            epoch_clips[-1].append(terms.clip_fraction)
            predicted_for_variance = (terms.value, rows)

        epochs_taken += 1
        # The break reads the epoch that just finished, so the damage an over-
        # long update does is bounded at one epoch past the ceiling rather
        # than three — the two recorded blowouts were fine at epoch 1 and
        # diverging by epoch 4. This is the one legitimate per-epoch sync: the
        # decision to keep taking epochs has to be made in Python, and
        # `config.kl_break > 0` short-circuits it away entirely at the
        # every-run-on-record default of 0.
        if config.kl_break > 0 and _stack_mean(epoch_kls[-1]) > config.kl_break:
            break

    with torch.no_grad():
        if predicted_for_variance is None:
            variance = 0.0
        else:
            predicted, rows = predicted_for_variance
            variance = _explained_variance(
                predicted[:, 0], batch.value_target[rows][:, 0]
            )

    return Stats(
        positions=size,
        policy_loss=_stack_mean(policy_losses),
        value_loss=_stack_mean(value_losses),
        entropy=_stack_mean(entropies),
        approx_kl=_stack_mean(kls),
        clip_fraction=_stack_mean(clipped),
        explained_variance=variance,
        approx_kl_first_minibatch=(
            float(epoch_kls[0][0]) if epoch_kls[0] else 0.0
        ),
        approx_kl_last_epoch=_stack_mean(epoch_kls[-1]),
        clip_fraction_last_epoch=_stack_mean(epoch_clips[-1]),
        grad_norm=_stack_median(grad_norms),
        value_target_variance=float(batch.value_target[:, 0].var()),
        lr=float(optimiser.param_groups[0]["lr"]),
        epochs_taken=epochs_taken,
        value_mse=_stack_mean(value_mses),
        margin_loss=_stack_mean(margin_losses),
    )
