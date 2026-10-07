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

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from hexset.encoding import global_columns, to_frame
from hexset.game import Phase
from ..model import Packing, pack, unpack
from ..policy import NetworkPolicy, masked_log_softmax

if TYPE_CHECKING:
    from ..steps import Steps
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
    # Rows per forward/backward inside a minibatch. A minibatch larger than
    # this is stepped through in micro-batches whose gradients accumulate,
    # each weighted by its share of the minibatch's rows, into the one
    # optimiser step the minibatch takes: same step, less device memory.
    # 0 (the default) is the whole minibatch in one pass.
    micro_batch: int = 0
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
    # Weight on KL(pi || prior): the learner's divergence from a fixed prior
    # policy over every legal action, averaged over the minibatch's rows and
    # added to the loss. The prior's log-probabilities ride on the batch
    # (`attach_prior`), taken once per update from a frozen network, so the
    # term costs no second forward per minibatch. The direction is the
    # learner's own: it is the learner's probability mass that is charged for
    # sitting where the prior puts little, which holds a warm-started policy
    # near the play it was initialised from without pulling it toward every
    # move the prior merely tolerates. 0.0, the default, adds nothing and
    # takes no prior.
    prior_kl: float = 0.0
    # Pay for fast wins: `fast_win[r - 1]` is what a win in round `r` is worth
    # (the last entry holds for every later round), and a loss or an unfinished
    # game is worth zero whenever it ends -- so a losing seat gains nothing by
    # stalling. The payoff reaches the policy through the fast-win head
    # (`hexn.model.Prediction.fast`), which the net must be built with; the
    # win head keeps its one-hot target. Empty, the default, pays every win 1.
    fast_win: tuple[float, ...] = ()
    # Pay `vp_reward` for every victory point a seat gains, at the decision
    # after it lands (and charge it for one lost, a road or army award
    # passing), on top of the win payoff. Over a game that sums to
    # `vp_reward * (final - opening points)`, so it is part of the objective,
    # weighted to stay well below a win. The critic for the extra return is
    # the margin head, retargeted to final points / 10 (`assemble`), trained
    # at `aux_margin_weight`, and recorded into the value by the policy
    # (`hexn.policy.NetworkPolicy.vp_reward`). 0.0, the default, pays nothing.
    vp_reward: float = 0.0
    # Leave out of the batch every position whose mover's own recorded win
    # estimate (`Transition.value[0]`) is below `decided_cut` or above
    # `1 - decided_cut`: the game is settled for that seat, whatever it plays.
    # The position still prices its neighbours -- GAE runs over the whole
    # trajectory first -- and only then is the row dropped, from every loss
    # term. 0.0, the default, keeps every position.
    decided_cut: float = 0.0
    # Every setup-phase position (`SETUP_SETTLEMENT`, `SETUP_ROAD`) enters the
    # batch this many times, so the opening carries that many times its weight
    # in every loss term. 1, the default, enters it once.
    setup_weight: int = 1
    # The update's forward and backward under autocast in this dtype, with a
    # gradient scaler stepping the optimiser (`update`'s `scaler`). "" (the
    # default) is fp32 throughout. "fp16" is the only other value: the network's
    # outputs are cast back to fp32 before the log-softmax and every loss term
    # (`NetworkPolicy.evaluate`), so only the trunk and heads run in half.
    # bfloat16 is refused: its 7-bit mantissa moves the recomputed log-probs
    # far enough from the recorded ones to clip positions before any step.
    amp: str = ""

    def __post_init__(self) -> None:
        if self.amp not in ("", "fp16"):
            raise ValueError(f"amp must be '' or 'fp16', got {self.amp!r}")
        if not 0.0 <= self.decided_cut < 0.5:
            raise ValueError(f"decided_cut must lie in [0, 0.5), got {self.decided_cut}")
        if self.decided_cut and (self.vp_reward or self.fast_win):
            raise ValueError(
                "decided_cut reads the recorded win estimate; under vp_reward or "
                "fast_win the recorded value is not one"
            )
        if self.setup_weight < 1:
            raise ValueError(f"setup_weight must be at least 1, got {self.setup_weight}")
        if self.micro_batch < 0:
            raise ValueError(f"micro_batch cannot be negative, got {self.micro_batch}")
        if self.vp_reward < 0.0:
            raise ValueError(f"vp_reward cannot be negative, got {self.vp_reward}")
        if self.vp_reward and (self.aux_margin_weight <= 0.0 or self.critic != "gae"):
            raise ValueError(
                "vp_reward's critic is the margin head: it needs aux_margin_weight > 0 and critic='gae'"
            )
        if self.vp_reward and self.pair_baseline:
            raise ValueError("vp_reward is not wired into the pair baseline")
        if any(not 0.0 <= w <= 1.0 for w in self.fast_win):
            raise ValueError(f"fast_win weights must lie in [0, 1], got {self.fast_win}")
        if self.fast_win and self.critic != "gae":
            raise ValueError("fast_win prices wins through the fast-win head; run it with critic='gae'")
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
        if self.prior_kl < 0.0:
            raise ValueError(f"prior_kl cannot be negative, got {self.prior_kl}")


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
    # Mean KL(pi || prior) over the update's minibatches, whenever the batch
    # carries a prior (`attach_prior`); 0.0 when it does not. The distance the
    # policy has walked from the prior, read directly rather than inferred
    # from ladder results against it.
    prior_kl: float = 0.0
    # The fast-win head's cross-entropy (`PPOConfig.fast_win`); 0.0 without it.
    fast_loss: float = 0.0
    # Under `amp`: optimiser steps the gradient scaler skipped (an inf or nan
    # in the scaled gradients), and its scale when the update finished.
    amp_skipped: int = 0
    amp_scale: float = 0.0


def win_round(outcome, players: int) -> int:
    """The round a game ended in: `outcome.turns` counts every seat's turns,
    so round `r` holds turns `players * (r - 1) + 1` through `players * r`."""
    return max(1, math.ceil(outcome.turns / players))


def fast_weight(curve: Sequence[float], outcome, players: int) -> float:
    """What `outcome`'s win is worth under `curve` (`PPOConfig.fast_win`):
    the entry for its round, the last entry past the curve's end, and 0.0 for
    a game with no winner."""
    if outcome.winner is None:
        return 0.0
    return float(curve[min(win_round(outcome, players), len(curve)) - 1])


def advantages(
    values: np.ndarray, terminal: float, lam: float, rewards: np.ndarray | None = None
) -> np.ndarray:
    """GAE over one seat's trajectory, with gamma fixed at 1.

    `values` is the seat's own value estimate at each of its decisions, in
    order. `terminal` is the reward the game ended on, from this seat's point of
    view. Without `rewards` the reward is zero at every step but the last, so
    the residual is just the change in the seat's own estimate, and only the
    final step sees a payoff; `rewards[t]` adds a reward earned between
    decision t and the next (`PPOConfig.vp_reward`).
    """
    steps = len(values)
    out = np.zeros(steps, dtype=np.float32)
    running = 0.0
    for t in reversed(range(steps)):
        # The bootstrap is the *next* estimate, except at the end of the game
        # where there is no next state and the payoff arrives instead.
        nxt = values[t + 1] if t + 1 < steps else 0.0
        payoff = terminal if t + 1 == steps else 0.0
        if rewards is not None:
            payoff += rewards[t]
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
    # A fixed prior policy's masked log-softmax over every action, one row per
    # position, for `PPOConfig.prior_kl`. `None` unless `attach_prior` filled
    # it, which is every batch a run without a prior ever builds.
    prior_log_probs: Tensor | None = None
    # The fast-win head's target (`PPOConfig.fast_win`): the round weight on
    # the winner's slot in each transition's own frame, the rest on the last
    # slot. `None` for a run that does not pay for speed.
    fast_target: Tensor | None = None

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
            },
            prior_log_probs=(
                None
                if self.prior_log_probs is None
                else self.prior_log_probs.to(device)
            ),
            fast_target=(
                None if self.fast_target is None else self.fast_target.to(device)
            ),
        )


def autocast(device: torch.device | str, amp: str):
    """The update's autocast context for `PPOConfig.amp`: fp16 on `device`'s
    type, or a disabled context that leaves fp32 untouched."""
    kind = torch.device(device).type
    return torch.autocast(kind, dtype=torch.float16, enabled=amp == "fp16")


def attach_prior(
    batch: Batch,
    prior: NetworkPolicy,
    layout: Packing,
    chunk: int = 4096,
    amp: str = "",
) -> Batch:
    """`batch` with `prior`'s masked log-softmax over every row attached.

    One no-grad forward of the prior per update, in chunks, on the prior's own
    device; the minibatch loop then indexes the rows it needs. `layout` is the
    one the batch was packed with, and the prior has to read rows the same
    way -- which any checkpoint of the same board graph and seat count does,
    since the packing depends on neither width nor rounds. `amp` is
    `PPOConfig.amp`: the prior's forward runs under the same autocast as the
    update's, its logits back in fp32 before the log-softmax.
    """
    if prior.layout != layout:
        raise ValueError("the prior packs positions differently from the batch")
    rows = []
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            buffer = batch.buffer[start : start + chunk].to(prior.device)
            mask = batch.mask[start : start + chunk].to(prior.device)
            with autocast(prior.device, amp):
                logits = prior.net(*unpack(layout, buffer)).logits.float()
            rows.append(masked_log_softmax(logits, mask).to(batch.buffer.device))
    return replace(batch, prior_log_probs=torch.cat(rows))


SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


def copies(transition: Transition, phase: slice, config: PPOConfig) -> int:
    """How many times `transition` enters the batch: 0 when its mover's own
    win estimate puts it outside `config.decided_cut`, `config.setup_weight`
    in a setup phase, else 1. `phase` is the observation's phase block
    (`hexset.encoding.global_columns(players)["phase"]`)."""
    if config.decided_cut:
        if not transition.value:
            raise ValueError("decided_cut needs every recorded transition's value estimate")
        own = transition.value[0]
        if own < config.decided_cut or own > 1.0 - config.decided_cut:
            return 0
    if config.setup_weight != 1:
        if int(np.argmax(transition.observation.globals[phase])) in SETUP_PHASES:
            return config.setup_weight
    return 1


def stakes_gauges(episodes: Sequence[Episode], config: PPOConfig) -> dict[str, float]:
    """What `decided_cut` and `setup_weight` did to one collection: the share
    of recorded positions dropped, the share of the dropped ones whose mover
    went on to win, on each side of the cut (a calibrated head wins about
    `decided_cut` of its low drops and misses about that share of its high
    ones), and the setup positions repeated."""
    recorded = low = low_won = high = high_won = setup = 0
    for episode in episodes:
        winner = episode.outcome.winner
        phase = global_columns(len(episode.outcome.points))["phase"]
        for seat, trajectory in enumerate(episode.trajectories):
            for transition in trajectory:
                recorded += 1
                n = copies(transition, phase, config)
                if n == 0:
                    if transition.value[0] < 0.5:
                        low += 1
                        low_won += winner == seat
                    else:
                        high += 1
                        high_won += winner == seat
                elif n > 1:
                    setup += 1
    return {
        "decided_share": (low + high) / recorded if recorded else 0.0,
        "decided_low_won": low_won / low if low else 0.0,
        "decided_high_won": high_won / high if high else 0.0,
        "setup_repeated": setup,
    }


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
    fast_targets: list[np.ndarray] = []
    counts: list[int] = []
    weighed = bool(config.decided_cut) or config.setup_weight != 1

    def paid(episode: Episode) -> tuple[float, ...]:
        # What each seat is paid: the win/loss vector, scaled by the winning
        # round's weight when the run pays for speed.
        wins = win_loss(episode.outcome)
        if not config.fast_win:
            return wins
        weight = fast_weight(config.fast_win, episode.outcome, len(wins))
        return tuple(weight * x for x in wins)

    # The pair baseline needs every game's mate in the same batch: a game is a
    # pure function of (seed, index), so the pairing key is the same pair.
    # `owned` keeps every episode (it empties seats, never drops games), so a
    # learner's slice of a paired cohort still carries both halves.
    dealt = (
        {(e.seed, e.index): paid(e) for e in episodes}
        if config.pair_baseline
        else {}
    )

    for episode in episodes:
        wins = win_loss(episode.outcome)
        pay = paid(episode)
        # In vp_reward mode the margin head is the critic for the VP still to
        # come, so its target is each seat's final points (/ 10, its scale).
        margins = (
            tuple(p / 10.0 for p in episode.outcome.points)
            if config.vp_reward
            else relative_points(episode.outcome.points)
        )
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
        phase = global_columns(len(wins))["phase"]
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            seen = to_frame(wins, seat)
            earned = to_frame(pay, seat)
            # Component 0 is this seat's own payoff, because `to_frame` puts it
            # first — the same convention the encoder and the value head use.
            own = np.float32(earned[0])
            if mate is not None:
                # (r - r')/2 is algebraically r - (r + r')/2 in exact
                # arithmetic, computed in the form whose float rounding makes
                # the two halves' adjusted payoffs bit-exact negatives —
                # negating a difference is exact, subtracting a rounded
                # midpoint is not. The raw `seen` still feeds `target` below:
                # only the policy-gradient terminal moves.
                own = np.float32((earned[0] - mate[seat]) / 2)
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
                steps_paid = None
                if config.vp_reward:
                    held = [t.points for t in trajectory] + [episode.outcome.points[seat]]
                    steps_paid = config.vp_reward * np.diff(np.asarray(held, dtype=np.float32))
                advantage_blocks.append(advantages(estimates, own, config.lam, steps_paid))
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
            fast_target = (
                np.asarray([*earned, 1.0 - sum(earned)], dtype=np.float32)
                if config.fast_win
                else None
            )
            for transition in trajectory:
                observations.append(transition.observation)
                masks.append(transition.mask)
                chosen.append(transition.index)
                log_probs.append(transition.log_prob)
                targets.append(target)
                margin_targets.append(margin_target)
                if fast_target is not None:
                    fast_targets.append(fast_target)
                if weighed:
                    counts.append(copies(transition, phase, config))

    advantage = np.concatenate(advantage_blocks) if advantage_blocks else None
    if weighed and observations:
        # Every row's advantage is taken over its whole trajectory above;
        # only now are rows dropped or repeated, all fields together.
        rows = np.repeat(np.arange(len(counts)), counts)
        observations = [observations[i] for i in rows]
        masks = [masks[i] for i in rows]
        chosen = [chosen[i] for i in rows]
        log_probs = [log_probs[i] for i in rows]
        targets = [targets[i] for i in rows]
        margin_targets = [margin_targets[i] for i in rows]
        if config.fast_win:
            fast_targets = [fast_targets[i] for i in rows]
        advantage = advantage[rows]

    if not observations:
        raise ValueError("no transitions to learn from")

    return Batch(
        buffer=pack(layout, observations),
        mask=torch.from_numpy(np.stack(masks)),
        chosen=torch.tensor(chosen, dtype=torch.int64),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        advantage=torch.from_numpy(advantage),
        value_target=torch.from_numpy(np.stack(targets)),
        margin_target=torch.from_numpy(np.stack(margin_targets)),
        fast_target=(
            torch.from_numpy(np.stack(fast_targets)) if config.fast_win else None
        ),
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
    for start, stop in _bounds(size, minibatch):
        yield order[start:stop]


def _bounds(size: int, minibatch: int) -> list[tuple[int, int]]:
    """`_minibatches`' chunk boundaries, known without drawing the shuffle."""
    bounds = [(start, min(start + minibatch, size)) for start in range(0, size, minibatch)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


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
    # `config.prior_kl * prior_kl`, attached, and the divergence itself,
    # detached. Both `None` when the minibatch came with no prior, so a caller
    # that never attaches one sees exactly the four summands it always has.
    prior_term: Tensor | None = None
    prior_kl: Tensor | None = None
    # The fast-win head's cross-entropy, detached; `None` without a target.
    fast_loss: Tensor | None = None


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
    prior_log_probs: Tensor | None = None,
    fast_target: Tensor | None = None,
) -> Terms:
    """The clipped surrogate, win-head loss, auxiliary margin loss and
    entropy for one minibatch -- plus, when `prior_log_probs` is given, the
    divergence to that prior (`PPOConfig.prior_kl`), and when `fast_target` is
    given, the fast-win head's cross-entropy, inside the value term.

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
    fast_loss = None
    if fast_target is not None:
        if evaluation.fast_logits is None:
            raise ValueError("a fast-win target needs a net built with fast_win")
        fast_loss = -(
            fast_target * torch.log_softmax(evaluation.fast_logits, dim=-1)
        ).sum(-1).mean()
        value_loss_total = value_loss + fast_loss
    else:
        value_loss_total = value_loss
    margin_loss = (evaluation.margin - margin_target).pow(2).mean()
    margin_term = config.aux_margin_weight * margin_loss
    entropy = evaluation.entropy.mean()
    # Under critic="none" the value term is absent from the loss, so no
    # gradient reaches the head or the trunk through it; `value_loss` is still
    # computed and logged, and reads as the untrained head's error.
    value_term = (
        config.value_coefficient * value_loss_total
        if config.critic != "none"
        else torch.zeros_like(policy_loss)
    )
    loss = (
        policy_loss
        + value_term
        + margin_term
        - config.entropy_coefficient * entropy
    )
    prior_term = prior_kl = None
    if prior_log_probs is not None:
        # Over the legal entries only. The masked ones hold ~NEG in both rows
        # and zero probability under the learner, which is a finite zero in
        # exact arithmetic and not worth trusting to float32.
        log_probs = evaluation.log_probs
        divergence = torch.where(
            mask, log_probs.exp() * (log_probs - prior_log_probs), 0.0
        ).sum(-1)
        prior_kl = divergence.mean()
        prior_term = config.prior_kl * prior_kl
        loss = loss + prior_term
        prior_kl = prior_kl.detach()
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
        prior_term=prior_term,
        prior_kl=prior_kl,
        fast_loss=None if fast_loss is None else fast_loss.detach(),
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


def _terms_for(
    policy: NetworkPolicy, batch: Batch, rows: Tensor, advantage: Tensor, config: PPOConfig
) -> Terms:
    """`minibatch_terms` on `rows` of `batch`, `advantage` already normalised
    and on the policy's device. The rows are gathered where the batch lives
    and moved to the device (a no-op when the batch is already there)."""
    device = policy.device

    def take(field: Tensor | None) -> Tensor | None:
        return None if field is None else field[rows].to(device)

    with autocast(device, config.amp):
        return minibatch_terms(
            policy,
            take(batch.buffer),
            take(batch.mask),
            take(batch.chosen),
            take(batch.log_prob),
            advantage,
            take(batch.value_target),
            take(batch.margin_target),
            config,
            prior_log_probs=take(batch.prior_log_probs),
            fast_target=take(batch.fast_target),
        )


def _backward(loss: Tensor, scaler: "torch.amp.GradScaler | None") -> None:
    """`loss.backward()`, through the gradient scaler when there is one."""
    (loss if scaler is None else scaler.scale(loss)).backward()


def _accumulate(
    policy: NetworkPolicy,
    batch: Batch,
    rows: Tensor,
    advantage: Tensor,
    config: PPOConfig,
    scaler: "torch.amp.GradScaler | None" = None,
) -> Terms:
    """One minibatch in `config.micro_batch`-row slices, gradients accumulated.

    Every term of the loss is a mean over rows, so weighting each slice's loss
    by its share of the minibatch's rows and summing the backwards gives the
    whole minibatch's gradient. The returned gauges are the same row-weighted
    means, so they read as the whole minibatch's; `value` is every slice's,
    in `rows` order. Everything returned is detached: the gradient is taken.
    """
    total = len(rows)
    sums: dict[str, Tensor | None] = {}
    values: list[Tensor] = []
    for start in range(0, total, config.micro_batch):
        piece = slice(start, start + config.micro_batch)
        terms = _terms_for(policy, batch, rows[piece], advantage[piece], config)
        share = len(rows[piece]) / total
        _backward(terms.loss * share, scaler)
        for field in _ACCUMULATED:
            got = getattr(terms, field)
            if got is None:
                sums[field] = None
            else:
                got = share * got.detach()
                sums[field] = got if start == 0 else sums[field] + got
        values.append(terms.value)
    return Terms(value=torch.cat(values), **sums)


# Every `Terms` field but `value`: the row means `_accumulate` weights by share.
_ACCUMULATED = (
    "loss", "policy_term", "value_term", "entropy_term", "margin_term",
    "policy_loss", "value_loss", "value_mse", "margin_loss", "entropy",
    "approx_kl", "clip_fraction", "prior_term", "prior_kl", "fast_loss",
)


def update(
    policy: NetworkPolicy,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    config: PPOConfig,
    *,
    generator: torch.Generator | None = None,
    steps: "Steps | None" = None,
    scaler: "torch.amp.GradScaler | None" = None,
) -> Stats:
    """One PPO update: `config.epochs` passes over `batch` in minibatches.

    With `steps` (`hexn.steps.Steps`) the update's progress goes to disk after
    its optimiser steps, and an update `steps` was opened to resume carries
    on from the step it holds.

    Under `config.amp` the losses are scaled by `scaler`, which the caller
    keeps for the whole run (its scale carries from update to update), and the
    gradients are unscaled before they are clipped; a step whose scaled
    gradients overflowed is skipped and counted in `Stats.amp_skipped`.
    """
    if bool(config.amp) != (scaler is not None):
        raise ValueError("amp and a gradient scaler must come together")
    from ..steps import Cursor

    if bool(config.fast_win) != (batch.fast_target is not None):
        raise ValueError("fast_win and the batch's fast-win target must come together (assemble)")
    if config.prior_kl > 0 and batch.prior_log_probs is None:
        # A weight with nothing to weigh would train as if it were zero and
        # log a zero divergence, which reads as "the policy has not moved".
        raise ValueError("prior_kl is set but the batch carries no prior (attach_prior)")
    # A micro-batched update leaves the batch where it is (the host, as
    # `assemble` builds it) and moves each slice's rows to the device as it
    # is used, so the device holds one slice rather than the whole batch.
    if not (config.micro_batch and config.micro_batch < config.minibatch):
        batch = batch.to(policy.device)
    size = len(batch)

    # Every per-step gauge, as device tensors read back once at the end, in
    # one dict so a step file can carry them (`hexn.steps`).
    g: dict = {
        name: []
        for name in (
            "policy_losses", "value_losses", "entropies", "kls", "clipped",
            "value_mses", "margin_losses", "prior_kls", "fast_losses",
            "grad_norms", "epoch_kls", "epoch_clips",
        )
    }
    g["variance"] = None
    cursor = Cursor(generator, steps, policy, optimiser, lambda: g)
    held = cursor.restored(policy.device)
    if held is not None:
        g = held
        if g["variance"] is not None:
            value, rows = g["variance"]
            g["variance"] = (value, rows.to(batch.buffer.device))

    epochs_taken = 0
    amp_skipped = 0
    for epoch in range(config.epochs):
        if len(g["epoch_kls"]) <= epoch:
            g["epoch_kls"].append([])
            g["epoch_clips"].append([])
        for rows in cursor.passes(size, config.minibatch):
            rows = rows.to(batch.buffer.device)
            advantage = batch.advantage[rows].to(policy.device)
            # Normalised per minibatch, which is what makes one clip range work
            # across a run whose reward scale is fixed but whose advantage
            # spread collapses as the value head improves. `_minibatches`
            # guarantees at least two rows, without which `std()` returns nan.
            # Over the whole minibatch even when it is micro-batched below.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            optimiser.zero_grad(set_to_none=True)
            if config.micro_batch and config.micro_batch < len(rows):
                terms = _accumulate(policy, batch, rows, advantage, config, scaler)
            else:
                terms = _terms_for(policy, batch, rows, advantage, config)
                _backward(terms.loss, scaler)
            if scaler is not None:
                scaler.unscale_(optimiser)
            # The return value is the norm *before* clipping, which is the only
            # way to know whether `max_grad_norm` is binding. Kept, not
            # dropped, and kept as a tensor -- read back once, with the rest of
            # the update's gauges, in `_stack_median` below.
            g["grad_norms"].append(
                torch.nn.utils.clip_grad_norm_(
                    policy.net.parameters(), config.max_grad_norm
                ).detach()
            )
            if scaler is None:
                optimiser.step()
            else:
                scale = scaler.get_scale()
                scaler.step(optimiser)
                scaler.update()
                amp_skipped += scaler.get_scale() < scale

            g["policy_losses"].append(terms.policy_loss)
            g["value_losses"].append(terms.value_loss)
            g["value_mses"].append(terms.value_mse)
            g["margin_losses"].append(terms.margin_loss)
            if terms.prior_kl is not None:
                g["prior_kls"].append(terms.prior_kl)
            if terms.fast_loss is not None:
                g["fast_losses"].append(terms.fast_loss)
            g["entropies"].append(terms.entropy)
            g["kls"].append(terms.approx_kl)
            g["clipped"].append(terms.clip_fraction)
            g["epoch_kls"][epoch].append(terms.approx_kl)
            g["epoch_clips"][epoch].append(terms.clip_fraction)
            g["variance"] = (terms.value, rows)
            cursor.stepped()

        epochs_taken += 1
        # The break reads the epoch that just finished, so the damage an over-
        # long update does is bounded at one epoch past the ceiling rather
        # than three — the two recorded blowouts were fine at epoch 1 and
        # diverging by epoch 4. This is the one legitimate per-epoch sync: the
        # decision to keep taking epochs has to be made in Python, and
        # `config.kl_break > 0` short-circuits it away entirely at the
        # every-run-on-record default of 0.
        if config.kl_break > 0 and _stack_mean(g["epoch_kls"][epoch]) > config.kl_break:
            break
    cursor.finish()
    predicted_for_variance = g["variance"]
    epoch_kls, epoch_clips = g["epoch_kls"], g["epoch_clips"]

    with torch.no_grad():
        if predicted_for_variance is None:
            variance = 0.0
        else:
            predicted, rows = predicted_for_variance
            variance = _explained_variance(
                predicted[:, 0], batch.value_target[rows][:, 0].to(predicted.device)
            )

    return Stats(
        positions=size,
        policy_loss=_stack_mean(g["policy_losses"]),
        value_loss=_stack_mean(g["value_losses"]),
        entropy=_stack_mean(g["entropies"]),
        approx_kl=_stack_mean(g["kls"]),
        clip_fraction=_stack_mean(g["clipped"]),
        explained_variance=variance,
        approx_kl_first_minibatch=(
            float(epoch_kls[0][0]) if epoch_kls[0] else 0.0
        ),
        approx_kl_last_epoch=_stack_mean(epoch_kls[-1]),
        clip_fraction_last_epoch=_stack_mean(epoch_clips[-1]),
        grad_norm=_stack_median(g["grad_norms"]),
        value_target_variance=float(batch.value_target[:, 0].var()),
        lr=float(optimiser.param_groups[0]["lr"]),
        epochs_taken=epochs_taken,
        value_mse=_stack_mean(g["value_mses"]),
        margin_loss=_stack_mean(g["margin_losses"]),
        prior_kl=_stack_mean(g["prior_kls"]),
        fast_loss=_stack_mean(g["fast_losses"]),
        amp_skipped=amp_skipped,
        amp_scale=scaler.get_scale() if scaler is not None else 0.0,
    )
