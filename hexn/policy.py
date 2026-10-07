# SPDX-License-Identifier: GPL-3.0-only
"""The torch `BatchPolicy`: one forward per tick, masked, sampled.

This is the seam between `hexn.selfplay` and `hexn.model`, and its shape
follows from one property of a forward pass: it costs a fixed dispatch toll
per call, plus a much smaller cost per position, so the cost of asking the
network anything at all dwarfs the cost of asking it about many things at
once. `Collector` therefore gathers one `Request` per lane and calls `act`
once; this module answers the whole batch with a single `forward`, a single
host-to-device copy and a single read-back. Everything else here exists to
keep it to one.

## Every action is one flat index again

Under contract 4 the flat categorical had one slot, `PROPOSE_TRADE`, that did
not name what it had chosen — an offer is ten numbers — so the policy carried
`give`/`want` heads, a masked pair distribution, a joint log-prob and a pair
mask riding along in `Choice.aux` for PPO to reuse. Contract 5 deletes the
trade actions outright (`hexset.trading`): trades clear once a turn from the
seats' published valuation vectors, which are *observation*, never action. So
`ActionSpace.decode` turns every sampled index back into exactly the `Action`
`legal_actions` emitted, the log-prob is the flat categorical's and nothing
else, and none of that apparatus has a successor.

## What the network brings to a trade instead

Contract 6's trading redesign (`hexset.trading`) deletes the public layer
outright — there is no published valuation vector any more, nothing is
advertised, and the mechanic's whole observation surface is each seat's
private gate. That gate is
`hexset.clients.netbot.NetworkBot`, and it is the engine's, not this
package's: it builds the post-trade position (both hands moved, the
counterparty's ledger row certified) and asks a `hexset.clients.policy.
Policy` what it is worth. `NetworkPolicy` is that `Policy` for torch, and
`trader` below seats it behind the engine's gate. There is still no new
parameter and no policy-gradient path through trading — the verdict is read
off the value head the run already trains, exactly as reading (A) always
was.

A torch copy of the gate lived here until 0.19.0 (`DerivedTrader`,
`after_exchange`) and drifted from the ONNX one three ways before anybody
noticed: the copy is gone, and what is left is the runtime.

## What is not optimised here, and why

The rollout is plumbing-bound rather than compute-bound: engine stepping,
encoding and legal-action enumeration in plain Python cost comparably to the
GPU forward itself, so shaving the torch side of a tick buys comparatively
little. More throughput comes from running collectors in parallel processes,
not from anything in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from hexset.actions import Action, ActionSpace, mask_of
from hexset.clients.netbot import NetworkBot
from hexset.encoding import encode_batch, from_frame
from hexset.game import Game
from .model import HexNet, Packing, pack, packing, unpack
from .selfplay import Choice, Request
from .trade import trade_params, trader_gate

__all__ = [
    "masked_log_softmax",
    "Evaluation",
    "NetworkPolicy",
]

# Large and finite rather than -inf. A row whose mask is entirely False would
# make `log_softmax` produce NaN under -inf and merely produce a uniform
# distribution here, which is recoverable and diagnosable. `Collector` raises
# `Stuck` before an empty flat mask can reach us.
NEG = -1e9


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """`log_softmax` over the legal entries of each row."""
    return torch.log_softmax(logits.masked_fill(~mask, NEG), dim=-1)


def _entropy(log_probs: Tensor) -> Tensor:
    return -(log_probs.exp() * log_probs).sum(-1)


@dataclass(frozen=True)
class Evaluation:
    """What the PPO update needs recomputed under the current parameters."""

    log_prob: Tensor
    entropy: Tensor
    # The win head's probability -- `softmax(value_logits)` -- unchanged from
    # `Prediction.value`, kept here under the same name for every existing
    # reader.
    value: Tensor
    # The same head's raw, pre-softmax output: what the cross-entropy loss
    # against the one-hot winner is actually taken against (`hexn.ppo`).
    value_logits: Tensor
    # The auxiliary VP-margin head's raw output, read only by `hexn.ppo`'s
    # own aux loss term -- never by the advantage or a gate.
    margin: Tensor
    # `None` for every value head but `"quantile"`, where it is the
    # `(B, players, Q)` tensor whose mean is `value_logits`. Carried through
    # so the value loss can be taken on the forward pass the ratio already
    # paid for; `hexn.ppo` is the only reader.
    quantiles: Tensor | None = None
    # The whole masked log-softmax row `log_prob` was gathered from, so a
    # divergence to another policy over every legal action (`hexn.ppo`'s
    # prior term) is taken on the same forward rather than a second one.
    log_probs: Tensor | None = None
    # The fast-win head's raw output (`Prediction.fast_logits`), `None` when
    # the net has none; `hexn.ppo`'s fast-win loss is its only reader.
    fast_logits: Tensor | None = None


class NetworkPolicy:
    """`HexNet` behind `hexn.selfplay.BatchPolicy` — and behind
    `hexset.clients.policy.Policy`.

    Two seams onto one net, because the two callers want different things.
    `act`/`values`/`score`/`evaluate` take pre-encoded rows, which is what the
    collector has already built for a whole tick and must not pay to rebuild.
    `act_rows`/`value_rows`/`score_rows` take live `(game, seat)` positions,
    which is what the engine's runtime-free bot, leaf evaluation, search and
    trade gate (`hexset.clients.netbot`) are stated in — they never learn
    which runtime they hold, and the onnxruntime half answers exactly the
    same three methods (`hexset.clients.onnxbot.V2Policy`).

    Sampling is stochastic by default because PPO is on-policy and needs the
    behaviour distribution it will later take ratios against. `greedy=True`
    takes the argmax instead, which is for evaluation duels only — a greedy
    policy's `log_prob` is not the one that generated anything.
    """

    def __init__(
        self,
        net: HexNet,
        space: ActionSpace,
        layout: Packing,
        *,
        device: torch.device | str = "cpu",
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.net = net
        self.space = space
        self.layout = layout
        self.device = torch.device(device)
        self.greedy = greedy
        self.generator = generator
        # Record the fast-win head's seat values (`Prediction.fast`) on each
        # `Choice` instead of the win head's -- what a run that pays for fast
        # wins takes its advantage from. The gates and a search still read the
        # win head; this only changes what `act` hands the trajectory.
        self.record_fast = False
        # Record the value of a per-VP reward (`hexn.ppo.PPOConfig.vp_reward`,
        # c per victory point gained): the mover's component becomes the win
        # (or fast-win) value plus c times the VP still to come, which the
        # margin head predicts as final points / 10 in that mode. 0 records
        # the plain value.
        self.vp_reward = 0.0

    def _sample(self, log_probs: Tensor) -> Tensor:
        if self.greedy:
            return log_probs.argmax(dim=-1)
        # `multinomial` over probabilities rather than Gumbel over logits: the
        # rows are already normalised by the masked log-softmax, so this needs
        # no second pass and stays one kernel.
        return torch.multinomial(
            log_probs.exp(), 1, generator=self.generator
        ).squeeze(-1)

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        if not requests:
            return []

        masks = np.stack([request.mask for request in requests])

        buffer = pack(self.layout, [request.observation for request in requests])
        mask = torch.from_numpy(masks).to(self.device, non_blocking=True)

        # Per-seat sampling temperatures (`Request.temperature`), applied to
        # the logits before the masked softmax so the recorded `log_prob` is
        # the tempered behaviour distribution's. The common case is every
        # row at 1.0, which pays one numpy compare and no device transfer.
        temperatures = np.fromiter(
            (request.temperature for request in requests),
            dtype=np.float32,
            count=len(requests),
        )
        scaled = not self.greedy and bool((temperatures != 1.0).any())

        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))

            logits = prediction.logits
            if scaled:
                logits = logits / torch.from_numpy(temperatures).to(
                    self.device, dtype=logits.dtype
                ).unsqueeze(1)
            log_probs = masked_log_softmax(logits, mask)
            chosen = self._sample(log_probs)
            log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)

            # One read-back rather than three. A transfer costs a fixed
            # overhead per tensor whatever its size, so the concatenation is
            # free and the two extra crossings are not.
            recorded = (
                prediction.fast[:, : prediction.value.shape[1]]
                if self.record_fast
                else prediction.value
            )
            if self.vp_reward:
                now = torch.tensor(
                    [float(request.points) for request in requests],
                    device=recorded.device,
                    dtype=recorded.dtype,
                )
                to_come = 10.0 * prediction.margin[:, 0] - now
                recorded = torch.cat(
                    [(recorded[:, 0] + self.vp_reward * to_come).unsqueeze(1), recorded[:, 1:]], dim=1
                )
            read = torch.cat(
                [
                    chosen.unsqueeze(1).to(prediction.value.dtype),
                    log_prob.unsqueeze(1),
                    recorded,
                ],
                dim=1,
            ).cpu().numpy()

        return [
            Choice(
                action=self.space.decode(int(read[row, 0])),
                log_prob=float(read[row, 1]),
                value=tuple(read[row, 2:].tolist()),
            )
            for row in range(len(requests))
        ]

    @property
    def players(self) -> int:
        """How many seats this policy was built for.

        Read off the action space rather than stored: the space is what the
        net's heads were sized against, so the two cannot disagree, and a
        test double wrapping `net` does not have to forward a field.
        """
        return int(self.space.num_players)

    def act_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[Action]:
        """`hexset.clients.policy.Policy.act_rows`: one action per position,
        drawn from that position's own options."""
        if not rows:
            return []
        observations = encode_batch(
            [game for game, _, _ in rows], [seat for _, seat, _ in rows]
        )
        requests = [
            Request(
                lane=lane,
                seat=seat,
                observation=observations[lane],
                mask=np.asarray(mask_of(self.space, options), dtype=bool),
                options=tuple(options),
                game=game,
            )
            for lane, (game, seat, options) in enumerate(rows)
        ]
        return [choice.action for choice in self.act(requests)]

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """`hexset.clients.policy.Policy.value_rows`: the value head alone,
        one vector per position, un-rotated back into board-seat order.

        The whole of what the engine's trade gate asks for
        (`hexset.clients.netbot.NetworkBot._score`), which is why it is
        `encode_batch` and one forward: the gate hands over the live position
        and every candidate's post-trade position together.
        """
        if not rows:
            return []
        seats = [seat for _, seat in rows]
        values = self.values(encode_batch([game for game, _ in rows], seats))
        return [
            from_frame(values[row].tolist(), seat) for row, seat in enumerate(seats)
        ]

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[np.ndarray, tuple[float, ...]]]:
        """`hexset.clients.policy.Policy.score_rows`: `(prior, value)` per
        position, as `hexset.mcts` wants a leaf scored.

        Every option has its own flat slot — the trade actions, and with them
        the one slot that stood for many offers, are gone (`hexset.trading`)
        — so the prior over a leaf's options is a plain gather from the
        masked row.
        """
        if not rows:
            return []
        seats = [seat for _, seat, _ in rows]
        observations = encode_batch([game for game, _, _ in rows], seats)
        masks = np.asarray(
            [mask_of(self.space, options) for _, _, options in rows], dtype=bool
        )
        slots, values = self.score(observations, masks)
        return [
            (
                self._prior(options, slots[row]),
                from_frame(values[row].tolist(), seats[row]),
            )
            for row, (_, _, options) in enumerate(rows)
        ]

    def _prior(self, options: Sequence[Action], slots: np.ndarray) -> np.ndarray:
        prior = np.exp(
            np.array([slots[self.space.index(option)] for option in options])
        )
        total = prior.sum()
        # Normalised rather than trusted: the mask makes this 1 up to float
        # error, and a search cannot use probability mass parked anywhere else.
        if total <= 0:
            return np.full(len(options), 1.0 / len(options))
        return prior / total

    def trader(
        self, game: Game, seat: int | None, max_offers: int | None = None,
        trader: str | None = None,
    ):
        """What this policy brings to `game`'s trade event at `seat`.

        A driver seats one of these per seat on `game.gates`; the engine asks
        it for a private-gate verdict (`hexset.trading.valued_many` — there
        is no public layer left for a gate to publish anything to). The gate
        is the engine's own (`hexset.clients.netbot.NetworkBot`), seated at
        this position exactly as `NetworkBot.choose` seats itself: it builds
        the post-trade positions and this policy only says what they are
        worth, so a self-play seat, a duelled checkpoint and a served `.onnx`
        file all trade through one implementation.

        `max_offers` is the network's own offer budget (`hexn.trade`), declared
        on the gate as HexSet's `TradeParams`: `0` does not trade, `None` is
        HexSet's default. `trader` seats that HexSet bot's gate instead
        (`hexn.trade.trader_gate`), and this policy's value head prices nothing.

        `seat` is carried onto the gate rather than left to the `View` to
        name: a gate wired to the wrong seat would answer for somebody else's
        hand, and `NetworkBot` raises on that instead of answering quietly.
        `seat_at` seats it at the position without asking it to move.
        """
        if trader is not None:
            return trader_gate(trader, game, seat)
        gate = NetworkBot(
            policy=self,
            players=self.players,
            trade=trade_params(max_offers),
            seat=seat,
        )
        gate.seat_at(game)
        return gate

    def values(self, observations: Sequence) -> np.ndarray:
        """The value head alone, `(B, players)`, in each row's own frame.

        For a search, which scores leaves and has no use for the policy logits.
        The forward computes them anyway — the heads share a trunk and splitting
        them would cost more than the gather does — so the saving here is the
        read-back and the sampling, not the network.

        Takes a list because that is the seam a leaf-batching search will want,
        even though `hexset.bots.SearchBot` currently hands over one at a time.
        """
        buffer = pack(self.layout, list(observations))
        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))
        return prediction.value.cpu().numpy()

    def score(
        self, observations: Sequence, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Masked log-probs and values for a batch.

        What a batched search needs and `act` does not give it: the distribution
        itself rather than a draw from it. A search wants the prior over every
        option, and `values` alone is not enough — a PUCT bonus without a prior
        is an untrained search. One forward and one read-back, for the same
        reason `act` is one of each.
        """
        buffer = pack(self.layout, list(observations))
        mask = torch.from_numpy(masks).to(self.device, non_blocking=True)
        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))
            slots = masked_log_softmax(prediction.logits, mask)
            read = torch.cat([slots, prediction.value], dim=1).cpu().numpy()
        width = slots.shape[1]
        return read[:, :width], read[:, width:]

    def distributions(
        self, buffer: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """The masked log-prob rows, the win-head probability, its raw
        logits and (only under a quantile head) its full spread -- all with
        grad.

        The third reader of the forward, and it wants what the other two throw
        away. `score` is these same quantities without grad and as numpy, for
        the search; `evaluate` reduces them to the one action PPO took. A
        cross-entropy against a visit distribution is defined over the whole
        row, so it can use neither.

        `value_logits` and `quantiles` exist for `hexn.exit`'s own
        win-head cross-entropy, mirroring `Evaluation`'s fields -- `hexn.exit`
        has no `chosen` action to key `evaluate` on, since its policy loss is
        a cross-entropy over the whole slot row rather than one taken action.
        """
        prediction = self.net(*unpack(self.layout, buffer))
        return (
            masked_log_softmax(prediction.logits, mask),
            prediction.value,
            prediction.value_logits,
            prediction.quantiles,
        )

    def evaluate(self, buffer: Tensor, mask: Tensor, chosen: Tensor) -> Evaluation:
        """Recompute a stored batch's log-probs, entropy and values, with grad.

        The mirror of `act`, and the two have to agree exactly or PPO's ratio is
        wrong at step zero rather than after it has learned something.
        `test_evaluate_reproduces_the_log_prob_act_recorded` pins that.
        """
        prediction = self.net(*unpack(self.layout, buffer))
        # Back to fp32 before anything is normalised or differenced: under
        # half-precision autocast (`PPOConfig.amp`) the heads come out in fp16,
        # whose range cannot even hold the mask's fill. A no-op in fp32.
        def wide(x: Tensor | None) -> Tensor | None:
            return None if x is None else x.float()

        log_probs = masked_log_softmax(prediction.logits.float(), mask)
        return Evaluation(
            log_prob=log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1),
            entropy=_entropy(log_probs),
            value=wide(prediction.value),
            value_logits=wide(prediction.value_logits),
            margin=wide(prediction.margin),
            quantiles=wide(prediction.quantiles),
            log_probs=log_probs,
            fast_logits=wide(prediction.fast_logits),
        )


def build(
    net: HexNet,
    space: ActionSpace,
    graph,
    players: int,
    **kwargs,
) -> NetworkPolicy:
    """A policy over `net`, with the packing layout derived from the graph."""
    return NetworkPolicy(net, space, packing(graph, players), **kwargs)
