# SPDX-License-Identifier: GPL-3.0-only
"""`hexn.trade`: a network's offer budget in HexSet's terms, and how a
checkpoint written before 0.24 reads."""

from __future__ import annotations

import pytest

from hexset.trading import UNLIMITED

from hexn.trade import offer_budget, recorded_budget, trade_params


def test_zero_is_the_no_trade_network_and_none_is_hexsets_default():
    assert trade_params(0).max_offers == 0
    assert not trade_params(0).trades
    assert trade_params(None) == UNLIMITED
    assert trade_params(3).max_offers == 3


def test_the_old_uncapped_spelling_reads_as_hexsets_unlimited():
    assert offer_budget(-1) is None
    assert offer_budget("") is None
    assert offer_budget("2") == 2
    with pytest.raises(ValueError):
        offer_budget(-2)


def test_a_checkpoint_from_before_the_rename_keeps_its_budget():
    """The pre-0.24 key is read, so a run that trained not to trade still
    does not, the way HexSet reads the same key out of an exported file."""
    assert recorded_budget({"max_trades": 0}) == 0
    assert recorded_budget({"max_trades": -1}) is None
    assert recorded_budget({"max_trades": None}) is None
    assert recorded_budget({"max_offers": 2, "max_trades": 0}) == 2
    assert recorded_budget({}) is None
