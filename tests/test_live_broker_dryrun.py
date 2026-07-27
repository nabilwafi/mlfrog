"""ponytail: minimal check that live dry-run broker + account helpers don't break."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from production.live.account import AccountSnapshot, snapshot_dict, sync_equity_into_state
from production.live.broker import LiveBroker, _RETCODE_HINT, _deal_exit_reason
from production.live.positions import BrokerOpenPosition, _atr_from_sl, _trade_id_from_comment, enrich_with_db
from production.paper.broker import FillResult
from production.paper.state import PortfolioState


def main() -> None:
    b = LiveBroker(symbol="XAUUSDc", execution_enabled=False)
    r = b.open_order(
        side="long",
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        lot=0.01,
        trade_id="selfcheck",
    )
    assert isinstance(r, FillResult) and r.success
    assert b.close_order("selfcheck", 2001.0).success
    snap = AccountSnapshot(1, "demo", "co", "USD", 512.34, 510.0, 0.0, 510.0, 0.0, 500.0, True, True)
    assert snapshot_dict(snap)["equity"] == 510.0
    state = PortfolioState(equity=80.0, peak_equity=80.0)
    assert state.day_start_equity == 80.0
    sync_equity_into_state(state, snap)
    assert state.equity == 510.0
    assert state.balance == 512.34
    assert state.day_start_equity == 510.0
    assert "TRADE_DISABLED" in _RETCODE_HINT[10017]
    assert _deal_exit_reason(4) == "SL"
    assert _trade_id_from_comment("xauusd:abc123", 99) == "recovered-abc123-99"
    assert _trade_id_from_comment("", 42) == "recovered-42"
    assert abs(_atr_from_sl(side="long", entry=2000.0, stop_loss=1990.0) - 10.0 / 1.5) < 1e-9
    b.register_recovered("recovered-42", 42)
    assert b._tickets["recovered-42"] == 42

    class FakeDb:
        enabled = True

        def fetch_open_trades_by_ticket(self, *, symbol=None, tickets=None):
            return {42: {"trade_id": "db-tid", "signal_id": "db-tid", "entry_price": 1.0, "lot": 0.01}}

    row = BrokerOpenPosition(
        trade_id="recovered-42",
        broker_ticket=42,
        side="long",
        entry_time=datetime.now(timezone.utc),
        entry_price=1.0,
        stop_loss=0.9,
        take_profit=0.0,
        lot=0.01,
        comment="",
    )
    en = enrich_with_db([row], FakeDb(), symbol="XAUUSDc")
    assert en[0].trade_id == "db-tid" and en[0].from_db
    print("ok")


if __name__ == "__main__":
    main()
