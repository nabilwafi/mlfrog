"""ponytail: minimal check that live dry-run broker + account helpers don't break."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from production.live.account import AccountSnapshot, snapshot_dict, sync_equity_into_state
from production.live.broker import LiveBroker, _RETCODE_HINT
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
    # sync must overwrite config starting_equity ($80) on equity + day_start + balance
    state = PortfolioState(equity=80.0, peak_equity=80.0)
    assert state.day_start_equity == 80.0
    sync_equity_into_state(state, snap)
    assert state.equity == 510.0
    assert state.balance == 512.34
    assert state.day_start_equity == 510.0
    assert "TRADE_DISABLED" in _RETCODE_HINT[10017]
    print("ok")


if __name__ == "__main__":
    main()
