"""ponytail: minimal check that live dry-run broker + account helpers don't break."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from production.live.account import AccountSnapshot, snapshot_dict
from production.live.broker import LiveBroker
from production.paper.broker import FillResult


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
    snap = AccountSnapshot(1, "demo", "co", "USD", 80.0, 80.0, 0.0, 80.0, 0.0, 500.0, True, True)
    assert snapshot_dict(snap)["equity"] == 80.0
    print("ok")


if __name__ == "__main__":
    main()
