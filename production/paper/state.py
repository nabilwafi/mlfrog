"""Paper portfolio / heat state (daily loss stop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from production import DAILY_LOSS_STOP_R, RISK_PCT


@dataclass
class OpenPosition:
    trade_id: str
    signal_id: str
    side: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float
    risk_pct: float
    atr: float
    correlation_id: str
    meta: dict[str, Any] = field(default_factory=dict)
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class PortfolioState:
    equity: float
    peak_equity: float
    day: date | None = None
    day_pnl: float = 0.0
    heat_triggered_today: int = 0
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    seen_signal_keys: set[str] = field(default_factory=set)  # idempotency
    meta_rejects: int = 0
    confidence_rejects: int = 0
    skips: int = 0
    trades_today: int = 0
    wins_today: int = 0
    pnl_today: float = 0.0

    def roll_day(self, now: datetime) -> None:
        d = now.date()
        if self.day != d:
            self.day = d
            self.day_pnl = 0.0
            self.heat_triggered_today = 0
            self.meta_rejects = 0
            self.confidence_rejects = 0
            self.skips = 0
            self.trades_today = 0
            self.wins_today = 0
            self.pnl_today = 0.0

    def r_unit(self) -> float:
        return self.equity * RISK_PCT

    def daily_heat_blocked(self) -> bool:
        ru = self.r_unit()
        if ru <= 0:
            return True
        return self.day_pnl <= -DAILY_LOSS_STOP_R * ru

    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def register_signal_key(self, key: str) -> bool:
        """Return True if new (not duplicate)."""
        if key in self.seen_signal_keys:
            return False
        self.seen_signal_keys.add(key)
        # ponytail: bound memory
        if len(self.seen_signal_keys) > 50_000:
            self.seen_signal_keys = set(list(self.seen_signal_keys)[-20_000:])
        return True

    def apply_pnl(self, pnl: float) -> None:
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.day_pnl += pnl
        self.pnl_today += pnl
