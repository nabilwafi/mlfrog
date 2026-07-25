"""MT5 account snapshot — equity/balance/margin/leverage (broker-agnostic).

Broker identity comes only from configs/config.yaml mt5.{login,password,server,path}.
Same code works for HF Markets, Finex, or any other MT5 broker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from production import ACCOUNT_LEVERAGE_FALLBACK

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountSnapshot:
    login: int
    server: str
    company: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float  # percent; 0 if no margin used
    leverage: float
    trade_allowed: bool
    trade_expert: bool


def fetch_account(*, leverage_fallback: float = ACCOUNT_LEVERAGE_FALLBACK) -> AccountSnapshot:
    """Read live account_info. Requires mt5_session already connected."""
    import MetaTrader5 as mt5

    from production.monitoring import mt5_session

    if not mt5_session.ensure_connected():
        raise RuntimeError("MT5 not connected — cannot fetch account")

    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")

    term = mt5.terminal_info()
    lev_raw = float(getattr(acc, "leverage", 0) or 0)
    # Some builds report leverage=0/1 when unset; fall back to Finex/HF-style default.
    lev = lev_raw if lev_raw > 1 else float(leverage_fallback)
    margin = float(getattr(acc, "margin", 0) or 0)
    equity = float(getattr(acc, "equity", 0) or 0)
    level = float(getattr(acc, "margin_level", 0) or 0)
    if level <= 0 and margin > 0 and equity > 0:
        level = 100.0 * equity / margin

    return AccountSnapshot(
        login=int(getattr(acc, "login", 0) or 0),
        server=str(getattr(acc, "server", "") or ""),
        company=str(getattr(term, "company", "") or "") if term else "",
        currency=str(getattr(acc, "currency", "USD") or "USD"),
        balance=float(getattr(acc, "balance", 0) or 0),
        equity=equity,
        margin=margin,
        free_margin=float(getattr(acc, "margin_free", 0) or 0),
        margin_level=level,
        leverage=lev,
        trade_allowed=bool(getattr(acc, "trade_allowed", False)),
        trade_expert=bool(getattr(acc, "trade_expert", True)),
    )


def sync_equity_into_state(state: Any, snap: AccountSnapshot) -> None:
    """Overwrite paper portfolio equity with broker equity (risk/heat use real money)."""
    state.equity = float(snap.equity)
    state.peak_equity = max(float(getattr(state, "peak_equity", 0) or 0), float(snap.equity))
    logger.info(
        "account_synced login=%s server=%s equity=%.2f balance=%.2f "
        "free_margin=%.2f leverage=1:%g trade_allowed=%s",
        snap.login,
        snap.server,
        snap.equity,
        snap.balance,
        snap.free_margin,
        snap.leverage,
        snap.trade_allowed,
    )


def snapshot_dict(snap: AccountSnapshot) -> dict[str, Any]:
    return {
        "login": snap.login,
        "server": snap.server,
        "company": snap.company,
        "currency": snap.currency,
        "balance": snap.balance,
        "equity": snap.equity,
        "margin": snap.margin,
        "free_margin": snap.free_margin,
        "margin_level": snap.margin_level,
        "leverage": snap.leverage,
        "trade_allowed": snap.trade_allowed,
        "trade_expert": snap.trade_expert,
    }
