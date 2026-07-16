"""MT5 liveness probe for health / Telegram (read-only)."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def probe_mt5(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Soft-check MetaTrader 5 terminal.
    Returns keys: mt5 (connected|down), plus terminal/account fields when up.
    Never raises — health path must not crash the runtime.
    """
    mt5_cfg = dict(cfg.get("mt5") or {})
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        return {"mt5": "down", "mt5_error": f"MetaTrader5 not installed: {exc}"}

    path = mt5_cfg.get("path") or None
    try:
        if not mt5.initialize(path=path):
            return {"mt5": "down", "mt5_error": f"initialize failed: {mt5.last_error()}"}

        login = int(mt5_cfg.get("login") or 0)
        if login:
            password = mt5_cfg.get("password", "")
            server = mt5_cfg.get("server", "")
            if not mt5.login(login, password=password, server=server):
                err = f"login failed: {mt5.last_error()}"
                mt5.shutdown()
                return {"mt5": "down", "mt5_error": err}

        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None:
            mt5.shutdown()
            return {"mt5": "down", "mt5_error": "terminal_info unavailable"}

        out: dict[str, Any] = {
            "mt5": "connected",
            "mt5_connected": bool(getattr(terminal, "connected", True)),
            "mt5_trade_allowed": bool(getattr(terminal, "trade_allowed", False)),
            "mt5_build": getattr(terminal, "build", None),
            "mt5_company": getattr(terminal, "company", None),
            "mt5_name": getattr(terminal, "name", None),
        }
        if account is not None:
            out["mt5_login"] = getattr(account, "login", None)
            out["mt5_server"] = getattr(account, "server", None)
            out["mt5_balance"] = getattr(account, "balance", None)
            out["mt5_equity"] = getattr(account, "equity", None)
        # Terminal process up but not connected to trade server
        if out.get("mt5_connected") is False:
            out["mt5"] = "down"
            out["mt5_error"] = "terminal running but not connected to trade server"
        mt5.shutdown()
        return out
    except Exception as exc:
        logger.exception("mt5_probe_failed")
        try:
            import MetaTrader5 as mt5

            mt5.shutdown()
        except Exception:
            pass
        return {"mt5": "down", "mt5_error": str(exc)}


def make_mt5_probe(cfg: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    return lambda: probe_mt5(cfg)
