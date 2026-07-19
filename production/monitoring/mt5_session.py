"""Process-wide MT5 session — live feed + health share one IPC connection."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cfg: dict[str, Any] | None = None
_connected = False


def _mt5_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("mt5") or {})


def _terminal_up() -> bool:
    try:
        import MetaTrader5 as mt5

        return mt5.terminal_info() is not None
    except Exception:
        return False


def connect(cfg: dict[str, Any]) -> None:
    """Initialize MT5 once for this process."""
    global _cfg, _connected
    with _lock:
        _cfg = cfg
        _connect_locked()


def _connect_locked() -> None:
    global _connected
    import MetaTrader5 as mt5

    if _terminal_up():
        _connected = True
        return

    if _cfg is None:
        raise RuntimeError("mt5_session: config not set")

    mt5_cfg = _mt5_cfg(_cfg)
    path = mt5_cfg.get("path") or None
    if not mt5.initialize(path=path):
        _connected = False
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    login = int(mt5_cfg.get("login") or 0)
    if login:
        password = mt5_cfg.get("password", "")
        server = mt5_cfg.get("server", "")
        if not mt5.login(login, password=password, server=server):
            mt5.shutdown()
            _connected = False
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    if mt5.terminal_info() is None:
        mt5.shutdown()
        _connected = False
        raise RuntimeError("MT5 terminal info unavailable")

    _connected = True
    info = mt5.terminal_info()
    logger.info("mt5_session_connected build=%s company=%s", info.build, info.company)


def ensure_connected(cfg: dict[str, Any] | None = None) -> bool:
    """Reconnect after IPC loss (-10004). Never shuts down a healthy session."""
    global _cfg, _connected
    if cfg is not None:
        _cfg = cfg
    with _lock:
        try:
            if _terminal_up():
                _connected = True
                return True
            if _cfg is None:
                return False
            try:
                import MetaTrader5 as mt5

                mt5.shutdown()
            except Exception:
                pass
            _connected = False
            _connect_locked()
            return True
        except Exception as exc:
            _connected = False
            logger.warning("mt5_ensure_connected_failed err=%s", exc)
            return False


def select_symbol(symbol: str) -> None:
    import MetaTrader5 as mt5

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol!r}) failed: {mt5.last_error()}")


def disconnect() -> None:
    """Call once on process shutdown."""
    global _connected
    with _lock:
        try:
            import MetaTrader5 as mt5

            if _connected or _terminal_up():
                mt5.shutdown()
        except Exception:
            pass
        _connected = False
        logger.info("mt5_session_disconnected")


def status(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Read-only health snapshot.
    ponytail: never calls shutdown — safe alongside live feed.
    """
    if cfg is not None:
        global _cfg
        _cfg = cfg
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        return {"mt5": "down", "mt5_error": f"MetaTrader5 not installed: {exc}"}

    try:
        if not _terminal_up():
            if _cfg is not None and ensure_connected(_cfg):
                pass
            else:
                return {"mt5": "down", "mt5_error": "terminal not running or IPC unavailable"}

        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None:
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
        if out.get("mt5_connected") is False:
            out["mt5"] = "down"
            out["mt5_error"] = "terminal running but not connected to trade server"
        return out
    except Exception as exc:
        logger.exception("mt5_status_failed")
        return {"mt5": "down", "mt5_error": str(exc)}
