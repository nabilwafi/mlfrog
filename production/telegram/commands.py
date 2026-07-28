"""Telegram slash commands: /check_candle, /check_summary."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from production.telegram.bot import TelegramNotifier, fmt_check_candle, fmt_check_positions, fmt_check_summary

logger = logging.getLogger(__name__)


def server_offset_hours(cfg: dict[str, Any] | None = None) -> int:
    """
    Hours to add to UTC unix bar time to match MT5 chart labels.
    HF Markets Live is commonly UTC+2/+3 — set paper_trading.mt5_server_utc_offset_hours.
    """
    paper = dict((cfg or {}).get("paper_trading") or {})
    raw = paper.get("mt5_server_utc_offset_hours")
    if raw is None:
        raw = (cfg or {}).get("mt5_server_utc_offset_hours", 3)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3


def fmt_mt5_time(unix_s: int | float, *, offset_hours: int) -> str:
    """Format rate time as MT5 chart clock (no WIB convert)."""
    dt = datetime.fromtimestamp(int(unix_s), tz=timezone.utc) + timedelta(hours=int(offset_hours))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_utc_time(unix_s: int | float) -> str:
    dt = datetime.fromtimestamp(int(unix_s), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _tf_seconds(timeframe: str) -> int:
    return {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
    }.get(str(timeframe).upper(), 3600)


def _candle_id(unix_s: int, timeframe: str) -> str:
    dt = datetime.fromtimestamp(int(unix_s), tz=timezone.utc)
    return f"#{str(timeframe).upper()}-{dt.strftime('%Y%m%d-%H%M')}"


def _ohlc_trend(o: float, h: float, l: float, c: float) -> str:
    rng = max(h - l, 1e-9)
    body = c - o
    if abs(body) / rng < 0.15:
        return "SIDEWAYS"
    return "BULLISH" if body > 0 else "BEARISH"


def _bot_name() -> str:
    try:
        from production import PIPELINE_VERSION

        return str(PIPELINE_VERSION)
    except Exception:
        return "mlfrog"


def build_candle_check(
    *,
    cfg: dict[str, Any],
    symbol: str,
    timeframe: str = "H1",
    count: int = 5,
    signal: str | None = None,
    trend: str | None = None,
) -> dict[str, Any]:
    """Fetch recent rates; Timestamp field is UTC (template); MT5 chart time kept in closed.time_mt5."""
    import MetaTrader5 as mt5

    from data.connectors.mt5_constants import resolve_timeframe
    from production.monitoring import mt5_session

    offset = server_offset_hours(cfg)
    bot = _bot_name()
    if not mt5_session.ensure_connected(cfg):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "server_offset_hours": offset,
            "bot_name": bot,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": "WAIT",
            "trend": "n/a",
            "status": "MT5 disconnected",
            "market_closed": True,
            "candle_id": "n/a",
        }
    mt5_session.select_symbol(symbol)
    tf = resolve_timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
    if rates is None or len(rates) == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "server_offset_hours": offset,
            "bot_name": bot,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": "WAIT",
            "trend": "n/a",
            "status": f"no rates: {mt5.last_error()}",
            "market_closed": True,
            "candle_id": "n/a",
        }

    now_utc = datetime.now(timezone.utc)
    server_now = (now_utc + timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M:%S")
    tf_sec = _tf_seconds(timeframe)
    rows: list[dict[str, Any]] = []
    for r in rates:
        t = int(r["time"])
        bar_end = datetime.fromtimestamp(t, tz=timezone.utc) + timedelta(seconds=tf_sec)
        closed = bar_end <= now_utc
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        rows.append(
            {
                "unix": t,
                "time": fmt_mt5_time(t, offset_hours=offset),
                "time_utc": fmt_utc_time(t),
                "time_mt5": fmt_mt5_time(t, offset_hours=offset),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "closed": closed,
                "trend": _ohlc_trend(o, h, l, c),
                "candle_id": _candle_id(t, timeframe),
            }
        )

    last = rows[-1]
    forming = None
    closed_bar = None
    if last["closed"]:
        closed_bar = dict(last)
        age_s = (
            now_utc - (datetime.fromtimestamp(last["unix"], tz=timezone.utc) + timedelta(seconds=tf_sec))
        ).total_seconds()
        closed_bar["age"] = f"{int(age_s // 3600)}h {int((age_s % 3600) // 60)}m"
        closed_bar["state"] = "STALE / MARKET CLOSED" if age_s > tf_sec * 2 else "READY"
        market_closed = age_s > tf_sec * 2
    else:
        forming = dict(last)
        if len(rows) >= 2:
            closed_bar = dict(rows[-2])
            closed_bar["age"] = "just closed"
            closed_bar["state"] = "READY"
        market_closed = False

    bar_trend = trend or (closed_bar or {}).get("trend") or "n/a"
    # ponytail: signal from caller (model) if provided; else WAIT (OHLC alone is not a trade signal)
    bar_signal = (signal or "WAIT").upper()
    status = (closed_bar or {}).get("state") or ("FORMING" if forming else "n/a")
    if market_closed:
        status = "MARKET CLOSED"
        bar_signal = "WAIT"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "server_offset_hours": offset,
        "server_now": server_now,
        "timestamp": (closed_bar or {}).get("time_utc")
        or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "closed": closed_bar,
        "forming": forming,
        "history": list(reversed(rows[-min(5, len(rows)) :])),
        "market_closed": market_closed,
        "trend": bar_trend,
        "signal": bar_signal,
        "status": status,
        "bot_name": bot,
        "candle_id": (closed_bar or {}).get("candle_id") or "n/a",
    }


def build_summary_check(
    *,
    state: Any,
    symbol: str,
    environment: str = "live",
    db: Any | None = None,
    use_mt5_account: bool = True,
) -> dict[str, Any]:
    """Balance/equity from MT5 when available; total PnL / W/L from history_trades."""
    balance = getattr(state, "balance", None)
    equity = float(getattr(state, "equity", 0) or 0)
    if use_mt5_account:
        try:
            from production.live.account import fetch_account

            snap = fetch_account()
            balance = float(snap.balance)
            equity = float(snap.equity)
        except Exception:
            logger.exception("summary_mt5_account_failed — falling back to state")

    hist = {"total_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
    if db is not None and getattr(db, "enabled", False):
        try:
            hist = dict(db.summarize_history() or hist)
        except Exception:
            logger.exception("summary_history_failed")

    # Prefer history aggregates; fall back to in-memory closed-only counters
    trades = int(hist.get("trades") or 0)
    wins = int(hist.get("wins") or 0)
    losses = int(hist.get("losses") or 0)
    pnl = float(hist.get("total_pnl") or 0.0)
    if trades <= 0:
        wins = int(getattr(state, "wins_today", 0) or 0)
        losses = int(getattr(state, "losses_today", 0) or 0)
        trades = wins + losses  # open/running excluded from W/L
        pnl = float(getattr(state, "pnl_today", 0) or 0)

    running = len(getattr(state, "open_positions", {}) or {})
    start_eq = float(getattr(state, "day_start_equity", 0) or equity or 0)
    if balance is None:
        balance = start_eq

    open_list = []
    for pos in (getattr(state, "open_positions", {}) or {}).values():
        open_list.append(
            {
                "side": pos.side,
                "entry_price": pos.entry_price,
                "stop_loss": pos.stop_loss,
                "lot": pos.lot,
                "ticket": getattr(pos, "broker_ticket", None),
            }
        )
    return {
        "symbol": symbol,
        "status": "ok",
        "environment": environment,
        "balance": float(balance),
        "equity": float(equity),
        "pnl": pnl,
        "pnl_pct": (pnl / float(balance) * 100.0) if float(balance or 0) > 0 else 0.0,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "running": running,
        "open_positions": running,
        "winrate": (wins / trades) if trades else 0.0,
        "best_trade_pnl": getattr(state, "best_trade_pnl_today", None),
        "worst_trade_pnl": getattr(state, "worst_trade_pnl_today", None),
        "open_list": open_list,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "bot_name": _bot_name(),
    }


def build_positions_check(
    *,
    cfg: dict[str, Any],
    state: Any,
    symbol: str,
    timeframe: str = "H1",
    magic: int = 27001,
) -> dict[str, Any]:
    """Live open positions from MT5 (preferred) + in-memory state fallback."""
    from production.monitoring import mt5_session

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trend = "n/a"
    positions: list[dict[str, Any]] = []

    # light trend from last closed H1 if MT5 up
    try:
        candle = build_candle_check(cfg=cfg, symbol=symbol, timeframe=timeframe, count=3)
        trend = str(candle.get("trend") or "n/a")
    except Exception:
        logger.exception("positions_trend_failed")

    try:
        if mt5_session.ensure_connected(cfg):
            import MetaTrader5 as mt5

            mt5_session.select_symbol(symbol)
            raw = mt5.positions_get(symbol=symbol) or ()
            tick = mt5.symbol_info_tick(symbol)
            for p in raw:
                if int(getattr(p, "magic", 0) or 0) != int(magic):
                    continue
                is_buy = int(getattr(p, "type", 0)) == mt5.POSITION_TYPE_BUY
                side = "long" if is_buy else "short"
                entry = float(getattr(p, "price_open", 0) or 0)
                cur = entry
                if tick:
                    cur = float(tick.bid if is_buy else tick.ask)
                profit = float(getattr(p, "profit", 0) or 0)
                profit += float(getattr(p, "swap", 0) or 0)
                # XAU pip ≈ 0.1
                raw_move = (cur - entry) if is_buy else (entry - cur)
                pips = raw_move / 0.1
                sl = float(getattr(p, "sl", 0) or 0)
                status = "OPEN"
                if abs(raw_move) < 1e-6:
                    status = "BREAKEVEN"
                positions.append(
                    {
                        "side": side,
                        "ticket": int(getattr(p, "ticket", 0) or 0),
                        "lot": float(getattr(p, "volume", 0) or 0),
                        "entry_price": entry,
                        "current_price": cur,
                        "profit": profit,
                        "pips": pips,
                        "stop_loss": sl,
                        "take_profit": float(getattr(p, "tp", 0) or 0),
                        "status": status,
                    }
                )
    except Exception:
        logger.exception("positions_mt5_failed")

    # fallback: in-memory bot state if MT5 empty
    if not positions:
        for pos in (getattr(state, "open_positions", {}) or {}).values():
            entry = float(pos.entry_price)
            positions.append(
                {
                    "side": pos.side,
                    "ticket": pos.broker_ticket or "n/a",
                    "lot": float(pos.lot),
                    "entry_price": entry,
                    "current_price": entry,
                    "profit": 0.0,
                    "pips": 0.0,
                    "stop_loss": float(pos.stop_loss),
                    "take_profit": float(pos.take_profit or 0),
                    "status": "OPEN",
                }
            )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": ts,
        "trend": trend,
        "positions": positions,
        "bot_name": _bot_name(),
    }


class TelegramCommandListener:
    """Background getUpdates loop for /candles /summary /positions."""

    def __init__(
        self,
        notifier: TelegramNotifier,
        *,
        allowed_chat_ids: set[str],
        on_check_candle: Callable[[], dict[str, Any]],
        on_check_summary: Callable[[], dict[str, Any]],
        on_check_positions: Callable[[], dict[str, Any]],
        poll_timeout: int = 25,
    ) -> None:
        self._tg = notifier
        self._allowed = {str(c).strip() for c in allowed_chat_ids if str(c).strip()}
        self._on_candle = on_check_candle
        self._on_summary = on_check_summary
        self._on_positions = on_check_positions
        self._poll_timeout = int(poll_timeout)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int | None = None

    def start(self) -> None:
        if not self._tg._token or not self._allowed:
            logger.warning("telegram_commands_disabled — missing token or allowed chats")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="telegram-commands", daemon=True)
        self._thread.start()
        logger.info("telegram_commands_started chats=%s", sorted(self._allowed))

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                updates = self._tg.get_updates(offset=self._offset, timeout=self._poll_timeout)
            except Exception:
                logger.exception("telegram_get_updates_failed")
                self._stop.wait(3.0)
                continue
            for upd in updates:
                try:
                    uid = int(upd.get("update_id", 0))
                    self._offset = uid + 1
                    self._handle(upd)
                except Exception:
                    logger.exception("telegram_command_handle_failed")
            if not updates:
                self._stop.wait(0.2)

    def _handle(self, upd: dict[str, Any]) -> None:
        msg = upd.get("message") or {}
        text = str(msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if chat_id not in self._allowed:
            logger.info("telegram_cmd_ignored chat=%s text=%s", chat_id, text[:40])
            return
        thread_id = msg.get("message_thread_id")
        cmd = text.split()[0].split("@", 1)[0].lower()
        if cmd in {"/candles", "/check_candle", "/candle"}:
            reply = fmt_check_candle(self._on_candle())
        elif cmd in {"/summary", "/check_summary"}:
            reply = fmt_check_summary(self._on_summary())
        elif cmd in {"/positions", "/position", "/check_position", "/check_positions"}:
            reply = fmt_check_positions(self._on_positions())
        elif cmd in {"/help", "/start"}:
            reply = (
                "<b>Commands</b>\n"
                "/candles — last H1 OHLC + signal status\n"
                "/summary — W/L/R + equity today\n"
                "/positions — open MT5 positions"
            )
        else:
            return
        self._tg.send_to(reply, chat_id=chat_id, message_thread_id=thread_id)
