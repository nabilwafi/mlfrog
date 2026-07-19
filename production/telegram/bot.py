"""Telegram Bot API notifier (urllib — no heavy SDK)."""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str | None,
        chat_id: str | None,
        *,
        message_thread_id: int | str | None = None,
        enabled: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        self._token = (bot_token or "").strip()
        self._chat_id = (chat_id or "").strip()
        self._thread_id = _parse_thread_id(message_thread_id)
        self._enabled = bool(enabled and self._token and self._chat_id)
        self._verify_ssl = bool(verify_ssl)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ssl_context(self, *, verify: bool) -> ssl.SSLContext | None:
        if verify:
            return None  # urllib default verification
        # ponytail: corp/AV MITM breaks Telegram TLS; upgrade = install corp CA + verify_ssl true
        return ssl._create_unverified_context()

    def send(self, text: str, *, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            logger.info("telegram_disabled msg=%s", text[:120])
            return False
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        # Forum topics / message threads
        if self._thread_id is not None:
            payload["message_thread_id"] = self._thread_id
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            return self._post(req, verify=self._verify_ssl)
        except urllib.error.URLError as exc:
            reason = str(exc.reason) if getattr(exc, "reason", None) else str(exc)
            if self._verify_ssl and "CERTIFICATE_VERIFY_FAILED" in reason:
                logger.warning("telegram_ssl_verify_failed retrying_insecure")
                try:
                    return self._post(req, verify=False)
                except Exception as exc2:
                    self._log_fail(exc2)
                    return False
            self._log_fail(exc)
            return False
        except TimeoutError as exc:
            self._log_fail(exc)
            return False

    def _post(self, req: urllib.request.Request, *, verify: bool) -> bool:
        ctx = self._ssl_context(verify=verify)
        kwargs: dict[str, Any] = {"timeout": 10}
        if ctx is not None:
            kwargs["context"] = ctx
        try:
            with urllib.request.urlopen(req, **kwargs) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                ok = 200 <= resp.status < 300
                if not ok:
                    logger.error("telegram_http status=%s body=%s", resp.status, raw[:300])
                    return False
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return True
                if not data.get("ok", True):
                    logger.error("telegram_api_error body=%s", raw[:300])
                    return False
                return True
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("telegram_http status=%s body=%s", exc.code, body[:300])
            raise

    def _log_fail(self, exc: BaseException) -> None:
        if isinstance(exc, urllib.error.HTTPError):
            return  # already logged
        logger.error("telegram_failed err=%s", exc)


def _parse_thread_id(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        tid = int(value)
    except (TypeError, ValueError):
        return None
    return tid if tid > 0 else None


def split_chat_and_thread(raw: str | None) -> tuple[str | None, int | None]:
    """
    Accept plain chat_id, or 'chat_id:thread_id' / 'chat_id_thread_id' shortcuts.
    Example: '-100123:2' or '-100123_2' → ('-100123', 2)
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    if ":" in s:
        chat, _, thread = s.partition(":")
        return chat.strip() or None, _parse_thread_id(thread.strip())
    # trailing _<digits> after a negative chat id (user shortcut)
    if "_" in s:
        left, _, right = s.rpartition("_")
        if right.isdigit() and left.startswith("-"):
            return left, int(right)
    return s, None


def _env(p: dict[str, Any]) -> str:
    return str(p.get("environment") or p.get("env") or "n/a").upper()


def _side_label(side: Any) -> str:
    s = str(side or "").lower()
    if s in {"long", "buy"}:
        return "BUY"
    if s in {"short", "sell"}:
        return "SELL"
    return str(side or "n/a").upper()


def _fmt_price(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v if v is not None else "n/a")


def _fmt_risk_pct(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    # accept 0.01 or 1.0 meaning 1%
    if 0 < x <= 1:
        x *= 100.0
    return f"{x:.0f}%"


def _fmt_confidence(v: Any) -> str:
    try:
        return f"{float(v):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_utc_time(raw: Any) -> str:
    if raw is None:
        return "n/a"
    try:
        from datetime import datetime, timezone

        if isinstance(raw, datetime):
            ts = raw
        else:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.strftime("%H:%M UTC")
    except ValueError:
        return str(raw)


def _fmt_duration(seconds: Any) -> str:
    try:
        s = max(0, int(seconds))
    except (TypeError, ValueError):
        return "n/a"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _fmt_money(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if x >= 0 else ""
    # trim trailing .0 for whole dollars
    body = f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"
    return f"{sign}{body}$"


def _fmt_r(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}R"


def _exit_result(reason: Any) -> str:
    r = str(reason or "").upper()
    if r == "TP":
        return "TP HIT"
    if r == "SL":
        return "SL HIT"
    return r or "n/a"


def _skip_reason_label(reason: Any) -> str:
    r = str(reason or "unknown")
    return r.replace("_", " ").strip().title()


def _skip_value_label(p: dict[str, Any]) -> str:
    reason = str(p.get("reason") or "").lower()
    val = p.get("current_value")
    if val is None:
        return "n/a"
    if reason == "confidence":
        return _fmt_confidence(val)
    if reason == "meta":
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)
    if reason == "heat":
        try:
            return f"{float(val):.2f}R"
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def _skip_threshold_label(p: dict[str, Any]) -> str:
    reason = str(p.get("reason") or "").lower()
    thr = p.get("threshold")
    if thr is None:
        return "n/a"
    if reason == "confidence":
        return _fmt_confidence(thr)
    if reason == "meta":
        try:
            return f"{float(thr):.2f}"
        except (TypeError, ValueError):
            return str(thr)
    return str(thr)


def fmt_new_trade(p: dict[str, Any]) -> str:
    return (
        "🟢 <b>NEW TRADE</b>\n\n"
        f"Environment:\n{_env(p)}\n\n"
        f"Side:\n{_side_label(p.get('side'))}\n\n"
        f"Entry:\n{_fmt_price(p.get('entry_price'))}\n\n"
        f"SL:\n{_fmt_price(p.get('stop_loss'))}\n\n"
        f"TP:\n{_fmt_price(p.get('take_profit'))}\n\n"
        f"Risk:\n{_fmt_risk_pct(p.get('risk_pct'))}\n\n"
        f"Confidence:\n{_fmt_confidence(p.get('confidence'))}\n\n"
        f"Time:\n{_fmt_utc_time(p.get('entry_time') or p.get('timestamp'))}"
    )


def fmt_trade_closed(p: dict[str, Any]) -> str:
    return (
        "🔴 <b>TRADE CLOSED</b>\n\n"
        f"Environment:\n{_env(p)}\n\n"
        f"Result:\n{_exit_result(p.get('exit_reason'))}\n\n"
        f"PnL:\n{_fmt_money(p.get('pnl'))}\n\n"
        f"R:\n{_fmt_r(p.get('pnl_r'))}\n\n"
        f"Duration:\n{_fmt_duration(p.get('duration_seconds'))}"
    )


def fmt_skipped(p: dict[str, Any]) -> str:
    return (
        "⏭ <b>TRADE SKIPPED</b>\n\n"
        f"Environment:\n{_env(p)}\n\n"
        f"Reason:\n{_skip_reason_label(p.get('reason'))}\n\n"
        f"Value:\n{_skip_value_label(p)}\n\n"
        f"Threshold:\n{_skip_threshold_label(p)}"
    )


def fmt_error(p: dict[str, Any]) -> str:
    return (
        "<b>EXECUTION ERROR</b>\n"
        f"Trade: <code>{p.get('trade_id')}</code>\n"
        f"Error: {p.get('error_message')}\n"
        f"Retries: {p.get('retry_count')}\n"
        f"TS: {p.get('timestamp')}"
    )


def _fmt_daily_date(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "n/a"
    try:
        from datetime import date, datetime

        if len(s) >= 10 and s[4] == "-":
            d = date.fromisoformat(s[:10])
        else:
            d = datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        return d.strftime("%d %b %Y")
    except ValueError:
        return s


def _fmt_daily_uptime(p: dict[str, Any]) -> str:
    if p.get("uptime"):
        return str(p["uptime"])
    try:
        seconds = float(p.get("uptime_seconds", 0) or 0)
    except (TypeError, ValueError):
        return "n/a"
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    if h > 0 and m == 0:
        return f"{h}h"
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m"
    return f"{s}s"


def _fmt_pnl_pct(p: dict[str, Any]) -> str:
    raw = p.get("pnl_pct")
    if raw is None:
        return "n/a"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def fmt_daily(p: dict[str, Any]) -> str:
    status = str(p.get("status", "ok")).lower()
    if status == "ok":
        status_line = "🟢 OK"
    elif status == "degraded":
        status_line = "🟡 DEGRADED"
    else:
        status_line = "🔴 DOWN"

    trades = int(p.get("trades") or 0)
    wins = int(p.get("wins") or 0)
    losses = p.get("losses")
    if losses is None:
        losses = max(0, trades - wins)
    skips = int(p.get("skipped") or 0)

    return (
        "📊 <b>DAILY REPORT</b>\n\n"
        f"📅 {_fmt_daily_date(p.get('date'))}\n\n"
        f"Status:\n{status_line}\n\n"
        f"Trades:\n{trades}\n\n"
        f"Win/Loss:\n{wins}/{int(losses)}\n\n"
        f"Skip:\n{skips}\n\n"
        f"PnL:\n{_fmt_pnl_pct(p)}\n\n"
        f"Uptime:\n{_fmt_daily_uptime(p)}"
    )


def fmt_health(p: dict[str, Any]) -> str:
    status = str(p.get("status", "ok")).lower()
    if status == "ok":
        status_line = "🟢 ONLINE"
        bot_line = "✅ Running"
    elif status == "degraded":
        status_line = "🟡 DEGRADED"
        bot_line = "✅ Running"
    else:
        status_line = "🔴 OFFLINE"
        bot_line = "❌ Down"

    mt5 = str(p.get("mt5") or "").lower()
    if mt5 == "connected":
        mt5_line = "✅ Connected"
    elif mt5 in {"down", "offline", "disconnected", "error"}:
        mt5_line = "❌ Disconnected"
    elif "mt5" in p:
        mt5_line = "❌ Disconnected"
    else:
        mt5_line = "⚪ N/A"

    env = str(p.get("environment") or p.get("env") or "n/a").upper()
    uptime = p.get("uptime") or p.get("uptime_human") or "n/a"
    last_check = p.get("last_ping") or p.get("last_check") or "n/a"
    return (
        "❤️ <b>HEALTHCHECK</b>\n\n"
        f"Status: {status_line}\n\n"
        f"🌍 Environment:\n{env}\n\n"
        f"🤖 Bot:\n{bot_line}\n\n"
        f"📡 MT5:\n{mt5_line}\n\n"
        f"⏱ Uptime:\n{uptime}\n\n"
        f"Last Check:\n{last_check}"
    )
