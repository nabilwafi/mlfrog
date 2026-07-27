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
        return self.send_to(
            text,
            chat_id=self._chat_id,
            message_thread_id=self._thread_id,
            parse_mode=parse_mode,
        )

    def send_to(
        self,
        text: str,
        *,
        chat_id: str | int | None = None,
        message_thread_id: int | None = None,
        parse_mode: str = "HTML",
    ) -> bool:
        if not self._token:
            logger.info("telegram_disabled msg=%s", text[:120])
            return False
        cid = str(chat_id or self._chat_id or "").strip()
        if not cid:
            logger.info("telegram_no_chat msg=%s", text[:120])
            return False
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": cid,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        thread = message_thread_id if message_thread_id is not None else self._thread_id
        if thread is not None:
            payload["message_thread_id"] = int(thread)
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

    def get_updates(self, *, offset: int | None = None, timeout: int = 25) -> list[dict[str, Any]]:
        """Long-poll Telegram getUpdates (commands)."""
        if not self._token:
            return []
        params: dict[str, Any] = {
            "timeout": int(timeout),
            "allowed_updates": ["message"],
        }
        if offset is not None:
            params["offset"] = int(offset)
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        body = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            data = self._post_json(req, verify=self._verify_ssl)
        except urllib.error.URLError as exc:
            reason = str(exc.reason) if getattr(exc, "reason", None) else str(exc)
            if self._verify_ssl and "CERTIFICATE_VERIFY_FAILED" in reason:
                try:
                    data = self._post_json(req, verify=False)
                except Exception as exc2:
                    self._log_fail(exc2)
                    return []
            else:
                self._log_fail(exc)
                return []
        except Exception as exc:
            self._log_fail(exc)
            return []
        if not data or not data.get("ok"):
            return []
        result = data.get("result") or []
        return result if isinstance(result, list) else []

    def _post_json(self, req: urllib.request.Request, *, verify: bool) -> dict[str, Any]:
        ctx = self._ssl_context(verify=verify)
        kwargs: dict[str, Any] = {"timeout": 35}
        if ctx is not None:
            kwargs["context"] = ctx
        with urllib.request.urlopen(req, **kwargs) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}

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


def _fmt_trade_id(p: dict[str, Any]) -> str:
    """Compact display id from trade_id / signal_id hex."""
    raw = str(p.get("trade_id") or p.get("signal_id") or "").replace("-", "").strip()
    if not raw:
        return "n/a"
    try:
        return f"#{int(raw[-4:], 16)}"
    except ValueError:
        return f"#{raw[:8]}"


def _fmt_risk_pct(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    # accept 0.01 or 1.0 meaning 1%
    if 0 < x <= 1:
        x *= 100.0
    return f"{x:.1f}%"


def _fmt_confidence(v: Any) -> str:
    try:
        return f"{float(v):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_utc_time(raw: Any) -> str:
    """Full timestamp for all notifs: YYYY-MM-DD HH:mm:ss UTC."""
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
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return str(raw)


def _fmt_now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _bot_name(p: dict[str, Any] | None = None) -> str:
    if p and p.get("bot_name"):
        return str(p["bot_name"])
    try:
        from production import PIPELINE_VERSION

        return str(PIPELINE_VERSION)
    except Exception:
        return "mlfrog"


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


def _fmt_money_dollar(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if x >= 0 else "-"
    ax = abs(x)
    body = f"{ax:.0f}" if abs(ax - round(ax)) < 1e-9 else f"{ax:.2f}"
    return f"{sign}${body}"


def _fmt_r(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}R"


def _fmt_closed_pnl(p: dict[str, Any]) -> str:
    r = _fmt_r(p.get("pnl_r"))
    money = _fmt_money_dollar(p.get("pnl"))
    if r == "n/a" and money == "n/a":
        return "n/a"
    if money == "n/a":
        return r
    if r == "n/a":
        return f"({money})"
    return f"{r} ({money})"


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


def _skip_required_label(p: dict[str, Any]) -> str:
    reason = str(p.get("reason") or "").lower()
    thr = p.get("threshold")
    if thr is None:
        return "n/a"
    if reason == "confidence":
        return f"≥{_fmt_confidence(thr)}"
    if reason == "meta":
        try:
            return f"≥{float(thr):.2f}"
        except (TypeError, ValueError):
            return f"≥{thr}"
    if reason == "heat":
        return f"≥{thr}R"
    return f"≥{thr}"


def _symbol(p: dict[str, Any]) -> str:
    return str(p.get("symbol") or "XAUUSD").upper()


def _lot(p: dict[str, Any]) -> str:
    try:
        return f"{float(p.get('lot')):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _primary_pct(p: dict[str, Any]) -> str:
    raw = p.get("probability")
    if raw is None:
        raw = p.get("edge_score")
    if raw is None:
        return "n/a"
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return "n/a"
    if x <= 1.0:
        x *= 100.0
    return f"{x:.1f}%"


def _signed_pips(side: Any, entry: Any, price: Any) -> str:
    try:
        e, px = float(entry), float(price)
        raw = (px - e) / 0.1 if str(side).lower() in {"long", "buy"} else (e - px) / 0.1
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if raw >= 0 else ""
    return f"{sign}{raw:.0f}"


def _label_map(value: Any, mapping: dict[str, str], *, fallback: str = "n/a") -> str:
    key = str(value or "").strip().lower()
    if not key or key == "unknown":
        return fallback
    return mapping.get(key, str(value).replace("_", " ").title())


def _market_headline(p: dict[str, Any]) -> str:
    trend = str(p.get("trend") or "").lower()
    structure = str(p.get("structure") or "").lower()
    if structure == "trend" and trend == "bull":
        return "🟢 TRENDING BULLISH"
    if structure == "trend" and trend == "bear":
        return "🔴 TRENDING BEARISH"
    if structure in {"range", "sideways"} or trend == "sideways":
        return "🟡 RANGE / SIDEWAYS"
    if structure == "compression":
        return "🟠 COMPRESSION"
    regime = str(p.get("regime") or "").strip()
    if regime and regime.lower() != "unknown":
        return f"⚪ {regime.replace('_', ' ').title()}"
    return "⚪ UNKNOWN"


def _market_block(p: dict[str, Any]) -> str:
    trend = _label_map(p.get("trend"), {"bull": "Bullish", "bear": "Bearish", "sideways": "Sideways"})
    vol = _label_map(p.get("volatility"), {"high": "High", "medium": "Medium", "low": "Low"})
    mom = _label_map(p.get("momentum"), {"strong": "Strong", "normal": "Normal", "weak": "Weak"})
    sess = _label_map(
        p.get("session"),
        {
            "london": "London",
            "newyork": "New York",
            "asia": "Asia",
            "overlap": "London/NY Overlap",
            "other": "Other",
        },
    )
    return (
        "━━━━━━━━━━━━━━\n"
        "📊 Market Context\n"
        f"{_market_headline(p)}\n\n"
        f"📈 Trend   : {trend}\n"
        f"🔥 Vol     : {vol}\n"
        f"⚡ Momentum: {mom}\n"
        f"📉 Session : {sess}"
    )


def _trail_policy_line(p: dict[str, Any]) -> str:
    try:
        mult = float(p.get("trail_atr_mult"))
    except (TypeError, ValueError):
        mult = None
    try:
        act = float(p.get("trail_activate_r"))
    except (TypeError, ValueError):
        act = None
    if mult is None:
        return "🔄 Exit  : ATR Trail"
    if act is None:
        return f"🔄 Exit  : ATR Trail {mult:.2f}"
    return f"🔄 Exit  : ATR Trail {mult:.2f} (after +{act:g}R)"


def fmt_new_trade(p: dict[str, Any]) -> str:
    side = _side_label(p.get("side"))
    side_emoji = "📈" if side == "BUY" else "📉"
    return (
        "🟢 <b>OPEN POSITION</b>\n\n"
        f"🟢 {_symbol(p)} | TRADE OPEN\n\n"
        f"{side_emoji} {side}\n\n"
        f"💰 Entry : {_fmt_price(p.get('entry_price'))}\n"
        f"🛑 SL    : {_fmt_price(p.get('stop_loss'))}\n"
        f"{_trail_policy_line(p)}\n"
        f"📌 TP ref: {_fmt_price(p.get('take_profit'))}\n\n"
        f"📦 Lot   : {_lot(p)}\n"
        f"⚖️ Risk  : {_fmt_risk_pct(p.get('risk_pct'))}\n"
        f"📊 Primary: {_primary_pct(p)}\n"
        f"🆔 {_fmt_trade_id(p)}\n\n"
        f"{_market_block(p)}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('entry_time') or p.get('timestamp'))}\n"
        f"🤖 Bot      : {_bot_name(p)}\n\n"
        "Status: RUNNING"
    )


def fmt_trail_update(p: dict[str, Any]) -> str:
    side = _side_label(p.get("side"))
    side_emoji = "📈" if side == "BUY" else "📉"
    pnl_money = _fmt_money_dollar(p.get("unrealized_pnl"))
    pips = _signed_pips(p.get("side"), p.get("entry_price"), p.get("mark_price"))
    return (
        "🟡 <b>UPDATE + TRAILING STOP</b>\n\n"
        f"🟡 {_symbol(p)} | POSITION UPDATE\n\n"
        f"{side_emoji} {side}\n\n"
        f"💵 Price : {_fmt_price(p.get('mark_price'))}\n"
        f"📈 Move  : {pips} Pips\n"
        f"💲 Profit: {pnl_money}\n\n"
        "🔄 Trailing Stop: ACTIVE\n"
        f"🛡️ New SL: {_fmt_price(p.get('stop_loss'))}\n\n"
        f"{_market_block(p)}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('timestamp'))}\n"
        f"🤖 Bot      : {_bot_name(p)}\n\n"
        "Status: PROFIT LOCKED"
    )


def fmt_trade_closed(p: dict[str, Any]) -> str:
    reason = str(p.get("exit_reason") or "").upper()
    side = _side_label(p.get("side"))
    try:
        pnl_f = float(p.get("pnl") or 0)
    except (TypeError, ValueError):
        pnl_f = 0.0
    if reason in {"SL", "BROKER"} or (reason == "TRAIL" and pnl_f < 0):
        header = "🔴 <b>STOP LOSS</b>"
        title = f"🔴 {_symbol(p)} | STOP LOSS HIT"
        status = "CLOSED · SL"
    elif reason == "TRAIL":
        header = "🟢 <b>TRAIL STOP</b>"
        title = f"🟢 {_symbol(p)} | TRAIL STOP HIT"
        status = "CLOSED · TRAIL"
    elif reason == "TP":
        header = "🟢 <b>TAKE PROFIT</b>"
        title = f"🟢 {_symbol(p)} | TAKE PROFIT HIT"
        status = "CLOSED · TP"
    elif reason == "TIMEOUT":
        header = "⚪ <b>TIMEOUT</b>"
        title = f"⚪ {_symbol(p)} | TIME EXIT"
        status = "CLOSED · TIMEOUT"
    else:
        header = "🔴 <b>STOP LOSS</b>" if pnl_f < 0 else "🟢 <b>POSITION CLOSED</b>"
        title = f"{'🔴' if pnl_f < 0 else '🟢'} {_symbol(p)} | CLOSED"
        status = "CLOSED · SL" if pnl_f < 0 else "CLOSED"

    side_line = f"{'📉' if side == 'BUY' else '📈'} {side} CLOSED"
    pips = _signed_pips(p.get("side"), p.get("entry_price"), p.get("exit_price"))
    return (
        f"{header}\n\n"
        f"{title}\n\n"
        f"{side_line}\n\n"
        f"💰 Entry : {_fmt_price(p.get('entry_price'))}\n"
        f"🏁 Exit  : {_fmt_price(p.get('exit_price'))}\n\n"
        f"📉 Move  : {pips} Pips\n"
        f"💲 PnL   : {_fmt_closed_pnl(p)}\n"
        f"⏱ Duration: {_fmt_duration(p.get('duration_seconds'))}\n"
        f"🆔 {_fmt_trade_id(p)}\n\n"
        f"{_market_block(p)}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('exit_time') or p.get('timestamp'))}\n"
        f"🤖 Bot      : {_bot_name(p)}\n\n"
        f"Status: {status}"
    )


def fmt_skipped(p: dict[str, Any]) -> str:
    return (
        "⏭️ <b>TRADE SKIPPED</b>\n\n"
        f"Price    : {_fmt_price(p.get('entry_price') or p.get('price'))}\n\n"
        f"Reason   : {_skip_reason_label(p.get('reason'))}\n"
        f"Value    : {_skip_value_label(p)}\n"
        f"Required : {_skip_required_label(p)}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('timestamp') or p.get('entry_time'))}\n"
        f"🤖 Bot      : {_bot_name(p)}"
    )


def fmt_error(p: dict[str, Any]) -> str:
    return (
        "🚨 <b>EXECUTION ERROR</b>\n\n"
        f"Trade: <code>{p.get('trade_id')}</code>\n"
        f"Error: {p.get('error_message')}\n"
        f"Retries: {p.get('retry_count')}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('timestamp'))}\n"
        f"🤖 Bot      : {_bot_name(p)}"
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
        bot_status = "🟢 Online"
    elif status == "degraded":
        bot_status = "🟡 Degraded"
    else:
        bot_status = "🔴 Down"

    trades = int(p.get("trades") or 0)
    wins = int(p.get("wins") or 0)
    losses = p.get("losses")
    if losses is None:
        losses = max(0, trades - wins)
    wr = p.get("winrate")
    if wr is None:
        wr = (wins / trades) if trades else 0.0
    try:
        wr_pct = float(wr) * 100.0 if float(wr) <= 1.0 else float(wr)
    except (TypeError, ValueError):
        wr_pct = 0.0

    balance = p.get("balance", p.get("day_start_equity", p.get("equity")))
    equity = p.get("equity")
    pnl = p.get("pnl")
    pnl_pct = _fmt_pnl_pct(p)
    best = p.get("best_trade_pnl")
    worst = p.get("worst_trade_pnl")

    def _money(v: Any, *, signed: bool = False) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if signed:
            sign = "+" if x >= 0 else "-"
            return f"{sign}${abs(x):.2f}"
        return f"${x:.2f}"

    symbol = _symbol(p)
    running = int(p.get("running") or p.get("open_positions") or 0)
    ts = _fmt_utc_time(p.get("timestamp")) if p.get("timestamp") else _fmt_now_utc()
    return (
        f"🟡 <b>{symbol} DAILY</b>\n\n"
        f"💰 Balance: {_money(balance)}\n"
        f"📈 Equity: {_money(equity)}\n"
        f"💵 PnL: {_money(pnl, signed=True)} ({pnl_pct})\n\n"
        f"📊 Trades: {trades}\n"
        f"✅ W: {wins} | ❌ L: {int(losses)} | 🔄 R: {running}\n"
        f"🎯 WR: {wr_pct:.0f}%\n\n"
        f"🔥 Best: {_money(best, signed=True) if best is not None else 'n/a'}\n"
        f"💀 Worst: {_money(worst, signed=True) if worst is not None else 'n/a'}\n\n"
        f"🤖 Status: {bot_status}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {ts}\n"
        f"🤖 Bot      : {_bot_name(p)}"
    )


def fmt_check_summary(p: dict[str, Any]) -> str:
    """On-demand /check_summary — same core stats + W/L/R."""
    payload = {**p, "status": p.get("status", "ok"), "timestamp": p.get("timestamp") or _fmt_now_utc()}
    text = fmt_daily(payload).replace("DAILY", "SUMMARY", 1)
    opens = p.get("open_list") or []
    if not opens:
        return text
    lines = [text, "", "<b>Open positions</b>"]
    for o in opens[:10]:
        side = _side_label(o.get("side"))
        lines.append(
            f"• {side} {_fmt_price(o.get('entry_price'))} "
            f"SL {_fmt_price(o.get('stop_loss'))} lot={o.get('lot', 'n/a')}"
        )
    return "\n".join(lines)


def fmt_check_candle(p: dict[str, Any]) -> str:
    """On-demand /candles — user template."""
    symbol = _symbol(p)
    tf = str(p.get("timeframe") or "H1").upper()
    closed = p.get("closed") or {}
    ts = str(p.get("timestamp") or closed.get("time_utc") or _fmt_now_utc())
    if not ts.endswith("UTC") and "UTC" not in ts:
        ts = f"{ts} UTC"
    trend = str(p.get("trend") or closed.get("trend") or "n/a")
    signal = str(p.get("signal") or "WAIT").upper()
    status = str(p.get("status") or closed.get("state") or "n/a")
    closed_line = (
        f"✅ {tf} candle has closed.\n🔍 Checking strategy..."
        if closed
        else f"⏸ No closed {tf} candle yet."
    )
    return (
        f"🔔 <b>CHECK CANDLE {tf}</b>\n\n"
        f"📌 Symbol    : {symbol}\n"
        f"🕒 Timeframe : {tf}\n"
        f"📅 Timestamp : {ts}\n\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"{closed_line}\n\n"
        f"💰 Open   : {_fmt_price(closed.get('open'))}\n"
        f"📈 High   : {_fmt_price(closed.get('high'))}\n"
        f"📉 Low    : {_fmt_price(closed.get('low'))}\n"
        f"💵 Close  : {_fmt_price(closed.get('close'))}\n\n"
        f"📊 Trend  : {trend}\n"
        f"🎯 Signal : {signal}\n"
        f"📌 Status : {status}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Bot      : {_bot_name(p)}\n"
        f"🆔 Candle   : {p.get('candle_id') or closed.get('candle_id') or 'n/a'}"
    )


def fmt_check_positions(p: dict[str, Any]) -> str:
    """On-demand /positions — open MT5/bot positions."""
    symbol = _symbol(p)
    tf = str(p.get("timeframe") or "H1").upper()
    ts = str(p.get("timestamp") or _fmt_now_utc())
    if "UTC" not in ts:
        ts = f"{ts} UTC"
    tag = f"#{symbol}"
    positions = p.get("positions") or []
    if not positions:
        return (
            "📋 <b>CHECK POSITION</b>\n\n"
            f"📌 Symbol    : {symbol}\n"
            f"🕒 Timeframe : {tf}\n"
            f"📅 Timestamp : {ts}\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "ℹ️ No active position found.\n\n"
            f"📊 Market Status : {p.get('trend') or 'n/a'}\n"
            "🎯 Next Action   : Waiting for valid setup.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Bot : {_bot_name(p)}\n\n"
            f"{tag} #NoPosition"
        )

    blocks: list[str] = []
    for i, pos in enumerate(positions):
        side = _side_label(pos.get("side"))
        if side not in {"BUY", "SELL"}:
            side = "NONE"
        status = str(pos.get("status") or "OPEN").upper()
        profit = pos.get("profit")
        try:
            pf = float(profit)
            profit_s = f"+${pf:.2f}" if pf >= 0 else f"-${abs(pf):.2f}"
        except (TypeError, ValueError):
            profit_s = "n/a"
        pips = pos.get("pips")
        try:
            pp = float(pips)
            pips_s = f"+{pp:.0f}" if pp >= 0 else f"{pp:.0f}"
        except (TypeError, ValueError):
            pips_s = "n/a"
        header = "📋 <b>CHECK POSITION</b>" if i == 0 else "📋 <b>CHECK POSITION</b> (cont.)"
        blocks.append(
            f"{header}\n\n"
            f"📌 Symbol    : {symbol}\n"
            f"🕒 Timeframe : {tf}\n"
            f"📅 Timestamp : {ts}\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 Position : {side}\n"
            f"🎫 Ticket   : {pos.get('ticket') or 'n/a'}\n"
            f"💵 Lot      : {_lot(pos)}\n"
            f"💲 Entry    : {_fmt_price(pos.get('entry_price'))}\n"
            f"📍 Current  : {_fmt_price(pos.get('current_price'))}\n\n"
            f"💹 Profit   : {profit_s}\n"
            f"📊 P/L Pips : {pips_s}\n\n"
            f"🛡️ Stop Loss  : {_fmt_price(pos.get('stop_loss'))}\n"
            f"🎯 Take Profit : {_fmt_price(pos.get('take_profit'))}\n\n"
            f"📌 Status : {status}\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Bot : {_bot_name(p)}\n\n"
            f"{tag} #Position"
        )
    return "\n\n".join(blocks)


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
    last_check = p.get("last_ping") or p.get("last_check") or _fmt_now_utc()
    return (
        "❤️ <b>HEALTHCHECK</b>\n\n"
        f"Status: {status_line}\n\n"
        f"🌍 Environment:\n{env}\n\n"
        f"🤖 Bot:\n{bot_line}\n\n"
        f"📡 MT5:\n{mt5_line}\n\n"
        f"⏱ Uptime:\n{uptime}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 Timestamp : {_fmt_utc_time(p.get('timestamp')) if p.get('timestamp') else last_check}\n"
        f"🤖 Bot      : {_bot_name(p)}"
    )
