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
        enabled: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        self._token = (bot_token or "").strip()
        self._chat_id = (chat_id or "").strip()
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
        body = json.dumps(
            {"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
        ).encode("utf-8")
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


def fmt_new_trade(p: dict[str, Any]) -> str:
    return (
        "<b>NEW TRADE</b>\n"
        f"ID: <code>{p.get('trade_id')}</code>\n"
        f"{p.get('symbol')} {str(p.get('side', '')).upper()}\n"
        f"Entry: {p.get('entry_price')} | SL: {p.get('stop_loss')} | TP: {p.get('take_profit')}\n"
        f"P: {p.get('probability')} | Meta: {p.get('meta_probability')} | Conf: {p.get('confidence')}\n"
        f"Risk: {p.get('risk_pct')} | Lots: {p.get('lot')}\n"
        f"Session: {p.get('session')} | Regime: {p.get('regime')}"
    )


def fmt_trade_closed(p: dict[str, Any]) -> str:
    return (
        "<b>TRADE CLOSED</b>\n"
        f"ID: <code>{p.get('trade_id')}</code>\n"
        f"Reason: {p.get('exit_reason')}\n"
        f"PnL: {p.get('pnl')} | R: {p.get('pnl_r')}\n"
        f"MAE: {p.get('mae')} | MFE: {p.get('mfe')}\n"
        f"Duration: {p.get('duration_seconds')}s\n"
        f"Equity: {p.get('equity')}"
    )


def fmt_skipped(p: dict[str, Any]) -> str:
    return (
        "<b>TRADE SKIPPED</b>\n"
        f"Reason: <b>{p.get('reason')}</b>\n"
        f"Threshold: {p.get('threshold')} | Value: {p.get('current_value')}\n"
        f"Symbol: {p.get('symbol')} | Corr: <code>{p.get('correlation_id')}</code>"
    )


def fmt_error(p: dict[str, Any]) -> str:
    return (
        "<b>EXECUTION ERROR</b>\n"
        f"Trade: <code>{p.get('trade_id')}</code>\n"
        f"Error: {p.get('error_message')}\n"
        f"Retries: {p.get('retry_count')}\n"
        f"TS: {p.get('timestamp')}"
    )


def fmt_daily(p: dict[str, Any]) -> str:
    return (
        "<b>DAILY SUMMARY</b>\n"
        f"Date: {p.get('date')}\n"
        f"Trades: {p.get('trades')} | WR: {p.get('winrate')}\n"
        f"PnL: {p.get('pnl')} | Daily R: {p.get('daily_r')}\n"
        f"DD: {p.get('drawdown')} | Skipped: {p.get('skipped')}\n"
        f"Heat: {p.get('heat_triggered')} | Equity: {p.get('equity')}"
    )


def fmt_health(p: dict[str, Any]) -> str:
    mt5 = p.get("mt5", "n/a")
    lines = [
        "<b>HEALTH</b>",
        f"Status: <b>{p.get('status', 'ok')}</b>",
        f"Reason: {p.get('reason')}",
        f"MT5: <b>{mt5}</b>",
    ]
    if p.get("mt5_error"):
        lines.append(f"MT5 err: {p.get('mt5_error')}")
    if mt5 == "connected":
        lines.append(
            f"MT5 acct: {p.get('mt5_login')} @ {p.get('mt5_server')} | "
            f"bal={p.get('mt5_balance')} eq={p.get('mt5_equity')}"
        )
        lines.append(
            f"Trade allowed: {p.get('mt5_trade_allowed')} | build={p.get('mt5_build')}"
        )
    lines.extend(
        [
            f"Equity: {p.get('equity')} | Open: {p.get('open_positions')}",
            f"Trades today: {p.get('trades_today')} | Skips: {p.get('skips')}",
            f"Bus pending: {p.get('bus_pending')} | Dropped: {p.get('bus_dropped')}",
            f"TS: {p.get('timestamp')}",
        ]
    )
    return "\n".join(lines)
