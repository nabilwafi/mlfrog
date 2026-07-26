"""Live MT5 broker — same FillResult surface as PaperBroker.

Safety: EXECUTION_ENABLED=False (default) → dry-run only (log + simulated fill).
Set production.EXECUTION_ENABLED=True (or --execute flag) to call order_send for real.

Broker-agnostic: any MT5 broker; symbol/filling mode resolved from symbol_info.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from production import EXECUTION_ENABLED
from production.paper.broker import FillResult, PaperBroker

logger = logging.getLogger(__name__)


def _order_type(side: str, *, closing: bool = False):
    import MetaTrader5 as mt5

    s = side.lower()
    if closing:
        return mt5.ORDER_TYPE_SELL if s == "long" else mt5.ORDER_TYPE_BUY
    return mt5.ORDER_TYPE_BUY if s == "long" else mt5.ORDER_TYPE_SELL


def _filling_mode(symbol: str) -> int:
    import MetaTrader5 as mt5

    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    # symbol_info.filling_mode is a SYMBOL_FILLING_* bitfield, not ORDER_FILLING_*.
    fm = int(getattr(info, "filling_mode", 0) or 0)
    if fm & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    if fm & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _price(symbol: str, side: str, *, closing: bool = False) -> float:
    import MetaTrader5 as mt5

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
    s = side.lower()
    if closing:
        # close long → sell at bid; close short → buy at ask
        return float(tick.bid if s == "long" else tick.ask)
    return float(tick.ask if s == "long" else tick.bid)


# Common MT5 retcodes we hit in live — short hint for logs / Telegram
_RETCODE_HINT: dict[int, str] = {
    10016: "invalid stops (SL/TP vs stops_level)",
    10017: "TRADE_DISABLED — AlgoTrading OFF, investor password, or symbol/account trade off",
    10018: "market closed",
    10019: "not enough money / margin",
    10027: "AutoTrading disabled in terminal (toolbar AlgoTrading button)",
    10030: "unsupported filling mode — try IOC/FOK",
}


def _trade_preflight(symbol: str) -> str | None:
    """Return a human reason if trading looks blocked before order_send; else None."""
    import MetaTrader5 as mt5

    term = mt5.terminal_info()
    acc = mt5.account_info()
    info = mt5.symbol_info(symbol)
    reasons: list[str] = []
    if term is not None and not bool(getattr(term, "trade_allowed", True)):
        reasons.append("terminal.trade_allowed=False (enable AlgoTrading button)")
    if acc is not None and not bool(getattr(acc, "trade_allowed", True)):
        reasons.append("account.trade_allowed=False (broker disabled trading / investor login)")
    if acc is not None and not bool(getattr(acc, "trade_expert", True)):
        reasons.append("account.trade_expert=False (EA trading not allowed on account)")
    if info is None:
        reasons.append(f"symbol_info({symbol!r}) is None — wrong name or not in Market Watch")
    else:
        # SYMBOL_TRADE_MODE_DISABLED = 0
        mode = int(getattr(info, "trade_mode", -1))
        if mode == 0:
            reasons.append(f"{symbol} trade_mode=DISABLED")
        if not bool(getattr(info, "visible", True)):
            reasons.append(f"{symbol} not visible in Market Watch")
    return "; ".join(reasons) if reasons else None


def _reject_message(retcode: int, comment: str | None, *, symbol: str) -> str:
    hint = _RETCODE_HINT.get(int(retcode), "")
    pre = _trade_preflight(symbol)
    parts = [f"retcode={retcode}"]
    if comment:
        parts.append(str(comment))
    if hint:
        parts.append(hint)
    if pre:
        parts.append(pre)
    return " | ".join(parts)


class LiveBroker:
    """Real-order broker with hard dry-run default.

    open_order / close_order match PaperBroker so ProductionPipeline can swap them.
    When execution is disabled, delegates fill simulation to an internal PaperBroker
    after logging the would-be request.
    """

    def __init__(
        self,
        *,
        symbol: str = "XAUUSDc",
        magic: int = 27001,
        execution_enabled: bool | None = None,
        deviation: int = 20,
    ) -> None:
        self.symbol = symbol
        self.magic = int(magic)
        self.execution_enabled = EXECUTION_ENABLED if execution_enabled is None else bool(execution_enabled)
        self.deviation = int(deviation)
        self._paper = PaperBroker()  # dry-run / fallback simulator
        self._tickets: dict[str, int] = {}  # trade_id → MT5 position ticket

    def open_order(
        self,
        *,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot: float,
        trade_id: str | None = None,
    ) -> FillResult:
        tid = trade_id or uuid.uuid4().hex
        if not self.execution_enabled:
            logger.warning(
                "DRY_RUN open side=%s lot=%.2f sl=%.2f tp=%.2f entry=%.2f trade_id=%s "
                "(set EXECUTION_ENABLED=True or --execute to send real order)",
                side,
                lot,
                stop_loss,
                take_profit,
                entry_price,
                tid,
            )
            return self._paper.open_order(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot=lot,
                trade_id=tid,
            )

        return self._send_open(
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot=lot,
            trade_id=tid,
        )

    def close_order(self, trade_id: str, exit_price: float) -> FillResult:
        if not self.execution_enabled:
            logger.warning(
                "DRY_RUN close trade_id=%s exit=%.2f (EXECUTION_ENABLED=False)",
                trade_id,
                exit_price,
            )
            return self._paper.close_order(trade_id, exit_price)

        ticket = self._tickets.get(trade_id)
        if ticket is None:
            return FillResult(
                success=False,
                trade_id=trade_id,
                fill_price=exit_price,
                spread=0.0,
                slippage=0.0,
                latency_ms=0.0,
                retry_count=0,
                broker_response="NOT_FOUND",
                error_message="no MT5 ticket for trade_id",
            )
        return self._send_close(trade_id=trade_id, ticket=ticket)

    def _send_open(
        self,
        *,
        side: str,
        stop_loss: float,
        take_profit: float,
        lot: float,
        trade_id: str,
    ) -> FillResult:
        import MetaTrader5 as mt5

        from production.monitoring import mt5_session

        mt5_session.select_symbol(self.symbol)
        blocked = _trade_preflight(self.symbol)
        if blocked:
            logger.error("LIVE open BLOCKED trade_id=%s %s", trade_id, blocked)
            # still attempt order_send — broker is source of truth; preflight only enriches errors
        price = _price(self.symbol, side)
        req: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lot),
            "type": _order_type(side),
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"xauusd:{trade_id[:12]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(self.symbol),
        }
        t0 = time.perf_counter()
        result = mt5.order_send(req)
        latency = (time.perf_counter() - t0) * 1000
        if result is None:
            err = str(mt5.last_error())
            if blocked:
                err = f"{err} | {blocked}"
            logger.error("order_send open failed trade_id=%s err=%s", trade_id, err)
            return FillResult(
                success=False,
                trade_id=trade_id,
                fill_price=price,
                spread=0.0,
                slippage=0.0,
                latency_ms=latency,
                retry_count=0,
                broker_response="ERROR",
                error_message=err,
            )
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            self._tickets[trade_id] = int(result.order or result.deal or 0)
            # Prefer position ticket if available
            positions = mt5.positions_get(symbol=self.symbol) or []
            for pos in positions:
                if int(getattr(pos, "magic", 0)) == self.magic:
                    self._tickets[trade_id] = int(pos.ticket)
                    break
            logger.info(
                "LIVE open OK trade_id=%s ticket=%s price=%.2f lot=%.2f",
                trade_id,
                self._tickets.get(trade_id),
                float(result.price or price),
                lot,
            )
            err_msg = None
        else:
            err_msg = _reject_message(int(result.retcode), result.comment, symbol=self.symbol)
            logger.error("LIVE open REJECT trade_id=%s %s", trade_id, err_msg)
        return FillResult(
            success=ok,
            trade_id=trade_id,
            fill_price=float(result.price or price),
            spread=0.0,
            slippage=0.0,
            latency_ms=latency,
            retry_count=0,
            broker_response=str(result.comment or result.retcode),
            error_message=err_msg,
        )

    def _send_close(self, *, trade_id: str, ticket: int) -> FillResult:
        import MetaTrader5 as mt5

        pos_list = mt5.positions_get(ticket=ticket) or ()
        if not pos_list:
            self._tickets.pop(trade_id, None)
            return FillResult(
                success=False,
                trade_id=trade_id,
                fill_price=0.0,
                spread=0.0,
                slippage=0.0,
                latency_ms=0.0,
                retry_count=0,
                broker_response="NOT_FOUND",
                error_message=f"position {ticket} gone",
            )
        pos = pos_list[0]
        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
        price = _price(self.symbol, side, closing=True)
        req: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(pos.volume),
            "type": _order_type(side, closing=True),
            "position": int(ticket),
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"xauusd:close:{trade_id[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(self.symbol),
        }
        t0 = time.perf_counter()
        result = mt5.order_send(req)
        latency = (time.perf_counter() - t0) * 1000
        if result is None:
            err = str(mt5.last_error())
            return FillResult(
                success=False,
                trade_id=trade_id,
                fill_price=price,
                spread=0.0,
                slippage=0.0,
                latency_ms=latency,
                retry_count=0,
                broker_response="ERROR",
                error_message=err,
            )
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            self._tickets.pop(trade_id, None)
            logger.info("LIVE close OK trade_id=%s ticket=%s price=%.2f", trade_id, ticket, float(result.price or price))
            err_msg = None
        else:
            err_msg = _reject_message(int(result.retcode), result.comment, symbol=self.symbol)
            logger.error("LIVE close REJECT trade_id=%s %s", trade_id, err_msg)
        return FillResult(
            success=ok,
            trade_id=trade_id,
            fill_price=float(result.price or price),
            spread=0.0,
            slippage=0.0,
            latency_ms=latency,
            retry_count=0,
            broker_response=str(result.comment or result.retcode),
            error_message=err_msg,
        )

    @staticmethod
    def pnl(side: str, entry: float, exit_px: float, lot: float) -> float:
        return PaperBroker.pnl(side, entry, exit_px, lot)

    @staticmethod
    def mark_excursions(side: str, entry: float, high: float, low: float) -> tuple[float, float]:
        return PaperBroker.mark_excursions(side, entry, high, low)
