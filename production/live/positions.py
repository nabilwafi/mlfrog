"""Recover open MT5 positions into pipeline state after restart."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from production import EXIT_MODE, RISK_PCT, TRAIL_ACTIVATE_R, TRAIL_ATR_MULT
from production.paper.state import OpenPosition
from settings.strategy import SL_ATR_MULT

logger = logging.getLogger(__name__)

_DEFAULT_MAGIC = 27001


@dataclass(frozen=True)
class BrokerOpenPosition:
    trade_id: str
    broker_ticket: int
    side: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float
    comment: str
    signal_id: str | None = None
    correlation_id: str | None = None
    risk_pct: float | None = None
    session: str | None = None
    regime: str | None = None
    probability: float | None = None
    meta_probability: float | None = None
    confidence: float | None = None
    from_db: bool = False


def _trade_id_from_comment(comment: str, ticket: int) -> str:
    c = str(comment or "").strip()
    if c.startswith("xauusd:"):
        tag = c.split(":", 1)[1].strip()
        if tag:
            return f"recovered-{tag}-{ticket}"
    return f"recovered-{ticket}"


def _side_from_mt5(pos_type: int) -> str:
    import MetaTrader5 as mt5

    return "long" if int(pos_type) == mt5.POSITION_TYPE_BUY else "short"


def _atr_from_sl(*, side: str, entry: float, stop_loss: float) -> float:
    sl = float(stop_loss or 0)
    if sl <= 0:
        return max(float(entry) * 0.001, 1e-6)
    if side == "long":
        dist = float(entry) - sl
    else:
        dist = sl - float(entry)
    if dist <= 0:
        return max(float(entry) * 0.001, 1e-6)
    return dist / float(SL_ATR_MULT)


def fetch_broker_open_positions(
    *,
    symbol: str,
    magic: int = _DEFAULT_MAGIC,
    ours_only: bool = True,
) -> list[BrokerOpenPosition]:
    """Read open MT5 positions for symbol (optionally filtered by bot magic)."""
    import MetaTrader5 as mt5

    from production.monitoring import mt5_session

    if not mt5_session.ensure_connected():
        raise RuntimeError("MT5 not connected — cannot recover positions")

    mt5_session.select_symbol(symbol)
    raw = mt5.positions_get(symbol=symbol) or ()
    out: list[BrokerOpenPosition] = []
    for p in raw:
        if ours_only and int(getattr(p, "magic", 0) or 0) != int(magic):
            continue
        ticket = int(getattr(p, "ticket", 0) or 0)
        if ticket <= 0:
            continue
        side = _side_from_mt5(getattr(p, "type", 0))
        entry = float(getattr(p, "price_open", 0) or 0)
        sl = float(getattr(p, "sl", 0) or 0)
        tp = float(getattr(p, "tp", 0) or 0)
        lot = float(getattr(p, "volume", 0) or 0)
        comment = str(getattr(p, "comment", "") or "")
        t_open = int(getattr(p, "time", 0) or 0)
        entry_time = datetime.fromtimestamp(t_open, tz=timezone.utc) if t_open else datetime.now(timezone.utc)
        out.append(
            BrokerOpenPosition(
                trade_id=_trade_id_from_comment(comment, ticket),
                broker_ticket=ticket,
                side=side,
                entry_time=entry_time,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                lot=lot,
                comment=comment,
            )
        )
    return out


def enrich_with_db(rows: list[BrokerOpenPosition], db: Any, *, symbol: str) -> list[BrokerOpenPosition]:
    """Prefer original signal_id / metadata from {schema}.trades by ticket_id when present."""
    if not rows or db is None or not getattr(db, "enabled", False):
        return rows
    fetch = getattr(db, "fetch_open_trades_by_ticket", None)
    if not callable(fetch):
        return rows
    tickets = [r.broker_ticket for r in rows]
    by_ticket = fetch(symbol=symbol, tickets=tickets) or {}
    if not by_ticket:
        return rows
    out: list[BrokerOpenPosition] = []
    for r in rows:
        db_row = by_ticket.get(int(r.broker_ticket))
        if not db_row:
            out.append(r)
            continue
        entry_time = db_row.get("entry_time") or r.entry_time
        if isinstance(entry_time, datetime) and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        out.append(
            replace(
                r,
                trade_id=str(db_row.get("trade_id") or r.trade_id),
                signal_id=str(db_row.get("signal_id") or r.trade_id),
                correlation_id=str(db_row.get("correlation_id") or "") or None,
                # live SL/TP from MT5 win over stale DB (trail may have moved)
                entry_price=float(db_row.get("entry_price") or r.entry_price),
                lot=float(db_row.get("lot") or r.lot),
                risk_pct=float(db_row["risk_pct"]) if db_row.get("risk_pct") is not None else r.risk_pct,
                entry_time=entry_time,
                session=db_row.get("session"),
                regime=db_row.get("regime"),
                probability=float(db_row["probability"]) if db_row.get("probability") is not None else None,
                meta_probability=(
                    float(db_row["meta_probability"]) if db_row.get("meta_probability") is not None else None
                ),
                confidence=float(db_row["confidence"]) if db_row.get("confidence") is not None else None,
                from_db=True,
            )
        )
        logger.info(
            "recover_matched_db trade_id=%s ticket=%s",
            out[-1].trade_id,
            r.broker_ticket,
        )
    return out


def broker_open_from_db_row(row: dict[str, Any]) -> BrokerOpenPosition:
    ticket = int(row["ticket_id"])
    entry_time = row.get("entry_time") or datetime.now(timezone.utc)
    if isinstance(entry_time, datetime) and entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    return BrokerOpenPosition(
        trade_id=str(row.get("trade_id") or row.get("signal_id") or f"recovered-{ticket}"),
        broker_ticket=ticket,
        side=str(row["side"]),
        entry_time=entry_time,
        entry_price=float(row["entry_price"]),
        stop_loss=float(row["stop_loss"]),
        take_profit=float(row.get("take_profit") or 0),
        lot=float(row["lot"]),
        comment="",
        signal_id=str(row.get("signal_id") or ticket),
        correlation_id=str(row.get("correlation_id") or "") or None,
        risk_pct=float(row["risk_pct"]) if row.get("risk_pct") is not None else None,
        session=row.get("session"),
        regime=row.get("regime"),
        probability=float(row["probability"]) if row.get("probability") is not None else None,
        meta_probability=float(row["meta_probability"]) if row.get("meta_probability") is not None else None,
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        from_db=True,
    )


def stale_db_opens(db: Any, *, symbol: str, live_tickets: set[int]) -> list[BrokerOpenPosition]:
    """DB open-book rows whose MT5 position is already gone (closed while bot was down)."""
    if db is None or not getattr(db, "enabled", False):
        return []
    fetch = getattr(db, "fetch_open_trades_by_ticket", None)
    if not callable(fetch):
        return []
    by_ticket = fetch(symbol=symbol) or {}
    live = {int(t) for t in live_tickets}
    out: list[BrokerOpenPosition] = []
    for ticket, row in by_ticket.items():
        if int(ticket) in live:
            continue
        out.append(broker_open_from_db_row(row))
    return out


def recover_positions_into_state(
    state: Any,
    broker: Any,
    recovered: list[BrokerOpenPosition],
    *,
    symbol: str,
) -> int:
    """Merge broker positions into PortfolioState; wire LiveBroker tickets."""
    n = 0
    for r in recovered:
        ticket_key = str(r.broker_ticket)
        if ticket_key in state.open_positions:
            logger.info("recover_skip_existing ticket=%s", r.broker_ticket)
            continue
        if any(int(p.broker_ticket or 0) == int(r.broker_ticket) for p in state.open_positions.values()):
            logger.info("recover_skip_ticket_dup ticket=%s", r.broker_ticket)
            continue
        atr = _atr_from_sl(side=r.side, entry=r.entry_price, stop_loss=r.stop_loss)
        corr = r.correlation_id or uuid.uuid4().hex
        pos = OpenPosition(
            trade_id=ticket_key,
            signal_id=str(r.signal_id or r.trade_id),
            side=r.side,
            entry_time=r.entry_time,
            entry_price=r.entry_price,
            stop_loss=r.stop_loss,
            take_profit=r.take_profit,
            lot=r.lot,
            risk_pct=float(r.risk_pct if r.risk_pct is not None else RISK_PCT),
            atr=atr,
            correlation_id=corr,
            broker_ticket=r.broker_ticket,
            extreme_fav=r.entry_price,
            bars_held=0,
            meta={
                "recovered": True,
                "from_db": bool(r.from_db),
                "broker_comment": r.comment,
                "initial_sl": r.stop_loss,
                "exit_mode": EXIT_MODE,
                "trail_atr_mult": TRAIL_ATR_MULT,
                "trail_activate_r": TRAIL_ACTIVATE_R,
                "session": r.session,
                "regime": r.regime,
                "probability": r.probability,
                "meta_probability": r.meta_probability,
                "confidence": r.confidence,
            },
        )
        state.open_positions[ticket_key] = pos
        state.register_signal_key(str(r.signal_id or ticket_key))
        register = getattr(broker, "register_recovered", None)
        if callable(register):
            register(ticket_key, r.broker_ticket)
        logger.info(
            "position_recovered ticket=%s side=%s lot=%.2f entry=%.2f sl=%.2f from_db=%s",
            r.broker_ticket,
            r.side,
            r.lot,
            r.entry_price,
            r.stop_loss,
            r.from_db,
        )
        n += 1
    if n:
        logger.info("recover_done symbol=%s count=%s open=%s", symbol, n, len(state.open_positions))
    return n
