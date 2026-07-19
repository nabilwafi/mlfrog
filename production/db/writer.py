"""Postgres writer — worker-side only; never called on broker hot path."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class PostgresWriter:
    """
    Thin SQL writer. Uses psycopg2 when available; otherwise no-op with warning
    so unit tests / dry-run paper can run without a live DB.
    """

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._conn = None
        self._enabled = bool(dsn)
        if self._enabled:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 not installed — DB writer disabled")
                self._enabled = False
                self._psycopg2 = None
        else:
            self._psycopg2 = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if not self._enabled or self._psycopg2 is None:
            yield None
            return
        conn = self._psycopg2.connect(self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def apply_schema(self, schema_sql: str) -> None:
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(schema_sql)

    def upsert_signal(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.signals (
            signal_id, correlation_id, timestamp, symbol, side,
            probability, meta_probability, confidence,
            threshold_meta, threshold_confidence,
            model_version, meta_version, feature_version, label_version, pipeline_version,
            accepted, session, regime
        ) VALUES (
            %(signal_id)s, %(correlation_id)s, %(timestamp)s, %(symbol)s, %(side)s,
            %(probability)s, %(meta_probability)s, %(confidence)s,
            %(threshold_meta)s, %(threshold_confidence)s,
            %(model_version)s, %(meta_version)s, %(feature_version)s, %(label_version)s, %(pipeline_version)s,
            %(accepted)s, %(session)s, %(regime)s
        )
        ON CONFLICT (signal_id) DO UPDATE SET
            accepted = EXCLUDED.accepted,
            confidence = EXCLUDED.confidence
        """
        self._execute(sql, row)

    def upsert_trade(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.trades (
            trade_id, signal_id, correlation_id, symbol, side,
            entry_time, exit_time, entry_price, exit_price, stop_loss, take_profit,
            lot, risk_pct, pnl, pnl_r, duration_seconds, mae, mfe, exit_reason, status,
            session, regime, probability, meta_probability, confidence, updated_at
        ) VALUES (
            %(trade_id)s, %(signal_id)s, %(correlation_id)s, %(symbol)s, %(side)s,
            %(entry_time)s, %(exit_time)s, %(entry_price)s, %(exit_price)s, %(stop_loss)s, %(take_profit)s,
            %(lot)s, %(risk_pct)s, %(pnl)s, %(pnl_r)s, %(duration_seconds)s, %(mae)s, %(mfe)s,
            %(exit_reason)s, %(status)s, %(session)s, %(regime)s,
            %(probability)s, %(meta_probability)s, %(confidence)s, NOW()
        )
        ON CONFLICT (trade_id) DO UPDATE SET
            exit_time = EXCLUDED.exit_time,
            exit_price = EXCLUDED.exit_price,
            pnl = EXCLUDED.pnl,
            pnl_r = EXCLUDED.pnl_r,
            duration_seconds = EXCLUDED.duration_seconds,
            mae = EXCLUDED.mae,
            mfe = EXCLUDED.mfe,
            exit_reason = EXCLUDED.exit_reason,
            status = EXCLUDED.status,
            updated_at = NOW()
        """
        self._execute(sql, row)

    def insert_skip(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.skip_logs (
            correlation_id, signal_id, timestamp, symbol, reason, threshold, current_value, detail
        ) VALUES (
            %(correlation_id)s, %(signal_id)s, %(timestamp)s, %(symbol)s, %(reason)s,
            %(threshold)s, %(current_value)s, %(detail)s::jsonb
        )
        """
        payload = dict(row)
        payload["detail"] = json.dumps(payload.get("detail") or {})
        self._execute(sql, payload)

    def insert_execution(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.execution_logs (
            correlation_id, trade_id, timestamp, latency_ms, broker_response,
            spread, slippage, retry_count, success, error_message
        ) VALUES (
            %(correlation_id)s, %(trade_id)s, %(timestamp)s, %(latency_ms)s, %(broker_response)s,
            %(spread)s, %(slippage)s, %(retry_count)s, %(success)s, %(error_message)s
        )
        """
        self._execute(sql, row)

    def upsert_daily(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.daily_statistics (
            date, equity, daily_r, drawdown, heat_triggered, trades, wins, winrate,
            pnl, skipped, meta_rejects, confidence_rejects, updated_at
        ) VALUES (
            %(date)s, %(equity)s, %(daily_r)s, %(drawdown)s, %(heat_triggered)s, %(trades)s,
            %(wins)s, %(winrate)s, %(pnl)s, %(skipped)s, %(meta_rejects)s, %(confidence_rejects)s, NOW()
        )
        ON CONFLICT (date) DO UPDATE SET
            equity = EXCLUDED.equity,
            daily_r = EXCLUDED.daily_r,
            drawdown = EXCLUDED.drawdown,
            heat_triggered = EXCLUDED.heat_triggered,
            trades = EXCLUDED.trades,
            wins = EXCLUDED.wins,
            winrate = EXCLUDED.winrate,
            pnl = EXCLUDED.pnl,
            skipped = EXCLUDED.skipped,
            meta_rejects = EXCLUDED.meta_rejects,
            confidence_rejects = EXCLUDED.confidence_rejects,
            updated_at = NOW()
        """
        self._execute(sql, row)

    def insert_metric(self, name: str, value: float, *, labels: dict | None = None, correlation_id: str | None = None) -> None:
        sql = """
        INSERT INTO trading.metrics (name, value, labels, correlation_id)
        VALUES (%(name)s, %(value)s, %(labels)s::jsonb, %(correlation_id)s)
        """
        self._execute(
            sql,
            {
                "name": name,
                "value": value,
                "labels": json.dumps(labels or {}),
                "correlation_id": correlation_id,
            },
        )

    def insert_audit(self, component: str, action: str, detail: dict, *, correlation_id: str | None = None) -> None:
        sql = """
        INSERT INTO trading.audit_logs (correlation_id, component, action, detail)
        VALUES (%(correlation_id)s, %(component)s, %(action)s, %(detail)s::jsonb)
        """
        self._execute(
            sql,
            {
                "correlation_id": correlation_id,
                "component": component,
                "action": action,
                "detail": json.dumps(detail),
            },
        )

    def upsert_candle(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO trading.candles (
            symbol, timeframe, timestamp, open, high, low, close,
            tick_volume, spread, real_volume, source, features
        ) VALUES (
            %(symbol)s, %(timeframe)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s,
            %(tick_volume)s, %(spread)s, %(real_volume)s, %(source)s, %(features)s::jsonb
        )
        ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            tick_volume = EXCLUDED.tick_volume,
            spread = EXCLUDED.spread,
            real_volume = EXCLUDED.real_volume,
            features = EXCLUDED.features
        """
        payload = dict(row)
        feats = payload.get("features")
        payload["features"] = json.dumps(feats) if feats is not None else None
        self._execute(sql, payload)

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self.connection() as conn:
            if conn is None:
                logger.debug("db_noop sql=%s keys=%s", sql.split()[0], list(params.keys()))
                return
            with conn.cursor() as cur:
                cur.execute(sql, params)
