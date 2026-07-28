"""Postgres writer — testing (rich) | production (lean). Worker-side only."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_SCHEMA_OK = re.compile(r"^[a-z_][a-z0-9_]*$", re.I)


class PostgresWriter:
    """
    testing  → rich tables (signals, skip, execution, audit, daily, metrics + trades/history/candles)
    production → lean (trades/history/candles only; no signal_id/correlation_id)
    PK for trades/history = ticket_id.
    """

    _EXIT_FIELDS = (
        "exit_time",
        "exit_price",
        "pnl",
        "pnl_r",
        "duration_seconds",
        "mae",
        "mfe",
        "exit_reason",
    )

    def __init__(self, dsn: str | None, *, schema: str = "testing") -> None:
        sch = str(schema or "testing").strip()
        if not _SCHEMA_OK.match(sch):
            raise ValueError(f"invalid schema name: {schema!r}")
        self._schema = sch
        self._dsn = dsn
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

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def is_rich(self) -> bool:
        """testing schema keeps observability tables."""
        return self._schema == "testing"

    def _t(self, table: str) -> str:
        return f"{self._schema}.{table}"

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

    def ensure_schema(self) -> None:
        """Apply matching SQL file content via CREATE IF NOT EXISTS helpers."""
        from settings.paths import ROOT

        name = "testing_schema.sql" if self.is_rich else "production_schema.sql"
        path = ROOT / "sql" / name
        if path.is_file():
            self.apply_schema(path.read_text(encoding="utf-8"))
        else:
            logger.warning("schema_sql_missing path=%s", path)

    def ensure_ticket_id_column(self) -> None:
        self.ensure_schema()

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        if payload.get("ticket_id") is None and payload.get("broker_ticket") is not None:
            payload["ticket_id"] = payload.get("broker_ticket")
        for k in (
            "signal_id",
            "correlation_id",
            "session",
            "regime",
            "probability",
            "meta_probability",
            "confidence",
        ):
            payload.setdefault(k, None)
        payload.setdefault("take_profit", 0)
        payload.setdefault("risk_pct", 0)
        return payload

    def insert_open_trade(self, row: dict[str, Any]) -> None:
        payload = self._normalize(row)
        if payload.get("ticket_id") is None:
            logger.warning("insert_open_trade skipped — missing ticket_id")
            return
        payload["ticket_id"] = int(payload["ticket_id"])
        if self.is_rich:
            cols = (
                "ticket_id, signal_id, correlation_id, symbol, side, "
                "entry_time, entry_price, stop_loss, take_profit, lot, risk_pct, "
                "session, regime, probability, meta_probability, confidence, updated_at"
            )
            vals = (
                "%(ticket_id)s, %(signal_id)s, %(correlation_id)s, %(symbol)s, %(side)s, "
                "%(entry_time)s, %(entry_price)s, %(stop_loss)s, %(take_profit)s, %(lot)s, %(risk_pct)s, "
                "%(session)s, %(regime)s, %(probability)s, %(meta_probability)s, %(confidence)s, NOW()"
            )
        else:
            cols = (
                "ticket_id, symbol, side, "
                "entry_time, entry_price, stop_loss, take_profit, lot, risk_pct, "
                "session, regime, probability, meta_probability, confidence, updated_at"
            )
            vals = (
                "%(ticket_id)s, %(symbol)s, %(side)s, "
                "%(entry_time)s, %(entry_price)s, %(stop_loss)s, %(take_profit)s, %(lot)s, %(risk_pct)s, "
                "%(session)s, %(regime)s, %(probability)s, %(meta_probability)s, %(confidence)s, NOW()"
            )
        t = self._t("trades")
        sql = f"""
        INSERT INTO {t} ({cols}) VALUES ({vals})
        ON CONFLICT (ticket_id) DO UPDATE SET
            stop_loss = COALESCE(EXCLUDED.stop_loss, {t}.stop_loss),
            take_profit = COALESCE(EXCLUDED.take_profit, {t}.take_profit),
            updated_at = NOW()
        """
        self._execute(sql, payload)

    def close_trade_to_history(self, row: dict[str, Any]) -> None:
        payload = self._normalize(row)
        ticket = payload.get("ticket_id")
        if ticket is None:
            logger.warning("close_trade_to_history skipped — missing ticket_id")
            return
        ticket = int(ticket)
        payload["ticket_id"] = ticket
        open_row = self.fetch_open_trade(ticket_id=ticket)
        if open_row:
            hist = dict(open_row)
            for key in self._EXIT_FIELDS:
                if payload.get(key) is not None:
                    hist[key] = payload[key]
            if self.is_rich:
                for key in (
                    "signal_id",
                    "correlation_id",
                    "session",
                    "regime",
                    "probability",
                    "meta_probability",
                    "confidence",
                ):
                    if hist.get(key) is None and payload.get(key) is not None:
                        hist[key] = payload[key]
            self._upsert_history_fill_empty(hist)
            self._delete_open_trade(ticket)
            return
        self._upsert_history_fill_empty(payload)

    def _upsert_history_fill_empty(self, row: dict[str, Any]) -> None:
        payload = self._normalize(row)
        if payload.get("ticket_id") is None:
            return
        payload["ticket_id"] = int(payload["ticket_id"])
        for key in self._EXIT_FIELDS:
            payload.setdefault(key, None)
        required = ("symbol", "side", "entry_time", "entry_price", "stop_loss", "lot")
        if any(payload.get(k) is None for k in required):
            self._fill_history_exit_by_ticket(payload)
            return
        ht = self._t("history_trades")
        if self.is_rich:
            cols = (
                "ticket_id, signal_id, correlation_id, symbol, side, "
                "entry_time, exit_time, entry_price, exit_price, stop_loss, take_profit, "
                "lot, risk_pct, pnl, pnl_r, duration_seconds, mae, mfe, exit_reason, "
                "session, regime, probability, meta_probability, confidence, updated_at"
            )
            vals = (
                "%(ticket_id)s, %(signal_id)s, %(correlation_id)s, %(symbol)s, %(side)s, "
                "%(entry_time)s, %(exit_time)s, %(entry_price)s, %(exit_price)s, %(stop_loss)s, %(take_profit)s, "
                "%(lot)s, %(risk_pct)s, %(pnl)s, %(pnl_r)s, %(duration_seconds)s, %(mae)s, %(mfe)s, "
                "%(exit_reason)s, %(session)s, %(regime)s, %(probability)s, %(meta_probability)s, "
                "%(confidence)s, NOW()"
            )
        else:
            cols = (
                "ticket_id, symbol, side, "
                "entry_time, exit_time, entry_price, exit_price, stop_loss, take_profit, "
                "lot, risk_pct, pnl, pnl_r, duration_seconds, mae, mfe, exit_reason, "
                "session, regime, probability, meta_probability, confidence, updated_at"
            )
            vals = (
                "%(ticket_id)s, %(symbol)s, %(side)s, "
                "%(entry_time)s, %(exit_time)s, %(entry_price)s, %(exit_price)s, %(stop_loss)s, %(take_profit)s, "
                "%(lot)s, %(risk_pct)s, %(pnl)s, %(pnl_r)s, %(duration_seconds)s, %(mae)s, %(mfe)s, "
                "%(exit_reason)s, %(session)s, %(regime)s, %(probability)s, %(meta_probability)s, "
                "%(confidence)s, NOW()"
            )
        sql = f"""
        INSERT INTO {ht} ({cols}) VALUES ({vals})
        ON CONFLICT (ticket_id) DO UPDATE SET
            exit_time = COALESCE({ht}.exit_time, EXCLUDED.exit_time),
            exit_price = COALESCE({ht}.exit_price, EXCLUDED.exit_price),
            pnl = COALESCE({ht}.pnl, EXCLUDED.pnl),
            pnl_r = COALESCE({ht}.pnl_r, EXCLUDED.pnl_r),
            duration_seconds = COALESCE({ht}.duration_seconds, EXCLUDED.duration_seconds),
            mae = COALESCE({ht}.mae, EXCLUDED.mae),
            mfe = COALESCE({ht}.mfe, EXCLUDED.mfe),
            exit_reason = COALESCE({ht}.exit_reason, EXCLUDED.exit_reason),
            updated_at = NOW()
        """
        self._execute(sql, payload)

    def _fill_history_exit_by_ticket(self, row: dict[str, Any]) -> None:
        sql = f"""
        UPDATE {self._t("history_trades")} SET
            exit_time = COALESCE(exit_time, %(exit_time)s),
            exit_price = COALESCE(exit_price, %(exit_price)s),
            pnl = COALESCE(pnl, %(pnl)s),
            pnl_r = COALESCE(pnl_r, %(pnl_r)s),
            duration_seconds = COALESCE(duration_seconds, %(duration_seconds)s),
            mae = COALESCE(mae, %(mae)s),
            mfe = COALESCE(mfe, %(mfe)s),
            exit_reason = COALESCE(exit_reason, %(exit_reason)s),
            updated_at = NOW()
        WHERE ticket_id = %(ticket_id)s
        """
        self._execute(sql, row)

    def _delete_open_trade(self, ticket_id: int) -> None:
        self._execute(
            f"DELETE FROM {self._t('trades')} WHERE ticket_id = %(ticket_id)s",
            {"ticket_id": int(ticket_id)},
        )

    def fetch_open_trade(self, *, ticket_id: int | None = None) -> dict[str, Any] | None:
        if not self._enabled or self._psycopg2 is None or ticket_id is None:
            return None
        sql = f"SELECT * FROM {self._t('trades')} WHERE ticket_id = %(ticket_id)s LIMIT 1"
        try:
            with self.connection() as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(sql, {"ticket_id": int(ticket_id)})
                    row = cur.fetchone()
                    if row is None or cur.description is None:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
        except Exception:
            logger.exception("fetch_open_trade_failed")
            return None

    def fetch_open_trades_by_ticket(
        self,
        *,
        symbol: str | None = None,
        tickets: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        if not self._enabled or self._psycopg2 is None:
            return {}
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if symbol:
            clauses.append("symbol = %(symbol)s")
            params["symbol"] = symbol
        if tickets:
            clauses.append("ticket_id = ANY(%(tickets)s)")
            params["tickets"] = [int(t) for t in tickets]
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM {self._t('trades')} {where}"
        out: dict[int, dict[str, Any]] = {}
        try:
            with self.connection() as conn:
                if conn is None:
                    return {}
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    cols = [d[0] for d in cur.description]
                    for row in cur.fetchall():
                        d = dict(zip(cols, row))
                        out[int(d["ticket_id"])] = d
        except Exception:
            logger.exception("fetch_open_trades_by_ticket_failed")
        return out

    def summarize_history(self) -> dict[str, Any]:
        empty = {"total_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        if not self._enabled or self._psycopg2 is None:
            return empty
        sql = f"""
        SELECT
            COALESCE(SUM(pnl), 0) AS total_pnl,
            COUNT(*)::int AS trades,
            COUNT(*) FILTER (WHERE pnl > 0)::int AS wins,
            COUNT(*) FILTER (WHERE pnl IS NOT NULL AND pnl <= 0)::int AS losses
        FROM {self._t("history_trades")}
        """
        try:
            with self.connection() as conn:
                if conn is None:
                    return empty
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                    if not row:
                        return empty
                    return {
                        "total_pnl": float(row[0] or 0),
                        "trades": int(row[1] or 0),
                        "wins": int(row[2] or 0),
                        "losses": int(row[3] or 0),
                    }
        except Exception:
            logger.exception("summarize_history_failed")
            return empty

    def upsert_candle(self, row: dict[str, Any]) -> None:
        sql = f"""
        INSERT INTO {self._t("candles")} (
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
        payload["features"] = json.dumps(feats, default=str) if feats is not None else None
        self._execute(sql, payload)

    # --- testing-only observability ---

    def upsert_signal(self, row: dict[str, Any]) -> None:
        if not self.is_rich:
            return
        sql = f"""
        INSERT INTO {self._t("signals")} (
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

    def insert_skip(self, row: dict[str, Any]) -> None:
        if not self.is_rich:
            return
        sql = f"""
        INSERT INTO {self._t("skip_logs")} (
            correlation_id, signal_id, timestamp, symbol, reason, threshold, current_value, detail
        ) VALUES (
            %(correlation_id)s, %(signal_id)s, %(timestamp)s, %(symbol)s, %(reason)s,
            %(threshold)s, %(current_value)s, %(detail)s::jsonb
        )
        """
        payload = dict(row)
        payload["detail"] = json.dumps(payload.get("detail") or {}, default=str)
        self._execute(sql, payload)

    def insert_execution(self, row: dict[str, Any]) -> None:
        if not self.is_rich:
            return
        payload = dict(row)
        if payload.get("ticket_id") is None and payload.get("trade_id") is not None:
            try:
                payload["ticket_id"] = int(payload["trade_id"])
            except (TypeError, ValueError):
                payload["ticket_id"] = None
        payload.setdefault("ticket_id", None)
        sql = f"""
        INSERT INTO {self._t("execution_logs")} (
            correlation_id, ticket_id, timestamp, latency_ms, broker_response,
            spread, slippage, retry_count, success, error_message
        ) VALUES (
            %(correlation_id)s, %(ticket_id)s, %(timestamp)s, %(latency_ms)s, %(broker_response)s,
            %(spread)s, %(slippage)s, %(retry_count)s, %(success)s, %(error_message)s
        )
        """
        self._execute(sql, payload)

    def insert_audit(self, component: str, action: str, detail: dict, *, correlation_id: str | None = None) -> None:
        if not self.is_rich:
            return
        sql = f"""
        INSERT INTO {self._t("audit_logs")} (correlation_id, component, action, detail)
        VALUES (%(correlation_id)s, %(component)s, %(action)s, %(detail)s::jsonb)
        """
        self._execute(
            sql,
            {
                "correlation_id": correlation_id,
                "component": component,
                "action": action,
                "detail": json.dumps(detail, default=str),
            },
        )

    def upsert_daily(self, row: dict[str, Any]) -> None:
        if not self.is_rich:
            return
        sql = f"""
        INSERT INTO {self._t("daily_statistics")} (
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

    def insert_metric(
        self,
        name: str,
        value: float,
        *,
        labels: dict | None = None,
        correlation_id: str | None = None,
    ) -> None:
        if not self.is_rich:
            return
        sql = f"""
        INSERT INTO {self._t("metrics")} (name, value, labels, correlation_id)
        VALUES (%(name)s, %(value)s, %(labels)s::jsonb, %(correlation_id)s)
        """
        self._execute(
            sql,
            {
                "name": name,
                "value": value,
                "labels": json.dumps(labels or {}, default=str),
                "correlation_id": correlation_id,
            },
        )

    def upsert_trade(self, row: dict[str, Any]) -> None:
        self.insert_open_trade(row)

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self.connection() as conn:
            if conn is None:
                logger.debug("db_noop sql=%s keys=%s", sql.split()[0], list(params.keys()))
                return
            with conn.cursor() as cur:
                cur.execute(sql, params)
