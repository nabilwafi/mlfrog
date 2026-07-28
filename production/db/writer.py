"""Postgres writer — lean runtime schemas (testing | production). Worker-side only."""

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
    Thin SQL writer for lean trading schemas.
    Tables: {schema}.trades (open), {schema}.history_trades, {schema}.candles.
    PK = ticket_id.
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
        """Idempotent CREATE for lean tables (same DDL as sql/*_schema.sql)."""
        s = self._schema
        self._execute(f"CREATE SCHEMA IF NOT EXISTS {s}", {})
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {s}.trades (
                ticket_id           BIGINT PRIMARY KEY,
                signal_id           TEXT,
                correlation_id      TEXT,
                symbol              TEXT NOT NULL,
                side                TEXT NOT NULL,
                entry_time          TIMESTAMPTZ NOT NULL,
                entry_price         DOUBLE PRECISION NOT NULL,
                stop_loss           DOUBLE PRECISION NOT NULL,
                take_profit         DOUBLE PRECISION NOT NULL DEFAULT 0,
                lot                 DOUBLE PRECISION NOT NULL,
                risk_pct            DOUBLE PRECISION NOT NULL DEFAULT 0,
                session             TEXT,
                regime              TEXT,
                probability         DOUBLE PRECISION,
                meta_probability    DOUBLE PRECISION,
                confidence          DOUBLE PRECISION,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            {},
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {s}.history_trades (
                ticket_id           BIGINT PRIMARY KEY,
                signal_id           TEXT,
                correlation_id      TEXT,
                symbol              TEXT NOT NULL,
                side                TEXT NOT NULL,
                entry_time          TIMESTAMPTZ NOT NULL,
                exit_time           TIMESTAMPTZ,
                entry_price         DOUBLE PRECISION NOT NULL,
                exit_price          DOUBLE PRECISION,
                stop_loss           DOUBLE PRECISION NOT NULL,
                take_profit         DOUBLE PRECISION NOT NULL DEFAULT 0,
                lot                 DOUBLE PRECISION NOT NULL,
                risk_pct            DOUBLE PRECISION NOT NULL DEFAULT 0,
                pnl                 DOUBLE PRECISION,
                pnl_r               DOUBLE PRECISION,
                duration_seconds    INTEGER,
                mae                 DOUBLE PRECISION,
                mfe                 DOUBLE PRECISION,
                exit_reason         TEXT,
                session             TEXT,
                regime              TEXT,
                probability         DOUBLE PRECISION,
                meta_probability    DOUBLE PRECISION,
                confidence          DOUBLE PRECISION,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            {},
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {s}.candles (
                symbol              TEXT NOT NULL,
                timeframe           TEXT NOT NULL,
                timestamp           TIMESTAMPTZ NOT NULL,
                open                DOUBLE PRECISION NOT NULL,
                high                DOUBLE PRECISION NOT NULL,
                low                 DOUBLE PRECISION NOT NULL,
                close               DOUBLE PRECISION NOT NULL,
                tick_volume         DOUBLE PRECISION,
                spread              DOUBLE PRECISION,
                real_volume         DOUBLE PRECISION,
                source              TEXT NOT NULL DEFAULT 'mt5_live',
                features            JSONB,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
            """,
            {},
        )

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
        sql = f"""
        INSERT INTO {self._t("trades")} (
            ticket_id, signal_id, correlation_id, symbol, side,
            entry_time, entry_price, stop_loss, take_profit, lot, risk_pct,
            session, regime, probability, meta_probability, confidence, updated_at
        ) VALUES (
            %(ticket_id)s, %(signal_id)s, %(correlation_id)s, %(symbol)s, %(side)s,
            %(entry_time)s, %(entry_price)s, %(stop_loss)s, %(take_profit)s, %(lot)s, %(risk_pct)s,
            %(session)s, %(regime)s, %(probability)s, %(meta_probability)s, %(confidence)s, NOW()
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            stop_loss = COALESCE(EXCLUDED.stop_loss, {self._t("trades")}.stop_loss),
            take_profit = COALESCE(EXCLUDED.take_profit, {self._t("trades")}.take_profit),
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
        sql = f"""
        INSERT INTO {ht} (
            ticket_id, signal_id, correlation_id, symbol, side,
            entry_time, exit_time, entry_price, exit_price, stop_loss, take_profit,
            lot, risk_pct, pnl, pnl_r, duration_seconds, mae, mfe, exit_reason,
            session, regime, probability, meta_probability, confidence, updated_at
        ) VALUES (
            %(ticket_id)s, %(signal_id)s, %(correlation_id)s, %(symbol)s, %(side)s,
            %(entry_time)s, %(exit_time)s, %(entry_price)s, %(exit_price)s, %(stop_loss)s, %(take_profit)s,
            %(lot)s, %(risk_pct)s, %(pnl)s, %(pnl_r)s, %(duration_seconds)s, %(mae)s, %(mfe)s,
            %(exit_reason)s, %(session)s, %(regime)s, %(probability)s, %(meta_probability)s,
            %(confidence)s, NOW()
        )
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
        """Aggregate closed-trade PnL from history_trades."""
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

    # --- legacy no-ops (call sites may still exist briefly) ---
    def ensure_ticket_id_column(self) -> None:
        self.ensure_schema()

    def upsert_trade(self, row: dict[str, Any]) -> None:
        self.insert_open_trade(row)

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self.connection() as conn:
            if conn is None:
                logger.debug("db_noop sql=%s keys=%s", sql.split()[0], list(params.keys()))
                return
            with conn.cursor() as cur:
                cur.execute(sql, params)
