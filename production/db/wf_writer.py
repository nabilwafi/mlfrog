"""Postgres writer for research rolling walk-forward results."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class WalkForwardWriter:
    """Thin writer for research.wf_* tables. No-op when DSN missing."""

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._enabled = bool(dsn)
        self._psycopg2 = None
        if self._enabled:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 not installed — WF DB writer disabled")
                self._enabled = False

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
                raise RuntimeError("DB disabled — cannot apply schema")
            with conn.cursor() as cur:
                cur.execute(schema_sql)

    def wipe_all_runs(self) -> int:
        """Delete every research.wf_runs row (CASCADE clears trainings/backtests/trades/equity)."""
        with self.connection() as conn:
            if conn is None:
                return 0
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM research.wf_runs")
                n = int(cur.fetchone()[0])
                cur.execute("DELETE FROM research.wf_runs")
            return n

    def upsert_run(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.wf_runs (
            run_id, symbol, timeframe, train_years, val_years, test_years,
            first_test_year, last_test_year, policy, feature_version,
            label_version, pipeline_version, status, notes
        ) VALUES (
            %(run_id)s, %(symbol)s, %(timeframe)s, %(train_years)s, %(val_years)s, %(test_years)s,
            %(first_test_year)s, %(last_test_year)s, %(policy)s::jsonb, %(feature_version)s,
            %(label_version)s, %(pipeline_version)s, %(status)s, %(notes)s
        )
        ON CONFLICT (run_id) DO UPDATE SET
            status = EXCLUDED.status,
            notes = EXCLUDED.notes,
            policy = EXCLUDED.policy
        """
        payload = dict(row)
        payload["policy"] = json.dumps(payload.get("policy") or {}, default=str)
        self._execute(sql, payload)

    def set_run_status(self, run_id: str, status: str, *, notes: str | None = None) -> None:
        sql = """
        UPDATE research.wf_runs
        SET status = %(status)s, notes = COALESCE(%(notes)s, notes)
        WHERE run_id = %(run_id)s
        """
        self._execute(sql, {"run_id": run_id, "status": status, "notes": notes})

    def upsert_training(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.wf_trainings (
            run_id, window_id, side, train_start_year, train_end_year, val_year, test_year,
            model_version, feature_version, label_version, n_train, n_val, n_features,
            best_iteration, val_logloss, feature_names, model_path, metrics
        ) VALUES (
            %(run_id)s, %(window_id)s, %(side)s, %(train_start_year)s, %(train_end_year)s,
            %(val_year)s, %(test_year)s, %(model_version)s, %(feature_version)s, %(label_version)s,
            %(n_train)s, %(n_val)s, %(n_features)s, %(best_iteration)s, %(val_logloss)s,
            %(feature_names)s::jsonb, %(model_path)s, %(metrics)s::jsonb
        )
        ON CONFLICT (run_id, window_id, side) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            n_train = EXCLUDED.n_train,
            n_val = EXCLUDED.n_val,
            best_iteration = EXCLUDED.best_iteration,
            val_logloss = EXCLUDED.val_logloss,
            feature_names = EXCLUDED.feature_names,
            model_path = EXCLUDED.model_path,
            metrics = EXCLUDED.metrics
        """
        payload = dict(row)
        payload["feature_names"] = json.dumps(payload.get("feature_names") or [], default=str)
        payload["metrics"] = json.dumps(payload.get("metrics") or {}, default=str)
        self._execute(sql, payload)

    def upsert_backtest(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.wf_backtests (
            run_id, window_id, test_year, scope, n_candidates, n_trades,
            total_return, final_equity, max_drawdown, profit_factor, win_rate,
            avg_holding_bars, starting_equity, metrics
        ) VALUES (
            %(run_id)s, %(window_id)s, %(test_year)s, %(scope)s, %(n_candidates)s, %(n_trades)s,
            %(total_return)s, %(final_equity)s, %(max_drawdown)s, %(profit_factor)s, %(win_rate)s,
            %(avg_holding_bars)s, %(starting_equity)s, %(metrics)s::jsonb
        )
        ON CONFLICT (run_id, window_id, scope) DO UPDATE SET
            n_candidates = EXCLUDED.n_candidates,
            n_trades = EXCLUDED.n_trades,
            total_return = EXCLUDED.total_return,
            final_equity = EXCLUDED.final_equity,
            max_drawdown = EXCLUDED.max_drawdown,
            profit_factor = EXCLUDED.profit_factor,
            win_rate = EXCLUDED.win_rate,
            avg_holding_bars = EXCLUDED.avg_holding_bars,
            metrics = EXCLUDED.metrics
        """
        payload = dict(row)
        payload["metrics"] = json.dumps(payload.get("metrics") or {}, default=str)
        self._execute(sql, payload)

    def insert_trades(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        def _clean(row: dict[str, Any]) -> dict[str, Any]:
            out = dict(row)
            for k, v in list(out.items()):
                if isinstance(v, float) and not (v == v):  # NaN
                    out[k] = None
            return out

        sql = """
        INSERT INTO research.wf_trades (
            run_id, window_id, test_year, trade_seq, timestamp, side, entry_price, lots,
            y_prob, net_return, pnl, equity, holding_bars, exit_reason,
            regime, trend_state, vol_state, mfe_pct, mae_pct, mfe_r, mae_r, r_multiple
        ) VALUES (
            %(run_id)s, %(window_id)s, %(test_year)s, %(trade_seq)s, %(timestamp)s, %(side)s,
            %(entry_price)s, %(lots)s, %(y_prob)s, %(net_return)s, %(pnl)s, %(equity)s,
            %(holding_bars)s, %(exit_reason)s,
            %(regime)s, %(trend_state)s, %(vol_state)s, %(mfe_pct)s, %(mae_pct)s,
            %(mfe_r)s, %(mae_r)s, %(r_multiple)s
        )
        """
        clean = [_clean(r) for r in rows]
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.executemany(sql, clean)

    def insert_equity(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        sql = """
        INSERT INTO research.wf_equity (
            run_id, scope, test_year, timestamp, equity, pnl
        ) VALUES (
            %(run_id)s, %(scope)s, %(test_year)s, %(timestamp)s, %(equity)s, %(pnl)s
        )
        """
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.executemany(sql, rows)

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self.connection() as conn:
            if conn is None:
                logger.debug("wf_db_noop sql=%s", sql.split()[0])
                return
            with conn.cursor() as cur:
                cur.execute(sql, params)
