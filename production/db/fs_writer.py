"""Thin Postgres writer for research.fs_* feature-selection experiments."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class FeatureSelectionWriter:
    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._enabled = bool(dsn)
        self._psycopg2 = None
        if self._enabled:
            try:
                import psycopg2

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 missing — fs writer disabled")
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

    def apply_schema(self, sql: str) -> None:
        with self.connection() as conn:
            if conn is None:
                raise RuntimeError("DB disabled")
            with conn.cursor() as cur:
                cur.execute(sql)

    def wipe_all(self) -> int:
        with self.connection() as conn:
            if conn is None:
                return 0
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM research.fs_runs")
                n = int(cur.fetchone()[0])
                cur.execute("DELETE FROM research.fs_runs")
            return n

    def upsert_run(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.fs_runs (run_id, status, baseline_pf, baseline_return, baseline_dd, notes)
        VALUES (%(run_id)s, %(status)s, %(baseline_pf)s, %(baseline_return)s, %(baseline_dd)s, %(notes)s)
        ON CONFLICT (run_id) DO UPDATE SET
            status = EXCLUDED.status,
            baseline_pf = COALESCE(EXCLUDED.baseline_pf, research.fs_runs.baseline_pf),
            baseline_return = COALESCE(EXCLUDED.baseline_return, research.fs_runs.baseline_return),
            baseline_dd = COALESCE(EXCLUDED.baseline_dd, research.fs_runs.baseline_dd),
            notes = COALESCE(EXCLUDED.notes, research.fs_runs.notes)
        """
        self._execute(sql, row)

    def upsert_experiment(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.fs_experiments (
            run_id, phase, experiment_id, features_json, n_features, removed_feature, removed_group,
            n_trades, win_rate, profit_factor, total_return, max_drawdown, final_equity,
            roc_auc, avg_pr, metrics
        ) VALUES (
            %(run_id)s, %(phase)s, %(experiment_id)s, %(features_json)s::jsonb, %(n_features)s,
            %(removed_feature)s, %(removed_group)s, %(n_trades)s, %(win_rate)s, %(profit_factor)s,
            %(total_return)s, %(max_drawdown)s, %(final_equity)s, %(roc_auc)s, %(avg_pr)s, %(metrics)s::jsonb
        )
        ON CONFLICT (run_id, phase, experiment_id) DO UPDATE SET
            features_json = EXCLUDED.features_json,
            n_features = EXCLUDED.n_features,
            n_trades = EXCLUDED.n_trades,
            win_rate = EXCLUDED.win_rate,
            profit_factor = EXCLUDED.profit_factor,
            total_return = EXCLUDED.total_return,
            max_drawdown = EXCLUDED.max_drawdown,
            final_equity = EXCLUDED.final_equity,
            roc_auc = EXCLUDED.roc_auc,
            avg_pr = EXCLUDED.avg_pr,
            metrics = EXCLUDED.metrics
        """
        payload = dict(row)
        payload["features_json"] = json.dumps(payload.get("features_json") or [], default=str)
        metrics = {k: (None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v)
                   for k, v in (payload.get("metrics") or {}).items()}
        payload["metrics"] = json.dumps(metrics, default=str)
        # NaN scalars → NULL for float columns too
        for k in ("roc_auc", "avg_pr", "win_rate", "profit_factor", "total_return", "max_drawdown"):
            v = payload.get(k)
            if isinstance(v, float) and v != v:
                payload[k] = None
        self._execute(sql, payload)

    def fetch_experiments(self, run_id: str) -> list[dict[str, Any]]:
        cols = (
            "phase", "experiment_id", "features_json", "n_features", "removed_feature", "removed_group",
            "n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "final_equity",
            "roc_auc", "avg_pr", "metrics",
        )
        with self.connection() as conn:
            if conn is None:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(cols)} FROM research.fs_experiments WHERE run_id = %s ORDER BY phase, experiment_id",
                    (run_id,),
                )
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(sql, params)
