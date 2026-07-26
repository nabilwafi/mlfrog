"""Append-only Postgres writer for research.exit_grid_* (exit engine grid search)."""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_RESULT_COLS = (
    "run_id", "combo_id", "params", "n_trades", "win_rate", "profit_factor",
    "total_return", "max_drawdown", "final_equity", "avg_win", "avg_loss",
    "avg_r", "expectancy_r", "avg_mfe_r", "avg_mae_r", "exit_reasons",
    "yearly", "years_positive", "ret_2026", "robustness",
)


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class ExitGridWriter:
    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._enabled = bool(dsn)
        self._psycopg2 = None
        if self._enabled:
            try:
                import psycopg2
                import psycopg2.extras  # noqa: F401

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 missing — exit grid writer disabled")
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

    def upsert_run(self, row: dict[str, Any]) -> None:
        sql = """
        INSERT INTO research.exit_grid_runs (run_id, status, n_combos, features, notes)
        VALUES (%(run_id)s, %(status)s, %(n_combos)s, %(features)s::jsonb, %(notes)s)
        ON CONFLICT (run_id) DO UPDATE SET
            status = EXCLUDED.status,
            n_combos = COALESCE(EXCLUDED.n_combos, research.exit_grid_runs.n_combos),
            notes = COALESCE(EXCLUDED.notes, research.exit_grid_runs.notes)
        """
        payload = dict(row)
        payload["features"] = json.dumps(payload.get("features") or [])
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(sql, payload)

    def insert_results(self, rows: list[dict[str, Any]]) -> None:
        """Batch append (ON CONFLICT DO NOTHING: never overwrite)."""
        if not rows:
            return
        from psycopg2.extras import execute_values

        values = []
        for r in rows:
            rec = []
            for c in _RESULT_COLS:
                v = r.get(c)
                if c in ("params", "exit_reasons", "yearly"):
                    v = json.dumps({k: _clean(x) for k, x in (v or {}).items()}, default=str)
                else:
                    v = _clean(v)
                rec.append(v)
            values.append(tuple(rec))
        sql = (
            f"INSERT INTO research.exit_grid_results ({', '.join(_RESULT_COLS)}) VALUES %s "
            "ON CONFLICT (run_id, combo_id) DO NOTHING"
        )
        with self.connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                execute_values(cur, sql, values, page_size=500)

    def done_combo_ids(self, run_id: str) -> set[str]:
        with self.connection() as conn:
            if conn is None:
                return set()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT combo_id FROM research.exit_grid_results WHERE run_id = %s", (run_id,)
                )
                return {r[0] for r in cur.fetchall()}

    def fetch_results(self, run_id: str) -> list[dict[str, Any]]:
        """Read grid results so reports do not need a large duplicate CSV."""
        with self.connection() as conn:
            if conn is None:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(_RESULT_COLS)} "
                    "FROM research.exit_grid_results WHERE run_id = %s",
                    (run_id,),
                )
                return [dict(zip(_RESULT_COLS, row)) for row in cur.fetchall()]
