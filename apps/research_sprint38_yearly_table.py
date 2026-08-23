"""Yearly book on the current production stack, research symbol XAUUSD.

Matches live: FEAT7, top 21%, trail a0.25/d0.08, lot 0.01, heat 3R, max 5,
daily 1R only if equity <= $80, isolated $280. Not XAUUSDc.

  python apps/research_sprint38_yearly_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.research_sprint38_exit_sds_rest import simulate_mapped
from apps.research_sprint38_iter3_exit import (
    EXIT_KW,
    _score_matrix,
    _yearly,
    simulate_scored,
)
from apps.run_exit_engine_grid import FEAT7, Paths, build_entry_panel, simulate_combo
from simulation.wf.sim import _load_side, load_h1, prepare_market

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint38_prod_yearly"
PANEL = OUT / "entry_panel_top21.parquet"
PROD_PORT = dict(fixed_lot_01=True, daily_on_below=80.0)


def _join_feat7(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    if all(c in panel.columns for c in FEAT7):
        return panel
    parts = []
    for side in ("long", "short"):
        d = _load_side(side)[["timestamp", *FEAT7]].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d["side"] = side
        parts.append(d)
    return panel.merge(pd.concat(parts, ignore_index=True), on=["timestamp", "side"], how="left")


def _load_panel() -> pd.DataFrame:
    if PANEL.is_file():
        print(f"load {PANEL.name}")
        return pd.read_parquet(PANEL)
    print("build FEAT7 WF top 21% entry panel (XAUUSD)")
    panel = _join_feat7(build_entry_panel(top_pct=0.21))
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)
    return panel


def _md(df: pd.DataFrame, name: str) -> str:
    sub = df[df["candidate"] == name].sort_values("year")
    lines = [
        f"### {name}",
        "",
        "| Year | Candidates | Trades | WR | Payoff | PF | DD | Return |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sub.itertuples(index=False):
        lines.append(
            f"| {int(r.year)} | {int(r.candidates)} | {int(r.trades)} | {r.wr:.3f} | "
            f"{r.payoff:.2f} | {r.pf:.2f} | {r.dd*100:.1f}% | {r.ret*100:.0f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(_load_panel(), mkt)
    atr_pct = np.full(p.n, 0.5)
    lot_mult = np.ones(p.n)
    probs = p.panel["y_prob"].astype(float).to_numpy()
    trend_ok = np.ones(p.n, dtype=bool)
    ykw = dict(
        p=p, lot_mult=lot_mult, atr_pct=atr_pct, probs=probs, trend_ok=trend_ok, **PROD_PORT,
    )

    print(f"n={p.n} production stack on XAUUSD")
    base = simulate_combo(p, **EXIT_KW)
    score = _score_matrix(p, base, until_exit_only=True)
    books = [
        ("baseline_prod", base),
        ("dyn_trail_3bucket", simulate_scored(p, score, mode="trail")),
        ("hold_if_score>=0.50", simulate_scored(p, score, mode="hold", hold_cut=0.50)),
        ("hybrid_cuts_0.40_0.70", simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True, lo=0.40, hi=0.70)),
    ]
    rows = []
    for name, sim in books:
        recs = _yearly(name, sim=sim, **ykw)
        rows.extend(recs)
        print(f"  {name}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "yearly_book.csv", index=False)
    block = (
        "# Yearly book — production stack on XAUUSD\n\n"
        "FEAT7, top **21%**, trail a0.25/d0.08, lot **0.01**, heat 3R, max 5, "
        "daily 1R only if equity ≤ $80, isolated **$280**. Symbol **XAUUSD** (not XAUUSDc).\n\n"
        + "\n".join(_md(df, n) for n, _ in books)
    )
    (OUT / "summary.md").write_text(block + "\n", encoding="utf-8")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
