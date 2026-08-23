"""Sprint 41 Iter 1 — Differential exit dataset (no ML).

Frozen production stack. Replay identical entries under P0–P3.
Labels are trade-level R; features are PIT bar state while P0 is alive.

  python apps/research_sprint41_iter1_diff.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint39_exit_state import _atr_pctile, _join_feat7, _load_panel
from apps.run_exit_engine_grid import FEAT7, Paths, simulate_combo
from simulation.wf.sim import entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter1_differential"
POLICIES = {
    "prod": dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None),
    "p1": dict(act=0.20, dist=0.06, tp=None, partials=(), tmax=None, be=None),
    "p2": dict(act=0.20, dist=0.08, tp=None, partials=(), tmax=None, be=None),
    "p3": dict(act=0.20, dist=0.10, tp=None, partials=(), tmax=None, be=None),
}
STATE_FEATS = [
    "y_prob",
    "atr_pct_entry",
    "atr_pct_now",
    "ema_dist_atr",
    "ema_trend_duration",
    "rolling_quantile",
    "hour_sin",
    "hour_cos",
    "hour_sin_now",
    "hour_cos_now",
    "mfe_so_far_R",
    "mae_so_far_R",
    "drawdown_from_MFE_R",
    "current_R",
    "mom_1R",
]
LABELS = [
    "R_prod", "R_p1", "R_p2", "R_p3",
    "diff_p1", "diff_p2", "diff_p3",
    "p1_better", "p2_better", "p3_better",
    "best_policy", "best_diff",
]
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def _replay(p: Paths) -> dict[str, dict]:
    out = {}
    for name, kw in POLICIES.items():
        sim = simulate_combo(p, **kw)
        out[name] = sim
        print(f"  {name}: meanR={float(np.mean(sim['r_multiple'])):.3f} hold={float(np.mean(sim['holding_bars'])):.2f}")
    return out


def _trade_results(p: Paths, sims: dict) -> pd.DataFrame:
    side = np.where(p.is_long, "long", "short")
    r0 = sims["prod"]["r_multiple"]
    r1 = sims["p1"]["r_multiple"]
    r2 = sims["p2"]["r_multiple"]
    r3 = sims["p3"]["r_multiple"]
    stack = np.vstack([r0, r1, r2, r3])
    names = np.array(["prod", "p1", "p2", "p3"])
    best_i = np.argmax(stack, axis=0)
    d1, d2, d3 = r1 - r0, r2 - r0, r3 - r0
    best_diff = np.maximum.reduce([np.zeros(p.n), d1, d2, d3])
    return pd.DataFrame({
        "trade_id": np.arange(p.n),
        "timestamp": p.ts,
        "year": p.year,
        "side": side,
        "R_prod": r0,
        "R_p1": r1,
        "R_p2": r2,
        "R_p3": r3,
        "diff_p1": d1,
        "diff_p2": d2,
        "diff_p3": d3,
        "p1_better": d1 > 0,
        "p2_better": d2 > 0,
        "p3_better": d3 > 0,
        "best_policy": names[best_i],
        "best_diff": best_diff,
        "hold_prod": sims["prod"]["holding_bars"],
        "hold_p1": sims["p1"]["holding_bars"],
        "hold_p2": sims["p2"]["holding_bars"],
        "hold_p3": sims["p3"]["holding_bars"],
    })


def _state_rows(p: Paths, mkt: dict, hold_prod: np.ndarray, trades: pd.DataFrame) -> pd.DataFrame:
    ei = entry_indices(p.panel, mkt["ts"])
    atr_s = mkt["atr"]
    close_s = mkt["close"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {name: k for k, name in enumerate(FEAT7)}
    side = np.where(p.is_long, "long", "short")
    ts0 = pd.DatetimeIndex(p.ts)
    rows = []
    for i in range(p.n):
        h = int(hold_prod[i])
        for j in range(1, h + 1):
            cr = p.close_r[i, j]
            if not np.isfinite(cr):
                continue
            mfe = float(np.nanmax(p.fav[i, 1 : j + 1]))
            mae = float(np.nanmax(p.adv[i, 1 : j + 1]))
            idx = int(ei[i] + j)
            if idx >= len(close_s):
                continue
            atr_e = float(p.atr[i])
            prev = p.close_r[i, j - 1] if j > 1 else 0.0
            hr = int((hour0[i] + j) % 24)
            sign = 1.0 if p.is_long[i] else -1.0
            ema_dist = sign * (close_s[idx] - ema[idx]) / max(atr_e, 1e-12)
            rec = {
                "trade_id": i,
                "timestamp": ts0[i] + pd.Timedelta(hours=j),
                "year": int(p.year[i]),
                "side": side[i],
                "bars_in_trade": j,
                "y_prob": float(y_prob[i]),
                "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
                "atr_pct_now": float(atr_pct_bar[idx]) if np.isfinite(atr_pct_bar[idx]) else 0.5,
                "ema_dist_atr": float(ema_dist),
                "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
                "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
                "hour_sin": float(f7[i, f7i["hour_sin"]]),
                "hour_cos": float(f7[i, f7i["hour_cos"]]),
                "hour_sin_now": math.sin(2 * math.pi * hr / 24),
                "hour_cos_now": math.cos(2 * math.pi * hr / 24),
                "mfe_so_far_R": mfe,
                "mae_so_far_R": mae,
                "drawdown_from_MFE_R": mfe - float(cr),
                "current_R": float(cr),
                "mom_1R": float(cr - prev) if np.isfinite(prev) else 0.0,
            }
            t = trades.iloc[i]
            for col in LABELS:
                rec[col] = t[col]
            rows.append(rec)
    return pd.DataFrame(rows)


def _policy_stats(trades: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, tdf, odf in [("pooled", trades, ds)] + [
        (str(y), trades[trades["year"] == y], ds[ds["year"] == y]) for y in YEARS
    ]:
        n_obs = len(odf)
        n_tr = len(tdf)
        for pol, rcol, dcol in (("p1", "R_p1", "diff_p1"), ("p2", "R_p2", "diff_p2"), ("p3", "R_p3", "diff_p3")):
            if n_tr == 0:
                continue
            r = tdf[rcol].to_numpy(dtype=float)
            r0 = tdf["R_prod"].to_numpy(dtype=float)
            d = tdf[dcol].to_numpy(dtype=float)
            rows.append({
                "scope": scope,
                "policy": pol,
                "observations": n_obs,
                "trades": n_tr,
                "trades_affected": int(np.sum(np.abs(d) > 1e-12)),
                "mean_R": float(np.mean(r)),
                "median_R": float(np.median(r)),
                "mean_R_prod": float(np.mean(r0)),
                "p_R_gt_prod": float(np.mean(r > r0)),
                "mean_diff": float(np.mean(d)),
                "median_diff": float(np.median(d)),
                "p_diff_gt_0": float(np.mean(d > 0)),
                "p_diff_ge_0.25": float(np.mean(d >= 0.25)),
                "p_diff_le_m0.25": float(np.mean(d <= -0.25)),
            })
    return pd.DataFrame(rows)


def _leakage() -> pd.DataFrame:
    rows = [
        ("y_prob", "frozen entry LGBM", "entry", "train<val<test", False),
        ("atr_pct_entry", "FEAT7 atr_percentile_252", "entry", "252", False),
        ("atr_pct_now", "H1 ATR rank 252 at ei+j", "bar j", "252", False),
        ("ema_dist_atr", "signed (close-ema20)/atr_entry at ei+j", "bar j", "20", False),
        ("ema_trend_duration", "FEAT7 at entry", "entry", "lookback", False),
        ("rolling_quantile", "FEAT7 at entry", "entry", "lookback", False),
        ("hour_sin", "FEAT7 at entry", "entry", "0", False),
        ("hour_cos", "FEAT7 at entry", "entry", "0", False),
        ("hour_sin_now", "clock at bar j", "bar j", "0", False),
        ("hour_cos_now", "clock at bar j", "bar j", "0", False),
        ("mfe_so_far_R", "max fav 1..j", "bar j", "j", False),
        ("mae_so_far_R", "max adv 1..j", "bar j", "j", False),
        ("drawdown_from_MFE_R", "mfe_so_far - current_R", "bar j", "j", False),
        ("current_R", "close in R at bar j", "bar j", "0", False),
        ("mom_1R", "close_R[j]-close_R[j-1]", "bar j", "1", False),
        ("R_prod/R_p*", "final trade R under each policy", "trade end", "CAP", True),
        ("diff_p*", "R_px - R_prod", "trade end", "CAP", True),
        ("p*_better / best_*", "derived from policy R", "trade end", "CAP", True),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "source", "timestamp", "lookback", "future_dependency"])


def _md(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)
    print(f"entries={p.n}")
    sims = _replay(p)
    trades = _trade_results(p, sims)
    print("bar states while production alive")
    ds = _state_rows(p, mkt, sims["prod"]["holding_bars"], trades)
    leak = _leakage()
    assert not any(x for x in leak.itertuples(index=False) if x.feature_name in STATE_FEATS and x.future_dependency)

    summ = _policy_stats(trades, ds)
    ds.to_csv(OUT / "exit_differential_dataset.csv", index=False)
    trades.to_csv(OUT / "exit_policy_trade_results.csv", index=False)
    summ.to_csv(OUT / "exit_policy_summary.csv", index=False)
    leak.to_csv(OUT / "leakage_audit.csv", index=False)
    ds.to_parquet(OUT / "exit_differential_dataset.parquet", index=False)

    cols = [
        "scope", "policy", "observations", "trades", "trades_affected",
        "mean_R", "median_R", "p_R_gt_prod", "mean_diff", "median_diff",
        "p_diff_gt_0", "p_diff_ge_0.25", "p_diff_le_m0.25",
    ]
    lines = [
        "# Sprint 41 Iter 1 — Differential exit dataset",
        "",
        "Frozen: FEAT7 top 21%, trail P0 a0.25/d0.08. Identical entries. Labels = trade R. Features = PIT bar state.",
        f"Trades **{len(trades)}**. Bar observations (P0 alive) **{len(ds)}**.",
        "",
        "STATE_FEATS have future_dependency=false. Policy R / diffs are labels only.",
        "",
        _md(summ[cols], cols),
        "",
        "Iter 2: Spearman/quantile of PIT vs differential, yearly. Production unchanged.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"iter": 1, "next_iter": 2, "n_trades": int(len(trades)), "n_obs": int(len(ds))}, indent=2),
        encoding="utf-8",
    )
    print(f"obs={len(ds)} trades={len(trades)}")
    print(summ[summ["scope"] == "pooled"][["policy", "mean_diff", "p_diff_gt_0", "p_diff_ge_0.25"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
