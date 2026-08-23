"""Sprint 39 iter 1–2 — Exit state dataset + future outcomes.

Frozen production stack. Research only. XAUUSD.

  python apps/research_sprint39_exit_state.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

def session_of_hour(h: int) -> str:
    if 0 <= h <= 6:
        return "ASIA"
    if 7 <= h <= 12:
        return "LONDON"
    if 13 <= h <= 16:
        return "LONDON_NY"
    if 17 <= h <= 21:
        return "NY"
    return "OFF"


from apps.run_exit_engine_grid import FEAT7, Paths, build_entry_panel, simulate_combo
from simulation.wf.sim import _load_side, entry_indices, load_h1, prepare_market

PANEL = (
    _ROOT
    / "artifacts"
    / "pipeline_backtest"
    / "rolling_wf"
    / "sprint38_prod_yearly"
    / "entry_panel_top21.parquet"
)
OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint39_exit_state"
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
HORIZONS = (8, 16, 24, 48)
MILESTONES = (0.25, 0.50, 1.0)
STATE_FEATS = [
    "current_R",
    "mfe_so_far_R",
    "mae_so_far_R",
    "drawdown_from_MFE_R",
    "bars_in_trade",
    "atr_pct_entry",
    "atr_pct_now",
    "atr_ratio",
    "ema_trend_duration",
    "rolling_quantile",
    "ctx_h4_swing_quality",
    "hour_sin",
    "hour_cos",
    "hour_sin_now",
    "hour_cos_now",
    "ema_dist_atr",
    "mom_1R",
    "y_prob",
    "is_long",
]


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
        return _join_feat7(pd.read_parquet(PANEL))
    print("build FEAT7 WF top 21% panel")
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(build_entry_panel(top_pct=0.21))
    panel.to_parquet(PANEL, index=False)
    return panel


def _atr_pctile(atr: np.ndarray, win: int = 252) -> np.ndarray:
    s = pd.Series(atr)
    return s.rolling(win, min_periods=50).apply(lambda x: float(np.mean(x <= x[-1])), raw=True).to_numpy()


def _outcomes(p: Paths, i: int, j: int, cr: float, mfe: float, mae: float, last: int) -> dict:
    rec = {}
    for H in HORIZONS:
        sl = p.fav[i, j + 1 : min(j + 1 + H, last + 1)]
        sl = sl[np.isfinite(sl)]
        rec[f"n_future_{H}"] = int(sl.size)
        rec[f"future_MFE_{H}"] = float(sl.max() - cr) if sl.size else np.nan
        adv = p.adv[i, j + 1 : min(j + 1 + H, last + 1)]
        adv = adv[np.isfinite(adv)]
        rec[f"future_MAE_{H}"] = float(np.nanmax(adv) - mae) if adv.size else np.nan
        cl = p.close_r[i, j : min(j + 1 + H, last + 1)]
        cl = cl[np.isfinite(cl)]
        rec[f"future_giveback_{H}"] = float(mfe - np.nanmin(cl)) if cl.size else np.nan
    post = p.fav[i, j + 1 : last + 1]
    post = post[np.isfinite(post)]
    rec["future_max_R"] = float(post.max()) if post.size else np.nan
    rec["future_giveback"] = rec["future_giveback_48"]
    return rec


def build_dataset(p: Paths, sim: dict, mkt: dict) -> pd.DataFrame:
    hold = np.asarray(sim["holding_bars"], dtype=int)
    reason = np.asarray(sim["reason"], dtype=int)
    ei = entry_indices(p.panel, mkt["ts"])
    atr_s = mkt["atr"]
    close_s = mkt["close"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {name: k for k, name in enumerate(FEAT7)}
    trend = p.panel["trend_state"].astype(str).to_numpy() if "trend_state" in p.panel.columns else np.full(p.n, "UNKNOWN")
    vol = p.panel["vol_state"].astype(str).to_numpy() if "vol_state" in p.panel.columns else np.full(p.n, "UNKNOWN")
    years = p.year
    rows: list[dict] = []

    def rec_at(i: int, j: int, kind: str) -> dict | None:
        cr = p.close_r[i, j]
        if not np.isfinite(cr):
            return None
        mfe = float(np.nanmax(p.fav[i, 1 : j + 1]))
        mae = float(np.nanmax(p.adv[i, 1 : j + 1]))
        idx = int(ei[i] + j)
        if idx >= len(close_s):
            return None
        atr_e = float(p.atr[i])
        atr_j = float(atr_s[idx]) if np.isfinite(atr_s[idx]) else atr_e
        sign = 1.0 if p.is_long[i] else -1.0
        ema_dist = sign * (close_s[idx] - ema[idx]) / max(atr_e, 1e-12)
        prev = p.close_r[i, j - 1] if j > 1 else 0.0
        hr = int((hour0[i] + j) % 24)
        last = int(p.last_off[i])
        rec = {
            "trade_i": i,
            "year": int(years[i]),
            "j": j,
            "kind": kind,
            "baseline_alive": int(j <= int(hold[i])),
            "trail_exit": int(reason[i] == 1),
            "hold": int(hold[i]),
            "current_R": float(cr),
            "mfe_so_far_R": mfe,
            "mae_so_far_R": mae,
            "drawdown_from_MFE_R": mfe - float(cr),
            "bars_in_trade": j,
            "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]) if "atr_percentile_252" in f7i else 0.5,
            "atr_pct_now": float(atr_pct_bar[idx]) if np.isfinite(atr_pct_bar[idx]) else 0.5,
            "atr_ratio": atr_j / max(atr_e, 1e-12),
            "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
            "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
            "ctx_h4_swing_quality": float(f7[i, f7i["ctx_h4_swing_quality"]]),
            "hour_sin": float(f7[i, f7i["hour_sin"]]),
            "hour_cos": float(f7[i, f7i["hour_cos"]]),
            "hour_sin_now": math.sin(2 * math.pi * hr / 24),
            "hour_cos_now": math.cos(2 * math.pi * hr / 24),
            "ema_dist_atr": float(ema_dist),
            "mom_1R": float(cr - prev) if np.isfinite(prev) else 0.0,
            "y_prob": float(y_prob[i]),
            "is_long": int(p.is_long[i]),
            "session": session_of_hour(hr),
            "trend": str(trend[i]),
            "vol_regime": str(vol[i]),
        }
        rec.update(_outcomes(p, i, j, float(cr), mfe, mae, last))
        return rec

    print("bar observations (while baseline trail alive)")
    for i in range(p.n):
        h = int(hold[i])
        for j in range(1, h + 1):
            r = rec_at(i, j, "bar")
            if r is not None:
                rows.append(r)
    print("MFE milestone first-hits")
    for i in range(p.n):
        last = int(p.last_off[i])
        seen = set()
        running = 0.0
        for j in range(1, last + 1):
            fv = p.fav[i, j]
            if not np.isfinite(fv):
                continue
            running = max(running, float(fv))
            for thr in MILESTONES:
                key = thr
                if key in seen:
                    continue
                if running >= thr:
                    r = rec_at(i, j, f"mfe_{thr:.2f}")
                    if r is not None:
                        rows.append(r)
                    seen.add(key)
            if len(seen) == len(MILESTONES):
                break
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(_load_panel(), mkt)
    sim = simulate_combo(p, **EXIT_KW)
    print(f"trades={p.n}")
    st = build_dataset(p, sim, mkt)
    st.to_parquet(OUT / "exit_state_dataset.parquet", index=False)
    print(f"n_obs={len(st)} kinds={st['kind'].value_counts().to_dict()}")

    leak = [
        ("current_R", "close in R at bar j", "bar j", "0", False),
        ("mfe_so_far_R", "max fav 1..j", "bar j", "j", False),
        ("mae_so_far_R", "max adv 1..j", "bar j", "j", False),
        ("drawdown_from_MFE_R", "mfe_so_far - current_R", "bar j", "j", False),
        ("bars_in_trade", "j", "bar j", "0", False),
        ("atr_pct_entry", "FEAT7 atr_percentile_252 at entry", "entry", "252", False),
        ("atr_pct_now", "H1 ATR rank 252 at ei+j", "bar j", "252", False),
        ("atr_ratio", "atr[ei+j] / atr_entry", "bar j", "0", False),
        ("ema_trend_duration", "FEAT7 at entry", "entry", "lookback", False),
        ("rolling_quantile", "FEAT7 at entry", "entry", "lookback", False),
        ("ctx_h4_swing_quality", "FEAT7 at entry", "entry", "lookback", False),
        ("hour_sin", "FEAT7 at entry", "entry", "0", False),
        ("hour_cos", "FEAT7 at entry", "entry", "0", False),
        ("hour_sin_now", "clock at bar j", "bar j", "0", False),
        ("hour_cos_now", "clock at bar j", "bar j", "0", False),
        ("ema_dist_atr", "signed (close-ema20)/atr_entry at ei+j", "bar j", "20", False),
        ("mom_1R", "close_R[j]-close_R[j-1]", "bar j", "1", False),
        ("y_prob", "frozen entry LGBM", "entry", "train<val<test", False),
        ("is_long", "entry side", "entry", "0", False),
        ("session", "hour bucket at bar j", "bar j", "0", False),
        ("trend", "entry H4 trend_state", "entry", "H4", False),
        ("vol_regime", "entry H4 vol_state", "entry", "H4", False),
        ("future_MFE_*", "max fav after j minus current_R", "j+1..j+H", "H", True),
        ("future_MAE_*", "max adv after j minus mae_so_far", "j+1..j+H", "H", True),
        ("future_max_R", "max fav after j (from entry)", "j+1..end", "CAP", True),
        ("future_giveback_*", "mfe_so_far - min close_R over next H", "j..j+H", "H", True),
    ]
    pd.DataFrame(leak, columns=["feature_name", "source", "timestamp", "lookback", "future_dependency"]).to_csv(
        OUT / "feature_leakage_audit.csv", index=False
    )
    assert not any(x[4] for x in leak if x[0] in STATE_FEATS)

    alive = st[st["kind"] == "bar"]
    lines = [
        "# Sprint 39 Iter 1–2 — Exit state dataset",
        "",
        "Frozen: FEAT7 top **21%**, trail a0.25/d0.08, lot 0.01, heat 3R, XAUUSD.",
        "Features are point-in-time. `future_*` columns are labels only.",
        "",
        f"Trades: **{p.n}**. Observations: **{len(st)}**.",
        f"- bar (alive under baseline trail): **{(st['kind']=='bar').sum()}**",
        f"- mfe_0.25 first-hit: **{(st['kind']=='mfe_0.25').sum()}** (alive {(st.loc[st['kind']=='mfe_0.25','baseline_alive']==1).sum()})",
        f"- mfe_0.50 first-hit: **{(st['kind']=='mfe_0.50').sum()}** (alive {(st.loc[st['kind']=='mfe_0.50','baseline_alive']==1).sum()})",
        f"- mfe_1.00 first-hit: **{(st['kind']=='mfe_1.00').sum()}** (alive {(st.loc[st['kind']=='mfe_1.00','baseline_alive']==1).sum()})",
        "",
        "Leakage audit: `feature_leakage_audit.csv`. All STATE_FEATS `future_dependency=false`.",
        "",
        "## Outcome rates (bar observations, all years)",
        "",
        "| Horizon | n | mean future_MFE | P(>=1R) | P(>=2R) | mean future_MAE | P(MAE>=0.5R) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for H in HORIZONS:
        col = f"future_MFE_{H}"
        mae = f"future_MAE_{H}"
        x = alive[col].to_numpy(dtype=float)
        a = alive[mae].to_numpy(dtype=float)
        ok = np.isfinite(x)
        lines.append(
            f"| {H} | {int(ok.sum())} | {np.nanmean(x):.2f} | {np.nanmean(x>=1):.3f} | "
            f"{np.nanmean(x>=2):.3f} | {np.nanmean(a):.2f} | {np.nanmean(a>=0.5):.3f} |"
        )
    lines += [
        "",
        "Iter 3: feature attribution. Production exit unchanged.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps({"iter": 2, "next": 3, "n_obs": int(len(st))}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
