"""A1 (frozen Sprint 46 trigger) with M5 execution.

H1 A1 died because 73% of triggers were same-bar P0 SL.
M5: act at first M5 close_R <= -0.50 with p>=0.80 IF SL not already tagged on that M5 bar.
Fill = next M5 open. No max(current_R, -0.50) cap.

Research only. Production unchanged.

  python apps/research_a1_m5.py
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
from apps.research_sprint40_iter1_time_aware_exit import STARTING
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import FOLDS, MODEL_FEATS, P0, _build_obs, _fit, _mc, _r_stats
from apps.research_sprint47_loss_action import CK, THR, _net
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/a1_m5"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
HORIZON_H = 48
ACT, DIST = 0.25, 0.08
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def _models(obs: pd.DataFrame) -> dict[int, object]:
    """Frozen expanding LGBM on H1 checkpoint obs. No retune."""
    import lightgbm as lgb

    early = obs[obs["checkpoint"] <= 0.75]
    X = early[MODEL_FEATS].astype(float).fillna(0.0)
    y = early["large_loss_1"].astype(int)
    years = early["year"].to_numpy()
    out: dict[int, object] = {}

    def fit(tr_mask, te_year):
        if tr_mask.sum() < 200 or y[tr_mask].min() == y[tr_mask].max():
            return
        clf = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.05, num_leaves=15,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
            verbosity=-1, class_weight="balanced",
        )
        clf.fit(X[tr_mask], y[tr_mask])
        out[te_year] = clf

    for te, tr_years, _va in FOLDS:
        fit(np.isin(years, tr_years), te)
    fit(years == 2021, 2022)
    return out


def _econ(r, r0, trades, trig) -> dict:
    ll = trades["large_loss_1"].to_numpy(dtype=bool)
    rec = trades["p0_R"].to_numpy() > 0
    tid = trades["trade_id"].to_numpy(dtype=int)
    hit = np.array([int(t) in trig for t in tid])
    g075 = (trades["mae_R"].to_numpy() >= 0.75) & rec
    ts = float(((r - r0)[ll & hit]).sum()) if (ll & hit).any() else 0.0
    rl = float(((r0 - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
    st = _r_stats(r)
    return {
        **st,
        "tail_saved_R": ts,
        "recovery_lost_R": rl,
        "net_benefit_R": ts - rl,
        "n_triggered": int(hit.sum()),
        "recovery_preservation": float(1.0 - (g075 & (r <= 0)).sum() / g075.sum()) if g075.sum() else 1.0,
        "tail_loss_rate": float((r <= -1.00).mean()),
        "p_le_m1_00": float((r <= -1.00).mean()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    trades, obs = _build_obs(p, mkt, sim0)
    models = _models(obs)
    print("fitted year models", sorted(models))

    close_s, atr_s = mkt["close"], mkt["atr"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    ei = entry_indices(p.panel, mkt["ts"]).astype(int)
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {n: k for k, n in enumerate(FEAT7)}
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()

    m5 = pd.read_parquet(M5_PATH)
    ts5 = pd.DatetimeIndex(pd.to_datetime(m5["timestamp"], utc=True)).asi8
    high, low, close = m5["high"].to_numpy(float), m5["low"].to_numpy(float), m5["close"].to_numpy(float)
    open5 = m5["open"].to_numpy(float)
    n5 = len(ts5)
    entry_ns = pd.DatetimeIndex(p.ts).asi8
    horizon_ns = HORIZON_H * 3_600_000_000_000
    h1_ts = pd.DatetimeIndex(mkt["ts"]).asi8

    r_a1 = np.array(sim0["r_multiple"], dtype=float)
    hold_a1 = np.array(sim0["holding_bars"], dtype=np.int64)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    trig: dict[int, dict] = {}
    n_block_sl = n_no_model = n_low_p = n_hit_ck = 0

    for i in range(p.n):
        y = int(p.year[i])
        if y not in TRUE_OOS:
            continue
        clf = models.get(y)
        if clf is None:
            n_no_model += 1
            continue
        e, atr = float(p.entry[i]), float(p.atr[i])
        one = 1.5 * atr
        if one <= 0 or e <= 0:
            continue
        lng = bool(p.is_long[i])
        start = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        end_ns = entry_ns[i] + horizon_ns
        sl = -1.0
        extreme = 0.0
        mae = 0.0
        prev_cr = 0.0
        seen_ck = False
        k = start
        while k < n5 and ts5[k] <= end_ns:
            h, l, c = high[k], low[k], close[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            mae = max(mae, adv)
            if extreme >= ACT:
                sl = max(sl, extreme - DIST * (atr / one))
            if adv >= -sl:
                n_block_sl += 1
                break
            if (not seen_ck) and cr <= -CK:
                seen_ck = True
                n_hit_ck += 1
                nxt = k + 1
                if nxt >= n5:
                    break
                hours = max(1, int((ts5[k] - entry_ns[i] + 3_599_999_999_999) // 3_600_000_000_000))
                h1_idx = int(np.searchsorted(h1_ts, ts5[k], side="right") - 1)
                h1_idx = max(int(ei[i]), min(h1_idx, len(close_s) - 1))
                hr = int((hour0[i] + hours) % 24)
                sign = 1.0 if lng else -1.0
                ema_dist = sign * (close_s[h1_idx] - ema[h1_idx]) / max(atr, 1e-12)
                feat = {
                    "y_prob": float(y_prob[i]),
                    "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
                    "atr_pct_now": float(atr_pct_bar[h1_idx]) if np.isfinite(atr_pct_bar[h1_idx]) else 0.5,
                    "ema_dist_atr": float(ema_dist),
                    "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
                    "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
                    "hour_sin": float(f7[i, f7i["hour_sin"]]),
                    "hour_cos": float(f7[i, f7i["hour_cos"]]),
                    "hour_sin_now": math.sin(2 * math.pi * hr / 24),
                    "hour_cos_now": math.cos(2 * math.pi * hr / 24),
                    "mfe_so_far_R": extreme,
                    "mae_so_far_R": mae,
                    "drawdown_from_MFE_R": extreme - cr,
                    "current_R": float(cr),
                    "mom_1R": float(cr - prev_cr),
                    "bars_in_trade": hours,
                }
                x = pd.DataFrame([feat])[MODEL_FEATS].astype(float).fillna(0.0)
                pr = float(clf.predict_proba(x)[0, 1])
                if pr < THR:
                    n_low_p += 1
                    prev_cr = cr
                    k += 1
                    continue
                fill_open = float(open5[nxt])
                fill_r = ((fill_open - e) if lng else (e - fill_open)) / one
                r_a1[i] = _net(fill_r, ru[i])
                hold_a1[i] = min(max(hours, 1), CAP)
                trig[i] = {
                    "j": hours, "fill": fill_r, "current_R": cr, "p": pr, "year": y,
                    "same_m5_sl": False,
                }
                break
            prev_cr = cr
            k += 1

    oos = trades[trades["year"].isin(TRUE_OOS)].copy()
    r0 = oos["p0_R"].to_numpy()
    r1 = np.array([r_a1[int(t)] for t in oos["trade_id"]])
    oos["A1_R"] = r1
    e0 = _econ(r0, r0, oos, {})
    e1 = _econ(r1, r0, oos, trig)
    e0["net_benefit_R"] = 0.0
    e0["tail_saved_R"] = 0.0
    e0["recovery_lost_R"] = 0.0
    pd.DataFrame([{"action": "A0", **e0}, {"action": "A1_M5", **e1}]).to_csv(OUT / "actions.csv", index=False)

    sim_a1 = dict(sim0)
    sim_a1["r_multiple"] = r_a1
    sim_a1["net_return"] = r_a1 * p.r_unit_pct
    sim_a1["holding_bars"] = hold_a1

    year_rows = []
    ports = {}
    for name, sim in (("A0", sim0), ("A1_M5", sim_a1)):
        yrows = [_year_row(y, _port(p, sim, y), sim) for y in YEARS]
        po_oos = _pooled([r for r in yrows if r["year"] in TRUE_OOS])
        mc = _mc(po_oos["pnl"], name)
        ports[name] = {"yrows": yrows, "po_oos": po_oos, "mc": mc}
        rmap = r0 if name == "A0" else r1
        ids = oos["trade_id"].to_numpy(dtype=int)
        rr_all = dict(zip(ids, rmap))
        for yr in yrows:
            y = yr["year"]
            ty = oos[oos["year"] == y] if y in TRUE_OOS else trades[trades["year"] == y]
            if y not in TRUE_OOS:
                year_rows.append({"book": name, **{k: yr[k] for k in yr if k != "pnl"},
                                  "trigger_count": 0, "net_benefit_R": 0.0, "tail_loss_rate": float((ty["p0_R"] <= -1).mean()) if len(ty) else 0.0})
                continue
            r = np.array([rr_all[int(t)] for t in ty["trade_id"]])
            b = ty["p0_R"].to_numpy()
            hit = np.array([int(t) in trig for t in ty["trade_id"]])
            ll = ty["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
            rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
            year_rows.append({
                "book": name, **{k: yr[k] for k in yr if k != "pnl"},
                "trigger_count": int(hit.sum()),
                "tail_loss_rate": float((r <= -1).mean()),
                "net_benefit_R": 0.0 if name == "A0" else ts - rl,
                "tail_saved_R": ts, "recovery_lost_R": rl,
                "recovery_preservation": float(1.0 - (((ty["mae_R"] >= 0.75) & rec & (r <= 0)).sum() /
                    max(((ty["mae_R"] >= 0.75) & rec).sum(), 1))),
            })
    pd.DataFrame(year_rows).to_csv(OUT / "yearly.csv", index=False)

    a0, a1p = ports["A0"], ports["A1_M5"]
    y_a0 = [r for r in year_rows if r["book"] == "A0" and r["year"] in TRUE_OOS]
    y_a1 = [r for r in year_rows if r["book"] == "A1_M5" and r["year"] in TRUE_OOS]
    fills = [t["fill"] for t in trig.values()]
    lines = [
        "# A1 + M5 execution",
        "",
        "Research only. Frozen trigger: first M5 `close_R ≤ −0.50` AND frozen Sprint 46 LGBM `p ≥ 0.80`.",
        "Fill = **next M5 open**. SL-first on the same M5 bar. No −0.50 cap. Production unchanged.",
        "",
        f"OOS trades={len(oos)} triggered={len(trig)} ck_hits={n_hit_ck} p<0.80={n_low_p} blocked_by_M5_SL={n_block_sl}",
        "",
        "## Actions (OOS 2022–2026)",
        "",
        _md(
            [{"action": "A0", **e0}, {"action": "A1_M5", **e1}],
            ["action", "mean_R", "pf", "wr", "p_le_m1_00", "tail_saved_R", "recovery_lost_R",
             "net_benefit_R", "recovery_preservation", "n_triggered"],
            {"mean_R": 3, "pf": 2, "wr": 3, "p_le_m1_00": 3, "tail_saved_R": 2, "recovery_lost_R": 2,
             "net_benefit_R": 2, "recovery_preservation": 3, "n_triggered": 0},
        ),
        "",
        f"median A1 fill R={float(np.median(fills)) if fills else 0:.3f} (not capped at −0.50)",
        "",
        "## Rolling walk-forward ($280/year)",
        "",
        _md(
            [{"year": a["year"], "A0_n": a["trades"], "A1_n": b["trades"], "A1_trig": b["trigger_count"],
              "A0_PF": a["pf"], "A1_PF": b["pf"], "A0_DD": a["dd"], "A1_DD": b["dd"],
              "A0_tail": a["tail_loss_rate"], "A1_tail": b["tail_loss_rate"],
              "net_R": b["net_benefit_R"], "A0_AvgR": a.get("avg_r", 0), "A1_AvgR": b.get("avg_r", 0)}
             for a, b in zip(y_a0, y_a1)],
            ["year", "A0_n", "A1_trig", "A0_PF", "A1_PF", "A0_DD", "A1_DD", "A0_tail", "A1_tail", "net_R"],
            {"year": 0, "A0_n": 0, "A1_trig": 0, "A0_PF": 2, "A1_PF": 2, "A0_DD": 3, "A1_DD": 3,
             "A0_tail": 3, "A1_tail": 3, "net_R": 2},
        ),
        "",
        "## Monte Carlo (OOS pooled, 1000 shuffles)",
        "",
        _md(
            [a0["mc"], a1p["mc"]],
            ["policy", "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3},
        ),
        "",
        f"OOS PF {a0['po_oos']['pf']:.2f} → {a1p['po_oos']['pf']:.2f} | DD {a0['po_oos']['dd']:.3f} → {a1p['po_oos']['dd']:.3f} | "
        f"AvgR {a0['po_oos']['avg_r']:.3f} → {a1p['po_oos']['avg_r']:.3f}",
        "",
        "## Leakage / execution",
        "",
        "- Model trained on H1 checkpoint obs (Sprint 46 folds), applied to M5 state (domain shift, documented).",
        "- Trigger uses M5 close_R (PIT). Fill is next M5 open, not the −0.50 cap.",
        "- If M5 low already tags P0 SL on the signal bar, A1 does not fire.",
        "- 2021 stays P0 (no prior train year).",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED. P0: UNCHANGED. A1 M5: NOT SHIPPED.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "production_changed": False,
        "n_oos": int(len(oos)),
        "n_triggered": len(trig),
        "n_ck_hits": n_hit_ck,
        "n_low_p": n_low_p,
        "net_benefit_R": e1["net_benefit_R"],
        "tail_saved_R": e1["tail_saved_R"],
        "recovery_lost_R": e1["recovery_lost_R"],
        "recovery_preservation": e1["recovery_preservation"],
        "p_le_m1_a0": e0["p_le_m1_00"],
        "p_le_m1_a1": e1["p_le_m1_00"],
        "pf_a0": a0["po_oos"]["pf"],
        "pf_a1": a1p["po_oos"]["pf"],
        "dd_a0": a0["po_oos"]["dd"],
        "dd_a1": a1p["po_oos"]["dd"],
    }, indent=2, default=float), encoding="utf-8")
    print(f"triggered={len(trig)} net={e1['net_benefit_R']:.2f} tail {e0['p_le_m1_00']:.3f}->{e1['p_le_m1_00']:.3f} pres={e1['recovery_preservation']:.3f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
