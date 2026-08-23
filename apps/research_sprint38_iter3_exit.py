"""Sprint 38 iter 3 — Dynamic exit policies from OOS continuation score.

Frozen entry/risk/multi-trade. $280. Does not change production.

  python apps/research_sprint38_iter3_exit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd
from apps.report_adaptive_vol_risk import enrich_vol, mapping_from_edges
from apps.report_multi_trade_engine import EngineCfg, run_multi
from apps.research_sprint38_continuation import (
    PANEL,
    STATE_FEATS,
    TEST_YEARS,
    _fit_predict,
    build_states,
)
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from simulation.wf.sim import COST, _load_side, load_h1, prepare_market

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint38_dynamic_exit"
STARTING = 280.0
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
ATR_MAP = {"edges": [0.30, 0.60, 0.80, 0.90], "risks": [1.0, 0.70, 0.40, 0.25, 0.10]}
CFG = EngineCfg(
    name="parallel_mo5_d0.0_cd0_h3.0",
    family="parallel",
    max_positions=5,
    min_distance_atr=0.0,
    cooldown_bars=0,
    heat_budget_r=3.0,
)
H_LABEL, THR_LABEL = 8, 2.0


def _load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL)
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


def simulate_scored(
    p: Paths,
    score: np.ndarray,
    *,
    mode: str,
    cut: float = 0.5,
    hold_cut: float = 0.6,
) -> dict[str, np.ndarray]:
    """Baseline trail + score overlay. score shape (n, CAP+1), nan = 0.5."""
    n = p.n
    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    active = np.ones(n, dtype=bool)
    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    trail_moved = np.zeros(n, dtype=bool)
    sc = np.where(np.isfinite(score), score, 0.5)

    for j in range(1, CAP + 1):
        ended = active & (p.last_off < j)
        if ended.any():
            lo = p.last_off[ended]
            exit_r[ended] = p.close_r[ended, lo]
            exit_off[ended] = lo
            reason[ended] = 4
            active[ended] = False
        m = active
        if not m.any():
            break
        fav_j = p.fav[m, j]
        adv_j = p.adv[m, j]
        extreme[m] = np.maximum(extreme[m], fav_j)
        mae[m] = np.maximum(mae[m], adv_j)
        sj = sc[np.where(m)[0], j]
        if mode == "trail":
            dist = np.full(int(m.sum()), 0.08)
            act = np.full(int(m.sum()), 0.25)
            dist[sj < 0.33] = 0.06
            act[sj < 0.33] = 0.20
            dist[sj >= 0.66] = 0.16
            act[sj >= 0.66] = 0.50
        else:
            dist = np.full(int(m.sum()), 0.08)
            act = np.full(int(m.sum()), 0.25)
        skip_trail = np.zeros(int(m.sum()), dtype=bool)
        if mode == "hold":
            skip_trail = sj >= hold_cut
        act_ok = (extreme[m] >= act) & ~skip_trail
        if act_ok.any():
            idx_m = np.where(m)[0][act_ok]
            new_sl = extreme[idx_m] - dist[act_ok] * p.atr_over_r[idx_m, j]
            moved = new_sl > sl[idx_m]
            sl[idx_m[moved]] = new_sl[moved]
            trail_moved[idx_m[moved]] = True
        hit_sl = m.copy()
        hit_sl[m] = adv_j >= -sl[m]
        if hit_sl.any():
            exit_r[hit_sl] = sl[hit_sl]
            exit_off[hit_sl] = j
            reason[hit_sl] = np.where(trail_moved[hit_sl], 1, 0)
            active[hit_sl] = False
        m = active
        if not m.any():
            break
        if mode == "binary":
            sj2 = sc[np.where(m)[0], j]
            cut_hit = np.zeros(p.n, dtype=bool)
            cut_hit[np.where(m)[0]] = sj2 < cut
            if cut_hit.any():
                cr = p.close_r[cut_hit, j]
                ok = np.isfinite(cr)
                take = cut_hit.copy()
                take[cut_hit] = ok
                exit_r[take] = p.close_r[take, j]
                exit_off[take] = j
                reason[take] = 4
                active[take] = False
        if j == CAP:
            fin = active & (p.last_off >= j)
            exit_r[fin] = p.close_r[fin, j]
            exit_off[fin] = j
            reason[fin] = 4
            active[fin] = False

    if active.any():
        lo = p.last_off[active]
        exit_r[active] = p.close_r[active, lo]
        exit_off[active] = lo
    net = exit_r * p.r_unit_pct - COST
    return {
        "net_return": net,
        "r_multiple": net / p.r_unit_pct,
        "holding_bars": np.maximum(exit_off, 1),
        "reason": reason,
        "mfe_r": extreme,
        "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
    }


def _score_matrix(p: Paths, sim: dict, *, until_exit_only: bool = False) -> np.ndarray:
    """OOS XGB P(future_max_R_8>=2). Train on in-trade bars year<Y; score 1..last_off.

    Bars after the baseline trail exit are OOD for the label (model never saw them
    in training). Hold-if-high uses those scores anyway — that's the test.
    until_exit_only: only build states while the baseline trade is alive (faster).
    """
    hold = np.asarray(sim["holding_bars"], dtype=int)
    st = build_states(p, {"holding_bars": hold if until_exit_only else p.last_off})
    fut = st[f"future_max_R_{H_LABEL}"].to_numpy()
    n_fut = st[f"n_future_{H_LABEL}"].to_numpy()
    y = (fut >= THR_LABEL).astype(int)
    ok = np.isfinite(fut) & (n_fut >= max(4, H_LABEL // 4))
    ti = st["trade_i"].to_numpy(dtype=int)
    jj = st["j"].to_numpy(dtype=int)
    in_trade = jj <= hold[ti]
    X = st[STATE_FEATS].astype(float).fillna(0.0).to_numpy()
    years = st["year"].to_numpy()
    oos = np.full(len(st), np.nan)
    for te in TEST_YEARS:
        tr = (years < te) & ok & in_trade
        te_m = years == te
        if tr.sum() < 200 or te_m.sum() < 20:
            continue
        oos[te_m] = _fit_predict("xgb", X[tr], y[tr], X[te_m])
    mat = np.full((p.n, CAP + 1), np.nan)
    good = np.isfinite(oos)
    mat[ti[good], jj[good]] = oos[good]
    last = np.full(p.n, 0.5)
    for j in range(1, CAP + 1):
        known = np.isfinite(mat[:, j])
        last[known] = mat[known, j]
        mat[~known, j] = last[~known]
    return mat


def _port(p, sim, lot_mult, atr_pct, probs, trend_ok, **port_kw):
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    return run_multi(
        order, p, sim, lot_mult, atr_pct, probs, trend_ok, CFG, starting=STARTING,
        **port_kw,
    )


def _yearly(
    name: str,
    p: Paths,
    sim: dict,
    lot_mult: np.ndarray,
    atr_pct: np.ndarray,
    probs: np.ndarray,
    trend_ok: np.ndarray,
    **port_kw,
) -> list[dict]:
    """Isolated $280 portfolio per test year (not a slice of the compound run)."""
    rows = []
    ts_ns = pd.DatetimeIndex(p.ts).as_unit("ns").asi8
    for y in sorted(set(int(v) for v in p.year)):
        idx = np.where(p.year == y)[0]
        order = idx[np.argsort(ts_ns[idx], kind="stable")]
        port = run_multi(
            order, p, sim, lot_mult, atr_pct, probs, trend_ok, CFG, starting=STARTING,
            **port_kw,
        )
        taken = np.asarray(port["taken"], dtype=int)
        r = sim["r_multiple"][taken] if taken.size else np.array([])
        pnl = np.asarray(port["pnl"], dtype=float)
        w, l = pnl[pnl > 0], pnl[pnl < 0]
        payoff = float(w.mean() / -l.mean()) if w.size and l.size else 0.0
        rows.append(
            {
                "candidate": name,
                "year": y,
                "candidates": int(idx.size),
                "trades": int(port["n_trades"]),
                "wr": float(port["win_rate"]),
                "payoff": payoff,
                "pf": float(port["profit_factor"]),
                "dd": float(port["max_drawdown"]),
                "ret": float(port["total_return"]),
                "n": int(port["n_trades"]),
                "avg_R": float(r.mean()) if r.size else 0.0,
                "trail_pct": float((sim["reason"][taken] == 1).mean()) if taken.size else 0.0,
                "blown": bool(port["blown"]),
            }
        )
    return rows


def _row(name: str, port: dict, sim: dict, p: Paths) -> dict:
    taken = np.asarray(port["taken"], dtype=int)
    r = sim["r_multiple"][taken]
    reason = sim["reason"][taken]
    years = p.year[taken]
    y26 = years == 2026
    gp, gl = float(r[r > 0].sum()), float(-r[r < 0].sum())
    mfe = sim["mfe_r"][taken]
    cap = np.where(mfe > 1e-9, r / mfe, np.nan)
    return {
        "candidate": name,
        "trades": int(port["n_trades"]),
        "pf": float(port["profit_factor"]),
        "dd": float(port["max_drawdown"]),
        "ret": float(port["total_return"]),
        "avg_R": float(r.mean()) if r.size else 0.0,
        "trail_pct": float((reason == 1).mean()) if r.size else 0.0,
        "sl_pct": float((reason == 0).mean()) if r.size else 0.0,
        "other_pct": float((reason >= 2).mean()) if r.size else 0.0,
        "mfe_capture_med": float(np.nanmedian(cap)) if r.size else 0.0,
        "n_2026": int(y26.sum()),
        "pf_2026": (
            float(r[y26][r[y26] > 0].sum() / max(-r[y26][r[y26] < 0].sum(), 1e-12))
            if y26.any()
            else 0.0
        ),
        "blown": bool(port["blown"]),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(_load_panel(), mkt)
    d = enrich_vol(p.panel, mkt)
    atr_pct = d["atr_pct"].astype(float).fillna(0.5).to_numpy()
    lot_mult = mapping_from_edges(ATR_MAP["edges"], ATR_MAP["risks"])(atr_pct)
    probs = p.panel["y_prob"].astype(float).to_numpy()
    trend_ok = np.ones(p.n, dtype=bool)
    print("baseline + OOS scores")
    base_sim = simulate_combo(p, **EXIT_KW)
    score = _score_matrix(p, base_sim)
    rows = [_row("baseline_a0.25_d0.08", _port(p, base_sim, lot_mult, atr_pct, probs, trend_ok), base_sim, p)]
    yearly_rows = _yearly("baseline_a0.25_d0.08", p, base_sim, lot_mult, atr_pct, probs, trend_ok)
    print(f"  baseline PF={rows[0]['pf']:.3f} DD={rows[0]['dd']*100:.1f}% n={rows[0]['trades']}")

    cands: list[tuple[str, dict]] = [("dyn_trail_3bucket", dict(mode="trail"))]
    for c in (0.30, 0.40, 0.50, 0.60, 0.70):
        cands.append((f"binary_cut_{c:.2f}", dict(mode="binary", cut=c)))
    for h in (0.50, 0.60, 0.70):
        cands.append((f"hold_if_score>={h:.2f}", dict(mode="hold", hold_cut=h)))

    for name, kw in cands:
        sim = simulate_scored(p, score, **kw)
        port = _port(p, sim, lot_mult, atr_pct, probs, trend_ok)
        rec = _row(name, port, sim, p)
        rows.append(rec)
        yearly_rows.extend(_yearly(name, p, sim, lot_mult, atr_pct, probs, trend_ok))
        print(f"  {name}: PF={rec['pf']:.3f} DD={rec['dd']*100:.1f}% n={rec['trades']} avgR={rec['avg_R']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exit_policy_grid.csv", index=False)
    ydf = pd.DataFrame(yearly_rows)
    ydf.to_csv(OUT / "yearly_comparison.csv", index=False)
    base = df.iloc[0]
    df["d_pf"] = df["pf"] - base["pf"]
    df["d_dd"] = df["dd"] - base["dd"]
    df["n_vs_base"] = df["trades"] / max(base["trades"], 1)
    win = df.loc[
        (df["candidate"] != "baseline_a0.25_d0.08")
        & (df["pf"] > base["pf"])
        & (df["dd"] <= base["dd"] + 0.01)
        & (df["n_vs_base"] >= 0.90)
        & (df["blown"] == False)
    ]
    verdict = "NO MATERIAL BENEFIT"
    best_name = "baseline_a0.25_d0.08"
    if len(win):
        best = win.sort_values(["pf", "dd"], ascending=[False, True]).iloc[0]
        best_name = str(best["candidate"])
        verdict = "PERFORMANCE WINNER" if best["pf"] > base["pf"] + 0.05 else "MARGINAL"
    lines = [
        "# Sprint 38 Iterasi 3 — Dynamic Exit from continuation score",
        "",
        "OOS XGB `future_max_R_8>=2`. Frozen FEAT7 top5%, ATR map, parallel_mo5 heat 3R, **$280**.",
        "Iterasi 2 was STOP (weak/unstable ranking). This run is diagnostic only.",
        "",
        f"Baseline PF **{base['pf']:.3f}** DD **{base['dd']*100:.1f}%** n={int(base['trades'])} avgR={base['avg_R']:.3f}",
        "",
        "| Candidate | n | PF | DD | avg R | TRAIL | 2026 PF | vs PF | vs DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['trades']} | {r['pf']:.3f} | {r['dd']*100:.1f}% | "
            f"{r['avg_R']:.3f} | {r['trail_pct']:.0%} | {r['pf_2026']:.2f} | "
            f"{r['pf']-base['pf']:+.3f} | {(r['dd']-base['dd'])*100:+.1f}pp |"
        )
    years = sorted(ydf["year"].unique())
    show = list(df["candidate"])
    lines += [
        "",
        "## Yearly (isolated $280)",
        "",
        "### PF",
        "",
        "| Year | " + " | ".join(show) + " |",
        "|---:|" + "|".join("---:" for _ in show) + "|",
    ]
    for y in years:
        cells = [str(int(y))]
        for c in show:
            hit = ydf[(ydf["year"] == y) & (ydf["candidate"] == c)]
            cells.append(f"{hit.iloc[0]['pf']:.2f}" if len(hit) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "### DD / n / avg R",
        "",
        "| Candidate | Year | n | PF | DD | avg R | ret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in yearly_rows:
        if rec["candidate"] not in show:
            continue
        lines.append(
            f"| {rec['candidate']} | {rec['year']} | {rec['n']} | {rec['pf']:.2f} | "
            f"{rec['dd']*100:.1f}% | {rec['avg_R']:.3f} | {rec['ret']:.2f}x |"
        )
    lines += [
        "",
        f"Best vs gates: **{best_name}**",
        f"Verdict: **{verdict}**",
        "",
        "Production exit unchanged unless PERFORMANCE WINNER with OOS robustness (iter 2 already failed that bar).",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"verdict": verdict, "best": best_name, "baseline_pf": float(base["pf"])}, indent=2),
        encoding="utf-8",
    )
    print(f"VERDICT {verdict} best={best_name}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
