"""Sprint 36 — Market State Engine Research (measurement only).

Uses identical 7-feat WF entries + exit a0.25_d0.08. No retrain / FE / exit / risk ship.

  python apps/report_market_state_engine.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.report_adaptive_vol_risk import (
    EXIT_KW,
    STARTING,
    apply_map,
    iter5_candidates,
    run_portfolio_scaled,
)
from apps.report_edge_attribution import session_of_hour
from apps.run_exit_engine_grid import Paths, build_entry_panel, simulate_combo
from simulation.wf.sim import _load_side, label_regime, load_h1, prepare_market

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint36_market_state"


def _pf(pnl: np.ndarray) -> float:
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _session(h: int) -> str:
    s = session_of_hour(h)
    return {"LONDON_NY": "OVERLAP", "OFF": "OTHER"}.get(s, s)


def build_trades() -> tuple[pd.DataFrame, Paths, dict, np.ndarray, dict[str, np.ndarray]]:
    panel = build_entry_panel()
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    # merge existing context features (no new FE)
    parts = []
    want = [
        "atr_percentile_252", "atr_percent", "ema_trend_duration", "ctx_h4_swing_quality",
        "ema20_distance_atr", "ema50_distance_atr", "ema_cross_distance", "rolling_volatility",
        "ctx_h4_trend_direction", "ctx_h4_volatility_regime",
    ]
    for side in ("long", "short"):
        df = _load_side(side)
        cols = ["timestamp"] + [c for c in want if c in df.columns]
        g = df[cols].copy(); g["side"] = side
        parts.append(g)
    feats = pd.concat(parts, ignore_index=True)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.merge(feats, on=["timestamp", "side"], how="left")

    p = Paths(panel, mkt)
    sim = simulate_combo(p, **EXIT_KW)
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    port = run_portfolio_scaled(order, p, sim, np.ones(p.n))
    taken = port["taken"]
    t = p.panel.iloc[taken].copy().reset_index(drop=True)
    t["pnl"] = port["pnl"]
    t["r_multiple"] = sim["r_multiple"][taken]
    t["mfe_r"] = sim["mfe_r"][taken]
    t["mae_r"] = sim["mae_r"][taken]
    t["net_return"] = sim["net_return"][taken]
    t["equity"] = port["equity"]
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t["year"] = t["timestamp"].dt.year
    t["hour"] = t["timestamp"].dt.hour
    t["entry_idx"] = taken  # index into Paths
    return t, p, mkt, order, sim


def label_states(t: pd.DataFrame) -> pd.DataFrame:
    d = t.copy()
    # Trend from H4 direction if present else ema_trend_duration sign proxy
    if "ctx_h4_trend_direction" in d.columns:
        tr = d["ctx_h4_trend_direction"].astype(float)
        d["trend"] = np.where(tr > 0.33, "UPTREND", np.where(tr < -0.33, "DOWNTREND", "SIDEWAYS"))
    else:
        dur = d.get("ema_trend_duration", pd.Series(0, index=d.index)).astype(float)
        d["trend"] = np.where(dur > 20, "UPTREND", np.where(dur < -20, "DOWNTREND", "SIDEWAYS"))

    atr = d["atr_percentile_252"].astype(float) if "atr_percentile_252" in d.columns else pd.Series(0.5, index=d.index)
    d["atr_pct"] = atr
    bins = [-0.01, 0.20, 0.40, 0.60, 0.80, 1.01]
    labs = ["VERY_LOW", "LOW", "NORMAL", "HIGH", "EXTREME"]
    d["volatility"] = pd.cut(atr.clip(0, 1), bins=bins, labels=labs).astype(str)

    d["session"] = d["hour"].map(_session)

    # Structure from existing ema distances + atr_percent compression/expansion
    above = None
    if "ema20_distance_atr" in d.columns:
        above = d["ema20_distance_atr"].astype(float) > 0
    elif "ema50_distance_atr" in d.columns:
        above = d["ema50_distance_atr"].astype(float) > 0
    else:
        above = pd.Series(True, index=d.index)

    atr_pct_price = d["atr_percent"].astype(float) if "atr_percent" in d.columns else atr
    med = float(atr_pct_price.median()) if atr_pct_price.notna().any() else 0.3
    expansion = atr_pct_price > med * 1.25
    compression = atr_pct_price < med * 0.75
    # precedence: Expansion/Compression override side-of-EMA tag when strong
    struct = np.where(expansion, "EXPANSION", np.where(compression, "COMPRESSION", np.where(above, "ABOVE_EMA", "BELOW_EMA")))
    d["structure"] = struct

    # Context buckets (existing)
    if "ctx_h4_swing_quality" in d.columns:
        sq = d["ctx_h4_swing_quality"].astype(float)
        d["structure_quality"] = np.where(sq >= sq.quantile(0.66), "HIGH_STRUCT", np.where(sq <= sq.quantile(0.33), "LOW_STRUCT", "MID_STRUCT"))
    else:
        d["structure_quality"] = "MID_STRUCT"

    if "ema_trend_duration" in d.columns:
        ed = d["ema_trend_duration"].astype(float).abs()
        d["trend_persistence"] = np.where(
            ed >= ed.quantile(0.66), "LONG_DUR", np.where(ed <= ed.quantile(0.33), "SHORT_DUR", "MID_DUR")
        )
    else:
        d["trend_persistence"] = "MID_DUR"

    # Full state key (Trend × Vol × Session × Structure)
    d["state"] = d["trend"] + "|" + d["volatility"] + "|" + d["session"] + "|" + d["structure"]
    # Richer state optional for inventory
    d["state_rich"] = d["state"] + "|" + d["structure_quality"]
    return d


def iter1_inventory(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = [
        "timestamp", "year", "side", "hour", "trend", "volatility", "session", "structure",
        "structure_quality", "trend_persistence", "state", "state_rich",
        "atr_pct", "pnl", "r_multiple", "mfe_r", "mae_r",
    ]
    for c in ("ctx_h4_swing_quality", "ema_trend_duration", "atr_percentile_252", "atr_percent"):
        if c in d.columns and c not in cols:
            cols.append(c)
    inv = d[[c for c in cols if c in d.columns]].copy()
    inv.to_csv(out / "market_states.csv", index=False)
    return inv


def state_perf(d: pd.DataFrame, key: str = "state") -> pd.DataFrame:
    rows = []
    total_loss = float((-d.loc[d["pnl"] < 0, "pnl"]).sum()) or 1e-12
    # DD contribution proxy: share of negative pnl
    peak = STARTING
    eq = STARTING
    # chronological dd attribution: when equity makes new DD, attribute to current trade's state
    dd_attr = {k: 0.0 for k in d[key].unique()}
    d_sorted = d.sort_values("timestamp")
    for _, r in d_sorted.iterrows():
        eq += float(r["pnl"])
        peak = max(peak, eq)
        dd_now = (peak - eq) / max(peak, 1e-12)
        # accumulate instantaneous dd increase
        # ponytail: attribute full trade loss magnitude when losing
        if r["pnl"] < 0:
            dd_attr[r[key]] = dd_attr.get(r[key], 0.0) + float(-r["pnl"])

    dd_tot = sum(dd_attr.values()) or 1e-12
    for st, g in d.groupby(key):
        pnl = g["pnl"].to_numpy(float)
        rows.append({
            "state": st,
            "trades": len(g),
            "years": int(g["year"].nunique()),
            "pf": _pf(pnl),
            "wr": float(np.mean(pnl > 0)),
            "expectancy": float(pnl.mean()),
            "avg_r": float(g["r_multiple"].mean()),
            "avg_mae": float(g["mae_r"].mean()),
            "avg_mfe": float(g["mfe_r"].mean()),
            "net_pnl": float(pnl.sum()),
            "loss_share": float((-pnl[pnl < 0]).sum() / total_loss),
            "dd_contribution": float(dd_attr.get(st, 0.0) / dd_tot),
            "mean_atr_pct": float(g["atr_pct"].mean()) if "atr_pct" in g else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["pf", "trades"], ascending=[False, False])


def iter2_performance(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    perf = state_perf(d, "state")
    perf.to_csv(out / "market_state_performance.csv", index=False)
    perf.head(20).to_csv(out / "market_state_top20.csv", index=False)
    perf.sort_values(["dd_contribution", "pf"], ascending=[False, True]).head(20).to_csv(out / "market_state_bottom20_dd.csv", index=False)
    # also bottom by PF with min trades
    liquid = perf[perf["trades"] >= 15]
    liquid.sort_values("pf").head(20).to_csv(out / "market_state_bottom20_pf.csv", index=False)
    return perf


def iter3_yearly(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for (st, y), g in d.groupby(["state", "year"]):
        pnl = g["pnl"].to_numpy(float)
        rows.append({
            "state": st, "year": int(y), "trades": len(g),
            "pf": _pf(pnl), "wr": float(np.mean(pnl > 0)),
            "expectancy": float(pnl.mean()), "net_pnl": float(pnl.sum()),
        })
    yearly = pd.DataFrame(rows)
    yearly.to_csv(out / "market_state_yearly.csv", index=False)

    # stability flags
    flags = []
    for st, g in yearly.groupby("state"):
        years = sorted(g["year"].unique())
        pfs = g.set_index("year")["pf"]
        flags.append({
            "state": st,
            "n_years": len(years),
            "years": ",".join(map(str, years)),
            "mean_pf": float(g["pf"].replace([np.inf], np.nan).mean()),
            "min_pf": float(g["pf"].replace([np.inf], np.nan).min()),
            "stable_pos": bool(len(years) >= 4 and (g["pf"].replace([np.inf], 3.0) >= 1.0).mean() >= 0.75),
            "collapse_any": bool((g["pf"].replace([np.inf], 3.0) < 0.8).any() and g["trades"].sum() >= 20),
            "appears_2026_only": bool(years == [2026]),
            "new_in_2026": bool(2026 in years and len(years) == 1),
            "total_trades": int(g["trades"].sum()),
        })
    stab = pd.DataFrame(flags).sort_values(["stable_pos", "mean_pf"], ascending=[False, False])
    stab.to_csv(out / "market_state_stability.csv", index=False)
    return yearly


def iter4_transitions(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    s = d.sort_values("timestamp").reset_index(drop=True)
    rows = []
    for i in range(1, len(s)):
        a, b = s.loc[i - 1, "state"], s.loc[i, "state"]
        # also coarser vol / trend transitions
        rows.append({
            "from_state": a, "to_state": b,
            "from_vol": s.loc[i - 1, "volatility"], "to_vol": s.loc[i, "volatility"],
            "from_trend": s.loc[i - 1, "trend"], "to_trend": s.loc[i, "trend"],
            "pnl": float(s.loc[i, "pnl"]),
            "r_multiple": float(s.loc[i, "r_multiple"]),
            "year": int(s.loc[i, "year"]),
        })
    tr = pd.DataFrame(rows)
    # aggregate dangerous transitions (vol / trend)
    agg_rows = []
    for keys in (("from_vol", "to_vol"), ("from_trend", "to_trend"), ("from_state", "to_state")):
        for k, g in tr.groupby(list(keys)):
            pnl = g["pnl"].to_numpy(float)
            if len(g) < 5:
                continue
            agg_rows.append({
                "kind": f"{keys[0]}->{keys[1]}",
                "from": k[0], "to": k[1], "n": len(g),
                "pf": _pf(pnl), "wr": float(np.mean(pnl > 0)),
                "expectancy": float(pnl.mean()),
                "avg_r": float(g["r_multiple"].mean()),
            })
    agg = pd.DataFrame(agg_rows).sort_values("expectancy")
    agg.to_csv(out / "market_state_transition.csv", index=False)
    return agg


def iter5_survival(d: pd.DataFrame, perf: pd.DataFrame, p: Paths, sim: dict, out: Path) -> pd.DataFrame:
    order_all = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    # rank states by PF among liquid
    liquid = perf[perf["trades"] >= 10].sort_values("pf", ascending=False)
    rows = []
    for top_n, label in ((10, "top10"), (20, "top20"), (30, "top30"), (10_000, "all")):
        allow = set(liquid.head(top_n)["state"]) if top_n < 10_000 else set(perf["state"])
        # mask: only take entries whose eventual trade state is in allow — approximate by labeling ALL entries
        # For fair survival: filter trades post-hoc by state (measurement of "if only traded these states")
        g = d[d["state"].isin(allow)] if top_n < 10_000 else d
        pnl = g["pnl"].to_numpy(float)
        # reconstruct equity path in time order
        gs = g.sort_values("timestamp")
        eq = STARTING
        peak = STARTING
        mdd = 0.0
        for x in gs["pnl"]:
            eq += float(x)
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / max(peak, 1e-12))
        rows.append({
            "universe": label,
            "n_states": len(allow) if top_n < 10_000 else int(perf["state"].nunique()),
            "trades": len(g),
            "pf": _pf(pnl),
            "wr": float(np.mean(pnl > 0)) if len(pnl) else 0.0,
            "ret": float(eq / STARTING - 1.0),
            "dd": mdd,
            "final_equity": eq,
        })
    surv = pd.DataFrame(rows)
    surv.to_csv(out / "state_survival.csv", index=False)
    return surv


def iter6_risk(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    total_loss = float((-d.loc[d["pnl"] < 0, "pnl"]).sum()) or 1e-12
    total_mae = float(d["mae_r"].sum()) or 1e-12
    # losing streaks attribution to state of streak-ending / each loss
    streak = 0
    streak_loss = {s: 0 for s in d["state"].unique()}
    for _, r in d.sort_values("timestamp").iterrows():
        if r["pnl"] < 0:
            streak += 1
            if streak >= 3:
                streak_loss[r["state"]] = streak_loss.get(r["state"], 0) + 1
        else:
            streak = 0
    for st, g in d.groupby("state"):
        pnl = g["pnl"].to_numpy(float)
        rows.append({
            "state": st,
            "trades": len(g),
            "loss_pnl": float((-pnl[pnl < 0]).sum()),
            "loss_share": float((-pnl[pnl < 0]).sum() / total_loss),
            "mae_share": float(g["mae_r"].sum() / total_mae),
            "losing_streak_hits": int(streak_loss.get(st, 0)),
            "pf": _pf(pnl),
            "avg_mae": float(g["mae_r"].mean()),
        })
    risk = pd.DataFrame(rows).sort_values("loss_share", ascending=False)
    risk.to_csv(out / "state_risk.csv", index=False)
    return risk


def iter7_adaptive_sim(d: pd.DataFrame, p: Paths, sim: dict, order: np.ndarray, out: Path) -> pd.DataFrame:
    """Compare ATR-only map_conservative vs state-tiered risk (measurement)."""
    # ATR-only
    atr_cand = next(c for c in iter5_candidates() if c["name"] == "map_conservative")
    # need atr on full Paths panel
    panel = p.panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    # merge atr_pct onto all entries
    atr_map = d.set_index("entry_idx")["atr_pct"] if "entry_idx" in d.columns else None
    atr_pct_all = np.full(p.n, 0.5)
    if "atr_percentile_252" in panel.columns:
        atr_pct_all = panel["atr_percentile_252"].astype(float).fillna(0.5).to_numpy()
    # build fake frame for apply_map
    fake = pd.DataFrame({"atr_pct": atr_pct_all})
    atr_mult = apply_map(fake, atr_cand)
    atr_port = run_portfolio_scaled(order, p, sim, atr_mult)

    # State policy: tier by state PF / risk buckets from traded states, apply to entries via nearest state label
    # Label ALL entries with same state function
    all_e = p.panel.copy().reset_index(drop=True)
    all_e["timestamp"] = pd.to_datetime(all_e["timestamp"], utc=True)
    all_e["hour"] = all_e["timestamp"].dt.hour
    # attach features already on panel
    all_e = label_states(all_e.assign(
        pnl=0.0, r_multiple=0.0, mfe_r=0.0, mae_r=0.0, year=all_e["timestamp"].dt.year
    ))
    # score states from trade performance
    perf = state_perf(d, "state")
    # tiers by PF among states with >=20 trades; else 1.0
    score = perf.set_index("state")["pf"].replace([np.inf], 3.0)
    def tier(pf: float) -> float:
        if pf >= 2.2:
            return 1.0
        if pf >= 1.6:
            return 0.7
        if pf >= 1.1:
            return 0.4
        return 0.1

    state_risk = {st: tier(float(score.get(st, 1.5))) for st in all_e["state"].unique()}
    # unknown/rare states → 0.7 default
    counts = d["state"].value_counts()
    for st in list(state_risk):
        if counts.get(st, 0) < 20:
            state_risk[st] = 0.7

    st_mult = all_e["state"].map(state_risk).astype(float).fillna(0.7).to_numpy()
    st_port = run_portfolio_scaled(order, p, sim, st_mult)
    fixed = run_portfolio_scaled(order, p, sim, np.ones(p.n))

    rows = [
        {"policy": "fixed_100", **{k: fixed[k] for k in ("n_trades", "profit_factor", "max_drawdown", "total_return", "win_rate", "sharpe")}},
        {"policy": "atr_map_conservative", **{k: atr_port[k] for k in ("n_trades", "profit_factor", "max_drawdown", "total_return", "win_rate", "sharpe")}},
        {"policy": "market_state_tiers", **{k: st_port[k] for k in ("n_trades", "profit_factor", "max_drawdown", "total_return", "win_rate", "sharpe")}},
    ]
    # yearly 2026 help
    for name, mult in (("fixed_100", np.ones(p.n)), ("atr_map_conservative", atr_mult), ("market_state_tiers", st_mult)):
        idx = np.where(pd.DatetimeIndex(p.ts).year == 2026)[0]
        if len(idx):
            o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8)]
            yp = run_portfolio_scaled(o, p, sim, mult, starting=STARTING)
            rows.append({
                "policy": f"{name}__y2026",
                "n_trades": yp["n_trades"], "profit_factor": yp["profit_factor"],
                "max_drawdown": yp["max_drawdown"], "total_return": yp["total_return"],
                "win_rate": yp["win_rate"], "sharpe": yp["sharpe"],
            })
    simdf = pd.DataFrame(rows)
    simdf.to_csv(out / "adaptive_policy_simulation.csv", index=False)
    # save mapping
    pd.DataFrame([{"state": k, "risk_mult": v} for k, v in sorted(state_risk.items())]).to_csv(
        out / "state_risk_mapping.csv", index=False
    )
    return simdf


def iter8_explain(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Simple mutual-info / variance share of state components (measurement)."""
    # How much of pnl variance is explained by each factor (ANOVA-like eta²)
    rows = []
    y = d["pnl"].to_numpy(float)
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    for col in ("trend", "volatility", "session", "structure", "structure_quality", "atr_pct"):
        if col == "atr_pct":
            # bin atr for fair compare
            x = pd.qcut(d["atr_pct"].rank(method="first"), 5, labels=False, duplicates="drop")
        else:
            x = d[col]
        ss_between = 0.0
        for _, g in d.groupby(x):
            yy = g["pnl"].to_numpy(float)
            ss_between += len(yy) * (yy.mean() - y.mean()) ** 2
        eta = ss_between / ss_tot
        rows.append({"factor": col, "eta_sq_pnl": float(eta), "levels": int(pd.Series(x).nunique())})
    # normalize to %
    expl = pd.DataFrame(rows).sort_values("eta_sq_pnl", ascending=False)
    s = expl["eta_sq_pnl"].sum() or 1e-12
    expl["share_pct"] = expl["eta_sq_pnl"] / s * 100
    expl.to_csv(out / "state_explainability.csv", index=False)
    return expl


def write_report(
    out: Path,
    d: pd.DataFrame,
    perf: pd.DataFrame,
    yearly: pd.DataFrame,
    trans: pd.DataFrame,
    surv: pd.DataFrame,
    risk: pd.DataFrame,
    asim: pd.DataFrame,
    expl: pd.DataFrame,
) -> str:
    n_states = int(d["state"].nunique())
    top = perf.iloc[0]
    worst_dd = risk.iloc[0]
    # stable states
    stab = pd.read_csv(out / "market_state_stability.csv")
    stable = stab[stab["stable_pos"] & (stab["total_trades"] >= 30)]
    # 2026 dominant
    y26 = d[d["year"] == 2026]
    top2026 = y26["state"].value_counts().head(5) if len(y26) else pd.Series(dtype=int)

    # NO TRADE candidates
    nt = perf[
        (perf["trades"] >= 100)
        & (perf["years"] >= 4)
        & (perf["pf"] < 0.8)
        & (perf["dd_contribution"] >= 0.05)
    ]
    # exclude 2026-only
    stab_map = stab.set_index("state")
    if not nt.empty:
        nt = nt[~nt["state"].map(lambda s: bool(stab_map.loc[s, "appears_2026_only"]) if s in stab_map.index else False)]

    atr_row = asim[asim["policy"] == "atr_map_conservative"].iloc[0]
    st_row = asim[asim["policy"] == "market_state_tiers"].iloc[0]
    fix_row = asim[asim["policy"] == "fixed_100"].iloc[0]

    state_beats_atr = (
        float(st_row["max_drawdown"]) <= float(atr_row["max_drawdown"]) + 0.005
        and float(st_row["profit_factor"]) >= float(atr_row["profit_factor"]) * 0.98
        and (
            float(st_row["max_drawdown"]) < float(atr_row["max_drawdown"]) - 0.01
            or float(st_row["profit_factor"]) > float(atr_row["profit_factor"]) + 0.02
        )
    )
    # explainability: is vol dominant?
    vol_share = float(expl.loc[expl["factor"] == "volatility", "share_pct"].iloc[0]) if (expl["factor"] == "volatility").any() else 0
    atr_share = float(expl.loc[expl["factor"] == "atr_pct", "share_pct"].iloc[0]) if (expl["factor"] == "atr_pct").any() else 0
    multi_needed = (vol_share + atr_share) < 70  # other factors matter

    # PASS/STOP
    has_stable = not stable.empty
    surv_top = surv[surv["universe"] == "top20"].iloc[0] if (surv["universe"] == "top20").any() else None
    explains_better = surv_top is not None and float(surv_top["dd"]) < float(surv[surv["universe"] == "all"].iloc[0]["dd"]) - 0.02

    if state_beats_atr and has_stable and explains_better:
        verdict = "YES"
        success = "PASS"
    elif has_stable or explains_better:
        verdict = "PARTIAL"
        success = "PARTIAL — useful for diagnosis; not clearly better than ATR adaptive for policy"
    else:
        verdict = "NO"
        success = "STOP — no significant stable improvement over ATR-only"

    no_trade_txt = (
        "Tidak ada state yang layak dijadikan NO TRADE."
        if nt.empty
        else nt[["state", "trades", "years", "pf", "dd_contribution"]].to_string(index=False)
    )

    lines = [
        "# Sprint 36 — Market State Engine Research",
        "",
        f"**Success:** {success}",
        f"**Policy Engine verdict: {verdict}**",
        "",
        f"Trades OOS: {len(d)} | Unique states (Trend×Vol×Session×Structure): **{n_states}**",
        "",
        "## 1. Total market state muncul",
        f"{n_states} kombinasi muncul di trade log (dari 3×5×5×4 = 300 teoritis).",
        "",
        "## 2. State paling profitable",
        f"`{top['state']}` — trades={int(top['trades'])} PF={top['pf']:.2f} WR={top['wr']:.1%} "
        f"E[R]={top['avg_r']:.3f} years={int(top['years'])}",
        "",
        "Top 5 by PF (trades≥15):",
        perf[perf["trades"] >= 15].head(5)[["state", "trades", "pf", "wr", "expectancy", "dd_contribution"]].to_string(index=False),
        "",
        "## 3. State paling merusak DD",
        f"`{worst_dd['state']}` — loss_share={worst_dd['loss_share']:.1%} streak_hits={int(worst_dd['losing_streak_hits'])} "
        f"PF={worst_dd['pf']:.2f} trades={int(worst_dd['trades'])}",
        "",
        risk.head(8)[["state", "trades", "loss_share", "mae_share", "losing_streak_hits", "pf"]].to_string(index=False),
        "",
        "## 4. Konsistensi antar tahun",
        f"Stable positive states (≥4y, mostly PF≥1, ≥30 trades): **{len(stable)}**",
        stable.head(10)[["state", "n_years", "mean_pf", "min_pf", "total_trades"]].to_string(index=False) if not stable.empty else "(none)",
        "",
        "## 5. State paling banyak di 2026",
        top2026.to_string() if len(top2026) else "(no 2026 trades)",
        "",
        "## 6. Volatility saja cukup?",
        f"Explainability share — volatility:{vol_share:.1f}% atr_pct:{atr_share:.1f}% | multi-factor needed? **{'YES' if multi_needed else 'NO / mostly vol'}**",
        expl.to_string(index=False),
        "",
        "Survival (filter to top states):",
        surv.to_string(index=False),
        "",
        "## 7. Adaptive Risk: ATR vs Market State",
        asim[asim["policy"].isin(["fixed_100", "atr_map_conservative", "market_state_tiers"])].to_string(index=False),
        "",
        f"Market State tiers {'MENANG' if state_beats_atr else 'TIDAK mengungguli'} ATR `map_conservative` "
        f"(DD {st_row['max_drawdown']:.1%} vs {atr_row['max_drawdown']:.1%}; "
        f"PF {st_row['profit_factor']:.2f} vs {atr_row['profit_factor']:.2f}).",
        "",
        "## 8. NO TRADE state?",
        no_trade_txt,
        "",
        "## 9. Market State sebagai Policy Engine?",
        f"**{verdict}**",
        "- YES = state menjelaskan + stabil + mengungguli ATR adaptive.",
        "- PARTIAL = bagus untuk diagnosis/attribution; ATR percentile tetap policy risk yang lebih sederhana & kuat.",
        "- NO = tidak ada lift stabil.",
        "",
        "## Dangerous transitions (worst expectancy)",
        trans.head(12).to_string(index=False) if not trans.empty else "(n/a)",
        "",
        "## Files",
        "market_states.csv, market_state_performance.csv, market_state_yearly.csv,",
        "market_state_transition.csv, state_survival.csv, state_risk.csv,",
        "adaptive_policy_simulation.csv, state_explainability.csv, summary.md",
    ]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return verdict


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Sprint 36 Market State Engine ===")
    print("Building trades (7-feat + exit a0.25_d0.08)…")
    t, p, mkt, order, sim = build_trades()
    d = label_states(t)
    print(f"trades={len(d)} states={d['state'].nunique()}")

    print("ITER 1 inventory…")
    iter1_inventory(d, OUT)
    print("ITER 2 performance…")
    perf = iter2_performance(d, OUT)
    print(perf.head(5).to_string(index=False))
    print("ITER 3 yearly…")
    yearly = iter3_yearly(d, OUT)
    print("ITER 4 transitions…")
    trans = iter4_transitions(d, OUT)
    print(trans.head(5).to_string(index=False))
    print("ITER 5 survival…")
    surv = iter5_survival(d, perf, p, sim, OUT)
    print(surv.to_string(index=False))
    print("ITER 6 risk…")
    risk = iter6_risk(d, OUT)
    print("ITER 7 adaptive sim…")
    asim = iter7_adaptive_sim(d, p, sim, order, OUT)
    print(asim.to_string(index=False))
    print("ITER 8 explainability…")
    expl = iter8_explain(d, OUT)
    print(expl.to_string(index=False))

    verdict = write_report(OUT, d, perf, yearly, trans, surv, risk, asim, expl)
    print(f"\nVERDICT={verdict}")
    print(f"DONE -> {OUT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
