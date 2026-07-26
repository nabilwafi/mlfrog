"""Sprint 35 — Adaptive Volatility Risk Research (measurement only).

Fixed: 7-feat FS entries, exit a0.25_d0.08, no retrain / no FE / no threshold change.
Only simulated lot multipliers by volatility regime.

  python apps/report_adaptive_vol_risk.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apps.run_exit_engine_grid import (
    FEAT7,
    Paths,
    STARTING,
    build_entry_panel,
    simulate_combo,
)
from simulation.wf.sim import (
    CONTRACT_SIZE,
    DAILY_LOSS_STOP_R,
    FIXED_LOT,
    LEVERAGE,
    RISK_BASE,
    SL_ATR,
    STOP_OUT_LEVEL,
    _load_side,
    load_h1,
    max_dd_from_equity,
    prepare_market,
    required_margin,
)

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint35_adaptive_vol_risk"
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
CLUSTERS = ("VERY_LOW", "LOW", "NORMAL", "HIGH", "EXTREME")
RISK_LEVELS = (1.0, 0.75, 0.50, 0.25, 0.10)
N_MC = 1000
RNG = np.random.default_rng(42)


def _pf(pnl: np.ndarray) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _sharpe(eq: np.ndarray, starting: float) -> float:
    if len(eq) < 2:
        return 0.0
    rets = np.diff(np.concatenate([[starting], eq])) / np.maximum(np.concatenate([[starting], eq[:-1]]), 1e-9)
    if rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(len(rets)))


def _ulcer(eq: np.ndarray, starting: float) -> float:
    curve = np.concatenate([[starting], eq.astype(float)])
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / np.maximum(peak, 1e-12)
    return float(np.sqrt(np.mean(dd ** 2)))


def _mar(total_return: float, max_dd: float) -> float:
    if max_dd <= 1e-12:
        return float("inf") if total_return > 0 else 0.0
    return float(total_return / max_dd)


def _recovery_factor(total_return: float, max_dd: float, starting: float, final: float) -> float:
    # classic: net profit / abs(max DD $)
    net = final - starting
    dd_cash = max_dd * starting  # approx; better use peak-based
    if abs(dd_cash) < 1e-12:
        return float("inf") if net > 0 else 0.0
    # use equity max drawdown dollars from curve if available elsewhere
    return float(net / (max_dd * max(starting, final, 1e-9)))


def run_portfolio_scaled(
    order: np.ndarray,
    p: Paths,
    sim: dict[str, np.ndarray],
    lot_mult: np.ndarray,
    *,
    starting: float = STARTING,
) -> dict[str, Any]:
    """Same portfolio rules; lot = FIXED_LOT * lot_mult[i] (risk scale)."""
    ts_ns = pd.DatetimeIndex(p.ts).as_unit("ns").asi8
    day_arr = ts_ns // 86_400_000_000_000
    hold = sim["holding_bars"]
    exit_ns = ts_ns + hold * 3_600_000_000_000
    net = sim["net_return"]
    entry = p.entry
    atr = p.atr

    equity = float(starting)
    day = -1
    day_pnl = 0.0
    open_exit = -1
    open_margin = 0.0
    blown = False
    taken: list[int] = []
    pnls: list[float] = []
    eq_curve: list[float] = []
    mults: list[float] = []

    for i in order:
        t = ts_ns[i]
        d = day_arr[i]
        if day != d:
            day = d
            day_pnl = 0.0
        if open_exit != -1 and open_exit <= t:
            open_exit = -1
            open_margin = 0.0
        if blown or equity <= 0:
            blown = True
            continue
        if open_exit != -1:
            continue
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            if equity <= 0:
                blown = True
            continue
        m = float(max(lot_mult[i], 0.0))
        lots = FIXED_LOT * m
        if lots <= 0:
            continue
        margin = required_margin(lots, entry[i], LEVERAGE)
        if equity - open_margin < margin:
            continue
        raw_pnl = lots * CONTRACT_SIZE * entry[i] * net[i]
        one_r_loss = lots * CONTRACT_SIZE * (SL_ATR * atr[i])
        stop_line = margin * STOP_OUT_LEVEL
        if equity - one_r_loss <= stop_line:
            pnl = -(equity - stop_line)
            equity = max(0.0, stop_line)
            if equity <= 0:
                blown = True
        elif equity + raw_pnl <= 0:
            pnl = -equity
            equity = 0.0
            blown = True
        else:
            pnl = raw_pnl
            equity += pnl
        day_pnl += pnl
        open_exit = int(exit_ns[i])
        open_margin = margin
        taken.append(int(i))
        pnls.append(float(pnl))
        eq_curve.append(float(equity))
        mults.append(m)

    if not taken:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_return": 0.0,
            "max_drawdown": 0.0, "final_equity": starting, "taken": np.array([], dtype=int),
            "pnl": np.array([]), "equity": np.array([]), "blown": blown, "sharpe": 0.0,
            "mar": 0.0, "ulcer": 0.0, "recovery_factor": 0.0, "mults": np.array([]),
        }
    pnl_a = np.asarray(pnls)
    eq = np.asarray(eq_curve)
    dd = max_dd_from_equity(eq, starting)
    tr = float(eq[-1] / starting - 1.0)
    return {
        "n_trades": len(taken),
        "win_rate": float(np.mean(pnl_a > 0)),
        "profit_factor": _pf(pnl_a),
        "total_return": tr,
        "max_drawdown": dd,
        "final_equity": float(eq[-1]),
        "taken": np.asarray(taken, dtype=int),
        "pnl": pnl_a,
        "equity": eq,
        "blown": blown,
        "sharpe": _sharpe(eq, starting),
        "mar": _mar(tr, dd),
        "ulcer": _ulcer(eq, starting),
        "recovery_factor": float((eq[-1] - starting) / max(dd * np.max(np.concatenate([[starting], eq])), 1e-9)),
        "mults": np.asarray(mults),
    }


def enrich_vol(panel: pd.DataFrame, mkt: dict) -> pd.DataFrame:
    """Attach ATR diagnostics + clusters on entry bars (derived from market ATR; not new FE library)."""
    d = panel.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    ts = pd.DatetimeIndex(mkt["ts"])
    atr_s = pd.Series(mkt["atr"], index=ts, dtype=float)
    close_s = pd.Series(mkt["close"], index=ts, dtype=float)
    atr_ratio = atr_s / close_s.replace(0, np.nan)
    roll_mean = atr_s.rolling(100, min_periods=20).mean()
    roll_std = atr_s.rolling(100, min_periods=20).std()
    atr_ratio_roll_mean = atr_ratio.rolling(252, min_periods=50).mean()
    atr_ratio_roll_std = atr_ratio.rolling(252, min_periods=50).std()
    # rolling percentile of atr_ratio (last point in window)
    def _last_pct(x: np.ndarray) -> float:
        if len(x) < 5:
            return float("nan")
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    atr_pctile = atr_ratio.rolling(252, min_periods=50).apply(_last_pct, raw=True)
    atr_z = (atr_ratio - atr_ratio_roll_mean) / atr_ratio_roll_std.replace(0, np.nan)
    vol_exp = atr_ratio / (roll_mean / close_s.replace(0, np.nan)).replace(0, np.nan)

    feat = pd.DataFrame(
        {
            "timestamp": ts,
            "atr_ratio": atr_ratio.to_numpy(),
            "atr_roll_mean": roll_mean.to_numpy(),
            "atr_roll_std": roll_std.to_numpy(),
            "atr_percentile": atr_pctile.to_numpy(),
            "atr_zscore": atr_z.to_numpy(),
            "vol_expansion_ratio": vol_exp.to_numpy(),
        }
    )
    d = d.merge(feat, on="timestamp", how="left")
    if "atr_percentile_252" in d.columns:
        d["atr_pct"] = pd.to_numeric(d["atr_percentile_252"], errors="coerce")
    else:
        d["atr_pct"] = pd.to_numeric(d["atr_percentile"], errors="coerce")
    d["atr_pct"] = d["atr_pct"].fillna(d["atr_percentile"])
    bins = [-0.01, 0.20, 0.40, 0.60, 0.80, 1.01]
    d["vol_cluster"] = pd.cut(d["atr_pct"].clip(0, 1), bins=bins, labels=CLUSTERS)
    d["year"] = d["timestamp"].dt.year
    # entry atr price alias
    if "atr_price" not in d.columns and "atr_entry" in d.columns:
        d["atr_price"] = d["atr_entry"]
    return d


def iter1_atr_dist(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = ["atr_price", "atr_pct", "atr_zscore", "atr_ratio", "atr_roll_mean", "atr_roll_std", "vol_expansion_ratio"]
    rows = []
    for y, g in d.groupby("year"):
        for c in cols:
            if c not in g.columns:
                continue
            v = g[c].astype(float).to_numpy()
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            rows.append({
                "year": int(y), "metric": c, "n": len(v),
                "mean": float(v.mean()), "median": float(np.median(v)),
                "std": float(v.std()), "p10": float(np.quantile(v, 0.1)),
                "p50": float(np.quantile(v, 0.5)), "p90": float(np.quantile(v, 0.9)),
                "p95": float(np.quantile(v, 0.95)),
            })
    dist = pd.DataFrame(rows)
    dist.to_csv(out / "atr_distribution.csv", index=False)
    # natural thresholds from 2021-25 p90/p95 of atr_pct
    base = d[d["year"] < 2026]
    thr = {
        "atr_pct_p70_2021_25": float(base["atr_pct"].quantile(0.70)),
        "atr_pct_p80_2021_25": float(base["atr_pct"].quantile(0.80)),
        "atr_pct_p90_2021_25": float(base["atr_pct"].quantile(0.90)),
        "atr_pct_p95_2021_25": float(base["atr_pct"].quantile(0.95)),
        "vol_exp_p90_2021_25": float(base["vol_expansion_ratio"].quantile(0.90)) if "vol_expansion_ratio" in base else float("nan"),
    }
    pd.DataFrame([thr]).to_csv(out / "atr_natural_thresholds.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for period, mask in (("2021-25", d["year"] < 2026), ("2026", d["year"] == 2026)):
        ax.hist(d.loc[mask, "atr_pct"].dropna(), bins=30, range=(0, 1), density=True, alpha=0.45, label=period)
    ax.set_title("ATR percentile at entry"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "chart_atr_pct_hist.png", dpi=120); plt.close(fig)
    return dist


def iter2_clusters(d: pd.DataFrame, sim: dict, order: np.ndarray, p: Paths, out: Path) -> pd.DataFrame:
    # stats on realized outcomes if trade taken — use full sim net_return per entry
    rows = []
    for cl in CLUSTERS:
        idx = np.where(d["vol_cluster"].astype(str).to_numpy() == cl)[0]
        if len(idx) == 0:
            continue
        # portfolio restricted to this cluster only (measurement)
        mask = np.zeros(p.n, dtype=float)
        mask[idx] = 1.0
        port = run_portfolio_scaled(order, p, sim, mask)
        nr = sim["net_return"][idx]
        rows.append({
            "vol_cluster": cl,
            "n_entries": int(len(idx)),
            "n_trades": port["n_trades"],
            "pf": port["profit_factor"] if np.isfinite(port["profit_factor"]) else None,
            "wr": port["win_rate"],
            "expectancy_net": float(nr.mean()),
            "dd": port["max_drawdown"],
            "total_return": port["total_return"],
            "mean_atr_pct": float(d.iloc[idx]["atr_pct"].mean()),
        })
    clus = pd.DataFrame(rows)
    clus.to_csv(out / "volatility_cluster.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(clus["vol_cluster"], clus["pf"].fillna(0))
    ax.set_title("PF by volatility cluster (cluster-only portfolio)"); ax.set_ylabel("PF")
    fig.tight_layout(); fig.savefig(out / "chart_cluster_pf.png", dpi=120); plt.close(fig)
    return clus


def iter3_transitions(d: pd.DataFrame, sim: dict, out: Path) -> pd.DataFrame:
    # bar-order transitions on entries
    cl = d["vol_cluster"].astype(str).to_numpy()
    nr = sim["net_return"]
    # transition matrix counts
    states = list(CLUSTERS)
    mat = pd.DataFrame(0.0, index=states, columns=states)
    post_exp = []
    for i in range(1, len(cl)):
        a, b = cl[i - 1], cl[i]
        if a in states and b in states:
            mat.loc[a, b] += 1
            # expectancy on trade AFTER transition
            post_exp.append({"from": a, "to": b, "net_return": float(nr[i]), "year": int(d.iloc[i]["year"])})
    # probabilities
    prob = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
    mat.to_csv(out / "volatility_transition_counts.csv")
    prob.to_csv(out / "volatility_transition.csv")
    pe = pd.DataFrame(post_exp)
    if not pe.empty:
        summ = pe.groupby(["from", "to"], as_index=False).agg(
            n=("net_return", "size"), mean_expectancy=("net_return", "mean"),
            wr=("net_return", lambda s: float((s > 0).mean())),
        )
        summ.to_csv(out / "volatility_transition_expectancy.csv", index=False)
    else:
        summ = pd.DataFrame()
    return summ


def iter4_risk_sensitivity(d: pd.DataFrame, sim: dict, order: np.ndarray, p: Paths, out: Path) -> pd.DataFrame:
    rows = []
    for cl in CLUSTERS:
        idx = set(np.where(d["vol_cluster"].astype(str).to_numpy() == cl)[0].tolist())
        for r in RISK_LEVELS:
            mult = np.ones(p.n)
            # scale only this cluster; others stay 1.0 (partial policy probe)
            for i in idx:
                mult[i] = r
            # also: scale ALL by r for global sensitivity
            port_cl = run_portfolio_scaled(order, p, sim, mult)
            rows.append({
                "mode": "scale_cluster_only", "vol_cluster": cl, "risk_mult": r,
                "n_trades": port_cl["n_trades"], "pf": port_cl["profit_factor"],
                "dd": port_cl["max_drawdown"], "ret": port_cl["total_return"],
                "sharpe": port_cl["sharpe"], "mar": port_cl["mar"],
                "ulcer": port_cl["ulcer"], "recovery_factor": port_cl["recovery_factor"],
                "blown": port_cl["blown"],
            })
    for r in RISK_LEVELS:
        mult = np.full(p.n, r)
        port = run_portfolio_scaled(order, p, sim, mult)
        rows.append({
            "mode": "scale_all", "vol_cluster": "ALL", "risk_mult": r,
            "n_trades": port["n_trades"], "pf": port["profit_factor"],
            "dd": port["max_drawdown"], "ret": port["total_return"],
            "sharpe": port["sharpe"], "mar": port["mar"],
            "ulcer": port["ulcer"], "recovery_factor": port["recovery_factor"],
            "blown": port["blown"],
        })
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "risk_simulation.csv", index=False)
    return sens


def mapping_from_edges(edges: list[float], risks: list[float]) -> Callable[[np.ndarray], np.ndarray]:
    edges = list(edges)
    risks = list(risks)

    def fn(pct: np.ndarray) -> np.ndarray:
        out = np.full(len(pct), risks[-1], dtype=float)
        # edges define upper bounds
        prev = -0.01
        for hi, rk in zip(edges, risks[:-1]):
            out[(pct > prev) & (pct <= hi)] = rk
            prev = hi
        out[pct > edges[-1]] = risks[-1]
        return out

    return fn


def iter5_candidates() -> list[dict[str, Any]]:
    """Fixed candidate mappings — measurement only, not optimized."""
    cands = []
    # example from brief
    cands.append({
        "name": "map_brief_default",
        "edges": [0.30, 0.60, 0.80, 0.90],
        "risks": [1.0, 0.80, 0.60, 0.40, 0.20],
    })
    cands.append({
        "name": "map_conservative",
        "edges": [0.30, 0.60, 0.80, 0.90],
        "risks": [1.0, 0.70, 0.40, 0.25, 0.10],
    })
    cands.append({
        "name": "map_mild",
        "edges": [0.40, 0.70, 0.85, 0.95],
        "risks": [1.0, 0.85, 0.70, 0.50, 0.30],
    })
    cands.append({
        "name": "map_step_high_only",
        "edges": [0.80, 0.90],
        "risks": [1.0, 0.50, 0.25],
    })
    cands.append({
        "name": "map_cut_p90",
        "edges": [0.90],
        "risks": [1.0, 0.25],
    })
    cands.append({
        "name": "map_cut_p80",
        "edges": [0.80],
        "risks": [1.0, 0.40],
    })
    cands.append({
        "name": "fixed_100",
        "edges": [1.0],
        "risks": [1.0, 1.0],
    })
    return cands


def apply_map(d: pd.DataFrame, cand: dict[str, Any]) -> np.ndarray:
    fn = mapping_from_edges(cand["edges"], cand["risks"])
    return fn(d["atr_pct"].astype(float).fillna(0.5).to_numpy())


def iter6_montecarlo(
    base_pnl: np.ndarray,
    dyn_pnl: np.ndarray,
    base_blown: bool,
    *,
    starting: float = STARTING,
) -> pd.DataFrame:
    """Bootstrap trade PnL sequences (order shuffle) → ruin / DD distribution."""
    rows = []
    for label, pnl0 in (("fixed", base_pnl), ("dynamic", dyn_pnl)):
        if len(pnl0) < 5:
            continue
        ruins = 0
        maxdds = []
        finals = []
        for _ in range(N_MC):
            seq = RNG.permutation(pnl0)
            eq = starting
            peak = starting
            mdd = 0.0
            blown = False
            for x in seq:
                eq += x
                if eq <= 0:
                    eq = 0.0
                    blown = True
                    break
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
            if blown:
                ruins += 1
            maxdds.append(mdd)
            finals.append(eq)
        rows.append({
            "policy": label,
            "n_mc": N_MC,
            "prob_ruin": ruins / N_MC,
            "median_dd": float(np.median(maxdds)),
            "worst_dd": float(np.max(maxdds)),
            "p95_dd": float(np.quantile(maxdds, 0.95)),
            "median_final": float(np.median(finals)),
            "equity_vol": float(np.std(finals) / max(starting, 1)),
        })
    return pd.DataFrame(rows)


def yearly_ports(d: pd.DataFrame, sim: dict, p: Paths, mult: np.ndarray) -> dict[int, dict]:
    out = {}
    years = sorted(d["year"].unique())
    for y in years:
        idx = np.where(d["year"].to_numpy() == y)[0]
        order = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
        out[int(y)] = run_portfolio_scaled(order, p, sim, mult, starting=STARTING)
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Sprint 35 Adaptive Vol Risk ===")
    print("Building identical 7-feat entries…")
    panel = build_entry_panel()
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    # merge dataset atr_percentile_252
    long_df = _load_side("long")
    short_df = _load_side("short")
    feat_parts = []
    for side, df in (("long", long_df), ("short", short_df)):
        cols = ["timestamp"] + [c for c in ("atr_percentile_252", "atr_percent", "rolling_volatility") if c in df.columns]
        g = df[cols].copy()
        g["side"] = side
        feat_parts.append(g)
    feats = pd.concat(feat_parts, ignore_index=True)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.merge(feats, on=["timestamp", "side"], how="left")

    p = Paths(panel, mkt)
    d = enrich_vol(p.panel, mkt)
    # align atr_pct on Paths panel index
    assert len(d) == p.n
    sim = simulate_combo(p, **EXIT_KW)
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    print(f"entries={p.n}")

    print("\nITER 1 ATR regime…")
    dist = iter1_atr_dist(d, OUT)
    print(dist[dist["metric"] == "atr_pct"].to_string(index=False))

    print("\nITER 2 clusters…")
    clus = iter2_clusters(d, sim, order, p, OUT)
    print(clus.to_string(index=False))

    print("\nITER 3 transitions…")
    trans = iter3_transitions(d, sim, OUT)
    if not trans.empty:
        print(trans.sort_values("mean_expectancy").head(8).to_string(index=False))

    print("\nITER 4 risk sensitivity…")
    sens = iter4_risk_sensitivity(d, sim, order, p, OUT)
    print(sens[sens["mode"] == "scale_all"].to_string(index=False))

    print("\nITER 5–8 candidates + MC + robustness…")
    cands = iter5_candidates()
    baseline = run_portfolio_scaled(order, p, sim, np.ones(p.n))
    base_yearly = yearly_ports(d, sim, p, np.ones(p.n))

    policy_rows = []
    robust_rows = []
    mc_rows = []
    adapt_rows = []

    for cand in cands:
        mult = apply_map(d, cand)
        port = run_portfolio_scaled(order, p, sim, mult)
        yports = yearly_ports(d, sim, p, mult)
        # survival vs baseline on same taken set length — MC on pnl vectors
        mc = iter6_montecarlo(baseline["pnl"], port["pnl"], baseline["blown"])
        mc["policy_name"] = cand["name"]
        mc_rows.append(mc)

        # robustness gates
        dd_ok = port["max_drawdown"] <= baseline["max_drawdown"] + 0.01
        pf_ok = (port["profit_factor"] >= baseline["profit_factor"] * 0.95) if np.isfinite(port["profit_factor"]) else False
        # yearly stability: PF not collapsing vs baseline each year (except allow 2026 rescue)
        year_ok = True
        for y, yp in yports.items():
            bp = base_yearly[y]
            robust_rows.append({
                "policy": cand["name"], "year": y,
                "n": yp["n_trades"], "pf": yp["profit_factor"], "dd": yp["max_drawdown"],
                "ret": yp["total_return"], "blown": yp["blown"],
                "base_pf": bp["profit_factor"], "base_dd": bp["max_drawdown"], "base_ret": bp["total_return"],
                "base_blown": bp["blown"],
            })
            if y < 2026 and yp["max_drawdown"] > bp["max_drawdown"] + 0.05:
                year_ok = False
            if y < 2026 and np.isfinite(bp["profit_factor"]) and yp["profit_factor"] < bp["profit_factor"] * 0.85:
                year_ok = False
        # 2026 help: higher ret or not blown when baseline blown, or lower dd
        b26 = base_yearly.get(2026, {})
        y26 = yports.get(2026, {})
        help_2026 = False
        if y26:
            help_2026 = (
                (y26.get("total_return", -1) > b26.get("total_return", -1) + 0.05)
                or (b26.get("blown") and not y26.get("blown"))
                or (y26.get("max_drawdown", 1) < b26.get("max_drawdown", 1) - 0.05)
            )

        mc_fix = mc[mc["policy"] == "fixed"].iloc[0] if (mc["policy"] == "fixed").any() else None
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if (mc["policy"] == "dynamic").any() else None
        ruin_down = bool(mc_dyn is not None and mc_fix is not None and mc_dyn["prob_ruin"] < mc_fix["prob_ruin"] - 0.01)

        passes = bool(dd_ok and pf_ok and year_ok and (help_2026 or ruin_down or port["max_drawdown"] < baseline["max_drawdown"] - 0.02))

        policy_rows.append({
            "policy": cand["name"],
            "edges": str(cand["edges"]),
            "risks": str(cand["risks"]),
            "n_trades": port["n_trades"],
            "pf": port["profit_factor"],
            "dd": port["max_drawdown"],
            "ret": port["total_return"],
            "sharpe": port["sharpe"],
            "mar": port["mar"],
            "ulcer": port["ulcer"],
            "blown": port["blown"],
            "base_pf": baseline["profit_factor"],
            "base_dd": baseline["max_drawdown"],
            "base_ret": baseline["total_return"],
            "dd_ok": dd_ok,
            "pf_ok": pf_ok,
            "year_ok": year_ok,
            "help_2026": help_2026,
            "ruin_down": ruin_down,
            "passes_success": passes,
            "mean_risk_mult": float(mult.mean()),
            "frac_risk_lt_1": float(np.mean(mult < 0.999)),
        })
        adapt_rows.append({
            "policy": cand["name"],
            "mapping": f"atr_pct edges={cand['edges']} risks={cand['risks']}",
            "passes": passes,
        })

    policies = pd.DataFrame(policy_rows).sort_values(["passes_success", "dd", "pf"], ascending=[False, True, False])
    policies.to_csv(out_path := OUT / "adaptive_policy.csv", index=False)
    policies.to_csv(OUT / "adaptive_risk_candidates.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(OUT / "robustness_test.csv", index=False)
    pd.concat(mc_rows, ignore_index=True).to_csv(OUT / "montecarlo_survival.csv", index=False)
    pd.DataFrame(adapt_rows).to_csv(OUT / "adaptive_policy_index.csv", index=False)

    print(policies[["policy", "pf", "dd", "ret", "passes_success", "help_2026", "ruin_down"]].to_string(index=False))

    # recommend best passer else best DD improvement without wrecking PF
    passed = policies[policies["passes_success"] == True]
    if not passed.empty:
        best = passed.iloc[0]
        stop_reason = "SUCCESS: at least one candidate meets criteria"
    else:
        # soft pick: lowest DD among pf_ok
        soft = policies[policies["pf_ok"] == True].sort_values("dd")
        best = soft.iloc[0] if not soft.empty else policies.iloc[0]
        stop_reason = "STOP: no candidate fully meets success criteria — report failures"

    write_summary(OUT, dist, clus, sens, policies, pd.concat(mc_rows, ignore_index=True), best, baseline, stop_reason, base_yearly)
    print(f"\n{stop_reason}")
    print(f"DONE -> {OUT / 'summary.md'}")
    return 0


def write_summary(out, dist, clus, sens, policies, mc, best, baseline, stop_reason, base_yearly):
    lines = [
        "# Sprint 35 — Adaptive Volatility Risk Summary",
        "",
        "Fixed entry (7-feat) + exit `a0.25_d0.08`. Measurement only — lot multiplier by ATR percentile.",
        "",
        f"**Baseline fixed 100% risk:** PF={baseline['profit_factor']:.3f} DD={baseline['max_drawdown']:.1%} "
        f"ret={baseline['total_return']:+.1%} trades={baseline['n_trades']} blown={baseline['blown']}",
        "",
        f"**Stop/Success:** {stop_reason}",
        "",
        "## 1. Apakah volatility penyebab utama DD?",
        clus.to_string(index=False) if clus is not None else "",
        "",
        "Lihat PF/DD per cluster. Jika EXTREME/HIGH punya expectancy lebih jelek / DD lebih tinggi → vol terkait risiko.",
        "",
        "## 2–3. Threshold alami (2021–25)",
        open(out / "atr_natural_thresholds.csv", encoding="utf-8").read() if (out / "atr_natural_thresholds.csv").exists() else "",
        "",
        "## 4. Risk mapping terbaik (candidate)",
        best.to_string() if hasattr(best, "to_string") else str(best),
        "",
        "## Policies ranked",
        policies[["policy", "pf", "dd", "ret", "mean_risk_mult", "passes_success", "help_2026", "ruin_down", "dd_ok", "pf_ok"]].to_string(index=False),
        "",
        "## 5. Trade yang diperkecil",
        "Entries dengan ATR percentile di bucket tinggi (mapping edges) — lihat policy risks[].",
        "",
        "## 6–9. Dynamic sizing / ruin / DD / return",
        mc.to_string(index=False) if mc is not None and not mc.empty else "",
        "",
        f"Recommended policy: **{best['policy']}** — PF {best['pf']:.3f} (base {best['base_pf']:.3f}), "
        f"DD {best['dd']:.1%} (base {best['base_dd']:.1%}), ret {best['ret']:+.1%} (base {best['base_ret']:+.1%}).",
        "",
        "## 10. Adaptive Risk Policy rekomendasi",
        f"- Mapping: edges={best['edges']} risks={best['risks']}",
        "- Implementasi production: kalikan FIXED_LOT dengan risk_mult(atr_percentile) **tanpa** ubah entry/exit.",
        "- Jika `passes_success=False`: **jangan ship** — gunakan hanya sebagai hipotesis; prioritas tetap equity guard.",
        "",
        "## Files",
        "atr_distribution.csv, volatility_cluster.csv, volatility_transition.csv, risk_simulation.csv,",
        "adaptive_risk_candidates.csv, adaptive_policy.csv, montecarlo_survival.csv, robustness_test.csv, summary.md",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
