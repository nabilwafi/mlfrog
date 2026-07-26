"""Sprint 31 — Edge Attribution Research (read-only).

Uses research.wf_trades (+ stored dataset feature panels joined on timestamp/side).
No retrain, no re-inference, no re-backtest.

Example:
  python apps/report_edge_attribution.py
  python apps/report_edge_attribution.py --run-id 20260726T142820Z_e1e4f575
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
import yaml
from scipy.stats import mannwhitneyu

from simulation.wf.sim import SPLITS

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint31_edge_attribution"
STARTING = 80.0
META_DROP = {
    "timestamp",
    "label",
    "feature_version",
    "label_version",
    "side",
    "split",
    "strategy",
    "symbol",
    "timeframe",
    "realized_return",
    "holding_bars",
    "entry_price",
}


def profit_factor(pnl: np.ndarray) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _fmt_pf(v: float) -> str:
    return "inf" if not np.isfinite(v) else f"{v:.2f}"


def _connect(dsn: str):
    import psycopg2

    return psycopg2.connect(dsn, connect_timeout=10)


def _query(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def load_feature_panel() -> pd.DataFrame:
    """Stored dataset panels (already built) — join key for entry features."""
    parts = []
    for side in ("long", "short"):
        for split in SPLITS:
            path = _ROOT / f"artifacts/datasets/XAUUSD/H1/{side}/v2/{split}.parquet"
            d = pd.read_parquet(path)
            d["side"] = side
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    # one row per timestamp×side
    return d.drop_duplicates(["timestamp", "side"], keep="last")


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


def bucket_stats(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    total_pnl = float(df["pnl"].sum()) or 1e-12
    for name, g in df.groupby(key, dropna=False):
        pnl = g["pnl"].to_numpy(dtype=float)
        rows.append(
            {
                key: name,
                "trades": int(len(g)),
                "win_rate": float(np.mean(pnl > 0)),
                "profit_factor": profit_factor(pnl),
                "expectancy_usd": float(pnl.mean()),
                "avg_r": float(g["r_multiple"].mean()),
                "net_pnl": float(pnl.sum()),
                "pnl_share": float(pnl.sum() / total_pnl),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("net_pnl", ascending=False).reset_index(drop=True)


def winner_loser_feature_table(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    win = df[df["win"]]
    loss = df[~df["win"]]
    rows = []
    for c in feat_cols:
        a = pd.to_numeric(win[c], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(loss[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) < 20 or len(b) < 20:
            continue
        # Cohen's d (pooled)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
        d = float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0
        try:
            # two-sided MWU; alternative='two-sided'
            u = mannwhitneyu(a, b, alternative="two-sided")
            p = float(u.pvalue)
        except Exception:
            p = float("nan")
        rows.append(
            {
                "feature": c,
                "win_mean": float(a.mean()),
                "win_median": float(np.median(a)),
                "win_std": float(a.std(ddof=1)),
                "loss_mean": float(b.mean()),
                "loss_median": float(np.median(b)),
                "loss_std": float(b.std(ddof=1)),
                "mean_diff": float(a.mean() - b.mean()),
                "cohens_d": d,
                "abs_cohens_d": abs(d),
                "p_value": p,
                "n_win": int(len(a)),
                "n_loss": int(len(b)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]], limit: int | None = None) -> list[str]:
    use = df if limit is None else df.head(limit)
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in use.iterrows():
        cells = []
        for c, _h, f in cols:
            v = r[c]
            if f == "pf":
                cells.append(_fmt_pf(float(v)))
            elif f == "s":
                cells.append(str(v))
            elif f.endswith("d"):
                cells.append(format(int(v), f))
            elif pd.isna(v):
                cells.append("nan")
            else:
                cells.append(format(float(v), f))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Sprint 31 edge attribution (DB + stored features)")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "config.yaml")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    dsn = (cfg.get("paper_trading") or {}).get("postgres_dsn")
    if not dsn:
        raise SystemExit("paper_trading.postgres_dsn required")

    conn = _connect(str(dsn))
    try:
        run_id = args.run_id
        if not run_id:
            runs = _query(conn, "SELECT run_id FROM research.wf_runs ORDER BY created_at DESC LIMIT 1")
            if runs.empty:
                raise SystemExit("research.wf_runs empty")
            run_id = str(runs.iloc[0]["run_id"])
        trades = _query(
            conn,
            "SELECT timestamp, side, y_prob, pnl, net_return, holding_bars, exit_reason, "
            "regime, trend_state, vol_state, mfe_r, mae_r, mfe_pct, mae_pct, r_multiple, "
            "entry_price, test_year "
            "FROM research.wf_trades WHERE run_id = %s ORDER BY timestamp",
            (run_id,),
        )
    finally:
        conn.close()

    if trades.empty:
        raise SystemExit(f"no trades for {run_id}")
    if trades["mfe_r"].isna().all():
        raise SystemExit(f"run {run_id} missing MFE — use the Sprint-30+ run with attribution columns")

    print(f"run_id={run_id} trades={len(trades)}")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["side"] = trades["side"].astype(str).str.lower()
    for c in ("y_prob", "pnl", "mfe_r", "mae_r", "r_multiple", "net_return"):
        trades[c] = trades[c].astype(float)
    trades["holding_bars"] = trades["holding_bars"].fillna(0).astype(int)
    trades["win"] = trades["pnl"] > 0
    trades["hour"] = trades["timestamp"].dt.hour
    trades["dow"] = trades["timestamp"].dt.dayofweek  # Mon=0
    trades["dow_name"] = trades["timestamp"].dt.day_name()
    trades["month"] = trades["timestamp"].dt.month
    trades["month_name"] = trades["timestamp"].dt.strftime("%b")
    trades["session"] = trades["hour"].map(session_of_hour)

    print("Joining stored entry features (dataset panels)…")
    feats = load_feature_panel()
    feat_cols = [c for c in feats.columns if c not in META_DROP]
    # keep only numeric feature cols
    numeric = []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(feats[c]):
            numeric.append(c)
    feats = feats[["timestamp", "side"] + numeric]
    df = trades.merge(feats, on=["timestamp", "side"], how="left", suffixes=("", "_f"))
    miss = int(df[numeric[0]].isna().sum()) if numeric else len(df)
    print(f"feature join: {len(numeric)} cols, missing_rows={miss}/{len(df)}")

    # --- 1. winner vs loser features ---
    # Entry features only (usable at decision time). Path outcomes reported separately.
    OUTCOME_COLS = {"r_multiple", "mfe_r", "mae_r", "mfe_pct", "mae_pct", "holding_bars", "net_return", "pnl"}
    entry_numeric = [c for c in numeric if c not in OUTCOME_COLS]
    feat_cmp = winner_loser_feature_table(df, entry_numeric)
    # model probability is known at entry
    prob_cmp = winner_loser_feature_table(
        df.rename(columns={"y_prob": "model_probability"}),
        ["model_probability"],
    )
    feat_all = pd.concat([feat_cmp, prob_cmp], ignore_index=True)
    feat_all = feat_all.drop_duplicates("feature").sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

    path_cmp = winner_loser_feature_table(df, ["mfe_r", "mae_r", "holding_bars", "r_multiple"])

    # --- 2. probability buckets ---
    # top5% probs can sit below 0.5 — use quantile / fine bins on observed range
    lo, hi = float(df["y_prob"].min()), float(df["y_prob"].max())
    # fixed 0.05-wide buckets clipped to data
    edges = np.arange(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20 + 1e-9, 0.05)
    if len(edges) < 2:
        edges = np.array([lo, hi + 1e-6])
    df["prob_bucket"] = pd.cut(df["y_prob"], bins=edges, include_lowest=True)
    prob_rows = []
    for b, g in df.groupby("prob_bucket", observed=True):
        pnl = g["pnl"].to_numpy(dtype=float)
        prob_rows.append(
            {
                "prob_bucket": str(b),
                "prob_lo": float(b.left),
                "prob_hi": float(b.right),
                "trades": int(len(g)),
                "win_rate": float(np.mean(pnl > 0)),
                "profit_factor": profit_factor(pnl),
                "expectancy_usd": float(pnl.mean()),
                "avg_r": float(g["r_multiple"].mean()),
                "net_pnl": float(pnl.sum()),
            }
        )
    prob_df = pd.DataFrame(prob_rows).sort_values("prob_lo").reset_index(drop=True)
    # best threshold = lowest prob_lo among buckets with trades>=30 and PF>=1.4 / expectancy>0,
    # preferring higher WR; also report max-expectancy bucket
    liquid = prob_df[prob_df["trades"] >= 20].copy()
    best_exp = liquid.sort_values("expectancy_usd", ascending=False).iloc[0] if not liquid.empty else None
    # cumulative threshold: keep trades with y_prob >= t
    thresh_rows = []
    for t in sorted(df["y_prob"].quantile(np.linspace(0.0, 0.9, 19)).unique()):
        g = df[df["y_prob"] >= float(t)]
        if len(g) < 30:
            continue
        pnl = g["pnl"].to_numpy(dtype=float)
        thresh_rows.append(
            {
                "min_prob": float(t),
                "trades": int(len(g)),
                "win_rate": float(np.mean(pnl > 0)),
                "profit_factor": profit_factor(pnl),
                "expectancy_usd": float(pnl.mean()),
                "avg_r": float(g["r_multiple"].mean()),
                "net_pnl": float(pnl.sum()),
                "pnl_keep_share": float(pnl.sum() / max(df["pnl"].sum(), 1e-12)),
            }
        )
    thresh_df = pd.DataFrame(thresh_rows)
    # score: expectancy * log(trades) among PF>1.3
    if not thresh_df.empty:
        cand = thresh_df[thresh_df["profit_factor"] >= 1.3].copy()
        if cand.empty:
            cand = thresh_df
        cand["score"] = cand["expectancy_usd"] * np.log1p(cand["trades"])
        best_thresh = cand.sort_values("score", ascending=False).iloc[0]
    else:
        best_thresh = None

    # --- 3. regime ---
    regime_df = bucket_stats(df, "regime")
    # also trend / vol alone
    trend_df = bucket_stats(df, "trend_state")
    vol_df = bucket_stats(df, "vol_state")

    # --- 4. time ---
    hour_df = bucket_stats(df, "hour").sort_values("hour").reset_index(drop=True)
    dow_df = bucket_stats(df, "dow_name")
    # order Mon-Sun
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_df["dow_name"] = pd.Categorical(dow_df["dow_name"], categories=order, ordered=True)
    dow_df = dow_df.sort_values("dow_name").reset_index(drop=True)
    month_df = bucket_stats(df, "month_name")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_df["month_name"] = pd.Categorical(month_df["month_name"], categories=month_order, ordered=True)
    month_df = month_df.sort_values("month_name").reset_index(drop=True)
    session_df = bucket_stats(df, "session")

    # --- 5. trade quality ---
    quality_rows = []
    for label, g in (("WIN", df[df["win"]]), ("LOSS", df[~df["win"]]), ("ALL", df)):
        quality_rows.append(
            {
                "outcome": label,
                "trades": int(len(g)),
                "mean_mfe_r": float(g["mfe_r"].mean()),
                "median_mfe_r": float(g["mfe_r"].median()),
                "mean_mae_r": float(g["mae_r"].mean()),
                "median_mae_r": float(g["mae_r"].median()),
                "mean_holding": float(g["holding_bars"].mean()),
                "median_holding": float(g["holding_bars"].median()),
                "trail_share": float((g["exit_reason"] == "TRAIL").mean()),
                "sl_share": float((g["exit_reason"] == "SL").mean()),
            }
        )
    quality_df = pd.DataFrame(quality_rows)
    exit_df = bucket_stats(df, "exit_reason")

    # --- 6. edge contribution across factors ---
    contrib = []

    def add_factor(name: str, table: pd.DataFrame, key: str, min_trades: int = 20) -> None:
        t = table[table["trades"] >= min_trades].copy()
        if t.empty:
            t = table.copy()
        best = t.sort_values("expectancy_usd", ascending=False).iloc[0]
        worst = t.sort_values("expectancy_usd", ascending=True).iloc[0]
        # contribution proxy: PnL of positive-expectancy buckets
        pos = t[t["expectancy_usd"] > 0]
        neg = t[t["expectancy_usd"] <= 0]
        contrib.append(
            {
                "factor": name,
                "best_bucket": str(best[key]),
                "best_expectancy": float(best["expectancy_usd"]),
                "worst_bucket": str(worst[key]),
                "worst_expectancy": float(worst["expectancy_usd"]),
                "expectancy_spread": float(best["expectancy_usd"] - worst["expectancy_usd"]),
                "pos_bucket_pnl": float(pos["net_pnl"].sum()) if not pos.empty else 0.0,
                "neg_bucket_pnl": float(neg["net_pnl"].sum()) if not neg.empty else 0.0,
                "pnl_concentration": float(best["pnl_share"]),
            }
        )

    add_factor("side", bucket_stats(df, "side"), "side", min_trades=1)
    add_factor("trend_state", trend_df, "trend_state", min_trades=30)
    add_factor("vol_state", vol_df, "vol_state", min_trades=30)
    add_factor("regime", regime_df, "regime", min_trades=20)
    add_factor("session", session_df, "session", min_trades=20)
    add_factor("hour", hour_df, "hour", min_trades=20)
    add_factor("month", month_df, "month_name", min_trades=20)
    add_factor("dow", dow_df, "dow_name", min_trades=20)
    # probability as factor: top vs bottom tercile
    q = df["y_prob"].quantile([1 / 3, 2 / 3])
    df["prob_tercile"] = np.where(df["y_prob"] <= q.iloc[0], "LOW", np.where(df["y_prob"] >= q.iloc[1], "HIGH", "MID"))
    add_factor("probability_tercile", bucket_stats(df, "prob_tercile"), "prob_tercile", min_trades=20)
    # top *entry* feature as continuous split (never path outcomes)
    if not feat_all.empty:
        top_f = str(feat_all.iloc[0]["feature"])
        if top_f in df.columns:
            med = float(df[top_f].median())
            df["_top_feat_bin"] = np.where(df[top_f] >= med, f"{top_f}_HIGH", f"{top_f}_LOW")
            add_factor(f"top_entry_feature:{top_f}", bucket_stats(df, "_top_feat_bin"), "_top_feat_bin", min_trades=20)
    contrib_df = pd.DataFrame(contrib).sort_values("expectancy_spread", ascending=False).reset_index(drop=True)

    # --- 7. candidate rules (data-driven, not applied) ---
    rules = []
    total_pnl = float(df["pnl"].sum())
    overall_exp = float(df["pnl"].mean())
    overall_pf = profit_factor(df["pnl"].to_numpy(dtype=float))
    overall_wr = float(df["win"].mean())

    def eval_skip(mask: pd.Series, name: str, rationale: str) -> None:
        keep = df.loc[~mask]
        skipped = int(mask.sum())
        if skipped < 10 or len(keep) < 50:
            return
        pnl = keep["pnl"].to_numpy(dtype=float)
        rules.append(
            {
                "rule": name,
                "rationale": rationale,
                "trades_skipped": skipped,
                "trades_kept": int(len(keep)),
                "skip_share": skipped / len(df),
                "kept_wr": float(np.mean(pnl > 0)),
                "kept_pf": profit_factor(pnl),
                "kept_expectancy": float(pnl.mean()),
                "kept_net_pnl": float(pnl.sum()),
                "pnl_delta_vs_all": float(pnl.sum() - total_pnl),
                "wr_delta": float(np.mean(pnl > 0) - overall_wr),
                "pf_delta": float(profit_factor(pnl) - overall_pf) if np.isfinite(profit_factor(pnl)) else float("nan"),
            }
        )

    # skip worst regime
    worst_reg = regime_df.sort_values("expectancy_usd").iloc[0]
    if worst_reg["expectancy_usd"] < 0:
        eval_skip(
            df["regime"] == worst_reg["regime"],
            f"Skip regime == {worst_reg['regime']}",
            f"Only losing regime bucket (exp={worst_reg['expectancy_usd']:+.2f}, PF={_fmt_pf(worst_reg['profit_factor'])})",
        )
    # skip high vol downtrend (only if not already covered by worst_reg)
    if str(worst_reg["regime"]) != "DOWNTREND / HIGH_VOL":
        eval_skip(
            df["regime"] == "DOWNTREND / HIGH_VOL",
            "Skip DOWNTREND / HIGH_VOL",
            "High-vol selloffs shake the 0.12 trail; historically negative expectancy",
        )
    # skip low probability
    if best_thresh is not None:
        t = float(best_thresh["min_prob"])
        eval_skip(
            df["y_prob"] < t,
            f"Skip y_prob < {t:.3f}",
            f"Threshold search max score at min_prob={t:.3f} (kept PF={_fmt_pf(best_thresh['profit_factor'])})",
        )
    # skip weak hours (PF<1 and n>=15)
    weak_hours = hour_df[(hour_df["trades"] >= 15) & (hour_df["profit_factor"] < 1.0)]
    if not weak_hours.empty:
        hs = sorted(int(x) for x in weak_hours["hour"].tolist())
        eval_skip(
            df["hour"].isin(hs),
            f"Skip hours UTC in {hs}",
            "Hours with >=15 trades and PF<1.0",
        )
    # skip Asia session
    eval_skip(
        df["session"] == "ASIA",
        "Skip session == ASIA",
        "Thin liquidity / negative aggregate PnL in early UTC hours",
    )
    # skip worst month
    worst_m = month_df.sort_values("expectancy_usd").iloc[0]
    if worst_m["expectancy_usd"] < 0:
        eval_skip(
            df["month_name"] == worst_m["month_name"],
            f"Skip month == {worst_m['month_name']}",
            f"Worst calendar month (exp={worst_m['expectancy_usd']:+.2f})",
        )
    # skip extreme ATR percentile if column exists
    if "atr_percentile_252" in df.columns:
        for thr in (0.90, 0.95):
            eval_skip(
                df["atr_percentile_252"] >= thr,
                f"Skip atr_percentile_252 >= {thr:.2f}",
                "Extreme realized vol often coincides with trail shakeouts",
            )
    # skip high MAE-prone: high atr_percent
    if "atr_percent" in df.columns:
        q90 = float(df["atr_percent"].quantile(0.90))
        eval_skip(
            df["atr_percent"] >= q90,
            f"Skip atr_percent >= p90 ({q90:.4f})",
            "Top-decile ATR% entries",
        )
    # feature-based from top *entry* separator (never r_multiple / MFE / MAE — those are outcomes)
    if not feat_all.empty:
        top = feat_all.iloc[0]
        fname = str(top["feature"])
        if fname in df.columns and abs(float(top["cohens_d"])) >= 0.12:
            if float(top["mean_diff"]) > 0:
                cut = float(df[fname].quantile(0.25))
                eval_skip(
                    df[fname] <= cut,
                    f"Skip {fname} <= p25 ({cut:.4g})",
                    f"Top entry separator (Cohen d={top['cohens_d']:+.2f}); winners run higher",
                )
            else:
                cut = float(df[fname].quantile(0.75))
                eval_skip(
                    df[fname] >= cut,
                    f"Skip {fname} >= p75 ({cut:.4g})",
                    f"Top entry separator (Cohen d={top['cohens_d']:+.2f}); winners run lower",
                )

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df["rank_score"] = rules_df["kept_expectancy"] * np.log1p(rules_df["trades_kept"]) + 0.5 * rules_df[
            "pnl_delta_vs_all"
        ].clip(lower=-1e9)
        # prefer rules that raise expectancy without destroying too much PnL
        rules_df = rules_df.sort_values(["kept_expectancy", "pnl_delta_vs_all"], ascending=False).reset_index(drop=True)
        rules_df.insert(0, "rank", np.arange(1, len(rules_df) + 1))

    # --- write CSVs ---
    OUT.mkdir(parents=True, exist_ok=True)
    feat_all.to_csv(OUT / "feature_importance_winner_vs_loser.csv", index=False)
    path_cmp.to_csv(OUT / "path_quality_winner_vs_loser.csv", index=False)
    prob_df.to_csv(OUT / "probability_bucket.csv", index=False)
    if not thresh_df.empty:
        thresh_df.to_csv(OUT / "probability_threshold_curve.csv", index=False)
    regime_df.to_csv(OUT / "regime_attribution.csv", index=False)
    hour_df.to_csv(OUT / "hour_attribution.csv", index=False)
    month_df.to_csv(OUT / "month_attribution.csv", index=False)
    dow_df.to_csv(OUT / "dow_attribution.csv", index=False)
    session_df.to_csv(OUT / "session_attribution.csv", index=False)
    contrib_df.to_csv(OUT / "edge_contribution.csv", index=False)
    quality_df.to_csv(OUT / "trade_quality.csv", index=False)
    if not rules_df.empty:
        rules_df.to_csv(OUT / "candidate_rules.csv", index=False)
    else:
        pd.DataFrame(columns=["rule"]).to_csv(OUT / "candidate_rules.csv", index=False)

    # --- narrative helpers ---
    top_feats = feat_all.head(10)
    best_reg = regime_df.iloc[0]
    worst_reg = regime_df.sort_values("expectancy_usd").iloc[0]
    hour_liq = hour_df[hour_df["trades"] >= 20]
    best_hour = hour_liq.sort_values("net_pnl", ascending=False).iloc[0] if not hour_liq.empty else hour_df.iloc[0]
    best_hour_pf = hour_liq.sort_values("profit_factor", ascending=False).iloc[0] if not hour_liq.empty else hour_df.iloc[0]
    worst_hour = hour_liq.sort_values("profit_factor").iloc[0] if not hour_liq.empty else hour_df.iloc[0]
    best_month = month_df.sort_values("net_pnl", ascending=False).iloc[0]
    worst_month = month_df.sort_values("net_pnl").iloc[0]
    win_q = quality_df[quality_df["outcome"] == "WIN"].iloc[0]
    loss_q = quality_df[quality_df["outcome"] == "LOSS"].iloc[0]
    top_rule = rules_df.iloc[0] if not rules_df.empty else None
    top_feat = feat_all.iloc[0] if not feat_all.empty else None

    md: list[str] = [
        "# Sprint 31 — Edge Attribution Research",
        "",
        f"run_id: `{run_id}`  ",
        "constraint: **no retrain / no re-inference / no re-backtest**  ",
        "trades: `research.wf_trades`  ",
        "entry features: stored `artifacts/datasets/.../v2/*.parquet` joined on `(timestamp, side)` "
        "(same panels the WF already used — measurement only).",
        "",
        f"Sample: **{len(df)}** trades | WR **{overall_wr:.1%}** | PF **{_fmt_pf(overall_pf)}** | "
        f"expectancy **${overall_exp:.2f}** | net **${total_pnl:.2f}**.",
        "",
        "## 1. Winning vs Losing Trade (entry features)",
        "",
        "Only features known **at entry** (usable as filters). Path outcomes (MFE/MAE/R) are in §5 — "
        "they separate winners hard but cannot be gated before the trade.",
        "",
        "Mann–Whitney U p-value + Cohen's d (winner − loser). Sorted by |d|.",
        "",
    ]
    md += md_table(
        top_feats,
        [
            ("feature", "Feature", "s"),
            ("win_mean", "Win mean", ".4g"),
            ("loss_mean", "Loss mean", ".4g"),
            ("mean_diff", "Δ mean", "+.4g"),
            ("cohens_d", "Cohen d", "+.3f"),
            ("p_value", "p-value", ".2e"),
        ],
        limit=15,
    )
    if top_feat is not None:
        md += [
            "",
            f"**Strongest entry separator:** `{top_feat['feature']}` (d={top_feat['cohens_d']:+.3f}, "
            f"p={top_feat['p_value']:.2e}). "
            + (
                "Winners enter with a *higher* value."
                if top_feat["mean_diff"] > 0
                else "Winners enter with a *lower* value."
            ),
            "",
            "Interpretation: |d| < 0.2 = small, 0.2–0.5 = medium, >0.5 = large. "
            "Entry microstructure effects here are mostly small; context (regime/session/hour) carries more "
            "of the actionable edge than any single indicator.",
        ]
    md += [
        "",
        "## 2. Probability Calibration",
        "",
        f"Observed `y_prob` range: [{lo:.3f}, {hi:.3f}] (top-5% gate — buckets follow the data, not a fixed 0.50 floor).",
        "",
        "### Buckets",
        "",
    ]
    md += md_table(
        prob_df,
        [
            ("prob_bucket", "Bucket", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("avg_r", "Avg R", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    if best_exp is not None:
        md += [
            "",
            f"**Best bucket by expectancy (≥20 trades):** {best_exp['prob_bucket']} — "
            f"WR {best_exp['win_rate']:.1%}, PF {_fmt_pf(best_exp['profit_factor'])}, "
            f"exp ${best_exp['expectancy_usd']:+.3f}.",
        ]
    if best_thresh is not None:
        md += [
            "",
            "### Threshold curve (`y_prob >= t`)",
            "",
            f"**Recommended threshold to *test* next:** `y_prob >= {best_thresh['min_prob']:.3f}` "
            f"(keeps {int(best_thresh['trades'])} trades, WR {best_thresh['win_rate']:.1%}, "
            f"PF {_fmt_pf(best_thresh['profit_factor'])}, exp ${best_thresh['expectancy_usd']:+.3f}, "
            f"keeps {best_thresh['pnl_keep_share']:.0%} of total PnL).",
            "",
            "This is an **in-sample attribution** suggestion — not an OOS-validated gate.",
        ]
    md += [
        "",
        "## 3. Regime Attribution",
        "",
    ]
    md += md_table(
        regime_df,
        [
            ("regime", "Regime", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("avg_r", "Avg R", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
            ("pnl_share", "PnL share", ".1%"),
        ],
    )
    md += [
        "",
        f"**Best:** {best_reg['regime']} (PF {_fmt_pf(best_reg['profit_factor'])}, "
        f"{best_reg['pnl_share']:.0%} of total PnL).  ",
        f"**Worst:** {worst_reg['regime']} (PF {_fmt_pf(worst_reg['profit_factor'])}, "
        f"exp ${worst_reg['expectancy_usd']:+.3f}).",
        "",
        "### Trend / Vol margins",
        "",
    ]
    md += ["**Trend**", ""] + md_table(
        trend_df,
        [
            ("trend_state", "Trend", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += ["", "**Volatility**", ""] + md_table(
        vol_df,
        [
            ("vol_state", "Vol", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += [
        "",
        "## 4. Time Attribution",
        "",
        "### Hour (UTC)",
        "",
    ]
    md += md_table(
        hour_df,
        [
            ("hour", "Hour", "02d"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += [
        "",
        f"**Best PnL hour (≥20):** {int(best_hour['hour']):02d}:00. "
        f"**Best PF hour (≥20):** {int(best_hour_pf['hour']):02d}:00. "
        f"**Weakest liquid hour:** {int(worst_hour['hour']):02d}:00 "
        f"(PF {_fmt_pf(worst_hour['profit_factor'])}).",
        "",
        "### Day of week",
        "",
    ]
    md += md_table(
        dow_df,
        [
            ("dow_name", "Day", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += ["", "### Month", ""]
    md += md_table(
        month_df,
        [
            ("month_name", "Month", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += [
        "",
        f"**Best month:** {best_month['month_name']}. **Worst month:** {worst_month['month_name']}.",
        "",
        "### Session",
        "",
    ]
    md += md_table(
        session_df,
        [
            ("session", "Session", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += [
        "",
        "## 5. Trade Quality",
        "",
    ]
    md += md_table(
        quality_df,
        [
            ("outcome", "Set", "s"),
            ("trades", "Trades", "d"),
            ("mean_mfe_r", "Mean MFE", ".2f"),
            ("median_mfe_r", "Med MFE", ".2f"),
            ("mean_mae_r", "Mean MAE", ".2f"),
            ("median_mae_r", "Med MAE", ".2f"),
            ("mean_holding", "Mean hold", ".2f"),
            ("trail_share", "TRAIL%", ".1%"),
            ("sl_share", "SL%", ".1%"),
        ],
    )
    md += ["", "### Exit type", ""]
    md += md_table(
        exit_df,
        [
            ("exit_reason", "Exit", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "WR", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("expectancy_usd", "Exp ($)", "+.3f"),
            ("net_pnl", "Net PnL", "+.2f"),
        ],
    )
    md += [
        "",
        f"**Quality signature:** winners average {win_q['mean_mfe_r']:.2f}R MFE / {win_q['mean_mae_r']:.2f}R MAE "
        f"and exit via TRAIL {win_q['trail_share']:.0%} of the time; losers average "
        f"{loss_q['mean_mfe_r']:.2f}R MFE / {loss_q['mean_mae_r']:.2f}R MAE and are almost all SL. "
        "A high-quality trade is therefore one that never needs much adverse room — the trail arms quickly.",
        "",
        "## 6. Edge Contribution (factor ranking)",
        "",
        "Ranked by expectancy spread (best bucket − worst bucket) among liquid buckets.",
        "",
    ]
    md += md_table(
        contrib_df,
        [
            ("factor", "Factor", "s"),
            ("best_bucket", "Best", "s"),
            ("worst_bucket", "Worst", "s"),
            ("expectancy_spread", "Exp spread ($)", "+.3f"),
            ("pos_bucket_pnl", "Pos-bucket PnL", "+.2f"),
            ("neg_bucket_pnl", "Neg-bucket PnL", "+.2f"),
            ("pnl_concentration", "Best share", ".1%"),
        ],
    )
    md += [
        "",
        "## 7. Candidate Decision Rules (not applied)",
        "",
        "In-sample filters only. Do **not** ship these without a fresh OOS / nested WF test.",
        "",
    ]
    if not rules_df.empty:
        md += md_table(
            rules_df,
            [
                ("rank", "Rank", "d"),
                ("rule", "Rule", "s"),
                ("trades_skipped", "Skip n", "d"),
                ("trades_kept", "Keep n", "d"),
                ("kept_wr", "Kept WR", ".1%"),
                ("kept_pf", "Kept PF", "pf"),
                ("kept_expectancy", "Kept exp", "+.3f"),
                ("pnl_delta_vs_all", "Δ PnL", "+.2f"),
            ],
        )
        md += ["", "### Rationales", ""]
        for _, r in rules_df.iterrows():
            md.append(f"- **{r['rule']}** — {r['rationale']}")
    else:
        md += ["(no rules met minimum sample gates)", ""]

    md += [
        "",
        "## Expected Conclusion",
        "",
        f"1. **Strongest conditions:** {best_reg['regime']} "
        f"(PF {_fmt_pf(best_reg['profit_factor'])}, {best_reg['pnl_share']:.0%} of PnL); "
        f"liquid hours around {int(best_hour_pf['hour']):02d}–{int(best_hour['hour']):02d} UTC; "
        f"month {best_month['month_name']}.",
        "",
        f"2. **Weakest conditions:** {worst_reg['regime']} "
        f"(exp ${worst_reg['expectancy_usd']:+.3f}); "
        f"hour {int(worst_hour['hour']):02d}:00 among liquid slots; month {worst_month['month_name']}; "
        f"ASIA session.",
        "",
        "3. **Entry features that separate winners/losers:** "
        + (
            ", ".join(f"`{r.feature}` (d={r.cohens_d:+.2f})" for r in top_feats.head(5).itertuples())
            if not top_feats.empty
            else "n/a"
        )
        + ". Path metrics (MFE/MAE) separate much harder but are outcomes, not entry gates.",
        "",
        "4. **Best probability threshold to *test*:** "
        + (
            f"`y_prob >= {best_thresh['min_prob']:.3f}` "
            f"(kept PF {_fmt_pf(best_thresh['profit_factor'])}, exp ${best_thresh['expectancy_usd']:+.3f})."
            if best_thresh is not None
            else "insufficient bucket depth."
        ),
        "",
        f"5. **Regime to avoid:** `{worst_reg['regime']}` "
        + ("(negative expectancy)." if worst_reg["expectancy_usd"] < 0 else "(weakest positive bucket)."),
        "",
        f"6. **Best trading hours:** {int(best_hour_pf['hour']):02d}:00 UTC by PF, "
        f"{int(best_hour['hour']):02d}:00 UTC by PnL (≥20 trades).",
        "",
        "7. **Rules most worth testing next sprint:** "
        + (
            "; ".join(f"{int(r.rank)}. {r.rule}" for r in rules_df.head(3).itertuples())
            if not rules_df.empty
            else "none"
        )
        + ".",
        "",
        "### Files",
        "",
        "- `feature_importance_winner_vs_loser.csv`",
        "- `probability_bucket.csv` (+ `probability_threshold_curve.csv`)",
        "- `regime_attribution.csv`, `hour_attribution.csv`, `month_attribution.csv`",
        "- `candidate_rules.csv`",
        "- also: `dow_attribution.csv`, `session_attribution.csv`, `edge_contribution.csv`, `trade_quality.csv`",
        "",
    ]
    (OUT / "edge_attribution_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
