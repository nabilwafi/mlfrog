"""Sprint 38 iter 2 — OOS continuation ranking (exit frozen).

  python apps/research_sprint38_continuation.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from simulation.wf.sim import _load_side, load_h1, prepare_market

PANEL = (
    _ROOT
    / "artifacts"
    / "pipeline_backtest"
    / "dynamic_exit"
    / "sprint38"
    / "entry_panel.parquet"
)
OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint38_continuation"
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
HORIZONS = (8, 16, 24, 48)
THRESH = (0.5, 1.0, 1.5, 2.0)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
STATE_FEATS = [
    "current_R",
    "bars_in_trade",
    "mfe_so_far",
    "mae_so_far",
    "dist_from_best_R",
    "y_prob",
    "is_long",
    "hour",
    "atr_pct_entry",
] + list(FEAT7)

LEAKAGE = [
    ("current_R", "path close at bar j", "bar j", "0", False),
    ("bars_in_trade", "j since entry", "bar j", "j", False),
    ("mfe_so_far", "max fav bars 1..j", "bar j", "j", False),
    ("mae_so_far", "max adv bars 1..j", "bar j", "j", False),
    ("dist_from_best_R", "mfe_so_far - current_R", "bar j", "j", False),
    ("y_prob", "frozen entry LGBM", "entry", "train<val<test", False),
    ("is_long", "entry side", "entry", "0", False),
    ("hour", "clock at bar j", "bar j", "0", False),
    ("atr_pct_entry", "atr_percentile_252 at entry", "entry", "252", False),
    *[
        (f, "FEAT7 at entry (frozen)", "entry", "lookback only", False)
        for f in FEAT7
    ],
    ("future_max_R_*", "max fav after j, minus current_R", "j+1..j+h", "h", True),
]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    if y.min() == y.max():
        return 0.5, float(y.mean()), float(np.mean((p - y) ** 2))
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
        float(brier_score_loss(y, p)),
    )


def _fit_predict(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "lr":
        sc = StandardScaler()
        X1 = sc.fit_transform(Xtr)
        X2 = sc.transform(Xte)
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(X1, ytr)
        return clf.predict_proba(X2)[:, 1]
    if model == "lgbm":
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.05, num_leaves=15,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
            verbosity=-1, class_weight="balanced",
        )
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    if model == "xgb":
        import xgboost as xgb

        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=80, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=spw, verbosity=0, n_jobs=1,
        )
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    if model == "cat":
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier(
            iterations=80, depth=4, learning_rate=0.05, verbose=False,
            auto_class_weights="Balanced",
        )
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    raise ValueError(model)


def build_states(p: Paths, sim: dict) -> pd.DataFrame:
    hold = np.asarray(sim["holding_bars"], dtype=int)
    years = p.year
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    if "atr_percentile_252" in p.panel.columns:
        atr_pct = p.panel["atr_percentile_252"].astype(float).to_numpy()
    else:
        atr_pct = np.full(p.n, 0.5)
    have_f7 = all(c in p.panel.columns for c in FEAT7)
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy() if have_f7 else None
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    rows = []
    for i in range(p.n):
        h = int(hold[i])
        last = int(p.last_off[i])
        for j in range(1, h + 1):
            cr = p.close_r[i, j]
            if not np.isfinite(cr):
                continue
            mfe = float(np.nanmax(p.fav[i, 1 : j + 1]))
            mae = float(np.nanmax(p.adv[i, 1 : j + 1]))
            rec = {
                "trade_i": i,
                "year": int(years[i]),
                "j": j,
                "current_R": float(cr),
                "bars_in_trade": j,
                "mfe_so_far": mfe,
                "mae_so_far": mae,
                "dist_from_best_R": mfe - float(cr),
                "y_prob": float(y_prob[i]),
                "is_long": int(p.is_long[i]),
                "hour": int((hour0[i] + j) % 24),
                "atr_pct_entry": float(atr_pct[i]) if np.isfinite(atr_pct[i]) else 0.5,
            }
            for k, name in enumerate(FEAT7):
                rec[name] = float(f7[i, k]) if f7 is not None else 0.0
            for H in HORIZONS:
                sl = p.fav[i, j + 1 : min(j + 1 + H, last + 1)]
                sl = sl[np.isfinite(sl)]
                rec[f"n_future_{H}"] = int(sl.size)
                rec[f"future_max_R_{H}"] = float(sl.max() - cr) if sl.size else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def eval_fold(pred: np.ndarray, y: np.ndarray, fut: np.ndarray) -> dict:
    roc, pr, brier = _auc(y, pred)
    sp = _spearman(pred, fut)
    q = pd.qcut(pred, 5, labels=False, duplicates="drop")
    bucket_mfe = []
    for b in sorted(set(q)):
        bucket_mfe.append(float(np.nanmean(fut[q == b])))
    mono = _spearman(np.arange(len(bucket_mfe), dtype=float), np.asarray(bucket_mfe))
    k = max(1, int(round(0.10 * len(pred))))
    order = np.argsort(pred)
    bot = float(np.nanmean(fut[order[:k]]))
    top = float(np.nanmean(fut[order[-k:]]))
    return {
        "n": int(len(pred)),
        "pos_rate": float(y.mean()),
        "roc": roc,
        "pr": pr,
        "brier": brier,
        "spearman": sp,
        "top10_mfe": top,
        "bot10_mfe": bot,
        "top_bot": (top / bot) if bot > 1e-9 else (float("inf") if top > 0 else 0.0),
        "mono": mono,
        "buckets": bucket_mfe,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    panel = pd.read_parquet(PANEL)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    if not all(c in panel.columns for c in FEAT7):
        print("join FEAT7 at entry timestamp (no future)")
        parts = []
        for side in ("long", "short"):
            d = _load_side(side)[["timestamp", *FEAT7]].copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d["side"] = side
            parts.append(d)
        feat = pd.concat(parts, ignore_index=True)
        panel = panel.merge(feat, on=["timestamp", "side"], how="left")
    p = Paths(panel, prepare_market(h1))
    sim = simulate_combo(p, **EXIT_KW)
    print("build position states")
    st = build_states(p, sim)
    st.to_parquet(OUT / "position_state_dataset.parquet", index=False)
    print(f"  n_states={len(st)} n_trades={st['trade_i'].nunique()}")

    pd.DataFrame(
        LEAKAGE, columns=["feature_name", "source", "timestamp", "lookback", "future_dependency"]
    ).to_csv(OUT / "feature_leakage_audit.csv", index=False)
    assert not any(x[4] for x in LEAKAGE if not str(x[0]).startswith("future_")), "feature leakage"

    Xall = st[STATE_FEATS].astype(float).fillna(0.0).to_numpy()
    years = st["year"].to_numpy()
    models = ("lr", "lgbm", "xgb", "cat")
    recs = []
    print("WF label x model")
    for H in HORIZONS:
        fut = st[f"future_max_R_{H}"].to_numpy()
        n_fut = st[f"n_future_{H}"].to_numpy()
        ok_h = np.isfinite(fut) & (n_fut >= max(4, H // 4))
        for thr in THRESH:
            y = (fut >= thr).astype(int)
            for model in models:
                oos_pred = np.full(len(st), np.nan)
                fold_rows = []
                for te in TEST_YEARS:
                    tr = (years < te) & ok_h
                    te_m = (years == te) & ok_h
                    if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
                        continue
                    pred = _fit_predict(model, Xall[tr], y[tr], Xall[te_m])
                    oos_pred[te_m] = pred
                    fr = eval_fold(pred, y[te_m], fut[te_m])
                    fr.update(year=te, model=model, horizon=H, thresh=thr)
                    fold_rows.append(fr)
                m = np.isfinite(oos_pred) & ok_h
                if m.sum() < 100 or not fold_rows:
                    continue
                ov = eval_fold(oos_pred[m], y[m], fut[m])
                n_mono = sum(1 for r in fold_rows if r["mono"] > 0.6)
                n_sep = sum(1 for r in fold_rows if r["top10_mfe"] > r["bot10_mfe"])
                recs.append(
                    {
                        "model": model,
                        "horizon": H,
                        "thresh": thr,
                        "label": f"future_max_R_{H}>={thr:g}",
                        **{f"oos_{k}": ov[k] for k in ("n", "pos_rate", "roc", "pr", "brier", "spearman", "top10_mfe", "bot10_mfe", "top_bot", "mono")},
                        "folds_mono": n_mono,
                        "folds_sep": n_sep,
                        "n_folds": len(fold_rows),
                        "mean_fold_roc": float(np.mean([r["roc"] for r in fold_rows])),
                        "mean_fold_sp": float(np.mean([r["spearman"] for r in fold_rows])),
                        "min_fold_sp": float(np.min([r["spearman"] for r in fold_rows])),
                    }
                )
                print(
                    f"  {model} h{H} t{thr:g}: roc={ov['roc']:.3f} sp={ov['spearman']:.3f} "
                    f"top/bot={ov['top_bot']:.2f} mono={ov['mono']:.2f}"
                )
    bench = pd.DataFrame(recs)
    bench.to_csv(OUT / "label_benchmark.csv", index=False)
    bench.to_csv(OUT / "model_benchmark.csv", index=False)

    # naive baselines on best horizon later; score by OOS spearman then top/bot then roc
    bench = bench.sort_values(["oos_spearman", "oos_top_bot", "oos_roc"], ascending=False)
    best = bench.iloc[0].to_dict() if len(bench) else {}
    go = False
    if best:
        go = (
            float(best["oos_spearman"]) >= 0.10
            and float(best["oos_top_bot"]) >= 1.40
            and int(best["folds_sep"]) >= 4
            and float(best["oos_mono"]) >= 0.60
            and float(best["min_fold_sp"]) > 0.0
        )
    verdict = "GO" if go else "STOP"
    yearly = []
    if best:
        H, thr, model = int(best["horizon"]), float(best["thresh"]), str(best["model"])
        fut = st[f"future_max_R_{H}"].to_numpy()
        n_fut = st[f"n_future_{H}"].to_numpy()
        ok_h = np.isfinite(fut) & (n_fut >= max(4, H // 4))
        y = (fut >= thr).astype(int)
        for te in TEST_YEARS:
            tr = (years < te) & ok_h
            te_m = (years == te) & ok_h
            if tr.sum() < 200 or te_m.sum() < 50:
                continue
            pred = _fit_predict(model, Xall[tr], y[tr], Xall[te_m])
            fr = eval_fold(pred, y[te_m], fut[te_m])
            fr.update(year=te)
            yearly.append(fr)
    pd.DataFrame(yearly).to_csv(OUT / "yearly_comparison.csv", index=False)

    lines = [
        "# Sprint 38 — Continuation Model (Iterasi 2)",
        "",
        "Frozen entry FEAT7 top 5%, trail a0.25/d0.08. Starting **$280** (not $80). "
        "Labels = additional future MFE from current bar. Features = position state + entry FEAT7 only.",
        "",
        f"States: {len(st)} from {st['trade_i'].nunique()} trades.",
        "",
        "## Leakage audit",
        "",
        "All STATE_FEATS have future_dependency=false. future_max_R_* is label-only.",
        "File: `feature_leakage_audit.csv`.",
        "",
        "## Best",
        "",
    ]
    if best:
        lines += [
            f"Model: **{best['model']}**",
            f"Label: **{best['label']}**",
            f"OOS ROC-AUC: {best['oos_roc']:.3f}",
            f"OOS PR-AUC: {best['oos_pr']:.3f}",
            f"OOS Spearman vs future MFE: {best['oos_spearman']:.3f}",
            f"Top10% future MFE: {best['oos_top10_mfe']:.3f}R",
            f"Bottom10% future MFE: {best['oos_bot10_mfe']:.3f}R",
            f"Top/Bot: {best['oos_top_bot']:.2f}",
            f"Quintile monotonicity (Spearman): {best['oos_mono']:.2f}",
            f"Folds with top>bot: {int(best['folds_sep'])}/{int(best['n_folds'])}",
            f"Min fold Spearman: {best['min_fold_sp']:.3f}",
            "",
        ]
    lines += ["## Yearly (best combo)", "", "| Year | n | ROC | Spearman | top10 MFE | bot10 MFE | top/bot | mono |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in yearly:
        lines.append(
            f"| {r['year']} | {r['n']} | {r['roc']:.3f} | {r['spearman']:.3f} | "
            f"{r['top10_mfe']:.3f} | {r['bot10_mfe']:.3f} | {r['top_bot']:.2f} | {r['mono']:.2f} |"
        )
    lines += [
        "",
        f"## Verdict: **{verdict}**",
        "",
        "GO if OOS Spearman>=0.10, top/bot>=1.40, mono>=0.60, top>bot in >=4 folds, min fold Spearman>0.",
        "",
        "If STOP: do not build Dynamic Exit. Keep production trail a0.25/d0.08.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"verdict": verdict, "best": best, "go": go}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"VERDICT {verdict}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
