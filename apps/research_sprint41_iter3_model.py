"""Sprint 41 Iter 3 — OOS differential model (train/val only; no OOS tuning).

Expanding WF on 2021-2026 panel (no 2015-2020 trades in this entry set):
  test 2023: train 2021 / val 2022
  test 2024: train 2021-22 / val 2023
  ...

  python apps/research_sprint41_iter3_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint41_iter1_diff import STATE_FEATS

ITER1 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter1_differential"
OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter3_model"
TARGETS = ("p1_better", "p2_better", "p3_better")
DIFF = {"p1_better": "diff_p1", "p2_better": "diff_p2", "p3_better": "diff_p3"}
FOLDS = (
    (2023, (2021,), 2022),
    (2024, (2021, 2022), 2023),
    (2025, (2021, 2022, 2023), 2024),
    (2026, (2021, 2022, 2023, 2024), 2025),
)


def _spearman(a, b) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _auc(y, p) -> tuple[float, float, float]:
    if y.min() == y.max():
        return 0.5, float(y.mean()), float(np.mean((p - y) ** 2))
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p)), float(brier_score_loss(y, p))


def _fit(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "lr":
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(sc.fit_transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
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
    raise ValueError(model)


def _eval(pred, y, diff) -> dict:
    roc, pr, brier = _auc(y, pred)
    k = max(1, int(round(0.20 * len(pred))))
    order = np.argsort(pred)
    bot, top = diff[order[:k]], diff[order[-k:]]
    return {
        "n": int(len(pred)),
        "pos_rate": float(y.mean()),
        "roc": roc,
        "pr": pr,
        "brier": brier,
        "spearman": _spearman(pred, diff),
        "top20_diff": float(np.mean(top)),
        "bot20_diff": float(np.mean(bot)),
        "top_minus_bot": float(np.mean(top) - np.mean(bot)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    d = ds.sort_values(["trade_id", "bars_in_trade"]).groupby("trade_id", as_index=False).tail(1)
    X = d[STATE_FEATS].astype(float).fillna(0.0).to_numpy()
    years = d["year"].to_numpy()

    recs = []
    fold_rows = []
    for tgt in TARGETS:
        y = d[tgt].astype(int).to_numpy()
        diff = d[DIFF[tgt]].to_numpy(dtype=float)
        for model in ("lr", "lgbm", "xgb"):
            oos_pred = np.full(len(d), np.nan)
            folds = []
            for te, tr_years, va in FOLDS:
                tr = np.isin(years, tr_years)
                va_m = years == va
                te_m = years == te
                if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
                    continue
                pred_va = _fit(model, X[tr], y[tr], X[va_m])
                pred_te = _fit(model, X[tr], y[tr], X[te_m])  # fit train only, ignore val for weights
                oos_pred[te_m] = pred_te
                fr = _eval(pred_te, y[te_m], diff[te_m])
                fr.update(year=te, model=model, target=tgt, val_spearman=_spearman(pred_va, diff[va_m]))
                folds.append(fr)
                fold_rows.append(fr)
            m = np.isfinite(oos_pred)
            if m.sum() < 100 or not folds:
                continue
            ov = _eval(oos_pred[m], y[m], diff[m])
            recs.append({
                "model": model,
                "target": tgt,
                **{f"oos_{k}": ov[k] for k in ("n", "pos_rate", "roc", "pr", "brier", "spearman", "top20_diff", "bot20_diff", "top_minus_bot")},
                "n_folds": len(folds),
                "mean_fold_roc": float(np.mean([r["roc"] for r in folds])),
                "mean_fold_sp": float(np.mean([r["spearman"] for r in folds])),
                "min_fold_sp": float(np.min([r["spearman"] for r in folds])),
                "min_fold_top_bot": float(np.min([r["top_minus_bot"] for r in folds])),
                "years_sp_pos": int(sum(r["spearman"] > 0 for r in folds)),
            })
            print(f"  {model} {tgt}: roc={ov['roc']:.3f} sp={ov['spearman']:.3f} top-bot={ov['top_minus_bot']:.3f}")

    folds_df = pd.DataFrame(fold_rows)
    bench = pd.DataFrame(recs)
    if len(folds_df) and len(bench):
        val_rank = folds_df.groupby(["model", "target"], as_index=False)["val_spearman"].mean()
        bench = bench.merge(val_rank, on=["model", "target"], how="left")
        bench = bench.sort_values(["val_spearman", "oos_spearman"], ascending=False)
    elif len(bench):
        bench = bench.sort_values(["oos_spearman"], ascending=False)
    bench.to_csv(OUT / "model_benchmark.csv", index=False)
    folds_df.to_csv(OUT / "fold_metrics.csv", index=False)

    go = False
    best = bench.iloc[0].to_dict() if len(bench) else {}
    if best:
        go = (
            float(best["oos_spearman"]) >= 0.10
            and float(best["oos_top_minus_bot"]) >= 0.05
            and int(best["years_sp_pos"]) >= 3
            and float(best["min_fold_sp"]) > 0.0
        )
        # path-mechanical: if only ranking MFE, still a stable OOS geometric fact
        if go and float(best["min_fold_top_bot"]) < 0:
            go = False
    verdict = "GO_ITER4" if go else "STOP"
    reasons = []
    if not best:
        reasons.append("no model")
    elif not go:
        if float(best.get("min_fold_sp", 0)) <= 0:
            reasons.append("min-fold Spearman <= 0")
        if int(best.get("years_sp_pos", 0)) < 3:
            reasons.append("Spearman not positive in enough folds")
        if float(best.get("oos_spearman", 0)) < 0.10:
            reasons.append("OOS Spearman < 0.10")
        if float(best.get("oos_top_minus_bot", 0)) < 0.05:
            reasons.append("top-bot differential too small")
        if float(best.get("min_fold_top_bot", 0)) < 0:
            reasons.append("a fold inverted top/bot")

    lines = [
        "# Sprint 41 Iter 3 — OOS differential model",
        "",
        "Train years only. Val year unused for weights (reported). No OOS tuning.",
        "One row per trade (last P0-alive bar). STATE_FEATS PIT only.",
        "",
        f"Best: **{best.get('model')}** / **{best.get('target')}**" if best else "No model.",
        "",
    ]
    if best:
        lines += [
            f"OOS ROC {best['oos_roc']:.3f}  PR {best['oos_pr']:.3f}  Spearman {best['oos_spearman']:.3f}",
            f"Top20-Bot20 mean diff {best['oos_top_minus_bot']:.3f}  min-fold Spearman {best['min_fold_sp']:.3f}",
            "",
        ]
    if len(bench):
        cols = [c for c in ("model", "target", "val_spearman", "oos_roc", "oos_pr", "oos_spearman", "oos_top_minus_bot", "min_fold_sp", "years_sp_pos") if c in bench.columns]
        show = bench[cols]
        lines += ["```", show.to_string(index=False), "```", ""]
    lines += [
        f"STOP reasons: {', '.join(reasons) if reasons else '(none)'}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Production exit unchanged.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 3,
                "verdict": verdict,
                "best": {k: best.get(k) for k in ("model", "target", "oos_roc", "oos_spearman", "oos_top_minus_bot", "min_fold_sp")} if best else {},
                "reasons": reasons,
                "next_iter": 4 if verdict == "GO_ITER4" else None,
                "production_exit": "a0.25_d0.08 unchanged",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"VERDICT {verdict} reasons={reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
