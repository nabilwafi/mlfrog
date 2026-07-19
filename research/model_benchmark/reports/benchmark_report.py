"""Markdown report answering Sprint-8 research questions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.model_benchmark.entities.benchmark_result import BenchmarkModelResult

BOOSTING = frozenset(
    {"hist_gradient_boosting", "xgboost", "lightgbm", "catboost"}
)
LINEAR = frozenset({"logistic_regression"})


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def build_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary[
        [
            "algorithm",
            "n_features",
            "mean_roc_auc",
            "std_roc_auc",
            "mean_pr_auc",
            "mean_f1",
            "mean_log_loss",
            "mean_calibration_error",
            "total_train_seconds",
            "total_infer_seconds",
            "proba_std",
        ]
    ].copy()
    out["rank_roc"] = out["mean_roc_auc"].rank(ascending=False, method="min")
    out["rank_generalization"] = (
        out["mean_roc_auc"].rank(ascending=False, method="min")
        + out["std_roc_auc"].rank(ascending=True, method="min")
    ) / 2.0
    out["rank_calibration"] = out["mean_calibration_error"].rank(ascending=True, method="min")
    out["rank_speed"] = out["total_train_seconds"].rank(ascending=True, method="min")
    out["rank_score"] = (
        out["rank_roc"] + out["rank_generalization"] + out["rank_calibration"]
    ) / 3.0
    return out.sort_values("rank_score").reset_index(drop=True)


class BenchmarkReportBuilder:
    def build(
        self,
        *,
        results: list[BenchmarkModelResult],
        summary: pd.DataFrame,
        rankings: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> str:
        by_algo = {r.algorithm: r for r in results}
        if summary.empty:
            return f"# Model Benchmark — {symbol} {timeframe} {side}\n\n_(no results)_\n"

        best_roc_row = summary.sort_values("mean_roc_auc", ascending=False).iloc[0]
        # Generalization: high mean ROC, low std across windows
        gen = summary.copy()
        gen["gen_score"] = gen["mean_roc_auc"] - gen["std_roc_auc"]
        best_gen_row = gen.sort_values("gen_score", ascending=False).iloc[0]
        fastest_row = summary.sort_values("total_train_seconds", ascending=True).iloc[0]
        best_cal_row = summary.sort_values("mean_calibration_error", ascending=True).iloc[0]
        # Best probability distribution: not collapsed (higher proba_std, mean near 0.3-0.7)
        dist = summary.copy()
        dist["dist_score"] = dist["proba_std"].fillna(0.0)
        best_dist_row = dist.sort_values("dist_score", ascending=False).iloc[0]

        prod_algo = str(rankings.iloc[0]["algorithm"]) if not rankings.empty else str(best_roc_row["algorithm"])
        prod = by_algo.get(prod_algo)

        boost_rocs = [
            float(r.mean_roc_auc)
            for a, r in by_algo.items()
            if a in BOOSTING and r.mean_roc_auc == r.mean_roc_auc
        ]
        linear_rocs = [
            float(r.mean_roc_auc)
            for a, r in by_algo.items()
            if a in LINEAR and r.mean_roc_auc == r.mean_roc_auc
        ]
        boost_mean = sum(boost_rocs) / len(boost_rocs) if boost_rocs else float("nan")
        linear_mean = sum(linear_rocs) / len(linear_rocs) if linear_rocs else float("nan")
        boost_beats = (
            boost_mean == boost_mean
            and linear_mean == linear_mean
            and (boost_mean - linear_mean) > 0.01
        )

        lgbm = by_algo.get("lightgbm")
        cat = by_algo.get("catboost")
        cat_beats = bool(
            lgbm
            and cat
            and cat.mean_roc_auc == cat.mean_roc_auc
            and lgbm.mean_roc_auc == lgbm.mean_roc_auc
            and cat.mean_roc_auc > lgbm.mean_roc_auc + 0.005
        )

        best_roc = float(best_roc_row["mean_roc_auc"])
        capacity_bottleneck = best_roc < 0.55  # near-chance after fair model sweep

        lines = [
            f"# Model Benchmark Report — {symbol} {timeframe} {side}",
            "",
            f"- Models: `{len(results)}`",
            f"- Features: `{int(summary['n_features'].iloc[0]) if not summary.empty else 'n/a'}`",
            f"- Protocol: walk-forward, identical dataset/metrics/threshold",
            "",
            "## 1. Which model has the best Walk Forward ROC?",
            "",
            f"**Answer:** `{best_roc_row['algorithm']}` "
            f"(mean ROC=`{_fmt(best_roc_row['mean_roc_auc'])}`, "
            f"std=`{_fmt(best_roc_row['std_roc_auc'])}`)",
            "",
            "## 2. Which model generalizes the best?",
            "",
            f"**Answer:** `{best_gen_row['algorithm']}` "
            f"(meanROC − stdROC = `{_fmt(best_gen_row['gen_score'])}`; "
            f"ROC=`{_fmt(best_gen_row['mean_roc_auc'])}`, std=`{_fmt(best_gen_row['std_roc_auc'])}`)",
            "",
            "## 3. Which model is the fastest?",
            "",
            f"**Answer:** `{fastest_row['algorithm']}` "
            f"(total train=`{_fmt(fastest_row['total_train_seconds'], 2)}`s, "
            f"infer=`{_fmt(fastest_row['total_infer_seconds'], 4)}`s)",
            "",
            "## 4. Which model is easiest to calibrate?",
            "",
            f"**Answer:** `{best_cal_row['algorithm']}` "
            f"(mean ECE=`{_fmt(best_cal_row['mean_calibration_error'])}`)",
            "",
            "## 5. Which model produces the best probability distribution?",
            "",
            f"**Answer:** `{best_dist_row['algorithm']}` "
            f"(pooled proba std=`{_fmt(best_dist_row['proba_std'])}`, "
            f"mean=`{_fmt(best_dist_row['proba_mean'])}`)",
            "",
            "## 6. Which model should become Production Base Model?",
            "",
        ]
        if prod is not None:
            lines.append(
                f"**Answer:** `{prod.algorithm}` — "
                f"ROC=`{_fmt(prod.mean_roc_auc)}` ECE=`{_fmt(prod.mean_calibration_error)}` "
                f"train=`{_fmt(prod.total_train_seconds, 2)}`s"
            )
            lines.append("")
            lines.append(
                "Rationale: best composite rank (ROC + generalization + calibration)."
            )
        else:
            lines.append("**Answer:** inconclusive.")

        lines.extend(
            [
                "",
                "## 7. Does boosting significantly outperform linear models?",
                "",
                f"**Answer:** `{'yes' if boost_beats else 'no / not clearly'}` "
                f"(boost mean ROC=`{_fmt(boost_mean)}` vs linear=`{_fmt(linear_mean)}`)",
                "",
                "## 8. Does CatBoost outperform LightGBM?",
                "",
            ]
        )
        if cat and lgbm:
            lines.append(
                f"**Answer:** `{'yes' if cat_beats else 'no'}` "
                f"(CatBoost=`{_fmt(cat.mean_roc_auc)}` vs LightGBM=`{_fmt(lgbm.mean_roc_auc)}`)"
            )
        else:
            lines.append("**Answer:** n/a (missing model).")

        lines.extend(
            [
                "",
                "## 9. Is model capacity now the bottleneck?",
                "",
                f"**Answer:** `{'unlikely — best ROC still near chance; labels/features/regime likely dominate' if capacity_bottleneck else 'possibly — best model clears chance; capacity/architecture may still matter'}` "
                f"(best mean WF ROC=`{_fmt(best_roc)}`)",
                "",
                "## Rankings",
                "",
                rankings.to_string(index=False) if not rankings.empty else "_(empty)_",
                "",
                "## Charts",
                "",
                "- `charts/roc_comparison.png`",
                "- `charts/pr_comparison.png`",
                "- `charts/calibration_comparison.png`",
                "- `charts/training_time_comparison.png`",
                "- `charts/inference_time_comparison.png`",
                "- `charts/probability_distribution.png`",
                "",
            ]
        )
        return "\n".join(lines)
