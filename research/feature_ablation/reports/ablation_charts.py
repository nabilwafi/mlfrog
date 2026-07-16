"""Ablation charts (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.feature_ablation.entities.experiment_result import ExperimentResult


class AblationChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        summary: pd.DataFrame,
        results: list[ExperimentResult],
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        paths["feature_contribution"] = self._feature_contribution(charts, results, summary)
        paths["category_contribution"] = self._category_contribution(charts, results)
        paths["walk_forward_comparison"] = self._walk_forward(charts, results)
        paths["roc_comparison"] = self._roc_bars(charts, summary)
        paths["feature_count_vs_roc"] = self._count_vs(charts, summary, "mean_roc_auc", "ROC-AUC")
        paths["feature_count_vs_calibration"] = self._count_vs(
            charts, summary, "mean_calibration_error", "Calibration Error (ECE)"
        )
        return paths

    def _feature_contribution(
        self, charts: Path, results: list[ExperimentResult], summary: pd.DataFrame
    ) -> Path:
        # Approximate per-experiment contribution vs baseline ROC
        path = charts / "feature_contribution.png"
        if summary.empty or "baseline" not in set(summary["experiment_id"]):
            self._empty(path, "Feature contribution")
            return path
        base = float(summary.loc[summary["experiment_id"] == "baseline", "mean_roc_auc"].iloc[0])
        df = summary.copy()
        df["delta_vs_baseline"] = df["mean_roc_auc"] - base
        df = df.sort_values("delta_vs_baseline")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["name"], df["delta_vs_baseline"], color="#4C78A8")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean WF ROC − baseline")
        ax.set_title("Experiment contribution vs baseline")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _category_contribution(self, charts: Path, results: list[ExperimentResult]) -> Path:
        path = charts / "category_contribution.png"
        by_id = {r.experiment_id: r for r in results}
        baseline = by_id.get("baseline")
        if baseline is None:
            self._empty(path, "Category contribution")
            return path
        mapping = {
            "trend": "remove_trend",
            "momentum": "remove_momentum",
            "volatility": "remove_volatility",
            "candle": "remove_candle",
            "session": "remove_session",
            "statistical": "remove_statistical",
        }
        cats, deltas = [], []
        for cat, exp_id in mapping.items():
            r = by_id.get(exp_id)
            if r is None:
                continue
            cats.append(cat)
            deltas.append(baseline.mean_roc_auc - r.mean_roc_auc)
        order = sorted(range(len(cats)), key=lambda i: deltas[i], reverse=True)
        cats = [cats[i] for i in order]
        deltas = [deltas[i] for i in order]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(cats, deltas, color="#F58518")
        ax.set_ylabel("ROC drop when removed")
        ax.set_title("Category contribution")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _walk_forward(self, charts: Path, results: list[ExperimentResult]) -> Path:
        path = charts / "walk_forward_comparison.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in results:
            ok = [w for w in r.windows if w.status == "ok"]
            if not ok:
                continue
            years = [w.valid_year for w in ok]
            rocs = [w.roc_auc for w in ok]
            ax.plot(years, rocs, marker="o", label=r.experiment_id)
        ax.set_xlabel("Validation year")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Walk-forward ROC by experiment")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _roc_bars(self, charts: Path, summary: pd.DataFrame) -> Path:
        path = charts / "roc_comparison.png"
        if summary.empty:
            self._empty(path, "ROC comparison")
            return path
        df = summary.sort_values("mean_roc_auc", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["name"], df["mean_roc_auc"], color="#54A24B")
        ax.set_xlabel("Mean walk-forward ROC-AUC")
        ax.set_title("ROC comparison across ablation experiments")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _count_vs(self, charts: Path, summary: pd.DataFrame, metric: str, ylabel: str) -> Path:
        safe = metric.replace("mean_", "")
        path = charts / f"feature_count_vs_{safe}.png"
        if summary.empty:
            self._empty(path, ylabel)
            return path
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary["n_features"], summary[metric], c="#B279A2")
        for _, row in summary.iterrows():
            ax.annotate(row["experiment_id"], (row["n_features"], row[metric]), fontsize=7)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Feature count vs {ylabel}")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _empty(self, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No data: {title}", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=100)
        plt.close(fig)
