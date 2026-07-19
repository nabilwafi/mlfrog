"""Model benchmark charts (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.model_benchmark.entities.benchmark_result import BenchmarkModelResult


class BenchmarkChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        summary: pd.DataFrame,
        results: list[BenchmarkModelResult],
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["roc"] = self._metric_bars(charts, summary, "mean_roc_auc", "ROC-AUC", "roc_comparison.png")
        paths["pr"] = self._metric_bars(charts, summary, "mean_pr_auc", "PR-AUC", "pr_comparison.png")
        paths["cal"] = self._metric_bars(
            charts, summary, "mean_calibration_error", "Calibration Error (ECE)", "calibration_comparison.png"
        )
        paths["train"] = self._metric_bars(
            charts, summary, "total_train_seconds", "Training Time (s)", "training_time_comparison.png"
        )
        paths["infer"] = self._metric_bars(
            charts, summary, "total_infer_seconds", "Inference Time (s)", "inference_time_comparison.png"
        )
        paths["proba"] = self._proba_hist(charts, results)
        return paths

    def _metric_bars(
        self,
        charts: Path,
        summary: pd.DataFrame,
        col: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        path = charts / filename
        if summary.empty or col not in summary.columns:
            self._empty(path, ylabel)
            return path
        df = summary.sort_values(col, ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(df["algorithm"], df[col], color="#4C78A8")
        ax.set_xlabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _proba_hist(self, charts: Path, results: list[BenchmarkModelResult]) -> Path:
        path = charts / "probability_distribution.png"
        ok = [r for r in results if r.pooled_proba]
        if not ok:
            self._empty(path, "Probability distribution")
            return path
        n = len(ok)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for i, r in enumerate(ok):
            ax = axes[i // cols][i % cols]
            ax.hist(r.pooled_proba, bins=30, color="#F58518", edgecolor="white")
            ax.set_title(r.algorithm, fontsize=9)
            ax.set_xlim(0, 1)
        for j in range(len(ok), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle("Pooled prediction probability distribution")
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
