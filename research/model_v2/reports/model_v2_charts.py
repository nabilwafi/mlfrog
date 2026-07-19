"""Model v2 validation charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.structure_selection.entities.selection_result import ExperimentResult


class ModelV2ChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        results: list[ExperimentResult],
        comparison: pd.DataFrame,
        feature_importance: pd.DataFrame,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["delta"] = self._delta(charts, comparison)
        for side in sorted({r.side for r in results}):
            paths[f"{side}_roc"] = self._side_windows(charts, results, side)
            paths[f"{side}_imp"] = self._side_importance(charts, feature_importance, side)
        return paths

    def _delta(self, charts: Path, comparison: pd.DataFrame) -> Path:
        path = charts / "comparison_delta.png"
        if comparison.empty:
            self._empty(path, "Delta")
            return path
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#54A24B" if d > 0 else "#E45756" for d in comparison["delta_roc"]]
        ax.bar(comparison["side"], comparison["delta_roc"], color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Delta ROC (v2 - baseline)")
        ax.set_title("Model v2 lift by side")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _side_windows(
        self, charts: Path, results: list[ExperimentResult], side: str
    ) -> Path:
        path = charts / f"{side}_roc_windows.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in results:
            if r.side != side:
                continue
            ok = [w for w in r.windows if w.status == "ok"]
            if not ok:
                continue
            ax.plot(
                [w.valid_year for w in ok],
                [w.roc_auc for w in ok],
                marker="o",
                label=r.experiment_id,
            )
        ax.set_xlabel("Valid year")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{side} walk-forward ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _side_importance(
        self, charts: Path, importance: pd.DataFrame, side: str
    ) -> Path:
        path = charts / f"{side}_feature_importance.png"
        df = importance.loc[importance["side"] == side].copy()
        if df.empty:
            self._empty(path, f"{side} importance")
            return path
        df = df.sort_values("mean_gain", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(df["feature"], df["mean_gain"], color="#4C78A8")
        ax.set_xlabel("Mean LightGBM gain")
        ax.set_title(f"{side} structure feature importance (v2)")
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
