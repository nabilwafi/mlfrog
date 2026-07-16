"""Context impact charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.context_impact.entities.experiment_result import ExperimentResult


class ContextImpactChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        ablation: pd.DataFrame,
        comparison: pd.DataFrame,
        results: list[ExperimentResult],
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["roc"] = self._roc_bars(charts, ablation)
        paths["delta"] = self._delta_side(charts, comparison)
        paths["category"] = self._category_ablation(charts, ablation)
        paths["sep"] = self._metric_bars(charts, ablation, "mean_probability_separation", "Probability Separation")
        paths["stability"] = self._metric_bars(charts, ablation, "wf_stability", "Walk-Forward Stability")
        paths["wf"] = self._walk_forward(charts, results)
        return paths

    def _roc_bars(self, charts: Path, ablation: pd.DataFrame) -> Path:
        path = charts / "roc_comparison.png"
        if ablation.empty:
            self._empty(path, "ROC")
            return path
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        sides = sorted(ablation["side"].unique())
        for ax, side in zip(axes, sides + [None] * (2 - len(sides))):
            if side is None:
                ax.axis("off")
                continue
            df = ablation.loc[ablation["side"] == side].sort_values("mean_roc_auc")
            ax.barh(df["experiment_id"], df["mean_roc_auc"], color="#4C78A8")
            ax.set_title(f"{side} — mean WF ROC")
            ax.set_xlabel("ROC-AUC")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _delta_side(self, charts: Path, comparison: pd.DataFrame) -> Path:
        path = charts / "delta_roc_by_side.png"
        if comparison.empty:
            self._empty(path, "Delta ROC")
            return path
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#54A24B" if d > 0 else "#E45756" for d in comparison["delta_roc"]]
        ax.bar(comparison["side"], comparison["delta_roc"], color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("ΔROC (all_context − baseline)")
        ax.set_title("Context lift by side")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _category_ablation(self, charts: Path, ablation: pd.DataFrame) -> Path:
        path = charts / "category_ablation.png"
        cats = ["trend_context_only", "volatility_context_only", "structure_context_only", "all_context"]
        if ablation.empty:
            self._empty(path, "Category ablation")
            return path
        fig, ax = plt.subplots(figsize=(8, 5))
        for side in sorted(ablation["side"].unique()):
            base = ablation.loc[
                (ablation["side"] == side) & (ablation["experiment_id"] == "baseline"),
                "mean_roc_auc",
            ]
            if base.empty:
                continue
            base_roc = float(base.iloc[0])
            deltas = []
            labels = []
            for exp_id in cats:
                row = ablation.loc[
                    (ablation["side"] == side) & (ablation["experiment_id"] == exp_id)
                ]
                if row.empty:
                    continue
                labels.append(exp_id.replace("_context_only", "").replace("_context", ""))
                deltas.append(float(row["mean_roc_auc"].iloc[0]) - base_roc)
            ax.plot(labels, deltas, marker="o", label=side)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("ΔROC vs baseline")
        ax.set_title("Context category ablation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _metric_bars(self, charts: Path, ablation: pd.DataFrame, col: str, title: str) -> Path:
        safe = col.replace("mean_", "")
        path = charts / f"{safe}.png"
        if col == "wf_stability":
            path = charts / "walk_forward_stability.png"
        if col == "mean_probability_separation":
            path = charts / "probability_separation.png"
        if ablation.empty or col not in ablation.columns:
            self._empty(path, title)
            return path
        fig, ax = plt.subplots(figsize=(10, 5))
        for side in sorted(ablation["side"].unique()):
            df = ablation.loc[ablation["side"] == side]
            ax.plot(df["experiment_id"], df[col], marker="o", label=side)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _walk_forward(self, charts: Path, results: list[ExperimentResult]) -> Path:
        path = charts / "walk_forward_roc.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sides = sorted({r.side for r in results})
        for ax, side in zip(axes, sides + [None] * (2 - len(sides))):
            if side is None:
                ax.axis("off")
                continue
            for r in results:
                if r.side != side:
                    continue
                if r.experiment_id not in ("baseline", "all_context"):
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
            ax.set_title(f"{side} WF ROC")
            ax.set_xlabel("Valid year")
            ax.set_ylabel("ROC-AUC")
            ax.legend(fontsize=8)
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
