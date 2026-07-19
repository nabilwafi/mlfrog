"""H4 structure research charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.h4_structure.entities.experiment_result import ExperimentResult


class StructureChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        ablation: pd.DataFrame,
        feature_statistics: pd.DataFrame,
        results: list[ExperimentResult],
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        paths = {
            "roc": self._roc(charts, ablation),
            "delta": self._delta(charts, results),
            "sep": self._sep(charts, ablation),
            "importance": self._importance(charts, feature_statistics),
        }
        return paths

    def _roc(self, charts: Path, ablation: pd.DataFrame) -> Path:
        path = charts / "roc_comparison.png"
        if ablation.empty:
            self._empty(path, "ROC")
            return path
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sides = sorted(ablation["side"].unique())
        for ax, side in zip(axes, sides + [None] * (2 - len(sides))):
            if side is None:
                ax.axis("off")
                continue
            df = ablation.loc[ablation["side"] == side].sort_values("mean_roc_auc")
            ax.barh(df["experiment_id"], df["mean_roc_auc"], color="#4C78A8")
            ax.set_title(f"{side} mean WF ROC")
            ax.set_xlabel("ROC-AUC")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _delta(self, charts: Path, results: list[ExperimentResult]) -> Path:
        path = charts / "delta_roc.png"
        by_key = {(r.side, r.experiment_id): r for r in results}
        rows = []
        for side in sorted({r.side for r in results}):
            base = by_key.get((side, "baseline"))
            if base is None:
                continue
            for exp_id in ("swing_quality_only", "new_structure", "all_structure"):
                r = by_key.get((side, exp_id))
                if r is None:
                    continue
                rows.append(
                    {
                        "side": side,
                        "experiment_id": exp_id,
                        "delta_roc": r.mean_roc_auc - base.mean_roc_auc,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            self._empty(path, "Delta ROC")
            return path
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for side in sorted(df["side"].unique()):
            part = df.loc[df["side"] == side]
            ax.plot(part["experiment_id"], part["delta_roc"], marker="o", label=side)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Delta ROC vs baseline")
        ax.set_title("Structure ablation lift")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _sep(self, charts: Path, ablation: pd.DataFrame) -> Path:
        path = charts / "probability_separation.png"
        if ablation.empty:
            self._empty(path, "Separation")
            return path
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for side in sorted(ablation["side"].unique()):
            df = ablation.loc[ablation["side"] == side]
            ax.plot(df["experiment_id"], df["mean_probability_separation"], marker="o", label=side)
        ax.set_ylabel("Probability separation")
        ax.set_title("Class probability separation")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _importance(self, charts: Path, stats: pd.DataFrame) -> Path:
        path = charts / "feature_importance.png"
        imp_cols = [c for c in stats.columns if c.startswith("importance_")]
        if stats.empty or not imp_cols:
            self._empty(path, "Importance")
            return path
        df = stats.copy()
        df["avg_importance"] = df[imp_cols].mean(axis=1)
        df = df.sort_values("avg_importance", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(df["feature"], df["avg_importance"], color="#F58518")
        ax.set_xlabel("Mean LightGBM gain (structure features)")
        ax.set_title("Structure feature importance")
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
