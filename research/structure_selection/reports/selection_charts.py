"""Structure selection charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.structure_selection.entities.selection_result import ExperimentResult


class SelectionChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        experiment_results: pd.DataFrame,
        feature_stability: pd.DataFrame,
        regime_results: pd.DataFrame,
        results: list[ExperimentResult],
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "roc": self._roc(charts, experiment_results),
            "delta": self._delta(charts, results),
            "stability": self._stability(charts, feature_stability),
            "regime": self._regime(charts, regime_results),
        }

    def _roc(self, charts: Path, df: pd.DataFrame) -> Path:
        path = charts / "experiment_roc.png"
        if df.empty:
            self._empty(path, "ROC")
            return path
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sides = sorted(df["side"].unique())
        for ax, side in zip(axes, sides + [None] * (2 - len(sides))):
            if side is None:
                ax.axis("off")
                continue
            part = df.loc[df["side"] == side].sort_values("mean_roc_auc")
            ax.barh(part["experiment_id"], part["mean_roc_auc"], color="#4C78A8")
            ax.set_title(f"{side} mean WF ROC")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _delta(self, charts: Path, results: list[ExperimentResult]) -> Path:
        path = charts / "delta_vs_baseline.png"
        by_key = {(r.side, r.experiment_id): r for r in results}
        rows = []
        for side in sorted({r.side for r in results}):
            base = by_key.get((side, "A_baseline"))
            if base is None:
                continue
            for r in results:
                if r.side != side or r.experiment_id == "A_baseline":
                    continue
                rows.append(
                    {
                        "side": side,
                        "experiment_id": r.experiment_id,
                        "delta_roc": r.mean_roc_auc - base.mean_roc_auc,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            self._empty(path, "Delta")
            return path
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for side in sorted(df["side"].unique()):
            part = df.loc[df["side"] == side]
            ax.plot(part["experiment_id"], part["delta_roc"], marker="o", label=side)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Delta ROC vs baseline")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _stability(self, charts: Path, stability: pd.DataFrame) -> Path:
        path = charts / "feature_stability.png"
        if stability.empty:
            self._empty(path, "Stability")
            return path
        # Average across experiments for E_top_structure preferentially
        focus = stability.loc[stability["experiment_id"] == "E_top_structure"]
        if focus.empty:
            focus = stability
        agg = (
            focus.groupby("feature")["stability_score"]
            .mean()
            .sort_values()
            .tail(10)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(agg.index, agg.values, color="#54A24B")
        ax.set_xlabel("Rank stability score")
        ax.set_title("Structure feature rank stability (higher=better)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _regime(self, charts: Path, regime: pd.DataFrame) -> Path:
        path = charts / "regime_roc.png"
        if regime.empty:
            self._empty(path, "Regime")
            return path
        summary = (
            regime.groupby(["regime_family", "regime_bin"])["roc_auc"].mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for fam in sorted(summary["regime_family"].unique()):
            part = summary.loc[summary["regime_family"] == fam]
            ax.plot(part["regime_bin"].astype(str), part["roc_auc"], marker="o", label=fam)
        ax.set_ylabel("Mean ROC")
        ax.set_title("Structure model ROC by regime bin")
        ax.legend()
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
