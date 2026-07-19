"""Charts for Sprint 16."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


class CalibrationChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        calibrated: pd.DataFrame,
        thresholds: pd.DataFrame,
        context: pd.DataFrame,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "reliability": self._reliability(charts, calibrated),
            "threshold": self._threshold(charts, thresholds),
            "context": self._context(charts, context),
        }

    def _reliability(self, charts: Path, calibrated: pd.DataFrame) -> Path:
        path = charts / "reliability_curve.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        methods = (
            ("y_prob_raw", "raw", "#4C78A8"),
            ("y_prob_platt", "platt", "#54A24B"),
            ("y_prob_isotonic", "isotonic", "#E45756"),
        )
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = calibrated.loc[calibrated["side"] == side]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
            if sub.empty:
                ax.set_title(f"{side} (empty)")
                continue
            y = sub["y_true"].astype(int).to_numpy()
            for col, name, color in methods:
                if col not in sub.columns:
                    continue
                p = sub[col].astype(float).to_numpy()
                try:
                    frac, mean_p = calibration_curve(
                        y, p, n_bins=min(10, max(3, len(sub) // 200)), strategy="quantile"
                    )
                except ValueError:
                    continue
                ax.plot(mean_p, frac, marker="o", color=color, label=name)
            ax.set_title(f"{side} reliability")
            ax.set_xlabel("mean predicted")
            ax.set_ylabel("fraction positive")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _threshold(self, charts: Path, thresholds: pd.DataFrame) -> Path:
        path = charts / "threshold_expectancy.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = thresholds.loc[thresholds["side"] == side]
            for prob_col, color in (
                ("y_prob_raw", "#4C78A8"),
                ("y_prob_platt", "#54A24B"),
                ("y_prob_isotonic", "#E45756"),
            ):
                g = sub.loc[sub["prob_col"] == prob_col].sort_values("percentile")
                if g.empty:
                    continue
                ax.plot(
                    g["percentile"] * 100,
                    g["expectancy"],
                    marker="o",
                    color=color,
                    label=prob_col.replace("y_prob_", ""),
                )
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
            ax.set_title(f"{side} expectancy vs top-%")
            ax.set_xlabel("top percentile %")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("expectancy (net)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _context(self, charts: Path, context: pd.DataFrame) -> Path:
        path = charts / "context_filter.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = context.loc[context["side"] == side]
            if sub.empty:
                ax.set_title(f"{side} (empty)")
                continue
            # best expectancy per feature at each percentile
            for feat in sorted(sub["context_feature"].unique()):
                g = sub.loc[sub["context_feature"] == feat].sort_values("percentile")
                ax.plot(g["percentile"] * 100, g["expectancy"], marker="o", label=feat.replace("ctx_h4_", ""))
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
            ax.set_title(f"{side} context+percentile")
            ax.set_xlabel("top percentile %")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("expectancy (net)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
