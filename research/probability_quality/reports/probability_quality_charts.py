"""Charts for probability quality."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


class ProbabilityQualityChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        predictions: pd.DataFrame,
        buckets: pd.DataFrame,
        deciles: pd.DataFrame,
        confidence: pd.DataFrame,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "histogram": self._histogram(charts, predictions),
            "buckets": self._buckets(charts, buckets, deciles),
            "calibration": self._calibration(charts, predictions),
            "confidence": self._confidence(charts, confidence),
        }

    def _histogram(self, charts: Path, predictions: pd.DataFrame) -> Path:
        path = charts / "probability_histogram.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, side in zip(axes, ("long", "short"), strict=True):
            p = predictions.loc[predictions["side"] == side, "y_prob"]
            if p.empty:
                ax.set_title(f"{side} (empty)")
                continue
            ax.hist(p, bins=30, range=(0, 1), color="#4C78A8", edgecolor="white")
            ax.axvline(float(p.mean()), color="#E45756", linestyle="--", label="mean")
            ax.set_title(f"{side} y_prob")
            ax.set_xlabel("probability")
            ax.legend()
        axes[0].set_ylabel("count")
        fig.suptitle("Probability distribution (OOF WF)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _buckets(
        self, charts: Path, buckets: pd.DataFrame, deciles: pd.DataFrame
    ) -> Path:
        path = charts / "bucket_performance.png"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        # Absolute buckets
        for ax, metric, title in zip(
            axes[0],
            ("win_rate", "expectancy"),
            ("Abs bucket win rate", "Abs bucket expectancy"),
            strict=True,
        ):
            for side, color in (("long", "#54A24B"), ("short", "#E45756")):
                sub = buckets.loc[buckets["side"] == side].sort_values("bucket_lo")
                if sub.empty:
                    continue
                ax.plot(sub["bucket"], sub[metric], marker="o", color=color, label=side)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            ax.legend()
            ax.axhline(0.5 if metric == "win_rate" else 0.0, color="gray", linewidth=0.8, linestyle=":")
        # Relative deciles
        for ax, metric, title in zip(
            axes[1],
            ("win_rate", "expectancy"),
            ("Decile win rate", "Decile expectancy"),
            strict=True,
        ):
            for side, color in (("long", "#54A24B"), ("short", "#E45756")):
                sub = deciles.loc[deciles["side"] == side].sort_values("decile")
                if sub.empty:
                    continue
                ax.plot(sub["bucket"], sub[metric], marker="o", color=color, label=side)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            ax.legend()
            ax.axhline(0.5 if metric == "win_rate" else 0.0, color="gray", linewidth=0.8, linestyle=":")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _calibration(self, charts: Path, predictions: pd.DataFrame) -> Path:
        path = charts / "calibration_curve.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        for side, color in (("long", "#54A24B"), ("short", "#E45756")):
            sub = predictions.loc[predictions["side"] == side]
            if sub.empty or sub["y_true"].nunique() < 2:
                continue
            y = sub["y_true"].astype(int).to_numpy()
            p = sub["y_prob"].astype(float).to_numpy()
            n_bins = min(10, max(3, len(sub) // 200))
            try:
                frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
            except ValueError:
                continue
            ax.plot(mean_p, frac, marker="o", color=color, label=side)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("fraction of positives")
        ax.set_title("Reliability diagram (uncalibrated v2)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _confidence(self, charts: Path, confidence: pd.DataFrame) -> Path:
        path = charts / "confidence_distribution.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        order = ["low", "medium", "high"]
        x = np.arange(len(order))
        width = 0.35
        for ax, metric, title in zip(
            axes,
            ("samples", "win_rate"),
            ("Samples by confidence", "Win rate by confidence"),
            strict=True,
        ):
            for i, (side, color) in enumerate((("long", "#54A24B"), ("short", "#E45756"))):
                vals = []
                for tier in order:
                    row = confidence.loc[
                        (confidence["side"] == side) & (confidence["confidence"] == tier)
                    ]
                    vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
                ax.bar(x + (i - 0.5) * width, vals, width=width, color=color, label=side)
            ax.set_xticks(x)
            ax.set_xticklabels(order)
            ax.set_title(title)
            ax.legend()
            if metric == "win_rate":
                ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
