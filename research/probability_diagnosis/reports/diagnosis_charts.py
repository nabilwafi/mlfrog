"""Charts for probability collapse diagnosis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class DiagnosisChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        predictions: pd.DataFrame,
        wf_drift: pd.DataFrame,
        compare: pd.DataFrame,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "histogram": self._histogram(charts, predictions),
            "wf_shift": self._wf_shift(charts, wf_drift),
            "baseline_v2": self._baseline_v2(charts, predictions, compare),
        }

    def _histogram(self, charts: Path, predictions: pd.DataFrame) -> Path:
        path = charts / "prediction_histogram.png"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        sides = ("long", "short")
        splits = ("train", "validation")
        for i, side in enumerate(sides):
            for j, split in enumerate(splits):
                ax = axes[i][j]
                for exp, color, label in (
                    ("A_baseline", "#4C78A8", "baseline"),
                    ("B_v2_context", "#54A24B", "v2"),
                ):
                    p = predictions.loc[
                        (predictions["side"] == side)
                        & (predictions["experiment_id"] == exp)
                        & (predictions["split"] == split),
                        "y_prob",
                    ]
                    if p.empty:
                        continue
                    ax.hist(
                        p,
                        bins=40,
                        range=(0.25, 0.65),
                        alpha=0.55,
                        color=color,
                        label=label,
                        density=True,
                    )
                ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
                ax.set_title(f"{side} / {split}")
                ax.legend(fontsize=8)
        fig.suptitle("Prediction density (zoomed 0.25-0.65)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _wf_shift(self, charts: Path, wf_drift: pd.DataFrame) -> Path:
        path = charts / "wf_probability_shift.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        v2 = wf_drift.loc[wf_drift["experiment_id"] == "B_v2_context"]
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = v2.loc[v2["side"] == side].sort_values("valid_year")
            if sub.empty:
                ax.set_title(f"{side} (empty)")
                continue
            ax.plot(sub["valid_year"], sub["mean"], marker="o", label="mean")
            ax.plot(sub["valid_year"], sub["p90"], marker="s", label="p90")
            ax.plot(sub["valid_year"], sub["p95"], marker="^", label="p95")
            ax.plot(sub["valid_year"], sub["p99"], marker="d", label="p99")
            ax.plot(
                sub["valid_year"],
                sub["pos_rate"],
                linestyle="--",
                color="gray",
                label="pos_rate",
            )
            ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":")
            ax.set_title(f"{side} v2 WF drift")
            ax.set_xlabel("valid year")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("probability")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _baseline_v2(
        self, charts: Path, predictions: pd.DataFrame, compare: pd.DataFrame
    ) -> Path:
        path = charts / "baseline_vs_v2_probability.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # left: std / range / sep bars
        ax = axes[0]
        if not compare.empty:
            x = range(len(compare))
            w = 0.25
            ax.bar([i - w for i in x], compare["baseline_std"], width=w, label="base std", color="#4C78A8")
            ax.bar([i for i in x], compare["v2_std"], width=w, label="v2 std", color="#54A24B")
            ax.bar(
                [i + w for i in x],
                compare["delta_roc"],
                width=w,
                label="delta ROC",
                color="#E45756",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(compare["side"].tolist())
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_title("Std and delta ROC")
            ax.legend(fontsize=8)
        # right: violin-ish via box of probs
        ax = axes[1]
        data = []
        labels = []
        for side in ("long", "short"):
            for exp, tag in (("A_baseline", "base"), ("B_v2_context", "v2")):
                p = predictions.loc[
                    (predictions["side"] == side)
                    & (predictions["experiment_id"] == exp)
                    & (predictions["split"] == "validation"),
                    "y_prob",
                ]
                if p.empty:
                    continue
                data.append(p.to_numpy())
                labels.append(f"{side[:1]}/{tag}")
        if data:
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title("Validation prob boxplot")
            ax.set_ylabel("y_prob")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
