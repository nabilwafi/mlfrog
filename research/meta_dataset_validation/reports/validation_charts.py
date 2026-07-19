"""Charts for meta-dataset validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class MetaDatasetChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        candidates: pd.DataFrame,
        summary: pd.DataFrame,
        wf: pd.DataFrame,
        focus_percentile: float | None = None,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        # Focus histograms on recommended-ish mid gate if present else 0.10
        pct = focus_percentile
        if pct is None:
            pct = 0.10
        focus = candidates.loc[candidates["percentile"] == pct]
        if focus.empty and not candidates.empty:
            focus = candidates.loc[
                candidates["percentile"] == float(candidates["percentile"].median())
            ]
        return {
            "return": self._hist(charts, focus, "net_return", "return_hist.png", "Net return"),
            "mae": self._hist(charts, focus, "mae", "mae_hist.png", "MAE"),
            "mfe": self._hist(charts, focus, "mfe", "mfe_hist.png", "MFE"),
            "holding": self._hist(
                charts, focus, "holding_bars", "holding_hist.png", "Holding bars"
            ),
            "prob": self._hist(
                charts,
                focus,
                "y_prob_raw" if "y_prob_raw" in focus.columns else "y_prob",
                "probability_hist.png",
                "Raw probability",
            ),
            "quality": self._quality(charts, summary),
            "wf": self._wf(charts, wf),
        }

    def _hist(
        self,
        charts: Path,
        frame: pd.DataFrame,
        col: str,
        filename: str,
        title: str,
    ) -> Path:
        path = charts / filename
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = frame.loc[frame["side"] == side, col] if col in frame.columns else pd.Series(dtype=float)
            sub = sub.dropna()
            if sub.empty:
                ax.set_title(f"{side} {title} (empty)")
                continue
            ax.hist(sub, bins=30, color="#4C78A8", edgecolor="white")
            ax.axvline(float(sub.mean()), color="#E45756", linestyle="--", label="mean")
            ax.set_title(f"{side} {title}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _quality(self, charts: Path, summary: pd.DataFrame) -> Path:
        path = charts / "percentile_quality.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in zip(
            axes,
            ("expectancy", "profit_factor"),
            ("Expectancy vs top-%", "Profit factor vs top-%"),
            strict=True,
        ):
            for side, color in (("long", "#54A24B"), ("short", "#E45756")):
                g = summary.loc[summary["side"] == side].sort_values("percentile")
                if g.empty:
                    continue
                ax.plot(g["percentile"] * 100, g[metric], marker="o", color=color, label=side)
            if metric == "expectancy":
                ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            else:
                ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("top percentile %")
            ax.set_title(title)
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _wf(self, charts: Path, wf: pd.DataFrame) -> Path:
        path = charts / "wf_expectancy.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, side in zip(axes, ("long", "short"), strict=True):
            sub = wf.loc[wf["side"] == side]
            for pct, g in sub.groupby("percentile"):
                g = g.sort_values("valid_year")
                ax.plot(
                    g["valid_year"],
                    g["expectancy"],
                    marker="o",
                    label=f"top {int(pct * 100)}%",
                )
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"{side} WF expectancy")
            ax.set_xlabel("valid year")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("expectancy")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
