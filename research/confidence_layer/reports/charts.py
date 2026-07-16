"""Charts for confidence layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ConfidenceChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        panel: pd.DataFrame,
        buckets: pd.DataFrame,
        regimes: pd.DataFrame,
        risk_search: pd.DataFrame,
        calibration_bins: pd.DataFrame,
        shap_importance: pd.DataFrame,
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._conf_dist(charts, panel)
        self._buckets(charts, buckets)
        self._regime(charts, regimes)
        self._risk(charts, risk_search)
        self._reliability(charts, calibration_bins)
        self._shap(charts, shap_importance)

    def _conf_dist(self, charts: Path, panel: pd.DataFrame) -> None:
        path = charts / "confidence_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if "confidence" in panel.columns:
            ax.hist(panel["confidence"], bins=20, color="#4C78A8", edgecolor="white")
        ax.set_title("Confidence score distribution")
        ax.set_xlabel("confidence (0-100)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _buckets(self, charts: Path, buckets: pd.DataFrame) -> None:
        path = charts / "bucket_expectancy.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not buckets.empty and "expectancy" in buckets.columns:
            ax.bar(buckets["bucket"], buckets["expectancy"], color="#54A24B")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("Expectancy by confidence bucket")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _regime(self, charts: Path, regimes: pd.DataFrame) -> None:
        path = charts / "regime_performance.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        if not regimes.empty:
            ax.barh(regimes["d1_regime"], regimes["expectancy"], color="#F58518")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title("Expectancy by D1 regime")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _risk(self, charts: Path, risk_search: pd.DataFrame) -> None:
        path = charts / "risk_schedule_compare.png"
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if not risk_search.empty:
            top = risk_search.head(7)
            ax.scatter(top["max_drawdown"], top["cagr"], s=60, c="#4C78A8")
            for _, r in top.iterrows():
                ax.annotate(str(r["schedule"])[:18], (r["max_drawdown"], r["cagr"]), fontsize=7)
        ax.set_xlabel("max drawdown")
        ax.set_ylabel("CAGR")
        ax.set_title("Dynamic risk schedules (CAGR vs DD)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _reliability(self, charts: Path, bins: pd.DataFrame) -> None:
        path = charts / "reliability_diagram.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if not bins.empty:
            ax.plot(bins["avg_confidence"], bins["empirical_win_rate"], marker="o", color="#E45756")
        ax.set_xlabel("confidence/100")
        ax.set_ylabel("empirical win rate")
        ax.set_title("Reliability diagram")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _shap(self, charts: Path, shap_importance: pd.DataFrame) -> None:
        path = charts / "shap_importance.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not shap_importance.empty:
            s = shap_importance.iloc[::-1]
            ax.barh(s["feature"], s["mean_abs_shap"], color="#72B7B2")
        ax.set_title("Component SHAP importance")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
