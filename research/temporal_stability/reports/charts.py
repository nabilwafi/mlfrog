"""Charts for temporal stability research."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalStabilityChartBuilder:
    def write_all(self, out_dir: Path, arts: dict) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        rolling = arts.get("rolling")
        if isinstance(rolling, pd.DataFrame) and not rolling.empty:
            self._line(rolling, "test_year", "roc_auc", charts / "roc_over_time.png", "ROC AUC over time")
            self._line(rolling, "test_year", "profit_factor", charts / "pf_over_time.png", "Profit Factor over time")
            self._line(rolling, "test_year", "total_return", charts / "return_over_time.png", "Return over time")
            self._line(rolling, "test_year", "ece", charts / "calibration_over_time.png", "ECE over time")

        drift = arts.get("drift")
        if isinstance(drift, pd.DataFrame) and not drift.empty:
            self._drift_heatmap(drift, charts / "drift_heatmap.png")

        fi = arts.get("feature_importance")
        if isinstance(fi, pd.DataFrame) and not fi.empty:
            self._feature_importance(fi, charts / "feature_importance_over_time.png")

        thr = arts.get("threshold")
        if isinstance(thr, pd.DataFrame) and not thr.empty:
            self._line(thr, "year", "best_threshold", charts / "threshold_stability.png", "Optimal threshold by year")

        aging = arts.get("aging")
        if isinstance(aging, pd.DataFrame) and not aging.empty:
            self._aging(aging, charts / "aging_windows.png")

        retrain = arts.get("retrain")
        if isinstance(retrain, pd.DataFrame) and not retrain.empty:
            self._retrain(retrain, charts / "retrain_degradation.png")

    def _line(self, df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
        if x not in df.columns or y not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df[x], df[y], marker="o")
        if y == "best_threshold":
            ax.axhline(0.45, color="gray", linestyle="--", label="production 0.45")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _drift_heatmap(self, drift: pd.DataFrame, path: Path) -> None:
        if "year_from" not in drift.columns or "feature_psi_mean" not in drift.columns:
            return
        years = sorted(set(drift["year_from"].astype(int)).union(set(drift["year_to"].astype(int))))
        mat = np.full((len(years), len(years)), np.nan)
        idx = {y: i for i, y in enumerate(years)}
        for _, r in drift.iterrows():
            i, j = idx[int(r["year_from"])], idx[int(r["year_to"])]
            v = r["feature_psi_mean"]
            mat[i, j] = float(v) if v == v else np.nan
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(years)))
        ax.set_yticks(range(len(years)))
        ax.set_xticklabels(years, rotation=90)
        ax.set_yticklabels(years)
        ax.set_title("Feature PSI drift (year_from → year_to)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _feature_importance(self, fi: pd.DataFrame, path: Path) -> None:
        top = (
            fi.groupby("feature")["gain_share"].mean().sort_values(ascending=False).head(8).index.tolist()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for feat in top:
            g = fi.loc[fi["feature"] == feat].sort_values("year")
            ax.plot(g["year"], g["gain_share"], marker="o", label=feat)
        ax.set_title("Feature gain share over time (top 8)")
        ax.set_xlabel("year")
        ax.set_ylabel("gain_share")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _aging(self, aging: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        for win, g in aging.groupby("history_window"):
            g = g.sort_values("test_year")
            ax.plot(g["test_year"], g["roc_auc"], marker="o", label=str(win))
        ax.set_title("ROC by history window")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _retrain(self, retrain: pd.DataFrame, path: Path) -> None:
        g = retrain.groupby("freeze_months", as_index=False)[["roc_auc", "profit_factor", "ece"]].mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(g["freeze_months"], g["roc_auc"], marker="o", label="ROC")
        ax.set_xlabel("months without retrain")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Degradation vs freeze horizon")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
