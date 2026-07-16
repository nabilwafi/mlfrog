"""Charts for Trade Quality Engine."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TradeQualityChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        panel: pd.DataFrame,
        tq_buckets: pd.DataFrame,
        yearly: pd.DataFrame,
        risk_search: pd.DataFrame,
        shap_df: pd.DataFrame,
        interactions: pd.DataFrame,
        capital: pd.DataFrame,
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._dist(charts, panel)
        self._buckets(charts, tq_buckets)
        self._yearly(charts, yearly)
        self._risk(charts, risk_search)
        self._shap(charts, shap_df)
        self._interact(charts, interactions)
        self._tq_exp(charts, tq_buckets)
        self._tq_dd(charts, tq_buckets)
        self._calib(charts, panel)
        self._rolling(charts, panel)
        self._capital(charts, capital)

    def _dist(self, charts: Path, panel: pd.DataFrame) -> None:
        path = charts / "tq_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(panel["trade_quality"].dropna(), bins=20, color="#4C78A8", edgecolor="white", alpha=0.85, label="TQ")
        if "confidence" in panel.columns:
            ax.hist(panel["confidence"].dropna(), bins=20, color="#54A24B", edgecolor="white", alpha=0.45, label="Conf")
        ax.legend()
        ax.set_title("Trade Quality distribution")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _buckets(self, charts: Path, buckets: pd.DataFrame) -> None:
        path = charts / "bucket_performance.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if not buckets.empty:
            axes[0].bar(buckets["bucket"], buckets["expectancy"], color="#54A24B")
            axes[0].set_title("Expectancy")
            axes[1].bar(buckets["bucket"], buckets["profit_factor"].replace([np.inf], np.nan), color="#F58518")
            axes[1].set_title("Profit factor")
            axes[2].bar(buckets["bucket"], buckets["trades"], color="#4C78A8")
            axes[2].set_title("Trades")
            for ax in axes:
                ax.tick_params(axis="x", rotation=30, labelsize=8)
        fig.suptitle("TQ bucket performance")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _yearly(self, charts: Path, yearly: pd.DataFrame) -> None:
        path = charts / "yearly_stability.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not yearly.empty:
            ax.plot(yearly["valid_year"], yearly["spearman_tq_net"], marker="o", label="spearman TQ-net")
            ax.plot(yearly["valid_year"], yearly["ranking_gap"] / 100.0, marker="s", label="ranking_gap/100")
            ax.legend(fontsize=8)
        ax.set_title("Yearly TQ stability")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _risk(self, charts: Path, risk: pd.DataFrame) -> None:
        path = charts / "risk_compare.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not risk.empty:
            ax.scatter(risk["max_drawdown"], risk["cagr"], s=55, c="#4C78A8")
            for _, r in risk.iterrows():
                ax.annotate(str(r["schedule"])[:16], (r["max_drawdown"], r["cagr"]), fontsize=7)
        ax.set_xlabel("max DD")
        ax.set_ylabel("CAGR")
        ax.set_title("Dynamic risk schedules")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _shap(self, charts: Path, shap_df: pd.DataFrame) -> None:
        path = charts / "shap_importance.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        if not shap_df.empty:
            s = shap_df.iloc[::-1]
            ax.barh(s["feature"], s["mean_abs_shap"], color="#72B7B2")
        ax.set_title("SHAP importance (TQ components)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _interact(self, charts: Path, interactions: pd.DataFrame) -> None:
        path = charts / "interaction_heatmap.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        if interactions.empty:
            ax.set_title("empty")
        else:
            top = interactions.head(15).iloc[::-1]
            ax.barh(top["segment"], top["tq_edge"], color="#E45756")
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_title("TQ edge by segment")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _tq_exp(self, charts: Path, buckets: pd.DataFrame) -> None:
        path = charts / "tq_vs_expectancy.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        if not buckets.empty:
            mid = [10, 30, 50, 70, 90][: len(buckets)]
            ax.plot(mid[: len(buckets)], buckets["expectancy"], marker="o")
        ax.set_xlabel("bucket mid TQ")
        ax.set_ylabel("expectancy")
        ax.set_title("Trade Quality vs Expectancy")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _tq_dd(self, charts: Path, buckets: pd.DataFrame) -> None:
        path = charts / "tq_vs_drawdown.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        if not buckets.empty:
            mid = [10, 30, 50, 70, 90][: len(buckets)]
            ax.plot(mid[: len(buckets)], buckets["drawdown"], marker="o", color="#E45756")
        ax.set_title("Trade Quality vs Drawdown")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _calib(self, charts: Path, panel: pd.DataFrame) -> None:
        path = charts / "calibration.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        tq = panel["trade_quality"].to_numpy(dtype=float) / 100.0
        y = panel["meta_label"].to_numpy(dtype=int)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (tq >= bins[i]) & (tq < bins[i + 1] if i < 9 else tq <= bins[i + 1])
            if m.any():
                xs.append(float(tq[m].mean()))
                ys.append(float(y[m].mean()))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if xs:
            ax.plot(xs, ys, marker="o", color="#4C78A8")
        ax.set_title("TQ calibration (not a probability claim)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _rolling(self, charts: Path, panel: pd.DataFrame) -> None:
        path = charts / "rolling_performance.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        p = panel.sort_values("timestamp").copy()
        if not p.empty:
            roll = p["net_return"].rolling(50, min_periods=10).mean()
            ax.plot(pd.to_datetime(p["timestamp"]), roll, color="#54A24B")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("Rolling mean net return (50 trades)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _capital(self, charts: Path, capital: pd.DataFrame) -> None:
        path = charts / "capital_allocation.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not capital.empty:
            ax.bar(capital["bucket"], capital["suggested_risk_pct"] * 100, color="#F58518")
            ax.set_ylabel("suggested risk %")
        ax.set_title("Capital allocation by TQ bucket")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
