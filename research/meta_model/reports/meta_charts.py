"""Charts for production meta model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.meta_model import REFERENCE_THRESHOLD


class MetaModelChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        summary: pd.DataFrame,
        by_year: pd.DataFrame,
        equity: pd.DataFrame,
        threshold_ref: float = REFERENCE_THRESHOLD,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "compare": self._compare(charts, summary, threshold_ref),
            "threshold": self._threshold(charts, summary),
            "equity": self._equity(charts, equity, threshold_ref),
            "wf": self._wf(charts, by_year, threshold_ref),
        }

    def _compare(self, charts: Path, summary: pd.DataFrame, thr: float) -> Path:
        path = charts / "primary_vs_meta.png"
        prim = summary.loc[summary["system"] == "primary"]
        meta = summary.loc[
            (summary["system"] == "primary_plus_meta") & (summary["threshold"] == thr)
        ]
        metrics = ["n_trades", "win_rate", "expectancy", "profit_factor", "annual_return", "sharpe", "max_drawdown"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
        for ax, m in zip(axes, metrics):
            vals = []
            labels = []
            if not prim.empty:
                vals.append(float(prim[m].iloc[0]))
                labels.append("primary")
            if not meta.empty:
                vals.append(float(meta[m].iloc[0]))
                labels.append("meta")
            ax.bar(labels, vals, color=["#4C78A8", "#54A24B"][: len(vals)])
            ax.set_title(m, fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle(f"Primary vs Primary+Meta (thr={thr})")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _threshold(self, charts: Path, summary: pd.DataFrame) -> Path:
        path = charts / "threshold_metrics.png"
        meta = summary.loc[summary["system"] == "primary_plus_meta"].sort_values("threshold")
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        if not meta.empty:
            x = meta["threshold"]
            axes[0].plot(x, meta["expectancy"], marker="o")
            axes[0].set_title("Expectancy")
            axes[1].plot(x, meta["annual_return"], marker="o", color="#E45756")
            axes[1].set_title("Annual return (sum)")
            axes[2].plot(x, meta["n_trades"], marker="o", color="#F58518")
            axes[2].set_title("N trades")
            for ax in axes:
                ax.set_xlabel("threshold")
        fig.suptitle("Meta metrics vs threshold (report only)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _equity(self, charts: Path, equity: pd.DataFrame, thr: float) -> Path:
        path = charts / "equity_curve.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub = equity.loc[equity["threshold"] == thr].sort_values("valid_year")
        if not sub.empty:
            ax.plot(sub["valid_year"], sub["primary_equity"], marker="o", label="primary", color="#4C78A8")
            ax.plot(sub["valid_year"], sub["meta_equity"], marker="o", label="primary+meta", color="#54A24B")
            ax.legend()
        ax.set_xlabel("validation year")
        ax.set_ylabel("equity ($)")
        ax.set_title(f"Compounded equity from $80 (thr={thr})")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _wf(self, charts: Path, by_year: pd.DataFrame, thr: float) -> Path:
        path = charts / "wf_comparison.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        prim = by_year.loc[by_year["system"] == "primary"].sort_values("valid_year")
        meta = by_year.loc[
            (by_year["system"] == "primary_plus_meta") & (by_year["threshold"] == thr)
        ].sort_values("valid_year")
        if not prim.empty:
            ax.plot(prim["valid_year"], prim["annual_return"], marker="o", label="primary")
        if not meta.empty:
            ax.plot(meta["valid_year"], meta["annual_return"], marker="o", label="meta")
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        ax.set_title(f"Walk-forward annual return (thr={thr})")
        ax.set_xlabel("year")
        ax.set_ylabel("annual_return (sum net)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
