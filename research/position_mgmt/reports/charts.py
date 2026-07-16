"""Charts for position management."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PositionMgmtChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        table: pd.DataFrame,
        base_curve: pd.DataFrame,
        best_curve: pd.DataFrame,
        mc_detail: pd.DataFrame,
        base_panel: pd.DataFrame,
        best_panel: pd.DataFrame | None,
        ranks: list[dict],
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._scatter(charts, table)
        self._equity(charts, base_curve, best_curve)
        self._rank(charts, ranks)
        self._mc(charts, mc_detail)
        self._returns(charts, base_panel, best_panel)

    def _scatter(self, charts: Path, table: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        if not table.empty:
            for fam, g in table.groupby("family"):
                ax.scatter(g["max_drawdown"], g["cagr"], s=50, label=str(fam))
            base = table.loc[table["policy"] == "baseline"]
            if not base.empty:
                ax.scatter(base["max_drawdown"], base["cagr"], c="red", s=120, marker="*", label="baseline")
        ax.set_xlabel("Max DD")
        ax.set_ylabel("CAGR")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("Position management policies")
        fig.tight_layout()
        fig.savefig(charts / "family_scatter.png", dpi=120)
        plt.close(fig)

    def _equity(self, charts: Path, base: pd.DataFrame, best: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        for df, lab, c in ((base, "baseline", "#E45756"), (best, "best", "#4C78A8")):
            if df is None or df.empty:
                continue
            d = df.dropna(subset=["timestamp"]).copy()
            if d.empty:
                continue
            ax.plot(pd.to_datetime(d["timestamp"]), d["equity"], label=lab, color=c)
        ax.legend()
        ax.set_title("Equity")
        fig.tight_layout()
        fig.savefig(charts / "best_equity.png", dpi=120)
        plt.close(fig)

    def _rank(self, charts: Path, ranks: list[dict]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if ranks:
            names = [r["technique"] for r in ranks][::-1]
            scores = [r["score"] for r in ranks][::-1]
            ax.barh(names, scores, color="#54A24B")
        ax.set_title("Technique ranking (higher=better)")
        fig.tight_layout()
        fig.savefig(charts / "ranking.png", dpi=120)
        plt.close(fig)

    def _mc(self, charts: Path, mc: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if mc is not None and not mc.empty:
            ax.hist(mc["final_equity"], bins=30, color="#72B7B2", edgecolor="white")
        ax.set_title("Monte Carlo final equity (best)")
        fig.tight_layout()
        fig.savefig(charts / "monte_carlo.png", dpi=120)
        plt.close(fig)

    def _returns(self, charts: Path, base: pd.DataFrame, best: pd.DataFrame | None) -> None:
        fig, ax = plt.subplots(figsize=(5, 5))
        if best is not None and not best.empty and "net_return_sim" in best.columns:
            ax.scatter(base["net_return"], best["net_return_sim"], s=12, alpha=0.5)
            lims = [
                min(base["net_return"].min(), best["net_return_sim"].min()),
                max(base["net_return"].max(), best["net_return_sim"].max()),
            ]
            ax.plot(lims, lims, "--", color="gray")
        ax.set_xlabel("base net_return")
        ax.set_ylabel("sim net_return")
        ax.set_title("Sim vs base trade returns")
        fig.tight_layout()
        fig.savefig(charts / "sim_vs_base_returns.png", dpi=120)
        plt.close(fig)
