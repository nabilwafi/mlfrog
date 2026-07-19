"""Charts for portfolio heat research."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioHeatChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        scored: pd.DataFrame,
        best_curve: pd.DataFrame,
        baseline_curve: pd.DataFrame,
        wf: pd.DataFrame,
        mc_detail: pd.DataFrame,
        best_log: pd.DataFrame,
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._scatter(charts, scored)
        self._equity(charts, baseline_curve, best_curve)
        self._wf(charts, wf)
        self._mc(charts, mc_detail)
        self._skips(charts, best_log)

    def _scatter(self, charts: Path, scored: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        if not scored.empty:
            acc = scored["accepted"].astype(bool) if "accepted" in scored.columns else np.zeros(len(scored), dtype=bool)
            ax.scatter(
                scored.loc[~acc, "max_drawdown"],
                scored.loc[~acc, "cagr"],
                c="#bbb",
                s=40,
                label="rejected",
            )
            ax.scatter(
                scored.loc[acc, "max_drawdown"],
                scored.loc[acc, "cagr"],
                c="#4C78A8",
                s=55,
                label="accepted",
            )
            base = scored.loc[scored["policy"] == "baseline_skip40_flat1"]
            if not base.empty:
                ax.scatter(base["max_drawdown"], base["cagr"], c="#E45756", s=90, marker="*", label="baseline")
        ax.set_xlabel("Max DD")
        ax.set_ylabel("CAGR")
        ax.legend(fontsize=8)
        ax.set_title("Policy scatter (CAGR vs DD)")
        fig.tight_layout()
        fig.savefig(charts / "policy_scatter.png", dpi=120)
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
        ax.set_title("Equity curve")
        fig.tight_layout()
        fig.savefig(charts / "best_equity.png", dpi=120)
        plt.close(fig)

    def _wf(self, charts: Path, wf: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not wf.empty:
            stand = wf.loc[wf["mode"] == "standalone_year"] if "mode" in wf.columns else wf
            if not stand.empty:
                ax.bar(stand["valid_year"].astype(str), stand["final_equity"], color="#54A24B")
        ax.set_title("Walk-forward final equity by year")
        fig.tight_layout()
        fig.savefig(charts / "yearly_wf.png", dpi=120)
        plt.close(fig)

    def _mc(self, charts: Path, mc: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not mc.empty:
            ax.hist(mc["final_equity"], bins=30, color="#72B7B2", edgecolor="white")
        ax.set_title("Monte Carlo final equity")
        fig.tight_layout()
        fig.savefig(charts / "monte_carlo.png", dpi=120)
        plt.close(fig)

    def _skips(self, charts: Path, log: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if log is not None and not log.empty and "skip_reason" in log.columns:
            sk = log.loc[log["skipped"].astype(bool), "skip_reason"].value_counts().head(12)
            if len(sk):
                ax.barh(sk.index.astype(str)[::-1], sk.values[::-1], color="#F58518")
        ax.set_title("Skip reasons (best policy)")
        fig.tight_layout()
        fig.savefig(charts / "skip_reasons.png", dpi=120)
        plt.close(fig)
