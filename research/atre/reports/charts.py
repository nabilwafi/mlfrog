"""ATRE charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AtreChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        lib: list,
        recovery_prob: pd.DataFrame,
        underwater: pd.DataFrame,
        table: pd.DataFrame,
        events: pd.DataFrame,
        importance: pd.DataFrame,
        wf: pd.DataFrame,
        mc: pd.DataFrame,
        base_curve: pd.DataFrame,
        best_curve: pd.DataFrame,
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._mae(charts, lib)
        self._uw(charts, underwater)
        self._rec(charts, recovery_prob)
        self._early(charts, table)
        self._score(charts, events)
        self._imp(charts, importance)
        self._wf(charts, wf)
        self._mc(charts, mc)
        self._eq(charts, base_curve, best_curve)

    def _mae(self, charts: Path, lib: list) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [p.get("max_adverse_atr", 0) for p in lib if p.get("ok")]
        if xs:
            ax.hist(xs, bins=20, color="#4C78A8", edgecolor="white")
        ax.set_title("MAE distribution (ATR)")
        fig.tight_layout()
        fig.savefig(charts / "mae_distribution.png", dpi=120)
        plt.close(fig)

    def _uw(self, charts: Path, underwater: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not underwater.empty:
            ax.plot(underwater["underwater_bars"], underwater["tp_rate"], marker="o")
            ax.set_ylabel("TP rate")
        ax.set_title("Time Under Water vs TP rate")
        fig.tight_layout()
        fig.savefig(charts / "time_underwater.png", dpi=120)
        plt.close(fig)

    def _rec(self, charts: Path, recovery_prob: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not recovery_prob.empty:
            ax.plot(recovery_prob["cond_mae_atr"], recovery_prob["p_tp"], marker="o", label="P(TP)")
            ax.plot(recovery_prob["cond_mae_atr"], recovery_prob["p_sl"], marker="s", label="P(SL)")
            ax.legend()
        ax.set_title("Recovery probability curve")
        fig.tight_layout()
        fig.savefig(charts / "recovery_probability.png", dpi=120)
        plt.close(fig)

    def _early(self, charts: Path, table: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        sub = table.loc[table["family"].isin(["baseline", "early_exit", "score_exit"])].head(12)
        if not sub.empty:
            ax.barh(sub["policy"].astype(str)[::-1], sub["cagr"][::-1], color="#54A24B")
        ax.set_title("Early exit comparison (CAGR)")
        fig.tight_layout()
        fig.savefig(charts / "early_exit_compare.png", dpi=120)
        plt.close(fig)

    def _score(self, charts: Path, events: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if events is not None and not events.empty and "recovery_score" in events.columns:
            ax.hist(events["recovery_score"].dropna(), bins=15, color="#F58518", edgecolor="white")
        ax.set_title("Recovery Score distribution")
        fig.tight_layout()
        fig.savefig(charts / "recovery_score_dist.png", dpi=120)
        plt.close(fig)

    def _imp(self, charts: Path, importance: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        if importance is not None and not importance.empty:
            s = importance.head(12).iloc[::-1]
            ax.barh(s["feature"], s["mean_abs_coef"], color="#72B7B2")
        ax.set_title("Recovery score feature importance")
        fig.tight_layout()
        fig.savefig(charts / "feature_importance.png", dpi=120)
        plt.close(fig)

    def _wf(self, charts: Path, wf: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not wf.empty and "final_equity" in wf.columns:
            stand = wf.loc[wf["mode"] == "standalone_year"] if "mode" in wf.columns else wf
            if not stand.empty:
                ax.bar(stand["valid_year"].astype(str), stand["final_equity"], color="#4C78A8")
        ax.set_title("Walk-forward final equity")
        fig.tight_layout()
        fig.savefig(charts / "walk_forward.png", dpi=120)
        plt.close(fig)

    def _mc(self, charts: Path, mc: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if mc is not None and not mc.empty:
            ax.hist(mc["final_equity"], bins=30, color="#72B7B2", edgecolor="white")
        ax.set_title("Monte Carlo final equity")
        fig.tight_layout()
        fig.savefig(charts / "monte_carlo.png", dpi=120)
        plt.close(fig)

    def _eq(self, charts: Path, base: pd.DataFrame, best: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        for df, lab, c in ((base, "baseline", "#E45756"), (best, "best", "#4C78A8")):
            if df is None or df.empty:
                continue
            d = df.dropna(subset=["timestamp"])
            if d.empty:
                continue
            ax.plot(pd.to_datetime(d["timestamp"]), d["equity"], label=lab, color=c)
        ax.legend()
        ax.set_title("Equity curve")
        fig.tight_layout()
        fig.savefig(charts / "equity_curve.png", dpi=120)
        plt.close(fig)
