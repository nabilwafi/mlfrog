"""Charts for meta feature research."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetaFeatureChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        univariate: pd.DataFrame,
        group_importance: pd.DataFrame,
        stability: pd.DataFrame,
        interactions: pd.DataFrame,
        session_table: pd.DataFrame,
        answers: dict,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "importance": self._importance(charts, univariate),
            "group": self._group(charts, group_importance),
            "stability": self._stability(charts, stability),
            "interaction": self._interaction(charts, interactions),
            "wf_stability": self._wf_stability(charts, stability),
            "session": self._session(charts, session_table),
            "m15_h4": self._m15_h4(charts, answers),
        }

    def _importance(self, charts: Path, univariate: pd.DataFrame) -> Path:
        path = charts / "feature_importance.png"
        top = univariate.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"], top["mutual_info"], color="#4C78A8")
        ax.set_xlabel("mutual information")
        ax.set_title("Top features by MI (meta_label)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _group(self, charts: Path, group_importance: pd.DataFrame) -> Path:
        path = charts / "feature_group_importance.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not group_importance.empty:
            g = group_importance.sort_values("mean_mutual_info")
            ax.barh(g["group"], g["mean_mutual_info"], color="#54A24B")
        ax.set_xlabel("mean MI")
        ax.set_title("Feature group importance")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _stability(self, charts: Path, stability: pd.DataFrame) -> Path:
        path = charts / "feature_stability.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        if not stability.empty:
            top = stability.head(25)
            ax.scatter(top["mean_importance"], top["std_importance"], s=40, c="#E45756")
            for _, r in top.head(10).iterrows():
                ax.annotate(str(r["feature"])[:24], (r["mean_importance"], r["std_importance"]), fontsize=7)
        ax.set_xlabel("mean MI across WF")
        ax.set_ylabel("std MI across WF")
        ax.set_title("Feature stability (mean vs std)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _interaction(self, charts: Path, interactions: pd.DataFrame) -> Path:
        path = charts / "interaction_heatmap.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        if interactions.empty:
            ax.set_title("interactions (empty)")
        else:
            top = interactions.head(12).copy()
            labels = [f"{a[:12]}×{b[:12]}" for a, b in zip(top["feature_a"], top["feature_b"])]
            ax.barh(labels[::-1], top["mi_interaction"].iloc[::-1], color="#72B7B2")
            ax.set_xlabel("interaction MI")
            ax.set_title("Top interactions")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _wf_stability(self, charts: Path, stability: pd.DataFrame) -> Path:
        path = charts / "walkforward_feature_stability.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        if not stability.empty:
            top = stability.head(15).iloc[::-1]
            ax.barh(top["feature"], top["top5_count"], color="#F58518")
        ax.set_xlabel("top-5 frequency across WF windows")
        ax.set_title("Walk-forward rank persistence")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _session(self, charts: Path, session_table: pd.DataFrame) -> Path:
        path = charts / "session_analysis.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if session_table.empty:
            axes[0].set_title("session empty")
        else:
            # Prefer hour_of_day or session flags
            hour = session_table.loc[session_table["feature"] == "hour_of_day"]
            flags = session_table.loc[
                session_table["feature"].isin(
                    [
                        "session_asia",
                        "session_london",
                        "session_newyork",
                        "session_london_ny_overlap",
                    ]
                )
                & (session_table["value"] == 1)
            ]
            if not hour.empty:
                hour = hour.sort_values("value")
                axes[0].plot(hour["value"], hour["win_rate"], marker="o")
                axes[0].set_title("Win rate by hour (UTC)")
                axes[0].set_xlabel("hour")
            if not flags.empty:
                axes[1].bar(flags["feature"].str.replace("session_", ""), flags["win_rate"], color="#4C78A8")
                axes[1].set_title("Win rate when session=1")
                axes[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _m15_h4(self, charts: Path, answers: dict) -> Path:
        path = charts / "m15_vs_h4_gain.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["H1", "H4", "M15"]
        vals = [
            float(answers.get("mi_h1") or 0),
            float(answers.get("mi_h4") or 0),
            float(answers.get("mi_m15") or 0),
        ]
        ax.bar(labels, vals, color=["#4C78A8", "#54A24B", "#E45756"])
        ax.set_ylabel("mean mutual information")
        ax.set_title("Group mean MI: H1 vs H4 vs M15")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
