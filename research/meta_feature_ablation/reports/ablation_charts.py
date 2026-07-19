"""Charts for meta feature ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetaAblationChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        pearson: pd.DataFrame,
        stage_summary: pd.DataFrame,
        window_metrics: pd.DataFrame,
        stability: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> dict[str, Path]:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        return {
            "corr": self._corr(charts, pearson),
            "ablation": self._ablation_curve(charts, stage_summary),
            "metric": self._metric_progression(charts, stage_summary),
            "wf": self._wf_progression(charts, window_metrics),
            "importance": self._importance_stability(charts, stability),
            "interaction": self._interaction(charts, interactions),
            "annual": self._annual_return(charts, stage_summary),
        }

    def _corr(self, charts: Path, pearson: pd.DataFrame) -> Path:
        path = charts / "feature_correlation_heatmap.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        if pearson.empty:
            ax.set_title("correlation (empty)")
        else:
            # Cap display size for readability
            cols = list(pearson.columns)
            if len(cols) > 35:
                cols = cols[:35]
                mat = pearson.loc[cols, cols]
            else:
                mat = pearson
            im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
            ax.set_yticklabels(mat.index, fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Pearson correlation (meta features)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _ablation_curve(self, charts: Path, stage_summary: pd.DataFrame) -> Path:
        path = charts / "ablation_curve.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not stage_summary.empty:
            x = stage_summary["stage_id"].astype(str)
            ax.plot(x, stage_summary["expectancy_mean"], marker="o", color="#4C78A8", label="E mean")
            ax.fill_between(
                range(len(x)),
                stage_summary["expectancy_mean"] - stage_summary["expectancy_std"].fillna(0),
                stage_summary["expectancy_mean"] + stage_summary["expectancy_std"].fillna(0),
                color="#4C78A8",
                alpha=0.2,
            )
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=25, ha="right", fontsize=8)
            ax.legend()
        ax.set_ylabel("expectancy (mean across WF)")
        ax.set_title("Ablation curve (fixed LGBM, thr=0.5)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _metric_progression(self, charts: Path, stage_summary: pd.DataFrame) -> Path:
        path = charts / "metric_progression.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if not stage_summary.empty:
            x = range(len(stage_summary))
            labels = stage_summary["stage_id"].astype(str)
            axes[0].plot(x, stage_summary["roc_mean"], marker="o", color="#54A24B")
            axes[0].set_title("ROC AUC mean")
            axes[1].plot(x, stage_summary["f1_mean"], marker="o", color="#E45756")
            axes[1].set_title("F1 mean")
            axes[2].plot(x, stage_summary["ece_mean"], marker="o", color="#F58518")
            axes[2].set_title("ECE mean")
            for ax in axes:
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        fig.suptitle("Classification metric progression")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _wf_progression(self, charts: Path, window_metrics: pd.DataFrame) -> Path:
        path = charts / "wf_metric_progression.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        if not window_metrics.empty:
            for stage, g in window_metrics.groupby("stage_id", sort=False):
                g = g.sort_values("valid_year")
                ax.plot(g["valid_year"], g["roc_auc"], marker="o", label=str(stage))
            ax.legend(fontsize=7)
            ax.set_xlabel("validation year")
            ax.set_ylabel("ROC AUC")
        ax.set_title("Walk-forward ROC by stage")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _importance_stability(self, charts: Path, stability: pd.DataFrame) -> Path:
        path = charts / "importance_stability.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        if not stability.empty:
            top = stability.head(25)
            ax.scatter(top["gain_mean"], top["gain_std"], s=45, c="#72B7B2")
            for _, r in top.head(12).iterrows():
                ax.annotate(str(r["feature"])[:22], (r["gain_mean"], r["gain_std"]), fontsize=7)
            ax.set_xlabel("mean gain importance")
            ax.set_ylabel("std gain across WF")
        ax.set_title("Importance stability (full stage)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _interaction(self, charts: Path, interactions: pd.DataFrame) -> Path:
        path = charts / "interaction_matrix.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        if interactions.empty:
            ax.set_title("interactions (empty)")
        else:
            feats = sorted(
                set(interactions["feature_a"]).union(set(interactions["feature_b"]))
            )
            idx = {f: i for i, f in enumerate(feats)}
            mat = np.full((len(feats), len(feats)), np.nan)
            for _, r in interactions.iterrows():
                i, j = idx[r["feature_a"]], idx[r["feature_b"]]
                v = float(r["mi_lift"]) if r["mi_lift"] == r["mi_lift"] else 0.0
                mat[i, j] = v
                mat[j, i] = v
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(feats)))
            ax.set_yticks(range(len(feats)))
            ax.set_xticklabels([f[:14] for f in feats], rotation=90, fontsize=6)
            ax.set_yticklabels([f[:14] for f in feats], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MI lift")
            ax.set_title("Top-pair interaction MI lift")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _annual_return(self, charts: Path, stage_summary: pd.DataFrame) -> Path:
        path = charts / "annual_return_progression.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not stage_summary.empty:
            x = stage_summary["stage_id"].astype(str)
            ax.bar(range(len(x)), stage_summary["annual_return_sum"], color="#4C78A8")
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("sum annual_return across WF years")
        ax.set_title("Annual return progression (additive trade PnL)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
