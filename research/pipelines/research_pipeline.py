"""Research pipeline — hypothesis evaluation over datasets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.entities.dataset import Dataset
from research.analyzers.feature_drift_analyzer import FeatureDriftAnalyzer
from research.analyzers.feature_stability_analyzer import FeatureStabilityAnalyzer
from research.analyzers.label_stability_analyzer import LabelStabilityAnalyzer
from research.analyzers.regime_analyzer import RegimeAnalyzer
from research.analyzers.walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardWindow
from research.analyzers._common import combine_splits
from research.entities.research_report import ResearchReport
from research.repositories.research_repository import ResearchRepository

logger = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


class ResearchPipeline:
    def __init__(self, repository: ResearchRepository) -> None:
        self._repo = repository
        self._stability = FeatureStabilityAnalyzer()
        self._drift = FeatureDriftAnalyzer()
        self._regime = RegimeAnalyzer()
        self._labels = LabelStabilityAnalyzer()

    def run(
        self,
        *,
        splits: dict[str, Dataset],
        algorithm: str,
        trainer_params: dict[str, Any],
        target_mode: str = "exclude_timeout",
        threshold: float = 0.5,
        random_seed: int = 42,
        dataset_version: str = "v1",
        windows: list[WalkForwardWindow] | None = None,
        label_frame: pd.DataFrame | None = None,
    ) -> tuple[ResearchReport, Path]:
        template = splits["train"]
        frame = combine_splits(splits)

        wf = WalkForwardAnalyzer(
            windows=windows,
            algorithm=algorithm,
            trainer_params=trainer_params,
            target_mode=target_mode,
            threshold=threshold,
            random_seed=random_seed,
            dataset_version=dataset_version,
        )
        wf_summary, wf_metrics, wf_md = wf.analyze(frame, template)
        stab_summary, stab_df = self._stability.analyze(frame)
        drift_summary, drift_df = self._drift.analyze(splits["train"], splits["validation"])
        regime_summary, regime_df, regime_md = self._regime.analyze(
            frame, label_frame=label_frame
        )
        label_summary, label_df = self._labels.analyze(frame, label_frame=label_frame)

        conclusions = self._conclude(
            wf_summary=wf_summary,
            stab_summary=stab_summary,
            drift_summary=drift_summary,
            regime_summary=regime_summary,
            label_summary=label_summary,
            regime_df=regime_df,
        )

        report = ResearchReport(
            symbol=template.symbol,
            timeframe=template.timeframe,
            side=template.side,
            algorithm=algorithm,
            created_at=datetime.now(tz=timezone.utc),
            walk_forward=wf_summary,
            feature_stability=stab_summary,
            feature_drift=drift_summary,
            regimes=regime_summary,
            label_stability=label_summary,
            conclusions=conclusions,
            metadata={
                "target_mode": target_mode,
                "threshold": threshold,
                "dataset_version": dataset_version,
                "feature_version": template.feature_version,
                "label_version": template.label_version,
            },
        )
        summary_md = self._research_summary_md(report)
        files: dict[str, str | pd.DataFrame] = {
            "walk_forward_report.md": wf_md,
            "walk_forward_metrics.csv": wf_metrics,
            "feature_stability.csv": stab_df,
            "feature_drift.csv": drift_df,
            "regime_report.md": regime_md,
            "regime_statistics.csv": regime_df,
            "label_stability.csv": label_df,
            "research_summary.md": summary_md,
        }
        out = self._repo.save(report, files=files)
        logger.info("Research complete | path=%s", out)
        return report, out

    def _conclude(
        self,
        *,
        wf_summary: dict[str, Any],
        stab_summary: dict[str, Any],
        drift_summary: dict[str, Any],
        regime_summary: dict[str, Any],
        label_summary: dict[str, Any],
        regime_df: pd.DataFrame,
    ) -> dict[str, Any]:
        mean_roc = wf_summary.get("mean_roc_auc")
        roc_std = wf_summary.get("std_roc_auc")
        n_severe = int(drift_summary.get("n_severe") or 0)
        label_drift = bool((label_summary.get("label_drift") or {}).get("detected"))
        top_drift = [r["feature"] for r in (drift_summary.get("top_drift_features") or [])[:5]]

        regime_spread = float("nan")
        if not regime_df.empty and regime_df["positive_rate"].notna().any():
            regime_spread = float(
                regime_df["positive_rate"].max() - regime_df["positive_rate"].min()
            )

        # Q1: feature quality vs regime
        feature_problem = bool(
            n_severe >= 3
            or (
                isinstance(drift_summary.get("max_psi"), float)
                and drift_summary["max_psi"] == drift_summary["max_psi"]
                and drift_summary["max_psi"] >= 0.25
            )
        )
        regime_problem = bool(
            regime_spread == regime_spread and regime_spread >= 0.08
        )
        if feature_problem and not regime_problem:
            failure_driver = "feature_quality_and_drift"
        elif regime_problem and not feature_problem:
            failure_driver = "market_regime"
        elif feature_problem and regime_problem:
            failure_driver = "both_feature_drift_and_regime"
        else:
            failure_driver = "weak_or_unstable_signal_overall"

        # Q6: global vs rolling
        unstable_wf = bool(
            isinstance(roc_std, float)
            and roc_std == roc_std
            and roc_std >= 0.03
        ) or bool(wf_summary.get("fail_years"))
        recommendation = (
            "prefer_rolling_retraining"
            if unstable_wf or feature_problem
            else "global_model_still_plausible_but_signal_weak"
        )

        # Q7: remove high-drift features?
        remove_high_drift = bool(n_severe >= 2)

        return {
            "failure_driver": failure_driver,
            "top_drifting_features": top_drift,
            "strong_years": list(wf_summary.get("strong_years") or []),
            "fail_years": list(wf_summary.get("fail_years") or []),
            "label_distribution_stable": not label_drift,
            "label_drift": label_summary.get("label_drift"),
            "model_strategy_recommendation": recommendation,
            "removing_high_drift_features_likely_helps": remove_high_drift,
            "mean_walk_forward_roc": mean_roc,
            "walk_forward_roc_std": roc_std,
            "regime_positive_rate_spread": regime_spread,
            "n_severe_drift_features": n_severe,
            "mean_feature_corr_drift": stab_summary.get("mean_corr_drift"),
            "notes": {
                "feature_problem": feature_problem,
                "regime_problem": regime_problem,
                "unstable_walk_forward": unstable_wf,
            },
        }

    def _research_summary_md(self, report: ResearchReport) -> str:
        c = report.conclusions
        lines = [
            f"# Research Summary — {report.symbol} {report.timeframe} {report.side}",
            "",
            f"- Algorithm under study: `{report.algorithm}`",
            f"- Created (UTC): `{report.created_at.isoformat()}`",
            "",
            "## 1. Feature quality or market regime?",
            "",
            f"**Verdict:** `{c.get('failure_driver')}`",
            "",
            f"- Severe-drift feature count: `{c.get('n_severe_drift_features')}`",
            f"- Regime positive-rate spread: `{_fmt(c.get('regime_positive_rate_spread'))}`",
            f"- Walk-forward mean ROC: `{_fmt(c.get('mean_walk_forward_roc'))}` "
            f"(std `{_fmt(c.get('walk_forward_roc_std'))}`)",
            "",
            "## 2. Which features drift the most?",
            "",
            f"`{c.get('top_drifting_features')}`",
            "",
            "See `feature_drift.csv` (sorted by PSI) and `feature_stability.csv`.",
            "",
            "## 3. Which years have the strongest predictive signal?",
            "",
            f"`{c.get('strong_years')}`",
            "",
            "## 4. Which years completely fail?",
            "",
            f"`{c.get('fail_years')}`",
            "",
            "## 5. Does label distribution remain stable?",
            "",
            f"**Stable:** `{c.get('label_distribution_stable')}`",
            "",
            f"Details: `{c.get('label_drift')}`",
            "",
            "## 6. One global model or rolling retraining?",
            "",
            f"**Recommendation:** `{c.get('model_strategy_recommendation')}`",
            "",
            "If walk-forward ROC is unstable or fail years appear, a single frozen global "
            "model is unlikely to generalize; rolling retraining is the safer research path.",
            "",
            "## 7. Would removing high-drift features likely improve generalization?",
            "",
            f"**Likely helpful:** `{c.get('removing_high_drift_features_likely_helps')}`",
            "",
            "This does not claim an automatic lift — it flags a hypothesis to test next. "
            "High-PSI features inject non-stationarity into any global fit.",
            "",
            "## Pointers",
            "",
            "- `walk_forward_report.md` / `walk_forward_metrics.csv`",
            "- `regime_report.md` / `regime_statistics.csv`",
            "- `feature_stability.csv` / `feature_drift.csv`",
            "- `label_stability.csv`",
            "",
        ]
        return "\n".join(lines)
