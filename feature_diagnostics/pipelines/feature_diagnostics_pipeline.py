"""Orchestrate feature diagnostics analyzers and write research artifacts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from feature_diagnostics.analyzers._common import (
    align_features_labels,
    load_feature_matrix,
    load_feature_metadata,
)
from feature_diagnostics.analyzers.correlation_analyzer import CorrelationAnalyzer
from feature_diagnostics.analyzers.mutual_information_analyzer import MutualInformationAnalyzer
from feature_diagnostics.analyzers.permutation_analyzer import PermutationImportanceAnalyzer
from feature_diagnostics.analyzers.redundancy_analyzer import RedundancyAnalyzer
from feature_diagnostics.analyzers.selection_analyzer import SelectionAnalyzer
from feature_diagnostics.analyzers.shap_analyzer import ShapImportanceAnalyzer, ShapWindow
from feature_diagnostics.analyzers.stability_analyzer import (
    StabilityAnalyzer,
    assign_regimes,
)
from feature_diagnostics.entities.feature_diagnostics_report import FeatureDiagnosticsReport
from feature_diagnostics.repositories.feature_diagnostics_repository import (
    FeatureDiagnosticsRepository,
)

logger = logging.getLogger(__name__)


class FeatureDiagnosticsPipeline:
    def __init__(self, repository: FeatureDiagnosticsRepository) -> None:
        self._repo = repository
        self._corr = CorrelationAnalyzer()
        self._redundancy = RedundancyAnalyzer()
        self._mi = MutualInformationAnalyzer()
        self._stability = StabilityAnalyzer()
        self._perm = PermutationImportanceAnalyzer()
        self._shap = ShapImportanceAnalyzer()
        self._selection = SelectionAnalyzer()

    def run(
        self,
        *,
        symbol: str,
        timeframe: str,
        side: str,
        feature_matrix_path: Path,
        feature_metadata_path: Path,
        labels: pd.DataFrame,
        target_mode: str = "exclude_timeout",
        corr_threshold: float = 0.95,
        shap_windows: list[ShapWindow] | None = None,
        random_state: int = 42,
    ) -> tuple[FeatureDiagnosticsReport, Path]:
        features = load_feature_matrix(feature_matrix_path)
        metadata = load_feature_metadata(feature_metadata_path)
        x, y, ts = align_features_labels(features, labels, target_mode=target_mode)
        feat_names = list(x.columns)
        logger.info(
            "Aligned features/labels | rows=%s features=%s",
            len(x),
            len(feat_names),
        )

        corr_summary, corr_df = self._corr.analyze(x)
        red_summary, red_df = self._redundancy.analyze(
            corr_df, threshold=corr_threshold, method="pearson"
        )
        mi_summary, mi_df = self._mi.analyze(
            x, y, random_state=random_state, metadata=metadata
        )
        year_summary, year_stab = self._stability.analyze_yearly(x, ts)
        regimes = assign_regimes(x)
        regime_stab = self._stability.analyze_regimes(x, regime=regimes)
        # merge regime max psi into yearly stability table
        if not regime_stab.empty:
            regime_max = (
                regime_stab.groupby("feature")["psi_vs_global"]
                .max()
                .rename("max_regime_psi")
            )
            year_stab = year_stab.merge(regime_max, on="feature", how="left")
        else:
            year_stab["max_regime_psi"] = float("nan")

        perm_summary, perm_df = self._perm.analyze(
            x, y, random_state=random_state, metadata=metadata
        )

        shap_frame = x.copy()
        shap_frame["timestamp"] = ts.values
        shap_frame["_y"] = y
        if shap_windows is not None:
            self._shap = ShapImportanceAnalyzer(windows=shap_windows, random_state=random_state)
        shap_summary, shap_df = self._shap.analyze(
            shap_frame, feat_names, metadata=metadata
        )

        selection_summary, candidates = self._selection.analyze(
            features=feat_names,
            mi=mi_df,
            permutation=perm_df,
            shap=shap_df,
            stability=year_stab,
            redundancy_groups=list(red_summary.get("groups") or []),
            metadata=metadata,
            corr_threshold=corr_threshold,
        )

        report = FeatureDiagnosticsReport(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            created_at=datetime.now(tz=timezone.utc),
            keep=list(selection_summary["keep"]),
            remove=list(selection_summary["remove"]),
            correlated_groups=list(selection_summary["correlated_groups"]),
            drift_sensitive=list(selection_summary["drift_sensitive"]),
            stable=list(selection_summary["stable"]),
            candidate_library_v3=list(selection_summary["candidate_library_v3"]),
            category_power=dict(selection_summary["category_power"]),
            summaries={
                "correlation": corr_summary,
                "redundancy": {
                    "n_groups": red_summary.get("n_groups"),
                    "threshold": red_summary.get("threshold"),
                },
                "mutual_information": mi_summary,
                "yearly_stability": year_summary,
                "permutation": perm_summary,
                "shap": shap_summary,
            },
            metadata={
                "target_mode": target_mode,
                "n_rows": int(len(x)),
                "n_features": int(len(feat_names)),
                "corr_threshold": corr_threshold,
                "feature_matrix": str(feature_matrix_path),
            },
        )
        markdown = self._build_report(report, regime_stab)
        tables = {
            "feature_correlation.csv": corr_df,
            "feature_mutual_information.csv": mi_df,
            "feature_permutation_importance.csv": perm_df,
            "feature_shap_importance.csv": shap_df,
            "feature_stability.csv": year_stab,
            "feature_redundancy.csv": red_df,
            "feature_selection_candidates.csv": candidates,
        }
        # attach regime detail as extra columns already in stability; also save regime rows appended? 
        # User asked feature_stability.csv only — yearly table includes max_regime_psi.
        out = self._repo.save(report, tables=tables, markdown=markdown)
        return report, out

    def _build_report(
        self,
        report: FeatureDiagnosticsReport,
        regime_stab: pd.DataFrame,
    ) -> str:
        lines = [
            f"# Feature Diagnostics Report — {report.symbol} {report.timeframe} {report.side}",
            "",
            f"- Created (UTC): `{report.created_at.isoformat()}`",
            f"- Rows analyzed: `{report.metadata.get('n_rows')}`",
            f"- Features analyzed: `{report.metadata.get('n_features')}`",
            f"- Target mode: `{report.metadata.get('target_mode')}`",
            "",
            "## Features to keep",
            "",
        ]
        for f in report.keep:
            lines.append(f"- `{f}`")
        lines.extend(["", "## Features to remove", ""])
        if not report.remove:
            lines.append("- _(none)_")
        else:
            for f in report.remove:
                lines.append(f"- `{f}`")

        lines.extend(["", "## Highly correlated feature groups", ""])
        if not report.correlated_groups:
            lines.append("- _(none above threshold)_")
        else:
            for i, g in enumerate(report.correlated_groups, start=1):
                lines.append(f"- Group {i}: `{', '.join(g)}`")

        lines.extend(["", "## Drift-sensitive features (yearly PSI)", ""])
        for f in report.drift_sensitive:
            lines.append(f"- `{f}`")
        if not report.drift_sensitive:
            lines.append("- _(none)_")

        lines.extend(["", "## Stable features", ""])
        for f in report.stable:
            lines.append(f"- `{f}`")
        if not report.stable:
            lines.append("- _(none)_")

        lines.extend(["", "## Category predictive power (sum MI on keep set)", ""])
        if not report.category_power:
            lines.append("- _(n/a)_")
        else:
            for cat, score in report.category_power.items():
                lines.append(f"- `{cat}`: `{score:.6f}`")

        lines.extend(
            [
                "",
                "## Candidate Feature Library v3",
                "",
                "Stable + kept features (exclude high-PSI and redundant losers).",
                "",
            ]
        )
        for f in report.candidate_library_v3:
            lines.append(f"- `{f}`")
        if not report.candidate_library_v3:
            lines.append("- _(empty — review thresholds)_")

        # regime highlights
        lines.extend(["", "## Regime instability highlights", ""])
        if regime_stab.empty:
            lines.append("- _(no regime rows)_")
        else:
            top = regime_stab.head(10)
            for _, r in top.iterrows():
                lines.append(
                    f"- `{r['feature']}` in `{r['regime']}` PSI=`{r['psi_vs_global']:.4f}`"
                )

        lines.extend(
            [
                "",
                "## Research answers",
                "",
                "1. Correlation pairs: see `feature_correlation.csv` (Pearson & Spearman).",
                "2. Redundancy groups: see `feature_redundancy.csv`.",
                "3. Mutual information: see `feature_mutual_information.csv`.",
                "4. Yearly PSI instability: see `feature_stability.csv` (`max_psi`, `drift_flag`).",
                "5. Regime instability: `max_regime_psi` in `feature_stability.csv`.",
                "6. Walk-forward SHAP: see `feature_shap_importance.csv`.",
                "7. Permutation importance: see `feature_permutation_importance.csv`.",
                "8. Removal list: section above + `feature_selection_candidates.csv`.",
                "9. Category power: section above.",
                "",
            ]
        )
        return "\n".join(lines)
