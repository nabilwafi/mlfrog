"""RunModelBenchmarkUseCase — fair algorithm comparison on fixed features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.feature_ablation.services.panel_builder import AblationPanelBuilder
from research.model_benchmark.entities.benchmark_result import BenchmarkModelResult
from research.model_benchmark.reports.benchmark_charts import BenchmarkChartBuilder
from research.model_benchmark.reports.benchmark_report import BenchmarkReportBuilder, build_rankings
from research.model_benchmark.repositories.benchmark_repository import BenchmarkRepository
from research.model_benchmark.services.benchmark_runner import BenchmarkWalkForwardRunner
from research.model_benchmark.services.model_catalog import BENCHMARK_MODELS

logger = logging.getLogger(__name__)


class RunModelBenchmarkUseCase:
    def __init__(
        self,
        repository: BenchmarkRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel_builder = AblationPanelBuilder()
        self._report_builder = BenchmarkReportBuilder()
        self._chart_builder = BenchmarkChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        side: str,
        feature_matrix_path: Path,
        feature_metadata_path: Path,
        labels: pd.DataFrame,
    ) -> tuple[list[BenchmarkModelResult], Path]:
        panel, all_features, _ = self._panel_builder.build(
            feature_matrix_path=feature_matrix_path,
            feature_metadata_path=feature_metadata_path,
            labels=labels,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
        )
        windows = self._parse_windows(self._cfg.get("windows"))
        algorithms = list(self._cfg.get("algorithms") or BENCHMARK_MODELS)
        runner = BenchmarkWalkForwardRunner(
            windows=windows,
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            model_params=dict(self._cfg.get("model_params") or {}),
        )

        results: list[BenchmarkModelResult] = []
        for algo in algorithms:
            logger.info("Benchmarking algorithm=%s | n_features=%s", algo, len(all_features))
            try:
                result = runner.run(panel, algo, feature_names=all_features)
                results.append(result)
            except Exception:
                logger.exception("Algorithm %s failed — recording empty result", algo)
                empty = BenchmarkModelResult(algorithm=algo, n_features=len(all_features))
                empty.aggregate()
                results.append(empty)

        summary = pd.DataFrame([r.summary_row() for r in results])
        metrics = pd.DataFrame([w.to_dict() for r in results for w in r.windows])
        rankings = build_rankings(summary)
        report_md = self._report_builder.build(
            results=results,
            summary=summary,
            rankings=rankings,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
        )
        out = self._repo.save(
            summary=summary,
            metrics=metrics,
            rankings=rankings,
            report_md=report_md,
        )
        self._chart_builder.write_all(out_dir=out, summary=summary, results=results)
        return results, out

    @staticmethod
    def _parse_windows(raw: Any) -> list[WalkForwardWindow] | None:
        if not raw:
            return None
        return [
            WalkForwardWindow(
                int(w["train_start_year"]),
                int(w["train_end_year"]),
                int(w["valid_year"]),
            )
            for w in raw
        ]
