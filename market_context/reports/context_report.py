"""Markdown report for the market context layer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_context.entities.context_feature import ContextFeatureSpec


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


class ContextReportBuilder:
    def build(
        self,
        *,
        symbol: str,
        base_tf: str,
        context_tf: str,
        specs: list[ContextFeatureSpec],
        joined: pd.DataFrame,
        h4_rows: int,
        dataset_v2_sides: list[str],
    ) -> str:
        cols = [s.name for s in specs]
        coverage = float(joined[cols[0]].notna().mean()) if cols and not joined.empty else 0.0
        lines = [
            f"# Market Context Report — {symbol} {context_tf} → {base_tf}",
            "",
            "Sprint 9 — Multi-Timeframe Context Layer (not entry, not execution).",
            "",
            "## Pipeline",
            "",
            f"1. Load `{context_tf}` OHLCV",
            f"2. Build H4 context features ({len(specs)} signals)",
            f"3. Causal asof-join onto every `{base_tf}` candle (latest **completed** bar only)",
            "4. Optionally write dataset v2 (features + context + labels) — **no H1 retrain**",
            "",
            "## Coverage",
            "",
            f"- H4 context rows (post warmup): `{h4_rows}`",
            f"- Base `{base_tf}` rows: `{len(joined)}`",
            f"- Rows with context attached: `{int(joined[cols[0]].notna().sum()) if cols else 0}` "
            f"({_fmt(coverage * 100, 1)}%)",
            "",
            "## Context Features",
            "",
        ]
        for s in specs:
            lines.append(f"- `{s.name}` ({s.category}): {s.description} range={s.value_range}")

        lines.extend(["", "## Distribution (attached rows)", ""])
        if cols and not joined.empty:
            ok = joined.dropna(subset=cols)
            for c in cols:
                lines.append(
                    f"- `{c}`: mean=`{_fmt(float(ok[c].mean()))}` "
                    f"std=`{_fmt(float(ok[c].std(ddof=1)) if len(ok) > 1 else 0.0)}` "
                    f"min=`{_fmt(float(ok[c].min()))}` max=`{_fmt(float(ok[c].max()))}`"
                )

        lines.extend(
            [
                "",
                "## Dataset v2",
                "",
            ]
        )
        if dataset_v2_sides:
            lines.append(
                f"Written for sides: `{', '.join(dataset_v2_sides)}` "
                f"under `artifacts/datasets/{symbol}/{base_tf}/{{side}}/v2/`."
            )
        else:
            lines.append("Skipped (disabled in config).")

        lines.extend(
            [
                "",
                "## Leakage rule",
                "",
                f"A `{context_tf}` bar with open-time `T` becomes available at `T + bar_duration`. "
                f"Each `{base_tf}` row receives only context with `available_at <= timestamp`.",
                "",
                "## Artifacts",
                "",
                "- `context.parquet`",
                "- `context_metadata.json`",
                "- `context_report.md`",
                "- `context_features.csv`",
                "",
            ]
        )
        return "\n".join(lines)
