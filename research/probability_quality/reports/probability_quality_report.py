"""Markdown report for probability quality."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.probability_quality.entities import DistSummary
from research.probability_quality.services.analyzers import collapsed_flag, tradable_range


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


def build_report(
    *,
    symbol: str,
    timeframe: str,
    dist: dict[str, DistSummary],
    buckets: pd.DataFrame,
    deciles: pd.DataFrame,
    confidence: pd.DataFrame,
    relative_confidence: pd.DataFrame,
    mono: dict[str, dict],
    mono_decile: dict[str, dict],
    side_compare: dict[str, Any],
) -> str:
    lines = [
        f"# Probability Quality Report — {symbol} {timeframe}",
        "",
        "Sprint 14 — research only. No calibration fitting, no execution.",
        "",
        "Models: Long/Short **v2** (H1 + Sprint-12 selected H4 structure).",
        "",
        "## Probability distribution",
        "",
    ]
    for side in ("long", "short"):
        d = dist.get(side)
        if d is None:
            continue
        flag = " **collapsed**" if collapsed_flag(d) else ""
        lines.extend(
            [
                f"### {side.title()}{flag}",
                "",
                f"- n=`{d.n}` mean=`{_fmt(d.mean)}` std=`{_fmt(d.std)}`",
                f"- percentiles: p10=`{_fmt(d.p10)}` p25=`{_fmt(d.p25)}` "
                f"p50=`{_fmt(d.p50)}` p75=`{_fmt(d.p75)}` p90=`{_fmt(d.p90)}` "
                f"p95=`{_fmt(d.p95)}` p99=`{_fmt(d.p99)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Absolute bucket performance (brief buckets ≥0.50)",
            "",
            "_Note: with collapsed probabilities most mass sits below 0.50; "
            "empty high buckets are an expected finding._",
            "",
        ]
    )
    if buckets.empty:
        lines.append("_(empty)_")
    else:
        show = buckets[
            [
                "side",
                "bucket",
                "samples",
                "win_rate",
                "avg_label_return",
                "expectancy",
                "precision",
                "recall",
            ]
        ].copy()
        lines.append(show.to_string(index=False))

    lines.extend(["", "## Relative probability deciles (ranking where mass lives)", ""])
    if deciles.empty:
        lines.append("_(empty)_")
    else:
        show_d = deciles[
            [
                "side",
                "bucket",
                "bucket_lo",
                "bucket_hi",
                "samples",
                "win_rate",
                "expectancy",
            ]
        ].copy()
        lines.append(show_d.to_string(index=False))

    lines.extend(
        [
            "",
            "## 1. Are probabilities monotonic with outcomes?",
            "",
            "_Primary evidence: relative deciles (absolute ≥0.50 buckets are sparse)._",
            "",
        ]
    )
    for side in ("long", "short"):
        m = mono_decile.get(side, mono.get(side, {}))
        wr_m = bool(m.get("win_rate_monotonic"))
        rho = m.get("win_rate_spearman")
        lines.append(
            f"- **{side}:** `{'yes' if wr_m or (rho == rho and float(rho) > 0.5) else 'partial / no'}` "
            f"(Spearman win_rate vs decile=`{_fmt(rho)}`, strict mono=`{wr_m}`)"
        )

    lines.extend(
        [
            "",
            "## 2. Does higher probability create higher expectancy?",
            "",
        ]
    )
    for side in ("long", "short"):
        m = mono_decile.get(side, mono.get(side, {}))
        exp_m = bool(m.get("expectancy_monotonic"))
        rho = m.get("expectancy_spearman")
        lines.append(
            f"- **{side}:** `{'yes' if exp_m or (rho == rho and float(rho) > 0.5) else 'partial / no'}` "
            f"(Spearman expectancy vs decile=`{_fmt(rho)}`, strict mono=`{exp_m}`)"
        )

    lines.extend(
        [
            "",
            "## 3. Which probability range is tradable?",
            "",
            "### Absolute (≥0.50 brief buckets)",
            "",
        ]
    )
    for side in ("long", "short"):
        lines.append(f"- **{side}:** `{tradable_range(buckets, side)}`")
    lines.extend(["", "### Relative (top deciles vs side mean win_rate, expectancy>0)", ""])
    for side in ("long", "short"):
        lines.append(f"- **{side}:** `{tradable_range(deciles, side, min_win_rate=None)}`")

    long_better = bool(side_compare.get("long_stronger_quality"))
    lines.extend(
        [
            "",
            "## 4. Does Long have better confidence quality than Short?",
            "",
            f"**Answer:** `{'yes' if long_better else 'no / mixed'}`",
            "",
            str(side_compare.get("quality_rationale", "")),
            "",
            "### Absolute confidence (≥0.50 / 0.60 / 0.70)",
            "",
        ]
    )
    if confidence.empty:
        lines.append("_(empty)_")
    else:
        lines.append(confidence.to_string(index=False))

    lines.extend(["", "### Relative confidence (within-side terciles)", ""])
    if relative_confidence.empty:
        lines.append("_(empty)_")
    else:
        lines.append(relative_confidence.to_string(index=False))

    proceed = bool(side_compare.get("proceed_to_calibration"))
    lines.extend(
        [
            "",
            "## 5. Should the models proceed to calibration?",
            "",
            f"**Answer:** `{'yes — with caveats' if proceed else 'not yet'}`",
            "",
            str(side_compare.get("calibration_rationale", "")),
            "",
            "## Long vs Short summary",
            "",
            f"- Stronger probability quality: `{side_compare.get('stronger_side', 'n/a')}`",
            f"- Thresholds should differ: `{side_compare.get('thresholds_should_differ', 'n/a')}`",
            "",
            "## Charts",
            "",
            "- `charts/probability_histogram.png`",
            "- `charts/bucket_performance.png`",
            "- `charts/calibration_curve.png`",
            "- `charts/confidence_distribution.png`",
            "",
        ]
    )
    return "\n".join(lines)


def compare_sides(
    *,
    buckets: pd.DataFrame,
    deciles: pd.DataFrame,
    confidence: pd.DataFrame,
    relative_confidence: pd.DataFrame,
    mono: dict[str, dict],
    mono_decile: dict[str, dict],
    dist: dict[str, DistSummary],
) -> dict[str, Any]:
    """Heuristic quality: prefer decile Spearman + relative high-vs-low lift."""

    def _side_score(side: str) -> float:
        m = mono_decile.get(side, mono.get(side, {}))
        rho_w = float(m.get("win_rate_spearman") or 0.0)
        rho_e = float(m.get("expectancy_spearman") or 0.0)
        if rho_w != rho_w:
            rho_w = 0.0
        if rho_e != rho_e:
            rho_e = 0.0
        conf = relative_confidence.loc[relative_confidence["side"] == side]
        high = conf.loc[conf["confidence"] == "high"]
        low = conf.loc[conf["confidence"] == "low"]
        lift = 0.0
        if not high.empty and not low.empty:
            hw = float(high["win_rate"].iloc[0])
            lw = float(low["win_rate"].iloc[0])
            if hw == hw and lw == lw:
                lift = hw - lw
        return 0.4 * rho_w + 0.4 * rho_e + 0.2 * max(lift, 0.0) * 5.0

    long_s = _side_score("long")
    short_s = _side_score("short")
    long_stronger = long_s >= short_s

    thr_differ = True  # asymmetric feature packs
    long_high = relative_confidence.loc[
        (relative_confidence["side"] == "long") & (relative_confidence["confidence"] == "high")
    ]
    short_high = relative_confidence.loc[
        (relative_confidence["side"] == "short") & (relative_confidence["confidence"] == "high")
    ]
    gap_note = ""
    if not long_high.empty and not short_high.empty:
        lh = float(long_high["win_rate"].iloc[0])
        sh = float(short_high["win_rate"].iloc[0])
        if lh == lh and sh == sh:
            gap_note = f"; relative high-conf win_rate long=`{_fmt(lh)}` short=`{_fmt(sh)}`"

    # Proceed if ranking exists in deciles OR absolute tradable mass exists
    proceed = False
    for side in ("long", "short"):
        m = mono_decile.get(side, {})
        rho = m.get("win_rate_spearman")
        if rho is not None and rho == rho and float(rho) > 0.3:
            proceed = True
            break
        if "none" not in tradable_range(buckets, side):
            proceed = True
            break

    collapsed_sides = [s for s, d in dist.items() if collapsed_flag(d)]
    collapse_note = (
        f" Distributions collapsed on: {collapsed_sides}."
        if collapsed_sides
        else ""
    )

    return {
        "long_score": long_s,
        "short_score": short_s,
        "long_stronger_quality": long_stronger,
        "stronger_side": "long" if long_stronger else "short",
        "thresholds_should_differ": thr_differ,
        "quality_rationale": (
            f"Quality score long=`{_fmt(long_s)}` short=`{_fmt(short_s)}` "
            f"(decile Spearman + relative tercile win-rate lift). "
            f"long relative tradable=`{tradable_range(deciles, 'long', min_win_rate=None)}`; "
            f"short relative tradable=`{tradable_range(deciles, 'short', min_win_rate=None)}`"
            + gap_note
            + collapse_note
        ),
        "proceed_to_calibration": proceed,
        "calibration_rationale": (
            (
                "Yes if relative ranking exists: calibration can stretch usable probability "
                "levels for thresholding. Caveat: absolute >=0.50 buckets are nearly empty — "
                "calibration alone will not invent high-confidence mass; combine with "
                "threshold redesign on the empirical distribution (e.g. percentile cuts)."
            )
            if proceed
            else "Ranking signal too weak — calibration will not invent tradable structure."
        )
        + collapse_note,
    }
