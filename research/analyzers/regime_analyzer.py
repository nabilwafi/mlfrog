"""Market regime classification from existing feature columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


from research.analyzers._common import frame_to_markdown


class RegimeAnalyzer:
    """Exclusive regimes: High Volatility, Low Volatility, Bull, Bear, Sideways."""

    def analyze(
        self,
        frame: pd.DataFrame,
        *,
        label_frame: pd.DataFrame | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame, str]:
        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = self._attach_labels(df, label_frame)
        df["regime"] = self._classify(df)

        rows = []
        for regime in ("Bull", "Bear", "Sideways", "High Volatility", "Low Volatility"):
            part = df.loc[df["regime"] == regime]
            lab = part["label"].astype(int) if "label" in part.columns and len(part) else pd.Series(dtype=int)
            n = max(len(part), 1)
            rows.append(
                {
                    "regime": regime,
                    "n_samples": int(len(part)),
                    "positive_rate": float((lab == 1).mean()) if len(lab) else float("nan"),
                    "negative_rate": float((lab == -1).mean()) if len(lab) else float("nan"),
                    "timeout_rate": float((lab == 0).mean()) if len(lab) else float("nan"),
                    "avg_atr": self._mean_col(part, ("atr_14", "atr")),
                    "avg_adx": self._mean_col(part, ("adx_14", "adx")),
                    "avg_return": self._mean_col(part, ("returns_1", "log_returns_1")),
                    "avg_holding_bars": (
                        float(part["holding_bars"].mean())
                        if "holding_bars" in part.columns and len(part)
                        else float("nan")
                    ),
                }
            )
        stats = pd.DataFrame(rows)
        summary = {
            "regime_counts": {r["regime"]: r["n_samples"] for r in rows},
            "highest_positive_regime": (
                stats.loc[stats["positive_rate"].idxmax(), "regime"]
                if stats["positive_rate"].notna().any()
                else None
            ),
            "lowest_positive_regime": (
                stats.loc[stats["positive_rate"].idxmin(), "regime"]
                if stats["positive_rate"].notna().any()
                else None
            ),
        }
        md = self._markdown(summary, stats)
        return summary, stats, md

    def _classify(self, df: pd.DataFrame) -> pd.Series:
        atr = self._series(df, ("atr_14", "atr", "atr_ratio_atr_14"))
        adx = self._series(df, ("adx_14", "adx"))
        ret = self._series(df, ("returns_1", "log_returns_1"))
        trend = self._series(
            df,
            (
                "ema_distance_ema_20_close",
                "ema_distance_ema_50_close",
            ),
        )
        if trend.isna().all() and {"ema_20_close", "ema_50_close"}.issubset(df.columns):
            trend = df["ema_20_close"].astype(float) - df["ema_50_close"].astype(float)

        atr_q75 = float(atr.quantile(0.75)) if atr.notna().any() else float("inf")
        atr_q25 = float(atr.quantile(0.25)) if atr.notna().any() else float("-inf")
        adx_med = float(adx.median()) if adx.notna().any() else 20.0

        regimes = []
        for i in range(len(df)):
            a = atr.iloc[i]
            d = adx.iloc[i] if i < len(adx) else np.nan
            t = trend.iloc[i] if i < len(trend) else np.nan
            r = ret.iloc[i] if i < len(ret) else np.nan
            if a == a and a >= atr_q75:
                regimes.append("High Volatility")
            elif a == a and a <= atr_q25:
                regimes.append("Low Volatility")
            elif (t == t and t > 0 and (d != d or d >= adx_med)) or (
                t != t and r == r and r > 0
            ):
                regimes.append("Bull")
            elif (t == t and t < 0 and (d != d or d >= adx_med)) or (
                t != t and r == r and r < 0
            ):
                regimes.append("Bear")
            else:
                regimes.append("Sideways")
        return pd.Series(regimes, index=df.index)

    @staticmethod
    def _series(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name].astype(float)
        return pd.Series(np.nan, index=df.index)

    @staticmethod
    def _mean_col(part: pd.DataFrame, names: tuple[str, ...]) -> float:
        for name in names:
            if name in part.columns and len(part):
                return float(part[name].astype(float).mean())
        return float("nan")

    @staticmethod
    def _attach_labels(
        df: pd.DataFrame, label_frame: pd.DataFrame | None
    ) -> pd.DataFrame:
        if label_frame is None or label_frame.empty:
            return df
        lab = label_frame.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        cols = ["timestamp", "label"]
        for extra in ("holding_bars", "exit_reason"):
            if extra in lab.columns:
                cols.append(extra)
        lab = lab[cols].drop_duplicates(subset=["timestamp"], keep="last")
        # prefer dataset label if already present; still merge holding/exit
        keep = [c for c in lab.columns if c != "label" or "label" not in df.columns]
        if "label" in df.columns:
            lab = lab.drop(columns=["label"], errors="ignore")
        return df.merge(lab, on="timestamp", how="left")

    def _markdown(self, summary: dict[str, Any], stats: pd.DataFrame) -> str:
        lines = [
            "# Regime Report",
            "",
            f"- Counts: `{summary.get('regime_counts')}`",
            f"- Highest positive-rate regime: `{summary.get('highest_positive_regime')}`",
            f"- Lowest positive-rate regime: `{summary.get('lowest_positive_regime')}`",
            "",
            "## Statistics",
            "",
            frame_to_markdown(stats),
            "",
            "Classification uses ATR percentiles for High/Low Volatility, then ",
            "EMA distance / returns with ADX for Bull / Bear, else Sideways.",
            "",
        ]
        return "\n".join(lines)
