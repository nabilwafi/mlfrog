"""Label distribution stability across year / quarter / month."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.analyzers._common import safe_ks


class LabelStabilityAnalyzer:
    def analyze(
        self,
        frame: pd.DataFrame,
        *,
        label_frame: pd.DataFrame | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        base = self._base_frame(frame, label_frame)
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        base["year"] = base["timestamp"].dt.year
        base["quarter"] = (
            base["timestamp"].dt.year.astype(str)
            + "Q"
            + base["timestamp"].dt.quarter.astype(str)
        )
        base["month"] = base["timestamp"].dt.strftime("%Y-%m")
        lab = base["label"].astype(int)

        rows: list[dict[str, Any]] = []
        for grain, col in (("year", "year"), ("quarter", "quarter"), ("month", "month")):
            for key, part in base.groupby(col):
                y = part["label"].astype(int)
                n = max(len(y), 1)
                row: dict[str, Any] = {
                    "grain": grain,
                    "period": str(key),
                    "n_samples": int(len(y)),
                    "positive_rate": float((y == 1).mean()),
                    "negative_rate": float((y == -1).mean()),
                    "timeout_rate": float((y == 0).mean()),
                }
                if "holding_bars" in part.columns:
                    hb = part["holding_bars"].astype(float)
                    row["holding_bars_mean"] = float(hb.mean())
                    row["holding_bars_median"] = float(hb.median())
                else:
                    row["holding_bars_mean"] = float("nan")
                    row["holding_bars_median"] = float("nan")
                if "exit_reason" in part.columns:
                    er = part["exit_reason"].astype(str).value_counts(normalize=True)
                    row["exit_tp"] = float(er.get("TP", 0.0))
                    row["exit_sl"] = float(er.get("SL", 0.0))
                    row["exit_timeout"] = float(er.get("TIMEOUT", 0.0))
                else:
                    row["exit_tp"] = float((y == 1).sum() / n)
                    row["exit_sl"] = float((y == -1).sum() / n)
                    row["exit_timeout"] = float((y == 0).sum() / n)
                rows.append(row)

        stability = pd.DataFrame(rows)

        yearly = stability.loc[stability["grain"] == "year"].sort_values("period")
        drift = self._detect_drift(yearly)
        summary = {
            "overall": {
                "positive_rate": float((lab == 1).mean()),
                "negative_rate": float((lab == -1).mean()),
                "timeout_rate": float((lab == 0).mean()),
            },
            "label_drift": drift,
            "n_periods": {
                "year": int((stability["grain"] == "year").sum()),
                "quarter": int((stability["grain"] == "quarter").sum()),
                "month": int((stability["grain"] == "month").sum()),
            },
        }
        return summary, stability

    def _base_frame(
        self, frame: pd.DataFrame, label_frame: pd.DataFrame | None
    ) -> pd.DataFrame:
        if label_frame is not None and not label_frame.empty:
            lab = label_frame.copy()
            lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
            keep = ["timestamp", "label"]
            for c in ("holding_bars", "exit_reason"):
                if c in lab.columns:
                    keep.append(c)
            return lab[keep].drop_duplicates(subset=["timestamp"], keep="last")
        out = frame[["timestamp", "label"]].copy()
        return out

    def _detect_drift(self, yearly: pd.DataFrame) -> dict[str, Any]:
        if yearly.empty or len(yearly) < 2:
            return {"detected": False, "reason": "insufficient years"}
        rates = yearly["positive_rate"].to_numpy(dtype=float)
        first = yearly.iloc[0]
        last = yearly.iloc[-1]
        abs_diff = float(abs(float(last["positive_rate"]) - float(first["positive_rate"])))
        max_diff = float(np.nanmax(rates) - np.nanmin(rates))
        # Reconstruct class indicators for KS between first/last year label mixes
        def _synth(row: pd.Series) -> np.ndarray:
            n = max(int(row["n_samples"]), 1)
            n_pos = int(round(float(row["positive_rate"]) * n))
            n_neg = int(round(float(row["negative_rate"]) * n))
            n_to = max(n - n_pos - n_neg, 0)
            return np.concatenate(
                [
                    np.ones(n_pos, dtype=float),
                    -np.ones(n_neg, dtype=float),
                    np.zeros(n_to, dtype=float),
                ]
            )

        ks = safe_ks(_synth(first), _synth(last))
        detected = bool(max_diff >= 0.08 or abs_diff >= 0.05)
        return {
            "detected": detected,
            "first_period": str(first["period"]),
            "last_period": str(last["period"]),
            "first_positive_rate": float(first["positive_rate"]),
            "last_positive_rate": float(last["positive_rate"]),
            "max_positive_rate_range": max_diff,
            "ks_first_vs_last": ks,
        }
