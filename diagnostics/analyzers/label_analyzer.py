"""Label distribution and barrier-exit diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from datasets.entities.dataset import Dataset


class LabelAnalyzer:
    def analyze(
        self,
        splits: dict[str, Dataset],
        *,
        label_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        # Prefer full LabelSet frame when available (holding bars / exit reason)
        if label_frame is not None and not label_frame.empty:
            base = label_frame.copy()
        else:
            parts = [ds.frame[["timestamp", "label"]].copy() for ds in splits.values()]
            base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if base.empty:
            return {
                "tp_pct": float("nan"),
                "sl_pct": float("nan"),
                "timeout_pct": float("nan"),
                "positive_rate": float("nan"),
                "negative_rate": float("nan"),
            }

        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        lab = base["label"].astype(int)
        n = max(len(lab), 1)
        tp = int((lab == 1).sum())
        sl = int((lab == -1).sum())
        to = int((lab == 0).sum())

        out: dict[str, Any] = {
            "n_labels": int(len(lab)),
            "tp_count": tp,
            "sl_count": sl,
            "timeout_count": to,
            "tp_pct": float(tp / n * 100.0),
            "sl_pct": float(sl / n * 100.0),
            "timeout_pct": float(to / n * 100.0),
            "positive_rate": float(tp / n),
            "negative_rate": float(sl / n),
            "timeout_rate": float(to / n),
        }

        base = base.assign(
            year=base["timestamp"].dt.year,
            month=base["timestamp"].dt.strftime("%Y-%m"),
        )
        yearly = (
            base.groupby("year")["label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .rename(columns={-1: "sl", 0: "timeout", 1: "tp"})
        )
        out["distribution_per_year"] = {
            str(idx): {str(c): float(yearly.loc[idx, c]) for c in yearly.columns}
            for idx in yearly.index
        }

        monthly = (
            base.groupby("month")["label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .rename(columns={-1: "sl", 0: "timeout", 1: "tp"})
        )
        # keep last 24 months for report size
        monthly_tail = monthly.tail(24)
        out["distribution_per_month"] = {
            str(idx): {str(c): float(monthly_tail.loc[idx, c]) for c in monthly_tail.columns}
            for idx in monthly_tail.index
        }

        if "holding_bars" in base.columns:
            hb = base["holding_bars"].astype(float)
            out["holding_period"] = {
                "mean": float(hb.mean()),
                "median": float(hb.median()),
                "std": float(hb.std(ddof=1)) if len(hb) > 1 else 0.0,
                "min": float(hb.min()),
                "max": float(hb.max()),
            }
        else:
            out["holding_period"] = {}

        if "exit_reason" in base.columns:
            er = base["exit_reason"].astype(str).value_counts(normalize=True)
            out["exit_reason_distribution"] = {str(k): float(v) for k, v in er.items()}
        else:
            # Infer from ternary label
            out["exit_reason_distribution"] = {
                "TP": float(tp / n),
                "SL": float(sl / n),
                "TIMEOUT": float(to / n),
            }

        return out
