"""Dataset-level diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset


class DatasetAnalyzer:
    def analyze(
        self,
        splits: dict[str, Dataset],
    ) -> dict[str, Any]:
        frames = {name: ds.frame for name, ds in splits.items()}
        total = int(sum(len(f) for f in frames.values()))
        class_dist: dict[str, dict[str, int]] = {}
        for name, frame in frames.items():
            counts = frame["label"].astype(int).value_counts().to_dict()
            class_dist[name] = {str(k): int(v) for k, v in sorted(counts.items())}

        missing = {
            name: int(frame.isna().sum().sum()) for name, frame in frames.items()
        }
        inf_count = {}
        for name, frame in frames.items():
            num = frame.select_dtypes(include=[np.number])
            inf_count[name] = int(np.isinf(num.to_numpy(dtype=float, copy=False)).sum())

        dup_ts = {}
        chrono_ok = {}
        for name, frame in frames.items():
            ts = pd.to_datetime(frame["timestamp"], utc=True)
            dup_ts[name] = int(ts.duplicated().sum())
            chrono_ok[name] = bool(ts.is_monotonic_increasing)

        # Imbalance on combined train+validation ternary labels
        combined = pd.concat(
            [frames[k] for k in ("train", "validation") if k in frames],
            ignore_index=True,
        )
        lab = combined["label"].astype(int)
        n = max(len(lab), 1)
        rates = {
            "tp_rate": float((lab == 1).mean()),
            "sl_rate": float((lab == -1).mean()),
            "timeout_rate": float((lab == 0).mean()),
        }
        binary_pos = float(((lab == 1).sum()) / max((lab != 0).sum(), 1))
        imbalance_ratio = float(
            max((lab == 1).sum(), (lab == -1).sum())
            / max(min((lab == 1).sum(), (lab == -1).sum()), 1)
        )

        return {
            "total_rows": total,
            "train_rows": int(len(frames["train"])) if "train" in frames else 0,
            "validation_rows": int(len(frames["validation"])) if "validation" in frames else 0,
            "test_rows": int(len(frames["test"])) if "test" in frames else 0,
            "sealed_rows": int(len(frames["sealed"])) if "sealed" in frames else 0,
            "class_distribution": class_dist,
            "missing_values": missing,
            "nan_count": missing,
            "infinite_values": inf_count,
            "duplicate_timestamps": dup_ts,
            "chronological_ordering": chrono_ok,
            "dataset_imbalance": {
                **rates,
                "binary_positive_rate_excl_timeout": binary_pos,
                "tp_sl_imbalance_ratio": imbalance_ratio,
            },
        }
