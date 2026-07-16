"""
ConfidenceEngine — calibrated probability + tier on top of raw model output.

Does NOT retrain the base model. Calibrator is fit separately (validation only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class CalibratorProto(Protocol):
    def predict(self, raw: np.ndarray) -> np.ndarray: ...


@dataclass
class TierThresholds:
    """Boundaries on *calibrated* probability. Tier = low | medium | high."""

    low_max: float  # calibrated < low_max → low
    medium_max: float  # low_max <= c < medium_max → medium; else high
    rationale: str = ""

    def as_dict(self) -> dict[str, float]:
        # user-facing style: lower edge of each tier
        return {"low": 0.0, "medium": self.low_max, "high": self.medium_max}


class PlattCalibrator:
    """Logistic regression on raw probability (1 feature)."""

    def __init__(self, lr: Any):
        self.lr = lr

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=float).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    def __init__(self, iso: Any):
        self.iso = iso

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=float).ravel()
        return np.clip(self.iso.predict(x), 0.0, 1.0)


class ConfidenceEngine:
    def __init__(self, calibrator: CalibratorProto, tier_thresholds: dict | TierThresholds):
        self.calibrator = calibrator
        if isinstance(tier_thresholds, TierThresholds):
            self.tiers = tier_thresholds
        else:
            # {"low": 0.0, "medium": m, "high": h} → edges
            self.tiers = TierThresholds(
                low_max=float(tier_thresholds["medium"]),
                medium_max=float(tier_thresholds["high"]),
                rationale=str(tier_thresholds.get("rationale", "")),
            )

    def get_calibrated_confidence(self, raw_probability: float) -> float:
        p = self.calibrator.predict(np.array([float(raw_probability)]))[0]
        return float(np.clip(p, 0.0, 1.0))

    def get_calibrated_batch(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(np.asarray(raw_probabilities, dtype=float)), 0.0, 1.0)

    def get_confidence_tier(self, calibrated_confidence: float) -> str:
        c = float(calibrated_confidence)
        if c < self.tiers.low_max:
            return "low"
        if c < self.tiers.medium_max:
            return "medium"
        return "high"

    def score(self, raw_probability: float) -> dict[str, Any]:
        cal = self.get_calibrated_confidence(raw_probability)
        return {
            "raw": float(raw_probability),
            "calibrated": cal,
            "tier": self.get_confidence_tier(cal),
        }

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceEngine":
        import pickle

        with open(path, "rb") as f:
            blob = pickle.load(f)
        return cls(blob["calibrator"], blob["tier_thresholds"])

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "calibrator": self.calibrator,
            "tier_thresholds": self.tiers.as_dict()
            | {"rationale": self.tiers.rationale},
            "meta": meta or {},
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
