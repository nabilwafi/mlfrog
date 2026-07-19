"""Validate train/validation datasets before fitting."""

from __future__ import annotations

import logging

from datasets.entities.dataset import Dataset
from models.exceptions import TrainingValidationError

logger = logging.getLogger(__name__)


class TrainingValidator:
    def validate(self, train: Dataset, validation: Dataset) -> None:
        if train.symbol != validation.symbol:
            raise TrainingValidationError(
                f"symbol mismatch train={train.symbol} val={validation.symbol}"
            )
        if train.timeframe != validation.timeframe:
            raise TrainingValidationError(
                f"timeframe mismatch train={train.timeframe} val={validation.timeframe}"
            )
        if train.side != validation.side:
            raise TrainingValidationError(
                f"side mismatch train={train.side} val={validation.side}"
            )
        if train.split != "train":
            raise TrainingValidationError(f"expected train split, got {train.split!r}")
        if validation.split != "validation":
            raise TrainingValidationError(
                f"expected validation split, got {validation.split!r}"
            )
        if train.size == 0 or validation.size == 0:
            raise TrainingValidationError("train/validation must be non-empty")

        train_feats = train.feature_names
        val_feats = validation.feature_names
        if not train_feats:
            raise TrainingValidationError("train dataset has no feature columns")
        if train_feats != val_feats:
            raise TrainingValidationError(
                "feature columns differ between train and validation"
            )

        for name, ds in (("train", train), ("validation", validation)):
            if ds.frame["label"].isna().any():
                raise TrainingValidationError(f"{name} has NaN labels")
            nan_feats = ds.frame[train_feats].isna().any()
            if bool(nan_feats.any()):
                bad = [c for c, v in nan_feats.items() if v]
                raise TrainingValidationError(f"{name} has NaN features: {bad[:10]}")

            labels = set(ds.frame["label"].astype(int).unique().tolist())
            allowed = {-1, 0, 1}
            if not labels.issubset(allowed):
                raise TrainingValidationError(
                    f"{name} has unexpected labels: {sorted(labels - allowed)}"
                )

        if train.feature_version != validation.feature_version:
            raise TrainingValidationError("feature_version mismatch train vs validation")
        if train.label_version != validation.label_version:
            raise TrainingValidationError("label_version mismatch train vs validation")

        logger.info(
            "TrainingValidator OK | symbol=%s tf=%s side=%s train=%s val=%s features=%s",
            train.symbol,
            train.timeframe,
            train.side,
            train.size,
            validation.size,
            len(train_feats),
        )
