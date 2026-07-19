"""Validate FeatureSet integrity."""

from __future__ import annotations

import logging
import math
from typing import Iterable

from features.entities.feature_set import FeatureSet
from features.exceptions import FeatureValidationError

logger = logging.getLogger(__name__)

_ALLOWED_DTYPES = frozenset({"float64", "float32", "float", "int64", "int32", "int", "bool"})


class FeatureValidator:
    def __init__(
        self,
        *,
        allow_nan: bool = False,
        max_nan_ratio: float = 0.0,
        min_rows: int = 1,
    ) -> None:
        self._allow_nan = allow_nan
        self._max_nan_ratio = max_nan_ratio
        self._min_rows = min_rows

    def validate(self, feature_set: FeatureSet) -> None:
        logger.info(
            "Validating FeatureSet | symbol=%s tf=%s features=%s rows=%s",
            feature_set.symbol,
            feature_set.timeframe,
            feature_set.feature_count,
            feature_set.row_count,
        )
        if feature_set.row_count < self._min_rows:
            raise FeatureValidationError(
                f"empty / too few rows: {feature_set.row_count} < {self._min_rows}"
            )
        if feature_set.feature_count == 0:
            raise FeatureValidationError("FeatureSet has no features")

        names = list(feature_set.feature_names)
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise FeatureValidationError(f"duplicate feature names: {dupes}")

        for feat in feature_set.features:
            if feat.dtype.split("[")[0] not in _ALLOWED_DTYPES and feat.dtype not in _ALLOWED_DTYPES:
                # accept numpy-like names float64 etc.
                base = feat.dtype.replace("numpy.", "")
                if base not in _ALLOWED_DTYPES:
                    raise FeatureValidationError(
                        f"unsupported dtype for {feat.name!r}: {feat.dtype!r}"
                    )
            self._check_values(feat.name, feat.values)

        # Window consistency: all features same length (already enforced in FeatureSet)
        logger.info("Feature validation success | features=%s", feature_set.feature_count)

    def _check_values(self, name: str, values: Iterable[float]) -> None:
        vals = list(values)
        if not vals:
            raise FeatureValidationError(f"feature {name!r} is empty")
        nan_count = sum(1 for v in vals if v is None or (isinstance(v, float) and math.isnan(v)))
        inf_count = sum(1 for v in vals if isinstance(v, float) and math.isinf(v))
        if inf_count:
            raise FeatureValidationError(f"feature {name!r} contains {inf_count} infinite values")
        ratio = nan_count / len(vals)
        if nan_count and not self._allow_nan and ratio > self._max_nan_ratio:
            raise FeatureValidationError(
                f"feature {name!r} contains {nan_count} NaN values (ratio={ratio:.4f})"
            )
