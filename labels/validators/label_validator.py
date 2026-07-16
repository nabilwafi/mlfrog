"""Validate LabelSet integrity and report imbalance / leakage risks."""

from __future__ import annotations

import logging
from collections import Counter

from labels.entities.label_set import LabelSet
from labels.exceptions import LabelValidationError

logger = logging.getLogger(__name__)

_ALLOWED_LABELS_DEFAULT = frozenset({-1, 0, 1})


class LabelValidator:
    def __init__(
        self,
        *,
        allowed_labels: frozenset[int] | None = None,
        min_size: int = 1,
        max_imbalance_ratio: float | None = None,
        expected_horizon: int | None = None,
    ) -> None:
        self._allowed = allowed_labels or _ALLOWED_LABELS_DEFAULT
        self._min_size = min_size
        self._max_imbalance_ratio = max_imbalance_ratio
        self._expected_horizon = expected_horizon

    def validate(self, label_set: LabelSet) -> dict[str, object]:
        logger.info(
            "Validating LabelSet | symbol=%s tf=%s side=%s strategy=%s n=%s",
            label_set.symbol,
            label_set.timeframe,
            label_set.side,
            label_set.strategy,
            label_set.size,
        )
        if label_set.size < self._min_size:
            raise LabelValidationError(f"missing labels / too few: {label_set.size}")

        report: dict[str, object] = {
            "size": label_set.size,
            "strategy": label_set.strategy,
            "side": label_set.side,
        }
        counts = Counter(lab.label for lab in label_set.labels)
        report["class_counts"] = dict(counts)
        total = sum(counts.values())
        report["class_balance"] = {k: v / total for k, v in counts.items()}
        report["class_distribution"] = dict(counts)

        if self._max_imbalance_ratio is not None and len(counts) >= 2:
            mx, mn = max(counts.values()), min(counts.values())
            ratio = mx / max(mn, 1)
            report["imbalance_ratio"] = ratio
            if ratio > self._max_imbalance_ratio:
                logger.warning(
                    "Class imbalance high | ratio=%.2f max_allowed=%.2f counts=%s",
                    ratio,
                    self._max_imbalance_ratio,
                    dict(counts),
                )

        seen_ts: set = set()
        prev_ts = None
        for i, lab in enumerate(label_set.labels):
            if lab.label not in self._allowed:
                raise LabelValidationError(
                    f"invalid class value {lab.label} at index {i}; allowed={sorted(self._allowed)}"
                )
            if lab.side != label_set.side:
                raise LabelValidationError(
                    f"side mismatch at index {i}: {lab.side} != {label_set.side}"
                )
            if lab.timestamp.tzinfo is None or lab.expire_timestamp.tzinfo is None:
                raise LabelValidationError(f"timezone-naive timestamp at index {i}")
            if lab.expire_timestamp < lab.timestamp:
                raise LabelValidationError(
                    f"look-ahead / expire before entry at index {i}"
                )
            if lab.timestamp in seen_ts:
                raise LabelValidationError(
                    f"duplicate timestamp at index {i}: {lab.timestamp.isoformat()}"
                )
            seen_ts.add(lab.timestamp)
            if prev_ts is not None and lab.timestamp < prev_ts:
                raise LabelValidationError(f"timestamps not ascending at index {i}")
            prev_ts = lab.timestamp

            if lab.entry_price <= 0 or lab.tp_price <= 0 or lab.sl_price <= 0:
                raise LabelValidationError(f"non-positive TP/SL/entry at index {i}")

            if lab.side == "long":
                if not (lab.sl_price < lab.entry_price < lab.tp_price):
                    raise LabelValidationError(
                        f"invalid long TP/SL geometry at index {i}: "
                        f"sl={lab.sl_price} entry={lab.entry_price} tp={lab.tp_price}"
                    )
            else:
                if not (lab.tp_price < lab.entry_price < lab.sl_price):
                    raise LabelValidationError(
                        f"invalid short TP/SL geometry at index {i}: "
                        f"tp={lab.tp_price} entry={lab.entry_price} sl={lab.sl_price}"
                    )

            if lab.holding_bars < 0:
                raise LabelValidationError(f"invalid holding period (<0) at index {i}")
            if self._expected_horizon is not None and lab.holding_bars > self._expected_horizon:
                raise LabelValidationError(
                    f"holding_bars {lab.holding_bars} > horizon {self._expected_horizon} at index {i}"
                )
            if lab.holding_bars == 0 and lab.exit_reason != "TIMEOUT":
                raise LabelValidationError(
                    f"invalid holding period (0 with {lab.exit_reason}) at index {i}"
                )

        if self._expected_horizon is not None:
            bad = [
                lab
                for lab in label_set.labels
                if lab.exit_reason == "TIMEOUT" and lab.holding_bars > self._expected_horizon
            ]
            report["timeout_horizon_mismatches"] = len(bad)

        report["leakage_checks"] = "pass"
        logger.info(
            "Label validation success | side=%s counts=%s",
            label_set.side,
            dict(counts),
        )
        return report
