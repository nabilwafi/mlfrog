"""Persist LabelSet under artifacts/labels/{symbol}/{timeframe}/{side}/."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from labels.entities.label import Label
from labels.entities.label_set import LabelSet
from labels.exceptions import LabelRepositoryError

logger = logging.getLogger(__name__)


class LabelRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def parquet_path(
        self, symbol: str, timeframe: str, side: str, strategy: str, label_version: str
    ) -> Path:
        safe = f"{strategy}_{label_version}".replace("/", "_")
        return self._dir(symbol, timeframe, side) / f"{safe}.parquet"

    def meta_path(
        self, symbol: str, timeframe: str, side: str, strategy: str, label_version: str
    ) -> Path:
        safe = f"{strategy}_{label_version}".replace("/", "_")
        return self._dir(symbol, timeframe, side) / f"{safe}.meta.json"

    def save_parquet(self, label_set: LabelSet) -> Path:
        path = self.parquet_path(
            label_set.symbol,
            label_set.timeframe,
            label_set.side,
            label_set.strategy,
            label_set.label_version,
        )
        logger.info(
            "Saving LabelSet | path=%s side=%s n=%s", path, label_set.side, label_set.size
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "timestamp": lab.timestamp.isoformat(),
                    "entry_price": lab.entry_price,
                    "tp_price": lab.tp_price,
                    "sl_price": lab.sl_price,
                    "expire_timestamp": lab.expire_timestamp.isoformat(),
                    "holding_bars": lab.holding_bars,
                    "realized_return": lab.realized_return,
                    "exit_reason": lab.exit_reason,
                    "side": lab.side,
                    "label": lab.label,
                    "metadata_json": json.dumps(dict(lab.metadata)),
                }
                for lab in label_set.labels
            ]
            pd.DataFrame(rows).to_parquet(path, index=False)
            meta = {
                "symbol": label_set.symbol,
                "timeframe": label_set.timeframe,
                "strategy": label_set.strategy,
                "side": label_set.side,
                "label_version": label_set.label_version,
                "created_at": label_set.created_at.isoformat(),
                "size": label_set.size,
                "class_counts": label_set.class_counts,
            }
            self.meta_path(
                label_set.symbol,
                label_set.timeframe,
                label_set.side,
                label_set.strategy,
                label_set.label_version,
            ).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:
            raise LabelRepositoryError(f"save_parquet failed: {exc}") from exc
        logger.info("Saved LabelSet | path=%s", path)
        return path

    def load_parquet(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        strategy: str,
        label_version: str,
        *,
        timezone: str = "UTC",
    ) -> LabelSet:
        path = self.parquet_path(symbol, timeframe, side, strategy, label_version)
        logger.info("Loading LabelSet | path=%s", path)
        if not path.is_file():
            raise LabelRepositoryError(f"label parquet not found: {path}")
        try:
            df = pd.read_parquet(path)
            meta_file = self.meta_path(symbol, timeframe, side, strategy, label_version)
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
            tz = ZoneInfo(timezone)
            labels: list[Label] = []
            for row in df.itertuples(index=False):
                ts = pd.Timestamp(row.timestamp).to_pydatetime()
                ts = ts.replace(tzinfo=tz) if ts.tzinfo is None else ts.astimezone(tz)
                exp = pd.Timestamp(row.expire_timestamp).to_pydatetime()
                exp = exp.replace(tzinfo=tz) if exp.tzinfo is None else exp.astimezone(tz)
                row_side = str(getattr(row, "side", side)).lower()
                md = json.loads(row.metadata_json) if row.metadata_json else {}
                labels.append(
                    Label(
                        timestamp=ts,
                        entry_price=float(row.entry_price),
                        tp_price=float(row.tp_price),
                        sl_price=float(row.sl_price),
                        expire_timestamp=exp,
                        holding_bars=int(row.holding_bars),
                        realized_return=float(row.realized_return),
                        exit_reason=str(row.exit_reason),  # type: ignore[arg-type]
                        side=row_side,  # type: ignore[arg-type]
                        label=int(row.label),
                        metadata=md,
                    )
                )
            created_raw = meta.get("created_at")
            created_at = (
                datetime.fromisoformat(created_raw)
                if created_raw
                else datetime.now(tz=tz)
            )
            return LabelSet(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                side=side.lower(),  # type: ignore[arg-type]
                label_version=str(meta.get("label_version", label_version)),
                created_at=created_at,
                labels=tuple(labels),
            )
        except LabelRepositoryError:
            raise
        except Exception as exc:
            raise LabelRepositoryError(f"load_parquet failed: {exc}") from exc
