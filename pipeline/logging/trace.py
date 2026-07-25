"""Per-signal trace logging (features hash + layer outputs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.types import TraceRecord


class TraceLogger:
    """Append-only JSONL traces. Optional path; no-op if unset."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []

    def log(self, record: TraceRecord) -> None:
        d = record.to_dict()
        # datetime → iso
        if hasattr(d.get("time"), "isoformat"):
            d["time"] = d["time"].isoformat()
        self.records.append(d)
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, default=str) + "\n")
