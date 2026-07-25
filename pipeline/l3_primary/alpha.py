"""L3 Primary Alpha — frozen LightGBM; probability logged, not used for sizing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.types import PrimaryOut
from production.live.inference import FrozenStackInference, ScoredSignal


class PrimaryAlpha:
    """Wraps frozen stack; score_row still returns meta/conf for live adapters."""

    def __init__(self, cfg: dict[str, Any], *, symbol: str, timeframe: str = "H1") -> None:
        self._inf = FrozenStackInference(cfg, symbol=symbol, timeframe=timeframe)

    @property
    def inference(self) -> FrozenStackInference:
        return self._inf

    def score_row(self, row: pd.Series, *, side: str, entry_price: float, atr: float) -> ScoredSignal | None:
        return self._inf.score_row(row, side=side, entry_price=entry_price, atr=atr)


def primary_from_scored(scored: ScoredSignal | None, *, side_hint: str = "") -> PrimaryOut:
    if scored is None:
        return PrimaryOut(side="NONE", raw_score=0.0, probability=None, reason="no_candidate")
    return PrimaryOut(
        side=str(scored.side).upper(),
        raw_score=float(scored.probability),
        probability=float(scored.probability),
        reason="primary_ok",
    )


def primary_from_signal(*, side: str, probability: float) -> PrimaryOut:
    s = str(side).upper()
    if s not in {"LONG", "SHORT"}:
        s = str(side).lower()
        s = "LONG" if s == "long" else "SHORT" if s == "short" else "NONE"
    return PrimaryOut(side=s, raw_score=float(probability), probability=float(probability), reason="replay")
