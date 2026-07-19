"""Live MT5 closed-bar polling → frozen inference → IncomingSignal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from production.live.features import LiveFeatureBuilder
from production.live.inference import FrozenStackInference, ScoredSignal
from production.live.mt5_candles import MT5CandleFeed
from production.paper.pipeline import IncomingSignal

logger = logging.getLogger(__name__)


@dataclass
class ClosedBar:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0
    spread: float = 0.0
    real_volume: float = 0.0
    features: dict[str, Any] | None = None


@dataclass
class LiveTick:
    """One poll: optional newly closed bar + signals to process."""

    bar: ClosedBar | None
    signals: list[IncomingSignal]


class LiveMT5SignalSource:
    """
    Poll MT5 for new closed H1 bars, run frozen stack, emit signals.
    Paper fills only — never calls order_send.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        symbol: str,
        timeframe: str = "H1",
        history_bars: int = 400,
    ) -> None:
        self._cfg = cfg
        self._symbol = symbol.upper()
        self._timeframe = timeframe.upper()
        self._history = int(history_bars)
        self._feed = MT5CandleFeed(cfg, symbol=symbol)
        self._features = LiveFeatureBuilder(cfg, symbol=symbol, timezone=str(cfg.get("timezone", "UTC")))
        self._inference = FrozenStackInference(cfg, symbol=symbol, timeframe=timeframe)
        self._last_bar_ts: pd.Timestamp | None = None

    def connect(self) -> None:
        self._feed.connect()

    def disconnect(self) -> None:
        self._feed.disconnect()

    def poll(self) -> LiveTick:
        try:
            closed = self._feed.latest_closed_bar(self._timeframe)
        except Exception:
            logger.exception("live_poll_failed")
            return LiveTick(bar=None, signals=[])
        if closed is None or closed.empty:
            return LiveTick(bar=None, signals=[])
        row = closed.iloc[0]
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if self._last_bar_ts is not None and ts <= self._last_bar_ts:
            return LiveTick(bar=None, signals=[])

        try:
            h1 = self._feed.fetch(self._timeframe, count=self._history)
            h4 = self._feed.fetch("H4", count=max(120, self._history // 4))
            d1 = self._feed.fetch("D1", count=120)
            m5 = self._feed.fetch("M5", count=min(2000, self._history * 12))
            if h1.empty:
                logger.warning("live_h1_empty ts=%s — MT5 history missing?", ts)
                return LiveTick(bar=None, signals=[])
            panel = self._features.build_panel(h1=h1, h4=h4, d1=d1, m5=m5)
        except Exception:
            logger.exception("live_feature_build_failed ts=%s", ts)
            return LiveTick(bar=None, signals=[])

        if panel.empty:
            logger.warning("live_feature_panel_empty ts=%s", ts)
            return LiveTick(bar=None, signals=[])

        # only advance cursor after we can actually emit a bar
        self._last_bar_ts = ts

        feat_row = panel.loc[panel["timestamp"] == ts]
        if feat_row.empty:
            # feature matrix may use close-time vs MT5 open-time — take nearest <= ts
            prior = panel.loc[panel["timestamp"] <= ts]
            feat_row = prior.iloc[[-1]] if not prior.empty else panel.iloc[[-1]]
            logger.warning(
                "live_feature_ts_mismatch bar_ts=%s feat_ts=%s",
                ts.isoformat(),
                feat_row.iloc[0]["timestamp"],
            )
        feat_row = feat_row.iloc[0]
        atr = float(feat_row.get("atr_percent", 0.01) or 0.01) * float(row["close"])
        if not pd.notna(atr) or atr <= 0:
            atr = float(row["close"]) * 0.001

        signals: list[IncomingSignal] = []
        for side in ("long", "short"):
            try:
                scored = self._inference.score_row(
                    feat_row,
                    side=side,
                    entry_price=float(row["close"]),
                    atr=atr,
                )
            except Exception:
                logger.exception("live_score_failed side=%s ts=%s", side, ts)
                continue
            if scored is None:
                continue
            signals.append(_to_incoming(scored, symbol=self._symbol))

        bar = ClosedBar(
            symbol=self._symbol,
            timeframe=self._timeframe,
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            tick_volume=float(row.get("tick_volume") or 0),
            spread=float(row.get("spread") or 0),
            real_volume=float(row.get("real_volume") or 0),
            features=_features_snapshot(feat_row),
        )
        logger.info(
            "live_bar_closed ts=%s close=%s signals=%s",
            ts.isoformat(),
            row["close"],
            len(signals),
        )
        return LiveTick(bar=bar, signals=signals)


def _to_incoming(scored: ScoredSignal, *, symbol: str) -> IncomingSignal:
    ts = scored.timestamp.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return IncomingSignal(
        timestamp=ts,
        symbol=symbol,
        side=scored.side,
        probability=scored.probability,
        meta_probability=scored.meta_probability,
        confidence=scored.confidence,
        entry_price=scored.entry_price,
        atr=scored.atr,
        session=scored.session,
        regime=scored.regime,
        bar_key=scored.bar_key,
    )


def _features_snapshot(row: pd.Series) -> dict[str, Any]:
    """Point-in-time feature row for DB eval (JSON-serializable)."""
    out: dict[str, Any] = {}
    for key, val in row.items():
        if key == "timestamp":
            continue
        if pd.isna(val):
            continue
        if isinstance(val, (np.floating, float)):
            out[str(key)] = float(val)
        elif isinstance(val, (np.integer, int)):
            out[str(key)] = int(val)
        else:
            out[str(key)] = val
    return out
