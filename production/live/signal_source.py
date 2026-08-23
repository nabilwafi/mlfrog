"""Live MT5 closed-bar polling → frozen inference → IncomingSignal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from production import TRAIL_TIMEFRAME
from production.live.features import LiveFeatureBuilder
from production.live.inference import FrozenStackInference, ScoredSignal
from production.live.mt5_candles import MT5CandleFeed, is_stale_closed_bar
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
    """One poll: optional newly closed H1 (entry) + M15 manage bar + signals."""

    bar: ClosedBar | None
    signals: list[IncomingSignal]
    manage_bar: ClosedBar | None = None
    m15_bars: pd.DataFrame | None = None


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
        model_symbol: str | None = None,
    ) -> None:
        self._cfg = cfg
        # Broker symbol is case-sensitive on HF (XAUUSDc). Model artifact paths use model_symbol.
        self._symbol = str(symbol)
        self._timeframe = timeframe.upper()
        self._history = int(history_bars)
        ms = str(model_symbol or symbol)
        self._feed = MT5CandleFeed(cfg, symbol=self._symbol)
        self._features = LiveFeatureBuilder(cfg, symbol=self._symbol, timezone=str(cfg.get("timezone", "UTC")))
        self._inference = FrozenStackInference(cfg, symbol=ms, timeframe=timeframe)
        self._last_bar_ts: pd.Timestamp | None = None
        self._last_trail_ts: pd.Timestamp | None = None
        self._market_was_closed: bool = False
        self._trail_tf = str(TRAIL_TIMEFRAME).upper()

    def connect(self) -> None:
        self._feed.connect()

    def disconnect(self) -> None:
        self._feed.disconnect()

    def _closed_bar(self, row: pd.Series, ts: pd.Timestamp, timeframe: str) -> ClosedBar:
        return ClosedBar(
            symbol=self._symbol,
            timeframe=timeframe,
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            tick_volume=float(row.get("tick_volume") or 0),
            spread=float(row.get("spread") or 0),
            real_volume=float(row.get("real_volume") or 0),
        )

    def _poll_manage_bar(self) -> ClosedBar | None:
        """New closed trail-clock bar. No inference. None if trail tf is the entry tf."""
        if self._trail_tf == self._timeframe:
            return None
        try:
            closed = self._feed.latest_closed_bar(self._trail_tf)
        except Exception:
            logger.exception("live_trail_poll_failed tf=%s", self._trail_tf)
            return None
        if closed is None or closed.empty:
            return None
        row = closed.iloc[0]
        ts = pd.Timestamp(row["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if self._last_trail_ts is not None and ts <= self._last_trail_ts:
            return None
        if is_stale_closed_bar(ts, self._trail_tf):
            if self._last_trail_ts is None or ts > self._last_trail_ts:
                self._last_trail_ts = ts
            return None
        self._last_trail_ts = ts
        logger.info("live_trail_bar_closed tf=%s ts=%s close=%s", self._trail_tf, ts.isoformat(), row["close"])
        return self._closed_bar(row, ts, self._trail_tf)

    def _fetch_m15_history(self) -> pd.DataFrame | None:
        """Closed M15 OHLC window for pullback entry evaluation."""
        if self._trail_tf != "M15":
            return None
        try:
            m15 = self._feed.fetch("M15", count=max(32, self._history * 4))
        except Exception:
            logger.exception("live_m15_history_failed")
            return None
        if m15 is None or m15.empty:
            return None
        out = m15.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.sort_values("timestamp").reset_index(drop=True)

    def poll(self) -> LiveTick:
        manage_bar = self._poll_manage_bar()
        m15_bars = self._fetch_m15_history() if manage_bar is not None else None
        try:
            closed = self._feed.latest_closed_bar(self._timeframe)
        except Exception:
            logger.exception("live_poll_failed")
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)
        if closed is None or closed.empty:
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)
        row = closed.iloc[0]
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if self._last_bar_ts is not None and ts <= self._last_bar_ts:
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)

        # Weekend / holiday: last MT5 bar is fully closed but stale — seed cursor only.
        # Without this, bot "eats" Friday as a live bar then looks stuck until next close.
        if is_stale_closed_bar(ts, self._timeframe):
            if self._last_bar_ts is None or ts > self._last_bar_ts:
                self._last_bar_ts = ts
                self._market_was_closed = True
                logger.info(
                    "live_market_closed_seed_cursor ts=%s — waiting for first bar after reopen",
                    ts.isoformat(),
                )
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)

        if self._market_was_closed:
            logger.info("live_market_reopened first_bar_ts=%s", ts.isoformat())
            self._market_was_closed = False

        try:
            h1 = self._feed.fetch(self._timeframe, count=self._history)
            h4 = self._feed.fetch("H4", count=max(120, self._history // 4))
            d1 = self._feed.fetch("D1", count=120)
            m5 = self._feed.fetch("M5", count=min(2000, self._history * 12))
            if h1.empty:
                logger.warning("live_h1_empty ts=%s — MT5 history missing?", ts)
                return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)
            panel = self._features.build_panel(h1=h1, h4=h4, d1=d1, m5=m5)
        except Exception:
            logger.exception("live_feature_build_failed ts=%s", ts)
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)

        if panel.empty:
            logger.warning("live_feature_panel_empty ts=%s", ts)
            return LiveTick(bar=None, signals=[], manage_bar=manage_bar, m15_bars=m15_bars)

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
        # atr_percent = (ATR/close)*100 from VolatilityBuilder — convert back to price ATR
        atr_pct = float(feat_row.get("atr_percent", 0.1) or 0.1)
        atr = atr_pct / 100.0 * float(row["close"])
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

        # One side per bar: keep highest meta (then primary prob)
        if len(signals) > 1:
            signals = [
                max(
                    signals,
                    key=lambda s: (float(s.meta_probability), float(s.probability)),
                )
            ]

        bar = self._closed_bar(row, ts, self._timeframe)
        bar.features = _features_snapshot(feat_row)
        logger.info(
            "live_bar_closed ts=%s close=%s signals=%s",
            ts.isoformat(),
            row["close"],
            len(signals),
        )
        return LiveTick(bar=bar, signals=signals, manage_bar=manage_bar, m15_bars=m15_bars)


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
        h1_ref_price=scored.entry_price,
        atr=scored.atr,
        atr_percentile=float(scored.atr_percentile),
        session=scored.session,
        regime=scored.regime,
        trend=scored.trend,
        volatility=scored.volatility,
        momentum=scored.momentum,
        structure=scored.structure,
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
