"""Score frozen Primary → Meta → Confidence on one live feature row."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from production import PRIMARY_TOP_PCT
from production.live.primary_export import ensure_frozen_primary
from production.paper.market_state import infer_market_state
from research.confidence_layer.services.confidence import attach_components
from research.meta_model import FINAL_FEATURES
from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT, SHORT_STRUCTURE_CONTEXT
from settings.paths import ROOT

logger = logging.getLogger(__name__)


@dataclass
class ScoredSignal:
    side: str
    timestamp: pd.Timestamp
    entry_price: float
    atr: float
    probability: float
    meta_probability: float
    confidence: float
    session: str
    regime: str
    bar_key: str
    trend: str = "unknown"
    volatility: str = "unknown"
    momentum: str = "unknown"
    structure: str = "unknown"


class FrozenStackInference:
    def __init__(self, cfg: dict[str, Any], *, symbol: str, timeframe: str = "H1") -> None:
        self._cfg = cfg
        self._symbol = symbol.upper()
        self._timeframe = timeframe.upper()
        mm = dict(cfg.get("meta_model") or {})
        meta_root = Path(str(mm.get("output_directory", "./artifacts/research/meta_model")))
        self._meta_root = meta_root if meta_root.is_absolute() else (ROOT / meta_root).resolve()
        self._meta_model = lgb.Booster(model_file=str(self._meta_root / "models" / "meta_full.txt"))
        med = pd.read_csv(self._meta_root / "feature_medians.csv", index_col=0)["median"]
        self._meta_medians = med
        self._meta_features = list(FINAL_FEATURES)

        conf_root = Path(str((cfg.get("confidence_layer") or {}).get("output_directory", "./artifacts/research/confidence_layer")))
        self._conf_root = conf_root if conf_root.is_absolute() else (ROOT / conf_root).resolve()
        weights = pd.read_csv(self._conf_root / "confidence_weights.csv")
        self._conf_coef = {str(r["component"]): float(r["coef_mean"]) for _, r in weights.iterrows()}

        self._primary: dict[str, lgb.Booster] = {}
        self._primary_features: dict[str, list[str]] = {}
        self._prob_history: dict[str, deque[float]] = {"long": deque(maxlen=500), "short": deque(maxlen=500)}
        # Prefer production policy top-pct; allow cfg.production.primary_top_pct override.
        prod_cfg = dict(cfg.get("production") or {})
        self._percentile = float(prod_cfg.get("primary_top_pct", PRIMARY_TOP_PCT))

    def _load_primary(self, side: str) -> lgb.Booster:
        side = side.lower()
        if side in self._primary:
            return self._primary[side]
        path = ensure_frozen_primary(self._cfg, symbol=self._symbol, timeframe=self._timeframe, side=side)
        booster = lgb.Booster(model_file=str(path))
        self._primary[side] = booster
        self._primary_features[side] = list(booster.feature_name())
        return booster

    def _context_cols(self, side: str) -> tuple[str, ...]:
        return LONG_STRUCTURE_CONTEXT if side.lower() == "long" else SHORT_STRUCTURE_CONTEXT

    def _session_label(self, row: pd.Series) -> str:
        if float(row.get("session_london_ny_overlap", 0) or 0) >= 0.5:
            return "overlap"
        if float(row.get("session_london", 0) or 0) >= 0.5:
            return "london"
        if float(row.get("session_newyork", 0) or 0) >= 0.5:
            return "newyork"
        if float(row.get("session_asia", 0) or 0) >= 0.5:
            return "asia"
        return "other"

    def _candidate_gate(self, side: str, prob: float) -> bool:
        hist = self._prob_history[side.lower()]
        hist.append(prob)
        if len(hist) < 50:
            return True  # ponytail: warm-up allows signals until history fills
        cutoff = float(np.percentile(list(hist), 100.0 * (1.0 - self._percentile)))
        return prob >= cutoff

    def _meta_row(self, row: pd.Series, *, side: str, raw_prob: float) -> pd.Series:
        out = row.copy()
        out["raw_probability"] = raw_prob
        hist = list(self._prob_history[side.lower()])
        out["probability_rank"] = float(sum(1 for p in hist if p >= raw_prob))
        out["probability_percentile"] = float(sum(1 for p in hist if p <= raw_prob) / max(len(hist), 1))
        out["probability_margin"] = raw_prob - (min(hist) if hist else raw_prob)
        x = pd.DataFrame([{f: float(out.get(f, np.nan)) for f in self._meta_features}])
        x = x.fillna(self._meta_medians)
        return x.iloc[0]

    def _confidence(self, row: pd.Series, *, side: str, raw_prob: float, meta_prob: float) -> float:
        frame = pd.DataFrame([dict(row)])
        frame["side"] = side
        frame["meta_proba"] = meta_prob
        frame["raw_probability"] = raw_prob
        comps = attach_components(frame).iloc[0]
        feat_map = {
            "meta_proba": float(comps["meta_proba"]),
            "raw_probability": float(comps["raw_probability"]),
            "h4_context": float(comps["h4_context"]),
            "d1_trend": float(comps["d1_trend"]),
            "m5_entry_quality": float(comps["m5_entry_quality_01"]),
        }
        z = sum(self._conf_coef.get(k, 0.0) * v for k, v in feat_map.items())
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p * 100.0, 0.0, 100.0))

    def score_row(self, row: pd.Series, *, side: str, entry_price: float, atr: float) -> ScoredSignal | None:
        side_l = side.lower()
        booster = self._load_primary(side_l)
        feats = self._primary_features[side_l]
        missing = [f for f in feats if f not in row.index]
        if missing:
            logger.warning("live_features_missing side=%s missing=%s", side_l, missing[:5])
            return None
        raw_prob = float(booster.predict(row[feats].astype(float).to_frame().T)[0])
        if not self._candidate_gate(side_l, raw_prob):
            return None
        meta_x = self._meta_row(row, side=side_l, raw_prob=raw_prob)
        meta_prob = float(self._meta_model.predict(meta_x[self._meta_features].astype(float).to_frame().T)[0])
        conf = self._confidence(row, side=side_l, raw_prob=raw_prob, meta_prob=meta_prob)
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ms = infer_market_state(row)
        return ScoredSignal(
            side=side_l,
            timestamp=ts,
            entry_price=float(entry_price),
            atr=float(atr),
            probability=raw_prob,
            meta_probability=meta_prob,
            confidence=conf,
            session=ms.session if ms.session != "unknown" else self._session_label(row),
            regime=ms.regime_raw,
            trend=ms.trend,
            volatility=ms.volatility,
            momentum=ms.momentum,
            structure=ms.structure,
            bar_key=ts.isoformat() + ":" + side_l,
        )
