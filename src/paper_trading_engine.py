"""
Paper trading engine — full decision stack, ZERO broker orders.

Decision path matches backtest:
  LightGBM LONG v3 @ thr=0.51 → ConfidenceEngine (Platt) → regime guard
  (Z_BLOCK=4.0, Z_REDUCE=2.0, W=4320) → paper position / skip / blocked hypothetical.

Run:
  python -m src.paper_trading_engine --once
  python -m src.paper_trading_engine --loop --poll-sec 30

HARD RULE: never call mt5.order_send (or any execution API).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from backtest.engine import (
    CONTRACT_SIZE,
    FALLBACK_SPREAD_POINTS,
    HORIZON_BARS,
    RISK_PER_TRADE_PCT,
    SL_ATR_MULT,
    STARTING_EQUITY,
    TP_ATR_MULT,
    VOLUME_MIN,
    _cost_price,
    _round_lots,
)
from backtest.regime import (
    ROLLING_WINDOW_BARS,
    Z_REDUCE_THRESHOLD,
    assign_regime_tier,
    compute_causal_atr_zscore,
)
from features.build_features_v2 import SMC_FEATURE_COLS
from features.feature_engineering import (
    FeatureConfig,
    compute_d1_features,
    compute_h1_features,
    compute_h4_features,
    merge_multi_tf_features,
)
from features.reversal import compute_reversal_features
from features.smc import compute_h4_bos_features, compute_smc_features
from models.data_prep import CAT_COLS
from src.confidence_engine import ConfidenceEngine
from src.live_feed_handler import LiveFeedHandler, load_raw_csv

# Guard defaults (Gate 4 settled)
Z_BLOCK = 4.0
Z_REDUCE = Z_REDUCE_THRESHOLD
RAW_THR = 0.51
PAPER_DIR = Path("data/paper_trading")
MODEL_PATH = Path("data/models_v3/long/lightgbm_long_v3.pkl")
CALIBRATOR_PATH = Path("models/calibration/confidence_calibrator_long_v3.pkl")


def _install_order_send_tripwire() -> None:
    """Any accidental order_send call raises — belt and suspenders."""
    import MetaTrader5 as mt5

    def _blocked(*_a, **_k):
        raise RuntimeError(
            "BLOCKED: paper trading must never call mt5.order_send / execution APIs"
        )

    mt5.order_send = _blocked  # type: ignore[assignment]
    if hasattr(mt5, "order_check"):
        mt5.order_check = _blocked  # type: ignore[assignment]


def _setup_log(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [paper] %(message)s",
    )
    return logging.getLogger("paper_trading")


@dataclass
class PaperPosition:
    entry_time: str
    entry_open_time: str  # H1 open time (raw Date)
    entry_ref_close: float
    entry_fill: float
    sl_level: float
    tp_level: float
    atr: float
    lots: float
    spread_points_entry: float
    regime_tier: str
    atr_zscore: float
    size_mult: float
    raw_probability: float
    calibrated_confidence: float
    confidence_tier: str
    bars_held: int = 0
    equity_at_entry: float = STARTING_EQUITY


@dataclass
class HypotheticalTracker:
    """Blocked-by-guard signal tracked like Gate 4.5."""

    signal_time: str
    entry_open_time: str
    entry_ref_close: float
    entry_fill: float
    sl_level: float
    tp_level: float
    atr: float
    lots: float
    spread_points_entry: float
    atr_zscore: float
    raw_probability: float
    calibrated_confidence: float
    bars_held: int = 0


@dataclass
class EngineState:
    last_processed_h1_open: str | None = None
    equity: float = STARTING_EQUITY
    open_position: dict | None = None
    open_hypotheticals: list[dict] = field(default_factory=list)


def _load_state(path: Path) -> EngineState:
    if not path.exists():
        return EngineState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EngineState(
        last_processed_h1_open=raw.get("last_processed_h1_open"),
        equity=float(raw.get("equity", STARTING_EQUITY)),
        open_position=raw.get("open_position"),
        open_hypotheticals=list(raw.get("open_hypotheticals") or []),
    )


def _save_state(path: Path, state: EngineState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")


def _append_parquet(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_parquet(path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(path, index=False)


def rebuild_features_v3(h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    """
    Reuse the SAME feature functions as offline v1→v2→v3 builds.
    Input Date = MT5 open time (raw CSV convention).
    """
    cfg = FeatureConfig()
    h1c = h1.copy()
    h4c = h4.copy()
    d1c = d1.copy()
    h1c["Date"] = pd.to_datetime(h1c["Date"], utc=True) + pd.Timedelta(hours=1)
    h4c["Date"] = pd.to_datetime(h4c["Date"], utc=True) + pd.Timedelta(hours=4)
    d1c["Date"] = pd.to_datetime(d1c["Date"], utc=True) + pd.Timedelta(days=1)

    v1 = merge_multi_tf_features(
        compute_h1_features(h1c, cfg),
        compute_h4_features(h4c, cfg),
        compute_d1_features(d1c, cfg),
        assert_no_lookahead_samples=0,
    )

    h1_smc = h1c.copy()
    h1_smc["atr_h1"] = AverageTrueRange(
        high=h1_smc["High"], low=h1_smc["Low"], close=h1_smc["Close"], window=14, fillna=False
    ).average_true_range()
    h4_smc = h4c.copy()
    h4_smc["atr_h4"] = AverageTrueRange(
        high=h4_smc["High"], low=h4_smc["Low"], close=h4_smc["Close"], window=14, fillna=False
    ).average_true_range()

    smc_h1 = compute_smc_features(h1_smc, atr_col="atr_h1")
    keep_h1 = ["Date"] + [c for c in SMC_FEATURE_COLS if c not in ("bos_bullish_h4", "bos_bearish_h4")]
    v2 = v1.merge(smc_h1[keep_h1], on="Date", how="left")
    bos_h4 = compute_h4_bos_features(h4_smc, atr_col="atr_h4")
    v2 = pd.merge_asof(v2.sort_values("Date"), bos_h4.sort_values("Date"), on="Date", direction="backward")
    v2 = v2.dropna(subset=["bos_bullish_h4", "bos_bearish_h4"]).reset_index(drop=True)

    # reversal needs OHLC + HTF EMA refs (same as build_features_v3)
    base = v2.merge(
        h1c[["Date", "Open", "High", "Low", "Close"]],
        on="Date",
        how="left",
        suffixes=("", "_ohlc"),
    )
    for c in ("Open", "High", "Low", "Close"):
        alt = f"{c}_ohlc"
        if alt in base.columns:
            base[c] = base[alt].fillna(base.get(c))
            base.drop(columns=[alt], inplace=True, errors="ignore")

    h4_ref = h4c.copy()
    h4_ref["ema_h4_ref"] = EMAIndicator(close=h4_ref["Close"], window=20, fillna=False).ema_indicator()
    d1_ref = d1c.copy()
    d1_ref["ema_d1_ref"] = EMAIndicator(close=d1_ref["Close"], window=20, fillna=False).ema_indicator()
    base = pd.merge_asof(
        base.sort_values("Date"),
        h4_ref[["Date", "ema_h4_ref"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    base = pd.merge_asof(
        base.sort_values("Date"),
        d1_ref[["Date", "ema_d1_ref"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    rev = compute_reversal_features(base)
    v3 = v2.merge(rev, on="Date", how="left")
    v3 = v3.dropna(subset=["dist_from_ema_h4_zscore", "dist_from_ema_d1_zscore"]).reset_index(drop=True)

    # attach OHLC+Spread at close-time for barriers / costs
    ohlc = h1c[["Date", "Open", "High", "Low", "Close", "Spread"]].copy()
    v3 = v3.drop(columns=[c for c in ("Open", "High", "Low", "Close", "Spread") if c in v3.columns], errors="ignore")
    v3 = v3.merge(ohlc, on="Date", how="left")
    return v3


def _predict_raw(model_bundle: dict, row: pd.Series) -> float:
    import lightgbm  # noqa: F401 — model unpickle

    model = model_bundle["model"]
    cols = model_bundle["feature_cols"]
    x = pd.DataFrame([{c: row[c] for c in cols}])
    for c in CAT_COLS:
        if c in x.columns:
            x[c] = x[c].astype("category")
    return float(model.predict_proba(x)[:, 1][0])


def _spread_pts(spread_val: float) -> float:
    if pd.isna(spread_val) or float(spread_val) <= 0:
        return float(FALLBACK_SPREAD_POINTS)
    return float(spread_val)


def _size_lots(equity: float, atr: float, size_mult: float) -> float:
    sl_dist = SL_ATR_MULT * atr
    risk = equity * RISK_PER_TRADE_PCT * size_mult
    lots = _round_lots(risk / (sl_dist * CONTRACT_SIZE))
    return lots if lots >= VOLUME_MIN else 0.0


def _resolve_barrier_bar(
    high: float,
    low: float,
    close: float,
    sl: float,
    tp: float,
    bars_held: int,
) -> str | None:
    """Return outcome if resolved this bar, else None. SL-first."""
    if low <= sl:
        return "sl"
    if high >= tp:
        return "tp"
    if bars_held >= HORIZON_BARS:
        return "timeout"
    return None


def _pnl_long(entry_fill: float, exit_raw: float, spread_exit: float, lots: float) -> float:
    from backtest.engine import BacktestConfig

    cfg = BacktestConfig(apply_costs=True)
    exit_fill = exit_raw - _cost_price(spread_exit, cfg)
    return (exit_fill - entry_fill) * CONTRACT_SIZE * lots


class PaperTradingEngine:
    def __init__(
        self,
        *,
        paper_dir: Path = PAPER_DIR,
        z_block: float = Z_BLOCK,
        log: logging.Logger | None = None,
    ) -> None:
        _install_order_send_tripwire()
        self.log = log or _setup_log()
        self.paper_dir = paper_dir
        self.z_block = z_block
        self.state_path = paper_dir / "state.json"
        self.decisions_path = paper_dir / "decisions_log.parquet"
        self.trades_path = paper_dir / "paper_trades.parquet"
        self.blocked_path = paper_dir / "blocked_hypothetical.parquet"
        self.feed = LiveFeedHandler(paper_dir=paper_dir, log=self.log)
        self.state = _load_state(self.state_path)

        import pickle

        with open(MODEL_PATH, "rb") as f:
            self.model_bundle = pickle.load(f)
        self.confidence = ConfidenceEngine.load(CALIBRATOR_PATH)
        self.log.info(
            "loaded model thr=%s calibrator=%s z_block=%s",
            self.model_bundle.get("selected_thr", RAW_THR),
            CALIBRATOR_PATH,
            z_block,
        )

    def step(self) -> dict[str, Any] | None:
        """Poll feed; if new closed H1, run full decision + tracking. Returns decision row or None."""
        newly = self.feed.poll_once()
        h1_new = newly.get("H1", pd.DataFrame())
        if h1_new is None or h1_new.empty:
            # still advance trackers if we already knew about latest bar? only process once per bar
            return None

        # Process each newly appended H1 in order (usually 1)
        last_row = None
        for _, bar in h1_new.sort_values("Date").iterrows():
            open_ts = pd.Timestamp(bar["Date"])
            last_key = open_ts.isoformat()
            if self.state.last_processed_h1_open and last_key <= self.state.last_processed_h1_open:
                continue
            last_row = self._process_h1_open(open_ts)
            self.state.last_processed_h1_open = last_key
            _save_state(self.state_path, self.state)
        return last_row

    def _process_h1_open(self, h1_open: pd.Timestamp) -> dict[str, Any]:
        """h1_open = MT5 open time of the newly closed bar."""
        h1 = load_raw_csv(self.feed._path("H1"))
        h4 = load_raw_csv(self.feed._path("H4"))
        d1 = load_raw_csv(self.feed._path("D1"))
        # only use bars up to and including this H1 (no future)
        h1 = h1[h1["Date"] <= h1_open].copy()
        # HTF: only fully closed relative to this H1 close time
        h1_close = h1_open + pd.Timedelta(hours=1)
        h4 = h4[h4["Date"] + pd.Timedelta(hours=4) <= h1_close].copy()
        d1 = d1[d1["Date"] + pd.Timedelta(days=1) <= h1_close].copy()

        feats = rebuild_features_v3(h1, h4, d1)
        close_ts = h1_close
        row = feats[feats["Date"] == close_ts]
        if row.empty:
            # warmup / missing HTF
            decision = {
                "timestamp": close_ts.isoformat(),
                "h1_open": h1_open.isoformat(),
                "raw_probability": float("nan"),
                "calibrated_confidence": float("nan"),
                "confidence_tier": "n/a",
                "atr_zscore": float("nan"),
                "regime_tier": "n/a",
                "decision": "skip_warmup",
                "reason": "feature row missing (warmup or HTF lag)",
            }
            _append_parquet(self.decisions_path, decision)
            self._tick_open_trackers(h1, h1_open)
            return decision

        r = row.iloc[-1]
        # regime on full atr history (causal)
        atr_series = feats["atr_h1"].copy()
        z_all = compute_causal_atr_zscore(atr_series, window=ROLLING_WINDOW_BARS)
        tier_all = assign_regime_tier(z_all, z_reduce=Z_REDUCE, z_block=self.z_block)
        atr_z = float(z_all.iloc[-1]) if len(z_all) else float("nan")
        regime = str(tier_all.iloc[-1]) if len(tier_all) else "normal"

        raw_p = _predict_raw(self.model_bundle, r)
        scored = self.confidence.score(raw_p)
        cal_p = scored["calibrated"]
        conf_tier = scored["tier"]

        # First: update open paper + hypotheticals on this bar's OHLC
        self._tick_open_trackers(h1, h1_open)

        decision = {
            "timestamp": close_ts.isoformat(),
            "h1_open": h1_open.isoformat(),
            "raw_probability": raw_p,
            "calibrated_confidence": cal_p,
            "confidence_tier": conf_tier,
            "atr_zscore": atr_z,
            "regime_tier": regime,
            "decision": "",
            "reason": "",
            "close": float(r["Close"]),
            "atr_h1": float(r["atr_h1"]),
            "spread": _spread_pts(r.get("Spread", np.nan)),
        }

        if self.state.open_position is not None:
            decision["decision"] = "skip_open_position"
            decision["reason"] = "paper position already open (no stacking)"
            _append_parquet(self.decisions_path, decision)
            return decision

        if raw_p < RAW_THR:
            decision["decision"] = "skip_threshold"
            decision["reason"] = f"raw_p {raw_p:.4f} < {RAW_THR}"
            _append_parquet(self.decisions_path, decision)
            return decision

        if regime == "extreme":
            decision["decision"] = "skip_guard"
            decision["reason"] = f"blocked by guard z={atr_z:.3f} >= {self.z_block}"
            self._open_hypothetical(r, h1_open, raw_p, cal_p, atr_z)
            _append_parquet(self.decisions_path, decision)
            return decision

        size_mult = 0.5 if regime == "elevated" else 1.0
        atr = float(r["atr_h1"])
        lots = _size_lots(self.state.equity, atr, size_mult)
        if lots <= 0:
            decision["decision"] = "skip_size"
            decision["reason"] = "lots rounded to 0"
            _append_parquet(self.decisions_path, decision)
            return decision

        sp = _spread_pts(r.get("Spread", np.nan))
        from backtest.engine import BacktestConfig

        cfg = BacktestConfig(apply_costs=True)
        entry_ref = float(r["Close"])
        entry_fill = entry_ref + _cost_price(sp, cfg)
        pos = PaperPosition(
            entry_time=close_ts.isoformat(),
            entry_open_time=h1_open.isoformat(),
            entry_ref_close=entry_ref,
            entry_fill=entry_fill,
            sl_level=entry_ref - SL_ATR_MULT * atr,
            tp_level=entry_ref + TP_ATR_MULT * atr,
            atr=atr,
            lots=lots,
            spread_points_entry=sp,
            regime_tier=regime,
            atr_zscore=atr_z,
            size_mult=size_mult,
            raw_probability=raw_p,
            calibrated_confidence=cal_p,
            confidence_tier=conf_tier,
            bars_held=0,
            equity_at_entry=self.state.equity,
        )
        self.state.open_position = asdict(pos)
        decision["decision"] = "trade"
        decision["reason"] = f"open paper lots={lots} size_mult={size_mult}"
        decision["lots"] = lots
        decision["size_mult"] = size_mult
        decision["sl"] = pos.sl_level
        decision["tp"] = pos.tp_level
        _append_parquet(self.decisions_path, decision)
        self.log.info("PAPER OPEN %s p=%.3f regime=%s", close_ts, raw_p, regime)
        return decision

    def _open_hypothetical(self, r: pd.Series, h1_open: pd.Timestamp, raw_p: float, cal_p: float, atr_z: float) -> None:
        atr = float(r["atr_h1"])
        sp = _spread_pts(r.get("Spread", np.nan))
        from backtest.engine import BacktestConfig

        cfg = BacktestConfig(apply_costs=True)
        entry_ref = float(r["Close"])
        lots = _size_lots(self.state.equity, atr, 1.0) or 0.01
        hyp = HypotheticalTracker(
            signal_time=(h1_open + pd.Timedelta(hours=1)).isoformat(),
            entry_open_time=h1_open.isoformat(),
            entry_ref_close=entry_ref,
            entry_fill=entry_ref + _cost_price(sp, cfg),
            sl_level=entry_ref - SL_ATR_MULT * atr,
            tp_level=entry_ref + TP_ATR_MULT * atr,
            atr=atr,
            lots=lots,
            spread_points_entry=sp,
            atr_zscore=atr_z,
            raw_probability=raw_p,
            calibrated_confidence=cal_p,
            bars_held=0,
        )
        self.state.open_hypotheticals.append(asdict(hyp))

    def _tick_open_trackers(self, h1: pd.DataFrame, current_open: pd.Timestamp) -> None:
        """Advance paper + hypothetical trackers using the bar at current_open."""
        bar = h1[h1["Date"] == current_open]
        if bar.empty:
            return
        b = bar.iloc[0]
        high, low, close = float(b["High"]), float(b["Low"]), float(b["Close"])
        sp_x = _spread_pts(b.get("Spread", np.nan))

        # Paper position
        if self.state.open_position is not None:
            pos = self.state.open_position
            entry_open = pd.Timestamp(pos["entry_open_time"])
            if current_open <= entry_open:
                pass  # entry bar itself: backtest starts checking from next bars
            else:
                pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
                outcome = _resolve_barrier_bar(
                    high, low, close, float(pos["sl_level"]), float(pos["tp_level"]), int(pos["bars_held"])
                )
                if outcome:
                    exit_raw = (
                        float(pos["sl_level"])
                        if outcome == "sl"
                        else float(pos["tp_level"])
                        if outcome == "tp"
                        else close
                    )
                    pnl = _pnl_long(float(pos["entry_fill"]), exit_raw, sp_x, float(pos["lots"]))
                    risk = float(pos["equity_at_entry"]) * RISK_PER_TRADE_PCT * float(pos.get("size_mult", 1.0))
                    trade = {
                        **pos,
                        "exit_time": (current_open + pd.Timedelta(hours=1)).isoformat(),
                        "exit_open_time": current_open.isoformat(),
                        "outcome": outcome,
                        "pnl_usd": pnl,
                        "pnl_R": pnl / risk if risk else float("nan"),
                        "bars_held": pos["bars_held"],
                        "spread_points_exit": sp_x,
                    }
                    _append_parquet(self.trades_path, trade)
                    self.state.equity = float(self.state.equity) + pnl
                    self.log.info("PAPER CLOSE %s pnl=%+.2f equity=%.2f", outcome, pnl, self.state.equity)
                    self.state.open_position = None

        # Hypotheticals (blocked)
        still: list[dict] = []
        for hyp in self.state.open_hypotheticals:
            entry_open = pd.Timestamp(hyp["entry_open_time"])
            if current_open <= entry_open:
                still.append(hyp)
                continue
            hyp["bars_held"] = int(hyp.get("bars_held", 0)) + 1
            outcome = _resolve_barrier_bar(
                high, low, close, float(hyp["sl_level"]), float(hyp["tp_level"]), int(hyp["bars_held"])
            )
            if not outcome:
                still.append(hyp)
                continue
            exit_raw = (
                float(hyp["sl_level"])
                if outcome == "sl"
                else float(hyp["tp_level"])
                if outcome == "tp"
                else close
            )
            pnl = _pnl_long(float(hyp["entry_fill"]), exit_raw, sp_x, float(hyp["lots"]))
            risk = self.state.equity * RISK_PER_TRADE_PCT  # approx
            _append_parquet(
                self.blocked_path,
                {
                    **hyp,
                    "exit_time": (current_open + pd.Timedelta(hours=1)).isoformat(),
                    "hypothetical_outcome": outcome,
                    "hypothetical_pnl_usd": pnl,
                    "hypothetical_pnl_R": pnl / risk if risk else float("nan"),
                    "bars_held": hyp["bars_held"],
                },
            )
            self.log.info("HYPOTHETICAL blocked → %s pnl=%+.2f", outcome, pnl)
        self.state.open_hypotheticals = still


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="XAUUSD paper trading (no real orders)")
    parser.add_argument("--once", action="store_true", help="single poll+process cycle")
    parser.add_argument("--loop", action="store_true", help="poll forever")
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="no MT5: seed from historical, rebuild features, score last H1 (sanity)",
    )
    args = parser.parse_args(argv)

    _install_order_send_tripwire()

    if args.offline_smoke:
        from src.live_feed_handler import seed_from_historical

        log = _setup_log()
        paper_dir = args.paper_dir
        seed_from_historical(paper_raw_dir=paper_dir / "raw")
        h1 = load_raw_csv(paper_dir / "raw/XAUUSD_H1.csv")
        h4 = load_raw_csv(paper_dir / "raw/XAUUSD_H4.csv")
        d1 = load_raw_csv(paper_dir / "raw/XAUUSD_D1.csv")
        log.info("offline smoke rebuild features n_h1=%s", len(h1))
        feats = rebuild_features_v3(h1, h4, d1)
        r = feats.iloc[-1]
        import pickle

        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        raw_p = _predict_raw(bundle, r)
        conf = ConfidenceEngine.load(CALIBRATOR_PATH).score(raw_p)
        z = compute_causal_atr_zscore(feats["atr_h1"], window=ROLLING_WINDOW_BARS)
        tier = assign_regime_tier(z, z_reduce=Z_REDUCE, z_block=Z_BLOCK)
        print(
            {
                "last_close_time": str(r["Date"]),
                "raw_p": raw_p,
                **conf,
                "atr_z": float(z.iloc[-1]),
                "regime": str(tier.iloc[-1]),
                "n_feat_rows": len(feats),
            }
        )
        return 0

    if not args.once and not args.loop:
        args.once = True  # safe default: one shot

    eng = PaperTradingEngine(paper_dir=args.paper_dir)
    if args.once:
        row = eng.step()
        print(row or {"status": "no_new_closed_h1"})
        _save_state(eng.state_path, eng.state)
        return 0

    eng.log.info("paper loop start poll_sec=%s (NO REAL ORDERS)", args.poll_sec)
    while True:
        try:
            eng.step()
            _save_state(eng.state_path, eng.state)
        except Exception:
            eng.log.exception("step failed")
        time.sleep(max(5, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
