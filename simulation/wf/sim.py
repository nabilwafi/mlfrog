"""Portfolio sim for rolling WF: fixed 0.01 lot, $80 start, ruin → stop trading.

ruin_stop=True: equity <= 0 (or wipe) → no further opens for that run/year.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_SIZE = 100.0
H1_HOURS = 1.0
COST = 1.5e-4
SL_ATR = 1.5
TRAIL = 0.12
ACTIVATE_R = 0.5
HORIZON = 16
MAX_OPEN = 1
RISK_BASE = 0.01
DAILY_LOSS_STOP_R = 1.0
FIXED_LOT = 0.01
LEVERAGE = 500.0
STOP_OUT_LEVEL = 0.20
SPLITS = ("train", "validation", "test", "sealed")


def required_margin(lots: float, price: float, leverage: float) -> float:
    if leverage <= 0:
        return 0.0
    return float(lots) * CONTRACT_SIZE * float(price) / float(leverage)


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def wilder_atr(h1: pd.DataFrame, period: int = 14) -> pd.Series:
    c = h1["close"].astype(float)
    h = h1["high"].astype(float)
    l = h1["low"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def load_h1(path: Path) -> pd.DataFrame:
    h = pd.read_parquet(path)
    h = h.sort_values("timestamp").reset_index(drop=True)
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    return h


def prepare_market(h1: pd.DataFrame) -> dict:
    h = h1.copy()
    h["atr"] = wilder_atr(h)
    return {
        "ts": pd.DatetimeIndex(h["timestamp"]),
        "high": h["high"].to_numpy(dtype=float),
        "low": h["low"].to_numpy(dtype=float),
        "close": h["close"].to_numpy(dtype=float),
        "atr": h["atr"].to_numpy(dtype=float),
    }


def entry_indices(entries: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
    ets = pd.to_datetime(entries["timestamp"], utc=True)
    return ts.searchsorted(ets, side="left").astype(int)


def _signed(side: str, entry: float, px: float) -> float:
    return (px - entry) / entry if side == "long" else (entry - px) / entry


def replay_trail(*, side: str, entry: float, atr: float, ei: int, mkt: dict, trail: float) -> dict | None:
    high, low, close, atr_s = mkt["high"], mkt["low"], mkt["close"], mkt["atr"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr <= 0 or entry <= 0:
        return None
    is_long = side == "long"
    one_r = SL_ATR * atr
    sl = entry - SL_ATR * atr if is_long else entry + SL_ATR * atr
    init_sl = sl
    extreme = entry
    hard = min(ei + HORIZON, n - 1)
    exit_px = float(close[hard])
    j_exit = hard
    reason = "TIMEOUT"
    for j in range(ei + 1, hard + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr
        if is_long:
            extreme = max(extreme, h)
            if (extreme - entry) >= ACTIVATE_R * one_r:
                sl = max(sl, extreme - trail * atr_j)
            hit_sl = l <= sl
        else:
            extreme = min(extreme, l)
            if (entry - extreme) >= ACTIVATE_R * one_r:
                sl = min(sl, extreme + trail * atr_j)
            hit_sl = h >= sl
        if hit_sl:
            exit_px = sl
            reason = "TRAIL" if abs(sl - init_sl) > 1e-9 else "SL"
            j_exit = j
            break
    return {
        "net_return": float(_signed(side, entry, exit_px) - COST),
        "holding_bars": int(j_exit - ei),
        "exit_reason": reason,
    }


def _load_side(side: str) -> pd.DataFrame:
    base = _ROOT / f"artifacts/datasets/XAUUSD/H1/{side}/v2"
    parts = [pd.read_parquet(base / f"{s}.parquet") for s in SPLITS]
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["side"] = side
    lab = pd.read_parquet(
        _ROOT / f"artifacts/labels/XAUUSD/H1/{side}/triple_barrier_v1.parquet",
        columns=["timestamp", "realized_return", "holding_bars", "entry_price"],
    )
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
    d = d.merge(lab, on="timestamp", how="left")
    return d.dropna(subset=["label", "realized_return"]).sort_values("timestamp")


def _feat_names(df: pd.DataFrame) -> list[str]:
    model_feat = lgb.Booster(
        model_file=str(_ROOT / "artifacts/models/frozen/primary_long.txt")
    ).feature_name()
    present = [f for f in model_feat if f in df.columns]
    missing = [f for f in model_feat if f not in df.columns]
    if missing:
        print(f"  warn: dropping {len(missing)} missing model feats: {missing}")
    return present


def max_dd_from_equity(eq: np.ndarray, starting: float) -> float:
    curve = np.concatenate([[starting], eq.astype(float)])
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def profit_factor_pnl(pnl: np.ndarray) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def run_portfolio(
    panel: pd.DataFrame,
    *,
    starting: float,
    ruin_stop: bool,
    leverage: float = LEVERAGE,
    stop_out_level: float = STOP_OUT_LEVEL,
) -> tuple[pd.DataFrame, dict]:
    """Fixed 0.01 lot + optional leverage margin / stop-out.

    Leverage does NOT multiply PnL. It only:
      - blocks opens when equity < required margin
      - stop-out if equity would fall below stop_out_level * margin
      - floor equity at 0 when wiped
    With ruin_stop: equity <= 0 → skip all further opens.
    """
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting)
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str, float]] = []
    rows: list[dict] = []
    blown = False
    blown_ts = None
    skipped_after_ruin = 0
    skipped_margin = 0
    stop_outs = 0
    margins: list[float] = []

    for _, row in t.iterrows():
        ts = _to_ts(row["timestamp"])
        d = ts.date()
        if day != d:
            day = d
            day_pnl = 0.0
        opens = [(e, s, m) for e, s, m in opens if e > ts]
        used_margin = float(sum(m for _, _, m in opens))

        if ruin_stop and equity <= 0:
            blown = True
            skipped_after_ruin += 1
            continue

        side = str(row["side"]).lower()
        if len(opens) >= MAX_OPEN:
            continue
        # ponytail: opposite-side block; with MAX_OPEN=1 this rarely runs
        if opens and side not in {s for _, s, _ in opens}:
            continue
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            if equity <= 0:
                blown = True
                skipped_after_ruin += 1
            continue

        lots = FIXED_LOT
        entry = float(row["entry_price"])
        atr = float(row.get("atr_price", 0.0) or 0.0)
        margin = required_margin(lots, entry, leverage) if leverage > 0 else 0.0
        free_margin = equity - used_margin
        if leverage > 0 and free_margin < margin:
            skipped_margin += 1
            continue

        raw_pnl = lots * CONTRACT_SIZE * entry * float(row["net_return"])
        one_r_loss = lots * CONTRACT_SIZE * (SL_ATR * atr) if atr > 0 else abs(min(raw_pnl, 0.0))
        trough_equity = equity - one_r_loss
        stop_line = margin * float(stop_out_level) if leverage > 0 else 0.0
        stopped = bool(leverage > 0 and trough_equity <= stop_line)

        if stopped:
            pnl = -(equity - stop_line)
            equity = max(0.0, float(stop_line))
            stop_outs += 1
            if equity <= 0:
                blown = True
                blown_ts = ts
        elif ruin_stop and equity + raw_pnl <= 0:
            pnl = -equity
            equity = 0.0
            blown = True
            blown_ts = ts
        else:
            pnl = raw_pnl
            equity += pnl
            if equity <= 0:
                blown = True
                blown_ts = ts

        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side, margin))
        margins.append(margin)
        rows.append(
            {
                "timestamp": ts,
                "year": int(ts.year),
                "side": side,
                "lots": lots,
                "margin": margin,
                "pnl": pnl,
                "equity": equity,
                "holding_bars": int(row["holding_bars"]),
                "stop_out": stopped,
                "wiped": bool(ruin_stop and equity <= 0 and pnl < 0),
            }
        )

    traded = pd.DataFrame(rows)
    empty = {
        "n_trades": 0,
        "total_return": 0.0,
        "final_equity": float(starting),
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "blown": blown,
        "blown_ts": None,
        "skipped_after_ruin": skipped_after_ruin,
        "skipped_margin": skipped_margin,
        "stop_outs": stop_outs,
        "avg_holding_bars": 0.0,
        "avg_margin": 0.0,
        "leverage": float(leverage),
    }
    if traded.empty:
        return traded, empty
    pnl = traded["pnl"].to_numpy(dtype=float)
    return traded, {
        "n_trades": int(len(traded)),
        "total_return": float(traded["equity"].iloc[-1] / starting - 1.0),
        "final_equity": float(traded["equity"].iloc[-1]),
        "max_drawdown": max_dd_from_equity(traded["equity"].to_numpy(dtype=float), starting),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "blown": blown,
        "blown_ts": blown_ts.isoformat() if blown_ts is not None else None,
        "skipped_after_ruin": int(skipped_after_ruin),
        "skipped_margin": int(skipped_margin),
        "stop_outs": int(stop_outs),
        "avg_holding_bars": float(traded["holding_bars"].mean()),
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "leverage": float(leverage),
    }


if __name__ == "__main__":
    # ponytail: ruin stop self-check — wipe then skip rest
    ts0 = pd.Timestamp("2020-01-01", tz="UTC")
    panel = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "side": "long",
                "entry_price": 2000.0,
                "atr_price": 10.0,
                "net_return": -1.0,  # wipe
                "holding_bars": 1,
            },
            {
                "timestamp": ts0 + pd.Timedelta(hours=2),
                "side": "long",
                "entry_price": 2000.0,
                "atr_price": 10.0,
                "net_return": 0.01,
                "holding_bars": 1,
            },
        ]
    )
    traded, m = run_portfolio(panel, starting=80.0, ruin_stop=True, leverage=0.0)
    assert m["blown"] and m["final_equity"] == 0.0 and m["n_trades"] == 1
    assert m["skipped_after_ruin"] >= 1
    print("ok: ruin_stop skips after wipe")
