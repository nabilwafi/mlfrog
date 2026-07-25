"""WF 2015-2026 @ $80 fixed 0.01 — stop trading for the year if equity <= 0.

Isolated = reset $80 each year (ruin only kills that year).
Continuous = compound; ruin kills the rest of the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

OUT = _ROOT / "artifacts/pipeline_backtest"
PANEL_CACHE = OUT / "wf_top5_trail012_panel.parquet"
# Exit-research trail panel (same ATR 0.12 path); used when WF rebuild isn't needed.
FROZEN_TRAIL = _ROOT / "artifacts/exit_research/atr_trail_0.12_trades.parquet"
STARTING = 80.0
YEARS = list(range(2015, 2027))
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
LEVERAGE = 500.0  # Finex-style; margin = notional / leverage
STOP_OUT_LEVEL = 0.20  # stop-out when equity / used_margin < 20%
TOP_PCT = 0.05
LOOKBACK = 7
SPLITS = ("train", "validation", "test", "sealed")


def required_margin(lots: float, price: float, leverage: float) -> float:
    """USD margin to open: (lots * contract_oz * price) / leverage."""
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
    """Intersect frozen model features with columns present on the dataset."""
    model_feat = lgb.Booster(
        model_file=str(_ROOT / "artifacts/models/frozen/primary_long.txt")
    ).feature_name()
    present = [f for f in model_feat if f in df.columns]
    missing = [f for f in model_feat if f not in df.columns]
    if missing:
        print(f"  warn: dropping {len(missing)} missing model feats: {missing}")
    return present


def _train_predict(train: pd.DataFrame, test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    n = len(train)
    cut = max(int(n * 0.8), 1)
    tr, va = train.iloc[:cut], train.iloc[cut:]
    if len(va) < 50:
        tr, va = train, train.iloc[-max(50, n // 10) :]
    y_tr = tr["label"].astype(int).to_numpy()
    y_va = va["label"].astype(int).to_numpy()
    spw = max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(spw),
        "verbosity": -1,
        "seed": 42,
    }
    dtr = lgb.Dataset(tr[feat], label=y_tr, feature_name=list(feat), free_raw_data=False)
    dva = lgb.Dataset(va[feat], label=y_va, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster.predict(test[feat])


def build_year_panel(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    h1: pd.DataFrame,
    feat: list[str],
    *,
    year: int,
) -> pd.DataFrame:
    lo = year - LOOKBACK
    rows = []
    for side, df in (("long", long_df), ("short", short_df)):
        years = df["timestamp"].dt.year
        train = df[(years >= lo) & (years < year)]
        test = df[years == year]
        if len(train) < 500 or test.empty:
            continue
        prob = _train_predict(train, test, feat)
        g = test[["timestamp", "realized_return", "holding_bars", "entry_price"]].copy()
        g["side"] = side
        g["y_prob"] = prob
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
    k = max(1, int(round(len(p) * float(TOP_PCT))))
    u = p.nlargest(k, "y_prob").sort_values("timestamp")
    h = h1.copy()
    h["atr"] = wilder_atr(h)
    m = u.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["entry_price"].fillna(m["close"]).astype(float)
    m["atr_price"] = m["atr"].astype(float)
    m["atr_entry"] = m["atr_price"]
    return m.dropna(subset=["entry_price", "atr_price"])


def load_frozen_trail_panel() -> pd.DataFrame:
    p = pd.read_parquet(FROZEN_TRAIL)
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    return p.sort_values("timestamp").reset_index(drop=True)


def build_trail_panel(*, rebuild: bool = False) -> pd.DataFrame:
    if not rebuild:
        if PANEL_CACHE.exists():
            print(f"loading cached panel {PANEL_CACHE}")
            p = pd.read_parquet(PANEL_CACHE)
            p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
            return p
        print(f"loading frozen trail panel {FROZEN_TRAIL}")
        return load_frozen_trail_panel()

    print("Building WF Primary top5% + ATR trail 0.12 panel…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    feat = _feat_names(long_df)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    parts = []
    for y in YEARS:
        e = build_year_panel(long_df, short_df, h1, feat, year=y)
        print(f"  year {y}: candidates={len(e)}")
        if e.empty:
            continue
        parts.append(e)
    entries = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    eis = entry_indices(entries, mkt["ts"])
    rows = []
    for i in range(len(entries)):
        src = entries.iloc[i]
        r = replay_trail(
            side=str(src["side"]),
            entry=float(src["entry_price"]),
            atr=float(src["atr_entry"]),
            ei=int(eis[i]),
            mkt=mkt,
            trail=TRAIL,
        )
        if r is None:
            continue
        rows.append(
            {
                "timestamp": src["timestamp"],
                "side": src["side"],
                "y_prob": float(src["y_prob"]),
                "entry_price": float(src["entry_price"]),
                "atr_price": float(src["atr_entry"]),
                "net_return": r["net_return"],
                "holding_bars": r["holding_bars"],
                "exit_reason": r["exit_reason"],
            }
        )
    panel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    print(f"cached {len(panel)} rows -> {PANEL_CACHE}")
    return panel


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
      - stop-out if equity would fall below stop_out_level * margin (approx via initial 1R)
      - floor equity at 0 when wiped
    """
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting)
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str, float]] = []  # exit_ts, side, margin
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

        # Ruin: equity already dead → no more opens this run/year.
        if ruin_stop and equity <= 0:
            blown = True
            skipped_after_ruin += 1
            continue

        side = str(row["side"]).lower()
        if len(opens) >= MAX_OPEN:
            continue
        if opens and side not in {s for _, s in opens}:
            continue
        r_unit = equity * RISK_BASE
        # Daily heat −1R (same as production). Also blocks if equity already <= 0.
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
        # Approx intra-trade trough: full initial 1R adverse (1.5×ATR) before trail helps.
        one_r_loss = lots * CONTRACT_SIZE * (SL_ATR * atr) if atr > 0 else abs(min(raw_pnl, 0.0))
        trough_equity = equity - one_r_loss
        stop_line = margin * float(stop_out_level) if leverage > 0 else 0.0
        stopped = bool(leverage > 0 and trough_equity <= stop_line)

        if stopped:
            # Stop-out realizes loss down to stop line (margin released after).
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


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_trail_panel(rebuild=rebuild)

    # Compare: no leverage gate vs 1:500 margin/stop-out
    modes = [
        ("no_lev", 0.0),
        ("lev_500", 500.0),
    ]

    all_payload: dict = {
        "starting_equity": STARTING,
        "fixed_lot": FIXED_LOT,
        "ruin_stop": True,
        "stop_out_level": STOP_OUT_LEVEL,
        "policy": "Primary WF top5% | ATR Trail 0.12 | max_open=1 | fixed 0.01",
        "modes": {},
    }

    md = [
        "# Walk-Forward 2015-2026 @ $80 — leverage + ruin stop",
        "",
        "Policy: Primary top 5% trail panel | ATR Trail 0.12 | max_open=1 | fixed 0.01 lot",
        "",
        "Leverage **does not multiply PnL**. At 1:500:",
        f"- margin ≈ `(0.01 × 100 × price) / 500` (e.g. gold $2000 → ~$4 margin)",
        f"- skip open if free margin < required margin",
        f"- stop-out approx if equity would fall below **{STOP_OUT_LEVEL:.0%}** of used margin (using 1R=1.5×ATR adverse)",
        "- ruin: equity ≤ 0 → no more trades until year reset",
        "",
    ]

    for mode_name, lev in modes:
        print(f"\n=== {mode_name} (leverage={lev:g}) ===")
        print(f"panel={len(panel)} lot={FIXED_LOT} start=${STARTING}")
        yearly = []
        for y in YEARS:
            g = panel[pd.to_datetime(panel["timestamp"], utc=True).dt.year == y]
            if g.empty:
                continue
            _, ym = run_portfolio(g, starting=STARTING, ruin_stop=True, leverage=lev)
            yearly.append({"year": y, **ym})
            flag = " BLOWN" if ym["blown"] else ""
            print(
                f"  {y}: n={ym['n_trades']:4d} ret={ym['total_return']:+7.1%} "
                f"final=${ym['final_equity']:.1f} dd={ym['max_drawdown']:.1%} "
                f"marg_skip={ym['skipped_margin']} stopouts={ym['stop_outs']} "
                f"avg_m=${ym['avg_margin']:.2f}{flag}"
            )

        traded, overall = run_portfolio(panel, starting=STARTING, ruin_stop=True, leverage=lev)
        all_payload["modes"][mode_name] = {
            "leverage": lev,
            "continuous": overall,
            "isolated_yearly": yearly,
        }
        pd.DataFrame(yearly).to_csv(
            OUT / f"wf_2015_2026_eq80_{mode_name}_isolated.csv", index=False
        )
        if not traded.empty:
            traded.to_parquet(
                OUT / f"wf_2015_2026_eq80_{mode_name}_continuous_trades.parquet",
                index=False,
            )

        md += [
            f"## {mode_name} (leverage={lev:g})",
            "",
            (
                f"Continuous: trades={overall['n_trades']} ret={overall['total_return']:+.1%} "
                f"pf={overall['profit_factor']:.2f} wr={overall['win_rate']:.1%} "
                f"dd={overall['max_drawdown']:.1%} final=${overall['final_equity']:.1f} "
                f"margin_skip={overall['skipped_margin']} stop_outs={overall['stop_outs']} "
                f"avg_margin=${overall['avg_margin']:.2f} blown={overall['blown']}"
            ),
            "",
            "| Year | Trades | Return | Final | DD | AvgMargin | MargSkip | StopOut | Blown |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in yearly:
            md.append(
                f"| {r['year']} | {r['n_trades']} | {r['total_return']:+.1%} | "
                f"${r['final_equity']:.1f} | {r['max_drawdown']:.1%} | "
                f"${r['avg_margin']:.2f} | {r['skipped_margin']} | {r['stop_outs']} | "
                f"{'YES' if r['blown'] else ''} |"
            )
        md.append("")

    (OUT / "wf_2015_2026_eq80_lev.json").write_text(
        json.dumps(all_payload, indent=2, default=float), encoding="utf-8"
    )
    (OUT / "wf_2015_2026_eq80_lev.md").write_text("\n".join(md), encoding="utf-8")
    # keep ruin.md alias pointing at lev500 summary line
    lev = all_payload["modes"]["lev_500"]
    (OUT / "wf_2015_2026_eq80_ruin.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote", OUT / "wf_2015_2026_eq80_lev.md")
    print("lev_500 continuous:", lev["continuous"])


if __name__ == "__main__":
    main()
