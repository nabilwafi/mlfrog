"""Realistic execution sim for trail policy (bid/ask, hour-spread, slip, latency, gap, requote).

H1 has no ticks — latency/gap/requote are bar-path approximations (documented below).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from apps.run_exit_engine_hunt import build_entries, metrics_for
from apps.run_primary_loosen_backtest import run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt.services.paths import entry_indices, prepare_market, wilder_atr
from settings.strategy import CONTRACT_SIZE, POINT, SL_ATR_MULT, TP_ATR_MULT

# --- Hour-varying spread (points). Asia tight-ish, London wider, NY open widest, late quieter.
# Finex baseline ~19pt mid; scale by session.
_SPREAD_PTS_BY_HOUR = {
    0: 22, 1: 24, 2: 26, 3: 28, 4: 26, 5: 22,
    6: 20, 7: 18, 8: 17, 9: 18, 10: 19, 11: 20,
    12: 22, 13: 24, 14: 23, 15: 21, 16: 19, 17: 18,
    18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22,
}


@dataclass
class ExecConfig:
    name: str
    # commission as fraction of notional per side (entry+exit both charged)
    commission_frac_side: float = 0.0
    # slip ~ U(0, slip_atr_max) * atr, adverse
    slip_atr_max: float = 0.05
    # latency seconds → adverse fill as fraction of bar range (1s≈1/3600 of H1; we use stronger proxy)
    latency_sec: tuple[float, float] = (1.0, 3.0)
    latency_range_frac: float = 0.15  # up to 15% of bar range adverse on stop fills
    requote_prob: float = 0.03
    requote_extra_slip_atr: float = 0.03
    gap_fill_at_open: bool = True
    seed: int = 42


def spread_price(hour: int, *, mult: float = 1.0) -> float:
    pts = _SPREAD_PTS_BY_HOUR.get(int(hour) % 24, 20)
    return float(pts) * float(POINT) * float(mult)


def _simulate_trade(
    *,
    side: str,
    ei: int,
    entry_mid: float,
    atr: float,
    mkt: dict,
    cfg: ExecConfig,
    rng: np.random.Generator,
    trail_atr: float = 0.12,
    activate_r: float = 0.5,
    horizon: int = 16,
    spread_mult: float = 1.0,
) -> dict:
    """
    Bid/Ask path on H1:
      ask = mid + spread/2, bid = mid - spread/2
    Long: enter ask+slip, exit bid-slip; Short opposite.
    Trail mirrors research: after +0.5R, ratchet SL by trail_atr*atr (entry atr).
    """
    high, low, close, ts = mkt["high"], mkt["low"], mkt["close"], mkt["ts"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr <= 0 or entry_mid <= 0:
        return {"ok": False}

    # Requote on entry: worse slip or skip
    requoted = False
    if rng.random() < cfg.requote_prob:
        requoted = True
        if rng.random() < 0.35:
            return {
                "ok": False,
                "reason": "REQUOTE_REJECT",
                "requoted": True,
            }

    hour0 = int(pd.Timestamp(ts[ei]).hour)
    sp0 = spread_price(hour0, mult=spread_mult)
    slip0 = float(rng.uniform(0.0, cfg.slip_atr_max)) * atr
    if requoted:
        slip0 += float(cfg.requote_extra_slip_atr) * atr

    is_long = side == "long"
    # entry on next bar open (decision at ei close) — use ei+1 open≈close[ei] proxy: close[ei]
    # Fill at bar ei (signal bar close) like rest of stack; apply ask/bid
    mid_e = float(entry_mid)
    if is_long:
        entry = mid_e + sp0 / 2.0 + slip0
    else:
        entry = mid_e - sp0 / 2.0 - slip0

    one_r = float(SL_ATR_MULT) * atr
    if is_long:
        sl = entry - float(SL_ATR_MULT) * atr
        tp = entry + float(TP_ATR_MULT) * atr
        extreme = entry
    else:
        sl = entry + float(SL_ATR_MULT) * atr
        tp = entry - float(TP_ATR_MULT) * atr
        extreme = entry

    hard_end = min(ei + horizon, n - 1)
    reason = "TIMEOUT"
    exit_px = float(close[hard_end])
    j_exit = hard_end
    lat_sec = float(rng.uniform(*cfg.latency_sec))

    for j in range(ei + 1, hard_end + 1):
        h = float(high[j])
        l = float(low[j])
        c = float(close[j])
        hour = int(pd.Timestamp(ts[j]).hour)
        sp = spread_price(hour, mult=spread_mult)
        # quote extremes inside bar
        ask_h, bid_h = h + sp / 2.0, h - sp / 2.0
        ask_l, bid_l = l + sp / 2.0, l - sp / 2.0
        ask_c, bid_c = c + sp / 2.0, c - sp / 2.0
        bar_range = max(h - l, atr * 0.05)

        # Gap: true open jumps through SL (not a normal intra-bar SL touch)
        o = float(mkt["open"][j]) if "open" in mkt else float(close[j - 1]) if j > 0 else c
        if cfg.gap_fill_at_open and j > 0:
            prev_c = float(close[j - 1])
            gapped_long = is_long and prev_c > sl and o < sl
            gapped_short = (not is_long) and prev_c < sl and o > sl
            if gapped_long or gapped_short:
                slip = float(rng.uniform(0.0, cfg.slip_atr_max)) * atr
                adverse = (lat_sec / 3.0) * cfg.latency_range_frac * bar_range
                if is_long:
                    # stop sell into gap — fill near open bid
                    exit_px = o - sp / 2.0 - slip - adverse
                else:
                    exit_px = o + sp / 2.0 + slip + adverse
                reason, j_exit = "GAP_SL", j
                break

        # update extreme / trail on mid path (favorable)
        if is_long:
            extreme = max(extreme, h)
            fav = extreme - entry
            if fav >= activate_r * one_r:
                sl = max(sl, extreme - trail_atr * atr)
        else:
            extreme = min(extreme, l)
            fav = entry - extreme
            if fav >= activate_r * one_r:
                sl = min(sl, extreme + trail_atr * atr)

        hit_sl = (bid_l <= sl) if is_long else (ask_h >= sl)
        hit_tp = (bid_h >= tp) if is_long else (ask_l <= tp)

        if hit_sl and hit_tp:
            hit_sl = True  # SL first
            hit_tp = False

        if hit_sl:
            slip = float(rng.uniform(0.0, cfg.slip_atr_max)) * atr
            # latency: stop trigger seen late → worse fill into bar
            adverse = (lat_sec / 3.0) * cfg.latency_range_frac * bar_range
            if is_long:
                exit_px = sl - slip - adverse  # sell into bid, worse
            else:
                exit_px = sl + slip + adverse
            init_sl = entry - float(SL_ATR_MULT) * atr if is_long else entry + float(SL_ATR_MULT) * atr
            reason = "TRAIL" if abs(sl - init_sl) > 1e-6 else "SL"
            j_exit = j
            break

        if hit_tp:
            slip = float(rng.uniform(0.0, cfg.slip_atr_max)) * atr
            if is_long:
                exit_px = tp - slip  # sell bid side of TP
                exit_px = min(exit_px, bid_h)
            else:
                exit_px = tp + slip
                exit_px = max(exit_px, ask_l)
            reason, j_exit = "TP", j
            break

        if j >= hard_end:
            slip = float(rng.uniform(0.0, cfg.slip_atr_max)) * atr
            if is_long:
                exit_px = bid_c - slip
            else:
                exit_px = ask_c + slip
            reason, j_exit = "TIMEOUT", j
            break

    if is_long:
        gross = (exit_px - entry) / entry
    else:
        gross = (entry - exit_px) / entry

    # commission both sides
    commission = 2.0 * float(cfg.commission_frac_side)
    net = gross - commission
    return {
        "ok": True,
        "net_return": float(net),
        "holding_bars": int(max(j_exit - ei, 0)),
        "exit_reason": reason,
        "entry": float(entry),
        "exit": float(exit_px),
        "requoted": requoted,
        "latency_sec": lat_sec,
        "spread_entry": sp0,
    }


def run_scenario(
    pool: pd.DataFrame,
    mkt: dict,
    eis: np.ndarray,
    cfg: ExecConfig,
    *,
    spread_mult: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    rejects = 0
    for pos in range(len(pool)):
        row = pool.iloc[pos]
        ei = int(eis[pos])
        atr = float(row["atr_entry"] if pd.notna(row.get("atr_entry")) else row.get("atr_price", np.nan))
        res = _simulate_trade(
            side=str(row["side"]),
            ei=ei,
            entry_mid=float(row["entry_price"]),
            atr=atr,
            mkt=mkt,
            cfg=cfg,
            rng=rng,
            spread_mult=spread_mult,
        )
        if not res.get("ok"):
            rejects += 1
            continue
        rows.append(
            {
                "timestamp": row["timestamp"],
                "side": row["side"],
                "y_prob": float(row["y_prob"]),
                "entry_price": res["entry"],
                "atr_price": atr,
                "net_return": res["net_return"],
                "holding_bars": res["holding_bars"],
                "exit_reason": res["exit_reason"],
                "requoted": res["requoted"],
                "latency_sec": res["latency_sec"],
                "spread_entry": res["spread_entry"],
            }
        )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel, {"n_trades": 0, "rejects": rejects}
    # portfolio settle (same as production max_open=1)
    traded, m = run_portfolio(panel, max_open=1, starting_equity=STARTING_EQUITY)
    # exit reason mix
    vc = panel["exit_reason"].value_counts(normalize=True).to_dict()
    m["rejects"] = rejects
    m["exit_mix"] = {str(k): float(v) for k, v in vc.items()}
    m["avg_spread_entry"] = float(panel["spread_entry"].mean())
    m["avg_latency_sec"] = float(panel["latency_sec"].mean())
    m["requote_rate"] = float(panel["requoted"].mean()) if len(panel) else 0.0
    m["median_hold"] = float(panel["holding_bars"].median())
    # isolated yearly avg WR/PF
    iso = metrics_for(panel, max_open=1)
    m["avg_iso_wr"] = iso.get("avg_iso_wr")
    m["avg_iso_pf"] = iso.get("avg_iso_pf")
    return panel, m


def main() -> None:
    print("Execution realism (H1 approx)")
    print("  - Spread: hour-varying points (Finex-ish ~17-28pt)")
    print("  - Bid/Ask fills (long buy ask / sell bid)")
    print("  - Slip: U(0, slip_atr_max)*ATR adverse")
    print("  - Commission: frac notional each side")
    print("  - Latency 1-3s: worse stop fill vs bar range")
    print("  - Requote: p reject or extra slip")
    print("  - Gap: open through SL fills worse than SL\n")

    years = list(range(2015, 2027))
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    if "open" not in h1.columns:
        h1["open"] = h1["close"].shift(1).fillna(h1["close"])
    mkt = prepare_market(h1)
    mkt["open"] = h1["open"].to_numpy(dtype=float)
    entries = build_entries(long_df, short_df, h1, top_pct=0.05, years=years)
    e = entries.copy()
    e["hour"] = pd.to_datetime(e["timestamp"], utc=True).dt.hour
    pool = e[(e["hour"] >= 9) & (e["hour"] <= 15)].drop(columns=["hour"]).reset_index(drop=True)
    eis = entry_indices(pool, mkt["ts"])

    scenarios = [
        (
            "lab_mid_no_friction",
            ExecConfig(name="lab", commission_frac_side=0.0, slip_atr_max=0.0, requote_prob=0.0, latency_sec=(0, 0), latency_range_frac=0.0, gap_fill_at_open=False),
            0.0,  # special: skip — use old trail sim? We'll still run bid/ask with 0 slip 0 spread via mult=0
        ),
        (
            "realistic_base",
            ExecConfig(
                name="base",
                commission_frac_side=0.00002,  # ~0.2bp/side
                slip_atr_max=0.05,
                latency_sec=(1.0, 3.0),
                latency_range_frac=0.12,
                requote_prob=0.03,
                seed=42,
            ),
            1.0,
        ),
        (
            "realistic_wider_spread",
            ExecConfig(
                name="wide_sp",
                commission_frac_side=0.00002,
                slip_atr_max=0.05,
                latency_sec=(1.0, 3.0),
                latency_range_frac=0.12,
                requote_prob=0.03,
                seed=42,
            ),
            1.5,
        ),
        (
            "harsh_slip_0.10atr",
            ExecConfig(
                name="harsh_slip",
                commission_frac_side=0.00003,
                slip_atr_max=0.10,
                latency_sec=(1.0, 3.0),
                latency_range_frac=0.20,
                requote_prob=0.05,
                requote_extra_slip_atr=0.05,
                seed=7,
            ),
            1.25,
        ),
        (
            "stress_latency_heavy",
            ExecConfig(
                name="lat",
                commission_frac_side=0.00002,
                slip_atr_max=0.05,
                latency_sec=(2.0, 3.0),
                latency_range_frac=0.35,
                requote_prob=0.05,
                seed=99,
            ),
            1.0,
        ),
    ]

    # lab: zero spread mult
    results = []
    for name, cfg, sp_mult in scenarios:
        if name == "lab_mid_no_friction":
            # mid-only: force spread_mult=0 and no slip/latency
            sp_mult = 0.0
        panel, m = run_scenario(pool, mkt, eis, cfg, spread_mult=sp_mult)
        m["scenario"] = name
        m["spread_mult"] = sp_mult
        results.append(m)
        print(
            f"{name:28} n={m.get('n_trades', 0):4d} rej={m.get('rejects', 0):3d} "
            f"PF={m.get('cont_pf', m.get('profit_factor', float('nan'))):5.2f} "
            f"ret={m.get('total_return', float('nan')):+7.1%} "
            f"DD={m.get('max_drawdown', float('nan')):5.1%} "
            f"WR={m.get('win_rate', float('nan')):5.1%} "
            f"isoWR={m.get('avg_iso_wr', float('nan')):5.1%} "
            f"sp={m.get('avg_spread_entry', 0):.3f} "
            f"req={m.get('requote_rate', 0):.1%}"
        )
        mix = m.get("exit_mix") or {}
        if mix:
            top = ", ".join(f"{k}={v:.0%}" for k, v in sorted(mix.items(), key=lambda x: -x[1])[:4])
            print(f"{'':28} exits: {top}")

    out = _ROOT / "artifacts/pipeline_backtest/trail012_execution_realism.json"
    out.write_text(
        json.dumps(
            {
                "note": (
                    "H1 bar approximation of tick execution. "
                    "Latency modeled as adverse fraction of bar range on stop fills. "
                    "Spread schedule is synthetic Finex-like points by UTC hour."
                ),
                "policy": "top5% + sess 09-15 + trail 0.12 + max_open 1",
                "results": results,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print("\nwrote", out)


if __name__ == "__main__":
    main()
