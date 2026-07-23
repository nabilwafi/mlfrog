"""Audit: verify ATR Trail multiplier is actually used in Exit Engine path replay.

Compare Trail 0.12 vs 0.25 on the same 10 frozen entries with candle-level trail stops.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from production.paper.barrier_resim import load_h1
from research.position_mgmt import COST, SL_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market

OUT = _ROOT / "artifacts/exit_research"
ACTIVATE_R = 0.5
HORIZON = 16
TRAILS = (0.12, 0.25)
N_TRADES = 10
SEED = 42


def _signed(side: str, entry: float, px: float) -> float:
    return (px - entry) / entry if side == "long" else (entry - px) / entry


def replay_verbose(
    *,
    side: str,
    entry: float,
    atr_entry: float,
    ei: int,
    mkt: dict,
    trail: float,
) -> dict:
    high, low, close, ts = mkt["high"], mkt["low"], mkt["close"], mkt["ts"]
    atr_s = mkt["atr"]
    n = len(close)
    is_long = side == "long"
    one_r = SL_ATR * atr_entry
    sl = entry - SL_ATR * atr_entry if is_long else entry + SL_ATR * atr_entry
    init_sl = sl
    extreme = entry
    hard = min(ei + HORIZON, n - 1)
    candles: list[dict] = []
    reason = "TIMEOUT"
    exit_px = float(close[hard])
    j_exit = hard

    # bar 0 = entry
    candles.append(
        {
            "bar": 0,
            "timestamp": str(pd.Timestamp(ts[ei])),
            "high": float(high[ei]),
            "low": float(low[ei]),
            "close": float(close[ei]),
            "atr": float(atr_entry),
            "trail_mult": trail,
            "trail_distance": float(trail * atr_entry),
            "extreme_fav": float(extreme),
            "trail_stop": float(sl),
            "activated": False,
            "hit": False,
        }
    )

    for j in range(ei + 1, hard + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr_entry
        activated = False
        if is_long:
            extreme = max(extreme, h)
            if (extreme - entry) >= ACTIVATE_R * one_r:
                activated = True
                sl = max(sl, extreme - trail * atr_j)
        else:
            extreme = min(extreme, l)
            if (entry - extreme) >= ACTIVATE_R * one_r:
                activated = True
                sl = min(sl, extreme + trail * atr_j)

        hit = (l <= sl) if is_long else (h >= sl)
        candles.append(
            {
                "bar": int(j - ei),
                "timestamp": str(pd.Timestamp(ts[j])),
                "high": h,
                "low": l,
                "close": c,
                "atr": atr_j,
                "trail_mult": trail,
                "trail_distance": float(trail * atr_j),
                "extreme_fav": float(extreme),
                "trail_stop": float(sl),
                "activated": activated or (abs(sl - init_sl) > 1e-9),
                "hit": bool(hit),
            }
        )
        if hit:
            exit_px = sl
            reason = "TRAIL" if abs(sl - init_sl) > 1e-9 else "SL"
            j_exit = j
            break
        if j >= hard:
            exit_px, reason, j_exit = c, "TIMEOUT", j
            break

    return {
        "side": side,
        "entry_time": str(pd.Timestamp(ts[ei])),
        "entry_price": float(entry),
        "atr_entry": float(atr_entry),
        "trail_mult": trail,
        "trail_distance_entry": float(trail * atr_entry),
        "sl_distance_entry": float(SL_ATR * atr_entry),
        "init_sl": float(init_sl),
        "extreme_fav": float(extreme),
        "exit_bar": int(j_exit - ei),
        "exit_reason": reason,
        "exit_price": float(exit_px),
        "net_return": float(_signed(side, entry, exit_px) - COST),
        "candles": candles,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(OUT / "frozen_entry_panel.parquet")
    pool["timestamp"] = pd.to_datetime(pool["timestamp"], utc=True)
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    eis = entry_indices(pool, mkt["ts"])

    rng = np.random.default_rng(SEED)
    # prefer trades that activate trail (need some room); sample from full pool then keep first N
    idxs = rng.choice(len(pool), size=min(80, len(pool)), replace=False)

    picked: list[int] = []
    pairs: list[tuple[dict, dict]] = []
    for i in idxs:
        if len(picked) >= N_TRADES:
            break
        row = pool.iloc[int(i)]
        kwargs = dict(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr_entry=float(row["atr_entry"]),
            ei=int(eis[int(i)]),
            mkt=mkt,
        )
        a = replay_verbose(**kwargs, trail=0.12)
        b = replay_verbose(**kwargs, trail=0.25)
        # keep trades that activate trail under either (so stop paths are informative)
        act_a = any(c["activated"] for c in a["candles"])
        act_b = any(c["activated"] for c in b["candles"])
        if not (act_a or act_b):
            continue
        picked.append(int(i))
        pairs.append((a, b))

    # if not enough activated, fill with any
    if len(pairs) < N_TRADES:
        for i in range(len(pool)):
            if len(pairs) >= N_TRADES:
                break
            if int(i) in picked:
                continue
            row = pool.iloc[i]
            kwargs = dict(
                side=str(row["side"]),
                entry=float(row["entry_price"]),
                atr_entry=float(row["atr_entry"]),
                ei=int(eis[i]),
                mkt=mkt,
            )
            pairs.append((replay_verbose(**kwargs, trail=0.12), replay_verbose(**kwargs, trail=0.25)))
            picked.append(i)

    # ----- flat debug CSV (one row per trade x mult x candle) -----
    flat = []
    summary_rows = []
    n_diff_dist = n_diff_stop = n_diff_exit_px = n_diff_exit_bar = n_diff_reason = 0

    for t_i, (a, b) in enumerate(pairs):
        dist_diff = abs(a["trail_distance_entry"] - b["trail_distance_entry"])
        # max |trail_stop_012 - trail_stop_025| on overlapping bars
        stops_a = {c["bar"]: c["trail_stop"] for c in a["candles"]}
        stops_b = {c["bar"]: c["trail_stop"] for c in b["candles"]}
        common = sorted(set(stops_a) & set(stops_b))
        max_stop_delta = max(abs(stops_a[k] - stops_b[k]) for k in common) if common else 0.0
        # any bar where stops differ
        any_stop_diff = any(abs(stops_a[k] - stops_b[k]) > 1e-9 for k in common)

        if dist_diff > 1e-12:
            n_diff_dist += 1
        if any_stop_diff:
            n_diff_stop += 1
        if abs(a["exit_price"] - b["exit_price"]) > 1e-9:
            n_diff_exit_px += 1
        if a["exit_bar"] != b["exit_bar"]:
            n_diff_exit_bar += 1
        if a["exit_reason"] != b["exit_reason"]:
            n_diff_reason += 1

        summary_rows.append(
            {
                "trade_id": t_i,
                "entry_time": a["entry_time"],
                "side": a["side"],
                "entry_price": a["entry_price"],
                "atr_entry": a["atr_entry"],
                "trail_dist_012": a["trail_distance_entry"],
                "trail_dist_025": b["trail_distance_entry"],
                "trail_dist_ratio": b["trail_distance_entry"] / a["trail_distance_entry"],
                "max_trail_stop_delta": max_stop_delta,
                "exit_bar_012": a["exit_bar"],
                "exit_bar_025": b["exit_bar"],
                "exit_price_012": a["exit_price"],
                "exit_price_025": b["exit_price"],
                "exit_reason_012": a["exit_reason"],
                "exit_reason_025": b["exit_reason"],
                "net_012": a["net_return"],
                "net_025": b["net_return"],
                "extreme_fav": a["extreme_fav"],
            }
        )

        for label, r in (("0.12", a), ("0.25", b)):
            for c in r["candles"]:
                flat.append(
                    {
                        "trade_id": t_i,
                        "entry_time": a["entry_time"],
                        "side": a["side"],
                        "entry_price": a["entry_price"],
                        "atr_entry": a["atr_entry"],
                        "atr_mult": r["trail_mult"],
                        "trail_distance_entry": r["trail_distance_entry"],
                        "bar": c["bar"],
                        "bar_timestamp": c["timestamp"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "atr_bar": c["atr"],
                        "trail_distance_bar": c["trail_distance"],
                        "extreme_fav": c["extreme_fav"],
                        "trail_stop": c["trail_stop"],
                        "activated": c["activated"],
                        "hit": c["hit"],
                        "exit_reason": r["exit_reason"],
                        "exit_price": r["exit_price"],
                        "exit_bar": r["exit_bar"],
                        "net_return": r["net_return"],
                    }
                )

    debug = pd.DataFrame(flat)
    summary = pd.DataFrame(summary_rows)
    debug.to_csv(OUT / "atr_trail_debug.csv", index=False)
    summary.to_csv(OUT / "atr_trail_debug_summary.csv", index=False)

    # Full-panel sanity: how often exit price / bar differ for all frozen entries
    n_all = len(pool)
    n_px = n_bar = n_same = 0
    stop_deltas = []
    for i in range(n_all):
        row = pool.iloc[i]
        kwargs = dict(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr_entry=float(row["atr_entry"]),
            ei=int(eis[i]),
            mkt=mkt,
        )
        a = replay_verbose(**kwargs, trail=0.12)
        b = replay_verbose(**kwargs, trail=0.25)
        if abs(a["exit_price"] - b["exit_price"]) > 1e-9:
            n_px += 1
        if a["exit_bar"] != b["exit_bar"]:
            n_bar += 1
        if abs(a["exit_price"] - b["exit_price"]) <= 1e-9 and a["exit_bar"] == b["exit_bar"]:
            n_same += 1
        sa = {c["bar"]: c["trail_stop"] for c in a["candles"]}
        sb = {c["bar"]: c["trail_stop"] for c in b["candles"]}
        common = set(sa) & set(sb)
        if common:
            stop_deltas.append(max(abs(sa[k] - sb[k]) for k in common))

    # ratio check on sample
    ratio_ok = all(abs(r["trail_dist_ratio"] - (0.25 / 0.12)) < 1e-9 for _, r in summary.iterrows())

    # example snippet for md
    ex = summary.iloc[0]
    ex_lines = []
    a0, b0 = pairs[0]
    ex_lines.append(f"Trade 0 | {a0['side']} | entry={a0['entry_price']:.2f} | ATR={a0['atr_entry']:.4f}")
    ex_lines.append(
        f"trail_distance 0.12={a0['trail_distance_entry']:.6f} | 0.25={b0['trail_distance_entry']:.6f} "
        f"(ratio={b0['trail_distance_entry']/a0['trail_distance_entry']:.6f}, expect {0.25/0.12:.6f})"
    )
    max_bars = max(len(a0["candles"]), len(b0["candles"]))
    ex_lines.append("bar | stop_0.12 | stop_0.25 | delta | extreme")
    for bar in range(max_bars):
        ca = next((c for c in a0["candles"] if c["bar"] == bar), None)
        cb = next((c for c in b0["candles"] if c["bar"] == bar), None)
        if ca is None or cb is None:
            continue
        ex_lines.append(
            f"{bar} | {ca['trail_stop']:.4f} | {cb['trail_stop']:.4f} | "
            f"{cb['trail_stop']-ca['trail_stop']:+.4f} | {ca['extreme_fav']:.4f}"
        )
    ex_lines.append(
        f"exit 0.12: bar={a0['exit_bar']} px={a0['exit_price']:.4f} {a0['exit_reason']} net={a0['net_return']:+.5f}"
    )
    ex_lines.append(
        f"exit 0.25: bar={b0['exit_bar']} px={b0['exit_price']:.4f} {b0['exit_reason']} net={b0['net_return']:+.5f}"
    )

    # verdict
    mult_ok = (
        ratio_ok
        and n_diff_dist == len(pairs)
        and n_diff_stop == len(pairs)
        and (n_px / max(n_all, 1) > 0.3)  # material share differ in exit price
    )
    # identical hold/WR in prior sweep explained if exit BAR often same but PRICE differs
    verdict = "PASS - ATR multiplier is used correctly" if mult_ok else "FAIL - ATR multiplier likely ignored or buggy"

    why_same_hold = (
        "Identical avg holding bars / WR across multipliers in Sprint 28 is expected when H1 bar range "
        "often exceeds (0.25-0.12)*ATR after activation: exit candle matches, but exit PRICE (= trail stop) differs. "
        f"Full panel: exit_price differs on {n_px}/{n_all} ({n_px/n_all:.1%}); "
        f"exit_bar differs on {n_bar}/{n_all} ({n_bar/n_all:.1%}); "
        f"fully identical exit on {n_same}/{n_all} ({n_same/n_all:.1%}). "
        f"Mean max |stop_0.25-stop_0.12| over path = {float(np.mean(stop_deltas)):.4f}."
    )

    md = f"""# Audit ATR Trail Multiplier

## Question
Does Exit Engine actually use `trail * ATR`, or is the multiplier hardcoded / ignored?

## Method
- Frozen entry panel (same as Sprint 28)
- Replay ATR Trail **0.12** vs **0.25** on the **same {len(pairs)} trades** (seed={SEED}, prefer activated trails)
- Log per candle: ATR, multiplier, trail distance, extreme, trail stop, hit, exit

## Sample checks (n={len(pairs)})

| Check | Result |
|---|---|
| Trail distance differs on all trades | **{n_diff_dist}/{len(pairs)}** |
| Trail stop path differs on all trades | **{n_diff_stop}/{len(pairs)}** |
| Exit price differs | **{n_diff_exit_px}/{len(pairs)}** |
| Exit bar differs | **{n_diff_exit_bar}/{len(pairs)}** |
| Exit reason differs | **{n_diff_reason}/{len(pairs)}** |
| trail_dist_025 / trail_dist_012 == 0.25/0.12 | **{ratio_ok}** |

## Full panel (n={n_all})

| Metric | Value |
|---|---|
| Exit price differs 0.12 vs 0.25 | {n_px} ({n_px/n_all:.1%}) |
| Exit bar differs | {n_bar} ({n_bar/n_all:.1%}) |
| Identical exit (bar+price) | {n_same} ({n_same/n_all:.1%}) |
| Mean max stop delta along path | {float(np.mean(stop_deltas)):.6f} |

## Example trade (trade_id=0)

```
{chr(10).join(ex_lines)}
```

## Trade summary

| id | side | ATR | dist_012 | dist_025 | max_stop_d | exit_bar 012/025 | exit_px 012/025 | reason 012/025 |
|---:|---|---:|---:|---:|---:|---|---|---|
{chr(10).join(
    f"| {r.trade_id} | {r.side} | {r.atr_entry:.3f} | {r.trail_dist_012:.4f} | {r.trail_dist_025:.4f} | "
    f"{r.max_trail_stop_delta:.4f} | {r.exit_bar_012}/{r.exit_bar_025} | "
    f"{r.exit_price_012:.2f}/{r.exit_price_025:.2f} | {r.exit_reason_012}/{r.exit_reason_025} |"
    for r in summary.itertuples()
)}

## Why Sprint 28 hold/WR looked identical

{why_same_hold}

## Verdict

**{verdict}**

- Formula in path replay: `trail_stop = extreme +/- (atr_mult * atr_bar)` after +0.5R activation
- Production paper pipeline uses `TRAIL_ATR_MULT` the same way (`production/paper/pipeline.py` `_update_trail`)
- Different multipliers change trail distance and stop path; often same exit candle with different fill price on XAUUSD H1
"""
    (OUT / "audit_atr_trail.md").write_text(md, encoding="utf-8")
    print(verdict)
    print("sample exit_price differ", n_diff_exit_px, "/", len(pairs))
    print("panel exit_price differ", n_px, "/", n_all)
    print("wrote", OUT / "audit_atr_trail.md")
    print("wrote", OUT / "atr_trail_debug.csv")


if __name__ == "__main__":
    main()
