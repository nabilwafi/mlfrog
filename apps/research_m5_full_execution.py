"""Research: P0 ATR trail with M5 entry fill + M5 exit (full M5 execution).

H1 frozen signals unchanged. Execution only:
  - entry fill = next M5 open after H1 signal
  - trail / hard SL / timeout = M5 closes

Baselines for comparison:
  - P0_H1: H1 fill + H1 trail (legacy)
  - P0_M15: H1 fill + M15 trail (current production clock)
  - P0_M5_exit: H1 fill + M5 trail (exit clock only)
  - P0_M5_full: M5 fill + M5 trail (entry + exit on M5)

Yearly portfolio $280 / lot 0.01 / heat 3R + Monte Carlo.
Research only. Production unchanged.

  python apps/research_m5_full_execution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_m5_trail_frequency import _hour_unit, _replay_m5, _score
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from apps.research_sprint42_attribution import TRUE_OOS, _md
from apps.research_sprint46_loss_reduction import P0
from apps.run_exit_engine_grid import CAP, Paths, simulate_combo
from simulation.wf.sim import COST, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/m5_full_execution"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
M15_PATH = _ROOT / "artifacts/raw/XAUUSD/M15/data.parquet"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
HORIZON_H = 48
ACT, DIST = 0.25, 0.08
ORDER = ("P0_H1", "P0_M15", "P0_M5_exit", "P0_M5_full", "P0_M5_legacy_lookahead")


def _pack(p: Paths, bars: pd.DataFrame) -> dict:
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp")
    idx = pd.DatetimeIndex(bars["timestamp"])
    ts = idx.asi8
    hour = _hour_unit(ts, idx)
    entry_idx = pd.DatetimeIndex(p.ts)
    entry_idx = entry_idx.tz_localize("UTC") if entry_idx.tz is None else entry_idx.tz_convert("UTC")
    return {
        "ts5": ts, "hour": hour, "n5": len(ts),
        "open": bars["open"].to_numpy(float),
        "high": bars["high"].to_numpy(float),
        "low": bars["low"].to_numpy(float),
        "close": bars["close"].to_numpy(float),
        "entry_ns": entry_idx.asi8,
    }


def _replay_m5_from_signal(
    p: Paths,
    pack: dict,
    *,
    act: float,
    dist: float,
    fill: str = "h1_close",
) -> dict:
    """Causal M5/M15 trail: management starts at H1 close (signal time), not H1 open.

    fill='h1_close' — entry price = H1 close (exit-clock research)
    fill='m5_open'  — entry price = M5 open at/after H1 close (full M5 execution)
    """
    ts5, hour, n5 = pack["ts5"], pack["hour"], pack["n5"]
    high, low, close = pack["high"], pack["low"], pack["close"]
    opn = pack.get("open")
    entry_ns = pack["entry_ns"]
    n = p.n
    r_g = np.zeros(n)
    hold = np.ones(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    fill_lag = np.full(n, np.nan)
    fill_slip = np.full(n, np.nan)
    horizon_ns = HORIZON_H * hour
    for i in range(n):
        atr = float(p.atr[i])
        one = 1.5 * atr
        h1_e = float(p.entry[i])
        if one <= 0 or h1_e <= 0:
            continue
        lng = bool(p.is_long[i])
        signal_ns = int(entry_ns[i] + hour)  # H1 close
        start = int(np.searchsorted(ts5, signal_ns, side="left"))
        if start >= n5 or int(ts5[start]) < signal_ns:
            start = int(np.searchsorted(ts5, signal_ns, side="right"))
        if start >= n5:
            continue
        if fill == "m5_open":
            if opn is None:
                raise ValueError("pack needs open for m5_open fill")
            e = float(opn[start])
            fill_lag[i] = float(ts5[start] - signal_ns) / (hour / 60.0)
            fill_slip[i] = ((e - h1_e) if lng else (h1_e - e)) / one
        else:
            e = h1_e
            fill_lag[i] = 0.0
            fill_slip[i] = 0.0
        end_ns = signal_ns + horizon_ns
        sl = -1.0
        extreme = 0.0
        last_r = 0.0
        last_j = 1
        k = start
        while k < n5 and ts5[k] <= end_ns:
            h, l, c = high[k], low[k], close[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            mae[i] = max(mae[i], adv)
            if extreme >= act:
                sl = max(sl, extreme - dist * (atr / one))
            hours = max(1, int((ts5[k] - signal_ns + hour - 1) // hour))
            if adv >= -sl:
                r_g[i] = sl
                hold[i] = hours
                reason[i] = 1 if extreme >= act else 0
                last_r = sl
                break
            last_r, last_j = cr, hours
            k += 1
        else:
            r_g[i] = last_r
            hold[i] = last_j
            reason[i] = 4
        mfe[i] = extreme
        hold[i] = min(max(int(hold[i]), 1), CAP)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    net = r_g * ru - COST
    out = {
        "net_return": net, "r_multiple": net / ru, "holding_bars": hold,
        "reason": reason, "mfe_r": mfe, "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
        "fill_lag_mins": fill_lag, "fill_slip_R": fill_slip,
    }
    return out


def _replay_m5_full(p: Paths, pack: dict, *, act: float, dist: float) -> dict:
    return _replay_m5_from_signal(p, pack, act=act, dist=dist, fill="m5_open")


def _rolling_wf(ports: dict) -> list[dict]:
    """Expanding prior-year mean AvgR as baseline; test year vs that baseline (descriptive)."""
    rows = []
    for name in ORDER:
        yrows = ports[name]["yrows"]
        by = {int(r["year"]): r for r in yrows}
        for y in YEARS:
            if y not in by:
                continue
            prior = [by[yy] for yy in YEARS if yy < y and yy in by]
            test = by[y]
            train_avg = float(np.mean([r["avg_r"] for r in prior])) if prior else float("nan")
            train_pf = float(np.mean([r["pf"] for r in prior])) if prior else float("nan")
            rows.append({
                "policy": name, "test_year": y, "n_train_years": len(prior),
                "train_avg_r": train_avg, "train_pf": train_pf,
                "test_avg_r": test["avg_r"], "test_pf": test["pf"],
                "test_dd": test["dd"], "test_ret": test["ret"], "test_trades": test["trades"],
                "delta_avg_r": test["avg_r"] - train_avg if prior else float("nan"),
                "oos": y in TRUE_OOS,
            })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(H1_PATH)))
    pack5 = _pack(p, pd.read_parquet(M5_PATH))
    pack15 = _pack(p, pd.read_parquet(M15_PATH))

    sim_h1 = simulate_combo(p, **P0)
    # Causal clocks: trail starts at H1 close (no signal-bar look-ahead)
    sim_m15 = _replay_m5_from_signal(p, pack15, act=ACT, dist=DIST, fill="h1_close")
    sim_m5_exit = _replay_m5_from_signal(p, pack5, act=ACT, dist=DIST, fill="h1_close")
    sim_m5_full = _replay_m5_from_signal(p, pack5, act=ACT, dist=DIST, fill="m5_open")
    # Legacy (prior research): walk from H1 open with entry=H1 close — includes signal-bar path
    sim_m5_legacy = _replay_m5(
        p, {k: pack5[k] for k in ("ts5", "hour", "n5", "high", "low", "close", "entry_ns")},
        act=ACT, dist=DIST,
    )

    sims = {
        "P0_H1": sim_h1,
        "P0_M15": sim_m15,
        "P0_M5_exit": sim_m5_exit,
        "P0_M5_full": sim_m5_full,
        "P0_M5_legacy_lookahead": sim_m5_legacy,
    }
    rows, ports = _score(p, sims)
    pd.DataFrame(rows).to_csv(OUT / "yearly_and_mc.csv", index=False)

    fill_diag = {
        "n": int(np.isfinite(sim_m5_full["fill_lag_mins"]).sum()),
        "median_fill_lag_mins": float(np.nanmedian(sim_m5_full["fill_lag_mins"])),
        "p90_fill_lag_mins": float(np.nanpercentile(sim_m5_full["fill_lag_mins"], 90)),
        "mean_fill_slip_R": float(np.nanmean(sim_m5_full["fill_slip_R"])),
        "median_fill_slip_R": float(np.nanmedian(sim_m5_full["fill_slip_R"])),
    }
    pd.DataFrame([fill_diag]).to_csv(OUT / "fill_diagnostics.csv", index=False)

    wf = _rolling_wf(ports)
    pd.DataFrame(wf).to_csv(OUT / "rolling_wf.csv", index=False)

    oos_cmp = []
    for name in ORDER:
        po, mc = ports[name]["po_oos"], ports[name]["mc"]
        oos_cmp.append({
            "policy": name, "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
            "ret": po["ret"], "trades": po["trades"],
            "prob_ruin": mc["prob_ruin"], "median_dd": mc["median_dd"],
            "p95_dd": mc["p95_dd"], "p99_dd": mc["p99_dd"], "worst_dd": mc["worst_dd"],
        })
    year_wide = []
    for i, y in enumerate(YEARS):
        row = {"year": y}
        for name in ORDER:
            yr = ports[name]["yrows"][i]
            row[f"{name}_n"] = yr["trades"]
            row[f"{name}_PF"] = yr["pf"]
            row[f"{name}_DD"] = yr["dd"]
            row[f"{name}_AvgR"] = yr["avg_r"]
            row[f"{name}_Ret"] = yr["ret"]
            row[f"{name}_WR"] = yr["wr"]
        year_wide.append(row)

    def _yt(prefix: str) -> str:
        return _md(year_wide, ["year"] + [f"{n}_{prefix}" for n in ORDER],
                   {"year": 0, **{f"{n}_{prefix}": (3 if prefix in ("DD", "AvgR", "WR") else 2) for n in ORDER}})

    lines = [
        "# M5 full execution — entry fill + exit trail",
        "",
        "Research only. Same **H1 frozen signals**. Production unchanged.",
        "",
        "| policy | entry fill | trail clock | notes |",
        "|---|---|---|---|",
        "| P0_H1 | H1 close | H1 close | legacy H1 trail |",
        "| P0_M15 | H1 close | M15 close **from H1 close** | causal; production clock without signal-bar look-ahead |",
        "| P0_M5_exit | H1 close | M5 close **from H1 close** | causal exit-only |",
        "| **P0_M5_full** | **M5 open at H1 close** | **M5 close** | **entry + exit on M5** |",
        "| P0_M5_legacy_lookahead | H1 close | M5 from H1 **open** | prior research path; includes signal-bar MFE (look-ahead) |",
        "",
        "P0 params: act 0.25R / dist 0.08 ATR / SL 1.5 ATR / timeout 48h / lot 0.01 / heat 3R / $280/year.",
        "H1 frozen signals unchanged. No P1/P2/P3. No HOLD/REDUCE/EXIT model.",
        "",
        "**Important:** earlier M5/M15 trail research walked from H1 **open** with entry=H1 **close**, so trail could book the signal bar’s excursion before entry. Causal books below start at H1 close.",
        "",
        "## Fill diagnostics (P0_M5_full)",
        "",
        f"n={fill_diag['n']} median_lag={fill_diag['median_fill_lag_mins']:.1f}m "
        f"p90_lag={fill_diag['p90_fill_lag_mins']:.1f}m "
        f"mean_slip_R={fill_diag['mean_fill_slip_R']:.4f} "
        f"median_slip_R={fill_diag['median_fill_slip_R']:.4f}",
        "",
        "Slip R = (M5 fill − H1 signal) / 1R, signed with trade direction (positive = worse fill).",
        "",
        "## OOS 2022–2026 + Monte Carlo (1000 shuffles, $280)",
        "",
        _md(oos_cmp,
            ["policy", "pf", "dd", "avg_r", "wr", "ret", "trades",
             "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "ret": 2, "trades": 0,
             "prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3}),
        "",
        "## Yearly PF",
        "",
        _yt("PF"),
        "",
        "## Yearly DD",
        "",
        _yt("DD"),
        "",
        "## Yearly AvgR",
        "",
        _yt("AvgR"),
        "",
        "## Yearly Return (× $280)",
        "",
        _yt("Ret"),
        "",
        "## Rolling walk-forward (expanding prior years → test year)",
        "",
        "Descriptive: mean AvgR/PF of all prior calendar years vs test year. Not a model retrain.",
        "",
        _md([r for r in wf if r["oos"]],
            ["policy", "test_year", "n_train_years", "train_avg_r", "test_avg_r", "delta_avg_r",
             "train_pf", "test_pf", "test_dd", "test_trades"],
            {"test_year": 0, "n_train_years": 0, "train_avg_r": 3, "test_avg_r": 3, "delta_avg_r": 3,
             "train_pf": 2, "test_pf": 2, "test_dd": 3, "test_trades": 0}),
        "",
        "2021 is not true OOS. 2026 is a partial year.",
        "",
        "## Production",
        "",
        "Entry signal: UNCHANGED (H1).",
        "",
        "P0 trail: UNCHANGED (M15 in live).",
        "",
        "M5 entry fill: NOT SHIPPED.",
        "",
        "M5 trail: NOT SHIPPED.",
        "",
        "P1/P2/P3: NOT USED.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "production_changed": False,
        "m5_entry_fill_shipped": False,
        "m5_trail_shipped": False,
        "fill_diagnostics": fill_diag,
        "oos": {r["policy"]: {k: r[k] for k in r if k != "policy"} for r in oos_cmp},
    }, indent=2, default=float), encoding="utf-8")

    print("FILL", {k: round(v, 3) if isinstance(v, float) else v for k, v in fill_diag.items()})
    for name in ORDER:
        po = ports[name]["po_oos"]
        print(name, {k: round(po[k], 3) if isinstance(po[k], float) else po[k]
                     for k in ("pf", "dd", "avg_r", "wr", "ret")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
