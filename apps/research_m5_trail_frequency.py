"""Research: Sprint 41 P0–P3 trails on M5 closes (P0_M5 … P3_M5).

Does not change production.

  python apps/research_m5_trail_frequency.py
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

from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import _mc
from apps.run_exit_engine_grid import CAP, Paths
from simulation.wf.sim import COST, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/trail_m5_frequency"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
HORIZON_H = 48
M5_POLICIES = {
    "P0_M5": dict(act=0.25, dist=0.08),
    "P1_M5": dict(act=0.20, dist=0.06),
    "P2_M5": dict(act=0.20, dist=0.08),
    "P3_M5": dict(act=0.20, dist=0.10),
}
ORDER = ("P0_M5", "P1_M5", "P2_M5", "P3_M5")


def _hour_unit(asi8: np.ndarray, idx: pd.DatetimeIndex) -> int:
    probe = min(2000, len(asi8) - 1)
    scale = float(asi8[probe] / pd.Timestamp(idx[probe]).value)
    return int(round(3_600_000_000_000 * scale))


def _m5_pack(p: Paths, m5: pd.DataFrame) -> dict:
    m5 = m5.copy()
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)
    m5 = m5.sort_values("timestamp").drop_duplicates("timestamp")
    m5_idx = pd.DatetimeIndex(m5["timestamp"])
    ts5 = m5_idx.asi8
    hour = _hour_unit(ts5, m5_idx)
    entry_idx = pd.DatetimeIndex(p.ts)
    if entry_idx.tz is None:
        entry_idx = entry_idx.tz_localize("UTC")
    else:
        entry_idx = entry_idx.tz_convert("UTC")
    return {
        "ts5": ts5, "hour": hour, "n5": len(ts5),
        "high": m5["high"].to_numpy(float), "low": m5["low"].to_numpy(float),
        "close": m5["close"].to_numpy(float), "entry_ns": entry_idx.asi8,
    }


def _replay_m5(p: Paths, pack: dict, *, act: float, dist: float) -> dict:
    ts5, hour, n5 = pack["ts5"], pack["hour"], pack["n5"]
    high, low, close, entry_ns = pack["high"], pack["low"], pack["close"], pack["entry_ns"]
    n = p.n
    r_g = np.zeros(n)
    hold = np.ones(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    horizon_ns = HORIZON_H * hour
    for i in range(n):
        e, atr = float(p.entry[i]), float(p.atr[i])
        one = 1.5 * atr
        if one <= 0 or e <= 0:
            continue
        lng = bool(p.is_long[i])
        start = int(np.searchsorted(ts5, entry_ns[i], side="left"))
        end_ns = entry_ns[i] + horizon_ns
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
            hours = max(1, int((ts5[k] - entry_ns[i] + hour - 1) // hour))
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
    return {
        "net_return": net, "r_multiple": net / ru, "holding_bars": hold,
        "reason": reason, "mfe_r": mfe, "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
    }


def _score(p: Paths, sims: dict) -> tuple[list[dict], dict]:
    rows, ports = [], {}
    for name, sim in sims.items():
        yrows = [_year_row(y, _port(p, sim, y), sim) for y in YEARS]
        po = _pooled(yrows)
        po_oos = _pooled([r for r in yrows if r["year"] in TRUE_OOS])
        mc = _mc(po_oos["pnl"], name)
        ports[name] = {"yrows": yrows, "po": po, "po_oos": po_oos, "mc": mc}
        for yr in yrows:
            rows.append({"book": name, **{k: yr[k] for k in yr if k != "pnl"}})
        rows.append({"book": name, "year": 0, **{k: po[k] for k in po if k != "pnl"},
                     **{k: mc[k] for k in ("prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd")}})
    return rows, ports


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    p = Paths(panel, prepare_market(h1))
    pack = _m5_pack(p, pd.read_parquet(M5_PATH))
    sims = {name: _replay_m5(p, pack, **kw) for name, kw in M5_POLICIES.items()}
    rows, ports = _score(p, sims)
    pd.DataFrame(rows).to_csv(OUT / "yearly_and_mc.csv", index=False)

    oos_cmp = []
    for name in ORDER:
        po, mc = ports[name]["po_oos"], ports[name]["mc"]
        oos_cmp.append({
            "policy": name, "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
            "ret": po["ret"], "prob_ruin": mc["prob_ruin"], "median_dd": mc["median_dd"],
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
    lines = [
        "# Sprint 41 P0–P3 trails on M5",
        "",
        "Research only. Same entries / $280 / lot 0.01 / heat 3R. Trail ratchets on every **M5 close**.",
        "P0_M5 / P1_M5 / P2_M5 / P3_M5 = Sprint 41 distances, trail ratchets on M5 close.",
        "Production **not** shipped.",
        "",
        "| policy | activation | distance | clock |",
        "|---|---:|---:|---|",
        "| P0_M5 | 0.25R | 0.08 ATR | M5 close |",
        "| P1_M5 | 0.20R | 0.06 ATR | M5 close |",
        "| P2_M5 | 0.20R | 0.08 ATR | M5 close |",
        "| P3_M5 | 0.20R | 0.10 ATR | M5 close |",
        "",
        "## OOS 2022–2026 + Monte Carlo (1000 shuffles, $280)",
        "",
        _md(oos_cmp,
            ["policy", "pf", "dd", "avg_r", "wr", "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3}),
        "",
        "## Yearly PF",
        "",
        _md(year_wide, ["year"] + [f"{n}_PF" for n in ORDER],
            {"year": 0, **{f"{n}_PF": 2 for n in ORDER}}),
        "",
        "## Yearly DD",
        "",
        _md(year_wide, ["year"] + [f"{n}_DD" for n in ORDER],
            {"year": 0, **{f"{n}_DD": 3 for n in ORDER}}),
        "",
        "## Yearly AvgR",
        "",
        _md(year_wide, ["year"] + [f"{n}_AvgR" for n in ORDER],
            {"year": 0, **{f"{n}_AvgR": 3 for n in ORDER}}),
        "",
        "## Yearly Return (× $280)",
        "",
        _md(year_wide, ["year"] + [f"{n}_Ret" for n in ORDER],
            {"year": 0, **{f"{n}_Ret": 2 for n in ORDER}}),
        "",
        "2021 is not true OOS. 2026 is a partial year.",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED. P0 H1-close trail: UNCHANGED. P1/P2/P3: NOT USED. M5 trail: NOT SHIPPED.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "production_changed": False,
        "oos": {r["policy"]: {k: r[k] for k in r if k != "policy"} for r in oos_cmp},
    }, indent=2, default=float), encoding="utf-8")
    for name in ORDER:
        po = ports[name]["po_oos"]
        print(name, {k: round(po[k], 3) if isinstance(po[k], float) else po[k] for k in ("pf", "dd", "avg_r", "wr")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
