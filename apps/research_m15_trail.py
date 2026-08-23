"""Research: P0 trail on M15 vs M5 vs production H1.

Same entries / $280 / lot 0.01 / heat 3R. Does not change production.

  python apps/research_m15_trail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_m5_trail_frequency import _m5_pack, _replay_m5, _score
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from apps.research_sprint42_attribution import TRUE_OOS, _md
from apps.research_sprint46_loss_reduction import P0
from apps.run_exit_engine_grid import Paths, simulate_combo
from simulation.wf.sim import load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/trail_m15_vs_m5"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
M15_PATH = _ROOT / "artifacts/raw/XAUUSD/M15/data.parquet"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
POLICIES = {
    "P0": dict(act=0.25, dist=0.08),
    "P1": dict(act=0.20, dist=0.06),
    "P2": dict(act=0.20, dist=0.08),
    "P3": dict(act=0.20, dist=0.10),
}
ORDER = (
    "P0_H1",
    "P0_M5", "P0_M15",
    "P1_M5", "P1_M15",
    "P2_M5", "P2_M15",
    "P3_M5", "P3_M15",
)
HEAD = ("P0_H1", "P0_M5", "P0_M15")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")))
    pack5 = _m5_pack(p, pd.read_parquet(M5_PATH))
    pack15 = _m5_pack(p, pd.read_parquet(M15_PATH))
    sims = {"P0_H1": simulate_combo(p, **P0)}
    for name, kw in POLICIES.items():
        sims[f"{name}_M5"] = _replay_m5(p, pack5, **kw)
        sims[f"{name}_M15"] = _replay_m5(p, pack15, **kw)
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
            row[f"{name}_PF"] = yr["pf"]
            row[f"{name}_DD"] = yr["dd"]
            row[f"{name}_AvgR"] = yr["avg_r"]
            row[f"{name}_Ret"] = yr["ret"]
        year_wide.append(row)

    def _yt(prefix: str, keys: tuple[str, ...]) -> str:
        return _md(year_wide, ["year"] + [f"{n}_{prefix}" for n in keys],
                   {"year": 0, **{f"{n}_{prefix}": (3 if prefix in ("DD", "AvgR") else 2) for n in keys}})

    lines = [
        "# P0 trail: H1 (production) vs M5 vs M15",
        "",
        "Research only. Same H1 entries / $280 / lot 0.01 / heat 3R.",
        "Trail ratchets on each **closed bar** of that clock. SL-first on the bar. Entry ATR for distance.",
        "Production **not** shipped.",
        "",
        "| policy | activation | distance | clock |",
        "|---|---:|---:|---|",
        "| P0_H1 | 0.25R | 0.08 ATR | H1 close (production) |",
        "| P0_M5 | 0.25R | 0.08 ATR | M5 close |",
        "| P0_M15 | 0.25R | 0.08 ATR | M15 close |",
        "| P1_* | 0.20R | 0.06 ATR | M5 / M15 |",
        "| P2_* | 0.20R | 0.08 ATR | M5 / M15 |",
        "| P3_* | 0.20R | 0.10 ATR | M5 / M15 |",
        "",
        "## OOS 2022–2026 + Monte Carlo (1000 shuffles, $280)",
        "",
        _md(oos_cmp,
            ["policy", "pf", "dd", "avg_r", "wr", "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3}),
        "",
        "## Production P0 clock — yearly PF",
        "",
        _yt("PF", HEAD),
        "",
        "## Production P0 clock — yearly DD",
        "",
        _yt("DD", HEAD),
        "",
        "## Production P0 clock — yearly AvgR",
        "",
        _yt("AvgR", HEAD),
        "",
        "## Production P0 clock — yearly Return (× $280)",
        "",
        _yt("Ret", HEAD),
        "",
        "## P1–P3 M5 vs M15 — yearly PF",
        "",
        _yt("PF", ("P1_M5", "P1_M15", "P2_M5", "P2_M15", "P3_M5", "P3_M15")),
        "",
        "## P1–P3 M5 vs M15 — yearly DD",
        "",
        _yt("DD", ("P1_M5", "P1_M15", "P2_M5", "P2_M15", "P3_M5", "P3_M15")),
        "",
        "## P1–P3 M5 vs M15 — yearly AvgR",
        "",
        _yt("AvgR", ("P1_M5", "P1_M15", "P2_M5", "P2_M15", "P3_M5", "P3_M15")),
        "",
        "2021 is not true OOS. 2026 is a partial year.",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED. P0 H1-close trail: UNCHANGED. M5 trail: NOT SHIPPED. M15 trail: NOT SHIPPED.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "production_changed": False,
        "m15_trail_shipped": False,
        "oos": {r["policy"]: {k: r[k] for k in r if k != "policy"} for r in oos_cmp},
    }, indent=2, default=float), encoding="utf-8")
    for name in ORDER:
        po = ports[name]["po_oos"]
        print(name, {k: round(po[k], 3) if isinstance(po[k], float) else po[k] for k in ("pf", "dd", "avg_r", "wr")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
