"""Sprint 39 Iter 5 — simple STRONG / EXHAUSTION / NORMAL rules.

Thresholds from train years only. OOS test. PIT features only.
No PF-chasing. No production change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
from research_sprint39_exit_state import STATE_FEATS  # noqa: E402

OUT = ROOT / "artifacts/pipeline_backtest/rolling_wf/sprint39_exit_state"
PARQUET = OUT / "exit_state_dataset.parquet"
SUMMARY = OUT / "summary.md"
DECISION = OUT / "decision.json"

PIT = [c for c in STATE_FEATS if not str(c).startswith("future_")]
# two-feature rules from iter 3/4: leftover anti-correlates with entry ATR;
# continuation paths sit further from EMA / higher mom
STRONG_FEAT_LO = "atr_pct_entry"
STRONG_FEAT_HI = "ema_dist_atr"
EXH_FEAT_HI = "atr_pct_entry"
EXH_FEAT_DD = "drawdown_from_MFE_R"


def _stats(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"n": 0, "mfe8": np.nan, "p1": np.nan, "p2": np.nan, "mfe48": np.nan}
    return {
        "n": n,
        "mfe8": float(g["future_MFE_8"].mean()),
        "p1": float((g["future_MFE_8"] >= 1).mean()),
        "p2": float((g["future_MFE_8"] >= 2).mean()),
        "mfe48": float(g["future_MFE_48"].mean()),
    }


def _assign(df: pd.DataFrame, atr_lo: float, ema_hi: float, atr_hi: float, dd_hi: float) -> pd.Series:
    strong = (df[STRONG_FEAT_LO] <= atr_lo) & (df[STRONG_FEAT_HI] >= ema_hi)
    exh = (df[EXH_FEAT_HI] >= atr_hi) & (df[EXH_FEAT_DD] >= dd_hi)
    # ponytail: if both fire, EXHAUSTION wins (conservative)
    out = pd.Series("NORMAL", index=df.index)
    out.loc[strong] = "STRONG"
    out.loc[exh] = "EXHAUSTION"
    return out


def main() -> None:
    ds = pd.read_parquet(PARQUET)
    bar = ds[ds["kind"] == "bar"].copy()
    years = sorted(bar["year"].unique().tolist())

    rows = []
    yearly = []
    # expanding: train all years < test_year
    for test_y in years[2:]:  # need at least 2 train years
        train = bar[bar["year"] < test_y]
        test = bar[bar["year"] == test_y]
        atr_lo = float(train[STRONG_FEAT_LO].quantile(0.30))
        ema_hi = float(train[STRONG_FEAT_HI].quantile(0.70))
        atr_hi = float(train[EXH_FEAT_HI].quantile(0.70))
        dd_hi = float(train[EXH_FEAT_DD].quantile(0.70))
        test = test.copy()
        test["state"] = _assign(test, atr_lo, ema_hi, atr_hi, dd_hi)
        for st, g in test.groupby("state"):
            s = _stats(g)
            s.update({"fold": f"test_{test_y}", "state": st, "atr_lo": atr_lo, "ema_hi": ema_hi, "atr_hi": atr_hi, "dd_hi": dd_hi})
            rows.append(s)
            yearly.append({"year": int(test_y), "state": st, **_stats(g)})

    oos = pd.DataFrame(rows)
    oos.to_csv(OUT / "iter5_oos_states.csv", index=False)
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "iter5_yearly_states.csv", index=False)

    # pooled OOS (2023+)
    oos_years = years[2:]
    pooled = bar[bar["year"].isin(oos_years)].copy()
    # thresholds from pre-first-OOS only (no peek)
    train0 = bar[bar["year"] < oos_years[0]]
    atr_lo = float(train0[STRONG_FEAT_LO].quantile(0.30))
    ema_hi = float(train0[STRONG_FEAT_HI].quantile(0.70))
    atr_hi = float(train0[EXH_FEAT_HI].quantile(0.70))
    dd_hi = float(train0[EXH_FEAT_DD].quantile(0.70))
    pooled["state"] = _assign(pooled, atr_lo, ema_hi, atr_hi, dd_hi)
    pooled_rows = []
    for st, g in pooled.groupby("state"):
        pooled_rows.append({"scope": "pooled_oos", "state": st, **_stats(g)})
    pooled_rows.append({"scope": "pooled_oos", "state": "ALL", **_stats(pooled)})
    pdf = pd.DataFrame(pooled_rows)
    pdf.to_csv(OUT / "iter5_pooled.csv", index=False)

    # separation: STRONG leftover minus EXHAUSTION leftover, by year
    sep = []
    for y, gy in ydf.groupby("year"):
        sm = gy.set_index("state")
        if "STRONG" in sm.index and "EXHAUSTION" in sm.index:
            sep.append({
                "year": int(y),
                "delta_mfe8": float(sm.loc["STRONG", "mfe8"] - sm.loc["EXHAUSTION", "mfe8"]),
                "delta_p1": float(sm.loc["STRONG", "p1"] - sm.loc["EXHAUSTION", "p1"]),
                "n_strong": int(sm.loc["STRONG", "n"]),
                "n_exh": int(sm.loc["EXHAUSTION", "n"]),
            })
    sdf = pd.DataFrame(sep)
    sdf.to_csv(OUT / "iter5_separation.csv", index=False)

    min_delta = float(sdf["delta_mfe8"].min()) if len(sdf) else np.nan
    mean_delta = float(sdf["delta_mfe8"].mean()) if len(sdf) else np.nan
    years_pos = int((sdf["delta_mfe8"] > 0.3).sum()) if len(sdf) else 0
    gate = "WEAK"
    if years_pos >= 4 and min_delta > 0.15:
        gate = "CANDIDATE"
    if years_pos <= 1 or (not np.isnan(mean_delta) and mean_delta < 0.15):
        gate = "NO_SIGNAL"

    def _md(df: pd.DataFrame) -> str:
        cols = [c for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" for _ in cols) + "|"]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    body = f"""
# Sprint 39 Iter 5 — Simple state rules (OOS)

Train-only quantiles: STRONG if atr_pct_entry <= p30 AND ema_dist_atr >= p70.
EXHAUSTION if atr_pct_entry >= p70 AND drawdown_from_MFE >= p70. Else NORMAL.
If both fire, EXHAUSTION wins.

### Pooled OOS

{_md(pdf)}

### Yearly STRONG minus EXHAUSTION leftover (MFE 8h)

{_md(sdf)}

mean delta MFE8={mean_delta:.3f}  min year={min_delta:.3f}  years delta>0.3R: {years_pos}/{len(sdf)}

Iter 5 gate: **{gate}**
Iter 6: rolling WF confirmation. Production exit unchanged.
"""
    text = SUMMARY.read_text(encoding="utf-8")
    marker = "# Sprint 39 Iter 5"
    SUMMARY.write_text(text.split(marker)[0].rstrip() + "\n" + body, encoding="utf-8")

    d = json.loads(DECISION.read_text(encoding="utf-8"))
    d["iter"] = 5
    d["next_iter"] = 6
    d["iter5_gate"] = gate
    d["iter5_mean_delta_mfe8"] = mean_delta
    d["iter5_min_delta_mfe8"] = min_delta
    DECISION.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"iter5 gate={gate} mean_delta={mean_delta:.3f} min={min_delta:.3f} years_pos={years_pos}")


if __name__ == "__main__":
    main()
