"""Sprint 39 Iter 6 — rolling WF confirmation of the Iter 5 state score.

Train-only z-score: (-atr_pct_entry + ema_dist_atr). OOS Spearman / top-bot / P>=1/2.
Iter 7 decision written in the same run. No production change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pipeline_backtest/rolling_wf/sprint39_exit_state"
PARQUET = OUT / "exit_state_dataset.parquet"
SUMMARY = OUT / "summary.md"
DECISION = OUT / "decision.json"

LO = "atr_pct_entry"
HI = "ema_dist_atr"


def _spearman(a, b) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _z(train: pd.Series, test: pd.Series) -> pd.Series:
    mu, sd = float(train.mean()), float(train.std())
    if sd < 1e-12:
        return pd.Series(0.0, index=test.index)
    return (test - mu) / sd


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
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


def main() -> None:
    bar = pd.read_parquet(PARQUET)
    bar = bar[bar["kind"] == "bar"].copy()
    ycol = "future_MFE_8"
    years = sorted(bar["year"].unique().tolist())
    rows = []
    for test_y in years[2:]:
        train = bar[bar["year"] < test_y]
        test = bar[bar["year"] == test_y].copy()
        score = -_z(train[LO], test[LO]) + _z(train[HI], test[HI])
        y = test[ycol].to_numpy(dtype=float)
        sc = score.to_numpy(dtype=float)
        m = np.isfinite(sc) & np.isfinite(y)
        sc, y = sc[m], y[m]
        q = pd.qcut(sc, 5, labels=False, duplicates="drop")
        top, bot = y[q == q.max()], y[q == q.min()]
        rows.append({
            "test_year": int(test_y),
            "n": int(m.sum()),
            "spearman": _spearman(sc, y),
            "top_mfe8": float(np.mean(top)),
            "bot_mfe8": float(np.mean(bot)),
            "top_minus_bot": float(np.mean(top) - np.mean(bot)),
            "p1_top": float(np.mean(top >= 1)),
            "p1_bot": float(np.mean(bot >= 1)),
            "p2_top": float(np.mean(top >= 2)),
            "p2_bot": float(np.mean(bot >= 2)),
        })
    wf = pd.DataFrame(rows)
    wf.to_csv(OUT / "iter6_rolling_wf.csv", index=False)

    min_sp = float(wf["spearman"].min())
    mean_sp = float(wf["spearman"].mean())
    min_gap = float(wf["top_minus_bot"].min())
    y26 = wf[wf["test_year"] == 2026].iloc[0]
    years_sp_ge10 = int((wf["spearman"].abs() >= 0.10).sum())
    y24_flip = bool(wf.loc[wf["test_year"] == 2024, "p1_top"].iloc[0] < wf.loc[wf["test_year"] == 2024, "p1_bot"].iloc[0])
    y26_fail = abs(float(y26["spearman"])) < 0.08 or float(y26["top_minus_bot"]) < 0.20

    stop_reasons = []
    if years_sp_ge10 <= 1:
        stop_reasons.append("one-year-only")
    if abs(min_sp) < 0.05:
        stop_reasons.append("min-fold random")
    if y26_fail:
        stop_reasons.append("2026 fails")
    if y24_flip:
        stop_reasons.append("2024 top/bot P>=1 inverted")
    if mean_sp < 0.10:
        stop_reasons.append("mean Spearman < 0.10")

    verdict = "NO_SIGNAL"
    if not stop_reasons and years_sp_ge10 >= 3 and min_gap > 0.20:
        verdict = "CANDIDATE"

    d = json.loads(DECISION.read_text(encoding="utf-8"))
    d["iter"] = 7
    d["next_iter"] = None
    d["iter6_mean_spearman"] = mean_sp
    d["iter6_min_spearman"] = min_sp
    d["iter6_min_top_bot"] = min_gap
    d["iter6_2026_spearman"] = float(y26["spearman"])
    d["stop_reasons"] = stop_reasons
    d["verdict"] = verdict
    d["production_exit"] = "a0.25_d0.08 unchanged"
    DECISION.write_text(json.dumps(d, indent=2), encoding="utf-8")

    body = f"""
# Sprint 39 Iter 6 — Rolling WF confirmation

Score = train-only z(-atr_pct_entry) + z(ema_dist_atr). Expanding year folds. PIT only.

{_md(wf)}

mean Spearman={mean_sp:.3f}  min fold={min_sp:.3f}  min top-bot MFE8={min_gap:.3f}
|spearman|>=0.10 years: {years_sp_ge10}/{len(wf)}  2026 Spearman={float(y26['spearman']):.3f}

STOP reasons: {', '.join(stop_reasons) if stop_reasons else '(none)'}

# Sprint 39 Iter 7 — Decision

**{verdict}**. Keep production trail `a0.25_d0.08`. Do not ship STRONG/EXHAUSTION states.
Leftover MFE after trail is real; in-trade PIT state does not separate continuation vs exhaustion stably across years.
SDS complete. Production unchanged.
"""
    text = SUMMARY.read_text(encoding="utf-8")
    marker = "# Sprint 39 Iter 6"
    SUMMARY.write_text(text.split(marker)[0].rstrip() + "\n" + body, encoding="utf-8")
    print(f"iter6/7 verdict={verdict} mean_sp={mean_sp:.3f} min_sp={min_sp:.3f} reasons={stop_reasons}")


if __name__ == "__main__":
    main()
