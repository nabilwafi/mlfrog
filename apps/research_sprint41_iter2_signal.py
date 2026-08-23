"""Sprint 41 Iter 2 — PIT vs differential signal (no ML).

  python apps/research_sprint41_iter2_signal.py
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

from apps.research_sprint41_iter1_diff import OUT as ITER1, STATE_FEATS, YEARS

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter2_signal"
DIFFS = ("diff_p1", "diff_p2", "diff_p3")


def _spearman(a, b) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _q20(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 40:
        return {"n": int(x.size), "sp": 0.0, "top_mean": np.nan, "bot_mean": np.nan, "top_p": np.nan, "bot_p": np.nan}
    lo, hi = np.quantile(x, 0.20), np.quantile(x, 0.80)
    bot, top = y[x <= lo], y[x >= hi]
    return {
        "n": int(x.size),
        "sp": _spearman(x, y),
        "top_mean": float(np.mean(top)),
        "bot_mean": float(np.mean(bot)),
        "top_minus_bot": float(np.mean(top) - np.mean(bot)),
        "top_p": float(np.mean(top > 0)),
        "bot_p": float(np.mean(bot > 0)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    # one row per trade: last alive bar (closest to production exit)
    last = ds.sort_values(["trade_id", "bars_in_trade"]).groupby("trade_id", as_index=False).tail(1)

    rows = []
    yearly = []
    for grain, d in (("bar", ds), ("trade_last", last)):
        for diff in DIFFS:
            y = d[diff].to_numpy(dtype=float)
            for feat in STATE_FEATS:
                x = d[feat].to_numpy(dtype=float)
                rec = {"grain": grain, "label": diff, "feature": feat, **_q20(x, y)}
                rows.append(rec)
                for yr in YEARS:
                    m = d["year"].to_numpy() == yr
                    q = _q20(x[m], y[m])
                    yearly.append({"grain": grain, "year": yr, "label": diff, "feature": feat, **q})

    attr = pd.DataFrame(rows)
    ydf = pd.DataFrame(yearly)
    attr.to_csv(OUT / "feature_spearman.csv", index=False)
    ydf.to_csv(OUT / "yearly_spearman.csv", index=False)

    # stability: trade_last grain, |sp|>=0.10 same sign in >=4 years, min-year |sp|>=0.05
    stab = []
    for diff in DIFFS:
        sub = ydf[(ydf["grain"] == "trade_last") & (ydf["label"] == diff)]
        for feat in STATE_FEATS:
            g = sub[sub["feature"] == feat].set_index("year")
            sps = g.loc[list(YEARS), "sp"].to_numpy(dtype=float)
            signs = np.sign(sps)
            n_ge10 = int(np.sum(np.abs(sps) >= 0.10))
            same = len(set(signs[np.abs(sps) >= 0.05])) <= 1 if np.any(np.abs(sps) >= 0.05) else False
            stab.append({
                "label": diff,
                "feature": feat,
                "mean_sp": float(np.mean(sps)),
                "min_abs_sp": float(np.min(np.abs(sps))),
                "n_years_abs_ge10": n_ge10,
                "same_sign": bool(same and n_ge10 >= 1),
                "stable": bool(n_ge10 >= 4 and same and float(np.min(np.abs(sps))) >= 0.05),
            })
    sdf = pd.DataFrame(stab).sort_values(["stable", "n_years_abs_ge10", "min_abs_sp"], ascending=False)
    sdf.to_csv(OUT / "stability.csv", index=False)

    n_stable = int(sdf["stable"].sum())
    best = sdf.iloc[0].to_dict() if len(sdf) else {}
    # STOP if no stable feature, or only one year, or min-year ~0
    stop = n_stable == 0
    reasons = []
    if n_stable == 0:
        reasons.append("no PIT feature stable vs differential across years")
    if int(sdf["n_years_abs_ge10"].max()) <= 1:
        reasons.append("signal one-year-only")
        stop = True
    if float(sdf["min_abs_sp"].max()) < 0.05:
        reasons.append("min-year effect near 0")
        stop = True

    verdict = "STOP" if stop else "GO_ITER3"
    top = attr[(attr["grain"] == "trade_last")].copy()
    top["abs_sp"] = top["sp"].abs()
    top = top.sort_values("abs_sp", ascending=False).head(12)

    def _md(df, cols):
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

    lines = [
        "# Sprint 41 Iter 2 — Differential signal discovery",
        "",
        "Grain `trade_last` = last bar while production trail still alive (one row per trade).",
        "No ML. Spearman + top/bot 20% vs `diff_p*`.",
        "",
        f"Stable features (|sp|>=0.10 in >=4 years, same sign, min |sp|>=0.05): **{n_stable}**",
        "",
        "### Top |Spearman| (trade_last, pooled)",
        "",
        _md(top, ["label", "feature", "n", "sp", "top_mean", "bot_mean", "top_minus_bot", "top_p", "bot_p"]),
        "",
        "### Stability (best 8)",
        "",
        _md(sdf.head(8), ["label", "feature", "mean_sp", "min_abs_sp", "n_years_abs_ge10", "same_sign", "stable"]),
        "",
        f"STOP reasons: {', '.join(reasons) if reasons else '(none)'}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "If STOP: do not train Iter 3. Production exit `a0.25/d0.08` unchanged.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 2,
                "verdict": verdict,
                "n_stable": n_stable,
                "best": {k: best.get(k) for k in ("label", "feature", "mean_sp", "n_years_abs_ge10", "stable")},
                "reasons": reasons,
                "next_iter": 3 if verdict == "GO_ITER3" else None,
                "production_exit": "a0.25_d0.08 unchanged",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"verdict={verdict} n_stable={n_stable} reasons={reasons}")
    if len(top):
        print(top[["label", "feature", "sp"]].head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
