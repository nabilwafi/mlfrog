"""Sprint 39 iter 3 — state attribution (no ML).

  python apps/research_sprint39_iter3_attr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.research_sprint39_exit_state import HORIZONS, OUT, STATE_FEATS

CATS = ("session", "trend", "vol_regime")
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _attr_numeric(d: pd.DataFrame, feat: str, ycol: str) -> dict:
    x = d[feat].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 50:
        return {"feature": feat, "label": ycol, "n": int(x.size), "spearman": 0.0}
    q = pd.qcut(x, 5, labels=False, duplicates="drop")
    top = y[q == q.max()]
    bot = y[q == q.min()]
    return {
        "feature": feat,
        "label": ycol,
        "n": int(x.size),
        "spearman": _spearman(x, y),
        "q1_mean": float(np.mean(y[q == q.min()])),
        "q5_mean": float(np.mean(y[q == q.max()])),
        "q5_minus_q1": float(np.mean(top) - np.mean(bot)),
        "p_mfe1_q1": float(np.mean(bot >= 1)) if ycol.startswith("future_MFE") else np.nan,
        "p_mfe1_q5": float(np.mean(top >= 1)) if ycol.startswith("future_MFE") else np.nan,
        "p_mfe2_q1": float(np.mean(bot >= 2)) if ycol.startswith("future_MFE") else np.nan,
        "p_mfe2_q5": float(np.mean(top >= 2)) if ycol.startswith("future_MFE") else np.nan,
    }


def main() -> int:
    ds = OUT / "exit_state_dataset.parquet"
    d = pd.read_parquet(ds)
    bar = d[d["kind"] == "bar"].copy()
    rows = []
    for feat in STATE_FEATS:
        for H in HORIZONS:
            rows.append(_attr_numeric(bar, feat, f"future_MFE_{H}"))
        rows.append(_attr_numeric(bar, feat, "future_MAE_8"))
    attr = pd.DataFrame(rows)
    attr.to_csv(OUT / "feature_attribution.csv", index=False)

    sp8 = attr[attr["label"] == "future_MFE_8"].sort_values("spearman", key=np.abs, ascending=False)
    best = str(sp8.iloc[0]["feature"])
    yearly = []
    y = bar["future_MFE_8"].to_numpy(dtype=float)
    for yr in YEARS:
        m = bar["year"].to_numpy() == yr
        yearly.append(
            {
                "year": yr,
                "n": int(m.sum()),
                "spearman_best": _spearman(bar.loc[m, best].to_numpy(dtype=float), y[m]),
                "spearman_dd": _spearman(bar.loc[m, "drawdown_from_MFE_R"].to_numpy(dtype=float), y[m]),
                "spearman_mfe": _spearman(bar.loc[m, "mfe_so_far_R"].to_numpy(dtype=float), y[m]),
                "p_mfe1": float(np.nanmean(y[m] >= 1)),
            }
        )
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "yearly_spearman.csv", index=False)

    cat_rows = []
    for c in CATS:
        g = bar.groupby(c, dropna=False)["future_MFE_8"].agg(["count", "mean"])
        g["p_ge1"] = bar.groupby(c)["future_MFE_8"].apply(lambda s: float(np.mean(s >= 1)))
        g["p_ge2"] = bar.groupby(c)["future_MFE_8"].apply(lambda s: float(np.mean(s >= 2)))
        g["feature"] = c
        cat_rows.append(g.reset_index().rename(columns={c: "level", "count": "n", "mean": "mean_mfe8"}))
    cats = pd.concat(cat_rows, ignore_index=True)
    cats.to_csv(OUT / "categorical_attribution.csv", index=False)

    hi = bar[bar["mfe_so_far_R"] >= 0.50].copy()
    soon = hi["trail_exit"].eq(1) & ((hi["hold"] - hi["j"]) <= 2)
    cont = hi["future_MFE_8"] >= 1.0
    a, b = hi[soon & ~cont], hi[cont & ~soon]
    cmp_rows = []
    for feat in STATE_FEATS + list(CATS):
        if feat in CATS:
            continue
        xa, xb = a[feat].to_numpy(dtype=float), b[feat].to_numpy(dtype=float)
        cmp_rows.append(
            {
                "feature": feat,
                "n_trail_soon": int(np.isfinite(xa).sum()),
                "n_continue": int(np.isfinite(xb).sum()),
                "mean_trail_soon": float(np.nanmean(xa)),
                "mean_continue": float(np.nanmean(xb)),
                "delta": float(np.nanmean(xb) - np.nanmean(xa)),
            }
        )
    cmp = pd.DataFrame(cmp_rows).sort_values("delta", key=np.abs, ascending=False)
    cmp.to_csv(OUT / "high_mfe_trail_vs_continue.csv", index=False)

    min_sp = float(ydf["spearman_best"].abs().min())
    top_sp = float(sp8.iloc[0]["spearman"])
    n_years_ok = int((ydf["spearman_best"].abs() >= 0.10).sum())
    stop = abs(top_sp) < 0.12 or min_sp < 0.05 or n_years_ok < 4
    verdict = "WEAK / likely STOP after iter 6" if stop else "SEPARATION worth rule search"
    top5 = sp8.head(8)[["feature", "spearman", "q5_minus_q1", "p_mfe1_q1", "p_mfe1_q5"]]

    prev = (OUT / "summary.md").read_text(encoding="utf-8") if (OUT / "summary.md").is_file() else ""
    extra = [
        "",
        "# Sprint 39 Iter 3 — Attribution",
        "",
        f"Bar observations only (n={len(bar)}). Label `future_MFE_8` unless noted.",
        f"Best |Spearman|: **{best}** = {top_sp:.3f}. Min-year |Spearman| on that feat: **{min_sp:.3f}** ({n_years_ok}/6 years |sp|>=0.10).",
        "",
        "### Top numeric features vs future_MFE_8",
        "",
        "| Feature | Spearman | Q5-Q1 MFE | P(>=1R) Q1 | P(>=1R) Q5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in top5.itertuples(index=False):
        extra.append(f"| {r.feature} | {r.spearman:.3f} | {r.q5_minus_q1:.2f} | {r.p_mfe1_q1:.3f} | {r.p_mfe1_q5:.3f} |")
    extra += [
        "",
        "### Yearly Spearman (best feat + drawdown_from_MFE + mfe_so_far)",
        "",
        "| Year | n | best | drawdown | mfe_so_far | P(MFE8>=1) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ydf.itertuples(index=False):
        extra.append(
            f"| {int(r.year)} | {int(r.n)} | {r.spearman_best:.3f} | {r.spearman_dd:.3f} | {r.spearman_mfe:.3f} | {r.p_mfe1:.3f} |"
        )
    extra += [
        "",
        f"HIGH MFE (>=0.50R) trail-soon n={len(a)} vs continue n={len(b)}. Biggest mean gaps: "
        + ", ".join(f"{r.feature} {r.delta:+.3f}" for r in cmp.head(5).itertuples(index=False)),
        "",
        f"Iter 3 gate: **{verdict}**",
        "Iter 4: trail-exit vs same-MFE continuation. Production exit unchanged.",
        "",
    ]
    marker = "# Sprint 39 Iter 3"
    if marker in prev:
        prev = prev.split(marker)[0].rstrip()
    (OUT / "summary.md").write_text(prev + "\n" + "\n".join(extra), encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"iter": 3, "best": best, "spearman": top_sp, "min_year_sp": min_sp, "verdict": verdict, "stop_hint": stop}, indent=2),
        encoding="utf-8",
    )
    print(f"best={best} sp={top_sp:.3f} min_year={min_sp:.3f} years_ok={n_years_ok} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
