"""Sprint 39 iter 4 — trail-exit point vs same-MFE continuation.

  python apps/research_sprint39_iter4_exitpoint.py
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

from apps.research_sprint39_exit_state import OUT, STATE_FEATS
from apps.research_sprint39_iter3_attr import CATS, YEARS, _spearman

FEATS = list(STATE_FEATS)


def _means(g: pd.DataFrame) -> dict[str, float]:
    out = {"n": float(len(g))}
    for f in FEATS:
        out[f] = float(np.nanmean(g[f].to_numpy(dtype=float)))
    for H in (8, 16, 24, 48):
        y = g[f"future_MFE_{H}"].to_numpy(dtype=float)
        out[f"future_MFE_{H}"] = float(np.nanmean(y))
        out[f"p_mfe{H}_ge1"] = float(np.nanmean(y >= 1))
        out[f"p_mfe{H}_ge2"] = float(np.nanmean(y >= 2))
    return out


def main() -> int:
    d = pd.read_parquet(OUT / "exit_state_dataset.parquet")
    bar = d[d["kind"] == "bar"]
    trail = bar[bar["trail_exit"].eq(1)].copy()
    exit_bar = trail[trail["j"] == trail["hold"]]
    pre1 = trail[trail["j"] == (trail["hold"] - 1)]
    pre1 = pre1[pre1["j"] >= 1]

    # same-MFE continuation: first hit 0.50/1.00 where path still adds >=1R
    mfe05 = d[d["kind"] == "mfe_0.50"]
    mfe10 = d[d["kind"] == "mfe_1.00"]
    cont05 = mfe05[mfe05["future_MFE_8"] >= 1.0]
    cont10 = mfe10[mfe10["future_MFE_8"] >= 1.0]
    # trail-too-early: hit 1.0 only after baseline already dead
    late10 = mfe10[mfe10["baseline_alive"].eq(0)]

    groups = {
        "trail_exit_bar": exit_bar,
        "trail_1bar_before": pre1,
        "mfe0.50_then_plus1R": cont05,
        "mfe1.00_then_plus1R": cont10,
        "mfe1.00_after_trail_dead": late10,
    }
    rows = []
    for name, g in groups.items():
        rec = {"group": name, **_means(g)}
        rows.append(rec)
    gdf = pd.DataFrame(rows)
    gdf.to_csv(OUT / "exit_point_groups.csv", index=False)

    # feature delta: exit bar vs mfe0.50 continuation
    deltas = []
    a, b = exit_bar, cont05
    for f in FEATS:
        xa, xb = a[f].to_numpy(dtype=float), b[f].to_numpy(dtype=float)
        deltas.append(
            {
                "feature": f,
                "mean_trail_exit": float(np.nanmean(xa)),
                "mean_mfe05_continue": float(np.nanmean(xb)),
                "delta": float(np.nanmean(xb) - np.nanmean(xa)),
            }
        )
    dlt = pd.DataFrame(deltas).sort_values("delta", key=np.abs, ascending=False)
    dlt.to_csv(OUT / "trail_vs_continue_delta.csv", index=False)

    yrows = []
    for yr in YEARS:
        e = exit_bar[exit_bar["year"] == yr]
        c = cont05[cont05["year"] == yr]
        ye = e["future_MFE_8"].to_numpy(dtype=float)
        yc = c["future_MFE_8"].to_numpy(dtype=float)
        yrows.append(
            {
                "year": yr,
                "n_trail_exit": len(e),
                "n_continue": len(c),
                "trail_mean_mfe8": float(np.nanmean(ye)) if e.size else np.nan,
                "cont_mean_mfe8": float(np.nanmean(yc)) if c.size else np.nan,
                "trail_p_ge1": float(np.nanmean(ye >= 1)) if e.size else np.nan,
                "cont_p_ge1": float(np.nanmean(yc >= 1)) if c.size else np.nan,
                "sp_atr_exit": _spearman(e["atr_pct_entry"].to_numpy(dtype=float), ye) if len(e) > 30 else 0.0,
            }
        )
    ydf = pd.DataFrame(yrows)
    ydf.to_csv(OUT / "exit_point_yearly.csv", index=False)

    top = dlt.head(6)
    # leftover after trail is large by construction; question is state separation
    sep = float(np.nanmean(np.abs(dlt["delta"].to_numpy())))
    weak = sep < 0.15  # mean |delta| across features small in mixed units — use ranked note
    lines = [
        "",
        "# Sprint 39 Iter 4 — Trail exit vs same-MFE continuation",
        "",
        f"TRAIL exit bars n={len(exit_bar)}. 1-bar-before n={len(pre1)}.",
        f"MFE 0.50 then +1R path n={len(cont05)}. MFE 1.00 then +1R n={len(cont10)}.",
        f"MFE 1.00 first-hit after trail already dead n={len(late10)} / {len(mfe10)}.",
        "",
        "After a production TRAIL exit bar, path leftover is still large "
        f"(mean future_MFE_8={gdf.loc[gdf['group']=='trail_exit_bar','future_MFE_8'].iloc[0]:.2f}, "
        f"P>=1R={gdf.loc[gdf['group']=='trail_exit_bar','p_mfe8_ge1'].iloc[0]:.3f}).",
        "",
        "### Group means",
        "",
        "| Group | n | MFE so far | current_R | DD from MFE | ATR pct entry | future_MFE_8 | P(>=1R) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = [
        ("trail_exit_bar", "trail exit bar"),
        ("trail_1bar_before", "1 bar before trail"),
        ("mfe0.50_then_plus1R", "MFE 0.50 then +1R"),
        ("mfe1.00_then_plus1R", "MFE 1.00 then +1R"),
        ("mfe1.00_after_trail_dead", "MFE 1.00 after trail dead"),
    ]
    for key, lab in show:
        r = gdf[gdf["group"] == key].iloc[0]
        lines.append(
            f"| {lab} | {int(r['n'])} | {r['mfe_so_far_R']:.2f} | {r['current_R']:.2f} | "
            f"{r['drawdown_from_MFE_R']:.2f} | {r['atr_pct_entry']:.2f} | {r['future_MFE_8']:.2f} | {r['p_mfe8_ge1']:.3f} |"
        )
    lines += [
        "",
        "### Biggest feature gaps (MFE 0.50-continue minus trail-exit bar)",
        "",
        "| Feature | trail exit | continue | delta |",
        "|---|---:|---:|---:|",
    ]
    for r in top.itertuples(index=False):
        lines.append(f"| {r.feature} | {r.mean_trail_exit:.3f} | {r.mean_mfe05_continue:.3f} | {r.delta:+.3f} |")
    lines += [
        "",
        "### Yearly leftover after trail exit vs MFE0.50-continue",
        "",
        "| Year | n trail | trail MFE8 | trail P>=1 | n cont | cont MFE8 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ydf.itertuples(index=False):
        lines.append(
            f"| {int(r.year)} | {int(r.n_trail_exit)} | {r.trail_mean_mfe8:.2f} | {r.trail_p_ge1:.3f} | "
            f"{int(r.n_continue)} | {r.cont_mean_mfe8:.2f} |"
        )
    q = (
        "Leftover after trail is consistent every year, but pre-exit state vs "
        "same-MFE continuation does not show a strong PIT separator "
        "(iter 3 Spearman already weak). Iter 5 will still try a simple OOS rule; "
        "default expectation is NO SIGNAL."
    )
    lines += ["", q, "", "Production exit unchanged.", ""]
    prev = (OUT / "summary.md").read_text(encoding="utf-8")
    marker = "# Sprint 39 Iter 4"
    if marker in prev:
        prev = prev.split(marker)[0].rstrip()
    (OUT / "summary.md").write_text(prev + "\n" + "\n".join(lines), encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"iter": 4, "next": 5, "late_mfe1_after_trail": int(len(late10)), "n_mfe1": int(len(mfe10))}, indent=2),
        encoding="utf-8",
    )
    print(f"exit_n={len(exit_bar)} cont05={len(cont05)} late1R={len(late10)}/{len(mfe10)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
