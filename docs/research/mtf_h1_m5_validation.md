# M5 Timing on H1 — Validation Summary

Full artifacts: `results/h4_h1_m5/validation/`

Runner: `python apps/research_mtf_h1_m5_validation.py`

## Verdict: **ITERATE**

M5 `pullback_recovery` on H1 baseline passes research gates vs production FEAT7 immediate, but is **not PROMOTE** until paper/live execution test. Production stack unchanged.

## OOS (2022–26, top 21%, P0 M15 trail, $280 start)

| Track | PF | Max DD | Avg R | Trades |
|---|---:|---:|---:|---:|
| Production FEAT7 | 1.33 | 53.8% | 0.059 | 5335 |
| H1 baseline (immediate) | 1.40 | 30.2% | 0.061 | 5358 |
| **H1 → M5 pullback** | **1.56** | **30.1%** | **0.108** | 5103 |

## Robustness

- **Cost stress (H1+M5):** PF 1.51 @ +25%, 1.47 @ +50%, 1.38 @ +100%
- **Yearly vs prod:** 4/5 OOS years better PF; 2022 slightly worse (−0.06 PF)
- **Direction:** LONG PF 1.54, SHORT PF 1.53 (both beat prod)
- **MC 10k:** prob ruin 0.1%, median DD 21.3%
- **M5 delay:** median 5 bars, p90 11 bars, invalidate 2.5%
- **Param grid:** stable around default (0.15 ATR / 0.50 recovery); 0.25/0.50 PF 1.57

## Interpretation

1. Most of the DD improvement vs production comes from **H1 baseline (6 features, no H4 ctx)** — not M5 alone.
2. M5 adds **timing edge**: +0.16 PF vs H1 immediate, avg R nearly doubles (0.061 → 0.108).
3. H4 layer still **REJECT** (separate track); validate H1→M5 only.

## Next steps (if continuing)

- Paper test M5 fill logic vs H1 close assumption
- Rolling WF re-train with frozen M5 rule
- Compare `pullback_recovery` vs production on identical FEAT7 panel (apples-to-apples entry set)
