# Final LONG Model Comparison (v1 / v2 / v3)

Sealed window (confirmed): **2026-01-02 04:00:00+00:00 -> 2026-07-08 11:00:00+00:00**

| metric | v1 @ full test | v1 @ sealed-2026 | v2 @ sealed-2026 | v3 @ full test | v3 @ sealed-2026 |
| --- | --- | --- | --- | --- | --- |
| Threshold | 0.52 | 0.52 | 0.54 | 0.51 | 0.51 |
| CAGR % (with cost) | 13.19 | 9.90 | -7.82 | 18.52 | 11.18 |
| Win-rate realized (TP/SL) | 0.5134 | 0.4815 | 0.4038 | 0.4943 | 0.4700 |
| n_trades | 247 | 59 | 56 | 480 | 107 |
| Sharpe (with cost) | 1.25 | 0.99 | -0.82 | 1.24 | 0.84 |
| Profit Factor | 1.31 | 1.20 | 0.84 | 1.19 | 1.12 |
| Max DD % | 17.06 | 6.26 | 7.32 | 12.66 | 6.49 |
| Ending equity | 13647 | 10492 | 9591 | 15327 | 10554 |

Sources: v1/v2 from `backtest_report.md`, `controlled_comparison_v1_vs_v2.md`, `backtest_report_v2.md`. v3 from this run (`backtest_report_v3.md`).

Notes:
- v3 **full** preds: one LGBM fit on v3 features, train<2023 / val 2023, thr frozen at WF majority 0.51 (same calendar philosophy as v1).
- v3 **sealed** preds: walk-forward final model test preds (same dates as v2 sealed).

## Required conclusions

### 1) Does v3 beat v1 AND v2 on sealed-2026?

**Beats v2 clearly; slightly beats v1 sealed on CAGR/equity.**

- v3 sealed: CAGR **+11.18%**, end equity **10554**, PF 1.12, WR 0.47 (profitable).
- v2 sealed: CAGR **-7.82%**, equity 9591 (loss).
- v1 sealed: CAGR **+9.90%**, equity 10492.

So reversal-augmented LONG **survives the 2026 stress window** that broke v2, and edges v1 on that short window (more trades: 107 vs 59).

### 2) Is v3 full-period comparable to v1 (CAGR 13.19%, PF 1.31)?

**Better on CAGR/equity; slightly weaker PF/WR.**

- v3 full: CAGR **18.52%**, equity **15327**, PF **1.19**, WR 0.494, MaxDD 12.7%, n_trades 480.
- v1 full: CAGR 13.19%, equity 13647, PF 1.31, WR 0.513, MaxDD 17.1%, n_trades 247.

v3 trades more often at thr 0.51 (lower bar than 0.52) - higher activity, lower PF, but higher CAGR and lower MaxDD. Not a collapse; headline return improves.

### 3) Operational LONG candidate

**Recommendation: v3 as primary LONG candidate; v1 as fallback benchmark.**

- **v2**: discarded (sealed-2026 failure).
- **v3**: wins sealed stress vs v2, slightly ahead of v1 sealed, and stronger full-window CAGR with acceptable PF>1.
- Keep v1 report as control; any live/paper promotion should still apply a future regime-guard pass on v3.

Next: regime guard on LONG v3; SHORT continues on stacking v4 track separately.
