# Controlled Comparison — v1 vs v2 (same sealed-test window)

## Sealed-test window (apple-to-apple)

Taken from walk-forward v2 LONG test preds (`lightgbm_long_v2_test_preds.parquet`):

| | |
|---|---|
| Start | **2026-01-02 04:00:00+00:00** |
| End | **2026-07-08 11:00:00+00:00** |
| Rows (labeled / preds) | 1923 |
| Span | ~0.51 years |

v1 preds filtered to the **same Date min/max** (also 1923 rows). No retrain. Thresholds frozen as originally selected:

- v1: **0.52** (classic single-split validation)
- v2: **0.54** (walk-forward majority) — only used for the v2 column, not applied to v1

Engine/rules identical: Method A costs, SL=1.5×ATR, TP=2.0×ATR, horizon=8, SL-first, no stacking, 1% risk.

---

## Side-by-side metrics (WITH cost unless noted)

| metric | v1 @ full test (2024-01 → 2026-07) | v1 @ sealed-test-v2 only | v2 @ sealed-test-v2 |
|---|---:|---:|---:|
| Threshold | 0.52 | 0.52 | 0.54 |
| Years | 2.510 | 0.509 | 0.513 |
| Ending equity (start 10k) | 13647.32 | **10492.05** | 9590.79 |
| Total return % | +36.47 | **+4.92** | -4.09 |
| CAGR % (with cost) | +13.19 | **+9.90** | -7.82 |
| CAGR % (no cost) | (see v1 report) | +10.79 | -6.77 |
| Max DD % (with cost) | 17.06 | **6.26** | 7.32 |
| Sharpe ann. approx (with cost) | 1.25 | **0.99** | -0.82 |
| Profit factor (with cost) | 1.31 | **1.20** | 0.84 |
| Win-rate TP/SL | 0.5134 | **0.4815** | 0.4038 |
| n_trades | 247 | **59** | 56 |
| n_tp / n_sl / n_timeout | 115 / 109 / 23 | 26 / 28 / 5 | 21 / 31 / 4 |
| n_signals raw / skipped | 878 / 631 | 235 / 175 | 175 / 114 |
| Model precision @ thr (labels) | 0.5228 (n=878) | 0.4298 (n=235) | 0.3257 (n=175) |

### v1 @ sealed window — with vs without cost (detail)

| | with cost | no cost |
|---|---:|---:|
| Ending equity | 10492.05 | 10535.27 |
| CAGR % | 9.90 | 10.79 |
| Max DD % | 6.26 | 6.47 |
| Sharpe | 0.99 | 1.06 |
| Profit factor | 1.20 | 1.22 |
| Win-rate TP/SL | 0.4815 | 0.4815 |
| n_trades | 59 | 59 |

---

## Bagian 1 interpretation (period isolated)

On the **identical** 2026 sealed window:

- **v1 stays profitable** (CAGR ~+10% with cost, PF>1, Sharpe~1).
- **v2 loses money** (CAGR ~-8%, PF<1, Sharpe negative).
- Trade counts are similar (59 vs 56); the gap is mostly **hit quality** (WR 0.48 vs 0.40; precision 0.43 vs 0.33), not sample-size mismatch.

That pattern supports **Hipotesis A** as the main explanation of the *v1 vs v2 gap*: something about the v2 model / SMC features / walk-forward threshold process underperforms relative to classic v1 on the same dates.

It does **not** support pure Hipotesis B (“2026 is so hard that every model fails”) — because v1 does **not** fail on that window.

Nuance (partial B): v1 on sealed 2026 is **weaker than** v1 on the full 2024–2026 test (precision drops to ~BE 0.43; WR 0.48 vs 0.51; CAGR 9.9% vs 13.2% on a short sample). So 2026 is a tougher slice for the classic model too — but still tradeable for v1, not for v2.

---

## Kesimpulan gabungan (Bagian 1 + Bagian 2)

See also `regime_characterization_2026.md`.

**Verdict: kombinasi, dengan A sebagai penyebab gap performa; B sebagai konteks regime.**

1. **Gap v1 vs v2 di jendela yang sama → Hipotesis A.** Revert / de-prioritize SMC v2 as an “upgrade”; classic v1 @0.52 remains the stronger candidate on this evidence.
2. **2026 sealed window is a volatility outlier (ATR z ≫ 1.5 vs 2020–2025 yearly means) and has a more balanced up/down D1 mix than the 2024–2025 uptrend years → Hipotesis B as regime context.** High-vol gold regime can stress any barrier strategy; it helps explain why v1 looks softer than on the full test, but it does **not** excuse v2’s loss while v1 stays green.
3. **Implikasi langkah lanjut:**
   - Operational baseline: keep **v1 classic LONG @0.52**; do not promote v2 on sealed-test strength alone.
   - SMC: treat as failed first pass — either drop, redesign (fewer/cleaner features, stronger leakage checks), or require walk-forward *and* same-window beat of v1 before another backtest cycle.
   - Any future model should be stress-checked explicitly on **high-ATR regimes** (2025–2026 style), not only average years.
   - Do not retrain or retune thresholds in this task; next work should be diagnostic (why v2 probs degrade) or feature ablation, not another opaque stack.
