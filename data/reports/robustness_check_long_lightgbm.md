# Robustness Check — LONG LightGBM @ threshold 0.52

## Setup
- Model artifact: `data\models\lightgbm_long.pkl`
- Test predictions: `data\models\lightgbm_long_test_preds.parquet`
- Frozen threshold: **0.52** (validation-selected; NOT re-tuned per sub-period)
- Breakeven win-rate: **0.4286**
- Test range target: 2024-01-02 -> 2026-07-08 (LONG kept rows before thr filter)
- Sanity target from prior report: precision~=0.5228, n~=878 (re-fit may differ slightly)

## Bagian 1 — Sub-period breakdown

| period               |   n_rows |   n_signals |   precision |   recall | breakeven_flag   |
|:---------------------|---------:|------------:|------------:|---------:|:-----------------|
| 2024 H1              |     1816 |         180 |      0.4222 |   0.0978 | BELOW breakeven  |
| 2024 H2              |     1845 |         176 |      0.5284 |   0.1197 | ABOVE breakeven  |
| 2025 H1              |     1917 |         134 |      0.6119 |   0.0979 | ABOVE breakeven  |
| 2025 H2              |     1888 |         153 |      0.6993 |   0.1269 | ABOVE breakeven  |
| 2026 H1              |     1837 |         215 |      0.4140 |   0.1388 | BELOW breakeven  |
| 2026 remainder       |       87 |          20 |      0.6000 |   0.3077 | ABOVE breakeven  |
| AGGREGATE (all test) |     9390 |         878 |      0.5228 |   0.1172 | ABOVE breakeven  |

### Robustness verdict
- **Label:** **ROBUST**
- Detail: Precision ABOVE breakeven in 4/6 sub-periods (std precision=0.1130).
- n_above=4, n_below=2, precision_std=0.1130

## Bagian 2 — Bootstrap 95% CI on thresholded test precision

- n_signals (p>= 0.52): **878**
- Point precision: **0.5228**
- Bootstrap iterations: 1000 (with replacement, seed=42)
- 95% CI: **[0.4874, 0.5547]**
- CI lower bound vs breakeven 0.4286: **STILL ABOVE**

### Keterbatasan metode
Bootstrap di atas mengasumsikan sample IID. Label triple-barrier punya overlapping forward windows (horizon=8), jadi antar baris saling berkorelasi. CI ini kemungkinan **under-estimate** ketidakpastian sebenarnya (terlalu sempit). Interpretasikan sebagai batas bawah optimis untuk ketidakpastian, bukan CI yang sepenuhnya time-series-aware.

## Rekomendasi akhir

**Layak dianggap kandidat kuat untuk lanjut ke backtest (Layer 5)** untuk LONG-LightGBM @ thr=0.52: sub-period=ROBUST, bootstrap 95% CI lower=0.4874 masih >= breakeven 0.4286, agregat precision=0.5228 (n=878).
