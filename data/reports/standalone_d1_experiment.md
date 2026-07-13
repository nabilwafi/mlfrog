# Standalone D1 Experiment

## Data
- Raw: `data\raw\XAUUSD_D1.csv`
- Native feature rows (post warm-up): 4237
- Close-time shift: 1 days 00:00:00
- Features: ['adx_d1', 'atr_d1', 'rsi_d1', 'ema_fast_d1', 'ema_slow_d1', 'ema_cross_signal_d1', 'return_1d', 'return_5d']

## Label grid (SL=1.5 ATR, TP=2.0 ATR)
- Chosen horizon: **10** bars
- Breakeven WR: 0.4286

| horizon | direction | win% | loss% | timeout% | wr_ex_to | n_kept |
|---:|---|---:|---:|---:|---:|---:|
| 5 | long | 18.81 | 25.32 | 55.86 | 0.4262 | 1870 |
| 5 | short | 14.30 | 31.74 | 53.95 | 0.3106 | 1951 |
| 8 | long | 29.53 | 34.72 | 35.76 | 0.4596 | 2722 |
| 8 | short | 22.11 | 42.98 | 34.91 | 0.3397 | 2758 |
| 10 | long | 34.79 | 38.33 | 26.88 | 0.4758 | 3098 |
| 10 | short | 26.03 | 47.96 | 26.01 | 0.3518 | 3135 |

## Models (time split train<2023 / val 2023 / test≥2024, embargo = horizon)

| direction | n_train | n_val | n_test | LGBM val AUC | LGBM test AUC | LGBM thr | LR val AUC | LR test AUC |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| long | 2417 | 187 | 474 | 0.5277 | 0.5340 | None | 0.3704 | 0.5967 |
- long LGBM selection: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
| short | 2450 | 176 | 494 | 0.5653 | 0.5608 | None | 0.3842 | 0.6278 |
- short LGBM selection: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

## Comparison vs H1 broadcast baseline (LGBM val AUC)

| | AUC (broadcast di H1, existing) | AUC (standalone D1, native LGBM val) | Delta |
|---|---:|---:|---:|
| LONG | 0.539 | 0.528 | -0.011 |
| SHORT | 0.494 | 0.565 | +0.071 |

## Conclusion

**Hipotesis 'fitur encer waktu broadcast' TERKONFIRMASI sebagian** (avg delta val AUC = +0.030; LONG -0.011, SHORT +0.071).

Caveats: D1 val sets are small (n≈176–187), so AUC has high variance; raw D1 SHORT win-rate ex-timeout is only 0.35 (strong bullish drift at D1 barriers) while LONG is 0.48 — not an apple-to-apple label base-rate vs H1.

Rekomendasi: lanjutkan hierarchical/stacking **dengan fokus SHORT meta-feature dari D1** (dan H4 sebagai pelengkap) — latih model D1/H4 standalone, ambil `predict_proba` pada close native-TF, lalu `merge_asof(direction='backward')` ke row H1 sebagai meta-feature (ganti/lengkapi slope & trend_direction broadcast mentah). H1 model tetap pakai label/horizon H1.

Jangan harapkan stacking saja menyelesaikan SHORT jika base-rate D1 SHORT tetap jelek; stacking = kompresi sinyal HTF, bukan pengganti fitur reversal.

## Artifacts
- `data\features\d1_native_features.parquet`
- `data\labels\d1_labeled.parquet`
