# Standalone H4 Experiment

## Data
- Raw: `data\raw\XAUUSD_H4.csv`
- Native feature rows (post warm-up): 18817
- Close-time shift: 0 days 04:00:00
- Features: ['adx_h4', 'atr_h4', 'rsi_h4', 'ema_fast_h4', 'ema_slow_h4', 'ema_cross_signal_h4', 'return_1h4', 'return_6h4']

## Label grid (SL=1.5 ATR, TP=2.0 ATR)
- Chosen horizon: **12** bars
- Breakeven WR: 0.4286

| horizon | direction | win% | loss% | timeout% | wr_ex_to | n_kept |
|---:|---|---:|---:|---:|---:|---:|
| 6 | long | 21.61 | 33.06 | 45.33 | 0.3954 | 10287 |
| 6 | short | 20.41 | 34.65 | 44.94 | 0.3706 | 10361 |
| 8 | long | 27.48 | 38.89 | 33.63 | 0.4141 | 12488 |
| 8 | short | 25.52 | 41.27 | 33.20 | 0.3821 | 12569 |
| 12 | long | 35.51 | 46.54 | 17.95 | 0.4328 | 15440 |
| 12 | short | 32.48 | 49.71 | 17.81 | 0.3952 | 15466 |

## Models (time split train<2023 / val 2023 / test≥2024, embargo = horizon)

| direction | n_train | n_val | n_test | LGBM val AUC | LGBM test AUC | LGBM thr | LR val AUC | LR test AUC |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| long | 10914 | 1311 | 3202 | 0.5448 | 0.5008 | None | 0.5470 | 0.5556 |
- long LGBM selection: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
| short | 10920 | 1291 | 3242 | 0.5254 | 0.5000 | None | 0.5714 | 0.5625 |
- short LGBM selection: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

## Note vs H1 baseline
- H1 broadcast LGBM val AUC LONG=0.539 SHORT=0.494
- Standalone H4 LGBM val AUC LONG=0.545 SHORT=0.525

## Conclusion
Lihat juga `standalone_d1_experiment.md` untuk keputusan stacking utama; H4 adalah cek tambahan di skala intermediate.

## Artifacts
- `data\features\h4_native_features.parquet`
- `data\labels\h4_labeled.parquet`
