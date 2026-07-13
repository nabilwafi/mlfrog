# Walk-Forward Validation Report — LightGBM v2 (classic + SMC)

## Setup
- Features: 28 (= 14 classic + 14 SMC)
- Embargo: ±8h at each train/val/test boundary
- Expanding train; val years 2021–2025; sealed test ≥2026
- Breakeven winrate: 0.4286
- Threshold grid: [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]; min_signals=200
- Final thr = majority-of-folds above BE (not best single fold)

## LONG
| fold | n_train | n_val | auc_roc | auc_pr | fold_selected_thr | fold_val_precision | fold_val_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | 21580 | 3457 | 0.5189 | 0.3870 | 0.5100 | 0.4759 | 290 |
| fold2 | 25046 | 3586 | 0.5579 | 0.4351 | 0.5000 | 0.4519 | 1538 |
| fold3 | 28637 | 3486 | 0.5335 | 0.4112 | 0.5300 | 0.4340 | 318 |
| fold4 | 32123 | 3655 | 0.5522 | 0.4647 | 0.5600 | 0.4930 | 284 |
| fold5 | 35778 | 3799 | 0.5300 | 0.4621 | 0.5400 | 0.5291 | 327 |

- Mean fold AUC-ROC: **0.5385** (std=0.0162)
- v1 single-split val AUC-ROC (LGBM): 0.5390
- Delta mean-fold vs v1: -0.0006
- Majority threshold: selected_thr=0.54 (above BE in 3/5 folds, mean_prec=0.4761)

### Threshold majority summary (LONG)
| threshold | n_folds_above_be | majority | mean_precision_when_ok | mean_n_signals |
| --- | --- | --- | --- | --- |
| 0.5000 | 3 | True | 0.4546 | 1502.2000 |
| 0.5100 | 4 | True | 0.4645 | 807.0000 |
| 0.5200 | 4 | True | 0.4596 | 597.8000 |
| 0.5300 | 4 | True | 0.4610 | 474.8000 |
| 0.5400 | 3 | True | 0.4761 | 376.0000 |
| 0.5500 | 2 | False | 0.4879 | 238.6000 |
| 0.5600 | 2 | False | 0.4657 | 167.6000 |
| 0.5700 | 1 | False | 0.4324 | 120.8000 |
| 0.5800 | 1 | False | 0.4303 | 78.0000 |
| 0.5900 | 0 | False | nan | 50.0000 |
| 0.6000 | 0 | False | nan | 30.8000 |

### Sealed test 2026 (LONG)
- n_test=1923
- test AUC-ROC=0.5678, AUC-PR=0.3868
- @selected_thr=0.54: precision=0.3257, n=175, BELOW breakeven
- val/test consistency (thr exists + test flag): INCONSISTENT

### Feature importance top 15 (LONG final model)
| feature | importance_gain |
| --- | --- |
| ema_slope_d1 | 2887.24 |
| ema_fast_h1 | 1849.30 |
| ema_slope_h4 | 1556.33 |
| adx_h4 | 1517.66 |
| ema_slow_h1 | 1470.69 |
| atr_h4 | 1411.71 |
| adx_h1 | 1218.14 |
| rsi_h1 | 1096.91 |
| atr_h1 | 857.12 |
| dist_to_last_swing_high_h1 | 634.69 |
| dist_to_bearish_ob_h1 | 626.65 |
| dist_to_last_swing_low_h1 | 600.29 |
| dist_to_nearest_fvg_bearish | 289.98 |
| session | 227.84 |
| dist_to_nearest_fvg_bullish | 210.04 |
- SMC features in top 15: ['dist_to_last_swing_high_h1', 'dist_to_bearish_ob_h1', 'dist_to_last_swing_low_h1', 'dist_to_nearest_fvg_bearish', 'dist_to_nearest_fvg_bullish']

## SHORT
| fold | n_train | n_val | auc_roc | auc_pr | fold_selected_thr | fold_val_precision | fold_val_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | 21625 | 3489 | 0.4844 | 0.4075 | nan | nan | nan |
| fold2 | 25122 | 3554 | 0.5492 | 0.4414 | 0.5900 | 0.4817 | 218.0000 |
| fold3 | 28681 | 3544 | 0.5584 | 0.4540 | 0.5200 | 0.4831 | 592.0000 |
| fold4 | 32225 | 3760 | 0.5583 | 0.3977 | nan | nan | nan |
| fold5 | 35985 | 3888 | 0.5468 | 0.3923 | nan | nan | nan |

- Mean fold AUC-ROC: **0.5394** (std=0.0312)
- v1 single-split val AUC-ROC (LGBM): 0.4941
- Delta mean-fold vs v1: +0.0453
- Majority threshold: no threshold above BE with min_signals>=200 in majority of folds (3/5)

### Threshold majority summary (SHORT)
| threshold | n_folds_above_be | majority | mean_precision_when_ok | mean_n_signals |
| --- | --- | --- | --- | --- |
| 0.5000 | 1 | False | 0.4536 | 2178.2000 |
| 0.5100 | 2 | False | 0.4551 | 531.8000 |
| 0.5200 | 2 | False | 0.4680 | 322.0000 |
| 0.5300 | 1 | False | 0.4673 | 167.2000 |
| 0.5400 | 1 | False | 0.4552 | 136.2000 |
| 0.5500 | 1 | False | 0.4498 | 111.6000 |
| 0.5600 | 1 | False | 0.4561 | 91.2000 |
| 0.5700 | 1 | False | 0.4521 | 73.0000 |
| 0.5800 | 1 | False | 0.4607 | 56.0000 |
| 0.5900 | 1 | False | 0.4817 | 43.6000 |
| 0.6000 | 0 | False | nan | 33.0000 |

### SHORT validity (key question vs v1)
- **NO** — still no threshold above BE with min_signals in majority of folds (same failure mode as v1 SHORT).

### Feature importance top 15 (SHORT final model)
| feature | importance_gain |
| --- | --- |
| ema_slope_d1 | 229.28 |
| dist_to_last_swing_low_h1 | 140.82 |
| dist_to_bearish_ob_h1 | 125.45 |
| session | 89.10 |
| atr_h4 | 85.82 |
| adx_h4 | 80.29 |
| ema_slow_h1 | 70.76 |
| dist_to_bullish_ob_h1 | 45.02 |
| ema_fast_h1 | 43.25 |
| dist_to_nearest_fvg_bearish | 28.03 |
| adx_h1 | 24.26 |
| ema_slope_h4 | 21.59 |
| dist_to_last_swing_high_h1 | 17.74 |
| rsi_h1 | 17.54 |
| ema_cross_signal_h1 | 0.00 |
- SMC features in top 15: ['dist_to_last_swing_low_h1', 'dist_to_bearish_ob_h1', 'dist_to_bullish_ob_h1', 'dist_to_nearest_fvg_bearish', 'dist_to_last_swing_high_h1']

## Artifacts
- `data\models_v2\long\lightgbm_long_v2.pkl`
- `data\models_v2\long\lightgbm_long_v2_test_preds.parquet`
- `data\models_v2\short\lightgbm_short_v2.pkl`
- `data\models_v2\short\lightgbm_short_v2_test_preds.parquet`
