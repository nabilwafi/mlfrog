# Walk-Forward Validation Report — SHORT v4 (v2 features + D1/H4 short proba)

## Leakage / Option A

D1 and H4 SHORT LGBMs were trained on native labels with train < 2023-01-01 (val=2023). Scoring those models on their own train window produces **in-sample** probabilities. If H1-SHORT trains on that window, it can overfit to memorized D1 scores rather than transferable HTF signal.

**Mitigation (Option A):** keep full-history `d1_short_proba` / `h4_short_proba` for feature engineering, but treat folds with val year < 2024 as contaminated for stacking claims. Prefer metrics on folds with `val_oos_vs_d1_model=True` (2024, 2025) and sealed test 2026 (always OOS vs D1/H4 train). Option B (out-of-fold HTF proba) is the rigorous follow-up if results look promising.

- Features: 30 (= 28 v2 + ['d1_short_proba', 'h4_short_proba'])
- Breakeven: 0.4286; min_signals=200

## Walk-forward folds
| fold | n_train | n_val | auc_roc | auc_pr | fold_selected_thr | fold_val_precision | fold_val_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | 21625 | 3489 | 0.5661 | 0.4917 | 0.5500 | 0.5794 | 252.0000 |
| fold2 | 25122 | 3554 | 0.6180 | 0.4952 | 0.5900 | 0.5494 | 739.0000 |
| fold3 | 28681 | 3544 | 0.5204 | 0.4054 | nan | nan | nan |
| fold4 | 32225 | 3760 | 0.5234 | 0.3680 | nan | nan | nan |
| fold5 | 35985 | 3888 | 0.5350 | 0.3847 | nan | nan | nan |

- Mean fold AUC (all folds): **0.5526** (std=0.0408)
- Mean fold AUC (OOS vs D1 only, 2024–2025): **0.5292**
- v2 mean fold AUC: 0.5394 | delta all-fold: +0.0132
- v3 mean fold AUC: 0.5177 | delta all-fold: +0.0349
- Majority threshold: no threshold above BE with min_signals>=200 in majority of folds (3/5)

### Threshold majority summary
| threshold | n_folds_above_be | majority | mean_precision_when_ok | mean_n_signals |
| --- | --- | --- | --- | --- |
| 0.5000 | 2 | False | 0.4859 | 2425.0000 |
| 0.5100 | 2 | False | 0.5022 | 1160.8000 |
| 0.5200 | 2 | False | 0.5192 | 431.0000 |
| 0.5300 | 2 | False | 0.5336 | 362.6000 |
| 0.5400 | 2 | False | 0.5360 | 306.6000 |
| 0.5500 | 2 | False | 0.5541 | 262.6000 |
| 0.5600 | 1 | False | 0.5346 | 233.0000 |
| 0.5700 | 1 | False | 0.5416 | 208.4000 |
| 0.5800 | 1 | False | 0.5427 | 181.8000 |
| 0.5900 | 1 | False | 0.5494 | 149.4000 |
| 0.6000 | 1 | False | 0.5466 | 128.0000 |

### Sealed test 2026
- n_test=1906; AUC=0.5274; AUPR=0.4473
- No majority-valid threshold — sealed not scored at operational thr.

### Feature importance top 15
| feature | importance_gain |
| --- | --- |
| h4_short_proba | 327.85 |
| d1_short_proba | 296.23 |
| ema_slope_h4 | 130.79 |
| ema_slope_d1 | 118.95 |
| dist_to_last_swing_low_h1 | 114.00 |
| ema_slow_h1 | 93.89 |
| session | 79.27 |
| atr_h4 | 78.35 |
| dist_to_bearish_ob_h1 | 73.66 |
| ema_fast_h1 | 62.95 |
| dist_to_last_swing_high_h1 | 38.76 |
| bos_bearish_h4 | 35.67 |
| dist_to_bullish_ob_h1 | 29.83 |
| atr_h1 | 26.82 |
| dist_to_nearest_fvg_bearish | 26.58 |
- Meta ranks: {'d1_short_proba': 2, 'h4_short_proba': 1}
- Meta in top 5: ['d1_short_proba', 'h4_short_proba']
- Meta in top 15: ['d1_short_proba', 'h4_short_proba']

## Required conclusions

1. **Is d1_short_proba important?** rank=2; in top15=True. h4_short_proba rank=1.

2. **Mean fold AUC vs v2/v3?** all-fold 0.5526 vs v2 0.5394 / v3 0.5177; OOS-vs-D1 folds 0.5292.

3. **Majority threshold for SHORT?** **NO** — still fails majority-of-folds rule.

4. **Recommendation:** Park SHORT for now; focus on LONG (v1/v3). Stacking did not unlock a valid SHORT threshold under Option A.

### Critical diagnostic (Option A side-effect)

HTF SHORT models produce **near-constant OOS probabilities**:

| window | d1_short_proba std | nunique | h4_short_proba |
|---|---:|---:|---|
| pre-2023 (in-sample vs D1 train) | 0.0113 | 31 | (varies in-sample) |
| 2023 (D1 val) | 0.0049 | 5 | - |
| post-2024 (OOS) | 0.0043 | **4** | **std=0 / constant** |

So fold1–2 AUC inflation is largely **in-sample stacking leakage**, not transferable HTF edge. OOS folds (2024–2025) and sealed 2026 see almost flat meta-features - Option B (OOF) is unlikely to rescue SHORT until the native D1/H4 SHORT models themselves produce dispersed, calibrated OOS scores.

**Bottom line:** do not invest further in SHORT stacking until HTF SHORT standalone models are fixed; prioritize LONG v3.

## Artifacts
- `data\models_v4\short\lightgbm_short_v4.pkl`
- `data\models_v4\short\lightgbm_short_v4_test_preds.parquet`
- `data/features/xauusd_h1_short_features_v4.parquet`
- `data/reports/d1_stacking_lookahead_check.md`
