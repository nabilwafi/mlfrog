# XAUUSD Baseline Modeling Report

- Features: `data\features\xauusd_h1_h4_d1_features.parquet`
- Labels: `data\labels\xauusd_triple_barrier_labels.parquet`
- Params: sl_mult=1.5, tp_mult=2.0, horizon_bars=8
- Breakeven win-rate: 0.4286
- Feature columns: ['adx_h1', 'atr_h1', 'rsi_h1', 'ema_fast_h1', 'ema_slow_h1', 'ema_cross_signal_h1', 'session', 'return_1h', 'return_4h', 'adx_h4', 'atr_h4', 'ema_slope_h4', 'ema_slope_d1', 'trend_direction_d1']

Models: Logistic Regression, LightGBM, XGBoost — trained separately for LONG and SHORT.
Accuracy is NOT used as primary metric (class imbalance).
Threshold selection: validation-only grid [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200; frozen for test evaluation.

# Data prep — LONG
- timeout dropped: 26758 (39.19%)
- kept win/loss: 41513 (60.81%) | win=16458 loss=25055

## Time-aware split (LONG) after embargo ±8h
Random split is NOT used: overlapping forward windows of length 8 would leak label information across train/test.
- train: rows=28637 | range=2010-02-24 01:00:00+00:00 -> 2022-12-31 00:00:00+00:00
- val: rows=3486 | range=2023-01-03 02:00:00+00:00 -> 2023-12-30 00:00:00+00:00
- test: rows=9390 | range=2024-01-02 02:00:00+00:00 -> 2026-07-08 11:00:00+00:00

## LogisticRegression — LONG

### Validation metrics
- AUC-ROC: 0.5111
- AUC-PR: 0.3927
- @0.5 (reference, no tuning) precision=0.4056 recall=0.4404 f1=0.4223 n_signals=1477

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4056 | 0.4404 | 0.4223 | 1477 | 0.4286 | BELOW breakeven |
| 0.51 | 0.3969 | 0.3338 | 0.3626 | 1144 | 0.4286 | BELOW breakeven |
| 0.52 | 0.3893 | 0.2559 | 0.3088 | 894 | 0.4286 | BELOW breakeven |
| 0.53 | 0.3984 | 0.1831 | 0.2509 | 625 | 0.4286 | BELOW breakeven |
| 0.54 | 0.3568 | 0.1081 | 0.1659 | 412 | 0.4286 | BELOW breakeven |
| 0.55 | 0.3374 | 0.0603 | 0.1023 | 243 | 0.4286 | BELOW breakeven |
| 0.56 | 0.3600 | 0.0397 | 0.0715 | 150 | 0.4286 | BELOW breakeven |
| 0.57 | 0.4023 | 0.0257 | 0.0484 | 87 | 0.4286 | BELOW breakeven |
| 0.58 | 0.2692 | 0.0051 | 0.0101 | 26 | 0.4286 | BELOW breakeven |
| 0.59 | 0.3333 | 0.0022 | 0.0044 | 9 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 1 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 349 | 0.4142 | 0.3954 | 0.0188 |
| 2 | 349 | 0.4449 | 0.3266 | 0.1183 |
| 3 | 348 | 0.4626 | 0.4310 | 0.0316 |
| 4 | 349 | 0.4768 | 0.3381 | 0.1387 |
| 5 | 348 | 0.4873 | 0.4167 | 0.0707 |
| 6 | 349 | 0.4973 | 0.3696 | 0.1277 |
| 7 | 348 | 0.5076 | 0.4425 | 0.0651 |
| 8 | 349 | 0.5205 | 0.3983 | 0.1222 |
| 9 | 348 | 0.5347 | 0.4282 | 0.1066 |
| 10 | 349 | 0.5602 | 0.3553 | 0.2049 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5490
- AUC-PR: 0.4436
- @0.5 (reference, no tuning) precision=0.4618 recall=0.2069 f1=0.2858 n_signals=1754

- No operational selected_thr — test metrics at a tuned threshold are not reported.

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4618 | 0.2069 | 0.2858 | 1754 | 0.4286 | ABOVE breakeven |
| 0.51 | 0.4525 | 0.1522 | 0.2278 | 1317 | 0.4286 | ABOVE breakeven |
| 0.52 | 0.4297 | 0.1006 | 0.1631 | 917 | 0.4286 | ABOVE breakeven |
| 0.53 | 0.4180 | 0.0651 | 0.1127 | 610 | 0.4286 | BELOW breakeven |
| 0.54 | 0.3770 | 0.0368 | 0.0670 | 382 | 0.4286 | BELOW breakeven |
| 0.55 | 0.3604 | 0.0181 | 0.0345 | 197 | 0.4286 | BELOW breakeven |
| 0.56 | 0.4217 | 0.0089 | 0.0175 | 83 | 0.4286 | BELOW breakeven |
| 0.57 | 0.3333 | 0.0028 | 0.0056 | 33 | 0.4286 | BELOW breakeven |
| 0.58 | 0.6250 | 0.0013 | 0.0025 | 8 | 0.4286 | ABOVE breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 939 | 0.3217 | 0.3206 | 0.0011 |
| 2 | 939 | 0.3791 | 0.3791 | -0.0001 |
| 3 | 939 | 0.4048 | 0.3770 | 0.0279 |
| 4 | 939 | 0.4253 | 0.4121 | 0.0131 |
| 5 | 939 | 0.4430 | 0.4228 | 0.0202 |
| 6 | 939 | 0.4591 | 0.4047 | 0.0544 |
| 7 | 939 | 0.4738 | 0.4643 | 0.0094 |
| 8 | 939 | 0.4891 | 0.4665 | 0.0227 |
| 9 | 939 | 0.5081 | 0.4899 | 0.0182 |
| 10 | 939 | 0.5385 | 0.4324 | 0.1061 |

### Feature importance / coefficients (top 15)
| feature                           |   coefficient |   abs_coef |
|:----------------------------------|--------------:|-----------:|
| num__rsi_h1                       |    0.177002   | 0.177002   |
| cat__trend_direction_d1_downtrend |   -0.0717581  | 0.0717581  |
| cat__session_asia                 |   -0.0638097  | 0.0638097  |
| num__ema_slope_d1                 |   -0.0559512  | 0.0559512  |
| cat__trend_direction_d1_uptrend   |    0.0538713  | 0.0538713  |
| num__atr_h4                       |   -0.0506314  | 0.0506314  |
| cat__session_london               |    0.0498032  | 0.0498032  |
| num__return_4h                    |   -0.0414827  | 0.0414827  |
| num__atr_h1                       |    0.0317832  | 0.0317832  |
| cat__session_london_ny_overlap    |    0.0279915  | 0.0279915  |
| cat__trend_direction_d1_sideways  |    0.0218325  | 0.0218325  |
| num__return_1h                    |   -0.0183665  | 0.0183665  |
| num__adx_h1                       |   -0.0165792  | 0.0165792  |
| cat__session_ny                   |   -0.0100392  | 0.0100392  |
| num__ema_fast_h1                  |   -0.00885154 | 0.00885154 |

## LightGBM — LONG

### Validation metrics
- AUC-ROC: 0.5390
- AUC-PR: 0.4163
- @0.5 (reference, no tuning) precision=0.4111 recall=0.5868 f1=0.4835 n_signals=1941

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4111 | 0.5868 | 0.4835 | 1941 | 0.4286 | BELOW breakeven |
| 0.51 | 0.4247 | 0.2301 | 0.2985 | 737 | 0.4286 | BELOW breakeven |
| 0.52 | 0.4686 | 0.1096 | 0.1776 | 318 | 0.4286 | ABOVE breakeven |
| 0.53 | 0.2000 | 0.0029 | 0.0058 | 20 | 0.4286 | BELOW breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** selected_thr=0.52 from validation (precision=0.4686, n=318, min_signals=200)

- selected_thr=0.52 | val precision=0.4686 | val recall=0.1096 | val f1=0.1776 | val n=318 | ABOVE breakeven

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 559 | 0.4785 | 0.3131 | 0.1655 |
| 2 | 184 | 0.4853 | 0.4022 | 0.0831 |
| 3 | 486 | 0.4949 | 0.3663 | 0.1286 |
| 4 | 316 | 0.4988 | 0.4272 | 0.0716 |
| 5 | 557 | 0.5006 | 0.4093 | 0.0912 |
| 6 | 537 | 0.5018 | 0.3650 | 0.1368 |
| 7 | 375 | 0.5099 | 0.4827 | 0.0272 |
| 8 | 131 | 0.5152 | 0.2901 | 0.2251 |
| 9 | 341 | 0.5211 | 0.4545 | 0.0666 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5489
- AUC-PR: 0.4552
- @0.5 (reference, no tuning) precision=0.4506 recall=0.5425 f1=0.4923 n_signals=4714

- **At selected_thr=0.52 (from val):** test precision=0.5228 | test recall=0.1172 | test f1=0.1915 | test n=878 | ABOVE breakeven
- **val_test_consistency_flag:** **CONSISTENT**

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4506 | 0.5425 | 0.4923 | 4714 | 0.4286 | ABOVE breakeven |
| 0.51 | 0.5017 | 0.1936 | 0.2794 | 1511 | 0.4286 | ABOVE breakeven |
| 0.52 | 0.5228 | 0.1172 | 0.1915 | 878 | 0.4286 | ABOVE breakeven |
| 0.53 | 0.4531 | 0.0074 | 0.0146 | 64 | 0.4286 | ABOVE breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 1480 | 0.4720 | 0.3230 | 0.1491 |
| 2 | 616 | 0.4773 | 0.4432 | 0.0341 |
| 3 | 775 | 0.4824 | 0.4297 | 0.0527 |
| 4 | 1198 | 0.4862 | 0.3781 | 0.1081 |
| 5 | 2366 | 0.4994 | 0.4332 | 0.0661 |
| 6 | 1152 | 0.5018 | 0.4184 | 0.0834 |
| 7 | 925 | 0.5112 | 0.4454 | 0.0658 |
| 8 | 878 | 0.5231 | 0.5228 | 0.0003 |

### Feature importance / coefficients (top 15)
| feature             |   importance_gain |
|:--------------------|------------------:|
| rsi_h1              |          568.861  |
| ema_slope_d1        |          461.487  |
| ema_slope_h4        |          396.485  |
| ema_fast_h1         |          294.036  |
| atr_h4              |          196.206  |
| ema_slow_h1         |          195.591  |
| atr_h1              |          182.881  |
| adx_h4              |          163.539  |
| adx_h1              |          147.322  |
| return_4h           |           92.5421 |
| session             |           78.447  |
| ema_cross_signal_h1 |            0      |
| return_1h           |            0      |
| trend_direction_d1  |            0      |

## XGBoost — LONG

### Validation metrics
- AUC-ROC: 0.5329
- AUC-PR: 0.4070
- @0.5 (reference, no tuning) precision=0.4284 recall=0.3522 f1=0.3866 n_signals=1118

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4284 | 0.3522 | 0.3866 | 1118 | 0.4286 | BELOW breakeven |
| 0.51 | 0.3985 | 0.1559 | 0.2241 | 532 | 0.4286 | BELOW breakeven |
| 0.52 | 0.3940 | 0.0971 | 0.1558 | 335 | 0.4286 | BELOW breakeven |
| 0.53 | 0.4044 | 0.0669 | 0.1148 | 225 | 0.4286 | BELOW breakeven |
| 0.54 | 0.4046 | 0.0515 | 0.0913 | 173 | 0.4286 | BELOW breakeven |
| 0.55 | 0.3402 | 0.0243 | 0.0453 | 97 | 0.4286 | BELOW breakeven |
| 0.56 | 0.3137 | 0.0118 | 0.0227 | 51 | 0.4286 | BELOW breakeven |
| 0.57 | 0.2400 | 0.0044 | 0.0087 | 25 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 1 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 374 | 0.4594 | 0.3690 | 0.0904 |
| 2 | 340 | 0.4752 | 0.3706 | 0.1046 |
| 3 | 333 | 0.4815 | 0.2973 | 0.1842 |
| 4 | 357 | 0.4868 | 0.3529 | 0.1339 |
| 5 | 342 | 0.4911 | 0.4211 | 0.0700 |
| 6 | 367 | 0.4946 | 0.3842 | 0.1104 |
| 7 | 344 | 0.4988 | 0.4564 | 0.0424 |
| 8 | 333 | 0.5038 | 0.3784 | 0.1254 |
| 9 | 349 | 0.5117 | 0.4871 | 0.0245 |
| 10 | 347 | 0.5409 | 0.3833 | 0.1577 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5421
- AUC-PR: 0.4576
- @0.5 (reference, no tuning) precision=0.4688 recall=0.2243 f1=0.3034 n_signals=1873

- No operational selected_thr — test metrics at a tuned threshold are not reported.

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4688 | 0.2243 | 0.3034 | 1873 | 0.4286 | ABOVE breakeven |
| 0.51 | 0.5013 | 0.1451 | 0.2250 | 1133 | 0.4286 | ABOVE breakeven |
| 0.52 | 0.5253 | 0.1034 | 0.1729 | 771 | 0.4286 | ABOVE breakeven |
| 0.53 | 0.5495 | 0.0822 | 0.1431 | 586 | 0.4286 | ABOVE breakeven |
| 0.54 | 0.5588 | 0.0582 | 0.1055 | 408 | 0.4286 | ABOVE breakeven |
| 0.55 | 0.5227 | 0.0235 | 0.0450 | 176 | 0.4286 | ABOVE breakeven |
| 0.56 | 0.4727 | 0.0066 | 0.0131 | 55 | 0.4286 | ABOVE breakeven |
| 0.57 | 0.6364 | 0.0036 | 0.0071 | 22 | 0.4286 | ABOVE breakeven |
| 0.58 | 0.4000 | 0.0005 | 0.0010 | 5 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 2 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 971 | 0.4373 | 0.3584 | 0.0789 |
| 2 | 908 | 0.4570 | 0.3700 | 0.0869 |
| 3 | 963 | 0.4672 | 0.4123 | 0.0550 |
| 4 | 930 | 0.4736 | 0.3968 | 0.0769 |
| 5 | 924 | 0.4798 | 0.4275 | 0.0523 |
| 6 | 960 | 0.4857 | 0.4177 | 0.0680 |
| 7 | 917 | 0.4906 | 0.4209 | 0.0696 |
| 8 | 944 | 0.4959 | 0.4290 | 0.0669 |
| 9 | 934 | 0.5059 | 0.4197 | 0.0862 |
| 10 | 939 | 0.5371 | 0.5176 | 0.0195 |

### Feature importance / coefficients (top 15)
| feature             |   importance_gain |
|:--------------------|------------------:|
| ema_slow_h1         |         0.097946  |
| rsi_h1              |         0.0974587 |
| ema_slope_d1        |         0.0948308 |
| ema_fast_h1         |         0.0917321 |
| ema_slope_h4        |         0.0846277 |
| atr_h4              |         0.0804665 |
| trend_direction_d1  |         0.0797604 |
| adx_h4              |         0.0767771 |
| atr_h1              |         0.0723586 |
| adx_h1              |         0.0682331 |
| session             |         0.0565852 |
| return_4h           |         0.0452645 |
| ema_cross_signal_h1 |         0.0336413 |
| return_1h           |         0.0203183 |

# Data prep — SHORT
- timeout dropped: 26479 (38.79%)
- kept win/loss: 41792 (61.21%) | win=16346 loss=25446

## Time-aware split (SHORT) after embargo ±8h
Random split is NOT used: overlapping forward windows of length 8 would leak label information across train/test.
- train: rows=28681 | range=2010-02-12 01:00:00+00:00 -> 2022-12-31 00:00:00+00:00
- val: rows=3544 | range=2023-01-03 02:00:00+00:00 -> 2023-12-30 00:00:00+00:00
- test: rows=9567 | range=2024-01-02 02:00:00+00:00 -> 2026-07-08 11:00:00+00:00

## LogisticRegression — SHORT

### Validation metrics
- AUC-ROC: 0.5202
- AUC-PR: 0.4018
- @0.5 (reference, no tuning) precision=0.3987 recall=0.6216 f1=0.4858 n_signals=2167

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.3987 | 0.6216 | 0.4858 | 2167 | 0.4286 | BELOW breakeven |
| 0.51 | 0.3971 | 0.5345 | 0.4557 | 1871 | 0.4286 | BELOW breakeven |
| 0.52 | 0.4011 | 0.4612 | 0.4290 | 1598 | 0.4286 | BELOW breakeven |
| 0.53 | 0.4039 | 0.3691 | 0.3857 | 1270 | 0.4286 | BELOW breakeven |
| 0.54 | 0.4236 | 0.2971 | 0.3493 | 975 | 0.4286 | BELOW breakeven |
| 0.55 | 0.4180 | 0.2144 | 0.2834 | 713 | 0.4286 | BELOW breakeven |
| 0.56 | 0.4194 | 0.1460 | 0.2166 | 484 | 0.4286 | BELOW breakeven |
| 0.57 | 0.3962 | 0.0892 | 0.1456 | 313 | 0.4286 | BELOW breakeven |
| 0.58 | 0.3876 | 0.0496 | 0.0880 | 178 | 0.4286 | BELOW breakeven |
| 0.59 | 0.3214 | 0.0194 | 0.0366 | 84 | 0.4286 | BELOW breakeven |
| 0.60 | 0.3158 | 0.0086 | 0.0168 | 38 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 355 | 0.4335 | 0.3380 | 0.0954 |
| 2 | 354 | 0.4630 | 0.3418 | 0.1212 |
| 3 | 354 | 0.4792 | 0.4209 | 0.0583 |
| 4 | 355 | 0.4946 | 0.4338 | 0.0608 |
| 5 | 354 | 0.5076 | 0.4011 | 0.1065 |
| 6 | 354 | 0.5196 | 0.3814 | 0.1383 |
| 7 | 355 | 0.5309 | 0.3606 | 0.1703 |
| 8 | 354 | 0.5433 | 0.4096 | 0.1337 |
| 9 | 354 | 0.5582 | 0.4407 | 0.1175 |
| 10 | 355 | 0.5828 | 0.3944 | 0.1884 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5464
- AUC-PR: 0.4192
- @0.5 (reference, no tuning) precision=0.3926 recall=0.7518 f1=0.5158 n_signals=6852

- No operational selected_thr — test metrics at a tuned threshold are not reported.

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.3926 | 0.7518 | 0.5158 | 6852 | 0.4286 | BELOW breakeven |
| 0.51 | 0.3964 | 0.6789 | 0.5005 | 6128 | 0.4286 | BELOW breakeven |
| 0.52 | 0.3993 | 0.6001 | 0.4795 | 5377 | 0.4286 | BELOW breakeven |
| 0.53 | 0.4043 | 0.5235 | 0.4562 | 4633 | 0.4286 | BELOW breakeven |
| 0.54 | 0.4085 | 0.4413 | 0.4243 | 3865 | 0.4286 | BELOW breakeven |
| 0.55 | 0.4175 | 0.3628 | 0.3882 | 3109 | 0.4286 | BELOW breakeven |
| 0.56 | 0.4140 | 0.2859 | 0.3382 | 2471 | 0.4286 | BELOW breakeven |
| 0.57 | 0.4188 | 0.2214 | 0.2896 | 1891 | 0.4286 | BELOW breakeven |
| 0.58 | 0.4224 | 0.1643 | 0.2366 | 1392 | 0.4286 | BELOW breakeven |
| 0.59 | 0.4483 | 0.1272 | 0.1981 | 1015 | 0.4286 | ABOVE breakeven |
| 0.60 | 0.4719 | 0.0869 | 0.1468 | 659 | 0.4286 | ABOVE breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 957 | 0.4422 | 0.3072 | 0.1350 |
| 2 | 957 | 0.4761 | 0.2999 | 0.1762 |
| 3 | 956 | 0.4945 | 0.3964 | 0.0981 |
| 4 | 957 | 0.5088 | 0.3469 | 0.1619 |
| 5 | 957 | 0.5215 | 0.3720 | 0.1495 |
| 6 | 956 | 0.5343 | 0.3860 | 0.1484 |
| 7 | 957 | 0.5467 | 0.3877 | 0.1591 |
| 8 | 956 | 0.5614 | 0.4027 | 0.1586 |
| 9 | 957 | 0.5796 | 0.3887 | 0.1909 |
| 10 | 957 | 0.6154 | 0.4525 | 0.1629 |

### Feature importance / coefficients (top 15)
| feature                           |   coefficient |   abs_coef |
|:----------------------------------|--------------:|-----------:|
| num__rsi_h1                       |    -0.151762  |  0.151762  |
| cat__session_asia                 |    -0.118341  |  0.118341  |
| cat__session_london               |     0.110641  |  0.110641  |
| cat__session_ny                   |    -0.103559  |  0.103559  |
| cat__session_london_ny_overlap    |     0.093102  |  0.093102  |
| cat__trend_direction_d1_downtrend |     0.09159   |  0.09159   |
| cat__trend_direction_d1_sideways  |    -0.0903195 |  0.0903195 |
| num__ema_slope_d1                 |     0.0675297 |  0.0675297 |
| num__atr_h1                       |     0.0599594 |  0.0599594 |
| num__atr_h4                       |    -0.0402218 |  0.0402218 |
| num__return_4h                    |     0.0329827 |  0.0329827 |
| num__adx_h4                       |     0.0200125 |  0.0200125 |
| num__return_1h                    |     0.0196589 |  0.0196589 |
| cat__trend_direction_d1_uptrend   |    -0.019428  |  0.019428  |
| num__ema_fast_h1                  |     0.0135154 |  0.0135154 |

## LightGBM — SHORT

### Validation metrics
- AUC-ROC: 0.4941
- AUC-PR: 0.3857
- @0.5 (reference, no tuning) precision=0.3933 recall=0.5252 f1=0.4498 n_signals=1856

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.3933 | 0.5252 | 0.4498 | 1856 | 0.4286 | BELOW breakeven |
| 0.51 | 0.3625 | 0.0209 | 0.0395 | 80 | 0.4286 | BELOW breakeven |
| 0.52 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.53 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 355 | 0.4946 | 0.3775 | 0.1171 |
| 2 | 511 | 0.4961 | 0.3992 | 0.0969 |
| 3 | 301 | 0.4969 | 0.4618 | 0.0351 |
| 4 | 521 | 0.4989 | 0.3512 | 0.1477 |
| 5 | 266 | 0.5005 | 0.3120 | 0.1885 |
| 6 | 460 | 0.5019 | 0.4587 | 0.0432 |
| 7 | 110 | 0.5025 | 0.5545 | -0.0521 |
| 8 | 444 | 0.5045 | 0.3851 | 0.1193 |
| 9 | 380 | 0.5075 | 0.3421 | 0.1654 |
| 10 | 196 | 0.5097 | 0.3776 | 0.1322 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5448
- AUC-PR: 0.3999
- @0.5 (reference, no tuning) precision=0.4020 recall=0.6685 f1=0.5021 n_signals=5950

- No operational selected_thr — test metrics at a tuned threshold are not reported.

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4020 | 0.6685 | 0.5021 | 5950 | 0.4286 | BELOW breakeven |
| 0.51 | 0.4316 | 0.0458 | 0.0829 | 380 | 0.4286 | ABOVE breakeven |
| 0.52 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.53 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 1321 | 0.4944 | 0.2801 | 0.2143 |
| 2 | 903 | 0.4961 | 0.3511 | 0.1450 |
| 3 | 963 | 0.4969 | 0.3562 | 0.1407 |
| 4 | 1261 | 0.5000 | 0.3632 | 0.1368 |
| 5 | 677 | 0.5024 | 0.4417 | 0.0607 |
| 6 | 1318 | 0.5046 | 0.3998 | 0.1048 |
| 7 | 366 | 0.5066 | 0.4617 | 0.0449 |
| 8 | 1421 | 0.5073 | 0.3807 | 0.1266 |
| 9 | 844 | 0.5075 | 0.4159 | 0.0916 |
| 10 | 493 | 0.5106 | 0.4118 | 0.0989 |

### Feature importance / coefficients (top 15)
| feature             |   importance_gain |
|:--------------------|------------------:|
| ema_slope_d1        |          196.82   |
| ema_slope_h4        |          144.032  |
| rsi_h1              |          130.877  |
| atr_h4              |          123.426  |
| atr_h1              |           89.127  |
| session             |           79.6451 |
| ema_slow_h1         |           57.4456 |
| adx_h4              |           43.5492 |
| adx_h1              |           39.3777 |
| ema_fast_h1         |           25.5194 |
| return_1h           |           22.5043 |
| ema_cross_signal_h1 |            0      |
| return_4h           |            0      |
| trend_direction_d1  |            0      |

## XGBoost — SHORT

### Validation metrics
- AUC-ROC: 0.5320
- AUC-PR: 0.4160
- @0.5 (reference, no tuning) precision=0.4269 recall=0.4583 f1=0.4421 n_signals=1492

#### Threshold grid on VALIDATION only (candidates=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], min_signals=200)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.4269 | 0.4583 | 0.4421 | 1492 | 0.4286 | BELOW breakeven |
| 0.51 | 0.4205 | 0.0856 | 0.1423 | 283 | 0.4286 | BELOW breakeven |
| 0.52 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.53 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

**Threshold selection (validation-only):** no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

#### Calibration (validation, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.4918 | 0.3867 | 0.1051 |
| 2 | 1540 | 0.4988 | 0.3604 | 0.1384 |
| 3 | 133 | 0.5008 | 0.3383 | 0.1624 |
| 4 | 468 | 0.5022 | 0.4338 | 0.0685 |
| 5 | 608 | 0.5095 | 0.4441 | 0.0654 |
| 6 | 283 | 0.5108 | 0.4205 | 0.0903 |

### Test metrics (selected_thr frozen from validation — no re-search)
- AUC-ROC: 0.5231
- AUC-PR: 0.3837
- @0.5 (reference, no tuning) precision=0.3950 recall=0.6635 f1=0.4952 n_signals=6010

- No operational selected_thr — test metrics at a tuned threshold are not reported.

#### Threshold grid on TEST (diagnostic only — NOT used for selection)
| threshold | precision | recall | f1 | n_signals | breakeven | flag |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.3950 | 0.6635 | 0.4952 | 6010 | 0.4286 | BELOW breakeven |
| 0.51 | 0.4069 | 0.1814 | 0.2509 | 1595 | 0.4286 | BELOW breakeven |
| 0.52 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.53 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.54 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.56 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.57 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.58 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.59 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.4286 | BELOW breakeven |

#### Calibration (test, 10 bins)
| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |
|---:|---:|---:|---:|---:|
| 1 | 3557 | 0.4977 | 0.3385 | 0.1592 |
| 2 | 891 | 0.5020 | 0.4400 | 0.0621 |
| 3 | 1040 | 0.5077 | 0.3875 | 0.1202 |
| 4 | 2484 | 0.5095 | 0.3744 | 0.1351 |
| 5 | 1122 | 0.5105 | 0.4216 | 0.0889 |
| 6 | 473 | 0.5145 | 0.3721 | 0.1424 |

### Feature importance / coefficients (top 15)
| feature             |   importance_gain |
|:--------------------|------------------:|
| trend_direction_d1  |        0.106009   |
| ema_slow_h1         |        0.0993418  |
| rsi_h1              |        0.0993239  |
| session             |        0.0939982  |
| ema_slope_h4        |        0.0812612  |
| atr_h4              |        0.0798275  |
| ema_slope_d1        |        0.0787374  |
| ema_fast_h1         |        0.0767533  |
| adx_h4              |        0.0731015  |
| adx_h1              |        0.0633674  |
| return_4h           |        0.0490145  |
| atr_h1              |        0.0488439  |
| return_1h           |        0.0461784  |
| ema_cross_signal_h1 |        0.00424202 |

# Final comparison & recommendation

Breakeven win-rate from SL/TP = 1.5/(1.5+2.0) = **0.4286**.

### Threshold methodology (fixed)
- Select `selected_thr` **only on validation** from grid [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6] with `n_signals >= 200` and precision ABOVE breakeven; pick highest validation precision.
- Apply that **same** threshold to test (no re-search on test).
- If no validation candidate qualifies: do **not** force a near-miss threshold.
- `val_test_consistency_flag`: CONSISTENT if val/test agree on above/below breakeven at selected_thr; else INCONSISTENT.

## Summary table
| direction   | model              |   val_auc_roc |   val_auc_pr |   test_auc_roc |   test_auc_pr |   selected_thr | selection_note                                                                                                                                                  |   val_precision_at_selected_thr |   val_n_at_selected_thr | val_flag_at_selected_thr   |   test_precision_at_selected_thr |   test_n_at_selected_thr | test_flag_at_selected_thr   | val_test_consistency_flag   |   val_precision_0_5 |   val_n_0_5 |   test_precision_0_5 |   test_n_0_5 |
|:------------|:-------------------|--------------:|-------------:|---------------:|--------------:|---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------:|------------------------:|:---------------------------|---------------------------------:|-------------------------:|:----------------------------|:----------------------------|--------------------:|------------:|---------------------:|-------------:|
| long        | LogisticRegression |      0.511097 |     0.39273  |       0.549044 |      0.443598 |         nan    | no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]) |                      nan        |                     nan | nan                        |                       nan        |                      nan | nan                         | N/A                         |            0.405552 |        1477 |             0.461802 |         1754 |
| long        | LightGBM           |      0.539043 |     0.416287 |       0.548857 |      0.455156 |           0.52 | selected_thr=0.52 from validation (precision=0.4686, n=318, min_signals=200)                                                                                    |                        0.468553 |                     318 | ABOVE breakeven            |                         0.522779 |                      878 | ABOVE breakeven             | CONSISTENT                  |            0.411128 |        1941 |             0.450573 |         4714 |
| long        | XGBoost            |      0.532868 |     0.406955 |       0.542084 |      0.457557 |         nan    | no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]) |                      nan        |                     nan | nan                        |                       nan        |                      nan | nan                         | N/A                         |            0.428444 |        1118 |             0.468767 |         1873 |
| short       | LogisticRegression |      0.520215 |     0.401814 |       0.546443 |      0.419239 |         nan    | no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]) |                      nan        |                     nan | nan                        |                       nan        |                      nan | nan                         | N/A                         |            0.398708 |        2167 |             0.392586 |         6852 |
| short       | LightGBM           |      0.494149 |     0.385749 |       0.544811 |      0.399939 |         nan    | no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]) |                      nan        |                     nan | nan                        |                       nan        |                      nan | nan                         | N/A                         |            0.393319 |        1856 |             0.402017 |         5950 |
| short       | XGBoost            |      0.531977 |     0.416047 |       0.523088 |      0.38367  |         nan    | no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]) |                      nan        |                     nan | nan                        |                       nan        |                      nan | nan                         | N/A                         |            0.426944 |        1492 |             0.395008 |         6010 |

## No valid selected_thr on validation
- **LONG / LogisticRegression**: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
- **LONG / XGBoost**: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
- **SHORT / LogisticRegression**: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
- **SHORT / LightGBM**: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
- **SHORT / XGBoost**: no threshold beats breakeven with sufficient sample size on validation (min_signals=200, grid=[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])

## Recommendation
- **LONG**: **LightGBM** (selected_thr=0.52, val_prec=0.4686, test_prec=0.5228, test_n=878, test AUC-PR=0.4552, CONSISTENT).
- **SHORT**: no model has a validation-selected threshold that stays ABOVE breakeven with CONSISTENT val/test behavior. **Do not force a recommendation.**

Honesty check: if all directions lack a valid CONSISTENT above-breakeven selected_thr, the baseline does not yet justify advancing to backtest on thresholded signals alone. Ranking metrics (AUC) may still be informative, but operational edge is not established.