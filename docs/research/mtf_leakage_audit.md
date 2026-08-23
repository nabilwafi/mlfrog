# MTF Leakage Audit

| # | Risk | Test | Result | Mitigation |
|---|---|---|---|---|
1 | H4 future into H1 | `assert_h4_causal_boundary()` + attach guard | **PASS** | `available_at` join |
2 | H1 future into M5 | M5 scan starts after H1 close | **PASS** | `searchsorted(side='right')` |
3 | H4 signal availability | H1 ts >= h4 available_at | **PASS** | attach raises on violation |
4 | Vol terciles peek | train-years-only per fold | **PASS** | `vol_terciles_from_train` |
5 | Forward-fill before available | merge_asof backward | **PASS** | no ffill before available |
6 | WF test peek | train/val < test | **PASS** | existing guards |
7 | Threshold on test | fixed top 21% | **PASS** | production gate |
8 | MC on IS trades | OOS PnL only | **PASS** | TRUE_OOS filter |
9 | Execution look-ahead | fill at next M5 open | **PASS** | documented |
10 | Naive H4 asof | prior research bug | **FIXED** | ContextJoinService |
