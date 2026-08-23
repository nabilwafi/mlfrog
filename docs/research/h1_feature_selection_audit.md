# H1 Feature Selection — Repository Audit (Phase 1)

**Date:** 2026-08-23  
**Scope:** H1 native features only (38). Exclude `ctx_h4_*`, `d1_*`, `m5_*`, session flags.  
**Stack under test (frozen protocol):** WF top 21% → M15 pullback entry → P0 M15 trail (a0.25/d0.08).  
**Baseline:** `H1_NATIVE` / `PRIMARY_FEATURES` = 6 features.

---

## 1. Current feature pipeline

| Layer | Path | Role |
|-------|------|------|
| FE service | `feature_engineering/services/feature_engineering_service.py` | OHLCV → matrix via registered builders |
| Builders | `feature_engineering/builders/{trend,volatility,momentum,candle,session,statistical}_builder.py` | Produce the 38 H1 features |
| Transforms | `feature_engineering/builders/_transforms.py` | Causal rolling / EMA / ATR / RSI / MACD |
| Config | `configs/config.example.yaml` → `feature_engineering.builders` | Default 6 builders (no return/volume/liquidity/MR) |
| Live panel | `production/live/features.py` | Runs FE + joins H4/D1/M5 (context **not** in this experiment) |
| Artifacts | `artifacts/features/XAUUSD/H1/feature_matrix.parquet` | On disk: 38 H1 + 10 research extensions (48) |

**Intermediates not exported:** `ema_20`, `ema_50`, `atr_14`, `rsi_14`, MACD line/signal/hist.

---

## 2. Current 38 H1 features

### Baseline CORE — 6 (`research/mtf_h4_h1_m5/h1_features.py` → `H1_NATIVE`)

| Feature | Group |
|---------|-------|
| `hour_sin`, `hour_cos` | session |
| `atr_percent`, `atr_percentile_252` | volatility |
| `ema_trend_duration` | trend |
| `rolling_quantile` | statistical |

### Research candidates — 32

| Group | Count | Features |
|-------|------:|----------|
| Trend | 8 | `ema20_distance_atr`, `ema50_distance_atr`, `ema20_distance_percent`, `ema50_distance_percent`, `ema_cross_distance`, `ema20_slope`, `ema50_slope`, `ema_alignment_score` |
| Volatility | 3 | `rolling_volatility`, `volatility_rank`, `volatility_regime_score` |
| Momentum | 8 | `macd_normalized`, `macd_histogram_zscore`, `macd_signal_distance`, `momentum_rank`, `rsi_percentile`, `roc_3`, `roc_12`, `momentum_acceleration` |
| Candle | 6 | `body_percent`, `upper_wick_percent`, `lower_wick_percent`, `close_position`, `range_percent`, `body_rank` |
| Session | 2 | `day_sin`, `day_cos` |
| Statistical | 5 | `rolling_zscore`, `rolling_percentile`, `rolling_rank`, `rolling_std`, `rolling_mean_distance` |

**Note:** `research/category_features/H1_ENGINE_V1` lists **35** (stale — missing `roc_3`, `roc_12`, `momentum_acceleration`). Canonical universe for this study = **38**.

---

## 3. Current model input

| Knob | Location | Value |
|------|----------|-------|
| Primary features | `production/__init__.py` → `PRIMARY_FEATURES` | 6 (H1 native) |
| Gate | `PRIMARY_TOP_PCT` | **0.21** |
| Frozen export | `production/live/primary_export.py` | LGBM on v2 train/val |
| Inference | `production/live/inference.py` | Uses booster `feature_name()` |

Meta / confidence gates are **OFF**. H4/M5/D1 joins remain for logging only — **excluded from feature selection**.

---

## 4. Dataset / labels / split

| Piece | Path | Notes |
|-------|------|-------|
| Dataset v2 | `artifacts/datasets/XAUUSD/H1/{long,short}/v2/*.parquet` | Present |
| Labels | Triple barrier (`labels/strategies/triple_barrier.py`) | SL 1.5 ATR, TP 2.0 ATR, horizon 8 |
| WF folds | `apps/run_rolling_walkforward.py` / `research/mtf_h4_h1_m5/backtest.py` | Test year T; train T−6…T−2; val T−1; test T; years 2021–2026 |
| Trainer | `train_frozen()` | LGBM binary, seed 42, early stop on val |

**Caveat:** `train_frozen` feeds ternary labels (−1/0/1) into binary LGBM without `exclude_timeout` remapping used elsewhere. Preserve existing protocol for fair comparison (do not change mid-study).

**Gate mismatch to avoid:** `run_rolling_walkforward.TOP_PCT = 0.05` ≠ production/research **0.21**. This study must use **0.21**.

---

## 5. Existing validation / ablation infrastructure (reuse)

| Asset | Path | Reuse |
|-------|------|-------|
| Panel builder | `research/mtf_h4_h1_m5/backtest.py::build_wf_entry_panel_feats` | Yes — arbitrary feature list |
| M15 entry + score | `apply_m15_execution`, `score_panel` | Yes — production-aligned |
| Monte Carlo | `monte_carlo_10k` | Yes |
| Category compare | `apps/research_category_feature_compare.py` | Pattern for orchestration |
| FS ablation CLI | `apps/run_feature_selection_ablation.py` | Ranking/forward helpers; **do not** bind universe to frozen 6 |
| Diagnostics | `feature_diagnostics/`, `apps/report_feature_library_research.py` | Corr / MI / SHAP / gain |
| Prior result | `results/h4_h1_m5/category_features/report.md` | Native 6 beat engine 35/48 on OOS PF/DD |

---

## 6. Possible leakage points

| Risk | Assessment |
|------|------------|
| Rolling / EMA / ATR / RSI / MACD | Causal (`min_periods`; no `center=True`) |
| `pct_change` / `shift` / slope | Backward only — OK |
| Same-bar candle OHLC | OK for H1-close decision |
| Session encodings | Calendar of bar timestamp — OK |
| Labels | Future barriers by design — must not enter X |
| Selection leakage | Must not pick features on final OOS then re-score same years without freeze protocol |
| Warmup NaNs | Early rows dropped if `drop_na=true` |

**Stop condition:** if any feature uses future bars, halt promotion.

---

## 7. Known / suspected redundancy (pre-analysis)

From code inspection (`statistical_builder.py`, `volatility_builder.py`):

| Pair / cluster | Relation |
|----------------|----------|
| `rolling_zscore` vs `rolling_mean_distance` | Same formula `(ret − mean) / std` — **near-identical** |
| `rolling_percentile` vs `rolling_rank` | Both rank current return in window → [0,1]; rank uses midrank ties |
| `rolling_quantile` | **Different:** window `quantile(q=0.75)` of returns — a **level**, not a rank of current ret |
| `volatility_regime_score` | `(atr_percentile_252 − 0.5) × 2` — deterministic transform of a **CORE** feature |
| `ema20_distance_atr` vs `ema20_distance_percent` | Same distance, different scale |
| `atr_percent` vs `rolling_volatility` | Related vol measures |

Phase 4 will quantify with Pearson/Spearman; do not auto-delete on correlation alone.

---

## 8. Recommended integration point

**Do not** rebuild FE, labels, or WF trainer.

**Do:**

1. Package `research/h1_feature_selection/` — inventory, quality, redundancy, experiment catalog (exact 38).
2. App `apps/research_h1_feature_selection.py` — thin orchestrator mirroring `research_category_feature_compare.py`.
3. Score via `build_wf_entry_panel_feats` + `apply_m15_execution` + `score_panel` + `monte_carlo_10k`.
4. Outputs under `results/h4_h1_m5/h1_feature_selection/` + charts + final report `docs/research/h1_feature_selection_report.md`.

**Excluded from X:** `ctx_h4_*`, `d1_*`, `m5_*`, `session_london` / NY / Asia flags.

---

## 9. Architecture constraint (unchanged)

```
H4 deterministic context  →  (out of scope for this study)
H1 native features (≤38)  →  H1 ML  →  trade signal
M15 pullback entry        →  M15 trail exit
```

---

## 10. Phase gate

Phase 1 complete. Next:

- Phase 2 — feature definition audit (all 38 formulas)
- Phase 3 — quality + leakage on `feature_matrix.parquet`
- Phase 4 — redundancy matrices
- Phase 5+ — baseline + group ablation under frozen trading protocol
