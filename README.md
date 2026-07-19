# XAUUSD ML Trading Platform

Production-oriented Machine Learning Trading Platform (train / backtest / paper / live).

## Design

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DOMAIN_MODEL.md](DOMAIN_MODEL.md)
- [LIFECYCLE.md](LIFECYCLE.md)

Settled strategy constants: `settings/strategy.py`

## Setup

```bash
pip install -r requirements.txt
```

Copy `configs/config.example.yaml` → `configs/config.yaml` and fill MT5 credentials.
Blobs live under `artifacts/` (gitignored).

## Sprint 1 — market data

```bash
python apps/fetch_market_data.py --provider mt5 --symbol XAUUSD --timeframe H1 --start 2020-01-01 --end 2025-12-31
```

## Sprint 2 — features

```bash
python apps/generate_features.py --symbol XAUUSD --timeframe H1
```

## Sprint 3 — labels (long + short)

```bash
python apps/generate_labels.py --symbol XAUUSD --timeframe H1 --strategy triple_barrier --sides long short
```

Output: `artifacts/labels/{SYMBOL}/{TF}/{side}/triple_barrier_v1.parquet`

## Sprint 4 — datasets (time splits)

```bash
python apps/build_dataset.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/datasets/{SYMBOL}/{TF}/{side}/{train,validation,test,sealed}.parquet`

## Sprint 5 — model training

```bash
python apps/train_model.py --symbol XAUUSD --timeframe H1 --side long --algorithm lightgbm
```

Output: `artifacts/models/{SYMBOL}/{TF}/{side}/` (`model.pkl`, `metadata.json`, `feature_importance.parquet`, `training_report.md`)

Algorithms are plug-ins (`logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost`) via `TrainerRegistry` — add a trainer module without changing `TrainingPipeline`.

## Sprint 5.5 — diagnostics

```bash
python apps/run_diagnostics.py --symbol XAUUSD --timeframe H1 --side long
```

Output: `artifacts/diagnostics/{SYMBOL}/{TF}/{side}/` (markdown report, CSVs, `metadata.json`).

Run after every training session, before calibration or backtesting.

## Sprint 5.6 — research

```bash
python apps/run_research.py --symbol XAUUSD --timeframe H1 --side long
```

Output: `artifacts/research/{SYMBOL}/{TF}/{side}/` (walk-forward, feature stability/drift, regimes, label stability, `research_summary.md`).

Hypothesis engine only — does not tune models.

## Sprint 6 — feature engineering (stationary / normalized)

```bash
python apps/run_feature_engineering.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/features/{SYMBOL}/{TF}/feature_matrix.parquet` (+ `feature_metadata.json`, `feature_report.md`).

Converts classic indicator levels into scale-free / regime-robust features. Does not add SMC or new indicator families.

## Sprint 6.5 — feature diagnostics

```bash
python apps/run_feature_diagnostics.py --symbol XAUUSD --timeframe H1 --side long
```

Output under `artifacts/research/{SYMBOL}/{TF}/{side}/`: correlation, MI, permutation, SHAP, stability, redundancy, selection candidates, `feature_diagnostics_report.md`.

Run after Sprint 6 feature engineering, before model retraining.

## Sprint 7 — feature ablation

```bash
python apps/run_feature_ablation.py --symbol XAUUSD --timeframe H1 --side long
```

Output: `artifacts/research/feature_ablation/` (summary/metrics/rankings/report + charts).

Measures category and subset contribution via walk-forward — not a tuning sprint.

Tests:

```bash
python -m unittest tests.test_feature_ablation -v
```

## Sprint 8 — model benchmark

```bash
python apps/run_model_benchmark.py --symbol XAUUSD --timeframe H1 --side long
```

Output: `artifacts/research/model_benchmark/` (`model_benchmark.csv`, `model_rankings.csv`, `model_benchmark.md`, charts).

Fair walk-forward comparison of 7 algorithms on the same 35-feature library — picks Production Base Model, does not tune hyperparameters.

```bash
python -m unittest tests.test_model_benchmark -v
```

## Sprint 9 — market context (H4 → H1)

```bash
python apps/run_market_context.py --symbol XAUUSD --base H1 --context H4
```

Output:
- `artifacts/context/{SYMBOL}/H1/` — `context.parquet`, `context_metadata.json`, `context_report.md`, `context_features.csv`
- `artifacts/features/{SYMBOL}/H4/` — optional H4 engineered feature matrix
- `artifacts/datasets/{SYMBOL}/H1/{side}/v2/` — dataset v2 (H1 features + H4 context + labels)

Attaches latest **completed** H4 context to every H1 candle. Not for entry/execution. Does **not** retrain the H1 model.

```bash
python -m unittest tests.test_market_context -v
```

## Sprint 10 — context impact & ablation

```bash
python apps/run_context_impact.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/context_impact/` (`comparison.csv`, `ablation_results.csv`, `context_impact_report.md`, charts).

Measures whether Sprint-9 H4 context improves H1 LightGBM (long and short run independently). Does not modify the context engine.

```bash
python -m unittest tests.test_context_impact -v
```

## Sprint 11 — H4 structure enhancement

```bash
python apps/run_h4_structure.py --symbol XAUUSD --base H1 --context H4
```

Output: `artifacts/research/h4_structure/` (`structure_features.parquet`, `structure_report.md`, `ablation_results.csv`, `feature_statistics.csv`, charts).

Richer causal H4 structure features (swings, BOS, liquidity, price location). Does not add volatility features or retrain production H1.

```bash
python -m unittest tests.test_h4_structure -v
```

## Sprint 12 — structure selection & stability

```bash
python apps/run_structure_selection.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/structure_selection/` (`selection_report.md`, `feature_stability.csv`, `experiment_results.csv`, charts).

Selects the minimal robust H4 structure subset (no new features). Requires Sprint-11 `structure_features.parquet`.

```bash
python -m unittest tests.test_structure_selection -v
```

## Sprint 13 — H1 Long/Short model v2 validation

```bash
python apps/run_model_v2_validation.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/models/v2/` (`long_model_v2_report.md`, `short_model_v2_report.md`, `comparison.csv`, `feature_importance.csv`, charts).

A/B validates Sprint-12 selected structure sets (Long top-5, Short `swing_quality`). No threshold tuning or execution.

```bash
python -m unittest tests.test_model_v2_validation -v
```

## Sprint 14 — Probability quality analysis

```bash
python apps/run_probability_quality.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/probability_quality/` (`probability_quality_report.md`, `probability_bucket_analysis.csv`, `confidence_analysis.csv`, charts).

Re-runs Long/Short v2 walk-forward to capture OOF probabilities; analyzes buckets, monotonicity, and confidence. No calibration fitting or execution.

```bash
python -m unittest tests.test_probability_quality -v
```

## Sprint 15 — Probability collapse diagnosis

```bash
python apps/run_probability_diagnosis.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/probability_diagnosis/` (`probability_diagnosis_report.md`, `prediction_distribution.csv`, `wf_probability_drift.csv`, charts).

Diagnoses why v2 LightGBM probs sit below 0.50 (imbalance, regularization, ranking vs confidence). No calibration or execution.

```bash
python -m unittest tests.test_probability_diagnosis -v
```

## Sprint 16 — Probability calibration & percentile thresholds

```bash
python apps/run_probability_calibration.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/probability_calibration/` (`probability_calibration_report.md`, `threshold_analysis.csv`, `context_filter_analysis.csv`, charts).

Walk-forward-safe Platt/Isotonic calibration on v2 OOF probs; percentile gates and H4 structure filters. Uses Sprint-15 diagnosis predictions when present.

```bash
python -m unittest tests.test_probability_calibration -v
```

## Meta dataset validation (research only)

```bash
python apps/run_meta_dataset_validation.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/meta_dataset_validation/` (`meta_dataset_validation_report.md`, percentile/WF/stability CSVs, charts).

Validates whether v2 OOF percentile candidates are suitable for a future meta-label dataset. Does **not** train a meta model.

```bash
python -m unittest tests.test_meta_dataset_validation -v
```

## Sprint 18 — Meta feature research

```bash
python apps/run_meta_feature_research.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/meta_feature_research/` (`meta_feature_research_report.md`, scores/stability/interactions CSVs, charts).

Model-free feature research for meta labeling (MI / IV / Spearman / WF stability). Does **not** train a meta classifier.

```bash
python -m unittest tests.test_meta_feature_research -v
```

## Sprint 19 ? Meta feature ablation

```bash
python apps/run_meta_feature_ablation.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/meta_feature_ablation/` (`meta_feature_ablation_report.md`, ablation/correlation CSVs, charts).

Fixed-param Meta LightGBM ablation across feature groups. Does **not** tune thresholds/HPs or train the final meta model.

```bash
python -m unittest tests.test_meta_feature_ablation -v
```

## Sprint 20 - Production Meta Model

```bash
python apps/run_meta_model.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/meta_model/` (`meta_model_report.md`, OOF preds, LOO + full LightGBM models, equity curves).

Trains the Trade/Skip Meta Model on Sprint-19 features. Primary Long/Short models are **not** retrained. Thresholds 0.40-0.60 are reported only (not optimized).

```bash
python -m unittest tests.test_meta_model -v
```

## Sprint 21 - Realistic portfolio backtest

```bash
python apps/run_portfolio_backtest.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/portfolio_backtest/` (`backtest_report.md`, `risk_report.md`, equity/trade logs, Monte Carlo, charts).

Portfolio money-management backtest on frozen Primary + Meta (thr=0.45). No retrain / no threshold search.

```bash
python -m unittest tests.test_portfolio_backtest -v
```

## Sprint 22 - Multi-timeframe confidence layer

```bash
python apps/run_confidence_layer.py --symbol XAUUSD --timeframe H1
```

Output: `artifacts/research/confidence_layer/` (`confidence_layer_report.md`, buckets/regime/risk/TP/trail/calibration/SHAP).

Execution layer above frozen Primary + Meta. No model retrain. Meta gate fixed at 0.45.

```bash
python -m unittest tests.test_confidence_layer -v
```

## Sprint 23 - Trade Quality Engine

```bash
python apps/run_trade_quality.py --symbol XAUUSD --timeframe H1
```

Requires Sprint 22 `confidence_panel.parquet`. Models frozen; Meta gate 0.45. Research only: composite Trade Quality score, bucket monotonicity, dynamic risk schedules, interactions, SHAP, LOO yearly stability.

Output: `artifacts/research/trade_quality/` (`trade_quality_report.md`, charts, CSVs).

```bash
python -m unittest tests.test_trade_quality -v
```

## Sprint 24 - Portfolio Heat Management

```bash
python apps/run_portfolio_heat.py --symbol XAUUSD --timeframe H1
```

Requires Sprint 22 `confidence_panel.parquet`. Models + Confidence schedule frozen (`skip40_flat1`). Research only: concurrent heat, daily/weekly stops, cooldowns, session caps, Monte Carlo, walk-forward.

Output: `artifacts/research/portfolio_heat/` (`portfolio_heat_report.md`).

```bash
python -m unittest tests.test_portfolio_heat -v
```

## Sprint 25 - Position Management

```bash
python apps/run_position_mgmt.py --symbol XAUUSD --timeframe H1
```

Requires Sprint 24 `best_policy_trades.parquet` + raw H1. Frozen Primary/Meta/Confidence/Heat. Same entries; H1 path simulation for BE, partial TP, ATR trail, time/vol exits, pyramid, scale-in.

Output: `artifacts/research/position_mgmt/`.

```bash
python -m unittest tests.test_position_mgmt -v
```

## Sprint 26 - Adverse Trade Recovery Engine (ATRE)

```bash
python apps/run_atre.py --symbol XAUUSD --timeframe H1
```

Frozen production stack. Same Heat entries. Causal MAE / underwater / momentum / structure / recovery-score early-exit research. Rejects policies that fail significance + WF + MC.

Output: `artifacts/research/atre/`.

```bash
python -m unittest tests.test_atre -v
```

## Sprint 27 - Production Paper Trading Infrastructure

Research frozen. Infrastructure only (paper fills, async workers, Postgres, Telegram, monitoring).

Docs: [docs/production/ARCHITECTURE.md](docs/production/ARCHITECTURE.md) · [docs/production/DEPLOYMENT.md](docs/production/DEPLOYMENT.md)

```bash
# Replay frozen Heat signals through production pipeline (no DB required)
python apps/run_paper_trading.py --mode replay --max-signals 50

# Idle paper loop + monitoring
python apps/run_paper_trading.py --mode loop
```

Schema: `sql/production_schema.sql`  
Monitoring: `http://<LAN-IP>:8787/health` · `/metrics` · `/portfolio`  
(`paper_trading.monitoring_host: 0.0.0.0` — use `127.0.0.1` for local-only)

```bash
python -m unittest tests.test_paper_production -v
```
