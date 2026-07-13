# Confidence-Weighted Sizing Experiment (isolated)

## Setup

- Model: LONG v3 @ thr=**0.51** (unchanged)
- Barriers / horizon / costs: unchanged
- Regime guard: Z_BLOCK=**4.0**, Z_REDUCE=2.0, elevated×0.5 (unchanged)
- ConfidenceEngine: Platt calibrator (unchanged)
- **Only** change: `final_size_mult = regime_mult × confidence_size_mult`

### Schemes

| scheme | medium | high |
|---|---:|---:|
| default | 0.75 | 1.25 |
| conservative | 0.85 | 1.15 |
| aggressive | 0.5 | 1.5 |

Baseline = guard only (confidence mult = 1.0 for all).

## Comparison vs baseline

### Full test (2024+)

| scheme | CAGR % | Sharpe | Sortino | Calmar | Max DD % | Expectancy R | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (flat×guard) | 15.82 | 1.1193 | 4.8995 | 1.3437 | 11.77 | 0.1142 | 471 |
| default 0.75/1.25 | 16.48 | 1.3271 | 4.3648 | 1.5400 | 10.70 | 0.1115 | 472 |
| conservative 0.85/1.15 | 16.03 | 1.2316 | 4.7230 | 1.5234 | 10.53 | 0.1115 | 472 |
| aggressive 0.5/1.5 | 17.77 | 1.5325 | 3.2106 | 1.3481 | 13.18 | 0.1059 | 470 |

### Sealed-2026

| scheme | CAGR % | Sharpe | Sortino | Calmar | Max DD % | Expectancy R | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (flat×guard) | 15.23 | 1.1051 | 5.4037 | 2.4149 | 6.31 | 0.1064 | 103 |
| default 0.75/1.25 | 15.63 | 1.0302 | 3.9419 | 2.2971 | 6.80 | 0.1064 | 103 |
| conservative 0.85/1.15 | 16.45 | 1.1100 | 5.1341 | 2.7673 | 5.95 | 0.1064 | 103 |
| aggressive 0.5/1.5 | 14.58 | 0.8433 | 2.3862 | 1.7654 | 8.26 | 0.0939 | 102 |

## Side-by-side (user table — default vs baseline)

| metric | baseline (flat 1%, guard only) | confidence-weighted (default) |
|---|---:|---:|
| CAGR % | 15.82 (full) / 15.23 (sealed) | 16.48 / 15.63 |
| Sharpe | 1.1193 / 1.1051 | 1.3271 / 1.0302 |
| Sortino | 4.8995 (full) / 5.4037 | 4.3648 / 3.9419 |
| Calmar | 1.3437 (full) / 2.4149 | 1.5400 / 2.2971 |
| Max DD % | 11.77 / 6.31 | 10.70 / 6.80 |
| Expectancy (R) | 0.1142 (full) / 0.1064 | 0.1115 / 0.1064 |

## Risk-adjusted deltas (default − baseline)

- Full: Sharpe ↑, Sortino ↓, Calmar ↑
- Sealed: Sharpe ↓, Sortino ↓, Calmar ↓

## Sensitivity trade-off notes

- **default**: full Sharpe 1.3271 (Δ0.2078), MaxDD 10.70 (Δ-1.07); sealed Sharpe 1.0302 (Δ-0.0750). score=+0.1328
- **conservative**: full Sharpe 1.2316 (Δ0.1123), MaxDD 10.53 (Δ-1.25); sealed Sharpe 1.1100 (Δ0.0048). score=+0.1172
- **aggressive**: full Sharpe 1.5325 (Δ0.4132), MaxDD 13.18 (Δ1.41); sealed Sharpe 0.8433 (Δ-0.2618). score=+0.1232

**Best raw score:** `default`. **Adoption pick:** `conservative` (default (raw score) but **conservative** is the best that clears adoption bar).

## Required conclusions

### 1. Does confidence sizing improve risk-adjusted return?

**PARTIAL YES** — **default** (0.75/1.25) is **mixed** (full Sharpe/Calmar ↑, Sortino ↓; sealed RA mostly ↓). **Conservative** (0.85/1.15) improves full Sharpe+Calmar and cuts MaxDD, with sealed Sharpe/Calmar flat-to-up and lower MaxDD — clearer cross-window confirmation.

### 2. Best multiplier scheme?

**`conservative`** ({'medium': 0.85, 'high': 1.15}) — best among schemes that pass the adoption bar (raw score leader `default` fails sealed stability).

### 3. Adopt into Risk Engine / paper trading?

**CONDITIONAL YES — adopt `conservative`** ({'medium': 0.85, 'high': 1.15}) as soft default sizing with `final = regime_mult × confidence_mult`. Merge into paper trading. Reject aggressive (0.5/1.5): sealed Sharpe/Sortino collapse and full MaxDD rises. Still soft tilt only — not hard R:R.

## Artifacts

- `data/reports/confidence_sizing_experiment.md`
- `data/reports/standard_report_long_v3_guarded_confsizing.md`
- `data/backtest_v3_guarded_confsizing/`
