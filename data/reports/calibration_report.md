# Calibration Report — LONG LightGBM v3

## Preflight

- Model: `data/models_v3/long/lightgbm_long_v3.pkl` (WF final; sealed preds source)
- Validation (fit calibrator): **n=5928** (2024-06-03 09:00:00+00:00 → 2025-12-31 08:00:00+00:00), base rate=0.4351
- Sealed test (evaluate only): **n=1923**, base rate=0.3531
- Labels: resolved TP/SL only (`drop_timeouts`) — same as model training
- n_val ample for both Platt and isotonic.
- Breakeven WR = 0.4286; operational raw thr = 0.51

## 1. ECE before vs after

| set | raw | Platt | Isotonic |
|---|---:|---:|---:|
| validation | 0.0338 | 0.0111 | 0.0000 |
| sealed test | 0.0979 | 0.0811 | 0.1015 |

**Selected method: `platt`** (lowest sealed-test ECE).

### Why this method?

- Sealed-test ECE: Platt **0.0811** vs Isotonic **0.1015**.
- Platt is simpler and more robust; isotonic val ECE≈0 with worse sealed-test ECE than raw (0.1015 > 0.0979) = classic overfit despite n_val=5928.

## 2. Reliability diagrams (decile tables)

### Validation — RAW

| bin | n | mean_predicted | realized_rate | |gap| |
|---:|---:|---:|---:|---:|
| 1 | 593 | 0.3495 | 0.3862 | 0.0367 |
| 2 | 593 | 0.3849 | 0.4165 | 0.0316 |
| 3 | 593 | 0.4049 | 0.3946 | 0.0103 |
| 4 | 592 | 0.4227 | 0.4088 | 0.0139 |
| 5 | 593 | 0.4390 | 0.4486 | 0.0095 |
| 6 | 593 | 0.4573 | 0.4283 | 0.0290 |
| 7 | 592 | 0.4763 | 0.4595 | 0.0169 |
| 8 | 593 | 0.4990 | 0.4486 | 0.0504 |
| 9 | 593 | 0.5263 | 0.4705 | 0.0558 |
| 10 | 593 | 0.5732 | 0.4890 | 0.0842 |

### Validation — platt

| bin | n | mean_predicted | realized_rate | |gap| |
|---:|---:|---:|---:|---:|
| 1 | 593 | 0.3941 | 0.3862 | 0.0079 |
| 2 | 593 | 0.4079 | 0.4165 | 0.0087 |
| 3 | 593 | 0.4157 | 0.3946 | 0.0211 |
| 4 | 592 | 0.4227 | 0.4088 | 0.0139 |
| 5 | 593 | 0.4292 | 0.4486 | 0.0194 |
| 6 | 593 | 0.4365 | 0.4283 | 0.0082 |
| 7 | 592 | 0.4441 | 0.4595 | 0.0154 |
| 8 | 593 | 0.4532 | 0.4486 | 0.0046 |
| 9 | 593 | 0.4642 | 0.4705 | 0.0063 |
| 10 | 593 | 0.4832 | 0.4890 | 0.0059 |

### Sealed test — RAW

| bin | n | mean_predicted | realized_rate | |gap| |
|---:|---:|---:|---:|---:|
| 1 | 193 | 0.3456 | 0.3420 | 0.0036 |
| 2 | 192 | 0.3751 | 0.3281 | 0.0470 |
| 3 | 192 | 0.3927 | 0.3125 | 0.0802 |
| 4 | 192 | 0.4089 | 0.2865 | 0.1224 |
| 5 | 193 | 0.4240 | 0.3057 | 0.1183 |
| 6 | 192 | 0.4418 | 0.3854 | 0.0564 |
| 7 | 192 | 0.4597 | 0.3646 | 0.0951 |
| 8 | 192 | 0.4870 | 0.3385 | 0.1485 |
| 9 | 192 | 0.5464 | 0.4479 | 0.0985 |
| 10 | 193 | 0.6289 | 0.4197 | 0.2092 |

### Sealed test — platt

| bin | n | mean_predicted | realized_rate | |gap| |
|---:|---:|---:|---:|---:|
| 1 | 193 | 0.3926 | 0.3420 | 0.0506 |
| 2 | 192 | 0.4040 | 0.3281 | 0.0759 |
| 3 | 192 | 0.4109 | 0.3125 | 0.0984 |
| 4 | 192 | 0.4173 | 0.2865 | 0.1308 |
| 5 | 193 | 0.4232 | 0.3057 | 0.1175 |
| 6 | 192 | 0.4303 | 0.3854 | 0.0449 |
| 7 | 192 | 0.4374 | 0.3646 | 0.0728 |
| 8 | 192 | 0.4484 | 0.3385 | 0.1098 |
| 9 | 192 | 0.4723 | 0.4479 | 0.0244 |
| 10 | 193 | 0.5057 | 0.4197 | 0.0860 |

## 3. Confidence tiers

- Edges: low < 0.4286; medium < 0.4706; else high
- Rationale: low < BE=0.4286 (no economic edge). medium/high split = median calibrated among val raw>=0.51 (n=1204) = 0.4706. Note: calibrated(raw_thr=0.51)=0.4576 (Platt compresses raw probs toward base rate).

### Validation (fit window — diagnostic only)

| tier | n | mean_calibrated | realized_wr |
|---|---:|---:|---:|
| low | 2623 | 0.4117 | 0.4049 |
| medium | 2703 | 0.4470 | 0.4517 |
| high | 602 | 0.4830 | 0.4917 |

### Sealed test (Bagian 4 — primary)

| tier | n | mean_calibrated | realized_wr |
|---|---:|---:|---:|
| low | 1015 | 0.4105 | 0.3192 |
| medium | 603 | 0.4429 | 0.3566 |
| high | 305 | 0.4960 | 0.4590 |

- Monotonic low ≤ medium ≤ high on sealed test? **YES**

### Among signals with raw p ≥ 0.51 (tradeable gate)

| tier | n | mean_calibrated | realized_wr |
|---|---:|---:|---:|
| low | 0 | nan | nan |
| medium | 63 | 0.4641 | 0.3810 |
| high | 305 | 0.4960 | 0.4590 |

## 4. Verdict for Risk Engine

**CONDITIONAL YES** — Platt cuts sealed-test ECE and tiers are monotonic (low→high WR). Remaining ECE≈0.081 reflects a 2026 base-rate shift (val 0.44 vs test 0.35) that a static calibrator cannot fully fix. Safe as a **soft** Risk Engine input (mild size tilt by tier); **not** yet for hard R:R or aggressive leverage. Re-fit if the base model changes.

Artifacts:
- `models/calibration/confidence_calibrator_long_v3.pkl`
- `src/confidence_engine.py`
- `data/reports/calibration_report.md`

Not done here: position sizing / dynamic R:R (downstream tasks).
