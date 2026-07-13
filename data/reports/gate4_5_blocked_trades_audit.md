# Gate 4.5 — Blocked Trades Audit (sealed 2026)

## Data source note

Prior guard run (`trades_with_guard.parquet`) only stores **executed** trades, not blocked signals. This audit **regenerated** blocked timestamps by replaying the sealed-2026 frame with causal regime tiers (W=4320, Z_REDUCE=2.0) and the same entry/no-stacking rules as the engine.

Window: `2026-01-02 04:00:00+00:00` → `2026-07-08 11:00:00+00:00` | model thr=0.51 | LONG barriers SL/TP/horizon = 1.5/2.0/8

Hypothetical sizing uses **no-guard** equity curve as-of signal time (1% risk, Method A costs).
Group z_block=4.0 is the subset of z_block=3.5 with `atr_zscore >= 4.0`.

Replay counts: blocked@3.5 = 13, blocked@4.0 = 7 (report previously said 13 / 7 — small diffs possible from equity-path / stacking interaction).

## Summary statistics

### Z_BLOCK = 3.5

- n_blocked = **13**
- outcomes: TP(win)=5 (38.5%), SL(loss)=6 (46.2%), timeout=2 (15.4%)
- would-be winrate TP/SL = 0.455
- total hypothetical PnL = **+66.55 USD** (+0.76 R)
- mean PnL/trade = +5.12 USD
- top-2 loss share of all SL abs PnL = 33.4%

### Z_BLOCK = 4.0

- n_blocked = **7**
- outcomes: TP(win)=1 (14.3%), SL(loss)=6 (85.7%), timeout=0 (0.0%)
- would-be winrate TP/SL = 0.143
- total hypothetical PnL = **-476.00 USD** (-4.69 R)
- mean PnL/trade = -68.00 USD
- top-2 loss share of all SL abs PnL = 33.4%

## Full table — Z_BLOCK 3.5

| timestamp | atr_zscore | regime_tier | outcome | pnl_usd | pnl_R | bars | mark |
|---|---:|---|---|---:|---:|---:|---|
| 2026-01-29 18:00:00+00:00 | 9.311 | extreme | tp | +135.09 | +1.330 | 2 | **WIN** |
| 2026-01-30 08:00:00+00:00 | 9.287 | extreme | sl | -101.90 | -1.003 | 3 | **LOSS** |
| 2026-01-30 09:00:00+00:00 | 8.981 | extreme | sl | -101.91 | -1.003 | 3 | **LOSS** |
| 2026-01-30 12:00:00+00:00 | 10.095 | extreme | sl | -101.87 | -1.003 | 8 | **LOSS** |
| 2026-02-02 02:00:00+00:00 | 10.955 | extreme | sl | -101.82 | -1.002 | 6 | **LOSS** |
| 2026-02-02 04:00:00+00:00 | 11.605 | extreme | sl | -101.80 | -1.002 | 4 | **LOSS** |
| 2026-02-02 05:00:00+00:00 | 11.074 | extreme | sl | -101.81 | -1.002 | 3 | **LOSS** |
| 2026-03-23 10:00:00+00:00 | 3.634 | extreme | tp | +132.40 | +1.329 | 4 | **WIN** |
| 2026-03-23 11:00:00+00:00 | 3.612 | extreme | tp | +132.40 | +1.329 | 3 | **WIN** |
| 2026-03-23 12:00:00+00:00 | 3.673 | extreme | tp | +132.41 | +1.329 | 2 | **WIN** |
| 2026-03-23 13:00:00+00:00 | 3.513 | extreme | tp | +132.39 | +1.329 | 1 | **WIN** |
| 2026-03-24 01:00:00+00:00 | 3.789 | extreme | timeout | -23.81 | -0.239 | 8 | **TIMEOUT** |
| 2026-03-24 02:00:00+00:00 | 3.550 | extreme | timeout | +36.76 | +0.369 | 8 | **TIMEOUT** |

## Full table — Z_BLOCK 4.0 (subset)

| timestamp | atr_zscore | regime_tier | outcome | pnl_usd | pnl_R | bars | mark |
|---|---:|---|---|---:|---:|---:|---|
| 2026-01-29 18:00:00+00:00 | 9.311 | extreme | tp | +135.09 | +1.330 | 2 | **WIN** |
| 2026-01-30 08:00:00+00:00 | 9.287 | extreme | sl | -101.90 | -1.003 | 3 | **LOSS** |
| 2026-01-30 09:00:00+00:00 | 8.981 | extreme | sl | -101.91 | -1.003 | 3 | **LOSS** |
| 2026-01-30 12:00:00+00:00 | 10.095 | extreme | sl | -101.87 | -1.003 | 8 | **LOSS** |
| 2026-02-02 02:00:00+00:00 | 10.955 | extreme | sl | -101.82 | -1.002 | 6 | **LOSS** |
| 2026-02-02 04:00:00+00:00 | 11.605 | extreme | sl | -101.80 | -1.002 | 4 | **LOSS** |
| 2026-02-02 05:00:00+00:00 | 11.074 | extreme | sl | -101.81 | -1.002 | 3 | **LOSS** |

## Candle context — highest-z blocked signals (from 3.5 set)

### Extreme case 1: z=11.605 @ 2026-02-02 04:00:00+00:00 → sl (PnL -101.80 USD / -1.00 R)

| idx | Date | O | H | L | C | notes |
|---:|---|---:|---:|---:|---:|---|
| 315 | 2026-01-30 23:00:00+00:00 | 4932.31 | 4932.52 | 4808.79 | 4847.31 |  |
| 316 | 2026-01-31 00:00:00+00:00 | 4847.07 | 4892.31 | 4831.41 | 4883.64 |  |
| 317 | 2026-02-02 02:00:00+00:00 | 4844.83 | 4875.53 | 4697.99 | 4740.43 |  |
| 318 | 2026-02-02 03:00:00+00:00 | 4740.99 | 4885.13 | 4739.78 | 4766.45 |  |
| 319 | 2026-02-02 04:00:00+00:00 | 4766.11 | 4784.30 | 4582.22 | 4737.19 | << SIGNAL |
| 320 | 2026-02-02 05:00:00+00:00 | 4737.02 | 4772.71 | 4698.28 | 4710.00 |  |
| 321 | 2026-02-02 06:00:00+00:00 | 4709.84 | 4728.88 | 4630.02 | 4646.53 |  |
| 322 | 2026-02-02 07:00:00+00:00 | 4646.46 | 4712.66 | 4603.05 | 4679.48 |  |
| 323 | 2026-02-02 08:00:00+00:00 | 4679.45 | 4696.30 | 4499.12 | 4550.95 |  |
| 324 | 2026-02-02 09:00:00+00:00 | 4551.03 | 4574.23 | 4403.34 | 4502.80 |  |
| 325 | 2026-02-02 15:00:00+00:00 | 4765.74 | 4811.83 | 4743.46 | 4794.78 |  |
| 326 | 2026-02-03 00:00:00+00:00 | 4652.57 | 4686.80 | 4643.67 | 4664.63 |  |
| 327 | 2026-02-03 20:00:00+00:00 | 4955.35 | 4969.38 | 4907.41 | 4928.97 |  |

### Extreme case 2: z=11.074 @ 2026-02-02 05:00:00+00:00 → sl (PnL -101.81 USD / -1.00 R)

| idx | Date | O | H | L | C | notes |
|---:|---|---:|---:|---:|---:|---|
| 316 | 2026-01-31 00:00:00+00:00 | 4847.07 | 4892.31 | 4831.41 | 4883.64 |  |
| 317 | 2026-02-02 02:00:00+00:00 | 4844.83 | 4875.53 | 4697.99 | 4740.43 |  |
| 318 | 2026-02-02 03:00:00+00:00 | 4740.99 | 4885.13 | 4739.78 | 4766.45 |  |
| 319 | 2026-02-02 04:00:00+00:00 | 4766.11 | 4784.30 | 4582.22 | 4737.19 |  |
| 320 | 2026-02-02 05:00:00+00:00 | 4737.02 | 4772.71 | 4698.28 | 4710.00 | << SIGNAL |
| 321 | 2026-02-02 06:00:00+00:00 | 4709.84 | 4728.88 | 4630.02 | 4646.53 |  |
| 322 | 2026-02-02 07:00:00+00:00 | 4646.46 | 4712.66 | 4603.05 | 4679.48 |  |
| 323 | 2026-02-02 08:00:00+00:00 | 4679.45 | 4696.30 | 4499.12 | 4550.95 |  |
| 324 | 2026-02-02 09:00:00+00:00 | 4551.03 | 4574.23 | 4403.34 | 4502.80 |  |
| 325 | 2026-02-02 15:00:00+00:00 | 4765.74 | 4811.83 | 4743.46 | 4794.78 |  |
| 326 | 2026-02-03 00:00:00+00:00 | 4652.57 | 4686.80 | 4643.67 | 4664.63 |  |
| 327 | 2026-02-03 20:00:00+00:00 | 4955.35 | 4969.38 | 4907.41 | 4928.97 |  |
| 328 | 2026-02-03 21:00:00+00:00 | 4928.97 | 4954.08 | 4885.24 | 4908.13 |  |

### Extreme case 3: z=10.955 @ 2026-02-02 02:00:00+00:00 → sl (PnL -101.82 USD / -1.00 R)

| idx | Date | O | H | L | C | notes |
|---:|---|---:|---:|---:|---:|---|
| 313 | 2026-01-30 21:00:00+00:00 | 4849.56 | 4893.04 | 4678.88 | 4880.19 |  |
| 314 | 2026-01-30 22:00:00+00:00 | 4880.16 | 4947.98 | 4839.44 | 4932.25 |  |
| 315 | 2026-01-30 23:00:00+00:00 | 4932.31 | 4932.52 | 4808.79 | 4847.31 |  |
| 316 | 2026-01-31 00:00:00+00:00 | 4847.07 | 4892.31 | 4831.41 | 4883.64 |  |
| 317 | 2026-02-02 02:00:00+00:00 | 4844.83 | 4875.53 | 4697.99 | 4740.43 | << SIGNAL |
| 318 | 2026-02-02 03:00:00+00:00 | 4740.99 | 4885.13 | 4739.78 | 4766.45 |  |
| 319 | 2026-02-02 04:00:00+00:00 | 4766.11 | 4784.30 | 4582.22 | 4737.19 |  |
| 320 | 2026-02-02 05:00:00+00:00 | 4737.02 | 4772.71 | 4698.28 | 4710.00 |  |
| 321 | 2026-02-02 06:00:00+00:00 | 4709.84 | 4728.88 | 4630.02 | 4646.53 |  |
| 322 | 2026-02-02 07:00:00+00:00 | 4646.46 | 4712.66 | 4603.05 | 4679.48 |  |
| 323 | 2026-02-02 08:00:00+00:00 | 4679.45 | 4696.30 | 4499.12 | 4550.95 |  |
| 324 | 2026-02-02 09:00:00+00:00 | 4551.03 | 4574.23 | 4403.34 | 4502.80 |  |
| 325 | 2026-02-02 15:00:00+00:00 | 4765.74 | 4811.83 | 4743.46 | 4794.78 |  |

### Extreme case 4: z=10.095 @ 2026-01-30 12:00:00+00:00 → sl (PnL -101.87 USD / -1.00 R)

| idx | Date | O | H | L | C | notes |
|---:|---|---:|---:|---:|---:|---|
| 300 | 2026-01-30 08:00:00+00:00 | 5213.56 | 5240.85 | 5179.21 | 5184.21 |  |
| 301 | 2026-01-30 09:00:00+00:00 | 5184.14 | 5205.00 | 5145.21 | 5155.49 |  |
| 302 | 2026-01-30 10:00:00+00:00 | 5155.49 | 5185.92 | 5117.35 | 5180.70 |  |
| 303 | 2026-01-30 11:00:00+00:00 | 5180.70 | 5180.70 | 5050.71 | 5116.61 |  |
| 304 | 2026-01-30 12:00:00+00:00 | 5116.52 | 5131.99 | 4941.97 | 5013.34 | << SIGNAL |
| 305 | 2026-01-30 13:00:00+00:00 | 5013.31 | 5122.29 | 5000.40 | 5119.63 |  |
| 306 | 2026-01-30 14:00:00+00:00 | 5119.49 | 5144.75 | 5088.97 | 5135.80 |  |
| 307 | 2026-01-30 15:00:00+00:00 | 5135.82 | 5145.60 | 5055.43 | 5064.20 |  |
| 308 | 2026-01-30 16:00:00+00:00 | 5064.18 | 5099.30 | 4992.45 | 5019.25 |  |
| 309 | 2026-01-30 17:00:00+00:00 | 5019.41 | 5112.96 | 5014.29 | 5032.62 |  |
| 310 | 2026-01-30 18:00:00+00:00 | 5032.62 | 5073.91 | 4980.40 | 5050.37 |  |
| 311 | 2026-01-30 19:00:00+00:00 | 5050.43 | 5051.91 | 4943.44 | 4985.61 |  |
| 312 | 2026-01-30 20:00:00+00:00 | 4985.37 | 4986.05 | 4799.91 | 4849.76 |  |

### Extreme case 5: z=9.311 @ 2026-01-29 18:00:00+00:00 → tp (PnL +135.09 USD / +1.33 R)

| idx | Date | O | H | L | C | notes |
|---:|---|---:|---:|---:|---:|---|
| 283 | 2026-01-29 14:00:00+00:00 | 5483.47 | 5528.72 | 5470.83 | 5522.58 |  |
| 284 | 2026-01-29 15:00:00+00:00 | 5522.50 | 5542.80 | 5518.45 | 5534.21 |  |
| 285 | 2026-01-29 16:00:00+00:00 | 5534.21 | 5541.14 | 5500.41 | 5510.29 |  |
| 286 | 2026-01-29 17:00:00+00:00 | 5510.40 | 5549.46 | 5491.00 | 5496.94 |  |
| 287 | 2026-01-29 18:00:00+00:00 | 5496.95 | 5496.95 | 5105.32 | 5181.95 | << SIGNAL |
| 288 | 2026-01-29 19:00:00+00:00 | 5182.00 | 5310.66 | 5171.36 | 5264.34 |  |
| 289 | 2026-01-29 20:00:00+00:00 | 5264.37 | 5375.37 | 5264.37 | 5334.54 |  |
| 290 | 2026-01-29 21:00:00+00:00 | 5334.49 | 5361.59 | 5307.92 | 5339.00 |  |
| 291 | 2026-01-29 22:00:00+00:00 | 5339.00 | 5345.01 | 5290.56 | 5305.08 |  |
| 292 | 2026-01-29 23:00:00+00:00 | 5305.20 | 5406.96 | 5282.86 | 5393.96 |  |
| 293 | 2026-01-30 00:00:00+00:00 | 5394.02 | 5425.77 | 5378.13 | 5379.69 |  |
| 294 | 2026-01-30 02:00:00+00:00 | 5378.78 | 5449.33 | 5377.87 | 5437.91 |  |
| 295 | 2026-01-30 03:00:00+00:00 | 5438.12 | 5450.77 | 5389.87 | 5406.44 |  |

## Pattern check (ADX on wrong vs right blocks)

- Among blocked-that-would-WIN: mean adx_h1=50.1, adx_h4=63.2
- Among blocked-that-would-LOSE: mean adx_h1=50.8, adx_h4=40.0
- ADX gap between wrong/right blocks is small in this sample.


## Required conclusions

1. **Verdict utama**
   - **Z_BLOCK=3.5 (n=13):** 6 LOSS / 5 WIN / 2 TIMEOUT — **mixed**. Net PnL if allowed: **+66.55 USD** (guard would have *left money on the table* on net at this looser threshold).
   - **Z_BLOCK=4.0 (n=7):** 6 LOSS / 1 WIN / 0 TIMEOUT — **much cleaner "mostly would have lost"** (86% SL). Net PnL if allowed: **-476 USD (−4.7 R)**.
   - The March-23 cluster (z ≈ 3.51–3.79) accounts for **all four extra wins** in the 3.5-only bag — volatile but continuing LONG that hit TP. That is why 3.5 looks lucky-harmful while 4.0 looks protective.

2. **Tail vs dispersed**
   - SL sizes are almost all **~−1.0 R** (same risk model) — not one catastrophic multi-R blow-up. Top-2 share of SL abs PnL ≈ 33% → **dispersed equal-R losses**, not a single fat-tail save.
   - Still valuable: blocking a **burst of consecutive SL** during the Jan 30–Feb 2 crash (z 9–11, wild H/L ranges in candle dumps) avoids a correlated losing streak, even if each ticket is only ~1R.

3. **Wrong-block / refine ideas**
   - ADX(H1) nearly identical on would-be wins vs losses (~50). H4 ADX higher on would-be wins (63 vs 40) — weak hint that extreme-z + strong H4 trend might be "trending vol" not chop, but **n too small** to encode into rules yet.
   - Practical takeaway already available without ADX: **prefer Z_BLOCK=4.0 over 3.5** on this window (matches sensitivity pick from the guard report).

4. **Gate 5 recommendation**
   - **Z_BLOCK=4.0:** audit supports the sealed CAGR lift as **mostly avoiding real losses** (6/7 would-be SL), not pure luck — but **n=7 is still tiny**.
   - **Z_BLOCK=3.5:** do **not** trust from this audit alone (net +PnL if unblocked).
   - Before calling Gate 4 "settled": run the same blocked-trade audit on **full 2024–2025** (larger blocked set). If the z≥4 bag stays net-negative there, Gate 5 paper trade with guard ON at **4.0** is reasonable. If historical bags flip net-positive, sealed improvement was fragile.


## Artifacts
- `data/backtest_v3_guarded/blocked_signals_audit_z35_sealed.parquet`
- `data/backtest_v3_guarded/blocked_signals_audit_z40_sealed.parquet`
- `data/reports/gate4_5_blocked_trades_audit.md`
