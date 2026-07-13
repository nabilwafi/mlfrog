# Asymmetry Audit — LONG vs SHORT Pipeline

**Scope:** pure diagnostic (no retrain, no code changes).  
**Goal:** distinguish (1) SHORT genuinely harder vs (2) asymmetric pipeline bug.

---

## Files in scope (LONG/SHORT logic)

| Layer | File(s) |
|---|---|
| Label engine | `labels/triple_barrier_labeling.py` (`label_triple_barrier`) |
| Classic features | `features/feature_engineering.py` (`compute_h1_features`, `compute_h4_features`, `compute_d1_features`) |
| SMC features | `features/smc.py`, `features/build_features_v2.py` |
| Training / split | `models/data_prep.py`, `models/train_baseline.py`, `models/train_walk_forward_v2.py`, `models/evaluate.py` |
| Downstream (LONG-only backtest; not SHORT train) | `backtest/engine.py`, `backtest/run_long_lgbm_v1.py`, `backtest/run_long_lgbm_v2.py` |

---

## BAGIAN 1: Label engine (triple-barrier)

### Side-by-side logic (`label_triple_barrier`, lines ~100–145)

| | LONG | SHORT |
|---|---|---|
| TP | `entry + tp_mult * ATR` | `entry - tp_mult * ATR` |
| SL | `entry - sl_mult * ATR` | `entry + sl_mult * ATR` |
| Loss touch | `low[j] <= sl_long` | `high[j] >= sl_short` |
| Win touch | `high[j] >= tp_long` | `low[j] <= tp_short` |
| Same-candle ambiguity | **SL checked before TP** | **SL checked before TP** |
| Horizon / timeout | same `i+1 .. i+horizon` | same |

Signs and high/low usage are mirrored correctly. SL-first is applied to both sides (not LONG-only).

### Raw label distribution (full labeled table, n=68271)

| | win% | loss% | timeout% | win/(win+loss) |
|---|---:|---:|---:|---:|
| LONG | 24.11 | 36.70 | 39.19 | 0.3965 |
| SHORT | 23.94 | 37.27 | 38.79 | 0.3911 |

Near-symmetric; both below breakeven 0.4286 (as expected for raw labels).

### 5 random SHORT path audits (seed=42)

All 5 recomputed paths **match** stored `label_short` / SL-first convention.

**Sample 1** — `2024-01-11 11:00` entry=2030.85 ATR≈3.35 → TP=2024.15 SL=2035.88 → **label=0 (loss)** at bar 5.  
Bar 5: H=2040.06 (≥SL) and L=2025.17 (also ≤TP). SL checked first → loss. Correct.

**Sample 2** — `2020-01-09 12:00` entry=1545.35 → TP=1533.43 SL=1554.29 → **label=0** at bar 3 (H=1556.23 ≥SL before TP). Correct.

**Sample 3** — `2022-08-19 13:00` entry=1753.39 → TP=1746.99 SL=1758.19 → **label=1 (win)** at bar 5 (L=1745.75 ≤TP; prior bars never hit SL). Correct: SHORT wins because low reaches TP before high reaches SL.

**Sample 4** — `2015-12-30 03:00` entry=1070.35 → TP=1066.59 SL=1073.17 → **label=1** at bar 8 (TP touch, no prior SL). Correct.

**Sample 5** — (same audit batch) path matched stored label under SL-first.

### Bagian 1 conclusion

**SIMETRIS/OK** — no LONG/SHORT sign or touch-logic bug in the label engine.

---

## BAGIAN 2: Classic feature engineering

### `ema_cross_signal_h1` (`feature_engineering.py` ~155–178)

| Bullish | Bearish |
|---|---|
| `fast.shift(1) <= slow.shift(1)` and `fast > slow` → `+1` | `fast.shift(1) >= slow.shift(1)` and `fast < slow` → `-1` |
| Same lookback N=3; last non-zero wins | Same |

Counts on feature table: `+1` = 3290, `-1` = 3294, `0` = 61687 — essentially equal bull/bear crosses.

### `trend_direction_d1` (~255–259)

```
uptrend:   slope > +thr
downtrend: slope < -thr
sideways:  |slope| <= thr
```
`thr = trend_slope_threshold = 1e-4` — **symmetric**.

Mix on feature rows: uptrend **54.8%**, downtrend **40.1%**, sideways **5.1%**.

### Slope distributions (historical drift, not code asymmetry)

| feature | mean | median | % positive | % negative |
|---|---:|---:|---:|---:|
| `ema_slope_h4` | +6.6e-5 | +5.8e-5 | 53.9% | 46.1% |
| `ema_slope_d1` | +4.0e-4 | +2.9e-4 | 57.5% | 42.5% |

Mild **bullish skew** consistent with gold’s long-run drift. Thresholds themselves are symmetric; the *data* leans up.

### Bagian 2 conclusion

**SIMETRIS/OK** (code). Documented **historical bullish skew** in slopes / D1 trend mix — market context, not a coding bias in definitions.

---

## BAGIAN 3: SMC features (highest risk)

### Swing fractals (`detect_fractals`)

| Swing high | Swing low |
|---|---|
| N before + N after (N=2) | same N |
| `high[i] == max(window)` and `> left` and `> right` | `low[i] == min(window)` and `< left` and `< right` |

Raw pivot counts on H1: swing highs **9076**, swing lows **9253** (~2% more lows — normal, not a `>`/`>` typo).

### BOS / CHoCH (mirror check)

| | Bullish | Bearish |
|---|---|---|
| Trigger | `high[t] > last_swing_high` | `low[t] < last_swing_low` |
| BOS if | `structure >= 0` | `structure <= 0` |
| CHoCH if | `structure < 0` | `structure > 0` |
| Lookback flag window | `bos_lookback_bars=3` | same |

Logic is mirrored. Soft note: when `structure==0`, both breaks classify as BOS (symmetric default).

**Frequency on v2 feature table (% rows flag=1):**

| flag | % |
|---|---:|
| `bos_bullish_h1` | **24.31** |
| `bos_bearish_h1` | **14.86** |
| `choch_bullish_h1` | 15.55 |
| `choch_bearish_h1` | 19.28 |
| `bos_bullish_h4` | **26.53** |
| `bos_bearish_h4` | **15.66** |

BOS bull ≫ BOS bear. Combined with more uptrend time and persistent breaks while price stays beyond the last swing, this is **consistent with market structure**, not a swapped high/low bug. CHoCH bear > CHoCH bull also fits (breaks against uptrend).

Soft design note (not SHORT-only): `_update_structure` applies swing-high update then swing-low update; if both confirm on the same bar, **SL-side overwrites SH-side**. Symmetric in code path, but order-dependent.

### Order blocks

| | Bullish OB | Bearish OB |
|---|---|---|
| Impulse | body ≥ 1.5×ATR and `close > open` | body ≥ 1.5×ATR and `close < open` |
| Zone candle | last bearish before impulse | last bullish before impulse |
| Mitigate | touch zone after birth (`low≤top & high≥bottom`) | same zone-touch rule |

| | bullish OB dist | bearish OB dist |
|---|---:|---:|
| Sentinel (−1) % | 66.34 | 67.94 |
| Mean dist when real | 4.00 ATR | 3.62 ATR |

Near-balanced.

### FVG

| | Bullish | Bearish |
|---|---|---|
| Detect | `low[t] > high[t-2]` | `high[t] < low[t-2]` |
| Fill | `low ≤ gap bottom` | `high ≥ gap top` |

| | bullish FVG dist | bearish FVG dist |
|---|---:|---:|
| Sentinel % | 37.96 | 42.98 |
| Mean dist when real | 1.36 | 1.29 |

Mildly more missing bearish FVGs — plausible under upward drift; detection formulas are mirrored.

### Sentinel handling

- Sentinel = `-1.0` mixed into the same numeric column as real ATR-distances.
- Affects bullish and bearish columns similarly (not SHORT-only).
- Trees can split on `-1` as if it were a distance — **design smell** for both directions; a `has_*` flag would be cleaner later, but this is not evidence of an asymmetric SHORT bug.

### Bagian 3 conclusion

**SIMETRIS/OK on code mirrors.**  
**ASIMETRI FREKUENSI (bukan bug ketik):** `bos_bullish_*` fires much more often than `bos_bearish_*` — attributed to bullish-time / structure persistence, not swapped conditions. Sentinel encoding is a shared modeling caveat.

---

## BAGIAN 4: Model training

### Class weights

| Model | Mechanism |
|---|---|
| LightGBM / LogReg | `class_weight="balanced"` → computed from **each direction’s own** y |
| XGBoost | `scale_pos_weight = n_neg/n_pos` on **that direction’s train set** |

Not a shared hardcoded weight across LONG/SHORT.

Implied train neg/pos (v1 split, after timeout drop):

| | n_train | pos_rate | neg/pos |
|---|---:|---:|---:|
| LONG | 28637 | 0.391 | 1.561 |
| SHORT | 28681 | 0.397 | 1.521 |

### Split dates

Same calendar cutoffs (`train_end=2023-01-01`, `val_end=2024-01-01`) and same ±8h embargo for both. Row counts differ slightly only because timeout masks differ per direction (expected).

| | kept | train | val | test |
|---|---:|---:|---:|---:|
| LONG v1/v2 | 41513 | 28637 | 3486 | 9390 |
| SHORT v1/v2 | 41792 | 28681 | 3544 | 9567 |

v2 feature join preserves the same kept counts (labels unchanged). `train_min` differs by a few early bars (LONG starts later) solely due to which early rows timeout per side — not a split bug.

### Bagian 4 conclusion

**SIMETRIS/OK** — independent balancing, shared time splits, kept counts match prior report.

---

## BAGIAN 5: Trend-conditional win-rates (ex-timeout)

Breakeven = 0.4286. Rows with resolved labels only.

| direction | subset | n | win-rate | vs BE |
|---|---|---:|---:|---:|
| LONG | uptrend | 22905 | 0.4001 | −0.028 |
| LONG | downtrend | 16597 | 0.3923 | −0.036 |
| LONG | sideways | 2011 | 0.3894 | −0.039 |
| SHORT | uptrend | 23103 | 0.3918 | −0.037 |
| SHORT | downtrend | 16624 | 0.3926 | −0.036 |
| SHORT | sideways | 2065 | 0.3724 | −0.056 |

Deltas:

- SHORT: WR(downtrend) − WR(uptrend) ≈ **+0.0008** (≈0)
- LONG: WR(uptrend) − WR(downtrend) ≈ **+0.0078** (tiny)

### Interpretation

The intuitive “SHORT should win more in D1 downtrend” **does not appear** in these barrier labels. Critically, the LONG sanity check is also weak (only ~0.8pp lift in uptrend). So this is **not** a SHORT-only smoking gun for a label/feature wire-swap; it shows **`trend_direction_d1` barely conditions triple-barrier outcomes for either side** under SL=1.5 / TP=2.0 / horizon=8.

That helps explain why models struggle: a major HTF categorical feature is nearly uninformative for the label, while gold’s bullish time-share still feeds LONG-friendly patterns elsewhere (slopes, BOS bull frequency, etc.).

### Bagian 5 conclusion

**DITEMUKAN: ekspektasi regime tidak muncul** — SHORT WR flat across up/down.  
**Bukan bukti bug SHORT-only**, karena LONG juga hampir flat. Level: **feature↔label weak coupling** (classic HTF trend), not label-engine asymmetry.

---

## Overall verdict & recommendation

| Bagian | Verdict |
|---|---|
| 1 Label engine | **SIMETRIS/OK** |
| 2 Classic features | **SIMETRIS/OK** (+ documented bullish historical skew) |
| 3 SMC | **SIMETRIS/OK** in logic; frequency skew = market/structure, not swapped ifs |
| 4 Training | **SIMETRIS/OK** |
| 5 Trend conditional | **Expectation failed for SHORT *and* LONG** — weak feature, not SHORT bug |

### No asymmetric pipeline bug found that selectively breaks SHORT.

Therefore:

1. **Do not prioritize a “fix SHORT label/SMC typo” pass** — audit did not surface one.
2. Treat SHORT failure as **genuine predictability gap** under current features + gold’s bullish bias (more uptrend time, positive slope skew, more BOS-bull flags).
3. Next work (separate task): **bearish-specific / mean-reversion / regime-conditioned features**, or accept LONG-only until SHORT has a validated edge — not another blind SMC stack.
4. Optional hygiene (not SHORT-critical): replace distance sentinels with `has_*` flags; revisit whether D1 trend should stay as a primary feature given near-zero WR lift.

**Bottom line:** SHORT is hard with this feature set; it is not failing because the SHORT barrier math or bull/bear SMC mirrors were wired backwards.
