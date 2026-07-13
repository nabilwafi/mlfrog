# Regime Characterization — Sealed-test 2026 vs 2020–2025

## Window

Same sealed-test-v2 window as controlled comparison:

- **2026-01-02 04:00:00+00:00 → 2026-07-08 11:00:00+00:00**
- Feature-table bars in window: **3031** (preds/labels subset is 1923 after timeout drop; regime stats use feature rows in the date window)
- Full calendar 2026 in feature table so far: 2026-01-01 → 2026-07-08 (3037 bars) — almost identical to sealed window

Source: `data/features/xauusd_h1_h4_d1_features.parquet` (no new features).

---

## Volatility & trend strength by year

Means and within-year std of ATR/ADX. `2026_sealed` = sealed window only.

| period | n_bars | atr_h1 mean | atr_h1 std | atr_h4 mean | atr_h4 std | adx_h1 mean | adx_h1 std | adx_h4 mean | adx_h4 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 5927 | 6.12 | 2.90 | 12.00 | 5.07 | 26.73 | 10.90 | 27.54 | 12.77 |
| 2021 | 5907 | 4.47 | 1.26 | 9.09 | 2.00 | 26.26 | 11.19 | 25.62 | 9.55 |
| 2022 | 5913 | 4.83 | 1.56 | 9.79 | 2.62 | 26.58 | 9.94 | 25.68 | 9.51 |
| 2023 | 5890 | 4.30 | 1.31 | 8.78 | 2.16 | 27.20 | 11.55 | 26.82 | 10.86 |
| 2024 | 5933 | 6.04 | 1.86 | 12.32 | 3.22 | 26.44 | 11.42 | 26.17 | 12.00 |
| 2025 | 5907 | 11.54 | 5.49 | 23.17 | 10.48 | 27.28 | 11.18 | 27.14 | 10.30 |
| **2026_sealed** | **3031** | **26.14** | **15.11** | **52.98** | **25.45** | **27.73** | **10.75** | **28.41** | **10.96** |

### Outlier check vs 2020–2025 yearly means

Compare sealed-2026 **mean** to the distribution of **yearly means** 2020–2025 (n=6). Flag if |z| > 1.5.

| series | mean of yearly means (2020–25) | std of yearly means | sealed-2026 mean | z-score | >1.5σ? |
|---|---:|---:|---:|---:|---|
| atr_h1 | 6.22 | 2.72 | 26.14 | **+7.33** | **YES** |
| atr_h4 | 12.52 | 5.42 | 52.98 | **+7.46** | **YES** |
| adx_h1 | 26.75 | 0.41 | 27.73 | +2.37 | **YES** |
| adx_h4 | 26.49 | 0.79 | 28.41 | +2.41 | **YES** |

Notes:

- **ATR is the dominant anomaly.** Sealed 2026 H1 ATR mean is ~4× 2020–25 average and ~2.3× even the already-elevated 2025.
- Within-year ATR std is also huge (15 vs ~1–5 historically) → choppier / wider barrier distances in absolute price terms.
- ADX means are only mildly high (~2.4σ of *year-mean* dispersion). Year-to-year ADX means barely move, so small absolute bumps look statistically large; economically ADX ~28 is still “normalish” trend strength, not a regime story by itself.

---

## Trend direction mix (`trend_direction_d1`)

% of H1 bars by D1 trend category:

| period | % uptrend | % downtrend | % sideways |
|---|---:|---:|---:|
| 2020 | 66.4 | 25.1 | 8.6 |
| 2021 | 44.5 | 51.2 | 4.3 |
| 2022 | 43.8 | 51.5 | 4.7 |
| 2023 | 56.8 | 38.9 | 4.3 |
| 2024 | 66.6 | 29.6 | 3.8 |
| 2025 | 81.0 | 15.5 | 3.5 |
| **2026_sealed** | **48.4** | **49.4** | **2.3** |

Reading:

- 2024–2025 were **strongly uptrend-dominated** (especially 2025 at ~81% up).
- Sealed 2026 is closer to **balanced up/down** (~48/49), more like 2021–2022 than like 2025.
- Sideways share is low (not a “dead range” year by this proxy); the shift is directional balance + extreme volatility, not a sideways takeover.

---

## Extra context (no new data fetch)

From the series itself (not an external calendar pull):

- Gold H1 ATR stepped up sharply in **2025**, then again in **early–mid 2026** — consistent with a high-volatility precious-metals regime, not a quiet continuation of 2021–2023.
- Barrier labels use ATR multiples, so absolute SL/TP distances in USD are much wider in 2026; cost in points is relatively smaller vs move size, but path noise and same-bar SL/TP ambiguity can still hurt classifiers trained mostly on quieter years.

---

## Regime conclusion

**Sealed-test 2026 looks like a volatility outlier**, not a “normal year” draw from 2020–2025:

- ATR (H1/H4) ≫ 1.5σ vs prior yearly means (z ~7).
- Direction mix breaks the 2024–2025 uptrend dominance.
- ADX is only mildly elevated; the story is **vol + two-way price action**, not “ADX collapsed.”

So Hipotesis B has **partial support as context**: this window is regime-unusual and a hard stress test.

It is **not** sufficient alone to explain v2’s loss, because controlled comparison shows **v1 remains profitable on the same dates** (`controlled_comparison_v1_vs_v2.md`). Prefer: hard regime (B) + v2 underperforms v1 on that regime (A).
