# H4 → H1 → M5 Hierarchical MTF Audit

**Phase 1** — repository audit before architectural changes  
**Date:** 2026-08-22  
**Architecture (this sprint):** H4 deterministic context → H1 ML → M5 deterministic execution  
**Out of scope:** D1, H4 ML, M5 ML

---

## 1. Existing architecture (production today)

```
H4 OHLCV → H4StructureBuilder → ContextJoinService → ctx_h4_* merged into H1 panel
H1 OHLCV → FeatureEngineeringService → H1 features
                              ↓
                    FEAT7 LGBM (top 21%)
                              ↓
                    H1 entry @ close
                              ↓
                    P0 trail @ M15 close
```

**Frozen production** (`production/__init__.py`):

| Component | Value |
|---|---|
| Model | FEAT7 LightGBM (`primary_feat7_frozen`) |
| Features | 6× H1-native + **`ctx_h4_swing_quality`** (raw H4 merge) |
| Gate | top 21% |
| Exit | P0 a0.25/d0.08, M15 trail, CAP 48 H1 bars |
| Portfolio | lot 0.01, heat 3R, max_open 5, daily 1R @ equity ≤ $80 |
| Pipeline | `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail` |

**Key difference from this sprint:** production merges **raw H4 feature** into H1; this sprint requires **abstract H4 signal contract** (no raw H4 columns in H1 input for Experiment B).

---

## 2. Data inventory

| TF | Path | Rows (approx) |
|---|---|---|
| H4 | `artifacts/raw/XAUUSD/H4/data.parquet` | ~19k |
| H1 | `artifacts/raw/XAUUSD/H1/data.parquet` | ~120k |
| M5 | `artifacts/raw/XAUUSD/M5/data.parquet` | ~1.4M |
| M15 | `artifacts/raw/XAUUSD/M15/data.parquet` | trail clock |
| H1 dataset v2 | `artifacts/datasets/XAUUSD/H1/{long,short}/v2/` | 68k/side |
| Labels | `artifacts/labels/XAUUSD/H1/{side}/triple_barrier_v1.parquet` | ternary −1/0/1 |

---

## 3. Reusable components

| Layer | Reuse |
|---|---|
| H4 native FE | `H4ContextBuilder`, `H4StructureBuilder` |
| Causal join | `ContextJoinService`, `available_at()` |
| H1 native FE | v2 parquet + `FeatureEngineeringService` |
| H1 ML | `train_frozen()`, `build_test_entries()` |
| WF windows | `build_windows()` 5y/1y/1y, 2021–2026 |
| Exit | `_replay_m5_from_signal` (M15, causal from H1 close) |
| Portfolio | `run_multi()` + `CFG_PROD` |
| MC | extended to 10k in `research/mtf_h4_h1_m5/backtest.py` |
| Tests | `tests/test_h4_structure.py` pattern |

---

## 4. Missing components (built in this sprint)

| Component | Module |
|---|---|
| H4 signal contract | `research/mtf_h4_h1_m5/contracts.py` |
| H4 context engine | `research/mtf_h4_h1_m5/h4_engine.py` |
| M5 execution rules | `research/mtf_h4_h1_m5/m5_engine.py` |
| WF + backtest harness | `research/mtf_h4_h1_m5/backtest.py` |
| Main runner | `apps/research_mtf_h4_h1_m5.py` |
| Alignment tests | `tests/test_mtf_h4_h1_m5.py` |

---

## 5. Existing assumptions

1. UTC open-time bars throughout.
2. H4 state available at `open + 4h` (`timeframe_utils`).
3. H1 label = triple barrier v1; **not changed** in this sprint.
4. Top gate fixed at **21%** (production) — no test-set threshold tuning.
5. OOS years = 2022–2026 (`TRUE_OOS`).
6. Starting equity **$280/year** isolated in research backtests.
7. M5 execution fill = **next M5 open** after rule fires (documented in `m5_engine.py`).
8. Immediate baseline = H1 close fill + M15 trail (production-faithful).

---

## 6. Leakage risks

| Risk | Severity | Mitigation |
|---|---|---|
| Naive H4 asof on bar open | **HIGH** | `available_at` join only; regression test |
| H4 vol terciles from full sample | MEDIUM | train-years-only per WF fold |
| M5 scan before H1 close | HIGH | `searchsorted(..., side='right')` |
| Legacy M5 trail from H1 open | HIGH | not used; M15 from H1 close |
| ctx_h4 duplicate + signal duplicate | MEDIUM | Experiment B excludes raw ctx |

See `docs/research/mtf_leakage_audit.md` after run.

---

## 7. Proposed integration points

| Phase | Action |
|---|---|
| H4 engine | `h4_engine.build_h4_feature_frame` → deterministic state → `H4Signal` |
| H4→H1 | `attach_h4_signals()` per WF fold (train vol terciles) |
| H1 A | `H1_NATIVE` (6 features, no H4) |
| H1 B | `H1_NATIVE + h4_sig_*` (5 contract columns) |
| Prod ref | `PRODUCTION_FEAT7` (raw ctx_h4_swing_quality) |
| M5 | `decide_m5_execution()` strategies A–E |
| Backtest | `score_panel()` + `monte_carlo_10k()` |
| Output | `results/h4_h1_m5/` |

---

## 8. Comparison framework

| Track | Description |
|---|---|
| **A — h1_baseline** | H1 native only → immediate M15 P0 |
| **B — h4_h1** | H4 signal + H1 native → immediate M15 P0 |
| **C — h4_h1_m5** | B + M5 execution rule → adjusted fill |
| **production_feat7** | Current shipped feature set (reference) |

Primary question:

> Does deterministic H4 context improve H1 OOS, and does M5 timing improve further?

---

## 9. What we deliberately do NOT do

- No H4 ML model
- No M5 ML model
- No D1
- No TP/SL/sizing changes
- No production code changes until PROMOTE verdict

---

## 10. Related prior research

| Artifact | Finding |
|---|---|
| `feat7_vs_schema31` | Flat 31-feat H1 beats FEAT7 on DD; not hierarchical |
| `m5_full_execution` | M5 entry/trail **REJECT** vs M15 production |
| Sprint 54–55 | M5 **management** signals; production unchanged |

---

## 11. Phase 1 status

**COMPLETE.** Implementation continues in `apps/research_mtf_h4_h1_m5.py`.

See also: `docs/research/mtf_h4_h1_m5_audit.md` (prior broader audit).
