# H1 Reversal & Position Decision Engine — Repository Audit

**Date:** 2026-08-23  
**Sprint goal:** Causal H1 reversal signal + deterministic position decision layer  
**Constraint:** Do not merge LONG/SHORT models; do not redesign them unless bug found.

---

## 1. Diagnosis (before adding models)

| Hypothesis | Evidence from repo |
|------------|-------------------|
| **A. Missing market/reversal information** | Partially researched (Sprint 45). Signal often weak / mechanical. |
| **B. Missing position-state information** | MAE/MFE tracked in paper; rich path research in Sprint 39–55. |
| **C. Missing decision logic** | **Primary gap.** No HOLD/EXIT/WAIT engine in production. |
| **D. Missing exit management beyond trail** | Production exit = ATR trail only. Thesis/recovery **not shipped** (53–54). |

**Working conclusion:** The system can enter LONG/SHORT but has **no position-aware decision layer**. Trail manages stop geometry, not “is my thesis still valid?”. Prior M5 thesis/recovery work concluded **do not ship** action overlays without better economics.

---

## 2. Current LONG / SHORT labels

| Item | Detail |
|------|--------|
| Strategy | `labels/strategies/triple_barrier.py` |
| Sides | Separate `long` / `short` datasets |
| Values | `1` = TP first, `-1` = SL first, `0` = TIMEOUT |
| Barriers | Entry = H1 close; ATR(14); TP **2.0×ATR**, SL **1.5×ATR** |
| Horizon | **8 H1 bars** |
| Same-bar both | SL wins (conservative) |

### What these labels are / are not

| Question | Answer |
|----------|--------|
| Continuation? | Indirect — TP before SL over 8 bars ≈ favorable path |
| Entry opportunity? | **Yes** — training target for entry models |
| Directional reversal while in trade? | **No** — labels are entry-bar outcomes, not in-position thesis failure |
| Exit condition? | **No** — not designed as management exit labels |

→ **Must define a separate reversal/thesis-failure label** for management research.

---

## 3. Current LONG / SHORT models

| Item | Detail |
|------|--------|
| Features | `PRIMARY_FEATURES` / `H1_NATIVE` — **6** (no `ctx_h4_*`) |
| Training | Separate LGBM per side (`train_frozen`) |
| Gate | Top **21%** by probability (research/production) |
| Caveat | Ternary labels fed to binary LGBM without `exclude_timeout` remap — keep frozen for comparability |
| Live | `production/live/inference.py` — rolling top-pct per side |

**Do not retrain for this sprint** unless a bug is found.

---

## 4. Current entry logic

```
H1 score (long & short) → best side → top 21%
  → PendingEntry (m15_pullback)
  → M15 pullback 0.15 ATR / 50% recovery (max 16 M15)
  → open (portfolio: max 5, block opposite, heat 3R)
```

Files: `production/paper/pipeline.py`, `production/live/entry_execution.py`, `research/mtf_h4_h1_m5/m5_engine.py`.

---

## 5. Current position lifecycle & exit

| State | Implementation |
|-------|----------------|
| FLAT | No open; may have pending M15 entry |
| LONG/SHORT | `OpenPosition` in `PortfolioState` |
| Exit | Initial SL 1.5 ATR; **TP off**; trail a0.25 / d0.08 on **M15 close**; TIMEOUT 192 M15 |
| MAE/MFE | `PaperBroker.mark_excursions` → `OpenPosition.mae/mfe` (price %) |

**No:** HOLD vs EXIT decision, WEAKENING state, reversal confirm, cooldown after exit (beyond portfolio heat).

---

## 6. H4 / M5 roles today

| Layer | Role now | Role target |
|-------|----------|-------------|
| H4 | Joined in live panel; **not** in primary features; meta gate OFF | Context only — never sole exit |
| M5 | Pullback **entry** + trail clock | Execution; tactical confirm researched separately |
| H1 | Entry ML only | Entry + reversal research + decision inputs |

---

## 7. Prior reversal / thesis research (reuse, do not duplicate)

| Sprint | Verdict / takeaway |
|--------|-------------------|
| **45** Reversal | Research-only; often weak / `NO_REVERSAL_SIGNAL` |
| **53** Thesis invalidation | Aspirational HOLD/REDUCE/EXIT — **not shipped** |
| **54** Recovery | Incremental signal underwater = mostly **time / adverse velocity**, not bounce — **do not ship** |
| **55** Price vs time | Attribution under water |

Path artifacts: `artifacts/pipeline_backtest/exit_structure/sprint52/sprint52_m5_path.parquet` (and sprint53/54 reports).

---

## 8. MAE / MFE infrastructure

| Source | Use |
|--------|------|
| Paper broker | Live/paper path MAE/MFE |
| Trail sims / exit grids | `mae_r`, `mfe_r` |
| Sprint 39–55 | PIT `mae_so_far_R`, `mfe_so_far_R`, recovery windows |

Reuse for: normal vs abnormal MAE, recovery vs deterioration states.

---

## 9. Calibration

- Package: `research/probability_calibration/`
- Production: **raw** LGBM proba (not calibrated in live gates)
- LONG vs SHORT proba **not** assumed comparable — calibrate separately if used in decision rules

---

## 10. Walk-forward / exit-only experiment hooks

| Asset | Path |
|-------|------|
| Entry panels | `research/mtf_h4_h1_m5/backtest.py::build_wf_entry_panel_feats` |
| Score + MC | `score_panel`, `monte_carlo_10k` |
| Exit grids | `apps/run_exit_engine_grid.py` |
| Path builders | Sprint 39/52 management path parquet |

**Exit-only rule:** freeze entries (top21 + M15 pullback); change exit/decision only.

---

## 11. Gaps (this sprint fills)

1. No position decision engine (`ENTER/HOLD/EXIT/WAIT`)
2. No position-aware reversal definition separate from TB entry labels
3. No WARNING → CONFIRMED two-stage state in production
4. No trade event log with `decision` / `decision_reason`
5. Trail ≠ thesis validity

---

## 12. Leakage risks (reversal labels)

1. Future path only in **label construction**, never in X
2. Do not broadcast trade-final `p0_R` / MAE to every bar (Sprint 54 lesson)
3. Do not tune thresholds on full OOS then claim OOS
4. Management horizon (trail CAP 48 H1) ≠ label horizon (TB 8) — align clocks explicitly
5. H4/D1 joins must remain asof completed bars

---

## 13. Recommended integration

```
research/h1_reversal/
  labels.py          # candidate A/B/C definitions (causal)
  baselines.py       # deterministic WARNING/CONFIRMED
  decision.py        # evaluate(...) → actions
  path_analysis.py   # MAE/MFE / prob-around-exit diagnostics

apps/research_h1_reversal.py   # WF exit overlays + report

production/ (later, after PROMOTE):
  decision engine wired behind flag; default OFF
```

Do **not** add reversal features into LONG/SHORT primary models by default.

---

## 14. Architecture target (research)

```
H4 context (optional evidence)
        ↓
H1: P(Long) | P(Short) | reversal state (research)
        ↓
Position state (FLAT / LONG / SHORT [+ WEAKENING])
        ↓
Decision Engine → ENTER_* | HOLD_* | EXIT_* | WAIT
        ↓
M5 execution / trail (unchanged unless exit overlay wins OOS)
```

---

## 15. Phase gate

Phase 1 complete. Next:

1. Formalize candidate reversal labels (A barrier / B thesis / C MAE-empirical)
2. Deterministic baselines + DecisionEngine contract + tests
3. Exit comparison vs P0 trail (identical entries)
4. Report: premature exit, missed reversal, ΔR, ΔDD, MC
