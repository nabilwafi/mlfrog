# SDS — 500 trades/year, PF ≥ 4, DD ≤ 20%

Locked stack unless a phase explicitly changes it: FEAT7, H1, trail `a0.25_d0.08`,
`parallel_mo5_d0.0_cd0_h3.0`, lot 0.01, heat 3R, daily 1R ON, yearly isolated $280.

## Target

Per calendar year 2021–2026, isolated $280: trades ≥ 500, PF ≥ 4.0, maxDD ≤ 20%, no blow.

Non-goals: production/live, compounding, RL exits, chasing return.

## Why (measured)

P0 (daily ON, $280): WR 0.84, payoff 0.54, concat PF 2.80. `PF = odds × payoff`.
PF 4 at payoff 0.50 needs WR ≈ 0.89. H1 500 trades = top 8.5% → WR 0.817 (measured).
Density on H1 and quality fight each other. DD ≤ 20% is already met with daily ON
except 2026 sitting on the line (20.0%).

## Phases

| Phase | What | Gate | Kill |
|---|---|---|---|
| P0 | WR / payoff / PF / DD / exit-reason baseline | reproduce WR≈0.82 PF≈2.3 | — |
| P1 | Exit re-grid via `run_multi`, rank PF≥4 ∧ DD≤20 | ≥1 combo PF≥3.0 **all years** @ ~250 trades | skip combo if DD>20% |
| P2 | Relabel to realized-R under P1 winner; retrain WF | WR +≥3pt on top 5% H1, payoff not down | stop if PF flat |
| P3 | Feature expansion (HOLD) | WF PF +≥0.3 vs P2 | drop families that fail |
| P4 | M15 primary (fallback M30) | 500 trades ∧ PF≥3.5 ∧ DD≤20 in ≥5/6 years | abort if COST/1R > ~15% |
| P5 | Meta gate (only if P4 passes) | PF≥4 ∧ trades≥500 ∧ DD≤20 | — |

This run: **P0 + P1**. If P1 max min-year PF < 3.0 → still **P2** (exit is not the lever). **Hold at P3**.

## Guardrails

- Artifacts: `artifacts/pipeline_backtest/rolling_wf/sprint38_*/`
- Never overwrite `sprint37_multi_trade/entry_panel.parquet` or label v1
- Yearly table + PF always in `summary.md`
- Daily ON. Daily-OFF-only pass = fail
- Research only; no `production/` edits

## P3 — Feature expansion (HOLD — specify only, do not execute)

P3 is not “add features until PF=4”. It is a ranking experiment on the **P2 label**
(or TB v1 if P2 is skipped): same entries count (~250–290/year), same exit, same
sizing. Success = PF +≥0.3 vs the P2 (or P0) baseline at that trade count.
500 trades is **out of scope** for P3; that is P4 (M15).

### Why this layer, not more of P1

P1 showed payoff cannot be bought from the exit grid: looser trail raises payoff
to ~1.75 and drops WR to ~0.54 (min PF 1.52, DD ~58%). The remaining H1 lever is
**which bars we pick**. FEAT7 was selected against TB labels (`1` = TP 2R first).
If P2 realigned the target, the old 7 features may be the wrong 7.

### What is actually missing today

Current v2 dataset: 43 columns, all H1 price + 7 H4 context. Raw H1 already has
`tick_volume`, `spread`, `real_volume` and **none of them are features**.
M15/M5/D1 candles exist on disk and are not joined into the primary set.
`M15ContextBuilder` already exists (`research/meta_feature_research/services/m15_builder.py`)
but is research-only.

### Families, in order (each is a separate ablation)

1. **Volume / cost (highest suspicion)**
   From H1 `tick_volume` + `spread`: z-score 20, rank 252, volume/range, spread
   percentile 252, volume spike vs 20-bar MA.
   Hypothesis: low-liquidity / wide-spread hours are the −1R tails that keep
   payoff at 0.50.
   Kill: if FS never picks any of these in the top 10, drop the family.

2. **Session structure**
   Asia / London / NY / London–NY overlap flags, minutes-into-session,
   session-relative close position. `hour_sin/cos` is already in FEAT7 but does
   not encode session boundaries.
   Hypothesis: 2026 WR 0.90 vs 2024 WR 0.81 is partly session mix.

3. **M15 context on the H1 bar** (not M15 primary)
   Promote `M15ContextBuilder` via `ContextJoinService` (causal `available_at`):
   `ctx_m15_trend_strength`, `momentum`, `ema_alignment`, `swing_quality`,
   `rejection_strength`, `breakout_strength`, `volume_spike`.
   This is **not** P4. We still enter on H1 close. M15 only describes the hour.

4. **D1 context**
   `artifacts/raw/XAUUSD/D1/data.parquet` is unused. Distance to D1 range mid,
   D1 trend direction, close vs prior D1 high/low. Slow features; cheap to try.

5. **H4 swing distance**
   `research/h4_structure` already exists; `ctx_h4_swing_quality` is already in
   FEAT7. Add distance-to-swing-high/low in ATR, not another quality score.

### How to run (when unblocked)

- Join families onto the existing H1 v2 rows (do not rebuild candles).
- Forward-select with `apps/run_feature_selection_ablation.py` **on the P2 label**.
- New FEAT-N replaces FEAT7 only if the selected set beats FEAT7 on val logloss
  **and** on WF PF (same top 5%, same exit).
- One family at a time. If a family fails, it is deleted from the candidate
  pool, not carried “just in case”.
- Artifacts: `sprint38_p3_feat_<family>/`. Never overwrite FEAT7 models.

### What P3 will not do

- Will not create 500 trades (H1 ceiling).
- Will not add a 50th oscillator.
- Will not touch production.
- Will not proceed to P4 automatically; P4 is a new primary TF and needs a
  separate go.

### Gate / kill

- Gate: WF min-year PF +≥ 0.3 vs P2 (or P0 if P2 skipped), DD still ≤ 20%,
  trades not below P2 by >10%.
- Kill: all five families fail → freeze FEAT7, skip to a product decision
  (accept PF~2.2 @ ~250 trades, or start P4 M15 for density).
