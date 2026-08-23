# H1 Reversal & Position Decision Report

**Verdict:** `KEEP_TRAIL_ONLY`

## 1. Current architecture

H1 native 6 → top 21% → M15 pullback → M15 ATR trail (P0). No HOLD/EXIT/WAIT engine in production.
Audit: `docs/research/h1_reversal_audit.md`.

## 2–3. LONG/SHORT labels & models

Triple-barrier entry labels (TP/SL/TIMEOUT, horizon 8). **Not** reversal labels.
Separate LONG/SHORT LGBM; H1 native 6. Untouched this sprint.

## 4–5. Reversal definition & candidates

- **A** Opposite 1 ATR before favor 1 ATR within 8 H1
- **B** Adverse 1 ATR with MFE < 0.25 ATR within 8 H1
- **C** IS loser MAE quantiles (see mae_summary)

```json
{
  "A_long_oos_rate": 0.4365651250478109,
  "A_short_oos_rate": 0.4560062903454882,
  "B_long_oos_rate": 0.11673486933954372,
  "B_short_oos_rate": 0.12190817223663745,
  "definition_A": "opposite 1ATR before favor 1ATR within 8 H1",
  "definition_B": "adverse 1ATR with MFE<0.25ATR within 8 H1",
  "note": "Entry TB labels are NOT these; separate management labels."
}
```

## 6. Feature research

Sprints 45/53/54: do not ship bounce-based recovery exits; time/adverse velocity dominate.
Keep reversal feature research separate from LONG/SHORT primary.

## 7. Baseline rules

DecisionEngine: NORMAL → WEAKENING → CONFIRMED; no direct reverse; cooldown; no pyramid.
Path overlay (a priori): 2 bars with current_R ≤ -0.5.

```json
{
  "weak_r": -0.5,
  "confirm_bars": 2,
  "n": 5524,
  "early_exit_rate": 0.2986965966690804,
  "premature_exit_rate": 0.14500362056480812,
  "saved_rate": 0.15369297610427227,
  "p0_avg_r": 0.24753776879590267,
  "early_avg_r": 0.1346704617467276,
  "p0_pf": 2.259666665414525,
  "early_pf": 1.5934190765293428,
  "delta_avg_r": -0.11286730704917508,
  "delta_pf": -0.6662475888851822,
  "note": "Path from Sprint52 (FEAT7-era entries). Overlay is diagnostic vs that P0, not prod v6 panel.",
  "p0_mc_median_dd": 0.01601146363745805,
  "early_mc_median_dd": 0.02243262743512628,
  "p0_mc_ruin": 0.0,
  "early_mc_ruin": 0.0
}
```

## 8–10. WF / directions

```json
{
  "policy": "prod_v6_p0_trail",
  "pf": 1.6001587219925002,
  "dd": 0.2706574463928747,
  "avg_r": 0.07727717034356676,
  "trades": 4988
}
```

Path overlay uses Sprint52 path (legacy FEAT7 entries) — directional evidence only.

## 11–12. MAE/MFE

```json
{
  "oos_trades": 5524,
  "oos_winrate": 0.8122737146994932,
  "median_final_MAE": 0.3549297177851761,
  "median_final_MFE": 0.4246097359275762,
  "recover_after_mae_ge_0.50": 0.634978229317852,
  "n_first_deep_mae": 2756,
  "is_loser_mae_p50": 0.8812306397913839,
  "is_loser_mae_p75": 0.9681977397337213
}
```

## 13–15. Decision engine / exit / MC

Code: `research/h1_reversal/decision.py`. Tests: `tests/test_h1_reversal.py`.
Artifacts: `results/h4_h1_m5/h1_reversal/`.

## 16. Final recommendation

**KEEP_TRAIL_ONLY**

1–2. Reversal = A/B position-aware labels, not TB entry.
5. Premature early-exit rate ≈ 0.14500362056480812
6–8. ΔAvgR ≈ -0.11286730704917508; simple weak_r overlay does not replace trail.
12. Running trade → HOLD / WAIT_FOR_CONFIRMATION / EXIT_confirmed → FLAT (no flip).

Diagnosis: missing **decision logic** layer. DecisionEngine research-ready; keep P0 trail until
an overlay beats OOS on **prod v6** entries.

Seed=42. Production unchanged.
