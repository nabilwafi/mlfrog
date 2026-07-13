# Gate 4 — Combined Verdict (2024–2025 + sealed-2026)

## Preflight

- `lightgbm_long_v3` @ thr=0.51 confirmed
- Features `xauusd_h1_h4_d1_features_v3.parquet` cover both windows
- Methodology identical to Gate 4.5 sealed audit
- Detail: `gate4_5_blocked_trades_audit_2024_2025.md`

## Combined summary table

| period | z_block | n_blocked | winrate_hipotetis | net_pnl_hipotetis | %_would_be_loss |
|---|---:|---:|---:|---:|---:|
| 2024-2025 | 3.5 | 35 | 0.519 | +691.59 | 37.1% |
| 2024-2025 | 4.0 | 23 | 0.400 | -47.14 | 39.1% |
| sealed-2026 | 3.5 | 13 | 0.455 | +66.55 | 46.2% |
| sealed-2026 | 4.0 | 7 | 0.143 | -476.00 | 85.7% |
| combined | 3.5 | 48 | 0.500 | +758.14 | 39.6% |
| combined | 4.0 | 30 | 0.318 | -523.14 | 50.0% |

## 1. Is the 2026 pattern consistent in 2024–2025?

**Ranking-nya konsisten; magnitude-nya tidak identik.**

- 2024–2025: Z=4.0 blocked-if-executed **-47.14** (protektif, lemah) vs Z=3.5 **+691.59** (opportunity cost).
- Arah sama dengan sealed-2026 (4.0 protektif, 3.5 tidak) — bukan kebetulan n=7.

## 2. Better threshold across periods?

- Combined: Z=3.5 → **+758.14 USD**; Z=4.0 → **-523.14 USD**.
- Recommended dual-period anchor: **Z_BLOCK=4.0**.
- On 2024–2025 alone, most protective z in scan is **3.9** (blocked pnl=-146.70). Jangan retune tanpa sealed mid-scan.

## 3. ADX H4 hypothesis (larger n)

| period | z_block | ΔADX_H4 (wrong−right) | n_wrong | n_right |
|---|---:|---:|---:|---:|
| 2024-2025 | 3.5 | -5.29 | 14 | 13 |
| 2024-2025 | 4.0 | -7.75 | 6 | 9 |
| sealed-2026 | 3.5 | +23.17 | — | — |
| sealed-2026 | 4.0 | +11.27 | — | — |

**Hipotesis ADX H4 tidak terkonfirmasi** di 2024–2025 (Δ terbalik vs sealed-2026). Jangan encode.

## 4. Final Gate 4 verdict → Gate 5?

**Gate 4 settled dengan Z_BLOCK=4.0. Siap lanjut Gate 5.**

- Combined blocked PnL: 3.5=+758.14, 4.0=-523.14.
- Freeze: W=4320, Z_REDUCE=2.0, Z_BLOCK=4.0 (ubah param di Gate 5, bukan di audit ini).

## Bottom line

- **Z_BLOCK rekomendasi gabungan: 4.0**
- Combined blocked PnL @3.5 = +758.14 USD; @4.0 = -523.14 USD
- Gate 5: YES — freeze Z_BLOCK=4.0 dan lanjut
