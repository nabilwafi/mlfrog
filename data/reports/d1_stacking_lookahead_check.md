# D1 Stacking Lookahead Check

Rule: H1 row at close-time T may only use D1 bars whose close-time <= T (merge_asof direction=backward on close-time indices).

D1 Date in native features is already OPEN+1d (close time).
H1 Date in feature tables is OPEN+1h (close time).

## Sample 1
- H1 Date (close)=2025-08-07 16:00:00+00:00
- Attached d1_short_proba=0.514773
- Last D1 close <= T: 2025-08-07 00:00:00+00:00 proba=0.514773
- Next D1 close > T (MUST NOT be used): 2025-08-08 00:00:00+00:00
- Match last usable: **PASS**

## Sample 2
- H1 Date (close)=2025-04-14 19:00:00+00:00
- Attached d1_short_proba=0.514773
- Last D1 close <= T: 2025-04-12 00:00:00+00:00 proba=0.514773
- Next D1 close > T (MUST NOT be used): 2025-04-15 00:00:00+00:00
- Match last usable: **PASS**

## Sample 3
- H1 Date (close)=2024-09-04 06:00:00+00:00
- Attached d1_short_proba=0.514773
- Last D1 close <= T: 2024-09-04 00:00:00+00:00 proba=0.514773
- Next D1 close > T (MUST NOT be used): 2024-09-05 00:00:00+00:00
- Match last usable: **PASS**

## Sample 4
- H1 Date (close)=2024-10-08 20:00:00+00:00
- Attached d1_short_proba=0.514773
- Last D1 close <= T: 2024-10-08 00:00:00+00:00 proba=0.514773
- Next D1 close > T (MUST NOT be used): 2024-10-09 00:00:00+00:00
- Match last usable: **PASS**

## Sample 5
- H1 Date (close)=2026-02-20 11:00:00+00:00
- Attached d1_short_proba=0.514773
- Last D1 close <= T: 2026-02-20 00:00:00+00:00 proba=0.514773
- Next D1 close > T (MUST NOT be used): 2026-02-21 00:00:00+00:00
- Match last usable: **PASS**
