# Sprint 27 policy experiment — steps 1–3

**Paper path (current):**

```text
Primary candidate
  → Portfolio: duplicate / single position / heat −1R
  → Risk: option C sizing from meta edge (NOT flat 1%)
  → PaperBroker
```

| Step | Change | Status |
|------|--------|--------|
| 1 | Single position (no concurrent / opposite) | ON |
| 2 | Confidence gate OFF | `CONFIDENCE_ENABLED=False` |
| 3 | Meta = edge for sizing only (not skip gate) | `META_AS_GATE=False` |

Sizing (option C), TP/SL still 2.0 / 1.5 ATR:

```text
expected_r = meta * (RR+1) - 1
risk_pct   = clip(0.01 * expected_r / 0.25, 0.0025, 0.015)
```

Heat still uses `RISK_BASE=1%` as 1R reference.

Restore old gates: set `META_AS_GATE=True` and/or `CONFIDENCE_ENABLED=True` in `production/__init__.py`.
