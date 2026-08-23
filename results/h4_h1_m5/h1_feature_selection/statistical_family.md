
### Rolling statistical family (verified in code)

| Feature | Formula | Role |
|---------|---------|------|
| `rolling_zscore` | (ret − mean20) / std20 | standardized current return |
| `rolling_mean_distance` | (ret − mean20) / std20 | **identical formula** to zscore |
| `rolling_percentile` | P(window ret ≤ current) | rank of current ret ∈ [0,1] |
| `rolling_rank` | midrank(current)/n | near-equivalent to percentile |
| `rolling_quantile` | quantile(0.75) of window rets | **distribution LEVEL**, not current rank |
| `rolling_std` | std20(ret) | vol of returns |

Conclusion: `rolling_quantile` (in CORE) is **not** interchangeable with percentile/rank.
`rolling_zscore` ≈ `rolling_mean_distance` → treat as redundant pair.
