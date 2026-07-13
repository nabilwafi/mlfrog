"""
Build feature table v2 = classic v1 features + SMC (H1 + H4 BOS).

Does NOT overwrite v1 parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ta.volatility import AverageTrueRange

from features.feature_engineering import load_ohlc_csv
from features.smc import compute_h4_bos_features, compute_smc_features, write_smc_sample_validation

SMC_FEATURE_COLS = [
    "dist_to_last_swing_high_h1",
    "dist_to_last_swing_low_h1",
    "bars_since_last_swing_high",
    "bars_since_last_swing_low",
    "bos_bullish_h1",
    "bos_bearish_h1",
    "choch_bullish_h1",
    "choch_bearish_h1",
    "dist_to_bullish_ob_h1",
    "dist_to_bearish_ob_h1",
    "dist_to_nearest_fvg_bullish",
    "dist_to_nearest_fvg_bearish",
    "bos_bullish_h4",
    "bos_bearish_h4",
]


def build_features_v2(
    *,
    v1_path: Path,
    h1_raw: Path,
    h4_raw: Path,
    out_path: Path,
    sample_report: Path,
) -> pd.DataFrame:
    v1 = pd.read_parquet(v1_path)
    v1["Date"] = pd.to_datetime(v1["Date"], utc=True)

    h1 = load_ohlc_csv(h1_raw)
    h1["Date"] = h1["Date"] + pd.Timedelta(hours=1)  # open -> close time
    h4 = load_ohlc_csv(h4_raw)
    h4["Date"] = h4["Date"] + pd.Timedelta(hours=4)

    # ATR on raw TF frames for SMC
    h1["atr_h1"] = AverageTrueRange(
        high=h1["High"], low=h1["Low"], close=h1["Close"], window=14, fillna=False
    ).average_true_range()
    h4["atr_h4"] = AverageTrueRange(
        high=h4["High"], low=h4["Low"], close=h4["Close"], window=14, fillna=False
    ).average_true_range()

    print("Computing H1 SMC features...")
    smc_h1 = compute_smc_features(h1, atr_col="atr_h1")
    write_smc_sample_validation(h1, smc_h1, sample_report)
    print(f"Wrote SMC sample validation -> {sample_report}")

    print("Computing H4 BOS features...")
    bos_h4 = compute_h4_bos_features(h4, atr_col="atr_h4")

    # Attach SMC to v1 on Date (both close-time)
    smc_cols = ["Date"] + SMC_FEATURE_COLS[:-2]  # without h4 bos yet
    # rename from compute_smc_features output - it uses h1 names already
    keep_h1 = ["Date"] + [
        c for c in SMC_FEATURE_COLS if c not in ("bos_bullish_h4", "bos_bearish_h4")
    ]
    smc_h1_keep = smc_h1[keep_h1].copy()

    merged = v1.merge(smc_h1_keep, on="Date", how="left")
    merged = pd.merge_asof(
        merged.sort_values("Date"),
        bos_h4.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Drop warm-up / incomplete SMC rows (sentinel-only early rows still ok;
    # require atr-based distances not all missing from merge failures)
    before = len(merged)
    merged = merged.dropna(subset=["bos_bullish_h4", "bos_bearish_h4"]).reset_index(drop=True)
    # Early rows may have sentinel -1 which is fine; drop if classic feats nan already gone in v1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Saved v2 features -> {out_path} | rows={len(merged)} (from {before})")
    print(f"Date range: {merged['Date'].min()} -> {merged['Date'].max()}")
    return merged


def main() -> None:
    base = Path("data")
    build_features_v2(
        v1_path=base / "features/xauusd_h1_h4_d1_features.parquet",
        h1_raw=base / "raw/XAUUSD_H1.csv",
        h4_raw=base / "raw/XAUUSD_H4.csv",
        out_path=base / "features/xauusd_h1_h4_d1_features_v2.parquet",
        sample_report=base / "reports/smc_feature_validation_samples.md",
    )


if __name__ == "__main__":
    main()
