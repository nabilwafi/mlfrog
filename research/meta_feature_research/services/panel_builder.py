"""Build meta-feature research panel for top-3% candidates (pre-entry only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from market_context.services.context_join_service import ContextJoinService
from research.meta_feature_research.services.m15_builder import M15ContextBuilder

logger = logging.getLogger(__name__)

LEAKAGE_COLS = {
    "realized_return",
    "net_return",
    "meta_label",
    "y_true",
    "mae",
    "mfe",
    "holding_bars",
    "exit_reason",
    "future_return",
}

# Point-in-time / entry-known diagnostics only (from barrier setup, not path)
ENTRY_DIAG_OK = {
    "initial_rr",
    "distance_to_sl",
    "distance_to_tp",
    "atr_stop_size",
    "atr_entry",
}


def _session_flags(ts: pd.Series) -> pd.DataFrame:
    """UTC session buckets (research proxies)."""
    hour = pd.to_datetime(ts, utc=True).dt.hour
    # Approximate FX session hours in UTC
    asia = ((hour >= 0) & (hour < 7)).astype(float)
    london = ((hour >= 7) & (hour < 16)).astype(float)
    newyork = ((hour >= 12) & (hour < 21)).astype(float)
    overlap = ((hour >= 12) & (hour < 16)).astype(float)
    return pd.DataFrame(
        {
            "session_asia": asia.to_numpy(),
            "session_london": london.to_numpy(),
            "session_newyork": newyork.to_numpy(),
            "session_london_ny_overlap": overlap.to_numpy(),
            "hour_of_day": hour.astype(float).to_numpy(),
            "day_of_week": pd.to_datetime(ts, utc=True).dt.dayofweek.astype(float).to_numpy(),
            "month": pd.to_datetime(ts, utc=True).dt.month.astype(float).to_numpy(),
        }
    )


def _parse_entry_diagnostics(row: pd.Series) -> dict[str, float]:
    entry = float(row.get("entry_price", np.nan)) if "entry_price" in row.index else float("nan")
    meta = {}
    raw = row.get("metadata_json")
    if isinstance(raw, str) and raw:
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {}
    atr_v = float(meta.get("atr", np.nan))
    tp_m = float(meta.get("tp_atr_mult", np.nan))
    sl_m = float(meta.get("sl_atr_mult", np.nan))
    # Distances in return units from multipliers if prices missing
    if np.isfinite(sl_m) and np.isfinite(atr_v) and np.isfinite(entry) and entry != 0:
        dist_sl = abs(sl_m * atr_v / entry)
        dist_tp = abs(tp_m * atr_v / entry) if np.isfinite(tp_m) else float("nan")
    else:
        dist_sl = float("nan")
        dist_tp = float("nan")
    rr = float(dist_tp / dist_sl) if dist_sl and dist_sl == dist_sl and dist_sl > 0 else float("nan")
    return {
        "atr_entry": atr_v,
        "atr_stop_size": sl_m if np.isfinite(sl_m) else float("nan"),
        "distance_to_sl": dist_sl,
        "distance_to_tp": dist_tp,
        "initial_rr": rr,
    }


class MetaFeaturePanelBuilder:
    def build(
        self,
        *,
        candidates: pd.DataFrame,
        h1_features_path: Path,
        h4_structure_path: Path,
        m15_candles_path: Path | None,
        percentile: float = 0.03,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        c = candidates.copy()
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
        c = c.loc[c["percentile"] == percentile].copy()
        if c.empty:
            raise ValueError(f"no candidates at percentile={percentile}")

        # Deduplicate: same timestamp may appear once per side
        # Keep window/side grain
        base = c[
            [
                col
                for col in (
                    "side",
                    "window",
                    "valid_year",
                    "timestamp",
                    "y_prob_raw",
                    "y_prob",
                    "meta_label",
                    "entry_price",
                    "metadata_json",
                )
                if col in c.columns
            ]
        ].copy()
        if "y_prob_raw" not in base.columns and "y_prob" in base.columns:
            base["y_prob_raw"] = base["y_prob"]

        # --- Group A: primary model derived (pre-entry) ---
        base["raw_probability"] = base["y_prob_raw"].astype(float)
        base["side_long"] = (base["side"].astype(str).str.lower() == "long").astype(float)
        # Rank / percentile within window+side (among candidates already top 3%,
        # recompute among ALL validation window? User said probability_rank on candidates.
        # For meta on candidates, rank within window universe of candidates is informative.
        base["probability_rank"] = base.groupby(["side", "window"])["raw_probability"].rank(
            ascending=False, method="average"
        )
        base["probability_percentile"] = base.groupby(["side", "window"])[
            "raw_probability"
        ].rank(pct=True)
        # Margin vs weakest selected trade in window (cutoff)
        cutoff = base.groupby(["side", "window"])["raw_probability"].transform("min")
        base["probability_margin"] = base["raw_probability"] - cutoff
        # Classification model has no predicted_return — omit fabricated column

        groups: dict[str, list[str]] = {
            "A_primary": [
                "raw_probability",
                "probability_rank",
                "probability_percentile",
                "side_long",
                "probability_margin",
            ],
            "B_h1": [],
            "C_h4": [],
            "D_m15": [],
            "E_session": [],
            "F_entry_diag": [],
        }

        # --- Group B: H1 ---
        h1 = pd.read_parquet(h1_features_path)
        h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
        h1_cols = [x for x in h1.columns if x != "timestamp"]
        base = base.merge(h1[["timestamp", *h1_cols]], on="timestamp", how="left")
        groups["B_h1"] = h1_cols

        # --- Group C: H4 structure (already H1-aligned in structure_features.parquet) ---
        h4 = pd.read_parquet(h4_structure_path)
        h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
        h4_cols = [x for x in h4.columns if x.startswith("ctx_h4_")]
        base = base.merge(h4[["timestamp", *h4_cols]].drop_duplicates("timestamp"), on="timestamp", how="left")
        groups["C_h4"] = h4_cols

        # --- Group D: M15 ---
        if m15_candles_path is not None and m15_candles_path.is_file():
            m15_raw = pd.read_parquet(m15_candles_path)
            m15_feat = M15ContextBuilder().build(m15_raw)
            m15_cols = [x for x in m15_feat.columns if x.startswith("ctx_m15_")]
            joined_m15 = ContextJoinService().join(
                base["timestamp"],
                m15_feat,
                context_timeframe="M15",
                context_cols=m15_cols,
            )
            for col in m15_cols:
                base[col] = joined_m15[col].to_numpy()
            groups["D_m15"] = m15_cols
            logger.info("Attached M15 research features | n=%s", len(m15_cols))
        else:
            logger.warning("M15 candles missing — Group D skipped")

        # --- Group E: session ---
        sess = _session_flags(base["timestamp"])
        for col in sess.columns:
            base[col] = sess[col].to_numpy()
        groups["E_session"] = list(sess.columns)

        # --- Group F: entry-known diagnostics only ---
        diag_rows = [_parse_entry_diagnostics(row) for _, row in base.iterrows()]
        diag = pd.DataFrame(diag_rows)
        for col in diag.columns:
            base[col] = diag[col].to_numpy()
        groups["F_entry_diag"] = list(diag.columns)

        # Ensure meta_label present
        if "meta_label" not in base.columns:
            raise ValueError("candidates missing meta_label")

        n_nan = int(base[groups["B_h1"]].isna().any(axis=1).sum()) if groups["B_h1"] else 0
        logger.info(
            "Meta feature panel | rows=%s h1_nan_rows=%s groups=%s",
            len(base),
            n_nan,
            {k: len(v) for k, v in groups.items()},
        )
        return base, groups
