"""
v3 feature factory — rebuild feature math here; legacy features/ removed.

Settled contract (when reimplemented):
  Input Date = MT5 bar OPEN time. Output Date = H1 CLOSE time.
"""

from __future__ import annotations

import pandas as pd


def build_v3_features(h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "build_v3_features: reimplement under mlfrog.features (legacy features/ wiped)"
    )


rebuild_features_v3 = build_v3_features
