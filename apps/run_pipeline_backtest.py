"""One-shot Sprint 27 scored-panel backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from pipeline.backtest.adapter import run_meta_expected_r_backtest
from research.portfolio_heat import STARTING_EQUITY


def main() -> None:
    panel_path = Path("artifacts/research/confidence_layer/confidence_panel.parquet")
    panel = pd.read_parquet(panel_path)
    n_meta = int((panel["meta_proba"].astype(float) >= 0.45).sum())
    print(f"panel rows={len(panel)} starting_equity={STARTING_EQUITY}")
    print(f"meta>=0.45 candidates={n_meta}")

    traded, equity, metrics = run_meta_expected_r_backtest(panel, starting_equity=STARTING_EQUITY)
    print("--- Sprint27 Meta+expected_r+heat1R ---")
    for k in (
        "n_trades",
        "final_equity",
        "total_return",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "policy",
        "meta_threshold",
        "confidence_enabled",
    ):
        v = metrics.get(k)
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if not traded.empty:
        print(
            "risk_pct mean/min/max:",
            float(traded["risk_pct"].mean()),
            float(traded["risk_pct"].min()),
            float(traded["risk_pct"].max()),
        )
        print("lot mean:", float(traded["lot"].mean()))
        print("date range:", traded["timestamp"].min(), "->", traded["timestamp"].max())

    out = Path("artifacts/pipeline_backtest")
    out.mkdir(parents=True, exist_ok=True)
    traded.to_parquet(out / "meta_expected_r_trades.parquet", index=False)
    equity.to_parquet(out / "meta_expected_r_equity.parquet", index=False)
    (out / "meta_expected_r_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float),
        encoding="utf-8",
    )
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
