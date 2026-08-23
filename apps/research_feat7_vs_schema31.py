"""Compare FEAT7 (production primary) vs 8-bucket schema-31 entry models.

Same stack otherwise: rolling WF 5y/1y/1y, top 21% gate, P0 trail on M15 close,
parallel max_open=5 / heat 3R / lot 0.01 / $280 isolated per calendar year.

  python apps/research_feat7_vs_schema31.py

Research only. Production unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_m5_full_execution import _hour_unit, _replay_m5_from_signal
from apps.research_sprint39_exit_state import session_of_hour
from apps.research_sprint40_iter1_time_aware_exit import CFG_PROD, STARTING
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import P0, _mc
from apps.run_exit_engine_grid import CAP, FEAT7, Paths
from apps.run_rolling_walkforward import (
    build_test_entries,
    build_windows,
    regime_thresholds,
    train_frozen,
    _slice_year,
    _slice_years,
)
from market_context.builders.h4_structure_builder import H4StructureBuilder
from market_context.services.context_join_service import ContextJoinService
from simulation.wf.sim import _load_side, load_h1, prepare_market, wilder_atr

OUT = _ROOT / "artifacts/pipeline_backtest/rolling_wf/feat7_vs_schema31"
FEAT7_PANEL = (
    _ROOT
    / "artifacts/pipeline_backtest/rolling_wf/sprint38_prod_yearly/entry_panel_top21.parquet"
)
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
H4_PATH = _ROOT / "artifacts/raw/XAUUSD/H4/data.parquet"
M15_PATH = _ROOT / "artifacts/raw/XAUUSD/M15/data.parquet"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
TOP_PCT = 0.21
ORDER = ("FEAT7", "SCHEMA31")

# User taxonomy → enriched column names (31 features)
SCHEMA31: list[str] = [
    "hour_sin",
    "hour_cos",
    "session_code",
    "ema_cross_distance",
    "ema20_slope",
    "ema50_slope",
    "ema_trend_duration",
    "atr_percent",
    "atr_percentile_252",
    "ctx_h4_expansion",
    "atr_ratio",
    "rsi_percentile",
    "rsi_slope",
    "macd_normalized",
    "close_position_20",
    "distance_high_20_atr",
    "distance_low_20_atr",
    "structure_direction",
    "swing_high_distance_atr",
    "swing_low_distance_atr",
    "structure_break",
    "structure_break_age",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_atr",
    "ctx_h4_trend_direction",
    "ctx_h4_trend_strength",
    "ctx_h4_swing_quality",
    "h4_distance_to_swing_high_atr",
    "h4_distance_to_swing_low_atr",
]

SESSION_CODE = {"ASIA": 0.0, "LONDON": 1.0, "LONDON_NY": 2.0, "NY": 3.0, "OFF": 4.0}


def _safe_div(a, b):
    b = np.where(np.abs(b) < 1e-12, np.nan, b)
    return a / b


def _h1_structure(h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.sort_values("timestamp").reset_index(drop=True).copy()
    high = h["high"].astype(float)
    low = h["low"].astype(float)
    close = h["close"].astype(float)
    atr = h["atr"].astype(float)
    swing_n = 5
    swing_high = high.rolling(swing_n, min_periods=swing_n).max()
    swing_low = low.rolling(swing_n, min_periods=swing_n).min()
    prev_sh = swing_high.shift(1)
    prev_sl = swing_low.shift(1)
    older_sh = swing_high.shift(swing_n)
    older_sl = swing_low.shift(swing_n)
    bull = (close > prev_sh).astype(float)
    bear = (close < prev_sl).astype(float)
    br = np.where(bull > 0.5, 1.0, np.where(bear > 0.5, -1.0, 0.0))
    ages = np.zeros(len(h), dtype=float)
    age = 0
    for i in range(len(h)):
        if br[i] != 0:
            age = 0
        else:
            age += 1
        ages[i] = float(age)
    hh = (swing_high > older_sh).astype(float)
    ll = (swing_low < older_sl).astype(float)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["timestamp"], utc=True),
            "close_position_20": _safe_div(
                close - low.rolling(20, min_periods=20).min(),
                high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min(),
            ),
            "distance_high_20_atr": _safe_div(
                high.rolling(20, min_periods=20).max() - close, atr
            ),
            "distance_low_20_atr": _safe_div(
                close - low.rolling(20, min_periods=20).min(), atr
            ),
            "swing_high_distance_atr": _safe_div(close - swing_high, atr),
            "swing_low_distance_atr": _safe_div(close - swing_low, atr),
            "structure_direction": hh - ll,
            "structure_break": br,
            "structure_break_age": ages,
        }
    )
    return out


def _h4_distances() -> pd.DataFrame:
    h4 = pd.read_parquet(H4_PATH)
    h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
    built, _ = H4StructureBuilder().build(h4)
    return built[
        [
            "timestamp",
            "ctx_h4_swing_high_distance_atr",
            "ctx_h4_swing_low_distance_atr",
        ]
    ].rename(
        columns={
            "ctx_h4_swing_high_distance_atr": "h4_distance_to_swing_high_atr",
            "ctx_h4_swing_low_distance_atr": "h4_distance_to_swing_low_atr",
        }
    )


def enrich_side(df: pd.DataFrame, h1_feats: pd.DataFrame, h4_feats: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp")
    d["session_code"] = d["timestamp"].dt.hour.map(
        lambda h: SESSION_CODE.get(session_of_hour(int(h)), 4.0)
    )
    d["atr_ratio"] = d["atr_percent"].astype(float) / 100.0
    d["body_ratio"] = d["body_percent"].astype(float) / 100.0
    d["upper_wick_ratio"] = d["upper_wick_percent"].astype(float) / 100.0
    d["lower_wick_ratio"] = d["lower_wick_percent"].astype(float) / 100.0
    d["range_atr"] = _safe_div(d["range_percent"].astype(float), d["atr_percent"].astype(float))
    d["rsi_slope"] = d.groupby("side")["rsi_percentile"].diff(3) / 3.0
    m = d.merge(h1_feats, on="timestamp", how="left")
    joined = ContextJoinService().join(
        m["timestamp"],
        h4_feats,
        context_timeframe="H4",
        context_cols=[
            "h4_distance_to_swing_high_atr",
            "h4_distance_to_swing_low_atr",
        ],
    )
    m = m.merge(
        joined[
            [
                "timestamp",
                "h4_distance_to_swing_high_atr",
                "h4_distance_to_swing_low_atr",
            ]
        ],
        on="timestamp",
        how="left",
    )
    return m


def load_enriched_sides(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_l = OUT / "enriched_long.parquet"
    cache_s = OUT / "enriched_short.parquet"
    if not force and cache_l.is_file() and cache_s.is_file():
        return pd.read_parquet(cache_l), pd.read_parquet(cache_s)
    h1 = load_h1(H1_PATH)
    h1["atr"] = wilder_atr(h1)
    h1_feats = _h1_structure(h1)
    h4_feats = _h4_distances()
    OUT.mkdir(parents=True, exist_ok=True)
    long_df = enrich_side(_load_side("long"), h1_feats, h4_feats)
    short_df = enrich_side(_load_side("short"), h1_feats, h4_feats)
    long_df.to_parquet(cache_l, index=False)
    short_df.to_parquet(cache_s, index=False)
    return long_df, short_df


def build_schema31_panel(long_df: pd.DataFrame, short_df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    cache = OUT / "entry_panel_schema31_top21.parquet"
    if not force and cache.is_file():
        print(f"  load cached {cache.name}")
        return pd.read_parquet(cache)
    missing = [f for f in SCHEMA31 if f not in long_df.columns or f not in short_df.columns]
    if missing:
        raise SystemExit(f"SCHEMA31 missing columns: {missing}")
    h1 = load_h1(H1_PATH)
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    panels = []
    for w in windows:
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", long_df), ("short", short_df)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _ = train_frozen(train, val, SCHEMA31)
            boosters[side] = booster
        if not boosters:
            continue
        e = build_test_entries(
            long_df=long_df,
            short_df=short_df,
            h1=h1,
            feat=SCHEMA31,
            window=w,
            boosters=boosters,
            top_pct=TOP_PCT,
            vol_lo=vol_lo,
            vol_hi=vol_hi,
        )
        if not e.empty:
            panels.append(e.assign(test_year=w.test_year))
        print(f"  schema31 window te{w.test_year}: {len(e)} entries")
    panel = pd.concat(panels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    panel.to_parquet(cache, index=False)
    return panel


def _pack_m15(p: Paths, m15: pd.DataFrame) -> dict:
    m15 = m15.copy()
    m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)
    m15 = m15.sort_values("timestamp").drop_duplicates("timestamp")
    idx = pd.DatetimeIndex(m15["timestamp"])
    ts = idx.asi8
    hour = _hour_unit(ts, idx)
    entry_idx = pd.DatetimeIndex(p.ts)
    entry_idx = entry_idx.tz_localize("UTC") if entry_idx.tz is None else entry_idx.tz_convert("UTC")
    return {
        "ts5": ts,
        "hour": hour,
        "n5": len(ts),
        "open": m15["open"].to_numpy(float),
        "high": m15["high"].to_numpy(float),
        "low": m15["low"].to_numpy(float),
        "close": m15["close"].to_numpy(float),
        "entry_ns": entry_idx.asi8,
    }


def _score_book(name: str, panel: pd.DataFrame, m15: pd.DataFrame) -> dict:
    p = Paths(panel, prepare_market(load_h1(H1_PATH)))
    pack15 = _pack_m15(p, m15)
    sim = _replay_m5_from_signal(p, pack15, act=P0["act"], dist=P0["dist"], fill="h1_close")
    yrows = [_year_row(y, _port(p, sim, y), sim) for y in YEARS]
    po = _pooled(yrows)
    po_oos = _pooled([r for r in yrows if r["year"] in TRUE_OOS])
    mc = _mc(po_oos["pnl"], name)
    return {"name": name, "yrows": yrows, "po": po, "po_oos": po_oos, "mc": mc}


def _rolling_wf(ports: dict[str, dict]) -> list[dict]:
    rows = []
    for name in ORDER:
        by = {int(r["year"]): r for r in ports[name]["yrows"]}
        for y in YEARS:
            if y not in by:
                continue
            prior = [by[yy] for yy in YEARS if yy < y and yy in by]
            test = by[y]
            train_avg = float(np.mean([r["avg_r"] for r in prior])) if prior else float("nan")
            train_pf = float(np.mean([r["pf"] for r in prior])) if prior else float("nan")
            rows.append(
                {
                    "book": name,
                    "test_year": y,
                    "n_train_years": len(prior),
                    "train_avg_r": train_avg,
                    "train_pf": train_pf,
                    "test_avg_r": test["avg_r"],
                    "test_pf": test["pf"],
                    "test_dd": test["dd"],
                    "test_ret": test["ret"],
                    "test_trades": test["trades"],
                    "delta_avg_r": test["avg_r"] - train_avg if prior else float("nan"),
                    "oos": y in TRUE_OOS,
                }
            )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not FEAT7_PANEL.is_file():
        raise SystemExit(f"missing FEAT7 panel: {FEAT7_PANEL}")

    force = "--force" in sys.argv
    if force:
        for p in (
            OUT / "enriched_long.parquet",
            OUT / "enriched_short.parquet",
            OUT / "entry_panel_schema31_top21.parquet",
        ):
            if p.is_file():
                p.unlink()

    print("Loading enriched datasets for SCHEMA31…")
    long_df, short_df = load_enriched_sides(force=force)
    print("Building SCHEMA31 WF entry panel (top 21%)…")
    schema_panel = build_schema31_panel(long_df, short_df, force=force)
    feat7_panel = pd.read_parquet(FEAT7_PANEL)
    feat7_panel["timestamp"] = pd.to_datetime(feat7_panel["timestamp"], utc=True)

    m15 = pd.read_parquet(M15_PATH)
    ports = {
        "FEAT7": _score_book("FEAT7", feat7_panel, m15),
        "SCHEMA31": _score_book("SCHEMA31", schema_panel, m15),
    }

    yearly_rows = []
    for name in ORDER:
        for yr in ports[name]["yrows"]:
            yearly_rows.append({"book": name, **{k: yr[k] for k in yr if k != "pnl"}})
        po = ports[name]["po"]
        mc = ports[name]["mc"]
        yearly_rows.append(
            {
                "book": name,
                "year": 0,
                "trades": po["trades"],
                "pf": po["pf"],
                "dd": po["dd"],
                "ret": po["ret"],
                "wr": po["wr"],
                "payoff": po["payoff"],
                "avg_r": po["avg_r"],
                "median_r": float("nan"),
                "max_loss_streak": po["max_loss_streak"],
                "blown": po["blown"],
                "prob_ruin": mc["prob_ruin"],
                "median_dd": mc["median_dd"],
                "p95_dd": mc["p95_dd"],
                "p99_dd": mc["p99_dd"],
                "worst_dd": mc["worst_dd"],
            }
        )
    pd.DataFrame(yearly_rows).to_csv(OUT / "yearly_and_mc.csv", index=False)

    wf = _rolling_wf(ports)
    pd.DataFrame(wf).to_csv(OUT / "rolling_wf.csv", index=False)

    oos_cmp = []
    for name in ORDER:
        po, mc = ports[name]["po_oos"], ports[name]["mc"]
        oos_cmp.append(
            {
                "book": name,
                "n_features": 7 if name == "FEAT7" else len(SCHEMA31),
                "pf": po["pf"],
                "dd": po["dd"],
                "avg_r": po["avg_r"],
                "wr": po["wr"],
                "ret": po["ret"],
                "trades": po["trades"],
                "prob_ruin": mc["prob_ruin"],
                "median_dd": mc["median_dd"],
                "p95_dd": mc["p95_dd"],
                "p99_dd": mc["p99_dd"],
                "worst_dd": mc["worst_dd"],
            }
        )
    pd.DataFrame(oos_cmp).to_csv(OUT / "oos_summary.csv", index=False)

    year_wide = []
    for y in YEARS:
        row = {"year": y}
        for name in ORDER:
            yr = next(r for r in ports[name]["yrows"] if r["year"] == y)
            row[f"{name}_trades"] = yr["trades"]
            row[f"{name}_PF"] = yr["pf"]
            row[f"{name}_DD"] = yr["dd"]
            row[f"{name}_AvgR"] = yr["avg_r"]
            row[f"{name}_Ret"] = yr["ret"]
            row[f"{name}_WR"] = yr["wr"]
        year_wide.append(row)
    pd.DataFrame(year_wide).to_csv(OUT / "yearly_wide.csv", index=False)

    def _yt(prefix: str) -> str:
        cols = ["year"] + [f"{n}_{prefix}" for n in ORDER]
        fmt = {"year": 0}
        for n in ORDER:
            fmt[f"{n}_{prefix}"] = 3 if prefix in ("DD", "AvgR", "WR") else 2 if prefix != "trades" else 0
        return _md(year_wide, cols, fmt)

    mapping_lines = [
        "| user name | column used |",
        "|---|---|",
        "| session | session_code (0=Asia … 4=Off) |",
        "| ema20_50_distance_atr | ema_cross_distance |",
        "| atr_expansion | ctx_h4_expansion |",
        "| atr_ratio | atr_percent / 100 |",
        "| rsi | rsi_percentile |",
        "| rsi_slope | Δ rsi_percentile / 3 bars |",
        "| close_position_20 / distance_*_20_atr | rolling 20 H1 OHLC |",
        "| structure_* | H1 swing/BOS (5-bar), causal |",
        "| body/wick/range ratios | body_percent etc /100; range_atr |",
        "| h4_distance_* | H4StructureBuilder + **ContextJoinService** (available_at) |",
        "",
        "Leakage audit: naive H4 asof on bar open inflated SCHEMA31 (PF>8); fixed with causal `available_at`.",
    ]

    delta_pf = oos_cmp[1]["pf"] - oos_cmp[0]["pf"]
    delta_dd = oos_cmp[1]["dd"] - oos_cmp[0]["dd"]
    winner = "SCHEMA31" if oos_cmp[1]["pf"] > oos_cmp[0]["pf"] and oos_cmp[1]["dd"] <= oos_cmp[0]["dd"] + 0.02 else (
        "FEAT7" if oos_cmp[0]["pf"] >= oos_cmp[1]["pf"] else "MIXED"
    )

    lines = [
        "# FEAT7 vs SCHEMA31 — entry model comparison",
        "",
        "Research only. **Exit unchanged:** P0 a0.25/d0.08 on **M15 close** (causal from H1 signal).",
        f"**Gate:** top {TOP_PCT:.0%} per WF test year. **Portfolio:** $280/year, lot 0.01, heat 3R, max_open 5.",
        "",
        "## Feature sets",
        "",
        f"- **FEAT7** (production frozen): {', '.join(FEAT7)}",
        f"- **SCHEMA31** (8-bucket taxonomy, {len(SCHEMA31)} cols): see mapping below",
        "",
        "## Column mapping (schema → repo)",
        "",
        *mapping_lines,
        "",
        f"Entry counts — FEAT7: {len(feat7_panel)}, SCHEMA31: {len(schema_panel)}",
        "",
        "## OOS 2022–2026 pooled + Monte Carlo (1000 shuffles, $280)",
        "",
        _md(
            oos_cmp,
            [
                "book",
                "n_features",
                "pf",
                "dd",
                "avg_r",
                "wr",
                "ret",
                "trades",
                "prob_ruin",
                "median_dd",
                "p95_dd",
                "p99_dd",
                "worst_dd",
            ],
            {
                "n_features": 0,
                "pf": 2,
                "dd": 3,
                "avg_r": 3,
                "wr": 3,
                "ret": 2,
                "trades": 0,
                "prob_ruin": 3,
                "median_dd": 3,
                "p95_dd": 3,
                "p99_dd": 3,
                "worst_dd": 3,
            },
        ),
        "",
        f"Δ PF (SCHEMA31 − FEAT7) OOS: **{delta_pf:+.2f}**. Δ max DD: **{delta_dd:+.3f}**. Headline: **{winner}** on OOS PF/DD.",
        "",
        "## Yearly PF",
        "",
        _yt("PF"),
        "",
        "## Yearly max DD",
        "",
        _yt("DD"),
        "",
        "## Yearly trades",
        "",
        _yt("trades"),
        "",
        "## Yearly AvgR",
        "",
        _yt("AvgR"),
        "",
        "## Yearly return (× $280)",
        "",
        _yt("Ret"),
        "",
        "## Yearly win rate",
        "",
        _yt("WR"),
        "",
        "## Rolling walk-forward (expanding prior calendar years → test year)",
        "",
        _md(
            [r for r in wf if r["oos"]],
            [
                "book",
                "test_year",
                "n_train_years",
                "train_avg_r",
                "test_avg_r",
                "delta_avg_r",
                "train_pf",
                "test_pf",
                "test_dd",
                "test_trades",
            ],
            {
                "test_year": 0,
                "n_train_years": 0,
                "train_avg_r": 3,
                "test_avg_r": 3,
                "delta_avg_r": 3,
                "train_pf": 2,
                "test_pf": 2,
                "test_dd": 3,
                "test_trades": 0,
            },
        ),
        "",
        "2021 = first WF test year (not true OOS). 2026 partial.",
        "",
        "## Production",
        "",
        "Entry: **FEAT7 unchanged** unless explicit promotion.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(
        json.dumps(
            {
                "production_changed": False,
                "winner_oos": winner,
                "feat7_oos": oos_cmp[0],
                "schema31_oos": oos_cmp[1],
                "delta_pf_oos": delta_pf,
                "delta_dd_oos": delta_dd,
                "schema31_features": SCHEMA31,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    for name in ORDER:
        po = ports[name]["po_oos"]
        print(name, {k: round(po[k], 3) if isinstance(po[k], float) else po[k] for k in ("pf", "dd", "avg_r", "trades")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
