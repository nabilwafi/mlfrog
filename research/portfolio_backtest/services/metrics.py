"""Portfolio performance metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe(v: float) -> float:
    if v != v or v == float("inf") or v == float("-inf"):
        return float("nan")
    return float(v)


def streaks(wins: np.ndarray) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
    return max_w, max_l


def sharpe_from_returns(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(len(r)))


def sortino_from_returns(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    down = r[r < 0]
    if len(down) == 0 or np.std(down, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(down, ddof=1) * np.sqrt(len(r)))


def profit_factor_pnl(pnl: np.ndarray) -> float:
    p = np.asarray(pnl, dtype=float)
    p = p[np.isfinite(p)]
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def max_dd_from_equity(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, np.nan)
    return float(np.nanmax(dd))


def cagr(start: float, end: float, years: float) -> float:
    if start <= 0 or years <= 0 or end <= 0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def compute_metrics(
    trade_log: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    starting_equity: float,
) -> dict[str, Any]:
    taken = trade_log.loc[~trade_log["skipped"].astype(bool)] if "skipped" in trade_log.columns else trade_log
    if taken.empty:
        nan = float("nan")
        return {
            "final_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else starting_equity,
            "total_return_pct": 0.0,
            "cagr": nan,
            "max_drawdown": 0.0,
            "profit_factor": nan,
            "sharpe": nan,
            "sortino": nan,
            "calmar": nan,
            "recovery_factor": nan,
            "avg_monthly_return": nan,
            "worst_month": nan,
            "best_month": nan,
            "avg_trade": nan,
            "median_trade": nan,
            "largest_win": nan,
            "largest_loss": nan,
            "longest_win_streak": 0,
            "longest_losing_streak": 0,
            "trades": 0,
            "trades_per_month": 0.0,
            "win_rate": nan,
            "avg_r": nan,
            "expectancy": nan,
            "n_skipped": int(trade_log["skipped"].sum()) if "skipped" in trade_log.columns else 0,
        }

    pnl = taken["pnl"].to_numpy(dtype=float)
    eq = equity_curve["equity"].to_numpy(dtype=float)
    final = float(eq[-1])
    total_ret = final / starting_equity - 1.0
    # years from first to last timestamp
    ts = pd.to_datetime(taken["timestamp"], utc=True)
    years = max((ts.max() - ts.min()).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    mdd = max_dd_from_equity(eq)
    pf = profit_factor_pnl(pnl)
    # trade-level sharpe on pnl / equity_before (return on capital per trade)
    trade_ret = (taken["pnl"] / taken["equity_before"].replace(0, np.nan)).to_numpy(dtype=float)
    sh = sharpe_from_returns(trade_ret)
    so = sortino_from_returns(trade_ret)
    cagr_v = cagr(starting_equity, final, years)
    calmar = cagr_v / mdd if mdd and mdd == mdd and mdd > 0 else float("nan")
    recovery = (final - starting_equity) / (mdd * starting_equity) if mdd and mdd > 0 else float("nan")

    # monthly returns from equity curve
    curve = equity_curve.dropna(subset=["timestamp"]).copy()
    monthly = pd.Series(dtype=float)
    if not curve.empty:
        curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
        curve = curve.set_index("timestamp").sort_index()
        monthly_eq = curve["equity"].resample("ME").last().dropna()
        monthly = monthly_eq.pct_change().dropna()

    wins = pnl > 0
    lw, ll = streaks(wins)
    # Average R: pnl / risk_amount approx equity_before * risk — use |median loss| as 1R proxy
    med_loss = float(np.median(np.abs(pnl[pnl < 0]))) if (pnl < 0).any() else float("nan")
    avg_r = float(np.mean(pnl) / med_loss) if med_loss and med_loss == med_loss and med_loss > 0 else float("nan")

    n_months = max(len(monthly), 1) if len(monthly) else max(years * 12, 1)

    return {
        "final_equity": final,
        "total_return_pct": float(total_ret * 100.0),
        "cagr": _safe(cagr_v),
        "max_drawdown": _safe(mdd),
        "profit_factor": _safe(pf),
        "sharpe": _safe(sh),
        "sortino": _safe(so),
        "calmar": _safe(calmar),
        "recovery_factor": _safe(recovery),
        "avg_monthly_return": float(monthly.mean()) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
        "avg_trade": float(np.mean(pnl)),
        "median_trade": float(np.median(pnl)),
        "largest_win": float(np.max(pnl)),
        "largest_loss": float(np.min(pnl)),
        "longest_win_streak": int(lw),
        "longest_losing_streak": int(ll),
        "trades": int(len(taken)),
        "trades_per_month": float(len(taken) / n_months),
        "win_rate": float(np.mean(wins)),
        "avg_r": _safe(avg_r),
        "expectancy": float(np.mean(pnl)),
        "n_skipped": int(trade_log["skipped"].sum()) if "skipped" in trade_log.columns else 0,
        "years": float(years),
    }


def yearly_report(trade_log: pd.DataFrame, *, starting_equity: float) -> pd.DataFrame:
    taken = trade_log.loc[~trade_log["skipped"].astype(bool)].copy() if "skipped" in trade_log.columns else trade_log.copy()
    if taken.empty:
        return pd.DataFrame()
    rows = []
    eq = float(starting_equity)
    for year, g in taken.groupby("valid_year", sort=True):
        start = eq
        peak = start
        max_dd = 0.0
        for pnl in g["pnl"].to_numpy(dtype=float):
            eq = eq + float(pnl)
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        pn = g["pnl"].to_numpy(dtype=float)
        rows.append(
            {
                "valid_year": int(year),
                "starting_equity": start,
                "ending_equity": eq,
                "return_pct": (eq / start - 1.0) * 100.0 if start > 0 else float("nan"),
                "drawdown": max_dd,
                "trades": int(len(g)),
                "win_rate": float(np.mean(pn > 0)),
                "profit_factor": profit_factor_pnl(pn),
            }
        )
    return pd.DataFrame(rows)


def monthly_returns_table(equity_curve: pd.DataFrame) -> pd.DataFrame:
    curve = equity_curve.dropna(subset=["timestamp"]).copy()
    if curve.empty:
        return pd.DataFrame(columns=["year", "month", "return"])
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.set_index("timestamp").sort_index()
    m = curve["equity"].resample("ME").last().dropna().pct_change().dropna()
    out = m.reset_index()
    out.columns = ["timestamp", "return"]
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    return out[["year", "month", "return", "timestamp"]]
