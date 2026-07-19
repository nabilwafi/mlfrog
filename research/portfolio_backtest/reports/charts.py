"""Charts for portfolio backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioChartBuilder:
    def write_all(
        self,
        *,
        out_dir: Path,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        monthly: pd.DataFrame,
        mc_detail: pd.DataFrame | None,
        prefix: str,
    ) -> None:
        charts = out_dir / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        self._equity(charts, equity_curve, prefix)
        self._drawdown(charts, equity_curve, prefix)
        self._monthly_heatmap(charts, monthly, prefix)
        self._rolling_sharpe(charts, equity_curve, prefix)
        self._rolling_dd(charts, equity_curve, prefix)
        self._trade_dist(charts, trade_log, prefix)
        self._monthly_dist(charts, monthly, prefix)
        if mc_detail is not None and not mc_detail.empty:
            self._mc(charts, mc_detail, prefix)

    def _equity(self, charts: Path, curve: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_equity_curve.png"
        fig, ax = plt.subplots(figsize=(9, 4.5))
        c = curve.dropna(subset=["timestamp"]) if "timestamp" in curve.columns else curve
        if not c.empty and c["timestamp"].notna().any():
            ax.plot(pd.to_datetime(c["timestamp"]), c["equity"], color="#4C78A8")
        else:
            ax.plot(c["equity"].to_numpy(), color="#4C78A8")
        ax.set_title(f"{prefix} portfolio equity")
        ax.set_ylabel("equity ($)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _drawdown(self, charts: Path, curve: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_drawdown_curve.png"
        fig, ax = plt.subplots(figsize=(9, 3.5))
        c = curve.dropna(subset=["timestamp"]) if "timestamp" in curve.columns else curve
        if not c.empty and c["timestamp"].notna().any():
            ax.fill_between(pd.to_datetime(c["timestamp"]), 0, -c["drawdown"], color="#E45756", alpha=0.7)
        else:
            ax.fill_between(range(len(c)), 0, -c["drawdown"].to_numpy(), color="#E45756", alpha=0.7)
        ax.set_title(f"{prefix} drawdown")
        ax.set_ylabel("drawdown")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _monthly_heatmap(self, charts: Path, monthly: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_monthly_heatmap.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        if monthly.empty:
            ax.set_title("monthly empty")
        else:
            pivot = monthly.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
            im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(range(12))
            ax.set_xticklabels(list(range(1, 13)))
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{prefix} monthly returns")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _rolling_sharpe(self, charts: Path, curve: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_rolling_sharpe.png"
        fig, ax = plt.subplots(figsize=(9, 3.5))
        c = curve.dropna(subset=["timestamp"]).copy()
        if len(c) > 10:
            c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
            r = c.set_index("timestamp")["equity"].pct_change().dropna()
            roll = r.rolling(30).apply(lambda x: (x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else np.nan)
            ax.plot(roll.index, roll.values, color="#54A24B")
        ax.set_title(f"{prefix} rolling Sharpe (30 trades)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _rolling_dd(self, charts: Path, curve: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_rolling_drawdown.png"
        fig, ax = plt.subplots(figsize=(9, 3.5))
        c = curve.dropna(subset=["timestamp"])
        if not c.empty:
            ax.plot(pd.to_datetime(c["timestamp"]), c["drawdown"], color="#F58518")
        ax.set_title(f"{prefix} rolling drawdown")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _trade_dist(self, charts: Path, trade_log: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_trade_returns_dist.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        taken = trade_log.loc[~trade_log["skipped"].astype(bool)] if "skipped" in trade_log.columns else trade_log
        if not taken.empty:
            ax.hist(taken["pnl"], bins=40, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{prefix} trade PnL distribution ($)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _monthly_dist(self, charts: Path, monthly: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_monthly_returns_dist.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        if not monthly.empty:
            ax.hist(monthly["return"], bins=20, color="#72B7B2", edgecolor="white")
        ax.set_title(f"{prefix} monthly return distribution")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _mc(self, charts: Path, mc: pd.DataFrame, prefix: str) -> None:
        path = charts / f"{prefix}_monte_carlo.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fe = mc["final_equity"].to_numpy(dtype=float)
        dd = mc["max_drawdown"].to_numpy(dtype=float)
        fe = fe[np.isfinite(fe)]
        dd = dd[np.isfinite(dd)]
        bins_fe = min(40, max(1, len(np.unique(np.round(fe, 6)))))
        bins_dd = min(40, max(1, len(np.unique(np.round(dd, 6)))))
        if len(fe):
            axes[0].hist(fe, bins=bins_fe, color="#4C78A8", edgecolor="white")
        axes[0].set_title("MC final equity")
        if len(dd):
            axes[1].hist(dd, bins=bins_dd, color="#E45756", edgecolor="white")
        axes[1].set_title("MC max drawdown")
        fig.suptitle(prefix)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def compare_systems(self, charts: Path, curves: dict[str, pd.DataFrame], title: str) -> None:
        path = charts / "primary_vs_meta_equity.png"
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = {"primary": "#4C78A8", "primary_plus_meta": "#54A24B"}
        for name, curve in curves.items():
            c = curve.dropna(subset=["timestamp"])
            if c.empty:
                continue
            ax.plot(
                pd.to_datetime(c["timestamp"]),
                c["equity"],
                label=name,
                color=colors.get(name, None),
            )
        ax.legend()
        ax.set_title(title)
        ax.set_ylabel("equity ($)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
