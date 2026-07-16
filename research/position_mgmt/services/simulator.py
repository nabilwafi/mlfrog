"""Bar-path position management simulator (same entries, alternate exits)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from research.position_mgmt import COST, DEFAULT_HORIZON, MAX_BARS, SL_ATR, TP_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market, window_end


ExitReason = str


@dataclass
class ManageConfig:
    name: str
    family: str  # baseline | be | partial | trail | time | vol | pyramid | scale_in
    # break-even
    be_trigger_r: float | None = None  # move SL to BE after +R
    be_atr_trigger: float | None = None  # move SL to BE after MFE >= k*ATR
    # partial TP
    partial_frac: float = 0.0
    partial_at_r: float | None = None
    # trail
    trail_atr: float | None = None
    trail_adaptive: bool = False
    # time
    time_exit_bars: int | None = None
    # vol exit
    vol_exit: str | None = None  # atr | adx | volume
    # pyramid (add-on into winner)
    pyramid_r: float | None = None  # risk multiple of base for 2nd position (0.25/0.5/1)
    pyramid_trigger_r: float = 1.0
    pyramid_min_conf: float = 60.0
    # scale-in
    scale_retrace_atr: float | None = None  # second 50% after adverse retrace
    horizon: int = DEFAULT_HORIZON
    max_bars: int = MAX_BARS


def build_experiments() -> list[ManageConfig]:
    exps = [ManageConfig(name="baseline", family="baseline")]
    for r in (0.3, 0.5, 0.7, 1.0):
        exps.append(ManageConfig(name=f"be_{r}R", family="be", be_trigger_r=r))
    exps.append(ManageConfig(name="be_atr_0_5", family="be", be_atr_trigger=0.5))
    for r in (0.5, 1.0, 1.2, 1.5):
        exps.append(
            ManageConfig(name=f"partial50_at_{r}R", family="partial", partial_frac=0.5, partial_at_r=r)
        )
    for a in (0.5, 0.8, 1.0, 1.5):
        exps.append(ManageConfig(name=f"trail_{a}atr", family="trail", trail_atr=a))
    exps.append(ManageConfig(name="trail_adaptive", family="trail", trail_atr=1.0, trail_adaptive=True))
    for b in (8, 12, 24, 36):
        exps.append(ManageConfig(name=f"time_{b}b", family="time", time_exit_bars=b, max_bars=max(b, MAX_BARS)))
    for v in ("atr", "adx", "volume"):
        exps.append(ManageConfig(name=f"vol_exit_{v}", family="vol", vol_exit=v))
    for pr in (0.25, 0.5, 1.0):
        exps.append(
            ManageConfig(
                name=f"pyramid_{pr}R",
                family="pyramid",
                pyramid_r=pr,
                pyramid_trigger_r=1.0,
                pyramid_min_conf=60.0,
            )
        )
    for ra in (0.25, 0.5, 0.75):
        exps.append(ManageConfig(name=f"scale_in_{ra}atr", family="scale_in", scale_retrace_atr=ra))
    return exps


@dataclass
class SimResult:
    net_return: float
    holding_bars: int
    exit_reason: str
    size_avg: float = 1.0
    pyramid_added: bool = False


def _signed_pnl_frac(side: str, entry: float, exit_px: float) -> float:
    if side == "long":
        return (exit_px - entry) / entry
    return (entry - exit_px) / entry


def simulate_one(
    *,
    side: str,
    entry: float,
    atr_entry: float,
    conf: float,
    d1_regime: str,
    daily_stop_hit: bool,
    ei: int,
    mkt: dict[str, Any],
    cfg: ManageConfig,
) -> SimResult:
    high, low, close = mkt["high"], mkt["low"], mkt["close"]
    atr_s, adx_s, vol = mkt["atr"], mkt["adx"], mkt["vol"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr_entry <= 0 or entry <= 0:
        return SimResult(net_return=-COST, holding_bars=0, exit_reason="bad_entry")

    sl_dist = SL_ATR * atr_entry
    tp_dist = TP_ATR * atr_entry
    one_r_px = sl_dist
    is_long = side == "long"

    if is_long:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    # scale-in: start 50% then add 50%
    if cfg.scale_retrace_atr is not None:
        size = 0.5
        scale_pending = True
        adverse_need = cfg.scale_retrace_atr * atr_entry
    else:
        size = 1.0
        scale_pending = False
        adverse_need = 0.0

    realized = 0.0  # fractional return * size pieces
    partial_done = False
    be_done = False
    max_fav = entry
    min_fav = entry  # extreme for trail: long uses max high, short min low
    pyramid_added = False
    pyramid_size = 0.0
    pyramid_entry = entry

    end = window_end(ei, n, cfg.max_bars)
    # default management horizon: use cfg.time_exit_bars or cfg.horizon for TP/SL walk
    hard_end = ei + (cfg.time_exit_bars or cfg.horizon)
    hard_end = min(hard_end, end)

    adx0 = float(adx_s[ei]) if np.isfinite(adx_s[ei]) else np.nan
    vol0 = float(np.nanmean(vol[max(0, ei - 10) : ei + 1]))

    j_exit = ei
    reason = "TIMEOUT"
    exit_px = float(close[min(hard_end, end)])

    for j in range(ei + 1, end + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr_entry

        # scale-in fill on adverse retrace without SL first
        if scale_pending:
            if is_long and (entry - l) >= adverse_need and l > sl:
                size = 1.0
                scale_pending = False
            elif (not is_long) and (h - entry) >= adverse_need and h < sl:
                size = 1.0
                scale_pending = False

        # trail update from favorable extreme
        if is_long:
            max_fav = max(max_fav, h)
        else:
            min_fav = min(min_fav, l)

        trail_mult = cfg.trail_atr
        if cfg.trail_adaptive and trail_mult is not None:
            # tighter when ATR expands vs entry
            ratio = atr_j / atr_entry if atr_entry > 0 else 1.0
            trail_mult = 0.5 if ratio >= 1.2 else (1.5 if ratio <= 0.8 else 1.0)

        if trail_mult is not None:
            if is_long:
                # activate trail after +0.5R
                if (max_fav - entry) >= 0.5 * one_r_px:
                    trail_sl = max_fav - trail_mult * atr_j
                    sl = max(sl, trail_sl)
            else:
                if (entry - min_fav) >= 0.5 * one_r_px:
                    trail_sl = min_fav + trail_mult * atr_j
                    sl = min(sl, trail_sl)

        # break-even
        fav_r = ((max_fav - entry) if is_long else (entry - min_fav)) / one_r_px if one_r_px > 0 else 0.0
        if not be_done:
            trig = False
            if cfg.be_trigger_r is not None and fav_r >= cfg.be_trigger_r:
                trig = True
            if cfg.be_atr_trigger is not None:
                fav_atr = ((max_fav - entry) if is_long else (entry - min_fav)) / atr_entry
                if fav_atr >= cfg.be_atr_trigger:
                    trig = True
            if trig:
                sl = entry  # BE
                be_done = True

        # same-bar order: SL then TP (conservative), then partial level
        hit_sl = (l <= sl) if is_long else (h >= sl)
        hit_tp = (h >= tp) if is_long else (l <= tp)

        # partial TP level
        if cfg.partial_frac > 0 and cfg.partial_at_r is not None and not partial_done:
            partial_px = entry + cfg.partial_at_r * one_r_px if is_long else entry - cfg.partial_at_r * one_r_px
            hit_partial = (h >= partial_px) if is_long else (l <= partial_px)
        else:
            hit_partial = False
            partial_px = tp

        if hit_sl and hit_tp:
            # SL first
            realized += size * _signed_pnl_frac(side, entry, sl)
            if pyramid_added:
                realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, sl)
            j_exit, reason, exit_px = j, "SL", sl
            size = 0.0
            pyramid_size = 0.0
            break
        if hit_sl:
            realized += size * _signed_pnl_frac(side, entry, sl)
            if pyramid_added:
                realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, sl)
            j_exit, reason, exit_px = j, "SL", sl
            size = 0.0
            pyramid_size = 0.0
            break
        if hit_partial:
            take = cfg.partial_frac * size
            realized += take * _signed_pnl_frac(side, entry, partial_px)
            size -= take
            partial_done = True
            # leave remainder to TP/SL/trail
        if hit_tp and size > 0:
            realized += size * _signed_pnl_frac(side, entry, tp)
            if pyramid_added:
                realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, tp)
            j_exit, reason, exit_px = j, "TP", tp
            size = 0.0
            pyramid_size = 0.0
            break

        # pyramid: one add into winner
        if (
            cfg.pyramid_r is not None
            and not pyramid_added
            and not daily_stop_hit
            and conf >= cfg.pyramid_min_conf
            and d1_regime in ("Strong Bull", "Strong Bear")
            and fav_r >= cfg.pyramid_trigger_r
        ):
            # add size proportional to risk multiple (base size=1 corresponds to 1R risk)
            pyramid_size = float(cfg.pyramid_r)
            pyramid_entry = c  # add at close of trigger bar (ponytail)
            pyramid_added = True
            # second position SL at BE of add or shared trail — use original SL initially
            # (already tracked via size+pyramid_size exits)

        # volatility exit
        if cfg.vol_exit == "atr" and atr_entry > 0 and atr_j / atr_entry < 0.6:
            realized += size * _signed_pnl_frac(side, entry, c)
            if pyramid_added:
                realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, c)
            j_exit, reason, exit_px = j, "VOL_ATR", c
            size = 0.0
            pyramid_size = 0.0
            break
        if cfg.vol_exit == "adx" and np.isfinite(adx0) and np.isfinite(adx_s[j]):
            if float(adx_s[j]) < 0.7 * adx0:
                realized += size * _signed_pnl_frac(side, entry, c)
                if pyramid_added:
                    realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, c)
                j_exit, reason, exit_px = j, "VOL_ADX", c
                size = 0.0
                pyramid_size = 0.0
                break
        if cfg.vol_exit == "volume" and vol0 > 0:
            vma = float(np.nanmean(vol[max(0, j - 5) : j + 1]))
            if vma < 0.5 * vol0:
                realized += size * _signed_pnl_frac(side, entry, c)
                if pyramid_added:
                    realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, c)
                j_exit, reason, exit_px = j, "VOL_VOL", c
                size = 0.0
                pyramid_size = 0.0
                break

        # time / horizon end
        if j >= hard_end:
            realized += size * _signed_pnl_frac(side, entry, c)
            if pyramid_added:
                realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, c)
            j_exit, reason, exit_px = j, "TIMEOUT", c
            size = 0.0
            pyramid_size = 0.0
            break

    if size > 0 or pyramid_size > 0:
        # safety close
        c = float(close[min(j_exit, end)])
        realized += size * _signed_pnl_frac(side, entry, c)
        if pyramid_added:
            realized += pyramid_size * _signed_pnl_frac(side, pyramid_entry, c)

    # cost: once for primary; pyramid adds half cost; scale incomplete still pays once
    cost = COST
    if pyramid_added:
        cost += COST * 0.5
    net = realized - cost
    hold = max(int(j_exit - ei), 0)
    avg_size = 1.0 + (float(cfg.pyramid_r) if pyramid_added and cfg.pyramid_r else 0.0)
    if cfg.scale_retrace_atr is not None and scale_pending:
        avg_size = 0.5
    return SimResult(net_return=float(net), holding_bars=hold, exit_reason=reason, size_avg=avg_size, pyramid_added=pyramid_added)


def simulate_panel(
    trades: pd.DataFrame,
    h1: pd.DataFrame | dict,
    cfg: ManageConfig,
    *,
    daily_stop_flags: np.ndarray | None = None,
    mkt: dict | None = None,
    eis: np.ndarray | None = None,
) -> pd.DataFrame:
    if mkt is None:
        mkt = prepare_market(h1 if isinstance(h1, pd.DataFrame) else h1)
    if eis is None:
        eis = entry_indices(trades, mkt["ts"])
    if daily_stop_flags is None:
        daily_stop_flags = np.zeros(len(trades), dtype=bool)

    rows_out = []
    for pos in range(len(trades)):
        row = trades.iloc[pos]
        atr = float(row["atr_entry"]) if pd.notna(row.get("atr_entry")) else float(row.get("atr_price", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = float(row["entry_price"]) * float(row.get("distance_to_sl", 0.001)) / SL_ATR
        res = simulate_one(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr_entry=atr,
            conf=float(row.get("confidence", 50) or 50),
            d1_regime=str(row.get("d1_regime", "") or ""),
            daily_stop_hit=bool(daily_stop_flags[pos]),
            ei=int(eis[pos]),
            mkt=mkt,
            cfg=cfg,
        )
        rows_out.append(
            {
                "net_return_sim": res.net_return,
                "holding_bars_sim": res.holding_bars,
                "exit_reason_sim": res.exit_reason,
                "pyramid_added": res.pyramid_added,
                "size_avg": res.size_avg,
            }
        )
    out = trades.reset_index(drop=True).copy()
    sim = pd.DataFrame(rows_out)
    for c in sim.columns:
        out[c] = sim[c]
    out["net_return_base"] = out["net_return"].astype(float)
    out["net_return"] = out["net_return_sim"].astype(float)
    out["policy"] = cfg.name
    out["family"] = cfg.family
    return out
