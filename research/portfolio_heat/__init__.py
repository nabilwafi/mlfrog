"""Sprint 24 — Portfolio Heat Management (execution only; frozen models)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

META_GATE: float = 0.45
CONF_SKIP: float = 40.0
BASE_RISK: float = 0.01
STARTING_EQUITY: float = 80.0
H1_HOURS: float = 1.0


@dataclass(frozen=True)
class HeatPolicy:
    """Portfolio heat knobs. None / inf means unlimited / off."""

    name: str
    max_open: float = float("inf")
    max_same_dir: float = float("inf")
    daily_loss_r: float | None = None  # stop if day PnL <= -R
    weekly_loss_r: float | None = None
    daily_profit_r: float | None = None  # lock if day PnL >= +R
    cooldown_after_losses: int | None = None
    cooldown_hours: float | None = None  # wall hours; or special via cooldown_mode
    cooldown_mode: str = "hours"  # hours | next_session | next_day
    cooldown_after_win_r: float | None = None  # large winner cooldown trigger in R
    cooldown_after_win_hours: float | None = None
    max_daily_trades: float = float("inf")
    max_london_trades: float = float("inf")
    max_ny_trades: float = float("inf")
    max_heat: float | None = None  # sum open risk_pct
    float_dd_stop: float | None = None  # peak equity DD stop new entries
    vol_scale: bool = False
    d1_scale: bool = False
    conf_scale: bool = False
    session_scale: bool = False
    stop_london_after_losses: int | None = None
    stop_ny_after_losses: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_policy_grid() -> list[HeatPolicy]:
    """Curated one-factor + a few combos. Not a full cartesian product."""
    pols: list[HeatPolicy] = [HeatPolicy(name="baseline_skip40_flat1")]

    for n in (1, 2, 3):
        pols.append(HeatPolicy(name=f"max_open_{n}", max_open=float(n)))
    for n in (1, 2, 3):
        pols.append(HeatPolicy(name=f"max_same_dir_{n}", max_same_dir=float(n)))
    for r in (1.0, 2.0, 3.0):
        pols.append(HeatPolicy(name=f"daily_loss_{-int(r)}R", daily_loss_r=r))
    for r in (3.0, 5.0, 8.0):
        pols.append(HeatPolicy(name=f"weekly_loss_{-int(r)}R", weekly_loss_r=r))
    for r in (2.0, 3.0, 5.0):
        pols.append(HeatPolicy(name=f"daily_profit_{int(r)}R", daily_profit_r=r))
    for losses in (2, 3, 4):
        for hours in (2.0, 4.0):
            pols.append(
                HeatPolicy(
                    name=f"cd_L{losses}_{int(hours)}h",
                    cooldown_after_losses=losses,
                    cooldown_hours=hours,
                    cooldown_mode="hours",
                )
            )
        pols.append(
            HeatPolicy(
                name=f"cd_L{losses}_next_day",
                cooldown_after_losses=losses,
                cooldown_mode="next_day",
            )
        )
        pols.append(
            HeatPolicy(
                name=f"cd_L{losses}_next_session",
                cooldown_after_losses=losses,
                cooldown_mode="next_session",
            )
        )
    for hours in (2.0, 4.0):
        pols.append(
            HeatPolicy(
                name=f"cd_win2R_{int(hours)}h",
                cooldown_after_win_r=2.0,
                cooldown_after_win_hours=hours,
            )
        )
    for n in (2, 3, 4):
        pols.append(HeatPolicy(name=f"max_day_trades_{n}", max_daily_trades=float(n)))
    for n in (2, 3):
        pols.append(HeatPolicy(name=f"max_london_{n}", max_london_trades=float(n)))
        pols.append(HeatPolicy(name=f"max_ny_{n}", max_ny_trades=float(n)))
    for h in (0.01, 0.02, 0.03, 0.04):
        pols.append(HeatPolicy(name=f"heat_cap_{int(h*100)}pct", max_heat=h))
    for d in (0.02, 0.03, 0.05):
        pols.append(HeatPolicy(name=f"float_dd_{int(d*100)}pct", float_dd_stop=d))

    pols.append(HeatPolicy(name="vol_scale", vol_scale=True))
    pols.append(HeatPolicy(name="d1_scale", d1_scale=True))
    pols.append(HeatPolicy(name="conf_scale", conf_scale=True))
    pols.append(HeatPolicy(name="session_scale", session_scale=True))
    pols.append(HeatPolicy(name="london_stop_3L", stop_london_after_losses=3))
    pols.append(HeatPolicy(name="ny_stop_3L", stop_ny_after_losses=3))

    # Combos that usually matter
    pols.append(
        HeatPolicy(
            name="combo_heat2_open2_dd3",
            max_open=2,
            max_heat=0.02,
            float_dd_stop=0.03,
        )
    )
    pols.append(
        HeatPolicy(
            name="combo_open2_dir1_day2R",
            max_open=2,
            max_same_dir=1,
            daily_loss_r=2.0,
        )
    )
    pols.append(
        HeatPolicy(
            name="combo_cd3_4h_day3_heat2",
            max_open=3,
            max_heat=0.02,
            cooldown_after_losses=3,
            cooldown_hours=4.0,
            max_daily_trades=3,
        )
    )
    pols.append(
        HeatPolicy(
            name="combo_d1_vol_heat2",
            max_heat=0.02,
            d1_scale=True,
            vol_scale=True,
        )
    )
    pols.append(
        HeatPolicy(
            name="combo_session_caps_day3",
            max_london_trades=3,
            max_ny_trades=3,
            max_daily_trades=3,
            daily_loss_r=2.0,
        )
    )
    pols.append(
        HeatPolicy(
            name="combo_full_conservative",
            max_open=2,
            max_same_dir=1,
            max_heat=0.02,
            float_dd_stop=0.03,
            daily_loss_r=2.0,
            weekly_loss_r=5.0,
            cooldown_after_losses=3,
            cooldown_hours=4.0,
            max_daily_trades=3,
            d1_scale=True,
            vol_scale=True,
        )
    )
    return pols
