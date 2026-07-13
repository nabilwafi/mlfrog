from __future__ import annotations

import MetaTrader5 as mt5

TIMEFRAME_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def resolve_timeframe(label: str) -> int:
    key = label.strip().upper()
    if key not in TIMEFRAME_MAP:
        valid = ", ".join(TIMEFRAME_MAP)
        raise ValueError(f"unknown timeframe: {label!r} (valid: {valid})")
    return TIMEFRAME_MAP[key]
