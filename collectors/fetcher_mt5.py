"""

Fetch XAUUSD OHLCV bars from MetaTrader 5.



Usage:

    python -m collectors.fetcher_mt5

    python -m collectors.fetcher_mt5 --config configs/config.yaml

"""



from __future__ import annotations



import argparse

import sys

import uuid

from dataclasses import dataclass

from datetime import date, datetime, timedelta

from pathlib import Path

from typing import Any, Iterator



import pandas as pd

import yaml



from connectors.mt5 import MT5Connector, resolve_timeframe

from pkg.logger import setup_logger





@dataclass(frozen=True)

class DateChunk:

    index: int

    start: datetime

    end: datetime  # exclusive upper bound





def load_config(path: Path) -> dict[str, Any]:

    with path.open(encoding="utf-8") as f:

        return yaml.safe_load(f)





def parse_date(value: str, *, end_of_day: bool = False) -> datetime:

    dt = datetime.strptime(value.strip(), "%Y-%m-%d")

    if end_of_day:

        return dt.replace(hour=23, minute=59, second=59)

    return dt





def add_years(d: date, years: int) -> date:

    try:

        return d.replace(year=d.year + years)

    except ValueError:

        return d.replace(year=d.year + years, day=28)





def add_months(d: date, months: int) -> date:

    month = d.month - 1 + months

    year = d.year + month // 12

    month = month % 12 + 1

    day = min(d.day, _days_in_month(year, month))

    return date(year, month, day)





def _days_in_month(year: int, month: int) -> int:

    if month == 12:

        nxt = date(year + 1, 1, 1)

    else:

        nxt = date(year, month + 1, 1)

    return (nxt - date(year, month, 1)).days





def iter_chunks(

    start: datetime,

    end: datetime,

    *,

    unit: str = "year",

    step: int = 1,

) -> Iterator[DateChunk]:

    if start >= end:

        raise ValueError(f"start_date ({start}) must be before end_date ({end})")

    if step < 1:

        raise ValueError("chunk.step must be >= 1")



    unit = unit.lower()

    if unit not in ("year", "month"):

        raise ValueError(f"unsupported chunk.unit: {unit!r} (use 'year' or 'month')")



    cursor = start

    idx = 0

    while cursor < end:

        if unit == "year":

            nxt = datetime.combine(add_years(cursor.date(), step), datetime.min.time())

        else:

            nxt = datetime.combine(add_months(cursor.date(), step), datetime.min.time())



        chunk_end = min(nxt, end)

        yield DateChunk(index=idx, start=cursor, end=chunk_end)

        cursor = chunk_end

        idx += 1





def fetch_chunk(

    connector: MT5Connector,

    symbol: str,

    timeframe: int,

    chunk: DateChunk,

    log: Any,

) -> pd.DataFrame:

    log.info(

        "Fetching chunk %d | %s -> %s",

        chunk.index,

        chunk.start.strftime("%Y-%m-%d %H:%M:%S"),

        chunk.end.strftime("%Y-%m-%d %H:%M:%S"),

    )



    df = connector.copy_rates_range(symbol, timeframe, chunk.start, chunk.end)

    if df.empty:

        log.warning("Chunk %d returned 0 bars", chunk.index)

        return df



    log.info("Chunk %d done | bars=%d", chunk.index, len(df))

    return df





def save_output(

    df: pd.DataFrame,

    out_dir: Path,

    symbol: str,

    timeframe_label: str,

    fmt: str,

    *,

    suffix: str = "",

    log: Any,

) -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{symbol}_{timeframe_label}{suffix}"

    fmt = fmt.lower()



    if fmt == "csv":

        path = out_dir / f"{stem}.csv"

        df.to_csv(path, index=False)

    elif fmt == "parquet":

        path = out_dir / f"{stem}.parquet"

        df.to_parquet(path, index=False)

    else:

        raise ValueError(f"unsupported output.format: {fmt!r}")



    log.info("Saved %d bars -> %s", len(df), path)

    return path





def run(config_path: Path) -> int:

    run_id = uuid.uuid4().hex[:8]

    cfg = load_config(config_path)



    log_cfg = cfg.get("logging", {})

    log = setup_logger(

        "xauusd.fetcher",

        level=log_cfg.get("level", "INFO"),

        log_dir=log_cfg.get("directory", "./logs"),

        filename=log_cfg.get("filename", "xauusd_fetcher.log"),

        max_bytes=int(log_cfg.get("max_bytes", 10_485_760)),

        backup_count=int(log_cfg.get("backup_count", 5)),

        console=bool(log_cfg.get("console", True)),

        run_id=run_id,

    )



    symbol = cfg.get("symbol", "XAUUSD")

    tf_label = str(cfg.get("timeframe", "H1")).upper()

    try:

        timeframe = resolve_timeframe(tf_label)

    except ValueError as exc:

        log.error("%s", exc)

        return 1



    dr = cfg["date_range"]

    start = parse_date(dr["start_date"])

    end = parse_date(dr["end_date"], end_of_day=True) + timedelta(seconds=1)



    chunk_cfg = cfg.get("chunk", {})

    unit = chunk_cfg.get("unit", "year")

    step = int(chunk_cfg.get("step", 1))



    out_cfg = cfg.get("output", {})

    out_dir = Path(out_cfg.get("directory", "./data/raw"))
    out_fmt = out_cfg.get("format", "csv")

    combine = bool(out_cfg.get("combine_chunks", True))



    chunks = list(iter_chunks(start, end, unit=unit, step=step))

    log.info(

        "Job start | run=%s | symbol=%s | tf=%s | range=%s -> %s | chunks=%d",

        run_id,

        symbol,

        tf_label,

        start.date(),

        (end - timedelta(seconds=1)).date(),

        len(chunks),

    )



    with MT5Connector(cfg, log) as connector:

        connector.select_symbol(symbol)



        frames: list[pd.DataFrame] = []

        for chunk in chunks:

            df = fetch_chunk(connector, symbol, timeframe, chunk, log)

            if df.empty:

                continue



            if combine:

                frames.append(df)

            else:

                suffix = (

                    f"_{chunk.start.strftime('%Y%m%d')}_{chunk.end.strftime('%Y%m%d')}"

                )

                save_output(

                    df, out_dir, symbol, tf_label, out_fmt, suffix=suffix, log=log

                )



        if combine:

            if not frames:

                log.warning("No bars fetched for the entire date range")

                return 0



            combined = (

                pd.concat(frames, ignore_index=True)

                .drop_duplicates(subset=["Date"])

                .sort_values("Date")

                .reset_index(drop=True)

            )

            save_output(combined, out_dir, symbol, tf_label, out_fmt, log=log)

            log.info(

                "Job complete | total_bars=%d | chunks_ok=%d/%d",

                len(combined),

                len(frames),

                len(chunks),

            )

        else:

            log.info("Job complete | chunks_saved=%d/%d", len(frames), len(chunks))



    return 0





def main() -> None:

    parser = argparse.ArgumentParser(description="Fetch XAUUSD bars from MetaTrader 5")

    parser.add_argument(

        "--config",

        type=Path,

        default=Path("configs/config.yaml"),

        help="Path to YAML config (default: configs/config.yaml)",

    )

    args = parser.parse_args()



    if not args.config.exists():

        print(f"Config not found: {args.config}", file=sys.stderr)

        sys.exit(1)



    sys.exit(run(args.config))





if __name__ == "__main__":

    main()


