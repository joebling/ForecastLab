"""Binance data downloader

Downloads spot klines from Binance public REST API and saves to a CSV in data/raw.

Notes:
- Binance 1d klines are returned with open_time/close_time in milliseconds since epoch (UTC).
- This script keeps both open_time and close_time for unambiguous alignment.
- No API key is required for public klines.

Example:
  python -m src.data.binance_downloader --symbol BTCUSDT --interval 1d --start 2022-01-01 --end 2025-12-12 --out data/raw/btc_binance_BTCUSDT_1d.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


def _parse_yyyymmdd(s: str) -> dt.datetime:
    # Treat as UTC date boundary.
    return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def _to_ms(ts: dt.datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def _ms_to_iso_utc(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat()


@dataclass
class KlineRow:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    sleep_s: float = 0.2,
) -> List[KlineRow]:
    """Fetch klines in UTC milliseconds, paginating forward from start_ms."""
    out: List[KlineRow] = []
    cur_start = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "limit": limit,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(BINANCE_BASE_URL + KLINES_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for r in data:
            # Binance kline array layout:
            # 0 open time
            # 1 open
            # 2 high
            # 3 low
            # 4 close
            # 5 volume
            # 6 close time
            # 7 quote asset volume
            # 8 number of trades
            # 9 taker buy base asset volume
            # 10 taker buy quote asset volume
            row = KlineRow(
                open_time_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                close_time_ms=int(r[6]),
                quote_asset_volume=float(r[7]),
                number_of_trades=int(r[8]),
                taker_buy_base_asset_volume=float(r[9]),
                taker_buy_quote_asset_volume=float(r[10]),
            )
            out.append(row)

        last_open_time = int(data[-1][0])
        # Stop if we didn't advance (safety)
        if last_open_time == cur_start:
            break

        # Next page starts right after the last open_time to avoid duplicates
        cur_start = last_open_time + 1

        # If we got less than limit, we're done
        if len(data) < limit:
            break

        time.sleep(sleep_s)

    # De-duplicate (in case of overlaps)
    uniq = {}
    for r in out:
        uniq[r.open_time_ms] = r
    return [uniq[k] for k in sorted(uniq.keys())]


def write_csv(rows: List[KlineRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_time_ms",
        "close_time_ms",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "timestamp": _ms_to_iso_utc(r.open_time_ms),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "open_time_ms": r.open_time_ms,
                    "close_time_ms": r.close_time_ms,
                    "quote_asset_volume": r.quote_asset_volume,
                    "number_of_trades": r.number_of_trades,
                    "taker_buy_base_asset_volume": r.taker_buy_base_asset_volume,
                    "taker_buy_quote_asset_volume": r.taker_buy_quote_asset_volume,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Download Binance klines and save to CSV")
    p.add_argument("--symbol", default="BTCUSDT", help="Spot symbol, e.g. BTCUSDT")
    p.add_argument("--interval", default="1d", help="Kline interval, e.g. 1d, 1h")
    p.add_argument("--start", default="2022-01-01", help="Start date (UTC) YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date (UTC) YYYY-MM-DD (exclusive). Optional")
    p.add_argument(
        "--out",
        default="data/raw/btc_binance_BTCUSDT_1d.csv",
        help="Output CSV path",
    )

    args = p.parse_args()

    start_dt = _parse_yyyymmdd(args.start)
    end_ms: Optional[int] = None
    if args.end:
        end_dt = _parse_yyyymmdd(args.end)
        end_ms = _to_ms(end_dt)

    rows = fetch_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_ms=_to_ms(start_dt),
        end_ms=end_ms,
    )

    if not rows:
        raise SystemExit("No data returned from Binance API. Check symbol/interval/date range.")

    write_csv(rows, Path(args.out))
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
