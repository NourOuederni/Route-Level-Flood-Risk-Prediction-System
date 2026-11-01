"""Hourly IMERG rain for Tunisian trigger zones, in the input format of `validate_trigger`.

    python -m flood_risk.real.imerg_tn --start 2014-09-01 --end 2025-09-01

Stand-in for INM gauge records while they are not available. Each IMERG cell (0.1 degree) whose centre
lies within `RADIUS_KM` of a zone's centre plays the part of a station. Zones and known floods are in
`data/tn_trigger/zones.csv` and `floods.csv`.

Downloading every half hour of eleven years is too slow, so the hourly series is built in two parts:

- Every day comes from IMERG Final daily.
- Days of a pair of consecutive days whose total reaches `CANDIDATE_MM` at some zone cell get the real
  half-hourly rain. Any 24 h window lies within two consecutive days, so outside those pairs no 24 h total
  can reach `CANDIDATE_MM`: every threshold from `CANDIDATE_MM` up is measured exactly.
- The other days are spread evenly over their 24 hours. That only changes how weak spells split, never
  whether a threshold >= `CANDIDATE_MM` is reached.

This is the same rain source as the calibration of the 44 mm threshold (IMERG Final), on Tunisian floods
that it was not fitted on (except Nabeul 2018). It does not replace a check on gauges or on the real-time
rain feed.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .areas import raw_dir
from .imerg import CELL, cell_index, half_hour
from .trigger import daily_grid

RADIUS_KM = 10.0
CANDIDATE_MM = 30.0
OUT = Path("data/tn_trigger")
_local = threading.local()


def zone_cells(zones: pd.DataFrame) -> pd.DataFrame:
    """IMERG cells whose centre lies within RADIUS_KM of each zone centre (at least the cell of the centre)."""
    rows = []
    for z in zones.itertuples():
        cx, cy = cell_index(z.lon, z.lat)
        for x in range(cx - 2, cx + 3):
            for y in range(cy - 2, cy + 3):
                lon, lat = (x + 0.5) * CELL - 180, (y + 0.5) * CELL - 90
                km = np.hypot((lon - z.lon) * 111.3 * np.cos(np.radians(z.lat)), (lat - z.lat) * 111.3)
                if km <= RADIUS_KM or (x, y) == (cx, cy):
                    rows.append({"zone": z.zone, "x": x, "y": y, "station": f"{x}_{y}"})
    return pd.DataFrame(rows)


def cookies() -> Path:
    """One cookie file per download thread: curl rewrites it on every call."""
    if not hasattr(_local, "file"):
        _local.file = raw_dir("imerg_tn") / f"cookies_{threading.get_ident()}.txt"
    return _local.file


def fetch(tasks, fn, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, tasks))


def build(zones: pd.DataFrame, start, end, workers=8) -> pd.DataFrame:
    cells = zone_cells(zones)
    x0, x1, y0, y1 = cells.x.min(), cells.x.max(), cells.y.min(), cells.y.max()
    cache = raw_dir("imerg_tn")
    days = pd.date_range(start, end, freq="D", inclusive="left")

    print(f"{len(days)} daily grids", flush=True)
    grids = fetch(days, lambda d: daily_grid(d, x0, x1, y0, y1, cookies(), cache), workers)
    daily = np.stack(grids)[:, cells.x - x0, cells.y - y0]                     # (day, zone cell)

    two_days = np.nan_to_num(daily[:-1]) + np.nan_to_num(daily[1:])
    pairs = np.flatnonzero((two_days >= CANDIDATE_MM).any(axis=1))
    detailed = days[np.union1d(pairs, pairs + 1)]
    halves = pd.DatetimeIndex([t for d in detailed for t in pd.date_range(d, periods=48, freq="30min")])
    print(f"{len(detailed)} days at half-hour resolution ({len(halves)} half hours)", flush=True)
    rates = fetch(halves, lambda t: half_hour(cookies(), t, x0, x1, y0, y1, cache), workers)
    rates = np.stack(rates)[:, cells.x - x0, cells.y - y0] if rates else np.empty((0, len(cells)))
    fine = (rates[0::2] + rates[1::2]) / 2                                       # mm in each hour

    hours = pd.date_range(start, end, freq="h", inclusive="left")
    hourly = np.repeat(daily / 24, 24, axis=0)                                   # days spread evenly
    pos = hours.get_indexer(pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in detailed for h in range(24)]))
    hourly[pos] = fine
    frame = pd.DataFrame(hourly, index=hours, columns=pd.MultiIndex.from_frame(cells[["zone", "station"]]))
    long = frame.stack(["zone", "station"], future_stack=True).rename("rain_mm").reset_index()
    long = long.rename(columns={"level_0": "time"})
    long["source"] = np.where(long.time.dt.normalize().isin(detailed), "half_hourly", "daily_spread")
    return long[["zone", "station", "time", "rain_mm", "source"]], cells, detailed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default=str(OUT / "zones.csv"))
    ap.add_argument("--start", default="2014-09-01")
    ap.add_argument("--end", default="2025-09-01", help="excluded; IMERG Final ends about 3.5 months back")
    ap.add_argument("--out", default=str(OUT / "imerg_hourly.csv.gz"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    zones = pd.read_csv(args.zones)
    long, cells, detailed = build(zones, pd.Timestamp(args.start), pd.Timestamp(args.end), args.workers)
    long.to_csv(args.out, index=False, float_format="%.3f")
    cells.to_csv(Path(args.out).with_name("zone_cells.csv"), index=False)
    print(f"{len(long):,} rows, {cells.station.nunique()} cells for {cells.zone.nunique()} zones, "
          f"{len(detailed)} days at half-hour resolution -> {args.out}")


if __name__ == "__main__":
    main()
