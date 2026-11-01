"""Rain trigger of a zone: when should the terrain ranking turn into alerts?

    python -m flood_risk.real.trigger

The terrain model ranks roads but cannot say when to alert, since it does not change with the weather.
The trigger is a rain threshold per zone. To set it, it must be tested on rain spells that flooded
*and* on rain spells that did not. For each mapped zone, the inventory's rain archive covers the
120 days before the Copernicus activation:

- positive: the wet spell that contains the zone's heaviest rain hour of the mapped flood;
- negative: every other wet spell of >= 5 mm in those 120 days. No Copernicus activation followed
  them, so no major flood was mapped; minor street flooding cannot be ruled out.

Rain is NASA GPM IMERG Final daily (0.1 degree), the source the model was trained with, taken at
the wettest cell of the zone. Thresholds are chosen leave-one-event-out: the threshold is fitted
on the other 12 events and checked on the held-out one.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ems
from .areas import raw_dir
from .imerg import HOSTS, NETRC

DAILY = ("https://{host}/opendap/GPM_L3/GPM_3IMERGDF.07/{t:%Y}/{t:%m}/"
         "3B-DAY.MS.MRG.3IMERG.{t:%Y%m%d}-S000000-E235959.V07B.nc4.ascii?precipitation[0:0][{x0}:{x1}][{y0}:{y1}]")
WET_MM, GAP_DAYS, MIN_SPELL_MM = 1.0, 1, 5.0
INDICATORS = ["max_1d_mm", "max_2d_mm", "total_mm"]


def daily_grid(day, x0, x1, y0, y1, cookies: Path, cache: Path) -> np.ndarray:
    """Daily rain (mm) on cells [x0..x1] x [y0..y1], shape (lon, lat)."""
    f = cache / f"imergday_{day:%Y%m%d}_{x0}_{x1}_{y0}_{y1}.txt"
    if not f.exists():
        for attempt in range(12):
            url = DAILY.format(host=HOSTS[attempt % len(HOSTS)], t=day, x0=x0, x1=x1, y0=y0, y1=y1)
            r = subprocess.run(["curl", "-s", "-g", "-L", "-m", "120", "--netrc-file", str(NETRC),
                                "-c", str(cookies), "-b", str(cookies), "-o", str(f), url], capture_output=True)
            if r.returncode == 0 and f.exists() and f.read_text(errors="ignore").lstrip().startswith("Dataset:"):   # error pages also echo the URL
                break
            f.unlink(missing_ok=True)
            time.sleep(min(2 * (attempt + 1), 20))
        else:
            raise RuntimeError(f"IMERG daily unavailable for {day:%Y-%m-%d}")
    rows = [line.split(",")[1:] for line in f.read_text().splitlines()
            if re.match(r"precipitation\.precipitation\[", line)]
    grid = np.array(rows, dtype=float)
    return np.where(grid < 0, np.nan, grid)


def spells(s: pd.Series) -> pd.DataFrame:
    """Wet spells: days >= WET_MM separated by at most GAP_DAYS dry days, kept if the total reaches MIN_SPELL_MM."""
    wet = s[s >= WET_MM]
    if wet.empty:
        return pd.DataFrame()
    new = wet.index.to_series().diff() > pd.Timedelta(days=GAP_DAYS + 1)
    out = []
    for _, block in wet.groupby(new.cumsum()):
        part = s[block.index[0]:block.index[-1]]
        if part.sum() >= MIN_SPELL_MM:
            out.append({"start": part.index[0], "end": part.index[-1], "total_mm": part.sum(),
                        "max_1d_mm": part.max(), "max_2d_mm": part.rolling(2, min_periods=1).sum().max()})
    return pd.DataFrame(out)


def scan(table: pd.DataFrame, col: str, thresholds) -> pd.DataFrame:
    pos, neg = table[table.flood == 1], table[table.flood == 0]
    zone_days = table.groupby("aoi").archive_days.first().sum()   # days of archive summed over zones
    return pd.DataFrame([{"threshold": t, "flood_zones_triggered": (pos[col] >= t).mean(),
                          "dry_spells_triggered": (neg[col] >= t).mean(),
                          "false_triggers_per_zone_per_year": (neg[col] >= t).sum() / zone_days * 365}
                         for t in thresholds])


def loo_thresholds(table: pd.DataFrame, col: str, target_recall: float) -> pd.DataFrame:
    """Leave-one-event-out: highest threshold that triggers at least `target_recall` of the other events' flood zones."""
    rows = []
    for ev in table.event_id.unique():
        train, test = table[table.event_id != ev], table[table.event_id == ev]
        pos = np.sort(train.loc[train.flood == 1, col].to_numpy())
        thr = float(pos[int(np.floor((1 - target_recall) * len(pos)))])
        rows.append({"event_id": ev, "threshold": thr,
                     "flood_zones_caught": (test.loc[test.flood == 1, col] >= thr).mean(),
                     "dry_spells": int((test.flood == 0).sum()),
                     "dry_spells_triggered": int((test.loc[test.flood == 0, col] >= thr).sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inventory_med_temporal.csv.gz")
    ap.add_argument("--out", default="reports/med_trigger")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(args.data, usecols=["event_id", "aoi", "imerg_cell", "offset_h", "prediction_time"])
    peaks = inv[inv.offset_h == 0].groupby("aoi").prediction_time.first()
    peaks = pd.to_datetime(peaks, utc=True, format="mixed").dt.tz_localize(None)
    cells = inv[inv.offset_h == 0].drop_duplicates(["aoi", "imerg_cell"])
    times = ems.activation_times()
    cookies = raw_dir("ems") / "earthdata_cookies.txt"
    rows = []
    for code, ev in cells.groupby("event_id", sort=False):
        xy = ev.imerg_cell.str.split("_", expand=True).astype(int)
        x0, x1, y0, y1 = xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()
        t0 = times[code].tz_localize(None).normalize()
        days = pd.date_range(t0 - pd.Timedelta(days=120), t0, freq="D")
        cache = raw_dir("ems", code)
        stack = np.stack([daily_grid(d, x0, x1, y0, y1, cookies, cache) for d in days])
        for aoi, z in ev.groupby("aoi"):
            zxy = z.imerg_cell.str.split("_", expand=True).astype(int)
            # Wettest cell of the zone each day: a trigger watches the worst part of the zone.
            s = pd.Series(np.nanmax(np.stack([stack[:, x - x0, y - y0] for x, y in zip(zxy[0], zxy[1])]), axis=0), index=days)
            sp = spells(s)
            peak_day = peaks[aoi].normalize()
            sp["flood"] = ((sp.start <= peak_day) & (sp.end >= peak_day)).astype(int)
            if sp.flood.sum() == 0:   # the peak hour fell on a dry day of the daily product: take the nearest spell
                sp.loc[(sp.start - peak_day).abs().idxmin(), "flood"] = 1
            sp["event_id"], sp["aoi"], sp["archive_days"] = code, aoi, len(days)
            rows.append(sp)
        print(f"{code}: {len(days)} days, {len(ev.aoi.unique())} zones", flush=True)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "spells.csv", index=False)
    print(f"\n{table.flood.sum()} flood spells, {(table.flood == 0).sum()} dry spells in {table.aoi.nunique()} zones")
    print(table.groupby("flood")[INDICATORS].describe(percentiles=[0.1, 0.5, 0.9]).T.round(1).to_string())
    for col in INDICATORS:
        loo = loo_thresholds(table, col, 0.9)
        loo.to_csv(out / f"loo_{col}.csv", index=False)
        print(f"\n{col}, threshold fitted for 90 % of training flood zones, checked on the held-out event:")
        print(f"  thresholds {loo.threshold.min():.0f}-{loo.threshold.max():.0f} mm, "
              f"held-out flood zones caught {loo.flood_zones_caught.mean():.0%}, "
              f"dry spells triggered {loo.dry_spells_triggered.sum()}/{loo.dry_spells.sum()}")


if __name__ == "__main__":
    main()
