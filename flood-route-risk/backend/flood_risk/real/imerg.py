"""Replace reanalysis rain in the satellite inventory by GPM IMERG rain (NASA, 0.1 degree, 30 min).

    python -m flood_risk.real.imerg

ERA5 (about 30 km) gave one rain value per mapped area and missed convective peaks: 4 to 14 mm
over Cap Bon on 22 September 2018 against more than 200 mm measured locally. IMERG merges
satellite microwave, infrared and rain gauges on a 0.1 degree grid, so rain varies inside an area.
Each segment takes the rain of its IMERG cell, at the worst hour of that cell during the storm.

Needs a NASA Earthdata login in ~/.netrc (or ~/_netrc on Windows) with the
"NASA GESDISC DATA ARCHIVE" application approved.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from . import ems, rain
from .areas import raw_dir

# gpm1 and gpm2 serve the same archive; whichever answers is used.
HOSTS = ["gpm2.gesdisc.eosdis.nasa.gov", "gpm1.gesdisc.eosdis.nasa.gov"]
OPENDAP = ("https://{host}/opendap/GPM_L3/GPM_3IMERGHH.07/{t:%Y}/{doy:03d}/"
           "3B-HHR.MS.MRG.3IMERG.{t:%Y%m%d}-S{t:%H%M%S}-E{e:%H%M%S}.{mn:04d}.V07B.HDF5.ascii"
           "?precipitation[0:0][{x0}:{x1}][{y0}:{y1}]")
CELL = 0.1
NETRC = next((p for p in [Path.home() / ".netrc", Path.home() / "_netrc"] if p.exists()), Path.home() / ".netrc")
RAIN_COLUMNS = ["rain_eta_mmh", "rain_1h_mm", "rain_3h_mm", "rain_24h_mm", "storm_total_mm"]


def cell_index(lon, lat):
    """IMERG grid indices: longitude from -180, latitude from -90, 0.1 degree cells."""
    return (np.floor((np.asarray(lon) + 180) / CELL).astype(int),
            np.floor((np.asarray(lat) + 90) / CELL).astype(int))


def half_hour(session: Path, t, x0, x1, y0, y1, cache) -> np.ndarray:
    """Rain rate (mm/h) on cells [x0..x1] x [y0..y1] for the half hour starting at t; `session` is a cookie file."""
    f = cache / f"imerg_{t:%Y%m%d%H%M}_{x0}_{x1}_{y0}_{y1}.txt"
    if not f.exists():
        for attempt in range(12):
            url = OPENDAP.format(host=HOSTS[attempt % len(HOSTS)], t=t, doy=t.dayofyear,
                                 e=t + pd.Timedelta(minutes=29, seconds=59),
                                 mn=t.hour * 60 + t.minute, x0=x0, x1=x1, y0=y0, y1=y1)
            # curl (Windows TLS stack) is used because the Earthdata servers reset Python's OpenSSL handshake here.
            r = subprocess.run(["curl", "-s", "-g", "-L", "-m", "120", "--netrc-file", str(NETRC),
                                "-c", str(session), "-b", str(session), "-o", str(f), url], capture_output=True)
            if r.returncode == 0 and f.exists() and f.read_text(errors="ignore").lstrip().startswith("Dataset:"):   # error pages also echo the URL
                break
            f.unlink(missing_ok=True)
            time.sleep(min(2 * (attempt + 1), 20))
        else:
            raise RuntimeError(f"IMERG unavailable for {t}")
    rows = [line.split(",")[1:] for line in f.read_text().splitlines() if re.match(r"precipitation\[0\]\[\d+\]", line)]
    grid = np.array(rows, dtype=float)
    return np.where(grid < 0, np.nan, grid)   # shape (lon, lat)


def event_rain(session, code, lon, lat, start, end, cache) -> dict:
    """Hourly rain series (mm) of every IMERG cell touched by the segments, between start and end."""
    xs, ys = cell_index(lon, lat)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    times = pd.date_range(start.floor("h"), end.ceil("h"), freq="30min", inclusive="left")
    stack = np.stack([half_hour(session, t, x0, x1, y0, y1, cache) for t in times])
    # Two half-hour rates (mm/h) make one hourly total (mm), labelled at the end of the hour like Open-Meteo.
    hourly = (stack[0::2] + stack[1::2]) / 2
    index = pd.date_range(times[0] + pd.Timedelta(hours=1), periods=len(hourly), freq="h")
    return {(x, y): pd.Series(np.nan_to_num(hourly[:, x - x0, y - y0]), index=index)
            for x, y in set(zip(xs, ys))}


def cell_features(series: pd.Series) -> dict:
    """Same definitions as rain.storm_features, at the wettest hour of the cell."""
    t, h = series.idxmax(), pd.Timedelta(hours=1)
    return {"rain_eta_mmh": series[t], "rain_1h_mm": series[t - h:t - h].sum(),
            "rain_3h_mm": series[t - 3 * h:t - h].sum(), "rain_24h_mm": series[t - 24 * h:t - h].sum(),
            "storm_total_mm": series.sum(), "peak_time": t}


def storm_window(code, ev: pd.DataFrame, geoms: dict, t0) -> tuple:
    """(start, end, storm_start, lon, lat) of an event: the reanalysis storm of each area, 24 h before to 12 h after."""
    lon, lat, windows = np.zeros(len(ev)), np.zeros(len(ev)), []
    cache = raw_dir("ems", code)
    for tag, idx in ev.groupby("aoi").indices.items():
        g = geoms[tag]
        crs = ems.utm_crs(g.centroid.x, g.centroid.y)
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon[idx], lat[idx] = to_wgs.transform(ev["x_km"].to_numpy()[idx] * 1000, ev["y_km"].to_numpy()[idx] * 1000)
        # The storm found in the reanalysis gives the window; IMERG then finds each cell's own peak.
        series = rain.hourly_rain(None, None, None, None, cache / f"rain_{tag.split('_', 1)[1]}.json")
        storms = rain.storms(series, min_total=5.0)
        near = storms[(storms.peak_time >= t0 - pd.Timedelta(days=10)) & (storms.peak_time <= t0 + pd.Timedelta(days=1))]
        s = near.sort_values("total_mm").iloc[-1]
        windows.append((s["start"], s["end"]))
    storm_start = min(w[0] for w in windows)
    return storm_start - pd.Timedelta(hours=24), max(w[1] for w in windows) + pd.Timedelta(hours=12), storm_start, lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inventory_med.csv.gz")
    ap.add_argument("--out", default="data/inventory_med_imerg.csv.gz")
    args = ap.parse_args()
    inv = pd.read_csv(args.data)
    times = ems.activation_times()
    session = raw_dir("ems") / "earthdata_cookies.txt"
    parts = []
    for code, ev in inv.groupby("event_id", sort=False):
        cache = raw_dir("ems", code)
        geoms = {f"{code}_{it['aoi']}": it["geometry"] for it in ems.mapped_areas(code, cache)}
        start, end, _, lon, lat = storm_window(code, ev, geoms, times[code])
        print(f"{code}: IMERG {start:%Y-%m-%d %H}h -> {end:%Y-%m-%d %H}h, {len(ev)} segments", flush=True)
        cells = event_rain(session, code, lon, lat, start, end, cache)
        feats = {k: cell_features(v) for k, v in cells.items()}
        xs, ys = cell_index(lon, lat)
        ev = ev.copy()
        for col in RAIN_COLUMNS:
            ev[col] = [feats[(x, y)][col] for x, y in zip(xs, ys)]
        ev["lon"], ev["lat"], ev["imerg_cell"] = lon, lat, [f"{x}_{y}" for x, y in zip(xs, ys)]
        ev["rain_interpolated"] = 0   # 30-min satellite product, no hourly-to-15-min interpolation
        print(f"  {len(cells)} cells, peak {ev.rain_eta_mmh.min():.1f}-{ev.rain_eta_mmh.max():.1f} mm/h, "
              f"total {ev.storm_total_mm.min():.0f}-{ev.storm_total_mm.max():.0f} mm", flush=True)
        parts.append(ev)
        pd.concat(parts).to_csv(args.out, index=False)   # keep progress
    out = pd.concat(parts)
    out.to_csv(args.out, index=False)
    print(out.groupby("event_id").agg(cells=("imerg_cell", "nunique"), peak=("rain_eta_mmh", "max"),
                                      total=("storm_total_mm", "max"), flooded=("flooded", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
