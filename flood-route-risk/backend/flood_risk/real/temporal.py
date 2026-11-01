"""Observations at several prediction times, built only from what is known at each time.

    python -m flood_risk.real.temporal

The satellite labels say whether a road flooded during an event, not when. The question that can be
evaluated honestly with them is: at time t, knowing the rain observed until t and a forecast of the
next hours, which roads will flood during the rest of this event? Each segment gets one row per
prediction time, from 12 h before to the hour of its area's heaviest rain.

- Observed rain uses only hours ending at or before t (IMERG Final here: in operation, gauges,
  radar or IMERG Early, which are faster and less accurate).
- Forecast rain is the rain that really fell after t times a log-normal error that grows with the
  horizon. No archived forecast exists for these storms, so it is a simulated forecast and is kept in
  separate columns, so models can be trained with or without it.
- The hour of the heaviest rain is found afterwards over the whole storm; it only places the
  prediction times and never enters a feature.
"""

import argparse

import numpy as np
import pandas as pd

from . import ems, imerg
from .areas import raw_dir

OFFSETS_H = (-12, -6, -3, -1, 0)
FORECAST_SIGMA = {"rain_eta_mmh": 0.5, "rain_next_6h_mm": 0.7}   # log-normal spread of the simulated forecast
STATIC = ["segment_id", "event_id", "aoi", "flooded", "elevation_rel_m", "slope_pct", "twi", "depression_depth_m",
          "dist_wadi_m", "impervious_frac", "drain_inlet_density", "osm_completeness", "underpass", "ford",
          "black_spot_count", "dry_spell_days", "elevation_rel_pct", "dist_wadi_pct", "twi_pct", "impervious_pct",
          "rain_interpolated", "data_age_min", "imerg_cell"]


def rain_at(s: pd.Series, t, storm_start) -> dict:
    """Observed inputs at t from an hourly series labelled at the end of each hour."""
    h = pd.Timedelta(hours=1)
    return {"rain_1h_mm": s.get(t, 0.0), "rain_3h_mm": s[t - 2 * h:t].sum(), "rain_24h_mm": s[t - 23 * h:t].sum(),
            "rain_event_mm": s[storm_start:t].sum(),
            "rain_eta_mmh": s.get(t + h, 0.0), "rain_next_6h_mm": s[t + h:t + 6 * h].sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inventory_med_imerg.csv.gz")
    ap.add_argument("--out", default="data/inventory_med_temporal.csv.gz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    inv = pd.read_csv(args.data)
    times = ems.activation_times()
    session = raw_dir("ems") / "earthdata_cookies.txt"
    rng = np.random.default_rng(args.seed)
    parts = []
    for code, ev in inv.groupby("event_id", sort=False):
        cache = raw_dir("ems", code)
        geoms = {f"{code}_{it['aoi']}": it["geometry"] for it in ems.mapped_areas(code, cache)}
        start, end, storm_start, _, _ = imerg.storm_window(code, ev, geoms, times[code])
        cells = imerg.event_rain(session, code, ev["lon"].to_numpy(), ev["lat"].to_numpy(), start, end, cache)
        key = [f"{x}_{y}" for x, y in cells]
        series = dict(zip(key, cells.values()))
        for aoi, seg in ev.groupby("aoi"):
            # Heaviest hour of the area, weighting each cell by its number of road segments.
            weights = seg["imerg_cell"].value_counts()
            area = sum(series[c] * w for c, w in weights.items()) / weights.sum()
            peak = area.idxmax()
            for off in OFFSETS_H:
                t = peak + pd.Timedelta(hours=off)
                feats = pd.DataFrame({c: rain_at(series[c], t, storm_start) for c in weights.index}).T
                for col, sigma in FORECAST_SIGMA.items():   # one forecast error per cell and time, shared by its roads
                    feats[col] *= rng.lognormal(-sigma ** 2 / 2, sigma, len(feats))
                rows = seg[STATIC].join(feats, on="imerg_cell")
                rows["offset_h"], rows["prediction_time"] = off, t
                parts.append(rows)
            print(f"{aoi}: peak {peak:%Y-%m-%d %H}h, area rain at peak {area[peak]:.1f} mm/h, "
                  f"before t-12h {area[:peak - pd.Timedelta(hours=12)].sum():.0f} mm", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\n{len(out):,} rows = {out.segment_id.nunique():,} segments x {len(OFFSETS_H)} prediction times")
    print(out.groupby("offset_h")[["rain_1h_mm", "rain_24h_mm", "rain_event_mm", "rain_eta_mmh", "rain_next_6h_mm"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
