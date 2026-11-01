"""Hourly rainfall from the Open-Meteo historical archive and storm detection (FR-05).

The archive blends ERA5 reanalysis and ECMWF IFS analyses (9 to 25 km, hourly): every
segment of a small area gets the same rain, coarser than the 15-min target of the spec.
Where airport stations exist (New York), measured hourly rain replaces it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def hourly_rain(lat, lon, start, end, cache: Path, model: str | None = None) -> pd.Series:
    """Hourly precipitation (mm), index in UTC. `model` picks an Open-Meteo archive model (e.g. ecmwf_ifs)."""
    if cache.exists():
        js = json.loads(cache.read_text())
    else:
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "hourly": "precipitation", "timezone": "GMT"}
        if model:
            params["models"] = model
        r = requests.get(ARCHIVE, params=params, timeout=300)
        r.raise_for_status()
        js = r.json()
        cache.write_text(json.dumps(js))
    s = pd.Series(js["hourly"]["precipitation"], index=pd.to_datetime(js["hourly"]["time"]), dtype=float)
    return s.dropna()


def asos_hourly(stations: dict, start, end, cache: Path) -> pd.DataFrame:
    """Measured hourly rain (mm) at airport weather stations (Iowa Environmental Mesonet archive).

    One column per station, hourly UTC index; NaN where the station did not report. A routine
    report at hh:51 holds the rain of the previous hour and is assigned to the next full hour,
    the same convention as Open-Meteo.
    """
    if not cache.exists():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        params = [("station", s) for s in stations] + [
            ("data", "p01i"), ("year1", start.year), ("month1", start.month), ("day1", start.day),
            ("year2", end.year), ("month2", end.month), ("day2", end.day), ("tz", "Etc/UTC"),
            ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"), ("trace", "0.0001"), ("report_type", "3")]
        r = requests.get(IEM_ASOS, params=params, timeout=900)
        r.raise_for_status()
        cache.write_text(r.text)
    raw = pd.read_csv(cache, na_values=["M"])
    raw["hour"] = pd.to_datetime(raw["valid"]).dt.ceil("h")
    raw["mm"] = raw["p01i"].astype(float) * 25.4
    table = raw.groupby(["hour", "station"])["mm"].max().unstack("station")
    full = pd.date_range(table.index.min(), table.index.max(), freq="h")
    return table.reindex(full)


def idw_weights(xs, ys, station_xy: dict, power=2.0) -> pd.DataFrame:
    """Inverse-distance weights of each station for each point (rows sum to 1)."""
    w = pd.DataFrame({name: 1 / np.maximum(np.hypot(xs - x, ys - y), 100.0) ** power
                      for name, (x, y) in station_xy.items()})
    return w.div(w.sum(axis=1), axis=0)


def storms(rain: pd.Series, min_total=15.0, gap_h=6, wet_mm=0.2) -> pd.DataFrame:
    """Storms = wet hours separated by less than gap_h dry hours, kept if total >= min_total mm."""
    wet = rain[rain >= wet_mm]
    if wet.empty:
        return pd.DataFrame()
    new = wet.index.to_series().diff() > pd.Timedelta(hours=gap_h)
    rows = []
    for _, block in wet.groupby(new.cumsum()):
        start, end = block.index[0], block.index[-1]
        total = rain[start:end].sum()
        if total >= min_total:
            rows.append({"start": start, "end": end, "total_mm": total,
                         "peak_time": rain[start:end].idxmax(), "peak_mmh": rain[start:end].max()})
    out = pd.DataFrame(rows)
    out["event_id"] = [f"S{t:%Y%m%d%H}" for t in out["start"]]
    return out


def storm_features(rain: pd.Series, storm) -> dict:
    """Rain features at the storm peak, i.e. the ETA of a driver crossing at the worst hour."""
    t = storm["peak_time"]
    h = pd.Timedelta(hours=1)
    daily = rain.resample("D").sum()
    before = daily[daily.index < storm["start"].normalize()]
    wet_days = before[before >= 5.0]
    dry = (storm["start"].normalize() - wet_days.index[-1]).days if len(wet_days) else 60
    return {
        "rain_eta_mmh": rain.get(t, 0.0),
        "rain_1h_mm": rain[t - h:t - h].sum(),
        "rain_3h_mm": rain[t - 3 * h:t - h].sum(),
        "rain_24h_mm": rain[t - 24 * h:t - h].sum(),
        "dry_spell_days": float(dry),
    }
