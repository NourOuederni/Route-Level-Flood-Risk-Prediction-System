"""Training inventory from NYC 311 street-flooding reports (Southeast Queens pilot).

    python -m flood_risk.real.nyc

One row per road segment x storm. A segment is labelled flooded when a street or highway
flooding report falls within 40 m of it between the start of the storm and 12 h after its end
(reports are made after the water is seen, so the exact hour is not reliable).

Sources: 311 reports (labels and history), NYC DEP catch basins (drain density), measured
hourly rain at JFK and LaGuardia interpolated to each segment (ERA5 fills station gaps),
OpenStreetMap roads, USGS 3DEP 10 m bare-earth DEM and ESA WorldCover.
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from . import osm, rain, terrain
from .areas import NYC, raw_dir

SOCRATA_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
SOCRATA_BASINS = "https://data.cityofnewyork.us/resource/2w2g-fk3i.csv"
FLOOD = ["Street Flooding (SJ)", "Flooding on Street", "Highway Flooding (SH)", "Flooding on Highway"]
BASIN = ["Catch Basin Clogged/Flooding (Use Comments) (SC)"]
MATCH_M = 40
NEIGHBOUR_M = 250
BASIN_M = 30
REPORT_DELAY = pd.Timedelta(hours=12)
KEEP = ["segment_id", "x_km", "y_km", "elevation_rel_m", "slope_pct", "twi", "depression_depth_m", "dist_wadi_m",
        "impervious_frac", "drain_inlet_density", "osm_completeness", "underpass", "ford"]


def socrata(url, where, select, cache):
    if not cache.exists():
        pages, offset = [], 0
        while True:
            r = requests.get(url, params={"$select": select, "$where": where, "$order": ":id",
                                          "$limit": 50000, "$offset": offset}, timeout=300)
            r.raise_for_status()
            page = pd.read_csv(pd.io.common.StringIO(r.text))
            pages.append(page)
            if len(page) < 50000:
                break
            offset += 50000
        pd.concat(pages).to_csv(cache, index=False)
    return pd.read_csv(cache)


def _box(area):
    return (f"latitude between {area['south']} and {area['north']} "
            f"AND longitude between {area['west']} and {area['east']}")


def complaints(area, cache) -> pd.DataFrame:
    kinds = " OR ".join(f"descriptor='{d}'" for d in FLOOD + BASIN)
    df = socrata(SOCRATA_311, f"({kinds}) AND {_box(area)}",
                 "unique_key,created_date,descriptor,latitude,longitude", cache).dropna(subset=["latitude", "longitude"])
    local = pd.to_datetime(df["created_date"])
    df["time_utc"] = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
                      .dt.tz_convert("UTC").dt.tz_localize(None))
    df["kind"] = np.where(df["descriptor"].isin(FLOOD), "flood", "basin")
    return df.dropna(subset=["time_utc"])


def catch_basins(area, cache) -> pd.DataFrame:
    return socrata(SOCRATA_BASINS, _box(area), "unitid,latitude,longitude", cache).dropna()


def _points(df, crs):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(crs)


def match_to_segments(df, seg) -> pd.DataFrame:
    hit = gpd.sjoin_nearest(_points(df, seg.crs), seg[["segment_id", "geometry"]], max_distance=MATCH_M, how="inner")
    return hit.drop_duplicates("unique_key")[["unique_key", "time_utc", "kind", "segment_id"]]


def neighbour_reports(reports, seg) -> pd.DataFrame:
    """(segment_id, time_utc) pairs for every flood report within NEIGHBOUR_M of the segment midpoint."""
    pts = _points(reports, seg.crs)
    mids = gpd.GeoDataFrame({"segment_id": seg["segment_id"]}, geometry=seg["mid"].buffer(NEIGHBOUR_M), crs=seg.crs)
    return gpd.sjoin(pts[["time_utc", "geometry"]], mids, predicate="within")[["segment_id", "time_utc"]]


def station_rain(area, start, end, d) -> tuple[pd.DataFrame, pd.Series]:
    """Measured hourly rain per station, gaps filled with the other station then ERA5; and the area mean."""
    st = rain.asos_hourly(area["stations"], start, end, d / "asos_hourly.csv")
    print("hours with a station report:", st.notna().mean().round(3).to_dict())
    lat, lon = (area["south"] + area["north"]) / 2, (area["west"] + area["east"]) / 2
    era5 = rain.hourly_rain(lat, lon, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", d / "rain_hourly.json")
    era5 = era5.reindex(st.index)
    mean = st.mean(axis=1)
    st = st.apply(lambda c: c.fillna(mean)).apply(lambda c: c.fillna(era5)).fillna(0)
    return st, st.mean(axis=1)


def build(min_storm_mm: float):
    d = raw_dir("nyc")
    seg = osm.build_segments(NYC, d)
    seg = terrain.add_terrain(seg, NYC, d)
    seg["osm_completeness"] = NYC["osm_completeness"]

    basins = _points(catch_basins(NYC, d / "catch_basins.csv"), seg.crs)
    buf = gpd.GeoDataFrame({"segment_id": seg["segment_id"]}, geometry=seg.geometry.buffer(BASIN_M), crs=seg.crs)
    per_seg = gpd.sjoin(basins[["geometry"]], buf, predicate="within").groupby("segment_id").size()
    seg["drain_inlet_density"] = (seg["segment_id"].map(per_seg).fillna(0) / (seg["length_m"] / 1000)).clip(upper=80)

    raw_reports = complaints(NYC, d / "311_flooding.csv")
    reports = match_to_segments(raw_reports, seg)
    flood, basin = reports[reports.kind == "flood"], reports[reports.kind == "basin"]
    near = neighbour_reports(raw_reports[raw_reports.kind == "flood"], seg)

    first, last = reports["time_utc"].min(), reports["time_utc"].max()
    stations, series = station_rain(NYC, first - pd.Timedelta(days=90), last, d)
    events = rain.storms(series, min_total=min_storm_mm)
    events = events[(events.start >= first) & (events.end + REPORT_DELAY <= last)].reset_index(drop=True)

    station_xy = {name: tuple(gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(seg.crs)
                              .iloc[0].coords[0]) for name, (lat, lon) in NYC["stations"].items()}
    weights = rain.idw_weights(seg["mid"].x.to_numpy(), seg["mid"].y.to_numpy(), station_xy)

    static = pd.DataFrame(seg.drop(columns=["geometry", "mid"]))
    static["x_km"] = (seg["mid"].x - seg["mid"].x.min()) / 1000
    static["y_km"] = (seg["mid"].y - seg["mid"].y.min()) / 1000
    static["ford"] = static["ford"].astype(int)
    static = static[KEEP]

    def prior(df, ev, since=None):
        sel = df[(df.time_utc < ev.start) & ((df.time_utc >= since) if since is not None else True)]
        return static["segment_id"].map(sel.groupby("segment_id").size()).fillna(0).to_numpy()

    rows = []
    for ev in events.itertuples():
        e = ev._asdict()
        part = static.copy()
        feats = pd.DataFrame({name: rain.storm_features(stations[name], e) for name in stations.columns})
        for k in ["rain_eta_mmh", "rain_1h_mm", "rain_3h_mm", "rain_24h_mm"]:
            part[k] = (weights[feats.columns] * feats.loc[k].to_numpy()).sum(axis=1).to_numpy()
        part["dry_spell_days"] = rain.storm_features(series, e)["dry_spell_days"]
        in_storm = flood[(flood.time_utc >= ev.start) & (flood.time_utc <= ev.end + REPORT_DELAY)]
        part["flooded"] = part["segment_id"].isin(in_storm["segment_id"]).astype(int)
        # History only counts reports made before the storm, so the label never leaks into the features.
        part["black_spot_count"] = prior(flood, ev)
        part["flood_reports_365d"] = prior(flood, ev, ev.start - pd.Timedelta(days=365))
        part["flood_reports_250m"] = prior(near, ev)
        part["basin_reports_prior"] = prior(basin, ev)
        part["event_id"], part["slot_min"], part["storm_total_mm"] = ev.event_id, 0, ev.total_mm
        rows.append(part)
    inv = pd.concat(rows, ignore_index=True)
    inv["rain_interpolated"], inv["data_age_min"] = 0, 60   # measured rain, hourly
    floats = inv.select_dtypes("float64").columns
    inv[floats] = inv[floats].astype("float32")

    out = raw_dir("..", "real")
    inv.to_csv("data/inventory_nyc.csv.gz", index=False)
    events.to_csv(out / "nyc_storms.csv", index=False)
    seg_out = seg.drop(columns=["mid"]).copy()
    seg_out["flood_reports"] = seg_out["segment_id"].map(flood.groupby("segment_id").size()).fillna(0)
    seg_out["basin_reports"] = seg_out["segment_id"].map(basin.groupby("segment_id").size()).fillna(0)
    seg_out.to_file(out / "nyc_segments.gpkg", driver="GPKG")

    print(f"segments: {len(seg)} | catch basins: {len(basins)} | storms >= {min_storm_mm} mm: {len(events)} | rows: {len(inv):,}")
    print(f"reports matched to a segment: {len(flood)} flood, {len(basin)} catch basin")
    print(f"flooded rows: {inv['flooded'].sum()} ({inv['flooded'].mean():.3%})")
    return inv, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-storm-mm", type=float, default=15.0)
    build(ap.parse_args().min_storm_mm)


if __name__ == "__main__":
    main()
