"""Flood inventory for V2 from Copernicus EMS satellite maps of Mediterranean flash floods.

    python -m flood_risk.real.ems

The specification asks V2 to learn from "documented events and satellite labels". Tunisia has
one mapped flood (EMSR319, Nabeul 2018), too few to train on, so the inventory adds flash
floods mapped the same way in similar climates (Libya, Spain, Greece, Italy, southern France).
Each road segment of a mapped area is one row; it is flooded when it touches (15 m) a mapped
"flooded area" or "flood trace" polygon. Maps are drawn from images taken after the storm:
they show where water spread, not street ponding at a given hour.
"""

import argparse
import io
import math
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import wkt

from . import osm, rain, terrain
from .areas import raw_dir

# Flash floods in Mediterranean or semi-arid settings, 2017 onwards (Open-Meteo rain archive).
EVENTS = {
    "EMSR319": "Nabeul, Tunisia, Sept 2018",
    "EMSR696": "Storm Daniel, Libya, Sept 2023",
    "EMSR773": "Valencia DANA, Spain, Oct 2024",
    "EMSR388": "South-east Spain DANA, Sept 2019",
    "EMSR569": "Valencian Community, Spain, Mar 2022",
    "EMSR326": "Malaga flash flood, Spain, Oct 2018",
    "EMSR327": "Andalusia flash flood, Spain, Oct 2018",
    "EMSR323": "Mallorca flash flood, Spain, Oct 2018",
    "EMSR324": "Aude flash flood, France, Oct 2018",
    "EMSR257": "Mandra flash flood, Attica, Greece, Nov 2017",
    "EMSR413": "Western Attica, Greece, Nov 2019",
    "EMSR480": "Crete, Greece, Nov 2020",
    "EMSR933": "Crete, Greece, Oct 2026",
    "EMSR548": "Eastern Sicily medicane, Italy, Oct 2021",
    "EMSR649": "Sicily, Italy, Feb 2023",
}
LIST_API = "https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=500"
NEW_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={}"
OLD_PAGE = "https://mapping.emergency.copernicus.eu/activations/{}/"
MAX_AOIS = 3
MAX_AOI_KM2 = 1500
OSM_COMPLETENESS = 0.5   # unknown outside the pilot: neutral value


def utm_crs(lon, lat):
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1 if lat >= 0 else 32700 + int((lon + 180) // 6) + 1}"


def activation_times() -> dict:
    out, url = {}, LIST_API
    while url:
        js = requests.get(url.replace("http://", "https://"), timeout=120).json()
        out.update({r["code"]: pd.Timestamp(r["activationTime"]) for r in js["results"]})
        url = js["next"]
    return out


def _priority(name: str) -> int:
    name = name.upper()
    if "MONIT" in name:
        return 0
    if "DEL_PRODUCT" in name or "DELINEATION" in name:
        return 3
    if "GRA_PRODUCT" in name or "GRADING" in name:
        return 2
    return 1 if "FEP" in name else 0


def _version(name: str) -> int:
    v = re.findall(r"_v(\d+)", name)
    return int(v[-1]) if v else 0


def _from_zip(path: Path):
    names = zipfile.ZipFile(path).namelist()
    def layer(*keys):
        hit = [n for n in names if n.lower().endswith(".shp") and all(k in n.lower().replace("_", "") for k in keys)]
        return gpd.read_file(f"zip://{path}!{hit[0]}").to_crs("EPSG:4326") if hit else None
    return layer("areaofinterest"), layer("observedevent")


def mapped_areas(code: str, cache: Path) -> list[dict]:
    """[{aoi, geometry (WGS84), flood (GeoDataFrame)}] for the best product of each area of interest."""
    out = []
    js = requests.get(NEW_API.format(code), timeout=120).json()
    js = js["results"][0] if "results" in js else js
    if js.get("aois"):
        for a in js["aois"]:
            prods = [p for p in a["products"] if not p.get("monitoring") and _priority(p["type"] + "_PRODUCT") > 0]
            prods = sorted(prods, key=lambda p: (_priority(p["type"] + "_PRODUCT"), p["version"]["number"]), reverse=True)
            for p in prods:
                obs = [l for l in p["layers"] if "observedEvent" in l["name"] and l.get("json")]
                if obs:
                    f = cache / f"{code}_AOI{a['number']:02d}_observed.json"
                    if not f.exists():
                        f.write_bytes(requests.get(obs[0]["json"], timeout=300).content)
                    out.append({"aoi": f"AOI{a['number']:02d}", "geometry": wkt.loads(a["extent"]),
                                "flood": gpd.read_file(f).set_crs("EPSG:4326", allow_override=True)})
                    break
        return out
    html = requests.get(OLD_PAGE.format(code), timeout=120).text
    zips = sorted(set(re.findall(r'https://[^"]+_vector\.zip', html)))
    best = {}
    for z in zips:
        m = re.search(rf"{code}_(AOI\d+|\d+[A-Z]+)_", z)
        if m and _priority(z) > 0:
            key = m.group(1)
            if key not in best or (_priority(z), _version(z)) > (_priority(best[key]), _version(best[key])):
                best[key] = z
    for key, z in sorted(best.items()):
        f = cache / z.rsplit("/", 1)[-1]
        if not f.exists():
            f.write_bytes(requests.get(z, timeout=300).content)
        aoi, flood = _from_zip(f)
        if aoi is not None and flood is not None and len(flood):
            out.append({"aoi": key, "geometry": aoi.geometry.union_all(), "flood": flood})
    return out


def build_area(code, item, t0) -> pd.DataFrame | None:
    geom = item["geometry"]
    lon, lat = geom.centroid.x, geom.centroid.y
    crs = utm_crs(lon, lat)
    km2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs).area.iloc[0] / 1e6
    if km2 > MAX_AOI_KM2:
        print(f"  {code} {item['aoi']}: {km2:.0f} km2, skipped (larger than {MAX_AOI_KM2})")
        return None
    west, south, east, north = geom.bounds
    tag = f"{code}_{item['aoi']}"
    area = {"name": tag, "south": south, "north": north, "west": west, "east": east, "crs": crs}
    d = raw_dir("ems", code)

    series = rain.hourly_rain(lat, lon, f"{t0 - pd.Timedelta(days=120):%Y-%m-%d}", f"{t0 + pd.Timedelta(days=1):%Y-%m-%d}",
                              d / f"rain_{item['aoi']}.json")
    events = rain.storms(series, min_total=5.0)
    # Activation can come a week after the storm (Nabeul: 22 -> 29 September): take the largest storm before it.
    near = events[(events.peak_time >= t0 - pd.Timedelta(days=10)) & (events.peak_time <= t0 + pd.Timedelta(days=1))]
    if near.empty:
        print(f"  {tag}: no storm >= 5 mm in the rain archive around {t0:%Y-%m-%d}, skipped")
        return None
    storm = near.sort_values("total_mm").iloc[-1]

    seg = osm.build_segments(area, d, clip=geom)
    if seg.empty:
        return None
    seg = terrain.add_terrain(seg, area, d)
    flood = item["flood"].to_crs(crs).geometry.buffer(15).union_all()
    seg["flooded"] = seg.geometry.intersects(flood).astype(int)
    for k, v in rain.storm_features(series, storm).items():
        seg[k] = v
    for col, src in [("elevation_rel_pct", "elevation_rel_m"), ("dist_wadi_pct", "dist_wadi_m"),
                     ("twi_pct", "twi"), ("impervious_pct", "impervious_frac")]:
        seg[col] = seg[src].rank(pct=True)
    seg["osm_completeness"] = OSM_COMPLETENESS
    seg["black_spot_count"] = 0
    seg["event_id"], seg["aoi"], seg["slot_min"] = code, tag, 0
    seg["storm_total_mm"] = storm["total_mm"]
    seg["rain_interpolated"], seg["data_age_min"] = 1, 60
    seg["x_km"], seg["y_km"] = seg["mid"].x / 1000, seg["mid"].y / 1000
    print(f"  {tag}: {len(seg)} segments, {seg.flooded.sum()} flooded ({seg.flooded.mean():.1%}), "
          f"storm total {storm['total_mm']:.0f} mm, peak {storm['peak_mmh']:.1f} mm/h", flush=True)
    return pd.DataFrame(seg.drop(columns=["geometry", "mid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", default=list(EVENTS))
    ap.add_argument("--out", default="data/inventory_med.csv.gz")
    args = ap.parse_args()
    times = activation_times()
    parts = []
    for code in args.events:
        print(f"{code}: {EVENTS.get(code, '')}", flush=True)
        try:
            items = mapped_areas(code, raw_dir("ems", code))
        except Exception as e:   # one unreachable activation must not stop the inventory
            print(f"  {code}: products unavailable ({e})")
            continue
        # Keep every Tunisian area (the validation target); elsewhere the most flooded areas only.
        limit = None if code == "EMSR319" else MAX_AOIS
        items = sorted(items, key=lambda it: it["flood"].to_crs(3857).area.sum(), reverse=True)[:limit]
        for item in items:
            try:
                part = build_area(code, item, times[code])
            except Exception as e:
                print(f"  {code} {item['aoi']}: failed ({e})")
                continue
            if part is not None:
                parts.append(part)
                pd.concat(parts, ignore_index=True).to_csv(args.out, index=False)   # keep progress
    inv = pd.concat(parts, ignore_index=True)
    inv.to_csv(args.out, index=False)
    print(f"\ninventory: {len(inv):,} segments, {inv.event_id.nunique()} events, {inv.aoi.nunique()} areas, "
          f"{inv.flooded.sum()} flooded ({inv.flooded.mean():.2%})")
    print(inv.groupby("event_id").agg(areas=("aoi", "nunique"), segments=("flooded", "size"),
                                      flooded=("flooded", "sum"), rain_mm=("storm_total_mm", "max")).to_string())


if __name__ == "__main__":
    main()
