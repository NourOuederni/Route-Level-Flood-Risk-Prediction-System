"""Build the terrain table of one zone for the V2 terrain model (docs/MODEL_CONTRACT.md, sections 2.1 to 2.3).

Run from backend/ after copying flood_risk/ and models/final/ into it:

    python -m flood_risk.real.build_zone --out data/zones/nabeul.csv

The output has one row per road segment of the zone. It can be passed as --area and, for a
first test, as --route (the same file works for both).

Not tested yet: it needs Overpass (roads) and the public Copernicus / ESA raster buckets.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import osm, terrain
from .areas import NABEUL_CRS, NABEUL_OSM_COMPLETENESS

# First run: the rectangle from areas.yaml. The handoff's validation used the Copernicus EMS
# area of interest for Nabeul, whose outline is not in the repo.
NABEUL_BOX = {"name": "nabeul", "south": 36.41, "north": 36.50, "west": 10.67, "east": 10.79,
              "crs": NABEUL_CRS}

# Columns the model reads directly (terrain features and percentiles), plus the metadata the
# confidence rules and the output join need.
OUTPUT_COLUMNS = [
    "segment_id", "aoi", "mid_lat", "mid_lon", "eta_min",
    "elevation_rel_m", "slope_pct", "twi", "depression_depth_m", "dist_wadi_m",
    "underpass", "ford", "elevation_rel_pct", "dist_wadi_pct", "twi_pct",
    "impervious_frac", "drain_inlet_density", "osm_completeness", "black_spot_count",
    "dry_spell_days", "rain_interpolated", "data_age_min",
]


def build(area: dict, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    seg = osm.build_segments(area, cache)
    seg = terrain.add_terrain(seg, area, cache)

    seg["aoi"] = area["name"]
    # Percentile of each segment inside its zone, 0..1 (contract section 2.3)
    seg["elevation_rel_pct"] = seg.groupby("aoi")["elevation_rel_m"].rank(pct=True)
    seg["dist_wadi_pct"] = seg.groupby("aoi")["dist_wadi_m"].rank(pct=True)
    seg["twi_pct"] = seg.groupby("aoi")["twi"].rank(pct=True)

    # Metadata used by the confidence rules and the output (not model inputs)
    seg["osm_completeness"] = NABEUL_OSM_COMPLETENESS
    seg["black_spot_count"] = 0          # no georeferenced flood history for this zone yet
    seg["dry_spell_days"] = 0.0          # no rain history: the terrain model does not use it
    seg["rain_interpolated"] = 0
    seg["data_age_min"] = 0
    seg["eta_min"] = 0.0

    mid_ll = gpd.GeoSeries(seg["mid"], crs=area["crs"]).to_crs(4326)
    seg["mid_lat"], seg["mid_lon"] = mid_ll.y.to_numpy(), mid_ll.x.to_numpy()
    seg["underpass"] = seg["underpass"].astype(int)
    seg["ford"] = seg["ford"].astype(int)
    return seg[OUTPUT_COLUMNS].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/zones/nabeul.csv")
    ap.add_argument("--cache", default="data/raw/zones")
    args = ap.parse_args()

    table = build(NABEUL_BOX, Path(args.cache))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"{len(table)} segments written to {out}")
    print(table[["elevation_rel_m", "slope_pct", "twi", "dist_wadi_m", "impervious_frac"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
