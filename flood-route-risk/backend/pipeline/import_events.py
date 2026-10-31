"""Import historical flood events / black spots (FR-23) into the inventory.

    python -m pipeline.import_events --area nabeul events.csv
    python -m pipeline.import_events --area nabeul emsr319_extent.geojson --source "Copernicus EMS EMSR319" \
        --date 2018-09-22T18:00:00Z --reliability high --test-set

CSV columns: event_date, lat, lon, source, reliability (high|medium|low), and optionally
source_url, spatial_acc_m, duration_h, observed_rain_mm, notes, test_set (true/false).
GeoJSON: any geometry; missing attributes are filled from the command-line options.

Events marked --test-set are held out for validation: they never feed f6 or DV
(guide 8.2, circularity). After importing, re-run: python -m pipeline.run --area <a> --only features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["event_date", "duration_h", "spatial_acc_m", "source", "source_url", "reliability",
           "observed_rain_mm", "test_set", "notes", "geometry"]


def read_any(path: Path, args) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs=4326)
    else:
        gdf = gpd.read_file(path).to_crs(4326)
    defaults = {"event_date": args.date, "source": args.source or path.name,
                "reliability": args.reliability, "test_set": args.test_set}
    for col, val in defaults.items():
        if col not in gdf or gdf[col].isna().all():
            gdf[col] = val
    for col in COLUMNS:
        if col not in gdf:
            gdf[col] = None
    gdf["event_date"] = pd.to_datetime(gdf["event_date"], utc=True).astype(str)
    gdf["test_set"] = gdf["test_set"].astype(str).str.lower().isin(["true", "1", "yes"])
    bad = ~gdf["reliability"].isin(["high", "medium", "low"])
    if bad.any():
        raise SystemExit(f"{bad.sum()} rows have reliability outside high|medium|low")
    return gdf[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--area", default="nabeul")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--date"); ap.add_argument("--source")
    ap.add_argument("--reliability", default="medium", choices=["high", "medium", "low"])
    ap.add_argument("--test-set", action="store_true")
    args = ap.parse_args()

    out = args.data_dir / "processed" / args.area / "events.geojson"
    frames = [gpd.read_file(out)] if out.is_file() else []
    frames += [read_any(f, args) for f in args.files]
    allev = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    out.parent.mkdir(parents=True, exist_ok=True)
    allev.to_file(out, driver="GeoJSON")
    print(f"{len(allev)} events in {out} ({int(allev['test_set'].astype(bool).sum())} held out for validation)")
    print(f"Now re-run: python -m pipeline.run --area {args.area} --only features")


if __name__ == "__main__":
    main()
