"""Validation inventory for the 22 September 2018 Cap Bon (Nabeul) flash floods, and transfer test.

    python -m flood_risk.real.nabeul

Labels come from the Copernicus EMS delineation EMSR319: a road segment is flooded when it
touches (15 m) a mapped "flooded area" or "flood trace" polygon. The maps were drawn from
satellite images taken days after the storm, so they show where water spread or left traces,
not street ponding at a given hour. Segments outside mapped polygons are taken as not flooded.
"""

import zipfile
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import v1
from . import osm, rain, terrain
from .areas import EMSR319_PRODUCTS, EMSR319_URL, NABEUL_CRS, NABEUL_OSM_COMPLETENESS, raw_dir

STORM_DAY = pd.Timestamp("2018-09-22")


def _layer(zip_path: Path, suffix: str) -> gpd.GeoDataFrame:
    name = next(n for n in zipfile.ZipFile(zip_path).namelist() if n.endswith(f"{suffix}.shp"))
    return gpd.read_file(f"zip://{zip_path}!{name}").to_crs("EPSG:4326")


def emsr_zip(product: str) -> Path:
    path = raw_dir("emsr319") / f"{product}.zip"
    if not path.exists():
        r = requests.get(EMSR319_URL.format(product), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def build_aoi(product: str) -> pd.DataFrame:
    z = emsr_zip(product)
    aoi = _layer(z, "area_of_interest_a").geometry.union_all()
    flood = _layer(z, "observed_event_a")
    west, south, east, north = aoi.bounds
    tag = product.split("_")[0]
    area = {"name": f"nabeul_{tag}", "south": south, "north": north, "west": west, "east": east, "crs": NABEUL_CRS}
    d = raw_dir("nabeul")

    seg = osm.build_segments(area, d, clip=aoi)
    seg = terrain.add_terrain(seg, area, d)
    flood_utm = flood.to_crs(NABEUL_CRS).geometry.buffer(15).union_all()
    seg["flooded"] = seg.geometry.intersects(flood_utm).astype(int)

    lat, lon = aoi.centroid.y, aoi.centroid.x
    # ECMWF IFS (9 km), which Open-Meteo's default already uses for 2018: 20 mm on 22/09 near Bou Argoub where
    # ERA5 gives 4 mm, but still far below the ~200 mm reported locally. No measured hourly rain is open.
    series = rain.hourly_rain(lat, lon, "2018-06-01", "2018-10-10", d / f"rain_ifs_{tag}.json", model="ecmwf_ifs")
    events = rain.storms(series, min_total=5.0)
    near = events[(events.peak_time >= STORM_DAY - pd.Timedelta(days=1)) & (events.peak_time < STORM_DAY + pd.Timedelta(days=2))]
    storm = near.sort_values("total_mm").iloc[-1]
    for k, v in rain.storm_features(series, storm).items():
        seg[k] = v

    seg["osm_completeness"] = NABEUL_OSM_COMPLETENESS
    seg["black_spot_count"] = 0          # no georeferenced flood history before 2018
    seg["event_id"], seg["slot_min"], seg["aoi"] = f"EMSR319_{tag}", 0, tag
    seg["storm_total_mm"] = storm["total_mm"]
    seg["rain_interpolated"], seg["data_age_min"] = 1, 60
    seg["x_km"], seg["y_km"] = seg["mid"].x / 1000, seg["mid"].y / 1000
    print(f"{tag}: {len(seg)} segments, {seg.flooded.sum()} flooded ({seg.flooded.mean():.1%}), "
          f"storm {storm['start']:%Y-%m-%d %H}h total {storm['total_mm']:.0f} mm, peak {storm['peak_mmh']:.1f} mm/h")
    return seg


def evaluate(inv: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """Transfer test: models trained in NYC, applied unchanged to Nabeul."""
    y = inv["flooded"].to_numpy()
    rows = []
    scores = {"V1 multi-criteria": v1.score(inv) / 100}
    bundles = {}
    for kind, name in [("hgb", "V2 gradient boosting"), ("logistic", "V2 logistic")]:
        b = joblib.load(models_dir / f"v2_{kind}.joblib")
        bundles[name] = b
        scores[name] = b.predict_proba(inv)
    for name, p in scores.items():
        # Exactly 10 % of segments: ties (near-constant scores) must not inflate the alert set.
        top = np.zeros(len(p), dtype=bool)
        top[np.argsort(-np.asarray(p), kind="stable")[: len(p) // 10]] = True
        row = {"model": name, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
               "base rate": y.mean(), "recall top 10%": (top & (y == 1)).sum() / y.sum(),
               "precision top 10%": y[top].mean()}
        if name in bundles:
            alert = np.isin(bundles[name].predict_level(inv), ["High", "Very high"])
            row["recall High+ (NYC thresholds)"] = (alert & (y == 1)).sum() / y.sum()
            row["false alarms High+"] = 1 - (alert & (y == 1)).sum() / max(alert.sum(), 1)
            row["alerts share"] = alert.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main():
    inv_path, models = Path("data/inventory_nabeul.csv"), Path("models/nyc")
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
    else:
        parts = [build_aoi(p) for p in EMSR319_PRODUCTS]
        seg = pd.concat(parts, ignore_index=True)
        seg = gpd.GeoDataFrame(seg, geometry="geometry", crs=NABEUL_CRS).drop_duplicates("segment_id")
        out = raw_dir("..", "real")
        seg.drop(columns=["mid"]).to_file(out / "nabeul_segments.gpkg", driver="GPKG")
        inv = pd.DataFrame(seg.drop(columns=["geometry", "mid"]))
        inv.to_csv(inv_path, index=False)
    print(f"total: {len(inv)} segments, {inv.flooded.sum()} flooded ({inv.flooded.mean():.1%})")
    if not (models / "v2_hgb.joblib").exists():
        print("models/nyc not trained yet: run python -m flood_risk.train --data data/inventory_nyc.csv.gz "
              "--out reports/nyc --models models/nyc")
        return

    res = evaluate(inv, models)
    Path("reports/nabeul").mkdir(parents=True, exist_ok=True)
    res.to_csv("reports/nabeul/transfer_metrics.csv")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
