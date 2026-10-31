"""Per-segment raw measurements: raster sampling in buffers + vector features (Phase 2 steps 2-5)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

from .common import METRIC_CRS

BUILT_UP_CLASS = 50   # ESA WorldCover "Built-up"


def _zonal(geoms, raster: Path, stat: str) -> np.ndarray:
    zs = zonal_stats(list(geoms), str(raster), stats=[stat], all_touched=True, nodata=np.nan)
    return np.array([np.nan if z[stat] is None else z[stat] for z in zs], dtype=float)


def terrain_measurements(seg: gpd.GeoDataFrame, rasters: dict[str, Path], buffer_m: float = 30) -> pd.DataFrame:
    buf = seg.geometry.buffer(buffer_m)
    out = pd.DataFrame(index=seg.index)
    out["depth_in_sink_m"] = np.nan_to_num(_zonal(buf, rasters["depth_in_sink"], "max"))
    out["twi_mean"] = _zonal(buf, rasters["twi"], "mean")
    out["hand_m"] = _zonal(buf, rasters["hand"], "min")
    out["slope_deg"] = _zonal(buf, rasters["slope"], "mean")
    out["dist_stream_m"] = _zonal(seg.geometry.buffer(1), rasters["dist_stream"], "min")
    # a crossing touches drainage cells but does not run along them (mean cover < 40%)
    near_max = np.nan_to_num(_zonal(seg.geometry.buffer(15), rasters["streams"], "max"))
    near_mean = np.nan_to_num(_zonal(seg.geometry.buffer(15), rasters["streams"], "mean"))
    out["wadi_crossing"] = (near_max > 0) & (near_mean < 0.4)
    for c in ["twi_mean", "hand_m", "slope_deg", "dist_stream_m"]:
        out[c] = out[c].fillna(out[c].median())
    return out


def impervious_share(seg: gpd.GeoDataFrame, worldcover_metric: Path, buffer_m: float = 75) -> np.ndarray:
    zs = zonal_stats(list(seg.geometry.buffer(buffer_m)), str(worldcover_metric), categorical=True, nodata=0)
    out = []
    for z in zs:
        total = sum(z.values()) if z else 0
        out.append(z.get(BUILT_UP_CLASS, 0) / total if total else 0.0)
    return np.array(out)


def drain_count(seg: gpd.GeoDataFrame, drains: gpd.GeoDataFrame | None, buffer_m: float = 50) -> np.ndarray:
    if drains is None or drains.empty:
        return np.zeros(len(seg))
    pts = drains.to_crs(METRIC_CRS)
    pts = gpd.GeoDataFrame(geometry=pts.geometry.representative_point(), crs=METRIC_CRS)
    buf = gpd.GeoDataFrame({"i": np.arange(len(seg))}, geometry=seg.geometry.buffer(buffer_m).values, crs=METRIC_CRS)
    j = gpd.sjoin(buf, pts, predicate="contains", how="left")
    counts = j.groupby("i")["index_right"].count()
    return counts.reindex(range(len(seg)), fill_value=0).to_numpy(dtype=float)


def maintenance_factor(seg: gpd.GeoDataFrame, records: gpd.GeoDataFrame | None,
                       horizon_days: float = 365, buffer_m: float = 100) -> np.ndarray:
    """FR-10: days since last cleaning near the segment, scaled to [0,1]; NaN = no data."""
    out = np.full(len(seg), np.nan)
    if records is None or records.empty or "last_cleaned" not in records:
        return out
    rec = records.to_crs(METRIC_CRS).copy()
    rec["last_cleaned"] = pd.to_datetime(rec["last_cleaned"], utc=True, errors="coerce")
    rec = rec.dropna(subset=["last_cleaned"])
    now = datetime.now(timezone.utc)
    mids = seg.geometry.interpolate(0.5, normalized=True)
    for i, m in enumerate(mids):
        near = rec[rec.distance(m) <= buffer_m]
        if not near.empty:
            days = (now - near["last_cleaned"].max()).days
            out[i] = min(1.0, max(0.0, days / horizon_days))
    return out
