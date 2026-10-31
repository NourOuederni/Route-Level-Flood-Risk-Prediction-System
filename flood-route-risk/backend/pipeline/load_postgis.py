"""Load processed tile segments into the PostGIS database.

Called by national.py after each tile completes feature computation.
Also handles weather-cell assignment and de-duplication of overlap segments.
"""
from __future__ import annotations

import logging
import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from sqlalchemy import text

from app.db.models import Base, Segment, WeatherCell, make_engine, make_session

log = logging.getLogger("pipeline.load_postgis")

# Weather grid: 0.25° spacing covering Tunisia -> ~500 cells
WEATHER_GRID_STEP = 0.25
TUNISIA_BBOX = [7.5, 30.2, 11.6, 37.5]


def _segment_id_global(tile_id: str, local_id: int) -> int:
    """Create a globally unique segment_id from tile coordinates and local id.

    Encodes tile grid position into the upper bits so IDs never collide across tiles.
    """
    parts = tile_id.split("_")
    lon_idx = int(float(parts[0]) * 10)
    lat_idx = int(float(parts[1]) * 10)
    tile_num = lon_idx * 1000 + lat_idx
    return tile_num * 1_000_000 + local_id


def load_tile_segments(seg: gpd.GeoDataFrame, tile_id: str, db_url: str) -> int:
    """Insert segments from one tile into PostGIS. Returns count inserted."""
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    Session = make_session(engine)
    session = Session()

    # Convert to WGS84 if needed
    if seg.crs and seg.crs.to_epsg() != 4326:
        seg = seg.to_crs(4326)

    inserted = 0
    try:
        for row in seg.itertuples(index=False):
            global_id = _segment_id_global(tile_id, int(row.segment_id))

            # Check for duplicate (overlap regions)
            existing = session.query(Segment.segment_id).filter(
                Segment.segment_id == global_id).first()
            if existing:
                continue

            coords = list(row.geometry.coords)
            mid_idx = len(coords) // 2
            mid_lon, mid_lat = coords[mid_idx]

            segment = Segment(
                segment_id=global_id,
                tile_id=tile_id,
                edge_uid=str(row.edge_uid),
                seq=int(row.seq),
                road_type=row.road_type if isinstance(row.road_type, str) else None,
                name=row.name if isinstance(row.name, str) else None,
                length_m=float(row.length_m),
                geom=f"SRID=4326;{row.geometry.wkt}",
                mid_lat=mid_lat,
                mid_lon=mid_lon,

                # Raw measurements
                depth_in_sink_m=_safe_float(getattr(row, "depth_in_sink_m", None)),
                twi_mean=_safe_float(getattr(row, "twi_mean", None)),
                hand_m=_safe_float(getattr(row, "hand_m", None)),
                slope_deg=_safe_float(getattr(row, "slope_deg", None)),
                dist_stream_m=_safe_float(getattr(row, "dist_stream_m", None)),
                impervious_share=_safe_float(getattr(row, "impervious_share", None)),

                # Normalised factors
                f_depression=float(row.f_depression),
                f_drainage_position=float(row.f_drainage_position),
                f_low_slope=float(row.f_low_slope),
                f_impervious=float(row.f_impervious),
                f_crossing=float(row.f_crossing),
                f_past_flood=float(row.f_past_flood),

                # Pre-computed susceptibility
                susceptibility=float(row.susceptibility),
                susceptibility_floored=float(row.susceptibility_floored),

                # Drainage vulnerability
                dv_urban=float(row.dv_urban),
                dv_blackspot=float(row.dv_blackspot),
                dv_drain_density_inv=float(row.dv_drain_density_inv),
                dv_maintenance=_safe_float(getattr(row, "dv_maintenance", None)),

                # Boolean flags
                is_underpass=bool(getattr(row, "is_underpass", False)),
                is_ford=bool(getattr(row, "is_ford", False)),
                is_bridge=bool(getattr(row, "is_bridge", False)),
                wadi_crossing=bool(getattr(row, "wadi_crossing", False)),
                poorly_mapped=bool(getattr(row, "poorly_mapped", False)),
                past_flood_confirmed=bool(getattr(row, "past_flood_confirmed", False)),
                crossing_kind=row.crossing_kind if isinstance(
                    getattr(row, "crossing_kind", None), str) else None,
            )
            session.add(segment)
            inserted += 1

            # Batch commit every 5000
            if inserted % 5000 == 0:
                session.commit()
                log.info("tile %s: committed %d segments", tile_id, inserted)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()

    return inserted


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def generate_weather_grid() -> list[dict]:
    """Generate the national weather grid (0.25° spacing)."""
    w, s, e, n = TUNISIA_BBOX
    cells = []
    lat = s
    while lat <= n:
        lon = w
        while lon <= e:
            cell_id = f"{lon:.2f}_{lat:.2f}"
            cells.append({"cell_id": cell_id, "lat": round(lat, 4), "lon": round(lon, 4)})
            lon += WEATHER_GRID_STEP
        lat += WEATHER_GRID_STEP
    return cells


def finalise_weather_cells(db_url: str) -> None:
    """Create weather grid and assign each segment to its nearest cell."""
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    Session = make_session(engine)
    session = Session()

    try:
        # Insert weather cells
        cells = generate_weather_grid()
        for c in cells:
            existing = session.query(WeatherCell).filter(WeatherCell.cell_id == c["cell_id"]).first()
            if not existing:
                session.add(WeatherCell(**c))
        session.commit()
        log.info("weather grid: %d cells", len(cells))

        # Assign each segment to its nearest weather cell
        # Using SQL for efficiency on 200k+ segments
        session.execute(text("""
            UPDATE segment s
            SET cell_id = (
                SELECT wc.cell_id
                FROM weather_cell wc
                ORDER BY (s.mid_lat - wc.lat)^2 + (s.mid_lon - wc.lon)^2
                LIMIT 1
            )
            WHERE s.cell_id IS NULL
        """))
        session.commit()
        log.info("segments assigned to weather cells")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()
