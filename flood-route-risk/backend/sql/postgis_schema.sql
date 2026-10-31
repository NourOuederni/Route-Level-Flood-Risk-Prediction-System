-- PostGIS version of the storage (guide Appendix B). The API creates plain tables itself
-- (SQLAlchemy); run this once if you want spatial columns and indexes for QGIS / SQL work.
-- psql "$DATABASE_URL" -f sql/postgis_schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- Segments with static factors (load with ogr2ogr from data/processed/<area>/segments_qa.gpkg)
CREATE TABLE IF NOT EXISTS segment (
  segment_id BIGINT PRIMARY KEY,
  edge_uid TEXT, seq INT, road_type TEXT, name TEXT, length_m REAL,
  geom_utm geometry(LineString, 32632),
  geom_wgs84 geometry(LineString, 4326),
  depth_in_sink_m REAL, twi_mean REAL, hand_m REAL, slope_deg REAL, dist_stream_m REAL,
  impervious_share REAL, is_underpass BOOLEAN, is_ford BOOLEAN, is_bridge BOOLEAN,
  f_depression REAL, f_drainage_position REAL, f_low_slope REAL, f_impervious REAL,
  f_crossing REAL, f_past_flood REAL,
  dv_urban REAL, dv_blackspot REAL, dv_drain_density_inv REAL, dv_maintenance REAL,
  poorly_mapped BOOLEAN, past_flood_confirmed BOOLEAN, crossing_kind TEXT,
  features_version TEXT
);
CREATE INDEX IF NOT EXISTS segment_geom_utm_idx ON segment USING GIST (geom_utm);

CREATE TABLE IF NOT EXISTS flood_event (
  event_id BIGSERIAL PRIMARY KEY,
  event_date TIMESTAMPTZ,
  duration_h REAL,
  geom geometry(Geometry, 4326),
  spatial_acc_m REAL,
  source TEXT, source_url TEXT,
  reliability TEXT CHECK (reliability IN ('high','medium','low')),
  observed_rain_mm REAL,
  test_set BOOLEAN DEFAULT FALSE,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS flood_event_geom_idx ON flood_event USING GIST (geom);

CREATE TABLE IF NOT EXISTS rain_obs_forecast (
  cell_id TEXT, valid_time TIMESTAMPTZ, model TEXT, retrieved_at TIMESTAMPTZ,
  mm REAL, interval_min INT, source TEXT,
  PRIMARY KEY (cell_id, valid_time, model, retrieved_at)
);

CREATE TABLE IF NOT EXISTS prediction_log (
  id BIGSERIAL PRIMARY KEY, request_id TEXT, segment_id BIGINT, eta TIMESTAMPTZ,
  score REAL, level TEXT, confidence REAL, mode TEXT, factors JSONB,
  rain_source TEXT, rain_retrieved_at TIMESTAMPTZ, engine_version TEXT,
  issued_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS prediction_log_seg_eta_idx ON prediction_log (segment_id, eta);

-- No user id, no origin/destination, no trip trace (NFR-07)
CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY, segment_id BIGINT, lat REAL, lon REAL,
  kind TEXT CHECK (kind IN ('flooded','not_flooded','drain_blocked')),
  comment TEXT, created_at TIMESTAMPTZ DEFAULT now()
);
