"""Column names shared by the data pipeline, the V1 index and the V2 models.

A real flood inventory must provide these columns (one row per segment x event x time
slot) to replace the synthetic data: see README.md.
"""

STATIC_FEATURES = [
    "elevation_rel_m",      # elevation above the local minimum (DEM)
    "slope_pct",            # road slope (DEM)
    "twi",                  # topographic wetness index (DEM flow accumulation)
    "depression_depth_m",   # fill-sink depth of the closed depression the segment sits in
    "dist_wadi_m",          # distance to the nearest wadi / drainage line (OSM, DEM)
    "impervious_frac",      # share of impervious surface in a 100 m buffer (land cover)
    "drain_inlet_density",  # mapped storm-drain inlets per km (OSM, municipality)
    "osm_completeness",     # 0..1 estimate of how well the area is mapped
    "underpass",            # 0/1
    "ford",                 # 0/1, road crosses a wadi without a bridge
    "black_spot_count",     # documented past floods on the segment
]

DYNAMIC_FEATURES = [
    "rain_eta_mmh",         # forecast rainfall intensity in the ETA window
    "rain_1h_mm",           # rainfall accumulated in the hour before ETA
    "rain_3h_mm",           # rainfall accumulated in the 3 hours before ETA
    "rain_24h_mm",          # antecedent rainfall, 24 h
    "dry_spell_days",       # days since the last significant rain
]

# Local history, available wherever floods and drain problems are reported (311 in New York,
# FR-21 user feedback in the application). Missing columns are read as "no history" (0).
HISTORY_FEATURES = [
    "flood_reports_250m",   # past flood reports within 250 m of the segment
    "flood_reports_365d",   # past flood reports on the segment in the last 365 days
    "basin_reports_prior",  # past "drain blocked" reports on the segment
]

# Position of the segment within its mapped area (percentiles, 0..1). They make terrain comparable
# between areas with different relief; missing columns are read as the area median (0.5).
RELATIVE_FEATURES = ["elevation_rel_pct", "dist_wadi_pct", "twi_pct", "impervious_pct"]

RAW_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

# Metadata used for confidence and splitting, never as model input.
META = ["segment_id", "event_id", "slot_min", "x_km", "y_km", "rain_interpolated", "data_age_min"]

LABEL = "flooded"

LEVELS = ["Low", "Moderate", "High", "Very high"]

# Features that carry the drainage-vulnerability hypothesis (FR-08); dropped in the ablation.
DRAINAGE_FEATURES = ["drain_inlet_density", "dry_spell_days", "osm_completeness", "basin_reports_prior"]

# Reference rainfall (mm/h) a working drainage network is assumed to absorb; FR-09 lowers it.
RAIN_REF_MMH = 30.0

# Target recall used to place the High threshold (section 8, proposed target).
TARGET_RECALL_HIGH = 0.70
TARGET_RECALL_MODERATE = 0.90
TARGET_RECALL_VERY_HIGH = 0.35

# Alternative level placement (FR-24): share of segment-storms put at each level or above.
ALERT_SHARE = {"Moderate": 0.10, "High": 0.02, "Very high": 0.005}
