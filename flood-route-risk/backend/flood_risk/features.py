"""Hybrid feature engineering: physically meaningful features fed to the learners (Ch. 2.2.7)."""

import numpy as np
import pandas as pd

from .schema import DRAINAGE_FEATURES, HISTORY_FEATURES, RAIN_REF_MMH, RAW_FEATURES, RELATIVE_FEATURES


def drainage_vulnerability(df: pd.DataFrame) -> pd.Series:
    """0..1 proxy of the likelihood that drainage is insufficient or blocked (FR-08)."""
    few_inlets = 1 - np.clip(df["drain_inlet_density"] / 30, 0, 1)
    # Missing inlets in a poorly mapped area are not evidence: shrink towards neutral.
    few_inlets = df["osm_completeness"] * few_inlets + (1 - df["osm_completeness"]) * 0.5
    dry = np.clip(df["dry_spell_days"] / 120, 0, 1)
    history = np.clip(df["black_spot_count"] / 3, 0, 1)
    return 0.35 * few_inlets + 0.30 * dry + 0.20 * df["impervious_frac"] + 0.15 * history


def effective_rain_threshold(dv: pd.Series) -> pd.Series:
    """Rainfall (mm/h) at which a segment becomes critical; a blocked drain lowers it (FR-09)."""
    return RAIN_REF_MMH * (1 - 0.5 * dv)


def engineer(df: pd.DataFrame, use_drainage: bool = True) -> pd.DataFrame:
    out = df[RAW_FEATURES].copy()
    for col in HISTORY_FEATURES:
        out[col] = df[col] if col in df.columns else 0.0
    for col in RELATIVE_FEATURES:
        out[col] = df[col] if col in df.columns else 0.5
    dv = drainage_vulnerability(df) if use_drainage else pd.Series(0.0, index=df.index)
    out["drainage_vuln"] = dv
    out["rain_excess_ratio"] = (df["rain_eta_mmh"] + 0.25 * df["rain_1h_mm"]) / effective_rain_threshold(dv)
    out["runoff_index"] = out["rain_excess_ratio"] * (0.3 + 0.7 * df["impervious_frac"])
    out["wadi_rain"] = df["rain_3h_mm"] * np.exp(-df["dist_wadi_m"] / 300)
    out["log_dist_wadi"] = np.log10(df["dist_wadi_m"].clip(lower=10))
    out["log_slope"] = np.log(df["slope_pct"].clip(lower=0.1))
    out = out.drop(columns=["dist_wadi_m", "slope_pct"])
    if not use_drainage:
        out = out.drop(columns=DRAINAGE_FEATURES + ["drainage_vuln"])
    return out.astype("float32")


# +1: risk can only increase with the feature, -1: only decrease, 0: free.
MONOTONIC = {
    "elevation_rel_m": -1, "twi": 1, "depression_depth_m": 1, "impervious_frac": 1,
    "drain_inlet_density": -1, "osm_completeness": 0, "underpass": 1, "ford": 1,
    "black_spot_count": 1, "rain_eta_mmh": 1, "rain_1h_mm": 1, "rain_3h_mm": 1,
    "rain_24h_mm": 1, "dry_spell_days": 1, "drainage_vuln": 1, "rain_excess_ratio": 1,
    "runoff_index": 1, "wadi_rain": 1, "log_dist_wadi": -1, "log_slope": -1,
    "flood_reports_250m": 1, "flood_reports_365d": 1, "basin_reports_prior": 1,
    "elevation_rel_pct": -1, "dist_wadi_pct": -1, "twi_pct": 1, "impervious_pct": 0,
    # v2.INTERACTIONS: more rain on a more exposed segment can only raise the risk; rain x elevation
    # mixes a rising and a falling factor, so it stays free.
    "rain1h_x_impervious": 1, "rain24h_x_depression": 1, "rain1h_x_twi": 1, "rain3h_x_elevation": 0,
    "rain_event_mm": 1, "rain_next_6h_mm": 1,
}

# Groups of raw columns shown to the user as "main contributing factors" (FR-13).
FACTOR_GROUPS = {
    "forecast rainfall": ["rain_eta_mmh"],
    "rain already fallen": ["rain_1h_mm", "rain_3h_mm", "rain_24h_mm"],
    "depression / low point": ["twi", "depression_depth_m", "elevation_rel_m", "slope_pct"],
    "wadi proximity": ["dist_wadi_m", "ford"],
    "drainage vulnerability": ["drain_inlet_density", "dry_spell_days", "basin_reports_prior"],
    "impervious surface": ["impervious_frac"],
    "past floods": ["black_spot_count", "flood_reports_250m", "flood_reports_365d"],
    "underpass": ["underpass"],
}
