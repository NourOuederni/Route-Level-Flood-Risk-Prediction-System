"""Version 1: weighted multi-criteria index (AHP-style fixed weights), 0..100. Baseline for V2."""

import numpy as np
import pandas as pd

from .features import drainage_vulnerability, effective_rain_threshold

WEIGHTS = {
    "rain": 0.30, "antecedent": 0.10, "depression": 0.15, "wadi": 0.12,
    "drainage": 0.12, "impervious": 0.08, "history": 0.08, "slope": 0.05,
}


def factors(df: pd.DataFrame) -> pd.DataFrame:
    dv = drainage_vulnerability(df)
    rain = (df["rain_eta_mmh"] + 0.25 * df["rain_1h_mm"]) / effective_rain_threshold(dv)
    return pd.DataFrame({
        "rain": np.clip(rain, 0, 1),
        "antecedent": np.clip(df["rain_24h_mm"] / 80, 0, 1),
        "depression": np.clip((df["twi"] - 4) / 10 + df["depression_depth_m"] / 2
                              + 0.5 * df["underpass"], 0, 1),
        "wadi": np.clip(np.exp(-df["dist_wadi_m"] / 400) + 0.5 * df["ford"], 0, 1),
        "drainage": dv,
        "impervious": df["impervious_frac"],
        "history": np.clip(df["black_spot_count"] / 3, 0, 1),
        "slope": 1 - np.clip(df["slope_pct"] / 8, 0, 1),
    }, index=df.index)


def relative_susceptibility(df: pd.DataFrame, area_col: str | None = None) -> np.ndarray:
    """0..100 static score from the segment's position within its area: low points near a watercourse rank highest.

    Percentiles are taken inside each area so the score does not depend on the area's absolute
    elevation or mapping density. It suits flash floods on plains (Nabeul 2018: ROC-AUC 0.80) but
    not urban street flooding (New York: 0.48). The rule was chosen after looking at the Nabeul
    medians, so it must be confirmed on another Tunisian event before being trusted.
    """
    groups = df.groupby(area_col) if area_col else None
    pct = lambda c: (groups[c] if groups is not None else df[c]).rank(pct=True)
    return (100 * (1 - (pct("elevation_rel_m") + pct("dist_wadi_m")) / 2)).to_numpy()


def score(df: pd.DataFrame) -> np.ndarray:
    f = factors(df)
    s = sum(w * f[k] for k, w in WEIGHTS.items())
    # Terrain alone never makes a dry road risky.
    return (100 * s * np.clip(f["rain"] * 2, 0, 1)).to_numpy()
