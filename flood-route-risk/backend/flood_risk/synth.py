"""Synthetic flood inventory with the same schema as the real one.

It exists only to exercise the V2 pipeline until documented Tunisian events are collected.
The hidden process includes clogged drains, under-reported floods and coarse rainfall, so
the metrics it produces show how the pipeline behaves, not how the real system performs.

    python -m flood_risk.synth --segments 800 --events 16
"""

import argparse

import numpy as np
import pandas as pd

SLOTS_MIN = [-60, -30, 0, 30]   # snapshot times relative to the storm peak


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_segments(n: int, rng: np.random.Generator) -> pd.DataFrame:
    x, y = rng.uniform(0, 20, n), rng.uniform(0, 20, n)
    r = np.hypot(x - 10, y - 10)
    urban = np.clip(1 - r / 9 + rng.normal(0, 0.1, n), 0, 1)
    slope = np.exp(rng.normal(np.log(2), 0.7, n))
    depression = np.where(rng.random(n) < 0.15, rng.exponential(0.6, n), 0.0)
    dist_wadi = np.clip(rng.exponential(900, n), 20, 5000)
    true_inlets = np.clip(rng.normal(5 + 25 * urban, 4), 0, None)
    completeness = np.clip(rng.beta(2 + 4 * urban, 2), 0.05, 1)
    seg = pd.DataFrame({
        "segment_id": [f"S{i:05d}" for i in range(n)],
        "x_km": x, "y_km": y,
        "elevation_rel_m": np.clip(5 + 3 * r + 8 * np.sin(x / 3) * np.cos(y / 4) + rng.normal(0, 3, n), 0, None),
        "slope_pct": slope,
        "twi": np.clip(6 + 2.5 * np.log1p(depression * 5) - 1.2 * np.log(slope) + rng.normal(0, 1, n), 2, 18),
        "depression_depth_m": depression,
        "dist_wadi_m": dist_wadi,
        "impervious_frac": np.clip(0.15 + 0.75 * urban + rng.normal(0, 0.08, n), 0, 1),
        "drain_inlet_density": rng.poisson(true_inlets * completeness).astype(float),
        "osm_completeness": completeness,
        "underpass": (rng.random(n) < 0.02 + 0.04 * urban).astype(int),
        "ford": ((dist_wadi < 150) & (rng.random(n) < 0.3) & (urban < 0.5)).astype(int),
        # Hidden truth, never given to a model.
        "_urban": urban, "_true_inlets": true_inlets, "_maintenance": rng.normal(0, 0.6, n),
    })
    # Past floods: segments that flood under a reference storm accumulate black spots.
    ref = {"rain_eta_mmh": 35.0, "rain_1h_mm": 25.0, "rain_3h_mm": 45.0, "rain_24h_mm": 50.0, "dry_spell_days": 90.0}
    p_ref = flood_probability(seg.assign(**ref))
    seg["black_spot_count"] = rng.poisson(2.5 * p_ref * report_rate(seg))
    return seg


def report_rate(seg: pd.DataFrame) -> np.ndarray:
    """Share of real floods that end up documented (press, civil protection): better in town."""
    return 0.55 + 0.35 * seg["_urban"].to_numpy()


def flood_probability(df: pd.DataFrame) -> np.ndarray:
    """Hidden flood process. The models never see this function."""
    clog = _sigmoid(-1.2 + 0.018 * df["dry_spell_days"] + 1.2 * df["impervious_frac"]
                    + 0.8 * df["_maintenance"] - 0.04 * df["_true_inlets"])
    capacity = (8 + 0.9 * df["_true_inlets"]) * (1 - 0.8 * clog)
    excess = np.clip(df["rain_eta_mmh"] + 0.25 * df["rain_1h_mm"] - capacity, 0, None)
    runoff = 0.3 + 0.6 * df["impervious_frac"] + 0.1 * np.clip(df["rain_24h_mm"] / 60, 0, 1)
    wadi = 2.2 * np.exp(-df["dist_wadi_m"] / 250) * np.clip((df["rain_3h_mm"] - 25) / 20, 0, 1.5)
    ford = 1.5 * df["ford"] * np.clip(df["rain_3h_mm"] / 30, 0, 1.5)
    logit = (-3.6 + 0.14 * runoff * excess + 0.35 * (df["twi"] - 8) + 1.5 * df["depression_depth_m"]
             + 1.6 * df["underpass"] + wadi + ford
             - 0.015 * df["elevation_rel_m"] - 0.3 * np.log(df["slope_pct"]))
    dry_road = (df["rain_eta_mmh"] + df["rain_1h_mm"]) < 5
    return _sigmoid(np.where(dry_road, logit - 4, logit))


def _intensity(peak, t, width):
    return peak * np.exp(-0.5 * (t / width) ** 2)


def _accum(peak, t_end, minutes, width):
    """Rain (mm) fallen in the `minutes` before t_end, with 5-min steps."""
    offsets = np.arange(-minutes, 0, 5) + 2.5
    ts = np.asarray(t_end, dtype=float).reshape(-1, 1) + offsets
    return _intensity(np.asarray(peak).reshape(-1, 1), ts, width).sum(axis=1) * 5 / 60


def make_event(rng: np.random.Generator, event_id: str) -> dict:
    return {
        "event_id": event_id,
        "peak_mmh": float(np.clip(rng.lognormal(np.log(28), 0.45), 6, 110)),
        "cx": rng.uniform(2, 18), "cy": rng.uniform(2, 18),
        "radius_km": rng.uniform(4, 12),
        "width_min": rng.uniform(20, 60),
        "dry_spell_days": float(rng.choice([rng.uniform(1, 15), rng.uniform(60, 160)], p=[0.6, 0.4])),
        "ante_mm": float(rng.exponential(8)),
        "interpolated": bool(rng.random() < 0.85),
    }


def rain_at(seg: pd.DataFrame, ev: dict, t_rel_min, rng: np.random.Generator) -> pd.DataFrame:
    """Rain features of each segment at time t_rel_min (scalar or per-segment array) from the peak."""
    d = np.hypot(seg["x_km"] - ev["cx"], seg["y_km"] - ev["cy"]).to_numpy()
    peak = ev["peak_mmh"] * np.exp(-((d / ev["radius_km"]) ** 2)) * rng.lognormal(0, 0.25, len(seg))
    t = np.broadcast_to(np.asarray(t_rel_min, dtype=float), (len(seg),))
    rain_3h = _accum(peak, t, 180, ev["width_min"])
    return pd.DataFrame({
        "rain_eta_mmh": _intensity(peak, t, ev["width_min"]),
        "rain_1h_mm": _accum(peak, t, 60, ev["width_min"]),
        "rain_3h_mm": rain_3h,
        "rain_24h_mm": ev["ante_mm"] * rng.lognormal(0, 0.3, len(seg)) + rain_3h,
        "dry_spell_days": ev["dry_spell_days"],
        "rain_interpolated": int(ev["interpolated"]),
        "data_age_min": rng.choice([5, 10, 15, 30, 90], len(seg), p=[0.35, 0.3, 0.2, 0.1, 0.05]),
    }, index=seg.index)


def make_inventory(n_segments: int, n_events: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    seg = make_segments(n_segments, rng)
    rows = []
    for k in range(n_events):
        ev = make_event(rng, f"E{k:02d}")
        for slot in SLOTS_MIN:
            df = pd.concat([seg, rain_at(seg, ev, slot, rng)], axis=1)
            df["event_id"], df["slot_min"] = ev["event_id"], slot
            p = flood_probability(df)
            df["flooded_true"] = (rng.random(len(df)) < p).astype(int)
            documented = rng.random(len(df)) < report_rate(df)
            mislocated = rng.random(len(df)) < 0.0005
            df["flooded"] = ((df["flooded_true"].astype(bool) & documented) | mislocated).astype(int)
            rows.append(df)
    inv = pd.concat(rows, ignore_index=True)
    return seg, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=800)
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/inventory_synthetic.csv")
    args = ap.parse_args()
    seg, inv = make_inventory(args.segments, args.events, args.seed)
    seg.to_csv("data/segments_synthetic.csv", index=False)
    inv.drop(columns=[c for c in inv.columns if c.startswith("_")]).to_csv(args.out, index=False)
    print(f"{len(inv)} rows, {inv['event_id'].nunique()} events, "
          f"flooded (documented) {inv['flooded'].mean():.2%}, flooded (true) {inv['flooded_true'].mean():.2%}")
    print(inv.groupby("event_id")[["flooded", "flooded_true"]].sum().T.to_string())


if __name__ == "__main__":
    main()
