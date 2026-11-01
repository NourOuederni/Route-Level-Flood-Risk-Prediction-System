"""Score the segments of a route at their ETA with a trained V2 model (FR-11 to FR-15).

    python -m flood_risk.predict --demo
    python -m flood_risk.predict --route my_route.csv

The route CSV has one row per segment with the raw features of schema.py at the segment's
ETA, plus `segment_id`, `eta_min`, `rain_interpolated` and `data_age_min`.

Levels follow the alert budget of each area (FR-24, schema.ALERT_SHARE) when the model was
trained with `--levels area`: the whole area is scored for the same forecast, and a route
segment is High if it is among the riskiest 2 % of its area. Pass the area with `--area`
(every segment of the zone, same columns plus `aoi` when several zones are served). Without
it, the fixed probability thresholds stored in the model are used.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import v2
from .features import FACTOR_GROUPS, drainage_vulnerability
from .schema import ALERT_SHARE, HISTORY_FEATURES, RAW_FEATURES


def confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based confidence (section 7.3); the model probability says nothing about data quality."""
    c = pd.Series(1.0, index=df.index)
    reasons = [[] for _ in range(len(df))]

    def apply(mask, delta, why):
        nonlocal c
        c = c + np.where(mask, delta, 0.0)
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(why)

    apply(df["rain_interpolated"] == 1, -0.20, "rainfall interpolated from hourly forecast")
    apply(df["data_age_min"] > 15, -0.20, "rainfall data older than 15 min")
    apply(df["data_age_min"] > 60, -0.15, "rainfall data older than 1 h")
    apply(df["osm_completeness"] < 0.5, -0.15, "area poorly mapped")
    apply(df["impervious_frac"] > 0.6, -0.10, "dense urban area, drain state unknown")
    apply(drainage_vulnerability(df) > 0.6, -0.10, "high estimated drainage vulnerability")
    apply(df["black_spot_count"] > 0, 0.10, "documented past floods here")
    c = c.clip(0.1, 1.0)
    label = np.select([c >= 0.7, c >= 0.45], ["High", "Medium"], "Low")
    return pd.DataFrame({"confidence": c.round(2), "confidence_level": label,
                         "confidence_notes": ["; ".join(r) for r in reasons]}, index=df.index)


def explain(bundle, df: pd.DataFrame, reference: pd.Series, top: int = 3) -> list:
    """Main contributing factors: score drop when a factor group is set to a typical value.

    Uses the model's raw score: calibration merges most of a new area into a few flat steps,
    where every factor would look negligible.
    """
    base = bundle.raw(df)
    drops = {}
    for name, cols in FACTOR_GROUPS.items():
        neutral = df.copy()
        for col in cols:
            if col in neutral.columns:
                neutral[col] = reference.get(col, 0.0)
        drops[name] = base - bundle.raw(neutral)
    drops = pd.DataFrame(drops, index=df.index)
    return [", ".join(f"{k} (+{100 * v:.0f})" for k, v in row.nlargest(top).items() if v > 0.005) or "none dominant"
            for _, row in drops.iterrows()]


def display_score(p, thresholds) -> np.ndarray:
    """0..100 score anchored on the levels (25 Moderate, 50 High, 75 Very high).

    Floods are rare, so a calibrated probability of a few percent can already be High; showing
    it raw as "7/100" would read as safe. The probability is reported next to the score.
    """
    knots = [0, thresholds["Moderate"], thresholds["High"], thresholds["Very high"], 1]
    # Tied thresholds (a flat area) would make np.interp ill-defined: keep the knots increasing.
    knots = np.maximum.accumulate(np.array(knots) + np.arange(5) * 1e-9)
    return np.round(np.interp(p, knots, [0, 25, 50, 75, 100])).astype(int)


def area_thresholds(bundle, area: pd.DataFrame, shares=ALERT_SHARE) -> dict:
    """Probability thresholds of each area for the current forecast: the top `shares[level]` reach the level.

    Returns {area id: {level: threshold}}; the id is `aoi` when present, otherwise a single "all".
    """
    p = pd.Series(bundle.predict_proba(area), index=area.index)
    groups = area["aoi"] if "aoi" in area.columns else pd.Series("all", index=area.index)
    return {aoi: {level: float(np.quantile(p[idx], 1 - share)) for level, share in shares.items()}
            for aoi, idx in groups.groupby(groups).groups.items()}


def route_thresholds(bundle, route: pd.DataFrame, area: pd.DataFrame | None) -> list:
    """Thresholds to apply to each route segment: those of its area, or the model's fixed ones."""
    use_area = area is not None and bundle.trained_on.get("levels") in ("area", "budget")
    if not use_area:
        return [bundle.thresholds] * len(route)
    by_area = area_thresholds(bundle, area)
    ids = route["aoi"] if "aoi" in route.columns else pd.Series("all", index=route.index)
    missing = set(ids) - set(by_area)
    if missing:
        raise ValueError(f"route segments in areas that were not scored: {sorted(missing)}")
    return [by_area[a] for a in ids]


def load_trigger(path) -> dict | None:
    """Rain trigger of a zone ({"column", "threshold_mm", ...}), or None while it has not been validated."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def trigger_status(route: pd.DataFrame, area: pd.DataFrame | None, trigger: dict | None) -> pd.Series:
    """Per route segment: "on" when its zone's rain reaches the trigger, "off" below it, "not validated" without one.

    The zone value is the wettest segment of the zone (the trigger watches the worst part of it).
    """
    if trigger is None:
        return pd.Series("not validated", index=route.index)
    src = area if area is not None else route
    zone = src["aoi"] if "aoi" in src.columns else pd.Series("all", index=src.index)
    wettest = src[trigger["column"]].groupby(zone).max()
    ids = route["aoi"] if "aoi" in route.columns else pd.Series("all", index=route.index)
    return ids.map(lambda a: "on" if wettest.get(a, 0) >= trigger["threshold_mm"] else "off")


def fill_unused(bundle, df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Raw columns the model does not use may be absent (e.g. rain for the terrain model): fill them with 0."""
    if df is None:
        return None
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    used = [c for c in missing if c in bundle.columns]
    if used:
        raise ValueError(f"inputs missing columns the model uses: {used}")
    return df.assign(**{c: 0.0 for c in missing})


def score_route(bundle, route: pd.DataFrame, data_time: str, area: pd.DataFrame | None = None,
                trigger: dict | None = None) -> pd.DataFrame:
    trig_cols = [trigger["column"]] if trigger else []
    route, area = fill_unused(bundle, route), fill_unused(bundle, area)
    for df in [d for d in (route, area) if d is not None]:
        missing = [c for c in trig_cols if c not in df.columns]
        if missing:
            raise ValueError(f"rain trigger column missing: {missing}")
    p = bundle.predict_proba(route)
    thresholds = route_thresholds(bundle, route, area)
    known = [c for c in RAW_FEATURES + HISTORY_FEATURES if c in route.columns]
    reference = pd.Series(bundle.trained_on.get("reference", {}), dtype=float).reindex(known)
    # Missing history means "no report": the neutral value is 0.
    reference = reference.fillna(route[known].median()).fillna(0.0)
    out = pd.DataFrame({
        "segment_id": route["segment_id"], "eta_min": route["eta_min"].round(1),
        "risk_score": [int(display_score(np.array([pi]), t)[0]) for pi, t in zip(p, thresholds)],
        "flood_prob_pct": np.round(100 * p, 1),
        "level": [v2.to_level(np.array([pi]), t)[0] for pi, t in zip(p, thresholds)],
    }, index=route.index)
    # Terrain models rank roads but do not see the weather: alert levels only apply once the zone's
    # rain trigger is reached. Below it, or while no trigger is validated, the level says how exposed
    # the road is (susceptibility) and is not an alert.
    status = trigger_status(route, area, trigger) if not set(bundle.columns) & {"rain_eta_mmh", "rain_1h_mm"} else         pd.Series("rain in model", index=route.index)
    out["susceptibility"] = out["level"]
    out["rain_trigger"] = status
    out.loc[status.isin(["off", "not validated"]), "level"] = "Low"
    out = out.join(confidence(route))
    out["main_factors"] = explain(bundle, route, reference)
    out["data_timestamp"] = data_time
    return out


def demo_route(rng) -> pd.DataFrame:
    """A 25-min trip across the synthetic area during a storm that was never seen in training."""
    from .synth import make_event, make_segments, rain_at
    seg = make_segments(800, np.random.default_rng(42))  # same segments as the inventory
    pts = np.linspace([2, 3], [17, 15], 120)
    idx = pd.unique(np.array([int(np.argmin(np.hypot(seg.x_km - x, seg.y_km - y))) for x, y in pts]))
    route = seg.iloc[idx].reset_index(drop=True)
    route["eta_min"] = np.linspace(0, 25, len(route))
    ev = make_event(rng, "DEMO")
    ev.update(peak_mmh=45.0, cx=10.0, cy=10.0, radius_km=7.0, dry_spell_days=130.0, width_min=40.0)
    # The storm peaks over the city 15 minutes after departure.
    return pd.concat([route, rain_at(route, ev, route["eta_min"].to_numpy() - 15, rng)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/v2_hgb.joblib")
    ap.add_argument("--route")
    ap.add_argument("--area", help="every segment of the route's area(s) at the same forecast time, for budget levels")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--alert-level", default="High")
    ap.add_argument("--trigger", default="models/final/trigger.json",
                    help="validated rain trigger of the zones; absent = terrain susceptibility only, no alert")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    route = demo_route(np.random.default_rng(7)) if args.demo else pd.read_csv(args.route)
    area = pd.read_csv(args.area) if args.area else None
    if area is None and bundle.trained_on.get("levels") in ("area", "budget"):
        print("warning: model expects budget levels but no --area was given; using its fixed thresholds")
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    trigger = load_trigger(args.trigger)
    res = score_route(bundle, route, now, area, trigger)
    if trigger is None and not set(bundle.columns) & {"rain_eta_mmh", "rain_1h_mm"}:
        print("note: no validated rain trigger: levels stay Low, see the 'susceptibility' column")

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    mode = f"budget levels from {len(area)} area segments" if area is not None else f"thresholds {bundle.thresholds}"
    print(f"model {bundle.kind}, trained {bundle.trained_on.get('trained_at')} on "
          f"{bundle.trained_on.get('events')} events; {mode}")
    print("Risk = possibility of flooding, not certainty. This complements official warnings: "
          "never drive into a flooded road.\n")
    print(res["level"].value_counts().reindex(["Low", "Moderate", "High", "Very high"], fill_value=0).to_string())
    order = ["Low", "Moderate", "High", "Very high"]
    alerts = res[res["level"].map(order.index) >= order.index(args.alert_level)]
    print(f"\n{len(alerts)} segment(s) at {args.alert_level} or above:")
    print(alerts.drop(columns=["data_timestamp", "confidence_notes"]).to_string(index=False) if len(alerts)
          else "none")
    if len(alerts):
        print(f"\nconfidence notes for first alert: {alerts.iloc[0]['confidence_notes']}")
    res.to_csv("reports/route_scores.csv", index=False)
    print("\nall segments: reports/route_scores.csv")


if __name__ == "__main__":
    main()
