"""Validation by replaying past events (guide Section 8, FR-25 evaluation part).

    python -m evaluation.replay_eval --area nabeul --rain openmeteo_forecast_2024-01-10.json
    python -m evaluation.replay_eval --area demo --scenario storm          # pipeline smoke test

Protocol (freeze it before looking at results!)
  * labels: test-set events only (never used for f6 / DV); a segment is "flooded" if it lies
    within --tolerance-m of an event geometry
  * for each event, score segments within --radius-km every 15 min from --lead-h before the
    event to 1 h after, with the rainfall that was available (replay file)
  * metrics per method: recall at High, share of High alerts on non-flooded segments, median
    lead time, plus a precision-recall table over score cut-offs
  * baselines: static-only (S' alone) and rain-only (H alone) - the full model must beat both
Report forecast-archive events and ERA5 events SEPARATELY.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from app.engine import EvalContext, default_config, evaluate_segment
from app.routing.area import Area
from app.weather import open_meteo, scenarios

ROOT = Path(__file__).resolve().parents[1]
STEP = timedelta(minutes=15)


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


def evaluate(area: Area, store, events: gpd.GeoDataFrame, tolerance_m: float, radius_km: float,
             lead_h: float) -> dict:
    cfg = default_config()
    high = float(cfg.level_cuts["high"])
    ante = int(cfg.section("rainfall")["antecedent_hours"])
    win = timedelta(minutes=float(cfg.section("rainfall")["window"]["min_minutes"]))
    store.simulated_age_minutes = 30.0       # replayed data are treated as fresh
    ctx = EvalContext(refresh_interval_minutes=float(cfg.section("weather")["refresh_interval_minutes"]))

    seg_ids = list(area.segments)
    mids = gpd.GeoSeries(gpd.points_from_xy([area.segments[i].mid[1] for i in seg_ids],
                                            [area.segments[i].mid[0] for i in seg_ids]), crs=4326).to_crs(32632)
    ev_m = events.to_crs(32632)
    rows = []
    for _, ev in ev_m.iterrows():
        t_ev = pd.Timestamp(ev["event_date"]).to_pydatetime().astimezone(timezone.utc)
        dist = mids.distance(ev.geometry).to_numpy()
        near = dist <= radius_km * 1000
        flooded = dist <= tolerance_m
        times = []
        t = t_ev - timedelta(hours=lead_h)
        while t <= t_ev + timedelta(hours=1):
            times.append(t); t += STEP
        for idx in np.flatnonzero(near):
            rec = area.segments[seg_ids[idx]]
            best = {"full": 0.0, "static": 0.0, "rain": 0.0}
            first_high = None
            for t in times:
                rain = store.window(rec.cell_id, t - win, t + win, t, ante)
                r = evaluate_segment(rec.static, rain, ctx, cfg)
                sp = cfg.s_min + (1 - cfg.s_min) * r.susceptibility
                best["full"] = max(best["full"], r.score)
                best["static"] = max(best["static"], 100 * sp)
                best["rain"] = max(best["rain"], 100 * (r.hazard or 0.0))
                if first_high is None and r.score >= high and t <= t_ev:
                    first_high = t
            rows.append({"event": int(_), "segment_id": rec.static.segment_id, "flooded": bool(flooded[idx]),
                         **{f"max_{k}": v for k, v in best.items()},
                         "lead_min": (t_ev - first_high).total_seconds() / 60 if first_high else None})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "no segments near the events"}

    report = {"segments_scored": int(len(df)), "flooded_segments": int(df["flooded"].sum()),
              "events": int(len(events)), "methods": {}}
    for m in ("full", "static", "rain"):
        alert = df[f"max_{m}"] >= high
        tp = int((alert & df["flooded"]).sum())
        fn = int((~alert & df["flooded"]).sum())
        fp = int((alert & ~df["flooded"]).sum())
        res = {
            "recall_at_high": round(tp / (tp + fn), 3) if tp + fn else None,
            "recall_ci95": wilson(tp, tp + fn),
            "false_alarm_share": round(fp / (tp + fp), 3) if tp + fp else None,
            "alerts": tp + fp,
        }
        if m == "full":
            leads = df.loc[df["flooded"] & df["lead_min"].notna(), "lead_min"]
            res["median_lead_min"] = float(leads.median()) if len(leads) else None
        curve = []
        for cut in range(5, 100, 5):
            a = df[f"max_{m}"] >= cut
            tp_c, fp_c = int((a & df["flooded"]).sum()), int((a & ~df["flooded"]).sum())
            curve.append({"cut": cut, "recall": round(tp_c / max(1, df["flooded"].sum()), 3),
                          "precision": round(tp_c / (tp_c + fp_c), 3) if tp_c + fp_c else None})
        res["pr_curve"] = curve
        report["methods"][m] = res
    report["note"] = ("Few events: treat as indicative. Labels near emergency-map extents are weak "
                      "evidence for street ponding. A missing label is not proof of no flooding.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="nabeul")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rain", help="replay file in data/raw/weather/")
    g.add_argument("--scenario", choices=list(scenarios.SCENARIOS))
    ap.add_argument("--tolerance-m", type=float, default=75)
    ap.add_argument("--radius-km", type=float, default=3)
    ap.add_argument("--lead-h", type=float, default=6)
    args = ap.parse_args()

    cfg = default_config()
    area = Area(args.data_dir / "processed" / args.area, float(cfg.section("weather")["grid_step_deg"]))
    events = gpd.read_file(area.dir / "events.geojson")
    events = events[events["test_set"].astype(str).str.lower().isin(["true", "1"])]
    if events.empty:
        raise SystemExit("no test-set events: import some with --test-set")
    if args.rain:
        store = open_meteo.load_raw(args.data_dir / "raw" / "weather" / args.rain)
        label = args.rain
    else:  # smoke test: put the synthetic storm peak on each event time
        t_ev = pd.Timestamp(events.iloc[0]["event_date"]).to_pydatetime()
        peak = scenarios.SCENARIOS[args.scenario].storms[0].peak_offset_h if scenarios.SCENARIOS[args.scenario].storms else 0
        store = scenarios.build_store(args.scenario, area.cells, area.centre, t_ev - timedelta(hours=peak))
        label = f"scenario:{args.scenario}"
    report = evaluate(area, store, events, args.tolerance_m, args.radius_km, args.lead_h)
    report |= {"rain": label, "engine_version": cfg.version,
               "run_at": datetime.now(timezone.utc).isoformat(),
               "protocol": {"tolerance_m": args.tolerance_m, "radius_km": args.radius_km, "lead_h": args.lead_h}}
    out = args.data_dir / "eval" / f"{args.area}_{label.replace(':', '_').replace('.json', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m, r in report.get("methods", {}).items():
        print(f"{m:7s} recall@High={r['recall_at_high']} CI={r['recall_ci95']} "
              f"false-alarm share={r['false_alarm_share']} alerts={r['alerts']}"
              + (f" median lead={r.get('median_lead_min')} min" if m == "full" else ""))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
