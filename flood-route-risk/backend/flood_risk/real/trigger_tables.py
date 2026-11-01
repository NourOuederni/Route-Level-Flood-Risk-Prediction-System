"""Flood and false-trigger tables of one or more `validate_trigger` runs on gauge rain, side by side.

    python -m flood_risk.real.trigger_tables \
        --run before reports/tn_trigger_synop/before_qc reports/tn_trigger_synop/before_qc/hourly.csv.gz \
        --run after reports/tn_trigger_synop/main data/tn_trigger/synop_hourly.csv.gz \
        --out reports/tn_trigger_synop/comparison

Replaces the one-off terminal scripts that gave "7/14" and "6/11". Definitions:

- **Zone-date**: one row of `floods.csv` (a flood reported in a zone on a day). Every date is kept, even when
  one spell holds several. Rows with `in_calibration = 1` (the event the threshold was fitted on) are
  reported but left out of the independent counts.
- **Event-station**: zone-dates that share a station and a date are one event, since they are the same rain
  (e.g. Tunis and Ariana both use Tunis-Carthage). An event is caught if its rain reaches the threshold.
- **Caught**: the matched spell's highest rolling 24 h rain reaches the threshold. A zone-date that matches
  no spell is "unmatched" (no rain recorded around it) and counts as not caught.
- **False trigger, zone-episode**: a spell of a zone with no listed flood whose highest rolling 24 h rain
  reaches the threshold (the "135" of the first run). **Station-episode**: the same, counted once per
  station; a station-episode counts as a flood if any zone using that station had a flood in it.
- **Episode groups**: station-episodes that overlap in time are merged (`validate_trigger.storms`). This is
  an algorithmic grouping, not a list of verified storms.
- **Flagged**: a false station-episode whose spell reach holds a total flagged `review_required` in
  `synop_quality_flags.csv` with at least `FLAG_MM` of rain at stake. It is counted, never removed: the
  count shows how much the result could move once those totals are reviewed.
- **Review** (`--review`): manual decisions on the flagged false station-episodes
  (`confirmed_false_alarm`, `possible_unlisted_flood`, `rainfall_data_unreliable`, `unresolved`). They only
  sort the false triggers into groups; no rain value is changed or removed. Station-episodes and
  zone-episodes are counted separately, since they are different units.

Resolution of each flood, on the 24 h window that gives the spell's highest total (`window_end`):

- `source_resolution`: share of the station's slots measured over 3 h, all period (`synop_coverage.csv`);
- `event_measured_share`: share of the window's rain measured over 3 h or derived by difference;
- `event_spread_share_12h_plus`: share spread evenly from totals of 12 h or more;
- `event_missing_hours`: hours without data in the window;
- `resolution_quality`: `high` if no missing hour and at least 80 % measured or derived; `insufficient` if
  more than 6 missing hours or at least half spread from 12 h totals or longer; `approximate` otherwise.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validate_trigger import storms, wilson

DATA = Path("data/tn_trigger")
FLAG_MM = 20.0


def window_resolution(hourly: pd.DataFrame, station: str, start, end) -> dict:
    """Resolution of the 24 h window with the highest rain within a spell's reach, at one station."""
    h = hourly[hourly.station == station].set_index("time").sort_index()
    reach = h.loc[start - pd.Timedelta(hours=23): end + pd.Timedelta(hours=23)]
    if reach.empty:
        return {}
    rolling = reach.rain_mm.rolling("24h", min_periods=1).sum()
    t = rolling.idxmax()
    w = reach.loc[t - pd.Timedelta(hours=23): t]
    rain = w.rain_mm.fillna(0)
    total = rain.sum()
    out = {"window_end": t + pd.Timedelta(hours=1), "event_missing_hours": int(w.rain_mm.isna().sum())}
    if "method" not in w:
        return out
    method = w.method.fillna("").astype(str)
    measured = method.isin(["measured"]) | method.str.startswith("difference")
    coarse = method.str.startswith("spread") & (w.resolution_h >= 12)
    out["event_measured_share"] = rain[measured].sum() / total if total else np.nan
    out["event_spread_share_12h_plus"] = rain[coarse].sum() / total if total else np.nan
    if out["event_missing_hours"] > 6 or out["event_spread_share_12h_plus"] >= 0.5:
        out["resolution_quality"] = "insufficient"
    elif out["event_missing_hours"] == 0 and out["event_measured_share"] >= 0.8:
        out["resolution_quality"] = "high"
    else:
        out["resolution_quality"] = "approximate"
    return out


def flagged_totals(flags: pd.DataFrame) -> pd.DataFrame:
    """Totals under review that put at least FLAG_MM of rain at stake (station, period start, end)."""
    f = flags[flags.decision.str.contains("review_required", na=False)].copy()
    value = pd.to_numeric(f.value_mm, errors="coerce")
    other = f.detail.str.extract(r"sum to ([\d.]+)")[0].astype(float)
    stake = (value - other).abs().fillna(value).where(f.flag != "large_spread", pd.to_numeric(f.value_used, errors="coerce"))
    f = f[stake.fillna(0) >= FLAG_MM]
    end = pd.to_datetime(f.end)
    return pd.DataFrame({"station": f.station.astype(str), "start": end - pd.to_timedelta(f.hours, unit="h"), "end": end})


def tables(run_dir: Path, hourly_path: Path, floods: pd.DataFrame, stations: pd.DataFrame, coverage: pd.DataFrame,
           flags: pd.DataFrame = None, review: pd.DataFrame = None):
    sp = pd.read_csv(run_dir / "spells.csv", parse_dates=["start", "end"])
    threshold = json.loads((run_dir / "settings.json").read_text())["threshold_mm"]
    hourly = pd.read_csv(hourly_path, parse_dates=["time"], dtype={"station": str})
    hourly = hourly.drop_duplicates(["station", "time"])
    zone_station = dict(zip(stations.zone, stations.station))

    # Zone-dates: every flood date, matched to the spell that claimed it.
    claimed = sp[sp.flood == 1].assign(date=sp.flood_dates.str.split(";")).explode("date")
    zd = floods.merge(claimed[["zone", "date", "start", "end", "rolling_24h_mm", "missing_hours"]],
                      on=["zone", "date"], how="left")
    # Only zones with a station, and dates inside the period the rain file covers.
    zd = zd[zd.zone.isin(zone_station)
            & (pd.to_datetime(zd.date) >= hourly.time.min()) & (pd.to_datetime(zd.date) <= hourly.time.max())].copy()
    zd["station"] = zd.zone.map(zone_station)
    zd["matched"] = zd.start.notna()
    zd["caught"] = zd.rolling_24h_mm >= threshold
    res = [window_resolution(hourly, r.station, r.start, r.end) if r.matched else {} for r in zd.itertuples()]
    zd = pd.concat([zd.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    if "resolution_quality" not in zd:   # rain files without a `method` column give no resolution
        zd["resolution_quality"] = np.nan
    zd["source_resolution"] = zd.station.map(dict(zip(coverage.station, coverage.share_res_3h)))

    # Event-stations: same station, same date.
    ev = (zd.groupby(["station", "date"])
            .agg(zones=("zone", lambda z: ",".join(sorted(z))), in_calibration=("in_calibration", "max"),
                 rolling_24h_mm=("rolling_24h_mm", "max"), caught=("caught", "max"),
                 resolution_quality=("resolution_quality", "first"))
            .reset_index())

    # False triggers.
    neg = sp[(sp.flood == 0) & (sp.rolling_24h_mm >= threshold)]
    sp_st = sp.assign(station=sp.zone.map(zone_station)).dropna(subset=["station"])
    st_ep = (sp_st.groupby(["station", "start", "end"])
                  .agg(flood=("flood", "max"), rolling_24h_mm=("rolling_24h_mm", "max")).reset_index())
    st_false = st_ep[(st_ep.flood == 0) & (st_ep.rolling_24h_mm >= threshold)].copy()
    if flags is not None:
        ft = flagged_totals(flags)
        st_false["flagged"] = [bool(((ft.station == r.station) & (ft.start < r.end + pd.Timedelta(hours=23))
                                     & (ft.end > r.start - pd.Timedelta(hours=23))).any()) for r in st_false.itertuples()]
        st_false.to_csv(run_dir.parent / "comparison" / f"false_station_episodes_{run_dir.name}.csv", index=False)
    summary = {"threshold_mm": threshold}
    for name, d in [("zone_dates", zd), ("event_stations", ev)]:
        ind = d[d.in_calibration == 0]
        k, n = int(ind.caught.sum()), len(ind)
        low, high = wilson(k, n)
        summary[f"{name}_caught_independent"] = f"{k}/{n}"
        summary[f"{name}_recall_95ci"] = f"{low:.2f}-{high:.2f}"
        for q in ["high", "approximate", "insufficient"]:
            if ind.resolution_quality.isna().all():   # rain file without a `method` column
                summary[f"{name}_{q}"] = "not computed: no method column in the rain file"
            else:
                sub = ind[ind.resolution_quality == q]
                summary[f"{name}_{q}"] = f"{int(sub.caught.sum())}/{len(sub)}"
    summary["zone_dates_unmatched"] = int((~zd.matched).sum())
    summary["false_zone_episodes"] = len(neg)
    summary["false_station_episodes"] = len(st_false)
    summary["false_episode_groups"] = storms(st_false)
    summary["spells"] = len(sp)
    if "flagged" in st_false:
        summary["false_station_episodes_flagged"] = int(st_false.flagged.sum())
    if review is not None:
        key = ["station", "start", "end"]
        st_false = st_false.merge(review[key + ["decision"]], on=key, how="left")
        unreviewed = int((st_false.flagged & st_false.decision.isna()).sum())
        summary["false_station_episodes_flagged_unreviewed"] = unreviewed
        if unreviewed:   # the review was made on other spells (another run): do not mix them
            for dec in ["not_flagged", "confirmed_false_alarm", "unresolved", "possible_unlisted_flood",
                        "rainfall_data_unreliable"]:
                for unit in ["station", "zone"]:
                    summary[f"false_{unit}_episodes_{dec}"] = f"not computed: {unreviewed} flagged episodes without a decision"
            return zd, ev, summary
        st_false["decision"] = st_false.decision.fillna("not_flagged")
        zone_false = neg.assign(station=neg.zone.map(zone_station)).merge(st_false[key + ["decision"]], on=key, how="left")
        for dec in ["not_flagged", "confirmed_false_alarm", "unresolved", "possible_unlisted_flood", "rainfall_data_unreliable"]:
            summary[f"false_station_episodes_{dec}"] = int((st_false.decision == dec).sum())
            summary[f"false_zone_episodes_{dec}"] = int((zone_false.decision == dec).sum())
    return zd, ev, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("NAME", "RUN_DIR", "HOURLY"), required=True)
    ap.add_argument("--floods", default=str(DATA / "floods.csv"))
    ap.add_argument("--stations", default=str(DATA / "synop_stations.csv"))
    ap.add_argument("--coverage", default=str(DATA / "synop_coverage.csv"))
    ap.add_argument("--flags", default=str(DATA / "synop_quality_flags.csv"))
    ap.add_argument("--review", default="reports/tn_trigger_synop/review/false_episodes_review.csv")
    ap.add_argument("--out", default="reports/tn_trigger_synop/comparison")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    floods = pd.read_csv(args.floods)
    stations = pd.read_csv(args.stations, dtype={"station": str})
    coverage = pd.read_csv(args.coverage, dtype={"station": str})
    flags = pd.read_csv(args.flags, dtype={"station": str})
    review = None
    if args.review and Path(args.review).exists():
        review = pd.read_csv(args.review, dtype={"station": str}, parse_dates=["start", "end"], keep_default_na=False)
        review = review[review.decision != ""]
    summaries = {}
    for name, run_dir, hourly in args.run:
        zd, ev, summaries[name] = tables(Path(run_dir), Path(hourly), floods, stations, coverage, flags, review)
        zd.to_csv(out / f"zone_dates_{name}.csv", index=False, float_format="%.3f")
        ev.to_csv(out / f"event_stations_{name}.csv", index=False, float_format="%.3f")
    s = pd.DataFrame(summaries)
    s.to_csv(out / "summary.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()
