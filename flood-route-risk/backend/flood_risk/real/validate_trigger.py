"""Check the provisional rain trigger on hourly rain records (INM gauges), with the window used in operation.

    python -m flood_risk.real.validate_trigger --rain inm_hourly.csv --floods inm_floods.csv

File formats and sources: `docs/TRIGGER_VALIDATION.md`. Every choice below is a command-line option and is
written to `<out>/settings.json`, so a result can always be traced back to its definitions.

- Zone: the area the trigger watches (an `aoi` of the model). A zone can have several stations; the 24 h
  total of the wettest station counts, as the trigger takes the wettest cell's 24 h total in operation.
- Wet spell: starts at the first hour with rain >= `wet_mmh`; ends at the last wet hour before more than
  `gap_h` hours without a wet hour. Kept if its total reaches `min_spell_mm`.
- Flood spell: each flood date of the zone goes to one spell only: the spell whose days contain it (the
  wettest that day if several do, possible only when `gap_h` < 24), or else the last spell that ended at
  most `match_days` before it (floods are often reported the day after the rain). A date with several
  candidate spells is flagged (`flood_ambiguous`) for a manual check, since missing hours can make the
  wrong spell look wettest. All other spells count as spells without a flood. Those that overlap a flood in another zone
  are flagged (`flood_other_zone`).
- Alert: a spell whose highest rolling 24 h rain reaches the threshold. A spell gives at most one alert,
  even if the rain crosses the threshold twice.
- Hours: each hour of a zone is `rain` or `zero` (at least one station measured it, > 0 or 0 mm),
  `not_recorded` (no value although a station was in service: transmission or gauge fault) or
  `no_station` (no station in service). A station is in service over the periods given with `--stations`
  (both ends included; a date without a time means the whole day), else from its first to its last row.
  A value recorded outside a declared period still counts: measurements take priority over the service
  file, and such values are counted in `zones.csv` (`values_outside_service`) so the file can be checked.
  A missing value is an empty `rain_mm` or an absent row, never 0.
- Duplicates: rows repeating the same station, hour and value are dropped and counted; the same station
  and hour with two different values stops the run, since summing or picking one would hide an error.
- Records shorter than an hour (e.g. every 10 minutes) need a `minutes` column: each row is the rain
  over [time, time + minutes), or (time - minutes, time] with `--time-label end`. Records of a station
  must not overlap (that would count rain twice) nor cross the hour; both stop the run. An hour gets the
  sum of its records only if they cover all 60 minutes with values, otherwise it is missing. Without a
  `minutes` column every record is one hour and must start on the hour.
- Complete spell: at most `max_missing_h` missing hours within its 24 h reach. Recall is given on all
  flood spells and on complete ones, since a flood missed during a gap says little about the threshold.
- False triggers per zone per year: alerts on spells without a flood in all zones, divided by the years of
  hourly rain recorded in all zones (an hour is recorded when at least one station of the zone has it;
  8766 hours per year). Gaps therefore lower both the count and the denominator. This pooled rate weights
  zones by their record length; `zones.csv` gives the rate of each zone.
- Time of alert: `observed_24h_reached` is the end of the first 24 h of observed rain above the threshold,
  i.e. when a trigger fed with observations only (t - 24 h to t) fires. The operational window
  (t - 12 h to t + 12 h) reaches the same total 12 h earlier only if the 12 h forecast is perfect; the
  forecast is not part of this validation, so the real alert time is not measured here.

The threshold in `models/final/trigger.json` is never changed by this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

DEFAULTS = {"wet_mmh": 0.2, "gap_h": 24, "min_spell_mm": 5.0, "match_days": 1, "max_missing_h": 0,
            "time_label": "start"}
THRESHOLDS = [30, 35, 40, 44, 50, 55, 60, 70]
WINDOWS = ["rolling_24h_mm", "calendar_day_mm"]
SPELL_COLUMNS = ["start", "end", "total_mm", "rolling_24h_mm", "calendar_day_mm", "missing_hours", "flood",
                 "flood_dates", "flood_ambiguous", "flood_other_zone"]


def intervals(rain: pd.DataFrame, label: str = "start") -> pd.DataFrame:
    """Check the time step of each record and give `time` as the start of its interval (module docstring)."""
    if "minutes" not in rain:
        if label == "end":
            rain = rain.assign(time=rain.time - pd.Timedelta(hours=1))
        off = rain[rain.time != rain.time.dt.floor("h")]
        if len(off):
            raise ValueError(f"{len(off)} records do not start on the hour; give their length in a `minutes` "
                             f"column. First ones: {off.head(5).to_dict('records')}")
        return rain.assign(minutes=60)
    step = pd.to_timedelta(rain.minutes, unit="min")
    rain = rain.assign(time=rain.time - step) if label == "end" else rain
    end = rain.time + step
    bad = rain[(rain.minutes <= 0) | (rain.minutes > 60) | (end > rain.time.dt.floor("h") + pd.Timedelta(hours=1))]
    if len(bad):
        raise ValueError(f"{len(bad)} records are longer than an hour or cross the hour. "
                         f"First ones: {bad.head(5).to_dict('records')}")
    r = rain.assign(end=end).sort_values(["zone", "station", "time"])
    overlap = r[(r.time < r.groupby(["zone", "station"]).end.shift()).to_numpy()]
    if len(overlap):
        raise ValueError(f"{len(overlap)} records overlap the previous record of their station; the rain "
                         f"would be counted twice. First ones: {overlap.head(5).to_dict('records')}")
    return rain


def zone_series(rain: pd.DataFrame) -> pd.DataFrame:
    """Hourly rain of a zone, one column per station, hours without a full hour of values left as NaN."""
    minutes = rain.minutes if "minutes" in rain else 60
    r = rain.assign(hour=rain.time.dt.floor("h"), covered=pd.Series(minutes, index=rain.index).where(rain.rain_mm.notna(), 0))
    g = r.groupby(["hour", "station"])
    # `intervals` rules out overlaps and records crossing the hour, so 60 covered minutes means no gap.
    hourly = g.rain_mm.sum(min_count=1).where(g.covered.sum() == 60).unstack("station")
    return hourly.resample("h").sum(min_count=1)


def check_rain(rain: pd.DataFrame):
    """Drop exact duplicate rows; stop on the same station and time with different values.

    A station may serve two zones: it then appears once per zone, with the same values.
    """
    n = len(rain)
    rain = rain.drop_duplicates(["zone", "station", "time", "rain_mm"])
    dropped = n - len(rain)
    values = rain.rain_mm.fillna(-1).groupby([rain.station, rain.time]).transform("nunique")
    clash = rain[values > 1]
    if len(clash):
        raise ValueError(f"{len(clash)} rows give different values for the same station and time; "
                         f"fix the input:\n{clash.sort_values(['station', 'time']).head(10).to_string(index=False)}")
    return rain, dropped


def in_service(st: pd.DataFrame, service: pd.DataFrame) -> pd.DataFrame:
    """True where a station of the zone is in service, one column per station, both ends included."""
    ins = pd.DataFrame(False, index=st.index, columns=st.columns)
    for _, r in service[service.station.isin(st.columns)].iterrows():
        ins.loc[r.start:r.end, r.station] = True
    return ins


def hour_status(st: pd.DataFrame, service: pd.DataFrame) -> pd.Series:
    """`rain`, `zero`, `not_recorded` or `no_station` for each hour of a zone (see the module docstring)."""
    s = st.max(axis=1)
    status = pd.Series("no_station", index=st.index)
    status[s.isna() & in_service(st, service).any(axis=1)] = "not_recorded"
    status[s == 0] = "zero"
    status[s > 0] = "rain"
    return status


def spells(stations, flood_days, other_flood_days=(), threshold=None, wet_mmh=0.2, gap_h=24,
           min_spell_mm=5.0, match_days=1) -> pd.DataFrame:
    """Wet spells of one zone with their rolling 24 h and calendar-day maxima.

    `stations` holds the hourly rain of each station (a Series for a single one). Spells are split on the
    wettest station each hour. The 24 h and daily totals are summed per station first and the wettest
    station is taken afterwards, as the trigger takes the wettest cell's 24 h total in operation.
    """
    stations = stations.to_frame() if isinstance(stations, pd.Series) else stations
    columns = SPELL_COLUMNS + ([] if threshold is None else ["observed_24h_reached"])
    s = stations.max(axis=1)
    wet = s[s >= wet_mmh]
    if wet.empty:
        return pd.DataFrame(columns=columns)
    new = wet.index.to_series().diff() > pd.Timedelta(hours=gap_h)
    # Trailing 24 h; its maximum over the spell equals the centred window's. Missing hours count as 0.
    rolling = stations.rolling("24h", min_periods=1).sum().max(axis=1)
    daily = stations.resample("D").sum(min_count=1).max(axis=1)
    blocks = [(b.index[0], b.index[-1]) for _, b in wet.groupby(new.cumsum())]
    blocks = [(a, b) for a, b in blocks if s[a:b].sum() >= min_spell_mm]
    first_day = [a.normalize() for a, _ in blocks]
    last_day = [b.normalize() for _, b in blocks]
    lag = pd.Timedelta(days=match_days)

    def owner(d):
        """The spell a flood date belongs to, and whether another spell could also claim it: the wettest
        one that day among those containing it, else the last one ended within `lag`."""
        inside = [i for i in range(len(blocks)) if first_day[i] <= d <= last_day[i]]
        before = [i for i in range(len(blocks)) if last_day[i] < d <= last_day[i] + lag]
        if inside:
            day = s[d:d + pd.Timedelta(hours=23)]
            return max(inside, key=lambda i: day[blocks[i][0]:blocks[i][1]].sum()), len(inside) + len(before) > 1
        return (before[-1], len(before) > 1) if before else (None, False)

    matched, ambiguous = {}, set()
    for d in sorted(set(flood_days)):
        i, amb = owner(d)
        matched.setdefault(i, []).append(d)
        if amb:
            ambiguous.add(i)
    others = {owner(d)[0] for d in set(other_flood_days)}
    out = []
    for i, (start, end) in enumerate(blocks):
        window = rolling[start:end + pd.Timedelta(hours=23)]
        row = {"start": start, "end": end, "total_mm": s[start:end].sum(),
               "rolling_24h_mm": window.max(),
               "calendar_day_mm": daily.reindex(pd.date_range(first_day[i], last_day[i], freq="D")).max(),
               # every hour that enters the spell's 24 h totals, so a gauge that stops mid-storm is caught
               "missing_hours": int(s[start - pd.Timedelta(hours=23):end + pd.Timedelta(hours=23)].isna().sum()),
               "flood": int(i in matched), "flood_dates": ";".join(f"{d:%Y-%m-%d}" for d in matched.get(i, [])),
               "flood_ambiguous": int(i in ambiguous), "flood_other_zone": int(i in others)}
        if threshold is not None:
            above = window[window >= threshold]
            row["observed_24h_reached"] = above.index[0] if len(above) else pd.NaT
        out.append(row)
    return pd.DataFrame(out, columns=columns)


def wilson(k: int, n: int, level: float = 0.95):
    """Confidence interval of a proportion k / n (Wilson score); (nan, nan) when n = 0."""
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def poisson_rate(k: int, years: float, level: float = 0.95):
    """Exact confidence interval of a rate of k events over `years`."""
    a = 1 - level
    low = chi2.ppf(a / 2, 2 * k) / 2 if k else 0.0
    return low / years, chi2.ppf(1 - a / 2, 2 * k + 2) / 2 / years


def storms(spells: pd.DataFrame) -> int:
    """Number of distinct storms among spells of several zones: spells that overlap in time are one storm."""
    count, end = 0, pd.Timestamp.min
    for s, e in spells.sort_values("start")[["start", "end"]].itertuples(index=False):
        if s > end:
            count += 1
        end = max(end, e)
    return count


def scan(table: pd.DataFrame, col: str, years: float, thresholds) -> pd.DataFrame:
    """Floods caught and false triggers per threshold, with 95 % intervals: with few floods the interval,
    not the point value, says what the data can support."""
    pos, neg = table[table.flood == 1], table[table.flood == 0]
    full = pos[pos.complete == 1]
    rows = []
    for t in thresholds:
        caught, false = int((full[col] >= t).sum()), int((neg[col] >= t).sum())
        rows.append({"window": col, "threshold": t,
                     "flood_spells_caught": f"{(pos[col] >= t).sum()}/{len(pos)}",
                     "recall": (pos[col] >= t).mean(),
                     "complete_flood_spells_caught": f"{caught}/{len(full)}",
                     "recall_complete": caught / len(full) if len(full) else np.nan,
                     "recall_complete_low": wilson(caught, len(full))[0],
                     "recall_complete_high": wilson(caught, len(full))[1],
                     # missed floods whose spell has missing hours: the miss may come from the gap
                     "missed_with_missing_hours": int(((pos[col] < t) & (pos.missing_hours > 0)).sum()),
                     "spells_triggered_without_flood": f"{false}/{len(neg)}",
                     "of_which_flood_other_zone": int(((neg[col] >= t) & (neg.flood_other_zone == 1)).sum()),
                     "false_triggers_per_zone_per_year": false / years,
                     "false_triggers_low": poisson_rate(false, years)[0],
                     "false_triggers_high": poisson_rate(false, years)[1],
                     # One storm can trigger several zones at once. The Poisson interval assumes independent
                     # triggers; when storms are much fewer than triggers, it is probably too narrow.
                     "false_trigger_storms": storms(neg[neg[col] >= t])})
    return pd.DataFrame(rows)


def restrict(rain: pd.DataFrame, floods: pd.DataFrame, start=None, end=None):
    """Keep the records and flood dates in [start, end), to choose a threshold on one period and test it on
    another. Cut in the dry season (e.g. 1 July) so that no storm, and none of the zones it flooded, falls
    on both sides."""
    start = pd.Timestamp(start) if start else pd.Timestamp.min
    end = pd.Timestamp(end) if end else pd.Timestamp.max
    dates = pd.to_datetime(floods.date)
    return (rain[(rain.time >= start) & (rain.time < end)],
            floods[(dates >= start) & (dates < end)])


def read_inputs(rain_path, floods_path, stations_path=None):
    """Read the CSV files. Empty `rain_mm` cells stay NaN (missing), never 0."""
    rain = pd.read_csv(rain_path, parse_dates=["time"], dtype={"station": str})
    floods = pd.read_csv(floods_path)
    service = None
    if stations_path:
        raw = pd.read_csv(stations_path, dtype=str)
        service = pd.DataFrame({"station": raw.station, "start": pd.to_datetime(raw.start)})
        end = raw.end.fillna("")
        # empty end: still in service; a date alone covers the whole day, up to its last hour
        service["end"] = [rain.time.max() if e == "" else
                          pd.Timestamp(e) + (pd.Timedelta(hours=23) if len(e.strip()) <= 10 else pd.Timedelta(0))
                          for e in end]
    return rain, floods, service


def run(rain: pd.DataFrame, floods: pd.DataFrame, threshold: float, settings: dict, service=None):
    """Spell table, per-zone coverage, threshold scan, matched floods, years recorded and a summary.

    `service`: periods each station was in service (`station, start, end`, both included). Stations it
    does not list are in service from their first to their last row in `rain`.
    """
    settings = dict(settings)
    max_missing = settings.pop("max_missing_h", 0)
    rain, duplicates = check_rain(rain)
    rain = intervals(rain, settings.pop("time_label", "start"))
    floods = floods.assign(date=pd.to_datetime(floods.date).dt.normalize())
    default = rain.groupby("station").time.agg(start="min", end="max").reset_index()
    service = default if service is None else pd.concat(
        [service, default[~default.station.isin(service.station)]], ignore_index=True)
    rows, coverage = [], []
    for zone, z in rain.groupby("zone"):
        st = zone_series(z)
        status = hour_status(st, service).value_counts()
        outside = int((st.notna() & ~in_service(st, service)).to_numpy().sum())
        recorded = int(status.get("rain", 0) + status.get("zero", 0))
        coverage.append({"zone": zone, "stations": st.shape[1], "first": st.index[0], "last": st.index[-1],
                         "hours": len(st), "hours_with_data": recorded, "coverage": recorded / len(st),
                         **{f"hours_{k}": int(status.get(k, 0)) for k in ["rain", "zero", "not_recorded", "no_station"]},
                         "values_outside_service": outside})
        sp = spells(st, floods.loc[floods.zone == zone, "date"], floods.loc[floods.zone != zone, "date"],
                    threshold, **settings)
        rows.append(sp.assign(zone=zone))
    table = pd.concat(rows, ignore_index=True)
    table["complete"] = (table.missing_hours <= max_missing).astype(int)
    coverage = pd.DataFrame(coverage)
    alert = table.rolling_24h_mm >= threshold
    per_zone = table.assign(caught=alert & (table.flood == 1), false=alert & (table.flood == 0)).groupby("zone").agg(
        spells=("flood", "size"), flood_spells=("flood", "sum"), caught=("caught", "sum"), false_triggers=("false", "sum"))
    coverage = coverage.merge(per_zone, left_on="zone", right_index=True, how="left")
    coverage[per_zone.columns] = coverage[per_zone.columns].fillna(0).astype(int)   # zones without any spell
    coverage["false_triggers_per_year"] = coverage.false_triggers / (coverage.hours_with_data / 8766)
    years = coverage.hours_with_data.sum() / 8766
    res = pd.concat([scan(table, c, years, sorted(set(THRESHOLDS) | {threshold})) for c in WINDOWS])

    matched = {(z, d) for z, ds in zip(table.zone, table.flood_dates) for d in ds.split(";") if d}
    floods["matched"] = [(z, f"{d:%Y-%m-%d}") in matched for z, d in zip(floods.zone, floods.date)]
    pos, neg = table[table.flood == 1], table[table.flood == 0]
    summary = {"zones": len(coverage), "zone_years_recorded": years, "duplicate_rows_dropped": duplicates,
               "values_outside_service": int(coverage.values_outside_service.sum()),
               "flood_dates": len(floods), "flood_dates_unmatched": int((~floods.matched).sum()),
               "flood_spells": len(pos), "flood_spells_incomplete": int((pos.complete == 0).sum()),
               "flood_spells_ambiguous": int(pos.flood_ambiguous.sum()),
               "spells_without_flood": len(neg), "spells_without_flood_incomplete": int((neg.complete == 0).sum())}
    return table, coverage, res, floods, years, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rain", required=True)
    ap.add_argument("--floods", required=True)
    ap.add_argument("--trigger", default="models/final/trigger.json")
    ap.add_argument("--stations", help="optional CSV station,start,end: periods each station was in service")
    ap.add_argument("--period-start", help="keep records from this date (included), e.g. 2015-07-01")
    ap.add_argument("--period-end", help="keep records before this date (excluded)")
    ap.add_argument("--out", default="reports/tn_trigger")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v,
                        **({"choices": ["start", "end"]} if k == "time_label" else {}))
    args = ap.parse_args()
    settings = {k: getattr(args, k) for k in DEFAULTS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    threshold = json.loads(Path(args.trigger).read_text())["threshold_mm"]

    rain, floods, service = read_inputs(args.rain, args.floods, args.stations)
    rain, floods = restrict(rain, floods, args.period_start, args.period_end)
    table, coverage, res, floods, years, summary = run(rain, floods, threshold, settings, service)
    table.to_csv(out / "spells.csv", index=False)
    coverage.to_csv(out / "zones.csv", index=False)
    floods.to_csv(out / "floods_matched.csv", index=False)
    res.to_csv(out / "thresholds.csv", index=False)
    (out / "settings.json").write_text(json.dumps({**settings, "threshold_mm": threshold, "rain": args.rain,
                                                   "floods": args.floods, "stations": args.stations,
                                                   "period": [args.period_start, args.period_end]}, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{summary['zones']} zones, {years:.1f} zone-years of hourly rain recorded, "
          f"{summary['flood_spells']} flood spells ({summary['flood_spells_incomplete']} incomplete, "
          f"{summary['flood_spells_ambiguous']} ambiguous), {summary['spells_without_flood']} spells without flood "
          f"({summary['spells_without_flood_incomplete']} incomplete)")
    if (~floods.matched).any():
        print(f"\n{(~floods.matched).sum()} flood dates match no wet spell (zone absent, rain missing or wrong date):")
        print(floods[~floods.matched].to_string(index=False))
    if (coverage.coverage < 0.9).any():
        print("\nzones with less than 90 % of hours recorded:")
        print(coverage[coverage.coverage < 0.9].round(3).to_string(index=False))
    if table.missing_hours.sum():
        print(f"\nwarning: {int((table.missing_hours > 0).sum())} spells have missing hours "
              f"({int(((table.missing_hours > 0) & (table.flood == 1)).sum())} of them flood spells)")
    if summary["duplicate_rows_dropped"]:
        print(f"\n{summary['duplicate_rows_dropped']} duplicate rows (same station, hour and value) dropped")
    if summary["values_outside_service"]:
        print(f"\n{summary['values_outside_service']} station-hours have a value outside the declared service "
              "periods; they are kept (see zones.csv)")
    if table.flood_ambiguous.sum():
        print(f"\ncheck by hand: {int(table.flood_ambiguous.sum())} flood spells could also be a neighbouring spell")
    print("\n" + res.round(3).to_string(index=False))
    print(f"\nPer zone at {threshold} mm (rolling window):")
    print(coverage.drop(columns=["first", "last", "hours"]).round(3).to_string(index=False))
    print(f"\nCurrent threshold {threshold} mm (rolling window, as in operation):")
    print(res[(res.threshold == threshold) & (res.window == "rolling_24h_mm")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
