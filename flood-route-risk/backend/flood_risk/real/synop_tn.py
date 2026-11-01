"""Hourly gauge rain of Tunisian synoptic stations, in the input format of `validate_trigger`.

    python -m flood_risk.real.synop_tn --start 2014-09-01 --end 2025-09-01

INM synoptic stations send their observations to the WMO network (SYNOP messages). NOAA archives them in the
Integrated Surface Database, freely available without a request:
https://noaa-global-hourly-pds.s3.amazonaws.com/<year>/<station>.csv

Rain comes in groups `AA1`-`AA4`: "hours,depth in 0.1 mm,condition,quality"; times are UTC and mark the end
of the period. Stations report different periods (3, 6, 12, 18 or 24 h), and many 3-hour reports are missing.

**Raw values are never changed.** Every check below is written to `synop_quality_flags.csv` with the original
value, the reason, the decision and the value used. Nothing is excluded for being large.

Reports (`reports`):

- 9999 means missing. A trace (condition 2) counts as 0.
- The same station, end time and length can come twice: SYNOP carries the 24 h total in two groups, one in
  whole mm (truncated) and one in tenths. A whole value less than `ROUNDING_MM` away from a tenths value is
  the same measurement: the tenths value is used and the pair is counted (`duplicate_rounding`, retained).
  Any other difference, such as 7 against 0.7, is a real conflict. Values further apart cannot both be right and
  nothing tells which: the total is not used (`duplicate_conflict`, treated as missing, review required).

Reconstruction on a 3-hour grid (`slots`), each slot labelled by its end:

1. A 3-hour report gives its slot directly (`method` = measured).
2. A longer total whose slots are all known but one gives that slot by difference (`method` = difference).
   A negative difference is set to 0 and flagged (`negative_residual`, review required).
3. Repeated until nothing changes. A total whose slots are still unknown is spread evenly over them
   (`method` = spread). The spread is an assumption, not a measurement: those slots carry the length of the
   total in `resolution_h`. A spread of `LARGE_SPREAD_MM` or more is flagged (`large_spread`, review required).
4. Every total is then compared with the sum of its rebuilt slots. A gap above `TOLERANCE_MM` is flagged
   (`total_inconsistent`, review required); the slots are kept as rebuilt.
5. Slots that no report covers stay missing.

Each 3-hour slot is spread evenly over its three hours, so an hourly value is never a measurement. The
highest rolling 24 h total is within about 3 h of its true window for measured and differenced slots, and
only as precise as `resolution_h` for spread ones.

These are a few airport and town gauges, not the full INM and DGRE networks.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .areas import raw_dir

S3 = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"
OUT = Path("data/tn_trigger")
SLOT = pd.Timedelta(hours=3)
ROUNDING_MM = 1.0
TOLERANCE_MM = 1.0
LARGE_SPREAD_MM = 50.0
FLAG_COLUMNS = ["station", "end", "hours", "value_mm", "quality", "flag", "detail", "decision", "value_used"]


def flag(station, end, hours, value, quality, kind, detail, decision, used):
    return dict(zip(FLAG_COLUMNS, [station, end, hours, value, quality, kind, detail, decision, used]))


def read_groups(station: str, folder: Path) -> pd.DataFrame:
    """Every AA group of a station, as read: end time, length (h), amount (mm), condition and quality."""
    rows = []
    for f in sorted(glob.glob(str(folder / f"*_{station}.csv"))):
        d = pd.read_csv(f, dtype=str, usecols=lambda c: c == "DATE" or c.startswith("AA"))
        for c in [c for c in d.columns if c.startswith("AA")]:
            x = d[["DATE", c]].dropna(subset=[c])
            p = x[c].str.split(",", expand=True)
            rows.append(pd.DataFrame({"end": pd.to_datetime(x.DATE), "hours": p[0].astype(int),
                                      "mm": p[1].astype(int) / 10, "cond": p[2], "quality": p[3]}))
    if not rows:
        return pd.DataFrame(columns=["end", "hours", "mm", "cond", "quality"])
    return pd.concat(rows, ignore_index=True)


def reports(station: str, folder: Path):
    """Usable rain totals of a station (end, hours, mm) and the flags raised while reading them."""
    r = read_groups(station, folder)
    r = r[(r.mm != 999.9) & r.hours.isin([3, 6, 9, 12, 15, 18, 21, 24])]
    r = r[(r.end.dt.minute == 0) & (r.end.dt.hour % 3 == 0)].copy()
    r.loc[r.cond == "2", "mm"] = 0.0
    r = r.drop_duplicates(["end", "hours", "mm"])
    flags, keep = [], []
    for (end, hours), g in r.groupby(["end", "hours"]):
        if len(g) == 1:
            keep.append((end, hours, g.mm.iloc[0]))
            continue
        values = ", ".join(f"{v:g} (q{q})" for v, q in zip(g.mm, g.quality))
        whole = g[(g.mm * 10).round() % 10 == 0]
        tenths = g[(g.mm * 10).round() % 10 != 0]
        if len(g) == 2 and len(whole) == 1 and len(tenths) == 1 and abs(g.mm.iloc[0] - g.mm.iloc[1]) < ROUNDING_MM:
            used = tenths.mm.iloc[0]
            keep.append((end, hours, used))
            flags.append(flag(station, end, hours, values, "", "duplicate_rounding",
                              "same total in whole mm and in tenths", "retained", used))
        else:
            flags.append(flag(station, end, hours, values, "", "duplicate_conflict",
                              f"values differ by {g.mm.max() - g.mm.min():g} mm",
                              "treated_as_missing; review_required", np.nan))
    return pd.DataFrame(keep, columns=["end", "hours", "mm"]), flags


def slots(rep: pd.DataFrame, start, end, station=""):
    """3-hour slots (labelled by their end) with rain, method and resolution, and the reconstruction flags."""
    grid = pd.date_range(pd.Timestamp(start) + SLOT, end, freq="3h")
    index = {t: i for i, t in enumerate(grid)}
    mm = np.full(len(grid), np.nan)
    res = np.full(len(grid), np.nan)
    method = np.full(len(grid), "", dtype=object)
    flags, totals = [], []
    for r in rep.sort_values("hours").itertuples():
        ids = [index.get(r.end - k * SLOT) for k in range(r.hours // 3)]
        if None not in ids:
            totals.append((r.end, r.hours, r.mm, ids))
    for end_t, h, total, ids in totals:
        if h == 3:
            mm[ids[0]], res[ids[0]], method[ids[0]] = total, 3, "measured"
    changed = True
    while changed:
        changed = False
        for end_t, h, total, ids in totals:
            unknown = [i for i in ids if np.isnan(mm[i])]
            if len(unknown) == 1:
                rest = total - np.nansum(mm[ids])
                if rest < 0:
                    flags.append(flag(station, end_t, h, total, "", "negative_residual",
                                      f"known slots sum to {np.nansum(mm[ids]):g} mm", "retained; review_required", 0.0))
                mm[unknown[0]], res[unknown[0]], method[unknown[0]] = max(rest, 0.0), 3, f"difference_{h}h"
                changed = True
    for end_t, h, total, ids in totals:
        unknown = [i for i in ids if np.isnan(mm[i])]
        if unknown:
            rest = total - np.nansum(mm[ids])
            if rest < 0:
                flags.append(flag(station, end_t, h, total, "", "negative_residual",
                                  f"known slots sum to {np.nansum(mm[ids]):g} mm", "retained; review_required", 0.0))
            rest = max(rest, 0.0)
            mm[unknown], res[unknown], method[unknown] = rest / len(unknown), h, f"spread_{h}h"
            if rest >= LARGE_SPREAD_MM:
                known = [i for i in ids if i not in unknown]
                flags.append(flag(station, end_t, h, total, "", "large_spread",
                                  f"{rest:g} mm spread over {len(unknown)} unknown slots; "
                                  f"{len(known)} known slots hold {np.nansum(mm[known]):g} mm",
                                  "retained; review_required", rest))
    for end_t, h, total, ids in totals:
        rebuilt = np.nansum(mm[ids])
        if abs(rebuilt - total) > TOLERANCE_MM:
            flags.append(flag(station, end_t, h, total, "", "total_inconsistent",
                              f"rebuilt slots sum to {rebuilt:g} mm", "retained; review_required", rebuilt))
    sl = pd.DataFrame({"end": grid, "mm": mm, "resolution_h": res, "method": method})
    return sl, flags


def hourly(sl: pd.DataFrame) -> pd.DataFrame:
    """Spread each slot over its three hours, labelled by the start of the hour."""
    out = []
    for k in (3, 2, 1):
        out.append(pd.DataFrame({"time": sl.end - pd.Timedelta(hours=k), "rain_mm": sl.mm / 3,
                                 "resolution_h": sl.resolution_h, "method": sl.method}))
    return pd.concat(out).sort_values("time").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default=str(OUT / "synop_stations.csv"))
    ap.add_argument("--start", default="2014-09-01")
    ap.add_argument("--end", default="2025-09-01")
    ap.add_argument("--out", default=str(OUT / "synop_hourly.csv.gz"))
    args = ap.parse_args()
    folder = raw_dir("synop_tn")
    stations = pd.read_csv(args.stations, dtype={"station": str})
    frames, summary, flags = [], [], []
    for st in stations.station.unique():
        rep, f_read = reports(st, folder)
        rep = rep[(rep.end > args.start) & (rep.end <= pd.Timestamp(args.end) + pd.Timedelta(days=1))]
        f_read = [f for f in f_read if pd.Timestamp(args.start) < f["end"] <= pd.Timestamp(args.end)]
        sl, f_slots = slots(rep, args.start, args.end, st)
        flags += f_read + f_slots
        h = hourly(sl)
        h = h[(h.time >= args.start) & (h.time < args.end)]
        known = sl.mm.notna()
        summary.append({"station": st, "name": stations.loc[stations.station == st, "name"].iloc[0],
                        "slots_known": known.mean(),
                        **{f"share_{m}": sl.method[known].str.startswith(m).mean()
                           for m in ("measured", "difference", "spread")},
                        **{f"share_res_{r}h": (sl.resolution_h[known] == r).mean() for r in (3, 6, 12, 24)}})
        for zone in stations.loc[stations.station == st, "zone"]:
            frames.append(h.assign(zone=zone, station=st))
    out = pd.concat(frames)[["zone", "station", "time", "rain_mm", "resolution_h", "method"]]
    out.to_csv(args.out, index=False, float_format="%.3f")
    base = Path(args.out).with_suffix("").with_suffix("")
    summary = pd.DataFrame(summary)
    summary.to_csv(base.with_name(base.name.replace("hourly", "coverage") + ".csv"), index=False, float_format="%.3f")
    flags = pd.DataFrame(flags, columns=FLAG_COLUMNS).sort_values(["station", "end", "hours"])
    flags.to_csv(base.with_name(base.name.replace("hourly", "quality_flags") + ".csv"), index=False)
    print(summary.round(2).to_string(index=False))
    print(flags.groupby(["flag", "decision"]).size().to_string())
    print(f"{len(out):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
