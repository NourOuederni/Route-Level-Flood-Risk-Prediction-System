"""Save the rainfall that was available around a past event, in the replay format (guide 4.5, 8.1).

    python -m pipeline.fetch_event_rain --area nabeul --start 2024-01-10 --end 2024-01-12
    python -m pipeline.fetch_event_rain --area nabeul --start 2018-09-21 --end 2018-09-23 --source era5

--source forecast  Open-Meteo historical FORECAST archive (from ~2021-2022): what a forecast said.
--source era5      ERA5 reanalysis (1940+): already "knows" what happened. Report separately!
Output: data/raw/weather/openmeteo_<source>_<start>.json, usable as WEATHER_MODE=replay:<file>
and by evaluation.replay_eval.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml

from app.engine import default_config
from app.weather.store import build_grid

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "era5": "https://archive-api.open-meteo.com/v1/archive",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="nabeul")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--source", choices=list(URLS), default="forecast")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    bbox = yaml.safe_load((Path(__file__).parent / "areas.yaml").read_text())[args.area]["bbox"]
    wcfg = default_config().section("weather")
    cells = build_grid(tuple(bbox), float(wcfg["grid_step_deg"]))
    models = list(wcfg["models"]) if args.source == "forecast" else ["era5"]
    common = {"latitude": ",".join(str(c.lat) for c in cells),
              "longitude": ",".join(str(c.lon) for c in cells),
              "timeformat": "unixtime", "timezone": "GMT"}
    hourly = {**common, "hourly": "precipitation", "start_date": args.start, "end_date": args.end}
    if args.source == "forecast":
        hourly["models"] = ",".join(models)
    with httpx.Client(timeout=60) as c:
        h = c.get(URLS[args.source], params=hourly); h.raise_for_status()
        start_hist = (datetime.fromisoformat(args.start) - timedelta(days=60)).date().isoformat()
        d = c.get(URLS["era5"], params={**common, "daily": "precipitation_sum",
                                        "start_date": start_hist, "end_date": args.start})
        d.raise_for_status()
    payload = h.json()
    if args.source == "era5":  # rename to look like a single model
        for loc in (payload if isinstance(payload, list) else [payload]):
            loc["hourly"]["precipitation_era5"] = loc["hourly"].pop("precipitation")
    out = args.data_dir / "raw" / "weather" / f"openmeteo_{args.source}_{args.start}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # retrieved_at = start of the period, so replayed data are "fresh" for the event
    out.write_text(json.dumps({
        "retrieved_at": datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).isoformat(),
        "models": models, "cells": [c.__dict__ for c in cells], "kind": args.source,
        "hourly": payload, "daily": d.json()}), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
