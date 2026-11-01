"""Pilot-zone terrain model (handoff V2, kind A_terrain) and zone rain trigger.

Where a road lies inside the zone table, its level comes from the trained model instead of the
algorithmic per-segment score. The model reads terrain only, so every zone segment is scored
once at start-up. The zone-wide rain trigger (contract: 24 h, t-12 h to t+12 h) decides whether
those levels are shown as alerts. Below the trigger, the terrain exposure is reported as
`susceptibility` and the level stays "low".

Outside the zone the planner keeps the algorithmic scorer.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..engine import SegmentRisk
from ..engine.types import Explanation

log = logging.getLogger(__name__)

# Pilot zone (Nabeul, rectangle from flood_risk/real/build_zone.py) and its centre
ZONE_BOX = {"south": 36.41, "north": 36.50, "west": 10.67, "east": 10.79}
ZONE_CENTRE = (36.455, 10.73)

M_PER_DEG_LAT = 111_320.0
MATCH_MAX_M = 200.0            # graph segment is matched to a zone segment within this distance

MODEL_LEVEL_TO_ENGINE = {"Low": "low", "Moderate": "moderate", "High": "high", "Very high": "very_high"}
ENGINE_LEVEL_INDEX = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}


class ZoneModel:
    def __init__(self, zone_csv: Path, model_path: Path, trigger_path: Path, cells) -> None:
        import joblib
        from flood_risk.predict import score_route

        self.bundle = joblib.load(model_path)
        self.trigger_cfg = json.loads(trigger_path.read_text(encoding="utf-8"))
        self.trigger_column = self.trigger_cfg["column"]
        self.threshold_mm = float(self.trigger_cfg["threshold_mm"])

        zone = pd.read_csv(zone_csv)
        # trigger=None: levels are computed without the rain gate; the gate is applied per request
        out = score_route(self.bundle, zone, "startup", area=zone, trigger=None)
        self.kx = M_PER_DEG_LAT * math.cos(math.radians(ZONE_CENTRE[0]))
        xy = np.column_stack([zone["mid_lon"].to_numpy() * self.kx, zone["mid_lat"].to_numpy() * M_PER_DEG_LAT])
        self.tree = cKDTree(xy)
        self.rows = [
            {
                "level": MODEL_LEVEL_TO_ENGINE[out.at[i, "susceptibility"]],
                "score": float(out.at[i, "risk_score"]),
                "conf": float(out.at[i, "confidence"]),
                "conf_label": str(out.at[i, "confidence_level"]).lower(),
                "factors": str(out.at[i, "main_factors"]),
            }
            for i in range(len(out))
        ]
        # Weather cells (0.25 deg, centre-based) whose footprint overlaps the zone rectangle
        half = 0.125
        self.cells = [c.cell_id for c in cells
                      if c.lat - half < ZONE_BOX["north"] and c.lat + half > ZONE_BOX["south"]
                      and c.lon - half < ZONE_BOX["east"] and c.lon + half > ZONE_BOX["west"]]
        counts = pd.Series([r["level"] for r in self.rows]).value_counts().to_dict()
        log.info("zone model: %d segments scored (%s), trigger cells %s",
                 len(self.rows), counts, self.cells)

    def lookup(self, lat: float, lon: float) -> int | None:
        d, i = self.tree.query([lon * self.kx, lat * M_PER_DEG_LAT], k=1)
        return int(i) if d <= MATCH_MAX_M else None

    def trigger(self, store, now: datetime) -> dict:
        """Zone rain trigger: largest 24 h total (12 h observed + 12 h forecast) over the zone cells.

        Uses the ensemble mean of the weather models, hour by hour.
        """
        base = {
            "threshold_mm": self.threshold_mm,
            "window": "12 h observed + 12 h forecast",
            "validation": self.trigger_cfg.get("status", "provisional"),
        }
        if store is None:
            return {**base, "status": "unknown", "max_rain_24h_mm": None}
        t0 = (now - timedelta(hours=12)).timestamp()
        t1 = (now + timedelta(hours=12)).timestamp()
        best = 0.0
        for cid in self.cells:
            cs = store.series.get(cid)
            if cs is None or cs.times.size == 0 or not cs.values:
                continue
            ens = np.mean(np.vstack(list(cs.values.values())), axis=0)
            m = (cs.times > t0) & (cs.times <= t1)
            if m.any():
                best = max(best, float(np.nansum(ens[m])))
        status = "on" if best >= self.threshold_mm else "off"
        return {**base, "status": status, "max_rain_24h_mm": round(best, 1)}

    def risk(self, idx: int, segment_id: int, trigger_on: bool) -> SegmentRisk:
        r = self.rows[idx]
        sus = ENGINE_LEVEL_INDEX[r["level"]] / 3.0
        level = r["level"] if trigger_on else "low"
        return SegmentRisk(
            segment_id=segment_id,
            score=r["score"],
            level=level,
            confidence=r["conf"],
            confidence_label=r["conf_label"],
            hazard=None,
            susceptibility=sus,
            drainage_vulnerability=0.0,
            threshold_mm_h=0.0,
            effective_rain_mm_h=None,
            mode="terrain_model",     # not "static_only": the zone has a rain trigger
            explanations=[Explanation(code="terrain_model",
                                      text=f"Terrain model (zone). Main factors: {r['factors']}")],
            contributions={},
        )
