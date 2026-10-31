"""Plain data types used by the risk engine. No I/O, no framework imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SegmentStatic:
    """Precomputed, normalised static factors of one road segment (all in [0, 1]).

    f_* follow guide Section 7.2 (1 = more flood-prone).
    dv_* are the static drainage-vulnerability proxies of Section 7.3.
    """

    segment_id: int
    f_depression: float = 0.0
    f_drainage_position: float = 0.0
    f_low_slope: float = 0.0
    f_impervious: float = 0.0
    f_crossing: float = 0.0
    f_past_flood: float = 0.0
    dv_urban: float = 0.0
    dv_blackspot: float = 0.0
    dv_drain_density_inv: float = 0.5
    dv_maintenance: float | None = None   # None = no authority data for this segment
    poorly_mapped: bool = False
    past_flood_confirmed: bool = False
    crossing_kind: str | None = None      # underpass | ford | low_crossing | None

    def factor_vector(self) -> dict[str, float]:
        return {
            "depression": self.f_depression,
            "drainage_position": self.f_drainage_position,
            "low_slope": self.f_low_slope,
            "impervious": self.f_impervious,
            "crossing": self.f_crossing,
            "past_flood": self.f_past_flood,
        }


@dataclass(frozen=True)
class RainWindow:
    """Rainfall seen by one segment over its ETA window, one entry per weather model."""

    intensity_mm_h: tuple[float, ...]      # I_W per model (max hourly intensity in window)
    antecedent_mm: tuple[float, ...]       # A_6h per model
    source: str
    retrieved_at: datetime
    age_minutes: float
    interval_minutes: int = 60
    interpolated: bool = True              # 15-min values interpolated from hourly (true in Tunisia)
    dry_spell_days: float | None = None
    models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.intensity_mm_h) != len(self.antecedent_mm) or not self.intensity_mm_h:
            raise ValueError("RainWindow needs one intensity and one antecedent value per model")


@dataclass(frozen=True)
class EvalContext:
    refresh_interval_minutes: float = 180.0
    recent_feedback: str | None = None     # "consistent_flooded" | "consistent_dry" | None


@dataclass
class Explanation:
    code: str
    params: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class SegmentRisk:
    segment_id: int
    score: float
    level: str
    confidence: float
    confidence_label: str
    hazard: float | None
    susceptibility: float
    drainage_vulnerability: float
    threshold_mm_h: float
    effective_rain_mm_h: float | None
    mode: str                              # "full" | "static_only"
    explanations: list[Explanation]
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "score": round(self.score, 1),
            "level": self.level,
            "confidence": round(self.confidence, 2),
            "confidence_label": self.confidence_label,
            "hazard": None if self.hazard is None else round(self.hazard, 3),
            "susceptibility": round(self.susceptibility, 3),
            "drainage_vulnerability": round(self.drainage_vulnerability, 3),
            "threshold_mm_h": round(self.threshold_mm_h, 2),
            "effective_rain_mm_h": None
            if self.effective_rain_mm_h is None
            else round(self.effective_rain_mm_h, 2),
            "mode": self.mode,
            "explanations": [
                {"code": e.code, "params": e.params, "text": e.text} for e in self.explanations
            ],
            "contributions": {k: round(v, 3) for k, v in self.contributions.items()},
        }
