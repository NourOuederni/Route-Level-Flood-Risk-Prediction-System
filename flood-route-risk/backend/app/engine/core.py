"""Version 1 risk engine: deterministic, transparent formulas (guide Section 7).

    Score(s, t) = 100 * H(s, t) * S'(s)
    S'          = s_min + (1 - s_min) * S,   S = sum_i w_i f_i
    R_eff       = I_W + kappa * A_6h
    DV          = sum_j v_j proxy_j
    T_eff       = T0 * (1 - beta * DV)
    H           = min(1, R_eff / T_eff)

No learned model is involved: every number comes from config/engine.yaml and the
precomputed segment factors, so every score can be decomposed and explained.
All functions here are pure (no I/O, no clock).
"""
from __future__ import annotations

from statistics import fmean

from .config import LEVEL_RANK, EngineConfig
from .types import EvalContext, Explanation, RainWindow, SegmentRisk, SegmentStatic

FACTOR_LABELS = {
    "depression": "deep local depression / wet ground",
    "drainage_position": "low above nearest drainage / close to a wadi",
    "low_slope": "flat road, water stays",
    "impervious": "highly paved surroundings",
    "crossing": "underpass, ford or low crossing",
    "past_flood": "flooded in the past / known black spot",
}


def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# --- static part -----------------------------------------------------------
def susceptibility(seg: SegmentStatic, cfg: EngineConfig) -> tuple[float, dict[str, float]]:
    """Return S in [0,1] and the weighted contribution of each factor."""
    contrib = {
        name: float(cfg.s_weights[name]) * clip01(value)
        for name, value in seg.factor_vector().items()
    }
    return clip01(sum(contrib.values())), contrib


def floored_susceptibility(s: float, cfg: EngineConfig) -> float:
    return cfg.s_min + (1.0 - cfg.s_min) * clip01(s)


def dry_spell_factor(days: float | None, cfg: EngineConfig) -> float:
    ds = cfg.section("drainage")["dry_spell"]
    if days is None:
        return float(ds["default_factor"])
    return clip01(max(days, 0.0) / float(ds["cap_days"]))


def drainage_vulnerability(seg: SegmentStatic, dry_spell_days: float | None,
                           cfg: EngineConfig) -> float:
    w = cfg.dv_weights
    maintenance = seg.dv_maintenance if seg.dv_maintenance is not None else 0.0
    dv = (
        w["urban"] * clip01(seg.dv_urban)
        + w["blackspot"] * clip01(seg.dv_blackspot)
        + w["dry_spell"] * dry_spell_factor(dry_spell_days, cfg)
        + w["drain_density_inv"] * clip01(seg.dv_drain_density_inv)
        + w.get("maintenance", 0.0) * clip01(maintenance)
    )
    return clip01(dv)


# --- dynamic part ----------------------------------------------------------
def effective_threshold(dv: float, cfg: EngineConfig) -> float:
    """T_eff = T0 (1 - beta DV). Higher DV never raises the threshold."""
    return cfg.t0 * (1.0 - cfg.beta * clip01(dv))


def effective_rain(intensity: float, antecedent: float, cfg: EngineConfig) -> float:
    kappa = float(cfg.section("rainfall")["kappa_per_h"])
    return max(intensity, 0.0) + kappa * max(antecedent, 0.0)


def hazard(r_eff: float, t_eff: float) -> float:
    return clip01(r_eff / t_eff) if t_eff > 0 else 1.0


def level_for(score: float, cfg: EngineConfig) -> str:
    cuts = cfg.level_cuts
    if score >= cuts["very_high"]:
        return "very_high"
    if score >= cuts["high"]:
        return "high"
    if score >= cuts["moderate"]:
        return "moderate"
    return "low"


def level_at_least(level: str, minimum: str) -> bool:
    return LEVEL_RANK[level] >= LEVEL_RANK[minimum]


# --- confidence ------------------------------------------------------------
def confidence(seg: SegmentStatic, rain: RainWindow | None, dv: float, t_eff: float,
               ctx: EvalContext, cfg: EngineConfig) -> tuple[float, list[Explanation]]:
    c = cfg.section("confidence")
    value = 1.0
    reasons: list[Explanation] = []

    if rain is None:
        value -= c["no_rain_data"]
        reasons.append(Explanation("conf_no_rain", {}, "no rainfall data available"))
    else:
        if rain.interpolated:
            value -= c["interpolated_rain"]
            reasons.append(Explanation("conf_interpolated", {"interval_min": rain.interval_minutes},
                                       "rainfall is hourly model data, not observations"))
        if len(rain.intensity_mm_h) > 1:
            r_models = [effective_rain(i, a, cfg)
                        for i, a in zip(rain.intensity_mm_h, rain.antecedent_mm)]
            spread = (max(r_models) - min(r_models)) / max(t_eff, 1e-6)
            pen = c["model_spread_max"] * clip01(spread)
            if pen >= 0.02:
                value -= pen
                reasons.append(Explanation(
                    "conf_model_spread",
                    {"min_mm_h": round(min(r_models), 1), "max_mm_h": round(max(r_models), 1)},
                    f"weather models disagree ({min(r_models):.1f} to {max(r_models):.1f} mm/h)"))
        refresh = ctx.refresh_interval_minutes
        if rain.age_minutes > 3 * refresh:
            value -= c["very_stale"]
            reasons.append(Explanation("conf_very_stale", {"age_min": round(rain.age_minutes)},
                                       f"rainfall data are {rain.age_minutes:.0f} min old"))
        elif rain.age_minutes > refresh:
            value -= c["stale"]
            reasons.append(Explanation("conf_stale", {"age_min": round(rain.age_minutes)},
                                       f"rainfall data are {rain.age_minutes:.0f} min old"))
        if rain.dry_spell_days is None:
            value -= c["dry_spell_unknown"]

    if seg.poorly_mapped:
        value -= c["poorly_mapped"]
        reasons.append(Explanation("conf_poorly_mapped", {}, "road poorly mapped in OpenStreetMap"))
    if seg.dv_maintenance is None:
        pen = c["unknown_drain_state_max"] * clip01(dv)
        if pen >= 0.02:
            value -= pen
            reasons.append(Explanation("conf_drain_unknown", {},
                                       "drain condition unknown (no authority data)"))
    if seg.past_flood_confirmed:
        value += c["past_flood_bonus"]
    if ctx.recent_feedback in ("consistent_flooded", "consistent_dry"):
        value += c["feedback_bonus"]

    return clip01(value), reasons


def confidence_label(value: float, cfg: EngineConfig) -> str:
    labels = cfg.section("confidence")["labels"]
    if value >= labels["high"]:
        return "high"
    if value >= labels["medium"]:
        return "medium"
    return "low"


# --- main entry point ------------------------------------------------------
def evaluate_segment(seg: SegmentStatic, rain: RainWindow | None, ctx: EvalContext,
                     cfg: EngineConfig) -> SegmentRisk:
    """Score one segment for one ETA window. Pure function."""
    s, contrib = susceptibility(seg, cfg)
    s_prime = floored_susceptibility(s, cfg)
    dry_days = rain.dry_spell_days if rain is not None else None
    dv = drainage_vulnerability(seg, dry_days, cfg)
    t_eff = effective_threshold(dv, cfg)

    explanations: list[Explanation] = []
    if rain is None:
        h = float(cfg.section("rainfall")["hazard_when_unknown"])
        r_eff = None
        mode = "static_only"
        explanations.append(Explanation(
            "static_only", {}, "no rainfall data: showing terrain susceptibility only"))
    else:
        # ensemble mean of models for the hazard; spread goes to confidence
        r_models = [effective_rain(i, a, cfg)
                    for i, a in zip(rain.intensity_mm_h, rain.antecedent_mm)]
        r_eff = fmean(r_models)
        h = hazard(r_eff, t_eff)
        mode = "full"
        explanations.append(Explanation(
            "rain_vs_threshold",
            {"rain_mm_h": round(r_eff, 1), "threshold_mm_h": round(t_eff, 1)},
            f"forecast {r_eff:.1f} mm/h against an effective threshold of {t_eff:.1f} mm/h"))

    score = 100.0 * h * s_prime
    level = level_for(score, cfg)

    # two largest static contributions
    for name, value in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:2]:
        if value > 0.02:
            explanations.append(Explanation(
                f"factor_{name}", {"contribution": round(value, 3)}, FACTOR_LABELS[name]))

    reduction = 1.0 - t_eff / cfg.t0
    if reduction >= 0.05:
        explanations.append(Explanation(
            "drainage_threshold", {"reduction_pct": round(100 * reduction)},
            f"drainage vulnerability lowers the threshold by {100 * reduction:.0f}%"))

    conf, conf_reasons = confidence(seg, rain, dv, t_eff, ctx, cfg)
    explanations.extend(conf_reasons)

    return SegmentRisk(
        segment_id=seg.segment_id,
        score=score,
        level=level,
        confidence=conf,
        confidence_label=confidence_label(conf, cfg),
        hazard=None if rain is None else h,
        susceptibility=s,
        drainage_vulnerability=dv,
        threshold_mm_h=t_eff,
        effective_rain_mm_h=r_eff,
        mode=mode,
        explanations=explanations,
        contributions=contrib,
    )
