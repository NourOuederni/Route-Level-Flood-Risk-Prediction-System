"""Engine properties required by guide Section 6.4, plus the worked example of Section 7.6."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.engine import EvalContext, RainWindow, SegmentStatic, default_config, evaluate_segment
from app.engine.ahp import aggregate, matrix_from_pairs, weights_and_cr
from app.engine.config import ConfigError, validate
from app.engine.core import effective_threshold

CFG = default_config()
CTX = EvalContext(refresh_interval_minutes=180)
NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)

unit = st.floats(min_value=0, max_value=1, allow_nan=False)
rain_mm = st.floats(min_value=0, max_value=200, allow_nan=False)


def seg(**kw) -> SegmentStatic:
    return SegmentStatic(segment_id=1, **kw)


def rain(i: float, a: float = 0.0, age: float = 10.0, dry: float | None = 10.0,
         models: int = 1) -> RainWindow:
    return RainWindow(
        intensity_mm_h=(i,) * models, antecedent_mm=(a,) * models, source="test",
        retrieved_at=NOW, age_minutes=age, dry_spell_days=dry)


segments = st.builds(
    SegmentStatic, segment_id=st.just(1), f_depression=unit, f_drainage_position=unit,
    f_low_slope=unit, f_impervious=unit, f_crossing=unit, f_past_flood=unit,
    dv_urban=unit, dv_blackspot=unit, dv_drain_density_inv=unit,
    poorly_mapped=st.booleans())


@given(segments)
def test_no_rain_gives_low_whatever_the_terrain(s):
    r = evaluate_segment(s, rain(0.0, 0.0), CTX, CFG)
    assert r.score == 0.0
    assert r.level == "low"


@given(segments, rain_mm, rain_mm, rain_mm)
def test_more_rain_never_lowers_the_score(s, i, a, extra):
    lo = evaluate_segment(s, rain(i, a), CTX, CFG).score
    hi_i = evaluate_segment(s, rain(i + extra, a), CTX, CFG).score
    hi_a = evaluate_segment(s, rain(i, a + extra), CTX, CFG).score
    assert hi_i >= lo - 1e-9
    assert hi_a >= lo - 1e-9


@given(unit, unit)
def test_higher_drainage_vulnerability_never_raises_threshold(dv1, dv2):
    lo, hi = sorted((dv1, dv2))
    assert effective_threshold(hi, CFG) <= effective_threshold(lo, CFG) + 1e-12


@given(segments, rain_mm, st.floats(min_value=0, max_value=5000), st.floats(min_value=0, max_value=5000))
def test_stale_data_never_raises_confidence(s, i, age1, age2):
    young, old = sorted((age1, age2))
    c_young = evaluate_segment(s, rain(i, age=young), CTX, CFG).confidence
    c_old = evaluate_segment(s, rain(i, age=old), CTX, CFG).confidence
    assert c_old <= c_young + 1e-12


@given(segments)
def test_missing_rain_never_alerts_and_lowers_confidence(s):
    r = evaluate_segment(s, None, CTX, CFG)
    assert r.mode == "static_only"
    assert r.level in ("low", "moderate")
    assert r.confidence <= evaluate_segment(s, rain(5.0), CTX, CFG).confidence


@settings(max_examples=50)
@given(segments, rain_mm)
def test_score_bounds(s, i):
    r = evaluate_segment(s, rain(i, i), CTX, CFG)
    assert 0 <= r.score <= 100
    assert 0 <= r.confidence <= 1


def test_worked_example_section_7_6():
    """S = 0.7, DV = 0.6, T0 = 20, beta = 0.5, R_eff = 12 -> ~65 High; without drainage ~46."""
    raw = copy.deepcopy(CFG.raw)
    # make S exactly 0.7 and DV exactly 0.6 with a single factor/proxy each
    raw["susceptibility"]["weights"] = {k: 0.0 for k in raw["susceptibility"]["weights"]}
    raw["susceptibility"]["weights"]["depression"] = 1.0
    raw["drainage"]["weights"] = {k: 0.0 for k in raw["drainage"]["weights"]}
    raw["drainage"]["weights"]["urban"] = 1.0
    cfg = validate(raw)

    s = seg(f_depression=0.7, dv_urban=0.6)
    r = evaluate_segment(s, rain(12.0), CTX, cfg)
    assert r.threshold_mm_h == pytest.approx(14.0)
    assert r.score == pytest.approx(100 * (12 / 14) * 0.76, abs=0.01)
    assert r.level == "high"

    r0 = evaluate_segment(seg(f_depression=0.7, dv_urban=0.0), rain(12.0), CTX, cfg)
    assert r0.score == pytest.approx(45.6, abs=0.01)
    assert r0.level == "moderate"


def test_explanations_present_for_alert():
    s = seg(f_depression=1, f_drainage_position=1, f_past_flood=1, dv_urban=1, dv_blackspot=1)
    r = evaluate_segment(s, rain(25.0, 10.0), CTX, CFG)
    codes = [e.code for e in r.explanations]
    assert codes[0] == "rain_vs_threshold"
    assert "drainage_threshold" in codes
    assert any(c.startswith("factor_") for c in codes)


def test_model_spread_lowers_confidence():
    agree = RainWindow((10.0, 10.0, 10.0), (0, 0, 0), "t", NOW, 10, dry_spell_days=5)
    disagree = RainWindow((2.0, 10.0, 25.0), (0, 0, 0), "t", NOW, 10, dry_spell_days=5)
    s = seg(f_depression=0.5)
    assert evaluate_segment(s, disagree, CTX, CFG).confidence < evaluate_segment(s, agree, CTX, CFG).confidence


def test_bad_weights_rejected():
    raw = copy.deepcopy(CFG.raw)
    raw["susceptibility"]["weights"]["depression"] = 0.9
    with pytest.raises(ConfigError):
        validate(raw)


def test_ahp_consistent_matrix():
    factors = ["a", "b", "c"]
    m = matrix_from_pairs(factors, {"a>b": 2, "a>c": 4, "b>c": 2})
    w, cr = weights_and_cr(m)
    assert np.allclose(w, [4 / 7, 2 / 7, 1 / 7], atol=1e-6)
    assert cr < 1e-9
    agg = aggregate([m, matrix_from_pairs(factors, {"a>b": "1/2", "a>c": 1, "b>c": 2})])
    assert agg[0, 1] == pytest.approx(1.0)
