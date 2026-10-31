"""End-to-end API tests on the synthetic demo area (build it first: python -m pipeline.make_demo_area)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "processed" / "demo" / "area.json"
pytestmark = pytest.mark.skipif(not DEMO.is_file(), reason="demo area not built")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["FLOOD_AREA"] = "demo"
    os.environ["WEATHER_MODE"] = "scenario:storm"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def _set(client, mode):
    r = client.post("/admin/weather", json={"mode": mode}, headers={"x-admin-token": "dev-admin"})
    assert r.status_code == 200, r.text
    return r.json()


def _trip(client):
    meta = client.get("/meta").json()
    w, s, e, n = meta["area"]["bbox"]
    # west-south-west to east-north-east: crosses the wadi valley
    return {"origin": {"lat": s + 0.15 * (n - s), "lon": w + 0.1 * (e - w)},
            "destination": {"lat": s + 0.85 * (n - s), "lon": w + 0.9 * (e - w)}}


def test_meta_and_health(client):
    m = client.get("/meta").json()
    assert m["area"]["synthetic"] is True
    assert "scenario:storm" in m["weather_modes"]
    assert client.get("/health").json()["segments"] > 100


def test_dry_scenario_all_low(client):
    _set(client, "scenario:dry")
    r = client.post("/routes", json=_trip(client) | {"alert_level": "high"}).json()
    assert 1 <= len(r["routes"]) <= 4
    for route in r["routes"]:
        assert route["max_level"] == "low"
        assert route["alert"]["count"] == 0
    assert r["fallback"] is None and r["safer_route_id"] is None
    assert r["compute_ms"] < 5000                                    # NFR-01


def test_storm_alerts_and_alternative(client):
    _set(client, "scenario:storm")
    r = client.post("/routes", json=_trip(client) | {"alert_level": "moderate"}).json()
    fastest = r["routes"][0]
    assert fastest["kind"] in ("fastest", "safer")
    assert fastest["max_level"] != "low"
    seg = fastest["segments"][0]
    for k in ("segment_id", "eta", "score", "level", "confidence", "confidence_label", "explanations", "coords"):
        assert k in seg
    assert seg["explanations"][0]["code"] == "rain_vs_threshold"
    # ETAs increase along the route (FR-04)
    etas = [s["eta"] for s in fastest["segments"]]
    assert etas == sorted(etas)
    # either a safer route or an explicit fallback warning (FR-18/FR-19)
    if fastest["alert"]["count"]:
        assert (r["safer_route_id"] is not None) ^ (r["fallback"] is not None)
    if r["safer_route_id"]:
        safer = next(x for x in r["routes"] if x["id"] == r["safer_route_id"])
        assert safer["alert"]["count"] < fastest["alert"]["count"] or safer["max_level"] != fastest["max_level"]
        assert "extra_duration_s" in safer


def test_outage_lowers_confidence(client):
    _set(client, "scenario:showers")
    trip = _trip(client)
    before = client.post("/routes", json=trip).json()["routes"][0]["min_confidence"]
    _set(client, "outage")
    after = client.post("/routes", json=trip).json()
    assert after["weather"]["simulated_outage"] is True
    assert after["routes"][0]["min_confidence"] < before
    assert any(e["code"] == "conf_very_stale" for e in after["routes"][0]["segments"][0]["explanations"])


def test_live_without_network_degrades(client, monkeypatch):
    from app.main import S
    from app.weather import open_meteo

    def boom(*a, **k):
        raise ConnectionError("network down")
    monkeypatch.setattr(open_meteo, "fetch", boom)
    _set(client, "live")
    r = client.post("/routes", json=_trip(client)).json()
    assert r["weather"]["available"] is False
    assert all(s["mode"] == "static_only" for s in r["routes"][0]["segments"])
    assert r["routes"][0]["max_level"] in ("low", "moderate")
    S.weather.set_mode("scenario:storm")


def test_outside_area_rejected(client):
    r = client.post("/routes", json={"origin": {"lat": 0, "lon": 0}, "destination": {"lat": 0.1, "lon": 0.1}})
    assert r.status_code == 422


def test_hotspots_and_feedback(client):
    _set(client, "scenario:storm")
    hs = client.get("/hotspots", params={"min_level": "moderate"}).json()["items"]
    assert hs and all(h["level"] in ("moderate", "high", "very_high") for h in hs)
    h = hs[0]
    fb = client.post("/feedback", json={"kind": "flooded", "lat": h["lat"], "lon": h["lon"]}).json()
    assert fb["segment_id"] == h["segment_id"]
    d = client.get(f"/segments/{h['segment_id']}").json()
    assert d["level"] == h["level"]


def test_admin_requires_token(client):
    r = client.post("/admin/weather", json={"mode": "scenario:dry"})
    assert r.status_code == 403
