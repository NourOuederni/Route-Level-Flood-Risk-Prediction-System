# Flood Route Risk — Version 1 (pilot)

Route-level flood risk for drivers in one Tunisian pilot area (default: **Nabeul**).
Version 1 is **purely algorithmic**: a transparent weighted multi-criteria engine, no machine learning.
Built from the Requirements Specification v1.0 and the *Version 1 Implementation Guide* (21 Oct 2026).

```
flood-route-risk/
├── backend/            Python 3.11+: engine, routing, API, data pipeline, validation
│   ├── app/engine/     the risk engine (pure functions) + AHP weighting tool
│   ├── app/routing/    precomputed area loader + risk-aware route planner
│   ├── app/weather/    Open-Meteo grid cache, synthetic scenarios, replay files
│   ├── app/storage/    prediction log, feedback, events (SQLite default, PostGIS optional)
│   ├── app/main.py     FastAPI endpoints
│   ├── config/         engine.yaml (all weights & thresholds), AHP expert template
│   ├── pipeline/       OSM + Copernicus DEM + WorldCover → per-segment factors
│   ├── evaluation/     replay validation with baselines (guide Section 8)
│   ├── sql/            PostGIS schema
│   └── tests/          engine properties + API end-to-end
└── mobile/             Expo (SDK 58) React Native app, TypeScript
```

Not prepared for deployment on purpose: no production server config, no HTTPS, no store builds.

---

## 1. The engine (what "algorithmic" means here)

Everything lives in `backend/app/engine/core.py` and `backend/config/engine.yaml`:

```
Score   = 100 · H · S'                    0–100, never a water depth
S'      = s_min + (1 − s_min) · S         floor: any road can flood in an extreme storm
S       = Σ wᵢ fᵢ                          6 static factors, percentile-normalised to [0,1]
R_eff   = I_W + κ · A_6h                   max hourly rain in the ETA window + antecedent rain
DV      = Σ vⱼ proxyⱼ                      urban density, black spots, dry spell, drain density
T_eff   = T0 · (1 − β · DV)                clogged drains lower the threshold (FR-09)
H       = min(1, R_eff / T_eff)
```

- **Levels**: Low < 25 ≤ Moderate < 50 ≤ High < 75 ≤ Very high.
- **Confidence** starts at 1 and loses points for interpolated hourly rain, model disagreement, stale data, poorly mapped roads and unknown drain state. It gains points for confirmed past floods and consistent user reports.
- **Explanations**: every score carries coded reasons (rain vs threshold, top two terrain factors, the drainage reduction, confidence reasons). The app translates them.
- **No rain data at all** gives degraded mode: terrain susceptibility only, capped below High, with low confidence (FR-15).
- **Rain is the trigger, terrain the modulator**: no rain means a score of 0, whatever the terrain.

All numbers are placeholders for calibration. Change them in `engine.yaml` (Git history = FR-24 change log) and bump `version`.

**Expert weights (AHP):** each expert fills a copy of `config/ahp_expert_template.yaml`, then:
```bash
python -m app.engine.ahp experts/*.yaml --mc 2000   # weights, consistency ratio, Monte-Carlo stability
```

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# A) offline demo, no downloads (synthetic town, same file format as real data)
python -m pipeline.make_demo_area
FLOOD_AREA=demo WEATHER_MODE=scenario:storm uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# B) real Nabeul data (needs internet: OSM, AWS public buckets, Open-Meteo)
python -m pipeline.run --area nabeul                      # ~10–20 min the first time
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  # live Open-Meteo by default
```

API docs: http://localhost:8000/docs · Tests: `pytest` (build the demo area first).

| Env var | Default | Meaning |
|---|---|---|
| `FLOOD_AREA` | `nabeul` | folder in `data/processed/` |
| `WEATHER_MODE` | `live` | `live`, `scenario:dry/showers/storm`, `replay:<file>`, `outage` |
| `DATABASE_URL` | SQLite in `data/` | e.g. `postgresql+psycopg://user:pw@localhost/flood` |
| `ADMIN_TOKEN` | `dev-admin` | needed to switch weather modes from the app |

**Endpoints:** `POST /routes` (up to 3 routes + safer alternative or fallback), `GET /hotspots`, `GET /segments/{id}`, `POST /feedback`, `GET /geocode`, `GET /meta`, `POST /admin/weather`.

### Pipeline steps (`python -m pipeline.run --area nabeul [--from STEP | --only STEP]`)
1. **osm**: drivable graph (OSMnx) with speeds and travel times, waterways, storm drains.
2. **rasters**: Copernicus GLO-30 DEM and ESA WorldCover 2021, read as windows from the public AWS buckets.
3. **terrain**: reprojects to UTM 32N. Computes depth in sink (before filling), flow accumulation, streams, slope, TWI, HAND and distance to drainage.
4. **features**: cuts roads into segments of 200 m or less, samples rasters in buffers, then derives impervious share, underpass/ford/low-crossing flags, past-flood proximity and normalisation.

Look at `data/interim/<area>/*.tif` and `data/processed/<area>/segments_qa.gpkg` in **QGIS** before trusting them.

### Events, authority data, validation
```bash
# historical floods / black spots (FR-23); --test-set keeps them OUT of f6/DV (no circularity)
python -m pipeline.import_events --area nabeul events.csv
python -m pipeline.import_events --area nabeul emsr319.geojson --source "Copernicus EMS EMSR319" \
       --date 2018-09-22T18:00:00Z --reliability high --test-set
# authority drainage data (FR-10): data/raw/<area>/drainage/inlets.geojson and
# maintenance.geojson (with a last_cleaned date), then:
python -m pipeline.run --area nabeul --only features

# the rain that was available at the time, then replay validation with baselines
python -m pipeline.fetch_event_rain --area nabeul --start 2024-01-10 --end 2024-01-12            # forecast archive
python -m pipeline.fetch_event_rain --area nabeul --start 2018-09-21 --end 2018-09-23 --source era5
python -m evaluation.replay_eval --area nabeul --rain openmeteo_forecast_2024-01-10.json
```
The report gives recall at High (with 95% CI), false-alarm share, median lead time and a precision-recall table. It covers the full model and both baselines (static-only, rain-only). Report forecast-archive and ERA5 events **separately**.

## 3. Run the mobile app (Expo)

```bash
cd mobile
npm install
npx expo start            # scan the QR code with Expo Go (Android / iOS)
```
- The app finds the API automatically on the same machine as Metro (`http://<your-LAN-IP>:8000`). Start uvicorn with `--host 0.0.0.0`. To override, use `EXPO_PUBLIC_API_URL=http://192.168.1.20:8000 npx expo start` or the app's Settings.
- Maps: Google Maps on Android, Apple Maps on iOS, both through `react-native-maps` (works in Expo Go). A standalone Android build will need a Google Maps API key. That is part of deployment, so it is not set up here.
- **Using it:** search a destination or long-press the map. Pick a departure time. Routes appear coloured by risk, each with an icon and a label as well as colour (NFR-09). Then:
  - An alert card offers the **safer route** (+time, +distance), or says no safer route exists.
  - Tap any route segment for its reasons and the age of the data, and to send feedback (flooded / not flooded / drain blocked).
  - **GO** starts trip mode. It re-checks every 5 minutes and whenever you leave the route.
- **Demo (guide 2.3):** Settings → *Demo weather* → `storm`, then `outage` to "unplug" the source (confidence drops, the app says why), then `dry`.
- Languages: Français / English / العربية. Arabic switches to right-to-left after the restart prompt.
- Trip traces are never sent or stored (NFR-07). The server logs only segment scores.

## 4. Known limits (be honest about them in the report)
- Rain is hourly model data at about 10 km. 15-minute values are interpolated, and small storm cells can be missed.
- A 30 m surface DEM cannot see kerbs or inlets. Terrain factors flag broad low zones and wadi corridors only.
- Drain condition is not observable. DV is a proxy until authority data arrive.
- Few labelled events exist: treat validation numbers as indicative, never as proven performance.
- The free Open-Meteo tier is **non-commercial**. See `LICENCES.md`.
