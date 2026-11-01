"""Study areas. Bounding boxes are WGS84 degrees; `crs` is the metric UTM zone used for geometry."""

from pathlib import Path

RAW = Path("data/raw")

NYC = {
    "name": "nyc_se_queens",
    "south": 40.66, "north": 40.74, "west": -73.84, "east": -73.74,
    "crs": "EPSG:32618",
    "dem": "usgs3dep",
    # Airport weather stations with measured hourly rain (lat, lon).
    "stations": {"JFK": (40.6398, -73.7789), "LGA": (40.7769, -73.8740)},
    # NYC buildings and addresses were imported into OSM in 2014: treat mapping as complete.
    "osm_completeness": 1.0,
}

# Copernicus EMS rapid mapping of the 22 September 2018 Cap Bon floods (EMSR319).
EMSR319_PRODUCTS = [
    "01TAKLISAH_01DELINEATION_MAP_v2",
    "02MANZILBUZALAFAH_01DELINEATION_MAP_v2",
    "03KORBA_01DELINEATION_MAP_v1",
    "04BOUARGOUB_01DELINEATION_MAP_v1",
    "05SOMAA_01DELINEATION_MAP_v2",
]
EMSR319_URL = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR319/EMSR319_{}_vector.zip"
NABEUL_CRS = "EPSG:32632"
# OSM completeness in Cap Bon is unknown: neutral value, see features.drainage_vulnerability.
NABEUL_OSM_COMPLETENESS = 0.5


def raw_dir(*parts) -> Path:
    path = RAW.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
