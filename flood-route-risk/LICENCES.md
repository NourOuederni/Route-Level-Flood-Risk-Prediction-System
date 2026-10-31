# Data and service licences (NFR-12)

Keep this table current from week 1. Verify terms before any non-academic use.

| Source | Used for | Licence | Obligation |
|---|---|---|---|
| OpenStreetMap | road graph, waterways, drains, basemap | ODbL 1.0 | Credit "© OpenStreetMap contributors"; share-alike on derived databases you publish |
| Copernicus DEM GLO-30 | elevation, slope, depressions, TWI, HAND | Copernicus DEM licence (free) | Required attribution text when data are adapted (in `pipeline/sources.py`) |
| ESA WorldCover 2021 v200 | built-up (impervious) share | CC BY 4.0 | Credit the ESA WorldCover project, cite the dataset |
| Open-Meteo API | rainfall forecast, recent past, archive | Data CC BY 4.0; free API **non-commercial** | "Weather data by Open-Meteo.com" with link; commercial use needs a plan or self-hosting |
| ERA5 / ERA5-Land (via Open-Meteo) | dry spell, back-tests | Copernicus licence | Credit Copernicus Climate Change Service |
| Copernicus EMS EMSR319 | validation labels (Nabeul, Sept 2018) | Copernicus EMS open data | Credit the activation |
| Nominatim (geocoding, dev only) | destination search | ODbL data; usage policy | Max 1 req/s, identify the app, cache results. Use another geocoder for anything beyond development |
| FABDEM (optional, not used by default) | comparison DEM | CC BY-NC-SA 4.0 | Non-commercial, share-alike |
| Authority data (ONAS, municipality, ...) | inlets, cleaning dates, black spots | to be negotiated | As agreed in writing |

Software dependencies are open source (MIT / BSD / Apache-2.0 / LGPL). `pysheds` is GPL-3.0 and is used only in the offline pipeline, not in the API.
