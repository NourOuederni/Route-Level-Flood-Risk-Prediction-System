"""Audit of the satellite flood labels: product registry, observed coverage and buffer sensitivity.

    python -m flood_risk.real.labels

For every mapped area of the inventory it records which Copernicus EMS product made the label
(type, version, sensor, resolution, image date and delay after the storm), and relabels the same
road segments with the flood polygons buffered by 0, 15 and 30 m. Each segment also gets
`observed`: 1 if it lies where the image was analysed, 0 if it falls in a "Not Analysed" area or
outside the image footprint. There, `flooded = 0` means "not seen", not "dry".
"""

import argparse
import io
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from . import ems, osm
from .areas import raw_dir

NEW_API = ems.NEW_API
BUFFERS_M = (0, 15, 30)


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (reads one zip member, not the whole zip)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=120)
        self.url = r.url
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300).content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _layer(zf: zipfile.ZipFile, *keys) -> gpd.GeoDataFrame | None:
    """Read the shapefile whose name contains every key (case and underscores ignored) from an open zip."""
    shp = [n for n in zf.namelist() if n.lower().endswith(".shp") and all(k in n.lower().replace("_", "") for k in keys)]
    if not shp:
        return None
    stem = shp[0][:-4]
    files = {ext: zf.read(stem + ext) for ext in (".shp", ".shx", ".dbf", ".prj") if stem + ext in zf.namelist()}
    tmp = raw_dir("ems", "_tmp")
    for ext, data in files.items():
        (tmp / f"layer{ext}").write_bytes(data)
    g = gpd.read_file(tmp / "layer.shp")
    return g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")


def _source_table(zf: zipfile.ZipFile) -> pd.DataFrame | None:
    dbf = [n for n in zf.namelist() if n.lower().endswith("source.dbf")]
    if not dbf:
        return None
    tmp = raw_dir("ems", "_tmp") / "source.dbf"
    tmp.write_bytes(zf.read(dbf[0]))
    return pd.DataFrame(gpd.read_file(tmp).drop(columns="geometry", errors="ignore"))


def products(code: str) -> list[dict]:
    """The product ems.mapped_areas used for each area, with its metadata and image footprint."""
    cache = raw_dir("ems", code)
    out = []
    js = requests.get(NEW_API.format(code), timeout=120).json()
    js = js["results"][0] if "results" in js else js
    if js.get("aois"):
        for a in js["aois"]:
            prods = [p for p in a["products"] if not p.get("monitoring") and ems._priority(p["type"] + "_PRODUCT") > 0]
            prods = sorted(prods, key=lambda p: (ems._priority(p["type"] + "_PRODUCT"), p["version"]["number"]), reverse=True)
            for p in prods:
                if any("observedEvent" in l["name"] and l.get("json") for l in p["layers"]):
                    times = [pd.Timestamp(i["acquisitionTime"]) for i in p["images"] if i.get("acquisitionTime")]
                    footprint, f = None, cache / f"{code}_AOI{a['number']:02d}_footprint.geojson"
                    if f.exists():
                        footprint = gpd.read_file(f)
                    elif p.get("downloadPath"):
                        try:
                            footprint = _layer(zipfile.ZipFile(io.BufferedReader(HttpFile(p["downloadPath"]), 1 << 20)), "imagefootprint")
                            if footprint is not None:
                                footprint.to_file(f, driver="GeoJSON")
                        except Exception as e:  # a missing zip only loses the coverage check
                            print(f"  {code} AOI{a['number']:02d}: footprint unavailable ({e})")
                    out.append({"aoi": f"{code}_AOI{a['number']:02d}", "product": f"{p['type']} v{p['version']['number']}",
                                "sensors": "; ".join(sorted({f"{i['sensorName']} ({i['resolutionClass']})" for i in p["images"]})),
                                "image_first": min(times) if times else pd.NaT, "image_last": max(times) if times else pd.NaT,
                                "delivered": pd.Timestamp(p["version"]["deliveryTime"]) if p["version"].get("deliveryTime") else pd.NaT,
                                "footprint": footprint})
                    break
        return out
    html = requests.get(ems.OLD_PAGE.format(code), timeout=120).text
    best = {}
    for z in sorted(set(re.findall(r'https://[^"]+_vector\.zip', html))):
        m = re.search(rf"{code}_(AOI\d+|\d+[A-Z]+)_", z)
        if m and ems._priority(z) > 0:
            key = m.group(1)
            if key not in best or (ems._priority(z), ems._version(z)) > (ems._priority(best[key]), ems._version(best[key])):
                best[key] = z
    for key, z in sorted(best.items()):
        f = cache / z.rsplit("/", 1)[-1]
        if not f.exists():
            continue
        zf = zipfile.ZipFile(f)
        src = _source_table(zf)
        post = src[src.eventphase.str.contains("Post", na=False)] if src is not None and "eventphase" in src else pd.DataFrame()
        # Dates are dd/mm/yyyy; times come as "10:00:30" or "T10:00:30Z" depending on the product generation.
        clock = post.get("source_tm", pd.Series(dtype=str)).astype(str).str.strip("TZ")
        times = pd.to_datetime(post.get("src_date", pd.Series(dtype=str)).astype(str) + " " + clock,
                               format="%d/%m/%Y %H:%M:%S", errors="coerce").dropna()
        out.append({"aoi": f"{code}_{key}", "product": z.rsplit("/", 1)[-1].replace(f"{code}_{key}_", "").replace("_vector.zip", ""),
                    "sensors": "; ".join(sorted(set(post.get("source_nam", pd.Series(dtype=str)).astype(str) + " ("
                                                    + post.get("sensor_gsd", pd.Series(dtype=str)).astype(str) + ")"))),
                    "image_first": times.min() if len(times) else pd.NaT, "image_last": times.max() if len(times) else pd.NaT,
                    "delivered": pd.NaT, "footprint": _layer(zf, "imagefootprint")})
    return out


def coverage(aoi_geom, footprint: gpd.GeoDataFrame | None):
    """Part of the area really analysed: image footprints, minus "Not Analysed" polygons."""
    if footprint is None or footprint.empty:
        return aoi_geom
    kind = footprint.get("obj_type", pd.Series("", index=footprint.index)).astype(str).str.lower()
    seen = footprint[~kind.str.contains("not analysed")]
    out = seen.geometry.union_all().intersection(aoi_geom) if len(seen) else aoi_geom
    blind = footprint[kind.str.contains("not analysed")]
    return out.difference(blind.geometry.union_all()) if len(blind) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inventory_med_imerg.csv.gz")
    ap.add_argument("--out", default="data/labels_med.csv.gz")
    ap.add_argument("--registry", default="reports/med_labels/product_registry.csv")
    args = ap.parse_args()
    inv = pd.read_csv(args.data, usecols=["segment_id", "event_id", "aoi", "flooded", "storm_total_mm"])
    times = ems.activation_times()
    parts, registry = [], []
    for code in inv.event_id.unique():
        prods = {p["aoi"]: p for p in products(code)}
        items = {f"{code}_{it['aoi']}": it for it in ems.mapped_areas(code, raw_dir("ems", code))}
        for tag in inv.loc[inv.event_id == code, "aoi"].unique():
            item, prod = items[tag], prods.get(tag, {})
            geom = item["geometry"]
            crs = ems.utm_crs(geom.centroid.x, geom.centroid.y)
            west, south, east, north = geom.bounds
            area = {"name": tag, "south": south, "north": north, "west": west, "east": east, "crs": crs}
            seg = osm.build_segments(area, raw_dir("ems", code), clip=geom)   # same segments as the inventory (cached OSM)
            flood = item["flood"].to_crs(crs).geometry
            lab = pd.DataFrame({"segment_id": seg["segment_id"], "aoi": tag, "event_id": code})
            for b in BUFFERS_M:
                lab[f"flooded_b{b}"] = seg.geometry.intersects(flood.buffer(b).union_all() if b else flood.union_all()).astype(int)
            seen = gpd.GeoSeries([coverage(geom, prod.get("footprint"))], crs="EPSG:4326").to_crs(crs).iloc[0]
            lab["observed"] = seg["mid"].within(seen).astype(int)
            parts.append(lab)
            first = prod.get("image_first", pd.NaT)
            registry.append({"event_id": code, "aoi": tag, "product": prod.get("product"), "sensors": prod.get("sensors"),
                             "image_first": first, "image_last": prod.get("image_last", pd.NaT),
                             "activation": times[code], "delivered": prod.get("delivered", pd.NaT),
                             "days_activation_to_image": (first.tz_localize(None) - times[code].tz_localize(None)).total_seconds() / 86400
                             if pd.notna(first) else np.nan,
                             "segments": len(lab), "observed_share": lab.observed.mean(),
                             **{f"flooded_b{b}": lab[f"flooded_b{b}"].mean() for b in BUFFERS_M}})
            print(f"{tag}: {len(lab)} segments, observed {lab.observed.mean():.1%}, flooded "
                  + " / ".join(f"{lab[f'flooded_b{b}'].mean():.2%}" for b in BUFFERS_M) + f" (buffers {BUFFERS_M} m)", flush=True)
    labels = pd.concat(parts, ignore_index=True).drop_duplicates(["segment_id", "aoi"])
    check = inv.merge(labels, on=["segment_id", "aoi"], how="left")
    print(f"\nsegments matched: {check.flooded_b15.notna().mean():.2%}; "
          f"same label as the inventory at 15 m: {(check.flooded == check.flooded_b15).mean():.3%}")
    labels.to_csv(args.out, index=False)
    from pathlib import Path
    Path(args.registry).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry).to_csv(args.registry, index=False)
    print(pd.DataFrame(registry).drop(columns=["sensors", "delivered"]).round(3).to_string())


if __name__ == "__main__":
    main()
