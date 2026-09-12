"""
One-off authoring tool for this project's map dashboard -- NOT part of the
tested pipeline and not run by CI. Needs `pip install shapely`.

Fetches Afghanistan's 34 province boundaries directly from the same GEE
dataset the extraction pipeline already uses (FAO/GAUL/2015/level1),
simplifies, projects (equirectangular, latitude-cosine x-compression), and
writes map_paths.json: ready-to-embed SVG <path> d-strings plus centroid
coordinates for the dashboard's map.
"""
import json
import math
import os

from dotenv import load_dotenv
load_dotenv()

import ee
from shapely.geometry import shape, MultiPolygon

GEE_SA = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
ee.Initialize(ee.ServiceAccountCredentials(GEE_SA, key_data=GEE_KEY))
print("GEE initialized OK")

gaul1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
provinces_fc = gaul1.filter(ee.Filter.eq("ADM0_NAME", "Afghanistan"))
geojson = provinces_fc.getInfo()
print(f"fetched {len(geojson['features'])} provinces from GEE")

REF_LAT = 34.0  # Afghanistan's approximate central latitude
COS_REF = math.cos(math.radians(REF_LAT))


def project(lon, lat):
    return (lon * COS_REF, -lat)


def geom_to_path(geom_geojson, tol):
    g = shape(geom_geojson).simplify(tol, preserve_topology=True)
    polys = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    d_parts = []
    for poly in polys:
        if poly.is_empty:
            continue
        for ring in [poly.exterior] + list(poly.interiors):
            coords = list(ring.coords)
            if len(coords) < 3:
                continue
            pts = [project(lon, lat) for lon, lat in coords]
            d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
            d_parts.append(d)
    return " ".join(d_parts)


province_paths = {}
province_centroids = {}
for feat in geojson["features"]:
    name = feat["properties"]["ADM1_NAME"]
    geom = feat["geometry"]
    province_paths[name] = geom_to_path(geom, tol=0.01)
    c = shape(geom).centroid
    province_centroids[name] = [round(v, 2) for v in project(c.x, c.y)]

out = {
    "province_paths": province_paths,
    "province_centroids": province_centroids,
    "ref_lat_cos": COS_REF,
}
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/map_paths.json", "w") as f:
    json.dump(out, f)

for name, p in list(province_paths.items())[:5]:
    print(name, len(p), "chars")
print("...")
total_chars = sum(len(p) for p in province_paths.values())
print(f"total path data: {total_chars} chars across {len(province_paths)} provinces")
