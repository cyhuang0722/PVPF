# Draw a street-map overlay for the camera field of view.
import os
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import osmnx as ox
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform


def make_metric_circle(lon, lat, radius_m):
    """Create an accurate meter-radius circle around lon/lat and return EPSG:3857 geometry."""
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    to_lonlat = Transformer.from_crs(local_crs, "epsg:4326", always_xy=True).transform
    to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform

    circle_ll = shapely_transform(to_lonlat, Point(0, 0).buffer(radius_m, resolution=256))
    return shapely_transform(to_merc, circle_ll)


def fetch_osm_layers(lat, lon, radius_m):
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path(__file__).resolve().parent / "cache")

    graph = ox.graph_from_point(
        (lat, lon),
        dist=radius_m,
        network_type="all",
        simplify=True,
    )
    _, roads = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    roads = roads.to_crs(epsg=3857)

    tags = {"building": True}
    try:
        buildings = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
    except AttributeError:
        buildings = ox.geometries_from_point((lat, lon), tags=tags, dist=radius_m)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(epsg=3857)
    return roads, buildings


def main():
    # HKUST camera location. Replace these if you have a more precise camera GPS.
    lat = float(os.environ.get("CAMERA_LAT", 22.3325417961502))
    lon = float(os.environ.get("CAMERA_LON", 114.26698453139196))
    fov_radius_m = float(os.environ.get("FOV_RADIUS_M", 2000))
    map_margin_m = float(os.environ.get("MAP_MARGIN_M", 350))
    output_path = Path(os.environ.get("OUTPUT", "draw/hkust_camera_fov_street_2km.png"))

    fetch_radius_m = fov_radius_m + map_margin_m
    roads_3857, buildings_3857 = fetch_osm_layers(lat, lon, fetch_radius_m)

    to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    cxm, cym = to_merc.transform(lon, lat)
    fov_circle = make_metric_circle(lon, lat, fov_radius_m)

    xmin, ymin, xmax, ymax = fov_circle.bounds
    xmin -= map_margin_m
    ymin -= map_margin_m
    xmax += map_margin_m
    ymax += map_margin_m

    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=220)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    try:
        cx.add_basemap(
            ax,
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom=14,
            attribution=False,
        )
    except Exception:
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.Positron,
            zoom=14,
            attribution=False,
        )

    if not buildings_3857.empty:
        buildings_3857.plot(
            ax=ax,
            facecolor="#f2efe7",
            edgecolor="#5c5751",
            linewidth=0.25,
            alpha=0.7,
            zorder=2,
        )

    if not roads_3857.empty:
        roads_3857.plot(ax=ax, color="#ffffff", linewidth=1.1, alpha=0.9, zorder=3)
        roads_3857.plot(ax=ax, color="#4c5564", linewidth=0.35, alpha=0.75, zorder=4)

    ax.fill(*fov_circle.exterior.xy, color="#ff6b35", alpha=0.13, zorder=5)
    ax.plot(*fov_circle.exterior.xy, color="#ff4d1f", linewidth=2.6, zorder=6)
    ax.scatter([cxm], [cym], marker="o", s=60, color="#00a6ff", edgecolor="white", linewidth=1.4, zorder=7)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#00a6ff",
                   markeredgecolor="white", markersize=16, label="Camera"),
            Patch(facecolor="#ff6b35", edgecolor="#ff4d1f", alpha=0.25, label="FOV radius: 2 km"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=18,
    )
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
