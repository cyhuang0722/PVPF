# 街景叠加辐亮度图
import os

import contextily as cx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import osmnx as ox
from pyproj import Transformer

from sat_vis import (
    _read_geo_params,
    lonlat_to_rowcol,
    rowcol_to_lonlat,
    read_roi_calibrated,
)

# 辐亮度：低值灰 → 中值淡蓝 → 高值深蓝（与固定 vmin/vmax 搭配便于跨图对比）
RADIANCE_CMAP = LinearSegmentedColormap.from_list(
    "radiance_gray_blue",
    [
        (0.80, 0.80, 0.80),
        (0.58, 0.78, 0.95),
        (0.02, 0.22, 0.48),
    ],
    N=256,
)


def fetch_buildings(lat, lon, radius_m=1000):
    tags = {"building": True}
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
    except AttributeError:
        gdf = ox.geometries_from_point((lat, lon), tags=tags, dist=radius_m)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf.to_crs(epsg=3857)


def compute_roi_radiance(file_path, lat, lon, half_km=1.0, ch=2):
    import h5py

    with h5py.File(file_path, "r") as f:
        geo = _read_geo_params(f)
        H, W = f[f"Data/NOMChannel{ch:02d}"].shape

    row_c, col_c = lonlat_to_rowcol(lon, lat, H, W, geo)
    roi, (r0, r1, c0, c1), _ = read_roi_calibrated(
        file_path, ch, row_c, col_c, half_km=half_km, geo=geo
    )

    lon_ul, lat_ul = rowcol_to_lonlat(r0, c0, H, W, geo)
    lon_lr, lat_lr = rowcol_to_lonlat(r1, c1, H, W, geo)
    extent_ll = [lon_ul, lon_lr, lat_lr, lat_ul]  # [xmin, xmax, ymin, ymax]
    return roi, extent_ll


def main():
    # HKUST 大致坐标；若需更准可自行替换
    lat = 22.3364
    lon = 114.2633
    radius_m = 600  # 建筑抓取半径，同时匹配辐亮度 1km
    half_km = 1.0
    ch = 2

    # file_path = os.environ.get(
    #     "FY4_FILE",
    #     # "../down/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20251214031500_20251214032959_2000M_V0001.HDF",
    #     "/Users/huangchouyue/Projects/PVPF/data/satellite/500m/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20260123160000_20260123161459_0500M_V0001.HDF",
    #     # "/Users/huangchouyue/Projects/PVPF/data/satellite/500m/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20260123161500_20260123162959_0500M_V0001.HDF",
    #     # "/Users/huangchouyue/Projects/PVPF/data/satellite/500m/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20260123163000_20260123164459_0500M_V0001.HDF",
    # )

    

    import re
    from pathlib import Path

    hdf_dir = Path("/Users/huangchouyue/Projects/PVPF/data/satellite/500m/")
    hdf_files = sorted(hdf_dir.glob("*.HDF"))
    if not hdf_files:
        raise FileNotFoundError("No FY4 HDF files found in {}".format(hdf_dir))
    print(hdf_files)
    # 设置colorbar的显示范围
    CLIM_MIN = 0.0
    CLIM_MAX = 1.4



    for file_path in hdf_files:

        # 从文件名中提取时间戳（如 20260123160000_20260123161459）
        m = re.search(r"(\d{14}_\d{14})", file_path.name)
        if m:
            time_stamp = m.group(1)
        else:
            time_stamp = None
            print(f"Warning: Could not extract time_stamp from {file_path.name}")


        # 1) OSM 建筑
        gdf_3857 = fetch_buildings(lat, lon, radius_m=radius_m)

        # 2) FY4 辐亮度 ROI
        roi, extent_ll = compute_roi_radiance(file_path, lat, lon, half_km=half_km, ch=ch)

        # 3) 经纬度 -> Web Mercator
        tf = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        xmin, ymin = tf.transform(extent_ll[0], extent_ll[2])
        xmax, ymax = tf.transform(extent_ll[1], extent_ll[3])
        extent_merc = [xmin, xmax, ymin, ymax]

        # 中心点投影
        cxm, cym = tf.transform(lon, lat)

        # 4) 画图
        fig, ax = plt.subplots(figsize=(8, 7), dpi=220)
        if not gdf_3857.empty:
            gdf_3857.boundary.plot(ax=ax, linewidth=0.8, color="black", zorder=3)

        im = ax.imshow(
            roi,
            extent=extent_merc,
            origin="upper",
            cmap=RADIANCE_CMAP,
            alpha=0.6,
            zorder=2,
            vmin=CLIM_MIN,
            vmax=CLIM_MAX,
        )
        ax.plot(cxm, cym, marker="x", color="cyan", markersize=10, linewidth=2, zorder=4)

        # 视野覆盖建筑与 ROI
        bounds = gdf_3857.total_bounds if not gdf_3857.empty else [xmin, ymin, xmax, ymax]
        ax.set_xlim(min(bounds[0], xmin), max(bounds[2], xmax))
        ax.set_ylim(min(bounds[1], ymin), max(bounds[3], ymax))

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Radiance (channel {:02d})".format(ch))

        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"hkust_radiance_overlay_{time_stamp}.png", dpi=320, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()

