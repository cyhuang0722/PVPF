# 海岸线叠加辐亮度图
from math import pi
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) 通用小工具
# -----------------------------
def _scalar_attr(v, default=None):
    """把 HDF attrs 里的 bytes / shape=(1,) 转成 python 标量"""
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    arr = np.array(v)
    if arr.ndim == 0:
        return arr.item()
    return arr.reshape(-1)[0].item()


def _get_dataset(f, path):
    if path not in f:
        raise KeyError(f"Dataset not found: {path}")
    return f[path]


# -----------------------------
# 2) GEO 参数 & 投影
# -----------------------------
def _read_geo_params(f: h5py.File):
    """
    从 FY4 L1 文件读取几何参数，兼容 RegCenterLon 作为后备。
    返回:
        a, b           : 椭球长/短半轴 (m)
        lon0_deg       : 星下点经度 (deg)
        r_sat          : 卫星到地心距离 (m)
        dx, dy         : 采样/步进角 (rad)
    """
    a = _scalar_attr(f.attrs.get("Semimajor axis of ellipsoid"), 6378137.0)
    b = _scalar_attr(f.attrs.get("Semiminor axis of ellipsoid"), 6356752.0)
    lon0_deg = _scalar_attr(f.attrs.get("NOMCenterLon"), None)
    if lon0_deg is None:
        lon0_deg = _scalar_attr(f.attrs.get("RegCenterLon"), None)
    if lon0_deg is None:
        raise KeyError("Cannot find NOMCenterLon/RegCenterLon in file attrs.")
    sat_h = _scalar_attr(f.attrs.get("NOMSatHeight"), None)
    if sat_h is None:
        raise KeyError("Cannot find NOMSatHeight in file attrs.")
    dx_urad = _scalar_attr(f.attrs.get("dSamplingAngle"), None)
    dy_urad = _scalar_attr(f.attrs.get("dSteppingAngle"), None)
    if dx_urad is None or dy_urad is None:
        raise KeyError("Cannot find dSamplingAngle/dSteppingAngle in file attrs.")

    r_sat = a + float(sat_h)
    dx = float(dx_urad) * 1e-6
    dy = float(dy_urad) * 1e-6

    return dict(a=float(a), b=float(b), lon0_deg=float(lon0_deg), r_sat=r_sat, dx=dx, dy=dy)


def lonlat_to_rowcol(lon_deg, lat_deg, H, W, geo):
    """
    经纬度 -> (row, col)，使用标准地球同步投影正解，东经为正。
    """
    a, b = geo["a"], geo["b"]
    lon0 = np.deg2rad(geo["lon0_deg"])
    r_sat = geo["r_sat"]
    dx, dy = geo["dx"], geo["dy"]

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    e2 = (a * a - b * b) / (a * a)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)

    lam = lon - lon0
    x = N * np.cos(lat) * np.cos(lam)
    y = N * np.cos(lat) * np.sin(lam)
    z = (b * b / (a * a)) * N * np.sin(lat)

    alpha = np.arctan2(y, (r_sat - x))  # 东为正
    beta = np.arctan2(z, np.sqrt((r_sat - x) ** 2 + y * y))

    col0 = (W - 1) / 2.0
    row0 = (H - 1) / 2.0

    col = col0 + alpha / dx
    row = row0 - beta / dy  # 行号向下为正
    return int(round(row)), int(round(col))


def rowcol_to_lonlat(row, col, H, W, geo):
    """
    (row, col) -> 经纬度，采用 NOAA/EUMETSAT/FY4 通用反演公式。
    """
    a, b = geo["a"], geo["b"]
    lon0 = np.deg2rad(geo["lon0_deg"])
    r_sat = geo["r_sat"]
    dx, dy = geo["dx"], geo["dy"]

    col0 = (W - 1) / 2.0
    row0 = (H - 1) / 2.0

    x = (col - col0) * dx  # E-W scan angle (rad), east positive
    y = (row0 - row) * dy  # N-S scan angle (rad), north positive

    cosx = np.cos(x)
    sinx = np.sin(x)
    cosy = np.cos(y)
    siny = np.sin(y)

    a2 = a * a
    b2 = b * b

    term = (r_sat * cosx * cosy) ** 2 - (cosy * cosy + (a / b) ** 2 * siny * siny) * (r_sat * r_sat - a2)
    if term < 0:
        return np.nan, np.nan

    sd = np.sqrt(term)
    sn = (r_sat * cosx * cosy - sd) / (cosy * cosy + (a / b) ** 2 * siny * siny)

    s1 = r_sat - sn * cosx * cosy
    s2 = sn * sinx * cosy
    s3 = sn * siny

    lon = lon0 + np.arctan2(s2, s1)
    lat = np.arctan((a2 / b2) * s3 / np.sqrt(s1 * s1 + s2 * s2))

    return float(np.rad2deg(lon)), float(np.rad2deg(lat))


# -----------------------------
# 3) 读取 ROI + 标定
# -----------------------------
def read_roi_calibrated(file_path, ch: int, row_center, col_center, half_km=20.0, geo=None):
    """
    读取 ROI 并查表标定。
    half_km: 方形半边长（公里），根据步进角换算像元。
    """
    with h5py.File(file_path, "r") as f:
        if geo is None:
            geo = _read_geo_params(f)
        data_path = f"Data/NOMChannel{ch:02d}"

        print('attr',  dict(f['Data/NOMChannel02'].attrs))
        cal_path = f"Calibration/CALChannel{ch:02d}"

        ds = _get_dataset(f, data_path)
        H, W = ds.shape

        # 以星下点 GSD 估算像元大小 (km)
        gsd_km = (geo["r_sat"] * (geo["dx"] + geo["dy"]) * 0.5) / 1000.0
        gsd_km = max(gsd_km, 1e-6)
        half_px = int(round(half_km / gsd_km))
        half_px = max(1, half_px)

        r0 = max(0, row_center - half_px)
        r1 = min(H, row_center + half_px)
        c0 = max(0, col_center - half_px)
        c1 = min(W, col_center + half_px)

        fill = _scalar_attr(ds.attrs.get("FillValue"), 65535)
        dn = ds[r0:r1, c0:c1].astype(np.int32, copy=False)

        if cal_path in f:
            lut = _get_dataset(f, cal_path)[...].astype(np.float32, copy=False)
            print('lut', lut.shape)
            out = np.full(dn.shape, np.nan, dtype=np.float32)
            m = (dn != fill) & (dn >= 0) & (dn < lut.shape[0])
            print('m', m.shape)
            out[m] = lut[dn[m]]
        else:
            out = dn.astype(np.float32, copy=False)
            out[out == fill] = np.nan

        # 辐亮度转换
        if ch <= 6:
            esun_path = f"Calibration/ESUN"
            esun = _get_dataset(f, esun_path)
            print('esun', esun.shape)
            out = out * esun[ch-1] / np.pi

        return out, (r0, r1, c0, c1), (H, W)


# -----------------------------
# 4) 绘图
# -----------------------------
def plot_roi(file_path, lon_deg, lat_deg, half_km=20.0, ch=2, title=None):
    with h5py.File(file_path, "r") as f:
        geo = _read_geo_params(f)
        H, W = _get_dataset(f, f"Data/NOMChannel{ch:02d}").shape

    row_c, col_c = lonlat_to_rowcol(lon_deg, lat_deg, H, W, geo)
    roi, (r0, r1, c0, c1), _ = read_roi_calibrated(file_path, ch, row_c, col_c, half_km=half_km, geo=geo)

    lon_ul, lat_ul = rowcol_to_lonlat(r0, c0, H, W, geo)
    lon_lr, lat_lr = rowcol_to_lonlat(r1, c1, H, W, geo)
    extent = [lon_ul, lon_lr, lat_lr, lat_ul]  # [xmin, xmax, ymin, ymax]
    print("extent", extent)

    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    plt.figure(figsize=(8, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dce9ff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f4f2", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), edgecolor="steelblue", facecolor="#d8e7ff", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="steelblue", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#444444", linewidth=1.0, linestyle="--", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=1.2, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.7, color="gray", linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=5)
    gl.ylocator = mticker.MaxNLocator(nbins=5)

    im = ax.imshow(roi, origin="upper", extent=extent, transform=ccrs.PlateCarree(), zorder=1)
    ax.plot([lon_deg], [lat_deg], marker="x", markersize=10, markeredgewidth=2, color="red", transform=ccrs.PlateCarree(), zorder=6)

    cbar = plt.colorbar(im, shrink=0.85, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Radiance / K")

    if title is None:
        title = f"FY-4 AGRI L1 ROI | Ch{ch:02d} | center=({lat_deg:.4f}N,{lon_deg:.4f}E) | half={half_km} km"
    plt.title(title)
    plt.tight_layout()
    plt.savefig("roi_hkust_16_30.png")
    plt.close()


# -----------------------------
# 5) 直接跑
# -----------------------------
if __name__ == "__main__":
    file_path = os.environ.get(
        "FY4_FILE",
        # "../data/satellite/down/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20260211174500_20260211175959_2000M_V0001.HDF",
        "/Users/huangchouyue/Projects/PVPF/data/satellite/500m/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20260123163000_20260123164459_0500M_V0001.HDF"
    )
    # 学校位置：22°20'N, 114°15'E
    lat = 22 + 20 / 60
    lon = 114 + 15 / 60
    plot_roi(file_path, lon_deg=lon, lat_deg=lat, half_km=20.0, ch=2)

