import csv
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
from pvlib import solarposition


def main() -> None:
    csv_in = "/Users/huangchouyue/Projects/PVPF/data/data_sort/sample.csv"
    csv_out = "/Users/huangchouyue/Projects/PVPF/data/data_sort/sample_with_sun.csv"
    plot_out = "/Users/huangchouyue/Projects/PVPF/data/data_sort/el_vs_time.png"

    lat = 22.33257724541564
    lon = 114.26699324044685
    altitude_m = 0.0

    rows = []
    local_times = []

    with open(csv_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            local_time = datetime.fromisoformat(r["local_time"])
            utc_time = datetime.fromisoformat(r["utc_time"])
            if local_time.tzinfo is None:
                local_time = local_time.replace(tzinfo=timezone.utc)
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            rows.append((r["filename"], local_time, utc_time))
            local_times.append(local_time)

    times_utc = pd.DatetimeIndex([utc for _, _, utc in rows]).tz_convert("UTC")
    solpos = solarposition.get_solarposition(times_utc, lat, lon, altitude_m)
    az = solpos["azimuth"].to_numpy()
    el = solpos["elevation"].to_numpy()
    zenith = 90.0 - el

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "local_time", "az_deg", "el_deg", "zenith_deg"])
        for (filename, local_time, _), a, e, z in zip(rows, az, el, zenith):
            writer.writerow([filename, local_time.isoformat(), f"{a:.6f}", f"{e:.6f}", f"{z:.6f}"])
            print(local_time.isoformat(), f"{a:.6f}", f"{e:.6f}", f"{z:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(local_times, el, marker="o", markersize=2, linewidth=1)
    plt.xlabel("Local time")
    plt.ylabel("Elevation (deg)")
    plt.title("Sun elevation vs time")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()

    print(f"Saved CSV: {csv_out}")
    print(f"Saved plot: {plot_out}")


if __name__ == "__main__":
    main()

