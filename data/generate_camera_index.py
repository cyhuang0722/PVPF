#!/usr/bin/env python3
# usage:
# python generate_camera_index.py \
#   --src camera_data/raw \
#   --out camera_data/index/raw_index.csv

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime

# 匹配 17 位时间戳
TIMESTAMP_PATTERN = re.compile(r"_(\d{17})_")

VALID_SUFFIXES = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp"
}


def extract_datetime(filename: str):
    match = TIMESTAMP_PATTERN.search(filename)
    if not match:
        return None

    ts = match.group(1)

    ts14 = ts[:14]
    ms = ts[14:]

    try:
        dt = datetime.strptime(ts14, "%Y%m%d%H%M%S")
        dt = dt.replace(microsecond=int(ms) * 1000)
        return dt
    except:
        return None


def iter_images(root: Path):

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in VALID_SUFFIXES:
            continue

        yield p


def generate_index(root: Path, out_csv: Path):

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(out_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "file_path",
            "relative_path",
            "date",
            "time",
            "year",
            "month",
            "day",
            "filename"
        ])

        for img in iter_images(root):

            dt = extract_datetime(img.name)

            if dt is None:
                skipped += 1
                continue

            writer.writerow([
                dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                str(img.resolve()),
                str(img.relative_to(root)),
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S.%f")[:-3],
                dt.strftime("%Y"),
                dt.strftime("%m"),
                dt.strftime("%d"),
                img.name
            ])

            written += 1

            if written % 50000 == 0:
                print("indexed", written)

    print("\nindex finished")
    print("indexed:", written)
    print("skipped:", skipped)
    print("output:", out_csv)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="image root directory"
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output csv path"
    )

    args = parser.parse_args()

    generate_index(args.src, args.out)


if __name__ == "__main__":
    main()