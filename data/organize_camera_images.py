#!/usr/bin/env python3
# usage: python organize_camera_images.py --src skyimg --dst camera_data/raw  --dry-run 
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
import re

# 匹配类似：
# 192.168.10.2_01_20260116184832903_TIMING.jpg
# 提取中间 17 位时间戳：20260116184832903
TIMESTAMP_PATTERN = re.compile(r"_(\d{17})_")

# 常见图片后缀
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_datetime_from_name(filename: str) -> datetime | None:
    """
    从文件名中提取 17 位时间戳，并解析为 datetime。
    格式: YYYYMMDDHHMMSSmmm
    其中最后 3 位是毫秒。
    """
    match = TIMESTAMP_PATTERN.search(filename)
    if not match:
        return None

    ts17 = match.group(1)  # e.g. 20260116184832903
    ts14 = ts17[:14]       # 20260116184832
    ms3 = ts17[14:]        # 903

    try:
        dt = datetime.strptime(ts14, "%Y%m%d%H%M%S")
        # datetime 只支持微秒，所以毫秒 * 1000
        dt = dt.replace(microsecond=int(ms3) * 1000)
        return dt
    except ValueError:
        return None


def build_target_path(dst_root: Path, dt: datetime, original_name: str) -> Path:
    """
    目标结构:
    dst_root/
        2026/
            2026-01/
                2026-01-16/
                    original_name
    """
    year_dir = dst_root / f"{dt:%Y}"
    month_dir = year_dir / f"{dt:%Y-%m}"
    day_dir = month_dir / f"{dt:%Y-%m-%d}"
    return day_dir / original_name


def safe_transfer_file(
    src: Path,
    dst: Path,
    *,
    dry_run: bool,
    copy_mode: bool,
) -> str:
    """
    安全移动/复制文件。
    如果目标已存在：
    - 若同名文件不存在冲突，则正常处理
    - 若存在，则自动追加 _dup1, _dup2... 避免覆盖
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    final_dst = dst
    if final_dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_dup{counter}{suffix}"
            if not candidate.exists():
                final_dst = candidate
                break
            counter += 1

    if dry_run:
        action = "COPY" if copy_mode else "MOVE"
        return f"[DRY-RUN {action}] {src} -> {final_dst}"

    if copy_mode:
        shutil.copy2(src, final_dst)
        return f"[COPIED] {src} -> {final_dst}"
    else:
        shutil.move(str(src), str(final_dst))
        return f"[MOVED] {src} -> {final_dst}"


def organize(
    src_dir: Path,
    dst_root: Path,
    *,
    recursive: bool,
    dry_run: bool,
    copy_mode: bool,
) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"源目录不存在或不是文件夹: {src_dir}")

    files = src_dir.rglob("*") if recursive else src_dir.iterdir()

    total = 0
    processed = 0
    skipped_nonfile = 0
    skipped_suffix = 0
    skipped_no_timestamp = 0
    failed = 0

    for path in files:
        total += 1

        if not path.is_file():
            skipped_nonfile += 1
            continue

        if path.suffix.lower() not in VALID_SUFFIXES:
            skipped_suffix += 1
            continue

        dt = extract_datetime_from_name(path.name)
        if dt is None:
            skipped_no_timestamp += 1
            print(f"[SKIP NO TIMESTAMP] {path}")
            continue

        dst = build_target_path(dst_root, dt, path.name)

        try:
            msg = safe_transfer_file(
                path,
                dst,
                dry_run=dry_run,
                copy_mode=copy_mode,
            )
            print(msg)
            processed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {path}: {e}", file=sys.stderr)

    print("\n===== SUMMARY =====")
    print(f"Scanned entries      : {total}")
    print(f"Processed images     : {processed}")
    print(f"Skipped non-files    : {skipped_nonfile}")
    print(f"Skipped bad suffix   : {skipped_suffix}")
    print(f"Skipped no timestamp : {skipped_no_timestamp}")
    print(f"Failed               : {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize camera images by date from filename timestamps."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="源文件夹（当前所有图片堆在这里）",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="目标根目录",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归扫描子目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不真的移动/复制",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制而不是移动（更安全，适合第一次整理）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize(
        src_dir=args.src,
        dst_root=args.dst,
        recursive=args.recursive,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )


if __name__ == "__main__":
    main()