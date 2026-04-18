#!/usr/bin/env python3
# usage:
# python downsample_camera_images.py \
#   --src camera_data/raw \
#   --dst camera_data/resized_64 \
#   --max-size 64 \
#   --recursive \
#   --dry-run

# python downsample_camera_images.py \
#   --src camera_data/raw \
#   --dst camera_data/resized_256 \
#   --max-size 256 \
#   --recursive \
#   --dry-run


from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageOps

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path, *, replace_if_empty_file: bool = False) -> None:
    """
    确保 path 是可写目录。
    若 path 是同名空文件且允许替换，则删除该空文件并创建目录。
    """
    if path.exists():
        if path.is_dir():
            return
        if replace_if_empty_file and path.is_file() and path.stat().st_size == 0:
            path.unlink()
            path.mkdir(parents=True, exist_ok=True)
            return
        raise ValueError(f"输出路径已存在但不是文件夹: {path}")

    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(root: Path, recursive: bool = True):
    files = root.rglob("*") if recursive else root.iterdir()
    for path in files:
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def resize_keep_aspect(img: Image.Image, max_size: int) -> Image.Image:
    """
    保持长宽比，把长边缩放到 max_size。
    若原图长边已经 <= max_size，则原样返回副本。
    """
    w, h = img.size
    long_side = max(w, h)

    if long_side <= max_size:
        return img.copy()

    scale = max_size / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def build_dst_path(src_file: Path, src_root: Path, dst_root: Path, out_ext: str | None) -> Path:
    rel = src_file.relative_to(src_root)

    if out_ext is None:
        return dst_root / rel

    new_name = rel.stem + out_ext
    return (dst_root / rel).with_name(new_name)


def ensure_rgb_for_jpeg(img: Image.Image) -> Image.Image:
    """
    JPEG 不支持 alpha，必要时转成 RGB。
    """
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def save_image(
    img: Image.Image,
    dst_path: Path,
    *,
    jpeg_quality: int,
    png_compress_level: int,
) -> None:
    ensure_dir(dst_path.parent)

    suffix = dst_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        img = ensure_rgb_for_jpeg(img)
        img.save(
            dst_path,
            quality=jpeg_quality,
            optimize=True,
        )
    elif suffix == ".png":
        img.save(
            dst_path,
            compress_level=png_compress_level,
            optimize=True,
        )
    else:
        img.save(dst_path)


def process_one(
    src_file: Path,
    src_root: Path,
    dst_root: Path,
    *,
    max_size: int,
    out_ext: str | None,
    overwrite: bool,
    dry_run: bool,
    jpeg_quality: int,
    png_compress_level: int,
) -> str:
    dst_path = build_dst_path(src_file, src_root, dst_root, out_ext)
    # 防御式创建输出目录，避免某些写入路径分支下目录缺失
    ensure_dir(dst_path.parent)

    if dst_path.exists() and not overwrite:
        return f"[SKIP EXISTS] {dst_path}"

    if dry_run:
        return f"[DRY-RUN] {src_file} -> {dst_path}"

    with Image.open(src_file) as img:
        img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋转
        resized = resize_keep_aspect(img, max_size=max_size)
        save_image(
            resized,
            dst_path,
            jpeg_quality=jpeg_quality,
            png_compress_level=png_compress_level,
        )

    return f"[OK] {src_file} -> {dst_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample camera images while preserving folder structure.")
    parser.add_argument("--src", type=Path, required=True, help="原始图片根目录，如 camera_data/raw")
    parser.add_argument("--dst", type=Path, required=True, help="输出目录，如 camera_data/resized_512")
    parser.add_argument("--max-size", type=int, default=512, help="输出图像长边最大尺寸，默认 512")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印操作，不真正写文件")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    parser.add_argument(
        "--out-ext",
        type=str,
        default=None,
        choices=[None, ".jpg", ".jpeg", ".png"],
        help="可选：统一输出格式，例如 .jpg；默认保持原后缀",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG 质量，默认 90")
    parser.add_argument("--png-compress-level", type=int, default=4, help="PNG 压缩级别，默认 4")
    args = parser.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        raise ValueError(f"源目录不存在或不是文件夹: {args.src}")

    # 预先创建输出根目录，避免首次写文件时目录不存在
    # 兼容历史残留的同名空文件（会自动替换为目录）
    ensure_dir(args.dst, replace_if_empty_file=True)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for src_file in iter_image_files(args.src, recursive=args.recursive):
        total += 1
        try:
            msg = process_one(
                src_file,
                args.src,
                args.dst,
                max_size=args.max_size,
                out_ext=args.out_ext,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                jpeg_quality=args.jpeg_quality,
                png_compress_level=args.png_compress_level,
            )
            print(msg)
            if msg.startswith("[OK]") or msg.startswith("[DRY-RUN]"):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {src_file}: {e}")

    print("\n===== SUMMARY =====")
    print(f"Scanned : {total}")
    print(f"OK      : {ok}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    main()