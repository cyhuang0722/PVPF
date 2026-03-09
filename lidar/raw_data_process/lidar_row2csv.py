#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from locale import D_FMT
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


# Default input (can be overridden by CLI)
INPUT_FILE = "HJYJY2026021912.2017"



def _read_n_lines_crlf(fp, n: int) -> List[bytes]:
    """Read exactly n lines separated by CRLF (\r\n).

    Returns list of raw line bytes WITHOUT trailing CRLF.
    Raises EOFError if not enough lines.
    """
    lines: List[bytes] = []
    for _ in range(n):
        line = fp.readline()
        if not line:
            raise EOFError(f"Unexpected EOF while reading header line {len(lines)+1}/{n}")
        # Strip one trailing \r\n or \n
        if line.endswith(b"\r\n"):
            line = line[:-2]
        elif line.endswith(b"\n"):
            line = line[:-1]
        lines.append(line)
    return lines


def _decode_line(b: bytes) -> str:
    # Try common encodings; header is usually ASCII/UTF-8.
    for enc in ("utf-8", "gbk", "gb18030", "latin1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin1", errors="replace")


def _split_tokens(s: str) -> List[str]:
    # Split by whitespace, keep simple.
    return [t for t in s.strip().split() if t]


def parse_header_1ch(line1: str, line2: str, line3: str, line4: str) -> Dict[str, Any]:
    """Parse 4-line header (1 channel) using the XLS as the source of truth.

    NOTE: This is a best-effort parser for printing/inspection.
    If some tokenization differs on your device, we will adjust once you show a sample header.
    """
    h: Dict[str, Any] = {
        "raw": {"line1": line1, "line2": line2, "line3": line3, "line4": line4},
        "line1": {},
        "line2": {},
        "line3": {},
        "channel": {},
    }

    # ---- Line 1 ----
    # XLS: filename, code
    t1 = _split_tokens(line1)
    if len(t1) >= 1:
        h["line1"]["filename"] = t1[0]
    if len(t1) >= 2:
        h["line1"]["code"] = t1[1]
    if len(t1) > 2:
        h["line1"]["extra"] = t1[2:]

    # ---- Line 2 ----
    # XLS keys (order): startDate startTime endDate endTime setAlt gpsLon gpsLat field1 field2 gpsAlt field3 field4 height field5 field6 windSpeed windDirection timeBatch rainFlag humi temp press angle
    t2 = _split_tokens(line2)
    keys2 = [
        "siteName",
        "startDate",
        "startTime",
        "endDate",
        "endTime",
        "setAlt",
        "gpsLon",
        "gpsLat",
        "field1",
        "field2",
        "gpsAlt",
        "StartElevation",
        "StopElevation",
        "height",
        "field5",
        "field6",
        "windSpeed",
        "windDirection",
        "timeBatch",
        "snow",
        "humi",
        "temp",
        "press",
        'tilt',
        'pan',
        'startPan',
        'stopPan',
        'intervalAngle',
        'compassX',
        'compassY',
        'compassZ',
        'compassXA',
        'compassYA',
        'compassZA',
        'compassXH',
        'compassYH',
        'compassZH',
        'mode',
        'scan_path_file',
        'con_rotate_direction',
        'lidar_id'
    ]
    for i, k in enumerate(keys2):
        if i < len(t2):
            h["line2"][k] = t2[i]

    # ---- Line 3 ----
    # XLS keys: accumulate, laserFrequency, laserField1, laserField2, channelCount
    t3 = _split_tokens(line3)
    keys3 = ["accumulate", "laserFrequency", "laserField1", "laserField2", "channelCount"]
    for i, k in enumerate(keys3):
        if i < len(t3):
            h["line3"][k] = t3[i]

    # ---- Line 4 (channel info) ----
    # XLS keys (order): enable signalType laserType dataLength field1 pmt resolution name field2 field3 field4 field5 adc pulse rangeOrThreshold bt dtype
    t4 = _split_tokens(line4)
    keys4 = [
        "enable",
        "signalType",
        "laserType",
        "dataLength",
        "field1",
        "pmt",
        "resolution",
        "name",
        "field2",
        "field3",
        "field4",
        "field5",
        "adc",
        "pulse",
        "rangeOrThreshold",
        "bt",
        "dtype",
    ]
    for i, k in enumerate(keys4):
        if i < len(t4):
            h["channel"][k] = t4[i]

    # Convenience numeric conversions
    try:
        if "channelCount" in h["line3"]:
            h["line3"]["channelCount_int"] = int(float(h["line3"]["channelCount"]))
    except Exception:
        pass

    try:
        if "dataLength" in h["channel"]:
            h["channel"]["dataLength_int"] = int(float(h["channel"]["dataLength"]))
    except Exception:
        pass

    try:
        if "resolution" in h["channel"]:
            h["channel"]["resolution_float"] = float(h["channel"]["resolution"])
    except Exception:
        pass

    return h



def read_header_and_offset(path: Path) -> tuple[list[str], Dict[str, Any], int]:
    """Read 4 header lines and return (decoded_lines, parsed_header_dict, data_offset_bytes)."""
    with path.open("rb") as fp:
        raw_lines = _read_n_lines_crlf(fp, 4)
        offset = fp.tell()  # byte offset after the 4th line ending
        lines = [_decode_line(b) for b in raw_lines]

    h = parse_header_1ch(lines[0], lines[1], lines[2], lines[3])
    return lines, h, offset


def read_data_block(
    path: Path,
    offset: int,
    data_len: int | None,
    endian: str = "little",
) -> np.ndarray:
    """Read int32 data block after header.

    Your expectation: data block should have TWO columns.

    Our interpretation (to be confirmed by you):
    - `data_len` (from header line4 "dataLength") is the number of range bins / samples *per column*.
    - Total int32 values = 2 * data_len.

    If `data_len` is missing or the file size does not match exactly, we will fall back to:
    - Read all remaining int32 values and reshape to (-1, 2) if possible.

    Returns a numpy array of shape (N, 2) with dtype int32.
    """
    if endian.lower().startswith("l"):
        dt = np.dtype("<i4")
    elif endian.lower().startswith("b"):
        dt = np.dtype(">i4")
    else:
        raise ValueError("endian must be 'little' or 'big'.")

    file_size = path.stat().st_size
    if offset >= file_size:
        raise ValueError(f"Data offset {offset} is beyond file size {file_size}.")

    with path.open("rb") as fp:
        fp.seek(offset)
        raw = fp.read()

    # Some files include CR/LF separators before/after the binary block.
    # Example from your error: remaining bytes = 16006 while 4000*4 = 16000,
    # suggesting 3 extra CRLF (6 bytes). We strip them and continue.
    raw0_len = len(raw)
    # strip leading
    while raw and raw[:1] in (b"\r", b"\n"):
        raw = raw[1:]
    # strip trailing
    while raw and raw[-1:] in (b"\r", b"\n"):
        raw = raw[:-1]

    if len(raw) != raw0_len:
        print(f"[WARN] Stripped {raw0_len - len(raw)} CR/LF bytes around data block (raw={raw0_len} -> {len(raw)} bytes).")

    data_bytes = len(raw)
    if data_bytes % 4 != 0:
        raise ValueError(
            f"After stripping CR/LF, data bytes {data_bytes} is still not multiple of 4. "
            f"This suggests separators embedded inside the binary block."
        )

    total_ints = data_bytes // 4
    arr = np.frombuffer(raw, dtype=dt)

    # Prefer header-provided length if available
    if data_len is not None:
        n = int(data_len)

        # Case A: exactly N int32 -> single-column signal (range will be computed from resolution)
        if len(arr) == n:
            return arr.reshape((n, 1))

        # Case B: at least 2N int32 -> two int32 columns stored
        if len(arr) >= 2 * n:
            expected = 2 * n
            arr_use = arr[:expected]
            if expected != len(arr):
                print(f"[WARN] Header expects {expected} int32 values (2*dataLength), but file has {len(arr)}. Using first {expected}.")
            return arr_use.reshape((n, 2))

        # Case C: shorter than N -> can't trust header, fall through to fallback
        print(f"[WARN] Header dataLength={n} but only {len(arr)} int32 values available. Falling back.")

    # Fallback: reshape everything into 2 columns
    if total_ints % 2 != 0:
        raise ValueError(f"Total int32 count {total_ints} is not even, cannot reshape to 2 columns.")

    return arr.reshape((-1, 2))


def main():
    ap = argparse.ArgumentParser(description="Simple test: read header + data block (2 columns) from ZKHG lidar raw file")
    ap.add_argument("raw_file", nargs="?", default=INPUT_FILE, help="Path to raw file")
    ap.add_argument("--endian", default="little", choices=["little", "big"], help="int32 endianness")
    args = ap.parse_args()

    path = Path(args.raw_file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines, h, offset = read_header_and_offset(path)

    print("=== RAW HEADER (4 lines) ===")
    for i, ln in enumerate(lines, 1):
        print(f"[{i}] {ln}")

    print("\n=== PARSED (XLS-based) ===")
    print("[Line1]", h["line1"])
    print("[Line2]")
    # Print all parsed fields in line2
    for k, v in h["line2"].items():
        print(f"  - {k}: {v}")
    print("[Line3]", h["line3"])
    print("[Channel]", h["channel"])

    data_len = h.get("channel", {}).get("dataLength_int")
    print(f"\n[INFO] data offset bytes: {offset}")
    print(f"[INFO] dataLength (from header): {data_len}")

    data = read_data_block(path, offset=offset, data_len=data_len, endian=args.endian)

    print("\n=== DATA BLOCK (interpreted) ===")
    print(f"shape={data.shape}, dtype={data.dtype}")

    resolution = h.get("channel", {}).get("resolution_float")
    if resolution is None:
        print("[WARN] Cannot parse resolution from header; range column cannot be computed.")

    n_show = min(10, data.shape[0])

    # Case: single-column int32 signal; compute range + scaled float signal to match vendor tool style
    if data.shape[1] == 1:
        sig_i32 = data[:, 0].astype(np.int64)
        if resolution is not None:
            rng = (np.arange(len(sig_i32), dtype=np.float64) * float(resolution))
        else:
            rng = np.arange(len(sig_i32), dtype=np.float64)

        # Heuristic scaling: many vendor tools output e.g. 19.188286 from raw 19188 by /1000.
        sig_f = sig_i32.astype(np.float64) / 1000.0

        print("first rows (range\tsignal_float\traw_int32):")
        for i in range(n_show):
            print(f"  {rng[i]:.6f}\t{sig_f[i]:.6f}\t{int(sig_i32[i])}")

        # Save to CSV: Distance (km), mV
        dist_km = rng / 1000.0
        mv = sig_f / 1000.0  # raw_int32 / 1e6

        out_csv = path.with_suffix(path.suffix + ".csv")
        # out_csv ="test.csv"

        data_out = np.column_stack([dist_km, mv])
        np.savetxt(
            out_csv,
            data_out,
            delimiter=",",
            header="Distance_km,Signal_mV",
            comments="",
            fmt="%.6f",
        )
        print(f"\n[OK] Saved CSV: {out_csv}")

        print("\n[STATS] signal_raw_int32: min/max/mean =", int(sig_i32.min()), int(sig_i32.max()), float(sig_i32.mean()))
        print("[STATS] signal_float(/1000): min/max/mean =", float(sig_f.min()), float(sig_f.max()), float(sig_f.mean()))

    # Case: two int32 columns stored (debug view)
    else:
        print(f"first {n_show} rows (int32 cols):")
        for i in range(n_show):
            print(f"  {i:04d}: {int(data[i,0])}\t{int(data[i,1])}")

        col0 = data[:, 0].astype(np.float64)
        col1 = data[:, 1].astype(np.float64)
        print("\n[STATS] col0: min/max/mean =", int(col0.min()), int(col0.max()), float(col0.mean()))
        print("[STATS] col1: min/max/mean =", int(col1.min()), int(col1.max()), float(col1.mean()))


if __name__ == "__main__":
    main()
