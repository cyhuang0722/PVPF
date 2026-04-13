from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WEATHER_CSV = ROOT / "data" / "weather.csv"
RUN_ROOTS = [
    ROOT / "new-model" / "artifacts" / "runs",
    ROOT / "SPM",
    ROOT / "pv_forecasting" / "ConvLSTM-encoder" / "model_output",
]
SPLITS = ("train", "val", "test")
SKY_LABELS = {
    1: "clear_sky",
    2: "cloudy_visible_sun",
    3: "overcast",
    4: "rain",
}


@dataclass
class MetricAccumulator:
    count: int = 0
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    target_sum: float = 0.0
    target_sq_sum: float = 0.0

    def add(self, target: float, pred: float) -> None:
        error = pred - target
        self.count += 1
        self.abs_error_sum += abs(error)
        self.sq_error_sum += error * error
        self.target_sum += target
        self.target_sq_sum += target * target

    def to_metrics(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"n_samples": 0, "mae": None, "rmse": None, "r2": None}

        mae = self.abs_error_sum / self.count
        rmse = math.sqrt(self.sq_error_sum / self.count)

        if self.count < 2:
            r2 = None
        else:
            mean = self.target_sum / self.count
            ss_tot = self.target_sq_sum - (self.count * mean * mean)
            if math.isclose(ss_tot, 0.0, abs_tol=1e-12):
                r2 = None
            else:
                r2 = 1.0 - (self.sq_error_sum / ss_tot)

        return {
            "n_samples": self.count,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }


def detect_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        delimiter = max([",", "\t", ";"], key=first_line.count)
        return csv.excel_tab if delimiter == "\t" else csv.excel


def load_weather_map(path: Path) -> dict[str, int]:
    dialect = detect_dialect(path)
    weather_by_day: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            date_key = row["Date"].strip()
            weather_by_day[date_key] = int(row["weather"])
    return weather_by_day


def format_day_key(ts_value: str) -> str:
    dt = datetime.fromisoformat(ts_value)
    return f"{dt.day}-{dt.strftime('%b')}"


def find_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []

    run_dirs: list[Path] = []
    for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((candidate / f"predictions_{split}.csv").exists() for split in SPLITS):
            run_dirs.append(candidate)
    return run_dirs


def select_metric_columns(fieldnames: list[str]) -> tuple[str, str]:
    if "target_pv_w" in fieldnames and "pred_w" in fieldnames:
        return "target_pv_w", "pred_w"
    if "target_value" in fieldnames and "pred_value" in fieldnames:
        return "target_value", "pred_value"
    raise ValueError(f"Unable to find target/prediction columns in {fieldnames}")


def compute_split_metrics(
    csv_path: Path,
    weather_by_day: dict[str, int],
) -> tuple[list[dict[str, float | int | str | None]], dict[str, object]]:
    dialect = detect_dialect(csv_path)
    accumulators: dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    missing_weather_dates: dict[str, int] = defaultdict(int)
    total_rows = 0
    matched_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")
        target_col, pred_col = select_metric_columns(reader.fieldnames)

        for row in reader:
            total_rows += 1
            day_key = format_day_key(row["ts_target"])
            sky = weather_by_day.get(day_key)
            if sky is None:
                missing_weather_dates[day_key] += 1
                continue

            matched_rows += 1
            accumulators[sky].add(float(row[target_col]), float(row[pred_col]))

    records: list[dict[str, float | int | str | None]] = []
    for sky in sorted(accumulators):
        metrics = accumulators[sky].to_metrics()
        records.append(
            {
                "sky": sky,
                "sky_label": SKY_LABELS.get(sky, f"sky_{sky}"),
                **metrics,
            }
        )

    meta = {
        "csv_path": str(csv_path),
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "missing_weather_rows": total_rows - matched_rows,
        "missing_weather_dates": dict(sorted(missing_weather_dates.items())),
    }
    return records, meta


def write_outputs(run_dir: Path, rows: list[dict[str, object]], meta: dict[str, object]) -> None:
    csv_path = run_dir / "weather_split_metrics.csv"
    json_path = run_dir / "weather_split_metrics.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "sky", "sky_label", "n_samples", "mae", "rmse", "r2"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weather_csv": str(WEATHER_CSV),
        "sky_labels": SKY_LABELS,
        "splits": meta,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    weather_by_day = load_weather_map(WEATHER_CSV)
    processed = 0

    for run_root in RUN_ROOTS:
        for run_dir in find_run_dirs(run_root):
            output_rows: list[dict[str, object]] = []
            split_meta: dict[str, object] = {}

            for split in SPLITS:
                split_csv = run_dir / f"predictions_{split}.csv"
                rows, meta = compute_split_metrics(split_csv, weather_by_day)
                for row in rows:
                    output_rows.append({"split": split, **row})
                split_meta[split] = meta

            write_outputs(run_dir, output_rows, split_meta)
            processed += 1
            print(f"Wrote weather metrics to {run_dir}")

    print(f"Processed {processed} run directories.")


if __name__ == "__main__":
    main()
