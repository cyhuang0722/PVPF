from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT_PREFIXES = (
    "/home/chuangbn/projects/PVPF",
    "/Users/huangchouyue/Projects/PVPF",
)


def project_root() -> Path:
    env_root = os.environ.get("PVPF_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, must_exist: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        relative = candidate
        for base in (Path.cwd(), project_root()):
            based = base / relative
            if based.exists():
                candidate = based
                break
        else:
            candidate = project_root() / relative
    text = str(candidate)
    root = project_root()
    for prefix in PROJECT_ROOT_PREFIXES:
        if text.startswith(prefix):
            candidate = root / text[len(prefix) :].lstrip("/")
            break
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path not found: {path} (resolved to {candidate})")
    return candidate


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, str):
        for prefix in PROJECT_ROOT_PREFIXES:
            if value.startswith(prefix):
                return str(resolve_project_path(value, must_exist=False))
    return value


def normalize_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    return _normalize_config_value(config)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_run_dir(root: str | Path, prefix: str = "run") -> Path:
    root = ensure_dir(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
