from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path("/Users/huangchouyue/Projects/PVPF")
REMOTE_ROOT = "/home/chuangbn/projects/PVPF"


def local_path(value: str | Path) -> Path:
    text = str(value)
    return Path(text.replace(REMOTE_ROOT, str(PROJECT_ROOT)))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    value = str(name).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def timestamped_run_dir(root: str | Path) -> Path:
    out = Path(root) / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out
