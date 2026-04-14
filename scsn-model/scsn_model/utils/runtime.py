from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache(root: str | Path) -> Path:
    cache_dir = Path(root) / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    return cache_dir

