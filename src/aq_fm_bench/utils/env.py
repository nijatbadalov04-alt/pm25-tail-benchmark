"""Minimal .env loader (no external dependency). Keeps the OpenAQ key out of code."""
from __future__ import annotations

import os
from pathlib import Path

from aq_fm_bench.paths import PROJECT_ROOT


def load_env(path: Path | None = None) -> None:
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_openaq_key() -> str:
    load_env()
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise RuntimeError("OPENAQ_API_KEY not set (environment or .env file)")
    return key
