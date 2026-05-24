"""
Data-pull manifest for reproducibility (source URLs + pull dates).

One row per pull: source, city, station, year, pull_date_utc, sha256, n_rows, notes.
Lets us cite the exact snapshot date and verify file integrity later, since OpenAQ
(and occasionally AURN) update retroactively.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

from aq_fm_bench.paths import MANIFEST

FIELDS = ["source", "city", "station", "year", "pull_date_utc", "sha256", "n_rows", "notes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(rows: list[dict]) -> None:
    """Append rows to the manifest CSV, writing a header if the file is new."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
