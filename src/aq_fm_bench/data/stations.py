"""
Station registry — the single source of truth for which stations are in the benchmark.

All stations are DEFRA AURN, reference-grade, with >=75% completeness over 2019-2024.

`pollutants` lists only the pollutants that PASSED the completeness check for that station
(e.g. MAHG is NO2-only: its PM2.5 was 6% complete).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    code: str            # source-specific id (AURN code, EEA/EPA id, ...)
    city: str
    tier: int            # 1 (UK deep) or 2 (global)
    env_type: str        # urban_background | urban_traffic | suburban_industrial
    source: str          # aurn | eea | epa
    pollutants: tuple    # subset of ("NO2", "PM2.5") that passed completeness
    koppen: str
    note: str = ""


TIER1: list[Station] = [
    Station("SHDG", "sheffield",  1, "urban_background",   "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("SHBR", "sheffield",  1, "urban_traffic",      "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("KC1",  "london",     1, "urban_background",   "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("MY1",  "london",     1, "urban_traffic",      "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("BMLD", "birmingham", 1, "urban_background",   "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("BIRR", "birmingham", 1, "urban_traffic",      "aurn", ("NO2", "PM2.5"), "Cfb",
            "A4540 — the Birmingham CAZ ring road (regime-shift site)"),
    Station("MAN3", "manchester", 1, "urban_background",   "aurn", ("NO2", "PM2.5"), "Cfb"),
    Station("MAHG", "manchester", 1, "suburban_industrial","aurn", ("NO2",),         "Cfb",
            "PM2.5 only 6% complete -> NO2-only station"),
]

# Additional station groups (none in this release).
TIER2: list[Station] = []

ALL_STATIONS = TIER1 + TIER2

# pollutant column names as they appear in the AURN parquet files
AURN_POLLUTANT_COL = {"NO2": "NO2", "PM2.5": "PM2.5"}
