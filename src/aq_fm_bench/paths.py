"""
Canonical project paths, derived from this file's location.

Lesson learned in setup: the shell tools default to a subfolder and the working
directory does not persist reliably, so NEVER rely on the CWD. Always import paths
from here:

    from aq_fm_bench.paths import PROCESSED, RAW_OPENAQ, ensure_dirs
"""
from __future__ import annotations

from pathlib import Path

# src/aq_fm_bench/paths.py -> parents[2] == project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data layers (raw -> validated_raw -> processed)
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
RAW_OPENAQ = RAW / "openaq"
RAW_AURN = RAW / "aurn"
RAW_ERA5 = RAW / "era5"
VALIDATED_RAW = DATA / "validated_raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MANIFEST = DATA / "manifest.csv"

# Code-adjacent
CONFIGS = PROJECT_ROOT / "configs"
CONFIGS_CITIES = CONFIGS / "cities"
CONFIGS_MODELS = CONFIGS / "models"
CONFIGS_CONFORMAL = CONFIGS / "conformal"

# Outputs
RESULTS = PROJECT_ROOT / "results"
RESULTS_LEADERBOARD = RESULTS / "leaderboard"
RESULTS_COVERAGE = RESULTS / "coverage"
RESULTS_REGIME = RESULTS / "regime_shift"
RESULTS_EPISODE = RESULTS / "episode_analysis"
RESULTS_TRANSFER = RESULTS / "transfer"

FIGURES = PROJECT_ROOT / "figures"
FIGURES_EDA = FIGURES / "eda"
FIGURES_RESULTS = FIGURES / "results"
FIGURES_PAPER = FIGURES / "paper"

LOGS = PROJECT_ROOT / "logs"

_ALL_DIRS = [
    RAW_OPENAQ, RAW_AURN, RAW_ERA5, VALIDATED_RAW, PROCESSED, SPLITS,
    RESULTS_LEADERBOARD, RESULTS_COVERAGE, RESULTS_REGIME, RESULTS_EPISODE, RESULTS_TRANSFER,
    FIGURES_EDA, FIGURES_RESULTS, FIGURES_PAPER, LOGS,
]


def ensure_dirs() -> None:
    """Create all output/data directories if missing (idempotent)."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    ensure_dirs()
    print("All data/result/figure directories ensured.")
