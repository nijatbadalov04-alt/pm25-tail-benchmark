"""
Hourly-grid construction and the chronological train/calibration/test splits.

    train  2019-01-01 .. 2022-12-31 23:00   (~35k h)
    cal    2023-01-01 .. 2023-06-30 23:00   (~4.3k h)  conformal residuals only
    test   2023-07-01 .. 2024-12-31 23:00   (~13k h)   all reporting
"""
from __future__ import annotations

import pandas as pd

FULL_START = "2019-01-01"
FULL_END = "2025-01-01"  # exclusive

SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2019-01-01", "2023-01-01"),
    "cal": ("2023-01-01", "2023-07-01"),
    "test": ("2023-07-01", "2025-01-01"),
}


def hourly_grid(start: str = FULL_START, end: str = FULL_END) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")


def to_hourly_series(df: pd.DataFrame, col: str, *, time_col: str = "timestamp_utc",
                     start: str = FULL_START, end: str = FULL_END) -> pd.Series:
    """Reindex a long df onto the canonical hourly UTC grid; NaN where missing."""
    idx = hourly_grid(start, end)
    s = (
        df.dropna(subset=[time_col])
        .set_index(time_col)[col]
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="first")]
    out = s.reindex(idx)
    out.name = col
    return out


def split_labels(idx: pd.DatetimeIndex) -> pd.Series:
    lab = pd.Series("none", index=idx, dtype="object")
    for name, (a, b) in SPLITS.items():
        a_ts, b_ts = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        lab[(idx >= a_ts) & (idx < b_ts)] = name
    return lab.astype("category")


def split_completeness(series: pd.Series) -> dict[str, float]:
    lab = split_labels(series.index)
    return {name: float(series[lab == name].notna().mean()) for name in SPLITS}
