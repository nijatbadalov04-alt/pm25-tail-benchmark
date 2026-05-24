"""
Build the processed per-station hourly dataset: AURN pollutants + ERA5 covariates on one
canonical UTC hourly grid, with split labels. Deterministic.

Output: data/processed/{code}_hourly.parquet
Schema: timestamp_utc, station, city, env_type, no2, pm25,
        t2m_c, sp_hpa, tp_mm, blh, ws10, wd10, u10, v10, split
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.processing import hourly_grid, split_labels, to_hourly_series  # noqa: E402
from aq_fm_bench.data.stations import TIER1  # noqa: E402
from aq_fm_bench.paths import PROCESSED, RAW_AURN, RAW_ERA5  # noqa: E402

MET_COLS = ["t2m_c", "sp_hpa", "tp_mm", "blh", "ws10", "wd10", "u10", "v10"]


def main() -> int:
    grid = hourly_grid()
    splits = split_labels(grid)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    for s in TIER1:
        aq = pd.read_parquet(RAW_AURN / f"{s.code}_2019_2024.parquet")
        met = pd.read_parquet(RAW_ERA5 / f"{s.code}_2019_2024.parquet")

        out = pd.DataFrame({"timestamp_utc": grid})
        out["station"] = s.code
        out["city"] = s.city
        out["env_type"] = s.env_type
        out["no2"] = to_hourly_series(aq, "NO2").to_numpy(dtype="float32") if "NO2" in aq.columns else np.nan
        out["pm25"] = to_hourly_series(aq, "PM2.5").to_numpy(dtype="float32") if "PM2.5" in aq.columns else np.nan

        m = met.drop_duplicates("timestamp_utc").set_index("timestamp_utc").reindex(grid)
        for c in MET_COLS:
            out[c] = m[c].to_numpy(dtype="float32") if c in m.columns else np.nan
        out["split"] = splits.to_numpy()

        path = PROCESSED / f"{s.code}_hourly.parquet"
        out.to_parquet(path, index=False)
        no2c = out["no2"].notna().mean()
        pm25c = out["pm25"].notna().mean()
        blhc = out["blh"].notna().mean()
        print(f"  {s.code:5} ({s.city:10}) rows={len(out):,} "
              f"no2={no2c:.0%} pm25={pm25c:.0%} blh={blhc:.0%} -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
