"""
Pull ERA5 (Open-Meteo) per Tier-1 station at its exact lat/lon. Saves
data/raw/era5/{code}_2019_2024.parquet and appends to the manifest.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.aurn import aurn_coords  # noqa: E402
from aq_fm_bench.data.era5 import fetch_openmeteo_era5  # noqa: E402
from aq_fm_bench.data.stations import TIER1  # noqa: E402
from aq_fm_bench.paths import RAW_AURN, RAW_ERA5  # noqa: E402
from aq_fm_bench.utils import manifest  # noqa: E402


def main() -> int:
    aurn_stations = [s for s in TIER1 if s.source == "aurn"]
    coords = aurn_coords([s.code for s in aurn_stations], RAW_AURN)
    RAW_ERA5.mkdir(parents=True, exist_ok=True)

    for s in aurn_stations:
        lat, lon = coords[s.code]
        print(f"\n=== ERA5 {s.code} ({s.city}) @ {lat:.4f},{lon:.4f} ===")
        df = fetch_openmeteo_era5(lat, lon)
        out = RAW_ERA5 / f"{s.code}_2019_2024.parquet"
        df.to_parquet(out, index=False)
        print(f"  {len(df):,} hrs  {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
        print(f"  BLH range {df['blh'].min():.0f}-{df['blh'].max():.0f} m | "
              f"ws {df['ws10'].mean():.1f} m/s | t2m {df['t2m_c'].mean():.1f}C | "
              f"NaN blh {df['blh'].isna().mean():.1%}")
        manifest.append([{
            "source": "era5_openmeteo", "city": s.city, "station": s.code, "year": "2019-2024",
            "pull_date_utc": manifest.utc_now(), "sha256": manifest.sha256_file(out),
            "n_rows": len(df), "notes": f"lat={lat},lon={lon}",
        }])
        time.sleep(2.0)  # be polite to the free API
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
