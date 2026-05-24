"""
ERA5 meteorological covariates via the Open-Meteo archive API (keyless).

Point extraction at each station's EXACT lat/lon (not a city centroid), so nearby cities
never collapse to one value. Includes boundary-layer height (the key PM2.5 covariate).

Returned columns (UTC hourly): t2m_c (degC), sp_hpa (hPa), tp_mm (mm), blh (m),
ws10 (m/s), wd10 (deg), u10/v10 (m/s, derived from speed+direction).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY = ["temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
           "precipitation", "boundary_layer_height"]


def fetch_openmeteo_era5(lat: float, lon: float, *, start: str = "2019-01-01",
                         end: str = "2024-12-31", timeout: int = 180, retries: int = 4) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(_HOURLY), "timezone": "GMT", "models": "era5",
        "wind_speed_unit": "ms",
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                break
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 + 3 * attempt)
    else:
        raise RuntimeError(f"Open-Meteo failed for ({lat},{lon}): {last}")

    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    ws = df["wind_speed_10m"].astype("float64")          # already m/s (wind_speed_unit=ms)
    wd = df["wind_direction_10m"].astype("float64")
    rad = np.deg2rad(wd)
    out = pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "t2m_c": df["temperature_2m"].astype("float32"),
        "sp_hpa": df["surface_pressure"].astype("float32"),
        "tp_mm": df["precipitation"].astype("float32"),
        "blh": df["boundary_layer_height"].astype("float32"),
        "ws10": ws.astype("float32"),
        "wd10": wd.astype("float32"),
        "u10": (-ws * np.sin(rad)).astype("float32"),     # meteorological "from" convention
        "v10": (-ws * np.cos(rad)).astype("float32"),
    })
    return out
