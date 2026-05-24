"""
DEFRA AURN (Automatic Urban and Rural Network) loader.

AURN is reference-grade UK air quality data, openly downloadable as per-site-per-year
R `.RData` files. No API key required. Values are already in µg/m³.

We parse with the pure-Python `rdata` library (librdata/pyreadr cannot read DEFRA's
serialization). The hourly object inside `{SITE}_{YEAR}.RData` is named `{SITE}_{YEAR}`
and has columns: date (epoch seconds, UTC), O3, NO, NO2, NOXasNO2, PM10, PM2.5, wd, ws,
temp, site, code.

Example
-------
    from aq_fm_bench.data.aurn import load_aurn_site
    df = load_aurn_site("SHDG", range(2019, 2025), raw_dir)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import rdata

AURN_BASE = "https://uk-air.defra.gov.uk/openair/R_data"
_UA = {"User-Agent": "aq-fm-bench/0.1 (research; contact omitted)"}

# AURN hourly columns we retain (others: site, code, aggregate objects -> dropped)
POLLUTANTS = ["NO2", "PM2.5", "PM10", "O3", "NO", "NOXasNO2"]
ONSITE_MET = ["ws", "wd", "temp"]


def download_aurn_rdata(site: str, year: int, dest_dir: Path, *, force: bool = False,
                        timeout: int = 120) -> Path:
    """Download one site-year .RData (idempotent: skips if already present)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{site}_{year}.RData"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    url = f"{AURN_BASE}/{site}_{year}.RData"
    resp = requests.get(url, timeout=timeout, headers=_UA)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def parse_aurn_rdata(path: Path, site: str, year: int) -> pd.DataFrame:
    """Parse a site-year .RData into a tidy hourly DataFrame with UTC timestamps."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence POSIXct constructor warnings
        parsed = rdata.read_rda(str(path))

    key = f"{site}_{year}"
    if key not in parsed:
        # fallback: the largest DataFrame (the hourly table)
        frames = {k: v for k, v in parsed.items() if isinstance(v, pd.DataFrame)}
        if not frames:
            raise ValueError(f"No DataFrame found in {path}")
        key = max(frames, key=lambda k: len(frames[k]))
    df = parsed[key].copy()

    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in {path} object {key}")
    # epoch seconds (UTC) -> tz-aware UTC timestamp
    df["timestamp_utc"] = pd.to_datetime(df["date"].astype("float64"), unit="s", utc=True)
    keep = ["timestamp_utc"] + [c for c in POLLUTANTS + ONSITE_MET if c in df.columns]
    df = df[keep].copy()
    df["site_code"] = site
    df["year"] = year
    return df


def load_aurn_site(site: str, years: Iterable[int], raw_dir: Path, *,
                   force: bool = False) -> pd.DataFrame:
    """Download + parse all requested years for a site; return one sorted DataFrame."""
    frames = []
    for y in years:
        path = download_aurn_rdata(site, int(y), raw_dir, force=force)
        try:
            frames.append(parse_aurn_rdata(path, site, int(y)))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {site} {y}: parse failed -> {e!r}")
    if not frames:
        raise RuntimeError(f"No data parsed for {site}")
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp_utc"], keep="first")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    return out


def load_aurn_metadata(raw_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Download + parse the AURN site catalogue (has lat/lon, type, parameters per site)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "AURN_metadata.RData"
    if not path.exists() or force:
        resp = requests.get(f"{AURN_BASE}/AURN_metadata.RData", timeout=120, headers=_UA)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.read_rda(str(path))
    dfs = [v for v in parsed.values() if isinstance(v, pd.DataFrame)]
    return max(dfs, key=len)


def aurn_coords(codes, raw_dir: Path) -> dict[str, tuple[float, float]]:
    """Return {site_id: (lat, lon)} for the requested AURN site codes."""
    md = load_aurn_metadata(raw_dir).drop_duplicates(subset=["site_id"])
    want = set(codes)
    return {
        row["site_id"]: (float(row["latitude"]), float(row["longitude"]))
        for _, row in md.iterrows()
        if row["site_id"] in want
    }


def completeness(df: pd.DataFrame, col: str, start: str, end: str) -> float:
    """Fraction of expected hourly slots in [start, end) that are non-null for `col`."""
    if col not in df.columns:
        return 0.0
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    s = df.set_index("timestamp_utc")[col].reindex(idx)
    return float(s.notna().mean())
