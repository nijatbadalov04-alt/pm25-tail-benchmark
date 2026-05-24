"""
Pull DEFRA AURN reference-grade data (no API key needed).

Usage:
    .venv\\Scripts\\python.exe scripts\\pull_aurn.py --sites SHDG --city sheffield --years 2019-2024
    .venv\\Scripts\\python.exe scripts\\pull_aurn.py --sites SHDG,MY1 --years 2019-2024

Saves:
    data/raw/aurn/{SITE}_{YEAR}.RData          (raw dumps, never modified)
    data/raw/aurn/{SITE}_{start}_{end}.parquet (parsed hourly, UTC)
and appends rows to data/manifest.csv.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

# allow running as a plain script without installation
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.aurn import load_aurn_site, completeness  # noqa: E402
from aq_fm_bench.paths import RAW_AURN  # noqa: E402
from aq_fm_bench.utils import manifest  # noqa: E402


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, help="comma-separated AURN site codes, e.g. SHDG,MY1")
    ap.add_argument("--city", default="", help="city label for the manifest")
    ap.add_argument("--years", default="2019-2024", help="e.g. 2019-2024 or 2019,2020")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    years = parse_years(args.years)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    start, end = f"{years[0]}-01-01", f"{years[-1] + 1}-01-01"

    for site in sites:
        print(f"\n=== AURN {site}  years {years[0]}-{years[-1]} ===")
        df = load_aurn_site(site, years, RAW_AURN, force=args.force)
        out = RAW_AURN / f"{site}_{years[0]}_{years[-1]}.parquet"
        df.to_parquet(out, index=False)
        print(f"  saved {out.name}: {len(df):,} rows, "
              f"{df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
        for pol in ("NO2", "PM2.5"):
            c = completeness(df, pol, start, end)
            flag = "OK" if c >= 0.75 else "LOW"
            print(f"  completeness {pol:6}: {c:6.1%}  [{flag}]")

        # manifest: one row per downloaded year file
        rows = []
        for y in years:
            rdata_path = RAW_AURN / f"{site}_{y}.RData"
            if rdata_path.exists():
                rows.append({
                    "source": "aurn", "city": args.city, "station": site, "year": y,
                    "pull_date_utc": manifest.utc_now(),
                    "sha256": manifest.sha256_file(rdata_path),
                    "n_rows": "", "notes": "reference-grade ug/m3",
                })
        manifest.append(rows)
        print(f"  manifest: appended {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
