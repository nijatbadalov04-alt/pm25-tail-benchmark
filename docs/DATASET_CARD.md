---
license: cc-by-4.0
language:
  - en
pretty_name: "AQ-FM-Bench v1 — UK Urban Air-Quality Forecasting Benchmark"
tags:
  - air-quality
  - time-series
  - forecasting
  - foundation-models
  - conformal-prediction
  - NO2
  - PM2.5
size_categories:
  - 1M<n<10M
task_categories:
  - time-series-forecasting
---

# AQ-FM-Bench v1 — Dataset Card

## Dataset description
Hourly NO₂ and PM₂.₅ concentrations from **eight reference-grade DEFRA AURN monitoring stations across
four UK cities** (2019–2024), merged on a canonical UTC hourly grid with **ERA5 meteorological
covariates** (10 m wind, 2 m temperature, surface pressure, boundary-layer height, precipitation). It is
the processed substrate for the AQ-FM-Bench benchmark, which evaluates naive, classical, and time-series
foundation-model forecasters with significance testing and conformal uncertainty, focusing on
high-pollution episodes.

- **Curated by:** Nijat Badalov
- **Language:** N/A (numeric time series)
- **License:** CC-BY-4.0 (processed dataset). See *Licensing* below for source attributions.
- **Repository:** https://github.com/nijatbadalov04-alt/pm25-tail-benchmark

## Stations
| code | city | environment | pollutants |
|---|---|---|---|
| SHDG | Sheffield | urban background | NO₂, PM₂.₅ |
| SHBR | Sheffield | urban traffic | NO₂, PM₂.₅ |
| KC1 | London | urban background | NO₂, PM₂.₅ |
| MY1 | London | urban traffic (Marylebone Rd) | NO₂, PM₂.₅ |
| BMLD | Birmingham | urban background | NO₂, PM₂.₅ |
| BIRR | Birmingham | urban traffic (A4540 CAZ ring road) | NO₂, PM₂.₅ |
| MAN3 | Manchester | urban background | NO₂, PM₂.₅ |
| MAHG | Manchester | suburban industrial | NO₂ only (PM₂.₅ 6% complete → excluded) |

All stations are Köppen **Cfb** (temperate maritime). **Scope is UK-temperate only** — no claim
generalises beyond it.

## Schema (`data/processed/{code}_hourly.parquet`)
| column | units | description |
|---|---|---|
| timestamp_utc | — | hourly UTC timestamp |
| station, city, env_type | — | station code, city, environment type |
| no2, pm25 | µg/m³ | pollutant concentrations (raw observed; NaN where missing) |
| t2m_c | °C | 2 m air temperature |
| sp_hpa | hPa | surface pressure |
| tp_mm | mm | total precipitation |
| blh | m | boundary-layer height (governs vertical mixing) |
| ws10, wd10 | m/s, ° | 10 m wind speed and direction |
| u10, v10 | m/s | 10 m wind components |
| split | — | `train` / `cal` / `test` |

## Splits (chronological, fixed, deterministic)
- **train:** 2019-01-01 – 2022-12-31
- **calibration:** 2023-01-01 – 2023-06-30 (conformal residuals only)
- **test:** 2023-07-01 – 2024-12-31 (all reporting)

Splits are **date-defined constants** (`aq_fm_bench.data.processing.SPLITS`), not index files — exactly
reproducible from the timestamp column. Evaluation is rolling-origin with a 7-day (168 h) stride.

## Provenance
- **Air quality:** DEFRA Automatic Urban and Rural Network (AURN), reference-grade, parsed from per-site
  `.RData` with the pure-Python `rdata` library. Completeness 2019–2024: NO₂ 78.8–99.0%, PM₂.₅
  88.2–99.5% (seven stations; MAHG PM₂.₅ excluded at 6%).
- **Meteorology:** ERA5 via the **Open-Meteo** archive, extracted at each station's exact lat/lon;
  cross-validated against authoritative native ERA5 (ECMWF ARCO-ERA5) — correlations ≈ 1.000 for all six
  variables, BLH RMSE 1.4 m.
- Pull dates and exact source URLs are in the repository `data/manifest.csv`.

## Intended use
Benchmarking probabilistic and point time-series forecasters (incl. zero-shot foundation models) on
urban air quality, with emphasis on **episode (top-decile) performance** and **calibrated uncertainty**.
Suitable for forecasting, conformal-prediction, and environmental-ML research.

## Known limitations
- **UK-temperate scope** (all Köppen Cfb); no cross-climate claim.
- **Manchester PM₂.₅ is single-station** (MAHG is NO₂-only).
- **Oracle-weather** features use ERA5 reanalysis at target time (an information ceiling); a
  lookback-only variant brackets the realistic forecast-weather case.
- Half-year calibration window (Jan–Jun 2023) may under-represent summer regimes for conformal.

## Licensing & attribution
- Processed dataset: **CC-BY-4.0**.
- DEFRA AURN: © Crown copyright, **Open Government Licence v3.0**.
- ERA5: ECMWF / Copernicus Climate Change Service; Open-Meteo archive **CC-BY-4.0**.
- OpenAQ (station discovery / cross-validation only): OpenAQ open data.

## Citation
See `CITATION.cff` in the repository. A Zenodo DOI is minted on the v1.0.0 release.
