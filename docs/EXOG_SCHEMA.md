# Exogenous covariate schema (FM-with-weather comparison)

All exogenous-capable models consume the **same 6 ERA5 covariates in the same units**, so the
comparison to oracle-weather XGBoost is apples-to-apples. Chronos receives **none** (no exog
support) — that asymmetry is RQ3.

## The 6 covariates (canonical, identical across models)

| name | source column | unit |
|---|---|---|
| u10 | `u10` | m/s |
| v10 | `v10` | m/s |
| t2m | `t2m_c` | °C |
| sp  | `sp_hpa` | hPa |
| blh | `blh` | m |
| tp  | `tp_mm` | mm/h |

Source: `data/processed/{code}_hourly.parquet` (Open-Meteo ERA5, per-station exact coords;
validated against native ERA5 — see `docs/ERA5_VALIDATION.md`). Covariate NaNs (~8% for BLH) are
forward-filled within the context/horizon window before being passed to the models.

## Time alignment & the lookback/forecast boundary

For a forecast origin `t0`, context length `C`, horizon `H`:
- **Lookback (past) window:** hours `[t0-C+1 .. t0]` — target series + covariates both observed.
- **Forecast (future) window:** hours `[t0+1 .. t0+H]` — target unknown; covariates **known**
  (we treat ERA5 reanalysis as the known-future weather, exactly the information oracle-weather
  XGBoost receives — this is what makes the comparison fair).

## TimesFM (main venv, GPU) — covariate API
- Entry: `forecast_with_covariates(...)`.
- Each covariate passed as a **dynamic numerical covariate** spanning the **full** `C+H` window
  (past + known future); the target series spans the past `C` only.
- `xreg_mode="xreg + timesfm"` (TimesFM core forecast + in-context linear regression on covariates).
- freq=0 (high-frequency/hourly). Point forecast = median output.

## Moirai-1.1-R (isolated `.venv-fm`, CPU, via subprocess) — covariate API
- uni2ts `MoiraiForecast` / GluonTS-style fields:
  - `past_feat_dynamic_real` ← covariates over the **lookback** `[t0-C+1 .. t0]`.
  - `feat_dynamic_real` ← covariates over the **full** series incl. the **known-future**
    `[t0+1 .. t0+H]` (Moirai's known-future-covariate channel).
- Target `past_target` ← series over lookback only. Patch/context length set to `C`.
- Quantile outputs {0.1, 0.5, 0.9}; point = 0.5.

## Chronos-Bolt — none
No exogenous support. Target-only. Its gap vs the exog models **is** the RQ3 measurement.

## Reporting
- Wall-clock **inference time per forecast** recorded per model (Moirai CPU vs FM/XGBoost GPU
  asymmetry made explicit — protects the comparison from reviewer challenge).
- Overall MAE + **top-decile MAE**, both pollutants, every row.
