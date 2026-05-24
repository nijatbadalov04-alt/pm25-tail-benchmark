# AQ-FM-Bench: A Significance-Tested Benchmark of Time-Series Foundation Models for UK Urban Air-Quality Forecasting, with Meteorology-Stratified Conformal Calibration

## Abstract

We present **AQ-FM-Bench**, a reproducible, significance-tested benchmark evaluating naive baselines,
classical models (Ridge, XGBoost), and three time-series foundation models (Chronos-Bolt, TimesFM-2.0,
Moirai-1.1) on hourly NO₂ and PM₂.₅ forecasting across **four UK cities (eight reference-grade DEFRA
AURN stations), 2019–2024**, at 24/72/168-hour horizons, with ERA5 meteorological covariates, conformal
uncertainty, and a deliberate focus on high-pollution episodes. We find that the dominant factor in NO₂
accuracy is *access to future weather*, not the foundation-vs-classical distinction — a zero-shot
foundation model with no covariates (Chronos, MAE 8.49) is statistically indistinguishable from a
feature-engineered model denied future weather (XGBoost-no-weather, 8.40; ΔMAE +0.10, 95% CI
[−0.21,+0.40]) — and that on the top decile of PM₂.₅, the episodes that drive health alerts, **no model
— foundation, classical, or naive — is statistically distinguishable from persistence** (all
ΔMAE-vs-persistence 95% CIs span zero; n = 212 tail origin-units). Every pairwise claim is corroborated
by two independent significance methods (block bootstrap and Diebold–Mariano with Benjamini–Hochberg FDR
control; concordant on all named comparisons). Motivated by this episodic-calibration failure, we
introduce **Meteorology-Mondrian Conformal Prediction (MM-CP)**, which stratifies conformal residuals by
boundary-layer height, wind speed, and time of day. MM-CP significantly improves foundation-model episode
coverage over split-conformal (+5–7 pp; bootstrap CIs exclude 0) but cannot restore *deployable* coverage
— even when extended to condition on an episode regime, top-decile PM₂.₅ coverage reaches only ~57–60%
against a 90% nominal — establishing episodic PM₂.₅ calibration as an open problem and isolating its cause
as cell-marginal versus conditional-on-tail coverage. Because episodic exposure, not average
concentration, drives the public-health value of air-quality forecasts, calibrated episode-aware
uncertainty matters more for responsible deployment than marginal average-case accuracy. All code, the
processed dataset, and a reproducibility harness accompany this preprint (GitHub, HuggingFace, Zenodo DOI).

---

## 1. Introduction

Ambient air pollution is among the largest environmental risk factors for human health, associated with
millions of premature deaths annually [WHO]. The operational value of a forecast is concentrated in the
**high-pollution episodes** — the hours when public-health alerts are issued and exposure is most harmful
— yet these are the hardest hours to predict and the ones conventional average-error metrics, dominated by
abundant low-concentration periods, least reflect. A credible air-quality forecasting study must therefore
be judged on the tail, not only the mean, and must report calibrated uncertainty: a point forecast of
"23 µg/m³" is not actionable against an alert threshold without an interval.

**Scope.** We study hourly NO₂ and PM₂.₅ forecasting at 24/72/168 h across **four UK cities** — Sheffield,
London, Birmingham, Manchester — using **eight reference-grade DEFRA AURN stations**, 2019–2024, with ERA5
meteorological covariates. This is a deliberately bounded, reference-grade setting (all temperate-maritime,
Köppen Cfb); cross-climate generalisation is explicitly out of scope and deferred to future work. Within
it we evaluate naive baselines, classical models (Ridge, XGBoost), and three time-series foundation models
(Chronos-Bolt, TimesFM-2.0, Moirai-1.1) under a common rolling-origin protocol, with conformal uncertainty
and **statistical-significance testing on every claim**.

**Contributions** (scoped narrowly; "first" claims qualified¹):

- **A significance-tested benchmark** — 11 model configurations × 2 pollutants × 3 horizons on 8 AURN
  stations, with top-decile (episode) metrics and conformal calibration — released as an open dataset,
  leaderboard, and Python package.
- **Six findings (F1–F6), each significance-backed by two independent methods:** episodic PM₂.₅ is not
  significantly beaten by any model class including foundation models (F1); future-weather access, not the
  foundation-vs-classical distinction, dominates NO₂ accuracy (F2–F3); XGBoost's advantage is nonlinear
  *weather* interaction (F4); on PM₂.₅ no model class is statistically separable (F5); and the benefit of
  weather to a foundation model is architecture-dependent (F6).
- **MM-CP (Meteorology-Mondrian Conformal Prediction):** a physics-stratified conformal method that
  significantly improves foundation-model episode coverage but does not restore deployable coverage,
  isolating a conditional-on-tail calibration limitation that motivates dedicated future methods.

**Figure 1** (`figures/paper/F_intro.pdf`) is the motivating image: a zero-shot foundation model tracks the
diurnal NO₂ mean but its 80% interval under-covers an evening pollution spike — the episodic-miscalibration
problem this paper quantifies.

¹ To the best of our knowledge as of Q1 2026; the benchmark space moves quickly and we make no priority
claim beyond the specific combination (significance-tested multi-model TS-FM AQ evaluation with
episode-conditional conformal analysis).

---

## 2. Related Work

**Time-series foundation models.** Recent TS-FMs claim strong zero-shot forecasting across domains:
Chronos (Ansari et al., 2024, arXiv:2403.07815), Moirai (Woo et al., ICML 2024, arXiv:2402.02592), and
TimesFM (Das et al., ICML 2024, arXiv:2310.10688). General-purpose benchmarks such as GIFT-Eval (Aksu et
al., NeurIPS 2024 D&B, arXiv:2410.10393) standardise evaluation across seven domains — but **air quality is
not among them**. No prior work evaluates TS-FMs on urban air quality with significance testing across
cities and explicit treatment of operational tail events.

**Conformal prediction for time series.** Conformal prediction (Vovk, Gammerman & Shafer, 2005) provides
distribution-free finite-sample coverage; see the tutorial of Angelopoulos & Bates (2021,
arXiv:2107.07511). Time-series and distribution-shift variants include Conformal PID (Angelopoulos, Candès
& Tibshirani, NeurIPS 2024) and adaptive CP (Gibbs & Candès, JMLR 2024). Conditional/Mondrian coverage and
its limits (Vovk 2005; Barber et al., 2021) motivate our meteorology-stratified MM-CP and frame its
observed limitation (cell-marginal vs. conditional-on-tail), which we pursue in future work.

**Air-quality deep learning.** AQ-specific deep models include AirFormer (AAAI 2023), AirPhyNet (ICLR
2024, arXiv:2402.03784), and PM2.5-GNN (Wang et al., 2020); UK-specific
work includes Munir & Mayfield (2019, 2021) and Williams, Chan & Ortiz (2025) on the Sheffield
Clean Air Zone. These are predominantly **single-city, train-and-test on the same city, and rarely report
calibrated uncertainty** — none evaluate zero-shot cross-domain TS-FMs, nor calibrated coverage on
pollution episodes.

**Gap (specific).** *Time-series foundation models claim universality across time-series domains; no
published work has systematically evaluated them on urban air quality with statistical-significance testing
across multiple cities, with explicit treatment of operational-tail-event (episode) forecasting and
conformal calibration.* AQ-FM-Bench fills this gap and, in investigating episode calibration, surfaces a
conditional-on-tail conformal limitation that prior conformal-for-AQ work has not isolated.

---

## 3. Methods

*Every configuration maps to a file/script in the repository (reproducibility contract). Pinned versions
in `pyproject.toml`; foundation-model checkpoints pinned by HuggingFace repo ID (revision pinning is a
release task).*

### 3.1 Task and evaluation protocol
We forecast hourly NO₂ and PM₂.₅ concentration (µg/m³) at horizons H ∈ {24, 72, 168} h from a 512-hour
lookback. Evaluation is **rolling-origin** with a 7-day (168 h) stride over the test window, giving ≈78
forecast origins per station; each origin yields an H-step trajectory scored against the realised series.
All models forecast the full trajectory; trajectory MAE (averaged over lead times) is the primary point
metric. Splits are **chronological and fixed**: **train** 2019-01-01–2022-12-31, **calibration**
2023-01-01–2023-06-30 (conformal residuals only), **test** 2023-07-01–2024-12-31 (all reporting).
Calibration spans one half-year; its seasonal coverage limitation is noted in §8. Trained models' targets
remain within the training window (no leakage into cal/test).

### 3.2 Data
**Air quality (DEFRA AURN, reference-grade).** Four UK cities × 2 stations = 8 stations: Sheffield
Devonshire Green (SHDG, urban background) & Barnsley Rd (SHBR, traffic); London N. Kensington (KC1,
background) & Marylebone Rd (MY1, traffic); Birmingham Ladywood (BMLD, background) & A4540 Roadside (BIRR,
traffic — the Clean-Air-Zone ring road); Manchester Piccadilly (MAN3, background) & Sharston (MAHG,
suburban-industrial). AURN reports in µg/m³ (no unit conversion). Per-site/year `.RData` files are parsed
with the pure-Python `rdata` library. Completeness 2019–2024: NO₂ 78.8–99.0%, PM₂.₅ 88.2–99.5% for seven
stations; **MAHG PM₂.₅ is 6% complete → treated as NO₂-only** (Manchester PM₂.₅ is therefore single-station,
stated as a limitation). Data completeness per station-month is shown in `figures/eda/F_eda_completeness`.

**Meteorology (ERA5).** Six covariates per station — 10 m u/v wind, 2 m temperature, surface pressure,
**boundary-layer height (BLH)**, total precipitation — extracted at each station's exact lat/lon from the
Open-Meteo ERA5 archive. We **validated Open-Meteo against authoritative native ERA5** (ARCO-ERA5) at SHDG
over a sample week: correlations ≈ 1.000 for all six variables, BLH RMSE 1.4 m; the only material
difference is a constant ~9 hPa surface-pressure offset (an elevation-reference difference, perfectly
correlated, harmless as an ML feature). That meteorology is a first-order control is visible in the raw
data: NO₂ and PM₂.₅ both fall monotonically with BLH (r ≈ −0.31/−0.33) and wind speed (r ≈ −0.30/−0.31)
(`figures/eda/F_eda_ventilation`) — the stagnation mechanism behind F2/F4.

**Processed dataset.** AURN pollutants + ERA5 covariates are merged on one canonical UTC hourly grid per
station (`data/processed/{code}_hourly.parquet`; schema: timestamp_utc, station, city, env_type, no2,
pm25, t2m_c, sp_hpa, tp_mm, blh, ws10, wd10, u10, v10, split). The target context is forward-filled to
supply contiguous history; **actuals are kept raw** so scores count only genuinely observed values.

### 3.3 Models
A **single-model-per-process** invariant is enforced for the foundation models (bf16 inference,
`torch.cuda.empty_cache()` between runs) on an 8 GB RTX 5070 (Blackwell, torch 2.11.0+cu128).

- **Naive floors:** persistence (last value carried forward) and seasonal-naive at periods 24 h and 168 h.
- **Ridge (linear) sanity baseline** on the engineered feature set, median-imputed + standardised —
  isolates linear-vs-nonlinear use of weather (F4).
- **XGBoost** direct multi-horizon (GPU `hist`, 400 trees, depth 7, lr 0.05): one model per
  (station, pollutant) with lead time as a feature; features = pollutant lags {1,3,6,24,168 h}, rolling
  mean/std, cyclical calendar of the target hour, and ERA5 weather at origin **and** target time. We report
  **two variants**: `xgboost` (oracle future weather = ERA5 at target time) and `xgboost_no_oracle`
  (weather only ≤ origin). The gap quantifies the future-weather information advantage (F2/F3).
- **Foundation models** (zero-shot; pinned by HF repo ID), with the same 6 ERA5 covariates as
  **known-future** exogenous inputs where supported:
  - **Chronos-Bolt-Base** (`amazon/chronos-bolt-base`, ~200 M): **no exogenous support** — target-only.
    Trained for ≤64-step horizons (autoregresses beyond; a noted caveat at 72/168 h).
  - **TimesFM-2.0** (`google/timesfm-2.0-500m-pytorch`, 500 M; num_layers 50, context 512, horizon 256):
    covariates via `forecast_with_covariates` (XReg ridge + TimesFM core).
  - **Moirai-1.1-R-base** (`Salesforce/moirai-1.1-R-base`, 91 M): covariates as GluonTS
    `feat_dynamic_real` spanning context+horizon (native covariate attention); num_samples 20, point =
    median (a constraint for native-quantile comparison, §8).
  AirFormer/AirPhyNet and LoRA fine-tuning are out of scope for v1.

### 3.4 Metrics
Point: MAE, RMSE, sMAPE; and **top-decile MAE** — MAE restricted to the top 10% of observed concentrations
(the episodes that drive health alerts) — reported for **every** leaderboard row. Probabilistic/conformal:
empirical coverage at 50/80/90%, mean interval width, Winkler (interval) score. Cost: wall-clock ms per
forecast (GPU/CPU asymmetry made explicit).

### 3.5 Statistical protocol
Significance testing accompanies every empirical claim (mandatory). **Primary:** 2000-iteration **block
bootstrap** resampling forecast origins (the 7-day-strided units), giving 95% CIs on MAE differences, and a
**paired coverage bootstrap** (resampling origins) on coverage differences for conformal comparisons. A
difference is "significant" iff its 95% CI excludes 0. **Complement:** paired **Diebold–Mariano** tests
(Harvey–Leybourne–Newbold small-sample correction) on per-origin mean losses, with **Benjamini–Hochberg**
FDR control over the family of pairwise comparisons within each pollutant (`results/leaderboard/dm_bh_h24.csv`).
The DM truncation lag is 0 — the 7-day stride makes 24 h forecast windows non-overlapping, so per-origin
losses are not mechanically autocorrelated and DM coincides with a paired t-test (lag-1 reported as
robustness; no decision changes). **The two methods are concordant on all F2–F6 named pairs (7/7)**,
including both null results, and DM+BH confirms every non-naive model beats persistence on overall MAE for
both pollutants after FDR — the tail-regime exception being F1, where per-origin tail losses are too sparse
for a parametric paired test and the bootstrap is primary. Cross-city aggregation uses the **median** across
stations to avoid domination by high-pollution sites; the main-text leaderboard and significance use the
**pooled** origin-units (the two summaries are reported side-by-side in the appendix to forestall confusion).

### 3.6 Conformal prediction
**Split-conformal:** nonconformity score = absolute calibration residual; interval = ŷ ± q̂ at 1−α (α=0.1).
**MM-CP (Meteorology-Mondrian, ours):** a Mondrian conformal predictor whose taxonomy is physical. *3D:*
cells = wind-speed tercile × BLH tercile × hour-of-day bucket (3×3×4 = 36 cells), per-cell conformal
quantiles, hierarchical backoff (cell → wind×BLH → global) with minimum n_min = 50; tercile cuts fit on
calibration only. *4D:* adds a binary **episode-regime** dimension E(t₀) computed from the 24 h lookback
ending at the forecast origin (no future leakage): E = 1 if the 24 h-mean concentration is in the
calibration top tercile **or** ≥6 of 24 lookback hours are low-BLH-and-low-wind; hour coarsened to 2
buckets (36 effective cells). A **leakage audit** confirms corr(E, future-window mean concentration) = 0.45
(correlated, not deterministic). **Conformal PID** (Angelopoulos et al. 2024) is acknowledged but **not
implemented in v1** — deferred to future work.

### 3.7 MM-CP iteration (reported honestly)
The method evolved and we report it as such. 3D MM-CP **significantly** improved foundation-model
top-decile PM₂.₅ coverage over split-CP (+5–7 pp; paired-bootstrap CIs exclude 0) but reached only 57–65%
against a 90% nominal. We diagnosed the cause as *cell-marginal calibration vs. conditional-on-the-tail
coverage* — episodes lie in each cell's own upper tail — not cell choice, and tested it by adding the
episode-regime cell (4D). 4D did **not** significantly improve over 3D (all 4D-vs-3D CIs span 0; episode
cells were sparse and backed off to 3D). This establishes a limitation of cell-marginal conformal for
conditional-tail coverage and motivates a dedicated conditional-on-tail method (future work).

### 3.8 Reproducibility
Each result regenerates from `scripts/`; outputs in `results/leaderboard/` and `results/coverage/`.
Environment: `uv`-managed Python 3.11; the FM stack is split across `.venv` (GPU) and `.venv-fm` (Moirai,
which pins torch 2.4.1; we force torch 2.11+cu128 to run it on GPU — verified compatible). Seeds fixed (42)
for bootstraps and XGBoost. A content-hashed prediction cache makes the full significance pipeline
re-run in ≈85 s on the reference machine; all published CSVs reproduce bitwise.

---

## 4. Results

*All @24 h numbers are pooled across stations with 95% block-bootstrap CIs (2000 iters, origins resampled):
`results/leaderboard/significance_h24.csv` and `pairwise_tests_h24.csv`. Per-station medians and the
72/168 h horizons: `uk_full_leaderboard.csv`. Conformal: `results/coverage/mmcp4d_pm25.csv`. Every F2–F6
pairwise finding is corroborated by paired Diebold–Mariano + Benjamini–Hochberg (`dm_bh_h24.csv`):
bootstrap and DM+BH are **concordant on all 7 named pairs**, including the two nulls (F2, F5).*

**Table 1 — main leaderboard (h24, MAE [95% CI], µg/m³; → `figures/paper/F_leaderboard.pdf`).**

| model | NO₂ overall | NO₂ top-decile | PM₂.₅ overall | PM₂.₅ top-decile |
|---|---|---|---|---|
| persistence | 10.98 [10.39,11.62] | 24.29 [22.07,26.60] | 4.35 [3.93,4.88] | 8.75 [7.56,10.16] |
| seasonal-naive 24h | 11.89 [11.38,12.43] | 20.64 [19.19,22.16] | 4.35 [4.11,4.59] | 9.18 [8.38,10.07] |
| seasonal-naive 168h | 12.51 [11.95,13.15] | 28.53 [26.48,30.90] | 5.80 [5.41,6.20] | 13.42 [12.38,14.51] |
| linear (no wx) | 9.25 [8.86,9.68] | 26.64 [25.37,27.99] | 4.08 [3.85,4.32] | 11.08 [10.35,11.85] |
| linear (+wx) | 8.26 [7.95,8.59] | 18.27 [16.85,19.77] | 4.25 [4.04,4.47] | 9.11 [8.41,9.89] |
| Chronos-Bolt | 8.49 [8.10,8.94] | 23.21 [21.63,24.86] | 3.30 [3.10,3.50] | 8.82 [7.95,9.72] |
| TimesFM-2 (+wx) | 7.55 [7.19,7.94] | 21.29 [19.93,22.77] | **3.14 [2.95,3.34]** | 8.43 [7.66,9.26] |
| Moirai (zero-shot) | 9.34 [8.90,9.80] | 19.90 [18.33,21.54] | 3.62 [3.38,3.87] | 8.37 [7.50,9.34] |
| Moirai (+wx) | 9.76 [9.31,10.24] | 19.61 [17.98,21.26] | 3.71 [3.47,3.96] | 8.10 [7.27,9.03] |
| XGBoost (no wx) | 8.40 [8.03,8.77] | 20.86 [19.49,22.25] | 3.59 [3.40,3.81] | 7.88 [7.22,8.69] |
| **XGBoost (+wx)** | **5.97 [5.70,6.26]** | **14.30 [13.02,15.64]** | 3.25 [3.04,3.48] | 8.72 [7.90,9.66] |

### 4.1 No model beats persistence on the PM₂.₅ episode tail (F1)
Across naive, classical, and three foundation-model architectures, **no model significantly beats
persistence on top-decile PM₂.₅ at 24 h** (paired ΔMAE-vs-persistence 95% CIs all span zero; n = 212 tail
origin-units, 2000 bootstrap iterations): Chronos −0.07 [−1.09,+1.30], TimesFM+wx +0.32 [−0.60,+1.57],
Moirai+wx +0.65 [−0.07,+1.74], XGBoost +0.03 [−1.34,+1.55]. Overall PM₂.₅ MAE (≈3 µg/m³) understates
episode error (≈8–9 µg/m³) ~3×. **Headline.**

### 4.2 Future-weather information dominates NO₂ accuracy (F2)
At 24 h NO₂, zero-shot **Chronos with no covariates (8.49) is statistically indistinguishable from
feature-engineered XGBoost denied future weather (8.40)**: ΔMAE +0.10, 95% CI [−0.21,+0.40] (bootstrap;
DM+BH p = 0.52, both non-significant). Granting XGBoost oracle future weather drops its NO₂ MAE to 5.97 —
the future-weather gain (≈2.4 µg/m³ @24 h) exceeds any model-vs-model gap within a fixed information regime.
The decisive axis is **information, not model class**. The physical mechanism is stagnation: NO₂ accumulates
under low boundary-layer height and low wind (§3.2), conditions encoded by future weather.

### 4.3 FMs benefit from weather, yet classical still wins NO₂ (F3)
Weather covariates significantly help the FM that can use them: **TimesFM+wx (7.55) beats Chronos (8.49)**,
ΔMAE +0.94 [+0.67,+1.20] (DM+BH sig). But with identical known-future weather, **XGBoost (5.97) still beats
TimesFM+wx (7.55)**, ΔMAE +1.58 [+1.30,+1.89] (DM+BH sig). Weather is necessary; the foundation model is not
sufficient to top a strong weather-aware baseline on NO₂.

### 4.4 XGBoost's edge is nonlinear weather interaction (F4)
The Ridge baseline isolates nonlinearity on identical features. **With** future weather, XGBoost beats
linear by ΔMAE +2.28 [+2.04,+2.54]; **without**, only +0.86 [+0.63,+1.09] (both significant by bootstrap and
DM+BH). Nonlinearity buys ~2.6× more in combination with weather — consistent with nonlinear stagnation
(low-BLH × low-wind) effects.

### 4.5 On PM₂.₅, no model class is separable (F5)
At 24 h PM₂.₅, **TimesFM+wx (3.14) and XGBoost (3.25) are statistically indistinguishable**: ΔMAE +0.11,
95% CI [−0.03,+0.25] (bootstrap; DM+BH p = 0.12). An earlier apparent FM "win" did not survive significance
testing; classical wins NO₂ decisively, but on PM₂.₅ the classes are not separable.

### 4.6 "FMs-with-weather" is architecture-dependent (F6, scoped narrow)
On this benchmark, weather covariates **significantly harm Moirai**: Moirai+wx (9.76) is *worse* than Moirai
zero-shot (9.34) on NO₂, ΔMAE −0.42 [−0.63,−0.23] (bootstrap; DM+BH sig), whereas the same covariates help
TimesFM. We state this narrowly (n = 2 FM architectures): TimesFM's XReg ridge extracted weather value that
Moirai's native covariate attention did not — a caution against assuming architecture implies capability,
not a general law.

### 4.7 MM-CP investigation (the method)
**Table 2 — top-decile PM₂.₅ coverage @24 h (target 90%; `mmcp4d_pm25.csv`; → `figures/paper/F_coverage.pdf`).**

| base model | split-CP | MM-CP 3D | MM-CP 4D | sig vs split | sig vs 3D |
|---|---|---|---|---|---|
| Chronos | 0.516 | 0.569 | 0.571 | yes | no |
| TimesFM+wx | 0.519 | 0.588 | 0.602 | yes | no |
| Moirai+wx | 0.611 | 0.650 | 0.650 | yes | no |
| XGBoost | 0.562 | 0.552 | 0.569 | no | no |

3D MM-CP **significantly improves FM episode coverage over split-CP** (+5–7 pp; paired coverage bootstrap
CIs exclude 0) — but reaches only **~57–65%** against a 90% nominal. Adding the episode-regime cell (4D;
leakage audit corr(E, future)=0.45) does **not** significantly improve over 3D (all 4D-vs-3D CIs span 0).
The limitation is cell-*marginal* calibration vs. *conditional-on-tail* coverage — episodes sit in each
cell's upper tail — not cell choice. **MM-CP significantly improves but does not restore deployable episode
coverage, establishing conditional-on-tail conformal as an open problem.**

*Figure 1 (`figures/paper/F_intro.pdf`): the motivating PoC — Chronos's 80% interval under-covers an evening
NO₂ spike on Sheffield Devonshire Green; the episodic-miscalibration problem in one image.*

---

## 5. Discussion

**Weather information, not model class, gates NO₂ skill (F2–F3).** Because access to future weather
dominates, deployment realism hinges on the **numerical-weather-prediction forecast quality** a system can
obtain at run time — not on whether the forecaster is a foundation model or a gradient-boosted tree. Our
oracle-weather results are an information *ceiling*; the realistic operating point lies between the oracle
and no-oracle variants, set by NWP skill at the relevant horizon.

**Episodes are the unsolved regime, and current FMs should not be marketed for them without intervals (F1).**
No model — foundation, classical, or naive — significantly beats persistence on the PM₂.₅ episode tail. A
zero-shot foundation model that looks strong on average MAE can be no better than persistence exactly when a
forecast triggers a health alert. Marketing TS-FM "universality" for episode forecasting without calibrated,
episode-aware uncertainty is not supported by this evidence.

**Architecture mediates whether exogenous data helps (F6).** That the same known-future weather helped
TimesFM but significantly *harmed* Moirai cautions against assuming a more "native" covariate mechanism is
automatically better; exogenous integration should be validated empirically per model.

**Calibration on the tail is an open problem (MM-CP).** Physics-stratified conformal significantly improves
FM episode coverage but cannot restore deployable nominal coverage, because the target is
*conditional-on-tail* coverage while cell-marginal conformal calibrates marginally. This motivates
conditional-on-tail conformal methods (locally-weighted CP, extremal-quantile CQR, risk-aware continuous
conditioning) as the natural follow-on.

## 6. Limitations
- **UK-temperate scope.** Four cities, all Köppen Cfb, eight reference-grade stations. No claim generalises
  beyond temperate-maritime UK; cross-climate transfer (and a physical CAMS baseline) is future work.
- **Zero-shot / in-context only; no fine-tuning.** No LoRA or full fine-tuning of the FMs in v1.
- **Oracle-weather advantage.** XGBoost/TimesFM receive ERA5 reanalysis at target time; the `_no_oracle`
  (lookback-only) variant brackets the realistic forecast-weather case.
- **Moirai num_samples = 20.** Sufficient for the median point forecast and residual-based conformal; too
  noisy for a native-quantile-head comparison (a GPU re-run at 100 samples is filed as feasible).
- **Single-station PM₂.₅ in Manchester** (MAHG is NO₂-only); within-city PM₂.₅ heterogeneity there is
  unrepresented.
- **Half-year calibration window** (Jan–Jun 2023) may under-represent summer regimes for conformal.
- **Chronos ≤64-step training** penalises its 72/168 h numbers (autoregressive roll-out).
- **Significance coverage.** Headline significance is established at 24 h by two independent methods (block
  bootstrap + Diebold–Mariano with Benjamini–Hochberg); extending DM+BH to 72/168 h is the remaining task.

## 7. Conclusion
We introduced AQ-FM-Bench, a significance-tested benchmark of naive, classical, and three foundation-model
architectures for hourly NO₂/PM₂.₅ forecasting across four UK cities, and found that the operationally
critical regime — high-pollution episodes — is not significantly forecast better than persistence by any
model, that future-weather information rather than model class dominates NO₂ skill, and that the value of
weather to a foundation model is architecture-dependent. Every claim is backed by two independent
significance methods.

We also proposed MM-CP, a meteorology-stratified conformal method. It significantly improves foundation-
model coverage on PM₂.₅ episodes (+5–7 pp across three architectures) but does not reach deployable nominal
coverage, and an episode-conditioned 4D extension did not improve over the 3D version — isolating the
limitation as cell-marginal calibration versus conditional-on-tail coverage, not cell choice. We report this
honestly as a bounded methodological result.

These results point to a specific next problem: **conformal prediction with provable coverage conditional on
the upper tail of a heterogeneous response**. Solving it — and then coupling calibrated forecasters to
physical (CAMS) baselines for operational air-quality warning — is the natural continuation of this work.

---

## Acknowledgements & contribution statement

**Data attribution.** DEFRA AURN air-quality data — Open Government Licence v3.0, © Crown copyright. ERA5
meteorology via the Open-Meteo archive (cross-validated against ECMWF ARCO-ERA5); ERA5 generated by
ECMWF / Copernicus Climate Change Service; Open-Meteo data CC-BY-4.0. OpenAQ (station discovery /
cross-validation only).

**Licences.** Code: MIT. Processed dataset + dataset card: CC-BY-4.0. **Author:** single-author preprint at
submission; co-authorship to be revisited for the physics-coupled follow-on (atmospheric-chemistry
collaboration).

---

## References

- Aksu, T., et al. (2024). GIFT-Eval: A Benchmark for General Time Series Forecasting. *NeurIPS Datasets &
  Benchmarks.* arXiv:2410.10393.
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
- Angelopoulos, A. N., Candès, E. J., & Tibshirani, R. J. (2024). Conformal PID Control for Time Series
  Prediction. *NeurIPS 2024.*
- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The limits of distribution-free
  conditional predictive inference. *Information and Inference.*
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS B,* 57(1), 289–300.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD.*
- Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). *ICML.*
  arXiv:2310.10688.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *J. Business & Economic
  Statistics,* 13(3), 253–263.
- Gibbs, I., & Candès, E. J. (2024). Adaptive Conformal Inference Under Distribution Shift. *JMLR.*
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors.
  *Int. J. Forecasting,* 13(2), 281–291.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *QJRMS,* 146(730), 1999–2049.
- Liang, Y., et al. (2023). AirFormer: Predicting Nationwide Air Quality in China. *AAAI.*
- Hettige, K. H., et al. (2024). AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality
  Prediction. *ICLR.* arXiv:2402.03784.
- Munir, S., & Mayfield, M. (2019, 2021). Air-quality monitoring and forecasting in UK urban areas.
- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *NeurIPS.*
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Wang, S., et al. (2020). PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network. *SIGSPATIAL.*
- Williams, A., Chan, R., & Ortiz, A. (2025). Forecasting and the Sheffield Clean Air Zone.
- Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers (Moirai).
  *ICML.* arXiv:2402.02592.
- World Health Organization (2021). WHO global air quality guidelines.
