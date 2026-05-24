# MM-CP — Meteorology-Mondrian Conformal Prediction

A meteorology-stratified conformal method introduced in this work.

## Motivation
Air-quality forecast errors are strongly heteroscedastic, and the heteroscedasticity is physically
governed: errors are systematically larger during **atmospheric stagnation** (low boundary-layer height
+ low wind speed), when pollutants accumulate and standard forecasters fail. Vanilla split-conformal uses
a single residual quantile across all conditions, so it **under-covers during stagnation** (intervals too
narrow exactly when forecasts matter for public health) and **over-covers in well-mixed conditions**
(intervals needlessly wide). MM-CP calibrates within physically homogeneous regimes.

## Method (3D)
MM-CP is a Mondrian conformal predictor (Vovk et al., 2005) whose taxonomy is defined by meteorology
rather than by labels. The nonconformity score is the absolute residual `r = |y − ŷ|`. The cell function
`κ(t)` uses covariates available at forecast time:

- `ws_bin` ∈ {low, mid, high} — terciles of 10 m wind speed `√(u10²+v10²)`, cut on the calibration set
- `blh_bin` ∈ {low, mid, high} — terciles of boundary-layer height, cut on the calibration set
- `hod_bin` ∈ {0–5, 6–11, 12–17, 18–23} — diurnal bucket

giving **3 × 3 × 4 = 36 cells**. For each cell *c*, the conformal quantile of its calibration residuals is

```
q̂_c = Quantile_{i: κ(i)=c}( |y_i − ŷ_i| ;  ⌈(n_c + 1)(1 − α)⌉ / n_c )
```

and the test interval for a point at time *t* is **[ ŷ_t − q̂_{κ(t)},  ŷ_t + q̂_{κ(t)} ]**.

**Small-cell backoff:** if `n_c < n_min` (default 50), fall back hierarchically — first to the
`ws_bin × blh_bin` marginal (ignore hour), then to the global split-CP quantile. This bounds variance in
sparse cells while preserving the guarantee where data is sufficient.

## Guarantee
Under exchangeability *within* each cell, Mondrian CP gives ≥ 1−α coverage **per cell** (conditional
coverage), not merely marginally. The contribution is the physically-motivated taxonomy: cells are chosen
so residuals are approximately homoscedastic inside each, targeting the stagnation regime where marginal
split-CP mis-calibrates.

## 4D extension: episode-conditioned MM-CP
3D MM-CP (wind × BLH × hour) significantly improves FM top-decile PM₂.₅ coverage over split-CP (+5–7 pp;
paired-bootstrap CIs exclude 0) but reaches only ~57–65%. The cause is that top-decile coverage is
*conditional-on-high*, which cell-*marginal* calibration does not target — episodes sit in each cell's own
upper tail. The 4D variant adds an episode-regime dimension so episodes calibrate against episode residuals.

**Episode-regime feature `E(t₀)`**, computed at/before the forecast origin (no future leakage). For origin
`t₀`, lookback `W = [t₀−23, t₀]` using only data ≤ `t₀`:

- `cbar` = mean observed (forward-filled) concentration over `W`.
- `n_stag` = number of hours `h ∈ W` with `BLH[h] < blh_lo` **and** `wind[h] < ws_lo`.
- Cuts fit on **calibration only**: `C3` = ⅔-quantile of `cbar` across calibration origins; `blh_lo`,
  `ws_lo` = ⅓-quantiles of BLH and wind across calibration points.
- **`E(t₀) = 1`** (episode/stagnation regime) if (`cbar ≥ C3`) **or** (`n_stag ≥ 6`); else `0`. Constant
  across the forecast's `H` steps.

**Leakage audit:** `corr(E(t₀), mean future concentration over [t₀+1, t₀+24])` across test origins should
be positive but not ≈ 1 (correlated — episodes persist — not deterministic). Measured: 0.45.

**4D cell taxonomy:** `cell = wind_bin(3) × BLH_bin(3) × hour_bin(2: day 06–18 / night) × regime(2)` =
**36 cells** (hour coarsened 4→2 to keep cells populated when adding the regime split — same total as 3D).
Backoff: 4D → drop regime → wind×BLH → global. Per-cell population counts are reported alongside coverage.

## Evaluation
- Mondrian-stratified coverage and mean interval width per cell, highlighting the stagnation cell
  (low-ws × low-blh).
- Marginal coverage at 50/80/90%; Winkler score at 90%; mean width.
- **Top-decile coverage** — coverage restricted to the top-decile PM₂.₅ episodes (links to F1).
- **Significance:** a paired block bootstrap on *coverage* (not MAE) — per test point, an indicator that
  its interval covers the truth; the coverage difference (MM-CP vs split-CP, and 4D vs 3D) is reported with
  a 95% CI from resampling origins 2000×.

## Result
3D MM-CP significantly improves FM episode coverage over split-CP but does not restore deployable nominal
coverage (~57–65% vs the 90% target); the 4D extension does not significantly improve over 3D. This
isolates the limitation as cell-marginal vs conditional-on-tail coverage, and establishes
conditional-on-tail conformal prediction as an open problem.

## Implementation
Requires ERA5 `u10, v10, blh` joined to each station-hour (already in the processed schema). Implemented in
`src/aq_fm_bench/conformal/mm_cp.py`, exposing the same `wrap(model, calibration_set) → conformal_model`
API as split-conformal. Cell cut-points are fit on the calibration set only (no leakage) and persisted with
the model. A unit test checks per-cell coverage on synthetic heteroscedastic data.

## Constraint
Moirai runs at `num_samples = 20` — sufficient for median-based MAE and residual-based MM-CP, but too noisy
for a comparison against the model's own native quantile head.
