# AQ-FM-Bench

**A significance-tested benchmark of time-series foundation models for UK urban air-quality forecasting,
with meteorology-stratified conformal calibration.**

We evaluate naive baselines, classical models (Ridge, XGBoost), and three time-series foundation models
(Chronos-Bolt, TimesFM-2.0, Moirai-1.1) on hourly NO₂ and PM₂.₅ forecasting across four UK cities (eight
reference-grade DEFRA AURN stations), 2019–2024, at 24/72/168-hour horizons, with ERA5 meteorological
covariates, conformal uncertainty, and a deliberate focus on high-pollution episodes. We find that the
dominant factor in NO₂ accuracy is *access to future weather*, not the foundation-vs-classical
distinction — and that on the top decile of PM₂.₅, the episodes that drive health alerts, **no model —
foundation, classical, or naive — is statistically distinguishable from persistence**.

> Scope is **UK-temperate** (all Köppen Cfb). No claim generalises beyond it. Every empirical claim is
> backed by two independent significance methods (block bootstrap + Diebold–Mariano with Benjamini–Hochberg).

## Headline results (24 h, MAE µg/m³ [95% CI])

| model | NO₂ overall | NO₂ episodes (top-decile) | PM₂.₅ overall | PM₂.₅ episodes |
|---|---|---|---|---|
| persistence (naive) | 10.98 [10.39,11.62] | 24.29 [22.07,26.60] | 4.35 [3.93,4.88] | 8.75 [7.56,10.16] |
| Chronos-Bolt (zero-shot FM) | 8.49 [8.10,8.94] | 23.21 [21.63,24.86] | 3.30 [3.10,3.50] | 8.82 [7.95,9.72] |
| TimesFM-2 (FM + weather) | 7.55 [7.19,7.94] | 21.29 [19.93,22.77] | 3.14 [2.95,3.34] | 8.43 [7.66,9.26] |
| **XGBoost (+ oracle weather)** | **5.97 [5.70,6.26]** | **14.30 [13.02,15.64]** | 3.25 [3.04,3.48] | 8.72 [7.90,9.66] |

**F1 (headline):** across naive, classical, and three FM architectures, *no model significantly beats
persistence on top-decile PM₂.₅ at 24 h* — every ΔMAE-vs-persistence 95% CI spans zero (n = 212 tail
origin-units). The episode regime is unsolved. (Full leaderboard + findings F1–F6 in `paper/paper.md`.)

## Quickstart — reproduce F1 in under a minute (no GPU)

The headline finding replicates from cached per-origin errors with no foundation-model downloads:

```bash
git clone https://github.com/nijatbadalov04-alt/pm25-tail-benchmark.git
cd pm25-tail-benchmark
uv venv --python 3.11                  # or: python -m venv .venv
uv pip install -e .                    # core deps only (numpy/pandas/scipy); no GPU/torch needed
uv run python scripts/reproduce_f1.py  # -> "F1 REPRODUCED: no model ... beats persistence"
```

For the **full pipeline** (foundation-model inference, leaderboard, conformal), install the GPU stack
(`requirements-main.txt`, plus `requirements-fm.txt` in a separate `.venv-fm` for Moirai — see comments in
those files) and run, e.g., `scripts/run_significance_leaderboard.py`, `scripts/run_pm25_mmcp4d.py`,
`scripts/run_dm_bh_h24.py`, `scripts/make_paper_figures.py`. All published CSVs reproduce bitwise
(`scripts/verify_results_unchanged.py`).

## Repository layout
```
src/aq_fm_bench/   package: data loaders, models, conformal (split-CP + MM-CP), metrics, stats (bootstrap, DM)
scripts/           pipeline + paper-artefact scripts (leaderboard, significance, MM-CP, figures, replication)
data/processed/    final hourly Parquet (8 stations); train/cal/test marked by a `split` column
data/manifest.csv  source URLs + pull dates (provenance)
paper/paper.md     the manuscript (preprint draft)
figures/paper/     final figures (F_intro, F_leaderboard, F_coverage)
docs/              dataset card, exogenous schema, MM-CP method spec
```

> **On directory layout:** model/city/conformal settings are inlined as code constants (there is no `configs/` YAML tree); the train/calibration/test splits are date constants in `aq_fm_bench.data.processing.SPLITS` (no separate `data/splits/` index files); and the motivating proof-of-concept ships as `scripts/poc_sheffield.py` rather than a `notebooks/` notebook.

## Models
Chronos-Bolt-Base (`amazon/chronos-bolt-base`, no exog), TimesFM-2.0 (`google/timesfm-2.0-500m-pytorch`,
XReg covariates), Moirai-1.1-R-base (`Salesforce/moirai-1.1-R-base`, native future covariates); XGBoost and
Ridge each with/without oracle future weather; persistence and seasonal-naive floors. Single-model-per-process
on an 8 GB GPU (bf16). The conformal contribution **MM-CP** (meteorology-Mondrian) is in
`src/aq_fm_bench/conformal/`.

## Links
- **Paper:** arXiv — *(link on posting)*
- **Dataset:** HuggingFace — *(link on release; card in `docs/DATASET_CARD.md`)*
- **Archived release:** Zenodo DOI — *(minted on the v1.0.0 tag)*

## License
Code: **MIT** (`LICENSE`). Processed dataset: **CC-BY-4.0** (`docs/DATASET_CARD.md`). DEFRA AURN © Crown
copyright (OGL v3.0); ERA5 via Open-Meteo / ECMWF–Copernicus (CC-BY-4.0).

## Citation
```bibtex
@misc{badalov2026aqfmbench,
  title  = {AQ-FM-Bench: A Significance-Tested Benchmark of Time-Series Foundation Models
            for UK Urban Air-Quality Forecasting, with Meteorology-Stratified Conformal Calibration},
  author = {Badalov, Nijat},
  year   = {2026},
  note   = {Preprint; code at https://github.com/nijatbadalov04-alt/pm25-tail-benchmark}
}
```
