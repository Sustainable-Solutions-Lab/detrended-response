# Detrended Response: Temperature–Growth Regression with Explicit Detrending

## Overview

This repository contains a research pipeline for estimating and stress‑testing *temperature response functions* in a country–year panel regression of per‑capita GDP growth on annual mean temperature.

The core objective is methodological: to make explicit what the common **country‑specific quadratic time trend** term in “Burke‑style” temperature–growth regressions is doing (i.e., what kind of *detrending* it implies), and to compare several alternative (but closely related) ways of separating:

- **short‑run temperature variation** (“departures from a smooth temperature trajectory”), and
- **slowly varying components** of growth (country trends) and global shocks (year effects).

The pipeline produces:
- point estimates for multiple “approaches” (alternative detrending/response specifications),
- bootstrap uncertainty estimates (cluster bootstrap by country, with optional year resampling),
- publication tables/figures,
- cumulative impact diagnostics, and
- an influence analysis identifying countries that systematically move coefficient estimates.

For a paper‑style, detailed description of the statistical methods, see **`METHODS_DETAIL.md`**.


## Quick start

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the included quick test (matches the command in the prompt):

```bash
python ./scripts/main.py --sample-years --n-bootstrap 5
```

This runs the full 7‑step pipeline and writes a timestamped directory under `data/output/`,
e.g. `data/output/pipeline_YYYYMMDD_HHMMSS/`.

A reference output directory is included in this repo under:
`data/output/pipeline_20260303_105544/`.


## Inputs

### Default input (recommended): pre‑processed CSV

By default, the pipeline reads:

- `data/input/Maddison_CRU_dataset.csv`

Expected columns (additional columns are ignored):
- `iso_id` (ISO3 country code)
- `year` (int)
- `pcGDP` (per‑capita GDP level)
- `growth_pcGDP` (log difference of per‑capita GDP)
- `temp` (annual mean temperature in °C)
- `Pop` (population; currently not used in estimation)

**Note:** The repository may include other columns (e.g., precipitation, legacy time indices).
The analysis in `src/` currently uses temperature only.

### Optional: build the merged dataset from Maddison + CRU inputs

If `data/input/mpd2023_web.xlsx` and `data/input/cru_climate_data.csv` are present, you can
(re)generate the merged CSV:

```bash
python scripts/create_Maddison_CRU_dataset.py \
  --maddison data/input/mpd2023_web.xlsx \
  --cru data/input/cru_climate_data.csv \
  --output data/input/Maddison_CRU_dataset.csv
```

Options include gap filling (`--max-gap`), output validation (`--validate`), and an optional
temperature permutation for synthetic null checks (`--randomT`).


## Running the pipeline

### Full pipeline (recommended)

```bash
python scripts/main.py
```

Key options:

- `--n-bootstrap N` (default 1000)
- `--random-seed SEED` (default 42)
- `--sample-years`  
  additionally resamples years with replacement and fits via weighted least squares
- `--loess-window YEARS`  
  LOESS bandwidth for approaches that use LOESS detrending
- `--output-dir PATH`  
  explicit output directory instead of a timestamped one

Example (fast, reproducible):

```bash
python scripts/main.py --n-bootstrap 50 --random-seed 42 --sample-years --output-dir data/output/test_run
```

### Individual steps

You can run any step in isolation:

- **Point estimation:** `python scripts/run_analysis.py`
- **Bootstrap:** `python scripts/run_bootstrap.py`
- **Publication tables/figures:** `python scripts/make_tables_and_figures.py`
- **Bootstrap uncertainty summary:** `python scripts/summarize_bootstrap_uncertainty.py`
- **QJ vs QP parameter comparison:** `python scripts/compare_Approach1J_Approach1P.py`
- **Cumulative effects:** `python scripts/calculate_cumulative_effects.py`
- **Influence analysis:** `python scripts/run_influence_analysis.py`


## Methods summary (high level)

### Data and dependent variable

Let country be indexed by *i* and year by *t*. The dependent variable is:

```
Δyᵢ,t = log(pcGDPᵢ,t) − log(pcGDPᵢ,t−1)
```

The climate variable is annual mean temperature `Tᵢ,t` (°C).

### Baseline model structure

Most approaches share the following additive decomposition:

```
Δyᵢ,t = climate_responseᵢ,t + jᵢ(t) + k(t) + εᵢ,t
```

- `jᵢ(t)`: a smooth, country‑specific growth component (trend)
- `k(t)`: year effects common across countries (global shocks)
- `εᵢ,t`: residual

The “climate response” is specified in several ways:

- **Quadratic (Q\*):** `h(T) = h₁·T + h₂·T²`
- **Piecewise quadratic (P\*):** different curvature below vs above an estimated optimum `T_opt`
- **Persistence/decay (D\*):** a distributed‑lag “converging” response with decay parameter `h₄`

### Approaches implemented

The code reports results under short labels:

- **QJ**: Quadratic climate response, *joint* OLS estimation of `h`, `jᵢ(t)`, and `k(t)`
- **QP**: Quadratic response, *polynomial detrending* (precompute `k(t)` and country trends; linear T trend)
- **QL**: Quadratic response, *LOESS detrending* (precompute `k(t)` and LOESS trends)
- **PJ / PP / PL**: Piecewise quadratic variants (joint / polynomial / LOESS detrending)
- **DJ / DP / DL**: Persistence/decay variants (joint / polynomial / LOESS detrending)
- **NJ / NP / NL**: “null” variants with **no climate response** (only `jᵢ(t)` and `k(t)`)

All approaches are defined precisely in **`METHODS_DETAIL.md`**.


## Outputs

The pipeline writes a parent directory with subfolders:

- `analysis/`  
  point estimates, diagnostic plots, coefficient tables
- `bootstrap/`  
  bootstrap coefficient samples, percentile summaries, and plots
- `publication/`  
  publication‑formatted tables and figures
- `comparison/`  
  QJ vs QP parameter comparison figure/tables
- `cumulative/`  
  cumulative climate effect calculations and plots
- `influence/`  
  influence regressions and ranked country lists

Each step also writes `run_metadata.json` capturing key options and inputs.


## Project structure (selected)

- `src/`  
  analysis code (loading, detrending, fitting, bootstrap, output utilities)
- `scripts/`  
  CLI entry points (pipeline and standalone steps)
- `data/input/`  
  input datasets
- `data/output/`  
  outputs (including a reference run)


## Reproducibility notes

- The analysis is deterministic given the same inputs and `--random-seed`.
- Bootstrap uncertainty depends on the random seed and whether `--sample-years` is enabled.
- LOESS‑based results depend on `--loess-window`.


## References

The code implements variants of a temperature–growth panel regression widely used in the climate‑econometrics literature (often associated with Burke et al.). This repository is intended as a **methods sandbox** for examining how detrending choices affect the estimated temperature response.


## Running a subset of approaches

Use `--approaches` to run only specific approach codes (first letter = response type, second = trend method):

```bash
python ./scripts/main.py --approaches QP LP DP QL LL DL \
  --use-csv ./data/input/ACCESS-ESM1-5_historical.csv \
  --sample-years --n-bootstrap 1000 --start-year 1960 \
  --output-dir ./data/output/pipeline_ACCESS-ESM1-5_subset
```

This runs only the Quadratic/Level/Decay response types with Polynomial and LOESS trend identification.


# example biogeochemistry full run
python ./scripts/main.py --output-dir ./data/output/pipeline_ACCESS-ESM1-5_2026-03-05_1000 --use-csv ./data/input/ACCESS-ESM1-5_historical.csv --sample-years --n-bootstrap 1000 --start-year 1960

python ./scripts/main.py --output-dir ./data/output/pipeline_CNRM-ESM2-1_2026-03-05_1000   --use-csv ./data/input/CNRM-ESM2-1_historical.csv   --sample-years --n-bootstrap 1000 --start-year 1960

python ./scripts/main.py --output-dir ./data/output/pipeline_MIROC-ES2L_2026-03-05_1000    --use-csv ./data/input/MIROC-ES2L_historical.csv    --sample-years --n-bootstrap 1000 --start-year 1960
