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
python ./scripts/main.py --n-bootstrap 5
```

This runs the full 7‑step pipeline and writes a timestamped directory under `data/output/`,
e.g. `data/output/pipeline_YYYYMMDD_HHMMSS/`.

A reference output directory is included in this repo under:
`data/output/pipeline_20260303_105544/`.


## Inputs

### Default input (recommended): pre‑processed CSV

By default, the pipeline reads:

- `data/input/Maddison_CRU_dataset.csv`

An alternative World Bank–based panel, `data/input/WorldBank_CRU_dataset.csv`, can be used via the
`--use-csv` flag on any analysis script (see "Building the merged datasets" below).

Expected columns (additional columns are ignored):
- `iso_id` (ISO3 country code)
- `year` (int)
- `pcGDP` (per‑capita GDP level)
- `growth_pcGDP` (log difference of per‑capita GDP)
- `temp` (annual mean temperature in °C)
- `Pop` (population; currently not used in estimation)

**Note:** The repository may include other columns (e.g., precipitation, legacy time indices).
The analysis in `src/` currently uses temperature only.

### Building the merged datasets from GDP + CRU inputs

`scripts/create_climate_gdp_dataset.py` builds a merged climate–GDP panel from a chosen GDP
source plus CRU climate. There are two canonical outputs:

| dataset | GDP source | units | span | countries |
|---|---|---|---|---|
| `Maddison_CRU_dataset.csv` | Maddison Project GDPpc | per-capita, PPP (constant int'l $) | 1961–2022 | 145 |
| `WorldBank_CRU_dataset.csv` | World Bank `NY.GDP.PCAP.KD` | per-capita, real (constant 2015 US$) | 1961–2022 | 190 |

Select the source with `--gdp-source` (default `maddison`):

```bash
# Maddison — balanced panel (full 1961→2022 record); default source
python scripts/create_climate_gdp_dataset.py \
  --gdp-source maddison --country-filter endpoints \
  --output data/input/Maddison_CRU_dataset.csv

# World Bank — unbalanced panel (all available country-years)
python scripts/create_climate_gdp_dataset.py \
  --gdp-source worldbank --country-filter none \
  --output data/input/WorldBank_CRU_dataset.csv
```

**World Bank source.** Reads the DataBank wide CSVs in `data/input/` (default: auto-detect the
`NY.GDP.*` GDP file and the `SP.POP.*` population file by series code among `WB_*_Data.csv`;
overridable with `--wb-gdp` / `--wb-pop`). Region/income aggregate rows (World, High income,
Sub-Saharan Africa, …) are dropped by keeping only real ISO-3166 country codes. Per-capita series
(`NY.GDP.PCAP.*`) are used directly; total-GDP series (`NY.GDP.MKTP.*`) are divided by population
automatically. Note: World Bank PPP series (`*.PP.*`) only start ~1990, which is why the per-capita
constant-US$ series `NY.GDP.PCAP.KD` (1960+) is used for the long panel.

**Country filtering** (`--country-filter`, default `none`):
- `none` — keep every available country-year (unbalanced; missing cells simply absent).
- `nearly-all` — keep countries with data in at least `--min-years` (default 30) years
  (or `--min-frac` of the span).
- `contiguous` — keep each country's longest contiguous run of years (after ≤`--max-gap` gap infill).
- `endpoints` — keep only countries present in both the first and last panel year (the balanced
  "1961→2022" rule); used for the canonical Maddison dataset.

**Territory exclusion** (`--exclude-iso`, default `BMU GRL HKG`): dependent territories (Bermuda,
Greenland, Hong Kong) that are valid ISO-3166 codes but not sovereign countries are excluded so the
panel is countries only. Pass `--exclude-iso` with no values to keep them.

**CRU country-name mapping.** CRU climate is keyed by country name and mapped to ISO3 via
`map_cru_country_to_iso3` + `CRU_COUNTRY_OVERRIDES` (`src/data_loader.py`). The overrides handle CRU
spelling/abbreviation variants (e.g. `Turkey`→TUR, `DR Congo`→COD, `Bosnia-Herzegovinia`→BIH,
`St Lucia`→LCA) that pycountry does not resolve; without them these real countries would be silently
dropped at the climate merge. North Korea (PRK) is in CRU but has no World Bank GDP, so it is absent
from the World Bank panel.

**Panel note.** The analysis loader (`load_data_from_csv`) uses an *unbalanced* panel: every country
in the CSV participates with whatever years it has (there is no requirement to span the full period at
load time). The balanced-panel choice is made at dataset-build time via `--country-filter endpoints`.

Other options: gap filling (`--max-gap`, ≤N-year interior interpolation via constant growth),
output validation (`--validate`), and a temperature permutation for synthetic null checks (`--randomT`).


## Running the pipeline

### Full pipeline (recommended)

```bash
python scripts/main.py
```

Key options:

- `--n-bootstrap N` (default 1000)
- `--random-seed SEED` (default 42)
- `--sample-countries-only`
  only resample countries in bootstrap (skip year resampling; by default both are sampled)
- `--loess-window YEARS`  
  LOESS bandwidth for approaches that use LOESS detrending
- `--output-dir PATH`  
  explicit output directory instead of a timestamped one

Example (fast, reproducible):

```bash
python scripts/main.py --n-bootstrap 50 --random-seed 42 --output-dir data/output/test_run
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
- **Segmented linear (S\*):** different slopes below vs above `T_opt` (V‑shaped response)
- **Ternary (T\*):** sum of three quadratic blocks centered on a shared `T_opt`: a growth block (`h₂_G·q`), a decay block (`h₂_D·(q − h₄·A_q,lag − corr_q)`) with persistence parameter `h₄`, and a level block (`h₂_L·(q − q_{t-1})`), where `q = (T − T_opt)²`
- **Persistence/decay (D\*):** a distributed‑lag “converging” response with decay parameter `h₄`
- **Level effect (L\*):** first‑difference of the climate response (`h₄ = 1`), testing whether temperature affects GDP *levels* rather than *growth*
- **GDP‑scaled quadratic (G\* / C\*):** the climate response is multiplied by a per‑capita‑GDP scaling factor `g = (pcGDP / Y_ref)^(−β)`, where `Y_ref = median(pcGDP)` and the exponent `β` is freely estimated. Two forms: **G** = free quadratic `g·(β₀ + β₁·T + β₂·T²)`; **C** = centered quadratic `g·β₂·(T − T_opt)²`. Larger `β` means poorer countries (lower `pcGDP`) are more temperature‑sensitive.
- **Log‑linear GDP‑dependent quadratic (I\*):** a *different* model of income dependence — the response is scaled by `s̃ = 1 − log(pcGDP/Y_ref)/β` instead of the power law. Because `log(pcGDP/Y_ref)` is bounded, `s̃` never explodes, so this form is identifiable on *pre‑detrended residuals* where the power‑law scaling degenerates. Response magnitude declines linearly with log‑income and crosses zero at `pcGDP = Y_ref·e^β`; larger `β` ⇒ weaker income dependence (`β → ∞` recovers the plain quadratic). Here `β` is a log‑income crossover, **not** comparable to the power‑law elasticity.

### Approaches implemented

The code reports results under short labels:

- **QJ**: Quadratic climate response, *joint* OLS estimation of `h`, `jᵢ(t)`, and `k(t)`
- **QP**: Quadratic response, *polynomial detrending* (precompute `k(t)` and country trends; linear T trend)
- **QL**: Quadratic response, *LOESS detrending* (precompute `k(t)` and LOESS trends)
- **PJ / PP / PL**: Piecewise quadratic variants (joint / polynomial / LOESS detrending)
- **SJ / SP / SL**: Segmented linear variants (joint / polynomial / LOESS detrending)
- **TJ / TP / TL**: Three‑interval variants (joint / polynomial / LOESS detrending)
- **DJ / DP / DL**: Persistence/decay variants (joint / polynomial / LOESS detrending)
- **LJ / LL**: Level effect variants (joint / LOESS detrending)
- **GJ**: GDP‑scaled *free* quadratic response `g·(β₀+β₁T+β₂T²)`, *joint* OLS (β and the quadratic coefficients estimated together)
- **CJ**: GDP‑scaled *centered* quadratic response `g·β₂·(T−T_opt)²`, *joint* OLS (β and `T_opt` profiled jointly)
- **IP / IL**: log‑linear GDP‑dependent quadratic `s̃·(β₀+β₁T+β₂T²)`, `s̃ = 1 − log(pcGDP/Y_ref)/β`, on polynomial‑ / LOESS‑detrended residuals (the P/L analogs of GJ, but with the well‑behaved log‑linear scale). β is profiled; coefficients are the response at `Y_ref` (`s̃ = 1`). Both land at an interior β (≈1.6 on the default data).
- **IJ**: the *joint* version of the log‑linear form. Provided for comparison, but log‑linear GDP‑dependence is **not identified in the joint specification** — β drifts to its bound and IJ collapses to ≈QJ. (Power‑law dependence is the reverse: identified jointly as GJ, but degenerate on residuals.)
- **NJ / NP / NL**: “null” variants with **no climate response** (only `jᵢ(t)` and `k(t)`)

All approaches are defined precisely in **`METHODS_DETAIL.md`**.


### Caveat: income-dependence of the response is confounded with temperature

The GDP-scaled families (G\* / C\* / R\* / M\* / W\* and the log-linear I\*) let the temperature
response depend on a country's income. **Treat these as exploratory**: across countries, income and
temperature level are strongly confounded — richer countries are systematically cooler
(`corr(mean temperature, mean log pcGDP) ≈ −0.50`) — so "poorer countries are more temperature-sensitive"
is nearly indistinguishable from "hotter countries are more temperature-sensitive."

A standalone diagnostic makes this explicit:

```bash
python scripts/run_two_stage_response.py            # runs both detrending methods
# --method {poly,loess,both}   Stage-1 detrending (default: both)
# --loess-window YEARS         LOESS bandwidth (default: 42.45, the repo-wide value)
```

It estimates each country's *own* temperature sensitivity `β₁ᵢ` (Stage 1) and then asks what explains
`β₁ᵢ` across countries (Stage 2: on mean `log(pcGDP)` and mean temperature). Stage 1 is available with
two detrendings, which give essentially identical results (per-country `β₁ᵢ` correlate at 0.99):

- **polynomial** — `Δy = β₁ᵢ·T + j0ᵢ + j1ᵢ·t + j2ᵢ·t²` per country;
- **LOESS** — LOESS-detrend each country's growth and temperature (bandwidth in years), then fit
  `y_resid = β0 + β1ᵢ·T_resid`.

Findings on the default dataset (both detrendings):

- Income and mean temperature each correlate with `β₁ᵢ` on their own, but when **both** are included
  neither survives cleanly (each drops to |t| ≈ 1.9 precision-weighted, and to insignificance unweighted) —
  they are tracing the same rich-cool gradient.
- The `β₁ᵢ`-vs-mean-temperature slope simply recovers `2·β₂` from the pooled quadratic (`QJ` for
  polynomial, `QL` for LOESS), i.e. it is the shape of a *single* global response, not evidence of
  heterogeneity.
- The income–temperature plane plot (`data/output/two_stage/slope_in_gdp_temp_plane*.pdf`) shows the cloud
  lying on a rich-cool / poor-hot ridge with the rich-and-hot corner essentially empty — so there is little
  off-diagonal variation with which to separate income from temperature.

Because the GDP-scaled point estimates draw ~84% of their identifying variation from this confounded
between-country dimension, their `β` should be read as reflecting the income–temperature confound rather
than a cleanly identified income effect. See `scripts/run_two_stage_response.py` / `src/two_stage_response.py`.


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
- Bootstrap uncertainty depends on the random seed and whether year resampling is enabled (default: yes; use `--sample-countries-only` to disable).
- LOESS‑based results depend on `--loess-window`.


## References

The code implements variants of a temperature–growth panel regression widely used in the climate‑econometrics literature (often associated with Burke et al.). This repository is intended as a **methods sandbox** for examining how detrending choices affect the estimated temperature response.


## Running a subset of approaches

Use `--approaches` to run only specific approach codes (first letter = response type, second = trend method):

```bash
python ./scripts/main.py --approaches QJ LJ DJ QL LL DL \
  --use-csv ./data/input/ACCESS-ESM1-5_historical.csv \
  --n-bootstrap 100 --start-year 1960 \
  --output-dir ./data/output/pipeline_ACCESS-ESM1-5
```

This runs only the Quadratic/Level/Decay response types with Polynomial and LOESS trend identification.


### Example: biogeochemistry model runs

```bash
python ./scripts/main.py --output-dir ./data/output/pipeline_ACCESS-ESM1-5_2026-03-05_1000 \
  --use-csv ./data/input/ACCESS-ESM1-5_historical.csv --n-bootstrap 1000 --start-year 1960

python ./scripts/main.py --output-dir ./data/output/pipeline_CNRM-ESM2-1_2026-03-05_1000 \
  --use-csv ./data/input/CNRM-ESM2-1_historical.csv --n-bootstrap 1000 --start-year 1960

python ./scripts/main.py --output-dir ./data/output/pipeline_MIROC-ES2L_2026-03-05_1000 \
  --use-csv ./data/input/MIROC-ES2L_historical.csv --n-bootstrap 1000 --start-year 1960
```
