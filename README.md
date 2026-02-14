# Detrended Response Analysis

## Overview

The Burke et al. (2015) equation contains a country-specific quadratic time trend. This work explores how this quadratic time trend can be interpreted. To simplify the discussion, we consider the economic response to temperature only, neglecting precipitation responses.

The time trend can be interpreted either as reflecting a linear detrending of the country-level temperature curve, or as a quadratic detrending of the country-level per capita GDP growth-rate curve, or as a combination of these two detrendings. One way of understanding the Burke et al. (2015) equation is that it is relating, at country level, departures from the linear temperature trend to departures from the quadratic GDP growth trend.

This work examines the effect of making this implicit detrending explicit.

## The Model

The Burke et al. (2015) equations, simplified to consider temperature only, can be written as:

```
Δyᵢ(t) = h(Tᵢ(t)) + jᵢ(t) + k(t)
```

where:
```
h(Tᵢ(t)) = h₁·Tᵢ(t) + h₂·Tᵢ(t)²
jᵢ(t) = j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t²
k(t) = kₜ
```

This can be collapsed into:
```
Δyᵢ(t) = h₁·Tᵢ(t) + h₂·Tᵢ(t)² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

**Variables:**
- `Δyᵢ(t)` — per capita GDP growth rate for country i in year t
- `Tᵢ(t)` — annual mean temperature
- `h₁, h₂` — temperature response coefficients
- `jᵢ(t)` — country-specific quadratic time trend
- `kₜ` — year fixed effects

**Optimal temperature** (where growth is maximized): `T_opt = -h₁ / (2·h₂)`

## Interpreting the Time Trend Function jᵢ(t)

The time trend function `jᵢ(t)` can be interpreted as:
1. A linear detrending of the country-level temperature curve
2. A quadratic detrending of the country-level per capita GDP growth-rate curve
3. A combination of both detrendings

If the quadratic time trend is meant to be one or both of these detrending functions, then these detrendings can be applied to the original datasets and the associated parameter values found prior to the main ordinary least squares solution for the climate response coefficients.

## Approaches

### Approach 0: Conjoined OLS Fit

Fit all terms in a single (conjoined) ordinary least squares fitting process. With this approach, it is unclear how much `jᵢ(t)` is acting to detrend temperature versus per capita GDP growth rates, and how much of the error term is being absorbed into these many additional degrees of freedom.

```
Δyᵢ(t) = h₁·T + h₂·T² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

Note: One could add any arbitrary quadratic in time to `k(t)` and subtract the same quadratic from all of the `jᵢ(t)`s. Therefore, three additional constraints must be added. Without loss of generality, we set `j₁(t) = 0` for the first country.

**Degrees of freedom:** 2 for h(T) + 3×(n_countries - 1) for jᵢ(t) + n_years for k(t)

### Approach 1: Combined Detrending (Mixed)

Combines linear temperature detrending with quadratic GDP growth detrending. If the purpose of `jᵢ(t)` is to effect both a linear detrending of the temperature record and a quadratic detrending of the per-capita GDP growth record:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 2: Linear Temperature Detrending

We know that at least `j₀,ᵢ` and `j₁,ᵢ` relate to the temperature scale, because their values would differ if temperature was measured in Celsius versus Kelvin. A natural assumption is that at least part of `jᵢ(t)` is meant to represent a linear detrending of temperature.

Pre-compute `T₀,ᵢ` and `T₁,ᵢ` for each country via least squares on the linear temperature trend, then estimate:

```
Δyᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 3: Quadratic GDP Growth Detrending

Another interpretation of `jᵢ(t)` is that it represents a quadratic detrending of the `Δyᵢ(t)` values.

Pre-compute `y₀,ᵢ`, `y₁,ᵢ`, and `y₂,ᵢ` for each country via least squares on the quadratic per capita GDP-growth trend, then estimate:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·T + h₂·T² + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 4: Combined Quadratic Detrending

Applies quadratic detrending to both per capita GDP growth and temperature. When combined with the 2nd order term in h(T), this results in `jᵢ(t)` becoming a 4th-order equation.

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)]
                                      + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 5: Pre-computed k(t) with Quadratic Trends

In Approaches 0-4, year fixed effects kₜ are estimated simultaneously with the temperature coefficients. An alternative is to pre-compute kₜ as year means before fitting.

1. Pre-compute year effects: `k(t) = mean_i(Δyᵢ(t))`
2. Fit country trends jᵢ(t) = j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² to `Δyᵢ(t) - k(t)`
3. Fit temperature trends T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t² (quadratic)
4. Final regression on residuals:

```
[Δyᵢ(t) - k(t)] - jᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)]
                         + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)²]
```

**Degrees of freedom:** 2 for h(T) (year effects pre-computed, not estimated)

#### Approach 5 Variants

Several variants explore different detrending combinations:

| Variant | GDP Trend | Temp Trend | Description |
|---------|-----------|------------|-------------|
| **5** | Quadratic | Quadratic | Full quadratic detrending |
| **5a** | Quadratic | Linear | Linear temperature only |
| **5b** | Quadratic | None | GDP detrending only |
| **5c** | Quadratic | Linear+Quadratic | Combined approach |
| **5d** | Quadratic | Quadratic + GDP-scaling | GDP-dependent response |

**Approach 5d** introduces GDP-dependent scaling:
```
h(Y,T) = (Y/Y_ref)^(-β) · (h₁·T* + h₂·T*²)
```
where β is the GDP scaling exponent (larger β = stronger income-based adaptation).

### Approach 6: LOESS Detrending

Replaces polynomial detrending with LOESS (Locally Weighted Scatterplot Smoothing), a non-parametric method that allows for more flexible trend shapes.

```
[Δyᵢ(t) - k(t)] - LOESS(Δyᵢ - k) = h₁·[T - LOESS(T)] + h₂·[T - LOESS(T)]²
```

Uses a 25-year LOESS window (configurable via `--loess-window`).

**Key difference from polynomial approaches:** Uses `h(T) - h(T_trend)` formulation where the response function is applied to both actual and trend temperatures, then differenced.

**Degrees of freedom:** 2 for h(T) (year effects pre-computed)

### Approach 6a: Separate Total/Trend Response

Extends Approach 6 by allowing different temperature response functions for the total temperature (T) vs trend temperature (T_trend) components.

```
Δy*(t) = h_total(T) - h_trend(T_trend)
```

where:
- `h_total(T) = h₁_total·T + h₂_total·T²` — response to actual (total) temperature
- `h_trend(T_trend) = h₁_trend·T_trend + h₂_trend·T_trend²` — response to trend temperature
- `Δy*(t) = Δy(t) - k(t) - LOESS(Δy - k)` — same detrending as Approach 6

**Parameters:**
- `h₁_total, h₂_total`: Total temperature response coefficients (response to actual T)
- `h₁_trend, h₂_trend`: Trend temperature response coefficients (response to T_trend)
- `T_optimal_total, T_optimal_trend`: Optimal temperatures for total and trend components

**Total Response:** When T = T_trend (i.e., T_delta = 0), the total climate effect is:
```
h_total(T) - h_trend(T) = (h₁_total - h₁_trend)·T + (h₂_total - h₂_trend)·T²
```

**Interpretation:** If climate adaptation occurs over time, the response to trend temperature changes may differ from the response to total temperature. Approach 6a tests whether economies respond differently to these two components.

**Degrees of freedom:** 4 for h(T) (h₁_total, h₂_total, h₁_trend, h₂_trend)

### Approach 6b: Trend Response Only

A restricted version of Approach 6a that sets the total temperature response to zero, attributing all temperature effects to the trend component only.

```
Δy*(t) = 0 - h_trend(T_trend)
```

where:
- `h_total(T) = 0` (no response to total temperature)
- `h_trend(T_trend) = h₁_trend·T_trend + h₂_trend·T_trend²`

**Interpretation:** Tests the hypothesis that only trend temperature affects economic growth, while deviations from the trend have no systematic effect.

**Degrees of freedom:** 2 for h(T) (h₁_trend, h₂_trend)

### Approach 8: Piecewise Quadratic Response with LOESS

Uses a piecewise quadratic temperature response that allows different curvatures for temperatures above vs below the optimum. This captures the asymmetry where warming may have different effects on hot vs cold countries.

**Model:**
```
h(T) = h₂_high · (T - T_opt)²   if T > T_opt
h(T) = h₂_low · (T - T_opt)²    if T ≤ T_opt
```

**Parameters:**
- `T_opt`: Optimal temperature (breakpoint where h(T) = 0)
- `h₂_low`: Curvature for cold countries (T ≤ T_opt)
- `h₂_high`: Curvature for hot countries (T > T_opt)

**Fitting uses the detrended formulation:**
```
[Δyᵢ(t) - k(t)] - LOESS(Δyᵢ - k) = h(T) - h(T_trend)
```

**Optimization strategy:**
1. Outer optimization: Search for T_opt using L-BFGS-B
2. Inner OLS: For each T_opt, solve for h₂_low and h₂_high via 2-column OLS

**Interpretation:**
- Both h₂_low and h₂_high should be negative (growth decreases away from optimum)
- If h₂_low ≈ h₂_high, model reduces to symmetric quadratic
- If |h₂_high| > |h₂_low|, warming hurts hot countries more than cooling hurts cold countries

**Degrees of freedom:** 3 (T_opt, h₂_low, h₂_high)

### Approach 8a: Separate Total/Trend Response with Shared T_opt

Combines the total/trend separation of Approach 6a with the piecewise quadratic structure of Approach 8. Uses separate curvature parameters for total temperature (T) and trend temperature (T_trend), but with a shared optimal temperature.

```
Δy*(t) = h₂_total·(T - T_opt)² - h₂_trend·(T_trend - T_opt)²
```

where:
- `h₂_total`: Curvature for total (actual) temperature response
- `h₂_trend`: Curvature for trend temperature response
- `T_opt`: Shared optimal temperature for both components

**Total Response:** When T = T_trend (i.e., T_delta = 0), the total climate effect is:
```
(h₂_total - h₂_trend)·(T - T_opt)²
```

**Key features:**
- Both functions share the same optimal temperature
- Linear terms (h₁) are implicitly zero (piecewise quadratic centered at T_opt)
- Allows different sensitivities to total vs trend temperature

**Interpretation:** Tests whether the curvature of the temperature-growth relationship differs between total temperature and trend temperature, while maintaining a common optimal temperature.

**Degrees of freedom:** 3 (T_opt, h₂_total, h₂_trend)

### Approach 8b: Squared-Deviation Modulated Temperature Response

Uses a symmetric modulation where the temperature response is scaled by the squared deviation from trend temperature. This captures effects that depend on the magnitude of deviation from trend, regardless of sign.

**Model:**
```
Δy*(t) = (1 + h₀ · (T - T_trend)²) · (h₁ · T + h₂ · T²)
```

**Parameters:**
- `h₀`: Squared-deviation modulation coefficient
- `h₁, h₂`: Standard quadratic temperature response coefficients
- `T_optimal`: Optimal temperature = -h₁/(2·h₂)

**Interpretation:**
- When T = T_trend, the multiplier is 1 (standard quadratic response)
- Any deviation from trend (warmer or cooler) scales the response by (1 + h₀·T_delta²)
- If h₀ > 0: deviations from trend amplify the temperature response
- If h₀ < 0: deviations from trend dampen the temperature response

**Fitting Strategy:**
- Outer optimization: Search for h₀ using L-BFGS-B
- Inner OLS: For each h₀, solve for h₁ and h₂ via 2-column OLS

**Degrees of freedom:** 3 (h₀, h₁, h₂)

### Null Models (No Climate Response)

Two null models for comparison:

| Model | Description |
|-------|-------------|
| **nocr0** | Joint OLS with country trends and year effects, but h₁=h₂=0 |
| **nocr5** | Precomputed k with country trends, but h₁=h₂=0 |

## Imbalance Metrics

### The Consistency Equation

If the choice of smoothing function and the determination of the climate response function were perfect, the following equation would be satisfied exactly:

```
0 = h(f_trend(Tᵢ(t))) + f_trend(Δyᵢ(t) - k(t)) + k(t)
```

where:
- `h(T) = h₁·T + h₂·T²` is the climate response function
- `f_trend(T)` is the temperature trend (linear, quadratic, or LOESS depending on approach)
- `f_trend(Δy - k)` is the country-specific GDP growth trend (after removing year means)
- `k(t)` is the year mean

In practice, the right side will not sum to zero. The root-mean-square of this imbalance provides a diagnostic metric.

### Three Metrics

For each approach, we compute:

| Metric | Description |
|--------|-------------|
| **RMS Imbalance** | `sqrt(mean((h(T_trend) + j_trend + k)²))` — measures internal consistency |
| **RMS h(T)** | `sqrt(mean(h(T)²))` — magnitude of climate response signal |
| **Imbalance Ratio** | RMS Imbalance / RMS h(T) — normalized imbalance |

### Interpretation

- **Low RMS Imbalance** suggests good internal consistency between the detrending and climate response
- **High RMS Imbalance** indicates a mismatch — the trend function may be absorbing signal that should be attributed to climate, or vice versa

### Important Limitation

**The imbalance metric can be "gamed" by null or weak detrending:**
- If `T_trend = 0` (no temperature detrending), then `h(T_trend) = 0` regardless of h₁ and h₂
- A degenerate case with no trends and a near-zero climate response would minimize the imbalance while explaining nothing meaningful

The **Imbalance Ratio** addresses this limitation by normalizing by the magnitude of the climate response.

## Data Sources

Two data input options are available:

### Option 1: Maddison + CRU (default)
- **GDP**: Maddison Project Database 2023 (Bolt & van Zanden, 2024)
- **Temperature**: CRU CY v4.09 country-level means (Harris et al., 2020)

### Option 2: Pre-processed CSV
- **Maddison_CRU_dataset.csv**: Pre-merged dataset with GDP growth and temperature already computed.

## Data Preparation

### Creating the Maddison/CRU Dataset

The `create_Maddison_CRU_dataset.py` script merges Maddison Project GDP/population data with CRU climate data to create a unified dataset for analysis.

```bash
python scripts/create_Maddison_CRU_dataset.py
```

**What it does:**
1. Loads GDP per capita from Maddison Project Database 2023 (GDPpc sheet)
2. Loads population from Maddison (Population sheet)
3. Fills GDP gaps of up to 4 years using constant growth rate interpolation
4. Loads CRU climate data and maps country names to ISO3 codes
5. Computes GDP growth rate as log difference: `growth = log(GDP_t) - log(GDP_{t-1})`
6. Computes log-transformed precipitation relative to country mean
7. Merges all data and filters to 1961-2022

**Options:**
```
--maddison PATH   Maddison Excel file (default: data/input/mpd2023_web.xlsx)
--cru PATH        CRU CSV file (default: data/input/cru_climate_data.csv)
--output PATH     Output CSV path (default: data/input/Maddison_CRU_dataset.csv)
--max-gap N       Maximum GDP gap to fill (default: 4 years)
--validate        Run validation checks on output
```

**Output columns:** `iso_id`, `year`, `pcGDP`, `growth_pcGDP`, `temp`, `precp`, `time`, `time2`, `Pop`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd detrended-response
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis with default settings:
```bash
python scripts/run_analysis.py
```

### Command Line Options

```
--use-csv PATH     Use pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--maddison PATH    Path to Maddison GDP Excel file
--cru PATH         Path to CRU temperature CSV file
--year-min YEAR    Minimum year to include
--year-max YEAR    Maximum year to include
--output-dir DIR   Output directory (default: timestamped directory in data/output/)
--loess-window N   Window size in years for LOESS smoothing (default: 25)
```

### Examples

Using pre-processed CSV:
```bash
python scripts/run_analysis.py --use-csv data/input/Maddison_CRU_dataset.csv
```

Custom year range:
```bash
python scripts/run_analysis.py --year-min 1970 --year-max 2010
```

### Bootstrap Uncertainty Analysis

The `run_bootstrap.py` script performs country-level cluster bootstrap resampling to compute confidence intervals for all parameters across all approaches.

```bash
python scripts/run_bootstrap.py
```

**What it does:**
1. Resamples countries with replacement (cluster bootstrap preserves within-country correlation)
2. Re-fits all approaches for each bootstrap iteration
3. Computes percentile-based confidence intervals (90% CI, IQR)
4. Generates distribution plots and summary statistics

**Options:**
```
--n-bootstrap N     Number of bootstrap iterations (default: 1000)
--random-seed SEED  Random seed for reproducibility (default: 42)
--use-csv PATH      Pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--year-min YEAR     Minimum year to include
--year-max YEAR     Maximum year to include
--output-dir DIR    Output directory (default: timestamped)
--loess-window N    LOESS window size (default: 25)
--quiet             Suppress progress messages
```

**Example:**
```bash
python scripts/run_bootstrap.py --n-bootstrap 1000 --output-dir results/bootstrap
```

### Country Influence Analysis

The `run_influence_analysis.py` script identifies which countries systematically skew bootstrap coefficient estimates upward or downward.

```bash
python scripts/run_influence_analysis.py
```

**What it does:**
1. For each bootstrap iteration, counts how many times each country appears in the resampled dataset
2. Computes percentile thresholds (5th, 25th, 75th, 95th) for each coefficient
3. Creates binary indicators for whether each iteration exceeds the threshold
4. Fits a Linear Probability Model: regresses the binary indicator on country counts
5. Ranks countries by regression coefficients to identify influential countries

**Options:**
```
--bootstrap-dir DIR    Bootstrap output directory
                       (default: most recent data/output/reference/bootstrap_*)
--output-dir DIR       Output directory (default: timestamped)
--approaches LIST      Approaches to analyze (default: all)
--coefficients LIST    Coefficients to analyze (default: approach-specific)
--percentiles LIST     Percentile thresholds (default: 5 25 75 95)
--regression-type      "linear" (default)
--n-top N              Number of top/bottom countries to report (default: 10)
```

**Default coefficients by approach:**
| Approach | Coefficients |
|----------|--------------|
| Standard (0-5, 5a-5d, 6, nocr0, nocr5) | h₁, h₂, T_optimal |
| Approach 6a/6b | h₁_total, h₂_total, h₁_trend, h₂_trend, T_optimal_total, T_optimal_trend |
| Approach 8 | h₂_low, h₂_high, T_optimal |
| Approach 8a | h₂_total, h₂_trend, T_optimal |

**Example:**
```bash
python scripts/run_influence_analysis.py --approaches "approach5 approach6" --n-top 15
```

**Interpretation:**
- Countries with **positive** influence coefficients tend to **increase** the parameter when included more frequently
- Countries with **negative** influence coefficients tend to **decrease** the parameter when included more frequently
- Large absolute coefficients indicate high sensitivity to that country's inclusion

### Approach 0 vs 5c Parameter Comparison

The `compare_approach0_5c.py` script generates scatter plots comparing parameters from Approach 0 (conjoined OLS) against parameters predicted from Approach 5c's pre-computed trends combined with Approach 0's h₁ and h₂.

```bash
python scripts/compare_approach0_5c.py
```

**What it does:**
1. Fits Approach 0 and Approach 5c to the same data
2. Extracts per-country j coefficients from Approach 0 residuals
3. Predicts j coefficients from Approach 5c's pre-computed trends (g, T₀, T₁) using Approach 0's h₁, h₂
4. Re-references by subtracting country 0's predicted j values (matching Approach 0's identification constraint), absorbing the subtracted quadratic into k
5. Applies a mean adjustment to align k levels
6. Generates a 2×2 scatter plot: (a) k(t), (b) j₀,ᵢ, (c) j₁,ᵢ, (d) j₂,ᵢ with best-fit regression equations, R², and correlation

**Options:**
```
--data-file PATH   Input CSV file (default: data/input/Maddison_CRU_dataset.csv)
--output-dir DIR   Output directory (default: data/output/approach0_vs_5c)
```

**Outputs:**
- `approach0_vs_5c_scatter.pdf` — 2×2 scatter plot with 1:1 reference and best-fit lines
- `approach0_vs_5c_scatter_data.csv` — Underlying scatter data for all panels
- `approach0_vs_5c_derivation.tex` — LaTeX derivation of the equations used

## Output Files

Results are saved to a timestamped directory in `data/output/`. Files include:

### Main Analysis Outputs

| File | Description |
|------|-------------|
| `comparison_summary.txt` | Text summary of all approaches |
| `comparison_table.csv` | Tabular comparison of coefficients and fit statistics |
| `comparison_table.xlsx` | Same as above in Excel format (includes variance decomposition) |
| `country_trends.csv` | Country-level trend coefficients |
| `temperature_response_*.pdf` | Plot of h(T) - h(T_opt) for each approach group |
| `temperature_derivative_*.pdf` | Plot of dh/dT for each approach group |
| `coefficient_comparison.pdf` | Bar chart comparing h₁ and h₂ across approaches |
| `optimal_temperature_comparison.pdf` | Bar chart of optimal temperatures |
| `year_effects.pdf` | Year fixed effects plot |
| `residuals_*.pdf` | Residual diagnostic plots for each approach |

### Bootstrap Outputs

| File | Description |
|------|-------------|
| `bootstrap_coefficients.csv` | All bootstrap samples (h₁, h₂, T_optimal, β, h₂_low, h₂_high) |
| `bootstrap_summary.txt` | Summary statistics and confidence intervals |
| `bootstrap_summary_table.csv` | Tabular summary with percentiles |
| `bootstrap_summary_table.xlsx` | Same as above in Excel format |
| `bootstrap_distributions.pdf` | Histograms of coefficient distributions |
| `bootstrap_temperature_response_*.pdf` | Temperature response curves with 90% uncertainty bands |
| `bootstrap_temperature_derivative.pdf` | Derivative curves with uncertainty bands |
| `bootstrap_T_optimal_comparison.pdf` | Optimal temperature comparison with error bars |

### Country Influence Outputs

| File | Description |
|------|-------------|
| `country_influence_coefficients.csv` | Full regression coefficients for all approach/coefficient/percentile combinations |
| `country_influence_rankings.csv` | Countries ranked by influence (all 157 countries per combination) |
| `country_influence_summary.txt` | Human-readable report with top 10 influential countries |

## Project Structure

```
detrended-response/
├── data/
│   ├── input/                   # Input data files
│   │   ├── mpd2023_web.xlsx     # Maddison GDP data
│   │   ├── cru_climate_data.csv # CRU temperature data
│   │   └── Maddison_CRU_dataset.csv  # Pre-processed merged dataset
│   └── output/                  # Analysis results (timestamped)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and merge GDP + temperature data
│   ├── detrending.py            # Country-level trend fitting (polynomial + LOESS)
│   ├── fitting.py               # OLS regression for each approach
│   ├── bootstrap.py             # Cluster bootstrap resampling
│   ├── influence.py             # Country influence analysis
│   └── output.py                # Results tables and plots
├── scripts/
│   ├── run_analysis.py              # Main entry point
│   ├── run_bootstrap.py             # Bootstrap uncertainty analysis
│   ├── run_influence_analysis.py    # Country influence on bootstrap coefficients
│   ├── compare_approach0_5c.py         # Scatter plots comparing Approach 0 vs 5c parameters
│   └── create_Maddison_CRU_dataset.py  # Create merged GDP/climate dataset
├── .gitignore
├── requirements.txt
├── DIAGNOSTICS.md               # Detailed diagnostics documentation
└── README.md
```

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.
- Bolt, J., & van Zanden, J. L. (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update. *Journal of Economic Surveys*.
- Harris, I., Osborn, T. J., Jones, P., & Lister, D. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. *Scientific Data*, 7(1), 109.
