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

## Parameter Naming Convention

All methods use a consistent naming scheme for output coefficients:

### Universal Parameters

| Parameter | Meaning |
|-----------|---------|
| h1 | Linear temperature response coefficient |
| h2 | Quadratic temperature response coefficient |
| T_opt | Optimal temperature (where h(T) is maximized) |
| k(t) | Year fixed effects |
| r_squared | R² on detrended residuals |
| total_r_squared | R² on original Δy |

### Method-Specific Parameters

**Methods 0, 1, 2** (standard quadratic): h1, h2, T_opt only

**Method 3** (piecewise quadratic):

| Parameter | Meaning |
|-----------|---------|
| h1 | Linear term (fixed at 0) |
| h2 | Curvature below T_opt |
| h4 | Curvature above T_opt |
| T_opt | Breakpoint temperature |

**Method 4** (T response with quadratic departure):

| Parameter | Meaning |
|-----------|---------|
| h1, h2 | Response to actual temperature T |
| h4 | Response to departure from trend (T - T_trend)² |
| T_opt | Optimal actual temperature |
| T_dep_opt | Optimal departure from trend |

Note: Method 4 uses only h4 (quadratic departure term), with h3=0.

**Method 5** (persistence decay):

| Parameter | Meaning |
|-----------|---------|
| h1 | Linear temperature coefficient |
| h2 | Quadratic temperature coefficient |
| h4 | Persistence decay parameter [0=full persistence, 1=no persistence] |
| T_opt | Optimal temperature |

### Standard Error Convention

All coefficients follow the pattern:
- Point estimate: `{name}` (e.g., h1, h2, T_opt, f1)
- Standard error: `{name}_se` in Python, `{name}_SE` in CSV
- Bootstrap samples: `{name}_samples`
- Bootstrap statistics: `{name}_point`, `{name}_median`, `{name}_p5`, etc.

## Interpreting the Time Trend Function jᵢ(t)

The time trend function `jᵢ(t)` can be interpreted as:
1. A linear detrending of the country-level temperature curve
2. A quadratic detrending of the country-level per capita GDP growth-rate curve
3. A combination of both detrendings

If the quadratic time trend is meant to be one or both of these detrending functions, then these detrendings can be applied to the original datasets and the associated parameter values found prior to the main ordinary least squares solution for the climate response coefficients.

## Methods

### Method 0: Conjoined OLS Fit

Fit all terms in a single (conjoined) ordinary least squares fitting process. With this approach, it is unclear how much `jᵢ(t)` is acting to detrend temperature versus per capita GDP growth rates, and how much of the error term is being absorbed into these many additional degrees of freedom.

```
Δyᵢ(t) = h₁·T + h₂·T² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

Note: One could add any arbitrary quadratic in time to `k(t)` and subtract the same quadratic from all of the `jᵢ(t)`s. Therefore, three additional constraints must be added. Without loss of generality, we set `j₁(t) = 0` for the first country.

**Degrees of freedom:** 2 for h(T) + 3×(n_countries - 1) for jᵢ(t) + n_years for k(t)

### Method 1: Combined Detrending (Mixed)

Combines linear temperature detrending with quadratic GDP growth detrending. If the purpose of `jᵢ(t)` is to effect both a linear detrending of the temperature record and a quadratic detrending of the per-capita GDP growth record:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Method 2: LOESS Detrending (Departure Response)

Replaces polynomial detrending with LOESS (Locally Weighted Scatterplot Smoothing), a non-parametric method that allows for more flexible trend shapes.

**GDP detrending:**
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function:**
```
h(T) = h₁·T + h₂·T²
```

**Regression model:**
```
Δy*(t) = h(T) - h(T_trend)
```

where `T_trend = LOESS(T)`. The regression is performed on the design matrix `[T - T_trend, T² - T_trend²]`.

**Key insight:** This method regresses GDP growth on the *departure of the climate response from its trend*, i.e., `h(T) - h(T_trend)`. It measures how GDP responds to year-to-year fluctuations in the climate effect, not to the absolute temperature level.

Uses a 25-year LOESS window (configurable via `--loess-window`).

**Degrees of freedom:** 2 for h(T) (year effects pre-computed)

### Method 3: Piecewise Quadratic Response with LOESS

Uses a piecewise quadratic temperature response that allows different curvatures for temperatures above vs below the optimum. This captures asymmetry where warming may have different effects on hot vs cold countries.

**GDP detrending** (same as Method 2):
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function:**
```
h(T) = h₂ · (T - T_opt)²   if T ≤ T_opt
h(T) = h₄ · (T - T_opt)²   if T > T_opt
```

**Regression model:**
```
Δy*(t) = h(T) - h(T_trend)
```

**Parameters:**
- `T_opt`: Optimal temperature (breakpoint where h(T) = 0)
- `h₂`: Curvature for cold countries (T ≤ T_opt)
- `h₄`: Curvature for hot countries (T > T_opt)

**Optimization strategy:**
1. Outer optimization: Search for T_opt using L-BFGS-B
2. Inner OLS: For each T_opt, solve for h₂ and h₄ via 2-column OLS

**Key insight:** Allows asymmetric curvature above vs below the optimum. Both h₂ and h₄ should be negative (growth decreases away from optimum). If |h₄| > |h₂|, warming hurts hot countries more than cooling hurts cold countries.

**Degrees of freedom:** 3 (T_opt, h₂, h₄)

### Method 4: T Response with Quadratic Departure

Quadratic departure term only (no linear departure), testing whether temperature volatility matters regardless of direction.

**GDP detrending** (same as Method 2):
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function:**
```
h(T, T_trend) = h₁·T + h₂·T² + h₄·(T - T_trend)²
```

**Regression model:**
```
Δy*(t) = h(T, T_trend) - h(T_trend, T_trend)
       = h₁·(T - T_trend) + h₂·(T² - T_trend²) + h₄·(T - T_trend)²
```

The regression is performed on the design matrix `[(T - T_trend), (T² - T_trend²), (T - T_trend)²]`.

**Parameters:**
- `h₁`: Linear departure coefficient
- `h₂`: Quadratic T² - T_trend² coefficient
- `h₄`: Quadratic departure coefficient
- `T_opt`: Optimal temperature = -h₁/(2h₂)
- `T_dep_opt`: Optimal departure = 0 (by construction, since h₃=0)

**Key insight:** A negative h₄ means larger temperature fluctuations reduce growth (volatility is harmful), regardless of whether the fluctuation is positive or negative.

**Degrees of freedom:** 3 for h(T) (h₁, h₂, h₄)

### Method 5: Persistence Decay Model

Models persistent effects of past temperatures on current GDP growth. The climate response includes an exponentially decaying memory of past temperature effects.

**GDP detrending** (same as Method 2):
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function with persistence:**

The net climate effect at time t accounts for the current temperature effect minus the decaying contribution of all prior temperatures:

```
h_net(T(t)) = h(T(t)) - h₄ · Σₖ (1-h₄)^(k-1) · h(T(t-k)) - (1-h₄)^(t-t₀) · h(T(t₀))
```

where `h(T) = h₁·T + h₂·T²`, the sum runs over all prior years in the country's record, and `t₀` is the first observation year (1961). The final term is the **pre-first-year correction**: it accounts for the assumption that temperature was constant at `T(t₀)` before the first observation.

**Efficient computation using accumulators:**
```
A_T(t) = T(t) + (1-h₄) · A_T(t-1)
A_T²(t) = T²(t) + (1-h₄) · A_T²(t-1)
```

**Pre-first-year correction:**
```
C_T(t) = (1-h₄)^(t-t₀) · T(t₀)
C_T²(t) = (1-h₄)^(t-t₀) · T²(t₀)
```

**Modified regressors:**
```
X₁(t) = [T(t) - h₄·A_T(t-1) - C_T(t)] - [T_trend(t) - h₄·A_T_trend(t-1) - C_T_trend(t)]
X₂(t) = [T²(t) - h₄·A_T²(t-1) - C_T²(t)] - [T_trend²(t) - h₄·A_T²_trend(t-1) - C_T²_trend(t)]
```

**Parameters:**
- `h₁`: Linear temperature coefficient
- `h₂`: Quadratic temperature coefficient
- `h₄`: Persistence decay parameter [0, 1]
- `T_opt`: Optimal temperature = -h₁/(2h₂)

**Edge cases:**
- `h₄ = 0`: Full persistence (accumulated temperature effects persist indefinitely). Note: when h₄=0, the pre-first-year correction C_T(t) = T(t₀) for all t, which cancels with the constant contribution from the infinite past.
- `h₄ = 1`: No persistence (first-difference behavior, only current year matters)

**Optimization strategy:**
1. Grid search over h₄ ∈ [0, 1] with 21 points
2. Refine with Brent's method in the best region
3. Inner OLS: For each h₄, solve for h₁ and h₂

**Key insight:** This model tests whether past temperatures have lingering effects on current growth, or whether only the current year's temperature matters. The pre-first-year correction ensures that h_net values are bounded and decay properly over time when h₄ > 0.

**Degrees of freedom:** 3 (h₁, h₂, h₄)

### Null Models (No Climate Response)

Two null models for comparison:

| Model | Description |
|-------|-------------|
| **method0h0** | Joint OLS with country trends and year effects, but h₁=h₂=0 |
| **method1h0** | Precomputed k with country trends, but h₁=h₂=0 |

## Imbalance Metrics

### The Consistency Equation

If the choice of smoothing function and the determination of the climate response function were perfect, the following equation would be satisfied exactly:

```
0 = h(f_trend(Tᵢ(t))) + f_trend(Δyᵢ(t) - k(t)) + k(t)
```

where:
- `h(T) = h₁·T + h₂·T²` is the climate response function
- `f_trend(T)` is the temperature trend (linear, quadratic, or LOESS depending on method)
- `f_trend(Δy - k)` is the country-specific GDP growth trend (after removing year means)
- `k(t)` is the year mean

In practice, the right side will not sum to zero. The root-mean-square of this imbalance provides a diagnostic metric.

### Three Metrics

For each method, we compute:

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
--use-csv PATH              Use pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--maddison PATH             Path to Maddison GDP Excel file
--cru PATH                  Path to CRU temperature CSV file
--year-min YEAR             Minimum year to include
--year-max YEAR             Maximum year to include
--output-dir DIR            Output directory (default: timestamped directory in data/output/)
--loess-window N            Window size in years for LOESS smoothing (default: 25)
--mean-weight-distance N    Mean weighting distance in years for LOESS (alternative to --loess-window)
```

#### LOESS Window Specification

The LOESS smoothing window can be specified in two ways:

1. **`--loess-window N`**: Directly specify the window size in years (default: 25)

2. **`--mean-weight-distance N`**: Specify the mean weighting distance, which is converted to a window size using the formula: `window = (44/7) × mean_weight_distance`

The mean weighting distance represents the average distance (in years) at which data points contribute to the LOESS fit, providing a more intuitive parameterization. When using `--mean-weight-distance`, the output directory automatically includes a `_mwXX` suffix (e.g., `analysis_mw10_20260219_065737`).

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

The `run_bootstrap.py` script performs cluster bootstrap resampling to compute confidence intervals for all parameters across all methods. Two bootstrap modes are available:

1. **Country-only bootstrap** (default): Resamples 140 countries with replacement
2. **Country + Year bootstrap**: Resamples both 140 countries AND 62 years with replacement

```bash
python scripts/run_bootstrap.py
```

**What it does:**
1. Resamples countries with replacement (cluster bootstrap preserves within-country correlation)
2. Optionally resamples years with replacement (when `--sample-years` is enabled)
3. Re-fits all methods for each bootstrap iteration
4. Computes percentile-based confidence intervals (90% CI, IQR)
5. Generates distribution plots and summary statistics

**Options:**
```
--n-bootstrap N           Number of bootstrap iterations (default: 1000)
--random-seed SEED        Random seed for reproducibility (default: 42)
--use-csv PATH            Pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--year-min YEAR           Minimum year to include
--year-max YEAR           Maximum year to include
--output-dir DIR          Output directory (default: timestamped)
--loess-window N          LOESS window size (default: 25)
--mean-weight-distance N  Mean weighting distance for LOESS (alternative to --loess-window)
--sample-years            Enable time-dimension bootstrap (sample years with replacement)
--quiet                   Suppress progress messages
```

**Examples:**
```bash
# Standard country-only bootstrap
python scripts/run_bootstrap.py --n-bootstrap 1000 --output-dir results/bootstrap

# Combined country + year bootstrap
python scripts/run_bootstrap.py --n-bootstrap 1000 --sample-years --output-dir results/bootstrap_time
```

#### Time-Dimension Bootstrap (`--sample-years`)

When `--sample-years` is enabled, the bootstrap additionally samples 62 years with replacement. This captures uncertainty from year-to-year variation in addition to country-to-country variation.

**How it works:**
- Within each bootstrap iteration, the same set of years is used for all countries
- Observation weights are computed as: `weight = country_count × year_count`
- Weighted least squares is used for trend fitting and regression
- Years not sampled have weight = 0 and k(t) = NaN for those years

**Technical details:**
- Weighted detrending: Year means and country trends are computed using observation weights
- Weighted OLS: All approach fitting functions accept weights and use weighted least squares
- LOESS with weights: Country-specific LOESS trends are fitted with observation weights
- Numerical stability: SVD-based least squares handles rank-deficient matrices when some years have zero weight

**Forward integration note:** For cumulative effects analysis, years with NaN k(t) values (not sampled in that bootstrap iteration) should be infilled with the central point estimate k(t) values. This is handled in `calculate_cumulative_effects.py`.

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
--methods LIST         Methods to analyze (default: all)
--coefficients LIST    Coefficients to analyze (default: method-specific)
--percentiles LIST     Percentile thresholds (default: 5 25 75 95)
--regression-type      "linear" (default)
--n-top N              Number of top/bottom countries to report (default: 10)
```

**Default coefficients by method:**
| Method | Coefficients |
|--------|--------------|
| Standard (method0, method1, method2, method0h0, method1h0) | h₁, h₂, T_opt |
| Method 3 | h₂, h₄, T_opt |
| Method 4 | h₁, h₂, h₄, T_opt, T_dep_opt |
| Method 5 | h₁, h₂, h₄, T_opt |

**Example:**
```bash
python scripts/run_influence_analysis.py --methods "method0 method2" --n-top 15
```

**Interpretation:**
- Countries with **positive** influence coefficients tend to **increase** the parameter when included more frequently
- Countries with **negative** influence coefficients tend to **decrease** the parameter when included more frequently
- Large absolute coefficients indicate high sensitivity to that country's inclusion

### Method 0 vs 5c Parameter Comparison

The `compare_method0_5c.py` script generates scatter plots comparing parameters from Method 0 (conjoined OLS) against parameters predicted from Approach 5c's pre-computed trends combined with Method 0's h₁ and h₂.

```bash
python scripts/compare_method0_5c.py
```

**What it does:**
1. Fits Method 0 and Approach 5c to the same data
2. Extracts per-country j coefficients from Method 0 residuals
3. Predicts j coefficients from Approach 5c's pre-computed trends (g, T₀, T₁) using Method 0's h₁, h₂
4. Re-references by subtracting country 0's predicted j values (matching Method 0's identification constraint), absorbing the subtracted quadratic into k
5. Generates a 2×2 scatter plot: (a) k(t), (b) j₀,ᵢ, (c) j₁,ᵢ, (d) j₂,ᵢ with best-fit regression equations, R², and correlation

**Options:**
```
--data-file PATH   Input CSV file (default: data/input/Maddison_CRU_dataset.csv)
--output-dir DIR   Output directory (default: data/output/method0_vs_5c)
```

**Outputs:**
- `method0_vs_5c_scatter.pdf` — 2×2 scatter plot with 1:1 reference and best-fit lines
- `method0_vs_5c_scatter_data.csv` — Underlying scatter data for all panels
- `method0_vs_5c_derivation.tex` — LaTeX derivation of the equations used

For a detailed step-by-step mathematical derivation of this analysis, see [METHODS_DETAIL.md](METHODS_DETAIL.md).

### Method 5 h₄ Sweep Analysis

The `sweep_h4_method5.py` script evaluates method5 at fixed h₄ values across a range, computing metrics at each value. This is useful for understanding how the persistence decay parameter affects the fit.

```bash
python scripts/sweep_h4_method5.py
```

**What it does:**
1. Loads data and computes LOESS trends
2. For each h₄ value in the specified range:
   - Computes persistence accumulators
   - Fits the inner OLS for h₁ and h₂
   - Computes SSE, RMSE, R², total R², and T_optimal
3. Identifies the h₄ value with minimum SSE
4. Outputs results as a table (and optionally to CSV)

**Options:**
```
--h4-min N                Minimum h₄ value (default: 0.0)
--h4-max N                Maximum h₄ value (default: 1.0)
--h4-steps N              Number of h₄ values to evaluate (default: 21)
--mean-weight-distance N  Mean weighting distance for LOESS
--loess-window N          LOESS window size (default: 25)
--use-csv PATH            Input CSV file (default: data/input/Maddison_CRU_dataset.csv)
--year-min YEAR           Minimum year to include
--year-max YEAR           Maximum year to include
--output-csv PATH         Save results to CSV file
```

**Examples:**
```bash
# Basic sweep with default settings
python scripts/sweep_h4_method5.py

# Sweep with mean weight distance of 10
python scripts/sweep_h4_method5.py --mean-weight-distance 10

# Custom range with more steps
python scripts/sweep_h4_method5.py --h4-min 0.0 --h4-max 0.5 --h4-steps 51

# Save to CSV
python scripts/sweep_h4_method5.py --output-csv data/output/h4_sweep.csv
```

**Output columns:**
| Column | Description |
|--------|-------------|
| h4 | Fixed h₄ value |
| h1 | Linear temperature coefficient |
| h2 | Quadratic temperature coefficient |
| T_optimal | Optimal temperature (-h₁ / 2h₂) |
| SSE | Sum of squared errors |
| RMSE | Root mean squared error |
| r_squared | R² of detrended regression |
| total_r_squared | R² of original Δy explained |

### Cumulative Effects Analysis

The `calculate_cumulative_effects.py` script computes cumulative climate effects on GDP from bootstrap h(T) values.

```bash
python scripts/calculate_cumulative_effects.py
```

**What it does:**
1. Loads bootstrap h(T) values for all countries
2. Computes h(T) - h(T₁₉₆₁) for each year relative to the 1961 baseline
3. Calculates cumulative climate effects as the sum of annual differences:
   ```
   h_int(t) = Σ_{τ=1961}^{t} (h(T(τ)) - h(T(1961)))
   ```
   Since h(T) represents changes in log GDP, the cumulative sum gives the integrated effect on log GDP over time.
4. Selects representative countries at percentiles (5, 25, 50, 75, 95)
5. Creates box-and-whisker visualizations grouped by country and by approach

**Options:**
```
--reference-dir DIR       Parent directory containing bootstrap_* subdirectory
                          (e.g., data/output/reference_mw10)
--input-dir DIR           Directory containing bootstrap_h_values.csv
                          (default: most recent in reference-dir/bootstrap_* or
                          data/output/reference/bootstrap_*)
--output-dir DIR          Output directory (default: timestamped)
--loess-window N          LOESS window size (default: 25)
--mean-weight-distance N  Mean weighting distance for LOESS (overrides --loess-window)
```

**Examples:**
```bash
# Use default reference directory
python scripts/calculate_cumulative_effects.py

# Use a specific reference directory
python scripts/calculate_cumulative_effects.py --reference-dir data/output/reference_mw10

# Use explicit input directory
python scripts/calculate_cumulative_effects.py --input-dir data/output/reference/bootstrap_20260215_120000
```

### Publication Tables and Figures

The `make_tables_and_figures.py` script generates publication-quality tables and figures from analysis outputs.

```bash
python scripts/make_tables_and_figures.py
```

**What it does:**
1. Loads pre-computed results from run_analysis.py and run_bootstrap.py outputs
2. Generates formatted tables (CSV/Excel) with coefficients and confidence intervals
3. Creates publication-quality figures for temperature response curves, year effects, etc.

**Options:**
```
--reference-dir DIR    Parent directory containing analysis_* and bootstrap_* subdirectories
                       (e.g., data/output/reference_mw10)
--analysis-dir DIR     Path to analysis output directory
                       (default: most recent in reference-dir/analysis_* or
                       data/output/reference/analysis_*)
--bootstrap-dir DIR    Path to bootstrap output directory
                       (default: most recent in reference-dir/bootstrap_* or
                       data/output/reference/bootstrap_*)
--output-dir DIR       Output directory for publication files (default: timestamped)
--data-file PATH       Path to input data CSV for temperature histogram
                       (default: data/input/Maddison_CRU_dataset.csv)
```

**Examples:**
```bash
# Use default reference directory
python scripts/make_tables_and_figures.py

# Use a specific reference directory (finds analysis_* and bootstrap_* subdirs)
python scripts/make_tables_and_figures.py --reference-dir data/output/reference_mw10

# Use explicit directories
python scripts/make_tables_and_figures.py \
    --analysis-dir data/output/reference/analysis_20260215_100000 \
    --bootstrap-dir data/output/reference/bootstrap_20260215_120000
```

## Output Files

Results are saved to a timestamped directory in `data/output/`. Files include:

### Main Analysis Outputs

| File | Description |
|------|-------------|
| `comparison_summary.txt` | Text summary of all methods |
| `comparison_table.csv` | Tabular comparison of coefficients and fit statistics |
| `comparison_table.xlsx` | Same as above in Excel format (includes variance decomposition) |
| `country_trends.csv` | Country-level trend coefficients |
| `temperature_response_*.pdf` | Plot of h(T) - h(T_opt) for each method group |
| `temperature_derivative_*.pdf` | Plot of dh/dT for each method group |
| `coefficient_comparison.pdf` | Bar chart comparing h₁ and h₂ across methods |
| `optimal_temperature_comparison.pdf` | Bar chart of optimal temperatures |
| `year_effects.pdf` | Year fixed effects plot |
| `residuals_*.pdf` | Residual diagnostic plots for each method |

### Bootstrap Outputs

| File | Description |
|------|-------------|
| `bootstrap_coefficients.csv` | All bootstrap samples (h1, h2, T_opt, f1, h3, h4, etc.) |
| `bootstrap_summary.txt` | Summary statistics and confidence intervals |
| `bootstrap_summary_table.csv` | Tabular summary with percentiles |
| `bootstrap_summary_table.xlsx` | Same as above in Excel format |
| `bootstrap_distributions.pdf` | Histograms of coefficient distributions |
| `bootstrap_temperature_response_*.pdf` | Temperature response curves with 90% uncertainty bands |
| `bootstrap_temperature_derivative.pdf` | Derivative curves with uncertainty bands |
| `bootstrap_T_opt_comparison.pdf` | Optimal temperature comparison with error bars |
| `bootstrap_country_samples.csv` | Country indices sampled in each bootstrap iteration |
| `bootstrap_year_samples.csv` | Year indices sampled (only when `--sample-years` used) |
| `bootstrap_k_samples.csv` | Year fixed effects k(t) for each bootstrap iteration |
| `bootstrap_h_values.csv` | h(T) values for each observation and bootstrap iteration |
| `bootstrap_h_baselines.csv` | h(T) baseline values at 1961 for each country |

### Country Influence Outputs

| File | Description |
|------|-------------|
| `country_influence_coefficients.csv` | Full regression coefficients for all method/coefficient/percentile combinations |
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
│   ├── fitting.py               # OLS regression for each method
│   ├── bootstrap.py             # Cluster bootstrap resampling
│   ├── influence.py             # Country influence analysis
│   └── output.py                # Results tables and plots
├── scripts/
│   ├── run_analysis.py              # Main entry point
│   ├── run_bootstrap.py             # Bootstrap uncertainty analysis
│   ├── run_influence_analysis.py    # Country influence on bootstrap coefficients
│   ├── compare_method0_5c.py        # Scatter plots comparing Method 0 vs 5c parameters
│   ├── sweep_h4_method5.py          # Sweep h₄ persistence decay parameter for method5
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
