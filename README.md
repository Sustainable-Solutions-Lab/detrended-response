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

## Six Approaches

### Approach 0: No Detrending

Fit all terms in a single fitting process. With this approach, it is unclear how much `jᵢ(t)` is acting to detrend temperature versus per capita GDP growth rates, and how much of the error term is being absorbed into these many additional degrees of freedom.

```
Δyᵢ(t) = h₁·T + h₂·T² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

Note: One could add any arbitrary quadratic in time to `k(t)` and subtract the same quadratic from all of the `jᵢ(t)`s. Therefore, three additional constraints must be added. Without loss of generality, we set `j₁(t) = 0` for the first country.

**Degrees of freedom:** 2 for h(T) + 3×(n_countries - 1) for jᵢ(t) + n_years for k(t)

### Approach 1: Linear Temperature Detrending

We know that at least `j₀,ᵢ` and `j₁,ᵢ` relate to the temperature scale, because their values would differ if temperature was measured in Celsius versus Kelvin. A natural assumption is that at least part of `jᵢ(t)` is meant to represent a linear detrending of temperature.

Pre-compute `T₀,ᵢ` and `T₁,ᵢ` for each country via least squares on the linear temperature trend, then estimate:

```
Δyᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kᵢ
```

**Degrees of freedom:** 2 for h(T) + n_countries for kᵢ

### Approach 2: Quadratic GDP Growth Detrending

Another interpretation of `jᵢ(t)` is that it represents a quadratic detrending of the `Δyᵢ(t)` values.

Pre-compute `y₀,ᵢ`, `y₁,ᵢ`, and `y₂,ᵢ` for each country via least squares on the quadratic per capita GDP-growth trend, then estimate:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·T + h₂·T² + kᵢ
```

**Degrees of freedom:** 2 for h(T) + n_countries for kᵢ

### Approach 3: Combined Detrending (Mixed)

Combines linear temperature detrending with quadratic GDP growth detrending. If the purpose of `jᵢ(t)` is to effect both a linear detrending of the temperature record and a quadratic detrending of the per-capita GDP growth record:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kᵢ
```

**Degrees of freedom:** 2 for h(T) + n_countries for kᵢ

### Approach 4: Combined Linear Detrending

There is something unsatisfying about correlating departures from a quadratic detrending of per capita GDP growth with departures from a linear detrending of temperature. This approach applies linear detrending to both variables.

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kᵢ
```

**Degrees of freedom:** 2 for h(T) + n_countries for kᵢ

### Approach 5: Combined Quadratic Detrending

Applies quadratic detrending to both per capita GDP growth and temperature. When combined with the 2nd order term in h(T), this results in `jᵢ(t)` becoming a 4th-order equation.

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)]
                                      + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)²] + kᵢ
```

**Degrees of freedom:** 2 for h(T) + n_countries for kᵢ

## Plot Color Scheme

In the output plots:

| Approach | Color | Line Style | Description |
|----------|-------|------------|-------------|
| 0 | Black | Solid | No detrending |
| 1 | Green | Dotted | Linear (temperature only) |
| 2 | Blue | Dashed | Quadratic (GDP growth only) |
| 3 | Red | Solid | Mixed (linear T + quadratic GDP) |
| 4 | Green | Solid | Linear (both) |
| 5 | Blue | Solid | Quadratic (both) |

**Color** indicates the degree of detrending (linear=green, quadratic=blue, mixed=red, none=black).
**Line style** indicates what is being detrended (temperature only=dotted, GDP growth only=dashed, both/none=solid).

## Data Sources

Two data input options are available:

### Option 1: Maddison + CRU (default)
- **GDP**: Maddison Project Database 2023 (Bolt & van Zanden, 2024)
- **Temperature**: CRU CY v4.09 country-level means (Harris et al., 2020)

### Option 2: Pre-processed CSV
- **df_base_withPop.csv**: Pre-merged dataset with GDP growth and temperature already computed. Contains columns: `iso_id`, `year`, `pcGDP`, `growth_pcGDP`, `temp`, `precp`, `time`, `Pop`.

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

Run the analysis with default settings (1960-2022):
```bash
python scripts/run_analysis.py
```

### Command Line Options

```
--use-csv PATH     Use pre-processed CSV file instead of Maddison/CRU
--maddison PATH    Path to Maddison GDP Excel file (default: data/input/mpd2023_web.xlsx)
--cru PATH         Path to CRU temperature CSV file (default: data/input/cru_climate_data.csv)
--year-min YEAR    Minimum year to include (default: 1960 for Maddison/CRU, all years for CSV)
--year-max YEAR    Maximum year to include (default: 2022 for Maddison/CRU, all years for CSV)
--output-dir DIR   Output directory (default: timestamped directory in data/output/)
```

### Examples

Using pre-processed CSV:
```bash
python scripts/run_analysis.py --use-csv data/input/df_base_withPop.csv
```

Custom year range:
```bash
python scripts/run_analysis.py --year-min 1970 --year-max 2010
```

## Output Files

Results are saved to a timestamped directory in `data/output/`. Files include:

| File | Description |
|------|-------------|
| `comparison_summary.txt` | Text summary of all approaches |
| `comparison_table.csv` | Tabular comparison of coefficients and fit statistics |
| `comparison_table.xlsx` | Same as above in Excel format |
| `country_trends.csv` | Country-level trend coefficients |
| `temperature_response.png` | Plot of h(T) - h(T_opt) for each approach |
| `temperature_derivative.png` | Plot of dh/dT = h₁ + 2h₂T |
| `coefficient_comparison.png` | Bar chart comparing h₁ and h₂ across approaches |
| `optimal_temperature_comparison.png` | Bar chart of optimal temperatures |
| `residuals_*.png` | Residual diagnostic plots for each approach |

## Project Structure

```
detrended-response/
├── data/
│   ├── input/                   # Input data files
│   │   ├── mpd2023_web.xlsx     # Maddison GDP data
│   │   ├── cru_climate_data.csv # CRU temperature data
│   │   └── df_base_withPop.csv  # Pre-processed alternative dataset
│   └── output/                  # Analysis results (timestamped)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and merge GDP + temperature data
│   ├── detrending.py            # Country-level trend fitting
│   ├── fitting.py               # OLS regression for each approach
│   └── output.py                # Results tables and plots
├── scripts/
│   └── run_analysis.py          # Main entry point
├── .gitignore
├── requirements.txt
└── README.md
```

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.
- Bolt, J., & van Zanden, J. L. (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update. *Journal of Economic Surveys*.
- Harris, I., Osborn, T. J., Jones, P., & Lister, D. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. *Scientific Data*, 7(1), 109.
