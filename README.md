# Detrended Response Analysis

This project examines different interpretations of the time trend terms in the Burke et al. (2015) climate-economy relationship. The Burke equation contains country-specific quadratic time trends that can be interpreted as:

1. Linear detrending of country-level temperature
2. Quadratic detrending of country-level per capita GDP growth
3. A combination of both detrendings

This work makes these implicit detrendings explicit and compares the resulting climate response estimates.

## The Model

The Burke et al. (2015) equation (simplified to temperature only) is:

```
Δyᵢ(t) = h₁·Tᵢ(t) + h₂·Tᵢ(t)² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

Where:
- `Δyᵢ(t)` is the per capita GDP growth rate for country i in year t
- `Tᵢ(t)` is the annual mean temperature
- `h₁, h₂` are the temperature response coefficients
- `j₀,ᵢ, j₁,ᵢ, j₂,ᵢ` are country-specific quadratic time trend coefficients
- `kₜ` are year fixed effects

The optimal temperature (where growth is maximized) is: `T* = -h₁ / (2·h₂)`

## Four Approaches

### Approach 0: Burke Original (No Pre-Detrending)
The standard Burke et al. (2015) specification with all j terms and year fixed effects estimated jointly via OLS. No detrending is performed prior to the fit:
```
Δyᵢ(t) = h₁·T + h₂·T² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```
All parameters (h₁, h₂, j terms, kₜ) are estimated simultaneously.

### Approach 1: Linear Temperature Detrending
Interprets the time trend as removing a linear temperature trend. Pre-computes `T₀,ᵢ` and `T₁,ᵢ` for each country via least squares, then estimates:
```
Δyᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kᵢ
```

### Approach 2: Quadratic GDP Growth Detrending
Interprets the time trend as removing a quadratic GDP growth trend. Pre-computes `y₀,ᵢ, y₁,ᵢ, y₂,ᵢ` for each country via least squares, then estimates:
```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·T + h₂·T² + kᵢ
```

### Approach 3: Combined Detrending
Applies both linear temperature detrending and quadratic GDP growth detrending:
```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kᵢ
```

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

Using default Maddison + CRU data:
```bash
python scripts/run_analysis.py
```

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
| `country_trends.csv` | Country-level trend coefficients (T₀,ᵢ, T₁,ᵢ, y₀,ᵢ, y₁,ᵢ, y₂,ᵢ) |
| `temperature_response.png` | Plot of h(T) = h₁T + h₂T² for each approach |
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

## Next Steps

The two data input options currently produce results with **opposite signs** for h₁ and h₂:

| Dataset | h₁ | h₂ | T_optimal |
|---------|----|----|-----------|
| Maddison + CRU | -0.0007 | +0.00002 | ~17°C |
| df_base_withPop.csv | +0.014 | -0.0005 | ~14°C |

The expected signs (based on Burke et al. 2015) are h₁ > 0 and h₂ < 0, which matches the pre-processed CSV but not the Maddison/CRU pipeline.

**Priority**: Diagnose why these two approaches give results with opposing signs. Potential issues to investigate:
- GDP variable: Is the Maddison/CRU pipeline using total GDP instead of per-capita GDP?
- Growth rate computation: Are the growth rates being computed consistently?
- Temperature data: Are there differences in how temperature is aggregated or scaled?
- Country/year coverage: Do the datasets have different country or time coverage that affects results?

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.
- Bolt, J., & van Zanden, J. L. (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update. *Journal of Economic Surveys*.
- Harris, I., Osborn, T. J., Jones, P., & Lister, D. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. *Scientific Data*, 7(1), 109.
