# Detrended Response Analysis

This project examines different interpretations of the time trend terms in the Burke et al. (2015) climate-economy relationship. The Burke equation contains country-specific quadratic time trends that can be interpreted as:

1. Linear detrending of country-level temperature
2. Quadratic detrending of country-level per capita GDP growth
3. A combination of both detrendings

This work makes these implicit detrendings explicit and compares the resulting climate response estimates.

## The Model

The Burke et al. (2015) equation (simplified to temperature only) is:

```
Δy_i(t) = h₁·T_i(t) + h₂·T_i(t)² + j_{0,i} + j_{1,i}·t + j_{2,i}·t² + k_t
```

Where:
- `Δy_i(t)` is the per capita GDP growth rate for country i in year t
- `T_i(t)` is the annual mean temperature
- `h₁, h₂` are the temperature response coefficients
- `j_{0,i}, j_{1,i}, j_{2,i}` are country-specific quadratic time trend coefficients
- `k_t` are year fixed effects

The optimal temperature (where growth is maximized) is: `T* = -h₁ / (2·h₂)`

## Four Approaches

### Approach 0: Burke Original (No Pre-Detrending)
The standard Burke et al. (2015) specification with all j terms and year fixed effects estimated jointly via OLS. No detrending is performed prior to the fit:
```
Δy_i(t) = h₁·T + h₂·T² + j_{0,i} + j_{1,i}·t + j_{2,i}·t² + k_t
```
All parameters (h₁, h₂, j terms, k_t) are estimated simultaneously.

### Approach 1: Linear Temperature Detrending
Interprets the time trend as removing a linear temperature trend. Pre-computes `T_{0,i}` and `T_{1,i}` for each country via least squares, then estimates:
```
Δy_i(t) = h₁·[T - (T₀ + T₁·t)] + h₂·[T² - (T₀ + T₁·t)²] + k_i
```

### Approach 2: Quadratic GDP Growth Detrending
Interprets the time trend as removing a quadratic GDP growth trend. Pre-computes `y_{0,i}, y_{1,i}, y_{2,i}` for each country via least squares, then estimates:
```
Δy_i(t) - (y₀ + y₁·t + y₂·t²) = h₁·T + h₂·T² + k_i
```

### Approach 3: Combined Detrending
Applies both linear temperature detrending and quadratic GDP growth detrending:
```
Δy_i(t) - (y₀ + y₁·t + y₂·t²) = h₁·[T - (T₀ + T₁·t)] + h₂·[T² - (T₀ + T₁·t)²] + k_i
```

## Data Sources

- **GDP**: Maddison Project Database 2023 (Bolt & van Zanden, 2024)
- **Temperature**: CRU CY v4.09 country-level means (Harris et al., 2020)

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
--maddison PATH    Path to Maddison GDP Excel file (default: data/input/mpd2023_web.xlsx)
--cru PATH         Path to CRU temperature CSV file (default: data/input/cru_climate_data.csv)
--year-min YEAR    Minimum year to include (default: 1960)
--year-max YEAR    Maximum year to include (default: 2022)
--output-dir DIR   Output directory (default: timestamped directory in data/output/)
```

Example with custom year range:
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
| `country_trends.csv` | Country-level trend coefficients (T₀, T₁, y₀, y₁, y₂) |
| `temperature_response.png` | Plot of h(T) = h₁T + h₂T² for each approach |
| `temperature_derivative.png` | Plot of dh/dT = h₁ + 2h₂T |
| `coefficient_comparison.png` | Bar chart comparing h₁ and h₂ across approaches |
| `optimal_temperature_comparison.png` | Bar chart of optimal temperatures |
| `residuals_*.png` | Residual diagnostic plots for each approach |

## Project Structure

```
detrended-response/
├── data/
│   ├── input/                  # Input data files
│   │   ├── mpd2023_web.xlsx    # Maddison GDP data
│   │   └── cru_climate_data.csv # CRU temperature data
│   └── output/                 # Analysis results (timestamped)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load and merge GDP + temperature data
│   ├── detrending.py           # Country-level trend fitting
│   ├── fitting.py              # OLS regression for each approach
│   └── output.py               # Results tables and plots
├── scripts/
│   └── run_analysis.py         # Main entry point
├── .gitignore
├── requirements.txt
└── README.md
```

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.
- Bolt, J., & van Zanden, J. L. (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update. *Journal of Economic Surveys*.
- Harris, I., Osborn, T. J., Jones, P., & Lister, D. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. *Scientific Data*, 7(1), 109.
