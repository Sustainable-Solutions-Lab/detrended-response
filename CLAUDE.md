# Claude Code Style Guide for Detrended Response Analysis

## Project Goal
This project investigates how the country-specific quadratic time trend in the Burke et al. (2015) climate-GDP equation can be interpreted as explicit detrending of temperature and/or GDP growth. It implements multiple methods (0-4) with varying detrending strategies and compares their climate response estimates.

When implementing new analysis features:
- Maintain consistency with the existing method numbering and naming conventions
- Ensure new methods follow the established pattern of precomputing trends vs joint estimation
- Document the degrees of freedom and interpretation for any new method
- Preserve comparability across methods by using consistent data filtering and year ranges

## Coding Philosophy
This project prioritizes elegant, fail-fast code that surfaces errors quickly rather than hiding them.

### Root Cause Analysis
- Always investigate and understand the root cause of problems before implementing solutions
- Avoid band-aid fixes that mask symptoms without addressing underlying issues
- When unexpected behavior occurs, trace it back to its source rather than applying quick patches
- Document the reasoning behind fixes to prevent similar issues

## Core Style Requirements

### Error Handling
- No input validation on function parameters (except for command-line interfaces)
- No defensive programming - let exceptions bubble up naturally
- Fail fast - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- No try-catch blocks unless absolutely necessary for program logic (not error suppression)
- Assume complete data - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- Minimize conditional statements - prefer functional approaches, mathematical expressions, and numpy vectorization
- Favor mathematical clarity over defensive checks
- Use numpy operations instead of loops and conditionals where possible
- Compute once, use many times - move invariant calculations outside loops and create centralized helper functions
- No backward compatibility - do not add conditional logic to support deprecated field names or old configurations. Update all code and configurations to use current conventions
- Use standard Python packages - prefer established numerical methods from scipy, numpy, statsmodels rather than implementing custom numerical algorithms

### Code Organization
- All imports at the top of the file - no imports inside functions or scattered throughout the code
- Source code belongs in `src/` with clear module responsibilities:
  - `data_loader.py` - data loading and merging
  - `detrending.py` - trend fitting (polynomial and LOESS)
  - `fitting.py` - OLS regression for each approach
  - `bootstrap.py` - cluster bootstrap resampling
  - `output.py` - results tables and plots
  - `publication.py` - publication-quality figure generation
- Scripts belong in `scripts/` and should be thin wrappers around src modules

### Protected Directories
- Never modify files in `./data/input/` - this directory contains reference data that must remain unchanged
- Reference outputs in `./data/output/reference/` should only be updated deliberately when establishing new baselines

### Naming Conventions
- **Approaches**: Use the systematic naming convention where:
  - First digit (0-3) indicates climate response function type:
    - 0 = No climate response (null models)
    - 1 = Quadratic response function
    - 2 = Piecewise quadratic response function
    - 3 = Quadratic with persistence time scale
  - Letter suffix indicates trend identification method:
    - J = Joint OLS solution
    - P = Polynomial trend identification (linear/quadratic)
    - L = LOESS trend identification
  - Examples: `Approach1J`, `Approach1P`, `Approach1L`, `Approach2L`, `Approach3L`, etc.
- **Coefficients**: Use `h1`, `h2`, `T_optimal`, `h2_low`, `h2_high` consistently
- **Bootstrap results**: Use `_samples` suffix for arrays of bootstrap values, `_point` for point estimates
- Descriptive names preferred - long, clear names are better than short, ambiguous ones

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling
- Clean fail-fast approach - if required arguments are not supplied, the code should fail immediately with a clear error

### Plotting Conventions
- **Temperature axis range**: Always use 0°C to 30°C for temperature axes in climate-growth plots
- **Approach colors**: Use consistent color scheme defined in `APPROACH_COLORS` in output.py:
  - Approach1J: black
  - Approach1P: red
  - Approach1L: orange
  - Approach2L: magenta
  - Approach3L: cyan
  - Approach2J: darkgreen
  - Approach3J: darkblue
  - Approach2P: olive
  - Approach3P: teal
  - Approach0J/Approach0P/Approach0L (null models): gray
- **Uncertainty bands**: Show 90% CI (lighter) and IQR (darker) for bootstrap results
- **Year effects**: Plot k(t) with bootstrap uncertainty bands, shared y-axis across methods
- **Diverging colormaps**: For difference plots or any plot using a diverging colormap (e.g., RdBu_r where white is in the middle), always use symmetric bounds with equal magnitude and opposite sign, so that white represents zero. Example: if data ranges from -0.03 to 0.05, use bounds of (-0.05, 0.05) not the raw data range.

### Bootstrap Analysis
- Use country-level cluster bootstrap (resample countries, not individual observations)
- Default to 1000 bootstrap iterations for publication results
- Store both full sample arrays and summary statistics (percentiles)
- Preserve year fixed effects (k_samples) for year effects plots
- **Testing**: Use 10 bootstrap samples (`--n-bootstrap 10`) when testing bootstrap-related code changes to minimize runtime

## Mathematical Conventions

### Response Function Centering
- Always plot `h(T) - h(T_opt)` so curves pass through zero at optimal temperature
- For piecewise models (method3), ensure continuity at T_optimal

### Detrending Order
1. Compute year means k(t) first (for methods 1+)
2. Fit country GDP trends on `Δy - k`
3. Fit country temperature trends
4. Final OLS on residuals

### Variance Decomposition
- Normalize all variance components by `Var(Δy)`
- Include all covariance terms (multiply by 2 for cross-terms)
- Sum should equal 1.0 as a consistency check
