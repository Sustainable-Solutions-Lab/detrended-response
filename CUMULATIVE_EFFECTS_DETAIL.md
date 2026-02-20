# Cumulative Effects Algorithm Documentation

This document describes the algorithms used in `scripts/calculate_cumulative_effects.py` to compute cumulative climate effects from bootstrap h(T) values.

## Overview

The script computes cumulative climate effects from 1961-2022 for each country and approach, showing how climate has cumulatively affected GDP relative to a 1961 baseline.

## What run_bootstrap.py Stores in bootstrap_h_values.csv

### Methods 0-4: Store h(T)

For methods 0-4, the h_T column contains:
```
h(T) = h1 * T + h2 * T²
```
This is the direct climate response function - the effect of temperature T on GDP growth rate.

### Method 5: Stores h_conv(T), NOT h(T)

For method 5, the h_T column contains the **converging climate response** which already incorporates persistence decay via accumulators:
```
h_conv(T) = h1 * (T - h4*A_T_lag) + h2 * (T² - h4*A_T2_lag)
```

Where the accumulators are computed recursively:
```
A_T(t) = T(t) + (1-h4) * A_T(t-1)      with A_T(first_year) = T(first_year)
A_T2(t) = T²(t) + (1-h4) * A_T2(t-1)   with A_T2(first_year) = T²(first_year)
```

And the **lagged values** A_T(t-1), A_T2(t-1) are used in the regressors.

**Key insight**: When h4=0, the regressors simplify to T and T², so h_conv = h(T) (same as method 2). This provides a consistency check.

## Key Mathematical Definitions

### Delta from Baseline

**h_T_delta(t)** = h_T(t) - h_T(1961)

This shows the climate effect in year t relative to 1961. By construction, h_T_delta(1961) = 0.

### Cumulative Effect: Different Formulas for Methods 0-4 vs Method5h4pos

**Methods 0-4: Subtract first, then compound**
```
h_T_delta(t) = h_T(t) - h_T(1961)
h_T_delta_cum(t) = (1 + h_T_delta_cum(t-1)) × (1 + h_T_delta(t)) - 1
```

This represents "GDP in year t relative to what it would have been if climate had stayed at 1961 levels."

**Method5h4pos (h4 > 0.001): Compound first, then subtract baseline cumulative**
```
GDP_cum_raw(t) = (1 + GDP_cum_raw(t-1)) × (1 + h_conv(t)) - 1
h_T_delta_cum(t) = GDP_cum_raw(t) - GDP_cum_raw(1961)
```

This is necessary because h_conv values for h4 > 0 decay towards zero as accumulated past effects build up. The h_conv(1961) value has no past to subtract (≈ h(T_1961)), while h_conv(2022) ≈ 0 (past effects cancel out). Using the standard "subtract first" formula would give large negative deltas that don't represent the physics correctly.

The "compound first" approach treats h_conv as the annual GDP growth contribution (which it is), compounds these to get total GDP effect, then makes it relative to 1961.

**Result**: Method5h4pos gives bounded cumulative effects that converge, rather than growing unboundedly like methods 0-4.

## Why Method5h4pos Uses a Different Cumulative Formula

The method 5 regression model is:
```
Δy = h1*(T - h4*A_T_lag) + h2*(T² - h4*A_T2_lag) + year_FE + country_trend + ε
```

The key insight is that h_conv represents the **net annual GDP growth contribution** after accounting for decay of past effects:
```
h_conv(t) = h(T(t)) - h4*h(T(t-1)) - h4*(1-h4)*h(T(t-2)) - ...
```

**Why the standard formula fails for h4 > 0:**

When h4 > 0, h_conv decays towards zero as the accumulated past effects build up:
- h_conv(1961) ≈ h(T_1961) (no past to subtract, so h_conv equals h(T))
- h_conv(2022) ≈ 0 (past accumulated effects cancel out current effect)

Using the standard "subtract first" formula:
```
h_T_delta(2022) = h_conv(2022) - h_conv(1961) ≈ 0 - h(T_1961) = -h(T_1961)
```

This gives large negative deltas for ALL countries (since h(T) > 0 for typical temperatures), which doesn't reflect the physics.

**The correct approach:**

h_conv represents the annual GDP growth contribution. Compounding these values gives the total GDP effect:
```
GDP_cum_raw(t) = compound(h_conv(1), h_conv(2), ..., h_conv(t))
```

Then subtracting the 1961 cumulative makes it relative to 1961:
```
h_T_delta_cum(t) = GDP_cum_raw(t) - GDP_cum_raw(1961)
```

This gives bounded cumulative effects that converge, reflecting the physics that past effects decay.

## Verification: h4=0 Case

When h4=0 for method 5:
- Accumulators: A_T_lag = A_T(t-1), but with decay rate 1, each year's accumulator equals just that year's T
- Regressors: X1 = T - 0*A_T_lag = T, X2 = T² - 0*A_T2_lag = T²
- h_conv = h1*T + h2*T² = h(T)

This means method 5 with h4=0 produces identical results to method 2, confirming the algorithm is correct.

## Processing Pipelines

### Pipeline 1: All Countries Point Estimate

**Purpose**: Generate data for `cumulative_effects_by_approach.pdf` (distribution across countries)

**Methods shown**: method0, method1, method2, method3, method4, method5h4pos

**Steps for methods 0-4**:
1. Load point estimate data (iteration = -1) from `bootstrap_h_values.csv`
2. For each (approach, iso3) group:
   - Extract years and h_T values
   - Set baseline: h_T_1961 = h_T at year 1961
   - Compute h_T_delta = h_T - h_T_1961
   - Compute h_T_delta_cum using persistent compounding

**Steps for method5h4pos point estimate**:
1. Load h4 values from bootstrap_coefficients.csv
2. Identify h4-positive iterations (h4 > 0.001)
3. For each h4-positive iteration, compute cumulative effects for all countries
4. Take the median cumulative effect across iterations for each (country, year)
5. This gives a representative central estimate for the h4-positive subset

### Pipeline 2: Representative Countries Full Bootstrap

**Purpose**: Generate data for box plots showing bootstrap uncertainty

**Methods shown**: method0, method1, method2, method3, method4, method5h4pos

**Selection of Representative Countries**:
- Uses method0 point estimate to rank countries by 2022 cumulative effect
- Selects countries at min, P5, P25, P50, P75, P95, max

**Steps**:
1. Load all bootstrap data (iterations 0-999 and -1) for representative countries
2. For each (iteration, approach, iso3) group:
   - Compute h_T_delta = h_T - h_T(1961)
   - Compute h_T_delta_cum using persistent compounding

**Creating method5h4pos for boxplots**:
1. Load h4 values from bootstrap_coefficients.csv
2. Identify h4-positive iterations (h4 > 0.001)
3. Re-process method5 raw data for these iterations using the correct cumulative formula:
   - Compound h_conv values directly
   - Subtract 1961's cumulative to make relative
4. Relabel as 'method5h4pos'
5. Boxplots show distribution of cumulative effects across these iterations

## Baseline Selection

For all methods, the baseline is the actual h_T value at 1961:
- For methods 0-3, 5: h_T = h(T_actual) or h_conv(T) at observed temperature
- For method 4: h_T = h(T_actual, T_trend) where T_trend is the LOESS trend

## Data Flow Summary

```
bootstrap_h_values.csv + bootstrap_coefficients.csv (for h4 filtering)
        │
        ├─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         ▼
All Countries (point estimates)                   Representative Countries (full bootstrap)
        │                                                         │
        ├── methods 0-4: iteration=-1                             ├── methods 0-4: all iterations
        │                                                         │
        └── method5h4pos: iteration with                          └── method5h4pos: filter method5
            median h4 (among h4>0.001)                                to h4 > 0.001 iterations
        │                                                         │
        ▼                                                         ▼
cumulative_effects_all_countries.csv              cumulative_h_values_summary.csv
        │                                                         │
        ▼                                                         ▼
cumulative_effects_by_approach.pdf                cumulative_effects_boxplot.pdf
(method0-4 + method5h4pos)                        cumulative_effects_by_method.pdf
                                                  (method0-4 + method5h4pos, bootstrap)
```

## Example: How h_conv Differs from h(T) and Why It Matters for Cumulative Effects

Consider a country with constant temperature T=20°C and h4=0.6:

**Method 2 (h(T))**:
- h(T) = h1*20 + h2*400 = some constant value V > 0 each year
- Cumulative: h_T_delta = 0 every year (since T=constant), so h_T_delta_cum = 0

**Method 5 with h4=0.6 (h_conv(T))**:
- First year: h_conv = h(T) = V (large positive)
- Second year: h_conv = 0.4*V (smaller)
- Third year: h_conv → 0 as accumulators reach equilibrium

**Why cumulative formulas differ:**

Using "subtract first" (WRONG for h4 > 0):
```
h_T_delta(2) = 0.4*V - V = -0.6*V  ← Large negative!
h_T_delta(∞) = 0 - V = -V         ← Even more negative!
```
Compounding these gives a large negative cumulative effect, which doesn't make physical sense.

Using "compound first" (CORRECT for h4 > 0):
```
GDP_cum_raw(1) = V
GDP_cum_raw(2) = (1+V)(1+0.4*V) - 1 ≈ V + 0.4*V  (approximately for small V)
GDP_cum_raw(∞) ≈ V + 0.4*V + 0.16*V + ... = V/(1-0.4) = 2.5*V  (converges!)
```
Then h_T_delta_cum = GDP_cum_raw - GDP_cum_raw(1961) gives bounded positive values.

**Key insight**: The decay model says effects don't compound forever. Method5h4pos correctly shows bounded cumulative effects (~5-8% typically), while methods 0-4 show unbounded effects (-60% to +114% depending on country).

## Note on method5h4pos

The h4-positive bootstrap iterations (h4 > 0.001) represent a subset of bootstrap samples where the decay parameter is meaningfully different from zero. These iterations often have systematically different h1 and h2 coefficients as well, resulting in a different response function.

This means method5h4pos shows the cumulative effects for a qualitatively different model, not just "method5 with decay".

**Point estimate selection**: For the all-countries plot, method5h4pos computes cumulative effects for each h4-positive iteration, then takes the median across iterations for each (country, year). This provides a representative central estimate for this subset of the bootstrap distribution.

**Bootstrap distribution**: For the representative country boxplots, method5h4pos shows the full distribution across all h4-positive iterations.
