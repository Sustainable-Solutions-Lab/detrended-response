# Methods Detail: Approach 0 vs 5c Parameter Comparison

This document provides a detailed mathematical derivation of the analysis performed by `scripts/compare_approach0_5c.py`, which demonstrates the equivalence between Approach 0 (conjoined OLS) and Approach 5c (precomputed k) under specific identification constraints.

## Overview

The script compares four sets of parameters between Approach 0 and Approach 5c:
- **k(t)**: Year fixed effects
- **j₀,ᵢ**: Country intercepts
- **j₁,ᵢ**: Country linear trend coefficients
- **j₂,ᵢ**: Country quadratic trend coefficients

The key insight is that both approaches estimate the same underlying model, but with different computational strategies. When properly re-referenced, the parameters should be nearly identical.

## Model Specification

### Approach 0: Conjoined OLS

Approach 0 fits the full model jointly:

```
Δyᵢ(t) = h₁·Tᵢ(t) + h₂·Tᵢ(t)² + jᵢ(t) + k(t) + εᵢ(t)
```

where:
- `Δyᵢ(t)` = GDP growth rate for country i at time t
- `h₁, h₂` = climate response coefficients (shared across countries)
- `jᵢ(t) = j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t²` = country-specific quadratic trend
- `k(t)` = year fixed effects (shared across countries)
- `εᵢ(t)` = residual

**Identification constraint**: Country 0 has j₀,₀ = j₁,₀ = j₂,₀ = 0.

### Approach 5c: Precomputed k

Approach 5c first computes year means k(t), then fits country trends to the residuals:

1. Compute `k_mean(t) = mean_i[Δyᵢ(t)]`
2. Fit country trends `gᵢ(t) = g₀,ᵢ + g₁,ᵢ·t + g₂,ᵢ·t²` to `Δyᵢ(t) - k_mean(t)`
3. Fit climate response `h₁, h₂` to the detrended residuals

## Step-by-Step Analysis

### Step 1: Load Data and Compute Trends

```python
data = load_data_from_csv(args.data_file)
trends = compute_country_trends(data)
year_means = compute_year_means(data)
trends_with_k = compute_country_trends_with_k(data, year_means)
```

- `trends`: Contains T₀,ᵢ (mean temperature) and T₁,ᵢ (temperature trend) for each country
- `year_means`: k_mean(t) = mean GDP growth across countries for each year
- `trends_with_k`: Contains g₀,ᵢ, g₁,ᵢ, g₂,ᵢ (GDP growth trends fitted to Δy - k_mean)

### Step 2: Fit Both Approaches

```python
result0 = fit_approach0_no_detrending(data)
result5c = fit_approach5c_precomputed_k_combined(data, trends_with_k, year_means)
```

Both approaches yield h₁ and h₂ estimates. Approach 0 also directly provides k(t) and implicitly defines jᵢ(t) through its residuals.

### Step 3: Extract Approach 0 j Coefficients

For each country, compute the residual after removing climate response and year effects:

```
rᵢ(t) = Δyᵢ(t) - h₁·Tᵢ(t) - h₂·Tᵢ(t)² - k(t)
```

Then fit a quadratic trend to get j₀,ᵢ, j₁,ᵢ, j₂,ᵢ:

```python
for c in range(data.n_countries):
    r_c = dy_c - h1 * T_c - h2 * T_c**2 - k_c
    j0[c], j1[c], j2[c] = fit_quadratic_trend(t_c, r_c)
```

### Step 4: Predict j Coefficients from Approach 5c Trends

The key mathematical relationship derives j coefficients from the pre-computed GDP trends (g) and temperature trends (T₀, T₁):

**Raw (un-referenced) j coefficients:**

```
j₀,ᵢ = g₀,ᵢ - (h₁·T₀,ᵢ + h₂·T₀,ᵢ²)
j₁,ᵢ = g₁,ᵢ - (h₁·T₁,ᵢ + 2·h₂·T₀,ᵢ·T₁,ᵢ)
j₂,ᵢ = g₂,ᵢ - h₂·T₁,ᵢ²
```

**Derivation:**

The GDP growth trend for country i can be written as:
```
gᵢ(t) = g₀,ᵢ + g₁,ᵢ·t + g₂,ᵢ·t²
```

The temperature for country i evolves approximately linearly:
```
Tᵢ(t) ≈ T₀,ᵢ + T₁,ᵢ·t
```

The climate response at this temperature is:
```
h(Tᵢ(t)) = h₁·Tᵢ(t) + h₂·Tᵢ(t)²
         = h₁·(T₀,ᵢ + T₁,ᵢ·t) + h₂·(T₀,ᵢ + T₁,ᵢ·t)²
         = h₁·T₀,ᵢ + h₁·T₁,ᵢ·t + h₂·T₀,ᵢ² + 2·h₂·T₀,ᵢ·T₁,ᵢ·t + h₂·T₁,ᵢ²·t²
```

Collecting terms by power of t:
```
h(Tᵢ(t)) = [h₁·T₀,ᵢ + h₂·T₀,ᵢ²] + [h₁·T₁,ᵢ + 2·h₂·T₀,ᵢ·T₁,ᵢ]·t + [h₂·T₁,ᵢ²]·t²
```

Since gᵢ(t) = jᵢ(t) + h(Tᵢ(t)), we get:
```
jᵢ(t) = gᵢ(t) - h(Tᵢ(t))
```

Matching coefficients gives the formulas above.

### Step 5: Re-reference to Match Approach 0's Identification

Approach 0 uses the constraint that country 0 has zero j coefficients. We apply the same constraint to the predicted j values:

```python
j0_ref = j0_raw[0]
j1_ref = j1_raw[0]
j2_ref = j2_raw[0]

j0_pred[c] = j0_raw[c] - j0_ref
j1_pred[c] = j1_raw[c] - j1_ref
j2_pred[c] = j2_raw[c] - j2_ref
```

The subtracted quadratic `j₀,ref + j₁,ref·t + j₂,ref·t²` must be absorbed somewhere. It gets absorbed into k(t).

### Step 6: Adjust k to Absorb the Reference Quadratic

Approach 5c's adjusted k becomes:

```
k'(t) = k_mean(t) + j₀,ref + j₁,ref·t + j₂,ref·t²
```

```python
k5c_values = [
    year_means[yr] + j0_ref + j1_ref * t + j2_ref * t**2
    for yr, t in zip(unique_years, t_years)
]
```

### Step 7: Apply Mean Adjustment

A final mean adjustment aligns the overall levels:

```python
k_mean_diff = np.mean(k5c_values) - np.mean(k0_values)
k5c_values = k5c_values - k_mean_diff
for c in j0_pred:
    j0_pred[c] -= k_mean_diff
```

This adjustment is necessary because there's an arbitrary constant that can shift between k and j₀ without changing the model fit.

## Expected Results

If both approaches estimate the same underlying model correctly, the scatter plots should show:

1. **k(t)**: Points fall on the 1:1 line (R² ≈ 1.0)
2. **j₀,ᵢ**: Points fall on the 1:1 line (R² ≈ 1.0)
3. **j₁,ᵢ**: Points fall on the 1:1 line (R² ≈ 1.0)
4. **j₂,ᵢ**: Points fall on the 1:1 line (R² ≈ 1.0)

Small deviations arise from:
- Numerical precision
- The linear approximation T(t) ≈ T₀ + T₁·t (actual temperature may have higher-order terms)
- Different estimation procedures (joint vs sequential)

## Output Files

The script generates:

| File | Description |
|------|-------------|
| `approach0_vs_5c_scatter.pdf` | 2×2 scatter plot with 1:1 reference lines, best-fit regression, R², and correlation |
| `approach0_vs_5c_scatter_data.csv` | Raw data for all scatter panels |

## Mathematical Summary

The key identity connecting the two approaches:

```
Approach 0:  Δy = h(T) + j(t) + k(t)
Approach 5c: Δy = h(T) + g(t) + k_mean(t)
```

where:
```
g(t) = j(t) + [h(T(t)) - h(T)]
     = j(t) + [linearized climate trend absorbed into country trend]
```

Re-referencing and absorbing the reference country's j into k recovers exact equivalence.
