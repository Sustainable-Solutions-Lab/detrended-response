# Methods Detail: Approach 0 vs 5c Parameter Comparison

This document provides a detailed mathematical derivation of the analysis performed by `scripts/compare_method0_5c.py`, which demonstrates the equivalence between Approach 0 (conjoined OLS) and Approach 5c (precomputed k) under specific identification constraints.

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
result0 = fit_method0_no_detrending(data)
result5c = fit_method1_precomputed_k_combined(data, trends_with_k, year_means)
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
| `method0_vs_5c_scatter.pdf` | 2×2 scatter plot with 1:1 reference lines, best-fit regression, R², and correlation |
| `method0_vs_5c_scatter_data.csv` | Raw data for all scatter panels |

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

---

# Persistence Decay Model: Bootstrap Methodology

This section documents how the persistence decay (Approach D) climate response is bootstrapped when using year-level resampling (`sample_years=True`).

## The Persistence Decay Model

The decay model generalizes the standard quadratic climate response `h(T) = h₁T + h₂T²` by allowing past temperature effects to persist with exponential decay:

```
h_conv(T(t)) = h(T(t)) - h₄ · Σ_{k=1}^{n} (1-h₄)^{k-1} · h(T(t-k))
```

where h₄ ∈ [0, 1] is the persistence decay parameter:
- h₄ = 0: Full persistence (all past effects accumulate indefinitely)
- h₄ = 1: No persistence (only current year temperature matters)

### Efficient Accumulator Representation

Rather than computing the infinite sum directly, the model uses recursive accumulators:

```
A_T(t) = T(t) + (1-h₄) · A_T(t-1)
A_T²(t) = T²(t) + (1-h₄) · A_T²(t-1)
```

The modified regressors become:
```
X₁(t) = T(t) - h₄ · A_T(t-1)
X₂(t) = T²(t) - h₄ · A_T²(t-1)
```

For the detrended approaches (DL, DP), the regressors subtract the corresponding trend accumulators:
```
X₁(t) = [T(t) - h₄·A_T(t-1) - c_T(t)] - [T_trend(t) - h₄·A_T_trend(t-1) - c_T_trend(t)]
```

where c_T(t) is a pre-first-year correction term accounting for the assumption that temperature was constant before the first observation.

### Three Variants

| Approach | Trend Method | Year Effects | h₄ Optimization |
|----------|-------------|--------------|-----------------|
| DJ | Joint OLS (estimated in design matrix) | Joint OLS | 1D search over h₄, inner OLS for all other params |
| DL | LOESS (pre-computed) | Pre-computed year means | 1D search over h₄, inner 2-column OLS for h₁, h₂ |
| DP | Polynomial (pre-computed) | Pre-computed year means | 1D search over h₄, inner 2-column OLS for h₁, h₂ |

## Bootstrap with Year Sampling

The cluster bootstrap resamples both countries and years with replacement. Rather than creating duplicate data, it uses a weighting scheme on the original data:

```
weight(i) = country_count(c_i) × year_count(yr_i)
```

where `country_count(c)` = number of times country c was drawn, and `year_count(yr)` = number of times year yr was drawn. Observations from unsampled years (year_count = 0) get weight 0.

### Why Accumulators Must Use ALL Years

The persistence accumulators iterate through **all** years chronologically for each country, including years with zero bootstrap weight. This is essential because:

1. **Physical continuity**: The accumulator A_T(t) represents the exponentially weighted temperature history. Skipping a year would break the decay chain: year t+1's accumulator depends on year t's value regardless of whether year t was sampled.

2. **Regressor correctness**: The modified regressors X₁(t) and X₂(t) at a sampled year depend on A_T(t-1), which may have been accumulated through unsampled years. Omitting those years would produce incorrect regressor values at sampled years.

3. **Separation of concerns**: The accumulators describe the physical temperature history (always the same), while the bootstrap weights describe which observations inform the statistical inference.

### SSE Uses Only Sampled Years

The h₄ optimization minimizes a weighted sum of squared errors:

```
SSE(h₄) = Σᵢ wᵢ · (yᵢ - ŷᵢ)²
```

Observations with weight 0 contribute nothing to the SSE. The inner OLS is also weighted:

```
β = argmin Σᵢ wᵢ · (yᵢ - Xᵢ·β)²
```

This is equivalent to transforming to `√w · y = √w · X · β` and solving standard OLS on the transformed system.

## LOESS Weighting Interaction with Bootstrap Weights

The LOESS trend fitting combines two weighting systems:

1. **Bootstrap observation weights** (from country × year sampling): control which observations are "in" the bootstrap sample
2. **Tricube proximity weights** (standard LOESS kernel): control local smoothing based on distance from evaluation point

At each evaluation point t₀, the combined weight for observation j is:

```
w_combined(j) = w_bootstrap(j) × tricube(|t_j - t₀| / bandwidth)
```

where `tricube(u) = (1 - u³)³` for u < 1, and 0 otherwise.

**Consequences**:
- **Unsampled years** (w_bootstrap = 0): contribute zero combined weight, so they do not influence the LOESS fit at any evaluation point
- **Years sampled multiple times** (w_bootstrap > 1): have proportionally more influence, equivalent to having multiple copies of that observation
- **LOESS evaluates at all years**: including unsampled ones, by interpolating from nearby sampled years using the kernel-weighted local polynomial. With a ~42-year bandwidth and ~62 years of data, there are always sufficient sampled years within the window for a stable fit
- **Result**: T_trend is estimated using only bootstrap-sampled data but is defined (finite) at all time points, which is exactly what the accumulators require

This satisfies the design requirement: **trends are fitted on the bootstrap sample but evaluated for all years**, providing a complete T_trend time series for the accumulator chain.

## NaN Handling Strategy

When `compute_year_means_weighted` encounters a year with zero total weight (unsampled year), it returns NaN for that year's mean. This propagates into the dependent variable:

```
y(i) = Δy(i) - k(yr) - j_trend(i)
```

For observations in unsampled years, k(yr) = NaN, so y(i) = NaN.

### The Problem

In IEEE 754 floating point arithmetic, `NaN × 0 = NaN`. When computing the weighted OLS inside the SSE function:

```python
y_w = y * sqrt(weights)    # NaN * 0 = NaN for unsampled years
```

This NaN propagates through `lstsq`, producing NaN coefficients and NaN SSE values. The h₄ optimizer cannot evaluate the objective function and converges to a fixed point determined by the algorithm's internal logic rather than the data.

### The Solution

Before the weighted OLS computation, replace NaN with 0 for zero-weight observations:

```python
y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
```

This is mathematically safe because zero-weight observations contribute nothing to the weighted OLS objective. The replacement value (0) is irrelevant — any finite value would give the same result since it is multiplied by weight 0.

This pattern is used consistently across all weighted fitting functions:
- `fit_ols_weighted` (src/fitting.py)
- `fit_linear_trend_weighted` (src/detrending.py)
- `fit_quadratic_trend_weighted` (src/detrending.py)
- `fit_loess_continuous_weighted` (src/detrending.py)
- `compute_sse_for_h4` closures in DL and DP fitting functions (src/fitting.py)
