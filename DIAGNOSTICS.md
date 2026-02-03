# Component Contribution Diagnostics

## Overview

This document explains the diagnostic metrics added to quantify the relative contributions of different model components to GDP growth variation.

## The Model

The general model structure is:

```
dy = h(T) + j(t) + k(t) + ε
```

where:
- `dy` = GDP growth rate (the dependent variable)
- `h(T)` = climate response (function of temperature)
- `j(t)` = country-specific growth trend (function of time)
- `k(t)` = year fixed effects (common across countries)
- `ε` = residuals

## Current Implementation

### Option A: RMS Magnitudes

We compute root-mean-square values to measure the typical magnitude of each component:

| Metric | Formula | Description |
|--------|---------|-------------|
| `rms_h` | √(mean(h²)) | Climate response magnitude |
| `rms_j` | √(mean(j²)) | Country trend magnitude |
| `rms_k` | √(mean(k²)) | Year effect magnitude |
| `rms_dy` | std(dy) | Total variation in dy |

### Option C: Variance Decomposition

We decompose the variance of dy into contributions from each component:

| Metric | Formula | Description |
|--------|---------|-------------|
| `var_frac_h` | Var(h) / Var(dy) | Climate variance fraction |
| `var_frac_j` | Var(j) / Var(dy) | Country trend variance fraction |
| `var_frac_k` | Var(k) / Var(dy) | Year effect variance fraction |
| `var_frac_resid` | Var(ε) / Var(dy) | Residual variance fraction |
| `cov_frac_hj` | 2·Cov(h,j) / Var(dy) | h-j covariance contribution |
| `cov_frac_hk` | 2·Cov(h,k) / Var(dy) | h-k covariance contribution |
| `cov_frac_jk` | 2·Cov(j,k) / Var(dy) | j-k covariance contribution |

**Mathematical identity**: If `dy = h + j + k + ε`, then:

```
Var(dy) = Var(h) + Var(j) + Var(k) + Var(ε)
        + 2·Cov(h,j) + 2·Cov(h,k) + 2·Cov(j,k)
        + 2·Cov(h,ε) + 2·Cov(j,ε) + 2·Cov(k,ε)
```

So the fractions should sum to 1 **only if** the covariances with residuals are zero.

---

## How Each Component is Computed

### Climate Response h(T)

For non-GDP approaches (0-7, 9):
```python
h_values = h1 * data.temp + h2 * data.temp ** 2
```

For GDP-dependent approaches (8, 10):
```python
g = (data.pcGDP / Y_ref) ** (-beta)
h_values = g * (h1 * data.temp + h2 * data.temp ** 2)
```

**Key point**: We use **actual temperature** (`data.temp`), not detrended temperature.

### Country Trends j(t)

Varies by approach:
- Approach 1: `j = 0` (no GDP detrending)
- Approaches 2, 3, 5, 7, 8: `j = y0 + y1*t + y2*t²` (quadratic)
- Approaches 4, 6: `j = y0_lin + y1_lin*t` (linear)
- Approaches 9, 10: `j = y_loess` (LOESS smoothed)

### Year Effects k(t)

- Approaches 1-5: `k` from regression coefficients (year fixed effects)
- Approaches 6-10: `k` = precomputed year means

### Residuals ε

```python
residuals = y_regression - X @ beta_coefficients
```

where `y_regression` is the dependent variable used in the OLS fit.

---

## Why the Sum Doesn't Equal 1

### Issue 1: Missing Covariances with Residuals

The sum of our fractions equals 1 only if:
```
Cov(h,ε) = Cov(j,ε) = Cov(k,ε) = 0
```

This is true for **Approach 0** (Conjoined OLS Fit), where h, j, k are all estimated together in one regression, so residuals are orthogonal to all predictors.

For other approaches, the residuals may be correlated with our computed h, j, k because:
1. The regression was performed on transformed variables
2. We compute h using actual T, but the regression used detrended T

### Issue 2: GDP Scaling Creates Massive h Variance

For approaches 8 and 10, the GDP scaling factor `g = (Y/Y_ref)^(-β)` varies enormously:
- Poor countries: g ≈ 3-8
- Rich countries: g ≈ 0.3-0.6

This means `Var(h)` is dominated by variation in GDP, not temperature:
```
Var(h) = Var(g · f(T)) >> Var(dy)
```

Result: `var_frac_h ≈ 8`, which seems nonsensical as a "fraction".

---

## Questions to Resolve

1. **Should h be computed using actual temperature or detrended temperature?**
   - Current: actual temperature
   - Alternative: detrended temperature (T - T_trend)

2. **Should the GDP scaling be included in h for approaches 8 and 10?**
   - Current: yes, h = g · (h1·T + h2·T²)
   - Alternative: no, h = h1·T + h2·T² (same as other approaches)

3. **Should we include the missing covariance terms with residuals?**
   - Current: no
   - Alternative: add `cov_frac_h_resid`, `cov_frac_j_resid`, `cov_frac_k_resid`

4. **What question are we trying to answer?**
   - "How much of the variance in dy is explained by each component?"
   - "How much does each component contribute to the predicted values?"
   - "What is the relative magnitude of each component?"

---

## Example Output (Current Implementation)

### Approach 0 (Conjoined OLS Fit) - Sum = 1.00
```
var_frac_h     =  0.1752
var_frac_j     =  0.3151
var_frac_k     =  0.1219
var_frac_resid =  0.8038
cov_frac_hj    = -0.3384
cov_frac_hk    = -0.0046
cov_frac_jk    = -0.0729
Sum            =  1.0000  ✓
```

### Approach 7 (Precomputed k Quadratic) - Sum = 1.13
```
var_frac_h     =  0.1304
var_frac_j     =  0.1039
var_frac_k     =  0.0902
var_frac_resid =  0.8042
cov_frac_hj    = -0.0028
cov_frac_hk    =  0.0005
cov_frac_jk    = -0.0014
Sum            =  1.1250
```

### Approach 8 (GDP-Response Quadratic) - Sum = 8.95
```
var_frac_h     =  8.3931  ← Very large due to GDP scaling
var_frac_j     =  0.1039
var_frac_k     =  0.0902
var_frac_resid =  0.8026
cov_frac_hj    = -0.4784
cov_frac_hk    =  0.0394
cov_frac_jk    = -0.0014
Sum            =  8.9494
```

---

## Next Steps

Please review and let me know:
1. What interpretation of "component contribution" would be most useful?
2. Should we change how h is computed?
3. Should we add the missing covariance terms?
4. Any other concerns about the current approach?
