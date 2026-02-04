# Variance Decomposition Diagnostics

## Overview

This document explains the variance decomposition that quantifies the relative contributions of different model components to GDP growth variation. The decomposition is exact: the sum of all variance and covariance fractions equals 1.0 by construction.

## The Model

The general model structure is:

```
Δy_i(t) = [climate response] + j_i(t) + k(t) + ε
```

where:
- `Δy` = GDP growth rate (the dependent variable)
- `j(t)` = country-specific growth trend (function of time)
- `k(t)` = year fixed effects (common across countries)
- `ε` = residual (defined as the remainder)

The climate response is decomposed differently depending on whether temperature detrending is used.

## Mathematical Decomposition

### Approach 0 (no detrending, joint OLS)

```
Δy_i(t) = h(T) + j_i(t) + k(t) + ε
```

**4 components**: `h_T`, `j`, `k`, `epsilon`

where `h(T) = h1·T + h2·T²`.

### Approaches 1-7, 9 (with detrending)

Since T = T\* + T\_trend, and h is quadratic:

```
h(T) = h1·(T* + T_trend) + h2·(T* + T_trend)²
     = [h1·T* + h2·T*²] + [h1·T_trend + h2·T_trend²] + [2·h2·T*·T_trend]
     = h_Tstar + h_Ttrend + h_cross
```

So:
```
Δy_i(t) = h_Tstar + h_Ttrend + h_cross + j_i(t) + k(t) + ε
```

**6 components**: `h_Tstar`, `h_Ttrend`, `h_cross`, `j`, `k`, `epsilon`

### Approaches 8, 10 (GDP-dependent)

Same 6-component structure but with g = (Y/Y\_ref)^(-β) applied to the h terms:

```
Δy_i(t) = g·h_Tstar + g·h_Ttrend + g·h_cross + j_i(t) + k(t) + ε
```

**6 components**: `g_h_Tstar`, `g_h_Ttrend`, `g_h_cross`, `j`, `k`, `epsilon`

### Degenerate Cases

- **Approach 2** (no T detrending): T\_trend=0, so h\_Ttrend=0, h\_cross=0, h\_Tstar=h(T). Still uses 6-component structure for consistency.
- **Approach 1** (no GDP detrending): j=0. Still uses 6-component structure.

## Variance Identity

For any set of components C₁, C₂, ..., Cₙ, ε where Δy = C₁ + C₂ + ... + Cₙ + ε:

```
Var(Δy) = Σᵢ Var(Cᵢ) + 2·Σᵢ<ⱼ Cov(Cᵢ, Cⱼ)
```

Dividing by Var(Δy):

```
1 = Σᵢ [Var(Cᵢ)/Var(Δy)] + 2·Σᵢ<ⱼ [Cov(Cᵢ,Cⱼ)/Var(Δy)]
```

Since ε is defined as Δy minus the sum of all other components, all covariance terms (including those with ε) are included, and **the sum equals 1.0 exactly**.

## Output Keys

The `var_decomp` dictionary contains:

| Key pattern | Description |
|-------------|-------------|
| `component_names` | List of component names (including 'epsilon') |
| `var_{name}` | Var(component) / Var(Δy) |
| `cov_{name1}_{name2}` | 2·Cov(C1,C2) / Var(Δy) |
| `rms_{name}` | √(mean(component²)) |
| `rms_dy` | std(Δy) |
| `sum_check` | Sum of all var\_ and cov\_ terms (should be 1.0) |

## Component Definitions per Approach

| Approach | T\_trend source | j source | Component names |
|----------|----------------|----------|-----------------|
| 0 | N/A (4-comp) | Fitted j₀+j₁t+j₂t² | h\_T, j, k |
| 1 | T0+T1·t | zeros | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 2 | zeros | y0+y1·t+y2·t² | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 3 | T0+T1·t | y0+y1·t+y2·t² | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 4 | T0+T1·t | y0\_lin+y1\_lin·t | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 5 | T0q+T1q·t+T2q·t² | y0+y1·t+y2·t² | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 6 | T0+T1·t | y0\_lin+y1\_lin·t | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 7 | T0q+T1q·t+T2q·t² | y0+y1·t+y2·t² | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 8 | T0q+T1q·t+T2q·t² | y0+y1·t+y2·t² | g\_h\_Tstar, g\_h\_Ttrend, g\_h\_cross, j, k |
| 9 | T\_loess | y\_loess | h\_Tstar, h\_Ttrend, h\_cross, j, k |
| 10 | T\_loess | y\_loess | g\_h\_Tstar, g\_h\_Ttrend, g\_h\_cross, j, k |

## Key Variance Ratios

In addition to the full decomposition, a small set of intuitive "key variance ratios" are reported. Each is simply Var(component) / Var(Δy), giving a quick sense of how much variability each major piece carries.

| Ratio | What it measures | How computed |
|-------|-----------------|--------------|
| `var_ratio_h_T` | Climate response to **full** temperature | Var(h(T)) / Var(Δy), or Var(g·h(T)) for GDP-dependent |
| `var_ratio_h_Tstar` | Climate response to **detrended** temperature | Var(h(T\*)) / Var(Δy), or Var(g·h(T\*)) for GDP-dependent |
| `var_ratio_j` | Country-specific growth trends | Var(j) / Var(Δy) |
| `var_ratio_k` | Year fixed effects | Var(k) / Var(Δy) |
| `var_ratio_cross` | Covariance remainder | Total R² - var\_ratio\_h\_T - var\_ratio\_j - var\_ratio\_k |

### Additive identity

These ratios satisfy:

```
Total R² = var_ratio_h_T + var_ratio_j + var_ratio_k + var_ratio_cross
```

where `var_ratio_cross` captures all cross-covariance terms between components (and with epsilon). This follows from the variance identity Var(Δy) = Σ Var(Cᵢ) + 2·Σ Cov(Cᵢ, Cⱼ) and Total R² = 1 - Var(ε)/Var(Δy).

### How `var_ratio_h_T` differs from the detailed decomposition

`var_ratio_h_T` is computed by summing the h-component arrays (h\_Tstar + h\_Ttrend + h\_cross, or their g-scaled equivalents) and then taking the variance of the sum:

```
var_ratio_h_T = Var(h_Tstar + h_Ttrend + h_cross) / Var(Δy)
```

This is **not** the same as `var_h_Tstar + var_h_Ttrend + var_h_cross` (which ignores within-h covariances). It captures the total variance attributable to the full climate response h(T).

### Special cases

- **Approach 0** (no detrending): h(T) has no sub-components, so `var_ratio_h_T` = `var_ratio_h_Tstar` = `var_h_T`.
- **Approach 2** (no T detrending): T\*=T, T\_trend=0, so `var_ratio_h_T` = `var_ratio_h_Tstar`.

### Relationship to the full decomposition

The key ratios provide an intuitive R²-like decomposition at a coarser level (4 components plus cross terms), while the full decomposition breaks down every individual component and cross-covariance to sum to exactly 1.0. Both are reported in the output.

## Using Key Variance Ratios for Approach Selection

The key variance ratios can guide the choice between approaches by revealing how much of the estimated climate impact comes from identifiable variation vs. trend extrapolation, and how cleanly the decomposition separates components.

### Signal Fraction: `var_ratio_h_Tstar / var_ratio_h_T`

This ratio measures what fraction of the total climate response variance comes from identified (detrended) temperature variation rather than trend extrapolation.

- **Approach 0**: Signal fraction = 100% by construction (no detrending, so all temperature variation is used directly).
- **Approaches with T detrending**: Signal fraction is typically ~3–5%, meaning detrended temperature fluctuations account for only a small share of the total climate response variance.
- **Interpretation**: A low signal fraction means the estimated climate impact relies heavily on extrapolating the fitted h(T) function along the temperature trend — variation that is not separately identified from other slow-moving processes.

### Confounding Ratio: `|var_ratio_cross| / Total R²`

This ratio measures how much the decomposition components overlap or cancel each other, relative to the model's overall explanatory power.

- **Lower is better** — indicates cleaner separation between the climate response, country trends, and year effects.
- **High values** mean the decomposition is unstable: small changes in the detrending method could shift variance between components, making the attribution fragile.

### Red Flags

Watch for these patterns when comparing approaches:

- **`var_ratio_h_Tstar ≈ 0`**: No identifiable climate signal from detrended temperature. Approach 2 exhibits this because it does not detrend temperature (T\* = T, T\_trend = 0), so all climate response variance appears in `var_ratio_h_T` rather than being split.
- **`var_ratio_h_T >> 1`**: GDP scaling amplifies the trend component to unreasonable levels. Approaches 8 and 10 can exhibit this when the g = (Y/Y\_ref)^(−β) scaling magnifies climate response variance beyond the total GDP growth variance.
- **`var_ratio_cross` very large and negative**: Heavy cancellation between components — the individual pieces are large but offset each other, indicating the decomposition is poorly conditioned.

### What the Metrics Cannot Tell You

- These are **statistical diagnostics**, not causal identification tests. They describe variance shares, not whether the estimated relationship is causal.
- A "clean" decomposition (high signal fraction, low confounding ratio) does not prove the climate effect is causal — it only means the components are well-separated statistically.
- The choice between approaches ultimately depends on which **structural assumptions** (form of temperature detrending, GDP detrending, functional form) are most defensible for the research question at hand.

## Interpretation Notes

- **var\_h\_Tstar**: Variance from climate response to detrended temperature fluctuations. This is the "signal" - the part of temperature variation that climate response is identified from.
- **var\_h\_Ttrend**: Variance from climate response to the temperature trend. This captures the effect of long-term warming.
- **var\_h\_cross**: Variance from the cross-term between detrended temperature and temperature trend. This arises because h is quadratic and T = T\* + T\_trend.
- **var\_j**: Variance from country-specific growth trends.
- **var\_k**: Variance from year fixed effects.
- **var\_epsilon**: Variance from residuals.
- **Covariance terms**: Positive covariance means two components tend to move together; negative means they offset each other.
- **sum\_check**: Should be 1.000000 to machine precision.
