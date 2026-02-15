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

All approaches use a consistent naming scheme for output coefficients:

### Universal Parameters

| Parameter | Meaning |
|-----------|---------|
| h1 | Linear temperature response coefficient |
| h2 | Quadratic temperature response coefficient |
| T_opt | Optimal temperature (where h(T) is maximized) |
| k(t) | Year fixed effects |
| r_squared | R² on detrended residuals |
| total_r_squared | R² on original Δy |

### Approach-Specific Parameters

**Approaches 0, 5, 5a-c, 7a-c** (standard quadratic): h1, h2, T_opt only

**Approach 5d** (GDP-dependent response):

| Parameter | Meaning |
|-----------|---------|
| f1 | GDP scaling exponent |
| f2 | Reference GDP level |

**Approaches 6a/6b/6d/6e** (separate T/departure responses):

| Parameter | Meaning |
|-----------|---------|
| h1, h2 | Response to actual temperature T |
| h3, h4 | Response to departure from trend (T - T_trend) |
| T_opt | Optimal actual temperature |
| T_dep_opt | Optimal departure from trend |

Note: 6d uses only h3 (linear departure), 6e uses only h4 (quadratic departure).

**Approach 6c** (departure/trend decomposition):

| Parameter | Meaning |
|-----------|---------|
| h1, h2 | Response to departure from trend (T - T_trend) |
| h3, h4 | Response to trend temperature T_trend |
| T_dep_opt | Optimal departure from trend |
| f2 | Optimal trend temperature |

**Approach 8** (piecewise quadratic):

| Parameter | Meaning |
|-----------|---------|
| h1 | Linear term (fixed at 0) |
| h2 | Curvature below T_opt |
| h4 | Curvature above T_opt |
| T_opt | Breakpoint temperature |

**Approach 8a** (shared T_opt, separate curvatures):

| Parameter | Meaning |
|-----------|---------|
| h2 | Curvature for actual T response |
| h4 | Curvature for trend T response |
| T_opt | Shared optimal temperature |

**Approach 8b** (modulated actual temperature response):

| Parameter | Meaning |
|-----------|---------|
| f1 | Linear departure modulation coefficient |
| f2 | Quadratic departure modulation coefficient |
| h1, h2 | Response to actual temperature T |
| T_opt | Optimal actual temperature |

**Approach 8c** (linear-only modulated response):

| Parameter | Meaning |
|-----------|---------|
| f1 | Linear departure modulation coefficient |
| h1, h2 | Response to actual temperature T |
| T_opt | Optimal actual temperature |

**Approach 8d** (quadratic-only modulated response):

| Parameter | Meaning |
|-----------|---------|
| f2 | Quadratic departure modulation coefficient |
| h1, h2 | Response to actual temperature T |
| T_opt | Optimal actual temperature |

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

## Approaches

### Approach 0: Conjoined OLS Fit

Fit all terms in a single (conjoined) ordinary least squares fitting process. With this approach, it is unclear how much `jᵢ(t)` is acting to detrend temperature versus per capita GDP growth rates, and how much of the error term is being absorbed into these many additional degrees of freedom.

```
Δyᵢ(t) = h₁·T + h₂·T² + j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² + kₜ
```

Note: One could add any arbitrary quadratic in time to `k(t)` and subtract the same quadratic from all of the `jᵢ(t)`s. Therefore, three additional constraints must be added. Without loss of generality, we set `j₁(t) = 0` for the first country.

**Degrees of freedom:** 2 for h(T) + 3×(n_countries - 1) for jᵢ(t) + n_years for k(t)

### Approach 1: Combined Detrending (Mixed)

Combines linear temperature detrending with quadratic GDP growth detrending. If the purpose of `jᵢ(t)` is to effect both a linear detrending of the temperature record and a quadratic detrending of the per-capita GDP growth record:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 2: Linear Temperature Detrending

We know that at least `j₀,ᵢ` and `j₁,ᵢ` relate to the temperature scale, because their values would differ if temperature was measured in Celsius versus Kelvin. A natural assumption is that at least part of `jᵢ(t)` is meant to represent a linear detrending of temperature.

Pre-compute `T₀,ᵢ` and `T₁,ᵢ` for each country via least squares on the linear temperature trend, then estimate:

```
Δyᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t)] + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 3: Quadratic GDP Growth Detrending

Another interpretation of `jᵢ(t)` is that it represents a quadratic detrending of the `Δyᵢ(t)` values.

Pre-compute `y₀,ᵢ`, `y₁,ᵢ`, and `y₂,ᵢ` for each country via least squares on the quadratic per capita GDP-growth trend, then estimate:

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·T + h₂·T² + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 4: Combined Quadratic Detrending

Applies quadratic detrending to both per capita GDP growth and temperature. When combined with the 2nd order term in h(T), this results in `jᵢ(t)` becoming a 4th-order equation.

```
Δyᵢ(t) - (y₀,ᵢ + y₁,ᵢ·t + y₂,ᵢ·t²) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)]
                                      + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)²] + kₜ
```

**Degrees of freedom:** 2 for h(T) + n_years for kₜ

### Approach 5: Pre-computed k(t) with Quadratic Trends

In Approaches 0-4, year fixed effects kₜ are estimated simultaneously with the temperature coefficients. An alternative is to pre-compute kₜ as year means before fitting.

1. Pre-compute year effects: `k(t) = mean_i(Δyᵢ(t))`
2. Fit country trends jᵢ(t) = j₀,ᵢ + j₁,ᵢ·t + j₂,ᵢ·t² to `Δyᵢ(t) - k(t)`
3. Fit temperature trends T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t² (quadratic)
4. Final regression on residuals:

```
[Δyᵢ(t) - k(t)] - jᵢ(t) = h₁·[T - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)]
                         + h₂·[T² - (T₀,ᵢ + T₁,ᵢ·t + T₂,ᵢ·t²)²]
```

**Degrees of freedom:** 2 for h(T) (year effects pre-computed, not estimated)

#### Approach 5 Variants

Several variants explore different detrending combinations:

| Variant | GDP Trend | Temp Trend | Description |
|---------|-----------|------------|-------------|
| **5** | Quadratic | Quadratic | Full quadratic detrending |
| **5a** | Quadratic | Linear | Linear temperature only |
| **5b** | Quadratic | None | GDP detrending only |
| **5c** | Quadratic | Linear+Quadratic | Combined approach |
| **5d** | Quadratic | Quadratic + GDP-scaling | GDP-dependent response |

**Approach 5d** introduces GDP-dependent scaling:
```
h(Y,T) = (Y/f₂)^(-f₁) · (h₁·T* + h₂·T*²)
```
where f₁ is the GDP scaling exponent (larger f₁ = stronger income-based adaptation) and f₂ is the reference GDP level.

### Approach 6: LOESS Detrending (Departure Response)

Replaces polynomial detrending with LOESS (Locally Weighted Scatterplot Smoothing), a non-parametric method that allows for more flexible trend shapes.

**GDP detrending** (same for all Approach 6 variants):
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

**Key insight:** This approach regresses GDP growth on the *departure of the climate response from its trend*, i.e., `h(T) - h(T_trend)`. It measures how GDP responds to year-to-year fluctuations in the climate effect, not to the absolute temperature level.

Uses a 25-year LOESS window (configurable via `--loess-window`).

**Degrees of freedom:** 2 for h(T) (year effects pre-computed)

### Approach 6a: LOESS Detrending (T + Departure Response)

Combines both actual temperature and departure effects, nesting Approaches 6 and 6b as special cases.

**GDP detrending** (same as Approach 6):
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function:**
```
h(T, T_trend) = h₁·T + h₂·T² + h₃·(T - T_trend) + h₄·(T - T_trend)²
```

**Regression model:**
```
Δy*(t) = h(T, T_trend)
```

The regression is performed on the design matrix `[T, T², (T - T_trend), (T - T_trend)²]`.

**Parameters:**
- `h₁, h₂`: Actual temperature response coefficients (response to T)
- `h₃, h₄`: Departure response coefficients (response to T - T_trend)
- `T_opt`: Optimal actual temperature = -h₁/(2h₂)
- `T_dep_opt`: Optimal departure = -h₃/(2h₄)

**At Trend:** When T = T_trend (i.e., departure = 0), the climate effect is:
```
h(T, T) = h₁·T + h₂·T²
```

**Key insight:** This approach allows GDP to respond to both the absolute temperature level (via h₁, h₂) and to temperature anomalies (via h₃, h₄). If h₃ = h₄ = 0, this reduces to Approach 6b. If h₁ = h₃ and h₂ = h₄, this is equivalent to Approach 6.

**Degrees of freedom:** 4 for h(T) (h₁, h₂, h₃, h₄)

### Approach 6b: LOESS Detrending (Actual T Response)

Uses the same GDP detrending as Approach 6, but regresses on actual temperature rather than temperature departures.

**GDP detrending** (same as Approach 6):
```
Δy*(t) = Δyᵢ(t) - k(t) - LOESS(Δyᵢ - k)
```

**Climate response function:**
```
h(T) = h₁·T + h₂·T²
```

**Regression model:**
```
Δy*(t) = h(T)
```

The regression is performed on the design matrix `[T, T²]` using actual temperature, not detrended temperature.

**Key insight:** This approach regresses GDP growth on *actual temperature*, testing whether the absolute temperature level affects growth. Unlike Approach 6, this assumes there is no separate effect from temperature trends—only the current year's temperature matters.

**Comparison to Approach 6:**
| Aspect | Approach 6 | Approach 6b |
|--------|------------|-------------|
| GDP detrending | LOESS | LOESS (same) |
| Independent variable | T - T_trend, T² - T_trend² | T, T² |
| Measures response to | Departures from climate trend | Actual temperature level |

**Degrees of freedom:** 2 for h(T) (h₁, h₂)

### Approach 6c: Departure/Trend Decomposition

Extends Approach 6 by decomposing the temperature response into separate responses to the departure from trend (T - T_trend) and to the trend temperature itself (T_trend).

```
h(T, T_trend) = h₁·(T - T_trend) + h₂·(T - T_trend)² + h₃·T_trend + h₄·T_trend²
```

where:
- `h₁·(T - T_trend) + h₂·(T - T_trend)²` — response to departure from trend
- `h₃·T_trend + h₄·T_trend²` — response to trend temperature
- `Δy*(t) = Δy(t) - k(t) - LOESS(Δy - k)` — same detrending as Approach 6

**Parameters:**
- `h₁, h₂`: Departure response coefficients (response to T - T_trend)
- `h₃, h₄`: Trend response coefficients (response to T_trend)
- `T_dep_opt`: Optimal departure from trend = -h₁/(2h₂)
- `f₂`: Optimal trend temperature = -h₃/(2h₄)

**At Trend:** When T = T_trend (i.e., departure = 0), the climate effect is:
```
h(T, T) = h₃·T_trend + h₄·T_trend²
```

**Interpretation:** This decomposition separates how economies respond to year-to-year temperature fluctuations (departures from trend) versus the underlying temperature level (trend). Unlike Approach 6a which uses actual temperature T and departure, Approach 6c cleanly separates the trend and fluctuation components.

**Degrees of freedom:** 4 for h(T) (h₁, h₂, h₃, h₄)

### Approach 6d: T Response with Linear Departure Only

A restricted version of Approach 6a that includes only the linear departure term, testing whether temperature fluctuations have a directional effect.

```
h(T, T_trend) = h₁·T + h₂·T² + h₃·(T - T_trend)
```

where:
- `h₁·T + h₂·T²` — response to actual temperature
- `h₃·(T - T_trend)` — linear response to departure from trend
- `Δy*(t) = Δy(t) - k(t) - LOESS(Δy - k)` — same detrending as Approach 6

**Parameters:**
- `h₁, h₂`: Actual temperature response coefficients (response to T)
- `h₃`: Linear departure coefficient
- `T_opt`: Optimal actual temperature = -h₁/(2h₂)

**Interpretation:** Tests whether warmer-than-trend and cooler-than-trend years have asymmetric effects on growth. A positive h₃ means warmer-than-trend years boost growth (or cooler-than-trend years reduce it). A negative h₃ means the opposite.

**Degrees of freedom:** 3 for h(T) (h₁, h₂, h₃)

### Approach 6e: T Response with Quadratic Departure Only

A restricted version of Approach 6a that includes only the quadratic departure term, testing whether the magnitude of temperature fluctuations matters regardless of direction.

```
h(T, T_trend) = h₁·T + h₂·T² + h₄·(T - T_trend)²
```

where:
- `h₁·T + h₂·T²` — response to actual temperature
- `h₄·(T - T_trend)²` — quadratic response to departure from trend
- `Δy*(t) = Δy(t) - k(t) - LOESS(Δy - k)` — same detrending as Approach 6

**Parameters:**
- `h₁, h₂`: Actual temperature response coefficients (response to T)
- `h₄`: Quadratic departure coefficient
- `T_opt`: Optimal actual temperature = -h₁/(2h₂)
- `T_dep_opt`: Optimal departure = 0 (by construction, since h₃=0)

**Interpretation:** Tests whether any deviation from trend temperature—warmer or cooler—affects growth symmetrically. A negative h₄ means larger fluctuations reduce growth regardless of direction (volatility is harmful). A positive h₄ means larger fluctuations boost growth (variability is beneficial).

**Degrees of freedom:** 3 for h(T) (h₁, h₂, h₄)

### Approach 8: Piecewise Quadratic Response with LOESS

Uses a piecewise quadratic temperature response that allows different curvatures for temperatures above vs below the optimum. This captures the asymmetry where warming may have different effects on hot vs cold countries.

**Model:**
```
h(T) = h₄ · (T - T_opt)²   if T > T_opt
h(T) = h₂ · (T - T_opt)²   if T ≤ T_opt
```

**Parameters:**
- `T_opt`: Optimal temperature (breakpoint where h(T) = 0)
- `h₂`: Curvature for cold countries (T ≤ T_opt)
- `h₄`: Curvature for hot countries (T > T_opt)

**Fitting uses the detrended formulation:**
```
[Δyᵢ(t) - k(t)] - LOESS(Δyᵢ - k) = h(T) - h(T_trend)
```

**Optimization strategy:**
1. Outer optimization: Search for T_opt using L-BFGS-B
2. Inner OLS: For each T_opt, solve for h₂ and h₄ via 2-column OLS

**Interpretation:**
- Both h₂ and h₄ should be negative (growth decreases away from optimum)
- If h₂ ≈ h₄, model reduces to symmetric quadratic
- If |h₄| > |h₂|, warming hurts hot countries more than cooling hurts cold countries

**Degrees of freedom:** 3 (T_opt, h₂, h₄)

### Approach 8a: Separate Total/Trend Response with Shared T_opt

Combines the total/trend separation of Approach 6a with the piecewise quadratic structure of Approach 8. Uses separate curvature parameters for total temperature (T) and trend temperature (T_trend), but with a shared optimal temperature.

```
Δy*(t) = h₂·(T - T_opt)² - h₄·(T_trend - T_opt)²
```

where:
- `h₂`: Curvature for total (actual) temperature response
- `h₄`: Curvature for trend temperature response
- `T_opt`: Shared optimal temperature for both components

**Total Response:** When T = T_trend (i.e., T_delta = 0), the total climate effect is:
```
(h₂ - h₄)·(T - T_opt)²
```

**Key features:**
- Both functions share the same optimal temperature
- Linear terms (h₁) are implicitly zero (piecewise quadratic centered at T_opt)
- Allows different sensitivities to total vs trend temperature

**Interpretation:** Tests whether the curvature of the temperature-growth relationship differs between total temperature and trend temperature, while maintaining a common optimal temperature.

**Degrees of freedom:** 3 (T_opt, h₂, h₄)

### Approach 8b: Modulated Actual Temperature Response

Uses a linear and quadratic modulation where the temperature response to actual temperature is scaled by the deviation from trend temperature. This captures effects that depend on both the direction and magnitude of deviation from trend.

**Model:**
```
h(T, Ttrend) = (1 + f₁ · (T - Ttrend) + f₂ · (T - Ttrend)²) · (h₁ · T + h₂ · T²)
```

**Parameters:**
- `f₁`: Linear departure modulation coefficient
- `f₂`: Quadratic departure modulation coefficient
- `h₁, h₂`: Standard quadratic temperature response coefficients (applied to actual temperature)
- `T_opt`: Optimal actual temperature = -h₁/(2·h₂)

**Interpretation:**
- When T = Ttrend, the modulation is 1 and h = h₁·T + h₂·T² (standard quadratic response)
- Linear term f₁ captures asymmetric effects (warmer vs cooler deviations have different effects)
- Quadratic term f₂ captures symmetric effects (magnitude of deviation matters regardless of sign)
- If f₁ = f₂ = 0, the model reduces to the standard h₁·T + h₂·T² response

**Fitting Strategy:**
- Outer optimization: 2D L-BFGS-B search over (f₁, f₂)
- Inner OLS: For each (f₁, f₂), solve for h₁ and h₂ via 2-column OLS

**Degrees of freedom:** 4 (f₁, f₂, h₁, h₂)

### Approach 8c: Linear-Only Modulated Response

Like Approach 8b but without the quadratic modulation term (f₂ = 0). The temperature response is modulated only by a linear function of the deviation from trend.

**Model:**
```
h(T, Ttrend) = (1 + f₁ · (T - Ttrend)) · (h₁ · T + h₂ · T²)
```

**Parameters:**
- `f₁`: Linear departure modulation coefficient
- `h₁, h₂`: Standard quadratic temperature response coefficients (applied to actual temperature)
- `T_opt`: Optimal actual temperature = -h₁/(2·h₂)

**Interpretation:**
- When T = Ttrend, the modulation is 1 and h = h₁·T + h₂·T² (standard quadratic response)
- Positive f₁ means the climate effect is amplified when T > Ttrend and dampened when T < Ttrend
- Negative f₁ means the opposite: amplified when cooler than trend, dampened when warmer
- If f₁ = 0, the model reduces to the standard h₁·T + h₂·T² response

**Fitting Strategy:**
- Outer optimization: 1D L-BFGS-B search over f₁
- Inner OLS: For each f₁, solve for h₁ and h₂ via 2-column OLS

**Degrees of freedom:** 3 (f₁, h₁, h₂)

### Approach 8d: Quadratic-Only Modulated Response

Like Approach 8b but without the linear modulation term (f₁ = 0). The temperature response is modulated only by a quadratic function of the deviation from trend.

**Model:**
```
h(T, Ttrend) = (1 + f₂ · (T - Ttrend)²) · (h₁ · T + h₂ · T²)
```

**Parameters:**
- `f₂`: Quadratic departure modulation coefficient
- `h₁, h₂`: Standard quadratic temperature response coefficients (applied to actual temperature)
- `T_opt`: Optimal actual temperature = -h₁/(2·h₂)

**Interpretation:**
- When T = Ttrend, the modulation is 1 and h = h₁·T + h₂·T² (standard quadratic response)
- Positive f₂ means the climate effect is amplified for any deviation from trend (warmer or cooler)
- Negative f₂ means the climate effect is dampened for any deviation from trend
- The effect is symmetric: |T - Ttrend| = 1°C has the same modulation whether warmer or cooler
- If f₂ = 0, the model reduces to the standard h₁·T + h₂·T² response

**Fitting Strategy:**
- Outer optimization: 1D L-BFGS-B search over f₂
- Inner OLS: For each f₂, solve for h₁ and h₂ via 2-column OLS

**Degrees of freedom:** 3 (f₂, h₁, h₂)

### Null Models (No Climate Response)

Two null models for comparison:

| Model | Description |
|-------|-------------|
| **nocr0** | Joint OLS with country trends and year effects, but h₁=h₂=0 |
| **nocr5** | Precomputed k with country trends, but h₁=h₂=0 |

## Imbalance Metrics

### The Consistency Equation

If the choice of smoothing function and the determination of the climate response function were perfect, the following equation would be satisfied exactly:

```
0 = h(f_trend(Tᵢ(t))) + f_trend(Δyᵢ(t) - k(t)) + k(t)
```

where:
- `h(T) = h₁·T + h₂·T²` is the climate response function
- `f_trend(T)` is the temperature trend (linear, quadratic, or LOESS depending on approach)
- `f_trend(Δy - k)` is the country-specific GDP growth trend (after removing year means)
- `k(t)` is the year mean

In practice, the right side will not sum to zero. The root-mean-square of this imbalance provides a diagnostic metric.

### Three Metrics

For each approach, we compute:

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
--use-csv PATH     Use pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--maddison PATH    Path to Maddison GDP Excel file
--cru PATH         Path to CRU temperature CSV file
--year-min YEAR    Minimum year to include
--year-max YEAR    Maximum year to include
--output-dir DIR   Output directory (default: timestamped directory in data/output/)
--loess-window N   Window size in years for LOESS smoothing (default: 25)
```

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

The `run_bootstrap.py` script performs country-level cluster bootstrap resampling to compute confidence intervals for all parameters across all approaches.

```bash
python scripts/run_bootstrap.py
```

**What it does:**
1. Resamples countries with replacement (cluster bootstrap preserves within-country correlation)
2. Re-fits all approaches for each bootstrap iteration
3. Computes percentile-based confidence intervals (90% CI, IQR)
4. Generates distribution plots and summary statistics

**Options:**
```
--n-bootstrap N     Number of bootstrap iterations (default: 1000)
--random-seed SEED  Random seed for reproducibility (default: 42)
--use-csv PATH      Pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv)
--year-min YEAR     Minimum year to include
--year-max YEAR     Maximum year to include
--output-dir DIR    Output directory (default: timestamped)
--loess-window N    LOESS window size (default: 25)
--quiet             Suppress progress messages
```

**Example:**
```bash
python scripts/run_bootstrap.py --n-bootstrap 1000 --output-dir results/bootstrap
```

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
--approaches LIST      Approaches to analyze (default: all)
--coefficients LIST    Coefficients to analyze (default: approach-specific)
--percentiles LIST     Percentile thresholds (default: 5 25 75 95)
--regression-type      "linear" (default)
--n-top N              Number of top/bottom countries to report (default: 10)
```

**Default coefficients by approach:**
| Approach | Coefficients |
|----------|--------------|
| Standard (0-5, 5a-5d, 6, nocr0, nocr5) | h₁, h₂, T_opt |
| Approach 6a/6b | h₁, h₂, h₃, h₄, T_opt, T_dep_opt |
| Approach 6c | h₁, h₂, h₃, h₄, T_dep_opt, f₂ |
| Approach 6d | h₁, h₂, h₃, T_opt |
| Approach 6e | h₁, h₂, h₄, T_opt, T_dep_opt |
| Approach 8 | h₂, h₄, T_opt |
| Approach 8a | h₂, h₄, T_opt |

**Example:**
```bash
python scripts/run_influence_analysis.py --approaches "approach5 approach6" --n-top 15
```

**Interpretation:**
- Countries with **positive** influence coefficients tend to **increase** the parameter when included more frequently
- Countries with **negative** influence coefficients tend to **decrease** the parameter when included more frequently
- Large absolute coefficients indicate high sensitivity to that country's inclusion

### Approach 0 vs 5c Parameter Comparison

The `compare_approach0_5c.py` script generates scatter plots comparing parameters from Approach 0 (conjoined OLS) against parameters predicted from Approach 5c's pre-computed trends combined with Approach 0's h₁ and h₂.

```bash
python scripts/compare_approach0_5c.py
```

**What it does:**
1. Fits Approach 0 and Approach 5c to the same data
2. Extracts per-country j coefficients from Approach 0 residuals
3. Predicts j coefficients from Approach 5c's pre-computed trends (g, T₀, T₁) using Approach 0's h₁, h₂
4. Re-references by subtracting country 0's predicted j values (matching Approach 0's identification constraint), absorbing the subtracted quadratic into k
5. Generates a 2×2 scatter plot: (a) k(t), (b) j₀,ᵢ, (c) j₁,ᵢ, (d) j₂,ᵢ with best-fit regression equations, R², and correlation

**Options:**
```
--data-file PATH   Input CSV file (default: data/input/Maddison_CRU_dataset.csv)
--output-dir DIR   Output directory (default: data/output/approach0_vs_5c)
```

**Outputs:**
- `approach0_vs_5c_scatter.pdf` — 2×2 scatter plot with 1:1 reference and best-fit lines
- `approach0_vs_5c_scatter_data.csv` — Underlying scatter data for all panels
- `approach0_vs_5c_derivation.tex` — LaTeX derivation of the equations used

For a detailed step-by-step mathematical derivation of this analysis, see [METHODS_DETAIL.md](METHODS_DETAIL.md).

## Output Files

Results are saved to a timestamped directory in `data/output/`. Files include:

### Main Analysis Outputs

| File | Description |
|------|-------------|
| `comparison_summary.txt` | Text summary of all approaches |
| `comparison_table.csv` | Tabular comparison of coefficients and fit statistics |
| `comparison_table.xlsx` | Same as above in Excel format (includes variance decomposition) |
| `country_trends.csv` | Country-level trend coefficients |
| `temperature_response_*.pdf` | Plot of h(T) - h(T_opt) for each approach group |
| `temperature_derivative_*.pdf` | Plot of dh/dT for each approach group |
| `coefficient_comparison.pdf` | Bar chart comparing h₁ and h₂ across approaches |
| `optimal_temperature_comparison.pdf` | Bar chart of optimal temperatures |
| `year_effects.pdf` | Year fixed effects plot |
| `residuals_*.pdf` | Residual diagnostic plots for each approach |

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

### Country Influence Outputs

| File | Description |
|------|-------------|
| `country_influence_coefficients.csv` | Full regression coefficients for all approach/coefficient/percentile combinations |
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
│   ├── fitting.py               # OLS regression for each approach
│   ├── bootstrap.py             # Cluster bootstrap resampling
│   ├── influence.py             # Country influence analysis
│   └── output.py                # Results tables and plots
├── scripts/
│   ├── run_analysis.py              # Main entry point
│   ├── run_bootstrap.py             # Bootstrap uncertainty analysis
│   ├── run_influence_analysis.py    # Country influence on bootstrap coefficients
│   ├── compare_approach0_5c.py         # Scatter plots comparing Approach 0 vs 5c parameters
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
