# Supplementary Methods: Explicit Detrending in Temperature–Growth Regressions

This document describes the statistical methods implemented in this repository at a level suitable for a **Supplementary Methods** section in an academic paper.

The code implements a family of closely related estimators (“approaches”) for a country–year panel regression of per‑capita GDP growth on annual mean temperature. The central theme is **making the detrending implicit in country‑specific time trends explicit**, and testing how sensitive estimated temperature response functions are to the detrending choice.

---

## 1. Data and variables

### 1.1 Panel structure and notation

- Countries are indexed by **i = 1,…,N**.
- Years are indexed by **t** (calendar year).
- The data form an unbalanced panel (countries may be missing interior years), but the analysis typically restricts to countries that appear in both the first and last year of the sample window to stabilize the country set.


In the bundled default input file (`data/input/Maddison_CRU_dataset.csv`), the sample spans **1961–2022** and includes **157 countries** (9404 country–year observations), though the code supports alternative year windows and datasets.

### 1.2 Dependent variable: per‑capita GDP growth

Let `pcGDPᵢ,t` denote per‑capita GDP (level). The dependent variable is the one‑year log difference:

```
Δyᵢ,t = log(pcGDPᵢ,t) − log(pcGDPᵢ,t−1).
```

This is approximately the annual percentage growth rate in per‑capita GDP.

### 1.3 Temperature

Let `Tᵢ,t` be annual mean near‑surface temperature in °C for country *i* in year *t*.

This repository’s estimation routines use **temperature only** (even if the input dataset includes additional climate variables).

### 1.4 Centered time index

For numerical stability in polynomial trend estimation, time is centered:

```
τᵢ,t = t − t̄,
```

where `t̄` is the midpoint of the sample’s first and last year (computed from the loaded dataset). The code stores `τ` as `data.time`.

---

## 2. Baseline regression structure

Across approaches, the starting point is an additive decomposition of growth into:

1. a **climate response** component,
2. a **country‑specific smooth component** (trend), and
3. a **global year effect** component.

A generic baseline specification is:

```
Δyᵢ,t = climate_responseᵢ,t + jᵢ(t) + k(t) + εᵢ,t.        (M1)
```

- `jᵢ(t)` is a country‑specific smooth function of time (typically a quadratic polynomial in τ, or a LOESS smoother).
- `k(t)` is a year effect shared across countries (a year fixed effect).
- `εᵢ,t` is a residual.

What differs across approaches is **(i)** how the climate response is parameterized, and **(ii)** whether `jᵢ(t)` and `k(t)` are estimated **jointly** with the climate response or **pre‑computed** via explicit detrending.

---

## 3. Climate response parameterizations

### 3.1 Quadratic temperature response

The baseline response function is quadratic in temperature:

```
h(T) = h₁·T + h₂·T².                                     (M2)
```

For `h₂ < 0`, this implies an “optimal” temperature:

```
T_opt = −h₁ / (2 h₂).
```

(Analytic standard errors for `T_opt` are not computed in the point‑estimate step; uncertainty is obtained via bootstrap.)

### 3.2 Piecewise quadratic response

The piecewise response allows different curvature below vs above a common optimum:

```
h(T; T_opt) =
  h₂_low · (T − T_opt)²          if T ≤ T_opt,
  h₂_high · (T − T_opt)²         if T > T_opt.           (M3)
```

By construction, `h(T_opt)=0`. (This is a convenient normalization; the model is identified through differences in growth, not levels.)

`T_opt` is estimated by outer optimization (Section 5.4).

### 3.3 Persistence / decay (“converging”) response

The persistence specification treats temperature effects as accumulating but decaying over time. Let `h(Tᵢ,t)` be the instantaneous quadratic response (M2). Define the **net annual contribution** as:

```
h_conv(t) = h(Tᵢ,t) − h₄ · Σ_{k=1}^{∞} (1 − h₄)^{k−1} · h(Tᵢ,t−k),   (M4)
```

where `h₄ ∈ [0,1]` controls the decay rate:
- `h₄ = 0` implies no persistence (reduces to the instantaneous quadratic response).
- larger `h₄` implies faster decay of past effects (stronger “convergence”).

The code implements (M4) efficiently via recursive accumulators (Section 5.5).

### 3.4 GDP‑dependent scaling of the response

The GDP‑scaled specifications allow the *magnitude* of the temperature response to depend on a country's per‑capita GDP **level** `pcGDPᵢ,t` (not its growth). The quadratic response is multiplied by a scaling factor:

```
g(pcGDPᵢ,t) = (pcGDPᵢ,t / Y_ref)^(−β),      Y_ref = median(pcGDP).      (M4b)
```

The exponent `β` is a freely estimated parameter (bounds `[0.001, 10]`); `Y_ref` is a fixed normalizer (the sample median of per‑capita GDP), computed once on the full dataset and held fixed across bootstrap resamples. Because `Y_ref` only rescales the multiplicative coefficients, its choice does not affect `β` or the fitted response; it keeps the coefficients well‑conditioned and interpretable. For `β > 0`, poorer countries (lower `pcGDP`) receive a larger response.

Two response shapes are used:

```
Free quadratic (G):     h(T, pcGDP) = g · (β₀ + β₁·T + β₂·T²),               (M4c)
Centered quadratic (C): h(T, pcGDP) = g · β₂ · (T − T_opt)²,                 (M4d)
Reference quadratic (R): h(T, pcGDP) = g · β₂ · (T − T_opt) · (T − Tref_i).  (M4e)
```

In (M4c) the vertex is `T_opt = −β₁ / (2 β₂)`, independent of `β₀` and of `g`. In (M4d) the response is zero at `T_opt` by construction, and `T_opt` is estimated directly (Section 5.7). In (M4e), `Tref_i` is the **mean temperature of country i across years**; the response is zero at both `T_opt` (a shared root) and the country's own mean temperature `Tref_i`, so each country's response is centered on its typical climate. At the reference GDP (`g = 1`) all three reduce to a quadratic in `T`; the plotted response curve is drawn at `Y_ref` (and, for R, at the global mean temperature `T̄` as the representative second root), and the GDP dependence is displayed separately as the scaling factor `g` versus `pcGDP`.

**Scaling scope (M/W).** In G/C/R the factor `g` multiplies the climate response *only*; the country trends `jᵢ(t)` and year effects `k_t` remain ordinary additive terms. Two further variants widen the scope of the GDP scaling (both use the free quadratic shape, with no `β₀` — the g‑scaled country intercept absorbs it):

```
Country model (M): Δyᵢ,t = g · (β₁·T + β₂·T² + jᵢ(t)) + k_t,          (M4f)
Whole model (W):   Δyᵢ,t = g · (β₁·T + β₂·T² + jᵢ(t) + k_t).          (M4g)
```

In (M4f) the GDP factor scales the country‑specific structural model (response **and** trend), while global year shocks enter additively. In (M4g) it scales the **entire** model, so year shocks also hit poorer countries more strongly (the stored `k_t` are shared‑shock coefficients, realized per observation as `g·k_t`). Estimation is detailed in Section 5.7.

**Log‑linear GDP dependence (I).** A distinct model replaces the power‑law factor with one that is *linear in log‑GDP*:

```
h(T, pcGDP) = s̃ · (β₀ + β₁·T + β₂·T²),   s̃ = 1 − log(pcGDP/Y_ref)/β.   (M4h)
```

`s̃` equals `(β − log(pcGDP/Y_ref))` up to an inner‑OLS rescaling, but is normalized so **s̃ = 1 at Y_ref** — so the fitted coefficients are the response at the reference GDP, directly comparable to GJ/QJ. Because `log(pcGDP/Y_ref)` is bounded (≈[−2.8, +3.3] on the default data), `s̃` cannot explode the way the power‑law factor does for low‑pcGDP observations. This is what makes the GDP‑dependent response identifiable on *pre‑detrended residuals* (Approaches IP/IL), where the power‑law form degenerates (β runs to a boundary and the response collapses; see Section 5.7). The response magnitude declines linearly with log‑income and crosses zero at `pcGDP = Y_ref·e^β`; `β → ∞` recovers the plain quadratic (no GDP dependence). Here `β` is a log‑income crossover, not the power‑law elasticity — the two are not comparable numbers.

---

## 4. Detrending components

Several approaches estimate the climate response using an explicit detrending workflow. This requires defining:
1. a year effect `k(t)`,
2. a country‑specific growth trend `jᵢ(t)`, and
3. a country‑specific temperature trend `T_trendᵢ,t`.

### 4.1 Year effect `k(t)`

For explicit detrending approaches, the year effect is computed as the (unweighted) cross‑country mean of growth in each year:


For **joint (conjoined) approaches** (QJ, PJ, DJ, NJ), `k(t)` is instead estimated as a set of **year dummy coefficients** in the joint regression. Because the model does not include a global intercept and imposes reference‑country constraints on `jᵢ(t)`, all years can be represented with dummies under that normalization.

```
k(t) = mean_i [ Δyᵢ,t ].                                 (M5)
```

In the time‑dimension bootstrap (Section 6.2), a **weighted** version is used:

```
k_w(t) = (Σ_i wᵢ,t Δyᵢ,t) / (Σ_i wᵢ,t),
```

with `k_w(t) = NaN` if the total weight in year *t* is zero.

### 4.2 Polynomial (parametric) trends

For the *polynomial detrending* approaches:

- **Temperature trend** (per country):
  fit a linear regression
  ```
  Tᵢ,t ≈ T₀,ᵢ + T₁,ᵢ · τᵢ,t,
  ```
  then define `T_trendᵢ,t = T₀,ᵢ + T₁,ᵢ τᵢ,t`.

- **Growth trend** (per country):
  compute residual growth after removing `k(t)`:
  ```
  gᵢ,t = Δyᵢ,t − k(t),
  ```
  then fit a quadratic regression
  ```
  gᵢ,t ≈ y₀,ᵢ + y₁,ᵢ · τᵢ,t + y₂,ᵢ · τᵢ,t²,
  ```
  and define `jᵢ(t) = y₀,ᵢ + y₁,ᵢ τ + y₂,ᵢ τ²`.

Weighted least squares analogues are used under year resampling.

### 4.3 LOESS (nonparametric) trends

For the *LOESS detrending* approaches, both the temperature trend and the growth trend are estimated with **local regression smoothing**.

For each country *i* and each target time point τ₀, LOESS solves a weighted local polynomial regression using tricube weights:

- Distance: `d = |τ − τ₀|`
- Normalized distance: `u = d / bandwidth`
- Tricube kernel:
  ```
  w(u) = (1 − u³)³    for u < 1, else 0.
  ```

A local polynomial (degree 1 by default) is fit using weights `w(u)`, and the fitted intercept at τ₀ is the smoothed value.

The bandwidth is specified in **years** via `--loess-window`. This choice controls the smoothness of `T_trendᵢ,t` and `jᵢ(t)`.

In the code:
- `T_trendᵢ,t` is `T_loessᵢ,t`,
- `jᵢ(t)` is `y_loessᵢ,t` applied to the series `Δyᵢ,t − k(t)`.

---

## 5. Estimation approaches

The code reports results using short approach names. For each, we describe the estimating equations and how parameters are obtained.


### 5.0 Approach summary

| Response family | Joint estimation | Explicit polynomial detrending | Explicit LOESS detrending |
|---|---:|---:|---:|
| Quadratic `h(T)=h₁T+h₂T²` | QJ | QP | QL |
| Piecewise quadratic | PJ | PP | PL |
| Persistence / decay | DJ | DP | DL |
| GDP‑scaled free quadratic `g·(β₀+β₁T+β₂T²)` | GJ | — | — |
| GDP‑scaled centered quadratic `g·β₂(T−T_opt)²` | CJ | — | — |
| GDP‑scaled reference quadratic `g·β₂(T−T_opt)(T−Tref_i)` | RJ | — | — |
| GDP‑scaled country model `g·(β₁T+β₂T²+jᵢ(t))+k_t` | MJ | — | — |
| GDP‑scaled whole model `g·(β₁T+β₂T²+jᵢ(t)+k_t)` | WJ | — | — |
| Log‑linear GDP quadratic `s̃·(β₀+β₁T+β₂T²)` | IJ | IP | IL |
| Null (no climate response) | NJ | NP | NL |

### 5.1 Quadratic, joint OLS: **Approach QJ**

**Model.** A joint fixed‑effects OLS regression:

```
Δyᵢ,t = h₁ Tᵢ,t + h₂ Tᵢ,t²
        + (j₀,ᵢ + j₁,ᵢ τᵢ,t + j₂,ᵢ τᵢ,t²)
        + k(t) + εᵢ,t.                                   (M6)
```

**Identification.** To avoid perfect multicollinearity, one country is treated as a reference country and its trend coefficients are set to zero:

```
j₀,ref = j₁,ref = j₂,ref = 0.
```

Year effects are represented by year dummy variables. The full parameter vector is estimated by OLS (or WLS in the year‑sampling bootstrap).

**Interpretation.** This is the “standard” conjoined estimation that simultaneously allocates variation among temperature, country trends, and year effects.

### 5.2 Quadratic, explicit polynomial detrending: **Approach QP**

This approach estimates `k(t)`, `jᵢ(t)`, and `T_trendᵢ,t` first, then estimates the climate response using detrended quantities.

**Step 1: compute year means** `k(t)` via (M5).

**Step 2: compute trends**
- compute `jᵢ(t)` as the quadratic trend in `(Δyᵢ,t − k(t))`,
- compute `T_trendᵢ,t` as the linear trend in temperature.

**Step 3: detrended regression.** Define the residual growth:

```
yᵢ,t = Δyᵢ,t − k(t) − jᵢ(t).                              (M7)
```

Define detrended temperature regressors:

```
X₁ᵢ,t = Tᵢ,t − T_trendᵢ,t
X₂ᵢ,t = Tᵢ,t² − T_trendᵢ,t².                              (M8)
```

Estimate by OLS (or WLS):

```
yᵢ,t = h₁ X₁ᵢ,t + h₂ X₂ᵢ,t + εᵢ,t.                         (M9)
```

Note that the right‑hand side equals:

```
h₁ (T − T_trend) + h₂ (T² − T_trend²) = h(T) − h(T_trend),
```

so this formulation estimates the climate response as a **difference between actual temperature and the smooth temperature trajectory**.

### 5.3 Quadratic, explicit LOESS detrending: **Approach QL**

Approach QL is identical in structure to QP, except that:

- `T_trendᵢ,t` is obtained via LOESS smoothing,
- `jᵢ(t)` is obtained via LOESS smoothing of `(Δyᵢ,t − k(t))`.

The regression is the same as (M9), with LOESS‑based trends.

### 5.4 Piecewise response: **Approaches PJ / PP / PL**

All piecewise approaches use response (M3), but differ in whether detrending is joint, polynomial, or LOESS.

#### Approach PJ (piecewise, joint)

Estimate:

```
Δyᵢ,t = h(Tᵢ,t; T_opt) + (country quadratic trend) + k(t) + εᵢ,t.     (M10)
```

`T_opt` is estimated by outer 1‑D optimization. For any candidate `T_opt`, the model is linear in `(h₂_low, h₂_high, j coefficients, year effects)`, so these are estimated by OLS/WLS, SSE is computed, and the optimizer updates `T_opt`.

#### Approach PP (piecewise, polynomial detrending)

Compute `k(t)`, `jᵢ(t)`, and `T_trendᵢ,t` as in QP. Define:

```
yᵢ,t = Δyᵢ,t − k(t) − jᵢ(t).                                         (M11)
```

Define the piecewise quadratic basis functions:

- For any temperature series `Z`:
  ```
  low(Z)  = (Z − T_opt)²  if Z ≤ T_opt, else 0
  high(Z) = (Z − T_opt)²  if Z > T_opt, else 0.
  ```

Then define detrended regressors:

```
X_low  = low(T)  − low(T_trend)
X_high = high(T) − high(T_trend).                                    (M12)
```

Estimate:

```
y = h₂_low · X_low + h₂_high · X_high + ε.                            (M13)
```

Again, the regression is linear in coefficients conditional on `T_opt`, enabling outer optimization over `T_opt`.

#### Approach PL (piecewise, LOESS detrending)

Same as PP, except the trend `T_trend` and `jᵢ(t)` are LOESS‑based.

**Standard errors for `T_opt`.** For piecewise approaches, the code computes an approximate standard error for `T_opt` from numerical curvature of the SSE profile at the optimum (finite‑difference second derivative), scaled by the residual variance.

### 5.5 Persistence/decay response: **Approaches DJ / DP / DL**

Persistence approaches use the distributed‑lag specification (M4) but implement it as a linear regression in modified regressors.

#### Accumulator representation

Define recursive accumulators for each country:

```
A_T(t)  = T(t)  + (1 − h₄) A_T(t−1)
A_T2(t) = T(t)² + (1 − h₄) A_T2(t−1).                                (M14)
```

Let `A_T_lag(t) = A_T(t−1)` (and similarly for `A_T2_lag`). Then the weighted sum in (M4) can be expressed using `A_T_lag` and `A_T2_lag`.

#### Pre‑history correction

Because each country’s time series begins at its first observed year, a pre‑sample assumption is required. The code applies a “constant pre‑history” correction term:

```
C_T(t)  = (1 − h₄)^{Δt} · T(first_year)
C_T2(t) = (1 − h₄)^{Δt} · T(first_year)²,                            (M15)
```

where `Δt` is years since the first observed year for that country.

- In **DJ/DP**, the correction uses a smoothed baseline temperature at the first year derived from a country‑specific linear regression in temperature.
- In **DL**, the correction uses the first observed temperature (typically the sample’s baseline year).

#### Approach DJ (persistence, joint)

Define modified regressors:

```
X₁ᵢ,t = Tᵢ,t   − h₄ A_T_lagᵢ,t  − C_Tᵢ,t
X₂ᵢ,t = Tᵢ,t² − h₄ A_T2_lagᵢ,t − C_T2ᵢ,t.                             (M16)
```

Estimate jointly:

```
Δyᵢ,t = h₁ X₁ᵢ,t + h₂ X₂ᵢ,t + (country quadratic trend) + k(t) + εᵢ,t.  (M17)
```

`h₄` is estimated by outer 1‑D optimization (bounded scalar minimization of SSE). For each candidate `h₄`, the model is linear in all other parameters.

#### Approach DP (persistence, polynomial detrending)

Compute `k(t)`, `jᵢ(t)`, and a linear `T_trendᵢ,t` as in QP. Define residual growth:

```
yᵢ,t = Δyᵢ,t − k(t) − jᵢ(t).                                           (M18)
```

Compute accumulator and correction terms for both observed temperature `T` and trend temperature `T_trend`, and define detrended regressors:

```
X₁ = (T − h₄ A_T_lag − C_T) − (T_trend − h₄ A_Ttrend_lag − C_Ttrend)
X₂ = (T² − h₄ A_T2_lag − C_T2) − (T_trend² − h₄ A_T2trend_lag − C_T2trend).  (M19)
```

Estimate:

```
y = h₁ X₁ + h₂ X₂ + ε.                                                 (M20)
```

As with DJ, `h₄` is estimated by outer 1‑D optimization with inner OLS/WLS for `(h₁,h₂)`.

#### Approach DL (persistence, LOESS detrending)

Same as DP, except `T_trend` is LOESS‑based and `jᵢ(t)` is LOESS‑based.

**Standard errors for `h₄`.** For persistence approaches, the code computes an approximate standard error for `h₄` from numerical curvature of the SSE profile at the optimum (finite‑difference second derivative), scaled by residual variance.

### 5.6 Null models: **Approaches NJ / NP / NL**

Null approaches fit the same detrending structure but impose **no climate response**:

- **NJ:** joint model with country trends and year effects only.
- **NP:** polynomial detrending of growth (`k(t)` and quadratic `jᵢ(t)`), no temperature regression.
- **NL:** LOESS detrending of growth, no temperature regression.

These provide baseline fit diagnostics and help assess how much explanatory power is coming from non‑climate components.

### 5.7 GDP‑scaled responses: **Approaches GJ / CJ / RJ / MJ / WJ / IJ / IP / IL**

These are joint fixed‑effects models (same country‑trend and year‑effect structure as QJ, M6) in which the climate response is multiplied by the per‑capita‑GDP scaling factor `g = (pcGDPᵢ,t / Y_ref)^(−β)` of Section 3.4. Only the climate columns are scaled by `g`; the country trends `jᵢ(t)` and year effects `k(t)` are unscaled.

**Approach GJ (free quadratic, M4c):**

```
Δyᵢ,t = gᵢ,t · (β₀ + β₁ Tᵢ,t + β₂ Tᵢ,t²)
        + (j₀,ᵢ + j₁,ᵢ τ + j₂,ᵢ τ²) + k(t) + εᵢ,t.        (M10)
```

**Approach CJ (centered quadratic, M4d):**

```
Δyᵢ,t = gᵢ,t · β₂ · (Tᵢ,t − T_opt)²
        + (j₀,ᵢ + j₁,ᵢ τ + j₂,ᵢ τ²) + k(t) + εᵢ,t.        (M11)
```

**Approach RJ (reference quadratic, M4e):**

```
Δyᵢ,t = gᵢ,t · β₂ · (Tᵢ,t − T_opt) · (Tᵢ,t − Tref_i)
        + (j₀,ᵢ + j₁,ᵢ τ + j₂,ᵢ τ²) + k(t) + εᵢ,t,        (M12)
```

where `Tref_i` is the mean temperature of country *i* across years (precomputed).

**Estimation.** The models are linear in the climate coefficients (`β₀,β₁,β₂` for GJ; `β₂` for CJ; see below for RJ) and the trend/year parameters given the *nonlinear* parameters, so estimation profiles the nonlinear parameters with an inner OLS:

- **GJ:** a bounded 1‑D search (Brent) over `β`; for each candidate `β`, an inner OLS solves `(β₀,β₁,β₂)` together with the country trends and year effects. `T_opt = −β₁/(2β₂)` is derived analytically.
- **CJ:** an alternating bounded 1‑D search over `β` and `T_opt` (as in the ternary/piecewise optimizers); for each candidate `(β, T_opt)`, an inner OLS solves the single climate coefficient `β₂` plus trends and year effects. The `T_opt` search window is wide (`[−30, +60] °C`) so the vertex is not forced into the `0–30 °C` display range.
- **RJ:** because `(T−T_opt)(T−Tref_i)` has **no `T_opt²` term**, the model is *linear* in `T_opt`. Writing `β₂(T−T_opt)(T−Tref_i) = β₂·(T−Tref_i)·T − β₂·T_opt·(T−Tref_i)`, an inner OLS over the two columns `g·(T−Tref_i)·T` and `g·(T−Tref_i)` recovers coefficients `a = β₂` and `b = −β₂·T_opt`; hence **only `β` is profiled** (one bounded Brent search) and `T_opt = −b/a` is derived, unbounded. This avoids the boundary‑pinning that the bounded CJ search can exhibit.

**Standard errors.** Climate‑coefficient SEs come from the inner OLS covariance at the optimum; the profiled `β` SE is obtained from the numerical curvature of the SSE profile. For CJ, `T_opt`'s SE also uses the SSE‑profile curvature; for RJ, `T_opt = −b/a` uses a delta‑method SE from the inner 2×2 covariance.

**Identification note.** In GJ the constant term is entered as the column `g` (i.e. `g·β₀`), which is a smooth function of country and year and can be weakly identified against the country trends `jᵢ(t)`; the response‑shape parameters (`β`, `β₁`, `β₂`, `T_opt`) are well identified, but `β₀` may carry a wide standard error.

**RJ curve and roots.** RJ's response is country‑specific (it is zero at each country's own mean temperature `Tref_i`), so it has no single `h(T)` curve. For display, the code draws a representative curve at the global mean temperature `T̄` (i.e. `β₂·(T−T_opt)·(T−T̄)` at `g = 1`); the reported `T_opt` is one *root* of the response, and the per‑country vertex lies at `(T_opt + Tref_i)/2`. `Tref_i` is invariant under country resampling and is held at the full‑sample country means under year‑weighted resampling.

**Reference GDP in the bootstrap.** `Y_ref = median(pcGDP)` is computed once on the full sample and reused for every resample, so bootstrap variation in `β` reflects sampling of countries/years, not a moving normalizer.

**Approaches MJ / WJ (scaling scope, M4f / M4g).** These widen the GDP factor from the response to the detrending structure. MJ scales the response **and** the country trend, `Δy = g·(β₁T + β₂T² + jᵢ(t)) + k_t`; WJ scales the **whole** model, `Δy = g·(β₁T + β₂T² + jᵢ(t) + k_t)`. Neither carries a `β₀` — the g‑scaled country intercept `v0ᵢ` plays that role, and country 0 is the dropped reference, so both have **one fewer free parameter than GJ**: `n_params = 2 + 3(n_countries − 1) + n_years + 1(β)`. Crucially, for a fixed `β` each model is still *linear in every remaining coefficient* (`β₁, β₂`, the country‑trend parameters, and the year effects), so estimation reuses the GJ machinery: a single bounded 1‑D Brent search over `β` with an inner OLS at each candidate. The *only* implementation difference from GJ is which columns of the design matrix are multiplied by `g` — the climate and country‑trend columns for MJ, and the entire design (including year effects) for WJ. The reported `(h₁, h₂, T_opt)` are the temperature‑shape at `Y_ref` (`g = 1`), so the standard quadratic plotting and bootstrap bands apply unchanged. For WJ, the stored `k_t` are the shared‑shock coefficients (before scaling); the realized per‑observation year effect is `g·k_t`, and that scaled version enters the variance decomposition.

**Identification (MJ / WJ).** Because `g = (pcGDP/Y_ref)^(−β)` varies within a country from year to year (per‑capita GDP changes annually), it is collinear with neither the country dummies/trends nor the year fixed effects; the profiled `β` and the response‑shape parameters are well identified even though the whole design is g‑scaled.

**Approaches IJ / IP / IL (log‑linear GDP dependence, M4h).** These fit `s̃·(β₀+β₁T+β₂T²)` with
`s̃ = 1 − log(pcGDP/Y_ref)/β`. Estimation profiles `β` with a bounded 1‑D search (`β ∈ [0.1, 50]`);
the inner OLS solves `(β₀,β₁,β₂)` — plus country trends and year effects for **IJ** (joint, like GJ),
or on the QP/QL residual `y = Δy − k − jᵢ(t)` for **IP** (polynomial) and **IL** (LOESS). Because
`s̃ = 1` at `Y_ref`, the fitted `(β₀,β₁,β₂)` are the response at the reference GDP, with clean inner‑OLS
SEs and `T_opt = −β₁/(2β₂)`. `n_params = n_climate(3) + 3(n−1) + n_years + 1` for IJ, and `4` for IP/IL.

The log‑linear form exists precisely because the **power‑law** GDP scaling is *not* identifiable on
pre‑detrended residuals: `(pcGDP/Y_ref)^(−β)` explodes for low‑pcGDP observations (reaching ~10¹² at
β=10), so once the country trends and year effects are removed — leaving no structure to compete for
that variance — β runs to the boundary and the response collapses. The bounded log‑linear scale avoids
this, giving a genuine interior optimum on residuals (β≈1.6 on the default data, T_opt≈19 °C, comparable
to GJ/QJ). **The asymmetry is exact and complementary:** power‑law dependence is identified in the joint
form (GJ) but degenerate on residuals (would‑be GP/GL); log‑linear dependence is identified on residuals
(IP/IL) but **not** in the joint form — **IJ**'s SSE(β) profile is nearly flat and β drifts to its upper
bound, collapsing to ≈QJ (the joint fit rejects the sign flip that log‑linear scaling imposes on the
richest countries, whereas the always‑positive power‑law shrink is accepted). IJ is therefore reported
for comparison only; its boundary β should be read as "no joint log‑linear dependence identified."

---

## 6. Fit statistics and diagnostics

For each approach, the code reports:


### 6.0 Analytic standard errors

For linear parameters estimated by ordinary least squares (OLS), the code reports classical homoskedastic standard errors based on:

```
Var(β̂) = σ̂² · (X'X)^{-1},
σ̂² = SSE / (n − p).
```

For weighted least squares (WLS) used in the time‑dimension bootstrap, the analogue is:

```
Var(β̂) = σ̂² · (X'WX)^{+},
```

where `W` is the diagonal matrix of observation weights and `(+ )` denotes a pseudoinverse (used to handle rank deficiency when some years receive zero total weight).

These are **not** heteroskedasticity‑robust or cluster‑robust standard errors. Uncertainty is therefore primarily summarized via the bootstrap distributions described in Section 7.

For outer‑optimized scalar parameters (`T_opt` in piecewise models; `h₄` in persistence models), the code computes an approximate standard error from the numerical curvature of the SSE profile at the optimum (finite‑difference second derivative), scaled by the residual variance.

- **RMSE** (root mean squared error)
- **R²** (in the regression’s estimation space)
  - For explicit detrending approaches, this is the R² of the detrended regression (e.g., (M9), (M13), (M20)).
- **Total R²** on the original `Δy` scale (computed using the full additive reconstruction implied by the approach, so approaches can be compared on a common scale).

### 6.1 Variance decomposition and attribution

To characterize how variation in `Δy` is partitioned among components, the code computes a variance decomposition identity of the form:

```
Var(Δy) = Σ Var(C_m) + 2 Σ_{m<n} Cov(C_m, C_n),
```

where the components `C_m` correspond to combinations of:
- climate terms (sometimes decomposed into “departure”, “trend”, and “cross” components for quadratic models),
- country trend `j`,
- year effect `k`, and
- the residual remainder `ε`.

An additional **variance attribution** summary allocates covariances symmetrically across components to produce additive “contributions” that sum to `Var(Δy)`.

These diagnostics are descriptive; they do not alter parameter estimates.

---

## 7. Uncertainty quantification via bootstrap

### 7.1 Cluster bootstrap by country

The default uncertainty analysis is a **cluster bootstrap** that resamples countries with replacement (preserving within‑country time dependence):

For each bootstrap iteration *b*:
1. Sample `N` countries with replacement from the original set of `N` countries.
2. Construct a bootstrap dataset by concatenating the sampled countries’ full time series (re‑indexing countries).
3. Recompute trend objects (`k(t)`, `jᵢ(t)`, `T_trend`) for the bootstrap sample.
4. Re‑fit all approaches, store coefficients and key derived quantities.

Point estimates are the fit on the original dataset. Uncertainty is summarized by empirical percentiles of bootstrap draws (default: 5, 25, 50, 75, 95).

### 7.2 Optional time‑dimension bootstrap (`--sample-years`)

When `--sample-years` is enabled, years are also sampled with replacement:

- Sample `N` countries with replacement (as above).
- Sample `Y` years with replacement, where `Y` is the number of unique years in the dataset.

Rather than physically duplicating observations, the code uses **weighted least squares** with observation weights:

```
wᵢ,t = (# times country i selected) × (# times year t selected).        (M22)
```

Trend estimation (`k(t)`, country polynomial fits, LOESS smoothing) and regression fitting are performed with these weights. Years with zero total weight effectively drop out of the fit.

This bootstrap variant is designed to reflect uncertainty associated with both country sampling and common year shocks.

---

## 8. Cumulative climate effects

The pipeline also computes cumulative impacts implied by fitted response functions using bootstrap draws.

### 8.1 From growth contributions to level effects

Because the dependent variable is a log difference, a sequence of annual climate contributions can be accumulated into a log‑level effect.

For each country and approach, define a baseline year `t₀` (the code uses 1961). Define the annual deviation in climate contribution relative to baseline:

```
Δhᵢ,t = h_termᵢ,t − h_termᵢ,t₀.                                        (M23)
```

For quadratic approaches, `h_termᵢ,t` is typically `h(Tᵢ,t)` (or the approach‑specific net contribution for persistence models).

The cumulative log effect through year *t* is:

```
Hᵢ,t = Σ_{s=t₀}^{t} Δhᵢ,s.                                             (M24)
```

A percent‑level interpretation can be obtained via:

```
pct_effectᵢ,t = (exp(Hᵢ,t) − 1) · 100.
```

### 8.2 Use of bootstrap draws

The bootstrap routine optionally stores `h_termᵢ,t` for each observation under each bootstrap draw (for selected approaches). The cumulative analysis applies (M23)–(M24) to each draw, producing distributions of cumulative effects by country.

---

## 9. Influence analysis (which countries move the coefficients?)

To identify countries that systematically shift estimated coefficients, the pipeline uses bootstrap resampling records.

For each bootstrap iteration, the country resampling vector can be converted into a count vector:

```
cᵦ,j = number of times country j appears in bootstrap iteration b.
```

For a given approach and coefficient (e.g., `h₁`, `h₂`, `T_opt`, or `h₄`), the code analyzes how coefficient values co‑vary with inclusion counts across bootstrap iterations. Two regression options are supported:

- **Linear:** coefficient value regressed on country counts.
- **Logistic:** indicator of being above a percentile threshold regressed on country counts.

The result is a ranked list of countries whose resampling frequency is most strongly associated with higher vs lower coefficient values.

---

## Appendix A. Relationship between QJ and QP (identification / re‑referencing)

Approach QJ (joint OLS) and Approach QP (explicit detrending) estimate closely related objects but use different parameterizations and identification constraints.

The script:

- `scripts/compare_Approach1J_Approach1P.py`

compares:
- year effects `k(t)`,
- country trend coefficients (`j₀,ᵢ`, `j₁,ᵢ`, `j₂,ᵢ`),

between QJ and quantities reconstructed from QP after appropriate re‑referencing. The key point is that QP computes trends relative to a different normalization than the joint fixed‑effects regression; aligning the normalizations yields near 1:1 correspondence in practice.

(For many analyses, this relationship is primarily a diagnostic that the explicit detrending workflow is algebraically consistent with the joint model up to identification choices.)
