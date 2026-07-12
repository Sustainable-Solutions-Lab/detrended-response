"""Two-stage random-coefficients diagnostic for temperature sensitivity.

A different conceptualization from the pooled "approaches" framework: instead of one
shared response function, estimate each country's *own* temperature sensitivity and then
explain the cross-country spread of those sensitivities.

Stage 1 (per country i, OLS):
    Δy_it = β₁ᵢ·T_it + j0ᵢ + j1ᵢ·t + j2ᵢ·t² + ε_it
    → a linear temperature sensitivity β₁ᵢ (net of a country-specific quadratic time
      trend, and optionally net of the global year means k_t).

Stage 2 (across countries, precision-weighted WLS):
    β̂₁ᵢ = γ0 + γ_G·(mean log pcGDPᵢ) + γ_D·(T̄ᵢ) + uᵢ
    fit as income-only / temp-only / both.

Interpretation caveat: under a *single* global quadratic response h(T)=β₁T+β₂T², the
country slope is β₁ᵢ = β₁ + 2β₂·T̄ᵢ — mechanically linear in mean temperature. So a
β̂₁ᵢ-vs-T̄ᵢ relationship is expected even with no true heterogeneity; the meaningful test
is whether income explains β̂₁ᵢ *beyond* T̄ᵢ (the "both" regression). β̂₁ᵢ is itself a noisy
regressand, so Stage 2 is precision-weighted (errors-in-variables still attenuates).
"""
from dataclasses import dataclass

import numpy as np

from .data_loader import AnalysisData
from .detrending import compute_year_means
from .fitting import fit_ols


@dataclass
class CountrySlopes:
    """Stage-1 per-country temperature sensitivities (retained countries only)."""
    iso: np.ndarray          # ISO3 codes
    beta1: np.ndarray        # temperature sensitivity β₁ᵢ
    se: np.ndarray           # standard error of β₁ᵢ
    n_obs: np.ndarray        # observations per country
    r_squared: np.ndarray    # first-stage R²
    mean_logGDP: np.ndarray  # mean log(pcGDP) per country
    mean_T: np.ndarray       # mean temperature per country
    growth_vol: np.ndarray   # std of Δy per country
    remove_year_means: bool


def fit_country_temperature_slopes(data: AnalysisData, remove_year_means: bool = True,
                                   min_years: int = 20) -> CountrySlopes:
    """Stage 1: fit Δy = β₁·T + j0 + j1·t + j2·t² separately for each country.

    Countries with fewer than `min_years` observations or no temperature variation are
    dropped. If `remove_year_means`, the global year means k_t are subtracted from Δy first.
    """
    y = data.growth_pcGDP.copy()
    if remove_year_means:
        k = compute_year_means(data)
        y = y - np.array([k[yr] for yr in data.year])

    logGDP = np.log(data.pcGDP)
    iso, beta1, se, n_obs, r2, m_logGDP, m_T, vol = [], [], [], [], [], [], [], []
    for c in range(data.n_countries):
        mask = data.country_idx == c
        n = int(mask.sum())
        T_c = data.temp[mask]
        if n < min_years or np.std(T_c) == 0:
            continue
        t_c = data.time[mask]
        y_c = y[mask]
        X = np.column_stack([T_c, np.ones(n), t_c, t_c * t_c])
        coef, residuals, _, cov = fit_ols(y_c, X)
        sst = np.sum((y_c - np.mean(y_c)) ** 2)
        iso.append(data.idx_to_iso[c])
        beta1.append(float(coef[0]))
        se.append(float(np.sqrt(max(cov[0, 0], 0.0))))
        n_obs.append(n)
        r2.append(float(1.0 - np.sum(residuals ** 2) / sst))
        m_logGDP.append(float(np.mean(logGDP[mask])))
        m_T.append(float(np.mean(T_c)))
        vol.append(float(np.std(data.growth_pcGDP[mask])))

    return CountrySlopes(
        iso=np.array(iso), beta1=np.array(beta1), se=np.array(se),
        n_obs=np.array(n_obs), r_squared=np.array(r2),
        mean_logGDP=np.array(m_logGDP), mean_T=np.array(m_T),
        growth_vol=np.array(vol), remove_year_means=remove_year_means,
    )


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> dict:
    """Weighted least squares via fit_ols on sqrt(w)-scaled data."""
    sw = np.sqrt(w)
    coef, residuals, _, cov = fit_ols(y * sw, X * sw[:, None])
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    return {'coef': coef, 'se': se, 'r_squared': 1.0 - sse / sst}


def explain_country_slopes(slopes: CountrySlopes, weighted: bool = True) -> dict:
    """Stage 2: regress β̂₁ᵢ on centered mean log(pcGDP) and mean temperature.

    Returns income-only / temp-only / both fits (each a dict with 'coef' ordered
    [intercept, <covariates...>], 'se', 'r_squared'), plus the raw Pearson correlations
    of β₁ with each covariate. `coef`/`se` label order is given under 'terms'.
    """
    b = slopes.beta1
    G = slopes.mean_logGDP - slopes.mean_logGDP.mean()   # centered income
    D = slopes.mean_T - slopes.mean_T.mean()             # centered temperature level
    ones = np.ones(len(b))
    w = 1.0 / slopes.se ** 2 if weighted else np.ones(len(b))

    return {
        'weighted': weighted,
        'n_countries': len(b),
        'terms': {'income_only': ['const', 'logGDP'],
                  'temp_only': ['const', 'meanT'],
                  'both': ['const', 'logGDP', 'meanT']},
        'income_only': _wls(b, np.column_stack([ones, G]), w),
        'temp_only': _wls(b, np.column_stack([ones, D]), w),
        'both': _wls(b, np.column_stack([ones, G, D]), w),
        'corr_income': float(np.corrcoef(b, slopes.mean_logGDP)[0, 1]),
        'corr_meanT': float(np.corrcoef(b, slopes.mean_T)[0, 1]),
    }
