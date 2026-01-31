"""OLS fitting for three detrending approaches.

Approach 1: Linear temperature detrending
    Δy_i(t) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i

Approach 2: Quadratic GDP growth detrending
    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*T + h2*T² + k_i

Approach 3: Combined detrending
    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass
from typing import Dict
from .data_loader import AnalysisData
from .detrending import (
    CountryTrends,
    compute_detrended_temperature,
    compute_detrended_temp_squared,
    compute_growth_trend_values,
)


@dataclass
class FitResult:
    """Container for regression results."""
    approach: str           # Name of the approach
    h1: float              # Linear temperature coefficient
    h2: float              # Quadratic temperature coefficient
    h1_se: float           # Standard error of h1
    h2_se: float           # Standard error of h2
    k: Dict[int, float]    # Country fixed effects (country_idx -> value)
    r_squared: float       # R-squared
    adj_r_squared: float   # Adjusted R-squared
    rmse: float            # Root mean squared error
    n_obs: int             # Number of observations
    n_params: int          # Number of parameters
    residuals: np.ndarray  # Residuals
    T_optimal: float       # Optimal temperature = -h1 / (2*h2)


def build_design_matrix(data: AnalysisData, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Build design matrix with temperature terms and country fixed effects.

    Args:
        data: AnalysisData object
        X1: First temperature term (n_obs,) - coefficient is h1
        X2: Second temperature term (n_obs,) - coefficient is h2

    Returns:
        Design matrix (n_obs, 2 + n_countries)
        Columns: [X1, X2, country_0, country_1, ..., country_{n-1}]
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Allocate design matrix
    X = np.zeros((n_obs, 2 + n_countries))

    # Temperature terms
    X[:, 0] = X1
    X[:, 1] = X2

    # Country fixed effects (one-hot encoding)
    for i in range(n_obs):
        X[i, 2 + data.country_idx[i]] = 1.0

    return X


def fit_ols(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit OLS regression y = X @ beta.

    Returns (beta, residuals, sigma_squared, cov_matrix).
    """
    # Solve least squares
    beta, residuals_sum, rank, s = linalg.lstsq(X, y)

    # Compute residuals
    y_pred = X @ beta
    residuals = y - y_pred

    # Degrees of freedom
    n = len(y)
    p = X.shape[1]
    df = n - p

    # Residual variance
    sse = np.sum(residuals ** 2)
    sigma_squared = sse / df

    # Covariance matrix of beta
    XtX_inv = linalg.inv(X.T @ X)
    cov_matrix = sigma_squared * XtX_inv

    return beta, residuals, sigma_squared, cov_matrix


def compute_fit_stats(y: np.ndarray, residuals: np.ndarray, n_params: int) -> tuple:
    """Compute R-squared, adjusted R-squared, and RMSE."""
    n_obs = len(y)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params)
    rmse = np.sqrt(ss_res / n_obs)

    return r_squared, adj_r_squared, rmse


def fit_approach1_temperature_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 1: Linear temperature detrending only.

    Δy_i(t) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i
    """
    # Compute detrended temperature terms
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

    # Build design matrix
    X = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    y = data.growth_pcGDP
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Country fixed effects
    k = {i: beta[2 + i] for i in range(data.n_countries)}

    # Fit statistics
    n_params = 2 + data.n_countries
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Temperature Detrending",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
    )


def fit_approach2_growth_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 2: Quadratic GDP growth detrending only.

    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*T + h2*T² + k_i
    """
    # Compute detrended dependent variable
    y_trend = compute_growth_trend_values(data, trends)
    y = data.growth_pcGDP - y_trend

    # Build design matrix with raw temperature
    T = data.temp
    T2 = data.temp ** 2
    X = build_design_matrix(data, T, T2)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Country fixed effects
    k = {i: beta[2 + i] for i in range(data.n_countries)}

    # Fit statistics
    n_params = 2 + data.n_countries
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="GDP Growth Detrending",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
    )


def fit_approach3_combined_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 3: Combined temperature and GDP growth detrending.

    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i
    """
    # Compute detrended dependent variable
    y_trend = compute_growth_trend_values(data, trends)
    y = data.growth_pcGDP - y_trend

    # Compute detrended temperature terms
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

    # Build design matrix
    X = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Country fixed effects
    k = {i: beta[2 + i] for i in range(data.n_countries)}

    # Fit statistics
    n_params = 2 + data.n_countries
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Combined Detrending",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
    )


def fit_burke_original(data: AnalysisData) -> FitResult:
    """Original Burke et al. (2015) model with country time trends and year fixed effects.

    Δy_i(t) = h1*T + h2*T² + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    This estimates:
    - h1, h2: temperature response coefficients
    - j_{0,i}, j_{1,i}, j_{2,i}: country-specific quadratic time trends (all countries)
    - k_t: year fixed effects (first year is reference, k_0 = 0)

    Setting k_0 = 0 is sufficient to identify the model. The j_0 terms absorb
    the country-specific baseline levels.
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years and create year index mapping
    unique_years = sorted(set(data.year))
    year_to_idx = {y: i for i, y in enumerate(unique_years)}
    n_years = len(unique_years)

    # Number of parameters:
    # - 2 for h1, h2
    # - 3 * n_countries for j terms (all countries)
    # - (n_years - 1) for k_t terms (first year is reference, k_0 = 0)
    n_j_params = 3 * n_countries
    n_k_params = n_years - 1
    n_params = 2 + n_j_params + n_k_params

    X = np.zeros((n_obs, n_params))

    # Temperature terms
    X[:, 0] = data.temp
    X[:, 1] = data.temp ** 2

    # Country-specific time trends (all countries)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]

        col_base = 2 + 3 * c
        X[i, col_base] = 1.0        # j0
        X[i, col_base + 1] = t      # j1
        X[i, col_base + 2] = t * t  # j2

    # Year fixed effects (skip first year as reference, k_0 = 0)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr_idx = year_to_idx[data.year[i]]
        if yr_idx > 0:
            X[i, k_col_start + yr_idx - 1] = 1.0

    # Fit OLS
    y = data.growth_pcGDP
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects (for reporting, we store by year index)
    k = {0: 0.0}  # Reference year (k_0 = 0)
    for yr_idx in range(1, n_years):
        k[yr_idx] = beta[k_col_start + yr_idx - 1]

    # Fit statistics
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Burke Original (j + k_t)",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
    )


def fit_all_approaches(data: AnalysisData, trends: CountryTrends) -> dict:
    """Fit all four approaches and return results.

    Returns dict with keys:
        'burke_original': Original Burke et al. (2015) with j terms and year fixed effects
        'approach1': Temperature detrending
        'approach2': GDP growth detrending
        'approach3': Combined detrending
    """
    return {
        'burke_original': fit_burke_original(data),
        'approach1': fit_approach1_temperature_detrending(data, trends),
        'approach2': fit_approach2_growth_detrending(data, trends),
        'approach3': fit_approach3_combined_detrending(data, trends),
    }
