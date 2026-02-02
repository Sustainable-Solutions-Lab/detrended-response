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
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Dict
from .data_loader import AnalysisData
from .detrending import (
    CountryTrends,
    CountryTrendsLoess,
    compute_detrended_temperature,
    compute_detrended_temp_squared,
    compute_detrended_temperature_quadratic,
    compute_detrended_temp_squared_quadratic,
    compute_detrended_temperature_loess,
    compute_detrended_temp_squared_loess,
    compute_growth_trend_values,
    compute_growth_trend_values_linear,
)


@dataclass
class FitResult:
    """Container for regression results."""
    approach: str           # Name of the approach
    h1: float              # Linear temperature coefficient
    h2: float              # Quadratic temperature coefficient
    h1_se: float           # Standard error of h1
    h2_se: float           # Standard error of h2
    k: Dict[int, float]    # Year fixed effects (year -> value)
    r_squared: float       # R-squared
    adj_r_squared: float   # Adjusted R-squared
    rmse: float            # Root mean squared error
    n_obs: int             # Number of observations
    n_params: int          # Number of parameters
    residuals: np.ndarray  # Residuals
    T_optimal: float       # Optimal temperature = -h1 / (2*h2)
    total_r_squared: float # Variance explained in original dy


@dataclass
class FitResultApproach8:
    """Container for Approach 8 results with GDP-dependent response.

    Model: h(Y,T) = (Y/Y_ref)^(-beta) * (h1*T + h2*T^2)
    """
    approach: str           # Name of the approach
    h1: float              # Linear temperature coefficient
    h2: float              # Quadratic temperature coefficient
    h1_se: float           # Standard error of h1
    h2_se: float           # Standard error of h2
    beta: float            # GDP scaling exponent
    beta_se: float         # Standard error of beta
    Y_ref: float           # Reference GDP used
    k: Dict[int, float]    # Year fixed effects (year -> value)
    r_squared: float       # R-squared
    adj_r_squared: float   # Adjusted R-squared
    rmse: float            # Root mean squared error
    n_obs: int             # Number of observations
    n_params: int          # Number of parameters (3: h1, h2, beta)
    residuals: np.ndarray  # Residuals
    T_optimal: float       # Optimal temperature = -h1 / (2*h2)
    total_r_squared: float # Variance explained in original dy


def build_design_matrix(data: AnalysisData, X1: np.ndarray, X2: np.ndarray) -> tuple:
    """Build design matrix with temperature terms and year fixed effects.

    Args:
        data: AnalysisData object
        X1: First temperature term (n_obs,) - coefficient is h1
        X2: Second temperature term (n_obs,) - coefficient is h2

    Returns:
        Tuple of (design_matrix, unique_years)
        - Design matrix (n_obs, 2 + n_years)
        - Columns: [X1, X2, year_0, year_1, ..., year_{n-1}]
        - unique_years: sorted list of years for k extraction
    """
    n_obs = data.n_obs

    # Get unique years and create year index mapping
    unique_years = sorted(set(data.year))
    n_years = len(unique_years)
    year_to_idx = {y: i for i, y in enumerate(unique_years)}

    # Allocate design matrix
    X = np.zeros((n_obs, 2 + n_years))

    # Temperature terms
    X[:, 0] = X1
    X[:, 1] = X2

    # Year fixed effects (one-hot encoding)
    for i in range(n_obs):
        yr_idx = year_to_idx[data.year[i]]
        X[i, 2 + yr_idx] = 1.0

    return X, unique_years


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


def compute_total_r_squared(residuals: np.ndarray, dy: np.ndarray) -> float:
    """Compute R² of original dy explained by full model.

    This provides a standardized metric across all approaches,
    measuring what fraction of variance in dy_i(t) is explained
    by h(T) + j_i(t) + k(t).
    """
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((dy - np.mean(dy)) ** 2)
    return 1 - ss_res / ss_tot


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
    X, unique_years = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    y = data.growth_pcGDP
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects
    k = {unique_years[i]: beta[2 + i] for i in range(len(unique_years))}

    # Fit statistics
    n_params = 2 + len(unique_years)
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Linear Temperature Detrending",
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
        total_r_squared=total_r_sq,
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
    X, unique_years = build_design_matrix(data, T, T2)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects
    k = {unique_years[i]: beta[2 + i] for i in range(len(unique_years))}

    # Fit statistics
    n_params = 2 + len(unique_years)
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Quadratic GDP Growth Detrending",
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
        total_r_squared=total_r_sq,
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
    X, unique_years = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects
    k = {unique_years[i]: beta[2 + i] for i in range(len(unique_years))}

    # Fit statistics
    n_params = 2 + len(unique_years)
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Combined Detrending (Mixed)",
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
        total_r_squared=total_r_sq,
    )


def fit_approach4_combined_linear_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 4: Combined detrending with linear GDP growth trend.

    Δy_i(t) - (y0 + y1*t) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i

    Like Approach 3, but uses linear GDP growth trend (y2 = 0).
    """
    # Compute detrended dependent variable using linear trend
    y_trend = compute_growth_trend_values_linear(data, trends)
    y = data.growth_pcGDP - y_trend

    # Compute detrended temperature terms
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

    # Build design matrix
    X, unique_years = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects
    k = {unique_years[i]: beta[2 + i] for i in range(len(unique_years))}

    # Fit statistics
    n_params = 2 + len(unique_years)
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Combined Linear Detrending",
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
        total_r_squared=total_r_sq,
    )


def fit_approach5_combined_quadratic_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 5: Combined detrending with quadratic temperature and GDP growth trends.

    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*[T - (T0 + T1*t + T2*t²)]
                                  + h2*[T² - (T0 + T1*t + T2*t²)²] + k_i

    Like Approach 3, but uses quadratic temperature trend instead of linear.
    """
    # Compute detrended dependent variable (quadratic GDP growth trend)
    y_trend = compute_growth_trend_values(data, trends)
    y = data.growth_pcGDP - y_trend

    # Compute detrended temperature terms using quadratic temperature trend
    T_star = compute_detrended_temperature_quadratic(data, trends)
    T2_detrend = compute_detrended_temp_squared_quadratic(data, trends)

    # Build design matrix
    X, unique_years = build_design_matrix(data, T_star, T2_detrend)

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects
    k = {unique_years[i]: beta[2 + i] for i in range(len(unique_years))}

    # Fit statistics
    n_params = 2 + len(unique_years)
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Combined Quadratic Detrending",
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
        total_r_squared=total_r_sq,
    )


def fit_approach6_precomputed_k_linear(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 6: Pre-computed k[t] with linear country/temperature trends.

    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = j_{0,i} + j_{1,i}*t are fit to dy_i[t] - k[t]
    3. Temperature is detrended with linear trend: T* = T - (T0 + T1*t)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Unlike Approaches 4/5, year effects k[t] are pre-computed as year means
    rather than estimated in the regression.
    """
    # Compute detrended temperature terms (linear)
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

    # Compute dependent variable: dy_i[t] - k[t] - j_i[t]
    # where j_i[t] is the linear trend fit to (dy - k)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        # j_i[t] = y0_lin + y1_lin * t (fit to dy - k)
        j_i_t = trends.y0_lin[c] + trends.y1_lin[c] * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    # Design matrix: just [T*, T*²] - no year fixed effects
    X = np.column_stack([T_star, T2_detrend])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # Just h1 and h2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Precomputed k Linear",
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
        total_r_squared=total_r_sq,
    )


def fit_approach7_precomputed_k_quadratic(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 7: Pre-computed k[t] with quadratic country/temperature trends.

    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = j_{0,i} + j_{1,i}*t + j_{2,i}*t² are fit to dy_i[t] - k[t]
    3. Temperature is detrended with quadratic trend: T* = T - (T0 + T1*t + T2*t²)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Unlike Approaches 4/5, year effects k[t] are pre-computed as year means
    rather than estimated in the regression.
    """
    # Compute detrended temperature terms (quadratic)
    T_star = compute_detrended_temperature_quadratic(data, trends)
    T2_detrend = compute_detrended_temp_squared_quadratic(data, trends)

    # Compute dependent variable: dy_i[t] - k[t] - j_i[t]
    # where j_i[t] is the quadratic trend fit to (dy - k)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        # j_i[t] = y0 + y1*t + y2*t² (fit to dy - k)
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    # Design matrix: just [T*, T*²] - no year fixed effects
    X = np.column_stack([T_star, T2_detrend])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # Just h1 and h2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Precomputed k Quadratic",
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
        total_r_squared=total_r_sq,
    )


def compute_beta_se_numerical(
    sse_func: callable,
    beta_opt: float,
    beta_bounds: tuple,
    n_obs: int,
    n_params: int = 3,
) -> float:
    """Compute standard error of beta from curvature of SSE profile.

    Uses numerical second derivative of the SSE function at the optimum.
    SE(beta) is approximated from the profile likelihood curvature.

    Args:
        sse_func: Function that computes SSE for a given beta
        beta_opt: Optimal beta value
        beta_bounds: Bounds for beta (to ensure step stays in bounds)
        n_obs: Number of observations
        n_params: Number of parameters (default 3: h1, h2, beta)

    Returns:
        Standard error of beta (or np.nan if curvature is non-positive)
    """
    # Step size for finite differences
    h = 0.01
    # Ensure we stay within bounds
    h = min(h, (beta_bounds[1] - beta_opt) / 2, (beta_opt - beta_bounds[0]) / 2)

    sse_minus = sse_func(beta_opt - h)
    sse_center = sse_func(beta_opt)
    sse_plus = sse_func(beta_opt + h)

    # Second derivative: d2(SSE)/d(beta)^2 ~ (f(x+h) - 2*f(x) + f(x-h)) / h^2
    d2_sse = (sse_plus - 2 * sse_center + sse_minus) / (h ** 2)

    if d2_sse <= 0:
        # Curvature is non-positive, cannot estimate SE this way
        return np.nan

    # Approximate SE from profile likelihood curvature
    # sigma^2 ~ SSE / (n - p)
    sigma_sq = sse_center / (n_obs - n_params)
    se = np.sqrt(2 * sigma_sq / d2_sse)

    return se


def fit_approach8_gdp_response(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    Y_ref: float,
    beta_bounds: tuple = (0.01, 0.99),
) -> FitResultApproach8:
    """Approach 8: GDP-dependent temperature response with quadratic detrending.

    Model: dy*_i(t) = (Y_i(t)/Y_ref)^(-beta) * [h1*T* + h2*T*^2]

    where:
    - dy* = dy_i(t) - k[t] - j_i(t) (same as Approach 7)
    - T* = T - (T0 + T1*t + T2*t^2) (quadratic detrended temperature)
    - (Y_i(t)/Y_ref)^(-beta) scales the response by GDP level

    Uses nested optimization:
    - Outer: minimize_scalar over beta in beta_bounds using Brent's method
    - Inner: for fixed beta, solve linear OLS for h1, h2

    Args:
        data: AnalysisData object
        trends: CountryTrends (with trends fit to dy - k[t])
        year_means: Pre-computed k[t] = mean(dy_i[t])
        Y_ref: Reference GDP (computed once on full dataset)
        beta_bounds: Bounds for beta optimization (default [0.01, 0.99])

    Returns:
        FitResultApproach8 with beta, h1, h2, and standard errors
    """
    # Compute detrended temperature terms (quadratic, same as Approach 7)
    T_star = compute_detrended_temperature_quadratic(data, trends)
    T2_star = compute_detrended_temp_squared_quadratic(data, trends)

    # Compute dependent variable: dy - k[t] - j_i[t] (same as Approach 7)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        # j_i[t] = y0 + y1*t + y2*t² (fit to dy - k)
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    # Define objective function: SSE for given beta
    def compute_sse_for_beta(beta):
        """Compute SSE for a given beta by solving inner OLS problem."""
        # Compute GDP scaling factor g = (Y/Y_ref)^(-beta)
        g = (data.pcGDP / Y_ref) ** (-beta)

        # Build design matrix: X = [g*T*, g*T*^2]
        X = np.column_stack([g * T_star, g * T2_star])

        # Solve OLS: min ||y - X @ [h1, h2]||^2
        try:
            beta_ols, _, _, _ = linalg.lstsq(X, y)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    # Optimize beta using Brent's method
    result = minimize_scalar(
        compute_sse_for_beta,
        bounds=beta_bounds,
        method='bounded',
        options={'xatol': 1e-6}
    )
    beta_opt = result.x

    # Re-fit at optimal beta to get h1, h2 and covariance
    g_opt = (data.pcGDP / Y_ref) ** (-beta_opt)
    X_opt = np.column_stack([g_opt * T_star, g_opt * T2_star])

    beta_ols, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute beta SE via numerical second derivative
    beta_se = compute_beta_se_numerical(
        compute_sse_for_beta, beta_opt, beta_bounds, data.n_obs
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1, h2, beta
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResultApproach8(
        approach="GDP-Response Quadratic",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        beta=beta_opt,
        beta_se=beta_se,
        Y_ref=Y_ref,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
        total_r_squared=total_r_sq,
    )


def fit_approach9_precomputed_k_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResult:
    """Approach 9: Pre-computed k[t] with LOESS country/temperature trends.

    LOESS version of Approach 7:
    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) are LOESS-smoothed (dy_i[t] - k[t])
    3. Temperature is detrended with LOESS: T* = T - T_loess
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResult
    """
    # Compute detrended temperature terms (LOESS)
    T_star = compute_detrended_temperature_loess(data, trends_loess)
    T2_detrend = compute_detrended_temp_squared_loess(data, trends_loess)

    # Compute dependent variable: dy_i[t] - k[t] - j_i[t]
    # where j_i[t] is the LOESS smoothed (dy - k)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: just [T*, T*²] - no year fixed effects
    X = np.column_stack([T_star, T2_detrend])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # Just h1 and h2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="Precomputed k LOESS",
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
        total_r_squared=total_r_sq,
    )


def fit_approach10_gdp_response_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    Y_ref: float,
    beta_bounds: tuple = (0.01, 0.99),
) -> FitResultApproach8:
    """Approach 10: GDP-dependent temperature response with LOESS detrending.

    LOESS version of Approach 8:
    Model: dy*_i(t) = (Y_i(t)/Y_ref)^(-beta) * [h1*T* + h2*T*^2]

    where:
    - dy* = dy_i(t) - k[t] - j_i(t) (same as Approach 9)
    - T* = T - T_loess (LOESS detrended temperature)
    - (Y_i(t)/Y_ref)^(-beta) scales the response by GDP level

    Uses nested optimization:
    - Outer: minimize_scalar over beta in beta_bounds using Brent's method
    - Inner: for fixed beta, solve linear OLS for h1, h2

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        Y_ref: Reference GDP (computed once on full dataset)
        beta_bounds: Bounds for beta optimization (default [0.01, 0.99])

    Returns:
        FitResultApproach8 with beta, h1, h2, and standard errors
    """
    # Compute detrended temperature terms (LOESS)
    T_star = compute_detrended_temperature_loess(data, trends_loess)
    T2_star = compute_detrended_temp_squared_loess(data, trends_loess)

    # Compute dependent variable: dy - k[t] - j_i[t] (same as Approach 9)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Define objective function: SSE for given beta
    def compute_sse_for_beta(beta):
        """Compute SSE for a given beta by solving inner OLS problem."""
        # Compute GDP scaling factor g = (Y/Y_ref)^(-beta)
        g = (data.pcGDP / Y_ref) ** (-beta)

        # Build design matrix: X = [g*T*, g*T*^2]
        X = np.column_stack([g * T_star, g * T2_star])

        # Solve OLS: min ||y - X @ [h1, h2]||^2
        try:
            beta_ols, _, _, _ = linalg.lstsq(X, y)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    # Optimize beta using Brent's method
    result = minimize_scalar(
        compute_sse_for_beta,
        bounds=beta_bounds,
        method='bounded',
        options={'xatol': 1e-6}
    )
    beta_opt = result.x

    # Re-fit at optimal beta to get h1, h2 and covariance
    g_opt = (data.pcGDP / Y_ref) ** (-beta_opt)
    X_opt = np.column_stack([g_opt * T_star, g_opt * T2_star])

    beta_ols, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute beta SE via numerical second derivative
    beta_se = compute_beta_se_numerical(
        compute_sse_for_beta, beta_opt, beta_bounds, data.n_obs
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1, h2, beta
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResultApproach8(
        approach="GDP-Response LOESS",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        beta=beta_opt,
        beta_se=beta_se,
        Y_ref=Y_ref,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_optimal=T_optimal,
        total_r_squared=total_r_sq,
    )


def fit_approach0_no_detrending(data: AnalysisData) -> FitResult:
    """Approach 0: No pre-detrending, with country time trends and year fixed effects.

    Δy_i(t) = h1*T + h2*T² + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    This estimates:
    - h1, h2: temperature response coefficients
    - j_{0,i}, j_{1,i}, j_{2,i}: country-specific quadratic time trends (countries i > 0)
    - k_t: year fixed effects (all years)

    For identifiability, we set j_{0,0} = j_{1,0} = j_{2,0} = 0 (first country is reference).
    This provides 3 constraints to pin down the arbitrary quadratic that could otherwise
    be added to all j_i(t) and subtracted from all k_t.
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years and create year index mapping
    unique_years = sorted(set(data.year))
    year_to_idx = {y: i for i, y in enumerate(unique_years)}
    n_years = len(unique_years)

    # Number of parameters:
    # - 2 for h1, h2
    # - 3 * (n_countries - 1) for j terms (first country is reference, j[0] = 0)
    # - n_years for k_t terms (all years)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_years
    n_params = 2 + n_j_params + n_k_params

    X = np.zeros((n_obs, n_params))

    # Temperature terms
    X[:, 0] = data.temp
    X[:, 1] = data.temp ** 2

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:  # Skip country 0 (reference country)
            t = data.time[i]
            # j0, j1, j2 for country c are at columns for (c-1)
            col_base = 2 + 3 * (c - 1)
            X[i, col_base] = 1.0        # j0[c]
            X[i, col_base + 1] = t      # j1[c]
            X[i, col_base + 2] = t * t  # j2[c]

    # Year fixed effects (all years)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr_idx = year_to_idx[data.year[i]]
        X[i, k_col_start + yr_idx] = 1.0

    # Fit OLS
    y = data.growth_pcGDP
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Year fixed effects (store by actual year for consistency with other approaches)
    k = {}
    for yr_idx in range(n_years):
        k[unique_years[yr_idx]] = beta[k_col_start + yr_idx]

    # Fit statistics
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_optimal = -h1 / (2 * h2) if h2 != 0 else np.nan

    return FitResult(
        approach="No Detrending",
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
        total_r_squared=total_r_sq,
    )


def fit_all_approaches(
    data: AnalysisData, trends: CountryTrends,
    trends_with_k: CountryTrends = None, year_means: dict = None,
    Y_ref: float = None, trends_loess: CountryTrendsLoess = None
) -> dict:
    """Fit all approaches and return results.

    Returns dict with keys:
        'approach0': No detrending, with j terms and year fixed effects
        'approach1': Temperature detrending (linear T trend)
        'approach2': GDP growth detrending (quadratic GDP trend)
        'approach3': Combined detrending (linear T trend, quadratic GDP trend)
        'approach4': Combined detrending (linear T trend, linear GDP trend)
        'approach5': Combined detrending (quadratic T trend, quadratic GDP trend)
        'approach6': Pre-computed k with linear trends (if trends_with_k and year_means provided)
        'approach7': Pre-computed k with quadratic trends (if trends_with_k and year_means provided)
        'approach8': GDP-dependent response (if Y_ref provided)
        'approach9': Pre-computed k with LOESS trends (if trends_loess provided)
        'approach10': GDP-dependent response with LOESS (if trends_loess and Y_ref provided)

    Args:
        data: AnalysisData object
        trends: CountryTrends for approaches 0-5
        trends_with_k: CountryTrends for approaches 6-8 (fit to dy - k)
        year_means: Pre-computed k[t] for approaches 6-10
        Y_ref: Reference GDP for approach 8 and 10 (computed once on full dataset)
        trends_loess: CountryTrendsLoess for approaches 9-10 (LOESS detrending)
    """
    results = {
        'approach0': fit_approach0_no_detrending(data),
        'approach1': fit_approach1_temperature_detrending(data, trends),
        'approach2': fit_approach2_growth_detrending(data, trends),
        'approach3': fit_approach3_combined_detrending(data, trends),
        'approach4': fit_approach4_combined_linear_detrending(data, trends),
        'approach5': fit_approach5_combined_quadratic_detrending(data, trends),
    }

    # Add approaches 6 and 7 if trends_with_k and year_means are provided
    if trends_with_k is not None and year_means is not None:
        results['approach6'] = fit_approach6_precomputed_k_linear(
            data, trends_with_k, year_means
        )
        results['approach7'] = fit_approach7_precomputed_k_quadratic(
            data, trends_with_k, year_means
        )

        # Add approach 8 if Y_ref is provided
        if Y_ref is not None:
            results['approach8'] = fit_approach8_gdp_response(
                data, trends_with_k, year_means, Y_ref
            )

    # Add approaches 9 and 10 if trends_loess and year_means are provided
    if trends_loess is not None and year_means is not None:
        results['approach9'] = fit_approach9_precomputed_k_loess(
            data, trends_loess, year_means
        )

        # Add approach 10 if Y_ref is provided
        if Y_ref is not None:
            results['approach10'] = fit_approach10_gdp_response_loess(
                data, trends_loess, year_means, Y_ref
            )

    return results
