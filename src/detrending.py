"""Country-level detrending functions.

This module computes:
1. Linear temperature trends: T_i(t) = T_{0,i} + T_{1,i} * t
2. Quadratic GDP growth trends: Δy_i(t) = y_{0,i} + y_{1,i} * t + y_{2,i} * t²
"""

import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Default window size in years for LOESS smoothing
# 42.45 years chosen to match Total R² of quadratic polynomial detrending (method1h0)
# with 140 countries (those having data in both 1961 and 2022)
DEFAULT_LOESS_WINDOW_YEARS = 42.447947771790915

# Minimum number of points required for LOESS fitting
MIN_LOESS_POINTS = 3

from dataclasses import dataclass
from typing import Dict
from .data_loader import AnalysisData


@dataclass
class CountryTrends:
    """Container for country-level trend coefficients."""
    # Temperature linear trend: T(t) = T0 + T1 * t
    T0: Dict[int, float]  # country_idx -> intercept
    T1: Dict[int, float]  # country_idx -> slope

    # Temperature quadratic trend: T(t) = T0_quad + T1_quad*t + T2_quad*t²
    T0_quad: Dict[int, float]  # country_idx -> intercept (quadratic fit)
    T1_quad: Dict[int, float]  # country_idx -> linear coef (quadratic fit)
    T2_quad: Dict[int, float]  # country_idx -> quadratic coef (quadratic fit)

    # GDP growth quadratic trend: Δy(t) = y0 + y1*t + y2*t²
    y0: Dict[int, float]  # country_idx -> constant
    y1: Dict[int, float]  # country_idx -> linear coef
    y2: Dict[int, float]  # country_idx -> quadratic coef

    # GDP growth linear trend: Δy(t) = y0_lin + y1_lin*t
    y0_lin: Dict[int, float]  # country_idx -> constant (linear fit)
    y1_lin: Dict[int, float]  # country_idx -> slope (linear fit)


@dataclass
class CountryTrendsLoess:
    """LOESS-smoothed trends (stores actual smoothed values at each observation).

    Unlike CountryTrends which stores polynomial coefficients, this stores
    the actual smoothed values since LOESS is non-parametric.
    """
    T_loess: np.ndarray   # Smoothed temperature at each observation
    y_loess: np.ndarray   # Smoothed (dy - k[t]) at each observation


def fit_linear_trend(t: np.ndarray, y: np.ndarray) -> tuple:
    """Fit linear trend y = a + b*t using least squares.

    Returns (intercept, slope).
    """
    n = len(t)
    sum_t = np.sum(t)
    sum_y = np.sum(y)
    sum_tt = np.sum(t * t)
    sum_ty = np.sum(t * y)

    denom = n * sum_tt - sum_t * sum_t
    slope = (n * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / n

    return intercept, slope


def fit_quadratic_trend(t: np.ndarray, y: np.ndarray) -> tuple:
    """Fit quadratic trend y = a + b*t + c*t² using least squares.

    Returns (constant, linear_coef, quadratic_coef).
    """
    # Build design matrix [1, t, t²]
    X = np.column_stack([np.ones(len(t)), t, t * t])

    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return coeffs[0], coeffs[1], coeffs[2]


def compute_country_trends(data: AnalysisData) -> CountryTrends:
    """Compute linear temperature and GDP growth trends for each country.

    For each country i:
    - Fits T_i(t) = T_{0,i} + T_{1,i} * t (temperature linear trend)
    - Fits T_i(t) = T_{0,i} + T_{1,i} * t + T_{2,i} * t² (temperature quadratic trend)
    - Fits Δy_i(t) = y_{0,i} + y_{1,i} * t + y_{2,i} * t² (GDP quadratic trend)
    - Fits Δy_i(t) = y_{0,lin,i} + y_{1,lin,i} * t (GDP linear trend)

    Args:
        data: AnalysisData object with observation arrays

    Returns:
        CountryTrends object with trend coefficients per country
    """
    T0 = {}
    T1 = {}
    T0_quad = {}
    T1_quad = {}
    T2_quad = {}
    y0 = {}
    y1 = {}
    y2 = {}
    y0_lin = {}
    y1_lin = {}

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]

        # Fit linear temperature trend
        T0[country_idx], T1[country_idx] = fit_linear_trend(t_country, temp_country)

        # Fit quadratic temperature trend
        T0_quad[country_idx], T1_quad[country_idx], T2_quad[country_idx] = fit_quadratic_trend(
            t_country, temp_country
        )

        # Fit quadratic GDP growth trend
        y0[country_idx], y1[country_idx], y2[country_idx] = fit_quadratic_trend(
            t_country, growth_country
        )

        # Fit linear GDP growth trend
        y0_lin[country_idx], y1_lin[country_idx] = fit_linear_trend(
            t_country, growth_country
        )

    return CountryTrends(
        T0=T0, T1=T1,
        T0_quad=T0_quad, T1_quad=T1_quad, T2_quad=T2_quad,
        y0=y0, y1=y1, y2=y2, y0_lin=y0_lin, y1_lin=y1_lin
    )


def _compute_temp_trend_value(
    trends: CountryTrends, country_idx: int, t: float, quadratic: bool = False
) -> float:
    """Compute temperature trend value for a single observation.

    Args:
        trends: CountryTrends object with trend coefficients
        country_idx: Index of the country
        t: Time value
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        Temperature trend value at time t
    """
    if quadratic:
        return (
            trends.T0_quad[country_idx]
            + trends.T1_quad[country_idx] * t
            + trends.T2_quad[country_idx] * t * t
        )
    else:
        return trends.T0[country_idx] + trends.T1[country_idx] * t


def _compute_growth_trend_value(
    trends: CountryTrends, country_idx: int, t: float, quadratic: bool = True
) -> float:
    """Compute GDP growth trend value for a single observation.

    Args:
        trends: CountryTrends object with trend coefficients
        country_idx: Index of the country
        t: Time value
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        Growth trend value at time t
    """
    if quadratic:
        return trends.y0[country_idx] + trends.y1[country_idx] * t + trends.y2[country_idx] * t * t
    else:
        return trends.y0_lin[country_idx] + trends.y1_lin[country_idx] * t


def compute_detrended_temperature(
    data: AnalysisData, trends: CountryTrends, quadratic: bool = False
) -> np.ndarray:
    """Compute temperature departure from trend.

    Args:
        data: AnalysisData object
        trends: CountryTrends object with trend coefficients
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        T*(t) = T(t) - trend(t) for each observation
    """
    T_star = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend = _compute_temp_trend_value(trends, c, t, quadratic)
        T_star[i] = data.temp[i] - T_trend

    return T_star


def compute_detrended_temp_squared(
    data: AnalysisData, trends: CountryTrends, quadratic: bool = False
) -> np.ndarray:
    """Compute T² - trend² for temperature detrending.

    This is the coefficient adjustment for h2 in the temperature-detrended model.

    Args:
        data: AnalysisData object
        trends: CountryTrends object with trend coefficients
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        T² - trend² for each observation
    """
    T2_detrend = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T = data.temp[i]
        T_trend = _compute_temp_trend_value(trends, c, t, quadratic)
        T2_detrend[i] = T * T - T_trend * T_trend

    return T2_detrend


def compute_detrended_growth(
    data: AnalysisData, trends: CountryTrends, quadratic: bool = True
) -> np.ndarray:
    """Compute GDP growth departure from trend.

    Args:
        data: AnalysisData object
        trends: CountryTrends object with trend coefficients
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        Δy*(t) = Δy(t) - trend(t) for each observation
    """
    growth_star = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        y_trend = _compute_growth_trend_value(trends, c, t, quadratic)
        growth_star[i] = data.growth_pcGDP[i] - y_trend

    return growth_star


def compute_growth_trend_values(
    data: AnalysisData, trends: CountryTrends, quadratic: bool = True
) -> np.ndarray:
    """Compute the GDP growth trend values.

    Args:
        data: AnalysisData object
        trends: CountryTrends object with trend coefficients
        quadratic: If True, use quadratic trend; otherwise use linear trend

    Returns:
        Trend values for each observation
    """
    y_trend = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        y_trend[i] = _compute_growth_trend_value(trends, c, t, quadratic)

    return y_trend


def compute_year_means(data: AnalysisData) -> Dict[int, float]:
    """Compute mean dy_i[t] for each year t (equally weighted across countries).

    Returns k[t] = mean_i(growth_pcGDP_i[t]) for each year t.
    """
    from collections import defaultdict

    sums = defaultdict(float)
    counts = defaultdict(int)

    for i in range(data.n_obs):
        yr = data.year[i]
        sums[yr] += data.growth_pcGDP[i]
        counts[yr] += 1

    return {yr: sums[yr] / counts[yr] for yr in sums}


def compute_country_trends_with_k(
    data: AnalysisData, year_means: Dict[int, float]
) -> CountryTrends:
    """Compute country trends on dy_i[t] - k[t] instead of dy_i[t].

    For Approaches 6 and 7, we first compute k[t] = mean(dy_i[t]), then fit
    country trends to the residual dy_i[t] - k[t]. Temperature trends remain
    unchanged (still fit to temperature data).

    Args:
        data: AnalysisData object with observation arrays
        year_means: Dictionary of year -> mean growth rate k[t]

    Returns:
        CountryTrends object with trend coefficients per country
    """
    T0 = {}
    T1 = {}
    T0_quad = {}
    T1_quad = {}
    T2_quad = {}
    y0 = {}
    y1 = {}
    y2 = {}
    y0_lin = {}
    y1_lin = {}

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]
        year_country = data.year[mask]

        # Subtract year means from growth to get dy - k[t]
        growth_adjusted = np.array([
            growth_country[j] - year_means[year_country[j]]
            for j in range(len(growth_country))
        ])

        # Fit linear temperature trend (unchanged)
        T0[country_idx], T1[country_idx] = fit_linear_trend(t_country, temp_country)

        # Fit quadratic temperature trend (unchanged)
        T0_quad[country_idx], T1_quad[country_idx], T2_quad[country_idx] = fit_quadratic_trend(
            t_country, temp_country
        )

        # Fit quadratic GDP growth trend to adjusted growth (dy - k[t])
        y0[country_idx], y1[country_idx], y2[country_idx] = fit_quadratic_trend(
            t_country, growth_adjusted
        )

        # Fit linear GDP growth trend to adjusted growth (dy - k[t])
        y0_lin[country_idx], y1_lin[country_idx] = fit_linear_trend(
            t_country, growth_adjusted
        )

    return CountryTrends(
        T0=T0, T1=T1,
        T0_quad=T0_quad, T1_quad=T1_quad, T2_quad=T2_quad,
        y0=y0, y1=y1, y2=y2, y0_lin=y0_lin, y1_lin=y1_lin
    )


def fit_loess_continuous(
    t: np.ndarray, y: np.ndarray, bandwidth: float, degree: int = 1
) -> np.ndarray:
    """LOESS with continuous bandwidth parameter using tricube weights.

    Args:
        t: Time values
        y: Values to smooth
        bandwidth: Half-width of window in t-units (e.g., years)
        degree: Local polynomial degree (1=linear, 2=quadratic)

    Returns:
        Smoothed values at each input time point
    """
    n = len(t)
    y_smooth = np.zeros(n)

    for i in range(n):
        # Distance from point i to all points
        d = np.abs(t - t[i])

        # Tricube weights with continuous bandwidth
        u = d / bandwidth
        w = np.where(u < 1, (1 - u**3)**3, 0)

        # Need at least degree+1 points with non-zero weight
        if np.sum(w > 0) <= degree:
            y_smooth[i] = y[i]
            continue

        # Weighted local polynomial regression
        X = np.column_stack([np.power(t - t[i], p) for p in range(degree + 1)])
        XtW = X.T * w  # More efficient than X.T @ diag(w)
        XtWX = XtW @ X
        XtWy = XtW @ y
        # Use lstsq for numerical stability with potentially singular matrices
        beta, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        y_smooth[i] = beta[0]  # Intercept = smoothed value at t[i]

    return y_smooth


def fit_loess_trend(
    t: np.ndarray, y: np.ndarray, window_years: float = DEFAULT_LOESS_WINDOW_YEARS
) -> np.ndarray:
    """Fit LOESS trend using continuous bandwidth with tricube weights.

    Args:
        t: Time values
        y: Values to smooth
        window_years: Bandwidth in years (points within this distance get non-zero weight)

    Returns:
        Smoothed values
    """
    n = len(t)
    if n < MIN_LOESS_POINTS:
        return y.copy()

    # Use window_years directly as the bandwidth
    bandwidth = window_years

    # Sort data by t
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]

    # Apply continuous LOESS
    smoothed = fit_loess_continuous(t_sorted, y_sorted, bandwidth, degree=1)

    # Reorder to match original input order
    result = np.zeros(n)
    result[sort_idx] = smoothed

    return result


def compute_country_trends_loess(
    data: AnalysisData, year_means: Dict[int, float],
    window_years: float = DEFAULT_LOESS_WINDOW_YEARS
) -> CountryTrendsLoess:
    """Compute LOESS-smoothed trends for temperature and GDP growth.

    For each country:
    - T_loess: LOESS smoothed temperature
    - y_loess: LOESS smoothed (dy - k[t]), where k[t] is the year mean

    Args:
        data: AnalysisData object with observation arrays
        year_means: Dictionary of year -> mean growth rate k[t]
        window_years: Window size for LOESS smoothing (default: 25 years)

    Returns:
        CountryTrendsLoess object with smoothed values at each observation
    """
    T_loess = np.zeros(data.n_obs)
    y_loess = np.zeros(data.n_obs)

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]
        year_country = data.year[mask]

        # Subtract year means from growth to get dy - k[t]
        growth_adjusted = np.array([
            growth_country[j] - year_means[year_country[j]]
            for j in range(len(growth_country))
        ])

        # Fit LOESS to temperature
        T_loess_country = fit_loess_trend(t_country, temp_country, window_years)

        # Fit LOESS to adjusted growth (dy - k[t])
        y_loess_country = fit_loess_trend(t_country, growth_adjusted, window_years)

        # Store results at observation level
        T_loess[mask] = T_loess_country
        y_loess[mask] = y_loess_country

    return CountryTrendsLoess(T_loess=T_loess, y_loess=y_loess)


def compute_detrended_temperature_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess
) -> np.ndarray:
    """Compute temperature departure from LOESS trend.

    Returns T*(t) = T(t) - T_loess(t)
    """
    return data.temp - trends_loess.T_loess


def compute_detrended_temp_squared_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess
) -> np.ndarray:
    """Compute T² - T_loess² for LOESS temperature detrending.

    This is the coefficient adjustment for h2 in the LOESS-detrended model.
    """
    return data.temp ** 2 - trends_loess.T_loess ** 2


# ==============================================================================
# Weighted Trend Functions (for time-dimension bootstrap)
# ==============================================================================

def fit_linear_trend_weighted(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple:
    """Fit weighted linear trend y = a + b*t using weighted least squares.

    Args:
        t: Time values
        y: Values to fit (may contain NaN for zero-weight observations)
        weights: Observation weights (higher = more influence)

    Returns (intercept, slope). Returns (0, 0) if all weights are zero.
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-10:
        # No effective observations, return zeros
        return 0.0, 0.0

    # Replace NaN with 0 for zero-weight observations to avoid NaN propagation
    y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)

    # Weighted design matrix
    X = np.column_stack([np.ones(len(t)), t])
    W = weights

    # Weighted normal equations: (X'WX) beta = X'Wy
    XtW = X.T * W
    XtWX = XtW @ X
    XtWy = XtW @ y_clean

    # Use lstsq for numerical stability
    coeffs, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
    return coeffs[0], coeffs[1]


def fit_quadratic_trend_weighted(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple:
    """Fit weighted quadratic trend y = a + b*t + c*t² using weighted least squares.

    Args:
        t: Time values
        y: Values to fit (may contain NaN for zero-weight observations)
        weights: Observation weights (higher = more influence)

    Returns (constant, linear_coef, quadratic_coef). Returns (0, 0, 0) if all weights are zero.
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-10:
        # No effective observations, return zeros
        return 0.0, 0.0, 0.0

    # Replace NaN with 0 for zero-weight observations to avoid NaN propagation
    y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)

    # Weighted design matrix
    X = np.column_stack([np.ones(len(t)), t, t * t])
    W = weights

    # Weighted normal equations: (X'WX) beta = X'Wy
    XtW = X.T * W
    XtWX = XtW @ X
    XtWy = XtW @ y_clean

    # Use lstsq for numerical stability
    coeffs, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
    return coeffs[0], coeffs[1], coeffs[2]


def compute_year_means_weighted(
    data: AnalysisData, weights: np.ndarray
) -> Dict[int, float]:
    """Compute weighted mean dy_i[t] for each year t.

    Returns k[t] = weighted_mean_i(growth_pcGDP_i[t]) for each year t.
    For years with zero total weight (unsampled in bootstrap), returns NaN.
    This allows code to access k[yr] for any year, but NaN indicates no data.

    Args:
        data: AnalysisData object
        weights: Observation weights, shape (n_obs,)

    Returns:
        Dictionary mapping year -> weighted mean growth rate (NaN for unsampled years)
    """
    from collections import defaultdict

    weighted_sums = defaultdict(float)
    weight_sums = defaultdict(float)

    # Collect all unique years from data
    unique_years = set(data.year)

    for i in range(data.n_obs):
        yr = data.year[i]
        w = weights[i]
        weighted_sums[yr] += w * data.growth_pcGDP[i]
        weight_sums[yr] += w

    # Return dict for ALL years - NaN for years with no weight (unsampled)
    result = {}
    for yr in unique_years:
        if weight_sums[yr] > 0:
            result[yr] = weighted_sums[yr] / weight_sums[yr]
        else:
            result[yr] = np.nan
    return result


def compute_country_trends_weighted(
    data: AnalysisData, weights: np.ndarray
) -> CountryTrends:
    """Compute weighted polynomial trends for each country.

    Same as compute_country_trends but using weighted least squares.

    Args:
        data: AnalysisData object with observation arrays
        weights: Observation weights, shape (n_obs,)

    Returns:
        CountryTrends object with trend coefficients per country
    """
    T0 = {}
    T1 = {}
    T0_quad = {}
    T1_quad = {}
    T2_quad = {}
    y0 = {}
    y1 = {}
    y2 = {}
    y0_lin = {}
    y1_lin = {}

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]
        w_country = weights[mask]

        # Fit weighted linear temperature trend
        T0[country_idx], T1[country_idx] = fit_linear_trend_weighted(
            t_country, temp_country, w_country
        )

        # Fit weighted quadratic temperature trend
        T0_quad[country_idx], T1_quad[country_idx], T2_quad[country_idx] = fit_quadratic_trend_weighted(
            t_country, temp_country, w_country
        )

        # Fit weighted quadratic GDP growth trend
        y0[country_idx], y1[country_idx], y2[country_idx] = fit_quadratic_trend_weighted(
            t_country, growth_country, w_country
        )

        # Fit weighted linear GDP growth trend
        y0_lin[country_idx], y1_lin[country_idx] = fit_linear_trend_weighted(
            t_country, growth_country, w_country
        )

    return CountryTrends(
        T0=T0, T1=T1,
        T0_quad=T0_quad, T1_quad=T1_quad, T2_quad=T2_quad,
        y0=y0, y1=y1, y2=y2, y0_lin=y0_lin, y1_lin=y1_lin
    )


def compute_country_trends_with_k_weighted(
    data: AnalysisData, year_means: Dict[int, float], weights: np.ndarray
) -> CountryTrends:
    """Compute weighted country trends on dy_i[t] - k[t] instead of dy_i[t].

    Same as compute_country_trends_with_k but using weighted least squares.

    Args:
        data: AnalysisData object with observation arrays
        year_means: Dictionary of year -> mean growth rate k[t]
        weights: Observation weights, shape (n_obs,)

    Returns:
        CountryTrends object with trend coefficients per country
    """
    T0 = {}
    T1 = {}
    T0_quad = {}
    T1_quad = {}
    T2_quad = {}
    y0 = {}
    y1 = {}
    y2 = {}
    y0_lin = {}
    y1_lin = {}

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]
        year_country = data.year[mask]
        w_country = weights[mask]

        # Subtract year means from growth to get dy - k[t]
        growth_adjusted = np.array([
            growth_country[j] - year_means[year_country[j]]
            for j in range(len(growth_country))
        ])

        # Fit weighted linear temperature trend (unchanged)
        T0[country_idx], T1[country_idx] = fit_linear_trend_weighted(
            t_country, temp_country, w_country
        )

        # Fit weighted quadratic temperature trend (unchanged)
        T0_quad[country_idx], T1_quad[country_idx], T2_quad[country_idx] = fit_quadratic_trend_weighted(
            t_country, temp_country, w_country
        )

        # Fit weighted quadratic GDP growth trend to adjusted growth (dy - k[t])
        y0[country_idx], y1[country_idx], y2[country_idx] = fit_quadratic_trend_weighted(
            t_country, growth_adjusted, w_country
        )

        # Fit weighted linear GDP growth trend to adjusted growth (dy - k[t])
        y0_lin[country_idx], y1_lin[country_idx] = fit_linear_trend_weighted(
            t_country, growth_adjusted, w_country
        )

    return CountryTrends(
        T0=T0, T1=T1,
        T0_quad=T0_quad, T1_quad=T1_quad, T2_quad=T2_quad,
        y0=y0, y1=y1, y2=y2, y0_lin=y0_lin, y1_lin=y1_lin
    )


def fit_loess_continuous_weighted(
    t: np.ndarray, y: np.ndarray, obs_weights: np.ndarray,
    bandwidth: float, degree: int = 1
) -> np.ndarray:
    """LOESS with observation weights and continuous bandwidth using tricube weights.

    The total weight for each point is obs_weights[j] * tricube_weight(distance[j]).

    Args:
        t: Time values
        y: Values to smooth (may contain NaN for zero-weight observations)
        obs_weights: Observation weights (from bootstrap sampling)
        bandwidth: Half-width of window in t-units (e.g., years)
        degree: Local polynomial degree (1=linear, 2=quadratic)

    Returns:
        Smoothed values at each input time point
    """
    n = len(t)
    y_smooth = np.zeros(n)

    # Replace NaN with 0 for zero-weight observations to avoid NaN propagation
    y_clean = np.where(np.isnan(y) & (obs_weights == 0), 0, y)

    for i in range(n):
        # Distance from point i to all points
        d = np.abs(t - t[i])

        # Tricube weights with continuous bandwidth
        u = d / bandwidth
        tricube_w = np.where(u < 1, (1 - u**3)**3, 0)

        # Combined weight: observation weight * tricube weight
        w = obs_weights * tricube_w

        # Need at least degree+1 points with non-zero weight
        if np.sum(w > 0) <= degree:
            y_smooth[i] = y_clean[i]
            continue

        # Weighted local polynomial regression
        X = np.column_stack([np.power(t - t[i], p) for p in range(degree + 1)])
        XtW = X.T * w
        XtWX = XtW @ X
        XtWy = XtW @ y_clean

        # Use lstsq for numerical stability
        try:
            beta, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
            y_smooth[i] = beta[0]  # Intercept = smoothed value at t[i]
        except np.linalg.LinAlgError:
            y_smooth[i] = y_clean[i]

    return y_smooth


def fit_loess_trend_weighted(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray,
    window_years: float = DEFAULT_LOESS_WINDOW_YEARS
) -> np.ndarray:
    """Fit weighted LOESS trend using observation weights and tricube kernel.

    Args:
        t: Time values
        y: Values to smooth
        weights: Observation weights (from bootstrap sampling)
        window_years: Bandwidth in years

    Returns:
        Smoothed values
    """
    n = len(t)
    if n < MIN_LOESS_POINTS:
        return y.copy()

    bandwidth = window_years

    # Sort data by t
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]
    w_sorted = weights[sort_idx]

    # Apply weighted LOESS
    smoothed = fit_loess_continuous_weighted(t_sorted, y_sorted, w_sorted, bandwidth, degree=1)

    # Reorder to match original input order
    result = np.zeros(n)
    result[sort_idx] = smoothed

    return result


def compute_country_trends_loess_weighted(
    data: AnalysisData, year_means: Dict[int, float], weights: np.ndarray,
    window_years: float = DEFAULT_LOESS_WINDOW_YEARS
) -> CountryTrendsLoess:
    """Compute weighted LOESS-smoothed trends for temperature and GDP growth.

    Same as compute_country_trends_loess but using observation weights.

    Args:
        data: AnalysisData object with observation arrays
        year_means: Dictionary of year -> mean growth rate k[t]
        weights: Observation weights, shape (n_obs,)
        window_years: Window size for LOESS smoothing

    Returns:
        CountryTrendsLoess object with smoothed values at each observation
    """
    T_loess = np.zeros(data.n_obs)
    y_loess = np.zeros(data.n_obs)

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]
        year_country = data.year[mask]
        w_country = weights[mask]

        # Subtract year means from growth to get dy - k[t]
        growth_adjusted = np.array([
            growth_country[j] - year_means[year_country[j]]
            for j in range(len(growth_country))
        ])

        # Fit weighted LOESS to temperature
        T_loess_country = fit_loess_trend_weighted(t_country, temp_country, w_country, window_years)

        # Fit weighted LOESS to adjusted growth (dy - k[t])
        y_loess_country = fit_loess_trend_weighted(t_country, growth_adjusted, w_country, window_years)

        # Store results at observation level
        T_loess[mask] = T_loess_country
        y_loess[mask] = y_loess_country

    return CountryTrendsLoess(T_loess=T_loess, y_loess=y_loess)
