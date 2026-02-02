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
DEFAULT_LOESS_WINDOW_YEARS = 25

# Minimum number of points required for LOESS fitting
MIN_LOESS_POINTS = 3

from dataclasses import dataclass
from typing import Dict
from statsmodels.nonparametric.smoothers_lowess import lowess
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


def compute_growth_trend_values_linear(data: AnalysisData, trends: CountryTrends) -> np.ndarray:
    """Compute the linear GDP growth trend values.

    This is a convenience wrapper for compute_growth_trend_values with quadratic=False.

    Returns y0_lin + y1_lin*t for each observation.
    """
    return compute_growth_trend_values(data, trends, quadratic=False)


def compute_detrended_temperature_quadratic(
    data: AnalysisData, trends: CountryTrends
) -> np.ndarray:
    """Compute temperature departure from quadratic trend.

    This is a convenience wrapper for compute_detrended_temperature with quadratic=True.

    Returns T*(t) = T(t) - (T0_quad + T1_quad*t + T2_quad*t²)
    """
    return compute_detrended_temperature(data, trends, quadratic=True)


def compute_detrended_temp_squared_quadratic(
    data: AnalysisData, trends: CountryTrends
) -> np.ndarray:
    """Compute T² - quadratic_trend² for quadratic temperature detrending.

    This is a convenience wrapper for compute_detrended_temp_squared with quadratic=True.
    """
    return compute_detrended_temp_squared(data, trends, quadratic=True)


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


def fit_loess_trend(
    t: np.ndarray, y: np.ndarray, window_years: int = DEFAULT_LOESS_WINDOW_YEARS
) -> np.ndarray:
    """Fit LOESS (locally weighted scatterplot smoothing) trend.

    Args:
        t: Time values (e.g., year or normalized time)
        y: Values to smooth
        window_years: Window size in years for LOESS smoothing

    Returns:
        Smoothed values at each input time point
    """
    n = len(t)
    if n < MIN_LOESS_POINTS:
        # Not enough points for LOESS, return original values
        return y.copy()

    # Calculate frac based on window_years and data span
    # frac is the fraction of data used for each local regression
    t_range = t.max() - t.min()
    if t_range == 0:
        return y.copy()

    # frac should give roughly window_years worth of data
    frac = min(window_years / t_range, 1.0)
    # Ensure we use at least MIN_LOESS_POINTS
    frac = max(frac, float(MIN_LOESS_POINTS) / n)

    # Sort data by t for LOESS
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]

    # Fit LOESS - returns sorted results
    smoothed = lowess(y_sorted, t_sorted, frac=frac, return_sorted=False)

    # Reorder to match original input order
    result = np.zeros(n)
    result[sort_idx] = smoothed

    return result


def compute_country_trends_loess(
    data: AnalysisData, year_means: Dict[int, float],
    window_years: int = DEFAULT_LOESS_WINDOW_YEARS
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
