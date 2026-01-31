"""Country-level detrending functions.

This module computes:
1. Linear temperature trends: T_i(t) = T_{0,i} + T_{1,i} * t
2. Quadratic GDP growth trends: Δy_i(t) = y_{0,i} + y_{1,i} * t + y_{2,i} * t²
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from .data_loader import AnalysisData


@dataclass
class CountryTrends:
    """Container for country-level trend coefficients."""
    # Temperature linear trend: T(t) = T0 + T1 * t
    T0: Dict[int, float]  # country_idx -> intercept
    T1: Dict[int, float]  # country_idx -> slope

    # GDP growth quadratic trend: Δy(t) = y0 + y1*t + y2*t²
    y0: Dict[int, float]  # country_idx -> constant
    y1: Dict[int, float]  # country_idx -> linear coef
    y2: Dict[int, float]  # country_idx -> quadratic coef


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
    """Compute linear temperature and quadratic GDP growth trends for each country.

    For each country i:
    - Fits T_i(t) = T_{0,i} + T_{1,i} * t
    - Fits Δy_i(t) = y_{0,i} + y_{1,i} * t + y_{2,i} * t²

    Args:
        data: AnalysisData object with observation arrays

    Returns:
        CountryTrends object with trend coefficients per country
    """
    T0 = {}
    T1 = {}
    y0 = {}
    y1 = {}
    y2 = {}

    for country_idx in range(data.n_countries):
        # Get observations for this country
        mask = data.country_idx == country_idx
        t_country = data.time[mask]
        temp_country = data.temp[mask]
        growth_country = data.growth_pcGDP[mask]

        # Fit linear temperature trend
        T0[country_idx], T1[country_idx] = fit_linear_trend(t_country, temp_country)

        # Fit quadratic GDP growth trend
        y0[country_idx], y1[country_idx], y2[country_idx] = fit_quadratic_trend(
            t_country, growth_country
        )

    return CountryTrends(T0=T0, T1=T1, y0=y0, y1=y1, y2=y2)


def compute_detrended_temperature(data: AnalysisData, trends: CountryTrends) -> np.ndarray:
    """Compute temperature departure from linear trend.

    Returns T*(t) = T(t) - (T0 + T1*t)
    """
    T_star = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend = trends.T0[c] + trends.T1[c] * t
        T_star[i] = data.temp[i] - T_trend

    return T_star


def compute_detrended_temp_squared(data: AnalysisData, trends: CountryTrends) -> np.ndarray:
    """Compute T² - (T0 + T1*t)² for the linear temperature detrending.

    This is the coefficient adjustment for h2 in the temperature-detrended model.
    """
    T2_detrend = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T = data.temp[i]
        T_trend = trends.T0[c] + trends.T1[c] * t
        T2_detrend[i] = T * T - T_trend * T_trend

    return T2_detrend


def compute_detrended_growth(data: AnalysisData, trends: CountryTrends) -> np.ndarray:
    """Compute GDP growth departure from quadratic trend.

    Returns Δy*(t) = Δy(t) - (y0 + y1*t + y2*t²)
    """
    growth_star = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        y_trend = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        growth_star[i] = data.growth_pcGDP[i] - y_trend

    return growth_star


def compute_growth_trend_values(data: AnalysisData, trends: CountryTrends) -> np.ndarray:
    """Compute the quadratic GDP growth trend values.

    Returns y0 + y1*t + y2*t² for each observation.
    """
    y_trend = np.zeros(data.n_obs)

    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        y_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t

    return y_trend
