"""Persistence/decay utilities.

This module contains the time-recursive accumulator helpers used by the
persistence/decay response family (Approaches DJ/DP/DL) and by downstream
post-processing (e.g., saving h(T) values).

The persistence models represent the *convolved* temperature response using
lagged accumulators:

    A_T(t)  = T(t)  + (1 - h4) * A_T(t-1)
    A_T2(t) = T(t)^2 + (1 - h4) * A_T2(t-1)

The regressors used in the convolved response are constructed from lagged
values A_T(t-1), A_T2(t-1) and a pre-first-year correction term.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import linalg

from .data_loader import AnalysisData


def compute_persistence_accumulators(data: AnalysisData, h4: float) -> tuple:
    """Compute lagged persistence accumulators for observed temperatures.

    For each country, computes:
    - A_T(t) = T(t) + (1-h4) * A_T(t-1), with A_T(first_year) = T(first_year)
    - A_T2(t) = T^2(t) + (1-h4) * A_T2(t-1), with A_T2(first_year) = T^2(first_year)

    Returns the LAGGED values A_T(t-1) and A_T2(t-1) for use in regressors.
    For the first year of each country, returns 0 (no lagged value available).

    Args:
        data: AnalysisData object
        h4: Persistence decay parameter [0, 1]

    Returns:
        Tuple of (A_T_lag, A_T2_lag), each of shape (n_obs,)
    """
    decay = 1 - h4
    A_T_lag = np.zeros(data.n_obs)
    A_T2_lag = np.zeros(data.n_obs)

    for c in range(data.n_countries):
        # Get observation indices for this country, sorted by year
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]
        # Sort by year
        years_for_country = data.year[country_indices]
        sorted_order = np.argsort(years_for_country)
        sorted_indices = country_indices[sorted_order]

        # Compute accumulators
        A_T = 0.0
        A_T2 = 0.0
        for i, idx in enumerate(sorted_indices):
            T_val = data.temp[idx]
            T2_val = T_val ** 2

            if i == 0:
                # First year: no lagged value available
                A_T_lag[idx] = 0.0
                A_T2_lag[idx] = 0.0
                # Initialize accumulator with first year's value
                A_T = T_val
                A_T2 = T2_val
            else:
                # Store lagged accumulator (from previous iteration)
                A_T_lag[idx] = A_T
                A_T2_lag[idx] = A_T2
                # Update accumulator: A(t) = T(t) + (1-h4) * A(t-1)
                A_T = T_val + decay * A_T
                A_T2 = T2_val + decay * A_T2

    return A_T_lag, A_T2_lag


def compute_persistence_accumulators_at_T(
    data: AnalysisData, h4: float, T_values: np.ndarray
) -> tuple:
    """Compute lagged persistence accumulators for arbitrary temperature values.

    Same as compute_persistence_accumulators but using provided T_values
    instead of data.temp. Used for computing accumulators at trend temperatures.

    Args:
        data: AnalysisData object (for country/year structure)
        h4: Persistence decay parameter [0, 1]
        T_values: Temperature values to use, shape (n_obs,)

    Returns:
        Tuple of (A_T_lag, A_T2_lag), each of shape (n_obs,)
    """
    decay = 1 - h4
    A_T_lag = np.zeros(data.n_obs)
    A_T2_lag = np.zeros(data.n_obs)

    for c in range(data.n_countries):
        # Get observation indices for this country, sorted by year
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]
        # Sort by year
        years_for_country = data.year[country_indices]
        sorted_order = np.argsort(years_for_country)
        sorted_indices = country_indices[sorted_order]

        # Compute accumulators
        A_T = 0.0
        A_T2 = 0.0
        for i, idx in enumerate(sorted_indices):
            T_val = T_values[idx]
            T2_val = T_val ** 2

            if i == 0:
                # First year: no lagged value available
                A_T_lag[idx] = 0.0
                A_T2_lag[idx] = 0.0
                # Initialize accumulator with first year's value
                A_T = T_val
                A_T2 = T2_val
            else:
                # Store lagged accumulator (from previous iteration)
                A_T_lag[idx] = A_T
                A_T2_lag[idx] = A_T2
                # Update accumulator: A(t) = T(t) + (1-h4) * A(t-1)
                A_T = T_val + decay * A_T
                A_T2 = T2_val + decay * A_T2

    return A_T_lag, A_T2_lag


def compute_pre_first_year_correction(
    data: AnalysisData, h4: float, T_values: np.ndarray = None
) -> tuple:
    """Compute pre-first-year correction for persistence decay model.

    The persistence model assumes temperature was constant at T(first_year) before
    the first observation. This creates a correction term:

        correction(t) = (1-h4)^(t - first_year) * T(first_year)

    This accounts for the accumulated effect of the assumed constant pre-history.

    Args:
        data: AnalysisData object
        h4: Persistence decay parameter [0, 1]
        T_values: Optional temperature values (default: data.temp)

    Returns:
        Tuple of (correction_T, correction_T2), each of shape (n_obs,)
    """
    if T_values is None:
        T_values = data.temp

    decay = 1 - h4
    correction_T = np.zeros(data.n_obs)
    correction_T2 = np.zeros(data.n_obs)

    for c in range(data.n_countries):
        # Get observation indices for this country, sorted by year
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]
        years_for_country = data.year[country_indices]
        sorted_order = np.argsort(years_for_country)
        sorted_indices = country_indices[sorted_order]
        sorted_years = years_for_country[sorted_order]

        # First year's temperature for this country
        first_year = sorted_years[0]
        first_idx = sorted_indices[0]
        T_first = T_values[first_idx]
        T2_first = T_first ** 2

        # Compute correction for each year
        for i, idx in enumerate(sorted_indices):
            years_since_first = sorted_years[i] - first_year
            decay_factor = decay ** years_since_first
            correction_T[idx] = decay_factor * T_first
            correction_T2[idx] = decay_factor * T2_first

    return correction_T, correction_T2


def compute_T_linear_at_first_year(data: AnalysisData, weights: np.ndarray = None) -> np.ndarray:
    """Compute linear temperature trend evaluated at each country's first year.

    For each country, fits a linear OLS regression T = a + b*t to all observations,
    then evaluates T_trend at the first year. This provides a smoothed baseline
    temperature for the pre-history assumption in persistence decay models.

    Returns an array where each observation has its country's T_linear(first_year).

    Args:
        data: AnalysisData object
        weights: Optional observation weights for weighted least squares

    Returns:
        Array of shape (n_obs,) with T_linear(first_year) for each observation's country
    """
    T_linear_first = np.zeros(data.n_obs)

    for c in range(data.n_countries):
        # Get observation indices for this country
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]

        # Get time and temperature for this country
        t_country = data.time[country_indices]
        T_country = data.temp[country_indices]

        n_c = len(t_country)
        X_lin = np.column_stack([np.ones(n_c), t_country])

        if weights is not None:
            w_country = weights[country_indices]
            total_weight = np.sum(w_country)
            if total_weight < 1e-10:
                # No effective observations for this country
                T_linear_first[country_mask] = 0.0
                continue
            # Weighted normal equations: (X'WX)^-1 X'Wy
            XtW = X_lin.T * w_country
            XtWX = XtW @ X_lin
            XtWy = XtW @ T_country
            coeffs, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        else:
            coeffs, _, _, _ = linalg.lstsq(X_lin, T_country)
        a, b = coeffs

        # Find first year for this country
        years_for_country = data.year[country_indices]
        first_year_idx = np.argmin(years_for_country)
        t_first = t_country[first_year_idx]

        # Evaluate T_linear at first year
        T_at_first = a + b * t_first

        # Set all observations for this country to T_linear(first_year)
        T_linear_first[country_mask] = T_at_first

    return T_linear_first

