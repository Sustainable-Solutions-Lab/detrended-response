"""OLS fitting for climate-GDP response approaches.

Approaches (publication-ready):
    approach0: Conjoined OLS with country time trends and year fixed effects
    approach1: Pre-computed k with linear T + quadratic GDP detrending
    approach2: Pre-computed k with LOESS trends
    approach3: Piecewise quadratic response with LOESS
    approach4: Persistence decay model with LOESS
    approach0h0: Null model (h1=h2=0) for approach0
    approach1h0: Null model (h1=h2=0) for approach1
    approach2h0: Null model (h1=h2=0) for approach2

Exploratory methods (kept as methods):
    method4: T response with quadratic departure term
    method6: Interaction model
    method7: Lagged deviation model
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf
from dataclasses import dataclass
from typing import Dict
from .data_loader import AnalysisData
from .detrending import (
    CountryTrends,
    CountryTrendsLoess,
    compute_detrended_temperature,
    compute_detrended_temp_squared,
    compute_detrended_temperature_loess,
    compute_detrended_temp_squared_loess,
)

# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_T_optimal(h1: float, h2: float) -> float:
    """Compute optimal temperature from quadratic response coefficients.

    For the quadratic response h(T) = h1*T + h2*T², the optimal temperature
    is at dh/dT = 0, which gives T_opt = -h1 / (2*h2).

    Args:
        h1: Linear temperature coefficient
        h2: Quadratic temperature coefficient

    Returns:
        Optimal temperature, or np.nan if h2 == 0
    """
    return -h1 / (2 * h2) if h2 != 0 else np.nan


def compute_rms_h(h1: float, h2: float, temp: np.ndarray) -> float:
    """Compute RMS of climate response h(T) = h1*T + h2*T².

    Args:
        h1: Linear temperature coefficient
        h2: Quadratic temperature coefficient
        temp: Temperature array

    Returns:
        RMS of h(T) across all observations
    """
    h_T = h1 * temp + h2 * temp ** 2
    return np.sqrt(np.mean(h_T ** 2))


def compute_variance_decomposition(components: Dict[str, np.ndarray], dy: np.ndarray, total_r_squared: float = None) -> dict:
    """Compute variance decomposition of dy into named components.

    Given dy = C1 + C2 + ... + Cn + epsilon (where epsilon is the remainder),
    decomposes Var(dy) into variance and covariance contributions from all
    components including epsilon. The sum equals 1.0 by construction.

    Args:
        components: Dict mapping component names to arrays (e.g.,
            {"h_Tstar": array, "h_Ttrend": array, "h_cross": array, "j": array, "k": array})
        dy: Original dependent variable (GDP growth rate)

    Returns:
        Dict with:
            - component_names: list of all component names (including 'epsilon')
            - var_{name}: Var(component) / Var(dy) for each component
            - cov_{name1}_{name2}: 2*Cov(C1,C2) / Var(dy) for each pair
            - rms_{name}: RMS of each component
            - rms_dy: std(dy)
            - sum_check: sum of all variance and covariance fractions (should be 1.0)
    """
    # Compute epsilon as remainder
    component_sum = sum(components.values())
    epsilon = dy - component_sum

    # All components including epsilon
    all_comps = dict(components)
    all_comps['epsilon'] = epsilon

    var_dy = np.var(dy)
    names = list(all_comps.keys())

    result = {
        'component_names': names,
        'rms_dy': np.std(dy),
    }

    # Variance fractions and RMS for each component
    for name in names:
        vals = all_comps[name]
        result[f'var_{name}'] = np.var(vals) / var_dy
        result[f'rms_{name}'] = np.sqrt(np.mean(vals ** 2))

    # Covariance fractions for all pairs (ddof=0 to match np.var default)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            cov_val = np.cov(all_comps[n1], all_comps[n2], ddof=0)[0, 1]
            result[f'cov_{n1}_{n2}'] = 2 * cov_val / var_dy

    # Sum check
    total = 0.0
    for key, val in result.items():
        if key.startswith('var_') or key.startswith('cov_'):
            total += val
    result['sum_check'] = total

    # Key variance ratios: Var(component) / Var(dy) for major groups
    # var_ratio_h_T: climate response to full temperature (sum of all h components)
    h_sum = sum(vals for name, vals in all_comps.items()
                if name not in ('j', 'k', 'epsilon'))
    result['var_ratio_h_T'] = np.var(h_sum) / var_dy

    # var_ratio_h_Tstar: climate response to detrended temperature only
    h_star_candidates = [n for n in names if n in ('h_Tstar', 'g_h_Tstar', 'h_T')]
    if h_star_candidates:
        result['var_ratio_h_Tstar'] = np.var(all_comps[h_star_candidates[0]]) / var_dy

    # Copy single-component ratios with var_ratio_ prefix for consistency
    result['var_ratio_j'] = result['var_j']
    result['var_ratio_k'] = result['var_k']

    # var_ratio_cross: covariance remainder so that
    # Total_R² = var_ratio_h_T + var_ratio_j + var_ratio_k + var_ratio_cross
    if total_r_squared is not None:
        result['var_ratio_cross'] = total_r_squared - result['var_ratio_h_T'] - result['var_ratio_j'] - result['var_ratio_k']

    return result


def compute_variance_attribution(
    Delta_u: np.ndarray,
    v: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    dy: np.ndarray
) -> dict:
    """Compute variance attribution across five additive components using equal-split covariance allocation.

    Given the fitted identity (must hold exactly in-sample):
        Δy_i(t) = Δu_i(t) + v_i(t) + j_i(t) + k(t) + ε_i(t)

    where:
        Δu_i(t) = h(T_i(t)) - h(T_trend(T_i(t)))  # increment from using actual T rather than trended T
        v_i(t)  = h(T_trend(T_i(t)))              # baseline climate term at trended temperature
        j_i(t)  = j_i(t)                          # country-specific growth trend component
        k(t)    = k(t)                            # time fixed effect component
        ε_i(t)  = regression residual             # Δy - [h(T) + j + k]

    Quantifies "how much of Var(Δy)" is attributable to each component using symmetric covariance allocation:

        C_Δu = Var(Δu) + Cov(Δu, v) + Cov(Δu, j) + Cov(Δu, k) + Cov(Δu, ε)
        C_v  = Var(v)  + Cov(v, Δu) + Cov(v, j)  + Cov(v, k)  + Cov(v, ε)
        C_j  = Var(j)  + Cov(j, Δu) + Cov(j, v)  + Cov(j, k)  + Cov(j, ε)
        C_k  = Var(k)  + Cov(k, Δu) + Cov(k, v)  + Cov(k, j)  + Cov(k, ε)
        C_ε  = Var(ε)  + Cov(ε, Δu) + Cov(ε, v)  + Cov(ε, j)  + Cov(ε, k)

    These contributions satisfy exactly: C_Δu + C_v + C_j + C_k + C_ε = Var(Δy)

    Args:
        Delta_u: Increment component Δu = h(T) - h(T_trend) for each observation
        v: Baseline component v = h(T_trend) for each observation
        j: Country-specific trend component for each observation
        k: Time fixed effect component for each observation
        epsilon: Regression residual ε = Δy - [Δu + v + j + k] for each observation
        dy: Original dependent variable (GDP growth rate)

    Returns:
        Dict with:
            - C_Delta_u, C_v, C_j, C_k, C_epsilon: Variance contributions (absolute)
            - s_Delta_u, s_v, s_j, s_k, s_epsilon: Variance contribution shares (fractions of Var(Δy))
            - var_dy: Var(Δy)
            - Sigma_*: Full covariance matrix entries (15 unique entries for 5x5 symmetric matrix)
            - sum_check: C_Δu + C_v + C_j + C_k + C_ε (should equal var_dy exactly)
            - cov_epsilon_*: Covariances between residual and fitted components (should be ~0 for OLS)

    Notes:
        - Negative C values are allowed and meaningful (variance cancellation)
        - Uses bias=True (ddof=0) for covariance calculations to ensure exact sum = Var(Δy)
        - All arrays are demeaned before computing covariances (constants don't affect relative contributions)
        - If ε is OLS residual, Cov(ε, fitted components) should be ~0 (orthogonality check)
    """
    # Demean all arrays to ensure constants don't affect covariance
    Delta_u_dm = Delta_u - np.mean(Delta_u)
    v_dm = v - np.mean(v)
    j_dm = j - np.mean(j)
    k_dm = k - np.mean(k)
    epsilon_dm = epsilon - np.mean(epsilon)
    dy_dm = dy - np.mean(dy)

    # Stack into matrix for covariance computation: shape (5, n_obs)
    components = np.vstack([Delta_u_dm, v_dm, j_dm, k_dm, epsilon_dm])

    # Compute covariance matrix Σ: shape (5, 5)
    # Use bias=True (ddof=0) to ensure sum of contributions equals variance exactly
    Sigma = np.cov(components, bias=True)

    # Variance contributions: row-sums of Σ
    # C_a = Var(a) + sum_{b≠a} Cov(a, b) = sum_b Σ_ab
    C = Sigma.sum(axis=1)

    C_Delta_u, C_v, C_j, C_k, C_epsilon = C

    # Variance of Δy (should equal sum of contributions)
    var_dy = np.var(dy_dm, ddof=0)

    # Shares: normalize by Var(Δy)
    s_Delta_u = C_Delta_u / var_dy if var_dy > 0 else 0
    s_v = C_v / var_dy if var_dy > 0 else 0
    s_j = C_j / var_dy if var_dy > 0 else 0
    s_k = C_k / var_dy if var_dy > 0 else 0
    s_epsilon = C_epsilon / var_dy if var_dy > 0 else 0

    # Sum check: should equal Var(Δy) exactly
    sum_check = C_Delta_u + C_v + C_j + C_k + C_epsilon

    return {
        # Variance contributions (absolute)
        'C_Delta_u': C_Delta_u,
        'C_v': C_v,
        'C_j': C_j,
        'C_k': C_k,
        'C_epsilon': C_epsilon,
        # Variance contribution shares (fractions)
        's_Delta_u': s_Delta_u,
        's_v': s_v,
        's_j': s_j,
        's_k': s_k,
        's_epsilon': s_epsilon,
        # Reference values
        'var_dy': var_dy,
        'sum_check': sum_check,
        # Full covariance matrix (15 unique entries for 5x5 symmetric matrix)
        'Sigma_Delta_u_Delta_u': Sigma[0, 0],
        'Sigma_Delta_u_v': Sigma[0, 1],
        'Sigma_Delta_u_j': Sigma[0, 2],
        'Sigma_Delta_u_k': Sigma[0, 3],
        'Sigma_Delta_u_epsilon': Sigma[0, 4],
        'Sigma_v_v': Sigma[1, 1],
        'Sigma_v_j': Sigma[1, 2],
        'Sigma_v_k': Sigma[1, 3],
        'Sigma_v_epsilon': Sigma[1, 4],
        'Sigma_j_j': Sigma[2, 2],
        'Sigma_j_k': Sigma[2, 3],
        'Sigma_j_epsilon': Sigma[2, 4],
        'Sigma_k_k': Sigma[3, 3],
        'Sigma_k_epsilon': Sigma[3, 4],
        'Sigma_epsilon_epsilon': Sigma[4, 4],
        # Combined h(T) = Delta_u + v terms (h(T) instead of separating h(T)-h(Ttr) and h(Ttr))
        'Sigma_h_h': Sigma[0, 0] + Sigma[1, 1] + 2 * Sigma[0, 1],  # Var(h(T)) = Var(Delta_u + v)
        'Sigma_h_j': Sigma[0, 2] + Sigma[1, 2],  # Cov(h(T), j)
        'Sigma_h_k': Sigma[0, 3] + Sigma[1, 3],  # Cov(h(T), k)
        'Sigma_h_epsilon': Sigma[0, 4] + Sigma[1, 4],  # Cov(h(T), epsilon)
        # Orthogonality checks (residual should be orthogonal to fitted components for OLS)
        'cov_epsilon_Delta_u': Sigma[4, 0],
        'cov_epsilon_v': Sigma[4, 1],
        'cov_epsilon_j': Sigma[4, 2],
        'cov_epsilon_k': Sigma[4, 3],
    }


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
    T_opt: float           # Optimal temperature = -h1 / (2*h2)
    total_r_squared: float # Variance explained in original dy
    rms_imbalance: float = None  # RMS of h(T_trend) + j_trend + k
    rms_h: float = None          # RMS of h(T) - climate response magnitude
    imbalance_ratio: float = None  # rms_imbalance / rms_h

    # Variance decomposition (replaces old var_frac/cov_frac fields)
    var_decomp: dict = None

    # Variance attribution (4-component decomposition with covariance allocation)
    var_attrib: dict = None


@dataclass
class FitResultApproach8:
    """Container for piecewise quadratic temperature response results.

    Model: h(T) = h4 * (T - T_opt)²  if T > T_opt
           h(T) = h2 * (T - T_opt)²   if T ≤ T_opt

    This model:
    - Has an optimum at T_opt where h(T_opt) = 0
    - Allows different curvatures above vs below T_opt
    - Is simple, interpretable, and requires nonlinear optimization for T_opt

    Parameters:
    - h1: Linear term (always 0 for piecewise model)
    - h2: Curvature for T ≤ T_opt (cold side)
    - h4: Curvature for T > T_opt (hot side)
    - T_opt: Optimal temperature (breakpoint)
    """
    approach: str
    h2: float              # Curvature for T ≤ T_opt (formerly h2_low)
    h2_se: float           # SE from inner OLS
    h4: float              # Curvature for T > T_opt (formerly h2_high)
    h4_se: float           # SE from inner OLS
    T_opt: float           # Optimal temperature (breakpoint)
    T_opt_se: float        # SE from numerical Hessian
    k: Dict[int, float]    # Year fixed effects
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int          # = 3 (h2, h4, T_opt)
    residuals: np.ndarray
    total_r_squared: float
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None
    # Compatibility field for plotting that expects h1
    h1: float = 0.0        # Linear term (always 0 for piecewise model)
    h1_se: float = 0.0     # SE for h1 (always 0)


@dataclass
class FitResultApproach4:
    """Container for approach4 (persistence decay) results.

    Model: h_conv(T(t)) = h(T(t)) - h4 * sum_{k=1}^{n} (1-h4)^{k-1} * h(T(t-k))

    where h(T) = h1*T + h2*T^2 and n = t - 1961.

    Using accumulators for efficient computation:
    - A_T(t) = T(t) + (1-h4) * A_T(t-1), with A_T(1961) = T(1961)
    - A_T2(t) = T^2(t) + (1-h4) * A_T2(t-1), with A_T2(1961) = T^2(1961)

    Modified regressors:
    - X1(t) = T(t) - h4 * A_T(t-1)
    - X2(t) = T^2(t) - h4 * A_T2(t-1)

    Edge cases:
    - h4 = 0: Full persistence (h_conv = h(T))
    - h4 = 1: No persistence (first-difference behavior)
    """
    approach: str
    h1: float              # Linear temperature coefficient
    h2: float              # Quadratic temperature coefficient
    h1_se: float
    h2_se: float
    h4: float              # Persistence decay parameter [0, 1]
    h4_se: float
    k: Dict[int, float]    # Year fixed effects
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int          # = 3
    residuals: np.ndarray
    T_opt: float           # -h1/(2*h2)
    total_r_squared: float
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None


@dataclass
class FitResultApproach6ab:
    """Container for Approach 6a/6b results with T/departure decomposition.

    Model 6a: h(T, Ttrend) = h1*T + h2*T² + h3*(T-Ttrend) + h4*(T-Ttrend)²
    Model 6b: h(T, Ttrend) = h1*T + h2*T²  (departure terms h3,h4 are zero)

    Where:
        - delta-y*(t) = dy(t) - k(t) - f_trend(dy(t) - k(t)) (same as approach 6)
        - h1, h2: Response to actual temperature T
        - h3, h4: Response to departure from trend (T - Ttrend)

    When T = Ttrend: h = h1*T + h2*T² (pure T response, no departure effect)

    Parameters:
    - h1, h2: Response to actual temperature T
    - h3, h4: Response to departure (T - Ttrend) (zero for 6b)
    - T_opt: Optimal actual temperature = -h1/(2*h2)
    - T_dep_opt: Optimal departure = -h3/(2*h4)
    """
    approach: str
    # Actual T coefficients
    h1: float              # Linear coef for actual T
    h2: float              # Quadratic coef for actual T
    h1_se: float
    h2_se: float
    # Departure (T - Ttrend) coefficients (zero for 6b)
    h3: float              # Linear coef for departure
    h4: float              # Quadratic coef for departure
    h3_se: float
    h4_se: float
    # Year effects
    k: Dict[int, float]
    # Fit stats
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int
    residuals: np.ndarray
    # Derived
    T_opt: float           # -h1/(2*h2) - optimal actual T
    T_dep_opt: float       # -h3/(2*h4) - optimal departure
    total_r_squared: float
    # Diagnostics
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None


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


def compute_rms_imbalance(
    h1: float, h2: float,
    T_trend: np.ndarray,
    j_trend: np.ndarray,
    k_values: np.ndarray
) -> float:
    """Compute RMS of imbalance: h(T_trend) + j_trend + k.

    If detrending and climate response were perfect, this would be zero:
        0 = h(T_trend) + j_trend + k

    where:
        h(T_trend) = h1*T_trend + h2*T_trend²
        j_trend = country-specific GDP growth trend (after removing year means)
        k = year mean GDP growth

    Args:
        h1: Linear temperature coefficient
        h2: Quadratic temperature coefficient
        T_trend: Temperature trend values at each observation
        j_trend: GDP growth trend values at each observation (dy_i - k subtracted)
        k_values: Year mean values at each observation

    Returns:
        RMS of the imbalance across all observations
    """
    # Climate response applied to temperature trend
    h_T_trend = h1 * T_trend + h2 * T_trend ** 2

    # Imbalance
    imbalance = h_T_trend + j_trend + k_values

    # RMS
    rms = np.sqrt(np.mean(imbalance ** 2))

    return rms


def fit_approach1_precomputed_k(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 1: Pre-computed k[t] with linear temp + quadratic GDP detrending.

    Like Approach 1 but with precomputed k:
    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = y0 + y1*t + y2*t² are fit to (dy_i[t] - k[t])
    3. Temperature is detrended with linear trend: T* = T - (T0 + T1*t)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Note: Linear temperature detrending + quadratic GDP detrending.
    """
    # Compute detrended temperature terms (linear)
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 5c: T_trend = T0 + T1*t (linear), j_trend = y0 + y1*t + y2*t²
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]
    rms_imb = compute_rms_imbalance(h1, h2, T_trend, j_trend, k_values)

    # Compute RMS of h(T) - climate response to actual temperature
    rms_h = compute_rms_h(h1, h2, data.temp)
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    T_star_vals = data.temp - T_trend
    h_Tstar = h1 * T_star_vals + h2 * T_star_vals ** 2
    h_Ttrend = h1 * T_trend + h2 * T_trend ** 2
    h_cross = 2 * h2 * T_star_vals * T_trend
    components = {
        'h_Tstar': h_Tstar, 'h_Ttrend': h_Ttrend, 'h_cross': h_cross,
        'j': j_trend, 'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Δu = h(T) - h(T_trend), v = h(T_trend), j = j_trend, k = k_values, ε = remainder
    h_T = h1 * data.temp + h2 * data.temp ** 2
    h_T_trend = h1 * T_trend + h2 * T_trend ** 2
    Delta_u = h_T - h_T_trend
    v = h_T_trend
    # Compute ε as remainder: ε = Δy - (Δu + v + j + k) for exact decomposition
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="1: Precomputed k Combined",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def compute_1d_se_numerical(
    sse_func: callable,
    param_opt: float,
    param_bounds: tuple,
    n_obs: int,
    n_params: int = 3,
) -> float:
    """Compute standard error of a 1D parameter from curvature of SSE profile.

    Uses numerical second derivative of the SSE function at the optimum.
    SE is approximated from the profile likelihood curvature.

    Args:
        sse_func: Function that computes SSE for a given parameter value
        param_opt: Optimal parameter value
        param_bounds: Bounds for parameter (to ensure step stays in bounds)
        n_obs: Number of observations
        n_params: Total number of parameters in the model

    Returns:
        Standard error of the parameter (or np.nan if curvature is non-positive)
    """
    # Step size for finite differences
    h = 0.01
    # Ensure we stay within bounds
    h = min(h, (param_bounds[1] - param_opt) / 2, (param_opt - param_bounds[0]) / 2)

    sse_minus = sse_func(param_opt - h)
    sse_center = sse_func(param_opt)
    sse_plus = sse_func(param_opt + h)

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


def compute_2d_se_numerical(
    sse_func: callable,
    params_opt: np.ndarray,
    bounds: list,
    n_obs: int,
    n_params: int = 3,
) -> tuple:
    """Compute SE for 2D parameters from Hessian curvature.

    Uses numerical second derivatives to estimate the Hessian matrix at the optimum.
    SE is derived from the diagonal of the inverse Hessian scaled by sigma^2.

    Args:
        sse_func: Function that computes SSE for given (T_opt, beta)
        params_opt: Optimal [T_opt, beta] values
        bounds: List of (min, max) tuples for each parameter
        n_obs: Number of observations
        n_params: Number of parameters (default 3: h2, T_opt, beta)

    Returns:
        Tuple of (T_opt_se, beta_se) standard errors
    """
    T_opt, beta = params_opt
    T_opt_bounds, beta_bounds = bounds

    # Step sizes for finite differences
    h_T = 0.1  # Step for T_opt
    h_b = 0.01  # Step for beta

    # Ensure we stay within bounds
    h_T = min(h_T, (T_opt_bounds[1] - T_opt) / 2, (T_opt - T_opt_bounds[0]) / 2, 0.5)
    h_b = min(h_b, (beta_bounds[1] - beta) / 2, (beta - beta_bounds[0]) / 2, 0.1)

    # Ensure minimum step sizes
    h_T = max(h_T, 1e-6)
    h_b = max(h_b, 1e-6)

    # Compute SSE at various points
    sse_center = sse_func([T_opt, beta])

    # d2SSE/dT_opt^2
    sse_T_plus = sse_func([T_opt + h_T, beta])
    sse_T_minus = sse_func([T_opt - h_T, beta])
    d2_T = (sse_T_plus - 2 * sse_center + sse_T_minus) / (h_T ** 2)

    # d2SSE/dbeta^2
    sse_b_plus = sse_func([T_opt, beta + h_b])
    sse_b_minus = sse_func([T_opt, beta - h_b])
    d2_b = (sse_b_plus - 2 * sse_center + sse_b_minus) / (h_b ** 2)

    # d2SSE/dT_opt*dbeta (mixed partial)
    sse_pp = sse_func([T_opt + h_T, beta + h_b])
    sse_pm = sse_func([T_opt + h_T, beta - h_b])
    sse_mp = sse_func([T_opt - h_T, beta + h_b])
    sse_mm = sse_func([T_opt - h_T, beta - h_b])
    d2_Tb = (sse_pp - sse_pm - sse_mp + sse_mm) / (4 * h_T * h_b)

    # Build Hessian matrix
    H = np.array([[d2_T, d2_Tb], [d2_Tb, d2_b]])

    # Compute sigma^2
    sigma_sq = sse_center / (n_obs - n_params)

    try:
        # Invert Hessian
        H_inv = np.linalg.inv(H)
        # Covariance matrix is 2 * sigma^2 * H^(-1) for profile likelihood
        cov = 2 * sigma_sq * H_inv
        # Standard errors from diagonal
        T_opt_se = np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.nan
        beta_se = np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan
    except np.linalg.LinAlgError:
        T_opt_se = np.nan
        beta_se = np.nan

    return T_opt_se, beta_se


def compute_3d_se_numerical(
    sse_func: callable,
    params_opt: np.ndarray,
    bounds: list,
    n_obs: int,
    n_params: int = 4,
) -> tuple:
    """Compute SE for 3D parameters from Hessian curvature.

    Uses numerical second derivatives to estimate the Hessian matrix at the optimum.
    SE is derived from the diagonal of the inverse Hessian scaled by sigma^2.

    Args:
        sse_func: Function that computes SSE for given (T_opt, sigma, alpha)
        params_opt: Optimal [T_opt, sigma, alpha] values
        bounds: List of (min, max) tuples for each parameter
        n_obs: Number of observations
        n_params: Number of parameters (default 4: h2, T_opt, sigma, alpha)

    Returns:
        Tuple of (T_opt_se, sigma_se, alpha_se) standard errors
    """
    T_opt, sigma, alpha = params_opt
    T_opt_bounds, sigma_bounds, alpha_bounds = bounds

    # Step sizes for finite differences
    h_T = 0.1   # Step for T_opt
    h_s = 0.1   # Step for sigma
    h_a = 0.1   # Step for alpha

    # Ensure we stay within bounds
    h_T = min(h_T, (T_opt_bounds[1] - T_opt) / 2, (T_opt - T_opt_bounds[0]) / 2, 0.5)
    h_s = min(h_s, (sigma_bounds[1] - sigma) / 2, (sigma - sigma_bounds[0]) / 2, 0.5)
    h_a = min(h_a, (alpha_bounds[1] - alpha) / 2, (alpha - alpha_bounds[0]) / 2, 0.5)

    # Ensure minimum step sizes
    h_T = max(h_T, 1e-6)
    h_s = max(h_s, 1e-6)
    h_a = max(h_a, 1e-6)

    # Compute SSE at center
    sse_center = sse_func([T_opt, sigma, alpha])

    # Second derivatives (diagonal of Hessian)
    # d2SSE/dT_opt^2
    sse_T_plus = sse_func([T_opt + h_T, sigma, alpha])
    sse_T_minus = sse_func([T_opt - h_T, sigma, alpha])
    d2_T = (sse_T_plus - 2 * sse_center + sse_T_minus) / (h_T ** 2)

    # d2SSE/dsigma^2
    sse_s_plus = sse_func([T_opt, sigma + h_s, alpha])
    sse_s_minus = sse_func([T_opt, sigma - h_s, alpha])
    d2_s = (sse_s_plus - 2 * sse_center + sse_s_minus) / (h_s ** 2)

    # d2SSE/dalpha^2
    sse_a_plus = sse_func([T_opt, sigma, alpha + h_a])
    sse_a_minus = sse_func([T_opt, sigma, alpha - h_a])
    d2_a = (sse_a_plus - 2 * sse_center + sse_a_minus) / (h_a ** 2)

    # Mixed partial derivatives
    # d2SSE/dT_opt*dsigma
    sse_Ts_pp = sse_func([T_opt + h_T, sigma + h_s, alpha])
    sse_Ts_pm = sse_func([T_opt + h_T, sigma - h_s, alpha])
    sse_Ts_mp = sse_func([T_opt - h_T, sigma + h_s, alpha])
    sse_Ts_mm = sse_func([T_opt - h_T, sigma - h_s, alpha])
    d2_Ts = (sse_Ts_pp - sse_Ts_pm - sse_Ts_mp + sse_Ts_mm) / (4 * h_T * h_s)

    # d2SSE/dT_opt*dalpha
    sse_Ta_pp = sse_func([T_opt + h_T, sigma, alpha + h_a])
    sse_Ta_pm = sse_func([T_opt + h_T, sigma, alpha - h_a])
    sse_Ta_mp = sse_func([T_opt - h_T, sigma, alpha + h_a])
    sse_Ta_mm = sse_func([T_opt - h_T, sigma, alpha - h_a])
    d2_Ta = (sse_Ta_pp - sse_Ta_pm - sse_Ta_mp + sse_Ta_mm) / (4 * h_T * h_a)

    # d2SSE/dsigma*dalpha
    sse_sa_pp = sse_func([T_opt, sigma + h_s, alpha + h_a])
    sse_sa_pm = sse_func([T_opt, sigma + h_s, alpha - h_a])
    sse_sa_mp = sse_func([T_opt, sigma - h_s, alpha + h_a])
    sse_sa_mm = sse_func([T_opt, sigma - h_s, alpha - h_a])
    d2_sa = (sse_sa_pp - sse_sa_pm - sse_sa_mp + sse_sa_mm) / (4 * h_s * h_a)

    # Build Hessian matrix
    H = np.array([
        [d2_T, d2_Ts, d2_Ta],
        [d2_Ts, d2_s, d2_sa],
        [d2_Ta, d2_sa, d2_a]
    ])

    # Compute sigma^2
    sigma_sq = sse_center / (n_obs - n_params)

    try:
        # Invert Hessian
        H_inv = np.linalg.inv(H)
        # Covariance matrix is 2 * sigma^2 * H^(-1) for profile likelihood
        cov = 2 * sigma_sq * H_inv
        # Standard errors from diagonal
        T_opt_se = np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.nan
        sigma_se = np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan
        alpha_se = np.sqrt(cov[2, 2]) if cov[2, 2] > 0 else np.nan
    except np.linalg.LinAlgError:
        T_opt_se = np.nan
        sigma_se = np.nan
        alpha_se = np.nan

    return T_opt_se, sigma_se, alpha_se


def fit_approach2_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResult:
    """Approach 2: Pre-computed k[t] with LOESS country/temperature trends.

    LOESS version of Approach 5:
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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 6: T_trend = T_loess, j_trend = y_loess (LOESS smoothed)
    T_trend = trends_loess.T_loess
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])
    rms_imb = compute_rms_imbalance(h1, h2, T_trend, j_trend, k_values)

    # Compute RMS of h(T) - climate response to actual temperature
    rms_h = compute_rms_h(h1, h2, data.temp)
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    T_star_vals = data.temp - T_trend
    h_Tstar = h1 * T_star_vals + h2 * T_star_vals ** 2
    h_Ttrend = h1 * T_trend + h2 * T_trend ** 2
    h_cross = 2 * h2 * T_star_vals * T_trend
    components = {
        'h_Tstar': h_Tstar, 'h_Ttrend': h_Ttrend, 'h_cross': h_cross,
        'j': j_trend, 'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Δu = h(T) - h(T_trend), v = h(T_trend), j = j_trend, k = k_values, ε = remainder
    h_T = h1 * data.temp + h2 * data.temp ** 2
    h_T_trend = h1 * T_trend + h2 * T_trend ** 2
    Delta_u = h_T - h_T_trend
    v = h_T_trend
    # Compute ε as remainder: ε = Δy - (Δu + v + j + k) for exact decomposition
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="2: Precomputed k LOESS",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_method4_quadratic_departure_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6ab:
    """Method 4: T response with quadratic departure term (LOESS detrending).

    Model: Δy* = h1*(T-Ttrend) + h2*(T²-Ttrend²) + h4*(T-Ttrend)²

    3-parameter OLS with design matrix: [(T-Ttrend), (T²-Ttrend²), (T-Ttrend)²]

    The h4*(T-Ttrend)² term captures quadratic effects of temperature departures
    from trend, allowing asymmetric response to warm vs cold anomalies.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResultApproach6ab with h3=0
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_dep = T - Ttrend  # Departure from trend
    T2_dep = T**2 - Ttrend**2  # T² - Ttrend²
    T_dep_sq = T_dep**2  # (T - Ttrend)²

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [(T-Ttrend), (T²-Ttrend²), (T-Ttrend)²]
    X = np.column_stack([T_dep, T2_dep, T_dep_sq])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1_linear = beta[0]   # Linear coef for (T - Ttrend)
    h2_T = beta[1]        # Quadratic coef for (T² - Ttrend²)
    h4_dep = beta[2]      # Quadratic coef for (T - Ttrend)²
    h1_linear_se = np.sqrt(cov[0, 0])
    h2_T_se = np.sqrt(cov[1, 1])
    h4_dep_se = np.sqrt(cov[2, 2])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1, h2, h4
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_T = compute_T_optimal(h1_linear, h2_T)
    # Optimal departure: -h1/(2*h4)
    T_optimal_dep = -h1_linear / (2 * h4_dep) if abs(h4_dep) > 1e-10 else 0.0

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_response = h1*(T-Ttrend) + h2*(T²-Ttrend²) + h4*(T-Ttrend)²
    h_response = h1_linear * T_dep + h2_T * T2_dep + h4_dep * T_dep_sq

    # RMS imbalance: at trend, response is 0 by construction
    imbalance = j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of total response magnitude
    rms_h = np.sqrt(np.mean(h_response ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_response': h_response,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution
    # Split h(T) into baseline at trend + departure
    # h(T_trend) = h1*Ttrend + h2*Ttrend² (since h4*(0)² = 0 at trend)
    h_Ttrend = h1_linear * Ttrend + h2_T * Ttrend**2
    Delta_u = h_response  # h(T) - h(Ttrend)
    v = h_Ttrend          # h(Ttrend)
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6ab(
        approach="4: T/Quadratic Departure LOESS",
        h1=h1_linear,
        h2=h2_T,
        h1_se=h1_linear_se,
        h2_se=h2_T_se,
        h3=0.0,  # Not in model
        h4=h4_dep,
        h3_se=0.0,
        h4_se=h4_dep_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_opt=T_optimal_T,
        T_dep_opt=T_optimal_dep,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def compute_1d_se_numerical(
    sse_func: callable,
    T_opt: float,
    T_opt_bounds: tuple,
    n_obs: int,
    n_params: int = 3,
) -> float:
    """Compute SE for T_opt from Hessian curvature.

    Uses numerical second derivative to estimate the curvature at the optimum.
    SE is derived from the inverse curvature scaled by sigma^2.

    Args:
        sse_func: Function that computes SSE for given T_opt
        T_opt: Optimal T_opt value
        T_opt_bounds: (min, max) tuple for T_opt
        n_obs: Number of observations
        n_params: Number of parameters (default 3: h2_low, h2_high, T_opt)

    Returns:
        Standard error for T_opt
    """
    # Step size for finite differences
    h_T = 0.1

    # Ensure we stay within bounds
    h_T = min(h_T, (T_opt_bounds[1] - T_opt) / 2, (T_opt - T_opt_bounds[0]) / 2, 0.5)
    h_T = max(h_T, 1e-6)

    # Compute SSE at center and neighbors
    sse_center = sse_func(T_opt)
    sse_plus = sse_func(T_opt + h_T)
    sse_minus = sse_func(T_opt - h_T)

    # Second derivative
    d2_T = (sse_plus - 2 * sse_center + sse_minus) / (h_T ** 2)

    # Compute sigma^2
    sigma_sq = sse_center / (n_obs - n_params)

    try:
        if d2_T > 0:
            # Variance is 2 * sigma^2 / d2SSE (profile likelihood)
            var_T_opt = 2 * sigma_sq / d2_T
            T_opt_se = np.sqrt(var_T_opt) if var_T_opt > 0 else np.nan
        else:
            T_opt_se = np.nan
    except Exception:
        T_opt_se = np.nan

    return T_opt_se


def compute_2d_se_numerical(
    sse_func: callable,
    f_opt: np.ndarray,
    f_bounds: tuple,
    n_obs: int,
    n_params: int = 4,
) -> tuple:
    """Compute SEs for f1 and f2 from 2D numerical Hessian.

    Uses numerical second derivatives to estimate the curvature at the optimum.
    SEs are derived from the inverse Hessian scaled by sigma^2.

    Args:
        sse_func: Function that computes SSE for given [f1, f2]
        f_opt: Optimal [f1, f2] values as numpy array
        f_bounds: Bounds for (f1, f2) as tuple of tuples
        n_obs: Number of observations
        n_params: Number of parameters (default 4: f1, f2, h1, h2)

    Returns:
        Tuple of (f1_se, f2_se)
    """
    # Step sizes for finite differences
    h_step = 0.01

    f1_opt, f2_opt = f_opt

    # Ensure we stay within bounds
    h1 = min(h_step, (f_bounds[0][1] - f1_opt) / 2, (f1_opt - f_bounds[0][0]) / 2, 0.5)
    h1 = max(h1, 1e-6)
    h2 = min(h_step, (f_bounds[1][1] - f2_opt) / 2, (f2_opt - f_bounds[1][0]) / 2, 0.5)
    h2 = max(h2, 1e-6)

    # Compute SSE at center and neighbors for Hessian
    sse_center = sse_func(f_opt)

    # d²SSE/df1²
    sse_f1_plus = sse_func(np.array([f1_opt + h1, f2_opt]))
    sse_f1_minus = sse_func(np.array([f1_opt - h1, f2_opt]))
    d2_f1 = (sse_f1_plus - 2 * sse_center + sse_f1_minus) / (h1 ** 2)

    # d²SSE/df2²
    sse_f2_plus = sse_func(np.array([f1_opt, f2_opt + h2]))
    sse_f2_minus = sse_func(np.array([f1_opt, f2_opt - h2]))
    d2_f2 = (sse_f2_plus - 2 * sse_center + sse_f2_minus) / (h2 ** 2)

    # d²SSE/df1df2 (mixed partial)
    sse_pp = sse_func(np.array([f1_opt + h1, f2_opt + h2]))
    sse_pm = sse_func(np.array([f1_opt + h1, f2_opt - h2]))
    sse_mp = sse_func(np.array([f1_opt - h1, f2_opt + h2]))
    sse_mm = sse_func(np.array([f1_opt - h1, f2_opt - h2]))
    d2_f1f2 = (sse_pp - sse_pm - sse_mp + sse_mm) / (4 * h1 * h2)

    # Compute sigma^2
    sigma_sq = sse_center / (n_obs - n_params)

    # Build Hessian matrix
    H = np.array([[d2_f1, d2_f1f2],
                  [d2_f1f2, d2_f2]])

    try:
        # Check if Hessian is positive definite
        if np.linalg.det(H) > 0 and d2_f1 > 0:
            # Inverse Hessian gives covariance (scaled by 2*sigma^2 for profile likelihood)
            H_inv = np.linalg.inv(H)
            var_f1 = 2 * sigma_sq * H_inv[0, 0]
            var_f2 = 2 * sigma_sq * H_inv[1, 1]
            f1_se = np.sqrt(var_f1) if var_f1 > 0 else np.nan
            f2_se = np.sqrt(var_f2) if var_f2 > 0 else np.nan
        else:
            f1_se = np.nan
            f2_se = np.nan
    except Exception:
        f1_se = np.nan
        f2_se = np.nan

    return f1_se, f2_se


def fit_approach3_piecewise(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
) -> FitResultApproach8:
    """Approach 3: Piecewise quadratic temperature response with LOESS detrending.

    Model: h(T) = h2_high * (T - T_opt)²  if T > T_opt
           h(T) = h2_low * (T - T_opt)²   if T ≤ T_opt

    This model:
    - Has an optimum at T_opt where h(T_opt) = 0
    - Allows different curvatures above vs below T_opt
    - Uses 1D optimization over T_opt with inner 2-column OLS for h2_low, h2_high

    For fitting we use the h(T) - h(T_trend) formulation.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])

    Returns:
        FitResultApproach8Piecewise with T_opt, h2_low, h2_high, and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t] (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Use raw temperature (NOT detrended) so T_opt represents actual optimal temperature
    T = data.temp
    T_trend = trends_loess.T_loess  # Temperature trend for h(T) - h(T_trend) formulation

    def piecewise_quad(T_vals, T_opt_val):
        """Compute piecewise quadratic: different curvature above/below T_opt.

        Returns two columns: one for low (T <= T_opt) and one for high (T > T_opt).
        """
        low_col = np.where(T_vals <= T_opt_val, (T_vals - T_opt_val) ** 2, 0.0)
        high_col = np.where(T_vals > T_opt_val, (T_vals - T_opt_val) ** 2, 0.0)
        return low_col, high_col

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving inner 2-column OLS for h2_low, h2_high."""
        # Compute columns for T
        low_T, high_T = piecewise_quad(T, T_opt_val)
        # Compute columns for T_trend
        low_trend, high_trend = piecewise_quad(T_trend, T_opt_val)

        # Design matrix: [h2_low, h2_high] columns using h(T) - h(T_trend) formulation
        X1 = low_T - low_trend   # Column for h2_low
        X2 = high_T - high_trend  # Column for h2_high
        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h2_low, h2_high]||^2
        try:
            beta_ols, _, _, _ = linalg.lstsq(X, y)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    # Initial guess: T_opt = 15°C
    x0 = 15.0

    # 1D optimization using L-BFGS-B
    result = minimize(
        lambda x: compute_sse_for_T_opt(x[0]),
        x0=[x0],
        bounds=[T_opt_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T_opt_opt = result.x[0]

    # Re-fit at optimal T_opt to get h2_low, h2_high, residuals, covariance
    low_T, high_T = piecewise_quad(T, T_opt_opt)
    low_trend, high_trend = piecewise_quad(T_trend, T_opt_opt)
    X1 = low_T - low_trend
    X2 = high_T - high_trend
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_low = beta_ols[0]
    h2_high = beta_ols[1]
    h2_low_se = np.sqrt(cov[0, 0])
    h2_high_se = np.sqrt(cov[1, 1])

    # Compute SE for T_opt using numerical Hessian
    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        data.n_obs,
        n_params=3
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h2_low, h2_high, T_opt
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values using h(T) - h(T_trend) formulation
    h_values = h2_low * X1 + h2_high * X2

    # For imbalance, we use h(T_trend) term
    h_of_T_trend = h2_low * low_trend + h2_high * high_trend
    imbalance = h_of_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # Compute RMS of h(T) - h(T_trend) - climate response to temperature fluctuations
    rms_h = np.sqrt(np.mean(h_values ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    Delta_u = h_values  # h(T) - h(T_trend)
    v = h_of_T_trend    # h(T_trend)
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="3: Piecewise Quadratic LOESS",
        h2=h2_low,
        h2_se=h2_low_se,
        h4=h2_high,
        h4_se=h2_high_se,
        T_opt=T_opt_opt,
        T_opt_se=T_opt_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_method6_piecewise_departure_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
) -> FitResultApproach8:
    """Method 6: Pure interaction model (LOESS detrending).

    Model: h(T) = h3 * (T - T_opt) * (T - T_trend)

    The interaction term (T - T_opt) * (T - T_trend) captures:
    - h3 < 0: hot + warming is bad, cold + cooling is bad
    - h3 > 0: hot + warming is good, cold + cooling is good

    Using h(T) - h(T_trend) formulation:
    - X = (T - T_opt) * (T - T_trend)  (interaction term, zero at T = T_trend)

    Model: y* = h3 * X (single parameter OLS after fixing T_opt)

    DOF: 1 parameter (h3) + 1 searched (T_opt) = 2 total

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])

    Returns:
        FitResultApproach8 with T_opt, h2=0, h4=h3 (interaction coef)
    """
    # Compute dependent variable: dy - k[t] - j_i[t] (same as method2)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Use raw temperature and trend
    T = data.temp
    T_trend = trends_loess.T_loess

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving single-parameter OLS for h3."""
        # X: interaction term (T - T_opt) * (T - T_trend)
        X = (T - T_opt_val) * (T - T_trend)

        # Solve OLS: min ||y - X * h3||^2
        # h3 = (X'X)^-1 X'y = sum(X*y) / sum(X^2)
        XtX = np.sum(X * X)
        if XtX < 1e-10:
            return np.inf
        h3 = np.sum(X * y) / XtX
        y_pred = X * h3
        sse = np.sum((y - y_pred) ** 2)
        return sse

    # Initial guess: T_opt = 15C
    x0 = 15.0

    # 1D optimization using L-BFGS-B
    result = minimize(
        lambda x: compute_sse_for_T_opt(x[0]),
        x0=[x0],
        bounds=[T_opt_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T_opt_opt = result.x[0]

    # Re-fit at optimal T_opt to get h3, residuals, covariance
    X = (T - T_opt_opt) * (T - T_trend)
    X_col = X.reshape(-1, 1)

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_col)
    h3_int = beta_ols[0]    # Interaction coefficient
    h3_int_se = np.sqrt(cov[0, 0])

    # Compute SE for T_opt using numerical Hessian
    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        data.n_obs,
        n_params=2
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # h3, T_opt
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R^2 (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values: h(T) - h(T_trend) = h3 * (T - T_opt) * (T - T_trend)
    # Note: h(T_trend) = h3 * (T_trend - T_opt) * 0 = 0
    h_values = h3_int * X

    # h(T_trend) = 0 since (T_trend - T_trend) = 0
    h_of_T_trend = np.zeros(data.n_obs)

    imbalance = h_of_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # Compute RMS of h(T) - h(T_trend) - climate response to temperature fluctuations
    rms_h = np.sqrt(np.mean(h_values ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    Delta_u = h_values      # h(T) - h(T_trend)
    v = h_of_T_trend        # h(T_trend) = 0
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    # Return using FitResultApproach8
    # Note: h2 = 0 (no quadratic term), h4 = h3 interaction coef
    return FitResultApproach8(
        approach="6: Interaction Model LOESS",
        h2=0.0,
        h2_se=0.0,
        h4=h3_int,          # interaction coefficient stored in h4 slot
        h4_se=h3_int_se,
        T_opt=T_opt_opt,
        T_opt_se=T_opt_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_method7_lagged_deviation_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
) -> FitResultApproach8:
    """Method 7: Quadratic with lagged deviation change (LOESS detrending).

    Model: h(T(t)) = h2 * (T(t) - T_opt)^2 + h4 * ((T(t) - T_opt)^2 - (T(t-1) - T_opt)^2)

    The h4 term captures the effect of moving toward or away from optimal:
    - h4 < 0: moving away from optimal (increasing squared deviation) is bad
    - h4 > 0: moving away from optimal is good

    Using h(T) - h(T_trend) formulation:
    - X1 = (T(t) - T_opt)^2 - (T_trend(t) - T_opt)^2
    - X2 = [(T(t) - T_opt)^2 - (T(t-1) - T_opt)^2] - [(T_trend(t) - T_opt)^2 - (T_trend(t-1) - T_opt)^2]

    Model: y* = h2 * X1 + h4 * X2 (linear in h2, h4 after fixing T_opt)

    DOF: 2 parameters (h2, h4) + 1 searched (T_opt) = 3 total (same as method3)

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])

    Returns:
        FitResultApproach8 with T_opt, h2, h4
    """
    # Compute dependent variable: dy - k[t] - j_i[t] (same as method2)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Use raw temperature and trend
    T = data.temp
    T_trend = trends_loess.T_loess

    # Compute lagged temperatures T(t-1) and T_trend(t-1) for each observation
    # Need to handle this per-country since lags don't cross country boundaries
    T_lag = np.zeros(data.n_obs)
    T_trend_lag = np.zeros(data.n_obs)
    valid_mask = np.ones(data.n_obs, dtype=bool)  # Track which obs have valid lags

    for country_idx in range(data.n_countries):
        mask = data.country_idx == country_idx
        idx = np.where(mask)[0]
        years = data.year[idx]
        sort_order = np.argsort(years)
        idx_sorted = idx[sort_order]
        years_sorted = years[sort_order]

        for j, obs_idx in enumerate(idx_sorted):
            if j == 0:
                # First year for this country: no lag available
                valid_mask[obs_idx] = False
                T_lag[obs_idx] = T[obs_idx]  # placeholder
                T_trend_lag[obs_idx] = T_trend[obs_idx]
            else:
                # Check if previous year is consecutive
                prev_idx = idx_sorted[j - 1]
                if years_sorted[j] == years_sorted[j - 1] + 1:
                    T_lag[obs_idx] = T[prev_idx]
                    T_trend_lag[obs_idx] = T_trend[prev_idx]
                else:
                    # Gap in years, no valid lag
                    valid_mask[obs_idx] = False
                    T_lag[obs_idx] = T[obs_idx]
                    T_trend_lag[obs_idx] = T_trend[obs_idx]

    # Filter to observations with valid lags
    y_valid = y[valid_mask]
    T_valid = T[valid_mask]
    T_trend_valid = T_trend[valid_mask]
    T_lag_valid = T_lag[valid_mask]
    T_trend_lag_valid = T_trend_lag[valid_mask]
    n_valid = np.sum(valid_mask)

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving 2-column OLS for h2, h4."""
        # X1: current squared deviation (detrended)
        X1 = (T_valid - T_opt_val)**2 - (T_trend_valid - T_opt_val)**2

        # X2: change in squared deviation (detrended)
        # [(T(t) - T_opt)^2 - (T(t-1) - T_opt)^2] - [(T_trend(t) - T_opt)^2 - (T_trend(t-1) - T_opt)^2]
        delta_sq = (T_valid - T_opt_val)**2 - (T_lag_valid - T_opt_val)**2
        delta_sq_trend = (T_trend_valid - T_opt_val)**2 - (T_trend_lag_valid - T_opt_val)**2
        X2 = delta_sq - delta_sq_trend

        X = np.column_stack([X1, X2])

        # Solve OLS
        try:
            beta_ols, _, _, _ = linalg.lstsq(X, y_valid)
            y_pred = X @ beta_ols
            sse = np.sum((y_valid - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    # Initial guess: T_opt = 15C
    x0 = 15.0

    # 1D optimization using L-BFGS-B
    result = minimize(
        lambda x: compute_sse_for_T_opt(x[0]),
        x0=[x0],
        bounds=[T_opt_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T_opt_opt = result.x[0]

    # Re-fit at optimal T_opt to get h2, h4, residuals, covariance
    X1 = (T_valid - T_opt_opt)**2 - (T_trend_valid - T_opt_opt)**2
    delta_sq = (T_valid - T_opt_opt)**2 - (T_lag_valid - T_opt_opt)**2
    delta_sq_trend = (T_trend_valid - T_opt_opt)**2 - (T_trend_lag_valid - T_opt_opt)**2
    X2 = delta_sq - delta_sq_trend
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y_valid, X_opt)
    h2_quad = beta_ols[0]
    h4_delta = beta_ols[1]
    h2_quad_se = np.sqrt(cov[0, 0])
    h4_delta_se = np.sqrt(cov[1, 1])

    # Compute SE for T_opt using numerical Hessian
    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        n_valid,
        n_params=3
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics (using valid observations only)
    n_params = 3  # h2, h4, T_opt
    r_sq, adj_r_sq, rmse = compute_fit_stats(y_valid, residuals, n_params)

    # Total R^2 (variance explained in original dy for valid obs)
    dy_valid = data.growth_pcGDP[valid_mask]
    total_r_sq = compute_total_r_squared(residuals, dy_valid)

    # Compute RMS values for valid observations
    j_trend_valid = trends_loess.y_loess[valid_mask]
    k_values_valid = np.array([year_means[data.year[i]] for i in np.where(valid_mask)[0]])

    # Climate response: h(T) - h(T_trend)
    h_values = h2_quad * X1 + h4_delta * X2

    # h(T_trend): need to compute using trend values
    # h(T_trend(t)) = h2 * (T_trend(t) - T_opt)^2 + h4 * ((T_trend(t) - T_opt)^2 - (T_trend(t-1) - T_opt)^2)
    h_of_T_trend = h2_quad * (T_trend_valid - T_opt_opt)**2 + h4_delta * delta_sq_trend

    imbalance = h_of_T_trend + j_trend_valid + k_values_valid
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    rms_h = np.sqrt(np.mean(h_values ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_values,
        'j': j_trend_valid,
        'k': k_values_valid,
    }
    var_decomp = compute_variance_decomposition(components, dy_valid, total_r_sq)

    # Compute variance attribution
    Delta_u = h_values
    v = h_of_T_trend
    epsilon = dy_valid - (Delta_u + v + j_trend_valid + k_values_valid)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_valid, k_values_valid, epsilon, dy_valid)

    return FitResultApproach8(
        approach="7: Lagged Deviation LOESS",
        h2=h2_quad,
        h2_se=h2_quad_se,
        h4=h4_delta,
        h4_se=h4_delta_se,
        T_opt=T_opt_opt,
        T_opt_se=T_opt_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=n_valid,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach4_persistence_decay(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    h4_bounds: tuple = (0.0, 1.0),
) -> FitResultApproach4:
    """Approach 4: Persistence decay model with LOESS detrending.

    Model: h_conv(T(t)) = h(T(t)) - h4 * sum_{k=1}^{n} (1-h4)^{k-1} * h(T(t-k))

    where h(T) = h1*T + h2*T^2.

    Using accumulators for efficient computation:
    - A_T(t) = T(t) + (1-h4) * A_T(t-1)
    - A_T2(t) = T^2(t) + (1-h4) * A_T2(t-1)

    Modified regressors (with detrending):
    - X1(t) = (T - h4*A_T_lag) - (T_trend - h4*A_T_trend_lag)
    - X2(t) = (T^2 - h4*A_T2_lag) - (T_trend^2 - h4*A_T2_trend_lag)

    Uses 1D optimization over h4 in [0, 1] with inner 2-column OLS for h1, h2.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        h4_bounds: Bounds for persistence decay parameter (default [0, 1])

    Returns:
        FitResultApproach4 with h1, h2, h4, T_opt, and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t] (same as approach2)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    T = data.temp
    T_trend = trends_loess.T_loess

    def compute_sse_for_h4(h4_val):
        """Compute SSE for given h4 by solving inner 2-column OLS for h1, h2."""
        # Compute accumulators for observed temperature
        A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
        # Compute accumulators for trend temperature
        A_T_trend_lag, A_T2_trend_lag = compute_persistence_accumulators_at_T(
            data, h4_val, T_trend
        )
        # Compute pre-first-year correction (assumes T was constant before first year)
        correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T)
        correction_T_trend, correction_T2_trend = compute_pre_first_year_correction(
            data, h4_val, T_trend
        )

        # Modified regressors with detrending and pre-first-year correction
        # X1 = (T - h4*A_T_lag - correction_T) - (T_trend - h4*A_T_trend_lag - correction_T_trend)
        X1 = (T - h4_val * A_T_lag - correction_T) - (T_trend - h4_val * A_T_trend_lag - correction_T_trend)
        # X2 = (T^2 - h4*A_T2_lag - correction_T2) - (T_trend^2 - h4*A_T2_trend_lag - correction_T2_trend)
        X2 = (T**2 - h4_val * A_T2_lag - correction_T2) - (T_trend**2 - h4_val * A_T2_trend_lag - correction_T2_trend)

        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h1, h2]||^2
        beta_ols, _, _, _ = linalg.lstsq(X, y)
        y_pred = X @ beta_ols
        sse = np.sum((y - y_pred) ** 2)
        return sse

    # 1D optimization using Brent's method (more robust for 1D bounded problems)
    # First do a coarse grid search to find a good starting region
    h4_grid = np.linspace(h4_bounds[0], h4_bounds[1], 21)
    sse_grid = [compute_sse_for_h4(h4_val) for h4_val in h4_grid]
    best_grid_idx = np.argmin(sse_grid)

    # Refine with Brent's method in the region around the best grid point
    search_lo = h4_grid[max(0, best_grid_idx - 1)]
    search_hi = h4_grid[min(len(h4_grid) - 1, best_grid_idx + 1)]

    result = minimize_scalar(
        compute_sse_for_h4,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    h4_opt = result.x

    # Re-fit at optimal h4 to get h1, h2, residuals, covariance
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_opt)
    A_T_trend_lag, A_T2_trend_lag = compute_persistence_accumulators_at_T(
        data, h4_opt, T_trend
    )
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_opt, T)
    correction_T_trend, correction_T2_trend = compute_pre_first_year_correction(
        data, h4_opt, T_trend
    )

    X1 = (T - h4_opt * A_T_lag - correction_T) - (T_trend - h4_opt * A_T_trend_lag - correction_T_trend)
    X2 = (T**2 - h4_opt * A_T2_lag - correction_T2) - (T_trend**2 - h4_opt * A_T2_trend_lag - correction_T2_trend)
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute SE for h4 using numerical Hessian
    h4_se = compute_1d_se_numerical(
        compute_sse_for_h4,
        h4_opt,
        h4_bounds,
        data.n_obs,
        n_params=3
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1, h2, h4
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values using h_conv formulation
    h_conv_values = h1 * X1 + h2 * X2

    # For imbalance, we use h(T_trend) term (no persistence correction at trend)
    h_of_T_trend = h1 * T_trend + h2 * T_trend ** 2
    imbalance = h_of_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # Compute RMS of h_conv
    rms_h = np.sqrt(np.mean(h_conv_values ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # h(T) at full temperature (without persistence correction for baseline)
    h_T_full = h1 * T + h2 * T ** 2
    h_T_trend_full = h1 * T_trend + h2 * T_trend ** 2
    Delta_u = h_conv_values  # Effective climate response increment
    v = h_T_trend_full       # Baseline at trend temperature
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach4(
        approach="4: Persistence Decay LOESS",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        h4=h4_opt,
        h4_se=h4_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach0_conjoined(data: AnalysisData) -> FitResult:
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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 0: T_trend = T (raw), j_trend from fitted coefficients
    T_trend = data.temp  # No temperature detrending, use raw T
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            # j coefficients for country c are at beta[2 + 3*(c-1) : 2 + 3*(c-1) + 3]
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        # else j_trend[i] = 0 (country 0 is reference)
        k_values[i] = k[yr]
    rms_imb = compute_rms_imbalance(h1, h2, T_trend, j_trend, k_values)

    # Compute RMS of h(T) - climate response to actual temperature
    rms_h = compute_rms_h(h1, h2, data.temp)
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition (approach 0: no detrending, 3 components)
    h_T = h1 * data.temp + h2 * data.temp ** 2
    components = {
        'h_T': h_T, 'j': j_trend, 'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Approach 0: no temperature detrending, so T_trend = T (raw temp)
    # Δu = h(T) - h(T), v = h(T), j = j_trend, k = k_values, ε = remainder
    # This means Δu = 0, v = h(T)
    h_T_full = h1 * data.temp + h2 * data.temp ** 2
    Delta_u = np.zeros(n_obs)  # No detrending, so increment is zero
    v = h_T_full  # Baseline is full climate response
    # Compute ε as remainder: ε = Δy - (Δu + v + j + k) for exact decomposition
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="0: Conjoined OLS Fit",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach0h0_joint(data: AnalysisData) -> FitResult:
    """Null model: No climate response, joint OLS fit with country trends and year effects.

    Δy_i(t) = j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    Like Approach 0 but without T and T² columns. Tests how much variance
    is explained by country trends and year effects alone.

    For identifiability, we set j_{0,0} = j_{1,0} = j_{2,0} = 0 (first country is reference).
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years and create year index mapping
    unique_years = sorted(set(data.year))
    year_to_idx = {y: i for i, y in enumerate(unique_years)}
    n_years = len(unique_years)

    # Number of parameters (no h1, h2):
    # - 3 * (n_countries - 1) for j terms (first country is reference, j[0] = 0)
    # - n_years for k_t terms (all years)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_years
    n_params = n_j_params + n_k_params

    X = np.zeros((n_obs, n_params))

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 3 * (c - 1)
            X[i, col_base] = 1.0        # j0[c]
            X[i, col_base + 1] = t      # j1[c]
            X[i, col_base + 2] = t * t  # j2[c]

    # Year fixed effects (all years)
    k_col_start = n_j_params
    for i in range(n_obs):
        yr_idx = year_to_idx[data.year[i]]
        X[i, k_col_start + yr_idx] = 1.0

    # Fit OLS
    y = data.growth_pcGDP
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # No climate response coefficients
    h1 = 0.0
    h2 = 0.0
    h1_se = 0.0
    h2_se = 0.0

    # Year fixed effects (store by actual year)
    k = {}
    for yr_idx in range(n_years):
        k[unique_years[yr_idx]] = beta[k_col_start + yr_idx]

    # Fit statistics
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature undefined (h1=h2=0)
    T_opt = np.nan

    # Build j_trend and k_values arrays
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_values[i] = k[yr]

    # Variance decomposition (no h components)
    components = {'j': j_trend, 'k': k_values}
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Variance attribution (Δu=0, v=0 since no climate response)
    Delta_u = np.zeros(n_obs)
    v = np.zeros(n_obs)
    epsilon = data.growth_pcGDP - (j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="approach0h0: No Climate Response (Joint)",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=None,
        rms_h=None,
        imbalance_ratio=None,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach1h0_precomputed_k(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Null model: No climate response, precomputed k with quadratic country trends.

    k(t) = mean_i(Δy_i(t)) is precomputed, then country quadratics j_i(t)
    are fit to Δy_i(t) - k(t). No regression needed — all components are
    already precomputed from trends_with_k and year_means.

    Δy_i(t) = j_i(t) + k(t) + ε_i(t)
    """
    n_obs = data.n_obs
    n_countries = data.n_countries
    unique_years = sorted(set(data.year))
    n_years = len(unique_years)

    # Build j_trend and k_values from precomputed trends
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]

    # Residuals = dy - (j + k)
    residuals = data.growth_pcGDP - (j_trend + k_values)

    # No climate response coefficients
    h1 = 0.0
    h2 = 0.0
    h1_se = 0.0
    h2_se = 0.0

    # Year fixed effects
    k = dict(year_means)

    # Fit statistics
    n_params = 3 * n_countries + n_years
    r_sq, adj_r_sq, rmse = compute_fit_stats(data.growth_pcGDP, residuals, n_params)

    # Total R²
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature undefined
    T_opt = np.nan

    # Variance decomposition (no h components)
    components = {'j': j_trend, 'k': k_values}
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Variance attribution (Δu=0, v=0 since no climate response)
    Delta_u = np.zeros(n_obs)
    v = np.zeros(n_obs)
    epsilon = data.growth_pcGDP - (j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="approach1h0: No Climate Response (Precomputed k)",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=None,
        rms_h=None,
        imbalance_ratio=None,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach2h0_precomputed_k_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResult:
    """Null model: No climate response, precomputed k with LOESS country trends.

    LOESS version of approach1h0:
    k(t) = mean_i(Δy_i(t)) is precomputed, then country LOESS trends j_i(t)
    are smoothed from Δy_i(t) - k(t). No regression needed — all components are
    already precomputed from trends_loess and year_means.

    Δy_i(t) = j_i(t) + k(t) + ε_i(t)
    """
    n_obs = data.n_obs
    unique_years = sorted(set(data.year))
    n_years = len(unique_years)

    # Build j_trend and k_values from precomputed LOESS trends
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(n_obs)])

    # Residuals = dy - (j + k)
    residuals = data.growth_pcGDP - (j_trend + k_values)

    # No climate response coefficients
    h1 = 0.0
    h2 = 0.0
    h1_se = 0.0
    h2_se = 0.0

    # Year fixed effects
    k = dict(year_means)

    # Fit statistics - LOESS effective degrees of freedom is harder to define,
    # use n_years as proxy (similar to method1h0's 3*n_countries + n_years but
    # LOESS doesn't have fixed polynomial parameters)
    n_params = n_years  # Conservative: just count year effects
    r_sq, adj_r_sq, rmse = compute_fit_stats(data.growth_pcGDP, residuals, n_params)

    # Total R²
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature undefined
    T_opt = np.nan

    # Variance decomposition (no h components)
    components = {'j': j_trend, 'k': k_values}
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Variance attribution (Δu=0, v=0 since no climate response)
    Delta_u = np.zeros(n_obs)
    v = np.zeros(n_obs)
    epsilon = data.growth_pcGDP - (j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="approach2h0: No Climate Response (LOESS Precomputed k)",
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
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        rms_imbalance=None,
        rms_h=None,
        imbalance_ratio=None,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_all_approaches(
    data: AnalysisData, trends: CountryTrends,
    trends_with_k: CountryTrends = None, year_means: dict = None,
    trends_loess: CountryTrendsLoess = None
) -> dict:
    """Fit all approaches and return results.

    Returns dict with keys:
        Publication-ready approaches:
        'approach0': Conjoined OLS fit, with j terms and year fixed effects
        'approach1': Pre-computed k with linear temp + quadratic GDP (if trends_with_k and year_means provided)
        'approach2': Pre-computed k with LOESS trends (if trends_loess provided)
        'approach3': Piecewise quadratic response with LOESS (if trends_loess provided)
        'approach4': Persistence decay model with LOESS (if trends_loess provided)
        'approach0h0': No climate response, joint OLS (country trends + year effects only)
        'approach1h0': No climate response, precomputed k (if trends_with_k and year_means provided)
        'approach2h0': No climate response, LOESS precomputed k (if trends_loess provided)

        Exploratory methods (kept as methods):
        'method4': T response with quadratic departure (T-Ttrend)^2 (if trends_loess provided)
        'method6': Interaction model h3*(T-Topt)*(T-Ttrend) (if trends_loess provided)
        'method7': Lagged deviation model (if trends_loess provided)

    Args:
        data: AnalysisData object
        trends: CountryTrends (unused, kept for API compatibility)
        trends_with_k: CountryTrends for approach1 (fit to dy - k)
        year_means: Pre-computed k[t] for approaches 1-4 and exploratory methods
        trends_loess: CountryTrendsLoess for approaches 2-4 and exploratory methods (LOESS detrending)
    """
    results = {
        'approach0': fit_approach0_conjoined(data),
        'approach0h0': fit_approach0h0_joint(data),
    }

    # Add approach1 and approach1h0 if trends_with_k and year_means are provided
    if trends_with_k is not None and year_means is not None:
        results['approach1'] = fit_approach1_precomputed_k(
            data, trends_with_k, year_means
        )
        results['approach1h0'] = fit_approach1h0_precomputed_k(
            data, trends_with_k, year_means
        )

    # Add approaches 2, 3, 4 and approach2h0 if trends_loess and year_means are provided
    # Also add exploratory methods 4, 6, 7
    if trends_loess is not None and year_means is not None:
        results['approach2'] = fit_approach2_loess(
            data, trends_loess, year_means
        )
        results['approach2h0'] = fit_approach2h0_precomputed_k_loess(
            data, trends_loess, year_means
        )
        results['approach3'] = fit_approach3_piecewise(
            data, trends_loess, year_means
        )
        results['approach4'] = fit_approach4_persistence_decay(
            data, trends_loess, year_means
        )
        # Exploratory methods (kept as methods)
        results['method4'] = fit_method4_quadratic_departure_loess(
            data, trends_loess, year_means
        )
        results['method6'] = fit_method6_piecewise_departure_loess(
            data, trends_loess, year_means
        )
        results['method7'] = fit_method7_lagged_deviation_loess(
            data, trends_loess, year_means
        )

    return results
