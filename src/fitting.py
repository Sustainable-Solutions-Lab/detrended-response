"""OLS fitting for three detrending approaches.

Approach 1: Combined detrending
    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i

Approach 2: Linear temperature detrending
    Δy_i(t) = h1*[T - (T0 + T1*t)] + h2*[T² - (T0 + T1*t)²] + k_i

Approach 3: Quadratic GDP growth detrending
    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*T + h2*T² + k_i
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
    compute_detrended_temperature_quadratic,
    compute_detrended_temp_squared_quadratic,
    compute_detrended_temperature_loess,
    compute_detrended_temp_squared_loess,
    compute_growth_trend_values,
)

# ==============================================================================
# Constants
# ==============================================================================

# Bounds for beta optimization in GDP-dependent response approaches
# Wider bounds allow for stronger GDP scaling effects
DEFAULT_BETA_BOUNDS = (0.001, 10.0)


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


def compute_gdp_scaling(pcGDP: np.ndarray, Y_ref: float, beta: float) -> np.ndarray:
    """Compute GDP scaling factors for GDP-dependent response approaches.

    Args:
        pcGDP: Per capita GDP values for each observation
        Y_ref: Reference GDP level
        beta: GDP scaling exponent

    Returns:
        Array of scaling factors g = (Y/Y_ref)^(-beta)
    """
    return (pcGDP / Y_ref) ** (-beta)


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
        # Orthogonality checks (residual should be orthogonal to fitted components for OLS)
        'cov_epsilon_Delta_u': Sigma[4, 0],
        'cov_epsilon_v': Sigma[4, 1],
        'cov_epsilon_j': Sigma[4, 2],
        'cov_epsilon_k': Sigma[4, 3],
    }


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
class FitResultApproach5d:
    """Container for GDP-dependent response results.

    Model: h(Y,T) = (Y/Y_ref)^(-f1) * (h1*T + h2*T^2)
    Used by approach 5d (GDP-only response).
    """
    approach: str           # Name of the approach
    h1: float              # Linear temperature coefficient
    h2: float              # Quadratic temperature coefficient
    h1_se: float           # Standard error of h1
    h2_se: float           # Standard error of h2
    f1: float              # GDP scaling exponent (formerly beta)
    f1_se: float           # Standard error of f1
    Y_ref: float           # Reference GDP used
    k: Dict[int, float]    # Year fixed effects (year -> value)
    r_squared: float       # R-squared
    adj_r_squared: float   # Adjusted R-squared
    rmse: float            # Root mean squared error
    n_obs: int             # Number of observations
    n_params: int          # Number of parameters (3: h1, h2, f1)
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


@dataclass
class FitResultApproach6c:
    """Container for Approach 6c results with departure/trend decomposition.

    Model: h(T, Ttrend) = h1*(T-Ttrend) + h2*(T-Ttrend)² + h3*Ttrend + h4*Ttrend²

    Where:
        - h1, h2: response to departure from trend (T - Ttrend)
        - h3, h4: response to trend temperature (Ttrend)

    When T = Ttrend: h = h3*Ttrend + h4*Ttrend² (pure trend response)

    Parameters:
    - h1, h2: Response to departure (T - T_trend)
    - h3, h4: Response to trend temperature T_trend
    - T_dep_opt: Optimal departure = -h1/(2*h2)
    - f2: Optimal trend temperature = -h3/(2*h4)
    """
    approach: str
    # Departure (T - Ttrend) coefficients
    h1: float              # Linear departure coef (formerly h1_dep)
    h2: float              # Quadratic departure coef (formerly h2_dep)
    h1_se: float
    h2_se: float
    # Trend (Ttrend) coefficients
    h3: float              # Linear trend coef (formerly h1_trend)
    h4: float              # Quadratic trend coef (formerly h2_trend)
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
    T_dep_opt: float       # -h1/(2*h2) - optimal departure
    f2: float              # -h3/(2*h4) - optimal trend T
    total_r_squared: float
    # Diagnostics
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None


@dataclass
class FitResultApproach8a:
    """Container for Approach 8a results with separate total/trend and shared T_opt.

    Model: delta-y*(t) = h2 * (T - T_opt)^2 - h4 * (Ttrend - T_opt)^2

    Where:
        - h2: curvature for total (actual T) response
        - h4: curvature for trend (Ttrend) response
        - T_opt: shared optimal temperature

    Total Response (when T = Ttrend): (h2 - h4) * (T - T_opt)^2

    Parameters:
    - h2: Curvature for actual T response (formerly h2_total)
    - h4: Curvature for trend T response (formerly h2_trend)
    - T_opt: Shared optimal temperature
    """
    approach: str
    # Curvature coefficients
    h2: float              # Curvature for actual T (formerly h2_total)
    h2_se: float
    h4: float              # Curvature for trend T (formerly h2_trend)
    h4_se: float
    # Shared optimal temperature
    T_opt: float
    T_opt_se: float
    # Year effects
    k: Dict[int, float]
    # Fit stats
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int  # = 3
    residuals: np.ndarray
    total_r_squared: float
    # Diagnostics
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None
    # Compatibility field for plotting that expects h1
    h1: float = 0.0        # Linear term (always 0 for piecewise model)
    h1_se: float = 0.0     # SE for h1 (always 0)


@dataclass
class FitResultApproach8b:
    """Container for Approach 8b: Modulated actual temperature response.

    Model: h(T, Ttrend) = (1 + f1·(T-Ttrend) + f2·(T-Ttrend)²) · (h1·T + h2·T²)

    The quadratic response h1·T + h2·T² is evaluated at the actual temperature,
    then modulated by a factor (1 + f1·T_delta + f2·T_delta²) where T_delta = T - T_trend.
    When temperature equals the trend (T = Ttrend), the modulation is 1.

    Parameters:
        f1: Linear departure modulation coefficient (nonlinear parameter)
        f2: Quadratic departure modulation coefficient (nonlinear parameter)
        h1, h2: Actual temperature response coefficients (linear parameters)
        T_opt: Optimal actual temperature = -h1/(2*h2)
    """
    approach: str
    f1: float              # Linear modulation coefficient
    f1_se: float
    f2: float              # Quadratic modulation coefficient
    f2_se: float
    h1: float
    h1_se: float
    h2: float
    h2_se: float
    T_opt: float           # Optimal actual temperature
    k: Dict[int, float]
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int  # = 4 (f1, f2, h1, h2)
    residuals: np.ndarray
    total_r_squared: float
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None


@dataclass
class FitResultApproach8c:
    """Container for Approach 8c: Linear-only modulated actual temperature response.

    Model: h(T, Ttrend) = (1 + f1·(T-Ttrend)) · (h1·T + h2·T²)

    Like 8b but without the quadratic modulation term (f2=0).
    The quadratic response h1·T + h2·T² is evaluated at the actual temperature,
    then modulated by a linear factor (1 + f1·T_delta) where T_delta = T - T_trend.

    Parameters:
        f1: Linear departure modulation coefficient (nonlinear parameter)
        h1, h2: Actual temperature response coefficients (linear parameters)
        T_opt: Optimal actual temperature = -h1/(2*h2)
    """
    approach: str
    f1: float              # Linear modulation coefficient
    f1_se: float
    h1: float
    h1_se: float
    h2: float
    h2_se: float
    T_opt: float           # Optimal actual temperature
    k: Dict[int, float]
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int  # = 3 (f1, h1, h2)
    residuals: np.ndarray
    total_r_squared: float
    rms_imbalance: float = None
    rms_h: float = None
    imbalance_ratio: float = None
    var_decomp: dict = None
    var_attrib: dict = None


@dataclass
class FitResultApproach8d:
    """Container for Approach 8d: Quadratic-only modulated actual temperature response.

    Model: h(T, Ttrend) = (1 + f2·(T-Ttrend)²) · (h1·T + h2·T²)

    Like 8b but without the linear modulation term (f1=0).
    The quadratic response h1·T + h2·T² is evaluated at the actual temperature,
    then modulated by a quadratic factor (1 + f2·T_delta²) where T_delta = T - T_trend.

    Parameters:
        f2: Quadratic departure modulation coefficient (nonlinear parameter)
        h1, h2: Actual temperature response coefficients (linear parameters)
        T_opt: Optimal actual temperature = -h1/(2*h2)
    """
    approach: str
    f2: float              # Quadratic modulation coefficient
    f2_se: float
    h1: float
    h1_se: float
    h2: float
    h2_se: float
    T_opt: float           # Optimal actual temperature
    k: Dict[int, float]
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_obs: int
    n_params: int  # = 3 (f2, h1, h2)
    residuals: np.ndarray
    total_r_squared: float
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


def fit_approach2_temperature_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 2: Linear temperature detrending only.

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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 2: T_trend = T0 + T1*t (linear), j_trend = 0
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t
        k_values[i] = k[yr]
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
        approach="2: Linear Temperature Detrending",
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


def fit_approach3_growth_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 3: Quadratic GDP growth detrending only.

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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 3: T_trend = 0 (no T detrending), j_trend = y0 + y1*t + y2*t²
    T_trend = np.zeros(data.n_obs)  # No temperature detrending
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = k[yr]
    rms_imb = compute_rms_imbalance(h1, h2, T_trend, j_trend, k_values)

    # Compute RMS of h(T) - climate response to actual temperature
    rms_h = compute_rms_h(h1, h2, data.temp)
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    # Approach 3: no T detrending, so T_trend=0, T_star=T
    T_star_vals = data.temp - T_trend  # T_trend is zeros
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
        approach="3: Quadratic GDP Growth Detrending",
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


def fit_approach1_combined_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 1: Combined temperature and GDP growth detrending.

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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 1: T_trend = T0 + T1*t (linear), j_trend = y0 + y1*t + y2*t² (quadratic)
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = k[yr]
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
        approach="1: Combined Detrending (Mixed)",
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


def fit_approach4_combined_quadratic_detrending(
    data: AnalysisData, trends: CountryTrends
) -> FitResult:
    """Approach 4: Combined detrending with quadratic temperature and GDP growth trends.

    Δy_i(t) - (y0 + y1*t + y2*t²) = h1*[T - (T0 + T1*t + T2*t²)]
                                  + h2*[T² - (T0 + T1*t + T2*t²)²] + k_i

    Like Approach 1, but uses quadratic temperature trend instead of linear.
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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 4: T_trend = T0_quad + T1_quad*t + T2_quad*t² (quadratic), j_trend = y0 + y1*t + y2*t² (quadratic)
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0_quad[c] + trends.T1_quad[c] * t + trends.T2_quad[c] * t * t
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = k[yr]
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
        approach="4: Combined Quadratic Detrending",
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


def fit_approach5_precomputed_k_quadratic(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 5: Pre-computed k[t] with quadratic country/temperature trends.

    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = j_{0,i} + j_{1,i}*t + j_{2,i}*t² are fit to dy_i[t] - k[t]
    3. Temperature is detrended with quadratic trend: T* = T - (T0 + T1*t + T2*t²)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Unlike Approaches 3/4, year effects k[t] are pre-computed as year means
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
    T_opt = compute_T_optimal(h1, h2)

    # Compute RMS imbalance: h(T_trend) + j_trend + k
    # Approach 5: T_trend = T0_quad + T1_quad*t + T2_quad*t² (quadratic), j_trend = y0 + y1*t + y2*t² (quadratic)
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0_quad[c] + trends.T1_quad[c] * t + trends.T2_quad[c] * t * t
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
        approach="5: Precomputed k Quadratic",
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


def fit_approach5a_precomputed_k_linear_temp(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 5a: Pre-computed k[t] with linear temperature detrending only.

    Like Approach 2 but with precomputed k:
    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) are fit to (dy_i[t] - k[t]) - NOT used for regression
    3. Temperature is detrended with linear trend: T* = T - (T0 + T1*t)
    4. Final regression: (dy_i[t] - k[t]) = h1*T* + h2*T*²

    Note: No GDP detrending (j_i not subtracted from dependent variable).
    """
    # Compute detrended temperature terms (linear)
    T_star = compute_detrended_temperature(data, trends)
    T2_detrend = compute_detrended_temp_squared(data, trends)

    # Compute dependent variable: dy_i[t] - k[t] (no j_i subtraction)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr]

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
    # Approach 5a: T_trend = T0 + T1*t (linear), j_trend = 0 (no GDP detrending)
    T_trend = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t
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
        approach="5a: Precomputed k Linear Temp",
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


def fit_approach5b_precomputed_k_gdp_only(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 5b: Pre-computed k[t] with GDP detrending only.

    Like Approach 3 but with precomputed k:
    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = y0 + y1*t + y2*t² are fit to (dy_i[t] - k[t])
    3. No temperature detrending: T* = T (raw temperature)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T + h2*T²

    Note: No temperature detrending, only GDP trend removal.
    """
    # No temperature detrending - use raw temperature
    T = data.temp
    T2 = data.temp ** 2

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

    # Design matrix: just [T, T²] - no year fixed effects
    X = np.column_stack([T, T2])

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
    # Approach 5b: T_trend = 0 (no T detrending), j_trend = y0 + y1*t + y2*t²
    T_trend = np.zeros(data.n_obs)  # No temperature detrending
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]
    rms_imb = compute_rms_imbalance(h1, h2, T_trend, j_trend, k_values)

    # Compute RMS of h(T) - climate response to actual temperature
    rms_h = compute_rms_h(h1, h2, data.temp)
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    # Approach 5b: no T detrending, so T_trend=0, T_star=T
    T_star_vals = data.temp - T_trend  # T_trend is zeros
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
        approach="5b: Precomputed k GDP Only",
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


def fit_approach5c_precomputed_k_combined(
    data: AnalysisData, trends: CountryTrends, year_means: dict
) -> FitResult:
    """Approach 5c: Pre-computed k[t] with linear temp + quadratic GDP detrending.

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
        approach="5c: Precomputed k Combined",
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


def fit_approach5d_precomputed_k_gdp_response(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    Y_ref: float,
    beta_bounds: tuple = DEFAULT_BETA_BOUNDS,
) -> FitResultApproach8:
    """Approach 5d: GDP-only response with precomputed k.

    Model: dy - k[t] - j_i[t] = h0 * (pcGDP/Y_ref)^(-beta)

    Like 5c but replaces h(T) with GDP-dependent response only.
    No temperature dependence - isolates GDP effects.

    Args:
        data: AnalysisData object
        trends: CountryTrends (fit to dy - k, with quadratic GDP trend)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        Y_ref: Reference GDP (computed once on full dataset)
        beta_bounds: Bounds for beta optimization (default [0.01, 0.99])

    Returns:
        FitResultApproach8 with beta, h0 (stored as h1), and h2=0
    """
    # Compute dependent variable: dy_i[t] - k[t] - j_i[t]
    # where j_i[t] is the quadratic trend fit to (dy - k)
    y = np.zeros(data.n_obs)
    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        # j_i[t] = y0 + y1*t + y2*t² (fit to dy - k)
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t
        j_trend[i] = j_i_t
        k_values[i] = year_means[yr]

    # Define objective function: SSE for given beta
    def compute_sse_for_beta(beta):
        """Compute SSE for a given beta by solving inner OLS problem."""
        # Compute GDP scaling factor g = (Y/Y_ref)^(-beta)
        g = compute_gdp_scaling(data.pcGDP, Y_ref, beta)

        # Build design matrix: X = [g] (single column, no T terms)
        X = g.reshape(-1, 1)

        # Solve OLS: min ||y - X @ [h0]||^2
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

    # Re-fit at optimal beta to get h0 and covariance
    g_opt = compute_gdp_scaling(data.pcGDP, Y_ref, beta_opt)
    X_opt = g_opt.reshape(-1, 1)

    beta_ols, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h0 = beta_ols[0]
    h0_se = np.sqrt(cov[0, 0])

    # Store h0 in h1 slot, h2 = 0 (no temperature dependence)
    h1 = h0
    h2 = 0.0
    h1_se = h0_se
    h2_se = 0.0

    # Compute beta SE via numerical second derivative
    beta_se = compute_1d_se_numerical(
        compute_sse_for_beta, beta_opt, beta_bounds, data.n_obs, n_params=2
    )

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # h0, beta
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # No optimal temperature (no temperature dependence)
    T_opt = np.nan

    # Compute RMS imbalance: h0 * g_trend + j_trend + k
    # g_trend uses trend GDP values (we need to compute what GDP would be along the trend)
    # For simplicity, we use actual GDP values for g
    g_h = h0 * g_opt
    imbalance = g_h + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # Compute RMS of h - GDP-scaled response magnitude
    rms_h = np.sqrt(np.mean(g_h ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    # Components: g_h (GDP response), j (country trends), k (year effects)
    components = {
        'g_h': g_h, 'j': j_trend, 'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (simplified for GDP-only model)
    # Δu = 0 (no temperature variation), v = g_h, j = j_trend, k = k_values, ε = remainder
    Delta_u = np.zeros(data.n_obs)  # No temperature effect
    v = g_h  # GDP response is the "trend" component
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach5d(
        approach="5d: Precomputed k GDP Response",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        f1=beta_opt,
        f1_se=beta_se,
        Y_ref=Y_ref,
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


def fit_approach6_precomputed_k_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResult:
    """Approach 6: Pre-computed k[t] with LOESS country/temperature trends.

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
        approach="6: Precomputed k LOESS",
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


def fit_approach6a_separate_high_low_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6ab:
    """Approach 6a: T/departure temperature decomposition with LOESS detrending.

    Model: h(T, Ttrend) = h1*T + h2*T² + h3*(T-Ttrend) + h4*(T-Ttrend)²

    Where:
        - delta-y*(t) = dy(t) - k(t) - y_loess (same as approach 6)
        - h1, h2: response to actual temperature T
        - h3, h4: response to departure from trend (T - Ttrend)
        - k(t) pre-computed (not in OLS)

    When T = Ttrend: h = h1*T + h2*T² (pure T response, no departure effect)

    4-parameter OLS with design matrix: [T, T², (T-Ttrend), (T-Ttrend)²]

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResultApproach6ab
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_dep = T - Ttrend  # Departure from trend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [T, T², (T-Ttrend), (T-Ttrend)²]
    X = np.column_stack([T, T**2, T_dep, T_dep**2])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1_T = beta[0]       # Linear coef for actual T
    h2_T = beta[1]       # Quadratic coef for actual T
    h1_dep = beta[2]     # Linear coef for departure
    h2_dep = beta[3]     # Quadratic coef for departure
    h1_T_se = np.sqrt(cov[0, 0])
    h2_T_se = np.sqrt(cov[1, 1])
    h1_dep_se = np.sqrt(cov[2, 2])
    h2_dep_se = np.sqrt(cov[3, 3])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 4  # h1_T, h2_T, h1_dep, h2_dep
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_T = compute_T_optimal(h1_T, h2_T)        # Optimal actual T
    T_optimal_dep = compute_T_optimal(h1_dep, h2_dep)  # Optimal departure

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_T(T) - response to actual T
    h_T_val = h1_T * T + h2_T * T**2
    # h_dep(T - Ttrend) - response to departure
    h_dep_val = h1_dep * T_dep + h2_dep * T_dep**2

    # RMS imbalance: h_T(T) + j_trend + k (at trend: departure=0, so h = h_T)
    imbalance = h_T_val + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of total response magnitude
    h_total = h_T_val + h_dep_val
    rms_h = np.sqrt(np.mean(h_total ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    # Delta_u = h_dep(T - Ttrend) (climate fluctuation effect from departures)
    Delta_u = h_dep_val
    components = {
        'h_T': h_T_val,
        'h_dep': h_dep_val,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # v = h_T(T) is the baseline at actual temperature
    v = h_T_val
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6ab(
        approach="6a: T/Departure LOESS",
        h1=h1_T,
        h2=h2_T,
        h1_se=h1_T_se,
        h2_se=h2_T_se,
        h3=h1_dep,
        h4=h2_dep,
        h3_se=h1_dep_se,
        h4_se=h2_dep_se,
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


def fit_approach6b_low_only_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6ab:
    """Approach 6b: T-only temperature response with LOESS detrending.

    Model: h(T, Ttrend) = h1*T + h2*T² (departure terms h3,h4 are zero)

    Same as 6a but departure response terms are zero.

    2-parameter OLS with design matrix: [T, T²]

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResultApproach6ab with departure coefficients set to zero
    """
    # Get temperature
    T = data.temp

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [T, T²]
    X = np.column_stack([T, T**2])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients (h1,h2 for T; h3,h4 for departure are zero)
    h1_T = beta[0]
    h2_T = beta[1]
    h1_T_se = np.sqrt(cov[0, 0])
    h2_T_se = np.sqrt(cov[1, 1])
    h1_dep = 0.0
    h2_dep = 0.0
    h1_dep_se = 0.0
    h2_dep_se = 0.0

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # h1_T, h2_T
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_T = compute_T_optimal(h1_T, h2_T)
    T_optimal_dep = np.nan  # No departure response

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_T(T)
    h_T_val = h1_T * T + h2_T * T**2

    # RMS imbalance: h_T(T) + j_trend + k
    imbalance = h_T_val + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response (just h_T since departure = 0)
    rms_h = np.sqrt(np.mean(h_T_val ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition (no departure component)
    Delta_u = np.zeros(data.n_obs)  # No departure effect
    components = {
        'h_T': h_T_val,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # v = h_T(T) is the baseline
    v = h_T_val
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6ab(
        approach="6b: T Only LOESS",
        h1=h1_T,
        h2=h2_T,
        h1_se=h1_T_se,
        h2_se=h2_T_se,
        h3=h1_dep,
        h4=h2_dep,
        h3_se=h1_dep_se,
        h4_se=h2_dep_se,
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


def fit_approach6d_linear_departure_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6ab:
    """Approach 6d: T response with linear departure term only (LOESS detrending).

    Model: h(T, Ttrend) = h1*T + h2*T² + h3*(T-Ttrend)

    Where:
        - delta-y*(t) = dy(t) - k(t) - y_loess (same as approach 6)
        - h1, h2: response to actual temperature T
        - h3: linear response to departure from trend (T - Ttrend)
        - h4 = 0 (no quadratic departure term)

    3-parameter OLS with design matrix: [T, T², (T-Ttrend)]

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResultApproach6ab with h4=0
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_dep = T - Ttrend  # Departure from trend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [T, T², (T-Ttrend)] - no quadratic departure term
    X = np.column_stack([T, T**2, T_dep])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1_T = beta[0]       # Linear coef for actual T
    h2_T = beta[1]       # Quadratic coef for actual T
    h1_dep = beta[2]     # Linear coef for departure
    h2_dep = 0.0         # No quadratic departure term
    h1_T_se = np.sqrt(cov[0, 0])
    h2_T_se = np.sqrt(cov[1, 1])
    h1_dep_se = np.sqrt(cov[2, 2])
    h2_dep_se = 0.0

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1_T, h2_T, h1_dep
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_T = compute_T_optimal(h1_T, h2_T)  # Optimal actual T
    T_optimal_dep = np.nan  # No quadratic term, so no optimal departure

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_T(T) - response to actual T
    h_T_val = h1_T * T + h2_T * T**2
    # h_dep(T - Ttrend) - linear departure only
    h_dep_val = h1_dep * T_dep

    # RMS imbalance: h_T(T) + j_trend + k (at trend: departure=0, so h = h_T)
    imbalance = h_T_val + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of total response magnitude
    h_total = h_T_val + h_dep_val
    rms_h = np.sqrt(np.mean(h_total ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    Delta_u = h_dep_val
    components = {
        'h_T': h_T_val,
        'h_dep': h_dep_val,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution
    v = h_T_val
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6ab(
        approach="6d: T/Linear Departure LOESS",
        h1=h1_T,
        h2=h2_T,
        h1_se=h1_T_se,
        h2_se=h2_T_se,
        h3=h1_dep,
        h4=h2_dep,
        h3_se=h1_dep_se,
        h4_se=h2_dep_se,
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


def fit_approach6e_quadratic_departure_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6ab:
    """Approach 6e: T response with quadratic departure term only (LOESS detrending).

    Model: h(T, Ttrend) = h1*T + h2*T² + h4*(T-Ttrend)²

    Where:
        - delta-y*(t) = dy(t) - k(t) - y_loess (same as approach 6)
        - h1, h2: response to actual temperature T
        - h4: quadratic response to departure from trend (T - Ttrend)
        - h3 = 0 (no linear departure term)

    3-parameter OLS with design matrix: [T, T², (T-Ttrend)²]

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

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [T, T², (T-Ttrend)²] - no linear departure term
    X = np.column_stack([T, T**2, T_dep**2])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1_T = beta[0]       # Linear coef for actual T
    h2_T = beta[1]       # Quadratic coef for actual T
    h1_dep = 0.0         # No linear departure term
    h2_dep = beta[2]     # Quadratic coef for departure
    h1_T_se = np.sqrt(cov[0, 0])
    h2_T_se = np.sqrt(cov[1, 1])
    h1_dep_se = 0.0
    h2_dep_se = np.sqrt(cov[2, 2])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # h1_T, h2_T, h2_dep
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_T = compute_T_optimal(h1_T, h2_T)  # Optimal actual T
    # For quadratic-only departure: h_dep = h4*(T-Ttrend)², optimal is at T-Ttrend = 0
    T_optimal_dep = 0.0

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_T(T) - response to actual T
    h_T_val = h1_T * T + h2_T * T**2
    # h_dep(T - Ttrend) - quadratic departure only
    h_dep_val = h2_dep * T_dep**2

    # RMS imbalance: h_T(T) + j_trend + k (at trend: departure=0, so h = h_T)
    imbalance = h_T_val + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of total response magnitude
    h_total = h_T_val + h_dep_val
    rms_h = np.sqrt(np.mean(h_total ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    Delta_u = h_dep_val
    components = {
        'h_T': h_T_val,
        'h_dep': h_dep_val,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution
    v = h_T_val
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6ab(
        approach="6e: T/Quadratic Departure LOESS",
        h1=h1_T,
        h2=h2_T,
        h1_se=h1_T_se,
        h2_se=h2_T_se,
        h3=h1_dep,
        h4=h2_dep,
        h3_se=h1_dep_se,
        h4_se=h2_dep_se,
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


def fit_approach6c_departure_trend_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResultApproach6c:
    """Approach 6c: Departure/trend decomposition with LOESS detrending.

    Model: Δy*(t) = h1_dep*(T-Ttrend) + h2_dep*(T-Ttrend)² + h1_trend*Ttrend + h2_trend*Ttrend²

    Where:
        - Δy*(t) = dy(t) - k(t) - y_loess (same as approach 6)
        - h1_dep, h2_dep: response to departure from trend (T - Ttrend)
        - h1_trend, h2_trend: response to trend temperature (Ttrend)

    When T = Ttrend: h = h1_trend*Ttrend + h2_trend*Ttrend² (pure trend response)

    4-parameter OLS with design matrix: [(T-Ttrend), (T-Ttrend)², Ttrend, Ttrend²]

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])

    Returns:
        FitResultApproach6c
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_dep = T - Ttrend  # Departure from trend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    # Design matrix: [(T-Ttrend), (T-Ttrend)², Ttrend, Ttrend²]
    X = np.column_stack([T_dep, T_dep**2, Ttrend, Ttrend**2])

    # Fit OLS
    beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1_dep = beta[0]
    h2_dep = beta[1]
    h1_trend = beta[2]
    h2_trend = beta[3]
    h1_dep_se = np.sqrt(cov[0, 0])
    h2_dep_se = np.sqrt(cov[1, 1])
    h1_trend_se = np.sqrt(cov[2, 2])
    h2_trend_se = np.sqrt(cov[3, 3])

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 4  # h1_dep, h2_dep, h1_trend, h2_trend
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperatures
    T_optimal_dep = compute_T_optimal(h1_dep, h2_dep)
    T_optimal_trend = compute_T_optimal(h1_trend, h2_trend)

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_dep(T - Ttrend)
    h_dep = h1_dep * T_dep + h2_dep * T_dep**2
    # h_trend(Ttrend)
    h_trend = h1_trend * Ttrend + h2_trend * Ttrend**2
    # Total climate response
    h_T = h_dep + h_trend

    # RMS imbalance: h_trend(Ttrend) + j_trend + k (baseline when T = Ttrend)
    imbalance = h_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response (departure component drives variation)
    rms_h = np.sqrt(np.mean(h_dep ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_T,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Δu = h_dep (response to departure from trend)
    # v = h_trend (baseline at trend temperature)
    Delta_u = h_dep
    v = h_trend
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach6c(
        approach="6c: Departure/Trend LOESS",
        h1=h1_dep,
        h2=h2_dep,
        h1_se=h1_dep_se,
        h2_se=h2_dep_se,
        h3=h1_trend,
        h4=h2_trend,
        h3_se=h1_trend_se,
        h4_se=h2_trend_se,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_dep_opt=T_optimal_dep,
        f2=T_optimal_trend,
        total_r_squared=total_r_sq,
        rms_imbalance=rms_imb,
        rms_h=rms_h,
        imbalance_ratio=imb_ratio,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_approach8a_shared_Topt_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
) -> FitResultApproach8a:
    """Approach 8a: Separate total/trend response with shared T_opt and LOESS detrending.

    Model: delta-y*(t) = h2_total * (T - T_opt)^2 - h2_trend * (Ttrend - T_opt)^2

    Both total (actual T) and trend (Ttrend) responses share a common optimal temperature T_opt.

    Total Response: When T = Ttrend, the net effect is (h2_total - h2_trend) * (T - T_opt)^2

    Nonlinear optimization:
        - Outer: L-BFGS-B over T_opt
        - Inner: 2-parameter OLS for h2_total (total), h2_trend (trend)

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])

    Returns:
        FitResultApproach8a with T_opt, h2_total (total), h2_trend (trend), and standard errors
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving inner 2-parameter OLS for h2_total, h2_trend."""
        # Design matrix: [(T - T_opt)^2, -(Ttrend - T_opt)^2]
        X1 = (T - T_opt_val) ** 2
        X2 = -(Ttrend - T_opt_val) ** 2
        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h2_total, h2_trend]||^2
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

    # Re-fit at optimal T_opt to get h2_total, h2_trend, residuals, covariance
    X1 = (T - T_opt_opt) ** 2
    X2 = -(Ttrend - T_opt_opt) ** 2
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_total = beta_ols[0]
    h2_trend = beta_ols[1]
    h2_total_se = np.sqrt(cov[0, 0])
    h2_trend_se = np.sqrt(cov[1, 1])

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
    n_params = 3  # h2_total, h2_trend, T_opt
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h_total(T) = h2_total * (T - T_opt)^2
    h_total_T = h2_total * (T - T_opt_opt) ** 2
    # h_trend(Ttrend) = h2_trend * (Ttrend - T_opt)^2
    h_trend_Ttrend = h2_trend * (Ttrend - T_opt_opt) ** 2

    # Climate response: h_total(T) - h_trend(Ttrend)
    Delta_u = h_total_T - h_trend_Ttrend

    # RMS imbalance: h_trend(Ttrend) + j_trend + k
    imbalance = h_trend_Ttrend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response
    rms_h = np.sqrt(np.mean(Delta_u ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_total_T': h_total_T,
        'h_trend_Ttrend': -h_trend_Ttrend,  # negative because model is h_total - h_trend
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    v = h_trend_Ttrend
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8a(
        approach="8a: Shared T_opt Total/Trend LOESS",
        h2=h2_total,
        h2_se=h2_total_se,
        h4=h2_trend,
        h4_se=h2_trend_se,
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


def fit_approach8b_modulated_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    f_bounds: tuple = ((-20.0, 20.0), (-20.0, 20.0)),
) -> FitResultApproach8b:
    """Approach 8b: Modulated actual temperature response with LOESS detrending.

    Model: h(T, Ttrend) = (1 + f₁·(T-Ttrend) + f₂·(T-Ttrend)²) · (h₁·T + h₂·T²)

    The quadratic response h₁·T + h₂·T² is evaluated at the actual temperature,
    then modulated by a factor (1 + f₁·T_delta + f₂·T_delta²) where T_delta = T - T_trend.
    When T = T_trend, the modulation is 1.

    Fitting Strategy:
        - Outer: 2D L-BFGS-B over (f₁, f₂)
        - Inner: 2-parameter OLS for h₁, h₂

    Given (f₁, f₂), the model is linear in h₁ and h₂:
        y = h₁ · [(1 + f₁·T_delta + f₂·T_delta²) · T] + h₂ · [(1 + f₁·T_delta + f₂·T_delta²) · T²]
          = h₁ · X₁ + h₂ · X₂

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        f_bounds: Bounds for (f₁, f₂) as tuple of tuples (default [(-20, 20), (-20, 20)])

    Returns:
        FitResultApproach8b with f₁, f₂, h₁, h₂, and standard errors
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_delta = T - Ttrend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    def compute_sse_for_f(f_vals):
        """Compute SSE for given (f₁, f₂) by solving inner 2-parameter OLS for h₁, h₂."""
        f1_val, f2_val = f_vals
        # Linear and quadratic departure modulation factor
        modulation = 1 + f1_val * T_delta + f2_val * T_delta ** 2

        # Design matrix: [modulation * T, modulation * T²]
        X1 = modulation * T
        X2 = modulation * (T ** 2)
        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h₁, h₂]||²
        beta_ols, _, _, _ = linalg.lstsq(X, y)
        y_pred = X @ beta_ols
        sse = np.sum((y - y_pred) ** 2)
        return sse

    # Initial guess: f₁ = 0, f₂ = 0 (no modulation)
    x0 = [0.0, 0.0]

    # 2D optimization using L-BFGS-B
    result = minimize(
        compute_sse_for_f,
        x0=x0,
        bounds=f_bounds,
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    f1_opt, f2_opt = result.x

    # Re-fit at optimal (f₁, f₂) to get h₁, h₂, residuals, covariance
    modulation = 1 + f1_opt * T_delta + f2_opt * T_delta ** 2
    X1 = modulation * T
    X2 = modulation * (T ** 2)
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute SE for (f₁, f₂) using numerical 2D Hessian
    f1_se, f2_se = compute_2d_se_numerical(
        compute_sse_for_f,
        np.array([f1_opt, f2_opt]),
        f_bounds,
        data.n_obs,
        n_params=4
    )

    # Compute T_optimal from h₁ and h₂ (applies to actual temperature)
    T_opt = compute_T_optimal(h1, h2)

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 4  # f₁, f₂, h₁, h₂
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response: (1 + f₁·T_delta + f₂·T_delta²) · (h₁·T + h₂·T²)
    h_T = modulation * (h1 * T + h2 * T ** 2)

    # Baseline at T = T_trend (modulation = 1): h₁·T_trend + h₂·T_trend²
    h_T_trend = h1 * Ttrend + h2 * Ttrend ** 2

    # Δu = h(T) - h(T_trend) (increment from deviation)
    Delta_u = h_T - h_T_trend

    # RMS imbalance: h(T_trend) + j_trend + k
    imbalance = h_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response
    rms_h = np.sqrt(np.mean(h_T ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_T,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    v = h_T_trend
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8b(
        approach="8b: Modulated Response LOESS",
        f1=f1_opt,
        f1_se=f1_se,
        f2=f2_opt,
        f2_se=f2_se,
        h1=h1,
        h1_se=h1_se,
        h2=h2,
        h2_se=h2_se,
        T_opt=T_opt,
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


def fit_approach8c_linear_modulated_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    f_bounds: tuple = (-20.0, 20.0),
) -> FitResultApproach8c:
    """Approach 8c: Linear-only modulated actual temperature response with LOESS detrending.

    Model: h(T, Ttrend) = (1 + f₁·(T-Ttrend)) · (h₁·T + h₂·T²)

    Like 8b but without the quadratic modulation term (f₂=0).
    The quadratic response h₁·T + h₂·T² is evaluated at the actual temperature,
    then modulated by a linear factor (1 + f₁·T_delta) where T_delta = T - T_trend.
    When T = T_trend, the modulation is 1.

    Fitting Strategy:
        - Outer: 1D L-BFGS-B over f₁
        - Inner: 2-parameter OLS for h₁, h₂

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        f_bounds: Bounds for f₁ as tuple (default (-20, 20))

    Returns:
        FitResultApproach8c with f₁, h₁, h₂, and standard errors
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_delta = T - Ttrend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    def compute_sse_for_f(f1_val):
        """Compute SSE for given f₁ by solving inner 2-parameter OLS for h₁, h₂."""
        # Linear departure modulation factor only
        modulation = 1 + f1_val * T_delta

        # Design matrix: [modulation * T, modulation * T²]
        X1 = modulation * T
        X2 = modulation * (T ** 2)
        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h₁, h₂]||²
        beta_ols, _, _, _ = linalg.lstsq(X, y)
        y_pred = X @ beta_ols
        sse = np.sum((y - y_pred) ** 2)
        return sse

    # Initial guess: f₁ = 0 (no modulation)
    x0 = [0.0]

    # 1D optimization using L-BFGS-B
    result = minimize(
        compute_sse_for_f,
        x0=x0,
        bounds=[f_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    f1_opt = result.x[0]

    # Re-fit at optimal f₁ to get h₁, h₂, residuals, covariance
    modulation = 1 + f1_opt * T_delta
    X1 = modulation * T
    X2 = modulation * (T ** 2)
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute SE for f₁ using numerical Hessian
    f1_se = compute_1d_se_numerical(
        compute_sse_for_f,
        f1_opt,
        f_bounds,
        data.n_obs,
        n_params=3
    )

    # Compute T_optimal from h₁ and h₂ (applies to actual temperature)
    T_opt = compute_T_optimal(h1, h2)

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # f₁, h₁, h₂
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response: (1 + f₁·T_delta) · (h₁·T + h₂·T²)
    h_T = modulation * (h1 * T + h2 * T ** 2)

    # Baseline at T = T_trend (modulation = 1): h₁·T_trend + h₂·T_trend²
    h_T_trend = h1 * Ttrend + h2 * Ttrend ** 2

    # Δu = h(T) - h(T_trend) (increment from deviation)
    Delta_u = h_T - h_T_trend

    # RMS imbalance: h(T_trend) + j_trend + k
    imbalance = h_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response
    rms_h = np.sqrt(np.mean(h_T ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_T,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    v = h_T_trend
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8c(
        approach="8c: Linear Modulated LOESS",
        f1=f1_opt,
        f1_se=f1_se,
        h1=h1,
        h1_se=h1_se,
        h2=h2,
        h2_se=h2_se,
        T_opt=T_opt,
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


def fit_approach8d_quadratic_modulated_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    f_bounds: tuple = (-20.0, 20.0),
) -> FitResultApproach8d:
    """Approach 8d: Quadratic-only modulated actual temperature response with LOESS detrending.

    Model: h(T, Ttrend) = (1 + f₂·(T-Ttrend)²) · (h₁·T + h₂·T²)

    Like 8b but without the linear modulation term (f₁=0).
    The quadratic response h₁·T + h₂·T² is evaluated at the actual temperature,
    then modulated by a quadratic factor (1 + f₂·T_delta²) where T_delta = T - T_trend.
    When T = T_trend, the modulation is 1.

    Fitting Strategy:
        - Outer: 1D L-BFGS-B over f₂
        - Inner: 2-parameter OLS for h₁, h₂

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        f_bounds: Bounds for f₂ as tuple (default (-20, 20))

    Returns:
        FitResultApproach8d with f₂, h₁, h₂, and standard errors
    """
    # Get temperature and trend temperature
    T = data.temp
    Ttrend = trends_loess.T_loess
    T_delta = T - Ttrend

    # Compute dependent variable: dy - k[t] - y_loess (same as approach 6)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    def compute_sse_for_f(f2_val):
        """Compute SSE for given f₂ by solving inner 2-parameter OLS for h₁, h₂."""
        # Quadratic departure modulation factor only
        modulation = 1 + f2_val * T_delta ** 2

        # Design matrix: [modulation * T, modulation * T²]
        X1 = modulation * T
        X2 = modulation * (T ** 2)
        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h₁, h₂]||²
        beta_ols, _, _, _ = linalg.lstsq(X, y)
        y_pred = X @ beta_ols
        sse = np.sum((y - y_pred) ** 2)
        return sse

    # Initial guess: f₂ = 0 (no modulation)
    x0 = [0.0]

    # 1D optimization using L-BFGS-B
    result = minimize(
        compute_sse_for_f,
        x0=x0,
        bounds=[f_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    f2_opt = result.x[0]

    # Re-fit at optimal f₂ to get h₁, h₂, residuals, covariance
    modulation = 1 + f2_opt * T_delta ** 2
    X1 = modulation * T
    X2 = modulation * (T ** 2)
    X_opt = np.column_stack([X1, X2])

    beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(cov[0, 0])
    h2_se = np.sqrt(cov[1, 1])

    # Compute SE for f₂ using numerical Hessian
    f2_se = compute_1d_se_numerical(
        compute_sse_for_f,
        f2_opt,
        f_bounds,
        data.n_obs,
        n_params=3
    )

    # Compute T_optimal from h₁ and h₂ (applies to actual temperature)
    T_opt = compute_T_optimal(h1, h2)

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 3  # f₂, h₁, h₂
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Compute RMS values
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response: (1 + f₂·T_delta²) · (h₁·T + h₂·T²)
    h_T = modulation * (h1 * T + h2 * T ** 2)

    # Baseline at T = T_trend (modulation = 1): h₁·T_trend + h₂·T_trend²
    h_T_trend = h1 * Ttrend + h2 * Ttrend ** 2

    # Δu = h(T) - h(T_trend) (increment from deviation)
    Delta_u = h_T - h_T_trend

    # RMS imbalance: h(T_trend) + j_trend + k
    imbalance = h_T_trend + j_trend + k_values
    rms_imb = np.sqrt(np.mean(imbalance ** 2))

    # RMS of climate response
    rms_h = np.sqrt(np.mean(h_T ** 2))
    imb_ratio = rms_imb / rms_h if rms_h > 0 else np.nan

    # Compute variance decomposition
    components = {
        'h_T': h_T,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    v = h_T_trend
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8d(
        approach="8d: Quadratic Modulated LOESS",
        f2=f2_opt,
        f2_se=f2_se,
        h1=h1,
        h1_se=h1_se,
        h2=h2,
        h2_se=h2_se,
        T_opt=T_opt,
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


def fit_approach8_piecewise_loess(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
) -> FitResultApproach8:
    """Approach 8: Piecewise quadratic temperature response with LOESS detrending.

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
        approach="8: Piecewise Quadratic LOESS",
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


# Keep backward-compatible alias
fit_approach8_gaussian_loess = fit_approach8_piecewise_loess


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


def fit_nocr0_joint(data: AnalysisData) -> FitResult:
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
        approach="nocr0: No Climate Response (Joint)",
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


def fit_nocr5_precomputed_k(
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
        approach="nocr5: No Climate Response (Precomputed k)",
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
    Y_ref: float = None, trends_loess: CountryTrendsLoess = None
) -> dict:
    """Fit all approaches and return results.

    Returns dict with keys:
        'approach0': Conjoined OLS fit, with j terms and year fixed effects
        'approach1': Combined detrending (linear T trend, quadratic GDP trend)
        'approach2': Temperature detrending (linear T trend)
        'approach3': GDP growth detrending (quadratic GDP trend)
        'approach4': Combined detrending (quadratic T trend, quadratic GDP trend)
        'approach5': Pre-computed k with quadratic trends (if trends_with_k and year_means provided)
        'approach5a': Pre-computed k with linear temp only (if trends_with_k and year_means provided)
        'approach5b': Pre-computed k with GDP only (if trends_with_k and year_means provided)
        'approach5c': Pre-computed k with linear temp + quadratic GDP (if trends_with_k and year_means provided)
        'approach5d': Pre-computed k with GDP-only response (if trends_with_k, year_means, and Y_ref provided)
        'approach6': Pre-computed k with LOESS trends (if trends_loess provided)
        'approach6a': Separate high/low frequency responses with LOESS (if trends_loess provided)
        'approach6b': Low frequency only response with LOESS (if trends_loess provided)
        'approach6c': Departure/trend decomposition with LOESS (if trends_loess provided)
        'approach8': Piecewise quadratic response with LOESS (if trends_loess provided)
        'approach8a': Shared T_opt for total/trend with LOESS (if trends_loess provided)
        'approach8b': Modulated temperature response with LOESS (if trends_loess provided)
        'approach8c': Linear-only modulated response with LOESS (if trends_loess provided)
        'approach8d': Quadratic-only modulated response with LOESS (if trends_loess provided)
        'nocr0': No climate response, joint OLS (country trends + year effects only)
        'nocr5': No climate response, precomputed k (if trends_with_k and year_means provided)

    Args:
        data: AnalysisData object
        trends: CountryTrends for approaches 0-4
        trends_with_k: CountryTrends for approach 5, 5a, 5b, 5c (fit to dy - k)
        year_means: Pre-computed k[t] for approaches 5, 5a, 5b, 5c, 6, 6a, 6b, 8, 8a
        Y_ref: Reference GDP for approach 5d (computed once on full dataset)
        trends_loess: CountryTrendsLoess for approaches 6-8a (LOESS detrending)
    """
    results = {
        'approach0': fit_approach0_no_detrending(data),
        'approach1': fit_approach1_combined_detrending(data, trends),
        'approach2': fit_approach2_temperature_detrending(data, trends),
        'approach3': fit_approach3_growth_detrending(data, trends),
        'approach4': fit_approach4_combined_quadratic_detrending(data, trends),
        'nocr0': fit_nocr0_joint(data),
    }

    # Add approach 5 and variants if trends_with_k and year_means are provided
    if trends_with_k is not None and year_means is not None:
        results['approach5'] = fit_approach5_precomputed_k_quadratic(
            data, trends_with_k, year_means
        )
        results['approach5a'] = fit_approach5a_precomputed_k_linear_temp(
            data, trends_with_k, year_means
        )
        results['approach5b'] = fit_approach5b_precomputed_k_gdp_only(
            data, trends_with_k, year_means
        )
        results['approach5c'] = fit_approach5c_precomputed_k_combined(
            data, trends_with_k, year_means
        )
        # Add approach 5d if Y_ref is provided
        if Y_ref is not None:
            results['approach5d'] = fit_approach5d_precomputed_k_gdp_response(
                data, trends_with_k, year_means, Y_ref
            )
        results['nocr5'] = fit_nocr5_precomputed_k(
            data, trends_with_k, year_means
        )

    # Add approaches 6, 6a, 6b, 8, 8a if trends_loess and year_means are provided
    if trends_loess is not None and year_means is not None:
        results['approach6'] = fit_approach6_precomputed_k_loess(
            data, trends_loess, year_means
        )
        results['approach6a'] = fit_approach6a_separate_high_low_loess(
            data, trends_loess, year_means
        )
        results['approach6b'] = fit_approach6b_low_only_loess(
            data, trends_loess, year_means
        )
        results['approach6c'] = fit_approach6c_departure_trend_loess(
            data, trends_loess, year_means
        )
        results['approach6d'] = fit_approach6d_linear_departure_loess(
            data, trends_loess, year_means
        )
        results['approach6e'] = fit_approach6e_quadratic_departure_loess(
            data, trends_loess, year_means
        )

        # Add approach 8, 8a, and 8b (piecewise/shared T_opt/modulated)
        results['approach8'] = fit_approach8_gaussian_loess(
            data, trends_loess, year_means
        )
        results['approach8a'] = fit_approach8a_shared_Topt_loess(
            data, trends_loess, year_means
        )
        results['approach8b'] = fit_approach8b_modulated_loess(
            data, trends_loess, year_means
        )
        results['approach8c'] = fit_approach8c_linear_modulated_loess(
            data, trends_loess, year_means
        )
        results['approach8d'] = fit_approach8d_quadratic_modulated_loess(
            data, trends_loess, year_means
        )

    return results
