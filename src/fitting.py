"""OLS fitting for climate-GDP response approaches.

Approaches (publication-ready):
    Approach NJ: Null model (h1=h2=0) with joint OLS (country trends + year effects only)
    Approach NP: Null model (h1=h2=0) with polynomial trend identification
    Approach NL: Null model (h1=h2=0) with LOESS trend identification
    Approach QJ: Quadratic response with joint OLS (country time trends and year fixed effects)
    Approach QP: Quadratic response with polynomial trend identification (linear T + quadratic GDP)
    Approach QL: Quadratic response with LOESS trend identification
    Approach PJ: Piecewise quadratic response with joint OLS
    Approach PP: Piecewise quadratic response with polynomial trend identification
    Approach PL: Piecewise quadratic response with LOESS trend identification
    Approach SJ: Segmented linear response with joint OLS
    Approach SP: Segmented linear response with polynomial trend identification
    Approach SL: Segmented linear response with LOESS trend identification
    Approach TJ: Three-interval response with joint OLS
    Approach TP: Three-interval response with polynomial trend identification
    Approach TL: Three-interval response with LOESS trend identification
    Approach DJ: Persistence decay model with joint OLS
    Approach DP: Persistence decay model with polynomial trend identification
    Approach DL: Persistence decay model with LOESS trend identification
    Approach LL: Level effect model (h4=1) with LOESS trend identification
    Approach LJ: Level effect model (h4=1) with joint OLS
"""

import time

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
from .persistence import (
    compute_persistence_accumulators,
    compute_persistence_accumulators_at_T,
    compute_pre_first_year_correction,
    compute_T_linear_at_first_year,
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
    var_decomp: dict = None
    var_attrib: dict = None
    # Compatibility field for plotting that expects h1
    h1: float = 0.0        # Linear term (always 0 for piecewise model)
    h1_se: float = 0.0     # SE for h1 (always 0)
    # Three-interval (T approach) specific fields
    T_crit_low: float = None    # Lower critical temperature (start of transition zone)
    T_crit_high: float = None   # Upper critical temperature (end of transition zone)


@dataclass
class FitResultApproach4:
    """Container for Approach DL (persistence decay) results.

    Model: h_conv(T(t)) = h(T(t)) - h4 * sum_{k=1}^{n} (1-h4)^{k-1} * h(T(t-k))

    where h(T) = h1*T + h2*T^2 and n = t - first_year.

    Using accumulators for efficient computation:
    - A_T(t) = T(t) + (1-h4) * A_T(t-1), with A_T(first_year) = T(first_year)
    - A_T2(t) = T^2(t) + (1-h4) * A_T2(t-1), with A_T2(first_year) = T^2(first_year)

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

    Uses normal equations (X'X)^-1 X'y for speed (~20x faster than lstsq).
    """
    # Solve using normal equations (much faster than lstsq for large matrices)
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

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

    # Covariance matrix of beta (reuse XtX)
    XtX_inv = linalg.inv(XtX)
    cov_matrix = sigma_squared * XtX_inv

    return beta, residuals, sigma_squared, cov_matrix


def fit_ols_weighted(y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> tuple:
    """Fit weighted OLS regression: minimize sum(w_i * (y_i - X_i @ beta)^2).

    The weighted least squares solution is:
        beta = (X'WX)^(-1) X'Wy

    where W = diag(weights).

    Uses SVD-based least squares to handle rank-deficient matrices (e.g., when
    some years have zero weight in bootstrap).

    Args:
        y: Target values, shape (n,)
        X: Design matrix, shape (n, p)
        weights: Observation weights, shape (n,). Higher weight = more influence.

    Returns:
        Tuple of (beta, residuals, sigma_squared, cov_matrix)
        - beta: Coefficients, shape (p,)
        - residuals: y - X @ beta, shape (n,)
        - sigma_squared: Weighted residual variance
        - cov_matrix: Covariance matrix of beta, shape (p, p)
    """
    # Replace NaN values with 0 for zero-weight observations
    # This is safe because 0-weight observations don't contribute to the fit
    # NaN can occur when year_means[yr] is NaN for unsampled years in bootstrap
    y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
    X_clean = np.where(np.isnan(X) & (weights[:, np.newaxis] == 0), 0, X)

    # Transform to standard OLS by weighting: sqrt(W) @ X, sqrt(W) @ y
    sqrt_W = np.sqrt(weights)
    X_weighted = X_clean * sqrt_W[:, np.newaxis]  # Scale each row by sqrt(w_i)
    y_weighted = y_clean * sqrt_W

    # Solve weighted least squares using normal equations (much faster than lstsq)
    # beta = (X'WX)^+ X'Wy (using pseudoinverse to handle rank-deficient matrices)
    XtW = X_clean.T * weights  # Shape (p, n)
    XtWX = XtW @ X_clean  # Shape (p, p)
    XtWy = XtW @ y_clean  # Shape (p,)

    # Use pseudoinverse to handle rank-deficient matrices (e.g., zero-weight years)
    XtWX_pinv = np.linalg.pinv(XtWX)
    beta = XtWX_pinv @ XtWy

    # Compute residuals (use cleaned versions to avoid NaN propagation)
    y_pred = X_clean @ beta
    residuals = y_clean - y_pred

    # Degrees of freedom: effective sample size - rank of X'WX
    # For WLS, use sum of weights as effective sample size
    n_eff = np.sum(weights)
    rank = np.linalg.matrix_rank(XtWX)
    df = n_eff - rank

    # Weighted residual variance: sum(w_i * e_i^2) / df
    sse_weighted = np.sum(weights * residuals ** 2)
    sigma_squared = sse_weighted / df if df > 0 else np.nan

    # Covariance matrix of beta: sigma^2 * (X'WX)^+
    cov_matrix = sigma_squared * XtWX_pinv

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


def fit_ApproachQP_precomputed_k(
    data: AnalysisData, trends: CountryTrends, year_means: dict,
    weights: np.ndarray = None
) -> FitResult:
    """Approach 1: Pre-computed k[t] with linear temp + quadratic GDP detrending.

    Like Approach 1 but with precomputed k:
    1. k[t] = mean(dy_i[t]) is computed first
    2. Country trends j_i(t) = y0 + y1*t + y2*t² are fit to (dy_i[t] - k[t])
    3. Temperature is detrended with linear trend: T* = T - (T0 + T1*t)
    4. Final regression: (dy_i[t] - k[t]) - j_i[t] = h1*T* + h2*T*²

    Note: Linear temperature detrending + quadratic GDP detrending.

    Args:
        data: AnalysisData object
        trends: CountryTrends with polynomial trends
        year_means: Pre-computed k[t] = mean(dy_i[t])
        weights: Optional observation weights for weighted least squares
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

    # Fit OLS (weighted if weights provided)
    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # Just h1 and h2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

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

    # Compute RMS of h(T) - climate response to actual temperature

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
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    # Compute ε as remainder: ε = Δy - (Δu + v + j_adjusted + k) for exact decomposition
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="Approach QP: Quadratic (Polynomial Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachQL_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict,
    weights: np.ndarray = None
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
        weights: Optional observation weights for weighted least squares

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

    # Fit OLS (weighted if weights provided)
    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics
    n_params = 2  # Just h1 and h2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

    # Approach 6: T_trend = T_loess, j_trend = y_loess (LOESS smoothed)
    T_trend = trends_loess.T_loess
    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Compute RMS of h(T) - climate response to actual temperature

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
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    # Compute ε as remainder: ε = Δy - (Δu + v + j_adjusted + k) for exact decomposition
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="Approach QL: Quadratic (LOESS Detrending)",
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


def fit_ApproachPL_piecewise(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
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
        weights: Optional observation weights for weighted least squares

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

        # Solve OLS: min ||y - X @ [h2_low, h2_high]||^2 (weighted if weights provided)
        try:
            if weights is not None:
                # Zero out NaN values where weights are 0 (unsampled years in bootstrap)
                # NaN * 0 = NaN (IEEE 754), which would poison lstsq
                y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
                # Weighted least squares (use lstsq for numerical stability)
                sqrt_W = np.sqrt(weights)
                X_w = X * sqrt_W[:, np.newaxis]
                y_w = y_clean * sqrt_W
                beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred = X @ beta_ols
                sse = np.sum(weights * (y_clean - y_pred) ** 2)
            else:
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

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_low = beta_ols[0]
    h2_high = beta_ols[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

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

    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values using h(T) - h(T_trend) formulation
    h_values = h2_low * X1 + h2_high * X2

    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    # Compute RMS of h(T) - h(T_trend) - climate response to temperature fluctuations

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
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach PL: Piecewise (LOESS Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachSL_segmented(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach SL: Segmented linear temperature response with LOESS detrending.

    Model: h(T) = h2 * (T - T_opt)  if T <= T_opt
           h(T) = h4 * (T - T_opt)  if T > T_opt

    This model:
    - Has an optimum at T_opt where h(T_opt) = 0
    - Allows different slopes above vs below T_opt (V-shaped response)
    - Uses 1D optimization over T_opt with inner 2-column OLS for h2, h4

    For fitting we use the h(T) - h(T_trend) formulation.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2 (slope below), h4 (slope above), and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    T = data.temp
    T_trend = trends_loess.T_loess

    def segmented_linear(T_vals, T_opt_val):
        """Compute segmented linear: different slopes above/below T_opt.

        Returns two columns: one for low (T <= T_opt) and one for high (T > T_opt).
        """
        low_col = np.where(T_vals <= T_opt_val, T_vals - T_opt_val, 0.0)
        high_col = np.where(T_vals > T_opt_val, T_vals - T_opt_val, 0.0)
        return low_col, high_col

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving inner 2-column OLS for h2, h4."""
        low_T, high_T = segmented_linear(T, T_opt_val)
        low_trend, high_trend = segmented_linear(T_trend, T_opt_val)

        X1 = low_T - low_trend
        X2 = high_T - high_trend
        X = np.column_stack([X1, X2])

        try:
            if weights is not None:
                y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
                sqrt_W = np.sqrt(weights)
                X_w = X * sqrt_W[:, np.newaxis]
                y_w = y_clean * sqrt_W
                beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred = X @ beta_ols
                sse = np.sum(weights * (y_clean - y_pred) ** 2)
            else:
                beta_ols, _, _, _ = linalg.lstsq(X, y)
                y_pred = X @ beta_ols
                sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    x0 = 15.0
    result = minimize(
        lambda x: compute_sse_for_T_opt(x[0]),
        x0=[x0],
        bounds=[T_opt_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T_opt_opt = result.x[0]

    # Re-fit at optimal T_opt
    low_T, high_T = segmented_linear(T, T_opt_opt)
    low_trend, high_trend = segmented_linear(T_trend, T_opt_opt)
    X1 = low_T - low_trend
    X2 = high_T - high_trend
    X_opt = np.column_stack([X1, X2])

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_low = beta_ols[0]
    h2_high = beta_ols[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        data.n_obs,
        n_params=3
    )

    k = dict(year_means)
    n_params = 3
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    h_values = h2_low * X1 + h2_high * X2
    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    Delta_u = h_values
    v = h_of_T_trend
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach SL: Segmented Linear (LOESS Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachDL_persistence_decay(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    h4_bounds: tuple = (0.0, 1.0),
    weights: np.ndarray = None,
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
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach4 with h1, h2, h4, T_opt, and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t] (same as Approach QL)
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

        # Solve OLS: min ||y - X @ [h1, h2]||^2 (weighted if weights provided)
        if weights is not None:
            # Zero out NaN values where weights are 0 (unsampled years in bootstrap)
            # NaN * 0 = NaN (IEEE 754), which would poison lstsq
            y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
            # Weighted least squares (use lstsq for numerical stability)
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y_clean * sqrt_W
            beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y_clean - y_pred) ** 2)
        else:
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

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

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

    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values using h_conv formulation
    h_conv_values = h1 * X1 + h2 * X2

    h_of_T_trend = h1 * T_trend + h2 * T_trend ** 2

    # Compute RMS of h_conv

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Delta_u = h_conv(T) - h_conv(T_trend), v = h_conv(T_trend)
    # So Delta_u + v = h_conv(T) = total persistence-weighted climate response
    Delta_u = h_conv_values  # h_conv(T) - h_conv(T_trend)
    # Compute h_conv at trend temperature (persistence-weighted baseline)
    h_conv_T_trend = h1 * (T_trend - h4_opt * A_T_trend_lag - correction_T_trend) \
                   + h2 * (T_trend**2 - h4_opt * A_T2_trend_lag - correction_T2_trend)
    v = h_conv_T_trend
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach4(
        approach="Approach DL: Decay (LOESS Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachLL_first_difference(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    weights: np.ndarray = None,
) -> FitResultApproach4:
    """Approach LL: Level effect model with LOESS detrending.

    This is Approach DL with h4 fixed at 1.0, which means:
    h_conv(T(t)) = h(T(t)) - h(T(t-1))

    The climate effect depends on the level of temperature, not just changes.

    With h4=1, the accumulator decay is (1-h4)=0, so A_T(t) = T(t) (no memory)
    and A_T_lag(t) = T(t-1). The pre-first-year correction vanishes since
    (1-h4)^k = 0 for k >= 1.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach4 with h1, h2, h4=1.0 (fixed), and standard errors
    """
    h4_val = 1.0

    # Compute dependent variable: dy - k[t] - j_i[t] (same as Approach QL)
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    T = data.temp
    T_trend = trends_loess.T_loess

    # Compute accumulators with h4=1 (decay=0, so only previous value matters)
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
    A_T_trend_lag, A_T2_trend_lag = compute_persistence_accumulators_at_T(
        data, h4_val, T_trend
    )
    # Pre-first-year corrections vanish at h4=1 since (1-h4)^k = 0
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T)
    correction_T_trend, correction_T2_trend = compute_pre_first_year_correction(
        data, h4_val, T_trend
    )

    # Modified regressors with detrending
    X1 = (T - h4_val * A_T_lag - correction_T) - (T_trend - h4_val * A_T_trend_lag - correction_T_trend)
    X2 = (T**2 - h4_val * A_T2_lag - correction_T2) - (T_trend**2 - h4_val * A_T2_trend_lag - correction_T2_trend)
    X_opt = np.column_stack([X1, X2])

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Year effects are pre-computed year means
    k = dict(year_means)

    # Fit statistics (only 2 free params: h1, h2; h4 is fixed)
    n_params = 2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # Climate response values using h_conv formulation
    h_conv_values = h1 * X1 + h2 * X2

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution
    Delta_u = h_conv_values
    h_conv_T_trend = h1 * (T_trend - h4_val * A_T_trend_lag - correction_T_trend) \
                   + h2 * (T_trend**2 - h4_val * A_T2_trend_lag - correction_T2_trend)
    v = h_conv_T_trend
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach4(
        approach="Approach LL: Level Effect (LOESS Detrending)",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        h4=h4_val,
        h4_se=0.0,  # h4 is fixed, not estimated
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachPJ_piecewise_conjoined(
    data: AnalysisData,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach 5: Piecewise quadratic with full OLS for j_i(t) and k(t).

    Combines:
    - Piecewise quadratic climate response (like Approach PL)
    - Full OLS estimation of country trends and year effects (like Approach QJ)

    Model: Δy_i(t) = h2_low*(T-T_opt)² [T≤T_opt] + h2_high*(T-T_opt)² [T>T_opt]
                     + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    Uses 1D optimization over T_opt with inner OLS solving for all other parameters.

    Design matrix structure:
    - Column 0: h2_low column - (T-T_opt)² where T <= T_opt, else 0
    - Column 1: h2_high column - (T-T_opt)² where T > T_opt, else 0
    - Columns 2 to 2+3*(n_countries-1)-1: Country quadratic trends (skip country 0)
    - Remaining columns: Year dummies (only years with non-zero weight)

    Args:
        data: AnalysisData object
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2 (below), h4 (above), and standard errors
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years
    unique_years = sorted(set(data.year))

    # When weights provided, only include years with non-zero total weight
    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters:
    # - 2 for h2_low, h2_high
    # - 3 * (n_countries - 1) for j terms (first country is reference)
    # - n_active_years for k_t terms (only years with non-zero weight)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
    n_total_params = 2 + n_j_params + n_k_params

    # Pre-compute constant parts of design matrix (country trends and year effects)
    X_base = np.zeros((n_obs, n_total_params))

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 2 + 3 * (c - 1)
            X_base[i, col_base] = 1.0        # j0[c]
            X_base[i, col_base + 1] = t      # j1[c]
            X_base[i, col_base + 2] = t * t  # j2[c]

    # Year fixed effects (only active years)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X_base[i, k_col_start + yr_idx] = 1.0

    T = data.temp
    y = data.growth_pcGDP

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving full OLS."""
        # Piecewise quadratic columns
        low_col = np.where(T <= T_opt_val, (T - T_opt_val) ** 2, 0.0)
        high_col = np.where(T > T_opt_val, (T - T_opt_val) ** 2, 0.0)

        # Build design matrix
        X = X_base.copy()
        X[:, 0] = low_col
        X[:, 1] = high_col

        # Solve OLS using lstsq for numerical stability with rank-deficient matrices
        # (can happen during bootstrap with year sampling)
        if weights is not None:
            # Weighted least squares
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y * sqrt_W
            XTX = X_w.T @ X_w
            XTy = X_w.T @ y_w
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y - y_pred) ** 2)
        else:
            XTX = X.T @ X
            XTy = X.T @ y
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
        return sse

    # 1D optimization: grid search then Brent's method
    T_opt_grid = np.linspace(T_opt_bounds[0], T_opt_bounds[1], 31)
    sse_grid = [compute_sse_for_T_opt(T_val) for T_val in T_opt_grid]
    best_grid_idx = np.argmin(sse_grid)

    search_lo = T_opt_grid[max(0, best_grid_idx - 1)]
    search_hi = T_opt_grid[min(len(T_opt_grid) - 1, best_grid_idx + 1)]

    result = minimize_scalar(
        compute_sse_for_T_opt,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    T_opt_opt = result.x

    # Re-fit at optimal T_opt to get coefficients and statistics
    low_col = np.where(T <= T_opt_opt, (T - T_opt_opt) ** 2, 0.0)
    high_col = np.where(T > T_opt_opt, (T - T_opt_opt) ** 2, 0.0)

    X_opt = X_base.copy()
    X_opt[:, 0] = low_col
    X_opt[:, 1] = high_col

    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h2_low = beta[0]
    h2_high = beta[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

    # Compute SE for T_opt using numerical Hessian
    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        n_obs,
        n_params=3  # h2_low, h2_high, T_opt are the climate parameters
    )

    # Extract year fixed effects (NaN for inactive years)
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    # Fit statistics
    n_params = 3  # Climate response: h2_low, h2_high, T_opt
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_total_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, y)

    # Compute j_trend and k_values for diagnostics
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    # Climate response values h(T)
    h_values = h2_low * low_col + h2_high * high_col

    # since there's no separate T_trend. Use T as T_trend (like Approach QJ).
    T_trend = T
    low_trend = np.where(T_trend <= T_opt_opt, (T_trend - T_opt_opt) ** 2, 0.0)
    high_trend = np.where(T_trend > T_opt_opt, (T_trend - T_opt_opt) ** 2, 0.0)
    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    # Compute variance decomposition
    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, y, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # For conjoined approach: Delta_u = 0, v = h(T)
    Delta_u = np.zeros(n_obs)
    v = h_values
    epsilon = y - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, y)

    return FitResultApproach8(
        approach="Approach PJ: Piecewise (Joint OLS)",
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
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachSJ_segmented_conjoined(
    data: AnalysisData,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach SJ: Segmented linear response with full OLS for j_i(t) and k(t).

    Combines:
    - Segmented linear climate response (like Approach SL but linear not quadratic)
    - Full OLS estimation of country trends and year effects (like Approach QJ)

    Model: Δy_i(t) = h2*(T-T_opt) [T≤T_opt] + h4*(T-T_opt) [T>T_opt]
                     + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    Uses 1D optimization over T_opt with inner OLS solving for all other parameters.

    Args:
        data: AnalysisData object
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2 (slope below), h4 (slope above), and standard errors
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    unique_years = sorted(set(data.year))

    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
    n_total_params = 2 + n_j_params + n_k_params

    # Pre-compute constant parts of design matrix
    X_base = np.zeros((n_obs, n_total_params))

    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 2 + 3 * (c - 1)
            X_base[i, col_base] = 1.0
            X_base[i, col_base + 1] = t
            X_base[i, col_base + 2] = t * t

    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X_base[i, k_col_start + yr_idx] = 1.0

    T = data.temp
    y = data.growth_pcGDP

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving full OLS."""
        low_col = np.where(T <= T_opt_val, T - T_opt_val, 0.0)
        high_col = np.where(T > T_opt_val, T - T_opt_val, 0.0)

        X = X_base.copy()
        X[:, 0] = low_col
        X[:, 1] = high_col

        if weights is not None:
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y * sqrt_W
            XTX = X_w.T @ X_w
            XTy = X_w.T @ y_w
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y - y_pred) ** 2)
        else:
            XTX = X.T @ X
            XTy = X.T @ y
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
        return sse

    # 1D optimization: grid search then Brent's method
    T_opt_grid = np.linspace(T_opt_bounds[0], T_opt_bounds[1], 31)
    sse_grid = [compute_sse_for_T_opt(T_val) for T_val in T_opt_grid]
    best_grid_idx = np.argmin(sse_grid)

    search_lo = T_opt_grid[max(0, best_grid_idx - 1)]
    search_hi = T_opt_grid[min(len(T_opt_grid) - 1, best_grid_idx + 1)]

    result = minimize_scalar(
        compute_sse_for_T_opt,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    T_opt_opt = result.x

    # Re-fit at optimal T_opt
    low_col = np.where(T <= T_opt_opt, T - T_opt_opt, 0.0)
    high_col = np.where(T > T_opt_opt, T - T_opt_opt, 0.0)

    X_opt = X_base.copy()
    X_opt[:, 0] = low_col
    X_opt[:, 1] = high_col

    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h2_low = beta[0]
    h2_high = beta[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        n_obs,
        n_params=3
    )

    # Extract year fixed effects
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    n_params = 3
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_total_params)
    total_r_sq = compute_total_r_squared(residuals, y)

    # Compute j_trend and k_values for diagnostics
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    h_values = h2_low * low_col + h2_high * high_col

    # For conjoined approach: no separate T_trend
    T_trend = T
    low_trend = np.where(T_trend <= T_opt_opt, T_trend - T_opt_opt, 0.0)
    high_trend = np.where(T_trend > T_opt_opt, T_trend - T_opt_opt, 0.0)
    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, y, total_r_sq)

    # For conjoined approach: Delta_u = 0, v = h(T)
    Delta_u = np.zeros(n_obs)
    v = h_values
    epsilon = y - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, y)

    return FitResultApproach8(
        approach="Approach SJ: Segmented Linear (Joint OLS)",
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
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachDJ_persistence_conjoined(
    data: AnalysisData,
    h4_bounds: tuple = (0.0, 1.0),
    weights: np.ndarray = None,
) -> FitResultApproach4:
    """Approach 6: Persistence decay with full OLS for j_i(t) and k(t).

    Combines:
    - Persistence decay climate response (like Approach DL)
    - Full OLS estimation of country trends and year effects (like Approach QJ)

    Model: Δy_i(t) = h1*X1 + h2*X2 + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    where:
    - X1 = T - h4*A_T_lag - correction_T
    - X2 = T² - h4*A_T2_lag - correction_T2
    - A_T_lag, A_T2_lag are lagged persistence accumulators
    - correction terms account for assumed constant pre-history

    Pre-history assumption: Temperature before each country's first observation
    was constant at T_linear(first_year), where T_linear is computed from a
    linear OLS fit to each country's temperature time series. This provides
    a smoothed baseline that avoids noise from year-to-year variability.

    Uses 1D optimization over h4 with inner OLS solving for all other parameters.

    Design matrix structure:
    - Column 0: X1 (modified linear temperature)
    - Column 1: X2 (modified quadratic temperature)
    - Columns 2 to 2+3*(n_countries-1)-1: Country quadratic trends (skip country 0)
    - Remaining columns: Year dummies (all n_years)

    Args:
        data: AnalysisData object
        h4_bounds: Bounds for persistence decay parameter (default [0, 1])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach4 with h1, h2, h4, T_opt, and standard errors
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years
    unique_years = sorted(set(data.year))

    # When weights provided, only include years with non-zero total weight
    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters:
    # - 2 for h1, h2
    # - 3 * (n_countries - 1) for j terms
    # - n_active_years for k_t terms (only years with non-zero weight)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
    n_total_params = 2 + n_j_params + n_k_params

    # Pre-compute constant parts of design matrix (country trends and year effects)
    X_base = np.zeros((n_obs, n_total_params))

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 2 + 3 * (c - 1)
            X_base[i, col_base] = 1.0        # j0[c]
            X_base[i, col_base + 1] = t      # j1[c]
            X_base[i, col_base + 2] = t * t  # j2[c]

    # Year fixed effects (only active years)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X_base[i, k_col_start + yr_idx] = 1.0

    T = data.temp
    y = data.growth_pcGDP

    # Compute T_linear at first year for each country (for pre-history correction)
    # This provides a smoothed baseline instead of using noisy actual T(first_year)
    T_linear_first = compute_T_linear_at_first_year(data, weights)

    def compute_sse_for_h4(h4_val):
        """Compute SSE for given h4 by solving full OLS."""
        # Compute accumulators and corrections
        # Use T_linear_first for pre-history correction (smoothed baseline)
        A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
        correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T_linear_first)

        # Modified temperature regressors
        X1 = T - h4_val * A_T_lag - correction_T
        X2 = T**2 - h4_val * A_T2_lag - correction_T2

        # Build design matrix
        X = X_base.copy()
        X[:, 0] = X1
        X[:, 1] = X2

        # Solve OLS using lstsq for numerical stability with rank-deficient matrices
        # (can happen during bootstrap with year sampling)
        if weights is not None:
            # Weighted least squares
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y * sqrt_W
            XTX = X_w.T @ X_w
            XTy = X_w.T @ y_w
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y - y_pred) ** 2)
        else:
            XTX = X.T @ X
            XTy = X.T @ y
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
        return sse

    # 1D optimization: grid search then Brent's method
    h4_grid = np.linspace(h4_bounds[0], h4_bounds[1], 21)
    sse_grid = [compute_sse_for_h4(h4_val) for h4_val in h4_grid]
    best_grid_idx = np.argmin(sse_grid)

    search_lo = h4_grid[max(0, best_grid_idx - 1)]
    search_hi = h4_grid[min(len(h4_grid) - 1, best_grid_idx + 1)]

    result = minimize_scalar(
        compute_sse_for_h4,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    h4_opt = result.x

    # Re-fit at optimal h4 to get coefficients and statistics
    # Use T_linear_first for pre-history correction (consistent with optimization)
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_opt)
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_opt, T_linear_first)

    X1 = T - h4_opt * A_T_lag - correction_T
    X2 = T**2 - h4_opt * A_T2_lag - correction_T2

    X_opt = X_base.copy()
    X_opt[:, 0] = X1
    X_opt[:, 1] = X2

    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Compute SE for h4 using numerical Hessian
    h4_se = compute_1d_se_numerical(
        compute_sse_for_h4,
        h4_opt,
        h4_bounds,
        n_obs,
        n_params=3
    )

    # Extract year fixed effects (NaN for inactive years)
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    # Fit statistics
    n_params = 3  # Climate response: h1, h2, h4
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_total_params)

    # Total R²
    total_r_sq = compute_total_r_squared(residuals, y)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

    # Compute j_trend and k_values for diagnostics
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    # Climate response values (modified h_conv)
    h_conv_values = h1 * X1 + h2 * X2

    h_T_full = h1 * T + h2 * T ** 2

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, y, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # For joint approaches: Delta_u = full climate response, v = 0
    # This ensures epsilon = OLS residual and Cov(h, epsilon) ≈ 0
    Delta_u = h_conv_values  # Persistence-modified climate response
    v = np.zeros(n_obs)  # No separate baseline for joint approach
    epsilon = y - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, y)

    return FitResultApproach4(
        approach="Approach DJ: Decay (Joint OLS)",
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
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachLJ_level_effect_conjoined(
    data: AnalysisData,
    weights: np.ndarray = None,
) -> FitResultApproach4:
    """Approach LJ: Level effect model with joint OLS.

    This is Approach DJ with h4 fixed at 1.0, which means:
    h_conv(T(t)) = h(T(t)) - h(T(t-1))

    The climate effect depends on the level of temperature, not just changes.
    With h4=1, the accumulator decay is (1-h4)=0, so the pre-first-year
    correction vanishes since (1-h4)^k = 0 for k >= 1.

    Combines:
    - Level effect climate response (h4=1 fixed)
    - Full OLS estimation of country trends and year effects (like Approach QJ)

    Model: Δy_i(t) = h1*X1 + h2*X2 + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    where X1 = T(t) - T(t-1), X2 = T²(t) - T²(t-1) (first-differenced).

    Args:
        data: AnalysisData object
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach4 with h1, h2, h4=1.0 (fixed), and standard errors
    """
    h4_val = 1.0
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years
    unique_years = sorted(set(data.year))

    # When weights provided, only include years with non-zero total weight
    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters:
    # - 2 for h1, h2 (h4 is fixed at 1.0)
    # - 3 * (n_countries - 1) for j terms
    # - n_active_years for k_t terms
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
    n_total_params = 2 + n_j_params + n_k_params

    # Pre-compute design matrix (country trends and year effects)
    X = np.zeros((n_obs, n_total_params))

    # Country-specific time trends (skip country 0 as reference)
    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 2 + 3 * (c - 1)
            X[i, col_base] = 1.0        # j0[c]
            X[i, col_base + 1] = t      # j1[c]
            X[i, col_base + 2] = t * t  # j2[c]

    # Year fixed effects (only active years)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X[i, k_col_start + yr_idx] = 1.0

    T = data.temp
    y = data.growth_pcGDP

    # Compute T_linear at first year for pre-history correction
    T_linear_first = compute_T_linear_at_first_year(data, weights)

    # Compute accumulators with h4=1 and pre-history correction
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T_linear_first)

    # Modified temperature regressors
    X1 = T - h4_val * A_T_lag - correction_T
    X2 = T**2 - h4_val * A_T2_lag - correction_T2

    X[:, 0] = X1
    X[:, 1] = X2

    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X)

    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Extract year fixed effects (NaN for inactive years)
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    # Fit statistics (h4 is fixed, only h1 and h2 are free climate params)
    n_params = 2
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_total_params)

    # Total R²
    total_r_sq = compute_total_r_squared(residuals, y)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

    # Compute j_trend and k_values for diagnostics
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    # Climate response values (modified h_conv)
    h_conv_values = h1 * X1 + h2 * X2

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, y, total_r_sq)

    # Compute variance attribution
    Delta_u = h_conv_values
    v = np.zeros(n_obs)  # No separate baseline for joint approach
    epsilon = y - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, y)

    return FitResultApproach4(
        approach="Approach LJ: Level Effect (Joint OLS)",
        h1=h1,
        h2=h2,
        h1_se=h1_se,
        h2_se=h2_se,
        h4=h4_val,
        h4_se=0.0,  # h4 is fixed, not estimated
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        T_opt=T_opt,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachQJ_conjoined(data: AnalysisData, weights: np.ndarray = None) -> FitResult:
    """Approach 0: No pre-detrending, with country time trends and year fixed effects.

    Δy_i(t) = h1*T + h2*T² + j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    This estimates:
    - h1, h2: temperature response coefficients
    - j_{0,i}, j_{1,i}, j_{2,i}: country-specific quadratic time trends (countries i > 0)
    - k_t: year fixed effects (all years)

    For identifiability, we set j_{0,0} = j_{1,0} = j_{2,0} = 0 (first country is reference).
    This provides 3 constraints to pin down the arbitrary quadratic that could otherwise
    be added to all j_i(t) and subtracted from all k_t.

    Args:
        data: AnalysisData object
        weights: Optional observation weights for weighted least squares
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years
    unique_years = sorted(set(data.year))

    # When weights provided, only include years with non-zero total weight
    # (avoids singular design matrix from zero-weight year columns)
    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters:
    # - 2 for h1, h2
    # - 3 * (n_countries - 1) for j terms (first country is reference, j[0] = 0)
    # - n_active_years for k_t terms (only years with non-zero weight)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
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

    # Year fixed effects (only active years)
    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X[i, k_col_start + yr_idx] = 1.0

    # Fit OLS (weighted if weights provided)
    y = data.growth_pcGDP
    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # Extract coefficients
    h1 = beta[0]
    h2 = beta[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

    # Year fixed effects (store by actual year, NaN for inactive years)
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    # Fit statistics
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)

    # Total R² (variance explained in original dy)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    # Optimal temperature
    T_opt = compute_T_optimal(h1, h2)

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
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    # Compute RMS of h(T) - climate response to actual temperature

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
        approach="Approach QJ: Quadratic (Joint OLS)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachNJ_joint(data: AnalysisData, weights: np.ndarray = None) -> FitResult:
    """Null model: No climate response, joint OLS fit with country trends and year effects.

    Δy_i(t) = j_{0,i} + j_{1,i}*t + j_{2,i}*t² + k_t

    Like Approach 0 but without T and T² columns. Tests how much variance
    is explained by country trends and year effects alone.

    For identifiability, we set j_{0,0} = j_{1,0} = j_{2,0} = 0 (first country is reference).

    Args:
        data: AnalysisData object
        weights: Optional observation weights for weighted least squares
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    # Get unique years
    unique_years = sorted(set(data.year))

    # When weights provided, only include years with non-zero total weight
    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    # Number of parameters (no h1, h2):
    # - 3 * (n_countries - 1) for j terms (first country is reference, j[0] = 0)
    # - n_active_years for k_t terms (only years with non-zero weight)
    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
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

    # Year fixed effects (only active years)
    k_col_start = n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X[i, k_col_start + yr_idx] = 1.0

    # Fit OLS (weighted if weights provided)
    y = data.growth_pcGDP
    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X)

    # No climate response coefficients
    h1 = 0.0
    h2 = 0.0
    h1_se = 0.0
    h2_se = 0.0

    # Year fixed effects (store by actual year, NaN for inactive years)
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

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
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    # Variance decomposition (no h components)
    components = {'j': j_trend, 'k': k_values}
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Variance attribution (Δu=0, v=0 since no climate response)
    Delta_u = np.zeros(n_obs)
    v = np.zeros(n_obs)
    epsilon = data.growth_pcGDP - (j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, data.growth_pcGDP)

    return FitResult(
        approach="Approach NJ: No Climate Response (Joint)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachNP_precomputed_k(
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
        approach="Approach NP: No Climate Response (Precomputed k)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachNL_precomputed_k_loess(
    data: AnalysisData, trends_loess: CountryTrendsLoess, year_means: dict
) -> FitResult:
    """Null model: No climate response, precomputed k with LOESS country trends.

    LOESS version of Approach NP:
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
        approach="Approach NL: No Climate Response (LOESS Precomputed k)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachPP_piecewise_linear_detrend(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach 7: Piecewise quadratic response with linear T + quadratic GDP detrending.

    Combines:
    - Piecewise quadratic climate response (like Approach PL)
    - Linear temperature + quadratic GDP detrending with pre-computed k (like Approach QP)

    Model: h(T) = h2 * (T - T_opt)²  if T ≤ T_opt
           h(T) = h4 * (T - T_opt)²  if T > T_opt

    Detrending:
    - y = Δy - k[t] - j_i[t] where j_i[t] = y0 + y1*t + y2*t²
    - T_trend = T0 + T1*t (linear)

    Uses h(T) - h(T_trend) formulation with detrended piecewise regressors.

    Args:
        data: AnalysisData object
        trends: CountryTrends (with linear T trend: T0, T1; quadratic GDP: y0, y1, y2)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2 (below), h4 (above), and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    # Use raw temperature (NOT detrended) so T_opt represents actual optimal temperature
    T = data.temp

    # Compute T_trend using linear polynomial fit (T0 + T1*t)
    T_trend = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t

    def piecewise_quad(T_vals, T_opt_val):
        """Compute piecewise quadratic: different curvature above/below T_opt."""
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

        # Solve OLS: min ||y - X @ [h2_low, h2_high]||^2 (weighted if weights provided)
        try:
            if weights is not None:
                # Zero out NaN values where weights are 0 (unsampled years in bootstrap)
                # NaN * 0 = NaN (IEEE 754), which would poison lstsq
                y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
                # Weighted least squares (use lstsq for numerical stability)
                sqrt_W = np.sqrt(weights)
                X_w = X * sqrt_W[:, np.newaxis]
                y_w = y_clean * sqrt_W
                beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred = X @ beta_ols
                sse = np.sum(weights * (y_clean - y_pred) ** 2)
            else:
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

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_low = beta_ols[0]
    h2_high = beta_ols[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

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

    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]

    # Climate response values using h(T) - h(T_trend) formulation
    h_values = h2_low * X1 + h2_high * X2

    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    # Compute RMS of h(T) - h(T_trend) - climate response to temperature fluctuations

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
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach PP: Piecewise (Polynomial Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachSP_segmented_linear_detrend(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    T_opt_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach SP: Segmented linear response with linear T + quadratic GDP detrending.

    Combines:
    - Segmented linear climate response (like Approach SL)
    - Linear temperature + quadratic GDP detrending with pre-computed k (like Approach QP)

    Model: h(T) = h2 * (T - T_opt)  if T <= T_opt
           h(T) = h4 * (T - T_opt)  if T > T_opt

    Args:
        data: AnalysisData object
        trends: CountryTrends (with linear T trend: T0, T1; quadratic GDP: y0, y1, y2)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_opt_bounds: Bounds for optimal temperature (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2 (slope below), h4 (slope above), and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    T = data.temp

    # Compute T_trend using linear polynomial fit (T0 + T1*t)
    T_trend = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t

    def segmented_linear(T_vals, T_opt_val):
        """Compute segmented linear: different slopes above/below T_opt."""
        low_col = np.where(T_vals <= T_opt_val, T_vals - T_opt_val, 0.0)
        high_col = np.where(T_vals > T_opt_val, T_vals - T_opt_val, 0.0)
        return low_col, high_col

    def compute_sse_for_T_opt(T_opt_val):
        """Compute SSE for given T_opt by solving inner 2-column OLS for h2, h4."""
        low_T, high_T = segmented_linear(T, T_opt_val)
        low_trend, high_trend = segmented_linear(T_trend, T_opt_val)

        X1 = low_T - low_trend
        X2 = high_T - high_trend
        X = np.column_stack([X1, X2])

        try:
            if weights is not None:
                y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
                sqrt_W = np.sqrt(weights)
                X_w = X * sqrt_W[:, np.newaxis]
                y_w = y_clean * sqrt_W
                beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred = X @ beta_ols
                sse = np.sum(weights * (y_clean - y_pred) ** 2)
            else:
                beta_ols, _, _, _ = linalg.lstsq(X, y)
                y_pred = X @ beta_ols
                sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    x0 = 15.0
    result = minimize(
        lambda x: compute_sse_for_T_opt(x[0]),
        x0=[x0],
        bounds=[T_opt_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T_opt_opt = result.x[0]

    # Re-fit at optimal T_opt
    low_T, high_T = segmented_linear(T, T_opt_opt)
    low_trend, high_trend = segmented_linear(T_trend, T_opt_opt)
    X1 = low_T - low_trend
    X2 = high_T - high_trend
    X_opt = np.column_stack([X1, X2])

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2_low = beta_ols[0]
    h2_high = beta_ols[1]
    h2_low_se = np.sqrt(max(cov[0, 0], 0))
    h2_high_se = np.sqrt(max(cov[1, 1], 0))

    T_opt_se = compute_1d_se_numerical(
        compute_sse_for_T_opt,
        T_opt_opt,
        T_opt_bounds,
        data.n_obs,
        n_params=3
    )

    k = dict(year_means)
    n_params = 3
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]

    h_values = h2_low * X1 + h2_high * X2
    h_of_T_trend = h2_low * low_trend + h2_high * high_trend

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    Delta_u = h_values
    v = h_of_T_trend
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach SP: Segmented Linear (Polynomial Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def fit_ApproachDP_persistence_linear_detrend(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    h4_bounds: tuple = (0.0, 1.0),
    weights: np.ndarray = None,
) -> FitResultApproach4:
    """Approach 8: Persistence decay with linear T + quadratic GDP detrending.

    Combines:
    - Persistence decay climate response (like Approach DL)
    - Linear temperature + quadratic GDP detrending with pre-computed k (like Approach QP)

    Model: h_conv(T(t)) = h(T(t)) - h4 * sum_{k=1}^{n} (1-h4)^{k-1} * h(T(t-k))

    where h(T) = h1*T + h2*T^2.

    Detrending:
    - y = Δy - k[t] - j_i[t] where j_i[t] = y0 + y1*t + y2*t²
    - T_trend = T0 + T1*t (linear)

    Uses h(T) - h(T_trend) formulation with persistence accumulators and corrections
    computed using linear T_trend.

    Args:
        data: AnalysisData object
        trends: CountryTrends (with linear T trend: T0, T1; quadratic GDP: y0, y1, y2)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        h4_bounds: Bounds for persistence decay parameter (default [0, 1])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach4 with h1, h2, h4, T_opt, and standard errors
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    T = data.temp

    # Compute T_trend using linear polynomial fit (T0 + T1*t)
    T_trend = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t

    # Compute T_linear at first year for pre-history correction (same as Approach DJ)
    T_linear_first = compute_T_linear_at_first_year(data, weights)

    def compute_sse_for_h4(h4_val):
        """Compute SSE for given h4 by solving inner 2-column OLS for h1, h2."""
        # Compute accumulators for observed temperature
        A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_val)
        # Compute accumulators for trend temperature
        A_T_trend_lag, A_T2_trend_lag = compute_persistence_accumulators_at_T(
            data, h4_val, T_trend
        )
        # Compute pre-first-year correction using T_linear at first year
        correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_val, T_linear_first)
        # Correction for trend temperature (using T_trend value at first year)
        correction_T_trend, correction_T2_trend = compute_pre_first_year_correction(
            data, h4_val, T_trend
        )

        # Modified regressors with detrending and pre-first-year correction
        X1 = (T - h4_val * A_T_lag - correction_T) - (T_trend - h4_val * A_T_trend_lag - correction_T_trend)
        X2 = (T**2 - h4_val * A_T2_lag - correction_T2) - (T_trend**2 - h4_val * A_T2_trend_lag - correction_T2_trend)

        X = np.column_stack([X1, X2])

        # Solve OLS: min ||y - X @ [h1, h2]||^2 (weighted if weights provided)
        if weights is not None:
            # Zero out NaN values where weights are 0 (unsampled years in bootstrap)
            # NaN * 0 = NaN (IEEE 754), which would poison lstsq
            y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
            # Weighted least squares (use lstsq for numerical stability)
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y_clean * sqrt_W
            beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y_clean - y_pred) ** 2)
        else:
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
    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_opt, T_linear_first)
    correction_T_trend, correction_T2_trend = compute_pre_first_year_correction(
        data, h4_opt, T_trend
    )

    X1 = (T - h4_opt * A_T_lag - correction_T) - (T_trend - h4_opt * A_T_trend_lag - correction_T_trend)
    X2 = (T**2 - h4_opt * A_T2_lag - correction_T2) - (T_trend**2 - h4_opt * A_T2_trend_lag - correction_T2_trend)
    X_opt = np.column_stack([X1, X2])

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h1 = beta_ols[0]
    h2 = beta_ols[1]
    h1_se = np.sqrt(max(cov[0, 0], 0))
    h2_se = np.sqrt(max(cov[1, 1], 0))

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

    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]

    # Climate response values using h_conv formulation
    h_conv_values = h1 * X1 + h2 * X2

    h_of_T_trend = h1 * T_trend + h2 * T_trend ** 2

    # Compute RMS of h_conv

    # Compute variance decomposition
    components = {
        'h_T': h_conv_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    # Compute variance attribution (5-component with covariance allocation)
    # Delta_u = h_conv(T) - h_conv(T_trend), v = h_conv(T_trend)
    # So Delta_u + v = h_conv(T) = total persistence-weighted climate response
    Delta_u = h_conv_values  # h_conv(T) - h_conv(T_trend)
    # Compute h_conv at trend temperature (persistence-weighted baseline)
    h_conv_T_trend = h1 * (T_trend - h4_opt * A_T_trend_lag - correction_T_trend) \
                   + h2 * (T_trend**2 - h4_opt * A_T2_trend_lag - correction_T2_trend)
    v = h_conv_T_trend
    # Adjust j_trend by subtracting climate response to temperature trend
    # This makes j_trend_adjusted comparable to Approach J's j (both net of climate trend response)
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach4(
        approach="Approach DP: Decay (Polynomial Detrending)",
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
        var_decomp=var_decomp,
        var_attrib=var_attrib,
    )


def three_interval_basis(T_vals, T_crit_low, delta_T_crit):
    """Compute three-interval basis functions f_low and f_high.

    The derivative transitions linearly from h2 (below T_lo) to h4 (above T_hi),
    where T_lo = T_crit_low and T_hi = T_lo + delta_T_crit.

    f_low(T):
      T <= T_lo:           T - T_lo
      T_lo < T < T_hi:    (T - T_lo) - (T - T_lo)^2 / (2*dT)
      T >= T_hi:           dT / 2

    f_high(T):
      T <= T_lo:           0
      T_lo < T < T_hi:    (T - T_lo)^2 / (2*dT)
      T >= T_hi:           T - T_lo - dT/2

    Args:
        T_vals: Temperature array
        T_crit_low: Lower critical temperature
        delta_T_crit: Width of transition zone (>= 0)

    Returns:
        Tuple of (f_low, f_high) arrays
    """
    T_lo = T_crit_low
    dT = delta_T_crit
    T_hi = T_lo + dT

    d = T_vals - T_lo  # shifted temperature

    below = T_vals <= T_lo
    above = T_vals >= T_hi
    middle = ~below & ~above

    f_low = np.where(below, d, np.where(above, dT / 2, 0.0))
    f_high = np.where(below, 0.0, np.where(above, d - dT / 2, 0.0))

    # Middle region: use np.where to avoid division by zero when dT=0
    # When dT=0, middle is empty so these values are never used
    if dT > 0:
        d_mid = np.where(middle, d, 0.0)
        f_low = np.where(middle, d_mid - d_mid**2 / (2 * dT), f_low)
        f_high = np.where(middle, d_mid**2 / (2 * dT), f_high)

    return f_low, f_high


def _optimize_three_interval(compute_sse_for_T0T1, T_bounds=(0.0, 30.0)):
    """Shared 2D optimization over (T0, T1) for three-interval approaches.

    Optimizes over (T0, T1) both in T_bounds. The SSE function should accept
    (T0, T1) and internally compute T_crit_low = min(T0, T1),
    delta_T_crit = |T1 - T0|.

    Returns:
        Tuple of (T_crit_low, delta_T_crit) at optimum
    """
    # Grid search: 12×12 over T0 × T1 (only upper triangle needed due to symmetry,
    # but searching both is simpler and the grid is small)
    T_grid = np.linspace(T_bounds[0], T_bounds[1], 12)
    best_sse = np.inf
    best_T0T1 = np.array([15.0, 20.0])
    for t0 in T_grid:
        for t1 in T_grid:
            sse = compute_sse_for_T0T1(np.array([t0, t1]))
            if sse < best_sse:
                best_sse = sse
                best_T0T1 = np.array([t0, t1])

    # Refine with L-BFGS-B
    result = minimize(
        compute_sse_for_T0T1,
        x0=best_T0T1,
        bounds=[T_bounds, T_bounds],
        method='L-BFGS-B',
        options={'ftol': 1e-8}
    )
    T0_opt, T1_opt = result.x
    T_crit_low = min(T0_opt, T1_opt)
    T_crit_high = max(T0_opt, T1_opt)
    return T_crit_low, T_crit_high


def _compute_g2(T, T_lo, T_hi):
    """Compute g2 basis: T² in middle, tangent-line extensions outside.

    g2(T) = T²              for T_lo <= T <= T_hi
    g2(T) = 2*T_lo*T - T_lo²  for T < T_lo
    g2(T) = 2*T_hi*T - T_hi²  for T > T_hi

    This is C1-continuous everywhere. In the middle, dg2/dT = 2T.
    Outside, dg2/dT is constant (2*T_lo or 2*T_hi).
    """
    return np.where(
        T < T_lo, 2 * T_lo * T - T_lo**2,
        np.where(T > T_hi, 2 * T_hi * T - T_hi**2, T**2)
    )


def _derive_T_opt(h2, h4, T_crit_low, T_crit_high):
    """Derive T_opt from three-interval parameters."""
    delta = T_crit_high - T_crit_low
    if h2 * h4 < 0:
        if delta > 0:
            return T_crit_low + h2 * delta / (h2 - h4)
        else:
            return T_crit_low
    return np.nan


def fit_ApproachTL_three_interval(
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    year_means: dict,
    T_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach TL: Three-interval response with LOESS detrending.

    The derivative dh/dT transitions linearly between T_crit_low and T_crit_high,
    creating three intervals: linear below, quadratic in the middle, linear above.

    Optimization is over (T0, T1) both in T_bounds, with T_crit_low = min(T0, T1)
    and delta_T_crit = |T1 - T0|. This avoids boundary issues when delta_T_crit = 0.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess (with LOESS trends)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_bounds: Bounds for T0 and T1 (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2, h4, T_crit_low, delta_T_crit
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    T = data.temp
    T_trend = trends_loess.T_loess

    def compute_sse_for_T0T1(params):
        """Compute SSE for given (T0, T1), using middle-only quadratic OLS.

        Uses g2 basis (T² in middle, tangent extensions outside).
        OLS on middle observations only, SSE evaluated on all data.
        If slopes at both boundaries have the same sign, returns inf.
        Falls back to full-data f_low/f_high OLS if < 3 middle observations.
        """
        T_lo = min(params[0], params[1])
        T_hi = max(params[0], params[1])

        # Count middle observations
        middle_mask = (T >= T_lo) & (T <= T_hi)
        n_middle = np.sum(middle_mask)

        if n_middle < 3:
            return np.inf

        # g2 basis
        g2_T = _compute_g2(T, T_lo, T_hi)
        g2_Ttrend = _compute_g2(T_trend, T_lo, T_hi)
        X1_all = T - T_trend
        X2_all = g2_T - g2_Ttrend
        X_all = np.column_stack([X1_all, X2_all])

        # Middle-only OLS
        X_mid = X_all[middle_mask]
        y_mid = y[middle_mask]

        try:
            if weights is not None:
                w_mid = weights[middle_mask]
                y_mid_clean = np.where(np.isnan(y_mid) & (w_mid == 0), 0, y_mid)
                sqrt_w = np.sqrt(w_mid)
                beta, _, _, _ = np.linalg.lstsq(
                    X_mid * sqrt_w[:, np.newaxis], y_mid_clean * sqrt_w, rcond=None
                )
            else:
                beta, _, _, _ = linalg.lstsq(X_mid, y_mid)
        except Exception:
            return np.inf

        h1a, h2a = beta[0], beta[1]

        # Check sign constraint: slopes at boundaries must have opposite signs
        slope_low = h1a + 2 * h2a * T_lo
        slope_high = h1a + 2 * h2a * T_hi
        if slope_low * slope_high > 0:
            return np.inf

        # SSE on all data
        y_pred = X_all @ beta
        if weights is not None:
            y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
            return np.sum(weights * (y_clean - y_pred) ** 2)
        return np.sum((y - y_pred) ** 2)

    T_crit_low_opt, T_crit_high_opt = _optimize_three_interval(compute_sse_for_T0T1, T_bounds)

    # Re-fit at optimal params using g2 basis with middle-only OLS
    middle_mask = (T >= T_crit_low_opt) & (T <= T_crit_high_opt)
    g2_T_mid = _compute_g2(T[middle_mask], T_crit_low_opt, T_crit_high_opt)
    g2_Ttrend_mid = _compute_g2(T_trend[middle_mask], T_crit_low_opt, T_crit_high_opt)
    X_mid = np.column_stack([
        T[middle_mask] - T_trend[middle_mask],
        g2_T_mid - g2_Ttrend_mid,
    ])
    y_mid = y[middle_mask]

    # Use lstsq for middle-only OLS (may be rank-deficient if interval is narrow)
    if weights is not None:
        w_mid = weights[middle_mask]
        y_mid_clean = np.where(np.isnan(y_mid) & (w_mid == 0), 0, y_mid)
        sqrt_w = np.sqrt(w_mid)
        beta_ols, _, _, _ = np.linalg.lstsq(
            X_mid * sqrt_w[:, np.newaxis], y_mid_clean * sqrt_w, rcond=None
        )
    else:
        beta_ols, _, _, _ = linalg.lstsq(X_mid, y_mid)

    h1a = beta_ols[0]
    h2a = beta_ols[1]

    # Derive slopes at boundaries and T_opt
    h2 = h1a + 2 * h2a * T_crit_low_opt   # slope below T_crit_low
    h4 = h1a + 2 * h2a * T_crit_high_opt   # slope above T_crit_high
    h2_se = np.nan  # SE not straightforward with middle-only OLS
    h4_se = np.nan

    # T_opt = where dh/dT = 0 in the quadratic region
    T_opt = -h1a / (2 * h2a) if abs(h2a) > 1e-15 else np.nan

    # Re-compute using three_interval_basis for compatibility with output/plotting
    delta = T_crit_high_opt - T_crit_low_opt
    f_low_T, f_high_T = three_interval_basis(T, T_crit_low_opt, delta)
    f_low_trend, f_high_trend = three_interval_basis(T_trend, T_crit_low_opt, delta)
    X1 = f_low_T - f_low_trend
    X2 = f_high_T - f_high_trend
    y_pred_all = h2 * X1 + h4 * X2
    y_for_resid = np.where(np.isnan(y) & (weights == 0), 0, y) if weights is not None else y
    residuals = y_for_resid - y_pred_all

    k = dict(year_means)
    n_params = 4  # h1a, h2a, T_crit_low, T_crit_high
    r_sq, adj_r_sq, rmse = compute_fit_stats(y_for_resid, residuals, n_params)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    j_trend = trends_loess.y_loess
    k_values = np.array([year_means[data.year[i]] for i in range(data.n_obs)])

    # h(T) - h(T_trend) using three_interval_basis
    h_values = h2 * X1 + h4 * X2
    h_of_T_trend = h2 * f_low_trend + h4 * f_high_trend

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    Delta_u = h_values
    v = h_of_T_trend
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach TL: Three-Interval (LOESS Detrending)",
        h2=h2,
        h2_se=h2_se,
        h4=h4,
        h4_se=h4_se,
        T_opt=T_opt,
        T_opt_se=np.nan,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
        T_crit_low=T_crit_low_opt,
        T_crit_high=T_crit_high_opt,
    )


def fit_ApproachTJ_three_interval_conjoined(
    data: AnalysisData,
    T_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach TJ: Three-interval response with full OLS for j_i(t) and k(t).

    Optimization is over (T0, T1) both in T_bounds, with T_crit_low = min(T0, T1)
    and delta_T_crit = |T1 - T0|.

    Args:
        data: AnalysisData object
        T_bounds: Bounds for T0 and T1 (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2, h4, T_crit_low, delta_T_crit
    """
    n_obs = data.n_obs
    n_countries = data.n_countries

    unique_years = sorted(set(data.year))

    if weights is not None:
        year_weights = {}
        for i in range(n_obs):
            yr = data.year[i]
            year_weights[yr] = year_weights.get(yr, 0) + weights[i]
        active_years = [yr for yr in unique_years if year_weights.get(yr, 0) > 0]
    else:
        active_years = unique_years

    active_year_to_idx = {y: i for i, y in enumerate(active_years)}
    n_active_years = len(active_years)

    n_j_params = 3 * (n_countries - 1)
    n_k_params = n_active_years
    n_total_params = 2 + n_j_params + n_k_params

    # Pre-compute constant parts of design matrix
    X_base = np.zeros((n_obs, n_total_params))

    for i in range(n_obs):
        c = data.country_idx[i]
        if c > 0:
            t = data.time[i]
            col_base = 2 + 3 * (c - 1)
            X_base[i, col_base] = 1.0
            X_base[i, col_base + 1] = t
            X_base[i, col_base + 2] = t * t

    k_col_start = 2 + n_j_params
    for i in range(n_obs):
        yr = data.year[i]
        if yr in active_year_to_idx:
            yr_idx = active_year_to_idx[yr]
            X_base[i, k_col_start + yr_idx] = 1.0

    T = data.temp
    y = data.growth_pcGDP

    def compute_sse_for_T0T1(params):
        """Compute SSE for given (T0, T1), using T_crit_low=min, delta=|T1-T0|."""
        T_crit_low_val = min(params[0], params[1])
        delta_T_crit_val = abs(params[1] - params[0])
        f_low, f_high = three_interval_basis(T, T_crit_low_val, delta_T_crit_val)

        X = X_base.copy()
        X[:, 0] = f_low
        X[:, 1] = f_high

        if weights is not None:
            y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
            sqrt_W = np.sqrt(weights)
            X_w = X * sqrt_W[:, np.newaxis]
            y_w = y_clean * sqrt_W
            XTX = X_w.T @ X_w
            XTy = X_w.T @ y_w
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum(weights * (y_clean - y_pred) ** 2)
        else:
            XTX = X.T @ X
            XTy = X.T @ y
            beta_ols, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
            y_pred = X @ beta_ols
            sse = np.sum((y - y_pred) ** 2)
        return sse

    T_crit_low_opt, T_crit_high_opt = _optimize_three_interval(compute_sse_for_T0T1, T_bounds)
    delta_T_crit = T_crit_high_opt - T_crit_low_opt

    # Re-fit at optimal params
    f_low, f_high = three_interval_basis(T, T_crit_low_opt, delta_T_crit)
    X_opt = X_base.copy()
    X_opt[:, 0] = f_low
    X_opt[:, 1] = f_high

    if weights is not None:
        beta, residuals, sigma_sq, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta, residuals, sigma_sq, cov = fit_ols(y, X_opt)

    h2 = beta[0]
    h4 = beta[1]
    h2_se = np.sqrt(max(cov[0, 0], 0))
    h4_se = np.sqrt(max(cov[1, 1], 0))

    T_opt = _derive_T_opt(h2, h4, T_crit_low_opt, T_crit_high_opt)

    # Extract year fixed effects
    k = {}
    for yr in unique_years:
        if yr in active_year_to_idx:
            k[yr] = beta[k_col_start + active_year_to_idx[yr]]
        else:
            k[yr] = np.nan

    n_params = 4
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_total_params)
    total_r_sq = compute_total_r_squared(residuals, y)

    # Compute j_trend and k_values for diagnostics
    j_trend = np.zeros(n_obs)
    k_values = np.zeros(n_obs)
    for i in range(n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        if c > 0:
            col_base = 2 + 3 * (c - 1)
            j0 = beta[col_base]
            j1 = beta[col_base + 1]
            j2 = beta[col_base + 2]
            j_trend[i] = j0 + j1 * t + j2 * t * t
        k_val = k[yr]
        k_values[i] = k_val if not np.isnan(k_val) else 0.0

    h_values = h2 * f_low + h4 * f_high

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, y, total_r_sq)

    # For conjoined approach: Delta_u = 0, v = h(T)
    Delta_u = np.zeros(n_obs)
    v = h_values
    epsilon = y - (Delta_u + v + j_trend + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend, k_values, epsilon, y)

    return FitResultApproach8(
        approach="Approach TJ: Three-Interval (Joint OLS)",
        h2=h2,
        h2_se=h2_se,
        h4=h4,
        h4_se=h4_se,
        T_opt=T_opt,
        T_opt_se=np.nan,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
        T_crit_low=T_crit_low_opt,
        T_crit_high=T_crit_high_opt,
    )


def fit_ApproachTP_three_interval_linear_detrend(
    data: AnalysisData,
    trends: CountryTrends,
    year_means: dict,
    T_bounds: tuple = (0.0, 30.0),
    weights: np.ndarray = None,
) -> FitResultApproach8:
    """Approach TP: Three-interval response with linear T + quadratic GDP detrending.

    Optimization is over (T0, T1) both in T_bounds, with T_crit_low = min(T0, T1)
    and delta_T_crit = |T1 - T0|.

    Args:
        data: AnalysisData object
        trends: CountryTrends (with linear T trend: T0, T1; quadratic GDP: y0, y1, y2)
        year_means: Pre-computed k[t] = mean(dy_i[t])
        T_bounds: Bounds for T0 and T1 (default [0, 30])
        weights: Optional observation weights for weighted least squares

    Returns:
        FitResultApproach8 with T_opt, h2, h4, T_crit_low, delta_T_crit
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_i_t = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        y[i] = data.growth_pcGDP[i] - year_means[yr] - j_i_t

    T = data.temp

    # Compute T_trend using linear polynomial fit (T0 + T1*t)
    T_trend = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        T_trend[i] = trends.T0[c] + trends.T1[c] * t

    def compute_sse_for_T0T1(params):
        """Compute SSE for given (T0, T1), using T_crit_low=min, delta=|T1-T0|."""
        T_crit_low_val = min(params[0], params[1])
        delta_T_crit_val = abs(params[1] - params[0])

        f_low_T, f_high_T = three_interval_basis(T, T_crit_low_val, delta_T_crit_val)
        f_low_trend, f_high_trend = three_interval_basis(T_trend, T_crit_low_val, delta_T_crit_val)

        X1 = f_low_T - f_low_trend
        X2 = f_high_T - f_high_trend
        X = np.column_stack([X1, X2])

        try:
            if weights is not None:
                y_clean = np.where(np.isnan(y) & (weights == 0), 0, y)
                sqrt_W = np.sqrt(weights)
                X_w = X * sqrt_W[:, np.newaxis]
                y_w = y_clean * sqrt_W
                beta_ols, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred = X @ beta_ols
                sse = np.sum(weights * (y_clean - y_pred) ** 2)
            else:
                beta_ols, _, _, _ = linalg.lstsq(X, y)
                y_pred = X @ beta_ols
                sse = np.sum((y - y_pred) ** 2)
            return sse
        except Exception:
            return np.inf

    T_crit_low_opt, T_crit_high_opt = _optimize_three_interval(compute_sse_for_T0T1, T_bounds)
    delta_T_crit = T_crit_high_opt - T_crit_low_opt

    # Re-fit at optimal params
    f_low_T, f_high_T = three_interval_basis(T, T_crit_low_opt, delta_T_crit)
    f_low_trend, f_high_trend = three_interval_basis(T_trend, T_crit_low_opt, delta_T_crit)
    X1 = f_low_T - f_low_trend
    X2 = f_high_T - f_high_trend
    X_opt = np.column_stack([X1, X2])

    if weights is not None:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols_weighted(y, X_opt, weights)
    else:
        beta_ols, residuals, sigma_sq_resid, cov = fit_ols(y, X_opt)
    h2 = beta_ols[0]
    h4 = beta_ols[1]
    h2_se = np.sqrt(max(cov[0, 0], 0))
    h4_se = np.sqrt(max(cov[1, 1], 0))

    T_opt = _derive_T_opt(h2, h4, T_crit_low_opt, T_crit_high_opt)

    k = dict(year_means)
    n_params = 4
    r_sq, adj_r_sq, rmse = compute_fit_stats(y, residuals, n_params)
    total_r_sq = compute_total_r_squared(residuals, data.growth_pcGDP)

    j_trend = np.zeros(data.n_obs)
    k_values = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        t = data.time[i]
        yr = data.year[i]
        j_trend[i] = trends.y0[c] + trends.y1[c] * t + trends.y2[c] * t * t
        k_values[i] = year_means[yr]

    h_values = h2 * X1 + h4 * X2
    h_of_T_trend = h2 * f_low_trend + h4 * f_high_trend

    components = {
        'h_T': h_values,
        'j': j_trend,
        'k': k_values,
    }
    var_decomp = compute_variance_decomposition(components, data.growth_pcGDP, total_r_sq)

    Delta_u = h_values
    v = h_of_T_trend
    j_trend_adjusted = j_trend - v
    epsilon = data.growth_pcGDP - (Delta_u + v + j_trend_adjusted + k_values)
    var_attrib = compute_variance_attribution(Delta_u, v, j_trend_adjusted, k_values, epsilon, data.growth_pcGDP)

    return FitResultApproach8(
        approach="Approach TP: Three-Interval (Polynomial Detrending)",
        h2=h2,
        h2_se=h2_se,
        h4=h4,
        h4_se=h4_se,
        T_opt=T_opt,
        T_opt_se=np.nan,
        k=k,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        rmse=rmse,
        n_obs=data.n_obs,
        n_params=n_params,
        residuals=residuals,
        total_r_squared=total_r_sq,
        var_decomp=var_decomp,
        var_attrib=var_attrib,
        T_crit_low=T_crit_low_opt,
        T_crit_high=T_crit_high_opt,
    )


def fit_all_approaches(
    data: AnalysisData, trends: CountryTrends,
    trends_with_k: CountryTrends = None, year_means: dict = None,
    trends_loess: CountryTrendsLoess = None,
    weights: np.ndarray = None,
    approaches: list = None,
) -> dict:
    """Fit all approaches and return results.

    Returns dict with keys:
        Publication-ready approaches:
        'Approach QJ': Conjoined OLS fit, with j terms and year fixed effects
        'Approach QP': Pre-computed k with linear temp + quadratic GDP (if trends_with_k and year_means provided)
        'Approach QL': Pre-computed k with LOESS trends (if trends_loess provided)
        'Approach PL': Piecewise quadratic response with LOESS (if trends_loess provided)
        'Approach DL': Persistence decay model with LOESS (if trends_loess provided)
        'Approach LL': Level effect model (h4=1) with LOESS (if trends_loess provided)
        'Approach PJ': Piecewise quadratic with full OLS (like Approach PL + Approach QJ)
        'Approach DJ': Persistence decay with full OLS (like Approach DL + Approach QJ)
        'Approach LJ': Level effect model (h4=1) with joint OLS
        'Approach NJ': No climate response, joint OLS (country trends + year effects only)
        'Approach NP': No climate response, precomputed k (if trends_with_k and year_means provided)
        'Approach NL': No climate response, LOESS precomputed k (if trends_loess provided)

    Args:
        data: AnalysisData object
        trends: CountryTrends (unused, kept for API compatibility)
        trends_with_k: CountryTrends for Approach QP (fit to dy - k)
        year_means: Pre-computed k[t] for approaches 1-4
        trends_loess: CountryTrendsLoess for approaches 2-4 (LOESS detrending)
        weights: Optional observation weights for weighted least squares (bootstrap)
        approaches: Optional list of approach names to fit (default: None = fit all)
    """
    results = {}
    timings = {}
    wanted = set(approaches) if approaches else None

    def should_fit(name):
        return wanted is None or name in wanted

    def timed_fit(name, fit_func, *args):
        """Fit an approach and record timing."""
        start = time.perf_counter()
        result = fit_func(*args)
        elapsed = time.perf_counter() - start
        results[name] = result
        timings[name] = elapsed
        print(f"      {name}: {elapsed:.3f}s")

    # Conjoined approaches
    if should_fit('Approach QJ'):
        timed_fit('Approach QJ', fit_ApproachQJ_conjoined, data, weights)
    if should_fit('Approach NJ'):
        timed_fit('Approach NJ', fit_ApproachNJ_joint, data, weights)
    if should_fit('Approach PJ'):
        timed_fit('Approach PJ', fit_ApproachPJ_piecewise_conjoined, data, (0.0, 30.0), weights)
    if should_fit('Approach SJ'):
        timed_fit('Approach SJ', fit_ApproachSJ_segmented_conjoined, data, (0.0, 30.0), weights)
    if should_fit('Approach TJ'):
        timed_fit('Approach TJ', fit_ApproachTJ_three_interval_conjoined, data, (0.0, 30.0), weights)
    if should_fit('Approach DJ'):
        timed_fit('Approach DJ', fit_ApproachDJ_persistence_conjoined, data, (0.0, 1.0), weights)
    if should_fit('Approach LJ'):
        timed_fit('Approach LJ', fit_ApproachLJ_level_effect_conjoined, data, weights)

    # Add Approach QP and Approach NP if trends_with_k and year_means are provided
    if trends_with_k is not None and year_means is not None:
        if should_fit('Approach QP'):
            timed_fit('Approach QP', fit_ApproachQP_precomputed_k,
                      data, trends_with_k, year_means, weights)
        if should_fit('Approach NP'):
            # NP uses precomputed trends - weighting is done in detrending step
            timed_fit('Approach NP', fit_ApproachNP_precomputed_k,
                      data, trends_with_k, year_means)

    # Add Approach QL, Approach PL, Approach DL, Approach LL and Approach NL if trends_loess and year_means are provided
    if trends_loess is not None and year_means is not None:
        if should_fit('Approach QL'):
            timed_fit('Approach QL', fit_ApproachQL_loess,
                      data, trends_loess, year_means, weights)
        if should_fit('Approach NL'):
            # NL uses precomputed trends - weighting is done in detrending step
            timed_fit('Approach NL', fit_ApproachNL_precomputed_k_loess,
                      data, trends_loess, year_means)
        if should_fit('Approach PL'):
            timed_fit('Approach PL', fit_ApproachPL_piecewise,
                      data, trends_loess, year_means, (0.0, 30.0), weights)
        if should_fit('Approach SL'):
            timed_fit('Approach SL', fit_ApproachSL_segmented,
                      data, trends_loess, year_means, (0.0, 30.0), weights)
        if should_fit('Approach TL'):
            timed_fit('Approach TL', fit_ApproachTL_three_interval,
                      data, trends_loess, year_means, (0.0, 30.0), weights)
        if should_fit('Approach DL'):
            timed_fit('Approach DL', fit_ApproachDL_persistence_decay,
                      data, trends_loess, year_means, (0.0, 1.0), weights)
        if should_fit('Approach LL'):
            timed_fit('Approach LL', fit_ApproachLL_first_difference,
                      data, trends_loess, year_means, weights)

    # Add Approach PP and Approach DP if trends_with_k and year_means are provided
    # These use linear T / quadratic GDP detrending with alternative climate response functions
    if trends_with_k is not None and year_means is not None:
        if should_fit('Approach PP'):
            timed_fit('Approach PP', fit_ApproachPP_piecewise_linear_detrend,
                      data, trends_with_k, year_means, (0.0, 30.0), weights)
        if should_fit('Approach SP'):
            timed_fit('Approach SP', fit_ApproachSP_segmented_linear_detrend,
                      data, trends_with_k, year_means, (0.0, 30.0), weights)
        if should_fit('Approach TP'):
            timed_fit('Approach TP', fit_ApproachTP_three_interval_linear_detrend,
                      data, trends_with_k, year_means, (0.0, 30.0), weights)
        if should_fit('Approach DP'):
            timed_fit('Approach DP', fit_ApproachDP_persistence_linear_detrend,
                      data, trends_with_k, year_means, (0.0, 1.0), weights)

    total_time = sum(timings.values())
    print(f"      Total fitting time: {total_time:.3f}s")

    return results
