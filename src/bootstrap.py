"""Country-level bootstrap resampling for uncertainty quantification.

This module implements cluster bootstrap (resampling countries with replacement)
for computing uncertainty estimates on h1, h2, and T_opt for all approaches.

The cluster bootstrap approach:
1. Fit the original model to get point estimates
2. For B bootstrap iterations:
   - Resample M countries with replacement from M countries
   - Build a new dataset from the selected countries (with re-indexed country IDs)
   - Recompute country trends for the bootstrap sample
   - Fit all 8 approaches to get bootstrap parameter estimates
3. Use the empirical distribution of parameters across bootstrap samples
   to compute percentile-based confidence intervals

This preserves within-country correlation structure across years.
"""

import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Default number of bootstrap iterations
DEFAULT_N_BOOTSTRAP = 1000

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED = 42

# Default percentiles for computing bootstrap statistics
DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)
from dataclasses import dataclass
from typing import Dict, Tuple

from .data_loader import AnalysisData
from .detrending import (
    CountryTrends,
    CountryTrendsLoess,
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
    DEFAULT_LOESS_WINDOW_YEARS,
    # Weighted functions for time-dimension bootstrap
    compute_year_means_weighted,
    compute_country_trends_weighted,
    compute_country_trends_with_k_weighted,
    compute_country_trends_loess_weighted,
)
from .fitting import (
    fit_all_approaches,
    FitResult,
    compute_persistence_accumulators,
    compute_persistence_accumulators_at_T,
    compute_pre_first_year_correction,
    compute_T_linear_at_first_year,
    fit_ols_weighted,
)


def _get_T_loess_at_base_year(
    data: AnalysisData,
    trends_loess: "CountryTrendsLoess",
    base_year: int = 1961
) -> np.ndarray:
    """Get T_loess at base year for each observation's country.

    For Approach DL's pre-history assumption, we want to use T_loess at 1961
    (not the actual temperature at first observation). This function creates
    an array where each observation has its country's T_loess at base_year.

    Args:
        data: AnalysisData with country/year info
        trends_loess: CountryTrendsLoess with T_loess values
        base_year: Base year (default: 1961)

    Returns:
        Array of shape (n_obs,) with T_loess at base_year for each observation's country
    """
    T_loess = trends_loess.T_loess
    year_arr = data.year.astype(int)
    result = np.zeros(data.n_obs)

    for c in range(data.n_countries):
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]
        years_for_country = year_arr[country_indices]

        # Find T_loess at base_year for this country
        base_year_mask = years_for_country == base_year
        if base_year_mask.any():
            base_idx = country_indices[np.where(base_year_mask)[0][0]]
            T_loess_base = T_loess[base_idx]
        else:
            # If no observation at base year, use earliest year
            earliest_idx = country_indices[np.argmin(years_for_country)]
            T_loess_base = T_loess[earliest_idx]

        # Set all observations for this country to T_loess_base
        result[country_mask] = T_loess_base

    return result


@dataclass
class BootstrapResult:
    """Container for bootstrap results for a single approach.

    Field naming convention follows the FitResult naming:
    - Universal: h1, h2, T_opt (for standard quadratic approaches)
    - Approach 5d: f1 = GDP scaling exponent
    - Approach 6b/6e: h1,h2 = actual T response; h3,h4 = departure response; T_opt, T_dep_opt = optimal temps
    - Approach 6c: h1,h2 = departure response; h3,h4 = trend response; T_dep_opt, f2 = optimal temps
    - Approach 6e: h3=0 (quadratic departure only)
    - Approach 8: h2 = curvature below T_opt; h4 = curvature above T_opt
    - Approach 8a: h2 = curvature for actual T; h4 = curvature for trend T
    - Approach 8b: f1 = linear modulation; f2 = quadratic modulation; h1,h2 = actual T response
    """
    approach: str

    # Point estimates (from original fit)
    h1_point: float
    h2_point: float
    T_opt_point: float             # (formerly T_optimal_point)
    r_squared_point: float
    total_r_squared_point: float

    # Bootstrap samples - shape (n_bootstrap,)
    h1_samples: np.ndarray
    h2_samples: np.ndarray
    T_opt_samples: np.ndarray      # (formerly T_optimal_samples)
    r_squared_samples: np.ndarray
    total_r_squared_samples: np.ndarray

    # Metadata
    n_bootstrap: int
    n_successful: int  # Number of successful fits

    # GDP-dependent approaches (5d) specific (optional)
    f1_point: float = None         # GDP scaling exponent (formerly beta_point)
    f1_samples: np.ndarray = None  # Bootstrap samples for f1 (formerly beta_samples)

    # Approach 8 (piecewise quadratic) specific (optional)
    # Note: h2 for approach 8 is stored in h2_point; h4 is the curvature above T_opt
    h4_point: float = None         # Curvature for T > T_opt (formerly h2_high_point)
    h4_samples: np.ndarray = None  # Bootstrap samples for h4 (formerly h2_high_samples)

    # Approach 6b/6e (separate actual T/departure response) specific (optional)
    h3_point: float = None         # Linear coef for departure (formerly h1_trend_point)
    h3_samples: np.ndarray = None  # Bootstrap samples for h3
    # Note: h4 reused from above for quadratic departure coef (formerly h2_trend)
    # For 6b/6e, T_opt is the optimal actual T; T_dep_opt is optimal departure
    T_dep_opt_point: float = None  # Optimal departure (-h3/(2*h4))
    T_dep_opt_samples: np.ndarray = None  # Bootstrap samples for T_dep_opt

    # Approach 8a (shared T_opt, total/trend) specific (optional)
    # Note: h2 is curvature for actual T; h4 is curvature for trend T (reuses h4_point)

    # Approach 8b (modulated response) specific (optional)
    # Note: f1 = linear modulation coefficient; f2 = quadratic modulation coefficient
    # h1,h2 = actual T response coefficients

    # Approach 6c (departure/trend decomposition) specific (optional)
    # Note: h1,h2 = departure coefficients; h3,h4 = trend coefficients (reuses h3/h4_point)
    # T_dep_opt = optimal departure (-h1/(2*h2)); f2 = optimal trend T (-h3/(2*h4))
    # For 8b: f2 = quadratic modulation coefficient
    f2_point: float = None         # 6c: T_opt_trend; 8b: quadratic modulation
    f2_samples: np.ndarray = None  # Bootstrap samples for f2

    # Variance decomposition
    var_decomp_point: dict = None   # From original fit
    var_decomp_samples: dict = None  # Dict mapping key -> np.ndarray of bootstrap samples

    # Variance attribution (5-component decomposition)
    var_attrib_point: dict = None      # From original fit
    var_attrib_samples: dict = None    # Dict mapping key -> np.ndarray of bootstrap samples

    # Year fixed effects k(t)
    k_point: Dict[int, float] = None      # Point estimates from original fit
    k_samples: Dict[int, np.ndarray] = None  # year -> array of shape (n_bootstrap,)


def create_bootstrap_data(
    data: AnalysisData,
    selected_country_indices: np.ndarray
) -> AnalysisData:
    """Create bootstrap sample by resampling countries with replacement.

    Args:
        data: Original panel data
        selected_country_indices: Array of length n_countries with country
            indices to include (may have duplicates for replacement)

    Returns:
        New AnalysisData with resampled observations and re-indexed countries
    """
    # Build list of observations to include, with new country indices
    obs_indices = []
    new_country_idx = []

    for new_idx, orig_country_idx in enumerate(selected_country_indices):
        # Find all observations for this original country
        country_obs = np.where(data.country_idx == orig_country_idx)[0]
        obs_indices.extend(country_obs)
        new_country_idx.extend([new_idx] * len(country_obs))

    obs_indices = np.array(obs_indices)
    new_country_idx = np.array(new_country_idx, dtype=np.int32)

    # Extract data for selected observations
    growth_pcGDP = data.growth_pcGDP[obs_indices]
    pcGDP = data.pcGDP[obs_indices]
    temp = data.temp[obs_indices]
    time = data.time[obs_indices]
    year = data.year[obs_indices]

    # Build new iso mappings for selected countries
    # Map idx -> original iso code
    new_idx_to_iso = {}
    new_iso_to_idx = {}
    for idx, orig_idx in enumerate(selected_country_indices):
        orig_iso = data.idx_to_iso[orig_idx]
        # If the same country is selected multiple times, append a suffix
        iso_key = f"{orig_iso}_{idx}"
        new_idx_to_iso[idx] = iso_key
        new_iso_to_idx[iso_key] = idx

    # Create new AnalysisData with resampled observations
    return AnalysisData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        time=time,
        country_idx=new_country_idx,
        year=year,
        iso_to_idx=new_iso_to_idx,
        idx_to_iso=new_idx_to_iso,
        n_obs=len(obs_indices),
        n_countries=len(selected_country_indices),
        n_years=data.n_years,
        year_range=data.year_range,
        time_offset=data.time_offset,
    )


def compute_bootstrap_weights(
    data: AnalysisData,
    selected_country_indices: np.ndarray,
    selected_year_indices: np.ndarray = None,
) -> np.ndarray:
    """Compute observation weights from bootstrap country/year sampling.

    When both countries and years are sampled with replacement, each observation's
    weight equals (country_count) × (year_count), where:
    - country_count = how many times this observation's country was sampled
    - year_count = how many times this observation's year was sampled

    Args:
        data: Original panel data (with all observations)
        selected_country_indices: Array of sampled country indices (length n_countries)
        selected_year_indices: Array of sampled year indices (length n_years), or None
            for country-only bootstrap (all year weights = 1)

    Returns:
        Observation weights, shape (n_obs,). Weights are 0 for observations whose
        year was not sampled (when year sampling is enabled).
    """
    # Count how many times each country was sampled
    country_counts = np.bincount(selected_country_indices, minlength=data.n_countries)

    # Count how many times each year was sampled
    unique_years = sorted(set(data.year))
    year_to_idx = {y: i for i, y in enumerate(unique_years)}

    if selected_year_indices is not None:
        year_counts = np.bincount(selected_year_indices, minlength=len(unique_years))
    else:
        # Country-only bootstrap: all years have weight 1
        year_counts = np.ones(len(unique_years))

    # Compute observation weights
    weights = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        c = data.country_idx[i]
        y_idx = year_to_idx[data.year[i]]
        weights[i] = country_counts[c] * year_counts[y_idx]

    return weights


def _fit_all_approaches_weighted(
    data: AnalysisData,
    trends: CountryTrends,
    weights: np.ndarray,
    trends_with_k: CountryTrends = None,
    year_means: dict = None,
    trends_loess: CountryTrendsLoess = None,
    loess_window: float = None,
) -> dict:
    """Fit all approaches using weighted OLS for bootstrap with year sampling.

    This is a simplified version of fit_all_approaches that uses weighted OLS.
    It extracts the essential coefficients needed for bootstrap analysis.

    Args:
        data: AnalysisData object (original data, not duplicated)
        trends: CountryTrends for polynomial detrending
        weights: Observation weights from country/year sampling
        trends_with_k: CountryTrends fit to (dy - k) for polynomial approaches
        year_means: Pre-computed weighted k[t]
        trends_loess: CountryTrendsLoess for LOESS approaches
        loess_window: LOESS window size (for consistency)

    Returns:
        Dict mapping approach name to a simple result object with h1, h2, T_opt, etc.
    """
    from scipy import linalg
    from .detrending import (
        compute_detrended_temperature,
        compute_detrended_temp_squared,
        compute_detrended_temperature_loess,
        compute_detrended_temp_squared_loess,
    )
    from .fitting import (
        compute_T_optimal,
        compute_fit_stats,
        compute_total_r_squared,
        FitResult,
        FitResultApproach8,
        FitResultApproach4,
    )

    results = {}

    # Helper: weighted OLS using pseudoinverse for numerical stability
    def weighted_ols(y, X, w):
        """Solve weighted least squares using pseudoinverse for stability."""
        # Remove columns that have zero effective weight (prevents singularity)
        effective_weight_per_col = np.abs(X.T @ w)
        valid_cols = effective_weight_per_col > 1e-10
        X_valid = X[:, valid_cols]

        if X_valid.shape[1] == 0:
            # No valid columns, return zeros
            beta = np.zeros(X.shape[1])
            residuals = y.copy()
            cov = np.zeros((X.shape[1], X.shape[1]))
            return beta, residuals, np.nan, cov

        # Weighted normal equations
        XtW = X_valid.T * w
        XtWX = XtW @ X_valid
        XtWy = XtW @ y

        # Use lstsq for numerical stability
        beta_valid, _, _, _ = linalg.lstsq(XtWX, XtWy)

        # Reconstruct full beta vector
        beta = np.zeros(X.shape[1])
        beta[valid_cols] = beta_valid

        y_pred = X @ beta
        residuals = y - y_pred

        # Covariance (simplified for bootstrap - SE not critical)
        n_eff = np.sum(w)
        p = X_valid.shape[1]
        df = n_eff - p
        sse_w = np.sum(w * residuals ** 2)
        sigma_sq = sse_w / df if df > 0 else np.nan

        # Use pseudoinverse for covariance
        cov = np.zeros((X.shape[1], X.shape[1]))
        try:
            XtWX_inv = linalg.pinv(XtWX)
            cov_valid = sigma_sq * XtWX_inv
            for i, ci in enumerate(np.where(valid_cols)[0]):
                for j, cj in enumerate(np.where(valid_cols)[0]):
                    cov[ci, cj] = cov_valid[i, j]
        except:
            pass

        return beta, residuals, sigma_sq, cov

    # Build year dummies for joint approaches
    # Only include years that have non-zero weight
    unique_years = sorted(set(data.year))
    year_weights = {}
    for yr in unique_years:
        yr_mask = data.year == yr
        year_weights[yr] = np.sum(weights[yr_mask])

    # Filter to years with non-zero weight
    active_years = [yr for yr in unique_years if year_weights[yr] > 0]
    n_years_local = len(active_years)
    year_to_idx_local = {y: i for i, y in enumerate(active_years)}

    def build_year_dummies():
        year_dummies = np.zeros((data.n_obs, n_years_local))
        for i in range(data.n_obs):
            yr = data.year[i]
            if yr in year_to_idx_local:
                yr_idx = year_to_idx_local[yr]
                year_dummies[i, yr_idx] = 1.0
        return year_dummies

    year_dummies = build_year_dummies()

    # =========================================================================
    # Approach QJ: Joint OLS with quadratic h(T)
    # =========================================================================
    # Design matrix: [T, T², year_dummies, country_time_dummies]
    # For simplicity in bootstrap, we skip country-specific j terms and just
    # estimate h1, h2, k
    X_QJ = np.column_stack([data.temp, data.temp ** 2, year_dummies])
    y = data.growth_pcGDP
    beta_QJ, residuals_QJ, _, cov_QJ = weighted_ols(y, X_QJ, weights)
    h1_QJ = beta_QJ[0]
    h2_QJ = beta_QJ[1]
    # Build k dict for all years (zero for inactive years)
    k_QJ = {}
    for yr in unique_years:
        if yr in year_to_idx_local:
            k_QJ[yr] = beta_QJ[2 + year_to_idx_local[yr]]
        else:
            k_QJ[yr] = 0.0
    T_opt_QJ = compute_T_optimal(h1_QJ, h2_QJ)

    # Compute fit stats
    r_sq_QJ, adj_r_sq_QJ, rmse_QJ = compute_fit_stats(y, residuals_QJ, X_QJ.shape[1])
    total_r_sq_QJ = compute_total_r_squared(residuals_QJ, data.growth_pcGDP)

    results['Approach QJ'] = FitResult(
        approach="Approach QJ: Quadratic (Joint OLS)",
        h1=h1_QJ, h2=h2_QJ,
        h1_se=np.sqrt(max(cov_QJ[0, 0], 0)), h2_se=np.sqrt(max(cov_QJ[1, 1], 0)),
        k=k_QJ,
        r_squared=r_sq_QJ, adj_r_squared=adj_r_sq_QJ, rmse=rmse_QJ,
        n_obs=data.n_obs, n_params=X_QJ.shape[1],
        residuals=residuals_QJ, T_opt=T_opt_QJ,
        total_r_squared=total_r_sq_QJ,
    )

    # =========================================================================
    # Approach NJ: Null model (h1=h2=0)
    # =========================================================================
    X_NJ = year_dummies
    beta_NJ, residuals_NJ, _, _ = weighted_ols(y, X_NJ, weights)
    # Build k dict for all years
    k_NJ = {}
    for yr in unique_years:
        if yr in year_to_idx_local:
            k_NJ[yr] = beta_NJ[year_to_idx_local[yr]]
        else:
            k_NJ[yr] = 0.0
    r_sq_NJ, adj_r_sq_NJ, rmse_NJ = compute_fit_stats(y, residuals_NJ, X_NJ.shape[1])
    total_r_sq_NJ = compute_total_r_squared(residuals_NJ, data.growth_pcGDP)

    results['Approach NJ'] = FitResult(
        approach="Approach NJ: Null (Joint OLS)",
        h1=0.0, h2=0.0, h1_se=0.0, h2_se=0.0,
        k=k_NJ,
        r_squared=r_sq_NJ, adj_r_squared=adj_r_sq_NJ, rmse=rmse_NJ,
        n_obs=data.n_obs, n_params=X_NJ.shape[1],
        residuals=residuals_NJ, T_opt=np.nan,
        total_r_squared=total_r_sq_NJ,
    )

    # =========================================================================
    # Pre-computed approaches (QP, QL, etc.) - use pre-computed trends
    # =========================================================================
    if trends_with_k is not None and year_means is not None:
        # Approach QP: Polynomial detrending
        T_star = compute_detrended_temperature(data, trends_with_k)
        T2_detrend = compute_detrended_temp_squared(data, trends_with_k)

        # Compute dependent variable: dy - k - j
        y_QP = np.zeros(data.n_obs)
        for i in range(data.n_obs):
            c = data.country_idx[i]
            t = data.time[i]
            yr = data.year[i]
            j_i = trends_with_k.y0[c] + trends_with_k.y1[c] * t + trends_with_k.y2[c] * t * t
            y_QP[i] = data.growth_pcGDP[i] - year_means[yr] - j_i

        X_QP = np.column_stack([T_star, T2_detrend])
        beta_QP, residuals_QP, _, cov_QP = weighted_ols(y_QP, X_QP, weights)
        h1_QP = beta_QP[0]
        h2_QP = beta_QP[1]
        T_opt_QP = compute_T_optimal(h1_QP, h2_QP)
        r_sq_QP, adj_r_sq_QP, rmse_QP = compute_fit_stats(y_QP, residuals_QP, 2)
        total_r_sq_QP = compute_total_r_squared(residuals_QP, data.growth_pcGDP)

        results['Approach QP'] = FitResult(
            approach="Approach QP: Quadratic (Polynomial Detrending)",
            h1=h1_QP, h2=h2_QP,
            h1_se=np.sqrt(cov_QP[0, 0]), h2_se=np.sqrt(cov_QP[1, 1]),
            k=dict(year_means),
            r_squared=r_sq_QP, adj_r_squared=adj_r_sq_QP, rmse=rmse_QP,
            n_obs=data.n_obs, n_params=2,
            residuals=residuals_QP, T_opt=T_opt_QP,
            total_r_squared=total_r_sq_QP,
        )

        # Approach NP: Null with polynomial
        results['Approach NP'] = FitResult(
            approach="Approach NP: Null (Polynomial Detrending)",
            h1=0.0, h2=0.0, h1_se=0.0, h2_se=0.0,
            k=dict(year_means),
            r_squared=0.0, adj_r_squared=0.0, rmse=np.std(y_QP),
            n_obs=data.n_obs, n_params=0,
            residuals=y_QP, T_opt=np.nan,
            total_r_squared=total_r_sq_QP,
        )

    if trends_loess is not None and year_means is not None:
        # Approach QL: LOESS detrending
        T_star_L = compute_detrended_temperature_loess(data, trends_loess)
        T2_detrend_L = compute_detrended_temp_squared_loess(data, trends_loess)

        y_QL = np.zeros(data.n_obs)
        for i in range(data.n_obs):
            yr = data.year[i]
            y_QL[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

        X_QL = np.column_stack([T_star_L, T2_detrend_L])
        beta_QL, residuals_QL, _, cov_QL = weighted_ols(y_QL, X_QL, weights)
        h1_QL = beta_QL[0]
        h2_QL = beta_QL[1]
        T_opt_QL = compute_T_optimal(h1_QL, h2_QL)
        r_sq_QL, adj_r_sq_QL, rmse_QL = compute_fit_stats(y_QL, residuals_QL, 2)
        total_r_sq_QL = compute_total_r_squared(residuals_QL, data.growth_pcGDP)

        results['Approach QL'] = FitResult(
            approach="Approach QL: Quadratic (LOESS Detrending)",
            h1=h1_QL, h2=h2_QL,
            h1_se=np.sqrt(cov_QL[0, 0]), h2_se=np.sqrt(cov_QL[1, 1]),
            k=dict(year_means),
            r_squared=r_sq_QL, adj_r_squared=adj_r_sq_QL, rmse=rmse_QL,
            n_obs=data.n_obs, n_params=2,
            residuals=residuals_QL, T_opt=T_opt_QL,
            total_r_squared=total_r_sq_QL,
        )

        # Approach NL: Null with LOESS
        results['Approach NL'] = FitResult(
            approach="Approach NL: Null (LOESS Detrending)",
            h1=0.0, h2=0.0, h1_se=0.0, h2_se=0.0,
            k=dict(year_means),
            r_squared=0.0, adj_r_squared=0.0, rmse=np.std(y_QL),
            n_obs=data.n_obs, n_params=0,
            residuals=y_QL, T_opt=np.nan,
            total_r_squared=total_r_sq_QL,
        )

        # =====================================================================
        # Approach PL: Piecewise quadratic with LOESS
        # =====================================================================
        # Optimize T_opt via grid search
        T_range = np.linspace(5, 25, 41)
        best_sse = np.inf
        best_T_opt_PL = 15.0
        best_h2_PL = 0.0
        best_h4_PL = 0.0

        for T_test in T_range:
            below = data.temp <= T_test
            X1_PL = np.where(below, (data.temp - T_test) ** 2, 0)
            X2_PL = np.where(~below, (data.temp - T_test) ** 2, 0)
            # Subtract LOESS trends
            X1_PL_star = X1_PL - np.where(below, (trends_loess.T_loess - T_test) ** 2, 0)
            X2_PL_star = X2_PL - np.where(~below, (trends_loess.T_loess - T_test) ** 2, 0)

            X_PL = np.column_stack([X1_PL_star, X2_PL_star])
            try:
                beta_PL_test, residuals_PL_test, _, _ = weighted_ols(y_QL, X_PL, weights)
                sse = np.sum(weights * residuals_PL_test ** 2)
                if sse < best_sse:
                    best_sse = sse
                    best_T_opt_PL = T_test
                    best_h2_PL = beta_PL_test[0]
                    best_h4_PL = beta_PL_test[1]
            except:
                pass

        results['Approach PL'] = FitResultApproach8(
            approach="Approach PL: Piecewise Quadratic (LOESS)",
            h2=best_h2_PL, h2_se=0.0,
            h4=best_h4_PL, h4_se=0.0,
            T_opt=best_T_opt_PL, T_opt_se=0.0,
            k=dict(year_means),
            r_squared=r_sq_QL, adj_r_squared=adj_r_sq_QL, rmse=rmse_QL,
            n_obs=data.n_obs, n_params=3,
            residuals=residuals_QL,
            total_r_squared=total_r_sq_QL,
        )

        # =====================================================================
        # Approach DL: Persistence decay with LOESS
        # =====================================================================
        # Optimize h4 (decay parameter) via grid search
        h4_range = np.linspace(0.01, 0.99, 20)
        best_sse_DL = np.inf
        best_h4_DL = 0.5
        best_h1_DL = 0.0
        best_h2_DL = 0.0

        for h4_test in h4_range:
            A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_test)
            # Correction for pre-history (using T_loess at base year)
            T_loess_base = _get_T_loess_at_base_year(data, trends_loess, base_year=1961)
            correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_test, T_loess_base)

            X1_DL = data.temp - h4_test * A_T_lag - correction_T
            X2_DL = data.temp ** 2 - h4_test * A_T2_lag - correction_T2
            # Subtract LOESS trends for persistence model
            X1_DL_star = X1_DL - (trends_loess.T_loess - h4_test * compute_persistence_accumulators_at_T(data, h4_test, trends_loess.T_loess)[0] - correction_T)
            X2_DL_star = X2_DL - (trends_loess.T_loess ** 2 - h4_test * compute_persistence_accumulators_at_T(data, h4_test, trends_loess.T_loess)[1] - correction_T2)

            X_DL = np.column_stack([X1_DL_star, X2_DL_star])
            try:
                beta_DL_test, residuals_DL_test, _, _ = weighted_ols(y_QL, X_DL, weights)
                sse = np.sum(weights * residuals_DL_test ** 2)
                if sse < best_sse_DL:
                    best_sse_DL = sse
                    best_h4_DL = h4_test
                    best_h1_DL = beta_DL_test[0]
                    best_h2_DL = beta_DL_test[1]
            except:
                pass

        T_opt_DL = compute_T_optimal(best_h1_DL, best_h2_DL)

        results['Approach DL'] = FitResultApproach4(
            approach="Approach DL: Persistence Decay (LOESS)",
            h1=best_h1_DL, h2=best_h2_DL,
            h1_se=0.0, h2_se=0.0,
            h4=best_h4_DL, h4_se=0.0,
            k=dict(year_means),
            r_squared=r_sq_QL, adj_r_squared=adj_r_sq_QL, rmse=rmse_QL,
            n_obs=data.n_obs, n_params=3,
            residuals=residuals_QL,
            T_opt=T_opt_DL,
            total_r_squared=total_r_sq_QL,
        )

    # =========================================================================
    # Joint approaches: PJ, DJ
    # =========================================================================
    # Approach PJ: Piecewise joint - simplified grid search
    T_range = np.linspace(5, 25, 41)
    best_sse_PJ = np.inf
    best_T_opt_PJ = 15.0
    best_h2_PJ = 0.0
    best_h4_PJ = 0.0

    for T_test in T_range:
        below = data.temp <= T_test
        X1_PJ = np.where(below, (data.temp - T_test) ** 2, 0)
        X2_PJ = np.where(~below, (data.temp - T_test) ** 2, 0)
        X_PJ = np.column_stack([X1_PJ, X2_PJ, year_dummies])
        try:
            beta_PJ_test, residuals_PJ_test, _, _ = weighted_ols(y, X_PJ, weights)
            sse = np.sum(weights * residuals_PJ_test ** 2)
            if sse < best_sse_PJ:
                best_sse_PJ = sse
                best_T_opt_PJ = T_test
                best_h2_PJ = beta_PJ_test[0]
                best_h4_PJ = beta_PJ_test[1]
        except:
            pass

    k_PJ = {yr: 0.0 for yr in unique_years}  # Simplified
    r_sq_PJ, _, rmse_PJ = compute_fit_stats(y, residuals_QJ, 3)  # Approximate
    total_r_sq_PJ = compute_total_r_squared(residuals_QJ, data.growth_pcGDP)

    results['Approach PJ'] = FitResultApproach8(
        approach="Approach PJ: Piecewise Quadratic (Joint)",
        h2=best_h2_PJ, h2_se=0.0,
        h4=best_h4_PJ, h4_se=0.0,
        T_opt=best_T_opt_PJ, T_opt_se=0.0,
        k=k_PJ,
        r_squared=r_sq_PJ, adj_r_squared=r_sq_PJ, rmse=rmse_PJ,
        n_obs=data.n_obs, n_params=3,
        residuals=residuals_QJ,
        total_r_squared=total_r_sq_PJ,
    )

    # Approach DJ: Persistence joint - simplified grid search
    h4_range = np.linspace(0.01, 0.99, 20)
    best_sse_DJ = np.inf
    best_h4_DJ = 0.5
    best_h1_DJ = 0.0
    best_h2_DJ = 0.0

    T_linear_first = compute_T_linear_at_first_year(data)
    for h4_test in h4_range:
        A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_test)
        correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_test, T_linear_first)
        X1_DJ = data.temp - h4_test * A_T_lag - correction_T
        X2_DJ = data.temp ** 2 - h4_test * A_T2_lag - correction_T2
        X_DJ = np.column_stack([X1_DJ, X2_DJ, year_dummies])
        try:
            beta_DJ_test, residuals_DJ_test, _, _ = weighted_ols(y, X_DJ, weights)
            sse = np.sum(weights * residuals_DJ_test ** 2)
            if sse < best_sse_DJ:
                best_sse_DJ = sse
                best_h4_DJ = h4_test
                best_h1_DJ = beta_DJ_test[0]
                best_h2_DJ = beta_DJ_test[1]
        except:
            pass

    T_opt_DJ = compute_T_optimal(best_h1_DJ, best_h2_DJ)
    k_DJ = {yr: 0.0 for yr in unique_years}

    results['Approach DJ'] = FitResultApproach4(
        approach="Approach DJ: Persistence Decay (Joint)",
        h1=best_h1_DJ, h2=best_h2_DJ,
        h1_se=0.0, h2_se=0.0,
        h4=best_h4_DJ, h4_se=0.0,
        k=k_DJ,
        r_squared=r_sq_QJ, adj_r_squared=r_sq_QJ, rmse=rmse_QJ,
        n_obs=data.n_obs, n_params=3,
        residuals=residuals_QJ,
        T_opt=T_opt_DJ,
        total_r_squared=total_r_sq_QJ,
    )

    # =========================================================================
    # PP and DP approaches (polynomial detrending variants)
    # =========================================================================
    if trends_with_k is not None and year_means is not None:
        # Approach PP: Piecewise with polynomial detrending
        T_range = np.linspace(5, 25, 41)
        best_sse_PP = np.inf
        best_T_opt_PP = 15.0
        best_h2_PP = 0.0
        best_h4_PP = 0.0

        for T_test in T_range:
            below = data.temp <= T_test
            # Temperature terms
            T_low = np.where(below, (data.temp - T_test) ** 2, 0)
            T_high = np.where(~below, (data.temp - T_test) ** 2, 0)
            # Trend terms
            T_trend = np.array([trends_with_k.T0[c] + trends_with_k.T1[c] * data.time[i]
                               for i, c in enumerate(data.country_idx)])
            T_low_trend = np.where(T_trend <= T_test, (T_trend - T_test) ** 2, 0)
            T_high_trend = np.where(T_trend > T_test, (T_trend - T_test) ** 2, 0)

            X1_PP = T_low - T_low_trend
            X2_PP = T_high - T_high_trend
            X_PP = np.column_stack([X1_PP, X2_PP])
            try:
                beta_PP_test, residuals_PP_test, _, _ = weighted_ols(y_QP, X_PP, weights)
                sse = np.sum(weights * residuals_PP_test ** 2)
                if sse < best_sse_PP:
                    best_sse_PP = sse
                    best_T_opt_PP = T_test
                    best_h2_PP = beta_PP_test[0]
                    best_h4_PP = beta_PP_test[1]
            except:
                pass

        results['Approach PP'] = FitResultApproach8(
            approach="Approach PP: Piecewise Quadratic (Polynomial)",
            h2=best_h2_PP, h2_se=0.0,
            h4=best_h4_PP, h4_se=0.0,
            T_opt=best_T_opt_PP, T_opt_se=0.0,
            k=dict(year_means),
            r_squared=r_sq_QP, adj_r_squared=r_sq_QP, rmse=rmse_QP,
            n_obs=data.n_obs, n_params=3,
            residuals=residuals_QP,
            total_r_squared=total_r_sq_QP,
        )

        # Approach DP: Persistence with polynomial detrending
        h4_range = np.linspace(0.01, 0.99, 20)
        best_sse_DP = np.inf
        best_h4_DP = 0.5
        best_h1_DP = 0.0
        best_h2_DP = 0.0

        for h4_test in h4_range:
            A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4_test)
            correction_T, correction_T2 = compute_pre_first_year_correction(data, h4_test, T_linear_first)
            X1_DP = data.temp - h4_test * A_T_lag - correction_T
            X2_DP = data.temp ** 2 - h4_test * A_T2_lag - correction_T2
            # Detrend
            T_star_DP = compute_detrended_temperature(data, trends_with_k)
            T2_star_DP = compute_detrended_temp_squared(data, trends_with_k)

            X_DP = np.column_stack([T_star_DP, T2_star_DP])
            try:
                beta_DP_test, residuals_DP_test, _, _ = weighted_ols(y_QP, X_DP, weights)
                sse = np.sum(weights * residuals_DP_test ** 2)
                if sse < best_sse_DP:
                    best_sse_DP = sse
                    best_h4_DP = h4_test
                    best_h1_DP = beta_DP_test[0]
                    best_h2_DP = beta_DP_test[1]
            except:
                pass

        T_opt_DP = compute_T_optimal(best_h1_DP, best_h2_DP)

        results['Approach DP'] = FitResultApproach4(
            approach="Approach DP: Persistence Decay (Polynomial)",
            h1=best_h1_DP, h2=best_h2_DP,
            h1_se=0.0, h2_se=0.0,
            h4=best_h4_DP, h4_se=0.0,
            k=dict(year_means),
            r_squared=r_sq_QP, adj_r_squared=r_sq_QP, rmse=rmse_QP,
            n_obs=data.n_obs, n_params=3,
            residuals=residuals_QP,
            T_opt=T_opt_DP,
            total_r_squared=total_r_sq_QP,
        )

    return results


def run_bootstrap(
    data: AnalysisData,
    trends: CountryTrends,
    original_results: Dict[str, FitResult],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = True,
    loess_window: int = None,
    h_T_approaches: list = None,
    sample_years: bool = False,
) -> Tuple[Dict[str, BootstrapResult], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Run bootstrap analysis for all methods.

    For each bootstrap iteration:
    1. Sample M countries with replacement
    2. Optionally sample Y years with replacement (if sample_years=True)
    3. Create bootstrap dataset (or compute weights for weighted fitting)
    4. Recompute country trends for bootstrap sample
    5. Fit all methods
    6. Store h1, h2, T_opt, R², Total R², and h4 (for method-specific coefficients)
    7. Optionally compute h(T) for selected methods

    Args:
        data: Original AnalysisData
        trends: Original CountryTrends (used for point estimates)
        original_results: Dict of original FitResult for each method
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        loess_window: Window size in years for LOESS smoothing
            (default: DEFAULT_LOESS_WINDOW_YEARS)
        h_T_approaches: List of method names to compute h(T) for (default: None means skip)
            Example: ['Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL']
        sample_years: If True, also sample years with replacement (time-dimension bootstrap).
            When True, uses weighted fitting instead of observation duplication.

    Returns:
        Tuple of:
        - Dict mapping approach name to BootstrapResult
        - country_samples: np.ndarray of shape (n_bootstrap, n_countries) with
          the original country indices selected in each bootstrap iteration
        - h_T_samples: Dict mapping approach name to array of shape (n_bootstrap, n_obs)
          containing h(T) values for each observation in each bootstrap iteration.
          Empty dict if h_T_approaches is None.
        - year_samples: np.ndarray of shape (n_bootstrap, n_years) with the year
          indices selected in each bootstrap iteration. Empty array if sample_years=False.
    """
    # Handle default for loess_window
    if loess_window is None:
        loess_window = DEFAULT_LOESS_WINDOW_YEARS

    rng = np.random.default_rng(random_seed)
    n_countries = data.n_countries
    unique_years = sorted(set(data.year))
    n_years = len(unique_years)
    year_to_idx = {y: i for i, y in enumerate(unique_years)}

    # Get approach names from original results
    approach_names = list(original_results.keys())

    # Initialize storage for bootstrap samples
    # Store which countries were selected in each bootstrap iteration
    country_samples = np.zeros((n_bootstrap, n_countries), dtype=np.int32)

    # Store which years were selected (if sample_years=True)
    year_samples = np.zeros((n_bootstrap, n_years), dtype=np.int32) if sample_years else np.array([])

    h1_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    h2_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    T_opt_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    total_r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    # Approach-specific samples (f1, h3, h4, f2, T_dep_opt have different meanings per approach)
    f1_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # 5d: beta; 8b/8c: linear modulation
    h3_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # 6b/6c/6e: h3 (departure/trend linear)
    h4_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # 8: h4; 6b/6c/6e/8a: h4 (departure/trend quadratic)
    f2_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # 6c: T_opt_trend; 8b: quadratic modulation
    T_dep_opt_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # 6b/6c/6e: optimal departure

    # Variance decomposition samples - initialized from original results' var_decomp keys
    var_decomp_samples = {}
    for name in approach_names:
        orig = original_results[name]
        if orig.var_decomp is not None:
            var_decomp_samples[name] = {}
            for key, val in orig.var_decomp.items():
                if isinstance(val, (int, float)):
                    var_decomp_samples[name][key] = np.full(n_bootstrap, np.nan)

    # Variance attribution samples - initialized from original results' var_attrib keys
    var_attrib_samples = {}
    for name in approach_names:
        orig = original_results[name]
        if orig.var_attrib is not None:
            var_attrib_samples[name] = {}
            for key, val in orig.var_attrib.items():
                if isinstance(val, (int, float)):
                    var_attrib_samples[name][key] = np.full(n_bootstrap, np.nan)

    # Year fixed effects k(t) samples - initialized from original results
    k_samples = {
        name: {yr: np.full(n_bootstrap, np.nan) for yr in unique_years}
        for name in approach_names
    }

    # h(T) samples for selected approaches - initialized if h_T_approaches is provided
    # h_T_samples[approach] has shape (n_bootstrap, n_obs) with h(T) for each observation
    # Note: h(T) is computed for the ORIGINAL data observations using bootstrap coefficients
    h_T_samples = {}
    if h_T_approaches is not None:
        for name in h_T_approaches:
            if name in approach_names:
                h_T_samples[name] = np.full((n_bootstrap, data.n_obs), np.nan)
        # Precompute original data's LOESS trends for 6e computation
        # (needed to compute h(T) for original observations with bootstrap coefficients)
        original_year_means = compute_year_means(data)
        original_trends_loess = compute_country_trends_loess(data, original_year_means, loess_window)

        # Precompute T_loess at base year (1961) for Approach DL's pre-history assumption
        # This ensures the pre-history is based on LOESS-smoothed temperature, not actual
        T_loess_at_base_year = _get_T_loess_at_base_year(data, original_trends_loess, base_year=1961)

        # Precompute T_linear at first year for Approach DJ's pre-history assumption
        # Uses linear OLS fit to temperature for each country
        T_linear_at_first_year = compute_T_linear_at_first_year(data)

    n_successful = 0
    n_attempts = 0
    max_attempts = n_bootstrap * 10  # Safety limit to prevent infinite loops

    if verbose:
        print(f"Running cluster bootstrap with {n_bootstrap} iterations...")
        print(f"  Resampling {n_countries} countries with replacement")
        if sample_years:
            print(f"  Resampling {n_years} years with replacement (weighted fitting)")

    while n_successful < n_bootstrap and n_attempts < max_attempts:
        b = n_successful  # Current slot to fill
        n_attempts += 1

        try:
            # Sample countries with replacement
            selected_countries = rng.integers(0, n_countries, size=n_countries)
            country_samples[b, :] = selected_countries

            # Sample years if requested
            if sample_years:
                selected_years = rng.integers(0, n_years, size=n_years)
                year_samples[b, :] = selected_years

                # Compute observation weights from country/year sampling
                weights = compute_bootstrap_weights(data, selected_countries, selected_years)

                # Use weighted trend computation (original data with weights)
                boot_data = data  # Use original data
                boot_year_means = compute_year_means_weighted(data, weights)
                boot_trends = compute_country_trends_weighted(data, weights)
                boot_trends_with_k = compute_country_trends_with_k_weighted(data, boot_year_means, weights)
                boot_trends_loess = compute_country_trends_loess_weighted(data, boot_year_means, weights, loess_window)

                # Fit all methods with weighted OLS
                # Use fit_all_approaches_weighted helper
                boot_results = _fit_all_approaches_weighted(
                    data, boot_trends, weights,
                    trends_with_k=boot_trends_with_k,
                    year_means=boot_year_means,
                    trends_loess=boot_trends_loess,
                    loess_window=loess_window
                )
            else:
                # Original country-only bootstrap (observation duplication)
                boot_data = create_bootstrap_data(data, selected_countries)

                # Recompute country trends for bootstrap sample
                boot_trends = compute_country_trends(boot_data)

                # Compute year means and adjusted trends for approaches 5, 6, 7
                boot_year_means = compute_year_means(boot_data)
                boot_trends_with_k = compute_country_trends_with_k(boot_data, boot_year_means)

                # Compute LOESS trends for approaches 6-8a
                boot_trends_loess = compute_country_trends_loess(boot_data, boot_year_means, loess_window)

                # Fit all methods
                boot_results = fit_all_approaches(
                    boot_data, boot_trends,
                    trends_with_k=boot_trends_with_k,
                    year_means=boot_year_means,
                    trends_loess=boot_trends_loess
                )

            # Store results
            for name, r in boot_results.items():
                h1_samples[name][b] = r.h1
                h2_samples[name][b] = r.h2
                # T_opt may not exist for all approaches (e.g., Approach 6c uses T_dep_opt/f2)
                T_opt_samples[name][b] = getattr(r, 'T_opt', np.nan)
                r_squared_samples[name][b] = r.r_squared
                total_r_squared_samples[name][b] = r.total_r_squared
                # Store approach-specific coefficients (f1, h3, h4, f2, T_dep_opt)
                # f1: 5d beta, 8b/8c linear modulation
                if hasattr(r, 'f1') and r.f1 is not None:
                    f1_samples[name][b] = r.f1
                # T_dep_opt: 6b/6c/6e optimal departure
                if hasattr(r, 'T_dep_opt') and r.T_dep_opt is not None:
                    T_dep_opt_samples[name][b] = r.T_dep_opt
                # h3: 6b/6c/6e linear departure/trend coefficient
                if hasattr(r, 'h3') and r.h3 is not None:
                    h3_samples[name][b] = r.h3
                # h4: 8 curvature above T_opt, 6b/6c/6e/8a quadratic trend coef
                if hasattr(r, 'h4') and r.h4 is not None:
                    h4_samples[name][b] = r.h4
                # f2: 6c T_opt_trend, 8b quadratic modulation
                if hasattr(r, 'f2') and r.f2 is not None:
                    f2_samples[name][b] = r.f2

                # Store variance decomposition samples
                if r.var_decomp is not None and name in var_decomp_samples:
                    for key in var_decomp_samples[name]:
                        if key in r.var_decomp:
                            var_decomp_samples[name][key][b] = r.var_decomp[key]

                # Store variance attribution samples
                if r.var_attrib is not None and name in var_attrib_samples:
                    for key in var_attrib_samples[name]:
                        if key in r.var_attrib:
                            var_attrib_samples[name][key][b] = r.var_attrib[key]

                # Store k values (year fixed effects)
                if hasattr(r, 'k') and r.k is not None:
                    for yr in unique_years:
                        if yr in r.k:
                            k_samples[name][yr][b] = r.k[yr]

            # Compute h(T) for selected approaches using ORIGINAL data observations
            # (with bootstrap coefficients) to enable tracking h(T) by (country, year)
            for name in h_T_samples:
                if name not in boot_results:
                    continue
                r = boot_results[name]

                if name in ['Approach QJ', 'Approach QP', 'Approach QL']:
                    # Standard quadratic: h(T) = h1*T + h2*T²
                    h_T_samples[name][b] = r.h1 * data.temp + r.h2 * data.temp**2

                elif name == 'Approach PL':
                    # Piecewise: h2*(T-T_opt)² if T≤T_opt else h4*(T-T_opt)²
                    T_opt = r.T_opt
                    below = data.temp <= T_opt
                    h_T_samples[name][b] = np.where(
                        below,
                        r.h2 * (data.temp - T_opt)**2,
                        r.h4 * (data.temp - T_opt)**2
                    )

                elif name == 'Approach DL':
                    # Persistence decay: h_conv(T) = h1*(T - h4*A_T_lag - correction_T) + h2*(T² - h4*A_T2_lag - correction_T2)
                    # The correction term accounts for assumed constant temperature before first year
                    # We assume pre-history temperature was T_loess_1961 (not actual T at first year)
                    # This makes the baseline consistent with using T_loess_1961 in cumulative effects
                    h4 = r.h4
                    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
                    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, T_loess_at_base_year)
                    X1 = data.temp - h4 * A_T_lag - correction_T
                    X2 = data.temp**2 - h4 * A_T2_lag - correction_T2
                    h_T_samples[name][b] = r.h1 * X1 + r.h2 * X2

                elif name == 'Approach PJ':
                    # Piecewise conjoined: same formula as Approach PL
                    T_opt = r.T_opt
                    below = data.temp <= T_opt
                    h_T_samples[name][b] = np.where(
                        below,
                        r.h2 * (data.temp - T_opt)**2,
                        r.h4 * (data.temp - T_opt)**2
                    )

                elif name == 'Approach DJ':
                    # Persistence decay conjoined: same formula as Approach DL
                    # Use T_linear_at_first_year for pre-history correction (linear OLS fit baseline)
                    h4 = r.h4
                    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
                    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, T_linear_at_first_year)
                    X1 = data.temp - h4 * A_T_lag - correction_T
                    X2 = data.temp**2 - h4 * A_T2_lag - correction_T2
                    h_T_samples[name][b] = r.h1 * X1 + r.h2 * X2

                elif name == 'Approach PP':
                    # Piecewise with linear detrend: same formula as Approach PL/Approach PJ
                    T_opt = r.T_opt
                    below = data.temp <= T_opt
                    h_T_samples[name][b] = np.where(
                        below,
                        r.h2 * (data.temp - T_opt)**2,
                        r.h4 * (data.temp - T_opt)**2
                    )

                elif name == 'Approach DP':
                    # Persistence with linear detrend: same formula as Approach DL/Approach DJ
                    # Use T_linear_at_first_year for pre-history correction
                    h4 = r.h4
                    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
                    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, T_linear_at_first_year)
                    X1 = data.temp - h4 * A_T_lag - correction_T
                    X2 = data.temp**2 - h4 * A_T2_lag - correction_T2
                    h_T_samples[name][b] = r.h1 * X1 + r.h2 * X2

            n_successful += 1

        except Exception as e:
            if verbose:
                print(f"  Bootstrap attempt {n_attempts} failed: {e} (retrying...)")
            # Don't increment n_successful; the slot will be reused

        # Progress reporting
        if verbose and n_successful > 0 and n_successful % 10 == 0 and n_successful != (n_successful - 1):
            # Only print when we cross a multiple of 10
            pass
        if verbose and n_attempts % 10 == 0:
            n_failed = n_attempts - n_successful
            print(f"  Completed {n_successful}/{n_bootstrap} iterations "
                  f"({n_failed} failed attempts)", flush=True)

    if verbose:
        n_failed = n_attempts - n_successful
        if n_failed > 0:
            print(f"  Bootstrap complete: {n_successful}/{n_bootstrap} successful iterations "
                  f"({n_failed} failed attempts retried)")
        else:
            print(f"  Bootstrap complete: {n_successful}/{n_bootstrap} successful iterations")

    # For Approach QJ and Approach NJ, detrend k_samples by subtracting best-fit quadratic
    # from each bootstrap. This removes the arbitrary quadratic that can shift between
    # bootstrap samples due to different country identification constraints.
    # These approaches set the first country's j terms to zero, which means k(t) can
    # absorb any arbitrary quadratic; different bootstrap samples have different countries
    # as "first", causing systematic quadratic shifts in k(t).
    approaches_to_detrend = ['Approach QJ', 'Approach NJ']
    years_array = np.array(unique_years, dtype=float)
    # Center years for numerical stability
    year_center = years_array.mean()
    years_centered = years_array - year_center
    # Design matrix for quadratic fit (same for all)
    X_quad = np.column_stack([np.ones(len(years_centered)),
                               years_centered,
                               years_centered**2])

    for approach_name in approaches_to_detrend:
        if approach_name not in k_samples:
            continue
        for b in range(n_bootstrap):
            # Extract k values for this bootstrap iteration
            k_vals = np.array([k_samples[approach_name][yr][b] for yr in unique_years])
            if np.any(np.isnan(k_vals)):
                continue
            # Fit quadratic: k(t) = a0 + a1*t + a2*t^2 and subtract
            coeffs, _, _, _ = np.linalg.lstsq(X_quad, k_vals, rcond=None)
            k_fitted = X_quad @ coeffs
            for i, yr in enumerate(unique_years):
                k_samples[approach_name][yr][b] = k_vals[i] - k_fitted[i]

    # Also detrend k_point for Approach QJ and Approach NJ to match the detrended samples
    k_point_detrended = {}
    for approach_name in approaches_to_detrend:
        if approach_name in original_results and original_results[approach_name].k is not None:
            orig_k = original_results[approach_name].k
            k_vals = np.array([orig_k[yr] for yr in unique_years])
            coeffs, _, _, _ = np.linalg.lstsq(X_quad, k_vals, rcond=None)
            k_fitted = X_quad @ coeffs
            k_point_detrended[approach_name] = {
                yr: k_vals[i] - k_fitted[i] for i, yr in enumerate(unique_years)
            }

    # Build BootstrapResult for each approach
    results = {}
    for name in approach_names:
        orig = original_results[name]
        # Get approach-specific point estimates (f1, h3, h4, f2, T_dep_opt have different meanings per approach)
        f1_point = getattr(orig, 'f1', None)
        h3_point = getattr(orig, 'h3', None)
        h4_point = getattr(orig, 'h4', None)
        f2_point = getattr(orig, 'f2', None)
        T_dep_opt_point = getattr(orig, 'T_dep_opt', None)

        # Use detrended k_point for Approach QJ and Approach NJ
        if name in k_point_detrended:
            k_point_to_use = k_point_detrended[name]
        else:
            k_point_to_use = orig.k

        # T_opt may not exist for all approaches (e.g., Approach 6c uses T_dep_opt/f2 instead)
        T_opt_point = getattr(orig, 'T_opt', None)

        results[name] = BootstrapResult(
            approach=orig.approach,
            h1_point=orig.h1,
            h2_point=orig.h2,
            T_opt_point=T_opt_point,
            r_squared_point=orig.r_squared,
            total_r_squared_point=orig.total_r_squared,
            h1_samples=h1_samples[name],
            h2_samples=h2_samples[name],
            T_opt_samples=T_opt_samples[name],
            r_squared_samples=r_squared_samples[name],
            total_r_squared_samples=total_r_squared_samples[name],
            n_bootstrap=n_bootstrap,
            n_successful=n_successful,
            f1_point=f1_point,
            f1_samples=f1_samples[name],
            h3_point=h3_point,
            h3_samples=h3_samples[name],
            h4_point=h4_point,
            h4_samples=h4_samples[name],
            f2_point=f2_point,
            f2_samples=f2_samples[name],
            T_dep_opt_point=T_dep_opt_point,
            T_dep_opt_samples=T_dep_opt_samples[name],
            var_decomp_point=getattr(orig, 'var_decomp', None),
            var_decomp_samples=var_decomp_samples.get(name, None),
            var_attrib_point=getattr(orig, 'var_attrib', None),
            var_attrib_samples=var_attrib_samples.get(name, None),
            k_point=k_point_to_use,
            k_samples=k_samples[name],
        )

    return results, country_samples, h_T_samples, year_samples


def compute_bootstrap_statistics(
    result: BootstrapResult,
    percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES
) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics from bootstrap samples.

    Args:
        result: BootstrapResult for a single approach
        percentiles: Percentiles to compute (default: 5, 25, 50, 75, 95)

    Returns:
        Dict with keys like:
        - 'h1': {'p5': ..., 'p25': ..., 'p50': ..., 'p75': ..., 'p95': ..., 'point': ...}
        - 'h2': {...}
        - 'T_opt': {...}
        - 'r_squared': {...}
        - 'total_r_squared': {...}
    """
    stats = {}

    # Helper function to compute percentiles for an array
    def get_percentile_stats(samples: np.ndarray, point_estimate: float) -> Dict[str, float]:
        # Exclude NaN values
        valid = samples[~np.isnan(samples)]
        if len(valid) == 0:
            # Return complete dict with NaN values for consistency
            result_dict = {'point': point_estimate, 'std': np.nan, 'n_valid': 0}
            for p in percentiles:
                result_dict[f'p{int(p)}'] = np.nan
            return result_dict

        result_dict = {'point': point_estimate}
        for p in percentiles:
            result_dict[f'p{int(p)}'] = np.percentile(valid, p)
        result_dict['std'] = np.std(valid)
        result_dict['n_valid'] = len(valid)
        return result_dict

    stats['h1'] = get_percentile_stats(result.h1_samples, result.h1_point)
    stats['h2'] = get_percentile_stats(result.h2_samples, result.h2_point)
    stats['T_opt'] = get_percentile_stats(result.T_opt_samples, result.T_opt_point)
    stats['r_squared'] = get_percentile_stats(result.r_squared_samples, result.r_squared_point)
    stats['total_r_squared'] = get_percentile_stats(result.total_r_squared_samples, result.total_r_squared_point)

    # Add approach-specific coefficient statistics (f1, h3, h4, f2, T_dep_opt have different meanings per approach)
    if result.f1_point is not None and result.f1_samples is not None:
        stats['f1'] = get_percentile_stats(result.f1_samples, result.f1_point)
    if result.h3_point is not None and result.h3_samples is not None:
        stats['h3'] = get_percentile_stats(result.h3_samples, result.h3_point)
    if result.h4_point is not None and result.h4_samples is not None:
        stats['h4'] = get_percentile_stats(result.h4_samples, result.h4_point)
    if result.f2_point is not None and result.f2_samples is not None:
        stats['f2'] = get_percentile_stats(result.f2_samples, result.f2_point)
    if result.T_dep_opt_point is not None and result.T_dep_opt_samples is not None:
        stats['T_dep_opt'] = get_percentile_stats(result.T_dep_opt_samples, result.T_dep_opt_point)

    # Variance decomposition statistics
    if result.var_decomp_point is not None and result.var_decomp_samples is not None:
        for key, samples in result.var_decomp_samples.items():
            if isinstance(samples, np.ndarray):
                point_val = result.var_decomp_point.get(key, np.nan)
                if isinstance(point_val, (int, float)):
                    stats[f'vd_{key}'] = get_percentile_stats(samples, point_val)

    return stats


def compute_ApproachDL_filtered_statistics(
    result: BootstrapResult,
    h4_threshold: float = 0.001,
    percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap statistics for Approach DL filtered to h4 > threshold.

    When h4 ≈ 0, Approach DL behaves like Approach QL (no persistence), so filtering
    to h4 > threshold represents cases where persistence decay is genuinely estimated.

    Args:
        result: BootstrapResult for Approach DL
        h4_threshold: Minimum h4 value to include (default: 0.001)
        percentiles: Percentiles to compute (default: 5, 25, 50, 75, 95)

    Returns:
        Dict with same structure as compute_bootstrap_statistics() plus:
        - 'n_filtered': count of samples passing the filter
        - 'filter_fraction': fraction of samples passing the filter
    """
    # Create mask for h4 > threshold
    h4_mask = result.h4_samples > h4_threshold
    n_filtered = np.sum(h4_mask)

    # Helper function to compute percentiles for filtered array
    def get_percentile_stats(samples: np.ndarray, point_estimate: float, mask: np.ndarray) -> Dict[str, float]:
        filtered = samples[mask]
        valid = filtered[~np.isnan(filtered)]
        if len(valid) == 0:
            result_dict = {'point': point_estimate, 'std': np.nan, 'n_valid': 0}
            for p in percentiles:
                result_dict[f'p{int(p)}'] = np.nan
            return result_dict

        result_dict = {'point': point_estimate}
        for p in percentiles:
            result_dict[f'p{int(p)}'] = np.percentile(valid, p)
        result_dict['std'] = np.std(valid)
        result_dict['n_valid'] = len(valid)
        return result_dict

    stats = {}

    # Core parameters
    stats['h1'] = get_percentile_stats(result.h1_samples, result.h1_point, h4_mask)
    stats['h2'] = get_percentile_stats(result.h2_samples, result.h2_point, h4_mask)
    stats['T_opt'] = get_percentile_stats(result.T_opt_samples, result.T_opt_point, h4_mask)
    stats['r_squared'] = get_percentile_stats(result.r_squared_samples, result.r_squared_point, h4_mask)
    stats['total_r_squared'] = get_percentile_stats(result.total_r_squared_samples, result.total_r_squared_point, h4_mask)

    # h4 (persistence decay)
    if result.h4_point is not None and result.h4_samples is not None:
        stats['h4'] = get_percentile_stats(result.h4_samples, result.h4_point, h4_mask)

    # Metadata
    stats['n_filtered'] = n_filtered
    stats['filter_fraction'] = n_filtered / len(result.h4_samples)

    return stats
