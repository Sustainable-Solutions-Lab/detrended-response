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
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
    DEFAULT_LOESS_WINDOW_YEARS,
)
from .fitting import (
    fit_all_approaches,
    FitResult,
    compute_persistence_accumulators,
    compute_persistence_accumulators_at_T,
    compute_pre_first_year_correction,
)


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


def run_bootstrap(
    data: AnalysisData,
    trends: CountryTrends,
    original_results: Dict[str, FitResult],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = True,
    loess_window: int = None,
    h_T_approaches: list = None,
) -> Tuple[Dict[str, BootstrapResult], np.ndarray, Dict[str, np.ndarray]]:
    """Run bootstrap analysis for all methods.

    For each bootstrap iteration:
    1. Sample M countries with replacement
    2. Create bootstrap dataset
    3. Recompute country trends for bootstrap sample
    4. Fit all methods
    5. Store h1, h2, T_opt, R², Total R², and h4 (for method-specific coefficients)
    6. Optionally compute h(T) for selected methods

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
            Example: ['approach0', 'approach1', 'approach2', 'method4', 'approach3']

    Returns:
        Tuple of:
        - Dict mapping approach name to BootstrapResult
        - country_samples: np.ndarray of shape (n_bootstrap, n_countries) with
          the original country indices selected in each bootstrap iteration
        - h_T_samples: Dict mapping approach name to array of shape (n_bootstrap, n_obs)
          containing h(T) values for each observation in each bootstrap iteration.
          Empty dict if h_T_approaches is None.
    """
    # Handle default for loess_window
    if loess_window is None:
        loess_window = DEFAULT_LOESS_WINDOW_YEARS

    rng = np.random.default_rng(random_seed)
    n_countries = data.n_countries

    # Get approach names from original results
    approach_names = list(original_results.keys())

    # Initialize storage for bootstrap samples
    # Store which countries were selected in each bootstrap iteration
    country_samples = np.zeros((n_bootstrap, n_countries), dtype=np.int32)

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
    # Get unique years from original data
    unique_years = sorted(set(data.year))
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

    n_successful = 0
    n_attempts = 0
    max_attempts = n_bootstrap * 10  # Safety limit to prevent infinite loops

    if verbose:
        print(f"Running cluster bootstrap with {n_bootstrap} iterations...")
        print(f"  Resampling {n_countries} countries with replacement")

    while n_successful < n_bootstrap and n_attempts < max_attempts:
        b = n_successful  # Current slot to fill
        n_attempts += 1

        try:
            # Sample countries with replacement
            selected_countries = rng.integers(0, n_countries, size=n_countries)
            country_samples[b, :] = selected_countries

            # Create bootstrap dataset
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

                if name in ['approach0', 'approach1', 'approach2']:
                    # Standard quadratic: h(T) = h1*T + h2*T²
                    h_T_samples[name][b] = r.h1 * data.temp + r.h2 * data.temp**2

                elif name == 'method4':
                    # Full model: h(T,Ttrend) = h1*T + h2*T² + h4*(T-Ttrend)²
                    # Use original data's Ttrend to compute for original observations
                    Ttrend = original_trends_loess.T_loess
                    h_T_samples[name][b] = (r.h1 * data.temp + r.h2 * data.temp**2
                                            + r.h4 * (data.temp - Ttrend)**2)

                elif name == 'approach3':
                    # Piecewise: h2*(T-T_opt)² if T≤T_opt else h4*(T-T_opt)²
                    T_opt = r.T_opt
                    below = data.temp <= T_opt
                    h_T_samples[name][b] = np.where(
                        below,
                        r.h2 * (data.temp - T_opt)**2,
                        r.h4 * (data.temp - T_opt)**2
                    )

                elif name == 'approach4':
                    # Persistence decay: h_conv(T) = h1*(T - h4*A_T_lag - correction_T) + h2*(T² - h4*A_T2_lag - correction_T2)
                    # The correction term accounts for assumed constant temperature before first year
                    # Store h_conv(T) without trend subtraction (like approach2)
                    # The cumulative effects script handles trend subtraction separately
                    h4 = r.h4
                    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
                    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, data.temp)
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

    # For approach0 and approach0h0, detrend k_samples by subtracting best-fit quadratic
    # from each bootstrap. This removes the arbitrary quadratic that can shift between
    # bootstrap samples due to different country identification constraints.
    # These approaches set the first country's j terms to zero, which means k(t) can
    # absorb any arbitrary quadratic; different bootstrap samples have different countries
    # as "first", causing systematic quadratic shifts in k(t).
    approaches_to_detrend = ['approach0', 'approach0h0']
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

    # Also detrend k_point for approach0 and approach0h0 to match the detrended samples
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

        # Use detrended k_point for approach0 and approach0h0
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

    return results, country_samples, h_T_samples


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


def compute_approach4_filtered_statistics(
    result: BootstrapResult,
    h4_threshold: float = 0.001,
    percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap statistics for approach4 filtered to h4 > threshold.

    When h4 ≈ 0, approach4 behaves like approach2 (no persistence), so filtering
    to h4 > threshold represents cases where persistence decay is genuinely estimated.

    Args:
        result: BootstrapResult for approach4
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
