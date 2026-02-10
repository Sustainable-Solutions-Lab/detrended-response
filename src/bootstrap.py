"""Country-level bootstrap resampling for uncertainty quantification.

This module implements cluster bootstrap (resampling countries with replacement)
for computing uncertainty estimates on h1, h2, and T_optimal for all 8 approaches.

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
from .fitting import fit_all_approaches, FitResult


@dataclass
class BootstrapResult:
    """Container for bootstrap results for a single approach."""
    approach: str

    # Point estimates (from original fit)
    h1_point: float
    h2_point: float
    T_optimal_point: float
    r_squared_point: float
    total_r_squared_point: float

    # Bootstrap samples - shape (n_bootstrap,)
    h1_samples: np.ndarray
    h2_samples: np.ndarray
    T_optimal_samples: np.ndarray
    r_squared_samples: np.ndarray
    total_r_squared_samples: np.ndarray

    # Metadata
    n_bootstrap: int
    n_successful: int  # Number of successful fits

    # Approach 7 specific (optional)
    beta_point: float = None       # GDP scaling exponent
    beta_samples: np.ndarray = None  # Bootstrap samples for beta

    # Approach 8/8a (piecewise quadratic / shared T_opt) specific (optional)
    h2_low_point: float = None     # Curvature for T ≤ T_opt
    h2_low_samples: np.ndarray = None  # Bootstrap samples for h2_low
    h2_high_point: float = None    # Curvature for T > T_opt
    h2_high_samples: np.ndarray = None  # Bootstrap samples for h2_high

    # Approach 6a/6b (separate high/low frequency) specific (optional)
    h1_high_point: float = None    # Linear coef for high-frequency (actual T)
    h1_high_samples: np.ndarray = None
    h1_low_point: float = None     # Linear coef for low-frequency (Ttrend)
    h1_low_samples: np.ndarray = None
    T_optimal_high_point: float = None  # Optimal T for high-frequency response
    T_optimal_high_samples: np.ndarray = None
    T_optimal_low_point: float = None   # Optimal T for low-frequency response
    T_optimal_low_samples: np.ndarray = None

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
    Y_ref: float = None,
    loess_window: int = None,
) -> Tuple[Dict[str, BootstrapResult], np.ndarray]:
    """Run bootstrap analysis for all approaches.

    For each bootstrap iteration:
    1. Sample M countries with replacement
    2. Create bootstrap dataset
    3. Recompute country trends for bootstrap sample
    4. Fit all approaches (including Approach 7 if Y_ref provided)
    5. Store h1, h2, T_optimal, R², Total R², and beta (for Approach 7)

    Args:
        data: Original AnalysisData
        trends: Original CountryTrends (used for point estimates)
        original_results: Dict of original FitResult for each approach
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        Y_ref: Reference GDP for Approach 7 (computed once on full dataset)
        loess_window: Window size in years for LOESS smoothing
            (default: DEFAULT_LOESS_WINDOW_YEARS)

    Returns:
        Tuple of:
        - Dict mapping approach name to BootstrapResult
        - country_samples: np.ndarray of shape (n_bootstrap, n_countries) with
          the original country indices selected in each bootstrap iteration
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
    T_optimal_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    total_r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    beta_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # For Approach 7
    h2_low_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # For Approach 8/8a
    h2_high_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # For Approach 8/8a
    # For Approach 6a/6b (separate high/low frequency)
    h1_high_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}
    h1_low_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}
    T_optimal_high_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}
    T_optimal_low_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}

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

    n_successful = 0

    if verbose:
        print(f"Running cluster bootstrap with {n_bootstrap} iterations...")
        print(f"  Resampling {n_countries} countries with replacement")

    for b in range(n_bootstrap):
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

            # Compute LOESS trends for approaches 6 and 7
            boot_trends_loess = compute_country_trends_loess(boot_data, boot_year_means, loess_window)

            # Fit all approaches (pass Y_ref for Approach 7)
            boot_results = fit_all_approaches(
                boot_data, boot_trends,
                trends_with_k=boot_trends_with_k,
                year_means=boot_year_means,
                Y_ref=Y_ref,
                trends_loess=boot_trends_loess
            )

            # Store results
            for name, r in boot_results.items():
                h1_samples[name][b] = r.h1
                h2_samples[name][b] = r.h2
                T_optimal_samples[name][b] = r.T_optimal
                r_squared_samples[name][b] = r.r_squared
                total_r_squared_samples[name][b] = r.total_r_squared
                # Store beta for Approach 7
                if hasattr(r, 'beta') and r.beta is not None:
                    beta_samples[name][b] = r.beta
                # Store h2_low and h2_high for Approach 8/8a (piecewise quadratic / shared T_opt)
                if hasattr(r, 'h2_low') and r.h2_low is not None:
                    h2_low_samples[name][b] = r.h2_low
                if hasattr(r, 'h2_high') and r.h2_high is not None:
                    h2_high_samples[name][b] = r.h2_high
                # Store h1_high, h1_low, T_optimal_high, T_optimal_low for Approach 6a/6b
                if hasattr(r, 'h1_high'):
                    h1_high_samples[name][b] = r.h1_high
                if hasattr(r, 'h1_low'):
                    h1_low_samples[name][b] = r.h1_low
                if hasattr(r, 'T_optimal_high'):
                    T_optimal_high_samples[name][b] = r.T_optimal_high
                if hasattr(r, 'T_optimal_low'):
                    T_optimal_low_samples[name][b] = r.T_optimal_low

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

            n_successful += 1

        except Exception as e:
            if verbose:
                print(f"  Bootstrap {b} failed: {e}")
            # Mark as NaN
            for name in approach_names:
                h1_samples[name][b] = np.nan
                h2_samples[name][b] = np.nan
                T_optimal_samples[name][b] = np.nan
                r_squared_samples[name][b] = np.nan
                total_r_squared_samples[name][b] = np.nan
                beta_samples[name][b] = np.nan
                h2_low_samples[name][b] = np.nan
                h2_high_samples[name][b] = np.nan
                h1_high_samples[name][b] = np.nan
                h1_low_samples[name][b] = np.nan
                T_optimal_high_samples[name][b] = np.nan
                T_optimal_low_samples[name][b] = np.nan

        # Progress reporting
        if verbose and (b + 1) % 10 == 0:
            print(f"  Completed {b + 1}/{n_bootstrap} iterations "
                  f"({n_successful} successful)", flush=True)

    if verbose:
        print(f"  Bootstrap complete: {n_successful}/{n_bootstrap} successful iterations")

    # For approach0 and nocr0, detrend k_samples by subtracting best-fit quadratic
    # from each bootstrap. This removes the arbitrary quadratic that can shift between
    # bootstrap samples due to different country identification constraints.
    # These approaches set the first country's j terms to zero, which means k(t) can
    # absorb any arbitrary quadratic; different bootstrap samples have different countries
    # as "first", causing systematic quadratic shifts in k(t).
    approaches_to_detrend = ['approach0', 'nocr0']
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

    # Also detrend k_point for approach0 and nocr0 to match the detrended samples
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
        # Get beta point estimate if available (Approach 7)
        beta_point = getattr(orig, 'beta', None)
        # Get h2_low and h2_high point estimates if available (Approach 8/8a piecewise)
        h2_low_point = getattr(orig, 'h2_low', None)
        h2_high_point = getattr(orig, 'h2_high', None)
        # Get h1_high, h1_low, T_optimal_high, T_optimal_low for Approach 6a/6b
        h1_high_point = getattr(orig, 'h1_high', None)
        h1_low_point = getattr(orig, 'h1_low', None)
        T_optimal_high_point = getattr(orig, 'T_optimal_high', None)
        T_optimal_low_point = getattr(orig, 'T_optimal_low', None)

        # Use detrended k_point for approach0 and nocr0
        if name in k_point_detrended:
            k_point_to_use = k_point_detrended[name]
        else:
            k_point_to_use = orig.k

        results[name] = BootstrapResult(
            approach=orig.approach,
            h1_point=orig.h1,
            h2_point=orig.h2,
            T_optimal_point=orig.T_optimal,
            r_squared_point=orig.r_squared,
            total_r_squared_point=orig.total_r_squared,
            h1_samples=h1_samples[name],
            h2_samples=h2_samples[name],
            T_optimal_samples=T_optimal_samples[name],
            r_squared_samples=r_squared_samples[name],
            total_r_squared_samples=total_r_squared_samples[name],
            n_bootstrap=n_bootstrap,
            n_successful=n_successful,
            beta_point=beta_point,
            beta_samples=beta_samples[name],
            h2_low_point=h2_low_point,
            h2_low_samples=h2_low_samples[name],
            h2_high_point=h2_high_point,
            h2_high_samples=h2_high_samples[name],
            h1_high_point=h1_high_point,
            h1_high_samples=h1_high_samples[name],
            h1_low_point=h1_low_point,
            h1_low_samples=h1_low_samples[name],
            T_optimal_high_point=T_optimal_high_point,
            T_optimal_high_samples=T_optimal_high_samples[name],
            T_optimal_low_point=T_optimal_low_point,
            T_optimal_low_samples=T_optimal_low_samples[name],
            var_decomp_point=getattr(orig, 'var_decomp', None),
            var_decomp_samples=var_decomp_samples.get(name, None),
            var_attrib_point=getattr(orig, 'var_attrib', None),
            var_attrib_samples=var_attrib_samples.get(name, None),
            k_point=k_point_to_use,
            k_samples=k_samples[name],
        )

    return results, country_samples


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
        - 'T_optimal': {...}
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
    stats['T_optimal'] = get_percentile_stats(result.T_optimal_samples, result.T_optimal_point)
    stats['r_squared'] = get_percentile_stats(result.r_squared_samples, result.r_squared_point)
    stats['total_r_squared'] = get_percentile_stats(result.total_r_squared_samples, result.total_r_squared_point)

    # Add beta statistics if present (Approach 7)
    if result.beta_point is not None and result.beta_samples is not None:
        stats['beta'] = get_percentile_stats(result.beta_samples, result.beta_point)

    # Add h2_low and h2_high statistics if present (Approach 8/8a piecewise)
    if result.h2_low_point is not None and result.h2_low_samples is not None:
        stats['h2_low'] = get_percentile_stats(result.h2_low_samples, result.h2_low_point)
    if result.h2_high_point is not None and result.h2_high_samples is not None:
        stats['h2_high'] = get_percentile_stats(result.h2_high_samples, result.h2_high_point)

    # Add h1_high, h1_low, T_optimal_high, T_optimal_low statistics if present (Approach 6a/6b)
    if result.h1_high_point is not None and result.h1_high_samples is not None:
        stats['h1_high'] = get_percentile_stats(result.h1_high_samples, result.h1_high_point)
    if result.h1_low_point is not None and result.h1_low_samples is not None:
        stats['h1_low'] = get_percentile_stats(result.h1_low_samples, result.h1_low_point)
    if result.T_optimal_high_point is not None and result.T_optimal_high_samples is not None:
        stats['T_optimal_high'] = get_percentile_stats(result.T_optimal_high_samples, result.T_optimal_high_point)
    if result.T_optimal_low_point is not None and result.T_optimal_low_samples is not None:
        stats['T_optimal_low'] = get_percentile_stats(result.T_optimal_low_samples, result.T_optimal_low_point)

    # Variance decomposition statistics
    if result.var_decomp_point is not None and result.var_decomp_samples is not None:
        for key, samples in result.var_decomp_samples.items():
            if isinstance(samples, np.ndarray):
                point_val = result.var_decomp_point.get(key, np.nan)
                if isinstance(point_val, (int, float)):
                    stats[f'vd_{key}'] = get_percentile_stats(samples, point_val)

    return stats
