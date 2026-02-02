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

    # Bootstrap samples - shape (n_bootstrap,)
    h1_samples: np.ndarray
    h2_samples: np.ndarray
    T_optimal_samples: np.ndarray
    r_squared_samples: np.ndarray
    total_r_squared_samples: np.ndarray

    # Metadata
    n_bootstrap: int
    n_successful: int  # Number of successful fits

    # Approach 8 specific (optional)
    beta_point: float = None       # GDP scaling exponent (for Approach 8)
    beta_samples: np.ndarray = None  # Bootstrap samples for beta


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
) -> Dict[str, BootstrapResult]:
    """Run bootstrap analysis for all approaches.

    For each bootstrap iteration:
    1. Sample M countries with replacement
    2. Create bootstrap dataset
    3. Recompute country trends for bootstrap sample
    4. Fit all approaches (including Approach 8, 9, 10 if Y_ref provided)
    5. Store h1, h2, T_optimal, R², Total R², and beta (for Approaches 8 and 10)

    Args:
        data: Original AnalysisData
        trends: Original CountryTrends (used for point estimates)
        original_results: Dict of original FitResult for each approach
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        Y_ref: Reference GDP for Approach 8 and 10 (computed once on full dataset)
        loess_window: Window size in years for LOESS smoothing
            (default: DEFAULT_LOESS_WINDOW_YEARS)

    Returns:
        Dict mapping approach name to BootstrapResult
    """
    # Handle default for loess_window
    if loess_window is None:
        loess_window = DEFAULT_LOESS_WINDOW_YEARS

    rng = np.random.default_rng(random_seed)
    n_countries = data.n_countries

    # Get approach names from original results
    approach_names = list(original_results.keys())

    # Initialize storage for bootstrap samples
    h1_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    h2_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    T_optimal_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    total_r_squared_samples = {name: np.zeros(n_bootstrap) for name in approach_names}
    beta_samples = {name: np.full(n_bootstrap, np.nan) for name in approach_names}  # For Approach 8 and 10

    n_successful = 0

    if verbose:
        print(f"Running cluster bootstrap with {n_bootstrap} iterations...")
        print(f"  Resampling {n_countries} countries with replacement")

    for b in range(n_bootstrap):
        try:
            # Sample countries with replacement
            selected_countries = rng.integers(0, n_countries, size=n_countries)

            # Create bootstrap dataset
            boot_data = create_bootstrap_data(data, selected_countries)

            # Recompute country trends for bootstrap sample
            boot_trends = compute_country_trends(boot_data)

            # Compute year means and adjusted trends for approaches 6, 7, 9, 10
            boot_year_means = compute_year_means(boot_data)
            boot_trends_with_k = compute_country_trends_with_k(boot_data, boot_year_means)

            # Compute LOESS trends for approaches 9 and 10
            boot_trends_loess = compute_country_trends_loess(boot_data, boot_year_means, loess_window)

            # Fit all approaches (pass Y_ref for Approach 8 and 10)
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
                # Store beta for Approach 8
                if hasattr(r, 'beta'):
                    beta_samples[name][b] = r.beta

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

        # Progress reporting
        if verbose and (b + 1) % 10 == 0:
            print(f"  Completed {b + 1}/{n_bootstrap} iterations "
                  f"({n_successful} successful)", flush=True)

    if verbose:
        print(f"  Bootstrap complete: {n_successful}/{n_bootstrap} successful iterations")

    # Build BootstrapResult for each approach
    results = {}
    for name in approach_names:
        orig = original_results[name]
        # Get beta point estimate if available (Approach 8)
        beta_point = getattr(orig, 'beta', None)
        results[name] = BootstrapResult(
            approach=orig.approach,
            h1_point=orig.h1,
            h2_point=orig.h2,
            T_optimal_point=orig.T_optimal,
            h1_samples=h1_samples[name],
            h2_samples=h2_samples[name],
            T_optimal_samples=T_optimal_samples[name],
            r_squared_samples=r_squared_samples[name],
            total_r_squared_samples=total_r_squared_samples[name],
            n_bootstrap=n_bootstrap,
            n_successful=n_successful,
            beta_point=beta_point,
            beta_samples=beta_samples[name],
        )

    return results


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
            return {f'p{int(p)}': np.nan for p in percentiles}

        result_dict = {'point': point_estimate}
        for p in percentiles:
            result_dict[f'p{int(p)}'] = np.percentile(valid, p)
        result_dict['std'] = np.std(valid)
        result_dict['n_valid'] = len(valid)
        return result_dict

    stats['h1'] = get_percentile_stats(result.h1_samples, result.h1_point)
    stats['h2'] = get_percentile_stats(result.h2_samples, result.h2_point)
    stats['T_optimal'] = get_percentile_stats(result.T_optimal_samples, result.T_optimal_point)
    stats['r_squared'] = get_percentile_stats(result.r_squared_samples, np.nan)
    stats['total_r_squared'] = get_percentile_stats(result.total_r_squared_samples, np.nan)

    # Add beta statistics if present (Approach 8)
    if result.beta_point is not None and result.beta_samples is not None:
        stats['beta'] = get_percentile_stats(result.beta_samples, result.beta_point)

    return stats
