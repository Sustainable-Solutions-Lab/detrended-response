"""Country influence analysis for bootstrap coefficient estimates.

This module identifies which countries systematically skew bootstrap coefficient
estimates upward or downward by analyzing the relationship between country
resampling frequencies and coefficient values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import pycountry


# Default coefficients to analyze for each approach type
STANDARD_COEFFICIENTS = ['h1', 'h2', 'T_optimal']
APPROACH_COEFFICIENTS = {
    # Standard approaches (0-5, 5a-5d, 6, nocr0, nocr5)
    'approach0': STANDARD_COEFFICIENTS,
    'approach1': STANDARD_COEFFICIENTS,
    'approach2': STANDARD_COEFFICIENTS,
    'approach3': STANDARD_COEFFICIENTS,
    'approach4': STANDARD_COEFFICIENTS,
    'approach5': STANDARD_COEFFICIENTS,
    'approach5a': STANDARD_COEFFICIENTS,
    'approach5b': STANDARD_COEFFICIENTS,
    'approach5c': STANDARD_COEFFICIENTS,
    'approach5d': STANDARD_COEFFICIENTS,
    'approach6': STANDARD_COEFFICIENTS,
    'nocr0': STANDARD_COEFFICIENTS,
    'nocr5': STANDARD_COEFFICIENTS,
    # Approach 6a/6b: total/trend variants
    'approach6a': ['h1_total', 'h2_total', 'h1_trend', 'h2_trend', 'T_optimal_total', 'T_optimal_trend'],
    'approach6b': ['h1_total', 'h2_total', 'h1_trend', 'h2_trend', 'T_optimal_total', 'T_optimal_trend'],
    # Approach 8: piecewise quadratic
    'approach8': ['h2_low', 'h2_high', 'T_optimal'],
    # Approach 8a: total/trend (shared T_optimal)
    'approach8a': ['h2_total', 'h2_trend', 'T_optimal'],
}


@dataclass
class CountryInfluenceResult:
    """Container for country influence analysis results."""
    approach: str
    coefficient: str
    percentile: int
    threshold_value: float
    n_valid_iterations: int
    n_above_threshold: int
    country_codes: List[str]  # 157 ISO codes
    influence_coefficients: np.ndarray  # 157 regression coefficients
    regression_type: str
    model_score: float


def get_country_name(iso_code: str) -> str:
    """Get full country name from ISO3 code.

    Returns the ISO code itself if lookup fails.
    """
    try:
        country = pycountry.countries.get(alpha_3=iso_code)
        if country:
            return country.name
    except (LookupError, AttributeError):
        pass
    return iso_code


def compute_country_counts(country_samples: np.ndarray, n_countries: int) -> np.ndarray:
    """Convert country indices to count matrix.

    Parameters
    ----------
    country_samples : np.ndarray
        Array of shape (n_bootstrap, n_slots) containing country indices.
        Each row is a bootstrap iteration, each column is a 'slot' in the
        resampled dataset. Values are original country indices (0 to n_countries-1).
    n_countries : int
        Number of unique countries.

    Returns
    -------
    np.ndarray
        Array of shape (n_bootstrap, n_countries) with counts per iteration.
        count[i, j] = number of times country j appears in bootstrap iteration i.
    """
    n_bootstrap = country_samples.shape[0]
    counts = np.zeros((n_bootstrap, n_countries), dtype=np.int32)

    for i in range(n_bootstrap):
        # Count occurrences of each country index in this iteration
        unique, count_vals = np.unique(country_samples[i], return_counts=True)
        for idx, cnt in zip(unique, count_vals):
            if 0 <= idx < n_countries:
                counts[i, idx] = cnt

    return counts


def compute_percentile_indicators(
    values: np.ndarray,
    percentiles: Tuple[int, ...] = (5, 25, 75, 95)
) -> Dict[int, Tuple[np.ndarray, float]]:
    """Create binary indicators for exceeding percentile thresholds.

    Parameters
    ----------
    values : np.ndarray
        1D array of coefficient values (one per bootstrap iteration).
    percentiles : tuple of int
        Percentile thresholds to compute.

    Returns
    -------
    dict
        {percentile: (binary_array, threshold_value)}
        binary_array[i] = 1 if values[i] > threshold, 0 otherwise
    """
    result = {}
    for pct in percentiles:
        threshold = np.percentile(values, pct)
        binary = (values > threshold).astype(np.int32)
        result[pct] = (binary, threshold)
    return result


def fit_influence_regression(
    country_counts: np.ndarray,
    binary_indicator: np.ndarray,
    regression_type: str = "linear"
) -> Tuple[np.ndarray, float]:
    """Fit regression to predict threshold exceedance from country counts.

    Parameters
    ----------
    country_counts : np.ndarray
        Array of shape (n_bootstrap, n_countries) with country counts.
    binary_indicator : np.ndarray
        1D array of binary indicators (1 if above threshold, 0 otherwise).
    regression_type : str
        "linear" for Linear Probability Model (only option, logistic not implemented).

    Returns
    -------
    tuple
        (coefficients_per_country, model_score)
        coefficients: shape (n_countries,)
        score: R^2
    """
    if regression_type != "linear":
        raise ValueError(
            f"Only 'linear' regression is supported (got '{regression_type}'). "
            "Install scikit-learn for logistic regression support."
        )

    # Linear regression using numpy least squares
    # Add intercept column
    X = np.column_stack([np.ones(len(binary_indicator)), country_counts])
    y = binary_indicator.astype(np.float64)

    # Solve least squares: X @ beta = y
    # Using lstsq for numerical stability
    result = np.linalg.lstsq(X, y, rcond=None)
    beta = result[0]

    # Extract coefficients (skip intercept)
    coefs = beta[1:]

    # Compute R^2
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return coefs, r_squared


def run_influence_analysis(
    bootstrap_dir: Path,
    approaches: Optional[List[str]] = None,
    coefficients: Optional[Dict[str, List[str]]] = None,
    percentiles: Tuple[int, ...] = (5, 25, 75, 95),
    regression_type: str = "linear"
) -> List[CountryInfluenceResult]:
    """Main analysis function for country influence on bootstrap coefficients.

    Parameters
    ----------
    bootstrap_dir : Path
        Path to bootstrap output directory containing bootstrap_coefficients.csv
        and bootstrap_country_samples.csv.
    approaches : list of str, optional
        Approaches to analyze. If None, analyzes all available approaches.
    coefficients : dict, optional
        {approach: [coefficient_names]} mapping. If None, uses defaults from
        APPROACH_COEFFICIENTS.
    percentiles : tuple of int
        Percentile thresholds to analyze.
    regression_type : str
        "linear" or "logistic" for the influence regression.

    Returns
    -------
    list of CountryInfluenceResult
        Results for each approach/coefficient/percentile combination.
    """
    bootstrap_dir = Path(bootstrap_dir)

    # Load bootstrap data
    coef_path = bootstrap_dir / "bootstrap_coefficients.csv"
    samples_path = bootstrap_dir / "bootstrap_country_samples.csv"

    if not coef_path.exists():
        raise FileNotFoundError(f"Bootstrap coefficients not found: {coef_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Country samples not found: {samples_path}")

    coef_df = pd.read_csv(coef_path, comment='#')
    samples_df = pd.read_csv(samples_path, comment='#')

    # Get country codes from column headers (skip 'iteration' column)
    country_codes = list(samples_df.columns[1:])
    n_countries = len(country_codes)

    # Convert samples to country indices array
    # The values in the CSV are already country indices
    country_samples = samples_df.drop(columns=['iteration']).values

    # Compute country counts matrix
    country_counts = compute_country_counts(country_samples, n_countries)

    # Determine approaches to analyze
    available_approaches = coef_df['approach'].unique()
    if approaches is None:
        approaches = list(available_approaches)
    else:
        # Validate requested approaches
        approaches = [a for a in approaches if a in available_approaches]

    results = []

    for approach in approaches:
        # Get coefficients to analyze for this approach
        if coefficients is not None and approach in coefficients:
            coef_names = coefficients[approach]
        elif approach in APPROACH_COEFFICIENTS:
            coef_names = APPROACH_COEFFICIENTS[approach]
        else:
            # Default to standard coefficients
            coef_names = STANDARD_COEFFICIENTS

        # Get data for this approach
        approach_df = coef_df[coef_df['approach'] == approach]

        for coef_name in coef_names:
            if coef_name not in approach_df.columns:
                continue

            # Get coefficient values aligned with iterations
            coef_values = approach_df[coef_name].values
            iterations = approach_df['iteration'].values

            # Filter out NaN values
            valid_mask = ~np.isnan(coef_values)
            n_valid = valid_mask.sum()

            if n_valid < 100:
                warnings.warn(
                    f"Only {n_valid} valid iterations for {approach}/{coef_name}. "
                    "Results may be unreliable."
                )

            if n_valid < 10:
                continue  # Skip if too few valid iterations

            valid_values = coef_values[valid_mask]
            valid_iterations = iterations[valid_mask]

            # Get corresponding country counts
            valid_counts = country_counts[valid_iterations]

            # Compute percentile indicators
            indicators = compute_percentile_indicators(valid_values, percentiles)

            for pct, (binary, threshold) in indicators.items():
                n_above = binary.sum()

                # Fit regression
                coefs, score = fit_influence_regression(
                    valid_counts, binary, regression_type
                )

                results.append(CountryInfluenceResult(
                    approach=approach,
                    coefficient=coef_name,
                    percentile=pct,
                    threshold_value=threshold,
                    n_valid_iterations=n_valid,
                    n_above_threshold=int(n_above),
                    country_codes=country_codes,
                    influence_coefficients=coefs,
                    regression_type=regression_type,
                    model_score=score,
                ))

    return results


def results_to_coefficients_df(results: List[CountryInfluenceResult]) -> pd.DataFrame:
    """Convert results to full coefficients DataFrame.

    Creates a wide-format DataFrame with one row per approach/coefficient/percentile
    and one column per country.
    """
    rows = []
    for r in results:
        row = {
            'approach': r.approach,
            'coefficient': r.coefficient,
            'percentile': r.percentile,
            'threshold_value': r.threshold_value,
            'n_valid': r.n_valid_iterations,
            'n_above': r.n_above_threshold,
            'model_score': r.model_score,
        }
        # Add country coefficients
        for code, coef in zip(r.country_codes, r.influence_coefficients):
            row[code] = coef
        rows.append(row)

    return pd.DataFrame(rows)


def results_to_rankings_df(
    results: List[CountryInfluenceResult],
    n_top: int = 10
) -> pd.DataFrame:
    """Convert results to rankings DataFrame.

    Creates a long-format DataFrame with top/bottom countries ranked by influence.
    """
    rows = []
    for r in results:
        # Sort countries by influence coefficient
        sorted_indices = np.argsort(r.influence_coefficients)[::-1]  # Descending

        for rank, idx in enumerate(sorted_indices, 1):
            code = r.country_codes[idx]
            coef = r.influence_coefficients[idx]
            direction = "increases" if coef > 0 else "decreases"

            rows.append({
                'approach': r.approach,
                'coefficient': r.coefficient,
                'percentile': r.percentile,
                'rank': rank,
                'country_iso': code,
                'country_name': get_country_name(code),
                'influence_coef': coef,
                'direction': direction,
            })

    return pd.DataFrame(rows)


def generate_summary_text(
    results: List[CountryInfluenceResult],
    n_top: int = 10
) -> str:
    """Generate human-readable summary text.

    Parameters
    ----------
    results : list of CountryInfluenceResult
    n_top : int
        Number of top/bottom countries to show per category.

    Returns
    -------
    str
        Formatted summary text.
    """
    lines = [
        "Country Influence Analysis Summary",
        "=" * 50,
        "",
    ]

    # Group by approach
    approaches = {}
    for r in results:
        if r.approach not in approaches:
            approaches[r.approach] = []
        approaches[r.approach].append(r)

    for approach in sorted(approaches.keys()):
        approach_results = approaches[approach]

        # Get approach name from first result
        lines.append(f"APPROACH: {approach}")
        lines.append("-" * 50)
        lines.append("")

        # Group by coefficient
        coefficients = {}
        for r in approach_results:
            if r.coefficient not in coefficients:
                coefficients[r.coefficient] = []
            coefficients[r.coefficient].append(r)

        for coef_name in sorted(coefficients.keys()):
            coef_results = coefficients[coef_name]
            lines.append(f"Coefficient: {coef_name}")
            lines.append("")

            # Show results for key percentiles (75th for "increases", 25th for "decreases")
            for r in sorted(coef_results, key=lambda x: x.percentile):
                sorted_indices = np.argsort(r.influence_coefficients)[::-1]

                # Countries that increase (positive coefficients, high percentiles)
                if r.percentile >= 75:
                    lines.append(f"  Countries that INCREASE {coef_name} (percentile > {r.percentile}th, threshold: {r.threshold_value:.6g}):")
                    for i, idx in enumerate(sorted_indices[:n_top]):
                        code = r.country_codes[idx]
                        coef = r.influence_coefficients[idx]
                        name = get_country_name(code)
                        lines.append(f"    {i+1:2d}. {code}  {coef:+.4f}  {name}")
                    lines.append("")

                # Countries that decrease (negative coefficients, low percentiles)
                elif r.percentile <= 25:
                    lines.append(f"  Countries that DECREASE {coef_name} (percentile > {r.percentile}th, threshold: {r.threshold_value:.6g}):")
                    for i, idx in enumerate(sorted_indices[-n_top:][::-1]):
                        code = r.country_codes[idx]
                        coef = r.influence_coefficients[idx]
                        name = get_country_name(code)
                        lines.append(f"    {i+1:2d}. {code}  {coef:+.4f}  {name}")
                    lines.append("")

            lines.append("")

        lines.append("")

    return "\n".join(lines)


def save_influence_results(
    results: List[CountryInfluenceResult],
    output_dir: Path,
    n_top: int = 10
) -> None:
    """Save all influence analysis results to files.

    Creates:
    - country_influence_coefficients.csv: Full regression coefficients
    - country_influence_rankings.csv: Simplified rankings
    - country_influence_summary.txt: Human-readable report

    Parameters
    ----------
    results : list of CountryInfluenceResult
    output_dir : Path
        Directory to save output files.
    n_top : int
        Number of top/bottom countries for summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save coefficients CSV
    coef_df = results_to_coefficients_df(results)
    coef_path = output_dir / "country_influence_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"  Saved: {coef_path}")

    # Save rankings CSV
    rank_df = results_to_rankings_df(results, n_top=157)  # Save all rankings
    rank_path = output_dir / "country_influence_rankings.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"  Saved: {rank_path}")

    # Save summary text
    summary = generate_summary_text(results, n_top=n_top)
    summary_path = output_dir / "country_influence_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")
