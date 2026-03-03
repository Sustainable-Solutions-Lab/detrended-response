"""Publication-specific formatting for tables and figures.

This module provides functions to generate publication-quality tables and figures
from pre-computed analysis and bootstrap results.
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.bootstrap import BootstrapResult
from src.data_loader import AnalysisData
from src.output import (
    plot_year_effects_2panel,
    # 3x3 and 4x3 grid plotting functions
    plot_temperature_response_3x3,
    plot_temperature_derivative_3x3,
    plot_T_optimal_histogram_3x3,
    plot_h2_histogram_4x3,
    plot_h4_histogram_1x3,
)


def reconstruct_bootstrap_results(
    coefficients_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    k_samples_df: pd.DataFrame = None,
) -> Dict[str, BootstrapResult]:
    """Reconstruct BootstrapResult objects from CSV data.

    Parameters
    ----------
    coefficients_df : pd.DataFrame
        DataFrame from bootstrap_coefficients.csv with columns:
        iteration, approach, approach_name, h1, h2, T_opt, r_squared, etc.
    summary_df : pd.DataFrame
        DataFrame from bootstrap_summary_table.csv with point estimates and percentiles
    k_samples_df : pd.DataFrame, optional
        DataFrame from bootstrap_k_samples.csv with columns:
        iteration, approach, approach_name, year, k_value

    Returns
    -------
    Dict[str, BootstrapResult]
        Dictionary mapping approach key to BootstrapResult object
    """
    results = {}

    # Get unique approaches
    approaches = coefficients_df['approach'].unique()

    for approach in approaches:
        # Get bootstrap samples for this approach (exclude iteration -1 which is the point estimate)
        mask = (coefficients_df['approach'] == approach) & (coefficients_df['iteration'] >= 0)
        samples = coefficients_df[mask].sort_values('iteration')

        # Get summary row for point estimates
        summary_mask = summary_df['approach'] == approach
        if not summary_mask.any():
            continue
        summary_row = summary_df[summary_mask].iloc[0]

        # Extract samples as numpy arrays
        h1_samples = samples['h1'].values
        h2_samples = samples['h2'].values
        T_opt_samples = samples['T_opt'].values
        r_squared_samples = samples['r_squared'].values
        total_r_squared_samples = samples['total_r_squared'].values

        # Get point estimates from summary
        h1_point = summary_row['h1_point']
        h2_point = summary_row['h2_point']
        T_opt_point = summary_row['T_opt_point']
        r_squared_point = summary_row['r_squared_point']
        total_r_squared_point = summary_row['total_r_squared_point']

        # Get approach name
        approach_name = summary_row['approach_name']

        # Handle optional fields (f1 for GDP-dependent/8b/8c, h3/h4 for departure/trend coefficients)
        f1_point = None
        f1_samples = None
        h3_point = None
        h3_samples = None
        h4_point = None
        h4_samples = None
        f2_point = None
        f2_samples = None
        T_dep_opt_point = None
        T_dep_opt_samples = None

        if 'f1' in samples.columns and not samples['f1'].isna().all():
            f1_samples = samples['f1'].values
            f1_point = summary_row.get('f1_point', None)

        if 'h3' in samples.columns and not samples['h3'].isna().all():
            h3_samples = samples['h3'].values
            h3_point = summary_row.get('h3_point', None)

        if 'h4' in samples.columns and not samples['h4'].isna().all():
            h4_samples = samples['h4'].values
            h4_point = summary_row.get('h4_point', None)

        if 'f2' in samples.columns and not samples['f2'].isna().all():
            f2_samples = samples['f2'].values
            f2_point = summary_row.get('f2_point', None)

        if 'T_dep_opt' in samples.columns and not samples['T_dep_opt'].isna().all():
            T_dep_opt_samples = samples['T_dep_opt'].values
            T_dep_opt_point = summary_row.get('T_dep_opt_point', None)

        # Reconstruct k_samples from k_samples_df if available
        k_point = None
        k_samples = None
        if k_samples_df is not None:
            k_mask = k_samples_df['approach'] == approach
            k_data = k_samples_df[k_mask]
            if len(k_data) > 0:
                # Separate point estimates (iteration -1) from bootstrap samples
                k_point_data = k_data[k_data['iteration'] == -1]
                k_bootstrap_data = k_data[k_data['iteration'] >= 0]

                years = sorted(k_bootstrap_data['year'].unique())

                # Build k_samples dict from bootstrap iterations only
                k_samples = {}
                for year in years:
                    year_data = k_bootstrap_data[k_bootstrap_data['year'] == year].sort_values('iteration')
                    k_samples[year] = year_data['k_value'].values

                # Use stored point estimates if available, else fall back to median
                k_point = {}
                if len(k_point_data) > 0:
                    for _, row in k_point_data.iterrows():
                        k_point[int(row['year'])] = row['k_value']
                else:
                    for year in years:
                        valid = k_samples[year][~np.isnan(k_samples[year])]
                        k_point[year] = np.median(valid) if len(valid) > 0 else np.nan

        # Create BootstrapResult
        result = BootstrapResult(
            approach=approach_name,
            h1_point=h1_point,
            h2_point=h2_point,
            T_opt_point=T_opt_point,
            r_squared_point=r_squared_point,
            total_r_squared_point=total_r_squared_point,
            h1_samples=h1_samples,
            h2_samples=h2_samples,
            T_opt_samples=T_opt_samples,
            r_squared_samples=r_squared_samples,
            total_r_squared_samples=total_r_squared_samples,
            n_bootstrap=int(summary_row['n_bootstrap']),
            n_successful=int(summary_row['n_successful']),
            f1_point=f1_point,
            f1_samples=f1_samples,
            h3_point=h3_point,
            h3_samples=h3_samples,
            h4_point=h4_point,
            h4_samples=h4_samples,
            f2_point=f2_point,
            f2_samples=f2_samples,
            T_dep_opt_point=T_dep_opt_point,
            T_dep_opt_samples=T_dep_opt_samples,
            k_point=k_point,
            k_samples=k_samples,
        )

        results[approach] = result

    return results


def generate_variance_decomposition_table(
    bootstrap_results: dict,
    output_dir: Path,
    approaches: list = None,
) -> None:
    """Generate variance decomposition table with bootstrap confidence intervals.

    Creates an Excel table with variance decomposition metrics as rows and
    approaches as columns, showing point estimates and percentiles in separate columns.

    Parameters
    ----------
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_var_attrib': DataFrame with variance attribution samples
        - 'bootstrap_summary': DataFrame with point estimates
    output_dir : Path
        Directory to save generated table
    approaches : list, optional
        List of approach keys to include. Defaults to main approaches.

    Table Structure
    ---------------
    Metric | Approach0_point | Approach0_p5 | Approach0_p25 | Approach0_p50 | Approach0_p75 | Approach0_p95 | Approach1_point | ...
    """
    var_attrib_df = bootstrap_results.get('bootstrap_var_attrib')
    summary_df = bootstrap_results.get('bootstrap_summary')

    if var_attrib_df is None:
        print("      [Tables] WARNING: bootstrap_var_attrib not loaded, skipping variance decomposition table")
        return
    if summary_df is None:
        print("      [Tables] WARNING: bootstrap_summary not loaded, skipping variance decomposition table")
        return

    # Default approaches to include (publication set)
    # Only includes approaches, not exploratory methods
    if approaches is None:
        approaches = ['Approach QJ', 'Approach NJ', 'Approach QP', 'Approach NP', 'Approach QL', 'Approach PL', 'Approach DL']

    # Filter to approaches that exist in the data
    available_approaches = [a for a in approaches if a in var_attrib_df['approach'].values]
    if not available_approaches:
        print("      [Tables] WARNING: No matching approaches found in var_attrib data")
        return

    # Define the metrics to include in the table (in order)
    # These correspond to var_attrib keys, normalized by var_dy
    # Using combined h(T) instead of separated h(T)-h(Ttr) and h(Ttr)
    variance_metrics = [
        ('Sigma_h_h', 'Var(h(T))/Var(Δy)'),
        ('Sigma_j_j', 'Var(j)/Var(Δy)'),
        ('Sigma_k_k', 'Var(k)/Var(Δy)'),
        ('Sigma_epsilon_epsilon', 'Var(ε)/Var(Δy)'),
    ]
    covariance_metrics = [
        ('Sigma_h_j', '2Cov(h(T),j)/Var(Δy)'),
        ('Sigma_h_k', '2Cov(h(T),k)/Var(Δy)'),
        ('Sigma_h_epsilon', '2Cov(h(T),ε)/Var(Δy)'),
        ('Sigma_j_k', '2Cov(j,k)/Var(Δy)'),
        ('Sigma_j_epsilon', '2Cov(j,ε)/Var(Δy)'),
        ('Sigma_k_epsilon', '2Cov(k,ε)/Var(Δy)'),
    ]

    # Percentiles to compute
    percentiles = [5, 25, 50, 75, 95]

    def compute_stats(samples: np.ndarray, point_estimate: float = None) -> dict:
        """Compute percentiles from samples, with optional explicit point estimate.

        Parameters
        ----------
        samples : np.ndarray
            Bootstrap samples for computing percentiles
        point_estimate : float, optional
            Explicit point estimate to use instead of median. If None, uses median.
        """
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            return {'point': np.nan, 'p5': np.nan, 'p25': np.nan, 'p50': np.nan, 'p75': np.nan, 'p95': np.nan}

        return {
            'point': point_estimate if point_estimate is not None else np.median(valid_samples),
            'p5': np.percentile(valid_samples, 5),
            'p25': np.percentile(valid_samples, 25),
            'p50': np.percentile(valid_samples, 50),
            'p75': np.percentile(valid_samples, 75),
            'p95': np.percentile(valid_samples, 95),
        }

    def get_metric_samples(approach_data: pd.DataFrame, key: str) -> np.ndarray:
        """Get metric samples, computing combined h(T) terms from separated terms if needed."""
        # If the key exists directly, use it
        if key in approach_data.columns:
            return approach_data[key].values

        # Compute combined h(T) terms from separated Delta_u and v terms
        # Sigma_h_h = Sigma_Delta_u_Delta_u + Sigma_v_v + 2*Sigma_Delta_u_v
        # Sigma_h_j = Sigma_Delta_u_j + Sigma_v_j
        # Sigma_h_k = Sigma_Delta_u_k + Sigma_v_k
        # Sigma_h_epsilon = Sigma_Delta_u_epsilon + Sigma_v_epsilon
        if key == 'Sigma_h_h':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_Delta_u', 'Sigma_v_v', 'Sigma_Delta_u_v']):
                return (approach_data['Sigma_Delta_u_Delta_u'].values +
                        approach_data['Sigma_v_v'].values +
                        2 * approach_data['Sigma_Delta_u_v'].values)
        elif key == 'Sigma_h_j':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_j', 'Sigma_v_j']):
                return approach_data['Sigma_Delta_u_j'].values + approach_data['Sigma_v_j'].values
        elif key == 'Sigma_h_k':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_k', 'Sigma_v_k']):
                return approach_data['Sigma_Delta_u_k'].values + approach_data['Sigma_v_k'].values
        elif key == 'Sigma_h_epsilon':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_epsilon', 'Sigma_v_epsilon']):
                return approach_data['Sigma_Delta_u_epsilon'].values + approach_data['Sigma_v_epsilon'].values

        # Key not available
        return None

    # Get approach display names
    approach_names = {}
    for approach in available_approaches:
        mask = summary_df['approach'] == approach
        if mask.any():
            approach_names[approach] = summary_df[mask].iloc[0]['approach_name']
        else:
            approach_names[approach] = approach

    # Build the table data
    rows = []

    # Add Total R² row first (from bootstrap_coefficients)
    total_r2_row = {'Metric': 'Total R²'}
    coefficients_df = bootstrap_results.get('bootstrap_coefficients')
    for approach in available_approaches:
        name = approach_names[approach]

        if coefficients_df is not None:
            coef_mask = coefficients_df['approach'] == approach
            approach_coef = coefficients_df[coef_mask]

            # Get point estimate from iteration -1
            point_mask = approach_coef['iteration'] == -1
            point_estimate = None
            if point_mask.any():
                point_estimate = approach_coef[point_mask].iloc[0]['total_r_squared']

            # Get bootstrap samples (iteration >= 0) for percentiles
            bootstrap_mask = approach_coef['iteration'] >= 0
            samples = approach_coef[bootstrap_mask]['total_r_squared'].values
        else:
            point_estimate = None
            samples = np.array([])

        stats = compute_stats(samples, point_estimate=point_estimate)
        total_r2_row[f'{name}_point'] = stats['point']
        total_r2_row[f'{name}_p5'] = stats['p5']
        total_r2_row[f'{name}_p25'] = stats['p25']
        total_r2_row[f'{name}_p50'] = stats['p50']
        total_r2_row[f'{name}_p75'] = stats['p75']
        total_r2_row[f'{name}_p95'] = stats['p95']
    rows.append(total_r2_row)

    # Add variance metrics
    for key, label in variance_metrics:
        row = {'Metric': label}
        for approach in available_approaches:
            name = approach_names[approach]
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if 'var_dy' not in approach_data.columns:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get point estimate from iteration -1
            point_estimate = None
            if 'iteration' in approach_data.columns:
                point_mask = approach_data['iteration'] == -1
                if point_mask.any():
                    point_data = approach_data[point_mask]
                    point_samples = get_metric_samples(point_data, key)
                    if point_samples is not None and len(point_samples) > 0:
                        var_dy_point = point_data['var_dy'].values[0]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            point_estimate = point_samples[0] / var_dy_point
                        if not np.isfinite(point_estimate):
                            point_estimate = None

            # Get bootstrap samples (iteration >= 0) for percentiles
            if 'iteration' in approach_data.columns:
                bootstrap_mask = approach_data['iteration'] >= 0
                bootstrap_data = approach_data[bootstrap_mask]
            else:
                bootstrap_data = approach_data

            metric_samples = get_metric_samples(bootstrap_data, key)
            if metric_samples is None:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get var_dy for normalization
            var_dy_samples = bootstrap_data['var_dy'].values

            # Normalize by var_dy
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            stats = compute_stats(normalized_samples, point_estimate=point_estimate)
            row[f'{name}_point'] = stats['point']
            row[f'{name}_p5'] = stats['p5']
            row[f'{name}_p25'] = stats['p25']
            row[f'{name}_p50'] = stats['p50']
            row[f'{name}_p75'] = stats['p75']
            row[f'{name}_p95'] = stats['p95']

        rows.append(row)

    # Add covariance metrics (multiply by 2 since we show 2*Cov)
    for key, label in covariance_metrics:
        row = {'Metric': label}
        for approach in available_approaches:
            name = approach_names[approach]
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if 'var_dy' not in approach_data.columns:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get point estimate from iteration -1 (with 2x factor)
            point_estimate = None
            if 'iteration' in approach_data.columns:
                point_mask = approach_data['iteration'] == -1
                if point_mask.any():
                    point_data = approach_data[point_mask]
                    point_samples = get_metric_samples(point_data, key)
                    if point_samples is not None and len(point_samples) > 0:
                        var_dy_point = point_data['var_dy'].values[0]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            point_estimate = (point_samples[0] * 2) / var_dy_point
                        if not np.isfinite(point_estimate):
                            point_estimate = None

            # Get bootstrap samples (iteration >= 0) for percentiles
            if 'iteration' in approach_data.columns:
                bootstrap_mask = approach_data['iteration'] >= 0
                bootstrap_data = approach_data[bootstrap_mask]
            else:
                bootstrap_data = approach_data

            metric_samples = get_metric_samples(bootstrap_data, key)
            if metric_samples is None:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get var_dy for normalization
            var_dy_samples = bootstrap_data['var_dy'].values

            # Multiply by 2 (2*Cov term) and normalize
            metric_samples = metric_samples * 2
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            stats = compute_stats(normalized_samples, point_estimate=point_estimate)
            row[f'{name}_point'] = stats['point']
            row[f'{name}_p5'] = stats['p5']
            row[f'{name}_p25'] = stats['p25']
            row[f'{name}_p50'] = stats['p50']
            row[f'{name}_p75'] = stats['p75']
            row[f'{name}_p95'] = stats['p95']

        rows.append(row)

    # Add Sum row (should equal 1.0)
    sum_row = {'Metric': 'Sum'}
    for approach in available_approaches:
        name = approach_names[approach]
        mask = var_attrib_df['approach'] == approach
        approach_data = var_attrib_df[mask]

        if 'var_dy' not in approach_data.columns:
            sum_row[f'{name}_point'] = np.nan
            sum_row[f'{name}_p5'] = np.nan
            sum_row[f'{name}_p25'] = np.nan
            sum_row[f'{name}_p50'] = np.nan
            sum_row[f'{name}_p75'] = np.nan
            sum_row[f'{name}_p95'] = np.nan
            continue

        # Get point estimate from iteration -1
        point_estimate = None
        if 'iteration' in approach_data.columns:
            point_mask = approach_data['iteration'] == -1
            if point_mask.any():
                point_data = approach_data[point_mask]
                var_dy_point = point_data['var_dy'].values[0]
                total_sum_point = 0.0
                for key, _ in variance_metrics:
                    samples = get_metric_samples(point_data, key)
                    if samples is not None and len(samples) > 0:
                        total_sum_point += samples[0]
                for key, _ in covariance_metrics:
                    samples = get_metric_samples(point_data, key)
                    if samples is not None and len(samples) > 0:
                        total_sum_point += 2 * samples[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    point_estimate = total_sum_point / var_dy_point
                if not np.isfinite(point_estimate):
                    point_estimate = None

        # Get bootstrap samples (iteration >= 0) for percentiles
        if 'iteration' in approach_data.columns:
            bootstrap_mask = approach_data['iteration'] >= 0
            bootstrap_data = approach_data[bootstrap_mask]
        else:
            bootstrap_data = approach_data

        var_dy_samples = bootstrap_data['var_dy'].values

        # Sum all variance and covariance terms (using get_metric_samples for combined h(T) terms)
        total_sum = np.zeros(len(bootstrap_data))
        for key, _ in variance_metrics:
            samples = get_metric_samples(bootstrap_data, key)
            if samples is not None:
                total_sum += samples
        for key, _ in covariance_metrics:
            samples = get_metric_samples(bootstrap_data, key)
            if samples is not None:
                total_sum += 2 * samples  # 2*Cov terms

        with np.errstate(divide='ignore', invalid='ignore'):
            sum_normalized = total_sum / var_dy_samples
        sum_normalized = np.where(np.isfinite(sum_normalized), sum_normalized, np.nan)

        stats = compute_stats(sum_normalized, point_estimate=point_estimate)
        sum_row[f'{name}_point'] = stats['point']
        sum_row[f'{name}_p5'] = stats['p5']
        sum_row[f'{name}_p25'] = stats['p25']
        sum_row[f'{name}_p50'] = stats['p50']
        sum_row[f'{name}_p75'] = stats['p75']
        sum_row[f'{name}_p95'] = stats['p95']

    rows.append(sum_row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to Excel
    xlsx_path = output_dir / 'variance_decomposition_table.xlsx'
    df.to_excel(xlsx_path, index=False, sheet_name='Variance Decomposition')
    print(f"      [Tables] Saved variance_decomposition_table.xlsx ({len(rows)} rows × {len(available_approaches)} approaches)")

    # Also save as CSV for easier inspection
    csv_path = output_dir / 'variance_decomposition_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"      [Tables] Saved variance_decomposition_table.csv")


def generate_bootstrap_comparison_table(
    bootstrap_results: dict,
    output_dir: Path,
    approaches: list = None,
) -> None:
    """Generate bootstrap coefficient comparison table with percentiles.

    Creates an Excel/CSV table showing point estimates and bootstrap percentiles
    (5th, 25th, 50th, 75th, 95th) for key parameters across approaches.

    Parameters
    ----------
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_summary': DataFrame with point estimates and percentiles
    output_dir : Path
        Directory to save generated table
    approaches : list, optional
        List of approach keys to include. Defaults to main approaches.

    Table Structure
    ---------------
    Approach | h1_point | h1_p5 | h1_p25 | h1_p50 | h1_p75 | h1_p95 | h2_point | ... | T_opt_point | h3_point | h4_point | ...
    """
    summary_df = bootstrap_results.get('bootstrap_summary')

    if summary_df is None:
        print("      [Tables] WARNING: bootstrap_summary not loaded, skipping bootstrap comparison table")
        return

    # Default approaches to include (publication set, same as variance decomposition table)
    # Only includes approaches, not exploratory methods
    if approaches is None:
        approaches = ['Approach QJ', 'Approach NJ', 'Approach QP', 'Approach NP', 'Approach QL', 'Approach PL', 'Approach DL']

    # Filter to approaches that exist in the data
    available_approaches = [a for a in approaches if a in summary_df['approach'].values]
    if not available_approaches:
        print("      [Tables] WARNING: No matching approaches found in summary data")
        return

    # Parameters to include
    # Standard parameters for all approaches
    standard_params = ['h1', 'h2', 'T_opt', 'total_r_squared']
    # Additional parameters: h4 for piecewise (Approach2*) and persistence (Approach3*)
    # For piecewise: h2 = curvature below T_opt, h4 = curvature above T_opt
    # For persistence: h4 = decay rate
    trend_params = ['h4']

    # Percentiles to include
    percentiles = ['p5', 'p25', 'p50', 'p75', 'p95']

    # Build the table
    rows = []

    for approach in available_approaches:
        mask = summary_df['approach'] == approach
        if not mask.any():
            continue
        row_data = summary_df[mask].iloc[0]

        row = {
            'Approach': row_data['approach_name'],
        }

        # Add standard parameters
        for param in standard_params:
            # Point estimate
            point_col = f'{param}_point'
            if point_col in row_data.index:
                row[f'{param}_point'] = row_data[point_col]
            else:
                row[f'{param}_point'] = np.nan

            # Percentiles (use median for p50)
            for pct in percentiles:
                if pct == 'p50':
                    pct_col = f'{param}_median'
                else:
                    pct_col = f'{param}_{pct}'

                if pct_col in row_data.index:
                    row[f'{param}_{pct}'] = row_data[pct_col]
                else:
                    row[f'{param}_{pct}'] = np.nan

        # Add trend parameters (h3, h4, T_dep_opt, f1, f2 - populated for 6b/6c/6e/8/8a)
        for param in trend_params:
            # Point estimate
            point_col = f'{param}_point'
            if point_col in row_data.index and not pd.isna(row_data[point_col]):
                row[f'{param}_point'] = row_data[point_col]
            else:
                row[f'{param}_point'] = np.nan

            # Percentiles
            for pct in percentiles:
                if pct == 'p50':
                    pct_col = f'{param}_median'
                else:
                    pct_col = f'{param}_{pct}'

                if pct_col in row_data.index and not pd.isna(row_data.get(pct_col)):
                    row[f'{param}_{pct}'] = row_data[pct_col]
                else:
                    row[f'{param}_{pct}'] = np.nan

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to Excel
    xlsx_path = output_dir / 'bootstrap_comparison_table.xlsx'
    df.to_excel(xlsx_path, index=False, sheet_name='Bootstrap Comparison')
    print(f"      [Tables] Saved bootstrap_comparison_table.xlsx ({len(rows)} approaches)")

    # Also save as CSV
    csv_path = output_dir / 'bootstrap_comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"      [Tables] Saved bootstrap_comparison_table.csv")


def format_small_number(value: float) -> str:
    """Format number with adaptive precision for small values.

    Default: 3 decimal places (e.g., 0.123)
    If |value| < 0.001 (would round to 0.000):
      - If |value| > 1e-6: Show enough digits to display first non-zero digit
      - If |value| <= 1e-6: Show in scientific notation with 1 significant digit (e.g., 6e-9)
    """
    if value == 0:
        return "0.000"
    if abs(value) < 0.001:  # Would round to 0.000
        if abs(value) > 1e-6:
            # Find first significant digit position
            digits = -int(math.floor(math.log10(abs(value))))
            return f"{value:.{digits}f}"
        else:
            # Scientific notation with 1 significant digit
            return f"{value:.0e}"
    return f"{value:.3f}"


def generate_variance_decomposition_by_response(
    bootstrap_results: dict,
    output_dir: Path,
) -> None:
    """Generate variance decomposition tables organized by response function type.

    Creates:
    - variance_decomposition_by_response.xlsx (4 sheets)
    - variance_decomposition_null.tex
    - variance_decomposition_quadratic.tex
    - variance_decomposition_piecewise.tex
    - variance_decomposition_persistence.tex

    Parameters
    ----------
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_var_attrib': DataFrame with variance attribution samples
        - 'bootstrap_summary': DataFrame with point estimates
        - 'bootstrap_coefficients': DataFrame with bootstrap samples
    output_dir : Path
        Directory to save generated tables
    """
    var_attrib_df = bootstrap_results.get('bootstrap_var_attrib')
    summary_df = bootstrap_results.get('bootstrap_summary')
    coefficients_df = bootstrap_results.get('bootstrap_coefficients')

    if var_attrib_df is None:
        print("      [Tables] WARNING: bootstrap_var_attrib not loaded, skipping variance decomposition by response tables")
        return
    if summary_df is None:
        print("      [Tables] WARNING: bootstrap_summary not loaded, skipping variance decomposition by response tables")
        return

    # Define response type groups
    response_types = {
        'Null': {
            'approaches': ['Approach NJ', 'Approach NP', 'Approach NL'],
            'description': '—',  # Em-dash for null model (no response)
            'latex_description': '---',  # LaTeX em-dash
        },
        'Quadratic': {
            'approaches': ['Approach QJ', 'Approach QP', 'Approach QL'],
            'description': 'Quadratic',
            'latex_description': r'$h_1 T + h_2 T^2$',
        },
        'Piecewise': {
            'approaches': ['Approach PJ', 'Approach PP', 'Approach PL'],
            'description': 'Piecewise quadratic',
            'latex_description': r'Piecewise quadratic',
        },
        'Persistence': {
            'approaches': ['Approach DJ', 'Approach DP', 'Approach DL'],
            'description': 'Quadratic with decay',
            'latex_description': r'Quadratic with decay',
        },
    }

    # Trend method column headers
    trend_methods = ['Joint', 'Polynomial', 'LOESS']
    approach_suffix_to_method = {'J': 'Joint', 'P': 'Polynomial', 'L': 'LOESS'}

    # Metrics to extract (variance components)
    variance_metrics = [
        ('Sigma_h_h', r'$\mathrm{Var}\!\big(h(T)\big)/\mathrm{Var}(\Delta y)$', 'Var(h(T))/Var(Δy)'),
        ('Sigma_j_j', r'$\mathrm{Var}(j)/\mathrm{Var}(\Delta y)$', 'Var(j)/Var(Δy)'),
        ('Sigma_k_k', r'$\mathrm{Var}(k)/\mathrm{Var}(\Delta y)$', 'Var(k)/Var(Δy)'),
        ('Sigma_epsilon_epsilon', r'$\mathrm{Var}(\varepsilon)/\mathrm{Var}(\Delta y)$', 'Var(ε)/Var(Δy)'),
    ]
    covariance_metrics = [
        ('Sigma_h_j', r'$2\,\mathrm{Cov}\!\big(h(T),j\big)/\mathrm{Var}(\Delta y)$', '2Cov(h(T),j)/Var(Δy)'),
        ('Sigma_h_k', r'$2\,\mathrm{Cov}\!\big(h(T),k\big)/\mathrm{Var}(\Delta y)$', '2Cov(h(T),k)/Var(Δy)'),
        ('Sigma_j_k', r'$2\,\mathrm{Cov}(j,k)/\mathrm{Var}(\Delta y)$', '2Cov(j,k)/Var(Δy)'),
        ('Sigma_h_epsilon', r'$2\,\mathrm{Cov}\!\big(h(T),\varepsilon\big)/\mathrm{Var}(\Delta y)$', '2Cov(h(T),ε)/Var(Δy)'),
        ('Sigma_j_epsilon', r'$2\,\mathrm{Cov}(j,\varepsilon)/\mathrm{Var}(\Delta y)$', '2Cov(j,ε)/Var(Δy)'),
        ('Sigma_k_epsilon', r'$2\,\mathrm{Cov}(k,\varepsilon)/\mathrm{Var}(\Delta y)$', '2Cov(k,ε)/Var(Δy)'),
    ]

    # Metrics involving h(T) that should be em-dash for null models
    h_metrics = {'Sigma_h_h', 'Sigma_h_j', 'Sigma_h_k', 'Sigma_h_epsilon'}

    def get_metric_samples(approach_data: pd.DataFrame, key: str) -> np.ndarray:
        """Get metric samples, computing combined h(T) terms from separated terms if needed."""
        if key in approach_data.columns:
            return approach_data[key].values

        # Compute combined h(T) terms from separated Delta_u and v terms
        if key == 'Sigma_h_h':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_Delta_u', 'Sigma_v_v', 'Sigma_Delta_u_v']):
                return (approach_data['Sigma_Delta_u_Delta_u'].values +
                        approach_data['Sigma_v_v'].values +
                        2 * approach_data['Sigma_Delta_u_v'].values)
        elif key == 'Sigma_h_j':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_j', 'Sigma_v_j']):
                return approach_data['Sigma_Delta_u_j'].values + approach_data['Sigma_v_j'].values
        elif key == 'Sigma_h_k':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_k', 'Sigma_v_k']):
                return approach_data['Sigma_Delta_u_k'].values + approach_data['Sigma_v_k'].values
        elif key == 'Sigma_h_epsilon':
            if all(k in approach_data.columns for k in ['Sigma_Delta_u_epsilon', 'Sigma_v_epsilon']):
                return approach_data['Sigma_Delta_u_epsilon'].values + approach_data['Sigma_v_epsilon'].values

        return None

    def get_point_estimate(approach: str, key: str, is_covariance: bool = False) -> float:
        """Get point estimate for a metric from the variance attribution data."""
        mask = var_attrib_df['approach'] == approach
        approach_data = var_attrib_df[mask]

        if 'iteration' not in approach_data.columns:
            return np.nan

        point_mask = approach_data['iteration'] == -1
        if not point_mask.any():
            return np.nan

        point_data = approach_data[point_mask]
        if 'var_dy' not in point_data.columns:
            return np.nan

        samples = get_metric_samples(point_data, key)
        if samples is None or len(samples) == 0:
            return np.nan

        var_dy = point_data['var_dy'].values[0]
        multiplier = 2 if is_covariance else 1
        with np.errstate(divide='ignore', invalid='ignore'):
            value = (samples[0] * multiplier) / var_dy
        return value if np.isfinite(value) else np.nan

    def get_total_r_squared(approach: str) -> float:
        """Get total R² point estimate."""
        if coefficients_df is None:
            return np.nan
        mask = (coefficients_df['approach'] == approach) & (coefficients_df['iteration'] == -1)
        if not mask.any():
            return np.nan
        return coefficients_df[mask].iloc[0]['total_r_squared']

    def compute_sum(approach: str) -> float:
        """Compute sum of all variance components."""
        total = 0.0
        for key, _, _ in variance_metrics:
            val = get_point_estimate(approach, key, is_covariance=False)
            if np.isfinite(val):
                total += val
        for key, _, _ in covariance_metrics:
            val = get_point_estimate(approach, key, is_covariance=True)
            if np.isfinite(val):
                total += val
        return total

    # Em-dash for missing values
    EM_DASH = '—'

    # Build tables for each response type
    excel_sheets = {}

    for response_type, config in response_types.items():
        approaches = config['approaches']
        is_null = response_type == 'Null'

        # Build DataFrame for this response type
        rows = []

        # Row 1: Response function description
        rows.append({
            'Metric': 'Response function',
            'Joint': config['description'],
            'Polynomial': config['description'],
            'LOESS': config['description'],
        })

        # Row 2: Trend method
        rows.append({
            'Metric': 'Trend method',
            'Joint': 'Joint',
            'Polynomial': 'Polynomial',
            'LOESS': 'LOESS',
        })

        # Row 3: Response fn ΔR²
        delta_r2_row = {'Metric': 'Response fn ΔR²'}
        for approach in approaches:
            suffix = approach[-1]  # J, P, or L
            method = approach_suffix_to_method[suffix]
            if is_null:
                delta_r2_row[method] = EM_DASH
            else:
                # Get null model R² for this trend method
                null_approach = f'Approach0{suffix}'
                r2_null = get_total_r_squared(null_approach)
                r2_this = get_total_r_squared(approach)
                delta_r2 = r2_this - r2_null if np.isfinite(r2_this) and np.isfinite(r2_null) else np.nan
                delta_r2_row[method] = format_small_number(delta_r2) if np.isfinite(delta_r2) else EM_DASH
        rows.append(delta_r2_row)

        # Row 4: Total R²
        r2_row = {'Metric': 'Total R²'}
        for approach in approaches:
            suffix = approach[-1]
            method = approach_suffix_to_method[suffix]
            r2 = get_total_r_squared(approach)
            r2_row[method] = format_small_number(r2) if np.isfinite(r2) else EM_DASH
        rows.append(r2_row)

        # Variance metrics
        for key, latex_label, excel_label in variance_metrics:
            row = {'Metric': excel_label}
            for approach in approaches:
                suffix = approach[-1]
                method = approach_suffix_to_method[suffix]
                if is_null and key in h_metrics:
                    row[method] = EM_DASH
                else:
                    val = get_point_estimate(approach, key, is_covariance=False)
                    row[method] = format_small_number(val) if np.isfinite(val) else EM_DASH
            rows.append(row)

        # Covariance metrics
        for key, latex_label, excel_label in covariance_metrics:
            row = {'Metric': excel_label}
            for approach in approaches:
                suffix = approach[-1]
                method = approach_suffix_to_method[suffix]
                if is_null and key in h_metrics:
                    row[method] = EM_DASH
                else:
                    val = get_point_estimate(approach, key, is_covariance=True)
                    row[method] = format_small_number(val) if np.isfinite(val) else EM_DASH
            rows.append(row)

        # Sum row
        sum_row = {'Metric': 'Sum'}
        for approach in approaches:
            suffix = approach[-1]
            method = approach_suffix_to_method[suffix]
            total = compute_sum(approach)
            sum_row[method] = format_small_number(total) if np.isfinite(total) else EM_DASH
        rows.append(sum_row)

        # Create DataFrame
        df = pd.DataFrame(rows)
        excel_sheets[response_type] = df

    # Save Excel file with multiple sheets
    xlsx_path = output_dir / 'variance_decomposition_by_response.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, df in excel_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"      [Tables] Saved variance_decomposition_by_response.xlsx (4 sheets)")

    # Generate LaTeX files
    for response_type, config in response_types.items():
        df = excel_sheets[response_type]
        is_null = response_type == 'Null'

        # Build LaTeX content
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{Variance decomposition: {response_type.lower()} climate response}}',
            f'\\label{{tab:variance_decomp_{response_type.lower()}}}',
            r'',
            r'\small',
            r'\begin{tabular}{@{}lccc@{}}',
            r'\toprule',
            r'Metric & Joint & Polynomial & LOESS \\',
            r'\midrule',
        ]

        # Trend method row
        latex_lines.append(r'Trend method & Joint & Polynomial & LOESS \\')
        latex_lines.append(r'\midrule')

        # Response fn ΔR² and Total R²
        delta_r2_row = df[df['Metric'] == 'Response fn ΔR²'].iloc[0]
        latex_lines.append(f"Response fn $\\Delta R^2$ & {delta_r2_row['Joint']} & {delta_r2_row['Polynomial']} & {delta_r2_row['LOESS']} \\\\")

        r2_row = df[df['Metric'] == 'Total R²'].iloc[0]
        latex_lines.append(f"Total $R^2$ & {r2_row['Joint']} & {r2_row['Polynomial']} & {r2_row['LOESS']} \\\\")
        latex_lines.append(r'\midrule')

        # Variance metrics
        for key, latex_label, excel_label in variance_metrics:
            row = df[df['Metric'] == excel_label].iloc[0]
            latex_lines.append(f"{latex_label} & {row['Joint']} & {row['Polynomial']} & {row['LOESS']} \\\\")
        latex_lines.append(r'\midrule')

        # Covariance metrics
        for key, latex_label, excel_label in covariance_metrics:
            row = df[df['Metric'] == excel_label].iloc[0]
            latex_lines.append(f"{latex_label} & {row['Joint']} & {row['Polynomial']} & {row['LOESS']} \\\\")
        latex_lines.append(r'\midrule')

        # Sum row
        sum_row = df[df['Metric'] == 'Sum'].iloc[0]
        latex_lines.append(f"Sum & {sum_row['Joint']} & {sum_row['Polynomial']} & {sum_row['LOESS']} \\\\")

        latex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        # Write LaTeX file
        tex_path = output_dir / f'variance_decomposition_{response_type.lower()}.tex'
        with open(tex_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"      [Tables] Saved variance_decomposition_{response_type.lower()}.tex")


def generate_covariance_tables(
    bootstrap_results: dict,
    output_dir: Path,
) -> None:
    """Generate separate covariance tables for J vs P/L approaches.

    J approaches (joint estimation) have 4 components: h(T), j, k, ε
    P/L approaches (sequential detrending) have 5 components: k_mean, GDP_trend, h(T), h(T_trend), ε

    The identity for P/L is: Δy = k_mean + GDP_trend + h(T) - h(T_trend) + ε
    Note: h(T_trend) enters with a negative sign, affecting covariance contributions.

    Creates:
    - covariance_table_J.xlsx / .csv: 4×4 covariance matrix for J approaches
    - covariance_table_PL.xlsx / .csv: 5×5 covariance matrix for P/L approaches

    Parameters
    ----------
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_var_attrib': DataFrame with variance attribution samples
    output_dir : Path
        Directory to save generated tables
    """
    var_attrib_df = bootstrap_results.get('bootstrap_var_attrib')

    if var_attrib_df is None:
        print("      [Tables] WARNING: bootstrap_var_attrib not loaded, skipping covariance tables")
        return

    # Define approach groups
    j_approaches = ['Approach NJ', 'Approach QJ', 'Approach PJ', 'Approach DJ']
    pl_approaches = ['Approach NP', 'Approach NL', 'Approach QP', 'Approach QL',
                     'Approach PP', 'Approach PL', 'Approach DP', 'Approach DL']

    # Filter to approaches that exist in the data
    available_j = [a for a in j_approaches if a in var_attrib_df['approach'].values]
    available_pl = [a for a in pl_approaches if a in var_attrib_df['approach'].values]

    percentiles = [5, 25, 50, 75, 95]

    def compute_stats(samples: np.ndarray, point_estimate: float = None) -> dict:
        """Compute percentiles from samples."""
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            return {'point': np.nan, 'p5': np.nan, 'p25': np.nan, 'p50': np.nan, 'p75': np.nan, 'p95': np.nan}
        return {
            'point': point_estimate if point_estimate is not None else np.median(valid_samples),
            'p5': np.percentile(valid_samples, 5),
            'p25': np.percentile(valid_samples, 25),
            'p50': np.percentile(valid_samples, 50),
            'p75': np.percentile(valid_samples, 75),
            'p95': np.percentile(valid_samples, 95),
        }

    def get_raw_samples(approach_data: pd.DataFrame, key: str) -> np.ndarray:
        """Get raw Sigma samples from data."""
        if key in approach_data.columns:
            return approach_data[key].values
        return None

    # =========================================================================
    # J-Approach Table: 4 components (h(T), j, k, ε)
    # Components: h(T) = Delta_u + v (since for J, Delta_u should be 0 anyway)
    # =========================================================================
    if available_j:
        # Define J metrics - 4 variances + 6 covariances = 10 entries
        j_variance_metrics = [
            ('Sigma_h_h', 'Var(h(T))/Var(Δy)', 1),  # Already combined
            ('Sigma_j_j', 'Var(j)/Var(Δy)', 1),
            ('Sigma_k_k', 'Var(k)/Var(Δy)', 1),
            ('Sigma_epsilon_epsilon', 'Var(ε)/Var(Δy)', 1),
        ]
        j_covariance_metrics = [
            ('Sigma_h_j', '2Cov(h(T),j)/Var(Δy)', 2),  # Already combined
            ('Sigma_h_k', '2Cov(h(T),k)/Var(Δy)', 2),  # Already combined
            ('Sigma_h_epsilon', '2Cov(h(T),ε)/Var(Δy)', 2),  # Already combined
            ('Sigma_j_k', '2Cov(j,k)/Var(Δy)', 2),
            ('Sigma_j_epsilon', '2Cov(j,ε)/Var(Δy)', 2),
            ('Sigma_k_epsilon', '2Cov(k,ε)/Var(Δy)', 2),
        ]

        def get_j_metric_samples(approach_data: pd.DataFrame, key: str) -> np.ndarray:
            """Get J-approach metric samples, computing combined h(T) terms if needed."""
            if key in approach_data.columns:
                return approach_data[key].values
            # Compute combined h(T) = Delta_u + v
            if key == 'Sigma_h_h':
                if all(k in approach_data.columns for k in ['Sigma_Delta_u_Delta_u', 'Sigma_v_v', 'Sigma_Delta_u_v']):
                    return (approach_data['Sigma_Delta_u_Delta_u'].values +
                            approach_data['Sigma_v_v'].values +
                            2 * approach_data['Sigma_Delta_u_v'].values)
            elif key == 'Sigma_h_j':
                if all(k in approach_data.columns for k in ['Sigma_Delta_u_j', 'Sigma_v_j']):
                    return approach_data['Sigma_Delta_u_j'].values + approach_data['Sigma_v_j'].values
            elif key == 'Sigma_h_k':
                if all(k in approach_data.columns for k in ['Sigma_Delta_u_k', 'Sigma_v_k']):
                    return approach_data['Sigma_Delta_u_k'].values + approach_data['Sigma_v_k'].values
            elif key == 'Sigma_h_epsilon':
                if all(k in approach_data.columns for k in ['Sigma_Delta_u_epsilon', 'Sigma_v_epsilon']):
                    return approach_data['Sigma_Delta_u_epsilon'].values + approach_data['Sigma_v_epsilon'].values
            return None

        rows_j = []
        all_j_metrics = j_variance_metrics + j_covariance_metrics

        for key, label, multiplier in all_j_metrics:
            row = {'Metric': label}
            for approach in available_j:
                mask = var_attrib_df['approach'] == approach
                approach_data = var_attrib_df[mask]

                if 'var_dy' not in approach_data.columns:
                    for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                        row[f'{approach}{suffix}'] = np.nan
                    continue

                # Get point estimate (iteration -1)
                point_estimate = None
                if 'iteration' in approach_data.columns:
                    point_mask = approach_data['iteration'] == -1
                    if point_mask.any():
                        point_data = approach_data[point_mask]
                        samples = get_j_metric_samples(point_data, key)
                        if samples is not None and len(samples) > 0:
                            var_dy = point_data['var_dy'].values[0]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                point_estimate = (samples[0] * multiplier) / var_dy
                            if not np.isfinite(point_estimate):
                                point_estimate = None

                # Get bootstrap samples (iteration >= 0)
                bootstrap_mask = approach_data['iteration'] >= 0
                bootstrap_data = approach_data[bootstrap_mask]

                metric_samples = get_j_metric_samples(bootstrap_data, key)
                if metric_samples is None:
                    for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                        row[f'{approach}{suffix}'] = np.nan
                    continue

                var_dy_samples = bootstrap_data['var_dy'].values
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = (metric_samples * multiplier) / var_dy_samples
                normalized = np.where(np.isfinite(normalized), normalized, np.nan)

                stats = compute_stats(normalized, point_estimate=point_estimate)
                row[f'{approach}_point'] = stats['point']
                row[f'{approach}_p5'] = stats['p5']
                row[f'{approach}_p25'] = stats['p25']
                row[f'{approach}_p50'] = stats['p50']
                row[f'{approach}_p75'] = stats['p75']
                row[f'{approach}_p95'] = stats['p95']

            rows_j.append(row)

        # Add Sum row for J approaches
        sum_row_j = {'Metric': 'Sum'}
        for approach in available_j:
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if 'var_dy' not in approach_data.columns:
                for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                    sum_row_j[f'{approach}{suffix}'] = np.nan
                continue

            # Point estimate sum
            point_sum = None
            if 'iteration' in approach_data.columns:
                point_mask = approach_data['iteration'] == -1
                if point_mask.any():
                    point_data = approach_data[point_mask]
                    var_dy = point_data['var_dy'].values[0]
                    total = 0.0
                    for key, _, mult in all_j_metrics:
                        samples = get_j_metric_samples(point_data, key)
                        if samples is not None and len(samples) > 0:
                            total += samples[0] * mult
                    with np.errstate(divide='ignore', invalid='ignore'):
                        point_sum = total / var_dy
                    if not np.isfinite(point_sum):
                        point_sum = None

            # Bootstrap sums
            bootstrap_mask = approach_data['iteration'] >= 0
            bootstrap_data = approach_data[bootstrap_mask]
            var_dy_samples = bootstrap_data['var_dy'].values

            total_sum = np.zeros(len(bootstrap_data))
            for key, _, mult in all_j_metrics:
                samples = get_j_metric_samples(bootstrap_data, key)
                if samples is not None:
                    total_sum += samples * mult

            with np.errstate(divide='ignore', invalid='ignore'):
                sum_normalized = total_sum / var_dy_samples
            sum_normalized = np.where(np.isfinite(sum_normalized), sum_normalized, np.nan)

            stats = compute_stats(sum_normalized, point_estimate=point_sum)
            sum_row_j[f'{approach}_point'] = stats['point']
            sum_row_j[f'{approach}_p5'] = stats['p5']
            sum_row_j[f'{approach}_p25'] = stats['p25']
            sum_row_j[f'{approach}_p50'] = stats['p50']
            sum_row_j[f'{approach}_p75'] = stats['p75']
            sum_row_j[f'{approach}_p95'] = stats['p95']

        rows_j.append(sum_row_j)

        df_j = pd.DataFrame(rows_j)
        xlsx_path_j = output_dir / 'covariance_table_J.xlsx'
        df_j.to_excel(xlsx_path_j, index=False, sheet_name='J Approaches')
        csv_path_j = output_dir / 'covariance_table_J.csv'
        df_j.to_csv(csv_path_j, index=False)
        print(f"      [Tables] Saved covariance_table_J.xlsx ({len(rows_j)} rows × {len(available_j)} approaches)")

    # =========================================================================
    # P/L-Approach Table: 5 components (k_mean, GDP_trend, h(T), h(T_trend), ε)
    # Identity: Δy = k + GDP_trend + h(T) - h(T_trend) + ε
    # Transformation from stored (Delta_u, v, j_adjusted, k, epsilon):
    #   k_mean = k (unchanged)
    #   GDP_trend = j_adjusted + v  (since j_adjusted = j_raw - v)
    #   h(T) = Delta_u + v
    #   h(T_trend) = v
    #   ε = epsilon (unchanged)
    # Note: h(T_trend) enters with negative sign in identity
    # =========================================================================
    if available_pl:

        def get_pl_metric(approach_data: pd.DataFrame, metric_key: str) -> np.ndarray:
            """Compute P/L metric by transforming from stored components.

            Stored: Delta_u, v, j (j_adjusted), k, epsilon
            Target: k, GDP_trend (=j+v), h(T) (=Delta_u+v), h(T_trend) (=v), epsilon
            """
            # Variance terms
            if metric_key == 'Var_k':
                return get_raw_samples(approach_data, 'Sigma_k_k')
            elif metric_key == 'Var_GDP_trend':
                # Var(j+v) = Var(j) + Var(v) + 2*Cov(j,v)
                j_j = get_raw_samples(approach_data, 'Sigma_j_j')
                v_v = get_raw_samples(approach_data, 'Sigma_v_v')
                v_j = get_raw_samples(approach_data, 'Sigma_v_j')
                if j_j is not None and v_v is not None and v_j is not None:
                    return j_j + v_v + 2 * v_j
                return None
            elif metric_key == 'Var_h_T':
                # Var(Delta_u + v) = Sigma_h_h (already stored)
                h_h = get_raw_samples(approach_data, 'Sigma_h_h')
                if h_h is not None:
                    return h_h
                # Compute from components
                du_du = get_raw_samples(approach_data, 'Sigma_Delta_u_Delta_u')
                v_v = get_raw_samples(approach_data, 'Sigma_v_v')
                du_v = get_raw_samples(approach_data, 'Sigma_Delta_u_v')
                if du_du is not None and v_v is not None and du_v is not None:
                    return du_du + v_v + 2 * du_v
                return None
            elif metric_key == 'Var_h_Ttrend':
                # Var(v) = Sigma_v_v
                return get_raw_samples(approach_data, 'Sigma_v_v')
            elif metric_key == 'Var_epsilon':
                return get_raw_samples(approach_data, 'Sigma_epsilon_epsilon')

            # Covariance terms (with appropriate signs for h(T_trend) which enters negatively)
            elif metric_key == 'Cov_k_GDP_trend':
                # Cov(k, j+v) = Cov(k,j) + Cov(k,v) = Sigma_j_k + Sigma_v_k
                j_k = get_raw_samples(approach_data, 'Sigma_j_k')
                v_k = get_raw_samples(approach_data, 'Sigma_v_k')
                if j_k is not None and v_k is not None:
                    return j_k + v_k
                return None
            elif metric_key == 'Cov_k_h_T':
                # Cov(k, Delta_u+v) = Sigma_h_k (already stored) or compute
                h_k = get_raw_samples(approach_data, 'Sigma_h_k')
                if h_k is not None:
                    return h_k
                du_k = get_raw_samples(approach_data, 'Sigma_Delta_u_k')
                v_k = get_raw_samples(approach_data, 'Sigma_v_k')
                if du_k is not None and v_k is not None:
                    return du_k + v_k
                return None
            elif metric_key == 'Cov_k_h_Ttrend':
                # Cov(k, v) = Sigma_v_k (NEGATIVE in identity contribution)
                return get_raw_samples(approach_data, 'Sigma_v_k')
            elif metric_key == 'Cov_k_epsilon':
                return get_raw_samples(approach_data, 'Sigma_k_epsilon')
            elif metric_key == 'Cov_GDP_trend_h_T':
                # Cov(j+v, Delta_u+v) = Cov(j,Du) + Cov(j,v) + Cov(v,Du) + Var(v)
                du_j = get_raw_samples(approach_data, 'Sigma_Delta_u_j')
                v_j = get_raw_samples(approach_data, 'Sigma_v_j')
                du_v = get_raw_samples(approach_data, 'Sigma_Delta_u_v')
                v_v = get_raw_samples(approach_data, 'Sigma_v_v')
                if all(x is not None for x in [du_j, v_j, du_v, v_v]):
                    return du_j + v_j + du_v + v_v
                return None
            elif metric_key == 'Cov_GDP_trend_h_Ttrend':
                # Cov(j+v, v) = Cov(j,v) + Var(v) (NEGATIVE in identity contribution)
                v_j = get_raw_samples(approach_data, 'Sigma_v_j')
                v_v = get_raw_samples(approach_data, 'Sigma_v_v')
                if v_j is not None and v_v is not None:
                    return v_j + v_v
                return None
            elif metric_key == 'Cov_GDP_trend_epsilon':
                # Cov(j+v, epsilon) = Cov(j,eps) + Cov(v,eps)
                j_eps = get_raw_samples(approach_data, 'Sigma_j_epsilon')
                v_eps = get_raw_samples(approach_data, 'Sigma_v_epsilon')
                if j_eps is not None and v_eps is not None:
                    return j_eps + v_eps
                return None
            elif metric_key == 'Cov_h_T_h_Ttrend':
                # Cov(Delta_u+v, v) = Cov(Du,v) + Var(v) (NEGATIVE in identity contribution)
                du_v = get_raw_samples(approach_data, 'Sigma_Delta_u_v')
                v_v = get_raw_samples(approach_data, 'Sigma_v_v')
                if du_v is not None and v_v is not None:
                    return du_v + v_v
                return None
            elif metric_key == 'Cov_h_T_epsilon':
                # Cov(Delta_u+v, epsilon) = Sigma_h_epsilon (already stored) or compute
                h_eps = get_raw_samples(approach_data, 'Sigma_h_epsilon')
                if h_eps is not None:
                    return h_eps
                du_eps = get_raw_samples(approach_data, 'Sigma_Delta_u_epsilon')
                v_eps = get_raw_samples(approach_data, 'Sigma_v_epsilon')
                if du_eps is not None and v_eps is not None:
                    return du_eps + v_eps
                return None
            elif metric_key == 'Cov_h_Ttrend_epsilon':
                # Cov(v, epsilon) (NEGATIVE in identity contribution)
                return get_raw_samples(approach_data, 'Sigma_v_epsilon')

            return None

        # Define P/L metrics - 5 variances + 10 covariances = 15 entries
        # multiplier: 1 for variance, 2 for positive covariance, -2 for negative covariance
        pl_metrics = [
            # Variances
            ('Var_k', 'Var(k)/Var(Δy)', 1),
            ('Var_GDP_trend', 'Var(GDP_trend)/Var(Δy)', 1),
            ('Var_h_T', 'Var(h(T))/Var(Δy)', 1),
            ('Var_h_Ttrend', 'Var(h(T_trend))/Var(Δy)', 1),
            ('Var_epsilon', 'Var(ε)/Var(Δy)', 1),
            # Covariances (with signs reflecting identity: Δy = k + GDP_trend + h(T) - h(T_trend) + ε)
            ('Cov_k_GDP_trend', '2Cov(k,GDP_trend)/Var(Δy)', 2),
            ('Cov_k_h_T', '2Cov(k,h(T))/Var(Δy)', 2),
            ('Cov_k_h_Ttrend', '-2Cov(k,h(T_trend))/Var(Δy)', -2),  # Negative sign
            ('Cov_k_epsilon', '2Cov(k,ε)/Var(Δy)', 2),
            ('Cov_GDP_trend_h_T', '2Cov(GDP_trend,h(T))/Var(Δy)', 2),
            ('Cov_GDP_trend_h_Ttrend', '-2Cov(GDP_trend,h(T_trend))/Var(Δy)', -2),  # Negative sign
            ('Cov_GDP_trend_epsilon', '2Cov(GDP_trend,ε)/Var(Δy)', 2),
            ('Cov_h_T_h_Ttrend', '-2Cov(h(T),h(T_trend))/Var(Δy)', -2),  # Negative sign
            ('Cov_h_T_epsilon', '2Cov(h(T),ε)/Var(Δy)', 2),
            ('Cov_h_Ttrend_epsilon', '-2Cov(h(T_trend),ε)/Var(Δy)', -2),  # Negative sign
        ]

        rows_pl = []
        for metric_key, label, multiplier in pl_metrics:
            row = {'Metric': label}
            for approach in available_pl:
                mask = var_attrib_df['approach'] == approach
                approach_data = var_attrib_df[mask]

                if 'var_dy' not in approach_data.columns:
                    for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                        row[f'{approach}{suffix}'] = np.nan
                    continue

                # Get point estimate (iteration -1)
                point_estimate = None
                if 'iteration' in approach_data.columns:
                    point_mask = approach_data['iteration'] == -1
                    if point_mask.any():
                        point_data = approach_data[point_mask]
                        samples = get_pl_metric(point_data, metric_key)
                        if samples is not None and len(samples) > 0:
                            var_dy = point_data['var_dy'].values[0]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                point_estimate = (samples[0] * multiplier) / var_dy
                            if not np.isfinite(point_estimate):
                                point_estimate = None

                # Get bootstrap samples (iteration >= 0)
                bootstrap_mask = approach_data['iteration'] >= 0
                bootstrap_data = approach_data[bootstrap_mask]

                metric_samples = get_pl_metric(bootstrap_data, metric_key)
                if metric_samples is None:
                    for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                        row[f'{approach}{suffix}'] = np.nan
                    continue

                var_dy_samples = bootstrap_data['var_dy'].values
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = (metric_samples * multiplier) / var_dy_samples
                normalized = np.where(np.isfinite(normalized), normalized, np.nan)

                stats = compute_stats(normalized, point_estimate=point_estimate)
                row[f'{approach}_point'] = stats['point']
                row[f'{approach}_p5'] = stats['p5']
                row[f'{approach}_p25'] = stats['p25']
                row[f'{approach}_p50'] = stats['p50']
                row[f'{approach}_p75'] = stats['p75']
                row[f'{approach}_p95'] = stats['p95']

            rows_pl.append(row)

        # Add Sum row for P/L approaches
        sum_row_pl = {'Metric': 'Sum'}
        for approach in available_pl:
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if 'var_dy' not in approach_data.columns:
                for suffix in ['_point', '_p5', '_p25', '_p50', '_p75', '_p95']:
                    sum_row_pl[f'{approach}{suffix}'] = np.nan
                continue

            # Point estimate sum
            point_sum = None
            if 'iteration' in approach_data.columns:
                point_mask = approach_data['iteration'] == -1
                if point_mask.any():
                    point_data = approach_data[point_mask]
                    var_dy = point_data['var_dy'].values[0]
                    total = 0.0
                    for metric_key, _, mult in pl_metrics:
                        samples = get_pl_metric(point_data, metric_key)
                        if samples is not None and len(samples) > 0:
                            total += samples[0] * mult
                    with np.errstate(divide='ignore', invalid='ignore'):
                        point_sum = total / var_dy
                    if not np.isfinite(point_sum):
                        point_sum = None

            # Bootstrap sums
            bootstrap_mask = approach_data['iteration'] >= 0
            bootstrap_data = approach_data[bootstrap_mask]
            var_dy_samples = bootstrap_data['var_dy'].values

            total_sum = np.zeros(len(bootstrap_data))
            for metric_key, _, mult in pl_metrics:
                samples = get_pl_metric(bootstrap_data, metric_key)
                if samples is not None:
                    total_sum += samples * mult

            with np.errstate(divide='ignore', invalid='ignore'):
                sum_normalized = total_sum / var_dy_samples
            sum_normalized = np.where(np.isfinite(sum_normalized), sum_normalized, np.nan)

            stats = compute_stats(sum_normalized, point_estimate=point_sum)
            sum_row_pl[f'{approach}_point'] = stats['point']
            sum_row_pl[f'{approach}_p5'] = stats['p5']
            sum_row_pl[f'{approach}_p25'] = stats['p25']
            sum_row_pl[f'{approach}_p50'] = stats['p50']
            sum_row_pl[f'{approach}_p75'] = stats['p75']
            sum_row_pl[f'{approach}_p95'] = stats['p95']

        rows_pl.append(sum_row_pl)

        df_pl = pd.DataFrame(rows_pl)
        xlsx_path_pl = output_dir / 'covariance_table_PL.xlsx'
        df_pl.to_excel(xlsx_path_pl, index=False, sheet_name='PL Approaches')
        csv_path_pl = output_dir / 'covariance_table_PL.csv'
        df_pl.to_csv(csv_path_pl, index=False)
        print(f"      [Tables] Saved covariance_table_PL.xlsx ({len(rows_pl)} rows × {len(available_pl)} approaches)")


def generate_tables(
    analysis_results: dict,
    bootstrap_results: dict,
    output_dir: Path,
) -> None:
    """Generate publication-quality tables.

    Parameters
    ----------
    analysis_results : dict
        Dictionary from load_analysis_results() containing:
        - 'comparison_table': DataFrame with main results
        - 'country_trends': DataFrame with country-level trends
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_coefficients': DataFrame with all samples
        - 'bootstrap_summary': DataFrame with percentiles
        - 'bootstrap_var_attrib': DataFrame with variance attribution samples
    output_dir : Path
        Directory to save generated tables
    """
    # Approaches to include in tables
    table_approaches = [
        'Approach NJ', 'Approach NP', 'Approach NL',
        'Approach QJ', 'Approach QP', 'Approach QL',
        'Approach PJ', 'Approach PP', 'Approach PL',
        'Approach DJ', 'Approach DP', 'Approach DL',
    ]

    # Generate variance decomposition table
    generate_variance_decomposition_table(bootstrap_results, output_dir, approaches=table_approaches)

    # Generate bootstrap comparison table
    generate_bootstrap_comparison_table(bootstrap_results, output_dir, approaches=table_approaches)

    # Generate variance decomposition tables by response function type
    generate_variance_decomposition_by_response(bootstrap_results, output_dir)

    # Generate covariance tables (separate for J vs P/L approaches)
    generate_covariance_tables(bootstrap_results, output_dir)


def generate_figures(
    analysis_results: dict,
    bootstrap_results: dict,
    output_dir: Path,
    data: AnalysisData = None,
) -> None:
    """Generate publication-quality figures.

    Parameters
    ----------
    analysis_results : dict
        Dictionary from load_analysis_results() containing:
        - 'comparison_table': DataFrame with main results
        - 'country_trends': DataFrame with country-level trends
    bootstrap_results : dict
        Dictionary from load_bootstrap_results() containing:
        - 'bootstrap_coefficients': DataFrame with all samples
        - 'bootstrap_summary': DataFrame with percentiles
    output_dir : Path
        Directory to save generated figures
    data : AnalysisData, optional
        Input data for temperature histogram overlay
    """
    # Check if we have the required data
    if bootstrap_results.get('bootstrap_coefficients') is None:
        print("      [Figures] ERROR: bootstrap_coefficients.csv not loaded")
        return
    if bootstrap_results.get('bootstrap_summary') is None:
        print("      [Figures] ERROR: bootstrap_summary_table.csv not loaded")
        return

    # Reconstruct BootstrapResult objects from CSV data
    print("      [Figures] Reconstructing bootstrap results from CSV...")
    results = reconstruct_bootstrap_results(
        bootstrap_results['bootstrap_coefficients'],
        bootstrap_results['bootstrap_summary'],
        bootstrap_results.get('bootstrap_k_samples'),
    )
    print(f"      [Figures] Reconstructed {len(results)} approaches")

    # Year effects figure (2 panels)
    if data is not None:
        print("      [Figures] Generating year effects figure...")
        plot_year_effects_2panel(
            results,
            data,
            output_dir,
            approaches=['Approach QJ', 'Approach QL'],
            filename='fig_year_effects_main.pdf',
            input_file=None,
        )
        print("      [Figures] Saved fig_year_effects_main.pdf")

    # =========================================================================
    # 3x3 and 4x3 Grid Figures (all 9 main approaches)
    # =========================================================================

    # Temperature response 3x3 grid
    if data is not None:
        print("      [Figures] Generating temperature response figure (3x3 grid)...")
        plot_temperature_response_3x3(
            results,
            data,
            output_dir,
            filename='fig_temperature_response_3x3.pdf',
            T_range=(0, 30),
            input_file=None,
        )
        print("      [Figures] Saved fig_temperature_response_3x3.pdf")

    # Figure 12: Temperature derivative 3x3 grid
    print("      [Figures] Generating temperature derivative figure (3x3 grid)...")
    plot_temperature_derivative_3x3(
        results,
        output_dir,
        filename='fig_temperature_derivative_3x3.pdf',
        T_range=(0, 30),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_derivative_3x3.pdf")

    # Figure 13: T_optimal histogram 3x3 grid
    print("      [Figures] Generating T_optimal histogram figure (3x3 grid)...")
    plot_T_optimal_histogram_3x3(
        results,
        output_dir,
        filename='fig_T_optimal_histogram_3x3.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_T_optimal_histogram_3x3.pdf")

    # Figure 14: h2 histogram 4x3 grid (separate rows for piecewise h2 and h4)
    print("      [Figures] Generating h2 histogram figure (4x3 grid)...")
    plot_h2_histogram_4x3(
        results,
        output_dir,
        filename='fig_h2_histogram_4x3.pdf',
        x_range=(-0.001, 0.0001),
        bin_width=0.00002,
        input_file=None,
    )
    print("      [Figures] Saved fig_h2_histogram_4x3.pdf")

    # Figure 15: h4 histogram 1x3 row (persistence approaches only)
    print("      [Figures] Generating h4 histogram figure (1x3 row)...")
    plot_h4_histogram_1x3(
        results,
        output_dir,
        filename='fig_h4_histogram_1x3.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_h4_histogram_1x3.pdf")

    # Note: Year effects figure is now fig_year_effects_main.pdf (separate from temperature response)
