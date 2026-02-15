"""Publication-specific formatting for tables and figures.

This module provides functions to generate publication-quality tables and figures
from pre-computed analysis and bootstrap results.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.bootstrap import BootstrapResult
from src.data_loader import AnalysisData
from src.output import (
    plot_bootstrap_temperature_response,
    plot_bootstrap_temperature_derivative,
    plot_T_optimal_histograms,
    plot_h2_histograms,
    plot_year_effects_bootstrap,
    plot_combined_temp_response_and_year_effects,
    plot_climate_response_contours,
    plot_temperature_response_2panel,
    plot_year_effects_2panel,
    plot_temperature_response_4panel_variants,
    plot_temperature_derivative_4panel_variants,
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
        # Get samples for this approach
        mask = coefficients_df['approach'] == approach
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
                # Get unique years
                years = sorted(k_data['year'].unique())
                n_bootstrap = int(summary_row['n_bootstrap'])

                # Build k_samples dict: year -> array of shape (n_bootstrap,)
                k_samples = {}
                for year in years:
                    year_data = k_data[k_data['year'] == year].sort_values('iteration')
                    k_samples[year] = year_data['k_value'].values

                # Compute k_point as median of each year's samples
                k_point = {}
                for year in years:
                    valid = k_samples[year][~np.isnan(k_samples[year])]
                    if len(valid) > 0:
                        k_point[year] = np.median(valid)
                    else:
                        k_point[year] = np.nan

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
    if approaches is None:
        approaches = ['approach0', 'nocr0', 'approach5c', 'nocr5', 'approach6', 'approach8', 'approach6e']

    # Filter to approaches that exist in the data
    available_approaches = [a for a in approaches if a in var_attrib_df['approach'].values]
    if not available_approaches:
        print("      [Tables] WARNING: No matching approaches found in var_attrib data")
        return

    # Define the metrics to include in the table (in order)
    # These correspond to var_attrib keys, normalized by var_dy
    variance_metrics = [
        ('Sigma_Delta_u_Delta_u', 'Var(h(T)-h(Ttr))/Var(Δy)'),
        ('Sigma_v_v', 'Var(h(Ttr))/Var(Δy)'),
        ('Sigma_j_j', 'Var(j)/Var(Δy)'),
        ('Sigma_k_k', 'Var(k)/Var(Δy)'),
        ('Sigma_epsilon_epsilon', 'Var(ε)/Var(Δy)'),
    ]
    covariance_metrics = [
        ('Sigma_Delta_u_v', '2Cov(h(T)-h(Ttr),h(Ttr))/Var(Δy)'),
        ('Sigma_Delta_u_j', '2Cov(h(T)-h(Ttr),j)/Var(Δy)'),
        ('Sigma_Delta_u_k', '2Cov(h(T)-h(Ttr),k)/Var(Δy)'),
        ('Sigma_Delta_u_epsilon', '2Cov(h(T)-h(Ttr),ε)/Var(Δy)'),
        ('Sigma_v_j', '2Cov(h(Ttr),j)/Var(Δy)'),
        ('Sigma_v_k', '2Cov(h(Ttr),k)/Var(Δy)'),
        ('Sigma_v_epsilon', '2Cov(h(Ttr),ε)/Var(Δy)'),
        ('Sigma_j_k', '2Cov(j,k)/Var(Δy)'),
        ('Sigma_j_epsilon', '2Cov(j,ε)/Var(Δy)'),
        ('Sigma_k_epsilon', '2Cov(k,ε)/Var(Δy)'),
    ]

    # Percentiles to compute
    percentiles = [5, 25, 50, 75, 95]

    def compute_stats(samples: np.ndarray) -> dict:
        """Compute point estimate (median) and percentiles from samples."""
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            return {'point': np.nan, 'p5': np.nan, 'p25': np.nan, 'p50': np.nan, 'p75': np.nan, 'p95': np.nan}

        return {
            'point': np.median(valid_samples),
            'p5': np.percentile(valid_samples, 5),
            'p25': np.percentile(valid_samples, 25),
            'p50': np.percentile(valid_samples, 50),
            'p75': np.percentile(valid_samples, 75),
            'p95': np.percentile(valid_samples, 95),
        }

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

    # Add Total R² row first (from bootstrap_summary)
    total_r2_row = {'Metric': 'Total R²'}
    coefficients_df = bootstrap_results.get('bootstrap_coefficients')
    for approach in available_approaches:
        name = approach_names[approach]

        # Get samples from coefficients
        if coefficients_df is not None:
            coef_mask = coefficients_df['approach'] == approach
            samples = coefficients_df[coef_mask]['total_r_squared'].values
        else:
            samples = np.array([])

        stats = compute_stats(samples)
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

            if key not in approach_data.columns or 'var_dy' not in approach_data.columns:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get var_dy for normalization
            var_dy_samples = approach_data['var_dy'].values

            # Get the metric samples and normalize
            metric_samples = approach_data[key].values
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            stats = compute_stats(normalized_samples)
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

            if key not in approach_data.columns or 'var_dy' not in approach_data.columns:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            # Get var_dy for normalization
            var_dy_samples = approach_data['var_dy'].values

            # Get the metric samples, multiply by 2, and normalize
            metric_samples = approach_data[key].values * 2  # 2*Cov term
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            stats = compute_stats(normalized_samples)
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

        var_dy_samples = approach_data['var_dy'].values

        # Sum all variance and covariance terms
        total_sum = np.zeros(len(approach_data))
        for key, _ in variance_metrics:
            if key in approach_data.columns:
                total_sum += approach_data[key].values
        for key, _ in covariance_metrics:
            if key in approach_data.columns:
                total_sum += 2 * approach_data[key].values  # 2*Cov terms

        with np.errstate(divide='ignore', invalid='ignore'):
            sum_normalized = total_sum / var_dy_samples
        sum_normalized = np.where(np.isfinite(sum_normalized), sum_normalized, np.nan)

        stats = compute_stats(sum_normalized)
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
    if approaches is None:
        approaches = ['approach0', 'nocr0', 'approach5c', 'nocr5', 'approach6', 'approach8', 'approach6e']

    # Filter to approaches that exist in the data
    available_approaches = [a for a in approaches if a in summary_df['approach'].values]
    if not available_approaches:
        print("      [Tables] WARNING: No matching approaches found in summary data")
        return

    # Parameters to include
    # Standard parameters for all approaches
    standard_params = ['h1', 'h2', 'T_opt', 'total_r_squared']
    # Additional parameters for approach 6b/6c/6e (departure/trend coefficients: h3, h4)
    trend_params = ['h3', 'h4', 'T_dep_opt', 'f1', 'f2']
    # Note: For piecewise approach8, h2 = curvature below T_opt, h4 = curvature above T_opt
    # For approach 8a, h2 = actual T curvature, h4 = trend T curvature

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
    # Generate variance decomposition table
    generate_variance_decomposition_table(bootstrap_results, output_dir)

    # Generate bootstrap comparison table
    generate_bootstrap_comparison_table(bootstrap_results, output_dir)


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

    # Define approach patterns for publication figures
    # 4-panel layout: [[row0: 0, 5c], [row1: 5a, 5b]]
    approaches_4panel = ['approach0', 'approach5c', 'approach5a', 'approach5b']
    # 2-panel layout: [col0: 6, col1: 8]
    approaches_2panel = ['approach6', 'approach8']

    # Figure 1a: Temperature response (2 panels)
    if data is not None:
        print("      [Figures] Generating temperature response figure...")
        plot_temperature_response_2panel(
            results,
            data,
            output_dir,
            approaches=['approach0', 'approach5c'],
            filename='fig_temperature_response_main.pdf',
            T_range=(0, 30),
            input_file=None,
        )
        print("      [Figures] Saved fig_temperature_response_main.pdf")

        # Figure 1b: Year effects (2 panels)
        print("      [Figures] Generating year effects figure...")
        plot_year_effects_2panel(
            results,
            data,
            output_dir,
            approaches=['approach0', 'approach6'],
            filename='fig_year_effects_main.pdf',
            input_file=None,
        )
        print("      [Figures] Saved fig_year_effects_main.pdf")
    else:
        print("      [Figures] Skipping main figures (data not loaded)")

    # Figure 2: Temperature derivative (dh/dT) - 4 panels (variants: 6, 8, 6e components)
    print("      [Figures] Generating temperature derivative figure (4 panels - variants)...")
    plot_temperature_derivative_4panel_variants(
        results,
        output_dir,
        filename='fig_temperature_derivative_4panel_variants.pdf',
        T_range=(0, 30),
        T_dep_range=(-5, 5),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_derivative_4panel_variants.pdf")

    # Figure 4: Temperature derivative (dh/dT) - 2 panels
    print("      [Figures] Generating temperature derivative figure (2 panels)...")
    plot_bootstrap_temperature_derivative(
        results,
        output_dir,
        approaches=approaches_2panel,
        filename='fig_temperature_derivative_2panel.pdf',
        T_range=(0, 30),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_derivative_2panel.pdf")

    # Figure 5: T_optimal histograms - 4 panels
    print("      [Figures] Generating T_optimal histogram figure (4 panels)...")
    plot_T_optimal_histograms(
        results,
        output_dir,
        approaches=approaches_4panel,
        filename='fig_T_optimal_histogram_4panel.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_T_optimal_histogram_4panel.pdf")

    # Figure 6: T_optimal histograms - 2 panels
    print("      [Figures] Generating T_optimal histogram figure (2 panels)...")
    plot_T_optimal_histograms(
        results,
        output_dir,
        approaches=approaches_2panel,
        filename='fig_T_optimal_histogram_2panel.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_T_optimal_histogram_2panel.pdf")

    # Figure 7: h2 coefficient histograms - 4 panels [[approach0, approach5c], [approach5a, approach5b]]
    print("      [Figures] Generating h2 coefficient histogram figure (4 panels)...")
    plot_h2_histograms(
        results,
        output_dir,
        approaches=['approach0', 'approach5c', 'approach5a', 'approach5b'],
        x_range=(-0.001, 0.0001),
        bin_width=0.00002,
        filename='fig_h2_histogram_4panel.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_h2_histogram_4panel.pdf")

    # Figure 8: h2 coefficient histograms - 3 panels (approach6 h2, approach8 h2_low, approach8 h2_high)
    print("      [Figures] Generating h2 coefficient histogram figure (3 panels)...")
    plot_h2_histograms(
        results,
        output_dir,
        approaches=['approach6', 'approach8'],
        x_range=(-0.001, 0.0001),
        bin_width=0.00002,
        x_range_h2_high=(-0.01, 0.001),
        bin_width_h2_high=0.0002,
        filename='fig_h2_histogram_3panel.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_h2_histogram_3panel.pdf")

    # Figure 9: Temperature response - 4 panels for variant approaches (6, 8, 6e components)
    print("      [Figures] Generating temperature response figure (4 panels - variants)...")
    plot_temperature_response_4panel_variants(
        results,
        data,
        output_dir,
        filename='fig_temperature_response_4panel_variants.pdf',
        T_range=(0, 30),
        T_dep_range=(-5, 5),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_response_4panel_variants.pdf")

    # Keep the 3-panel variants for 6b, 6e, 8a
    approaches_variants = ['approach6b', 'approach6e', 'approach8a']

    # Figure 10: Temperature derivative - 3 panels for variant approaches (6b, 6e, 8a)
    print("      [Figures] Generating temperature derivative figure (3 panels - variants)...")
    plot_bootstrap_temperature_derivative(
        results,
        output_dir,
        approaches=approaches_variants,
        filename='fig_temperature_derivative_3panel_variants.pdf',
        T_range=(0, 30),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_derivative_3panel_variants.pdf")

    # Figure 11: T_optimal histograms - 3 panels for variant approaches (6b, 6e, 8a)
    print("      [Figures] Generating T_optimal histogram figure (3 panels - variants)...")
    plot_T_optimal_histograms(
        results,
        output_dir,
        approaches=approaches_variants,
        filename='fig_T_optimal_histogram_3panel_variants.pdf',
        input_file=None,
    )
    print("      [Figures] Saved fig_T_optimal_histogram_3panel_variants.pdf")

    # Figure 12: Climate response contours for approaches 6e and 8a
    print("      [Figures] Generating climate response contour figure (6e, 8a)...")
    plot_climate_response_contours(
        results,
        output_dir,
        approaches=['approach6e', 'approach8a'],
        filename='fig_climate_response_contours.pdf',
        Ttrend_range=(0, 30),
        deltaT_range=(-5, 5),
        input_file=None,
    )
    print("      [Figures] Saved fig_climate_response_contours.pdf")

    # Note: Year effects figure is now fig_year_effects_main.pdf (separate from temperature response)
