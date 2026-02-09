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
        iteration, approach, approach_name, h1, h2, T_optimal, r_squared, etc.
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
        T_optimal_samples = samples['T_optimal'].values
        r_squared_samples = samples['r_squared'].values
        total_r_squared_samples = samples['total_r_squared'].values

        # Get point estimates from summary
        h1_point = summary_row['h1_point']
        h2_point = summary_row['h2_point']
        T_optimal_point = summary_row['T_optimal_point']
        r_squared_point = summary_row['r_squared_point']
        total_r_squared_point = summary_row['total_r_squared_point']

        # Get approach name
        approach_name = summary_row['approach_name']

        # Handle optional fields (beta for approach7, h2_low/h2_high for approach8)
        beta_point = None
        beta_samples = None
        h2_low_point = None
        h2_low_samples = None
        h2_high_point = None
        h2_high_samples = None

        if 'beta' in samples.columns and not samples['beta'].isna().all():
            beta_samples = samples['beta'].values
            beta_point = summary_row.get('beta_point', None)

        if 'h2_low' in samples.columns and not samples['h2_low'].isna().all():
            h2_low_samples = samples['h2_low'].values
            h2_low_point = summary_row.get('h2_low_point', None)

        if 'h2_high' in samples.columns and not samples['h2_high'].isna().all():
            h2_high_samples = samples['h2_high'].values
            h2_high_point = summary_row.get('h2_high_point', None)

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
            T_optimal_point=T_optimal_point,
            r_squared_point=r_squared_point,
            total_r_squared_point=total_r_squared_point,
            h1_samples=h1_samples,
            h2_samples=h2_samples,
            T_optimal_samples=T_optimal_samples,
            r_squared_samples=r_squared_samples,
            total_r_squared_samples=total_r_squared_samples,
            n_bootstrap=int(summary_row['n_bootstrap']),
            n_successful=int(summary_row['n_successful']),
            beta_point=beta_point,
            beta_samples=beta_samples,
            h2_low_point=h2_low_point,
            h2_low_samples=h2_low_samples,
            h2_high_point=h2_high_point,
            h2_high_samples=h2_high_samples,
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
    approaches as columns, showing point estimates with 90% confidence intervals.

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
    Row Labels              | Approach 0        | Approach 1        | ...
    ------------------------|-------------------|-------------------|----
    Total R²                | 0.196 [0.18,0.21] | 0.192 [0.17,0.20] | ...
    Var(h(T)-h(Ttr))/Var    | 0.002 [0.00,0.01] | —                 | ...
    Var(h(Ttr))/Var(dy)     | 0.175 [0.15,0.20] | —                 | ...
    ...                     | ...               | ...               | ...
    Sum                     | 1.000             | 1.000             | ...
    """
    var_attrib_df = bootstrap_results.get('bootstrap_var_attrib')
    summary_df = bootstrap_results.get('bootstrap_summary')

    if var_attrib_df is None:
        print("      [Tables] WARNING: bootstrap_var_attrib not loaded, skipping variance decomposition table")
        return
    if summary_df is None:
        print("      [Tables] WARNING: bootstrap_summary not loaded, skipping variance decomposition table")
        return

    # Default approaches to include
    if approaches is None:
        approaches = ['approach0', 'nocr0', 'approach5c', 'nocr5', 'approach5a', 'approach5b', 'approach6', 'approach8']

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

    def format_value_with_ci(point: float, samples: np.ndarray, is_covariance: bool = False) -> str:
        """Format point estimate with 90% CI, or em-dash for zero-by-construction values."""
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            return '—'

        # Check if all samples are identically zero (zero by construction)
        if np.allclose(valid_samples, 0, atol=1e-15):
            return '—'

        p5 = np.percentile(valid_samples, 5)
        p95 = np.percentile(valid_samples, 95)

        # Format with appropriate precision
        if abs(point) < 0.001:
            return f'{point:.4f} [{p5:.4f},{p95:.4f}]'
        else:
            return f'{point:.3f} [{p5:.3f},{p95:.3f}]'

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
        # Get point estimate from summary
        summary_mask = summary_df['approach'] == approach
        if summary_mask.any():
            point = summary_df[summary_mask].iloc[0]['total_r_squared_point']
        else:
            point = np.nan

        # Get samples from coefficients
        if coefficients_df is not None:
            coef_mask = coefficients_df['approach'] == approach
            samples = coefficients_df[coef_mask]['total_r_squared'].values
        else:
            samples = np.array([])

        total_r2_row[approach_names[approach]] = format_value_with_ci(point, samples)
    rows.append(total_r2_row)

    # Add variance metrics
    for key, label in variance_metrics:
        row = {'Metric': label}
        for approach in available_approaches:
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if key not in approach_data.columns:
                row[approach_names[approach]] = '—'
                continue

            # Get var_dy for normalization
            if 'var_dy' in approach_data.columns:
                var_dy_samples = approach_data['var_dy'].values
            else:
                row[approach_names[approach]] = '—'
                continue

            # Get the metric samples and normalize
            metric_samples = approach_data[key].values
            # Normalize by var_dy
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            # Point estimate is mean of samples (or could use median)
            valid = normalized_samples[~np.isnan(normalized_samples)]
            if len(valid) > 0:
                point = np.median(valid)
                row[approach_names[approach]] = format_value_with_ci(point, normalized_samples)
            else:
                row[approach_names[approach]] = '—'

        rows.append(row)

    # Add covariance metrics (multiply by 2 since we show 2*Cov)
    for key, label in covariance_metrics:
        row = {'Metric': label}
        for approach in available_approaches:
            mask = var_attrib_df['approach'] == approach
            approach_data = var_attrib_df[mask]

            if key not in approach_data.columns:
                row[approach_names[approach]] = '—'
                continue

            # Get var_dy for normalization
            if 'var_dy' in approach_data.columns:
                var_dy_samples = approach_data['var_dy'].values
            else:
                row[approach_names[approach]] = '—'
                continue

            # Get the metric samples, multiply by 2, and normalize
            metric_samples = approach_data[key].values * 2  # 2*Cov term
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_samples = metric_samples / var_dy_samples
            normalized_samples = np.where(np.isfinite(normalized_samples), normalized_samples, np.nan)

            valid = normalized_samples[~np.isnan(normalized_samples)]
            if len(valid) > 0:
                point = np.median(valid)
                row[approach_names[approach]] = format_value_with_ci(point, normalized_samples, is_covariance=True)
            else:
                row[approach_names[approach]] = '—'

        rows.append(row)

    # Add Sum row (should equal 1.0)
    sum_row = {'Metric': 'Sum'}
    for approach in available_approaches:
        mask = var_attrib_df['approach'] == approach
        approach_data = var_attrib_df[mask]

        if 'var_dy' not in approach_data.columns:
            sum_row[approach_names[approach]] = '—'
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

        valid = sum_normalized[~np.isnan(sum_normalized)]
        if len(valid) > 0:
            point = np.median(valid)
            sum_row[approach_names[approach]] = f'{point:.3f}'
        else:
            sum_row[approach_names[approach]] = '—'

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

    # Figure 1: Temperature response (h(T) - h(T*)) - 4 panels
    print("      [Figures] Generating temperature response figure (4 panels)...")
    plot_bootstrap_temperature_response(
        results,
        output_dir,
        approaches=approaches_4panel,
        filename='fig_temperature_response_4panel.pdf',
        T_range=(0, 30),
        data=data,  # Include temperature histogram
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_response_4panel.pdf")

    # Figure 2: Temperature response (h(T) - h(T*)) - 2 panels
    print("      [Figures] Generating temperature response figure (2 panels)...")
    plot_bootstrap_temperature_response(
        results,
        output_dir,
        approaches=approaches_2panel,
        filename='fig_temperature_response_2panel.pdf',
        T_range=(0, 30),
        data=data,  # Include temperature histogram
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_response_2panel.pdf")

    # Figure 3: Temperature derivative (dh/dT) - 4 panels
    print("      [Figures] Generating temperature derivative figure (4 panels)...")
    plot_bootstrap_temperature_derivative(
        results,
        output_dir,
        approaches=approaches_4panel,
        filename='fig_temperature_derivative_4panel.pdf',
        T_range=(0, 30),
        input_file=None,
    )
    print("      [Figures] Saved fig_temperature_derivative_4panel.pdf")

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

    # Figure 9: Year effects k(t) - 2 panels (approach0 and approach6)
    if data is not None:
        print("      [Figures] Generating year effects figure (2 panels)...")
        plot_year_effects_bootstrap(
            results,
            data,
            output_dir,
            approaches_to_plot=['approach0', 'approach6'],
            filename='fig_year_effects_2panel.pdf',
            show_title=False,
            input_file=None,
        )
        print("      [Figures] Saved fig_year_effects_2panel.pdf")
    else:
        print("      [Figures] Skipping year effects figure (data not loaded)")
