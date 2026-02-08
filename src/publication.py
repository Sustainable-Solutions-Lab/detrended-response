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
)


def reconstruct_bootstrap_results(
    coefficients_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Dict[str, BootstrapResult]:
    """Reconstruct BootstrapResult objects from CSV data.

    Parameters
    ----------
    coefficients_df : pd.DataFrame
        DataFrame from bootstrap_coefficients.csv with columns:
        iteration, approach, approach_name, h1, h2, T_optimal, r_squared, etc.
    summary_df : pd.DataFrame
        DataFrame from bootstrap_summary_table.csv with point estimates and percentiles

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
        )

        results[approach] = result

    return results


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
    output_dir : Path
        Directory to save generated tables
    """
    print("      [Tables] Placeholder - no tables generated yet")


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
