#!/usr/bin/env python3
"""Calculate cumulative climate effects from bootstrap h(T) values.

This script processes bootstrap_h_values.csv to compute cumulative climate effects
from 1961-2022, select representative countries, and create a box-and-whisker
visualization.

Usage:
    python scripts/calculate_cumulative_effects.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detrending import fit_quadratic_trend, DEFAULT_LOESS_WINDOW_YEARS
from src.output import METHOD_COLORS, create_output_dir, add_input_file_annotation


def find_most_recent_dir(pattern: str) -> str:
    """Find the most recent directory matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern to match directories (e.g., 'data/output/reference/bootstrap_*')

    Returns
    -------
    str
        Path to the most recent matching directory, or None if no match found
    """
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return None


# ==============================================================================
# Constants
# ==============================================================================

# Approaches that use quadratic OLS for h(T) trend
QUADRATIC_METHODS = {'method0', 'method1'}

# Approaches that use LOESS for h(T) trend
LOESS_METHODS = {'method2', 'method4', 'method3'}

# Central approaches for analysis (in display order)
CENTRAL_METHODS = ['method0', 'method1', 'method2', 'method3', 'method4']

# Base year for cumulative effect calculation
BASE_YEAR = 1961

# Percentiles for selecting representative countries
REPRESENTATIVE_PERCENTILES = (5, 25, 50, 75, 95)


# ==============================================================================
# Core Functions
# ==============================================================================

def fit_h_T_trend_1961(
    years: np.ndarray,
    h_T_values: np.ndarray,
    approach: str,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS
) -> float:
    """Fit trend to h(T) and evaluate at 1961.

    For quadratic approaches (method0, method1): fit h_T ~ year + year²
    For LOESS approaches (method2, method4, method3): use LOESS smoothing

    Args:
        years: Array of year values
        h_T_values: Array of h(T) values
        approach: Approach name
        loess_window: Window size in years for LOESS smoothing

    Returns:
        Trend value at 1961
    """
    if approach in QUADRATIC_METHODS:
        # Fit quadratic: h_T = a + b*year + c*year²
        a, b, c = fit_quadratic_trend(years, h_T_values)
        return a + b * BASE_YEAR + c * BASE_YEAR ** 2
    else:
        # LOESS smoothing
        n = len(years)
        t_range = years.max() - years.min()
        frac = min(loess_window / t_range, 1.0) if t_range > 0 else 1.0
        frac = max(frac, 3.0 / n)  # Ensure minimum points

        # Sort data for LOESS
        sort_idx = np.argsort(years)
        years_sorted = years[sort_idx]
        h_T_sorted = h_T_values[sort_idx]

        # Fit LOESS
        smoothed = lowess(h_T_sorted, years_sorted, frac=frac, return_sorted=False)

        # Interpolate to get value at 1961
        return np.interp(BASE_YEAR, years_sorted, smoothed)


def calculate_h_T_delta_cumulative(h_T_delta: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Calculate compound cumulative effect.

    For year 1961: h_T_delta_cum = h_T_delta
    For year > 1961: h_T_delta_cum(year) = (1 + h_T_delta_cum(year-1)) * (1 + h_T_delta(year)) - 1

    Args:
        h_T_delta: Array of h_T - h_T_trend_1961 values
        years: Array of year values (must be sorted)

    Returns:
        Array of compound cumulative effects
    """
    # Sort by year to ensure proper ordering
    sort_idx = np.argsort(years)
    h_T_delta_sorted = h_T_delta[sort_idx]

    # Calculate compound cumulative
    h_T_delta_cum = np.zeros(len(h_T_delta))
    h_T_delta_cum[0] = h_T_delta_sorted[0]

    for i in range(1, len(h_T_delta_sorted)):
        h_T_delta_cum[i] = (1 + h_T_delta_cum[i - 1]) * (1 + h_T_delta_sorted[i]) - 1

    # Restore original order
    result = np.zeros(len(h_T_delta))
    result[sort_idx] = h_T_delta_cum

    return result


def log_transform(pct):
    """Transform percentage to log scale: log(1 + pct/100)."""
    return np.log(1 + pct / 100)


def inv_log_transform(y):
    """Inverse transform: from log scale back to percentage."""
    return (np.exp(y) - 1) * 100


def get_country_ordering(representatives: dict) -> list:
    """Get ordered list of country keys: min, P5, P25, P50, P75, P95, max.

    Args:
        representatives: Dictionary from select_representative_countries

    Returns:
        List of keys in proper order
    """
    # Define the display order
    ordering = []
    if 'min' in representatives:
        ordering.append('min')
    # Add percentiles in sorted order
    percentile_keys = sorted([k for k in representatives.keys() if isinstance(k, int)])
    ordering.extend(percentile_keys)
    if 'max' in representatives:
        ordering.append('max')
    return ordering


def get_country_label(key, representatives: dict) -> str:
    """Get display label for a country key.

    Args:
        key: Either 'min', 'max', or an integer percentile
        representatives: Dictionary with country info

    Returns:
        Formatted label string
    """
    iso3 = representatives[key]['iso3']
    if key == 'min':
        return f"{iso3}\n(Min)"
    elif key == 'max':
        return f"{iso3}\n(Max)"
    else:
        return f"{iso3}\n(P{key})"


def plot_cumulative_effects_boxplot(
    df: pd.DataFrame,
    representatives: dict,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create clustered box-and-whisker plot of cumulative effects.

    Uses log(1 + pct/100) transform so that -50% and +100% are equidistant from 0.
    Groups by country (min, P5, P25, P50, P75, P95, max) with 5 method bars per country.

    Args:
        df: DataFrame with cumulative effects for representative countries
        representatives: Dictionary from select_representative_countries
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use central approaches in consistent order
    approaches = CENTRAL_METHODS
    n_approaches = len(approaches)

    # Get ordered country keys (min, P5, P25, P50, P75, P95, max)
    country_keys = get_country_ordering(representatives)
    n_clusters = len(country_keys)

    # Spacing parameters
    cluster_width = 0.8
    box_width = cluster_width / (n_approaches + 1)  # Extra space between clusters

    # Filter to year 2022 for final values
    df_2022 = df[df['year'] == 2022].copy()

    # Create box plots
    for i, country_key in enumerate(country_keys):
        iso3 = representatives[country_key]['iso3']
        cluster_center = i

        for j, approach in enumerate(approaches):
            # Get bootstrap samples (iterations 0-999) for this country/approach
            mask = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] >= 0)
            bootstrap_values_pct = df_2022.loc[mask, 'h_T_delta_cum'].values * 100  # Convert to percent

            # Transform to log scale
            bootstrap_values = log_transform(bootstrap_values_pct)

            # Get point estimate (iteration -1)
            mask_point = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] == -1)
            point_estimate_pct = df_2022.loc[mask_point, 'h_T_delta_cum'].values
            point_estimate_pct = point_estimate_pct[0] * 100 if len(point_estimate_pct) > 0 else np.nan
            point_estimate = log_transform(point_estimate_pct)

            # Position for this box
            pos = cluster_center + (j - (n_approaches - 1) / 2) * box_width

            # Draw box
            color = METHOD_COLORS.get(approach, 'gray')
            box = ax.boxplot(
                [bootstrap_values],
                positions=[pos],
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=False,
                whis=[5, 95],  # Whiskers at 5th and 95th percentile
                medianprops=dict(color='black', linewidth=1),
            )

            # Color the box
            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add point estimate as diamond marker
            ax.plot(pos, point_estimate, 'd', color='white', markersize=6,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

    # X-axis labels (country codes with percentile info)
    x_labels = [get_country_label(k, representatives) for k in country_keys]
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(x_labels)

    # Y-axis: set ticks at nice percentage values, but plot at log-transformed positions
    # Choose tick values that span the data range nicely
    tick_pcts = [-75, -50, -25, 0, 25, 50, 100, 200]
    tick_positions = [log_transform(p) for p in tick_pcts]
    tick_labels = [f'{p}%' for p in tick_pcts]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Formatting
    ax.set_ylabel('Cumulative Climate Effect')
    ax.set_xlabel('Representative Country (Percentile)')
    ax.set_title('Cumulative Climate Effect on GDP Growth (1961-2022)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Legend for approaches
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS.get(a, 'gray'), alpha=0.7)
        for a in approaches
    ]
    ax.legend(legend_handles, approaches, loc='best', fontsize=8)

    # Add diamond marker explanation to legend
    ax.plot([], [], 'd', color='white', markersize=6,
            markeredgecolor='black', markeredgewidth=1, label='Point estimate')

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_boxplot.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


# Colors for representative countries (min to max)
COUNTRY_COLORS = {
    'min': '#1f77b4',    # Blue
    5: '#ff7f0e',        # Orange
    25: '#2ca02c',       # Green
    50: '#d62728',       # Red
    75: '#9467bd',       # Purple
    95: '#8c564b',       # Brown
    'max': '#e377c2',    # Pink
}


def plot_cumulative_effects_by_method(
    df: pd.DataFrame,
    representatives: dict,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create clustered box-and-whisker plot grouped by method.

    Uses log(1 + pct/100) transform so that -50% and +100% are equidistant from 0.
    Groups by method (5 clusters) with 7 country bars per method.

    Args:
        df: DataFrame with cumulative effects for representative countries
        representatives: Dictionary from select_representative_countries
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use central approaches in consistent order
    approaches = CENTRAL_METHODS
    n_approaches = len(approaches)

    # Get ordered country keys (min, P5, P25, P50, P75, P95, max)
    country_keys = get_country_ordering(representatives)
    n_countries = len(country_keys)

    # Spacing parameters
    cluster_width = 0.85
    box_width = cluster_width / (n_countries + 1)  # Extra space between clusters

    # Filter to year 2022 for final values
    df_2022 = df[df['year'] == 2022].copy()

    # Create box plots - grouped by method
    for i, approach in enumerate(approaches):
        cluster_center = i

        for j, country_key in enumerate(country_keys):
            iso3 = representatives[country_key]['iso3']

            # Get bootstrap samples (iterations 0-999) for this country/approach
            mask = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] >= 0)
            bootstrap_values_pct = df_2022.loc[mask, 'h_T_delta_cum'].values * 100  # Convert to percent

            # Transform to log scale
            bootstrap_values = log_transform(bootstrap_values_pct)

            # Get point estimate (iteration -1)
            mask_point = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] == -1)
            point_estimate_pct = df_2022.loc[mask_point, 'h_T_delta_cum'].values
            point_estimate_pct = point_estimate_pct[0] * 100 if len(point_estimate_pct) > 0 else np.nan
            point_estimate = log_transform(point_estimate_pct)

            # Position for this box
            pos = cluster_center + (j - (n_countries - 1) / 2) * box_width

            # Draw box - color by country
            color = COUNTRY_COLORS.get(country_key, 'gray')
            box = ax.boxplot(
                [bootstrap_values],
                positions=[pos],
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=False,
                whis=[5, 95],  # Whiskers at 5th and 95th percentile
                medianprops=dict(color='black', linewidth=1),
            )

            # Color the box
            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add point estimate as diamond marker
            ax.plot(pos, point_estimate, 'd', color='white', markersize=6,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

    # X-axis labels (methods)
    ax.set_xticks(range(n_approaches))
    ax.set_xticklabels(approaches)

    # Y-axis: set ticks at nice percentage values, but plot at log-transformed positions
    tick_pcts = [-75, -50, -25, 0, 25, 50, 100, 200]
    tick_positions = [log_transform(p) for p in tick_pcts]
    tick_labels = [f'{p}%' for p in tick_pcts]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Formatting
    ax.set_ylabel('Cumulative Climate Effect')
    ax.set_xlabel('Approach')
    ax.set_title('Cumulative Climate Effect on GDP Growth (1961-2022) by Approach')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Legend for countries
    legend_handles = []
    legend_labels = []
    for country_key in country_keys:
        color = COUNTRY_COLORS.get(country_key, 'gray')
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7))
        iso3 = representatives[country_key]['iso3']
        if country_key == 'min':
            legend_labels.append(f'{iso3} (Min)')
        elif country_key == 'max':
            legend_labels.append(f'{iso3} (Max)')
        else:
            legend_labels.append(f'{iso3} (P{country_key})')
    ax.legend(legend_handles, legend_labels, loc='best', fontsize=8)

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_by_method.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


def process_group(group: pd.DataFrame, approach: str, loess_window: int) -> pd.DataFrame:
    """Process a single (iteration, approach, iso3) group to compute cumulative effects.

    Args:
        group: DataFrame for a single group with columns [year, temp, h_T]
        approach: Approach name
        loess_window: Window size for LOESS smoothing

    Returns:
        DataFrame with added columns [h_T_trend_1961, h_T_delta, h_T_delta_cum]
    """
    years = group['year'].values
    h_T = group['h_T'].values

    # Calculate h_T_trend_1961
    h_T_trend_1961 = fit_h_T_trend_1961(years, h_T, approach, loess_window)

    # Calculate h_T_delta
    h_T_delta = h_T - h_T_trend_1961

    # Calculate cumulative
    h_T_delta_cum = calculate_h_T_delta_cumulative(h_T_delta, years)

    result = group.copy()
    result['h_T_trend_1961'] = h_T_trend_1961
    result['h_T_delta'] = h_T_delta
    result['h_T_delta_cum'] = h_T_delta_cum

    return result


def process_all_countries_point_estimate(
    input_path: Path,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS
) -> pd.DataFrame:
    """Process point estimate data for all countries and approaches.

    Args:
        input_path: Path to bootstrap_h_values.csv
        loess_window: Window size for LOESS smoothing

    Returns:
        DataFrame with cumulative effects for all countries (point estimate only)
    """
    print("      Loading point estimate data (iteration=-1, all approaches)...")

    # Read CSV in chunks, filtering to only point estimates
    chunks = []
    for chunk in pd.read_csv(input_path, comment='#', chunksize=100000):
        filtered = chunk[chunk['iteration'] == -1]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Loaded {len(df):,} rows")

    # Process each (approach, iso3) group
    groups = df.groupby(['approach', 'iso3'])
    total_groups = len(groups)

    results = []
    for idx, ((approach, iso3), group) in enumerate(groups):
        if idx % 500 == 0:
            print(f"      Progress: {idx:,}/{total_groups:,} groups...")

        processed = process_group(group, approach, loess_window)
        results.append(processed)

    print(f"      Completed processing {total_groups:,} groups")
    return pd.concat(results, ignore_index=True)


def plot_cumulative_effects_by_approach(
    df: pd.DataFrame,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create multi-panel plot showing cumulative effect distribution across countries.

    Each panel shows one approach with:
    - 90% range (5th-95th percentile) as light shading
    - 50% range (25th-75th percentile) as darker shading
    - Median line

    Uses log(1 + pct/100) transform so that -50% and +100% are equidistant from 0.

    Args:
        df: DataFrame with cumulative effects for all countries (point estimate)
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

    for ax, approach in zip(axes, CENTRAL_METHODS):
        # Filter to this approach
        df_approach = df[df['approach'] == approach]

        # Get unique years
        years = sorted(df_approach['year'].unique())

        # Calculate percentiles across countries for each year
        percentiles_by_year = []
        for year in years:
            values = df_approach[df_approach['year'] == year]['h_T_delta_cum'].values * 100
            percentiles_by_year.append({
                'year': year,
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'min': np.min(values),
                'max': np.max(values),
            })

        df_pct = pd.DataFrame(percentiles_by_year)

        color = METHOD_COLORS.get(approach, 'gray')

        # Transform percentiles to log scale
        p5_log = log_transform(df_pct['p5'])
        p25_log = log_transform(df_pct['p25'])
        p50_log = log_transform(df_pct['p50'])
        p75_log = log_transform(df_pct['p75'])
        p95_log = log_transform(df_pct['p95'])
        min_log = log_transform(df_pct['min'])
        max_log = log_transform(df_pct['max'])

        # Min/max lines (thin)
        ax.plot(df_pct['year'], min_log, color=color, linewidth=0.5, linestyle='-', alpha=0.5, label='Min/Max')
        ax.plot(df_pct['year'], max_log, color=color, linewidth=0.5, linestyle='-', alpha=0.5)

        # 90% range (lighter)
        ax.fill_between(
            df_pct['year'], p5_log, p95_log,
            alpha=0.2, color=color, label='90% range'
        )

        # 50% range (darker)
        ax.fill_between(
            df_pct['year'], p25_log, p75_log,
            alpha=0.4, color=color, label='50% range'
        )

        # Median line
        ax.plot(df_pct['year'], p50_log, color=color, linewidth=2, label='Median')

        # Formatting
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Year')
        ax.set_title(approach)

        if ax == axes[0]:
            ax.set_ylabel('Cumulative Climate Effect')
            ax.legend(loc='lower left', fontsize=7)

    # Set y-axis ticks at nice percentage values (in log-transformed positions)
    tick_pcts = [-75, -50, -25, 0, 25, 50, 100, 200]
    tick_positions = [log_transform(p) for p in tick_pcts]
    tick_labels = [f'{p}%' for p in tick_pcts]
    for ax in axes:
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_by_approach.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


def select_representative_countries_from_file(
    input_path: Path,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS,
    percentiles: tuple = REPRESENTATIVE_PERCENTILES,
    include_min_max: bool = True
) -> dict:
    """Select representative countries using only point estimate data.

    Loads only iteration=-1, approach=method0 to minimize memory usage.

    Args:
        input_path: Path to bootstrap_h_values.csv
        loess_window: Window size for LOESS smoothing
        percentiles: Percentiles for selecting representatives
        include_min_max: Whether to include min and max countries

    Returns:
        Dictionary mapping percentile (or 'min'/'max') -> {'iso3': str, 'value': float, 'target': float}
    """
    print("      Loading point estimate data (iteration=-1, method0)...")

    # Read CSV in chunks, filtering to only needed rows
    chunks = []
    for chunk in pd.read_csv(input_path, comment='#', chunksize=100000):
        filtered = chunk[(chunk['iteration'] == -1) & (chunk['approach'] == 'method0')]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Loaded {len(df):,} rows for country selection")

    # Process each country to get cumulative effects
    results = []
    for iso3, group in df.groupby('iso3'):
        processed = process_group(group, 'method0', loess_window)
        # Get 2022 value
        row_2022 = processed[processed['year'] == 2022]
        if len(row_2022) > 0:
            results.append({
                'iso3': iso3,
                'h_T_delta_cum': row_2022['h_T_delta_cum'].values[0]
            })

    df_results = pd.DataFrame(results)

    # Calculate target percentile values
    values = df_results['h_T_delta_cum'].values
    targets = {p: np.percentile(values, p) for p in percentiles}

    # Find country closest to each percentile
    representatives = {}
    for p, target in targets.items():
        idx = np.argmin(np.abs(df_results['h_T_delta_cum'].values - target))
        row = df_results.iloc[idx]
        representatives[p] = {
            'iso3': row['iso3'],
            'value': row['h_T_delta_cum'],
            'target': target
        }

    # Add min and max countries
    if include_min_max:
        min_idx = np.argmin(values)
        max_idx = np.argmax(values)
        representatives['min'] = {
            'iso3': df_results.iloc[min_idx]['iso3'],
            'value': values[min_idx],
            'target': values[min_idx]
        }
        representatives['max'] = {
            'iso3': df_results.iloc[max_idx]['iso3'],
            'value': values[max_idx],
            'target': values[max_idx]
        }

    return representatives


def process_representative_countries(
    input_path: Path,
    representative_iso3s: list,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS
) -> pd.DataFrame:
    """Process bootstrap data for representative countries only.

    Args:
        input_path: Path to bootstrap_h_values.csv
        representative_iso3s: List of ISO3 country codes to process
        loess_window: Window size for LOESS smoothing

    Returns:
        DataFrame with cumulative effects for representative countries
    """
    print(f"      Loading data for {len(representative_iso3s)} representative countries...")

    # Convert to set for faster lookup
    iso3_set = set(representative_iso3s)

    # Read CSV in chunks, filtering to only needed countries
    chunks = []
    for chunk in pd.read_csv(input_path, comment='#', chunksize=100000):
        filtered = chunk[chunk['iso3'].isin(iso3_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Loaded {len(df):,} rows for representative countries")

    # Process each (iteration, approach, iso3) group
    groups = df.groupby(['iteration', 'approach', 'iso3'])
    total_groups = len(groups)

    results = []
    for idx, ((iteration, approach, iso3), group) in enumerate(groups):
        if idx % 1000 == 0:
            print(f"      Progress: {idx:,}/{total_groups:,} groups...")

        processed = process_group(group, approach, loess_window)
        results.append(processed)

    print(f"      Completed processing {total_groups:,} groups")
    return pd.concat(results, ignore_index=True)


def main():
    # Find most recent bootstrap directory for default
    default_bootstrap_dir = find_most_recent_dir("data/output/reference/bootstrap_*")

    parser = argparse.ArgumentParser(
        description="Calculate cumulative climate effects from bootstrap h(T) values"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=default_bootstrap_dir,
        help="Directory containing bootstrap_h_values.csv (default: most recent in data/output/reference/bootstrap_*)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped in data/output)",
    )
    parser.add_argument(
        "--loess-window",
        type=int,
        default=DEFAULT_LOESS_WINDOW_YEARS,
        help=f"Window size in years for LOESS smoothing (default: {DEFAULT_LOESS_WINDOW_YEARS})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Calculate Cumulative Climate Effects")
    print("=" * 70)

    # Validate input directory
    if args.input_dir is None:
        print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
        print("       Please specify --input-dir explicitly")
        sys.exit(1)

    # Input file
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    input_path = input_dir / "bootstrap_h_values.csv"
    if not input_path.exists():
        print(f"ERROR: bootstrap_h_values.csv not found in {input_dir}")
        sys.exit(1)

    print(f"      Input dir: {input_dir}")
    input_file = str(input_path)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="cumulative_")

    # Phase 1: Process all countries with point estimate
    print("\n[1/7] Processing all countries (point estimate only)...")
    df_all_countries = process_all_countries_point_estimate(input_path, args.loess_window)

    # Save all countries cumulative effects
    all_countries_path = output_dir / 'cumulative_effects_all_countries.csv'
    with open(all_countries_path, 'w') as f:
        f.write(f"# Input data: {Path(input_file).name}\n")
        f.write("# Point estimate cumulative effects for all countries\n")
    df_all_countries.to_csv(all_countries_path, mode='a', index=False)
    print(f"      Saved: {all_countries_path} ({len(df_all_countries):,} rows)")

    # Phase 2: Create multi-panel visualization by approach
    print("\n[2/7] Creating cumulative effects by approach visualization...")
    plot_cumulative_effects_by_approach(df_all_countries, output_dir, input_file)

    # Phase 3: Select representative countries using point estimate only
    print("\n[3/7] Selecting representative countries (using point estimate)...")
    representatives = select_representative_countries_from_file(
        input_path, args.loess_window
    )

    # Print selected countries
    print("      Selected countries:")
    country_order = get_country_ordering(representatives)
    for key in country_order:
        info = representatives[key]
        if key == 'min':
            label = "Min"
        elif key == 'max':
            label = "Max"
        else:
            label = f"P{key}"
        print(f"        {label}: {info['iso3']} (value={info['value']:.4f}, target={info['target']:.4f})")

    # Save representative countries info
    rep_rows = []
    for key in country_order:
        info = representatives[key]
        rep_rows.append({
            'percentile': key,
            'iso3': info['iso3'],
            'h_T_delta_cum_2022': info['value'],
            'target_percentile_value': info['target']
        })
    rep_df = pd.DataFrame(rep_rows)
    rep_path = output_dir / 'representative_countries.csv'
    rep_df.to_csv(rep_path, index=False)
    print(f"      Saved: {rep_path}")

    # Phase 4: Process full bootstrap data for representative countries only
    print("\n[4/7] Processing bootstrap data for representative countries...")
    rep_iso3s = [representatives[p]['iso3'] for p in representatives]
    df_summary = process_representative_countries(
        input_path, rep_iso3s, args.loess_window
    )

    # Save cumulative effects summary
    print("\n[5/7] Saving cumulative effects summary...")

    # Add header comment to CSV
    summary_path = output_dir / 'cumulative_h_values_summary.csv'
    with open(summary_path, 'w') as f:
        f.write(f"# Input data: {Path(input_file).name}\n")
    df_summary.to_csv(summary_path, mode='a', index=False)
    print(f"      Saved: {summary_path} ({len(df_summary):,} rows)")

    # Create visualizations
    print("\n[6/7] Creating box plot visualization (grouped by country)...")
    plot_cumulative_effects_boxplot(df_summary, representatives, output_dir, input_file)

    print("\n[7/7] Creating box plot visualization (grouped by method)...")
    plot_cumulative_effects_by_method(df_summary, representatives, output_dir, input_file)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
