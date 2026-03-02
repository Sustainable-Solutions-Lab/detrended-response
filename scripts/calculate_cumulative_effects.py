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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detrending import DEFAULT_LOESS_WINDOW_YEARS
from src.output import APPROACH_COLORS, create_output_dir, add_input_file_annotation


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


def load_run_metadata(directory: Path) -> dict:
    """Load run metadata from a directory.

    Looks for run_metadata.json in the directory.

    Parameters
    ----------
    directory : Path
        Directory to search for metadata

    Returns
    -------
    dict
        Metadata dictionary, or empty dict if not found
    """
    import json
    metadata_path = directory / 'run_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


# ==============================================================================
# Constants
# ==============================================================================

# Central approaches for analysis (in display order)
CENTRAL_APPROACHES_POINT = ['Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL', 'Approach DL']
# All 9 approaches grouped by method (J, P, L) for boxplot figures
CENTRAL_APPROACHES_BOXPLOT = [
    'Approach QJ', 'Approach PJ', 'Approach DJ',
    'Approach QP', 'Approach PP', 'Approach DP',
    'Approach QL', 'Approach PL', 'Approach DL',
]

# Base year for cumulative effect calculation
BASE_YEAR = 1961

# Percentiles for selecting representative countries
REPRESENTATIVE_PERCENTILES = (5, 25, 50, 75, 95)


# ==============================================================================
# Core Functions
# ==============================================================================

def calculate_h_T_delta_cumulative(h_T_delta: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Calculate cumulative effect as simple sum of annual differences.

    Since h_T represents changes in log GDP, the cumulative effect is simply:
        h_int(t) = Σ_{τ=1961}^{t} (h(T(τ)) - h(T(1961)))

    This is the sum of annual climate effects relative to the baseline year.

    Note: For Approach DL, the h_T values in bootstrap_h_values.csv already incorporate
    persistence decay via h_conv(T), so no additional decay is applied here.

    Args:
        h_T_delta: Array of h_T - h_T(1961) values
        years: Array of year values

    Returns:
        Array of cumulative effects (sum of h_T_delta from first year to each year)
    """
    # Sort by year to ensure proper ordering
    sort_idx = np.argsort(years)
    h_T_delta_sorted = h_T_delta[sort_idx]

    # Calculate cumulative sum of annual effects
    h_T_delta_cum = np.cumsum(h_T_delta_sorted)

    # Restore original order
    result = np.zeros(len(h_T_delta))
    result[sort_idx] = h_T_delta_cum

    return result


def log_transform(pct):
    """Transform percentage to log scale: log(1 + pct/100).

    Used for y-axis tick positioning: converts percent values to log positions.
    Example: log_transform(100) ≈ 0.693, log_transform(-50) ≈ -0.693
    """
    return np.log(1 + pct / 100)


def inv_log_transform(y):
    """Inverse transform: from log scale back to percentage.

    Converts cumulative log changes to percent: (exp(y) - 1) * 100
    Example: inv_log_transform(-0.5) ≈ -39.3%, inv_log_transform(0.5) ≈ 64.9%
    """
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

    Labels clarify that percentiles are of the cumulative effect distribution:
    - Low percentiles (P5) = most negative effect = most hurt by climate
    - High percentiles (P95) = most positive effect = most helped by climate

    Args:
        key: Either 'min', 'max', or an integer percentile
        representatives: Dictionary with country info

    Returns:
        Formatted label string
    """
    iso3 = representatives[key]['iso3']
    if key == 'min':
        return f"{iso3}\n(min)"
    elif key == 'max':
        return f"{iso3}\n(max)"
    elif key == 5:
        return f"{iso3}\n(P5)"
    elif key == 95:
        return f"{iso3}\n(P95)"
    else:
        return f"{iso3}\n(P{key})"


def plot_cumulative_effects_boxplot(
    df: pd.DataFrame,
    representatives: dict,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create clustered box-and-whisker plot of cumulative effects.

    Plots h_T_delta_cum values directly (already in log space as sum of log changes).
    Y-axis labels show equivalent percent changes, with symmetric scaling so that
    -50% and +100% are equidistant from 0.
    Groups by country (min, P5, P25, P50, P75, P95, max) with approach bars per country.

    Args:
        df: DataFrame with cumulative effects for representative countries
        representatives: Dictionary from select_representative_countries
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Use central approaches in consistent order
    approaches = CENTRAL_APPROACHES_BOXPLOT
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
            # h_T_delta_cum is already in log space (sum of log changes), use directly
            bootstrap_values = df_2022.loc[mask, 'h_T_delta_cum'].values

            # Get point estimate (iteration -1)
            mask_point = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] == -1)
            point_estimate_arr = df_2022.loc[mask_point, 'h_T_delta_cum'].values
            point_estimate = point_estimate_arr[0] if len(point_estimate_arr) > 0 else np.nan

            # Position for this box
            pos = cluster_center + (j - (n_approaches - 1) / 2) * box_width

            # Draw box (only if we have valid bootstrap data)
            color = APPROACH_COLORS.get(approach, 'gray')
            if len(bootstrap_values) > 0:
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

            # Add point estimate as diamond marker (only if we have a valid estimate)
            if not np.isnan(point_estimate):
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
        plt.Rectangle((0, 0), 1, 1, facecolor=APPROACH_COLORS.get(a, 'gray'), alpha=0.7)
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


def plot_cumulative_effects_by_approach_grouped(
    df: pd.DataFrame,
    representatives: dict,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create clustered box-and-whisker plot grouped by approach.

    Plots h_T_delta_cum values directly (already in log space as sum of log changes).
    Y-axis labels show equivalent percent changes, with symmetric scaling so that
    -50% and +100% are equidistant from 0.
    Groups by approach with country bars per approach cluster.

    Args:
        df: DataFrame with cumulative effects for representative countries
        representatives: Dictionary from select_representative_countries
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Use central approaches in consistent order
    approaches = CENTRAL_APPROACHES_BOXPLOT
    n_approaches = len(approaches)

    # Get ordered country keys (min, P5, P25, P50, P75, P95, max)
    country_keys = get_country_ordering(representatives)
    n_countries = len(country_keys)

    # Spacing parameters
    cluster_width = 0.85
    box_width = cluster_width / (n_countries + 1)  # Extra space between clusters
    group_gap = 0.6  # Extra space between method groups (J, P, L)

    # Filter to year 2022 for final values
    df_2022 = df[df['year'] == 2022].copy()

    # Compute x-positions with gaps between groups
    # Groups: [0,1,2] = J methods, [3,4,5] = P methods, [6,7,8] = L methods
    x_positions = []
    for i in range(n_approaches):
        group_idx = i // 3  # Which group (0=J, 1=P, 2=L)
        pos_in_group = i % 3  # Position within group
        x_pos = group_idx * (3 + group_gap) + pos_in_group
        x_positions.append(x_pos)

    # Create box plots - grouped by approach
    for i, approach in enumerate(approaches):
        cluster_center = x_positions[i]

        for j, country_key in enumerate(country_keys):
            iso3 = representatives[country_key]['iso3']

            # Get bootstrap samples (iterations 0-999) for this country/approach
            mask = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] >= 0)
            # h_T_delta_cum is already in log space (sum of log changes), use directly
            bootstrap_values = df_2022.loc[mask, 'h_T_delta_cum'].values

            # Get point estimate (iteration -1)
            mask_point = (df_2022['iso3'] == iso3) & (df_2022['approach'] == approach) & (df_2022['iteration'] == -1)
            point_estimate_arr = df_2022.loc[mask_point, 'h_T_delta_cum'].values
            point_estimate = point_estimate_arr[0] if len(point_estimate_arr) > 0 else np.nan

            # Position for this box
            pos = cluster_center + (j - (n_countries - 1) / 2) * box_width

            # Draw box - color by country (only if we have valid bootstrap data)
            color = COUNTRY_COLORS.get(country_key, 'gray')
            if len(bootstrap_values) > 0:
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

            # Add point estimate as diamond marker (only if we have a valid estimate)
            if not np.isnan(point_estimate):
                ax.plot(pos, point_estimate, 'd', color='white', markersize=6,
                        markeredgecolor='black', markeredgewidth=1, zorder=10)

    # X-axis labels (approaches)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(approaches)

    # Add vertical separator lines between method groups
    # Lines go between positions 2 and 3, and between 5 and 6
    y_min, y_max = ax.get_ylim()
    sep1_x = (x_positions[2] + x_positions[3]) / 2
    sep2_x = (x_positions[5] + x_positions[6]) / 2
    ax.axvline(x=sep1_x, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=sep2_x, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add group labels at top
    group_centers = [
        (x_positions[0] + x_positions[2]) / 2,  # J group center
        (x_positions[3] + x_positions[5]) / 2,  # P group center
        (x_positions[6] + x_positions[8]) / 2,  # L group center
    ]
    group_labels = ['Joint (J)', 'Polynomial (P)', 'LOESS (L)']

    # Y-axis: set ticks at nice percentage values, but plot at log-transformed positions
    tick_pcts = [-75, -50, -25, 0, 25, 50, 100, 200]
    tick_positions = [log_transform(p) for p in tick_pcts]
    tick_labels = [f'{p}%' for p in tick_pcts]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Formatting
    ax.set_ylabel('Cumulative Climate Effect')
    ax.set_title('Cumulative Climate Effect on GDP Growth (1961-2022) by Approach', pad=25)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Add group labels above the plot
    for center, label in zip(group_centers, group_labels):
        ax.text(center, ax.get_ylim()[1] * 1.02, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Legend for countries
    # Labels clarify that percentiles are of the cumulative effect distribution:
    # - Low percentiles (P5) = most negative effect = most hurt by climate
    # - High percentiles (P95) = most positive effect = most helped by climate
    legend_handles = []
    legend_labels = []
    for country_key in country_keys:
        color = COUNTRY_COLORS.get(country_key, 'gray')
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7))
        iso3 = representatives[country_key]['iso3']
        if country_key == 'min':
            legend_labels.append(f'{iso3} (min)')
        elif country_key == 'max':
            legend_labels.append(f'{iso3} (max)')
        elif country_key == 5:
            legend_labels.append(f'{iso3} (P5)')
        elif country_key == 95:
            legend_labels.append(f'{iso3} (P95)')
        else:
            legend_labels.append(f'{iso3} (P{country_key})')
    ax.legend(legend_handles, legend_labels, loc='best', fontsize=8)

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_by_approach_grouped.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


def process_group(group: pd.DataFrame, approach: str, loess_window: int, h_T_baseline: float = None) -> pd.DataFrame:
    """Process a single (iteration, approach, iso3) group to compute cumulative effects.

    Args:
        group: DataFrame for a single group with columns [year, temp, h_T]
        approach: Approach name (not used in computation, kept for API compatibility)
        loess_window: Window size (unused, kept for API compatibility)
        h_T_baseline: h(T_loess_1961) baseline value. If provided, uses this instead of
                      the actual h(T) at 1961. This provides a stable baseline that
                      isn't affected by inter-annual temperature variability.

    Returns:
        DataFrame with added columns [h_T_trend_1961, h_T_delta, h_T_delta_cum]
    """
    years = group['year'].values
    h_T = group['h_T'].values

    # Sort by year to ensure proper ordering
    sort_idx = np.argsort(years)
    years_sorted = years[sort_idx]
    h_T_sorted = h_T[sort_idx]

    # Use LOESS baseline if provided, otherwise fall back to actual 1961 value
    if h_T_baseline is not None:
        h_T_1961 = h_T_baseline
    else:
        # Find 1961 index and baseline value
        year_1961_mask = years_sorted == BASE_YEAR
        if year_1961_mask.any():
            idx_1961 = np.where(year_1961_mask)[0][0]
            h_T_1961 = h_T_sorted[idx_1961]
        else:
            idx_1961 = 0
            h_T_1961 = h_T_sorted[0]

    # All approaches use the same logic: subtract 1961 baseline, then sum
    #
    # For approaches Approach QJ, Approach QP, Approach QL, Approach PL: h_T = h(T), so h_T_delta = h(T) - h(T_baseline)
    #
    # For Approach DL: h_T = h_conv which already incorporates persistence decay.
    # We use the same formula: h_T_delta = h_conv - h_conv(baseline)
    # When h4 > 0, h_conv values are smaller due to built-in decay, so cumulative
    # effects will be smaller/bounded compared to Approach QL.
    # When h4 ≈ 0, h_conv ≈ h(T), so results match Approach QL.
    #
    # This unified approach avoids the "double-decay" bug where we incorrectly
    # applied additional decay to the baseline.
    h_T_delta = h_T - h_T_1961
    h_T_delta_cum = calculate_h_T_delta_cumulative(h_T_delta, years)

    result = group.copy()
    result['h_T_trend_1961'] = h_T_1961  # Keep column name for compatibility
    result['h_T_delta'] = h_T_delta
    result['h_T_delta_cum'] = h_T_delta_cum

    return result


def load_baselines(bootstrap_dir: Path, iteration: int = None) -> pd.DataFrame:
    """Load baseline values from bootstrap_h_baselines.csv.

    Args:
        bootstrap_dir: Directory containing bootstrap_h_baselines.csv
        iteration: If specified, filter to only this iteration (-1 for point estimate)

    Returns:
        DataFrame with columns [iteration, approach, iso3, T_loess_base, h_T_baseline]
    """
    baselines_path = bootstrap_dir / 'bootstrap_h_baselines.csv'
    if not baselines_path.exists():
        print(f"      Warning: {baselines_path} not found, using actual 1961 values")
        return None

    df = pd.read_csv(baselines_path, comment='#')
    if iteration is not None:
        df = df[df['iteration'] == iteration]
    return df


def process_all_countries_point_estimate(
    input_path: Path,
    bootstrap_dir: Path,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS
) -> pd.DataFrame:
    """Process point estimate data for all countries and approaches.

    Uses point estimates (iteration=-1) for all approaches.

    Args:
        input_path: Path to bootstrap_h_values.csv
        bootstrap_dir: Directory containing bootstrap_coefficients.csv
        loess_window: Window size for LOESS smoothing

    Returns:
        DataFrame with cumulative effects for all countries
    """
    print("      Loading point estimate data (iteration=-1, all approaches)...")

    # Read CSV in chunks, filtering to only point estimates
    chunks = []
    for chunk in pd.read_csv(input_path, comment='#', chunksize=100000):
        filtered = chunk[chunk['iteration'] == -1]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Loaded {len(df):,} rows for point estimates")

    # Load baselines (point estimates only)
    baselines = load_baselines(bootstrap_dir, iteration=-1)
    if baselines is not None:
        # Create lookup dict for baselines: (approach, iso3) -> h_T_baseline
        baselines_dict = {
            (row['approach'], row['iso3']): row['h_T_baseline']
            for _, row in baselines.iterrows()
        }
        print(f"      Loaded {len(baselines_dict):,} baseline values")
    else:
        baselines_dict = {}

    # Process each (approach, iso3) group for all approaches
    groups = df.groupby(['approach', 'iso3'])
    total_groups = len(groups)

    results = []
    for idx, ((approach, iso3), group) in enumerate(groups):
        if idx % 500 == 0:
            print(f"      Progress: {idx:,}/{total_groups:,} groups...")

        # Get baseline value if available
        h_T_baseline = baselines_dict.get((approach, iso3))
        processed = process_group(group, approach, loess_window, h_T_baseline=h_T_baseline)
        results.append(processed)

    print(f"      Completed processing {total_groups:,} groups")

    return pd.concat(results, ignore_index=True)


def plot_cumulative_effects_by_approach(
    df: pd.DataFrame,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create 3x3 multi-panel plot showing cumulative effect distribution across countries.

    Layout:
        Rows = response functions (1=quadratic, 2=piecewise, 3=persistence)
        Columns = methods (J=Joint, P=Polynomial, L=LOESS)

        Row 0: Approach QJ, Approach QP, Approach QL
        Row 1: Approach PJ, Approach PP, Approach PL
        Row 2: Approach DJ, Approach DP, Approach DL

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
    # 3x3 grid: rows = response functions (1,2,3), columns = methods (J,P,L)
    approach_order = [
        ['Approach QJ', 'Approach QP', 'Approach QL'],
        ['Approach PJ', 'Approach PP', 'Approach PL'],
        ['Approach DJ', 'Approach DP', 'Approach DL'],
    ]

    # Check for missing approaches
    available_approaches = set(df['approach'].unique())
    all_approaches = [a for row in approach_order for a in row]
    missing_approaches = [a for a in all_approaches if a not in available_approaches]
    if missing_approaches:
        raise ValueError(
            f"Missing approaches in data: {missing_approaches}. "
            f"Available: {sorted(available_approaches)}. "
            f"You may need to re-run run_bootstrap.py to generate data for all approaches."
        )

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True, sharex=True)

    # Row and column labels
    row_labels = ['Quadratic (Q)', 'Piecewise (P)', 'Decay (D)']
    col_labels = ['Joint (J)', 'Polynomial (P)', 'LOESS (L)']

    for row_idx, row_approaches in enumerate(approach_order):
        for col_idx, approach in enumerate(row_approaches):
            ax = axes[row_idx, col_idx]

            # Filter to this approach
            df_approach = df[df['approach'] == approach]

            # Get unique years
            years = sorted(df_approach['year'].unique())

            # Calculate percentiles across countries for each year
            # h_T_delta_cum is already in log space (sum of log changes), use directly
            percentiles_by_year = []
            for year in years:
                values = df_approach[df_approach['year'] == year]['h_T_delta_cum'].values
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

            color = APPROACH_COLORS.get(approach, 'gray')

            # Values are already in log space, use directly
            p5 = df_pct['p5']
            p25 = df_pct['p25']
            p50 = df_pct['p50']
            p75 = df_pct['p75']
            p95 = df_pct['p95']
            min_val = df_pct['min']
            max_val = df_pct['max']

            # Min/max lines (thin)
            ax.plot(df_pct['year'], min_val, color=color, linewidth=0.5, linestyle='-', alpha=0.5, label='Min/Max')
            ax.plot(df_pct['year'], max_val, color=color, linewidth=0.5, linestyle='-', alpha=0.5)

            # 90% range (lighter)
            ax.fill_between(
                df_pct['year'], p5, p95,
                alpha=0.2, color=color, label='90% range'
            )

            # 50% range (darker)
            ax.fill_between(
                df_pct['year'], p25, p75,
                alpha=0.4, color=color, label='50% range'
            )

            # Median line
            ax.plot(df_pct['year'], p50, color=color, linewidth=2, label='Median')

            # Formatting
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_title(approach, fontsize=10)

            # Column headers (top row only)
            if row_idx == 0:
                ax.text(0.5, 1.15, col_labels[col_idx], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(f'{row_labels[row_idx]}\nCumulative Effect', fontsize=9)

            # X-axis label (bottom row only)
            if row_idx == 2:
                ax.set_xlabel('Year')

            # Legend (top-left panel only)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower left', fontsize=7)

    # Set y-axis limits to -66.67% to 150% (symmetric in log space)
    y_min = log_transform(-200/3)  # -66.67%
    y_max = log_transform(150)
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)

    # Set y-axis ticks at nice percentage values (in log-transformed positions)
    tick_pcts = [-50, -25, 0, 25, 50, 100]
    tick_positions = [log_transform(p) for p in tick_pcts]
    tick_labels = [f'{p}%' for p in tick_pcts]
    for ax in axes.flat:
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_by_year.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


def plot_cumulative_effects_ApproachDL(
    df_all_countries: pd.DataFrame,
    df_representative: pd.DataFrame,
    representatives: dict,
    output_dir: Path,
    input_file: str = None
) -> None:
    """Create 2x3 panel plot for persistence approaches (Approach DJ, Approach DP, Approach DL).

    Layout:
        Top row: Cumulative effects by year (all countries) for each approach
        Bottom row: Box-and-whisker plot for representative countries (2022) for each approach
        Columns: J, P, L methods

    Args:
        df_all_countries: DataFrame with cumulative effects for all countries (point estimate)
        df_representative: DataFrame with cumulative effects for representative countries (bootstrap)
        representatives: Dictionary mapping percentile -> country info
        output_dir: Directory to save plot
        input_file: Input file for annotation
    """
    approaches = ['Approach DJ', 'Approach DP', 'Approach DL']
    col_labels = ['Joint (J)', 'Polynomial (P)', 'LOESS (L)']

    # Check for missing approaches
    available = set(df_all_countries['approach'].unique())
    missing = [a for a in approaches if a not in available]
    if missing:
        print(f"      Warning: Missing approaches {missing}, skipping persistence plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Get ordered country keys for bottom row
    country_keys = get_country_ordering(representatives)
    n_countries = len(country_keys)

    for col_idx, approach in enumerate(approaches):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        # Filter to this approach
        df_approach = df_all_countries[df_all_countries['approach'] == approach]
        color = APPROACH_COLORS.get(approach, 'gray')

        # ==================== TOP ROW: Cumulative effects by year ====================
        years = sorted(df_approach['year'].unique())

        # Calculate percentiles across countries for each year
        percentiles_by_year = []
        for year in years:
            values = df_approach[df_approach['year'] == year]['h_T_delta_cum'].values
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

        # Values are already in log space
        p5 = df_pct['p5']
        p25 = df_pct['p25']
        p50 = df_pct['p50']
        p75 = df_pct['p75']
        p95 = df_pct['p95']
        min_val = df_pct['min']
        max_val = df_pct['max']

        # Min/max lines (thin)
        ax_top.plot(df_pct['year'], min_val, color=color, linewidth=0.5, linestyle='-', alpha=0.5, label='Min/Max')
        ax_top.plot(df_pct['year'], max_val, color=color, linewidth=0.5, linestyle='-', alpha=0.5)

        # 90% range (lighter)
        ax_top.fill_between(
            df_pct['year'], p5, p95,
            alpha=0.2, color=color, label='90% range'
        )

        # 50% range (darker)
        ax_top.fill_between(
            df_pct['year'], p25, p75,
            alpha=0.4, color=color, label='50% range'
        )

        # Median line
        ax_top.plot(df_pct['year'], p50, color=color, linewidth=2, label='Median')

        # Formatting
        ax_top.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax_top.set_title(f'{approach}\n{col_labels[col_idx]}', fontsize=10)

        if col_idx == 0:
            ax_top.set_ylabel('Cumulative Effect')
            ax_top.legend(loc='upper left', fontsize=7)

        # ==================== BOTTOM ROW: Box-and-whisker for representative countries ====================
        # Filter to year 2022 and this approach
        df_2022 = df_representative[(df_representative['year'] == 2022) & (df_representative['approach'] == approach)].copy()

        box_width = 0.7

        for j, country_key in enumerate(country_keys):
            iso3 = representatives[country_key]['iso3']

            # Get bootstrap samples (iterations 0-999) for this country
            mask = (df_2022['iso3'] == iso3) & (df_2022['iteration'] >= 0)
            bootstrap_values = df_2022.loc[mask, 'h_T_delta_cum'].values

            # Get point estimate (iteration -1)
            mask_point = (df_2022['iso3'] == iso3) & (df_2022['iteration'] == -1)
            point_estimate_arr = df_2022.loc[mask_point, 'h_T_delta_cum'].values
            point_estimate = point_estimate_arr[0] if len(point_estimate_arr) > 0 else np.nan

            # Draw box - color by country
            box_color = COUNTRY_COLORS.get(country_key, 'gray')
            if len(bootstrap_values) > 0:
                box = ax_bottom.boxplot(
                    [bootstrap_values],
                    positions=[j],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    whis=[5, 95],
                    medianprops=dict(color='black', linewidth=1),
                )

                for patch in box['boxes']:
                    patch.set_facecolor(box_color)
                    patch.set_alpha(0.7)

            # Add point estimate as diamond marker
            if not np.isnan(point_estimate):
                ax_bottom.plot(j, point_estimate, 'd', color='white', markersize=6,
                              markeredgecolor='black', markeredgewidth=1, zorder=10)

        # X-axis labels (countries)
        x_labels = []
        for country_key in country_keys:
            iso3 = representatives[country_key]['iso3']
            if country_key == 'min':
                x_labels.append(f'{iso3}\n(min)')
            elif country_key == 'max':
                x_labels.append(f'{iso3}\n(max)')
            elif country_key == 5:
                x_labels.append(f'{iso3}\n(P5)')
            elif country_key == 95:
                x_labels.append(f'{iso3}\n(P95)')
            else:
                x_labels.append(f'{iso3}\n(P{country_key})')

        ax_bottom.set_xticks(range(n_countries))
        ax_bottom.set_xticklabels(x_labels, fontsize=7)
        ax_bottom.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        if col_idx == 0:
            ax_bottom.set_ylabel('Cumulative Effect (2022)')

    # ==================== Y-axis scaling ====================
    # Top row: -3% to +6%, ticks every 1%
    top_tick_pcts = list(range(-3, 7, 1))
    top_tick_positions = [log_transform(p) for p in top_tick_pcts]
    top_tick_labels = [f'{p}%' for p in top_tick_pcts]
    for ax in axes[0, :]:
        ax.set_ylim(log_transform(-3), log_transform(6))
        ax.set_yticks(top_tick_positions)
        ax.set_yticklabels(top_tick_labels)

    # Bottom row: -6% to +12%, ticks every 2%
    bottom_tick_pcts = list(range(-6, 13, 2))
    bottom_tick_positions = [log_transform(p) for p in bottom_tick_pcts]
    bottom_tick_labels = [f'{p}%' for p in bottom_tick_pcts]
    for ax in axes[1, :]:
        ax.set_ylim(log_transform(-6), log_transform(12))
        ax.set_yticks(bottom_tick_positions)
        ax.set_yticklabels(bottom_tick_labels)

    plt.tight_layout()

    # Add input file annotation
    add_input_file_annotation(fig, input_file)

    # Save
    output_path = output_dir / 'cumulative_effects_Approach3_2x3.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"      Saved: {output_path}")


def select_representative_countries_from_file(
    input_path: Path,
    bootstrap_dir: Path,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS,
    percentiles: tuple = REPRESENTATIVE_PERCENTILES,
    include_min_max: bool = True
) -> dict:
    """Select representative countries using only point estimate data.

    Loads only iteration=-1, approach=Approach QJ to minimize memory usage.

    Args:
        input_path: Path to bootstrap_h_values.csv
        bootstrap_dir: Directory containing bootstrap_h_baselines.csv
        loess_window: Window size for LOESS smoothing
        percentiles: Percentiles for selecting representatives
        include_min_max: Whether to include min and max countries

    Returns:
        Dictionary mapping percentile (or 'min'/'max') -> {'iso3': str, 'value': float, 'target': float}
    """
    print("      Loading point estimate data (iteration=-1, Approach QJ)...")

    # Read CSV in chunks, filtering to only needed rows
    chunks = []
    for chunk in pd.read_csv(input_path, comment='#', chunksize=100000):
        filtered = chunk[(chunk['iteration'] == -1) & (chunk['approach'] == 'Approach QJ')]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Loaded {len(df):,} rows for country selection")

    # Load baselines (point estimates only, Approach QJ)
    baselines = load_baselines(bootstrap_dir, iteration=-1)
    if baselines is not None:
        baselines_ApproachQJ = baselines[baselines['approach'] == 'Approach QJ']
        baselines_dict = {
            row['iso3']: row['h_T_baseline']
            for _, row in baselines_ApproachQJ.iterrows()
        }
        print(f"      Loaded {len(baselines_dict):,} baseline values for Approach QJ")
    else:
        baselines_dict = {}

    # Process each country to get cumulative effects
    results = []
    for iso3, group in df.groupby('iso3'):
        h_T_baseline = baselines_dict.get(iso3)
        processed = process_group(group, 'Approach QJ', loess_window, h_T_baseline=h_T_baseline)
        # Get 2022 value
        row_2022 = processed[processed['year'] == 2022]
        if len(row_2022) > 0:
            results.append({
                'iso3': iso3,
                'h_T_delta_cum': row_2022['h_T_delta_cum'].values[0]
            })

    df_results = pd.DataFrame(results)

    # Diagnostic: print distribution statistics
    values = df_results['h_T_delta_cum'].values
    print(f"      Cumulative effect distribution (2022):")
    print(f"        Min: {np.min(values):.4f} ({inv_log_transform(np.min(values)):.1f}%)")
    print(f"        P5:  {np.percentile(values, 5):.4f} ({inv_log_transform(np.percentile(values, 5)):.1f}%)")
    print(f"        P25: {np.percentile(values, 25):.4f} ({inv_log_transform(np.percentile(values, 25)):.1f}%)")
    print(f"        P50: {np.percentile(values, 50):.4f} ({inv_log_transform(np.percentile(values, 50)):.1f}%)")
    print(f"        P75: {np.percentile(values, 75):.4f} ({inv_log_transform(np.percentile(values, 75)):.1f}%)")
    print(f"        P95: {np.percentile(values, 95):.4f} ({inv_log_transform(np.percentile(values, 95)):.1f}%)")
    print(f"        Max: {np.max(values):.4f} ({inv_log_transform(np.max(values)):.1f}%)")

    # Diagnostic: check specific countries
    for test_iso in ['FIN', 'SDN', 'USA', 'BRA']:
        row = df_results[df_results['iso3'] == test_iso]
        if len(row) > 0:
            val = row['h_T_delta_cum'].values[0]
            pct_rank = (values < val).sum() / len(values) * 100
            print(f"        {test_iso}: {val:.4f} ({inv_log_transform(val):.1f}%) - percentile rank: {pct_rank:.1f}")

    # Calculate target percentile values
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
    bootstrap_dir: Path,
    loess_window: int = DEFAULT_LOESS_WINDOW_YEARS
) -> pd.DataFrame:
    """Process bootstrap data for representative countries only.

    Args:
        input_path: Path to bootstrap_h_values.csv
        representative_iso3s: List of ISO3 country codes to process
        bootstrap_dir: Directory containing bootstrap_h_baselines.csv
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

    # Load all baselines for representative countries
    baselines = load_baselines(bootstrap_dir)
    if baselines is not None:
        # Filter to representative countries
        baselines = baselines[baselines['iso3'].isin(iso3_set)]
        # Create lookup dict: (iteration, approach, iso3) -> h_T_baseline
        baselines_dict = {
            (row['iteration'], row['approach'], row['iso3']): row['h_T_baseline']
            for _, row in baselines.iterrows()
        }
        print(f"      Loaded {len(baselines_dict):,} baseline values for representative countries")
    else:
        baselines_dict = {}

    # Process each (iteration, approach, iso3) group
    groups = df.groupby(['iteration', 'approach', 'iso3'])
    total_groups = len(groups)

    results = []
    for idx, ((iteration, approach, iso3), group) in enumerate(groups):
        if idx % 1000 == 0:
            print(f"      Progress: {idx:,}/{total_groups:,} groups...")

        # Get baseline value if available
        h_T_baseline = baselines_dict.get((iteration, approach, iso3))
        processed = process_group(group, approach, loess_window, h_T_baseline=h_T_baseline)
        results.append(processed)

    print(f"      Completed processing {total_groups:,} groups")
    return pd.concat(results, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calculate cumulative climate effects from bootstrap h(T) values"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Parent directory containing bootstrap_* subdirectory (e.g., data/output/reference_mw10)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing bootstrap_h_values.csv (default: most recent in reference-dir/bootstrap_* or data/output/reference/bootstrap_*)",
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
    parser.add_argument(
        "--mean-weight-distance",
        type=float,
        default=None,
        help="Mean weighting distance in years for LOESS. Window = 44/7 * this value. "
             "Overrides --loess-window if specified.",
    )

    args = parser.parse_args(argv)

    # Resolve input directory: explicit --input-dir > --reference-dir > default
    if args.input_dir is not None:
        input_dir = args.input_dir
    elif args.reference_dir is not None:
        input_dir = find_most_recent_dir(f"{args.reference_dir}/bootstrap_*")
    else:
        input_dir = find_most_recent_dir("data/output/reference/bootstrap_*")

    print("=" * 70)
    print("Calculate Cumulative Climate Effects")
    print("=" * 70)

    # Validate input directory
    if input_dir is None:
        if args.reference_dir:
            print(f"ERROR: No bootstrap directory found matching {args.reference_dir}/bootstrap_*")
        else:
            print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
        print("       Please specify --input-dir or --reference-dir explicitly")
        sys.exit(1)

    # Input file
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    input_path = input_dir / "bootstrap_h_values.csv"
    if not input_path.exists():
        print(f"ERROR: bootstrap_h_values.csv not found in {input_dir}")
        sys.exit(1)

    print(f"      Input dir: {input_dir}")
    input_file = str(input_path)

    # Determine LOESS window: explicit args > metadata > default
    # Try to load metadata from bootstrap directory
    metadata = load_run_metadata(input_dir)

    if args.mean_weight_distance is not None:
        loess_window = (44 / 7) * args.mean_weight_distance
        print(f"      LOESS window: {loess_window:.1f} years (from --mean-weight-distance {args.mean_weight_distance})")
    elif args.loess_window != DEFAULT_LOESS_WINDOW_YEARS:
        loess_window = args.loess_window
        print(f"      LOESS window: {loess_window} years (from --loess-window)")
    elif 'loess_window' in metadata:
        loess_window = metadata['loess_window']
        mwd = metadata.get('mean_weight_distance')
        if mwd is not None:
            print(f"      LOESS window: {loess_window:.1f} years (from metadata, mean_weight_distance={mwd})")
        else:
            print(f"      LOESS window: {loess_window:.1f} years (from metadata)")
    else:
        loess_window = DEFAULT_LOESS_WINDOW_YEARS
        print(f"      LOESS window: {loess_window} years (default)")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="cumulative_")

    # Phase 1: Process all countries with point estimates
    print("\n[1/7] Processing all countries (point estimates)...")
    df_all_countries = process_all_countries_point_estimate(input_path, input_dir, loess_window)

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
        input_path, input_dir, loess_window
    )

    # Print selected countries with percentage interpretation
    print("      Selected countries (log value → percent equivalent):")
    country_order = get_country_ordering(representatives)
    for key in country_order:
        info = representatives[key]
        if key == 'min':
            label = "Min"
        elif key == 'max':
            label = "Max"
        else:
            label = f"P{key:2d}"
        pct = inv_log_transform(info['value'])
        print(f"        {label}: {info['iso3']} (log={info['value']:+.4f} → {pct:+.1f}%)")

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
        input_path, rep_iso3s, input_dir, loess_window
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

    print("\n[7/7] Creating box plot visualization (grouped by approach)...")
    plot_cumulative_effects_by_approach_grouped(df_summary, representatives, output_dir, input_file)

    # Additional: Create 2-panel Approach DL plot
    print("\n[Bonus] Creating Approach DL 2-panel visualization...")
    plot_cumulative_effects_ApproachDL(df_all_countries, df_summary, representatives, output_dir, input_file)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return output_dir


if __name__ == "__main__":
    main()
