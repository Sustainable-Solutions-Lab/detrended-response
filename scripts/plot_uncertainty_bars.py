#!/usr/bin/env python3
"""Generate bar charts showing bootstrap uncertainty ranges for coefficient estimates.

This script creates bar charts comparing IQR and 90% CI ranges for h2, T_opt, and h4
across different approaches (ApproachQ, ApproachP, ApproachD) and estimation methods
(J=Joint, P=Polynomial, L=LOESS).

Usage:
    python scripts/plot_uncertainty_bars.py
    python scripts/plot_uncertainty_bars.py --bootstrap-dir data/output/reference/bootstrap_20260226_025028
    python scripts/plot_uncertainty_bars.py --output-dir data/output/figures
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.output import APPROACH_COLORS


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


def plot_uncertainty_range_bars(
    bootstrap_summary: pd.DataFrame,
    output_dir: Path,
    range_type: str = 'iqr',
) -> None:
    """Create bar charts showing uncertainty ranges for bootstrap coefficient estimates.

    Parameters
    ----------
    bootstrap_summary : pd.DataFrame
        Bootstrap summary table with percentile columns for each variable
    output_dir : Path
        Directory to save output figures
    range_type : str
        Type of range to plot: 'iqr' for interquartile range (p75-p25) or
        '90ci' for 90% confidence interval (p95-p5)
    """
    # Define approach groups: rows are approach types, columns are variables
    # Row 1: ApproachQ (h2, T_opt, [blank])
    # Row 2: ApproachP (h2, T_opt, h4)
    # Row 3: ApproachD (h2, T_opt, h4)
    approach_groups = [
        ('ApproachQ', 'Quadratic', ['ApproachQJ', 'ApproachQP', 'ApproachQL']),
        ('ApproachP', 'Piecewise', ['ApproachPJ', 'ApproachPP', 'ApproachPL']),
        ('ApproachD', 'Persistence', ['ApproachDJ', 'ApproachDP', 'ApproachDL']),
    ]

    # Variables for each column
    variables = ['h2', 'T_opt', 'h4']

    # Consistent colors for J, P, L variants
    method_colors = {
        'J': '#2c3e50',  # Dark blue-gray
        'P': '#e74c3c',  # Red
        'L': '#f39c12',  # Orange
    }

    # Variable display names
    var_titles = {
        'h2': r'$h_2$',
        'T_opt': r'$T_{opt}$',
        'h4': r'$h_4$',
    }

    # Determine percentile columns based on range type
    if range_type == 'iqr':
        low_suffix, high_suffix = '_p25', '_p75'
        title_suffix = 'IQR (p25-p75)'
        file_suffix = 'iqr'
    else:
        low_suffix, high_suffix = '_p5', '_p95'
        title_suffix = '90% CI (p5-p95)'
        file_suffix = '90ci'

    # Create 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f'Bootstrap Uncertainty Ranges: {title_suffix}', fontsize=14, fontweight='bold')

    methods = ['J', 'P', 'L']
    x = np.arange(len(methods))
    width = 0.6

    for row_idx, (group_key, group_name, approaches) in enumerate(approach_groups):
        for col_idx, var in enumerate(variables):
            ax = axes[row_idx, col_idx]

            # ApproachQ doesn't have h4, so leave that cell blank
            if row_idx == 0 and col_idx == 2:
                ax.axis('off')
                continue

            # Get ranges for each method (J, P, L)
            ranges = []
            colors = []
            for method in methods:
                approach = f'{group_key}{method}'
                row_data = bootstrap_summary[bootstrap_summary['approach'] == approach]

                if row_data.empty:
                    ranges.append(0)
                    colors.append(method_colors[method])
                    continue

                low_col = f'{var}{low_suffix}'
                high_col = f'{var}{high_suffix}'

                if low_col in row_data.columns and high_col in row_data.columns:
                    low_val = row_data[low_col].values[0]
                    high_val = row_data[high_col].values[0]
                    if pd.notna(low_val) and pd.notna(high_val):
                        ranges.append(high_val - low_val)
                    else:
                        ranges.append(0)
                else:
                    ranges.append(0)
                colors.append(method_colors[method])

            # Plot bars
            bars = ax.bar(x, ranges, width, color=colors, edgecolor='white', linewidth=0.5)

            # Customize subplot
            ax.set_xticks(x)
            ax.set_xticklabels(methods)

            # Add title to top row panels (h4 title goes on row 1 since row 0 is blank)
            if row_idx == 0 and col_idx < 2:
                ax.set_title(var_titles[var], fontsize=12, fontweight='bold')
            elif row_idx == 1 and col_idx == 2:
                ax.set_title(var_titles[var], fontsize=12, fontweight='bold')

            # Add row labels on the left
            if col_idx == 0:
                ax.set_ylabel(f'{group_key}\n({group_name})', fontsize=10, fontweight='bold')

            # Add grid for readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'bootstrap_uncertainty_{file_suffix}.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"      Saved: {output_path}")
    plt.close(fig)


def plot_combined_uncertainty_bars(
    bootstrap_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create bar charts showing both IQR and 90% CI ranges on the same panels.

    Bars are grouped by range type (90% CI, IQR) with J, P, L bars within each group.

    Parameters
    ----------
    bootstrap_summary : pd.DataFrame
        Bootstrap summary table with percentile columns for each variable
    output_dir : Path
        Directory to save output figures
    """
    # Define approach groups: rows are approach types, columns are variables
    approach_groups = [
        ('ApproachQ', 'Quadratic', ['ApproachQJ', 'ApproachQP', 'ApproachQL']),
        ('ApproachP', 'Piecewise', ['ApproachPJ', 'ApproachPP', 'ApproachPL']),
        ('ApproachD', 'Persistence', ['ApproachDJ', 'ApproachDP', 'ApproachDL']),
    ]

    # Variables for each column
    variables = ['h2', 'T_opt', 'h4']

    # Consistent colors for J, P, L variants
    method_colors = {
        'J': '#2c3e50',  # Dark blue-gray
        'P': '#e74c3c',  # Red
        'L': '#f39c12',  # Orange
    }

    # Variable display names
    var_titles = {
        'h2': r'$h_2$',
        'T_opt': r'$T_{opt}$',
        'h4': r'$h_4$',
    }

    # Create 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('Bootstrap Uncertainty Ranges: IQR and 90% CI', fontsize=14, fontweight='bold')

    methods = ['J', 'P', 'L']
    n_methods = len(methods)
    range_types = ['90% CI', 'IQR']
    n_range_types = len(range_types)

    # X positions for the two clusters (90% CI and IQR)
    x_clusters = np.arange(n_range_types)
    width = 0.25  # Width for each bar
    cluster_width = n_methods * width

    for row_idx, (group_key, group_name, approaches) in enumerate(approach_groups):
        for col_idx, var in enumerate(variables):
            ax = axes[row_idx, col_idx]

            # ApproachQ doesn't have h4, so leave that cell blank
            if row_idx == 0 and col_idx == 2:
                ax.axis('off')
                continue

            # Get ranges for each method (J, P, L)
            iqr_ranges = []
            ci90_ranges = []
            for method in methods:
                approach = f'{group_key}{method}'
                row_data = bootstrap_summary[bootstrap_summary['approach'] == approach]

                if row_data.empty:
                    iqr_ranges.append(0)
                    ci90_ranges.append(0)
                    continue

                # IQR (p25-p75)
                iqr_low_col = f'{var}_p25'
                iqr_high_col = f'{var}_p75'
                if iqr_low_col in row_data.columns and iqr_high_col in row_data.columns:
                    low_val = row_data[iqr_low_col].values[0]
                    high_val = row_data[iqr_high_col].values[0]
                    if pd.notna(low_val) and pd.notna(high_val):
                        iqr_ranges.append(high_val - low_val)
                    else:
                        iqr_ranges.append(0)
                else:
                    iqr_ranges.append(0)

                # 90% CI (p5-p95)
                ci_low_col = f'{var}_p5'
                ci_high_col = f'{var}_p95'
                if ci_low_col in row_data.columns and ci_high_col in row_data.columns:
                    low_val = row_data[ci_low_col].values[0]
                    high_val = row_data[ci_high_col].values[0]
                    if pd.notna(low_val) and pd.notna(high_val):
                        ci90_ranges.append(high_val - low_val)
                    else:
                        ci90_ranges.append(0)
                else:
                    ci90_ranges.append(0)

            # Plot bars grouped by range type
            # First cluster: 90% CI with J, P, L bars (solid)
            # Second cluster: IQR with J, P, L bars (half-tone/hatched)
            for i, method in enumerate(methods):
                offset = (i - (n_methods - 1) / 2) * width
                # 90% CI cluster (x=0) - solid bars
                bar_90 = ax.bar(x_clusters[0] + offset, ci90_ranges[i], width,
                                color=method_colors[method], edgecolor='white', linewidth=0.5)
                # IQR cluster (x=1) - lighter tone bars (no hatching)
                bar_iqr = ax.bar(x_clusters[1] + offset, iqr_ranges[i], width,
                                 color=method_colors[method], edgecolor='white', linewidth=0.5,
                                 alpha=2/3)

                # Add method labels (J, P, L) in the middle of each bar
                # 90% CI bar label
                if ci90_ranges[i] > 0:
                    ax.text(x_clusters[0] + offset, ci90_ranges[i] / 2, method,
                            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
                # IQR bar label
                if iqr_ranges[i] > 0:
                    ax.text(x_clusters[1] + offset, iqr_ranges[i] / 2, method,
                            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # Customize subplot
            ax.set_xticks(x_clusters)
            ax.set_xticklabels(range_types)

            # Add title to top row panels (h4 title goes on row 1 since row 0 is blank)
            if row_idx == 0 and col_idx < 2:
                ax.set_title(var_titles[var], fontsize=12, fontweight='bold')
            elif row_idx == 1 and col_idx == 2:
                ax.set_title(var_titles[var], fontsize=12, fontweight='bold')

            # Add row labels on the left and unit labels
            # Units: h2 = °C⁻², T_opt = °C, h4 = °C⁻² (ApproachP) or year⁻¹ (ApproachD)
            if col_idx == 0:
                # h2 column: °C⁻²
                ax.set_ylabel(f'{group_key}\n({group_name})\n[°C$^{{-2}}$]', fontsize=10, fontweight='bold')
            elif col_idx == 1:
                # T_opt column: °C
                ax.set_ylabel('[°C]', fontsize=10)
            elif col_idx == 2:
                # h4 column: depends on approach
                if row_idx == 1:  # ApproachP (Piecewise)
                    ax.set_ylabel('[°C$^{{-2}}$]', fontsize=10)
                elif row_idx == 2:  # ApproachD (Persistence)
                    ax.set_ylabel('[year$^{{-1}}$]', fontsize=10)

            # Add grid for readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

    # Add legend to the blank cell (row 0, col 2)
    ax_legend = axes[0, 2]
    ax_legend.axis('off')
    # Create proxy artists for legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='90% CI'),
        Patch(facecolor='gray', edgecolor='black', alpha=2/3, label='IQR'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'bootstrap_uncertainty_combined.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"      Saved: {output_path}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate bar charts showing bootstrap uncertainty ranges"
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=str,
        default=None,
        help="Path to bootstrap output directory (default: most recent in data/output/reference/bootstrap_*)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: same as bootstrap directory)",
    )

    args = parser.parse_args(argv)

    # Find bootstrap directory
    if args.bootstrap_dir:
        bootstrap_dir = Path(args.bootstrap_dir)
    else:
        bootstrap_dir_str = find_most_recent_dir("data/output/reference/bootstrap_*")
        if bootstrap_dir_str is None:
            print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
            print("       Please specify --bootstrap-dir explicitly")
            sys.exit(1)
        bootstrap_dir = Path(bootstrap_dir_str)

    print("=" * 70)
    print("Bootstrap Uncertainty Range Bar Charts")
    print("=" * 70)

    print(f"\n[1/3] Loading bootstrap summary...")
    print(f"      Bootstrap dir: {bootstrap_dir}")

    summary_path = bootstrap_dir / "bootstrap_summary_table.csv"
    if not summary_path.exists():
        print(f"ERROR: bootstrap_summary_table.csv not found at {summary_path}")
        sys.exit(1)

    bootstrap_summary = pd.read_csv(summary_path, comment='#')
    print(f"      Loaded {len(bootstrap_summary)} approaches")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else bootstrap_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Output directory: {output_dir}")

    # Generate combined chart
    print(f"\n[3/3] Generating bar chart...")

    plot_combined_uncertainty_bars(bootstrap_summary, output_dir)

    print("\n" + "=" * 70)
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)

    return output_dir


if __name__ == "__main__":
    main()
