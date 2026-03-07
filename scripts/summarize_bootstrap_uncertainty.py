#!/usr/bin/env python3
"""Summarize bootstrap uncertainty: width comparisons, bar charts, and LaTeX tables.

This script computes bootstrap-of-bootstrap width comparisons (P/L vs J),
generates bar charts showing IQR and 90% CI ranges, and produces a LaTeX
summary table with point estimates, uncertainty ranges, and significance stars.

Outputs:
    - bootstrap_width_comparison.csv  (width comparison p-values)
    - bootstrap_uncertainty_combined.pdf  (bar chart figure)
    - approach_summary.tex  (LaTeX table with \\estciqci macro)

Usage:
    python scripts/summarize_bootstrap_uncertainty.py
    python scripts/summarize_bootstrap_uncertainty.py --bootstrap-dir data/output/reference/bootstrap_20260226_025028
    python scripts/summarize_bootstrap_uncertainty.py --output-dir data/output/figures
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

from compare_bootstrap_widths import compute_width_comparisons
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
        ('Approach Q', 'Quadratic', ['Approach QJ', 'Approach QP', 'Approach QL']),
        ('Approach P', 'Piecewise', ['Approach PJ', 'Approach PP', 'Approach PL']),
        ('Approach D', 'Persistence', ['Approach DJ', 'Approach DP', 'Approach DL']),
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

            # Approach Q doesn't have h4, so leave that cell blank
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


def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def plot_combined_uncertainty_bars(
    bootstrap_summary: pd.DataFrame,
    output_dir: Path,
    width_comparisons: pd.DataFrame = None,
) -> None:
    """Create bar charts showing both IQR and 90% CI ranges on the same panels.

    Bars are grouped by range type (90% CI, IQR) with J, P, L bars within each group.

    Parameters
    ----------
    bootstrap_summary : pd.DataFrame
        Bootstrap summary table with percentile columns for each variable
    output_dir : Path
        Directory to save output figures
    width_comparisons : pd.DataFrame, optional
        Width comparison results from compute_width_comparisons(). If provided,
        significance stars are shown above P and L bars.
    """
    # Define approach groups: rows are approach types, columns are variables
    approach_groups = [
        ('Approach Q', 'Quadratic', ['Approach QJ', 'Approach QP', 'Approach QL']),
        ('Approach P', 'Piecewise', ['Approach PJ', 'Approach PP', 'Approach PL']),
        ('Approach D', 'Persistence', ['Approach DJ', 'Approach DP', 'Approach DL']),
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

            # Approach Q doesn't have h4, so leave that cell blank
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

                # Add significance stars for P and L methods
                if width_comparisons is not None and method in ('P', 'L'):
                    response_type = group_key.split()[-1]  # e.g. 'Q' from 'Approach Q'
                    comparison_label = f'{method} vs J'
                    mask = (
                        (width_comparisons['response'] == response_type) &
                        (width_comparisons['coefficient'] == var) &
                        (width_comparisons['comparison'] == comparison_label)
                    )
                    for range_type_key, cluster_idx, bar_height in [('90ci', 0, ci90_ranges[i]), ('iqr', 1, iqr_ranges[i])]:
                        row = width_comparisons[mask & (width_comparisons['range_type'] == range_type_key)]
                        if not row.empty and bar_height > 0:
                            stars = p_to_stars(row['p_alt_narrower'].values[0])
                            if stars:
                                ax.text(x_clusters[cluster_idx] + offset, bar_height,
                                        stars, ha='center', va='bottom', fontsize=8, fontweight='bold')

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


def _format_value(value, decimals):
    """Format a numeric value for LaTeX, handling infinity."""
    if abs(value) > 100:
        return r'$\infty$'
    return f'{value:.{decimals}f}'


def generate_approach_summary_latex(
    bootstrap_summary: pd.DataFrame,
    bootstrap_coefficients: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate LaTeX table summarizing approach parameters with bootstrap uncertainty.

    Each cell uses the \\estciq macro:
        \\estciq{point}{iqr_low}{iqr_high}{ci_low}{ci_high}

    Parameters
    ----------
    bootstrap_summary : pd.DataFrame
        Bootstrap summary table with percentile columns
    bootstrap_coefficients : pd.DataFrame
        Raw bootstrap samples (needed for computing 1/h4 percentiles)
    output_dir : Path
        Directory to save the .tex file
    """
    # Table row definitions: (approach_key, short_label)
    rows = [
        ('Approach QJ', 'QJ'),
        ('Approach QP', 'QP'),
        ('Approach QL', 'QL'),
        ('Approach PJ', 'PJ'),
        ('Approach PP', 'PP'),
        ('Approach PL', 'PL'),
        ('Approach DJ', 'DJ'),
        ('Approach DP', 'DP'),
        ('Approach DL', 'DL'),
    ]

    # Response type section headers
    section_headers = {
        'Approach QJ': r'\multicolumn{5}{l}{\textit{Quadratic response}} \\',
        'Approach PJ': r'\multicolumn{5}{l}{\textit{Piecewise quadratic response}} \\',
        'Approach DJ': r'\multicolumn{5}{l}{\textit{Decay (persistence) response}} \\',
    }

    # Precompute 1/h4 percentiles for D approaches from raw bootstrap samples
    boot_samples = bootstrap_coefficients[bootstrap_coefficients['iteration'] >= 0]
    inv_h4_stats = {}
    for approach in ['Approach DJ', 'Approach DP', 'Approach DL']:
        h4_samples = boot_samples[boot_samples['approach'] == approach]['h4'].dropna().values
        inv_samples = 1.0 / h4_samples
        summary_row = bootstrap_summary[bootstrap_summary['approach'] == approach].iloc[0]
        inv_h4_stats[approach] = {
            'point': 1.0 / summary_row['h4_point'],
            'p25': np.percentile(inv_samples, 25),
            'p75': np.percentile(inv_samples, 75),
            'p5': np.percentile(inv_samples, 5),
            'p95': np.percentile(inv_samples, 95),
        }

    def make_cell(point, p25, p75, p5, p95, decimals):
        """Format a single \\estciq cell."""
        fmt = lambda v: _format_value(v, decimals)
        return (f'\\estciq{{{fmt(point)}}}'
                f'{{{fmt(p25)}}}{{{fmt(p75)}}}'
                f'{{{fmt(p5)}}}{{{fmt(p95)}}}')

    # Build LaTeX
    lines = []
    lines.append(r'% Auto-generated by summarize_bootstrap_uncertainty.py')
    lines.append(r'% three-line numeric cell')
    lines.append(r'\newcommand{\estciq}[5]{\makecell[c]{#1\\\footnotesize[#2, #3]\\\footnotesize[#4, #5]}}')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Climate response parameter estimates with bootstrap uncertainty.')
    lines.append(r'  Point estimates shown with IQR [p25, p75] and 90\% CI [p5, p95].}')
    lines.append(r'\label{tab:approach_summary}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r' & {$1000 \cdot h_2$}'
                 r' & {$1000 \cdot h_2$ ($T>T_\mathrm{opt}$)}'
                 r' & {$T_\mathrm{opt}$}'
                 r' & {$1/f$} \\')
    lines.append(r'Approach'
                 r' & {(\si{\per\celsius\squared})}'
                 r' & {(\si{\per\celsius\squared})}'
                 r' & {(\si{\celsius})}'
                 r' & {(\si{yr})} \\')
    lines.append(r'\midrule')

    for approach, label in rows:
        # Section header
        if approach in section_headers:
            if approach != 'Approach QJ':
                lines.append(r'\addlinespace[4pt]')
            lines.append(section_headers[approach])

        summary_row = bootstrap_summary[bootstrap_summary['approach'] == approach].iloc[0]
        response_type = label[0]  # Q, P, or D

        # Column 1: 1000*h2 (all approaches)
        h2_cell = make_cell(
            summary_row['h2_point'] * 1000,
            summary_row['h2_p25'] * 1000,
            summary_row['h2_p75'] * 1000,
            summary_row['h2_p5'] * 1000,
            summary_row['h2_p95'] * 1000,
            decimals=3,
        )

        # Column 2: 1000*h4 (Piecewise only)
        if response_type == 'P':
            h4_cell = make_cell(
                summary_row['h4_point'] * 1000,
                summary_row['h4_p25'] * 1000,
                summary_row['h4_p75'] * 1000,
                summary_row['h4_p5'] * 1000,
                summary_row['h4_p95'] * 1000,
                decimals=3,
            )
        else:
            h4_cell = ''

        # Column 3: T_opt (all approaches)
        topt_cell = make_cell(
            summary_row['T_opt_point'],
            summary_row['T_opt_p25'],
            summary_row['T_opt_p75'],
            summary_row['T_opt_p5'],
            summary_row['T_opt_p95'],
            decimals=1,
        )

        # Column 4: 1/f (Decay only)
        if response_type == 'D':
            stats = inv_h4_stats[approach]
            inv_f_cell = make_cell(
                stats['point'], stats['p25'], stats['p75'],
                stats['p5'], stats['p95'],
                decimals=2,
            )
        else:
            inv_f_cell = ''

        lines.append(f'{label} & {h2_cell} & {h4_cell} & {topt_cell} & {inv_f_cell} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_path = output_dir / 'approach_summary.tex'
    tex_path.write_text('\n'.join(lines) + '\n')
    print(f'      Saved: {tex_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize bootstrap uncertainty: width comparisons, bar charts, and LaTeX tables"
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
    parser.add_argument(
        "--n-resample",
        type=int,
        default=10000,
        help="Number of meta-bootstrap resamples for width comparison (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for width comparison (default: 42)",
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
    print("Bootstrap Uncertainty Summary")
    print("=" * 70)

    print(f"\n[1/5] Loading bootstrap data...")
    print(f"      Bootstrap dir: {bootstrap_dir}")

    summary_path = bootstrap_dir / "bootstrap_summary_table.csv"
    if not summary_path.exists():
        print(f"ERROR: bootstrap_summary_table.csv not found at {summary_path}")
        sys.exit(1)

    bootstrap_summary = pd.read_csv(summary_path, comment='#')
    print(f"      Loaded {len(bootstrap_summary)} approaches")

    coeff_path = bootstrap_dir / "bootstrap_coefficients.csv"
    coeff_df_full = None
    if coeff_path.exists():
        coeff_df_full = pd.read_csv(coeff_path, comment='#')
        print(f"      Loaded {len(coeff_df_full)} coefficient samples")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else bootstrap_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/5] Output directory: {output_dir}")

    # Compute width comparisons
    width_comparisons = None
    if coeff_df_full is not None:
        print(f"\n[3/5] Computing bootstrap width comparisons...")
        coeff_df = coeff_df_full[coeff_df_full['iteration'] >= 0]
        width_comparisons = compute_width_comparisons(coeff_df, args.n_resample, args.seed)
        csv_path = output_dir / "bootstrap_width_comparison.csv"
        width_comparisons.to_csv(csv_path, index=False)
        print(f"      Saved: {csv_path}")
    else:
        print(f"\n[3/5] No bootstrap_coefficients.csv found, skipping width comparisons")

    # Generate bar chart
    print(f"\n[4/5] Generating uncertainty bar chart...")
    plot_combined_uncertainty_bars(bootstrap_summary, output_dir, width_comparisons)

    # Generate LaTeX approach summary table
    print(f"\n[5/5] Generating LaTeX approach summary table...")
    if coeff_df_full is not None:
        generate_approach_summary_latex(
            bootstrap_summary, coeff_df_full, output_dir,
        )
    else:
        print("      Skipped: bootstrap_coefficients.csv required for 1/h4 computation")

    print("\n" + "=" * 70)
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70)

    return output_dir


if __name__ == "__main__":
    main()
