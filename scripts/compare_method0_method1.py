#!/usr/bin/env python3
"""Compare parameters from Approach 0 (conjoined OLS) vs Approach 5c (precomputed k).

Generates a 2x2 scatter plot comparing:
  (a) k(t) year fixed effects
  (b) j_{0,i} country intercepts
  (c) j_{1,i} country linear trend coefficients
  (d) j_{2,i} country quadratic trend coefficients

Each panel shows Approach 5c predictions on x-axis vs Approach 0 values on y-axis,
with a 1:1 reference line and R^2 annotation.

For the j coefficients, Approach 5c's raw predictions are re-referenced by subtracting
country 0's values (to match Approach 0's identification constraint j=0 for country 0).
The subtracted quadratic is absorbed into k, so Approach 5c's k becomes:
  k'(t) = k_mean(t) + j_{0,0} + j_{1,0}*t + j_{2,0}*t^2
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_country_trends_with_k,
    compute_year_means,
    fit_quadratic_trend,
)
from src.fitting import fit_method0_no_detrending, fit_method1_precomputed_k_combined
from src.output import add_input_file_annotation


def extract_method0_j_coefficients(data, result0):
    """Extract per-country j coefficients from Approach 0 results.

    For each country, compute residual r_i(t) = dy_i(t) - h1*T_i(t) - h2*T_i(t)^2 - k(t),
    then fit quadratic j_{0,i} + j_{1,i}*t + j_{2,i}*t^2 to r_i(t).

    Returns:
        Tuple of (j0_dict, j1_dict, j2_dict), each mapping country_idx -> coefficient.
    """
    h1 = result0.h1
    h2 = result0.h2
    k = result0.k

    j0 = {}
    j1 = {}
    j2 = {}

    for c in range(data.n_countries):
        mask = data.country_idx == c
        t_c = data.time[mask]
        dy_c = data.growth_pcGDP[mask]
        T_c = data.temp[mask]
        yr_c = data.year[mask]

        # Residual after removing climate response and year effects
        k_c = np.array([k[yr] for yr in yr_c])
        r_c = dy_c - h1 * T_c - h2 * T_c**2 - k_c

        # Fit quadratic trend to residual
        j0[c], j1[c], j2[c] = fit_quadratic_trend(t_c, r_c)

    return j0, j1, j2


def compute_predicted_j(data, trends_with_k, trends, h1, h2):
    """Compute predicted j coefficients from Approach 5c trends and given h1, h2.

    Raw (un-referenced) j coefficients:
        j_{0,i} = g_{0,i} - (h1 * T_{0,i} + h2 * T_{0,i}^2)
        j_{1,i} = g_{1,i} - (h1 * T_{1,i} + 2*h2 * T_{0,i} * T_{1,i})
        j_{2,i} = g_{2,i} - h2 * T_{1,i}^2

    Then re-referenced by subtracting country 0's values (to match Approach 0's
    identification constraint j=0 for country 0). The subtracted quadratic
    j_{0,0} + j_{1,0}*t + j_{2,0}*t^2 is returned separately to be added to k.

    Returns:
        Tuple of (j0_pred, j1_pred, j2_pred, j_ref) where j_ref = (j0_ref, j1_ref, j2_ref)
        are the reference country's raw j coefficients.
    """

    j0_raw = {}
    j1_raw = {}
    j2_raw = {}

    for c in range(data.n_countries):
        g0 = trends_with_k.y0[c]
        g1 = trends_with_k.y1[c]
        g2 = trends_with_k.y2[c]
        T0 = trends.T0[c]
        T1 = trends.T1[c]

        j0_raw[c] = g0 - (h1 * T0 + h2 * T0**2)
        j1_raw[c] = g1 - (h1 * T1 + 2 * h2 * T0 * T1)
        j2_raw[c] = g2 - h2 * T1**2

    # Re-reference: subtract country 0's values so country 0 has j=0
    j0_ref = j0_raw[0]
    j1_ref = j1_raw[0]
    j2_ref = j2_raw[0]

    j0_pred = {c: j0_raw[c] - j0_ref for c in range(data.n_countries)}
    j1_pred = {c: j1_raw[c] - j1_ref for c in range(data.n_countries)}
    j2_pred = {c: j2_raw[c] - j2_ref for c in range(data.n_countries)}

    return j0_pred, j1_pred, j2_pred, (j0_ref, j1_ref, j2_ref)


def make_scatter_panel(ax, x, y, label, panel_letter):
    """Create a single scatter panel with 1:1 line and R^2 annotation."""
    ax.scatter(x, y, s=10, alpha=0.6, edgecolors='none')

    # Set equal axis ranges so 1:1 line goes corner to corner
    all_vals = np.concatenate([x, y])
    lo, hi = np.min(all_vals), np.max(all_vals)
    margin = 0.05 * (hi - lo)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    # 1:1 reference line (corner to corner)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            '--', color='gray', linewidth=0.8, zorder=0)

    # Best-fit linear regression: y = slope*x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    # Plot best-fit line
    ax.plot([lo - margin, hi + margin],
            [slope * (lo - margin) + intercept, slope * (hi + margin) + intercept],
            '-', color='red', linewidth=0.8, alpha=0.7, zorder=1)

    # R^2 (relative to 1:1 line)
    ss_res = np.sum((y - x)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Format intercept sign
    sign = '+' if intercept >= 0 else '\u2212'
    abs_int = abs(intercept)

    ax.text(0.05, 0.95,
            f'y = {slope:.4f}x {sign} {abs_int:.4f}\n'
            f'R$^2$ = {r2:.4f}   r = {corr:.4f}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(f'Approach 5c ({label})', fontsize=9)
    ax.set_ylabel(f'Approach 0 ({label})', fontsize=9)
    ax.set_title(f'({panel_letter}) {label}', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser(
        description="Compare Approach 0 vs 5c parameters via scatter plots."
    )
    parser.add_argument(
        "--data-file", type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Path to input CSV (default: data/input/Maddison_CRU_dataset.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/output/method0_vs_5c",
        help="Output directory (default: data/output/method0_vs_5c)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data and compute trends ---
    print("Loading data...")
    data = load_data_from_csv(args.data_file)

    print("Computing trends...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)

    # --- Fit both approaches ---
    print("Fitting Approach 0...")
    result0 = fit_method0_no_detrending(data)

    print("Fitting Approach 5c...")
    result5c = fit_method1_precomputed_k_combined(data, trends_with_k, year_means)

    # --- Plots 2-4 data: j coefficients ---
    print("Extracting Approach 0 j coefficients...")
    j0_actual, j1_actual, j2_actual = extract_method0_j_coefficients(data, result0)

    print("Computing predicted j coefficients from Approach 5c trends + Approach 0 h1,h2...")
    j0_pred, j1_pred, j2_pred, j_ref = compute_predicted_j(
        data, trends_with_k, trends, result0.h1, result0.h2
    )
    j0_ref, j1_ref, j2_ref = j_ref

    # --- Plot 1 data: k(t) ---
    # Approach 0: k(t) raw
    unique_years = sorted(result0.k.keys())
    k0_values = np.array([result0.k[yr] for yr in unique_years])
    t_years = np.array([yr - data.time_offset for yr in unique_years])

    # Approach 5c: k_mean(t) + reference country's j quadratic
    # (absorb the subtracted j_ref into k to match Approach 0's parameterization)
    k5c_values = np.array([
        year_means[yr] + j0_ref + j1_ref * (yr - data.time_offset)
        + j2_ref * (yr - data.time_offset)**2
        for yr in unique_years
    ])

    # Build arrays for scatter (all countries including reference country 0)
    countries = list(range(data.n_countries))
    j0_act_arr = np.array([j0_actual[c] for c in countries])
    j0_pred_arr = np.array([j0_pred[c] for c in countries])
    j1_act_arr = np.array([j1_actual[c] for c in countries])
    j1_pred_arr = np.array([j1_pred[c] for c in countries])
    j2_act_arr = np.array([j2_actual[c] for c in countries])
    j2_pred_arr = np.array([j2_pred[c] for c in countries])

    # --- Generate 2x2 figure ---
    print("Creating scatter plots...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    make_scatter_panel(axes[0, 0], k5c_values, k0_values,
                       'k(t)', 'a')
    make_scatter_panel(axes[0, 1], j0_pred_arr, j0_act_arr,
                       r'$j_{0,i}$', 'b')
    make_scatter_panel(axes[1, 0], j1_pred_arr, j1_act_arr,
                       r'$j_{1,i}$', 'c')
    make_scatter_panel(axes[1, 1], j2_pred_arr, j2_act_arr,
                       r'$j_{2,i}$', 'd')

    fig.suptitle('Approach 0 vs Approach 5c: Parameter Comparison',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    add_input_file_annotation(fig, args.data_file)

    # Save figure
    pdf_path = output_dir / "method0_vs_5c_scatter.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure to {pdf_path}")

    # Save scatter data to CSV
    csv_path = output_dir / "method0_vs_5c_scatter_data.csv"
    with open(csv_path, 'w') as f:
        # k(t) section
        f.write("# k(t) scatter data\n")
        f.write("year,k_method0,k_method1\n")
        for i, yr in enumerate(unique_years):
            f.write(f"{yr},{k0_values[i]:.8f},{k5c_values[i]:.8f}\n")
        f.write("\n")

        # j coefficients section
        f.write("# j coefficient scatter data\n")
        f.write("country_idx,iso,j0_method0,j0_method1_pred,"
                "j1_method0,j1_method1_pred,"
                "j2_method0,j2_method1_pred\n")
        for c in countries:
            iso = data.idx_to_iso[c]
            f.write(f"{c},{iso},"
                    f"{j0_actual[c]:.8f},{j0_pred[c]:.8f},"
                    f"{j1_actual[c]:.8f},{j1_pred[c]:.8f},"
                    f"{j2_actual[c]:.8f},{j2_pred[c]:.8f}\n")
    print(f"Saved scatter data to {csv_path}")

    # Print summary statistics
    print("\n--- Summary ---")
    for label, x, y in [
        ("k(t)", k5c_values, k0_values),
        ("j0", j0_pred_arr, j0_act_arr),
        ("j1", j1_pred_arr, j1_act_arr),
        ("j2", j2_pred_arr, j2_act_arr),
    ]:
        ss_res = np.sum((y - x)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        corr = np.corrcoef(x, y)[0, 1]
        print(f"  {label:20s}  R^2 = {r2:.6f}  r = {corr:.6f}")


if __name__ == "__main__":
    main()
