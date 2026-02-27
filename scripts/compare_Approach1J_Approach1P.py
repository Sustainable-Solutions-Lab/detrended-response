#!/usr/bin/env python3
"""Compare parameters from ApproachQJ (conjoined OLS) vs ApproachQP (precomputed k).

Generates a 2x2 scatter plot comparing:
  (a) k(t) year fixed effects
  (b) j_{0,i} country intercepts
  (c) j_{1,i} country linear trend coefficients
  (d) j_{2,i} country quadratic trend coefficients

Each panel shows ApproachQP predictions on x-axis vs ApproachQJ values on y-axis,
with a 1:1 reference line and R^2 annotation.

For the j coefficients, ApproachQP's raw predictions are re-referenced by subtracting
country 0's values (to match ApproachQJ's identification constraint j=0 for country 0).
The subtracted quadratic is absorbed into k, so ApproachQP's k becomes:
  k'(t) = k_mean(t) + j_{0,0} + j_{1,0}*t + j_{2,0}*t^2

Outputs:
  - ApproachQJ_vs_ApproachQP_scatter.pdf: 2x2 scatter plot figure
  - ApproachQJ_vs_ApproachQP_scatter_data.csv: raw scatter data
  - ApproachQJ_vs_ApproachQP_table.tex: LaTeX table with slope/intercept/R²/r
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_country_trends_with_k,
    compute_year_means,
    fit_quadratic_trend,
)
from src.fitting import fit_ApproachQJ_conjoined, fit_ApproachQP_precomputed_k
from src.output import add_input_file_annotation


def extract_ApproachQJ_j_coefficients(data, result0):
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
    """Compute predicted j coefficients from ApproachQP trends and given h1, h2.

    Raw (un-referenced) j coefficients:
        j_{0,i} = g_{0,i} - (h1 * T_{0,i} + h2 * T_{0,i}^2)
        j_{1,i} = g_{1,i} - (h1 * T_{1,i} + 2*h2 * T_{0,i} * T_{1,i})
        j_{2,i} = g_{2,i} - h2 * T_{1,i}^2

    Then re-referenced by subtracting country 0's values (to match ApproachQJ's
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

    ax.set_xlabel(f'ApproachQP ({label})', fontsize=9)
    ax.set_ylabel(f'ApproachQJ ({label})', fontsize=9)
    ax.set_title(f'({panel_letter}) {label}', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser(
        description="Compare ApproachQJ vs ApproachQP parameters via scatter plots."
    )
    parser.add_argument(
        "--data-file", type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Path to input CSV (default: data/input/Maddison_CRU_dataset.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/output/ApproachQJ_vs_ApproachQP",
        help="Output directory (default: data/output/ApproachQJ_vs_ApproachQP)"
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
    print("Fitting ApproachQJ...")
    result0 = fit_ApproachQJ_conjoined(data)

    print("Fitting ApproachQP...")
    result1 = fit_ApproachQP_precomputed_k(data, trends_with_k, year_means)

    # --- Plots 2-4 data: j coefficients ---
    print("Extracting ApproachQJ j coefficients...")
    j0_actual, j1_actual, j2_actual = extract_ApproachQJ_j_coefficients(data, result0)

    print("Computing predicted j coefficients from ApproachQP trends + ApproachQJ h1,h2...")
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

    fig.tight_layout(rect=[0, 0.02, 1, 1.0])

    add_input_file_annotation(fig, args.data_file)

    # Save figure
    pdf_path = output_dir / "ApproachQJ_vs_ApproachQP_scatter.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure to {pdf_path}")

    # Save scatter data to CSV
    csv_path = output_dir / "ApproachQJ_vs_ApproachQP_scatter_data.csv"
    with open(csv_path, 'w') as f:
        # k(t) section
        f.write("# k(t) scatter data\n")
        f.write("year,k_ApproachQJ,k_ApproachQP\n")
        for i, yr in enumerate(unique_years):
            f.write(f"{yr},{k0_values[i]:.8f},{k5c_values[i]:.8f}\n")
        f.write("\n")

        # j coefficients section
        f.write("# j coefficient scatter data\n")
        f.write("country_idx,iso,j0_ApproachQJ,j0_ApproachQP_pred,"
                "j1_ApproachQJ,j1_ApproachQP_pred,"
                "j2_ApproachQJ,j2_ApproachQP_pred\n")
        for c in countries:
            iso = data.idx_to_iso[c]
            f.write(f"{c},{iso},"
                    f"{j0_actual[c]:.8f},{j0_pred[c]:.8f},"
                    f"{j1_actual[c]:.8f},{j1_pred[c]:.8f},"
                    f"{j2_actual[c]:.8f},{j2_pred[c]:.8f}\n")
    print(f"Saved scatter data to {csv_path}")

    # Compute statistics for each parameter
    param_stats = []
    for label, latex_label, x, y in [
        ("k(t)", r"$k(t)$", k5c_values, k0_values),
        ("j0", r"$j_{0,i}$", j0_pred_arr, j0_act_arr),
        ("j1", r"$j_{1,i}$", j1_pred_arr, j1_act_arr),
        ("j2", r"$j_{2,i}$", j2_pred_arr, j2_act_arr),
    ]:
        # Use scipy linregress for slope, intercept, and standard errors
        result = stats.linregress(x, y)
        slope = result.slope
        intercept = result.intercept
        slope_se = result.stderr
        intercept_se = result.intercept_stderr

        # p-value for slope ≠ 1 (t-test)
        n = len(x)
        t_slope = (slope - 1.0) / slope_se
        p_slope = 2 * stats.t.sf(abs(t_slope), df=n - 2)

        # p-value for intercept ≠ 0 (already provided by linregress for intercept ≠ 0)
        t_intercept = intercept / intercept_se
        p_intercept = 2 * stats.t.sf(abs(t_intercept), df=n - 2)

        # R² relative to 1:1 line
        ss_res = np.sum((y - x)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        corr = np.corrcoef(x, y)[0, 1]
        param_stats.append({
            'label': label,
            'latex_label': latex_label,
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'corr': corr,
            'p_slope': p_slope,
            'p_intercept': p_intercept,
        })

    # Print summary statistics
    print("\n--- Summary ---")
    for ps in param_stats:
        print(f"  {ps['label']:10s}  slope={ps['slope']:.4f}  R^2={ps['r2']:.6f}  "
              f"r={ps['corr']:.6f}  p(slope≠1)={ps['p_slope']:.2e}")

    # Generate LaTeX table
    def format_pvalue(p):
        """Format p-value for LaTeX."""
        if p < 0.001:
            return f"{p:.1e}".replace("e-0", "e-").replace("e-", r"\times 10^{-") + "}"
        else:
            return f"{p:.3f}"

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Parameter & Slope & Intercept & $R^2$ & $r$ & $p(\mathrm{slope} \neq 1)$ \\",
        r"\midrule",
    ]
    for ps in param_stats:
        sign = '+' if ps['intercept'] >= 0 else '-'
        abs_int = abs(ps['intercept'])
        p_str = format_pvalue(ps['p_slope'])
        latex_lines.append(
            f"{ps['latex_label']:13s} & {ps['slope']:.4f} & ${sign}{abs_int:.4f}$ "
            f"& {ps['r2']:.4f} & {ps['corr']:.4f} & ${p_str}$ \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Best-fit regression of ApproachQJ values (y-axis) vs ApproachQP predictions (x-axis). "
        r"Slopes near~1 and intercepts near~0 confirm consistency. "
        r"$R^2$ is computed relative to the 1:1 line, not the best-fit line. "
        r"$p(\mathrm{slope} \neq 1)$ tests whether the slope differs significantly from unity.}",
        r"\label{tab:ApproachQJ_vs_ApproachQP}",
        r"\end{table}",
    ])
    latex_table = '\n'.join(latex_lines)

    # Save LaTeX table
    tex_path = output_dir / "ApproachQJ_vs_ApproachQP_table.tex"
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {tex_path}")

    # Also print to console
    print("\n--- LaTeX Table ---")
    print(latex_table)


if __name__ == "__main__":
    main()
