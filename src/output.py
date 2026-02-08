"""Output and visualization for detrended response analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict
from scipy.special import erf
from .data_loader import AnalysisData
from .detrending import CountryTrends
from .fitting import FitResult

# Import for type hints - bootstrap module imported at end to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bootstrap import BootstrapResult

# ==============================================================================
# Constants
# ==============================================================================

# Number of points for temperature response plots
TEMPERATURE_PLOT_POINTS = 200

# Number of bins for histogram plots
HISTOGRAM_BINS = 30

# Z-score for 95% confidence interval
CONFIDENCE_Z_SCORE = 1.96

# Color scheme for approaches (degree of detrending)
# - Conjoined OLS fit: black
# - Mixed (linear T + quadratic GDP): red
# - Linear: green
# - Quadratic: blue
# - GDP-dependent: purple
# - LOESS: orange/brown
APPROACH_COLORS = {
    'approach0': 'black',
    'approach1': 'red',
    'approach2': 'green',
    'approach3': 'blue',
    'approach4': 'blue',
    'approach5': 'blue',
    'approach5a': 'green',
    'approach5b': 'blue',
    'approach5c': 'red',
    'approach5d': 'purple',
    'approach6': 'orange',
    'approach7': 'brown',
    'approach8': 'magenta',
    'nocr0': 'gray',
    'nocr5': 'gray',
}

# Line style scheme for approaches
# - Conjoined OLS or combined (both): solid
# - GDP growth detrending only: dashed
# - Temperature detrending only: dotted
# - Precomputed k approaches: dash-dot
# - LOESS approaches: densely dashed
APPROACH_LINESTYLES = {
    'approach0': '-',
    'approach1': '-',
    'approach2': ':',
    'approach3': '--',
    'approach4': '-',
    'approach5': '-.',
    'approach5a': '-.',
    'approach5b': '-.',
    'approach5c': '-.',
    'approach5d': '-.',
    'approach6': (0, (5, 1)),   # densely dashed
    'approach7': (0, (5, 1)),   # densely dashed
    'approach8': (0, (5, 1)),   # densely dashed
    'nocr0': '--',
    'nocr5': ':',
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_valid_bootstrap_samples(
    result: "BootstrapResult"
) -> tuple:
    """Extract valid (non-NaN) bootstrap samples for h1 and h2.

    Args:
        result: BootstrapResult containing h1_samples and h2_samples

    Returns:
        Tuple of (h1_valid, h2_valid, valid_mask) where valid_mask is a boolean
        array indicating which samples are valid.
    """
    valid_mask = ~np.isnan(result.h1_samples) & ~np.isnan(result.h2_samples)
    return result.h1_samples[valid_mask], result.h2_samples[valid_mask], valid_mask


def is_piecewise_result(result) -> bool:
    """Check if result is from piecewise quadratic model (approach 8).

    Piecewise results have T_opt, h2_low, and h2_high as primary parameters and h1=0.
    """
    return hasattr(result, 'T_opt') and hasattr(result, 'h2_low') and hasattr(result, 'h2_high')


# Keep backward compatibility aliases
is_skewnorm_result = is_piecewise_result
is_gaussian_result = is_piecewise_result


def piecewise_quad_shape(T: np.ndarray, T_opt: float) -> tuple:
    """Compute piecewise quadratic shape.

    Returns two arrays: one for T <= T_opt contribution, one for T > T_opt.

    Args:
        T: Temperature array
        T_opt: Optimal temperature (breakpoint)

    Returns:
        Tuple of (low_component, high_component)
    """
    low_component = np.where(T <= T_opt, (T - T_opt) ** 2, 0.0)
    high_component = np.where(T > T_opt, (T - T_opt) ** 2, 0.0)
    return low_component, high_component


def compute_h_response(T: np.ndarray, result) -> np.ndarray:
    """Compute h(T) - h(T_opt) for any approach type.

    For quadratic model: h(T) = h1*T + h2*T², so h(T) - h(T_opt) = ...
    For piecewise model: h(T) - h(T_opt) = h2_low*(T-T_opt)² for T≤T_opt, h2_high*(T-T_opt)² for T>T_opt
        Since h(T_opt) = 0, h(T) - h(T_opt) = h(T)

    Args:
        T: Temperature array
        result: FitResult or similar with h1, h2, T_optimal (and h2_low, h2_high, T_opt for piecewise)

    Returns:
        Array of h(T) - h(T_opt) values
    """
    if is_piecewise_result(result):
        # Piecewise quadratic: h(T_opt) = 0, so h(T) - h(T_opt) = h(T)
        T_opt = result.T_opt
        h2_low = result.h2_low
        h2_high = result.h2_high
        low_comp, high_comp = piecewise_quad_shape(T, T_opt)
        return h2_low * low_comp + h2_high * high_comp
    else:
        # Quadratic: h(T) = h1*T + h2*T²
        h1, h2 = result.h1, result.h2
        h_T = h1 * T + h2 * T ** 2
        # h(T_opt) at optimal temperature
        if not np.isnan(result.T_optimal) and h2 != 0:
            h_T_opt = -h1 ** 2 / (4 * h2)
        else:
            h_T_opt = 0
        return h_T - h_T_opt


def compute_dh_dT(T: np.ndarray, result) -> np.ndarray:
    """Compute dh/dT for any approach type.

    For quadratic model: dh/dT = h1 + 2*h2*T
    For piecewise model: dh/dT = 2*h2_low*(T-T_opt) for T≤T_opt, 2*h2_high*(T-T_opt) for T>T_opt

    Args:
        T: Temperature array
        result: FitResult or similar with h1, h2 (and h2_low, h2_high, T_opt for piecewise)

    Returns:
        Array of dh/dT values
    """
    if is_piecewise_result(result):
        # Piecewise quadratic derivative
        T_opt = result.T_opt
        h2_low = result.h2_low
        h2_high = result.h2_high

        # dh/dT = 2*h2*(T - T_opt) with appropriate h2 based on which side of T_opt
        return np.where(
            T <= T_opt,
            2 * h2_low * (T - T_opt),
            2 * h2_high * (T - T_opt)
        )
    else:
        # Quadratic derivative: h1 + 2*h2*T
        return result.h1 + 2 * result.h2 * T


def create_output_dir(base_dir: str = "data/output") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def add_input_file_annotation(fig, input_file: str = None) -> None:
    """Add input filename annotation to the lower right corner of a figure.

    Args:
        fig: matplotlib Figure object
        input_file: Path to input data file (if None, no annotation is added)
    """
    if input_file:
        # Get just the filename, not the full path
        filename = Path(input_file).name
        fig.text(0.99, 0.01, f"Data: {filename}",
                 fontsize=6, color='gray', alpha=0.7,
                 ha='right', va='bottom',
                 transform=fig.transFigure)


def save_summary_table(
    results: Dict[str, FitResult], output_dir: Path, input_file: str = None
) -> None:
    """Save comparison table of all approaches."""
    rows = []
    for name, result in results.items():
        row = {
            'Approach': result.approach,
            'h1': result.h1,
            'h1_SE': result.h1_se,
            'h2': result.h2,
            'h2_SE': result.h2_se,
            'T_optimal': result.T_optimal,
            'R_squared': result.r_squared,
            'Total_R_squared': result.total_r_squared,
            'Adj_R_squared': result.adj_r_squared,
            'RMSE': result.rmse,
            'RMS_Imbalance': result.rms_imbalance,
            'RMS_h_T': result.rms_h,
            'Imbalance_Ratio': result.imbalance_ratio,
            'n_obs': result.n_obs,
            'n_params': result.n_params,
        }
        # Add beta for Approach 7
        if hasattr(result, 'beta') and result.beta is not None:
            row['beta'] = result.beta
            row['beta_SE'] = result.beta_se
        # Add Y_ref for Approach 7
        if hasattr(result, 'Y_ref'):
            row['Y_ref'] = result.Y_ref
        # Add T_opt, h2_low, h2_high for Approach 8 (piecewise quadratic)
        if hasattr(result, 'T_opt') and hasattr(result, 'h2_low'):
            row['T_opt'] = result.T_opt
            row['T_opt_SE'] = result.T_opt_se
            row['h2_low'] = result.h2_low
            row['h2_low_SE'] = result.h2_low_se
            row['h2_high'] = result.h2_high
            row['h2_high_SE'] = result.h2_high_se
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add input file as first row comment in CSV
    csv_path = output_dir / 'comparison_table.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)

    # Build variance decomposition DataFrame
    # Decomposition: dy = [h(T)-h(Ttr)] + h(Ttr) + j + k + err
    # All terms divided by Var(dy) so they sum to 1.0
    vd_rows = []
    for name, result in results.items():
        va = result.var_attrib
        if va is None:
            continue
        var_dy = va['var_dy']
        if var_dy <= 0:
            continue
        vd_row = {
            'Approach': result.approach,
            'Var(dy)': var_dy,
            'Var(h(T)-h(Ttr))/Var(dy)': va['Sigma_Delta_u_Delta_u'] / var_dy,
            'Var(h(Ttr))/Var(dy)': va['Sigma_v_v'] / var_dy,
            'Var(j)/Var(dy)': va['Sigma_j_j'] / var_dy,
            'Var(k)/Var(dy)': va['Sigma_k_k'] / var_dy,
            'Var(err)/Var(dy)': va['Sigma_epsilon_epsilon'] / var_dy,
            '2Cov(h(T)-h(Ttr),h(Ttr))/Var(dy)': 2 * va['Sigma_Delta_u_v'] / var_dy,
            '2Cov(h(T)-h(Ttr),j)/Var(dy)': 2 * va['Sigma_Delta_u_j'] / var_dy,
            '2Cov(h(T)-h(Ttr),k)/Var(dy)': 2 * va['Sigma_Delta_u_k'] / var_dy,
            '2Cov(h(T)-h(Ttr),err)/Var(dy)': 2 * va['Sigma_Delta_u_epsilon'] / var_dy,
            '2Cov(h(Ttr),j)/Var(dy)': 2 * va['Sigma_v_j'] / var_dy,
            '2Cov(h(Ttr),k)/Var(dy)': 2 * va['Sigma_v_k'] / var_dy,
            '2Cov(h(Ttr),err)/Var(dy)': 2 * va['Sigma_v_epsilon'] / var_dy,
            '2Cov(j,k)/Var(dy)': 2 * va['Sigma_j_k'] / var_dy,
            '2Cov(j,err)/Var(dy)': 2 * va['Sigma_j_epsilon'] / var_dy,
            '2Cov(k,err)/Var(dy)': 2 * va['Sigma_k_epsilon'] / var_dy,
        }
        # Sum check: all variance and covariance fractions should sum to 1.0
        frac_sum = sum(val for key, val in vd_row.items()
                       if key not in ('Approach', 'Var(dy)'))
        vd_row['Sum'] = frac_sum
        vd_rows.append(vd_row)

    vd_df = pd.DataFrame(vd_rows)

    # For Excel, add input file info in a header row
    xlsx_path = output_dir / 'comparison_table.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Write header info
        if input_file:
            header_df = pd.DataFrame({'Input Data': [Path(input_file).name]})
            header_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0)
            df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=2)
        else:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        # Write variance decomposition on separate sheet
        if len(vd_rows) > 0:
            vd_df.to_excel(writer, sheet_name='Variance Decomposition', index=False)

    # Also save as formatted text
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("Detrended Response Analysis - Comparison of Approaches\n")
        f.write("=" * 70 + "\n")
        if input_file:
            f.write(f"Input data: {Path(input_file).name}\n")
        f.write("=" * 70 + "\n\n")

        for name, result in results.items():
            f.write(f"{result.approach}\n")
            f.write("-" * 50 + "\n")
            # Special handling for Approach 8 (piecewise quadratic)
            if hasattr(result, 'h2_low') and hasattr(result, 'h2_high'):
                f.write(f"  h2_low = {result.h2_low:.6f}  (SE: {result.h2_low_se:.6f})\n")
                f.write(f"  h2_high = {result.h2_high:.6f}  (SE: {result.h2_high_se:.6f})\n")
                T_opt_se = result.T_opt_se if not np.isnan(result.T_opt_se) else 0.0
                f.write(f"  T_opt = {result.T_opt:.4f}  (SE: {T_opt_se:.4f})\n")
            else:
                f.write(f"  h1 = {result.h1:12.6f}  (SE: {result.h1_se:.6f})\n")
                f.write(f"  h2 = {result.h2:12.6f}  (SE: {result.h2_se:.6f})\n")
                # Add beta for Approach 7
                if hasattr(result, 'beta') and result.beta is not None:
                    f.write(f"  beta = {result.beta:10.4f}  (SE: {result.beta_se:.4f})\n")
                    if hasattr(result, 'Y_ref'):
                        f.write(f"  Y_ref = {result.Y_ref:.2f}\n")
            if np.isnan(result.T_optimal):
                f.write(f"  T_optimal = N/A\n")
            else:
                f.write(f"  T_optimal = {result.T_optimal:.2f} C\n")
            f.write(f"  R² = {result.r_squared:.4f}\n")
            f.write(f"  Total R² = {result.total_r_squared:.4f}\n")
            f.write(f"  Adjusted R² = {result.adj_r_squared:.4f}\n")
            f.write(f"  RMSE = {result.rmse:.6f}\n")
            if result.rms_imbalance is not None:
                f.write(f"  RMS Imbalance = {result.rms_imbalance:.6f}\n")
            if result.rms_h is not None:
                f.write(f"  RMS h(T) = {result.rms_h:.6f}\n")
            if result.imbalance_ratio is not None:
                f.write(f"  Imbalance Ratio = {result.imbalance_ratio:.4f}\n")
            f.write(f"  Observations: {result.n_obs}\n")
            f.write(f"  Parameters: {result.n_params}\n")
            # Key variance ratios
            if result.var_decomp is not None:
                vd = result.var_decomp
                f.write(f"  Key Variance Ratios (Var(component) / Var(dy)):\n")
                if 'var_ratio_h_T' in vd:
                    f.write(f"    h(T) total:    {vd['var_ratio_h_T']:.4f}   (climate response, full temperature)\n")
                if 'var_ratio_h_Tstar' in vd:
                    f.write(f"    h(T*) detrend: {vd['var_ratio_h_Tstar']:.4f}   (climate response, detrended temperature)\n")
                if 'var_ratio_j' in vd:
                    f.write(f"    j (trends):    {vd['var_ratio_j']:.4f}   (country growth trends)\n")
                if 'var_ratio_k' in vd:
                    f.write(f"    k (year FE):   {vd['var_ratio_k']:.4f}   (year fixed effects)\n")
                if 'var_ratio_cross' in vd:
                    f.write(f"    cross terms:   {vd['var_ratio_cross']:.4f}   (covariance remainder)\n")
                if all(k in vd for k in ('var_ratio_h_T', 'var_ratio_j', 'var_ratio_k', 'var_ratio_cross')):
                    ratio_sum = vd['var_ratio_h_T'] + vd['var_ratio_j'] + vd['var_ratio_k'] + vd['var_ratio_cross']
                    f.write(f"    sum:           {ratio_sum:.4f}   (= Total R²)\n")
            # Detailed variance decomposition
            if result.var_decomp is not None:
                vd = result.var_decomp
                comp_names = vd.get('component_names', [])
                f.write(f"  Detailed Variance Decomposition:\n")
                f.write(f"    RMS dy = {vd.get('rms_dy', 0):.6f}\n")
                # RMS of each component
                for cn in comp_names:
                    rms_key = f'rms_{cn}'
                    if rms_key in vd:
                        f.write(f"    RMS {cn} = {vd[rms_key]:.6f}\n")
                # Variance fractions
                for cn in comp_names:
                    var_key = f'var_{cn}'
                    if var_key in vd:
                        f.write(f"    Var frac {cn} = {vd[var_key]:.4f}\n")
                # Covariance fractions
                for i in range(len(comp_names)):
                    for j in range(i + 1, len(comp_names)):
                        cov_key = f'cov_{comp_names[i]}_{comp_names[j]}'
                        if cov_key in vd:
                            f.write(f"    Cov frac {comp_names[i]}-{comp_names[j]} = {vd[cov_key]:.4f}\n")
                f.write(f"    Sum check = {vd.get('sum_check', 0):.6f}\n")
            # Variance attribution (5-component with covariance allocation)
            if result.var_attrib is not None:
                va = result.var_attrib
                f.write(f"  Variance Attribution (5-component with covariance allocation):\n")
                f.write(f"    Identity: Δy = Δu + v + j + k + ε  (exact, in-sample)\n")
                f.write(f"    Components:\n")
                f.write(f"      Δu = h(T) - h(T_trend)  [increment from actual vs trended T]\n")
                f.write(f"      v  = h(T_trend)         [baseline climate at trended T]\n")
                f.write(f"      j  = j_i(t)             [country growth trends]\n")
                f.write(f"      k  = k(t)               [time fixed effects]\n")
                f.write(f"      ε  = residuals          [unexplained variation]\n")
                f.write(f"    Var(Δy) = {va.get('var_dy', 0):.6f}\n")
                f.write(f"    Contributions (C_x = row-sum of covariance matrix Σ):\n")
                f.write(f"      C_Δu = {va.get('C_Delta_u', 0):+.6f}   (share: {va.get('s_Delta_u', 0):+.4f})\n")
                f.write(f"      C_v  = {va.get('C_v', 0):+.6f}   (share: {va.get('s_v', 0):+.4f})\n")
                f.write(f"      C_j  = {va.get('C_j', 0):+.6f}   (share: {va.get('s_j', 0):+.4f})\n")
                f.write(f"      C_k  = {va.get('C_k', 0):+.6f}   (share: {va.get('s_k', 0):+.4f})\n")
                f.write(f"      C_ε  = {va.get('C_epsilon', 0):+.6f}   (share: {va.get('s_epsilon', 0):+.4f})\n")
                f.write(f"      Sum  = {va.get('sum_check', 0):.6f}   (should equal Var(Δy) exactly)\n")
                f.write(f"    Orthogonality checks (residual vs fitted, should be ~0 for OLS):\n")
                f.write(f"      Cov(ε, Δu) = {va.get('cov_epsilon_Delta_u', 0):+.6e}\n")
                f.write(f"      Cov(ε, v)  = {va.get('cov_epsilon_v', 0):+.6e}\n")
                f.write(f"      Cov(ε, j)  = {va.get('cov_epsilon_j', 0):+.6e}\n")
                f.write(f"      Cov(ε, k)  = {va.get('cov_epsilon_k', 0):+.6e}\n")
                f.write(f"    Covariance matrix Σ (all 15 unique entries, ddof=0 for exact sum):\n")
                f.write(f"      Σ[Δu,Δu] = {va.get('Sigma_Delta_u_Delta_u', 0):+.6f}\n")
                f.write(f"      Σ[Δu,v]  = {va.get('Sigma_Delta_u_v', 0):+.6f}\n")
                f.write(f"      Σ[Δu,j]  = {va.get('Sigma_Delta_u_j', 0):+.6f}\n")
                f.write(f"      Σ[Δu,k]  = {va.get('Sigma_Delta_u_k', 0):+.6f}\n")
                f.write(f"      Σ[Δu,ε]  = {va.get('Sigma_Delta_u_epsilon', 0):+.6f}\n")
                f.write(f"      Σ[v,v]   = {va.get('Sigma_v_v', 0):+.6f}\n")
                f.write(f"      Σ[v,j]   = {va.get('Sigma_v_j', 0):+.6f}\n")
                f.write(f"      Σ[v,k]   = {va.get('Sigma_v_k', 0):+.6f}\n")
                f.write(f"      Σ[v,ε]   = {va.get('Sigma_v_epsilon', 0):+.6f}\n")
                f.write(f"      Σ[j,j]   = {va.get('Sigma_j_j', 0):+.6f}\n")
                f.write(f"      Σ[j,k]   = {va.get('Sigma_j_k', 0):+.6f}\n")
                f.write(f"      Σ[j,ε]   = {va.get('Sigma_j_epsilon', 0):+.6f}\n")
                f.write(f"      Σ[k,k]   = {va.get('Sigma_k_k', 0):+.6f}\n")
                f.write(f"      Σ[k,ε]   = {va.get('Sigma_k_epsilon', 0):+.6f}\n")
                f.write(f"      Σ[ε,ε]   = {va.get('Sigma_epsilon_epsilon', 0):+.6f}\n")
            f.write("\n")


def save_country_trends(
    data: AnalysisData, trends: CountryTrends, output_dir: Path,
    input_file: str = None
) -> None:
    """Save country-level trend coefficients."""
    rows = []
    for i in range(data.n_countries):
        iso = data.idx_to_iso[i]
        rows.append({
            'iso_id': iso,
            'T0': trends.T0[i],
            'T1': trends.T1[i],
            'y0': trends.y0[i],
            'y1': trends.y1[i],
            'y2': trends.y2[i],
        })

    df = pd.DataFrame(rows)

    # Add input file as first row comment in CSV
    csv_path = output_dir / 'country_trends.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)

    # For Excel, add input file info in a header row
    xlsx_path = output_dir / 'country_trends.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        if input_file:
            header_df = pd.DataFrame({'Input Data': [Path(input_file).name]})
            header_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0)
            df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=2)
        else:
            df.to_excel(writer, sheet_name='Sheet1', index=False)


def _plot_temperature_response_subset(
    results: Dict[str, FitResult], output_dir: Path,
    approaches: list, filename: str, title_suffix: str = "",
    T_range: tuple = (0, 30), input_file: str = None
) -> None:
    """Plot h(T) - h(T*) for a subset of approaches.

    This shows the temperature response relative to the optimal temperature,
    so the maximum is at y=0 for each curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.linspace(T_range[0], T_range[1], TEMPERATURE_PLOT_POINTS)

    for name in approaches:
        if name not in results:
            continue
        r = results[name]

        # Use helper function that handles both quadratic and power-law models
        h_relative = compute_h_response(T, r)

        label = f"{r.approach} (T_opt = {r.T_optimal:.1f}°C)"
        ax.plot(T, h_relative, color=APPROACH_COLORS.get(name, 'gray'),
                linestyle=APPROACH_LINESTYLES.get(name, '-'), label=label, linewidth=2)

        # Mark optimal temperature
        ax.axvline(r.T_optimal, color=APPROACH_COLORS.get(name, 'gray'),
                   linestyle=':', alpha=0.5)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('h(T) - h(T_opt)', fontsize=12)
    title = 'Temperature Response Relative to Optimum'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename)
    plt.close()


def plot_temperature_response(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30),
    input_file: str = None
) -> None:
    """Plot h(T) - h(T*) for approaches, generating three separate plots."""
    # Plot 1: Approaches 0-4 (basic approaches)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3', 'approach4'],
        filename='temperature_response_all.pdf',
        title_suffix='Approaches 0-4',
        T_range=T_range,
        input_file=input_file
    )
    # Plot 2: Approaches 0, 5, 5a, 5b, 5c (precomputed k approaches)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach0', 'approach5', 'approach5a', 'approach5b', 'approach5c'],
        filename='temperature_response_precomputed_k.pdf',
        title_suffix='Precomputed k Variants',
        T_range=T_range,
        input_file=input_file
    )
    # Plot 3: Quadratic vs LOESS comparison (5 vs 6, 7, 8)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach5', 'approach6', 'approach7', 'approach8'],
        filename='temperature_response_loess.pdf',
        title_suffix='Quadratic vs LOESS',
        T_range=T_range,
        input_file=input_file
    )


def _plot_temperature_derivative_subset(
    results: Dict[str, FitResult], output_dir: Path,
    approaches: list, filename: str, title_suffix: str = "",
    T_range: tuple = (0, 30), input_file: str = None
) -> None:
    """Plot dh/dT for a subset of approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.linspace(T_range[0], T_range[1], TEMPERATURE_PLOT_POINTS)

    for name in approaches:
        if name not in results:
            continue
        r = results[name]
        # Use helper function that handles both quadratic and power-law models
        dh_dT = compute_dh_dT(T, r)
        label = f"{r.approach}"
        ax.plot(T, dh_dT, color=APPROACH_COLORS.get(name, 'gray'),
                linestyle=APPROACH_LINESTYLES.get(name, '-'), label=label, linewidth=2)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('dh/dT', fontsize=12)
    title = 'Temperature Derivative by Approach'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename)
    plt.close()


def plot_temperature_derivative(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30),
    input_file: str = None
) -> None:
    """Plot dh/dT for approaches, generating three separate plots."""
    # Plot 1: Approaches 0-4 (basic approaches)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3', 'approach4'],
        filename='temperature_derivative_all.pdf',
        title_suffix='Approaches 0-4',
        T_range=T_range,
        input_file=input_file
    )
    # Plot 2: Approaches 0, 5, 5a, 5b, 5c (precomputed k approaches)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach0', 'approach5', 'approach5a', 'approach5b', 'approach5c'],
        filename='temperature_derivative_precomputed_k.pdf',
        title_suffix='Precomputed k Variants',
        T_range=T_range,
        input_file=input_file
    )
    # Plot 3: Quadratic vs LOESS comparison (5 vs 6, 7, 8)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach5', 'approach6', 'approach7', 'approach8'],
        filename='temperature_derivative_loess.pdf',
        title_suffix='Quadratic vs LOESS',
        T_range=T_range,
        input_file=input_file
    )


def plot_coefficient_comparison(
    results: Dict[str, FitResult], output_dir: Path, input_file: str = None
) -> None:
    """Plot T_opt and h2 coefficients for each approach."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    approaches = list(results.keys())
    labels = [results[a].approach for a in approaches]

    # T_optimal values — filter out NaN (no climate response approaches)
    valid_t = [(a, results[a]) for a in approaches if not np.isnan(results[a].T_optimal)]
    t_labels = [r.approach for _, r in valid_t]
    T_opt_vals = [r.T_optimal for _, r in valid_t]
    x_t = np.arange(len(valid_t))

    axes[0].bar(x_t, T_opt_vals, color='steelblue', alpha=0.7)
    axes[0].set_xticks(x_t)
    axes[0].set_xticklabels(t_labels, rotation=45, ha='right')
    axes[0].set_ylabel('Optimal Temperature (°C)')
    axes[0].set_title('Optimal Temperature (T_opt)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, val in enumerate(T_opt_vals):
        axes[0].text(i, val + 0.3, f'{val:.1f}°C', ha='center', va='bottom', fontsize=9)

    # h2 coefficients
    x = np.arange(len(approaches))
    h2_vals = [results[a].h2 for a in approaches]
    h2_errs = [results[a].h2_se * CONFIDENCE_Z_SCORE for a in approaches]  # 95% CI

    axes[1].bar(x, h2_vals, yerr=h2_errs, capsize=5, color='coral', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('h₂ coefficient')
    axes[1].set_title('Quadratic Temperature Coefficient (h₂)')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'coefficient_comparison.pdf')
    plt.close()


def plot_optimal_temperature_comparison(
    results: Dict[str, FitResult], output_dir: Path, input_file: str = None
) -> None:
    """Plot optimal temperature comparison across approaches."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter out approaches with NaN T_optimal (no climate response)
    valid = [(a, results[a]) for a in results.keys() if not np.isnan(results[a].T_optimal)]
    labels = [r.approach for _, r in valid]
    T_opt = [r.T_optimal for _, r in valid]
    valid_keys = [a for a, _ in valid]

    colors = [APPROACH_COLORS.get(a, 'gray') for a in valid_keys]
    x = np.arange(len(valid))

    bars = ax.bar(x, T_opt, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optimal Temperature (°C)')
    ax.set_title('Optimal Temperature by Approach')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, T_opt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}°C', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'optimal_temperature_comparison.pdf')
    plt.close()


def plot_year_effects(
    results: Dict[str, FitResult], data: AnalysisData, output_dir: Path,
    input_file: str = None
) -> None:
    """Plot year fixed effects k(t) for all approaches.

    All approaches now use year fixed effects k_t.

    For Approach 0 (Conjoined OLS fit), we subtract a least-squares best-fit quadratic
    from k_t. This shows what the year effects would look like if the quadratic
    trend were absorbed into the country-specific j_i(t) terms. The subtracted
    quadratic is what would be added to all j_i(t) under an alternative
    identifiability constraint.
    """
    # Get unique years from data
    unique_years = sorted(set(data.year))
    years_array = np.array(unique_years)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot year means k[t] used by approaches 5, 6, 7, 8 as a heavy line
    for name in ('approach5', 'approach6', 'approach7', 'approach8'):
        if name in results:
            k_year_means = np.array([results[name].k[yr] for yr in unique_years])
            ax.plot(unique_years, k_year_means, color='black', linestyle='-', linewidth=3,
                    label='Year means k[t] (Approaches 5, 6, 7, 8)')
            break  # Only plot once since all three share the same k

    for name, result in results.items():
        # Skip approaches that use the same k values as approach5 (already plotted above)
        if name in ('approach5', 'approach5a', 'approach5b', 'approach5c', 'approach5d', 'approach6', 'approach7', 'approach8'):
            continue

        # k is stored with actual year as key
        k_values = np.array([result.k[yr] for yr in unique_years])

        if name == 'approach0':
            # For Approach 0, subtract least-squares best-fit quadratic
            # Fit quadratic: k(t) = a + b*t + c*t^2
            # Use normalized time for numerical stability
            t_normalized = years_array - years_array[0]
            A = np.column_stack([
                np.ones(len(t_normalized)),
                t_normalized,
                t_normalized ** 2
            ])
            coeffs, _, _, _ = np.linalg.lstsq(A, k_values, rcond=None)
            quadratic_fit = A @ coeffs
            k_values_plot = k_values - quadratic_fit
            label = "Conjoined OLS Fit (minus best-fit quadratic)"
        else:
            k_values_plot = k_values
            label = f"{result.approach}"

        ax.plot(unique_years, k_values_plot, color=APPROACH_COLORS.get(name, 'gray'),
                linestyle=APPROACH_LINESTYLES.get(name, '-'), linewidth=1.5,
                label=label)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('k(t) - Year Fixed Effect', fontsize=12)
    ax.set_title('Year Fixed Effects by Approach', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'year_effects.pdf')
    plt.close()


def plot_residual_diagnostics(
    results: Dict[str, FitResult], data: AnalysisData, output_dir: Path,
    input_file: str = None
) -> None:
    """Plot residual diagnostics for each approach."""
    for name, result in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        residuals = result.residuals

        # Histogram of residuals
        axes[0, 0].hist(residuals, bins=HISTOGRAM_BINS, density=True, alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].axvline(0, color='red', linestyle='--')

        # Residuals vs fitted
        fitted = data.growth_pcGDP - residuals
        axes[0, 1].scatter(fitted, residuals, alpha=0.3, s=1)
        axes[0, 1].axhline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Fitted')

        # Residuals vs temperature
        axes[1, 0].scatter(data.temp, residuals, alpha=0.3, s=1)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Temperature')

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')

        fig.suptitle(f'Residual Diagnostics: {result.approach}', fontsize=14)
        plt.tight_layout()
        add_input_file_annotation(fig, input_file)

        # Safe filename
        safe_name = name.replace(' ', '_').lower()
        plt.savefig(output_dir / f'residuals_{safe_name}.pdf')
        plt.close()


def plot_gdp_scaling_factor(
    results: Dict[str, FitResult],
    output_dir: Path,
    data: AnalysisData = None,
    Y_range: tuple = None,
    input_file: str = None,
) -> None:
    """Plot the GDP scaling factor (Y/Y_ref)^(-beta) for Approach 7.

    This shows how the temperature response is scaled by per capita GDP level.
    Countries with lower GDP have larger scaling factors (more affected).
    Creates a two-panel figure when both approaches are present.

    Args:
        results: Dictionary of FitResult objects
        output_dir: Output directory
        data: AnalysisData for adding GDP histogram (optional)
        Y_range: GDP range for x-axis (default: from data min to max)
        input_file: Path to input data file (for annotation)
    """
    # Collect panels to plot
    panels = []
    for key, title, color in [
        ('approach7', 'GDP-Response LOESS (Approach 7)', 'brown'),
    ]:
        if key in results:
            r = results[key]
            if hasattr(r, 'beta') and hasattr(r, 'Y_ref'):
                panels.append((r, title, color))

    if not panels:
        return

    # Default Y range
    if Y_range is None:
        Y_range = (500, 100000)

    # Create GDP array (log-spaced for better visualization)
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 6), squeeze=False)

    def _draw_panel(ax, r, title, color):
        beta = r.beta
        Y_ref = r.Y_ref
        g = (Y / Y_ref) ** (-beta)

        # Add GDP histogram on secondary y-axis (if data provided)
        if data is not None:
            max_year = data.year_range[1]
            mask_recent = data.year == max_year
            gdp_recent = data.pcGDP[mask_recent]

            ax2 = ax.twinx()
            bins = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 30)
            ax2.hist(gdp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel(f'Data density ({max_year})', fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            ax2.set_ylim(bottom=0)
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        ax.plot(Y, g, color=color, linewidth=2, label=f'β = {beta:.3f}')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='g = 1 (at Y = Y_ref)')
        ax.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref = ${Y_ref:,.0f}')

        ax.set_xscale('log')
        ax.set_xlabel('Per Capita GDP ($)', fontsize=12)
        ax.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-β)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        ax.annotate('Poorer countries:\nmore affected',
                    xy=(Y_range[0] * 2, g[0] * 0.9),
                    fontsize=10, color='darkred')
        ax.annotate('Richer countries:\nless affected',
                    xy=(Y_range[1] * 0.3, g[-1] * 1.1),
                    fontsize=10, color='darkgreen')

    for i, (r, title, color) in enumerate(panels):
        _draw_panel(axes[0, i], r, title, color)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'gdp_scaling_factor.pdf')
    plt.close()


def save_all_outputs(
    data: AnalysisData,
    trends: CountryTrends,
    results: Dict[str, FitResult],
    output_dir: Path = None,
    input_file: str = None,
) -> Path:
    """Save all outputs to the specified directory.

    Args:
        data: AnalysisData object
        trends: CountryTrends object
        results: Dictionary of FitResult objects
        output_dir: Output directory (created if None)
        input_file: Path to input data file (for annotation)

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = create_output_dir()

    print(f"Saving outputs to: {output_dir}")

    # Save tables
    save_summary_table(results, output_dir, input_file=input_file)
    save_country_trends(data, trends, output_dir, input_file=input_file)

    # Generate plots
    plot_temperature_response(results, output_dir, input_file=input_file)
    plot_temperature_derivative(results, output_dir, input_file=input_file)
    plot_coefficient_comparison(results, output_dir, input_file=input_file)
    plot_optimal_temperature_comparison(results, output_dir, input_file=input_file)
    plot_year_effects(results, data, output_dir, input_file=input_file)
    plot_residual_diagnostics(results, data, output_dir, input_file=input_file)

    # Plot GDP scaling factor for Approach 7
    if 'approach7' in results:
        plot_gdp_scaling_factor(results, output_dir, data=data, input_file=input_file)

    print("All outputs saved.")
    return output_dir


def save_bootstrap_coefficients_csv(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap samples to CSV for each approach.

    Creates: bootstrap_coefficients.csv with columns:
    - iteration
    - approach
    - h1, h2, T_optimal, r_squared, total_r_squared
    - beta (for approaches 8 and 10 where it's a free parameter)
    """
    rows = []
    for name, result in results.items():
        for i in range(result.n_bootstrap):
            row = {
                'iteration': i,
                'approach': name,
                'approach_name': result.approach,
                'h1': result.h1_samples[i],
                'h2': result.h2_samples[i],
                'T_optimal': result.T_optimal_samples[i],
                'r_squared': result.r_squared_samples[i],
                'total_r_squared': result.total_r_squared_samples[i],
            }
            # Add beta for approaches where it's a free parameter (7)
            if result.beta_samples is not None:
                row['beta'] = result.beta_samples[i]
            # Add h2_low and h2_high for Approach 8 (piecewise quadratic)
            if result.h2_low_samples is not None:
                row['h2_low'] = result.h2_low_samples[i]
            if result.h2_high_samples is not None:
                row['h2_high'] = result.h2_high_samples[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'bootstrap_coefficients.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_coefficients.csv ({len(df)} rows)")


def save_bootstrap_summary_txt(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save text summary with confidence intervals.

    For each approach, reports:
    - Point estimate
    - Bootstrap median
    - 90% CI: [5th, 95th percentiles]
    - IQR: [25th, 75th percentiles]
    """
    with open(output_dir / 'bootstrap_summary.txt', 'w') as f:
        f.write("Bootstrap Uncertainty Analysis - Summary\n")
        f.write("=" * 70 + "\n")
        if input_file:
            f.write(f"Input data: {Path(input_file).name}\n")
        f.write("=" * 70 + "\n\n")

        # Write metadata
        first_result = next(iter(results.values()))
        f.write(f"Bootstrap iterations: {first_result.n_bootstrap}\n")
        f.write(f"Successful iterations: {first_result.n_successful}\n\n")

        for name, result in results.items():
            stats = all_stats[name]
            f.write(f"{result.approach}\n")
            f.write("-" * 50 + "\n")

            # T_optimal
            f.write(f"  T_optimal (Optimal Temperature, C):\n")
            f.write(f"    Point estimate:  {result.T_optimal_point:10.2f}\n")
            f.write(f"    Bootstrap median:{stats['T_optimal']['p50']:10.2f}\n")
            f.write(f"    90% CI:          [{stats['T_optimal']['p5']:8.2f}, {stats['T_optimal']['p95']:8.2f}]\n")
            f.write(f"    IQR:             [{stats['T_optimal']['p25']:8.2f}, {stats['T_optimal']['p75']:8.2f}]\n")
            f.write(f"    Std:             {stats['T_optimal']['std']:10.4f}\n")

            # h1
            f.write(f"  h1 (Linear temperature coefficient):\n")
            f.write(f"    Point estimate:  {result.h1_point:10.6f}\n")
            f.write(f"    Bootstrap median:{stats['h1']['p50']:10.6f}\n")
            f.write(f"    90% CI:          [{stats['h1']['p5']:10.6f}, {stats['h1']['p95']:10.6f}]\n")
            f.write(f"    IQR:             [{stats['h1']['p25']:10.6f}, {stats['h1']['p75']:10.6f}]\n")
            f.write(f"    Std:             {stats['h1']['std']:10.6f}\n")

            # h2
            f.write(f"  h2 (Quadratic temperature coefficient):\n")
            f.write(f"    Point estimate:  {result.h2_point:10.6f}\n")
            f.write(f"    Bootstrap median:{stats['h2']['p50']:10.6f}\n")
            f.write(f"    90% CI:          [{stats['h2']['p5']:10.6f}, {stats['h2']['p95']:10.6f}]\n")
            f.write(f"    IQR:             [{stats['h2']['p25']:10.6f}, {stats['h2']['p75']:10.6f}]\n")
            f.write(f"    Std:             {stats['h2']['std']:10.6f}\n")

            # beta (for approaches where it's a free parameter)
            if result.beta_point is not None and 'beta' in stats:
                f.write(f"  beta (GDP scaling exponent):\n")
                f.write(f"    Point estimate:  {result.beta_point:10.4f}\n")
                f.write(f"    Bootstrap median:{stats['beta']['p50']:10.4f}\n")
                f.write(f"    90% CI:          [{stats['beta']['p5']:10.4f}, {stats['beta']['p95']:10.4f}]\n")
                f.write(f"    IQR:             [{stats['beta']['p25']:10.4f}, {stats['beta']['p75']:10.4f}]\n")
                f.write(f"    Std:             {stats['beta']['std']:10.4f}\n")

            # h2_low and h2_high (for Approach 8 piecewise quadratic)
            if result.h2_low_point is not None and 'h2_low' in stats:
                f.write(f"  h2_low (Curvature for T <= T_opt):\n")
                f.write(f"    Point estimate:  {result.h2_low_point:10.6f}\n")
                f.write(f"    Bootstrap median:{stats['h2_low']['p50']:10.6f}\n")
                f.write(f"    90% CI:          [{stats['h2_low']['p5']:10.6f}, {stats['h2_low']['p95']:10.6f}]\n")
                f.write(f"    IQR:             [{stats['h2_low']['p25']:10.6f}, {stats['h2_low']['p75']:10.6f}]\n")
                f.write(f"    Std:             {stats['h2_low']['std']:10.6f}\n")
            if result.h2_high_point is not None and 'h2_high' in stats:
                f.write(f"  h2_high (Curvature for T > T_opt):\n")
                f.write(f"    Point estimate:  {result.h2_high_point:10.6f}\n")
                f.write(f"    Bootstrap median:{stats['h2_high']['p50']:10.6f}\n")
                f.write(f"    90% CI:          [{stats['h2_high']['p5']:10.6f}, {stats['h2_high']['p95']:10.6f}]\n")
                f.write(f"    IQR:             [{stats['h2_high']['p25']:10.6f}, {stats['h2_high']['p75']:10.6f}]\n")
                f.write(f"    Std:             {stats['h2_high']['std']:10.6f}\n")

            # Key variance ratios with bootstrap CIs
            if result.var_decomp_point is not None:
                vd = result.var_decomp_point
                f.write(f"  Key Variance Ratios (Var(component) / Var(dy)):\n")
                for ratio_key, label in [
                    ('var_ratio_h_T', 'h(T) total'),
                    ('var_ratio_h_Tstar', 'h(T*) detrend'),
                    ('var_ratio_j', 'j (trends)'),
                    ('var_ratio_k', 'k (year FE)'),
                    ('var_ratio_cross', 'cross terms'),
                ]:
                    if ratio_key in vd:
                        point_val = vd[ratio_key]
                        stat_key = f'vd_{ratio_key}'
                        if stat_key in stats:
                            f.write(f"    {label:20s} {point_val:8.4f}  "
                                    f"[{stats[stat_key]['p5']:7.4f}, {stats[stat_key]['p95']:7.4f}]\n")
                        else:
                            f.write(f"    {label:20s} {point_val:8.4f}\n")

            # Detailed variance decomposition
            if result.var_decomp_point is not None:
                vd = result.var_decomp_point
                comp_names = vd.get('component_names', [])
                f.write(f"  Detailed Variance Decomposition (sum = 1.0 by construction):\n")
                # Variance fractions
                for cn in comp_names:
                    var_key = f'var_{cn}'
                    stat_key = f'vd_{var_key}'
                    point_val = vd.get(var_key, np.nan)
                    if stat_key in stats:
                        f.write(f"    {var_key:30s} {point_val:8.4f}  "
                                f"[{stats[stat_key]['p5']:7.4f}, {stats[stat_key]['p95']:7.4f}]\n")
                    else:
                        f.write(f"    {var_key:30s} {point_val:8.4f}\n")
                # Covariance fractions
                for i in range(len(comp_names)):
                    for j in range(i + 1, len(comp_names)):
                        cov_key = f'cov_{comp_names[i]}_{comp_names[j]}'
                        stat_key = f'vd_{cov_key}'
                        point_val = vd.get(cov_key, np.nan)
                        if stat_key in stats:
                            f.write(f"    {cov_key:30s} {point_val:8.4f}  "
                                    f"[{stats[stat_key]['p5']:7.4f}, {stats[stat_key]['p95']:7.4f}]\n")
                        else:
                            f.write(f"    {cov_key:30s} {point_val:8.4f}\n")
                f.write(f"    {'sum_check':30s} {vd.get('sum_check', 0):8.6f}\n")

            f.write("\n")

    print(f"  Saved bootstrap_summary.txt")


def save_bootstrap_summary_table(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap summary as CSV and XLSX with approaches as rows.

    Creates: bootstrap_summary_table.csv and bootstrap_summary_table.xlsx

    Each row is an approach, with columns for each parameter's statistics:
    - Point estimate, median, p5, p25, p75, p95, std for h1, h2, T_optimal, total_r_squared
    - Beta statistics included for approaches where it's a free parameter
    """
    rows = []
    for name, result in results.items():
        stats = all_stats[name]

        row = {
            'approach': name,
            'approach_name': result.approach,
            'n_bootstrap': result.n_bootstrap,
            'n_successful': result.n_successful,

            # h1 statistics
            'h1_point': result.h1_point,
            'h1_median': stats['h1']['p50'],
            'h1_p5': stats['h1']['p5'],
            'h1_p25': stats['h1']['p25'],
            'h1_p75': stats['h1']['p75'],
            'h1_p95': stats['h1']['p95'],
            'h1_std': stats['h1']['std'],

            # h2 statistics
            'h2_point': result.h2_point,
            'h2_median': stats['h2']['p50'],
            'h2_p5': stats['h2']['p5'],
            'h2_p25': stats['h2']['p25'],
            'h2_p75': stats['h2']['p75'],
            'h2_p95': stats['h2']['p95'],
            'h2_std': stats['h2']['std'],

            # T_optimal statistics
            'T_optimal_point': result.T_optimal_point,
            'T_optimal_median': stats['T_optimal']['p50'],
            'T_optimal_p5': stats['T_optimal']['p5'],
            'T_optimal_p25': stats['T_optimal']['p25'],
            'T_optimal_p75': stats['T_optimal']['p75'],
            'T_optimal_p95': stats['T_optimal']['p95'],
            'T_optimal_std': stats['T_optimal']['std'],

            # total_r_squared statistics
            'total_r_squared_point': result.total_r_squared_point,
            'total_r_squared_median': stats['total_r_squared']['p50'],
            'total_r_squared_p5': stats['total_r_squared']['p5'],
            'total_r_squared_p25': stats['total_r_squared']['p25'],
            'total_r_squared_p75': stats['total_r_squared']['p75'],
            'total_r_squared_p95': stats['total_r_squared']['p95'],
            'total_r_squared_std': stats['total_r_squared']['std'],

            # r_squared statistics (within-model R²)
            'r_squared_point': result.r_squared_point,
            'r_squared_median': stats['r_squared']['p50'],
            'r_squared_p5': stats['r_squared']['p5'],
            'r_squared_p25': stats['r_squared']['p25'],
            'r_squared_p75': stats['r_squared']['p75'],
            'r_squared_p95': stats['r_squared']['p95'],
            'r_squared_std': stats['r_squared']['std'],
        }

        # Add beta statistics for approaches where it's a free parameter
        if result.beta_point is not None and 'beta' in stats:
            row['beta_point'] = result.beta_point
            row['beta_median'] = stats['beta']['p50']
            row['beta_p5'] = stats['beta']['p5']
            row['beta_p25'] = stats['beta']['p25']
            row['beta_p75'] = stats['beta']['p75']
            row['beta_p95'] = stats['beta']['p95']
            row['beta_std'] = stats['beta']['std']
        else:
            # Fill with NaN for approaches without beta
            row['beta_point'] = np.nan
            row['beta_median'] = np.nan
            row['beta_p5'] = np.nan
            row['beta_p25'] = np.nan
            row['beta_p75'] = np.nan
            row['beta_p95'] = np.nan
            row['beta_std'] = np.nan

        # Add h2_low and h2_high statistics for Approach 8 (piecewise quadratic)
        if result.h2_low_point is not None and 'h2_low' in stats:
            row['h2_low_point'] = result.h2_low_point
            row['h2_low_median'] = stats['h2_low']['p50']
            row['h2_low_p5'] = stats['h2_low']['p5']
            row['h2_low_p25'] = stats['h2_low']['p25']
            row['h2_low_p75'] = stats['h2_low']['p75']
            row['h2_low_p95'] = stats['h2_low']['p95']
            row['h2_low_std'] = stats['h2_low']['std']
        else:
            row['h2_low_point'] = np.nan
            row['h2_low_median'] = np.nan
            row['h2_low_p5'] = np.nan
            row['h2_low_p25'] = np.nan
            row['h2_low_p75'] = np.nan
            row['h2_low_p95'] = np.nan
            row['h2_low_std'] = np.nan

        if result.h2_high_point is not None and 'h2_high' in stats:
            row['h2_high_point'] = result.h2_high_point
            row['h2_high_median'] = stats['h2_high']['p50']
            row['h2_high_p5'] = stats['h2_high']['p5']
            row['h2_high_p25'] = stats['h2_high']['p25']
            row['h2_high_p75'] = stats['h2_high']['p75']
            row['h2_high_p95'] = stats['h2_high']['p95']
            row['h2_high_std'] = stats['h2_high']['std']
        else:
            row['h2_high_point'] = np.nan
            row['h2_high_median'] = np.nan
            row['h2_high_p5'] = np.nan
            row['h2_high_p25'] = np.nan
            row['h2_high_p75'] = np.nan
            row['h2_high_p95'] = np.nan
            row['h2_high_std'] = np.nan

        # Variance decomposition
        if result.var_decomp_point is not None:
            for key, val in result.var_decomp_point.items():
                if isinstance(val, (int, float)):
                    stat_key = f'vd_{key}'
                    row[f'vd_{key}_point'] = val
                    if stat_key in stats:
                        row[f'vd_{key}_median'] = stats[stat_key]['p50']
                        row[f'vd_{key}_p5'] = stats[stat_key]['p5']
                        row[f'vd_{key}_p25'] = stats[stat_key]['p25']
                        row[f'vd_{key}_p75'] = stats[stat_key]['p75']
                        row[f'vd_{key}_p95'] = stats[stat_key]['p95']
                        row[f'vd_{key}_std'] = stats[stat_key]['std']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / 'bootstrap_summary_table.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_summary_table.csv ({len(df)} rows)")

    # Save as XLSX
    xlsx_path = output_dir / 'bootstrap_summary_table.xlsx'
    df.to_excel(xlsx_path, index=False, sheet_name='Bootstrap Summary')
    print(f"  Saved bootstrap_summary_table.xlsx")


def compute_h_response_uncertainty_bands(
    result: "BootstrapResult",
    T_range: np.ndarray,
    percentiles: tuple = (5, 50, 95),
    approach_key: str = None
) -> tuple:
    """Compute h(T) - h(T*) uncertainty bands from bootstrap samples.

    For each bootstrap sample, computes h(T) - h(T*) over the temperature range.
    Returns percentile bands across all bootstrap samples.

    For quadratic models: h(T) = h1*T + h2*T²
    For piecewise (approach8): h(T) - h(T_opt) = h2_low*(T-T_opt)² or h2_high*(T-T_opt)²

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)
        approach_key: Approach identifier (e.g., 'approach8' for piecewise)

    Returns:
        Tuple of arrays (h_lower, h_median, h_upper) each with shape (len(T_range),)
    """
    is_piecewise = (approach_key == 'approach8')

    if is_piecewise:
        # Piecewise quadratic model: need h2_low, h2_high, T_optimal samples
        h2_low_samples = getattr(result, 'h2_low_samples', None)
        h2_high_samples = getattr(result, 'h2_high_samples', None)

        if h2_low_samples is None or h2_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(result.T_optimal_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_opt_valid = result.T_optimal_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        h_relative_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2_low = h2_low_valid[i]
            h2_high = h2_high_valid[i]
            T_opt = T_opt_valid[i]
            # h(T) - h(T_opt) = h(T) since h(T_opt) = 0
            low_comp, high_comp = piecewise_quad_shape(T_range, T_opt)
            h_relative_samples[i, :] = h2_low * low_comp + h2_high * high_comp
    else:
        # Quadratic model
        h1_valid, h2_valid, _ = get_valid_bootstrap_samples(result)

        if len(h1_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h1_valid)
        n_T = len(T_range)
        h_relative_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h1 = h1_valid[i]
            h2 = h2_valid[i]
            h_T = h1 * T_range + h2 * T_range ** 2
            # Evaluate h at T_optimal
            if h2 != 0:
                T_opt = -h1 / (2 * h2)
                h_T_opt = -h1 ** 2 / (4 * h2)
            else:
                h_T_opt = 0
            h_relative_samples[i, :] = h_T - h_T_opt

    # Compute percentiles at each temperature
    h_bands = []
    for p in percentiles:
        h_bands.append(np.percentile(h_relative_samples, p, axis=0))

    return tuple(h_bands)


def plot_bootstrap_parameter_distributions(
    result: "BootstrapResult",
    stats: Dict[str, Dict],
    output_dir: Path,
    approach_key: str
) -> None:
    """Plot h1, h2, T_optimal distributions for one approach.

    Creates a (1, 3) subplot with histograms showing:
    - Point estimate (red solid line)
    - Bootstrap median (blue dashed line)
    - 90% CI bounds (gray dotted lines)

    Args:
        result: BootstrapResult for this approach
        stats: Statistics dict from compute_bootstrap_statistics
        output_dir: Directory to save the plot
        approach_key: Key like 'approach0' for filename
    """
    params = [
        ('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁ (Linear Coefficient)'),
        ('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (Quadratic Coefficient)'),
        ('T_optimal', result.T_optimal_samples, result.T_optimal_point, stats['T_optimal'], 'T_optimal (°C)'),
    ]

    n_panels = len(params)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (param_name, samples, point_est, param_stats, xlabel) in zip(axes, params):
        # Filter valid samples
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(xlabel, fontsize=12)
            continue

        # Histogram
        ax.hist(valid_samples, bins=50, density=True, alpha=0.7, color='steelblue')

        # Point estimate (red solid)
        ax.axvline(x=point_est, color='red', linestyle='-', linewidth=2, label=f'Point est: {point_est:.4f}')

        # Bootstrap median (blue dashed)
        median = param_stats['p50']
        ax.axvline(x=median, color='blue', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')

        # 90% CI bounds (gray dotted)
        p5 = param_stats['p5']
        p95 = param_stats['p95']
        ax.axvline(x=p5, color='gray', linestyle=':', linewidth=1.5, label=f'5th pct: {p5:.4f}')
        ax.axvline(x=p95, color='gray', linestyle=':', linewidth=1.5, label=f'95th pct: {p95:.4f}')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Bootstrap Distributions: {result.approach}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'bootstrap_distributions_{approach_key}.pdf')
    plt.close()


def plot_all_bootstrap_distributions(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    filename: str = "bootstrap_distributions.pdf",
    input_file: str = None
) -> None:
    """Plot h1, h2, T_optimal distributions for all approaches in a single PDF.

    Creates a multi-panel figure with one row per approach and 3 columns
    (h1, h2, T_optimal).

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save the plot
        filename: Output filename (should end in .pdf)
        input_file: Path to input data file (for annotation)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    approach_names = list(results.keys())
    n_approaches = len(approach_names)

    n_cols = 3

    with PdfPages(output_dir / filename) as pdf:
        # Create a figure with all approaches - one row per approach, 3 columns
        fig, axes = plt.subplots(n_approaches, n_cols, figsize=(4.5 * n_cols, 4 * n_approaches))

        if n_approaches == 1:
            axes = axes.reshape(1, -1)

        # First pass: determine x-axis ranges for each column
        col_ranges = {i: [] for i in range(n_cols)}  # h1, h2, T_optimal
        for name in approach_names:
            result = results[name]
            sample_lists = [result.h1_samples, result.h2_samples, result.T_optimal_samples]
            for col_idx, samples in enumerate(sample_lists):
                valid_samples = samples[~np.isnan(samples)]
                if len(valid_samples) > 0:
                    col_ranges[col_idx].extend([valid_samples.min(), valid_samples.max()])

        # Compute min/max for each column with small padding
        col_xlims = {}
        for col_idx, values in col_ranges.items():
            if values:
                xmin, xmax = min(values), max(values)
                padding = (xmax - xmin) * 0.05
                col_xlims[col_idx] = (xmin - padding, xmax + padding)
            else:
                col_xlims[col_idx] = None

        for row_idx, name in enumerate(approach_names):
            result = results[name]
            stats = all_stats[name]

            params = [
                ('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁'),
                ('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂'),
                ('T_optimal', result.T_optimal_samples, result.T_optimal_point, stats['T_optimal'], 'T_optimal (°C)'),
            ]

            for col_idx, (param_name, samples, point_est, param_stats, xlabel) in enumerate(params):
                ax = axes[row_idx, col_idx]

                # Filter valid samples
                valid_samples = samples[~np.isnan(samples)]
                if len(valid_samples) == 0:
                    ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel(xlabel, fontsize=10)
                    if col_xlims[col_idx]:
                        ax.set_xlim(col_xlims[col_idx])
                    continue

                # Histogram - use fixed bins based on column range for consistency
                if col_xlims[col_idx]:
                    bin_edges = np.linspace(col_xlims[col_idx][0], col_xlims[col_idx][1], 51)
                    ax.hist(valid_samples, bins=bin_edges, density=True, alpha=0.7, color='steelblue')
                else:
                    ax.hist(valid_samples, bins=50, density=True, alpha=0.7, color='steelblue')

                # Point estimate (red solid)
                ax.axvline(x=point_est, color='red', linestyle='-', linewidth=2, label=f'Point: {point_est:.4f}')

                # Bootstrap median (blue dashed)
                median = param_stats['p50']
                ax.axvline(x=median, color='blue', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')

                # 90% CI bounds (gray dotted)
                p5 = param_stats['p5']
                p95 = param_stats['p95']
                ax.axvline(x=p5, color='gray', linestyle=':', linewidth=1.5, label=f'5%: {p5:.4f}')
                ax.axvline(x=p95, color='gray', linestyle=':', linewidth=1.5, label=f'95%: {p95:.4f}')

                ax.set_xlabel(xlabel, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f'{result.approach}\n\nDensity', fontsize=10)
                else:
                    ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

                # Set consistent x-axis range for all panels in this column
                if col_xlims[col_idx]:
                    ax.set_xlim(col_xlims[col_idx])

                # Add title only on top row
                if row_idx == 0:
                    ax.set_title(xlabel, fontsize=11)

        fig.suptitle('Bootstrap Parameter Distributions by Approach', fontsize=14, y=1.01)
        plt.tight_layout()
        add_input_file_annotation(fig, input_file)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def plot_bootstrap_temperature_response(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = "bootstrap_temperature_response.pdf",
    T_range: tuple = (0, 30),
    data: AnalysisData = None,
    input_file: str = None,
) -> None:
    """Plot h(T) - h(T*) with 90% CI bands in multi-panel layout.

    Each approach gets its own panel to avoid overlapping uncertainty bands.
    All panels share the same y-axis range for easy comparison.
    Output is saved as PDF.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        approaches: List of approach keys to include (default: all)
        filename: Output filename (should end in .pdf)
        T_range: Temperature range for x-axis
        data: AnalysisData for adding temperature histogram (optional)
    """
    T = np.linspace(T_range[0], T_range[1], 200)

    if approaches is None:
        approaches = list(results.keys())

    # Filter to only approaches that exist in results
    approaches = [name for name in approaches if name in results]
    n_approaches = len(approaches)

    if n_approaches == 0:
        return

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for name in approaches:
        result = results[name]

        # Compute uncertainty bands (90% CI and IQR)
        # Pass approach_key for power-law handling
        h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
        )

        # Compute point estimate response
        # Check if this is Gaussian model (approach8)
        if name == 'approach8' and result.beta_point is not None:
            # Gaussian: h(T) - h(T_opt) = h2 * (2*pi*sigma)^(-0.5) * [exp(...) - 1]
            h2_point = result.h2_point
            T_opt_point = result.T_optimal_point
            sigma_point = result.beta_point  # sigma stored as beta for compatibility
            norm_factor = 1.0 / np.sqrt(2 * np.pi * sigma_point)
            gauss_shape = np.exp(-((T - T_opt_point) ** 2) / (2 * sigma_point ** 2))
            h_point = h2_point * norm_factor * (gauss_shape - 1.0)
        else:
            # Quadratic model
            h1_point = result.h1_point
            h2_point = result.h2_point
            h_T_point = h1_point * T + h2_point * T ** 2
            if h2_point != 0:
                h_T_opt_point = -h1_point ** 2 / (4 * h2_point)
            else:
                h_T_opt_point = 0
            h_point = h_T_point - h_T_opt_point

        plot_data[name] = {
            'h_p5': h_p5,
            'h_p25': h_p25,
            'h_p75': h_p75,
            'h_p95': h_p95,
            'h_point': h_point,
        }

        # Update global y range
        y_min = min(y_min, np.nanmin(h_p5), np.nanmin(h_point))
        y_max = max(y_max, np.nanmax(h_p95), np.nanmax(h_point))

    # Add some padding to y range
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Determine grid layout
    if n_approaches <= 3:
        n_rows, n_cols = 1, n_approaches
    elif n_approaches <= 4:
        n_rows, n_cols = 2, 2
    elif n_approaches <= 6:
        n_rows, n_cols = 2, 3
    elif n_approaches <= 8:
        n_rows, n_cols = 4, 2
    elif n_approaches <= 9:
        n_rows, n_cols = 3, 3
    elif n_approaches <= 12:
        n_rows, n_cols = 4, 3
    else:
        n_rows, n_cols = 4, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_approaches == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Get temperature data from most recent year for histogram (if data provided)
    temp_recent = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

    # Second pass: create the plots
    for idx, name in enumerate(approaches):
        ax = axes[idx]
        result = results[name]
        color = APPROACH_COLORS.get(name, 'steelblue')
        pdata = plot_data[name]

        # Add temperature histogram on secondary y-axis (if data provided)
        if temp_recent is not None:
            ax2 = ax.twinx()
            # Create histogram
            bins = np.linspace(T_range[0], T_range[1], 30)
            ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel('Data density', fontsize=8, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
            ax2.set_ylim(bottom=0)
            # Ensure histogram is behind main plot
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        # Plot 90% CI band
        ax.fill_between(T, pdata['h_p5'], pdata['h_p95'], alpha=0.2, color=color, label='90% CI')

        # Plot IQR band
        ax.fill_between(T, pdata['h_p25'], pdata['h_p75'], alpha=0.3, color=color, label='IQR')

        # Plot point estimate
        ax.plot(T, pdata['h_point'], color=color, linestyle='-', linewidth=2, label='Point estimate')

        # Mark optimal temperature
        ax.axvline(result.T_optimal_point, color=color, linestyle=':', alpha=0.7,
                   label=f'T_opt = {result.T_optimal_point:.1f}°C')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
        ax.set_title(f'{result.approach}', fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

    # Hide unused subplots
    for idx in range(n_approaches, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Temperature Response with Bootstrap 90% CI and IQR', fontsize=14, y=1.02)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def plot_bootstrap_T_optimal_comparison(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Horizontal error bar plot: point estimate + 90% CI + IQR for each approach.

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    approach_names = list(results.keys())
    n_approaches = len(approach_names)
    y_positions = np.arange(n_approaches)

    for i, name in enumerate(approach_names):
        result = results[name]
        stats = all_stats[name]['T_optimal']
        color = APPROACH_COLORS.get(name, 'gray')

        point_est = result.T_optimal_point
        p5, p25, p50, p75, p95 = stats['p5'], stats['p25'], stats['p50'], stats['p75'], stats['p95']

        # Plot 90% CI as error bar (use abs to handle cases where point est is outside CI)
        ci_lower = abs(point_est - p5)
        ci_upper = abs(p95 - point_est)
        ax.errorbar(point_est, i, xerr=[[ci_lower], [ci_upper]],
                    fmt='o', color=color, capsize=5, capthick=2, markersize=8,
                    label=f'{result.approach}' if i == 0 else None)

        # Plot IQR as a thick bar
        ax.plot([p25, p75], [i, i], color=color, linewidth=4, alpha=0.5)

        # Add label with point estimate value
        ax.annotate(f'{point_est:.1f}°C', xy=(point_est, i), xytext=(5, 0),
                    textcoords='offset points', fontsize=9, va='center')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([results[name].approach for name in approach_names])
    ax.set_xlabel('Optimal Temperature (°C)', fontsize=12)
    ax.set_title('T_optimal with Bootstrap 90% CI and IQR', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend explaining markers
    ax.annotate('Circle: point estimate, Thin line: 90% CI, Thick line: IQR',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'bootstrap_T_optimal_comparison.pdf')
    plt.close()


def compute_derivative_uncertainty_bands(
    result: "BootstrapResult",
    T_range: np.ndarray,
    percentiles: tuple = (5, 50, 95),
    approach_key: str = None
) -> tuple:
    """Compute dh/dT uncertainty bands from bootstrap samples.

    For quadratic models: dh/dT = h1 + 2*h2*T
    For skew-normal (approach8): dh/dT = h2 * df/dT where
        df/dT = (C/σ) * g(z) * [-z*s(z) + s'(z)]

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)
        approach_key: Approach identifier (e.g., 'approach8' for skew-normal)

    Returns:
        Tuple of arrays (dh_lower, dh_median, dh_upper) each with shape (len(T_range),)
    """
    is_skewnorm = (approach_key == 'approach8')

    if is_skewnorm:
        # Skew-normal model: need h2, T_optimal, sigma, and alpha samples
        sigma_samples = getattr(result, 'sigma_samples', None)
        alpha_samples = getattr(result, 'alpha_samples', None)

        if sigma_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(result.h2_samples) &
                      ~np.isnan(result.T_optimal_samples) &
                      ~np.isnan(sigma_samples))
        if alpha_samples is not None:
            valid_mask = valid_mask & ~np.isnan(alpha_samples)

        h2_valid = result.h2_samples[valid_mask]
        T_opt_valid = result.T_optimal_samples[valid_mask]
        sigma_valid = sigma_samples[valid_mask]
        alpha_valid = alpha_samples[valid_mask] if alpha_samples is not None else np.zeros_like(sigma_valid)

        if len(h2_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_valid)
        n_T = len(T_range)
        dh_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2 = h2_valid[i]
            T_opt = T_opt_valid[i]
            sigma = sigma_valid[i]
            alpha = alpha_valid[i]

            # Skew-normal derivative
            z = (T_range - T_opt) / sigma
            C = 1.0 / (sigma * np.sqrt(2 * np.pi))
            g_z = np.exp(-0.5 * z ** 2)
            s_z = 1.0 + erf(alpha * z / np.sqrt(2))
            # Derivative of erf term: s'(z) = α√(2/π) * exp(-α²z²/2)
            s_prime_z = alpha * np.sqrt(2 / np.pi) * np.exp(-0.5 * (alpha * z) ** 2)
            # df/dT = (C/σ) * g(z) * [-z*s(z) + s'(z)]
            df_dT = (C / sigma) * g_z * (-z * s_z + s_prime_z)
            dh_samples[i, :] = h2 * df_dT
    else:
        # Quadratic model
        h1_valid, h2_valid, _ = get_valid_bootstrap_samples(result)

        if len(h1_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h1_valid)
        n_T = len(T_range)
        dh_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h1 = h1_valid[i]
            h2 = h2_valid[i]
            dh_samples[i, :] = h1 + 2 * h2 * T_range

    # Compute percentiles at each temperature
    dh_bands = []
    for p in percentiles:
        dh_bands.append(np.percentile(dh_samples, p, axis=0))

    return tuple(dh_bands)


def plot_bootstrap_temperature_derivative(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = "bootstrap_temperature_derivative.pdf",
    T_range: tuple = (0, 30),
    input_file: str = None
) -> None:
    """Plot dh/dT = h1 + 2*h2*T with 90% CI bands in multi-panel layout.

    Each approach gets its own panel to avoid overlapping uncertainty bands.
    All panels share the same y-axis range for easy comparison.
    Output is saved as PDF.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        approaches: List of approach keys to include (default: all)
        filename: Output filename (should end in .pdf)
        T_range: Temperature range for x-axis
        input_file: Path to input data file (for annotation)
    """
    T = np.linspace(T_range[0], T_range[1], 200)

    if approaches is None:
        approaches = list(results.keys())

    # Filter to only approaches that exist in results
    approaches = [name for name in approaches if name in results]
    n_approaches = len(approaches)

    if n_approaches == 0:
        return

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for name in approaches:
        result = results[name]

        # Compute uncertainty bands
        # Pass approach_key for power-law handling
        dh_lower, dh_median, dh_upper = compute_derivative_uncertainty_bands(
            result, T, approach_key=name
        )

        # Compute point estimate derivative
        # Check if this is Gaussian model (approach8)
        if name == 'approach8' and result.beta_point is not None:
            # Gaussian: dh/dT = h2 * (2*pi*sigma)^(-0.5) * exp(...) * (-(T-T_opt)/sigma^2)
            h2_point = result.h2_point
            T_opt_point = result.T_optimal_point
            sigma_point = result.beta_point  # sigma stored as beta for compatibility
            T_diff = T - T_opt_point
            norm_factor = 1.0 / np.sqrt(2 * np.pi * sigma_point)
            gauss_shape = np.exp(-(T_diff ** 2) / (2 * sigma_point ** 2))
            dh_point = h2_point * norm_factor * gauss_shape * (-T_diff / (sigma_point ** 2))
        else:
            # Quadratic model
            h1_point = result.h1_point
            h2_point = result.h2_point
            dh_point = h1_point + 2 * h2_point * T

        plot_data[name] = {
            'dh_lower': dh_lower,
            'dh_upper': dh_upper,
            'dh_point': dh_point,
        }

        # Update global y range
        y_min = min(y_min, np.nanmin(dh_lower), np.nanmin(dh_point))
        y_max = max(y_max, np.nanmax(dh_upper), np.nanmax(dh_point))

    # Add some padding to y range
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Determine grid layout
    if n_approaches <= 3:
        n_rows, n_cols = 1, n_approaches
    elif n_approaches <= 4:
        n_rows, n_cols = 2, 2
    elif n_approaches <= 6:
        n_rows, n_cols = 2, 3
    elif n_approaches <= 8:
        n_rows, n_cols = 4, 2
    elif n_approaches <= 9:
        n_rows, n_cols = 3, 3
    elif n_approaches <= 12:
        n_rows, n_cols = 4, 3
    else:
        n_rows, n_cols = 4, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_approaches == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Second pass: create the plots
    for idx, name in enumerate(approaches):
        ax = axes[idx]
        result = results[name]
        color = APPROACH_COLORS.get(name, 'steelblue')
        data = plot_data[name]

        # Plot CI band
        ax.fill_between(T, data['dh_lower'], data['dh_upper'], alpha=0.3, color=color, label='90% CI')

        # Plot point estimate
        ax.plot(T, data['dh_point'], color=color, linestyle='-', linewidth=2, label='Point estimate')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('dh/dT = h₁ + 2h₂T', fontsize=10)
        ax.set_title(f'{result.approach}', fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Hide unused subplots
    for idx in range(n_approaches, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Temperature Derivative with Bootstrap 90% CI', fontsize=14, y=1.02)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def plot_T_optimal_histograms(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = "T_optimal_histograms.pdf",
    input_file: str = None,
) -> None:
    """Plot T_optimal bootstrap distributions as histograms in multi-panel layout.

    Each approach gets its own panel showing the histogram of T_optimal samples
    with point estimate, median, and 90% CI bounds marked.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        approaches: List of approach keys to include (default: all)
        filename: Output filename (should end in .pdf)
        input_file: Path to input data file (for annotation)
    """
    if approaches is None:
        approaches = list(results.keys())

    # Filter to only approaches that exist in results
    approaches = [name for name in approaches if name in results]
    n_approaches = len(approaches)

    if n_approaches == 0:
        return

    # Determine grid layout
    if n_approaches == 1:
        n_rows, n_cols = 1, 1
    elif n_approaches == 2:
        n_rows, n_cols = 1, 2
    elif n_approaches <= 4:
        n_rows, n_cols = 2, 2
    elif n_approaches <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_approaches == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # First pass: find global x-axis range for consistent scaling
    all_samples = []
    all_points = []
    for name in approaches:
        result = results[name]
        valid_samples = result.T_optimal_samples[~np.isnan(result.T_optimal_samples)]
        if len(valid_samples) > 0:
            all_samples.extend(valid_samples)
            all_points.append(result.T_optimal_point)

    if len(all_samples) == 0:
        plt.close()
        return

    # Compute common x-axis range with some padding
    x_min = min(np.percentile(all_samples, 1), min(all_points)) - 2
    x_max = max(np.percentile(all_samples, 99), max(all_points)) + 2

    # Second pass: create the plots
    for idx, name in enumerate(approaches):
        ax = axes[idx]
        result = results[name]
        color = APPROACH_COLORS.get(name, 'steelblue')

        # Get valid samples
        valid_samples = result.T_optimal_samples[~np.isnan(result.T_optimal_samples)]

        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{result.approach}', fontsize=11)
            continue

        # Compute statistics
        point_est = result.T_optimal_point
        median = np.median(valid_samples)
        p5 = np.percentile(valid_samples, 5)
        p95 = np.percentile(valid_samples, 95)

        # Plot histogram
        ax.hist(valid_samples, bins=40, density=True, alpha=0.7, color=color,
                edgecolor='white', linewidth=0.5)

        # Point estimate (solid line)
        ax.axvline(x=point_est, color='black', linestyle='-', linewidth=2,
                   label=f'Point: {point_est:.1f}°C')

        # Bootstrap median (dashed line)
        ax.axvline(x=median, color=color, linestyle='--', linewidth=2,
                   label=f'Median: {median:.1f}°C')

        # 90% CI bounds (dotted lines)
        ax.axvline(x=p5, color='gray', linestyle=':', linewidth=1.5,
                   label=f'90% CI: [{p5:.1f}, {p95:.1f}]')
        ax.axvline(x=p95, color='gray', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Optimal Temperature (°C)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{result.approach}', fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_approaches, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Bootstrap Distribution of Optimal Temperature', fontsize=14, y=1.02)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def plot_bootstrap_gdp_scaling(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    Y_ref: float,
    Y_range: tuple = None,
    filename: str = 'bootstrap_gdp_scaling.pdf',
    data: AnalysisData = None,
    input_file: str = None,
) -> None:
    """Plot GDP scaling factor with bootstrap uncertainty bands for Approach 7.

    Shows the spread of (Y/Y_ref)^(-beta) curves across bootstrap samples.
    Creates a two-panel figure when both approaches are present.

    Args:
        results: Dict of BootstrapResult
        output_dir: Directory to save the plot
        Y_ref: Reference GDP value (same as used in fitting)
        Y_range: GDP range for x-axis (default: 500 to 100000)
        filename: Output filename
        data: AnalysisData for adding GDP histogram (optional)
        input_file: Path to input data file (for annotation)
    """
    # Collect panels to plot
    panels = []
    for key, title, color in [
        ('approach7', 'GDP-Response LOESS (Approach 7)', 'brown'),
    ]:
        if key in results:
            result = results[key]
            if result.beta_point is not None and result.beta_samples is not None:
                valid_betas = result.beta_samples[~np.isnan(result.beta_samples)]
                if len(valid_betas) > 0:
                    panels.append((result.beta_point, valid_betas, title, color))

    if not panels:
        return

    # Default Y range
    if Y_range is None:
        Y_range = (500, 100000)

    # Create GDP array (log-spaced)
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 6), squeeze=False)

    def _draw_panel(ax, beta_point, valid_betas, title, color):
        # Plot individual bootstrap samples (thin lines)
        n_samples_to_plot = min(100, len(valid_betas))
        sample_indices = np.linspace(0, len(valid_betas) - 1, n_samples_to_plot, dtype=int)

        for idx in sample_indices:
            beta_b = valid_betas[idx]
            g_b = (Y / Y_ref) ** (-beta_b)
            ax.plot(Y, g_b, color=color, alpha=0.05, linewidth=0.5)

        # Compute percentile bands
        g_samples = np.zeros((len(valid_betas), len(Y)))
        for i, beta_b in enumerate(valid_betas):
            g_samples[i, :] = (Y / Y_ref) ** (-beta_b)

        g_p5 = np.percentile(g_samples, 5, axis=0)
        g_p25 = np.percentile(g_samples, 25, axis=0)
        g_p75 = np.percentile(g_samples, 75, axis=0)
        g_p95 = np.percentile(g_samples, 95, axis=0)

        # Add GDP histogram on secondary y-axis (if data provided)
        if data is not None:
            max_year = data.year_range[1]
            mask_recent = data.year == max_year
            gdp_recent = data.pcGDP[mask_recent]

            ax2 = ax.twinx()
            bins = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 30)
            ax2.hist(gdp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel(f'Data density ({max_year})', fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            ax2.set_ylim(bottom=0)
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        # Plot uncertainty bands
        ax.fill_between(Y, g_p5, g_p95, color=color, alpha=0.2, label='90% CI')
        ax.fill_between(Y, g_p25, g_p75, color=color, alpha=0.3, label='IQR')

        # Plot point estimate
        g_point = (Y / Y_ref) ** (-beta_point)
        ax.plot(Y, g_point, color=color, linewidth=2.5,
                label=f'Point estimate (β = {beta_point:.3f})')

        # Reference lines
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref ≈ ${Y_ref:,.0f}')

        ax.set_xscale('log')
        ax.set_xlabel('Per Capita GDP ($)', fontsize=12)
        ax.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-β)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add beta distribution inset
        ax_inset = ax.inset_axes([0.72, 0.42, 0.25, 0.30])
        ax_inset.hist(valid_betas, bins=30, color=color, alpha=0.7, density=True)
        ax_inset.axvline(beta_point, color='red', linewidth=1.5, label='Point est.')
        ax_inset.set_xlabel('β', fontsize=9)
        ax_inset.set_ylabel('Density', fontsize=9)
        ax_inset.set_title('Bootstrap β distribution', fontsize=9)
        ax_inset.tick_params(labelsize=8)

    for i, (beta_point, valid_betas, title, color) in enumerate(panels):
        _draw_panel(axes[0, i], beta_point, valid_betas, title, color)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename)
    plt.close()


def plot_bootstrap_approach7_combined(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    Y_ref: float,
    T_range: tuple = (0, 30),
    Y_range: tuple = None,
    filename: str = "bootstrap_temperature_response_gdp.pdf",
    data: AnalysisData = None,
    input_file: str = None,
) -> None:
    """Plot Approach 7 h(T) response and GDP scaling side by side in one row.

    Left panel: temperature response h(T) - h(T*) with bootstrap CI
    Right panel: GDP scaling factor (Y/Y_ref)^(-beta) with bootstrap CI

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        Y_ref: Reference GDP value
        T_range: Temperature range for x-axis of left panel
        Y_range: GDP range for x-axis of right panel (default: 500 to 100000)
        filename: Output filename
        data: AnalysisData for adding data density histograms (optional)
        input_file: Path to input data file (for annotation)
    """
    if 'approach7' not in results:
        return

    result = results['approach7']
    if result.beta_point is None or result.beta_samples is None:
        return

    valid_betas = result.beta_samples[~np.isnan(result.beta_samples)]
    if len(valid_betas) == 0:
        return

    if Y_range is None:
        Y_range = (500, 100000)

    color = APPROACH_COLORS.get('approach7', 'brown')

    fig, (ax_temp, ax_gdp) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: Temperature response h(T) - h(T*) ---
    T = np.linspace(T_range[0], T_range[1], 200)

    h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
        result, T, percentiles=(5, 25, 50, 75, 95)
    )

    h1_point = result.h1_point
    h2_point = result.h2_point
    h_T_point = h1_point * T + h2_point * T ** 2
    if h2_point != 0:
        h_T_opt_point = -h1_point ** 2 / (4 * h2_point)
    else:
        h_T_opt_point = 0
    h_point = h_T_point - h_T_opt_point

    # Temperature histogram
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]
        ax_temp2 = ax_temp.twinx()
        bins = np.linspace(T_range[0], T_range[1], 30)
        ax_temp2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax_temp2.set_ylabel('Data density', fontsize=8, color='gray')
        ax_temp2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax_temp2.set_ylim(bottom=0)
        ax_temp2.set_zorder(ax_temp.get_zorder() - 1)
        ax_temp.set_zorder(ax_temp2.get_zorder() + 1)
        ax_temp.patch.set_visible(False)

    ax_temp.fill_between(T, h_p5, h_p95, alpha=0.2, color=color, label='90% CI')
    ax_temp.fill_between(T, h_p25, h_p75, alpha=0.3, color=color, label='IQR')
    ax_temp.plot(T, h_point, color=color, linestyle='-', linewidth=2, label='Point estimate')
    ax_temp.axvline(result.T_optimal_point, color=color, linestyle=':', alpha=0.7,
                    label=f'T_opt = {result.T_optimal_point:.1f}°C')
    ax_temp.axhline(0, color='gray', linewidth=0.5)
    ax_temp.set_xlabel('Temperature (°C)', fontsize=10)
    ax_temp.set_ylabel('h(T) - h(T_opt)', fontsize=10)
    ax_temp.set_title(f'{result.approach}', fontsize=11)
    ax_temp.set_xlim(T_range)
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(fontsize=8, loc='lower left')

    # --- Right panel: GDP scaling factor ---
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    # Plot individual bootstrap samples (thin lines)
    n_samples_to_plot = min(100, len(valid_betas))
    sample_indices = np.linspace(0, len(valid_betas) - 1, n_samples_to_plot, dtype=int)
    for idx in sample_indices:
        beta_b = valid_betas[idx]
        g_b = (Y / Y_ref) ** (-beta_b)
        ax_gdp.plot(Y, g_b, color=color, alpha=0.05, linewidth=0.5)

    # Compute percentile bands
    g_samples = np.zeros((len(valid_betas), len(Y)))
    for i, beta_b in enumerate(valid_betas):
        g_samples[i, :] = (Y / Y_ref) ** (-beta_b)
    g_p5 = np.percentile(g_samples, 5, axis=0)
    g_p25 = np.percentile(g_samples, 25, axis=0)
    g_p75 = np.percentile(g_samples, 75, axis=0)
    g_p95 = np.percentile(g_samples, 95, axis=0)

    # GDP histogram
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        gdp_recent = data.pcGDP[mask_recent]
        ax_gdp2 = ax_gdp.twinx()
        bins = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 30)
        ax_gdp2.hist(gdp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax_gdp2.set_ylabel(f'Data density ({max_year})', fontsize=10, color='gray')
        ax_gdp2.tick_params(axis='y', labelcolor='gray', labelsize=8)
        ax_gdp2.set_ylim(bottom=0)
        ax_gdp2.set_zorder(ax_gdp.get_zorder() - 1)
        ax_gdp.set_zorder(ax_gdp2.get_zorder() + 1)
        ax_gdp.patch.set_visible(False)

    ax_gdp.fill_between(Y, g_p5, g_p95, color=color, alpha=0.2, label='90% CI')
    ax_gdp.fill_between(Y, g_p25, g_p75, color=color, alpha=0.3, label='IQR')
    g_point = (Y / Y_ref) ** (-result.beta_point)
    ax_gdp.plot(Y, g_point, color=color, linewidth=2.5,
                label=f'Point estimate (β = {result.beta_point:.3f})')
    ax_gdp.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax_gdp.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref ≈ ${Y_ref:,.0f}')
    ax_gdp.set_xscale('log')
    ax_gdp.set_xlabel('Per Capita GDP ($)', fontsize=12)
    ax_gdp.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-β)', fontsize=12)
    ax_gdp.set_title('GDP Scaling Factor', fontsize=11)
    ax_gdp.legend(loc='upper right', fontsize=8)
    ax_gdp.grid(True, alpha=0.3)

    # Beta distribution inset
    ax_inset = ax_gdp.inset_axes([0.72, 0.42, 0.25, 0.30])
    ax_inset.hist(valid_betas, bins=30, color=color, alpha=0.7, density=True)
    ax_inset.axvline(result.beta_point, color='red', linewidth=1.5, label='Point est.')
    ax_inset.set_xlabel('β', fontsize=9)
    ax_inset.set_ylabel('Density', fontsize=9)
    ax_inset.set_title('Bootstrap β distribution', fontsize=9)
    ax_inset.tick_params(labelsize=8)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename)
    plt.close()


def save_all_bootstrap_plots(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    T_range: tuple = (0, 30),
    Y_ref: float = None,
    data: AnalysisData = None,
    input_file: str = None,
) -> None:
    """Generate all bootstrap plots.

    Calls:
    - plot_all_bootstrap_distributions() for all approaches in single PDF
    - plot_bootstrap_temperature_response() for basic approaches (0,1,2,3) and precomputed k (4,5,6)
    - plot_bootstrap_approach7_combined() for Approach 7 h(T) + GDP scaling
    - plot_bootstrap_temperature_derivative() for all approaches
    - plot_bootstrap_T_optimal_comparison() for all approaches

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save plots
        T_range: Temperature range for response plots
        Y_ref: Reference GDP for Approach 7 GDP scaling plot
        data: AnalysisData for adding data density histograms (optional)
        input_file: Path to input data file (for annotation)
    """
    # Generate combined distribution plot for all approaches
    plot_all_bootstrap_distributions(results, all_stats, output_dir, input_file=input_file)
    print("      Saved bootstrap_distributions.pdf")

    # Temperature response PDF 1: Basic approaches (2x2: row1=[0,1], row2=[2,3])
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3'],
        filename='bootstrap_temperature_response_basic.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_basic.pdf")

    # Temperature response PDF 2: Precomputed k approaches (including 5a, 5b, 5c)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['approach4', 'approach5', 'approach5a', 'approach5b', 'approach5c', 'approach6'],
        filename='bootstrap_temperature_response_precomputed.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_precomputed.pdf")

    # Temperature response PDF 3: LOESS approaches (6, 7, 8)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['approach6', 'approach7', 'approach8'],
        filename='bootstrap_temperature_response_loess.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_loess.pdf")

    # Temperature response PDF 4: Approach 7 h(T) + GDP scaling combined
    if Y_ref is not None and 'approach7' in results:
        plot_bootstrap_approach7_combined(
            results, output_dir, Y_ref,
            T_range=T_range,
            data=data,
            input_file=input_file,
            filename='bootstrap_temperature_response_gdp.pdf',
        )
        print("      Saved bootstrap_temperature_response_gdp.pdf")

    # Temperature derivative plot - all approaches in one PDF
    plot_bootstrap_temperature_derivative(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3',
                    'approach4', 'approach5', 'approach5a', 'approach5b', 'approach5c',
                    'approach6', 'approach7', 'approach8'],
        filename='bootstrap_temperature_derivative.pdf',
        T_range=T_range,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_derivative.pdf")

    # T_optimal comparison across all approaches
    plot_bootstrap_T_optimal_comparison(results, all_stats, output_dir, input_file=input_file)
    print("      Saved bootstrap_T_optimal_comparison.pdf")
