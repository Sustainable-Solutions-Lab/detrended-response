"""Output and visualization for detrended response analysis."""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict
from .data_loader import AnalysisData
from .detrending import CountryTrends, CountryTrendsLoess, compute_year_means, compute_country_trends_loess
from .fitting import FitResult
from .persistence import (
    compute_persistence_accumulators,
    compute_pre_first_year_correction,
)
from .bootstrap import _get_T_loess_at_base_year
from .axes import get_axis_bounds_and_ticks

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

# Color scheme for approaches
# - Approach QJ (Quadratic Joint OLS): black
# - Approach QP (Quadratic Polynomial): red
# - Approach QL (Quadratic LOESS): orange
# - Approach PL (Piecewise LOESS): magenta
# - Approach DL (Decay LOESS): cyan
# - Null models (Approach NJ/NP/NL): gray
APPROACH_COLORS = {
    'Approach QJ': 'black',
    'Approach QP': 'red',
    'Approach QL': 'orange',
    'Approach PL': 'magenta',
    'Approach DL': 'cyan',
    'Approach PJ': 'darkgreen',
    'Approach DJ': 'darkblue',
    'Approach PP': 'olive',
    'Approach DP': 'teal',
    'Approach LL': 'blue',
    'Approach LJ': 'purple',
    'Approach SL': 'gold',
    'Approach SJ': 'darkkhaki',
    'Approach SP': 'goldenrod',
    'Approach TL': 'darkorange',
    'Approach TJ': 'sienna',
    'Approach TP': 'peru',
    'Approach NJ': 'gray',
    'Approach NP': 'gray',
    'Approach NL': 'gray',
}

# Line style scheme for approaches
# - Approach QJ (Quadratic Joint): solid
# - Approach QP (Quadratic Polynomial): dash-dot
# - LOESS approaches (QL, PL, DL): densely dashed
# - Null models: dashed/dotted
APPROACH_LINESTYLES = {
    'Approach QJ': '-',
    'Approach QP': '-.',
    'Approach QL': (0, (5, 1)),   # densely dashed
    'Approach PL': (0, (5, 1)),   # densely dashed
    'Approach DL': (0, (5, 1)),   # densely dashed
    'Approach PJ': '-',           # solid (conjoined)
    'Approach DJ': '-',           # solid (conjoined)
    'Approach PP': '-.',          # dash-dot (like Approach QP)
    'Approach DP': '-.',          # dash-dot (like Approach QP)
    'Approach SL': (0, (5, 1)),   # densely dashed
    'Approach SJ': '-',           # solid (conjoined)
    'Approach SP': '-.',          # dash-dot
    'Approach TL': (0, (5, 1)),   # densely dashed
    'Approach TJ': '-',           # solid (conjoined)
    'Approach TP': '-.',          # dash-dot
    'Approach NJ': '--',
    'Approach NP': ':',
    'Approach NL': ':',
}


def get_color(name: str, default: str = 'gray') -> str:
    """Get color for approach or method name.

    Returns the color if found in APPROACH_COLORS, otherwise returns the default.
    """
    return APPROACH_COLORS.get(name, default)


def get_linestyle(name: str, default='-'):
    """Get linestyle for approach or method name.

    Returns the linestyle if found in APPROACH_LINESTYLES, otherwise returns the default.
    """
    return APPROACH_LINESTYLES.get(name, default)


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
    """Check if result is from piecewise quadratic model (Approach PL/Approach PJ/Approach PP).

    Piecewise results have T_opt, h2, and h4 as primary parameters and h1=0.
    For Approach PL/Approach PJ/Approach PP, h2 is curvature below T_opt, h4 is curvature above T_opt.

    Checks approach name first (most reliable), then falls back to h1==0 check.
    """
    # Check approach name first (works for both FitResult and BootstrapResult)
    approach = getattr(result, 'approach', '')
    if 'piecewise' in approach.lower() or approach.startswith(('Approach PL', 'Approach PJ', 'Approach PP')):
        return True
    return hasattr(result, 'T_opt') and hasattr(result, 'h4') and getattr(result, 'h1', None) == 0.0


# Keep backward compatibility aliases
is_skewnorm_result = is_piecewise_result
is_gaussian_result = is_piecewise_result


def is_persistence_result(result) -> bool:
    """Check if result is from persistence model (Approach DL/Approach DJ/Approach DP).

    Persistence results have h4 as the decay rate parameter.
    """
    approach = getattr(result, 'approach', '')
    if 'decay' in approach.lower() or approach.startswith(('Approach DL', 'Approach DJ', 'Approach DP')):
        return True
    return False


def is_three_interval_result(result) -> bool:
    """Check if result is from three-interval model (Approach TL/Approach TJ/Approach TP).

    Three-interval results have T_crit_low and T_crit_high parameters.
    """
    approach = getattr(result, 'approach', '')
    if 'three-interval' in approach.lower() or approach.startswith(('Approach TL', 'Approach TJ', 'Approach TP')):
        return True
    return getattr(result, 'T_crit_low', None) is not None


def is_segmented_result(result) -> bool:
    """Check if result is from segmented linear model (Approach SL/Approach SJ/Approach SP).

    Segmented results have T_opt, h2, and h4 as primary parameters (slopes, not curvatures).
    """
    approach = getattr(result, 'approach', '')
    return 'segmented' in approach.lower() or approach.startswith(('Approach SL', 'Approach SJ', 'Approach SP'))


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


def segmented_linear_shape(T: np.ndarray, T_opt: float) -> tuple:
    """Compute segmented linear shape.

    Returns two arrays: one for T <= T_opt contribution, one for T > T_opt.

    Args:
        T: Temperature array
        T_opt: Optimal temperature (breakpoint)

    Returns:
        Tuple of (low_component, high_component)
    """
    low_component = np.where(T <= T_opt, T - T_opt, 0.0)
    high_component = np.where(T > T_opt, T - T_opt, 0.0)
    return low_component, high_component


def three_interval_shape(T: np.ndarray, T_crit_low: float, delta_T_crit: float) -> tuple:
    """Compute three-interval basis functions f_low and f_high.

    Args:
        T: Temperature array
        T_crit_low: Lower critical temperature
        delta_T_crit: Width of transition zone

    Returns:
        Tuple of (f_low, f_high) arrays
    """
    T_lo = T_crit_low
    dT = delta_T_crit
    T_hi = T_lo + dT

    d = T - T_lo

    below = T <= T_lo
    above = T >= T_hi
    middle = ~below & ~above

    f_low = np.where(below, d, np.where(above, dT / 2, 0.0))
    f_high = np.where(below, 0.0, np.where(above, d - dT / 2, 0.0))

    if dT > 0:
        d_mid = np.where(middle, d, 0.0)
        f_low = np.where(middle, d_mid - d_mid**2 / (2 * dT), f_low)
        f_high = np.where(middle, d_mid**2 / (2 * dT), f_high)

    return f_low, f_high


def compute_h_response(T: np.ndarray, result) -> np.ndarray:
    """Compute h(T) - h(T_opt) for any approach type.

    For quadratic model: h(T) = h1*T + h2*T², so h(T) - h(T_opt) = ...
    For piecewise model: h(T) - h(T_opt) = h2*(T-T_opt)² for T≤T_opt, h4*(T-T_opt)² for T>T_opt
        Since h(T_opt) = 0, h(T) - h(T_opt) = h(T)

    Args:
        T: Temperature array
        result: FitResult or similar with h1, h2, T_opt (and h4 for piecewise)

    Returns:
        Array of h(T) - h(T_opt) values
    """
    if is_three_interval_result(result):
        # Three-interval: h(T) = h2*f_low(T) + h4*f_high(T)
        T_crit_low = result.T_crit_low
        T_crit_high = result.T_crit_high
        h2 = result.h2
        h4 = result.h4
        f_low, f_high = three_interval_shape(T, T_crit_low, T_crit_high - T_crit_low)
        return h2 * f_low + h4 * f_high
    elif is_segmented_result(result):
        # Segmented linear: h(T_opt) = 0, so h(T) - h(T_opt) = h(T)
        T_opt = result.T_opt
        h2 = result.h2   # Slope below T_opt
        h4 = result.h4   # Slope above T_opt
        low_comp, high_comp = segmented_linear_shape(T, T_opt)
        return h2 * low_comp + h4 * high_comp
    elif is_piecewise_result(result):
        # Piecewise quadratic: h(T_opt) = 0, so h(T) - h(T_opt) = h(T)
        T_opt = result.T_opt
        h2 = result.h2   # Curvature below T_opt
        h4 = result.h4   # Curvature above T_opt
        low_comp, high_comp = piecewise_quad_shape(T, T_opt)
        return h2 * low_comp + h4 * high_comp
    else:
        # Quadratic: h(T) = h1*T + h2*T²
        h1, h2 = result.h1, result.h2
        h_T = h1 * T + h2 * T ** 2
        # h(T_opt) at optimal temperature
        T_opt = getattr(result, 'T_opt', np.nan)
        if not np.isnan(T_opt) and h2 != 0:
            h_T_opt = -h1 ** 2 / (4 * h2)
        else:
            h_T_opt = 0
        return h_T - h_T_opt


def compute_dh_dT(T: np.ndarray, result) -> np.ndarray:
    """Compute dh/dT for any approach type.

    For quadratic model: dh/dT = h1 + 2*h2*T
    For piecewise model: dh/dT = 2*h2*(T-T_opt) for T≤T_opt, 2*h4*(T-T_opt) for T>T_opt

    Args:
        T: Temperature array
        result: FitResult or similar with h1, h2 (and h4, T_opt for piecewise)

    Returns:
        Array of dh/dT values
    """
    if is_three_interval_result(result):
        # Three-interval derivative: piecewise linear
        T_crit_low = result.T_crit_low
        T_crit_high = result.T_crit_high
        h2 = result.h2
        h4 = result.h4
        below = T <= T_crit_low
        above = T >= T_crit_high
        delta = T_crit_high - T_crit_low
        if delta > 0:
            frac = np.clip((T - T_crit_low) / delta, 0, 1)
            return np.where(below, h2, np.where(above, h4, h2 + (h4 - h2) * frac))
        else:
            return np.where(T <= T_crit_low, h2, h4)
    elif is_segmented_result(result):
        # Segmented linear derivative: step function
        T_opt = result.T_opt
        h2 = result.h2   # Slope below T_opt
        h4 = result.h4   # Slope above T_opt
        return np.where(T <= T_opt, h2, h4)
    elif is_piecewise_result(result):
        # Piecewise quadratic derivative
        T_opt = result.T_opt
        h2 = result.h2   # Curvature below T_opt
        h4 = result.h4   # Curvature above T_opt

        # dh/dT = 2*h*(T - T_opt) with appropriate h based on which side of T_opt
        return np.where(
            T <= T_opt,
            2 * h2 * (T - T_opt),
            2 * h4 * (T - T_opt)
        )
    else:
        # Quadratic derivative: h1 + 2*h2*T
        return result.h1 + 2 * result.h2 * T


def create_output_dir(base_dir: str = "data/output", prefix: str = "", suffix: str = "") -> Path:
    """Create timestamped output directory.

    Args:
        base_dir: Base directory for output (default: "data/output")
        prefix: Optional prefix for the timestamped folder (e.g., "analysis_", "bootstrap_")
        suffix: Optional suffix inserted between prefix and timestamp (e.g., "mw10")

    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_part = f"{suffix}_" if suffix else ""
    output_dir = Path(base_dir) / f"{prefix}{suffix_part}{timestamp}"
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
    results: Dict[str, FitResult], output_dir: Path, input_file: str = None,
    approaches: list = None
) -> None:
    """Save comparison table of all approaches.

    Parameters
    ----------
    results : dict
        Dictionary mapping approach keys to FitResult objects
    output_dir : Path
        Directory to save output files
    input_file : str, optional
        Path to input data file for annotation
    approaches : list, optional
        List of approach keys to include, in desired order.
        If None, uses all approaches in results dict order.
    """
    # Determine approach order
    if approaches is None:
        approach_order = list(results.keys())
    else:
        # Filter to approaches that exist in results
        approach_order = [a for a in approaches if a in results]

    rows = []
    for name in approach_order:
        result = results[name]
        row = {
            'Approach': result.approach,
            'h1': result.h1,
            'h1_SE': result.h1_se,
            'h2': result.h2,
            'h2_SE': result.h2_se,
            'T_opt': getattr(result, 'T_opt', None),
            'T_opt_SE': getattr(result, 'T_opt_se', None),
            'h4': None,  # For piecewise (Approach2*) and persistence (Approach3*)
            'h4_SE': None,
            'R_squared': result.r_squared,
            'Total_R_squared': result.total_r_squared,
            'Adj_R_squared': result.adj_r_squared,
            'RMSE': result.rmse,
            'n_obs': result.n_obs,
            'n_params': result.n_params,
        }
        # Add h4 for piecewise, segmented, three-interval, and persistence approaches
        if is_three_interval_result(result) or is_piecewise_result(result) or is_segmented_result(result) or is_persistence_result(result):
            row['h4'] = result.h4
            row['h4_SE'] = result.h4_se
        # Add T_crit_low and T_crit_high for three-interval approaches
        if is_three_interval_result(result):
            row['T_crit_low'] = result.T_crit_low
            row['T_crit_high'] = result.T_crit_high
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add input file as first row comment in CSV
    csv_path = output_dir / 'comparison_table.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)

    # Build variance decomposition DataFrame
    # Decomposition: dy = h(T) + j + k + err
    # All terms divided by Var(dy) so they sum to 1.0
    vd_rows = []
    for name in approach_order:
        result = results[name]
        va = result.var_attrib
        if va is None:
            continue
        var_dy = va['var_dy']
        if var_dy <= 0:
            continue
        vd_row = {
            'Approach': result.approach,
            'Var(dy)': var_dy,
            'Var(h(T))/Var(dy)': va['Sigma_h_h'] / var_dy,
            'Var(j)/Var(dy)': va['Sigma_j_j'] / var_dy,
            'Var(k)/Var(dy)': va['Sigma_k_k'] / var_dy,
            'Var(err)/Var(dy)': va['Sigma_epsilon_epsilon'] / var_dy,
            '2Cov(h(T),j)/Var(dy)': 2 * va['Sigma_h_j'] / var_dy,
            '2Cov(h(T),k)/Var(dy)': 2 * va['Sigma_h_k'] / var_dy,
            '2Cov(h(T),err)/Var(dy)': 2 * va['Sigma_h_epsilon'] / var_dy,
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
            # Special handling for Approach 6a/6b (separate actual T/departure response)
            # h1,h2 = actual T response; h3,h4 = departure response
            if hasattr(result, 'h3') and hasattr(result, 'T_dep_opt') and not hasattr(result, 'f2'):
                f.write(f"  h1 = {result.h1:.6f}  (SE: {result.h1_se:.6f})  [actual T linear]\n")
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [actual T quadratic]\n")
                f.write(f"  h3 = {result.h3:.6f}  (SE: {result.h3_se:.6f})  [departure linear]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [departure quadratic]\n")
                if not np.isnan(result.T_opt):
                    f.write(f"  T_opt = {result.T_opt:.2f} C  [optimal actual T]\n")
                else:
                    f.write(f"  T_opt = N/A\n")
                if not np.isnan(result.T_dep_opt):
                    f.write(f"  T_dep_opt = {result.T_dep_opt:.2f} C  [optimal departure]\n")
                else:
                    f.write(f"  T_dep_opt = N/A\n")
            # Special handling for Approach 6c (departure/trend decomposition)
            # h1,h2 = departure response; h3,h4 = trend response; T_dep_opt = opt departure; f2 = opt trend T
            elif hasattr(result, 'h3') and hasattr(result, 'f2'):
                f.write(f"  h1 = {result.h1:.6f}  (SE: {result.h1_se:.6f})  [departure linear]\n")
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [departure quadratic]\n")
                f.write(f"  h3 = {result.h3:.6f}  (SE: {result.h3_se:.6f})  [trend T linear]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [trend T quadratic]\n")
                if not np.isnan(result.T_dep_opt):
                    f.write(f"  T_dep_opt = {result.T_dep_opt:.2f} C  [optimal departure]\n")
                else:
                    f.write(f"  T_dep_opt = N/A\n")
                if not np.isnan(result.f2):
                    f.write(f"  f2 = {result.f2:.2f} C  [optimal trend T]\n")
                else:
                    f.write(f"  f2 = N/A\n")
            # Special handling for Approach 8a (shared T_opt, total/trend)
            # h2 = curvature for actual T; h4 = curvature for trend T
            elif hasattr(result, 'h4') and hasattr(result, 'T_opt_se') and result.h1 == 0.0 and not is_piecewise_result(result) and not is_segmented_result(result) and not is_three_interval_result(result):
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [actual T curvature]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [trend T curvature]\n")
                T_opt_se = result.T_opt_se if not np.isnan(result.T_opt_se) else 0.0
                f.write(f"  T_opt = {result.T_opt:.4f}  (SE: {T_opt_se:.4f})\n")
            # Special handling for three-interval
            elif is_three_interval_result(result):
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [slope below T_crit_low]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [slope above T_crit_high]\n")
                f.write(f"  T_crit_low = {result.T_crit_low:.4f}\n")
                f.write(f"  T_crit_high = {result.T_crit_high:.4f}\n")
                T_opt_str = f"{result.T_opt:.4f}" if not np.isnan(result.T_opt) else "N/A"
                f.write(f"  T_opt = {T_opt_str}\n")
            # Special handling for segmented linear
            # h2 = slope below T_opt; h4 = slope above T_opt
            elif is_segmented_result(result):
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [slope below T_opt]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [slope above T_opt]\n")
                T_opt_se = result.T_opt_se if not np.isnan(result.T_opt_se) else 0.0
                f.write(f"  T_opt = {result.T_opt:.4f}  (SE: {T_opt_se:.4f})\n")
            # Special handling for Approach 8 (piecewise quadratic)
            # h2 = curvature below T_opt; h4 = curvature above T_opt
            elif is_piecewise_result(result):
                f.write(f"  h2 = {result.h2:.6f}  (SE: {result.h2_se:.6f})  [curvature below T_opt]\n")
                f.write(f"  h4 = {result.h4:.6f}  (SE: {result.h4_se:.6f})  [curvature above T_opt]\n")
                T_opt_se = result.T_opt_se if not np.isnan(result.T_opt_se) else 0.0
                f.write(f"  T_opt = {result.T_opt:.4f}  (SE: {T_opt_se:.4f})\n")
            else:
                # Add f1, f2 for Approach 8b/8c/8d (modulated response)
                is_modulated = not hasattr(result, 'Y_ref') and not hasattr(result, 'h3')
                if is_modulated and hasattr(result, 'f1') and result.f1 is not None:
                    f.write(f"  f1 = {result.f1:12.6f}  (SE: {result.f1_se:.6f})  [linear modulation]\n")
                if is_modulated and hasattr(result, 'f2') and result.f2 is not None:
                    f.write(f"  f2 = {result.f2:12.6f}  (SE: {result.f2_se:.6f})  [quadratic modulation]\n")
                f.write(f"  h1 = {result.h1:12.6f}  (SE: {result.h1_se:.6f})\n")
                f.write(f"  h2 = {result.h2:12.6f}  (SE: {result.h2_se:.6f})\n")
                # Add f1 for GDP-dependent approaches (5d)
                if hasattr(result, 'f1') and result.f1 is not None and hasattr(result, 'Y_ref'):
                    f.write(f"  f1 = {result.f1:10.4f}  (SE: {result.f1_se:.4f})  [GDP scaling]\n")
                    f.write(f"  Y_ref = {result.Y_ref:.2f}\n")
                if np.isnan(result.T_opt):
                    f.write(f"  T_opt = N/A\n")
                else:
                    f.write(f"  T_opt = {result.T_opt:.2f} C\n")
            f.write(f"  R² = {result.r_squared:.4f}\n")
            f.write(f"  Total R² = {result.total_r_squared:.4f}\n")
            f.write(f"  Adjusted R² = {result.adj_r_squared:.4f}\n")
            f.write(f"  RMSE = {result.rmse:.6f}\n")
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

        # Skip approaches without T_opt (e.g., 6c which decomposes into departure/trend)
        T_opt = getattr(r, 'T_opt', np.nan)
        if np.isnan(T_opt):
            continue

        # Use helper function that handles both quadratic and power-law models
        h_relative = compute_h_response(T, r)

        label = f"{r.approach} (T_opt = {T_opt:.1f}°C)"
        ax.plot(T, h_relative, color=get_color(name, 'gray'),
                linestyle=get_linestyle(name, '-'), label=label, linewidth=2)

        # Mark optimal temperature
        ax.axvline(T_opt, color=get_color(name, 'gray'),
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


def _plot_temperature_response_panels(
    results: Dict[str, FitResult], output_dir: Path,
    T_range: tuple = (0, 30), input_file: str = None
) -> None:
    """Plot h(T) - h(T*) with one subplot per approach."""
    valid = [(name, r) for name, r in results.items()
             if not np.isnan(getattr(r, 'T_opt', np.nan))]
    if not valid:
        return

    n = len(valid)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)

    T = np.linspace(T_range[0], T_range[1], TEMPERATURE_PLOT_POINTS)

    for idx, (name, r) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        h_relative = compute_h_response(T, r)
        ax.plot(T, h_relative, color=get_color(name, 'gray'),
                linestyle=get_linestyle(name, '-'), linewidth=2)
        ax.axvline(r.T_opt, color=get_color(name, 'gray'), linestyle=':', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(f"{r.approach} (T_opt = {r.T_opt:.1f}°C)", fontsize=11)
        ax.set_xlim(T_range)
        ax.grid(True, alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Temperature (°C)')
        if idx % ncols == 0:
            ax.set_ylabel('h(T) - h(T_opt)')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Temperature Response by Approach', fontsize=14)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'temperature_response_panels.pdf')
    plt.close()


def plot_temperature_response(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30),
    input_file: str = None
) -> None:
    """Plot h(T) - h(T*): combined plot and per-approach panels."""
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=list(results.keys()),
        filename='temperature_response_combined.pdf',
        title_suffix='All Approaches',
        T_range=T_range,
        input_file=input_file
    )
    _plot_temperature_response_panels(
        results, output_dir, T_range=T_range, input_file=input_file
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

        # Skip approaches without T_opt (e.g., 6c which decomposes into departure/trend)
        T_opt = getattr(r, 'T_opt', np.nan)
        if np.isnan(T_opt):
            continue

        # Use helper function that handles both quadratic and power-law models
        dh_dT = compute_dh_dT(T, r)
        label = f"{r.approach}"
        ax.plot(T, dh_dT, color=get_color(name, 'gray'),
                linestyle=get_linestyle(name, '-'), label=label, linewidth=2)

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


def _plot_temperature_derivative_panels(
    results: Dict[str, FitResult], output_dir: Path,
    T_range: tuple = (0, 30), input_file: str = None
) -> None:
    """Plot dh/dT with one subplot per approach."""
    valid = [(name, r) for name, r in results.items()
             if not np.isnan(getattr(r, 'T_opt', np.nan))]
    if not valid:
        return

    n = len(valid)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)

    T = np.linspace(T_range[0], T_range[1], TEMPERATURE_PLOT_POINTS)

    for idx, (name, r) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        dh_dT = compute_dh_dT(T, r)
        ax.plot(T, dh_dT, color=get_color(name, 'gray'),
                linestyle=get_linestyle(name, '-'), linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(r.approach, fontsize=11)
        ax.set_xlim(T_range)
        ax.grid(True, alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Temperature (°C)')
        if idx % ncols == 0:
            ax.set_ylabel('dh/dT')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Temperature Derivative by Approach', fontsize=14)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / 'temperature_derivative_panels.pdf')
    plt.close()


def plot_temperature_derivative(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30),
    input_file: str = None
) -> None:
    """Plot dh/dT: combined plot and per-approach panels."""
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=list(results.keys()),
        filename='temperature_derivative_combined.pdf',
        title_suffix='All Approaches',
        T_range=T_range,
        input_file=input_file
    )
    _plot_temperature_derivative_panels(
        results, output_dir, T_range=T_range, input_file=input_file
    )


def plot_coefficient_comparison(
    results: Dict[str, FitResult], output_dir: Path, input_file: str = None
) -> None:
    """Plot T_opt and h2 coefficients for each approach."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    approaches = list(results.keys())
    labels = [results[a].approach for a in approaches]

    # T_opt values — filter out NaN and approaches without T_opt (e.g., Approach 6c)
    valid_t = [(a, results[a]) for a in approaches
               if hasattr(results[a], 'T_opt') and not np.isnan(getattr(results[a], 'T_opt', np.nan))]
    t_labels = [r.approach for _, r in valid_t]
    T_opt_vals = [r.T_opt for _, r in valid_t]
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

    # Filter out approaches with NaN T_opt or missing T_opt (e.g., Approach 6c)
    valid = [(a, results[a]) for a in results.keys()
             if hasattr(results[a], 'T_opt') and not np.isnan(getattr(results[a], 'T_opt', np.nan))]
    labels = [r.approach for _, r in valid]
    T_opt = [r.T_opt for _, r in valid]
    valid_keys = [a for a, _ in valid]

    colors = [get_color(a, 'gray') for a in valid_keys]
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

    # Plot year means k[t] used by methods 1-4 as a heavy line
    for name in ('Approach QP', 'Approach QL', 'Approach PL'):
        if name in results:
            k_year_means = np.array([results[name].k[yr] for yr in unique_years])
            ax.plot(unique_years, k_year_means, color='black', linestyle='-', linewidth=3,
                    label='Year means k[t] (Methods 1-4)')
            break  # Only plot once since all share the same k

    for name, result in results.items():
        # Skip methods that use the same k values as Approach QP (already plotted above)
        if name in ('Approach QP', 'Approach QL', 'Approach PL'):
            continue

        # k is stored with actual year as key
        k_values = np.array([result.k[yr] for yr in unique_years])

        if name == 'Approach QJ':
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

        ax.plot(unique_years, k_values_plot, color=get_color(name, 'gray'),
                linestyle=get_linestyle(name, '-'), linewidth=1.5,
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
        n_residuals = len(residuals)
        n_data = len(data.growth_pcGDP)

        # Histogram of residuals
        axes[0, 0].hist(residuals, bins=HISTOGRAM_BINS, density=True, alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].axvline(0, color='red', linestyle='--')

        # Residuals vs fitted - handle length mismatch for methods with lagged data
        if n_residuals == n_data:
            fitted = data.growth_pcGDP - residuals
            temp_for_plot = data.temp
        else:
            # For methods with fewer residuals (e.g., persistence decay with lagged data),
            # use only the last n_residuals observations (assuming they correspond to later years)
            fitted = data.growth_pcGDP[-n_residuals:] - residuals
            temp_for_plot = data.temp[-n_residuals:]

        axes[0, 1].scatter(fitted, residuals, alpha=0.3, s=1)
        axes[0, 1].axhline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Fitted')

        # Residuals vs temperature
        axes[1, 0].scatter(temp_for_plot, residuals, alpha=0.3, s=1)
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
    """Plot the GDP scaling factor (Y/Y_ref)^(-beta) for GDP-dependent approaches.

    This shows how the temperature response is scaled by per capita GDP level.
    Countries with lower GDP have larger scaling factors (more affected).
    Currently disabled (no GDP-dependent approaches in panels list).

    Args:
        results: Dictionary of FitResult objects
        output_dir: Output directory
        data: AnalysisData for adding GDP histogram (optional)
        Y_range: GDP range for x-axis (default: from data min to max)
        input_file: Path to input data file (for annotation)
    """
    # Collect panels to plot (no GDP-dependent approaches currently)
    panels = []
    for key, title, color in [
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
    approaches: list = None,
) -> Path:
    """Save all outputs to the specified directory.

    Args:
        data: AnalysisData object
        trends: CountryTrends object
        results: Dictionary of FitResult objects
        output_dir: Output directory (created if None)
        input_file: Path to input data file (for annotation)
        approaches: List of approach keys to include, in desired order

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = create_output_dir()

    print(f"Saving outputs to: {output_dir}")

    # Save tables
    save_summary_table(results, output_dir, input_file=input_file, approaches=approaches)
    save_country_trends(data, trends, output_dir, input_file=input_file)

    # Generate plots
    plot_temperature_response(results, output_dir, input_file=input_file)
    plot_temperature_derivative(results, output_dir, input_file=input_file)
    plot_coefficient_comparison(results, output_dir, input_file=input_file)
    plot_optimal_temperature_comparison(results, output_dir, input_file=input_file)
    plot_year_effects(results, data, output_dir, input_file=input_file)
    plot_residual_diagnostics(results, data, output_dir, input_file=input_file)


    print("All outputs saved.")
    return output_dir


def save_bootstrap_coefficients_csv(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap samples to CSV for each approach.

    Creates: bootstrap_coefficients.csv with columns:
    - iteration (-1 for point estimate, 0+ for bootstrap)
    - approach (uses approach naming for publication methods, method naming for exploratory)
    - approach_name (human-readable description)
    - h1, h2, T_opt, r_squared, total_r_squared
    - f1 (GDP scaling exponent for approach 5d, linear modulation for 8b/8c)
    - h4 (curvature above T_opt for approach 8)
    - h3, h4 (trend coefficients for approaches 6a/6b/6c)
    - T_dep_opt (optimal departure for approaches 6a/6b/6c)
    - f2 (optimal trend temp for approach 6c)
    """
    rows = []
    for name, result in results.items():
        # Use the approach/method key directly
        approach_key = name
        # Write point estimates as iteration -1
        point_row = {
            'iteration': -1,
            'approach': approach_key,
            'approach_name': result.approach,
            'h1': result.h1_point,
            'h2': result.h2_point,
            'T_opt': result.T_opt_point,
            'r_squared': result.r_squared_point,
            'total_r_squared': result.total_r_squared_point,
        }
        if result.f1_point is not None:
            point_row['f1'] = result.f1_point
        if result.h4_point is not None:
            point_row['h4'] = result.h4_point
        if result.h3_point is not None:
            point_row['h3'] = result.h3_point
        if result.f2_point is not None:
            point_row['f2'] = result.f2_point
        if getattr(result, 'T_crit_low_point', None) is not None:
            point_row['T_crit_low'] = result.T_crit_low_point
        if getattr(result, 'T_crit_high_point', None) is not None:
            point_row['T_crit_high'] = result.T_crit_high_point
        rows.append(point_row)

        # Write bootstrap samples (iteration 0, 1, ..., N-1)
        for i in range(result.n_bootstrap):
            row = {
                'iteration': i,
                'approach': approach_key,
                'approach_name': result.approach,
                'h1': result.h1_samples[i],
                'h2': result.h2_samples[i],
                'T_opt': result.T_opt_samples[i],
                'r_squared': result.r_squared_samples[i],
                'total_r_squared': result.total_r_squared_samples[i],
            }
            # Add f1 for approach 5d (GDP scaling exponent)
            if result.f1_samples is not None:
                row['f1'] = result.f1_samples[i]
            # Add h4 for Approach 8 (piecewise quadratic - curvature above T_opt)
            if result.h4_samples is not None and not np.isnan(result.h4_samples[i]):
                row['h4'] = result.h4_samples[i]
            # Add h3 for approaches 6a/6b/6c (trend linear coefficient)
            if result.h3_samples is not None and not np.isnan(result.h3_samples[i]):
                row['h3'] = result.h3_samples[i]
            # Add f2 for approach 6c (optimal trend temperature)
            if result.f2_samples is not None and not np.isnan(result.f2_samples[i]):
                row['f2'] = result.f2_samples[i]
            # Add T_crit_low and T_crit_high for three-interval approaches
            T_crit_low_samp = getattr(result, 'T_crit_low_samples', None)
            T_crit_high_samp = getattr(result, 'T_crit_high_samples', None)
            if T_crit_low_samp is not None and not np.isnan(T_crit_low_samp[i]):
                row['T_crit_low'] = T_crit_low_samp[i]
            if T_crit_high_samp is not None and not np.isnan(T_crit_high_samp[i]):
                row['T_crit_high'] = T_crit_high_samp[i]
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

            # T_opt (not available for all approaches, e.g., Approach 6c uses T_dep_opt/f2)
            if result.T_opt_point is not None:
                f.write(f"  T_opt (Optimal Temperature, C):\n")
                f.write(f"    Point estimate:  {result.T_opt_point:10.2f}\n")
                f.write(f"    Bootstrap median:{stats['T_opt']['p50']:10.2f}\n")
                f.write(f"    90% CI:          [{stats['T_opt']['p5']:8.2f}, {stats['T_opt']['p95']:8.2f}]\n")
                f.write(f"    IQR:             [{stats['T_opt']['p25']:8.2f}, {stats['T_opt']['p75']:8.2f}]\n")
                f.write(f"    Std:             {stats['T_opt']['std']:10.4f}\n")

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

            # f1 (GDP scaling exponent for approach 5d, linear modulation for 8b/8c)
            if result.f1_point is not None and 'f1' in stats:
                f.write(f"  f1 (GDP scaling exponent / linear modulation):\n")
                f.write(f"    Point estimate:  {result.f1_point:10.4f}\n")
                f.write(f"    Bootstrap median:{stats['f1']['p50']:10.4f}\n")
                f.write(f"    90% CI:          [{stats['f1']['p5']:10.4f}, {stats['f1']['p95']:10.4f}]\n")
                f.write(f"    IQR:             [{stats['f1']['p25']:10.4f}, {stats['f1']['p75']:10.4f}]\n")
                f.write(f"    Std:             {stats['f1']['std']:10.4f}\n")

            # T_dep_opt (optimal departure for approaches 6a/6b/6c)
            if result.T_dep_opt_point is not None and 'T_dep_opt' in stats:
                f.write(f"  T_dep_opt (Optimal departure, C):\n")
                f.write(f"    Point estimate:  {result.T_dep_opt_point:10.2f}\n")
                f.write(f"    Bootstrap median:{stats['T_dep_opt']['p50']:10.2f}\n")
                f.write(f"    90% CI:          [{stats['T_dep_opt']['p5']:10.2f}, {stats['T_dep_opt']['p95']:10.2f}]\n")
                f.write(f"    IQR:             [{stats['T_dep_opt']['p25']:10.2f}, {stats['T_dep_opt']['p75']:10.2f}]\n")
                f.write(f"    Std:             {stats['T_dep_opt']['std']:10.2f}\n")

            # h4 (curvature above T_opt for Approach 8 piecewise quadratic)
            # Note: h2 is the curvature below T_opt, h4 is curvature above
            if result.h4_point is not None and 'h4' in stats and result.h3_point is None:
                f.write(f"  h4 (Curvature for T > T_opt):\n")
                f.write(f"    Point estimate:  {result.h4_point:10.6f}\n")
                f.write(f"    Bootstrap median:{stats['h4']['p50']:10.6f}\n")
                f.write(f"    90% CI:          [{stats['h4']['p5']:10.6f}, {stats['h4']['p95']:10.6f}]\n")
                f.write(f"    IQR:             [{stats['h4']['p25']:10.6f}, {stats['h4']['p75']:10.6f}]\n")
                f.write(f"    Std:             {stats['h4']['std']:10.6f}\n")

            # For Approach 8a: h2 is curvature for actual T, h4 is curvature for trend T
            # These are covered by the universal h2 output above and h4 below

            # h3, h4, T_dep_opt (for Approach 6a/6b - departure coefficients and optimal departure)
            # h1/h2 are universal and output above; T_opt is for actual T response
            if result.h3_point is not None and 'h3' in stats:
                f.write(f"  h3 (Linear coef for trend response T_trend):\n")
                f.write(f"    Point estimate:  {result.h3_point:10.6f}\n")
                f.write(f"    Bootstrap median:{stats['h3']['p50']:10.6f}\n")
                f.write(f"    90% CI:          [{stats['h3']['p5']:10.6f}, {stats['h3']['p95']:10.6f}]\n")
                f.write(f"    Std:             {stats['h3']['std']:10.6f}\n")
            if result.h4_point is not None and 'h4' in stats and result.h3_point is not None:
                f.write(f"  h4 (Quadratic coef for trend response T_trend):\n")
                f.write(f"    Point estimate:  {result.h4_point:10.6f}\n")
                f.write(f"    Bootstrap median:{stats['h4']['p50']:10.6f}\n")
                f.write(f"    90% CI:          [{stats['h4']['p5']:10.6f}, {stats['h4']['p95']:10.6f}]\n")
                f.write(f"    Std:             {stats['h4']['std']:10.6f}\n")

            # f2 (for Approach 6c - optimal trend temperature)
            if result.f2_point is not None and 'f2' in stats:
                f.write(f"  f2 (Optimal trend temperature, C):\n")
                f.write(f"    Point estimate:  {result.f2_point:10.2f}\n")
                f.write(f"    Bootstrap median:{stats['f2']['p50']:10.2f}\n")
                f.write(f"    90% CI:          [{stats['f2']['p5']:8.2f}, {stats['f2']['p95']:8.2f}]\n")
                f.write(f"    IQR:             [{stats['f2']['p25']:8.2f}, {stats['f2']['p75']:8.2f}]\n")
                f.write(f"    Std:             {stats['f2']['std']:10.4f}\n")

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
    - Point estimate, median, p5, p25, p75, p95, std for h1, h2, T_opt, total_r_squared
    - f1 statistics included for approaches where it's used (5d, 8b, 8c)
    - T_dep_opt statistics for approaches 6a/6b/6c (optimal departure)
    - f2 statistics included for approach 6c (T_opt_trend) or 8b/8d (quadratic modulation)
    """
    rows = []
    for name, result in results.items():
        stats = all_stats[name]

        # Use the approach/method key directly
        approach_key = name

        row = {
            'approach': approach_key,
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

            # T_opt statistics
            'T_opt_point': result.T_opt_point,
            'T_opt_median': stats['T_opt']['p50'],
            'T_opt_p5': stats['T_opt']['p5'],
            'T_opt_p25': stats['T_opt']['p25'],
            'T_opt_p75': stats['T_opt']['p75'],
            'T_opt_p95': stats['T_opt']['p95'],
            'T_opt_std': stats['T_opt']['std'],

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

        # Add f1 statistics for approaches where it's used (5d, 8b, 8c)
        if result.f1_point is not None and 'f1' in stats:
            row['f1_point'] = result.f1_point
            row['f1_median'] = stats['f1']['p50']
            row['f1_p5'] = stats['f1']['p5']
            row['f1_p25'] = stats['f1']['p25']
            row['f1_p75'] = stats['f1']['p75']
            row['f1_p95'] = stats['f1']['p95']
            row['f1_std'] = stats['f1']['std']
        else:
            # Fill with NaN for approaches without f1
            row['f1_point'] = np.nan
            row['f1_median'] = np.nan
            row['f1_p5'] = np.nan
            row['f1_p25'] = np.nan
            row['f1_p75'] = np.nan
            row['f1_p95'] = np.nan
            row['f1_std'] = np.nan

        # Add T_dep_opt statistics for approaches 6a/6b/6c (optimal departure)
        if result.T_dep_opt_point is not None and 'T_dep_opt' in stats:
            row['T_dep_opt_point'] = result.T_dep_opt_point
            row['T_dep_opt_median'] = stats['T_dep_opt']['p50']
            row['T_dep_opt_p5'] = stats['T_dep_opt']['p5']
            row['T_dep_opt_p25'] = stats['T_dep_opt']['p25']
            row['T_dep_opt_p75'] = stats['T_dep_opt']['p75']
            row['T_dep_opt_p95'] = stats['T_dep_opt']['p95']
            row['T_dep_opt_std'] = stats['T_dep_opt']['std']
        else:
            # Fill with NaN for approaches without T_dep_opt
            row['T_dep_opt_point'] = np.nan
            row['T_dep_opt_median'] = np.nan
            row['T_dep_opt_p5'] = np.nan
            row['T_dep_opt_p25'] = np.nan
            row['T_dep_opt_p75'] = np.nan
            row['T_dep_opt_p95'] = np.nan
            row['T_dep_opt_std'] = np.nan

        # Add h3 statistics for approaches 6a/6b/6c (trend linear coefficient)
        if result.h3_point is not None and 'h3' in stats:
            row['h3_point'] = result.h3_point
            row['h3_median'] = stats['h3']['p50']
            row['h3_p5'] = stats['h3']['p5']
            row['h3_p25'] = stats['h3']['p25']
            row['h3_p75'] = stats['h3']['p75']
            row['h3_p95'] = stats['h3']['p95']
            row['h3_std'] = stats['h3']['std']
        else:
            row['h3_point'] = np.nan
            row['h3_median'] = np.nan
            row['h3_p5'] = np.nan
            row['h3_p25'] = np.nan
            row['h3_p75'] = np.nan
            row['h3_p95'] = np.nan
            row['h3_std'] = np.nan

        # Add h4 statistics for approaches 6a/6b/6c/8/8a (trend or high-T curvature)
        if result.h4_point is not None and 'h4' in stats:
            row['h4_point'] = result.h4_point
            row['h4_median'] = stats['h4']['p50']
            row['h4_p5'] = stats['h4']['p5']
            row['h4_p25'] = stats['h4']['p25']
            row['h4_p75'] = stats['h4']['p75']
            row['h4_p95'] = stats['h4']['p95']
            row['h4_std'] = stats['h4']['std']
        else:
            row['h4_point'] = np.nan
            row['h4_median'] = np.nan
            row['h4_p5'] = np.nan
            row['h4_p25'] = np.nan
            row['h4_p75'] = np.nan
            row['h4_p95'] = np.nan
            row['h4_std'] = np.nan

        # Add f2 statistics for approach 6c (optimal trend T) or 8b/8d (quadratic modulation)
        if result.f2_point is not None and 'f2' in stats:
            row['f2_point'] = result.f2_point
            row['f2_median'] = stats['f2']['p50']
            row['f2_p5'] = stats['f2']['p5']
            row['f2_p25'] = stats['f2']['p25']
            row['f2_p75'] = stats['f2']['p75']
            row['f2_p95'] = stats['f2']['p95']
            row['f2_std'] = stats['f2']['std']
        else:
            row['f2_point'] = np.nan
            row['f2_median'] = np.nan
            row['f2_p5'] = np.nan
            row['f2_p25'] = np.nan
            row['f2_p75'] = np.nan
            row['f2_p95'] = np.nan
            row['f2_std'] = np.nan

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


def save_bootstrap_k_samples_csv(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap samples of year fixed effects k(t) to CSV.

    Creates: bootstrap_k_samples.csv with columns:
    - iteration: bootstrap iteration number
    - approach: approach key (uses approach naming for publication methods)
    - approach_name: human-readable approach name
    - year: calendar year
    - k_value: k(t) value for that year in that bootstrap iteration
    """
    rows = []
    for name, result in results.items():
        if result.k_samples is None:
            continue

        # Use the approach/method key directly
        approach_key = name

        # Add point estimates as iteration -1
        if result.k_point is not None:
            for year in sorted(result.k_point.keys()):
                rows.append({
                    'iteration': -1,
                    'approach': approach_key,
                    'approach_name': result.approach,
                    'year': year,
                    'k_value': result.k_point[year],
                })

        for year in sorted(result.k_samples.keys()):
            k_array = result.k_samples[year]
            for i in range(result.n_bootstrap):
                rows.append({
                    'iteration': i,
                    'approach': approach_key,
                    'approach_name': result.approach,
                    'year': year,
                    'k_value': k_array[i],
                })

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'bootstrap_k_samples.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_k_samples.csv ({len(df)} rows)")


def save_bootstrap_var_attrib_csv(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap samples of variance attribution to CSV.

    Creates: bootstrap_var_attrib_samples.csv with columns:
    - iteration: bootstrap iteration number (-1 for point estimate, 0+ for bootstrap)
    - approach: approach key (uses approach naming for publication methods)
    - approach_name: human-readable approach name
    - All var_attrib keys (Sigma_Delta_u_Delta_u, Sigma_Delta_u_v, etc.)

    This allows reconstructing variance attribution confidence intervals
    in postprocessing for publication tables.
    """
    rows = []
    for name, result in results.items():
        if result.var_attrib_samples is None:
            continue

        # Use the approach/method key directly
        approach_key = name

        # Get all keys from var_attrib_samples
        keys = sorted(result.var_attrib_samples.keys())
        if not keys:
            continue

        # Write point estimates as iteration -1
        if result.var_attrib_point is not None:
            row = {
                'iteration': -1,
                'approach': approach_key,
                'approach_name': result.approach,
            }
            for key in keys:
                row[key] = result.var_attrib_point.get(key, np.nan)
            rows.append(row)

        # Write bootstrap samples (iteration 0, 1, ..., N-1)
        for i in range(result.n_bootstrap):
            row = {
                'iteration': i,
                'approach': approach_key,
                'approach_name': result.approach,
            }
            for key in keys:
                row[key] = result.var_attrib_samples[key][i]
            rows.append(row)

    if not rows:
        print("  No var_attrib samples to save")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'bootstrap_var_attrib_samples.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_var_attrib_samples.csv ({len(df)} rows)")


def save_variance_decomposition_table(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save variance decomposition table with bootstrap statistics.

    Creates: variance_decomposition_table.xlsx and .csv with columns:
    - Metric: variance/covariance metric name
    - For each approach: point, p5, p25, p50, p75, p95 statistics

    This table includes ALL approaches in the bootstrap results.
    """
    # Define variance metrics (using combined h(T) instead of separated h(T)-h(Ttr) and h(Ttr))
    variance_metrics = [
        ('Sigma_h_h', 'Var(h(T))/Var(Δy)'),
        ('Sigma_j_j', 'Var(j)/Var(Δy)'),
        ('Sigma_k_k', 'Var(k)/Var(Δy)'),
        ('Sigma_epsilon_epsilon', 'Var(ε)/Var(Δy)'),
    ]
    covariance_metrics = [
        ('Sigma_h_j', '2Cov(h(T),j)/Var(Δy)'),
        ('Sigma_h_k', '2Cov(h(T),k)/Var(Δy)'),
        ('Sigma_h_epsilon', '2Cov(h(T),ε)/Var(Δy)'),
        ('Sigma_j_k', '2Cov(j,k)/Var(Δy)'),
        ('Sigma_j_epsilon', '2Cov(j,ε)/Var(Δy)'),
        ('Sigma_k_epsilon', '2Cov(k,ε)/Var(Δy)'),
    ]

    def compute_stats(samples: np.ndarray, point_estimate: float = None) -> dict:
        """Compute percentiles from samples, with optional explicit point estimate.

        If point_estimate is provided, use it directly. Otherwise use median of samples.
        """
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            return {'point': point_estimate if point_estimate is not None else np.nan,
                    'p5': np.nan, 'p25': np.nan, 'p50': np.nan, 'p75': np.nan, 'p95': np.nan}
        return {
            'point': point_estimate if point_estimate is not None else np.median(valid_samples),
            'p5': np.percentile(valid_samples, 5),
            'p25': np.percentile(valid_samples, 25),
            'p50': np.percentile(valid_samples, 50),
            'p75': np.percentile(valid_samples, 75),
            'p95': np.percentile(valid_samples, 95),
        }

    def get_metric_samples(var_attrib: dict, key: str) -> np.ndarray:
        """Get metric samples, computing combined h(T) terms from separated terms if needed."""
        # If the key exists directly, use it
        if key in var_attrib:
            return var_attrib[key]

        # Compute combined h(T) terms from separated Delta_u and v terms
        if key == 'Sigma_h_h':
            if all(k in var_attrib for k in ['Sigma_Delta_u_Delta_u', 'Sigma_v_v', 'Sigma_Delta_u_v']):
                return (var_attrib['Sigma_Delta_u_Delta_u'] +
                        var_attrib['Sigma_v_v'] +
                        2 * var_attrib['Sigma_Delta_u_v'])
        elif key == 'Sigma_h_j':
            if all(k in var_attrib for k in ['Sigma_Delta_u_j', 'Sigma_v_j']):
                return var_attrib['Sigma_Delta_u_j'] + var_attrib['Sigma_v_j']
        elif key == 'Sigma_h_k':
            if all(k in var_attrib for k in ['Sigma_Delta_u_k', 'Sigma_v_k']):
                return var_attrib['Sigma_Delta_u_k'] + var_attrib['Sigma_v_k']
        elif key == 'Sigma_h_epsilon':
            if all(k in var_attrib for k in ['Sigma_Delta_u_epsilon', 'Sigma_v_epsilon']):
                return var_attrib['Sigma_Delta_u_epsilon'] + var_attrib['Sigma_v_epsilon']

        return None

    def get_metric_point(var_attrib_point: dict, key: str) -> float:
        """Get point estimate for a metric, computing combined h(T) terms if needed."""
        if var_attrib_point is None:
            return np.nan
        # If the key exists directly, use it
        if key in var_attrib_point:
            return var_attrib_point[key]

        # Compute combined h(T) terms from separated Delta_u and v terms
        if key == 'Sigma_h_h':
            if all(k in var_attrib_point for k in ['Sigma_Delta_u_Delta_u', 'Sigma_v_v', 'Sigma_Delta_u_v']):
                return (var_attrib_point['Sigma_Delta_u_Delta_u'] +
                        var_attrib_point['Sigma_v_v'] +
                        2 * var_attrib_point['Sigma_Delta_u_v'])
        elif key == 'Sigma_h_j':
            if all(k in var_attrib_point for k in ['Sigma_Delta_u_j', 'Sigma_v_j']):
                return var_attrib_point['Sigma_Delta_u_j'] + var_attrib_point['Sigma_v_j']
        elif key == 'Sigma_h_k':
            if all(k in var_attrib_point for k in ['Sigma_Delta_u_k', 'Sigma_v_k']):
                return var_attrib_point['Sigma_Delta_u_k'] + var_attrib_point['Sigma_v_k']
        elif key == 'Sigma_h_epsilon':
            if all(k in var_attrib_point for k in ['Sigma_Delta_u_epsilon', 'Sigma_v_epsilon']):
                return var_attrib_point['Sigma_Delta_u_epsilon'] + var_attrib_point['Sigma_v_epsilon']

        return np.nan

    # Get approaches that have var_attrib_samples
    available_approaches = [name for name, result in results.items()
                           if result.var_attrib_samples is not None]
    if not available_approaches:
        print("  No variance attribution data available")
        return

    # Build approach names mapping
    approach_names = {name: results[name].approach for name in available_approaches}

    # Build table rows
    rows = []

    # Add Total R² row first
    total_r2_row = {'Metric': 'Total R²'}
    for approach in available_approaches:
        name = approach_names[approach]
        result = results[approach]
        point_estimate = result.total_r_squared_point
        samples = result.total_r_squared_samples
        stats = compute_stats(samples, point_estimate)
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
            result = results[approach]
            var_attrib = result.var_attrib_samples
            var_attrib_point = result.var_attrib_point

            # Get metric samples (may compute combined h(T) terms from separated terms)
            metric_samples = get_metric_samples(var_attrib, key)
            if metric_samples is None or 'var_dy' not in var_attrib:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            var_dy_samples = var_attrib['var_dy']

            # Compute point estimate from var_attrib_point (normalized by var_dy)
            if var_attrib_point is not None:
                metric_point = get_metric_point(var_attrib_point, key)
                var_dy_point = var_attrib_point.get('var_dy', np.nan) if var_attrib_point else np.nan
                with np.errstate(divide='ignore', invalid='ignore'):
                    point_estimate = metric_point / var_dy_point if var_dy_point else np.nan
            else:
                point_estimate = None

            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = metric_samples / var_dy_samples
            normalized = np.where(np.isfinite(normalized), normalized, np.nan)
            stats = compute_stats(normalized, point_estimate)
            row[f'{name}_point'] = stats['point']
            row[f'{name}_p5'] = stats['p5']
            row[f'{name}_p25'] = stats['p25']
            row[f'{name}_p50'] = stats['p50']
            row[f'{name}_p75'] = stats['p75']
            row[f'{name}_p95'] = stats['p95']
        rows.append(row)

    # Add covariance metrics (multiply by 2)
    for key, label in covariance_metrics:
        row = {'Metric': label}
        for approach in available_approaches:
            name = approach_names[approach]
            result = results[approach]
            var_attrib = result.var_attrib_samples
            var_attrib_point = result.var_attrib_point

            # Get metric samples (may compute combined h(T) terms from separated terms)
            metric_samples = get_metric_samples(var_attrib, key)
            if metric_samples is None or 'var_dy' not in var_attrib:
                row[f'{name}_point'] = np.nan
                row[f'{name}_p5'] = np.nan
                row[f'{name}_p25'] = np.nan
                row[f'{name}_p50'] = np.nan
                row[f'{name}_p75'] = np.nan
                row[f'{name}_p95'] = np.nan
                continue

            var_dy_samples = var_attrib['var_dy']

            # Compute point estimate from var_attrib_point (normalized by var_dy, with 2x factor)
            if var_attrib_point is not None:
                metric_point = get_metric_point(var_attrib_point, key)
                var_dy_point = var_attrib_point.get('var_dy', np.nan) if var_attrib_point else np.nan
                with np.errstate(divide='ignore', invalid='ignore'):
                    point_estimate = (2 * metric_point) / var_dy_point if var_dy_point else np.nan
            else:
                point_estimate = None

            metric_samples = metric_samples * 2  # 2*Cov term
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = metric_samples / var_dy_samples
            normalized = np.where(np.isfinite(normalized), normalized, np.nan)
            stats = compute_stats(normalized, point_estimate)
            row[f'{name}_point'] = stats['point']
            row[f'{name}_p5'] = stats['p5']
            row[f'{name}_p25'] = stats['p25']
            row[f'{name}_p50'] = stats['p50']
            row[f'{name}_p75'] = stats['p75']
            row[f'{name}_p95'] = stats['p95']
        rows.append(row)

    # Add Sum row
    sum_row = {'Metric': 'Sum'}
    for approach in available_approaches:
        name = approach_names[approach]
        result = results[approach]
        var_attrib = result.var_attrib_samples
        var_attrib_point = result.var_attrib_point

        if 'var_dy' not in var_attrib:
            sum_row[f'{name}_point'] = np.nan
            sum_row[f'{name}_p5'] = np.nan
            sum_row[f'{name}_p25'] = np.nan
            sum_row[f'{name}_p50'] = np.nan
            sum_row[f'{name}_p75'] = np.nan
            sum_row[f'{name}_p95'] = np.nan
            continue

        var_dy_samples = var_attrib['var_dy']
        n_samples = len(var_dy_samples)
        total_sum = np.zeros(n_samples)

        # Compute point estimate sum from var_attrib_point
        if var_attrib_point is not None:
            var_dy_point = var_attrib_point.get('var_dy', np.nan) if var_attrib_point else np.nan
            total_sum_point = 0.0
            for key, _ in variance_metrics:
                metric_point = get_metric_point(var_attrib_point, key)
                if not np.isnan(metric_point):
                    total_sum_point += metric_point
            for key, _ in covariance_metrics:
                metric_point = get_metric_point(var_attrib_point, key)
                if not np.isnan(metric_point):
                    total_sum_point += 2 * metric_point
            with np.errstate(divide='ignore', invalid='ignore'):
                point_estimate = total_sum_point / var_dy_point if var_dy_point else np.nan
        else:
            point_estimate = None

        # Sum all variance and covariance terms (using get_metric_samples for combined h(T) terms)
        for key, _ in variance_metrics:
            samples = get_metric_samples(var_attrib, key)
            if samples is not None:
                total_sum += samples
        for key, _ in covariance_metrics:
            samples = get_metric_samples(var_attrib, key)
            if samples is not None:
                total_sum += 2 * samples

        with np.errstate(divide='ignore', invalid='ignore'):
            sum_normalized = total_sum / var_dy_samples
        sum_normalized = np.where(np.isfinite(sum_normalized), sum_normalized, np.nan)
        stats = compute_stats(sum_normalized, point_estimate)
        sum_row[f'{name}_point'] = stats['point']
        sum_row[f'{name}_p5'] = stats['p5']
        sum_row[f'{name}_p25'] = stats['p25']
        sum_row[f'{name}_p50'] = stats['p50']
        sum_row[f'{name}_p75'] = stats['p75']
        sum_row[f'{name}_p95'] = stats['p95']
    rows.append(sum_row)

    # Create and save DataFrame
    df = pd.DataFrame(rows)

    xlsx_path = output_dir / 'variance_decomposition_table.xlsx'
    df.to_excel(xlsx_path, index=False, sheet_name='Variance Decomposition')
    print(f"  Saved variance_decomposition_table.xlsx ({len(rows)} rows × {len(available_approaches)} approaches)")

    csv_path = output_dir / 'variance_decomposition_table.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
        df.to_csv(f, index=False)
    print(f"  Saved variance_decomposition_table.csv")


def save_bootstrap_country_samples_csv(
    country_samples: np.ndarray,
    data: "AnalysisData",
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap country resampling indices to CSV.

    Creates: bootstrap_country_samples.csv with shape (n_bootstrap, n_countries).
    Each row is one bootstrap iteration, each column is a "slot" in the resampled
    dataset, and the value is the original country index (0 to n_countries-1).

    This allows analysis of country influence on bootstrap results by checking
    how often each country appears in each bootstrap sample.

    Args:
        country_samples: Array of shape (n_bootstrap, n_countries) with country indices
        data: AnalysisData to get country ISO codes for column headers
        output_dir: Directory to save the CSV
        input_file: Optional input file path for annotation
    """
    n_bootstrap, n_countries = country_samples.shape

    # Create column headers using original country ISO codes
    col_names = [data.idx_to_iso.get(i, f"country_{i}") for i in range(n_countries)]

    df = pd.DataFrame(country_samples, columns=col_names)
    df.insert(0, 'iteration', range(n_bootstrap))

    csv_path = output_dir / 'bootstrap_country_samples.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
            f.write(f"# Each row is a bootstrap iteration, each column is a 'slot' in the resampled dataset\n")
            f.write(f"# Values are original country indices (0 to {n_countries-1})\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_country_samples.csv ({n_bootstrap} x {n_countries})")


def save_bootstrap_year_samples_csv(
    year_samples: np.ndarray,
    data: "AnalysisData",
    output_dir: Path,
    input_file: str = None
) -> None:
    """Save bootstrap year resampling indices to CSV.

    Creates: bootstrap_year_samples.csv with shape (n_bootstrap, n_years).
    Each row is one bootstrap iteration, each column is a "slot" in the resampled
    year set, and the value is the original year index (0 to n_years-1).

    This is only created when sample_years=True (time-dimension bootstrap).

    Args:
        year_samples: Array of shape (n_bootstrap, n_years) with year indices.
            Empty array if sample_years=False.
        data: AnalysisData to get year values for column headers
        output_dir: Directory to save the CSV
        input_file: Optional input file path for annotation
    """
    if year_samples.size == 0:
        return  # No year sampling, skip

    n_bootstrap, n_years = year_samples.shape
    unique_years = sorted(set(data.year))

    # Create column headers using actual year values
    col_names = [str(yr) for yr in unique_years]

    df = pd.DataFrame(year_samples, columns=col_names)
    df.insert(0, 'iteration', range(n_bootstrap))

    csv_path = output_dir / 'bootstrap_year_samples.csv'
    with open(csv_path, 'w') as f:
        if input_file:
            f.write(f"# Input data: {Path(input_file).name}\n")
            f.write(f"# Time-dimension bootstrap: each row is a bootstrap iteration\n")
            f.write(f"# Each column is a 'slot' in the resampled year set\n")
            f.write(f"# Values are year indices (0 to {n_years-1}), mapping to years {unique_years[0]}-{unique_years[-1]}\n")
        df.to_csv(f, index=False)
    print(f"  Saved bootstrap_year_samples.csv ({n_bootstrap} x {n_years})")


def save_bootstrap_h_values(
    h_T_samples: Dict[str, np.ndarray],
    data: AnalysisData,
    output_dir: Path,
    input_file: str = None,
    original_results: Dict[str, FitResult] = None,
    trends_loess: CountryTrendsLoess = None,
) -> None:
    """Save h(T) values by observation for each bootstrap iteration.

    Creates: bootstrap_h_values.csv with columns:
    - iteration: bootstrap iteration number (-1 for point estimate, 0+ for bootstrap)
    - approach: approach key (uses approach naming for publication methods)
    - iso3: country ISO3 code
    - year: calendar year
    - temp: temperature value
    - h_T: computed h(T) value

    The h(T) formula varies by approach:
    - Approach QJ, Approach QP, Approach QL: h(T) = h1*T + h2*T²
    - Approach PL: h(T) = h2*(T-T_opt)² if T≤T_opt else h4*(T-T_opt)²
    - Approach DL: h_conv = h1*X1 + h2*X2 with persistence-decay accumulators

    Note: This file can be large (~2GB for 1000 iterations × 5 approaches × 9405 obs).
    Consider gzip compression after generation if needed.

    Args:
        h_T_samples: Dict mapping method name to array of shape (n_bootstrap, n_obs)
        data: AnalysisData with country/year info
        output_dir: Output directory
        input_file: Input data filename for header comment
        original_results: Dict of FitResult for point estimates (iteration=-1)
    """
    if not h_T_samples:
        print("  No h(T) samples to save")
        return

    output_path = output_dir / 'bootstrap_h_values.csv'

    # Count total rows for progress reporting
    total_rows = sum(arr.shape[0] * arr.shape[1] for arr in h_T_samples.values())
    if original_results is not None:
        total_rows += len(h_T_samples) * data.n_obs

    # Pre-compute metadata arrays once (avoids repeated lookups in inner loops)
    iso3_arr = np.array([data.idx_to_iso[idx] for idx in data.country_idx])
    year_arr = data.year.astype(int)
    temp_arr = data.temp
    n_obs = data.n_obs

    # Chunk size for buffered writing (balance memory vs I/O overhead)
    CHUNK_SIZE = 50000

    with open(output_path, 'w') as f:
        if input_file:
            f.write(f'# Input data: {Path(input_file).name}\n')
        f.write('iteration,approach,iso3,year,temp,h_T\n')

        # Write point estimates (iteration = -1)
        if original_results is not None:
            for name in h_T_samples.keys():
                if name not in original_results:
                    continue
                r = original_results[name]

                # Use the approach/method key directly
                approach_key = name

                # Compute h(T) for each observation based on approach type
                response_type = name.split()[-1][0]  # 'Q', 'P', 'S', 'T', 'D', or 'L'
                if response_type == 'Q':
                    h_T_point = r.h1 * temp_arr + r.h2 * temp_arr**2
                elif response_type == 'T':
                    # Three-interval: h2*f_low + h4*f_high
                    f_low, f_high = three_interval_shape(temp_arr, r.T_crit_low, r.T_crit_high - r.T_crit_low)
                    h_T_point = r.h2 * f_low + r.h4 * f_high
                elif response_type == 'S':
                    # Segmented linear: h2*(T-T_opt) below, h4*(T-T_opt) above
                    below = temp_arr <= r.T_opt
                    h_T_point = np.where(below, r.h2 * (temp_arr - r.T_opt), r.h4 * (temp_arr - r.T_opt))
                elif response_type == 'P':
                    # Piecewise quadratic: h2 below T_opt, h4 above T_opt
                    below = temp_arr <= r.T_opt
                    h_T_point = np.where(below, r.h2 * (temp_arr - r.T_opt)**2, r.h4 * (temp_arr - r.T_opt)**2)
                elif response_type in ('D', 'L'):
                    # Persistence/level decay: h_conv(T) = h1*X1 + h2*X2
                    h4 = r.h4
                    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
                    T_loess_base = _get_T_loess_at_base_year(data, trends_loess, base_year=data.year_range[0])
                    correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, T_loess_base)
                    X1 = temp_arr - h4 * A_T_lag - correction_T
                    X2 = temp_arr**2 - h4 * A_T2_lag - correction_T2
                    h_T_point = r.h1 * X1 + r.h2 * X2
                else:
                    continue

                # Vectorized formatting for point estimates
                lines = [
                    f'-1,{approach_key},{iso3_arr[i]},{year_arr[i]},{temp_arr[i]:.4f},{h_T_point[i]:.8f}\n'
                    for i in range(n_obs)
                ]
                f.write(''.join(lines))

        # Write bootstrap samples (iteration = 0, 1, ..., N-1)
        for name, arr in h_T_samples.items():
            # Use the approach/method key directly
            approach_key = name
            n_bootstrap = arr.shape[0]
            buffer = []

            for b in range(n_bootstrap):
                h_T_row = arr[b]
                # Build lines for this bootstrap iteration
                for i in range(n_obs):
                    buffer.append(f'{b},{approach_key},{iso3_arr[i]},{year_arr[i]},{temp_arr[i]:.4f},{h_T_row[i]:.8f}\n')

                # Flush buffer when it reaches chunk size
                if len(buffer) >= CHUNK_SIZE:
                    f.write(''.join(buffer))
                    buffer = []

            # Flush remaining buffer for this approach
            if buffer:
                f.write(''.join(buffer))

    print(f"  Saved bootstrap_h_values.csv ({total_rows} rows)")


def save_bootstrap_h_baselines(
    bootstrap_results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    trends_loess: CountryTrendsLoess,
    output_dir: Path,
    base_year: int,
    input_file: str = None,
    original_results: Dict[str, FitResult] = None,
) -> None:
    """Save h(T_loess_base_year) baseline values for cumulative effects calculation.

    For each country, computes h(T_loess_base) where T_loess_base is the LOESS-smoothed
    temperature at the first year. This provides a stable baseline that isn't affected
    by inter-annual temperature variability.

    Creates: bootstrap_h_baselines.csv with columns:
    - iteration: bootstrap iteration number (-1 for point estimate, 0+ for bootstrap)
    - approach: approach key
    - iso3: country ISO3 code
    - T_loess_base: LOESS temperature at base year for this country
    - h_T_baseline: h(T_loess_base) value

    The h(T) formula varies by approach:
    - Approach QJ, Approach QP, Approach QL: h(T) = h1*T + h2*T²
    - Approach PL: h(T) = h2*(T-T_opt)² if T≤T_opt else h4*(T-T_opt)²
    - Approach DL: h_T_baseline = 0 if h4 > 0, else h1*T + h2*T²

    Note: For Approach DL with h4 > 0, constant temperature gives X1=X2=0 due to
    the persistence decay structure, so h_T_baseline = 0. If h4 = 0, Approach DL
    reduces to Approach QL and uses the standard quadratic baseline.

    Args:
        bootstrap_results: Dict mapping method name to BootstrapResult
        data: AnalysisData with country/year info
        trends_loess: CountryTrendsLoess with T_loess values
        output_dir: Output directory
        input_file: Input data filename for header comment
        original_results: Dict of FitResult for point estimates (iteration=-1)
        base_year: First year of the data (e.g., data.year_range[0])
    """
    output_path = output_dir / 'bootstrap_h_baselines.csv'

    # Pre-compute country info
    iso3_arr = np.array([data.idx_to_iso[idx] for idx in data.country_idx])
    year_arr = data.year.astype(int)
    T_loess = trends_loess.T_loess

    # Build mapping: iso3 -> T_loess at base_year
    # Find the observation index for each country at base_year
    country_T_loess_base = {}
    for c in range(data.n_countries):
        iso3 = data.idx_to_iso[c]
        country_mask = data.country_idx == c
        country_indices = np.where(country_mask)[0]
        years_for_country = year_arr[country_indices]

        # Find the base year observation
        base_year_mask = years_for_country == base_year
        if base_year_mask.any():
            base_idx = country_indices[np.where(base_year_mask)[0][0]]
            country_T_loess_base[iso3] = T_loess[base_idx]
        else:
            # If no observation at base year, use earliest year
            earliest_idx = country_indices[np.argmin(years_for_country)]
            country_T_loess_base[iso3] = T_loess[earliest_idx]

    # Approaches to process (matching h_T_samples keys)
    available_approaches = [a for a in bootstrap_results if not a.startswith('Approach N')]

    rows = []

    for approach_key in available_approaches:
        br = bootstrap_results[approach_key]
        n_bootstrap = br.n_bootstrap

        # Get point estimates
        h1_point = br.h1_point
        h2_point = br.h2_point
        h4_point = getattr(br, 'h4_point', None)
        T_opt_point = br.T_opt_point

        # Get bootstrap samples
        h1_samples = br.h1_samples
        h2_samples = br.h2_samples
        h4_samples = getattr(br, 'h4_samples', None)
        T_opt_samples = br.T_opt_samples

        # Process each country
        for iso3, T_base in country_T_loess_base.items():
            # Point estimate (iteration = -1)
            response_type = approach_key.split()[-1][0]  # 'Q', 'P', 'S', 'T', 'D', or 'L'
            if response_type == 'T':
                # Three-interval: h2*f_low + h4*f_high
                T_crit_low_pt = getattr(br, 'T_crit_low_point', None)
                T_crit_high_pt = getattr(br, 'T_crit_high_point', None)
                if T_crit_low_pt is not None and T_crit_high_pt is not None:
                    f_low, f_high = three_interval_shape(np.array([T_base]), T_crit_low_pt, T_crit_high_pt - T_crit_low_pt)
                    h_T_baseline = h2_point * f_low[0] + h4_point * f_high[0]
                else:
                    h_T_baseline = h1_point * T_base + h2_point * T_base ** 2
            elif response_type == 'S':
                # Segmented linear: h2*(T-T_opt) below, h4*(T-T_opt) above
                if T_base <= T_opt_point:
                    h_T_baseline = h2_point * (T_base - T_opt_point)
                else:
                    h_T_baseline = h4_point * (T_base - T_opt_point)
            elif response_type == 'P':
                # Piecewise: h2*(T-T_opt)² below, h4*(T-T_opt)² above
                if T_base <= T_opt_point:
                    h_T_baseline = h2_point * (T_base - T_opt_point) ** 2
                else:
                    h_T_baseline = h4_point * (T_base - T_opt_point) ** 2
            elif response_type in ('D', 'L') and h4_point > 0:
                # Persistence/level with decay: baseline = 0
                # (constant temperature gives X1=X2=0 due to persistence decay)
                h_T_baseline = 0.0
            else:
                # Standard quadratic: h1*T + h2*T²
                h_T_baseline = h1_point * T_base + h2_point * T_base ** 2

            rows.append({
                'iteration': -1,
                'approach': approach_key,
                'iso3': iso3,
                'T_loess_base': T_base,
                'h_T_baseline': h_T_baseline,
            })

            # Bootstrap samples (iteration = 0, 1, ..., N-1)
            for b in range(n_bootstrap):
                h1_b = h1_samples[b]
                h2_b = h2_samples[b]

                if response_type == 'T':
                    h4_b = h4_samples[b] if h4_samples is not None else 0.0
                    T_crit_low_samp = getattr(br, 'T_crit_low_samples', None)
                    T_crit_high_samp = getattr(br, 'T_crit_high_samples', None)
                    if T_crit_low_samp is not None and T_crit_high_samp is not None:
                        f_low_b, f_high_b = three_interval_shape(
                            np.array([T_base]), T_crit_low_samp[b], T_crit_high_samp[b] - T_crit_low_samp[b])
                        h_T_baseline_b = h2_b * f_low_b[0] + h4_b * f_high_b[0]
                    else:
                        h_T_baseline_b = h1_b * T_base + h2_b * T_base ** 2
                elif response_type == 'S':
                    h4_b = h4_samples[b] if h4_samples is not None else 0.0
                    T_opt_b = T_opt_samples[b]
                    if T_base <= T_opt_b:
                        h_T_baseline_b = h2_b * (T_base - T_opt_b)
                    else:
                        h_T_baseline_b = h4_b * (T_base - T_opt_b)
                elif response_type == 'P':
                    h4_b = h4_samples[b] if h4_samples is not None else 0.0
                    T_opt_b = T_opt_samples[b]
                    if T_base <= T_opt_b:
                        h_T_baseline_b = h2_b * (T_base - T_opt_b) ** 2
                    else:
                        h_T_baseline_b = h4_b * (T_base - T_opt_b) ** 2
                elif response_type in ('D', 'L'):
                    h4_b = h4_samples[b] if h4_samples is not None else 0.0
                    if h4_b > 0:
                        # Persistence/level: baseline = 0
                        h_T_baseline_b = 0.0
                    else:
                        h_T_baseline_b = h1_b * T_base + h2_b * T_base ** 2
                else:
                    h_T_baseline_b = h1_b * T_base + h2_b * T_base ** 2

                rows.append({
                    'iteration': b,
                    'approach': approach_key,
                    'iso3': iso3,
                    'T_loess_base': T_base,
                    'h_T_baseline': h_T_baseline_b,
                })

    # Write to CSV
    df = pd.DataFrame(rows)
    with open(output_path, 'w') as f:
        if input_file:
            f.write(f'# Input data: {Path(input_file).name}\n')
        f.write(f'# Baseline values at T_loess({base_year}) for cumulative effects\n')
    df.to_csv(output_path, mode='a', index=False)

    total_rows = len(df)
    print(f"  Saved bootstrap_h_baselines.csv ({total_rows} rows)")


def plot_year_effects_bootstrap(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    input_file: str = None,
    approaches_to_plot: list = None,
    filename: str = 'year_effects_bootstrap.pdf',
    show_title: bool = True,
) -> None:
    """Plot year fixed effects k(t) with bootstrap uncertainty bands.

    Creates a multi-panel figure with one panel per approach, showing:
    - Point estimate line
    - IQR band (25th-75th percentile, darker shading)
    - 90% CI band (5th-95th percentile, lighter shading)

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for getting year range
        output_dir: Directory to save the plot
        input_file: Optional input file path for annotation
        approaches_to_plot: List of approach keys to plot. If None, plots all
            approaches that have k_samples. Default focuses on main approaches.
        filename: Output filename (default: 'year_effects_bootstrap.pdf')
        show_title: Whether to show the figure title (default: True)
    """
    # Default to methods that have meaningful year effects
    if approaches_to_plot is None:
        approaches_to_plot = [
            'Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL', 'Approach DL'
        ]

    # Filter to approaches that exist and have k_samples
    available_approaches = [
        name for name in approaches_to_plot
        if name in results and results[name].k_samples is not None
    ]

    if not available_approaches:
        print("  No approaches with k_samples available for plotting")
        return

    # Determine grid layout
    n_approaches = len(available_approaches)
    n_cols = min(3, n_approaches)
    n_rows = (n_approaches + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False, sharex=True)
    axes = axes.flatten()

    # Get years from data
    unique_years = sorted(set(data.year))
    years_array = np.array(unique_years)

    # Compute global y-axis range across all approaches
    all_k_values = []
    for name in available_approaches:
        result = results[name]
        if result.k_point is not None:
            # Filter out NaN values from k_point
            k_point_vals = [result.k_point[yr] for yr in unique_years if yr in result.k_point]
            all_k_values.extend([v for v in k_point_vals if not np.isnan(v)])
        if result.k_samples is not None:
            for yr in unique_years:
                if yr in result.k_samples:
                    valid = result.k_samples[yr][~np.isnan(result.k_samples[yr])]
                    if len(valid) > 0:
                        all_k_values.extend([np.percentile(valid, 5), np.percentile(valid, 95)])

    # Filter any remaining NaN values and compute range
    all_k_values = [v for v in all_k_values if not np.isnan(v)]
    if all_k_values:
        y_min = min(all_k_values)
        y_max = max(all_k_values)
        y_margin = (y_max - y_min) * 0.1
        y_range = (y_min - y_margin, y_max + y_margin)
    else:
        y_range = (-0.05, 0.05)

    for idx, name in enumerate(available_approaches):
        ax = axes[idx]
        result = results[name]

        # Extract point estimates and bootstrap percentiles
        k_point = []
        k_p5 = []
        k_p25 = []
        k_p75 = []
        k_p95 = []

        for yr in unique_years:
            # Point estimate
            if result.k_point is not None and yr in result.k_point:
                k_point.append(result.k_point[yr])
            else:
                k_point.append(np.nan)

            # Bootstrap samples
            if result.k_samples is not None and yr in result.k_samples:
                samples = result.k_samples[yr]
                valid = samples[~np.isnan(samples)]
                if len(valid) > 0:
                    k_p5.append(np.percentile(valid, 5))
                    k_p25.append(np.percentile(valid, 25))
                    k_p75.append(np.percentile(valid, 75))
                    k_p95.append(np.percentile(valid, 95))
                else:
                    k_p5.append(np.nan)
                    k_p25.append(np.nan)
                    k_p75.append(np.nan)
                    k_p95.append(np.nan)
            else:
                k_p5.append(np.nan)
                k_p25.append(np.nan)
                k_p75.append(np.nan)
                k_p95.append(np.nan)

        k_point = np.array(k_point)
        k_p5 = np.array(k_p5)
        k_p25 = np.array(k_p25)
        k_p75 = np.array(k_p75)
        k_p95 = np.array(k_p95)

        color = get_color(name, 'blue')

        # Plot 90% CI band
        ax.fill_between(years_array, k_p5, k_p95, alpha=0.2, color=color, linewidth=0)
        # Plot IQR band
        ax.fill_between(years_array, k_p25, k_p75, alpha=0.3, color=color, linewidth=0)
        # Plot point estimate
        ax.plot(years_array, k_point, color=color, linewidth=1.5)

        # Formatting
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(result.approach, fontsize=10)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Year')
        if idx % n_cols == 0:
            ax.set_ylabel('k(t)')

    # Hide unused axes
    for idx in range(n_approaches, len(axes)):
        axes[idx].set_visible(False)

    if show_title:
        fig.suptitle('Year Fixed Effects k(t) with Bootstrap Uncertainty', fontsize=12)
    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename)
    plt.close()
    print(f"  Saved {filename}")


def plot_combined_temp_response_and_year_effects(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    temp_response_approaches: list = None,
    year_effects_approaches: list = None,
    filename: str = 'fig_combined_temp_year.pdf',
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot combined figure with temperature response (top row) and year effects (bottom row).

    Creates a 2x2 figure combining:
    - Top row: Temperature response h(T) - h(T*) with uncertainty bands
    - Bottom row: Year fixed effects k(t) with uncertainty bands

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for getting year range and temperature histogram
        output_dir: Directory to save the plot
        temp_response_approaches: List of 2 approach keys for temperature response panels
        year_effects_approaches: List of 2 approach keys for year effects panels
        filename: Output filename (default: 'fig_combined_temp_year.pdf')
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    if temp_response_approaches is None:
        temp_response_approaches = ['Approach QJ', 'Approach QP']
    if year_effects_approaches is None:
        year_effects_approaches = ['Approach QJ', 'Approach QL']

    # Validate approaches exist
    temp_approaches = [a for a in temp_response_approaches if a in results]
    year_approaches = [a for a in year_effects_approaches if a in results and results[a].k_samples is not None]

    if len(temp_approaches) < 2 or len(year_approaches) < 2:
        print("  WARNING: Not enough valid approaches for combined figure")
        return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Temperature array for response plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Get temperature data from most recent year for histogram
    temp_recent = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

    # Get years for year effects plots
    unique_years = sorted(set(data.year))
    years_array = np.array(unique_years)

    # ========== TOP ROW: Temperature Response ==========
    # First pass: compute y-axis range for temperature response panels
    y_min_temp, y_max_temp = np.inf, -np.inf
    temp_plot_data = {}

    for name in temp_approaches[:2]:
        result = results[name]
        h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
        )
        h_point, T_opt = _compute_point_estimate_response(result, T, name, None)

        temp_plot_data[name] = {
            'h_p5': h_p5, 'h_p25': h_p25, 'h_p75': h_p75, 'h_p95': h_p95,
            'h_point': h_point, 'T_opt': T_opt
        }

        if not np.all(np.isnan(h_p5)):
            y_min_temp = min(y_min_temp, np.nanmin(h_p5), np.nanmin(h_point))
        if not np.all(np.isnan(h_p95)):
            y_max_temp = max(y_max_temp, np.nanmax(h_p95), np.nanmax(h_point))

    if np.isinf(y_min_temp) or np.isinf(y_max_temp):
        y_min_temp, y_max_temp = -0.05, 0.05
    y_padding_temp = (y_max_temp - y_min_temp) * 0.05
    y_min_temp -= y_padding_temp
    y_max_temp += y_padding_temp

    # Plot temperature response panels (top row)
    for col, name in enumerate(temp_approaches[:2]):
        ax = axes[0, col]
        result = results[name]
        color = get_color(name, 'steelblue')
        pdata = temp_plot_data[name]

        # Add temperature histogram on secondary y-axis
        if temp_recent is not None:
            ax2 = ax.twinx()
            bins = np.linspace(T_range[0], T_range[1], 30)
            ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel('Data density', fontsize=8, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
            ax2.set_ylim(bottom=0)
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
        T_opt = pdata['T_opt']
        if T_opt is not None and not np.isnan(T_opt):
            ax.axvline(T_opt, color=color, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt:.1f}°C')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
        ax.set_title(result.approach, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min_temp, y_max_temp)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    # ========== BOTTOM ROW: Year Effects ==========
    # First pass: compute y-axis range for year effects panels
    all_k_values = []
    for name in year_approaches[:2]:
        result = results[name]
        if result.k_point is not None:
            # Filter out NaN values from k_point
            k_point_vals = [result.k_point[yr] for yr in unique_years if yr in result.k_point]
            all_k_values.extend([v for v in k_point_vals if not np.isnan(v)])
        if result.k_samples is not None:
            for yr in unique_years:
                if yr in result.k_samples:
                    valid = result.k_samples[yr][~np.isnan(result.k_samples[yr])]
                    if len(valid) > 0:
                        all_k_values.extend([np.percentile(valid, 5), np.percentile(valid, 95)])

    # Filter any remaining NaN values and compute range
    all_k_values = [v for v in all_k_values if not np.isnan(v)]
    if all_k_values:
        y_min_year = min(all_k_values)
        y_max_year = max(all_k_values)
        y_margin = (y_max_year - y_min_year) * 0.1
        y_range_year = (y_min_year - y_margin, y_max_year + y_margin)
    else:
        y_range_year = (-0.05, 0.05)

    # Plot year effects panels (bottom row)
    for col, name in enumerate(year_approaches[:2]):
        ax = axes[1, col]
        result = results[name]

        # Extract point estimates and bootstrap percentiles
        k_point = []
        k_p5 = []
        k_p25 = []
        k_p75 = []
        k_p95 = []

        for yr in unique_years:
            if result.k_point is not None and yr in result.k_point:
                k_point.append(result.k_point[yr])
            else:
                k_point.append(np.nan)

            if result.k_samples is not None and yr in result.k_samples:
                samples = result.k_samples[yr]
                valid = samples[~np.isnan(samples)]
                if len(valid) > 0:
                    k_p5.append(np.percentile(valid, 5))
                    k_p25.append(np.percentile(valid, 25))
                    k_p75.append(np.percentile(valid, 75))
                    k_p95.append(np.percentile(valid, 95))
                else:
                    k_p5.append(np.nan)
                    k_p25.append(np.nan)
                    k_p75.append(np.nan)
                    k_p95.append(np.nan)
            else:
                k_p5.append(np.nan)
                k_p25.append(np.nan)
                k_p75.append(np.nan)
                k_p95.append(np.nan)

        k_point = np.array(k_point)
        k_p5 = np.array(k_p5)
        k_p25 = np.array(k_p25)
        k_p75 = np.array(k_p75)
        k_p95 = np.array(k_p95)

        color = get_color(name, 'blue')

        # Plot 90% CI band
        ax.fill_between(years_array, k_p5, k_p95, alpha=0.2, color=color, linewidth=0)
        # Plot IQR band
        ax.fill_between(years_array, k_p25, k_p75, alpha=0.3, color=color, linewidth=0)
        # Plot point estimate
        ax.plot(years_array, k_point, color=color, linewidth=1.5)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(result.approach, fontsize=10)
        ax.set_ylim(y_range_year)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('k(t)')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_temperature_response_4panel(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    approaches: list = None,
    filename: str = 'fig_temperature_response_4panel.pdf',
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot 4-panel temperature response figure (h(T) - h(T_opt)).

    Creates a 2x2 figure with temperature response curves and uncertainty bands.
    Default layout: Approach QJ, Approach QP on top row; Approach QL, Approach PL on bottom row.

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for temperature histogram
        output_dir: Directory to save the plot
        approaches: List of 4 approach keys for panels (row-major order)
        filename: Output filename
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    if approaches is None:
        approaches = ['Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL']

    # Validate approaches exist
    valid_approaches = [a for a in approaches if a in results]
    if len(valid_approaches) < 4:
        print(f"  WARNING: Only {len(valid_approaches)} valid approaches for 4-panel temperature response figure")
        if len(valid_approaches) < 2:
            return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Temperature array for response plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Get temperature data from most recent year for histogram
    temp_recent = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

    # Fixed y-axis range for publication consistency
    y_min, y_max = -0.15, 0.00
    y_ticks = np.arange(-0.15, 0.01, 0.03)

    # Compute plot data
    plot_data = {}
    for name in valid_approaches[:4]:
        result = results[name]
        h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
        )
        h_point, T_opt = _compute_point_estimate_response(result, T, name, None)

        plot_data[name] = {
            'h_p5': h_p5, 'h_p25': h_p25, 'h_p75': h_p75, 'h_p95': h_p95,
            'h_point': h_point, 'T_opt': T_opt
        }

    # Plot panels in 2x2 layout
    for idx, name in enumerate(valid_approaches[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        result = results[name]
        color = get_color(name, 'steelblue')
        pdata = plot_data[name]

        # Add temperature histogram on secondary y-axis
        if temp_recent is not None:
            ax2 = ax.twinx()
            bins = np.linspace(T_range[0], T_range[1], 30)
            ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel('Data density', fontsize=8, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
            ax2.set_ylim(bottom=0)
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
        T_opt = pdata['T_opt']
        if T_opt is not None and not np.isnan(T_opt):
            ax.axvline(T_opt, color=color, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt:.1f}°C')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
        ax.set_title(result.approach, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_temperature_derivative_4panel(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = 'fig_temperature_derivative_4panel.pdf',
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot 4-panel temperature derivative figure (dh/dT).

    Creates a 2x2 figure with temperature derivative curves and uncertainty bands.
    Default layout: Approach QJ, Approach QP on top row; Approach QL, Approach PL on bottom row.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        approaches: List of 4 approach keys for panels (row-major order)
        filename: Output filename
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    if approaches is None:
        approaches = ['Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL']

    # Validate approaches exist
    valid_approaches = [a for a in approaches if a in results]
    if len(valid_approaches) < 4:
        print(f"  WARNING: Only {len(valid_approaches)} valid approaches for 4-panel derivative figure")
        if len(valid_approaches) < 2:
            return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Temperature array for derivative plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Compute plot data
    plot_data = {}
    for name in valid_approaches[:4]:
        result = results[name]
        dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
        )
        dh_point = _compute_derivative_point_estimate(result, T, name, None)

        plot_data[name] = {
            'dh_p5': dh_p5, 'dh_p25': dh_p25, 'dh_p75': dh_p75, 'dh_p95': dh_p95,
            'dh_point': dh_point
        }

    # Plot panels in 2x2 layout
    for idx, name in enumerate(valid_approaches[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        result = results[name]
        color = get_color(name, 'steelblue')
        pdata = plot_data[name]

        # Plot 90% CI band
        ax.fill_between(T, pdata['dh_p5'], pdata['dh_p95'], alpha=0.2, color=color, label='90% CI')
        # Plot IQR band
        ax.fill_between(T, pdata['dh_p25'], pdata['dh_p75'], alpha=0.3, color=color, label='IQR')
        # Plot point estimate
        ax.plot(T, pdata['dh_point'], color=color, linestyle='-', linewidth=2, label='Point estimate')

        # Mark zero line
        ax.axhline(0, color='gray', linewidth=0.5)

        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('dh/dT', fontsize=10)
        ax.set_title(result.approach, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(-0.025, 0.015)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_temperature_response_2panel(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    approaches: list = None,
    filename: str = 'fig_temperature_response_2panel.pdf',
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot 2-panel temperature response figure (h(T) - h(T_opt)).

    Creates a 1x2 figure with temperature response curves and uncertainty bands.

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for temperature histogram
        output_dir: Directory to save the plot
        approaches: List of 2 approach keys for panels
        filename: Output filename
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    if approaches is None:
        approaches = ['Approach QJ', 'Approach QP']

    # Validate approaches exist
    valid_approaches = [a for a in approaches if a in results]
    if len(valid_approaches) < 2:
        print("  WARNING: Not enough valid approaches for temperature response figure")
        return

    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Temperature array for response plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Get temperature data from most recent year for histogram
    temp_recent = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

    # Fixed y-axis range for publication consistency
    y_min, y_max = -0.15, 0.00
    y_ticks = np.arange(-0.15, 0.01, 0.03)

    # Compute plot data
    plot_data = {}
    for name in valid_approaches[:2]:
        result = results[name]
        h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
        )
        h_point, T_opt = _compute_point_estimate_response(result, T, name, None)

        plot_data[name] = {
            'h_p5': h_p5, 'h_p25': h_p25, 'h_p75': h_p75, 'h_p95': h_p95,
            'h_point': h_point, 'T_opt': T_opt
        }

    # Plot panels
    for col, name in enumerate(valid_approaches[:2]):
        ax = axes[col]
        result = results[name]
        color = get_color(name, 'steelblue')
        pdata = plot_data[name]

        # Add temperature histogram on secondary y-axis
        if temp_recent is not None:
            ax2 = ax.twinx()
            bins = np.linspace(T_range[0], T_range[1], 30)
            ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax2.set_ylabel('Data density', fontsize=8, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
            ax2.set_ylim(bottom=0)
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
        T_opt = pdata['T_opt']
        if T_opt is not None and not np.isnan(T_opt):
            ax.axvline(T_opt, color=color, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt:.1f}°C')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
        ax.set_title(result.approach, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_year_effects_2panel(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    approaches: list = None,
    filename: str = 'fig_year_effects_2panel.pdf',
    input_file: str = None,
) -> None:
    """Plot 2-panel year effects figure (k(t)).

    Creates a 1x2 figure with year fixed effects and uncertainty bands.

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for getting year range
        output_dir: Directory to save the plot
        approaches: List of 2 approach keys for panels
        filename: Output filename
        input_file: Optional input file path for annotation
    """
    if approaches is None:
        approaches = ['Approach QJ', 'Approach QL']

    # Validate approaches exist and have k_samples
    valid_approaches = [a for a in approaches if a in results and results[a].k_samples is not None]
    if len(valid_approaches) < 2:
        print("  WARNING: Not enough valid approaches for year effects figure")
        return

    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Get years
    unique_years = sorted(set(data.year))
    years_array = np.array(unique_years)

    # First pass: collect IQR extremes for y-axis range
    iqr_extremes = []
    for name in valid_approaches[:2]:
        result = results[name]
        if result.k_samples is not None:
            for yr in unique_years:
                if yr in result.k_samples:
                    valid = result.k_samples[yr][~np.isnan(result.k_samples[yr])]
                    if len(valid) > 0:
                        iqr_extremes.extend([np.percentile(valid, 25), np.percentile(valid, 75)])

    if iqr_extremes:
        (y_min, y_max), y_ticks = get_axis_bounds_and_ticks(iqr_extremes)
        y_range = (y_min, y_max)
    else:
        y_range = (-0.05, 0.05)
        y_ticks = np.arange(-0.05, 0.06, 0.01)

    # Plot panels
    for col, name in enumerate(valid_approaches[:2]):
        ax = axes[col]
        result = results[name]

        # Extract point estimates and bootstrap percentiles
        k_point = []
        k_p5 = []
        k_p25 = []
        k_p75 = []
        k_p95 = []

        for yr in unique_years:
            if result.k_point is not None and yr in result.k_point:
                k_point.append(result.k_point[yr])
            else:
                k_point.append(np.nan)

            if result.k_samples is not None and yr in result.k_samples:
                samples = result.k_samples[yr]
                valid = samples[~np.isnan(samples)]
                if len(valid) > 0:
                    k_p5.append(np.percentile(valid, 5))
                    k_p25.append(np.percentile(valid, 25))
                    k_p75.append(np.percentile(valid, 75))
                    k_p95.append(np.percentile(valid, 95))
                else:
                    k_p5.append(np.nan)
                    k_p25.append(np.nan)
                    k_p75.append(np.nan)
                    k_p95.append(np.nan)
            else:
                k_p5.append(np.nan)
                k_p25.append(np.nan)
                k_p75.append(np.nan)
                k_p95.append(np.nan)

        k_point = np.array(k_point)
        k_p5 = np.array(k_p5)
        k_p25 = np.array(k_p25)
        k_p75 = np.array(k_p75)
        k_p95 = np.array(k_p95)

        color = get_color(name, 'blue')

        # Plot 90% CI band
        ax.fill_between(years_array, k_p5, k_p95, alpha=0.2, color=color, linewidth=0)
        # Plot IQR band
        ax.fill_between(years_array, k_p25, k_p75, alpha=0.3, color=color, linewidth=0)
        # Plot point estimate
        ax.plot(years_array, k_point, color=color, linewidth=1.5)

        ax.axhline(0, color='gray', linewidth=0.5)
        # Custom titles for the two panels
        if col == 0:
            ax.set_title('Approach QJ: k(t) minus quadratic OLS fit', fontsize=10)
        else:
            ax.set_title(r'$k_{mean}(t)$', fontsize=10)
        ax.set_ylim(y_range)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('k(t)')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_climate_response_contours(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = 'fig_climate_response_contours.pdf',
    Ttrend_range: tuple = (0, 30),
    deltaT_range: tuple = (-5, 5),
    n_points: int = 100,
    input_file: str = None,
) -> None:
    """Plot contour maps of climate response as function of Ttrend/T and deltaT.

    Creates a multi-page PDF:
    - Page 1: Response h(T) contours
      - Row 0: h_total(T) only, x=Ttrend
      - Row 1: h_total(T) - h_trend(Ttrend) vs Ttrend (where T = Ttrend + deltaT)
      - Row 2: h_total(T) - h_trend(T - deltaT) vs T
    - Page 2: Derivative dh/dT contours
      - Row 0: dh_total/dT only, x=Ttrend
      - Row 1: dh_total/dT(T) - dh_trend/dT(Ttrend) vs Ttrend
      - Row 2: dh_total/dT(T) - dh_trend/dT(T - deltaT) vs T
    - Page 3: Simple T vs Ttrend plot (2x2 grid)
      - Row 0: h_total(T) - h_trend(Ttrend) with T on x-axis and Ttrend on y-axis
      - Row 1: dh_total/dT - dh_trend/dTtrend (derivatives)

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        approaches: List of approach keys to include (default: ['Approach QL'])
        filename: Output filename
        Ttrend_range: Trend temperature range for x-axis (default: (0, 30))
        deltaT_range: Temperature deviation (T - Ttrend) range for y-axis (default: (-5, 5))
        n_points: Number of grid points per axis (default: 100)
        input_file: Optional input file path for annotation
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if approaches is None:
        approaches = ['Approach QL']

    # Filter to approaches that exist
    available = [a for a in approaches if a in results]
    if not available:
        print("  No valid approaches for contour plot")
        return

    n_cols = len(available)

    # Create meshgrid with Ttrend on x-axis, deltaT on y-axis (for rows 0-1)
    Ttrend = np.linspace(Ttrend_range[0], Ttrend_range[1], n_points)
    deltaT = np.linspace(deltaT_range[0], deltaT_range[1], n_points)
    Ttrend_grid, deltaT_grid = np.meshgrid(Ttrend, deltaT)

    # Actual temperature T = Ttrend + deltaT (for rows 0-1)
    T_grid = Ttrend_grid + deltaT_grid

    # Create separate meshgrid for bottom row (row 2) with T on x-axis
    # T ranges from 0 to 30, deltaT from -5 to 5, Ttrend = T - deltaT
    T_row2 = np.linspace(Ttrend_range[0], Ttrend_range[1], n_points)
    T_grid_row2, deltaT_grid_row2 = np.meshgrid(T_row2, deltaT)
    Ttrend_grid_row2 = T_grid_row2 - deltaT_grid_row2  # Ttrend = T - deltaT

    # Precompute all responses and derivatives for each approach
    approach_data = {}
    for approach in available:
        result = results[approach]

        if approach == 'method3a':
            # h2 for actual T curvature; h4 for trend T curvature
            h2_total = getattr(result, 'h2_point', 0) or 0
            h2_trend = getattr(result, 'h4_point', 0) or 0
            T_opt = result.T_opt_point

            # Response functions for rows 0-1 (Ttrend on x-axis)
            response_total_only = h2_total * (T_grid - T_opt)**2
            response_full = h2_total * (T_grid - T_opt)**2 - h2_trend * (Ttrend_grid - T_opt)**2

            # Response functions for row 2 (T on x-axis)
            # h_total(T) - h_trend(T - deltaT)
            response_full_row2 = h2_total * (T_grid_row2 - T_opt)**2 - h2_trend * (Ttrend_grid_row2 - T_opt)**2

            # Derivatives for rows 0-1: dh/dT = 2*h2*(T - T_opt)
            deriv_total = 2 * h2_total * (T_grid - T_opt)
            deriv_trend = 2 * h2_trend * (Ttrend_grid - T_opt)
            deriv_full = deriv_total - deriv_trend

            # Derivatives for row 2 (T on x-axis)
            deriv_total_row2 = 2 * h2_total * (T_grid_row2 - T_opt)
            deriv_trend_row2 = 2 * h2_trend * (Ttrend_grid_row2 - T_opt)
            deriv_full_row2 = deriv_total_row2 - deriv_trend_row2

            T_opt_trend = T_opt

        else:
            continue

        approach_data[approach] = {
            'result': result,
            'response_total_only': response_total_only,
            'response_full': response_full,
            'response_full_row2': response_full_row2,
            'deriv_total': deriv_total,
            'deriv_trend': deriv_trend,
            'deriv_full': deriv_full,
            'deriv_full_row2': deriv_full_row2,
            'T_opt_trend': T_opt_trend,
        }

    # Create multi-page PDF
    with PdfPages(output_dir / filename) as pdf:
        # ===== PAGE 1: Response h(T) =====
        fig1, axes1 = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14))
        if n_cols == 1:
            axes1 = axes1.reshape(3, 1)

        for col_idx, approach in enumerate(available):
            data = approach_data[approach]
            result = data['result']
            T_opt_trend = data['T_opt_trend']

            for row_idx, (response, title_suffix, x_grid, y_grid, x_label, x_range) in enumerate([
                (data['response_total_only'], 'Total Response h(T)', Ttrend_grid, deltaT_grid, 'Trend Temperature Ttrend (°C)', Ttrend_range),
                (data['response_full'], 'Total - Trend (x=Ttrend)', Ttrend_grid, deltaT_grid, 'Trend Temperature Ttrend (°C)', Ttrend_range),
                (data['response_full_row2'], 'Total - Trend (x=T)', T_grid_row2, deltaT_grid_row2, 'Temperature T (°C)', Ttrend_range)
            ]):
                ax = axes1[row_idx, col_idx]

                if row_idx == 0:
                    # Top row: h_total only, use viridis colormap
                    levels = np.linspace(np.nanmin(response), np.nanmax(response), 21)
                    contour = ax.contourf(x_grid, y_grid, response, levels=levels, cmap='viridis')
                else:
                    # Rows 1-2: difference field, symmetric bounds, red=negative
                    max_abs = max(abs(np.nanmin(response)), abs(np.nanmax(response)))
                    levels = np.linspace(-max_abs, max_abs, 21)
                    contour = ax.contourf(x_grid, y_grid, response, levels=levels, cmap='RdBu')
                ax.contour(x_grid, y_grid, response, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

                if row_idx >= 1:
                    ax.contour(x_grid, y_grid, response, levels=[0], colors='black', linewidths=1.5)

                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label('Climate Response', fontsize=9)

                if T_opt_trend is not None and not np.isnan(T_opt_trend):
                    ax.axvline(T_opt_trend, color='blue', linestyle='--', linewidth=1.5,
                              label=f'T_opt = {T_opt_trend:.1f}°C')

                ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('deltaT = T - Ttrend (°C)', fontsize=10)
                ax.set_title(f'{result.approach}: {title_suffix}', fontsize=11)
                ax.set_xlim(x_range)
                ax.set_ylim(deltaT_range)
                ax.legend(loc='upper right', fontsize=8)

        fig1.suptitle('Climate Response h(T)', fontsize=14, y=1.01)
        plt.tight_layout()
        add_input_file_annotation(fig1, input_file)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # ===== PAGE 2: Derivative dh/dT =====
        fig2, axes2 = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14))
        if n_cols == 1:
            axes2 = axes2.reshape(3, 1)

        for col_idx, approach in enumerate(available):
            data = approach_data[approach]
            result = data['result']
            T_opt_trend = data['T_opt_trend']

            for row_idx, (response, title_suffix, x_grid, y_grid, x_label, x_range) in enumerate([
                (data['deriv_total'], 'Total Derivative dh/dT', Ttrend_grid, deltaT_grid, 'Trend Temperature Ttrend (°C)', Ttrend_range),
                (data['deriv_full'], 'Total - Trend Derivative (x=Ttrend)', Ttrend_grid, deltaT_grid, 'Trend Temperature Ttrend (°C)', Ttrend_range),
                (data['deriv_full_row2'], 'Total - Trend Derivative (x=T)', T_grid_row2, deltaT_grid_row2, 'Temperature T (°C)', Ttrend_range)
            ]):
                ax = axes2[row_idx, col_idx]

                if row_idx == 0:
                    # Top row: total derivative only, use viridis colormap
                    levels = np.linspace(np.nanmin(response), np.nanmax(response), 21)
                    contour = ax.contourf(x_grid, y_grid, response, levels=levels, cmap='viridis')
                    ax.contour(x_grid, y_grid, response, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
                    # Add zero contour line
                    ax.contour(x_grid, y_grid, response, levels=[0], colors='black', linewidths=1.5)
                else:
                    # Rows 1-2: difference field, symmetric bounds, red=negative
                    max_abs = max(abs(np.nanmin(response)), abs(np.nanmax(response)))
                    levels = np.linspace(-max_abs, max_abs, 21)
                    contour = ax.contourf(x_grid, y_grid, response, levels=levels, cmap='RdBu')
                    ax.contour(x_grid, y_grid, response, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
                    # Add zero contour line
                    ax.contour(x_grid, y_grid, response, levels=[0], colors='black', linewidths=1.5)

                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label('dh/dT', fontsize=9)

                if T_opt_trend is not None and not np.isnan(T_opt_trend):
                    ax.axvline(T_opt_trend, color='blue', linestyle='--', linewidth=1.5,
                              label=f'T_opt = {T_opt_trend:.1f}°C')

                ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('deltaT = T - Ttrend (°C)', fontsize=10)
                ax.set_title(f'{result.approach}: {title_suffix}', fontsize=11)
                ax.set_xlim(x_range)
                ax.set_ylim(deltaT_range)
                ax.legend(loc='upper right', fontsize=8)

        fig2.suptitle('Climate Response Derivative dh/dT', fontsize=14, y=1.01)
        plt.tight_layout()
        add_input_file_annotation(fig2, input_file)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

        # ===== PAGE 3: Simple T vs Ttrend (2x2 grid: response and derivative) =====
        # Create meshgrid with T on x-axis, Ttrend on y-axis (both 0-30)
        T_page3 = np.linspace(Ttrend_range[0], Ttrend_range[1], n_points)
        Ttrend_page3 = np.linspace(Ttrend_range[0], Ttrend_range[1], n_points)
        T_grid_p3, Ttrend_grid_p3 = np.meshgrid(T_page3, Ttrend_page3)

        fig3, axes3 = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
        if n_cols == 1:
            axes3 = axes3.reshape(2, 1)

        for col_idx, approach in enumerate(available):
            result = results[approach]

            if approach == 'method3a':
                # h2 for actual T curvature; h4 for trend T curvature
                h2_total = getattr(result, 'h2_point', 0) or 0
                h2_trend = getattr(result, 'h4_point', 0) or 0
                T_opt = result.T_opt_point

                # h_total(T) - h_trend(Ttrend)
                response = h2_total * (T_grid_p3 - T_opt)**2 - h2_trend * (Ttrend_grid_p3 - T_opt)**2

                # Derivatives: dh_total/dT - dh_trend/dTtrend
                deriv_total = 2 * h2_total * (T_grid_p3 - T_opt)
                deriv_trend = 2 * h2_trend * (Ttrend_grid_p3 - T_opt)
                deriv = deriv_total - deriv_trend

                T_opt_trend = T_opt
                T_opt_total = T_opt

            else:
                continue

            # Row 0: Response h_total(T) - h_trend(Ttrend)
            ax = axes3[0, col_idx]
            max_abs = max(abs(np.nanmin(response)), abs(np.nanmax(response)))
            levels = np.linspace(-max_abs, max_abs, 21)
            contour = ax.contourf(T_grid_p3, Ttrend_grid_p3, response, levels=levels, cmap='RdBu')
            ax.contour(T_grid_p3, Ttrend_grid_p3, response, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
            ax.contour(T_grid_p3, Ttrend_grid_p3, response, levels=[0], colors='black', linewidths=1.5)

            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('h_total(T) - h_trend(Ttrend)', fontsize=9)

            if T_opt_total is not None and not np.isnan(T_opt_total):
                ax.axvline(T_opt_total, color='red', linestyle='--', linewidth=1.5,
                          label=f'T_opt (total) = {T_opt_total:.1f}°C')
            if T_opt_trend is not None and not np.isnan(T_opt_trend):
                ax.axhline(T_opt_trend, color='blue', linestyle='--', linewidth=1.5,
                          label=f'T_opt (trend) = {T_opt_trend:.1f}°C')
            ax.plot([Ttrend_range[0], Ttrend_range[1]], [Ttrend_range[0], Ttrend_range[1]],
                   'g--', linewidth=1.5, label='T = Ttrend')

            ax.set_xlabel('Temperature T (°C)', fontsize=10)
            ax.set_ylabel('Trend Temperature Ttrend (°C)', fontsize=10)
            ax.set_title(f'{result.approach}: Total - Trend Response', fontsize=11)
            ax.set_xlim(Ttrend_range)
            ax.set_ylim(Ttrend_range)
            ax.set_aspect('equal')
            ax.legend(loc='upper left', fontsize=8)

            # Row 1: Derivative dh_total/dT - dh_trend/dTtrend
            ax = axes3[1, col_idx]
            max_abs = max(abs(np.nanmin(deriv)), abs(np.nanmax(deriv)))
            levels = np.linspace(-max_abs, max_abs, 21)
            contour = ax.contourf(T_grid_p3, Ttrend_grid_p3, deriv, levels=levels, cmap='RdBu')
            ax.contour(T_grid_p3, Ttrend_grid_p3, deriv, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
            ax.contour(T_grid_p3, Ttrend_grid_p3, deriv, levels=[0], colors='black', linewidths=1.5)

            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('dh_total/dT - dh_trend/dTtrend', fontsize=9)

            if T_opt_total is not None and not np.isnan(T_opt_total):
                ax.axvline(T_opt_total, color='red', linestyle='--', linewidth=1.5,
                          label=f'T_opt (total) = {T_opt_total:.1f}°C')
            if T_opt_trend is not None and not np.isnan(T_opt_trend):
                ax.axhline(T_opt_trend, color='blue', linestyle='--', linewidth=1.5,
                          label=f'T_opt (trend) = {T_opt_trend:.1f}°C')
            ax.plot([Ttrend_range[0], Ttrend_range[1]], [Ttrend_range[0], Ttrend_range[1]],
                   'g--', linewidth=1.5, label='T = Ttrend')

            ax.set_xlabel('Temperature T (°C)', fontsize=10)
            ax.set_ylabel('Trend Temperature Ttrend (°C)', fontsize=10)
            ax.set_title(f'{result.approach}: Total - Trend Derivative', fontsize=11)
            ax.set_xlim(Ttrend_range)
            ax.set_ylim(Ttrend_range)
            ax.set_aspect('equal')
            ax.legend(loc='upper left', fontsize=8)

        fig3.suptitle('Climate Response: T vs Ttrend', fontsize=14, y=1.01)
        plt.tight_layout()
        add_input_file_annotation(fig3, input_file)
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)

    print(f"  Saved {filename} (3 pages)")


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
    For piecewise (Approach PL/Approach PJ): h(T) - h(T_opt) = h2*(T-T_opt)² or h4*(T-T_opt)²
    For method2b/8a variants: uses appropriate coefficients (h3,h4 for trend)

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)
        approach_key: Approach identifier (e.g., 'Approach PL'/'Approach PJ' for piecewise,
                      'method2b', 'method3a_high', 'method3a_low')

    Returns:
        Tuple of arrays (h_lower, h_median, h_upper) each with shape (len(T_range),)
    """
    is_piecewise = (approach_key in ('Approach PL', 'Approach PJ', 'Approach PP'))
    is_segmented = (approach_key in ('Approach SL', 'Approach SJ', 'Approach SP'))
    is_three_interval = (approach_key in ('Approach TL', 'Approach TJ', 'Approach TP'))

    # Handle approach 6b (trend only)
    if approach_key == 'method2b':
        h1_samples = getattr(result, 'h3_samples', None)
        h2_samples = getattr(result, 'h4_samples', None)
        if h1_samples is None or h2_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h1_samples) & ~np.isnan(h2_samples)
        h1_valid = h1_samples[valid_mask]
        h2_valid = h2_samples[valid_mask]
        return _compute_quadratic_bands(h1_valid, h2_valid, T_range, percentiles)

    # Handle approach 8a total response: (h2 - h4) * (T - T_opt)^2
    # This matches the contour plot at T_delta = 0 (where T = T_trend)
    if approach_key == 'method3a_total':
        h2_total_samples = getattr(result, 'h2_samples', None)
        h2_trend_samples = getattr(result, 'h4_samples', None)
        T_opt_samples = getattr(result, 'T_opt_samples', None)
        if h2_total_samples is None or h2_trend_samples is None or T_opt_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h2_total_samples) & ~np.isnan(h2_trend_samples) & ~np.isnan(T_opt_samples)
        # Net curvature: h2_net = h2 - h4
        h2_net = h2_total_samples[valid_mask] - h2_trend_samples[valid_mask]
        T_opt_valid = T_opt_samples[valid_mask]
        return _compute_symmetric_piecewise_bands(h2_net, T_opt_valid, T_range, percentiles)

    # Handle approach 8a trend response (piecewise with shared T_opt)
    if approach_key == 'method3a_low':
        h2_samples = getattr(result, 'h4_samples', None)
        T_opt_samples = getattr(result, 'T_opt_samples', None)
        if h2_samples is None or T_opt_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h2_samples) & ~np.isnan(T_opt_samples)
        h2_valid = h2_samples[valid_mask]
        T_opt_valid = T_opt_samples[valid_mask]
        return _compute_symmetric_piecewise_bands(h2_valid, T_opt_valid, T_range, percentiles)

    if is_three_interval:
        # Three-interval model: h2*f_low + h4*f_high
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)
        T_crit_low_samples = getattr(result, 'T_crit_low_samples', None)
        T_crit_high_samples = getattr(result, 'T_crit_high_samples', None)

        if h2_low_samples is None or h2_high_samples is None or T_crit_low_samples is None or T_crit_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(T_crit_low_samples) &
                      ~np.isnan(T_crit_high_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_crit_low_valid = T_crit_low_samples[valid_mask]
        T_crit_high_valid = T_crit_high_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        h_relative_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            f_low, f_high = three_interval_shape(T_range, T_crit_low_valid[i], T_crit_high_valid[i] - T_crit_low_valid[i])
            h_relative_samples[i, :] = h2_low_valid[i] * f_low + h2_high_valid[i] * f_high
    elif is_segmented:
        # Segmented linear model: h2 for T <= T_opt, h4 for T > T_opt (linear, not squared)
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)

        if h2_low_samples is None or h2_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(result.T_opt_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_opt_valid = result.T_opt_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        h_relative_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2_low = h2_low_valid[i]
            h2_high = h2_high_valid[i]
            T_opt = T_opt_valid[i]
            low_comp, high_comp = segmented_linear_shape(T_range, T_opt)
            h_relative_samples[i, :] = h2_low * low_comp + h2_high * high_comp
    elif is_piecewise:
        # Piecewise quadratic model: h2 for T <= T_opt, h4 for T > T_opt
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)

        if h2_low_samples is None or h2_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(result.T_opt_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_opt_valid = result.T_opt_samples[valid_mask]

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

        return _compute_quadratic_bands(h1_valid, h2_valid, T_range, percentiles)

    # Compute percentiles at each temperature
    h_bands = []
    for p in percentiles:
        h_bands.append(np.percentile(h_relative_samples, p, axis=0))

    return tuple(h_bands)


def _compute_quadratic_bands(h1_valid, h2_valid, T_range, percentiles):
    """Helper to compute uncertainty bands for quadratic h(T) = h1*T + h2*T²."""
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
            h_T_opt = -h1 ** 2 / (4 * h2)
        else:
            h_T_opt = 0
        h_relative_samples[i, :] = h_T - h_T_opt

    h_bands = []
    for p in percentiles:
        h_bands.append(np.percentile(h_relative_samples, p, axis=0))
    return tuple(h_bands)


def _compute_symmetric_piecewise_bands(h2_valid, T_opt_valid, T_range, percentiles):
    """Helper to compute uncertainty bands for symmetric piecewise h(T) = h2*(T-T_opt)²."""
    if len(h2_valid) == 0:
        return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

    n_samples = len(h2_valid)
    n_T = len(T_range)
    h_relative_samples = np.zeros((n_samples, n_T))

    for i in range(n_samples):
        h2 = h2_valid[i]
        T_opt = T_opt_valid[i]
        # h(T) - h(T_opt) = h2*(T - T_opt)² since h(T_opt) = 0
        h_relative_samples[i, :] = h2 * (T_range - T_opt) ** 2

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
    """Plot h1, h2, T_opt distributions for one approach.

    Creates a (1, 3) subplot with histograms showing:
    - Point estimate (red solid line)
    - Bootstrap median (blue dashed line)
    - 90% CI bounds (gray dotted lines)

    Args:
        result: BootstrapResult for this approach
        stats: Statistics dict from compute_bootstrap_statistics
        output_dir: Directory to save the plot
        approach_key: Key like 'Approach QJ' for filename
    """
    params = [
        ('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁ (Linear Coefficient)'),
        ('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (Quadratic Coefficient)'),
        ('T_opt', result.T_opt_samples, result.T_opt_point, stats['T_opt'], 'T_opt (°C)'),
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


def _get_distribution_params_for_approach(name: str, result, stats: dict) -> list:
    """Get the parameter tuples for distribution plotting based on approach type.

    Returns list of tuples: (param_name, samples, point_est, param_stats, xlabel)
    """
    # Approach 6b: T only (uses h1, h2, T_opt - departure terms h3, h4 are zero)
    if name == 'method2b':
        params = []
        if result.h1_samples is not None and 'h1' in stats:
            params.append(('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁ (actual T)'))
        if result.h2_samples is not None and 'h2' in stats:
            params.append(('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (actual T)'))
        if result.T_opt_samples is not None and 'T_opt' in stats:
            params.append(('T_opt', result.T_opt_samples, result.T_opt_point, stats['T_opt'], 'T_opt (°C)'))
        return params if params else _get_standard_params(result, stats)

    # Approach 8: piecewise quadratic (h2 for T<=T_opt, h4 for T>T_opt)
    if name in ('Approach PL', 'Approach PL'):
        params = []
        # h2 is curvature below T_opt
        if result.h2_samples is not None and 'h2' in stats:
            params.append(('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (T ≤ T_opt)'))
        # h4 is curvature above T_opt
        h4_samples = getattr(result, 'h4_samples', None)
        if h4_samples is not None and 'h4' in stats:
            params.append(('h4', h4_samples, getattr(result, 'h4_point', np.nan), stats['h4'], 'h₄ (T > T_opt)'))
        if 'T_opt' in stats:
            params.append(('T_opt', result.T_opt_samples, result.T_opt_point, stats['T_opt'], 'T_opt (°C)'))
        return params if params else _get_standard_params(result, stats)

    # Approach 8a: h2 for actual T, h4 for trend T, shared T_opt
    if name == 'method3a':
        params = []
        # h2 is curvature for actual T
        if result.h2_samples is not None and 'h2' in stats:
            params.append(('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (actual T)'))
        # h4 is curvature for trend T
        h4_samples = getattr(result, 'h4_samples', None)
        if h4_samples is not None and 'h4' in stats:
            params.append(('h4', h4_samples, getattr(result, 'h4_point', np.nan), stats['h4'], 'h₄ (trend T)'))
        if 'T_opt' in stats:
            params.append(('T_opt', result.T_opt_samples, result.T_opt_point, stats['T_opt'], 'T_opt (°C)'))
        return params if params else _get_standard_params(result, stats)

    # Standard approaches: h1, h2, T_opt
    return _get_standard_params(result, stats)


def _get_standard_params(result, stats: dict) -> list:
    """Get standard parameter tuples (h1, h2, T_opt)."""
    params = []
    if 'h1' in stats:
        params.append(('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁'))
    if 'h2' in stats:
        params.append(('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂'))
    if 'T_opt' in stats:
        params.append(('T_opt', result.T_opt_samples, result.T_opt_point,
                      stats['T_opt'], 'T_opt (°C)'))
    return params


def plot_all_bootstrap_distributions(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    filename: str = "bootstrap_distributions.pdf",
    input_file: str = None
) -> None:
    """Plot parameter distributions for all approaches in a single PDF.

    Creates a multi-page PDF with each approach on its own page.
    Handles different parameter structures for different approaches:
    - Standard (0-5, 5a-5d, 6): h1, h2, T_opt
    - Approach 6a: h1, h2, T_opt (actual T), h3, h4, T_dep_opt (departure)
    - Approach 6b: h1, h2, T_opt (actual T only, departure terms are zero)
    - Approach 8: h2 (T ≤ T_opt), h4 (T > T_opt), T_opt
    - Approach 8a: h2 (actual T), h4 (trend T), T_opt

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save the plot
        filename: Output filename (should end in .pdf)
        input_file: Path to input data file (for annotation)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    approach_names = list(results.keys())

    with PdfPages(output_dir / filename) as pdf:
        for name in approach_names:
            result = results[name]
            stats = all_stats[name]

            # Get parameters for this approach type
            params = _get_distribution_params_for_approach(name, result, stats)

            if not params:
                continue

            n_params = len(params)
            # Arrange in rows of 3
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, (param_name, samples, point_est, param_stats, xlabel) in enumerate(params):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row_idx, col_idx]

                if samples is None:
                    ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel(xlabel, fontsize=10)
                    continue

                # Filter valid samples
                valid_samples = samples[~np.isnan(samples)]
                if len(valid_samples) == 0:
                    ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel(xlabel, fontsize=10)
                    continue

                # Histogram
                ax.hist(valid_samples, bins=50, density=True, alpha=0.7, color='steelblue')

                # Point estimate (red solid)
                if point_est is not None and not np.isnan(point_est):
                    ax.axvline(x=point_est, color='red', linestyle='-', linewidth=2,
                              label=f'Point: {point_est:.4f}')

                # Bootstrap median (blue dashed)
                if param_stats and 'p50' in param_stats:
                    median = param_stats['p50']
                    ax.axvline(x=median, color='blue', linestyle='--', linewidth=2,
                              label=f'Median: {median:.4f}')

                    # 90% CI bounds (gray dotted)
                    p5 = param_stats.get('p5', np.nan)
                    p95 = param_stats.get('p95', np.nan)
                    if not np.isnan(p5):
                        ax.axvline(x=p5, color='gray', linestyle=':', linewidth=1.5,
                                  label=f'5%: {p5:.4f}')
                    if not np.isnan(p95):
                        ax.axvline(x=p95, color='gray', linestyle=':', linewidth=1.5,
                                  label=f'95%: {p95:.4f}')

                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(xlabel, fontsize=11)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_params, n_rows * n_cols):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                axes[row_idx, col_idx].set_visible(False)

            fig.suptitle(f'Bootstrap Distributions: {result.approach}', fontsize=14, y=1.02)
            plt.tight_layout()
            add_input_file_annotation(fig, input_file)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def _expand_approaches_for_plotting(approaches: list, results: dict) -> list:
    """Expand approaches that need multiple panels (6a, 8a) into sub-approaches.

    Args:
        approaches: List of approach names
        results: Dict of BootstrapResult

    Returns:
        List of tuples: (plot_key, result_key, display_name, coefficient_variant)
        - plot_key: unique key for this panel
        - result_key: key to look up in results dict
        - display_name: title for the panel
        - coefficient_variant: 'high', 'low', or None for standard handling
    """
    expanded = []
    for name in approaches:
        if name not in results:
            continue

        if name == 'method3a':
            # Expand into total response and low-frequency panels
            # Total response = (h2_high - h2_low)*(T - T_opt)^2, matching contour at T_delta=0
            expanded.append(('method3a_total', name, '8a: Total Response', 'total'))
            expanded.append(('method3a_low', name, '8a: Trend Response', 'low'))
        else:
            # Standard approach - one panel
            result = results[name]
            display_name = getattr(result, 'approach', name)
            expanded.append((name, name, display_name, None))

    return expanded


def _compute_point_estimate_response(result, T, approach_key, variant=None):
    """Compute point estimate h(T) - h(T_opt) for a given approach.

    Args:
        result: BootstrapResult
        T: Temperature array
        approach_key: The approach key (may include _high/_low suffix)
        variant: 'high', 'low', or None

    Returns:
        tuple: (h_point array, T_opt for vertical line)
    """
    # Handle method2b (T only - departure terms h3,h4 are zero)
    if approach_key == 'method2b':
        h1 = getattr(result, 'h1_point', 0) or 0
        h2 = getattr(result, 'h2_point', 0) or 0
        T_opt = getattr(result, 'T_opt_point', None)
        h_T = h1 * T + h2 * T ** 2
        if h2 != 0:
            h_T_opt = -h1 ** 2 / (4 * h2)
            if T_opt is None:
                T_opt = -h1 / (2 * h2)
        else:
            h_T_opt = 0
            T_opt = T_opt or np.nan
        return h_T - h_T_opt, T_opt

    # Handle method3a total response: (h2 - h4) * (T - T_opt)^2
    # This matches the contour plot at T_delta = 0 (where T = T_trend)
    if approach_key == 'method3a_total' or (approach_key == 'method3a' and variant == 'total'):
        h2_total = getattr(result, 'h2_point', 0) or 0
        h2_trend = getattr(result, 'h4_point', 0) or 0
        T_opt = result.T_opt_point
        # Net response = h2*(T-T_opt)² - h4*(T-T_opt)² = (h2 - h4)*(T-T_opt)²
        h2_net = h2_total - h2_trend
        h_point = h2_net * (T - T_opt) ** 2
        return h_point, T_opt

    # Handle method3a trend (piecewise with shared T_opt)
    if approach_key == 'method3a_low' or (approach_key == 'method3a' and variant == 'low'):
        h2 = getattr(result, 'h4_point', 0) or 0
        T_opt = result.T_opt_point
        # h(T) - h(T_opt) = h4 * (T - T_opt)^2 since h(T_opt) = 0
        h_point = h2 * (T - T_opt) ** 2
        return h_point, T_opt

    # Handle three-interval approaches Approach TL/Approach TJ/Approach TP
    if approach_key in ('Approach TL', 'Approach TJ', 'Approach TP') and result.h4_point is not None:
        T_crit_low = getattr(result, 'T_crit_low_point', None)
        T_crit_high = getattr(result, 'T_crit_high_point', None)
        T_opt = result.T_opt_point
        if T_crit_low is not None and T_crit_high is not None:
            f_low, f_high = three_interval_shape(T, T_crit_low, T_crit_high - T_crit_low)
            h_point = result.h2_point * f_low + result.h4_point * f_high
        else:
            h_point = np.zeros_like(T)
        return h_point, T_opt

    # Handle segmented linear approaches Approach SL/Approach SJ/Approach SP: h2*(T-T_opt) for T<=T_opt, h4*(T-T_opt) for T>T_opt
    if approach_key in ('Approach SL', 'Approach SJ', 'Approach SP') and result.h4_point is not None:
        T_opt = result.T_opt_point
        h2_low = result.h2_point
        h2_high = result.h4_point
        h_point = np.where(
            T <= T_opt,
            h2_low * (T - T_opt),
            h2_high * (T - T_opt)
        )
        return h_point, T_opt

    # Handle piecewise approaches Approach PL/Approach PJ/Approach PP (asymmetric): h2 for T <= T_opt, h4 for T > T_opt
    if approach_key in ('Approach PL', 'Approach PJ', 'Approach PP') and result.h4_point is not None:
        T_opt = result.T_opt_point
        h2_low = result.h2_point
        h2_high = result.h4_point
        h_point = np.where(
            T <= T_opt,
            h2_low * (T - T_opt) ** 2,
            h2_high * (T - T_opt) ** 2
        )
        return h_point, T_opt

    # Standard quadratic model
    h1 = result.h1_point
    h2 = result.h2_point
    h_T = h1 * T + h2 * T ** 2
    if h2 != 0:
        T_opt = -h1 / (2 * h2)
        h_T_opt = -h1 ** 2 / (4 * h2)
    else:
        T_opt = result.T_opt_point
        h_T_opt = 0
    return h_T - h_T_opt, T_opt


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

    For approaches with separate high/low frequency responses (6a, 8a),
    two panels are created - one for each frequency band.

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

    if len(approaches) == 0:
        return

    # Expand approaches that need multiple panels
    plot_entries = _expand_approaches_for_plotting(approaches, results)
    n_panels = len(plot_entries)

    if n_panels == 0:
        return

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for plot_key, result_key, display_name, variant in plot_entries:
        result = results[result_key]

        # Compute uncertainty bands (90% CI and IQR)
        h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=plot_key
        )

        # Compute point estimate response
        h_point, T_opt = _compute_point_estimate_response(result, T, plot_key, variant)

        plot_data[plot_key] = {
            'h_p5': h_p5,
            'h_p25': h_p25,
            'h_p75': h_p75,
            'h_p95': h_p95,
            'h_point': h_point,
            'T_opt': T_opt,
            'display_name': display_name,
            'result_key': result_key,
        }

        # Update global y range
        if not np.all(np.isnan(h_p5)):
            y_min = min(y_min, np.nanmin(h_p5), np.nanmin(h_point))
        if not np.all(np.isnan(h_p95)):
            y_max = max(y_max, np.nanmax(h_p95), np.nanmax(h_point))

    # Handle case where all data is NaN
    if np.isinf(y_min) or np.isinf(y_max):
        y_min, y_max = -0.05, 0.05

    # Add some padding to y range
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Determine grid layout
    if n_panels <= 3:
        n_rows, n_cols = 1, n_panels
    elif n_panels <= 4:
        n_rows, n_cols = 2, 2
    elif n_panels <= 6:
        n_rows, n_cols = 2, 3
    elif n_panels <= 8:
        n_rows, n_cols = 4, 2
    elif n_panels <= 9:
        n_rows, n_cols = 3, 3
    elif n_panels <= 12:
        n_rows, n_cols = 4, 3
    elif n_panels <= 16:
        n_rows, n_cols = 4, 4
    elif n_panels <= 20:
        n_rows, n_cols = 5, 4
    else:
        n_rows, n_cols = 6, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
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
    for idx, (plot_key, result_key, display_name, variant) in enumerate(plot_entries):
        ax = axes[idx]
        result = results[result_key]
        color = get_color(result_key, 'steelblue')
        pdata = plot_data[plot_key]

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
        T_opt = pdata['T_opt']
        if T_opt is not None and not np.isnan(T_opt):
            ax.axvline(T_opt, color=color, linestyle=':', alpha=0.7,
                       label=f'T_opt = {T_opt:.1f}°C')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
        ax.set_title(display_name, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

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

    # Filter to approaches that have T_opt (e.g., exclude Approach 6c which uses T_dep_opt/f2)
    approach_names = [name for name in results.keys() if results[name].T_opt_point is not None]
    n_approaches = len(approach_names)
    y_positions = np.arange(n_approaches)

    for i, name in enumerate(approach_names):
        result = results[name]
        stats = all_stats[name]['T_opt']
        color = get_color(name, 'gray')

        point_est = result.T_opt_point
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
    For piecewise quadratic (method3): dh/dT = 2*h2_low*(T-T_opt) or 2*h2_high*(T-T_opt)
    For method2b/8a variants: uses appropriate high/low frequency coefficients

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)
        approach_key: Approach identifier (e.g., 'Approach PL' for piecewise quadratic,
                      'method2b', 'method3a_high', 'method3a_low')

    Returns:
        Tuple of arrays (dh_lower, dh_median, dh_upper) each with shape (len(T_range),)
    """
    is_piecewise = (approach_key in ('Approach PL', 'Approach PJ', 'Approach PP'))
    is_segmented = (approach_key in ('Approach SL', 'Approach SJ', 'Approach SP'))
    is_three_interval = (approach_key in ('Approach TL', 'Approach TJ', 'Approach TP'))

    # Handle approach 6b (trend only - uses h3,h4 for trend response)
    if approach_key == 'method2b':
        h1_samples = getattr(result, 'h3_samples', None)
        h2_samples = getattr(result, 'h4_samples', None)
        if h1_samples is None or h2_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h1_samples) & ~np.isnan(h2_samples)
        h1_valid = h1_samples[valid_mask]
        h2_valid = h2_samples[valid_mask]
        return _compute_quadratic_derivative_bands(h1_valid, h2_valid, T_range, percentiles)

    # Handle approach 8a high-frequency derivative (uses h2 for actual T curvature)
    if approach_key == 'method3a_high':
        h2_samples = getattr(result, 'h2_samples', None)
        T_opt_samples = getattr(result, 'T_opt_samples', None)
        if h2_samples is None or T_opt_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h2_samples) & ~np.isnan(T_opt_samples)
        h2_valid = h2_samples[valid_mask]
        T_opt_valid = T_opt_samples[valid_mask]
        return _compute_symmetric_piecewise_derivative_bands(h2_valid, T_opt_valid, T_range, percentiles)

    # Handle approach 8a low-frequency derivative (uses h4 for trend T curvature)
    if approach_key == 'method3a_low':
        h2_samples = getattr(result, 'h4_samples', None)
        T_opt_samples = getattr(result, 'T_opt_samples', None)
        if h2_samples is None or T_opt_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)
        valid_mask = ~np.isnan(h2_samples) & ~np.isnan(T_opt_samples)
        h2_valid = h2_samples[valid_mask]
        T_opt_valid = T_opt_samples[valid_mask]
        return _compute_symmetric_piecewise_derivative_bands(h2_valid, T_opt_valid, T_range, percentiles)

    if is_three_interval:
        # Three-interval model: piecewise linear derivative
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)
        T_crit_low_samples = getattr(result, 'T_crit_low_samples', None)
        T_crit_high_samples = getattr(result, 'T_crit_high_samples', None)

        if h2_low_samples is None or h2_high_samples is None or T_crit_low_samples is None or T_crit_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(T_crit_low_samples) &
                      ~np.isnan(T_crit_high_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_crit_low_valid = T_crit_low_samples[valid_mask]
        T_crit_high_valid = T_crit_high_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        dh_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2_lo = h2_low_valid[i]
            h4_hi = h2_high_valid[i]
            T_lo = T_crit_low_valid[i]
            dT = T_crit_high_valid[i] - T_crit_low_valid[i]
            T_hi = T_lo + dT
            below = T_range <= T_lo
            above = T_range >= T_hi
            if dT > 0:
                frac = np.clip((T_range - T_lo) / dT, 0, 1)
                dh_samples[i, :] = np.where(below, h2_lo, np.where(above, h4_hi, h2_lo + (h4_hi - h2_lo) * frac))
            else:
                dh_samples[i, :] = np.where(T_range <= T_lo, h2_lo, h4_hi)
    elif is_segmented:
        # Segmented linear model: derivative is step function (constant h2 below, h4 above T_opt)
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)

        if h2_low_samples is None or h2_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(result.T_opt_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_opt_valid = result.T_opt_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        dh_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2_low = h2_low_valid[i]
            h2_high = h2_high_valid[i]
            T_opt = T_opt_valid[i]

            # Segmented linear derivative: step function
            dh_samples[i, :] = np.where(T_range <= T_opt, h2_low, h2_high)
    elif is_piecewise:
        # Piecewise quadratic model: h2 for T <= T_opt, h4 for T > T_opt
        h2_low_samples = getattr(result, 'h2_samples', None)
        h2_high_samples = getattr(result, 'h4_samples', None)

        if h2_low_samples is None or h2_high_samples is None:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        valid_mask = (~np.isnan(h2_low_samples) &
                      ~np.isnan(h2_high_samples) &
                      ~np.isnan(result.T_opt_samples))

        h2_low_valid = h2_low_samples[valid_mask]
        h2_high_valid = h2_high_samples[valid_mask]
        T_opt_valid = result.T_opt_samples[valid_mask]

        if len(h2_low_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        n_samples = len(h2_low_valid)
        n_T = len(T_range)
        dh_samples = np.zeros((n_samples, n_T))

        for i in range(n_samples):
            h2_low = h2_low_valid[i]
            h2_high = h2_high_valid[i]
            T_opt = T_opt_valid[i]

            # Piecewise quadratic derivative: dh/dT = 2*h2*(T-T_opt)
            T_diff = T_range - T_opt
            dh_samples[i, :] = np.where(
                T_range <= T_opt,
                2 * h2_low * T_diff,
                2 * h2_high * T_diff
            )
    else:
        # Quadratic model
        h1_valid, h2_valid, _ = get_valid_bootstrap_samples(result)

        if len(h1_valid) == 0:
            return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

        return _compute_quadratic_derivative_bands(h1_valid, h2_valid, T_range, percentiles)

    # Compute percentiles at each temperature
    dh_bands = []
    for p in percentiles:
        dh_bands.append(np.percentile(dh_samples, p, axis=0))

    return tuple(dh_bands)


def _compute_quadratic_derivative_bands(h1_valid, h2_valid, T_range, percentiles):
    """Helper to compute derivative bands for quadratic dh/dT = h1 + 2*h2*T."""
    if len(h1_valid) == 0:
        return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

    n_samples = len(h1_valid)
    n_T = len(T_range)
    dh_samples = np.zeros((n_samples, n_T))

    for i in range(n_samples):
        h1 = h1_valid[i]
        h2 = h2_valid[i]
        dh_samples[i, :] = h1 + 2 * h2 * T_range

    dh_bands = []
    for p in percentiles:
        dh_bands.append(np.percentile(dh_samples, p, axis=0))
    return tuple(dh_bands)


def _compute_symmetric_piecewise_derivative_bands(h2_valid, T_opt_valid, T_range, percentiles):
    """Helper to compute derivative bands for symmetric piecewise dh/dT = 2*h2*(T-T_opt)."""
    if len(h2_valid) == 0:
        return tuple(np.full_like(T_range, np.nan) for _ in percentiles)

    n_samples = len(h2_valid)
    n_T = len(T_range)
    dh_samples = np.zeros((n_samples, n_T))

    for i in range(n_samples):
        h2 = h2_valid[i]
        T_opt = T_opt_valid[i]
        # dh/dT = 2*h2*(T - T_opt)
        dh_samples[i, :] = 2 * h2 * (T_range - T_opt)

    dh_bands = []
    for p in percentiles:
        dh_bands.append(np.percentile(dh_samples, p, axis=0))
    return tuple(dh_bands)


def _compute_derivative_point_estimate(result, T, approach_key, variant=None):
    """Compute point estimate dh/dT for a given approach.

    Args:
        result: BootstrapResult
        T: Temperature array
        approach_key: The approach key (may include _high/_low suffix)
        variant: 'high', 'low', or None

    Returns:
        dh_point array
    """
    # Handle method2b (trend only - uses h3,h4 for trend response)
    if approach_key == 'method2b':
        h1 = getattr(result, 'h3_point', 0) or 0
        h2 = getattr(result, 'h4_point', 0) or 0
        return h1 + 2 * h2 * T

    # Handle method3a total (uses h2 for actual T curvature)
    if approach_key == 'method3a_high' or (approach_key == 'method3a' and variant == 'high'):
        h2 = getattr(result, 'h2_point', 0) or 0
        T_opt = result.T_opt_point
        return 2 * h2 * (T - T_opt)

    # Handle method3a low-frequency (uses h4 for trend T curvature)
    if approach_key == 'method3a_low' or (approach_key == 'method3a' and variant == 'low'):
        h2 = getattr(result, 'h4_point', 0) or 0
        T_opt = result.T_opt_point
        return 2 * h2 * (T - T_opt)

    # Handle three-interval approaches: piecewise linear derivative
    if approach_key in ('Approach TL', 'Approach TJ', 'Approach TP') and result.h4_point is not None:
        T_crit_low = getattr(result, 'T_crit_low_point', None)
        T_crit_high = getattr(result, 'T_crit_high_point', None)
        h2 = result.h2_point
        h4 = result.h4_point
        if T_crit_low is not None and T_crit_high is not None:
            below = T <= T_crit_low
            above = T >= T_crit_high
            delta = T_crit_high - T_crit_low
            if delta > 0:
                frac = np.clip((T - T_crit_low) / delta, 0, 1)
                return np.where(below, h2, np.where(above, h4, h2 + (h4 - h2) * frac))
            else:
                return np.where(T <= T_crit_low, h2, h4)
        return np.where(T <= result.T_opt_point, h2, h4)

    # Handle segmented linear approaches: derivative is step function
    if approach_key in ('Approach SL', 'Approach SJ', 'Approach SP') and result.h4_point is not None:
        T_opt = result.T_opt_point
        return np.where(T <= T_opt, result.h2_point, result.h4_point)

    # Handle piecewise approaches Approach PL/Approach PJ/Approach PP (asymmetric: h2 for T <= T_opt, h4 for T > T_opt)
    if approach_key in ('Approach PL', 'Approach PJ', 'Approach PP') and result.h4_point is not None:
        T_opt = result.T_opt_point
        h2_low = result.h2_point
        h2_high = result.h4_point
        T_diff = T - T_opt
        return np.where(
            T <= T_opt,
            2 * h2_low * T_diff,
            2 * h2_high * T_diff
        )

    # Standard quadratic model
    h1 = result.h1_point
    h2 = result.h2_point
    return h1 + 2 * h2 * T


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

    For approaches with separate high/low frequency responses (6a, 8a),
    two panels are created - one for each frequency band.

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

    if len(approaches) == 0:
        return

    # Expand approaches that need multiple panels
    plot_entries = _expand_approaches_for_plotting(approaches, results)
    n_panels = len(plot_entries)

    if n_panels == 0:
        return

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for plot_key, result_key, display_name, variant in plot_entries:
        result = results[result_key]

        # Compute uncertainty bands (90% CI and IQR)
        dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
            result, T, percentiles=(5, 25, 50, 75, 95), approach_key=plot_key
        )

        # Compute point estimate derivative
        dh_point = _compute_derivative_point_estimate(result, T, plot_key, variant)

        plot_data[plot_key] = {
            'dh_p5': dh_p5,
            'dh_p25': dh_p25,
            'dh_p75': dh_p75,
            'dh_p95': dh_p95,
            'dh_point': dh_point,
            'display_name': display_name,
        }

        # Update global y range
        if not np.all(np.isnan(dh_p5)):
            y_min = min(y_min, np.nanmin(dh_p5), np.nanmin(dh_point))
        if not np.all(np.isnan(dh_p95)):
            y_max = max(y_max, np.nanmax(dh_p95), np.nanmax(dh_point))

    # Handle case where all data is NaN
    if np.isinf(y_min) or np.isinf(y_max):
        y_min, y_max = -0.01, 0.01

    # Add some padding to y range
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Determine grid layout
    if n_panels <= 3:
        n_rows, n_cols = 1, n_panels
    elif n_panels <= 4:
        n_rows, n_cols = 2, 2
    elif n_panels <= 6:
        n_rows, n_cols = 2, 3
    elif n_panels <= 8:
        n_rows, n_cols = 4, 2
    elif n_panels <= 9:
        n_rows, n_cols = 3, 3
    elif n_panels <= 12:
        n_rows, n_cols = 4, 3
    elif n_panels <= 16:
        n_rows, n_cols = 4, 4
    elif n_panels <= 20:
        n_rows, n_cols = 5, 4
    else:
        n_rows, n_cols = 6, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Second pass: create the plots
    for idx, (plot_key, result_key, display_name, variant) in enumerate(plot_entries):
        ax = axes[idx]
        color = get_color(result_key, 'steelblue')
        pdata = plot_data[plot_key]

        # Plot 90% CI band
        ax.fill_between(T, pdata['dh_p5'], pdata['dh_p95'], alpha=0.2, color=color, label='90% CI')

        # Plot IQR band
        ax.fill_between(T, pdata['dh_p25'], pdata['dh_p75'], alpha=0.3, color=color, label='IQR')

        # Plot point estimate
        ax.plot(T, pdata['dh_point'], color=color, linestyle='-', linewidth=2, label='Point estimate')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('dh/dT', fontsize=10)
        ax.set_title(display_name, fontsize=11)
        ax.set_xlim(T_range)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

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

    # Fixed x-axis range (same as temperature response/derivative plots)
    x_min, x_max = 0, 30
    # Bins with 0.5°C width
    bins = np.arange(x_min, x_max + 0.5, 0.5)

    # Second pass: create the plots
    for idx, name in enumerate(approaches):
        ax = axes[idx]
        result = results[name]
        color = get_color(name, 'steelblue')

        # Get valid samples
        valid_samples = result.T_opt_samples[~np.isnan(result.T_opt_samples)]

        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            continue

        # Compute statistics
        point_est = result.T_opt_point
        p5 = np.percentile(valid_samples, 5)
        p25 = np.percentile(valid_samples, 25)
        p75 = np.percentile(valid_samples, 75)
        p95 = np.percentile(valid_samples, 95)

        # Plot bands and lines first (bottom layer)
        # 90% CI vertical band
        ax.axvspan(p5, p95, alpha=0.2, color=color,
                   label=f'90% CI: [{p5:.1f}, {p95:.1f}]°C')

        # IQR vertical band
        ax.axvspan(p25, p75, alpha=0.3, color=color,
                   label=f'IQR: [{p25:.1f}, {p75:.1f}]°C')

        # Point estimate (solid line, behind histogram)
        ax.axvline(x=point_est, color='black', linestyle='-', linewidth=2,
                   label=f'Point: {point_est:.1f}°C')

        # Plot histogram on top (fully saturated)
        n, _, _ = ax.hist(valid_samples, bins=bins, density=True, alpha=1.0, color=color,
                          edgecolor=color, linewidth=0.5)

        # Extend y-axis upper bound with ~10% padding
        y_max = np.max(n) * 1.1
        ax.set_ylim(0, y_max)

        ax.set_xlabel('Optimal Temperature (°C)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_approaches, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def plot_h2_histograms(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = "h2_histograms.pdf",
    approaches: list = None,
    x_range: tuple = None,
    bin_width: float = 0.0001,
    x_range_h2_high: tuple = None,
    bin_width_h2_high: float = None,
    input_file: str = None,
) -> None:
    """Plot h2 coefficient bootstrap distributions as histograms in multi-panel layout.

    Creates a figure with panels for h2 coefficients:
    - For standard approaches: h2 (quadratic coefficient)
    - For method3: h2_low (T <= T_opt) and h2_high (T > T_opt)

    Uses shaded bands for uncertainty visualization:
    - 90% CI band (light shading)
    - IQR band (darker shading)
    - Point estimate line (black)
    - Histogram bars (fully saturated)

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename (should end in .pdf)
        approaches: List of approaches to include (default: ['Approach QL', 'Approach PL'])
        x_range: Fixed x-axis range as (x_min, x_max) for h2/h2_low panels
        bin_width: Width of histogram bins for h2/h2_low panels (default: 0.0001)
        x_range_h2_high: Fixed x-axis range for h2_high panel (default: same as x_range)
        bin_width_h2_high: Width of histogram bins for h2_high panel (default: same as bin_width)
        input_file: Path to input data file (for annotation)
    """
    if approaches is None:
        approaches = ['Approach QL', 'Approach PL']

    # Default h2_high settings to main settings if not specified
    if x_range_h2_high is None:
        x_range_h2_high = x_range
    if bin_width_h2_high is None:
        bin_width_h2_high = bin_width

    # Build list of panels to plot: (samples, point_est, color, title, is_h4)
    panels = []

    for approach in approaches:
        if approach not in results:
            continue

        result = results[approach]

        if approach in ('Approach PL', 'Approach PL'):
            # Piecewise quadratic: h2 for T <= T_opt, h4 for T > T_opt
            if result.h2_samples is not None:
                valid_low = result.h2_samples[~np.isnan(result.h2_samples)]
                if len(valid_low) > 0:
                    panels.append((
                        valid_low,
                        result.h2_point,
                        get_color('Approach PL', 'magenta'),
                        'Approach 8: h₂ (T ≤ T_opt)',
                        False  # not h4
                    ))
            h4_samples = getattr(result, 'h4_samples', None)
            if h4_samples is not None:
                valid_high = h4_samples[~np.isnan(h4_samples)]
                if len(valid_high) > 0:
                    panels.append((
                        valid_high,
                        result.h4_point,
                        get_color('Approach PL', 'magenta'),
                        'Approach 8: h₄ (T > T_opt)',
                        True  # is h4
                    ))
        else:
            # Standard approaches: single h2
            if result.h2_samples is not None:
                valid_samples = result.h2_samples[~np.isnan(result.h2_samples)]
                if len(valid_samples) > 0:
                    # Format approach name for title
                    approach_num = approach.replace('approach', 'Approach ')
                    panels.append((
                        valid_samples,
                        result.h2_point,
                        get_color(approach, 'gray'),
                        f'{approach_num}: h₂',
                        False  # not h2_high
                    ))

    n_panels = len(panels)
    if n_panels == 0:
        return

    # Determine grid layout
    if n_panels <= 3:
        n_rows, n_cols = 1, n_panels
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4))
    elif n_panels <= 4:
        n_rows, n_cols = 2, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    elif n_panels <= 6:
        n_rows, n_cols = 2, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    else:
        n_cols = 3
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, (samples, point_est, color, title, is_h4) in enumerate(panels):
        ax = axes[idx]

        # Determine x-axis range and bin width for this panel
        if is_h4:
            panel_x_range = x_range_h2_high  # reuse h2_high settings for h4
            panel_bin_width = bin_width_h2_high
        else:
            panel_x_range = x_range
            panel_bin_width = bin_width

        # Compute x limits
        if panel_x_range is not None:
            x_min, x_max = panel_x_range
        else:
            x_min = np.floor(np.min(samples) * 1000) / 1000 - 0.001
            x_max = np.ceil(np.max(samples) * 1000) / 1000 + 0.001

        # Bins for histogram with fixed width
        bins = np.arange(x_min, x_max + panel_bin_width, panel_bin_width)

        # Compute statistics
        p5 = np.percentile(samples, 5)
        p25 = np.percentile(samples, 25)
        p75 = np.percentile(samples, 75)
        p95 = np.percentile(samples, 95)

        # Plot bands and lines first (bottom layer)
        # 90% CI vertical band
        ax.axvspan(p5, p95, alpha=0.2, color=color,
                   label=f'90% CI: [{p5:.4f}, {p95:.4f}]')

        # IQR vertical band
        ax.axvspan(p25, p75, alpha=0.3, color=color,
                   label=f'IQR: [{p25:.4f}, {p75:.4f}]')

        # Point estimate (solid line, behind histogram)
        ax.axvline(x=point_est, color='black', linestyle='-', linewidth=2,
                   label=f'Point: {point_est:.4f}')

        # Plot histogram on top (fully saturated)
        n, _, _ = ax.hist(samples, bins=bins, density=True, alpha=1.0, color=color,
                          edgecolor=color, linewidth=0.5)

        # Extend y-axis upper bound with ~10% padding
        y_max = np.max(n) * 1.1
        ax.set_ylim(0, y_max)

        ax.set_xlabel('h₂ Coefficient', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def plot_persistence_decay(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    data: AnalysisData = None,
    T_range: tuple = (0, 30),
    filename: str = 'fig_ApproachDL_persistence_decay.pdf',
    input_file: str = None,
) -> None:
    """Plot Approach DL (persistence decay) 2-panel figure.

    Panel layout:
        [0] h(T) - h(T_opt) temperature response (all samples)
        [1] h4 distribution histogram (all samples)

    Args:
        results: Dict of BootstrapResult (must contain 'Approach DL')
        output_dir: Directory to save the plot
        data: AnalysisData for temperature histogram overlay
        T_range: Temperature range for x-axis (default: 0-30°C)
        filename: Output filename
        input_file: Path to input data file (for annotation)
    """
    result = results['Approach DL']
    color = get_color('Approach DL', 'cyan')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T = np.linspace(T_range[0], T_range[1], 200)

    # Get bootstrap samples
    h1_samples = result.h1_samples
    h2_samples = result.h2_samples
    h4_samples = result.h4_samples

    # Point estimates
    h1_point = result.h1_point
    h2_point = result.h2_point
    h4_point = result.h4_point
    T_opt_point = result.T_opt_point if hasattr(result, 'T_opt_point') else -h1_point / (2 * h2_point)

    # Compute point estimate h(T) curve
    h_point = h1_point * T + h2_point * T**2
    if not np.isnan(T_opt_point):
        h_at_opt_point = h1_point * T_opt_point + h2_point * T_opt_point**2
        h_point = h_point - h_at_opt_point

    # Helper function to compute h(T) samples and percentiles
    def compute_h_samples_and_percentiles(h1_vals, h2_vals):
        """Compute h(T) for each bootstrap sample and return percentiles."""
        valid_mask = ~np.isnan(h1_vals) & ~np.isnan(h2_vals)
        h1_valid = h1_vals[valid_mask]
        h2_valid = h2_vals[valid_mask]

        n_valid = len(h1_valid)
        h_samples = np.zeros((n_valid, len(T)))
        for i in range(n_valid):
            h1_i, h2_i = h1_valid[i], h2_valid[i]
            h_i = h1_i * T + h2_i * T**2
            T_opt_i = -h1_i / (2 * h2_i) if h2_i != 0 else np.nan
            if not np.isnan(T_opt_i) and T_range[0] <= T_opt_i <= T_range[1]:
                h_at_opt = h1_i * T_opt_i + h2_i * T_opt_i**2
                h_samples[i] = h_i - h_at_opt
            else:
                h_samples[i] = np.nan

        h_p5 = np.nanpercentile(h_samples, 5, axis=0)
        h_p25 = np.nanpercentile(h_samples, 25, axis=0)
        h_p75 = np.nanpercentile(h_samples, 75, axis=0)
        h_p95 = np.nanpercentile(h_samples, 95, axis=0)
        return h_p5, h_p25, h_p75, h_p95

    # Helper function to plot h(T) panel
    def plot_h_T_panel(ax, h_p5, h_p25, h_p75, h_p95, title, add_temp_hist=True,
                       T_opt_override=None, T_opt_label=None):
        """Plot h(T) response panel.

        Args:
            T_opt_override: If provided, use this T_opt value instead of T_opt_point
            T_opt_label: If provided, use this label for T_opt (e.g., "T_opt (median)")
        """
        # Add temperature histogram on secondary y-axis
        if add_temp_hist and data is not None:
            ax_twin = ax.twinx()
            max_year = data.year_range[1]
            mask_recent = data.year == max_year
            temp_recent = data.temp[mask_recent]
            bins = np.linspace(T_range[0], T_range[1], 30)
            ax_twin.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
            ax_twin.set_ylabel('Data density', fontsize=8, color='gray')
            ax_twin.tick_params(axis='y', labelcolor='gray', labelsize=7)
            ax_twin.set_ylim(bottom=0)

        # Plot uncertainty bands
        ax.fill_between(T, h_p5, h_p95, alpha=0.2, color=color, label='90% CI')
        ax.fill_between(T, h_p25, h_p75, alpha=0.3, color=color, label='IQR')
        ax.plot(T, h_point, color=color, linewidth=2, label='Point estimate')

        # Mark optimal temperature with label
        T_opt_to_plot = T_opt_override if T_opt_override is not None else T_opt_point
        if not np.isnan(T_opt_to_plot):
            label = T_opt_label if T_opt_label else f'T_opt = {T_opt_to_plot:.1f}°C'
            ax.axvline(T_opt_to_plot, color=color, linestyle=':', alpha=0.7, label=label)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('h(T) - h(T_opt)')
        ax.set_title(title)
        ax.set_xlim(T_range)
        ax.set_ylim(-0.15, 0.0)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_yticks([-0.15, -0.12, -0.09, -0.06, -0.03, 0.00])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc='lower left', fontsize=8)

    # Helper function to plot h4 histogram panel
    def plot_h4_panel(ax, h4_vals, title, show_count_annotation=False, total_count=None,
                      show_median=False):
        """Plot h4 distribution panel.

        Args:
            show_median: If True, show median of displayed values instead of point estimate
        """
        valid_h4 = h4_vals[~np.isnan(h4_vals)]

        if len(valid_h4) > 0:
            # Compute statistics
            p5 = np.percentile(valid_h4, 5)
            p25 = np.percentile(valid_h4, 25)
            median = np.median(valid_h4)
            p75 = np.percentile(valid_h4, 75)
            p95 = np.percentile(valid_h4, 95)

            # Fixed x-axis range for h4 (0 to 1)
            x_min, x_max = 0, 1
            bin_width = 0.02
            bins = np.arange(x_min, x_max + bin_width, bin_width)

            # Plot histogram
            ax.hist(valid_h4, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Mark statistics with vertical lines
            if show_median:
                ax.axvline(median, color='black', linewidth=2, label=f'Median: {median:.3f}')
            else:
                ax.axvline(h4_point, color='black', linewidth=2, label=f'Point: {h4_point:.3f}')
            ax.axvline(p5, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(p95, color=color, linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'90% CI: [{p5:.3f}, {p95:.3f}]')

            # Shade IQR region
            ax.axvspan(p25, p75, alpha=0.2, color=color, label=f'IQR: [{p25:.3f}, {p75:.3f}]')

            ax.set_xlabel('h₄ (persistence decay parameter)')
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.set_xlim(x_min, x_max)
            ax.legend(loc='upper right', fontsize=8)

            # Add interpretation text
            interpretation_text = 'h₄ = 0: full persistence\nh₄ = 1: no persistence'
            if show_count_annotation and total_count is not None:
                interpretation_text += f'\n\nn = {len(valid_h4)} of {total_count}'
            ax.text(0.05, 0.95, interpretation_text,
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No h4 samples available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

    # ==================== LEFT PANEL: Temperature response ====================
    h_p5_all, h_p25_all, h_p75_all, h_p95_all = compute_h_samples_and_percentiles(h1_samples, h2_samples)
    plot_h_T_panel(axes[0], h_p5_all, h_p25_all, h_p75_all, h_p95_all,
                   'Approach 4: Temperature Response')

    # ==================== RIGHT PANEL: h4 distribution ====================
    plot_h4_panel(axes[1], h4_samples, 'Approach 4: h₄ Distribution')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_persistence_decay_derivative(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    T_range: tuple = (0, 30),
    filename: str = 'fig_ApproachDL_persistence_decay_derivative.pdf',
    input_file: str = None,
) -> None:
    """Plot Approach DL (persistence decay) derivative figure (dh/dT).

    Single panel showing dh/dT with bootstrap uncertainty.

    Args:
        results: Dict of BootstrapResult (must contain 'Approach DL')
        output_dir: Directory to save the plot
        T_range: Temperature range for x-axis (default: 0-30°C)
        filename: Output filename
        input_file: Path to input data file (for annotation)
    """
    result = results['Approach DL']
    color = get_color('Approach DL', 'cyan')

    fig, ax = plt.subplots(figsize=(8, 5))

    T = np.linspace(T_range[0], T_range[1], 200)

    # Get bootstrap samples
    h1_samples = result.h1_samples
    h2_samples = result.h2_samples

    # Point estimates
    h1_point = result.h1_point
    h2_point = result.h2_point

    # Compute point estimate dh/dT curve: dh/dT = h1 + 2*h2*T
    dh_point = h1_point + 2 * h2_point * T

    # Compute dh/dT samples and percentiles
    valid_mask = ~np.isnan(h1_samples) & ~np.isnan(h2_samples)
    h1_valid = h1_samples[valid_mask]
    h2_valid = h2_samples[valid_mask]

    n_valid = len(h1_valid)
    dh_samples = np.zeros((n_valid, len(T)))
    for i in range(n_valid):
        h1_i, h2_i = h1_valid[i], h2_valid[i]
        dh_samples[i] = h1_i + 2 * h2_i * T

    dh_p5 = np.nanpercentile(dh_samples, 5, axis=0)
    dh_p25 = np.nanpercentile(dh_samples, 25, axis=0)
    dh_p75 = np.nanpercentile(dh_samples, 75, axis=0)
    dh_p95 = np.nanpercentile(dh_samples, 95, axis=0)

    # Plot uncertainty bands
    ax.fill_between(T, dh_p5, dh_p95, alpha=0.2, color=color, label='90% CI')
    ax.fill_between(T, dh_p25, dh_p75, alpha=0.3, color=color, label='IQR')
    ax.plot(T, dh_point, color=color, linewidth=2, label='Point estimate')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('dh/dT')
    ax.set_title('Approach 4: dh/dT')
    ax.set_xlim(T_range)
    ax.set_ylim(-0.025, 0.015)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_bootstrap_gdp_scaling(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    Y_ref: float,
    Y_range: tuple = None,
    filename: str = 'bootstrap_gdp_scaling.pdf',
    data: AnalysisData = None,
    input_file: str = None,
) -> None:
    """Plot GDP scaling factor with bootstrap uncertainty bands for GDP-dependent approaches.

    Shows the spread of (Y/Y_ref)^(-beta) curves across bootstrap samples.
    Currently disabled (no GDP-dependent approaches in panels list).

    Args:
        results: Dict of BootstrapResult
        output_dir: Directory to save the plot
        Y_ref: Reference GDP value (same as used in fitting)
        Y_range: GDP range for x-axis (default: 500 to 100000)
        filename: Output filename
        data: AnalysisData for adding GDP histogram (optional)
        input_file: Path to input data file (for annotation)
    """
    # Collect panels to plot (GDP-dependent approaches use f1 for scaling exponent)
    panels = []
    for key, title, color in [
    ]:
        if key in results:
            result = results[key]
            if result.f1_point is not None and result.f1_samples is not None:
                valid_f1s = result.f1_samples[~np.isnan(result.f1_samples)]
                if len(valid_f1s) > 0:
                    panels.append((result.f1_point, valid_f1s, title, color))

    if not panels:
        return

    # Default Y range
    if Y_range is None:
        Y_range = (500, 100000)

    # Create GDP array (log-spaced)
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 6), squeeze=False)

    def _draw_panel(ax, f1_point, valid_f1s, title, color):
        # Plot individual bootstrap samples (thin lines)
        n_samples_to_plot = min(100, len(valid_f1s))
        sample_indices = np.linspace(0, len(valid_f1s) - 1, n_samples_to_plot, dtype=int)

        for idx in sample_indices:
            f1_b = valid_f1s[idx]
            g_b = (Y / Y_ref) ** (-f1_b)
            ax.plot(Y, g_b, color=color, alpha=0.05, linewidth=0.5)

        # Compute percentile bands
        g_samples = np.zeros((len(valid_f1s), len(Y)))
        for i, f1_b in enumerate(valid_f1s):
            g_samples[i, :] = (Y / Y_ref) ** (-f1_b)

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
        g_point = (Y / Y_ref) ** (-f1_point)
        ax.plot(Y, g_point, color=color, linewidth=2.5,
                label=f'Point estimate (f₁ = {f1_point:.3f})')

        # Reference lines
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref ≈ ${Y_ref:,.0f}')

        ax.set_xscale('log')
        ax.set_xlabel('Per Capita GDP ($)', fontsize=12)
        ax.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-f₁)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add f1 distribution inset
        ax_inset = ax.inset_axes([0.72, 0.42, 0.25, 0.30])
        ax_inset.hist(valid_f1s, bins=30, color=color, alpha=0.7, density=True)
        ax_inset.axvline(f1_point, color='red', linewidth=1.5, label='Point est.')
        ax_inset.set_xlabel('f₁', fontsize=9)
        ax_inset.set_ylabel('Density', fontsize=9)
        ax_inset.set_title('Bootstrap f₁ distribution', fontsize=9)
        ax_inset.tick_params(labelsize=8)

    for i, (f1_point, valid_f1s, title, color) in enumerate(panels):
        _draw_panel(axes[0, i], f1_point, valid_f1s, title, color)

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
    - plot_bootstrap_temperature_response() for basic, precomputed k, and LOESS approaches
    - plot_bootstrap_temperature_derivative() for all approaches
    - plot_bootstrap_T_optimal_comparison() for all approaches

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save plots
        T_range: Temperature range for response plots
        Y_ref: Reference GDP (unused, kept for API compatibility)
        data: AnalysisData for adding data density histograms (optional)
        input_file: Path to input data file (for annotation)
    """
    # Generate combined distribution plot for all approaches
    plot_all_bootstrap_distributions(results, all_stats, output_dir, input_file=input_file)
    print("      Saved bootstrap_distributions.pdf")

    # Temperature response PDF 1: Basic methods (method0, method1)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach QJ', 'Approach QP'],
        filename='bootstrap_temperature_response_basic.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_basic.pdf")

    # Temperature response PDF 2: Precomputed k methods (method1, method2)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach QP', 'Approach QL'],
        filename='bootstrap_temperature_response_precomputed.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_precomputed.pdf")

    # Temperature response PDF 3: LOESS methods (Approach QL, Approach PL, Approach DL)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach QL', 'Approach PL', 'Approach DL'],
        filename='bootstrap_temperature_response_loess.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_loess.pdf")

    # Temperature response PDF 4: Conjoined approaches (Approach PJ, Approach DJ)
    # These combine piecewise/persistence climate response with full OLS for j_i(t) and k(t)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach PJ', 'Approach DJ'],
        filename='bootstrap_temperature_response_conjoined.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_conjoined.pdf")

    # Temperature response PDF 5: Comparison of piecewise approaches (Approach PL vs Approach PJ)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach PL', 'Approach PJ'],
        filename='bootstrap_temperature_response_piecewise_comparison.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_piecewise_comparison.pdf")

    # Temperature response PDF 6: Comparison of persistence approaches (Approach DL vs Approach DJ)
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['Approach DL', 'Approach DJ'],
        filename='bootstrap_temperature_response_persistence_comparison.pdf',
        T_range=T_range,
        data=data,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_response_persistence_comparison.pdf")

    # Temperature derivative plot - all methods in one PDF
    plot_bootstrap_temperature_derivative(
        results, output_dir,
        approaches=['Approach QJ', 'Approach QP', 'Approach QL', 'Approach PL', 'Approach DL', 'Approach PJ', 'Approach DJ'],
        filename='bootstrap_temperature_derivative.pdf',
        T_range=T_range,
        input_file=input_file
    )
    print("      Saved bootstrap_temperature_derivative.pdf")

    # T_optimal comparison across all approaches
    plot_bootstrap_T_optimal_comparison(results, all_stats, output_dir, input_file=input_file)
    print("      Saved bootstrap_T_optimal_comparison.pdf")

    # Year effects k(t) with bootstrap uncertainty bands
    if data is not None:
        plot_year_effects_bootstrap(results, data, output_dir, input_file=input_file)


def plot_temperature_response_4panel_variants(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    filename: str = 'fig_temperature_response_4panel_variants.pdf',
    T_range: tuple = (0, 30),
    T_dep_range: tuple = (-1.5, 1.5),
    input_file: str = None,
) -> None:
    """Plot 4-panel temperature response figure with approach 2, 3, and 4 components.

    Creates a 2x2 figure:
    - Top-left: Approach 2 h(T) bootstrap response (LOESS quadratic)
    - Top-right: Approach 3 h(T) bootstrap response (piecewise quadratic)
    - Bottom-left: Approach 4 h(T) bootstrap response (persistence decay)
    - Bottom-right: Approach 4 persistence decay component

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for temperature histogram
        output_dir: Directory to save the plot
        filename: Output filename
        T_range: Temperature range for panels (default: (0, 30))
        T_dep_range: Unused, kept for API compatibility
        input_file: Optional input file path for annotation
    """
    required = ['Approach QL', 'Approach PL', 'Approach DL']
    valid_approaches = [a for a in required if a in results]
    if len(valid_approaches) < 3:
        print(f"  WARNING: Not enough valid approaches for 4-panel figure (found {valid_approaches})")
        return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Temperature arrays
    T = np.linspace(T_range[0], T_range[1], 200)
    T_dep = np.linspace(T_dep_range[0], T_dep_range[1], 200)

    # Fixed y-axis range for publication consistency
    y_min, y_max = -0.15, 0.00
    y_ticks = np.arange(-0.15, 0.01, 0.03)

    # Get temperature data from most recent year for histogram
    temp_recent = None
    T_dep_actual = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

        # Compute T_dep (temperature departure from LOESS trend) for bottom-right panel histogram
        year_means = compute_year_means(data)
        trends_loess = compute_country_trends_loess(data, year_means)
        T_dep_actual = data.temp - trends_loess.T_loess

    # Top-left: Approach 6 (standard quadratic)
    ax = axes[0, 0]
    result6 = results['Approach QL']
    color6 = get_color('Approach QL', 'orange')
    h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
        result6, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach QL'
    )
    h_point6, T_opt6 = _compute_point_estimate_response(result6, T, 'Approach QL', None)

    if temp_recent is not None:
        ax2 = ax.twinx()
        bins = np.linspace(T_range[0], T_range[1], 30)
        ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax2.set_ylabel('Data density', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax2.set_ylim(bottom=0)
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.fill_between(T, h_p5, h_p95, alpha=0.2, color=color6, label='90% CI')
    ax.fill_between(T, h_p25, h_p75, alpha=0.3, color=color6, label='IQR')
    ax.plot(T, h_point6, color=color6, linestyle='-', linewidth=2, label='Point estimate')
    if T_opt6 is not None and not np.isnan(T_opt6):
        ax.axvline(T_opt6, color=color6, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt6:.1f}°C')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
    ax.set_title('Approach 6: LOESS quadratic', fontsize=11)
    ax.set_xlim(T_range)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    # Top-right: Approach 8 (piecewise quadratic)
    ax = axes[0, 1]
    result8 = results['Approach PL']
    color8 = get_color('Approach PL', 'magenta')
    h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
        result8, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach PL'
    )
    h_point8, T_opt8 = _compute_point_estimate_response(result8, T, 'Approach PL', None)

    if temp_recent is not None:
        ax2 = ax.twinx()
        bins = np.linspace(T_range[0], T_range[1], 30)
        ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax2.set_ylabel('Data density', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax2.set_ylim(bottom=0)
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.fill_between(T, h_p5, h_p95, alpha=0.2, color=color8, label='90% CI')
    ax.fill_between(T, h_p25, h_p75, alpha=0.3, color=color8, label='IQR')
    ax.plot(T, h_point8, color=color8, linestyle='-', linewidth=2, label='Point estimate')
    if T_opt8 is not None and not np.isnan(T_opt8):
        ax.axvline(T_opt8, color=color8, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt8:.1f}°C')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
    ax.set_title('Approach 8: Piecewise quadratic', fontsize=11)
    ax.set_xlim(T_range)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    # Bottom-left: Approach 4 (persistence decay)
    ax = axes[1, 0]
    result4 = results['Approach DL']
    color4 = get_color('Approach DL', 'cyan')
    h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
        result4, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach DL'
    )
    h_point4, T_opt4 = _compute_point_estimate_response(result4, T, 'Approach DL', None)

    if temp_recent is not None:
        ax2 = ax.twinx()
        bins = np.linspace(T_range[0], T_range[1], 30)
        ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax2.set_ylabel('Data density', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax2.set_ylim(bottom=0)
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.fill_between(T, h_p5, h_p95, alpha=0.2, color=color4, label='90% CI')
    ax.fill_between(T, h_p25, h_p75, alpha=0.3, color=color4, label='IQR')
    ax.plot(T, h_point4, color=color4, linestyle='-', linewidth=2, label='Point estimate')
    if T_opt4 is not None and not np.isnan(T_opt4):
        ax.axvline(T_opt4, color=color4, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt4:.1f}°C')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
    ax.set_title('Approach 4: Persistence decay', fontsize=11)
    ax.set_xlim(T_range)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    # Bottom-right: Approach 4 persistence parameter distribution
    ax = axes[1, 1]
    h4_samples = getattr(result4, 'h4_samples', None)
    h4_point = getattr(result4, 'h4_point', None)

    if h4_samples is not None and h4_point is not None:
        h4_valid = h4_samples[~np.isnan(h4_samples)]
        # Plot histogram of h4 (persistence parameter)
        ax.hist(h4_valid, bins=30, density=True, color=color4, alpha=0.6, edgecolor='black')
        ax.axvline(h4_point, color='red', linestyle='-', linewidth=2, label=f'Point est: {h4_point:.3f}')
        ax.axvline(np.median(h4_valid), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(h4_valid):.3f}')
        ax.axvline(np.percentile(h4_valid, 5), color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(np.percentile(h4_valid, 95), color='gray', linestyle=':', linewidth=1.5)
        ax.set_xlabel('h₄ (persistence parameter)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Approach 4: Persistence parameter distribution', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    else:
        ax.text(0.5, 0.5, 'No h4 samples available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Approach 4: Persistence parameter', fontsize=11)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_temperature_derivative_4panel_variants(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = 'fig_temperature_derivative_4panel_variants.pdf',
    T_range: tuple = (0, 30),
    T_dep_range: tuple = (-1.5, 1.5),
    input_file: str = None,
) -> None:
    """Plot 4-panel temperature derivative figure with approach 2, 3, and 4 components.

    Creates a 2x2 figure:
    - Top-left: Approach 2 dh/dT = h1 + 2*h2*T (LOESS quadratic)
    - Top-right: Approach 3 piecewise dh/dT
    - Bottom-left: Approach 4 dh/dT (persistence decay)
    - Bottom-right: Approach 4 persistence parameter distribution

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename
        T_range: Temperature range for panels (default: (0, 30))
        T_dep_range: Unused, kept for API compatibility
        input_file: Optional input file path for annotation
    """
    required = ['Approach QL', 'Approach PL', 'Approach DL']
    valid_approaches = [a for a in required if a in results]
    if len(valid_approaches) < 3:
        print(f"  WARNING: Not enough valid approaches for 4-panel derivative figure (found {valid_approaches})")
        return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Temperature arrays
    T = np.linspace(T_range[0], T_range[1], 200)
    T_dep = np.linspace(T_dep_range[0], T_dep_range[1], 200)

    # Top-left: Approach 6 (standard quadratic derivative)
    ax = axes[0, 0]
    result6 = results['Approach QL']
    color6 = get_color('Approach QL', 'orange')

    dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
        result6, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach QL'
    )
    dh_point6 = _compute_derivative_point_estimate(result6, T, 'Approach QL', None)

    ax.fill_between(T, dh_p5, dh_p95, alpha=0.2, color=color6, label='90% CI')
    ax.fill_between(T, dh_p25, dh_p75, alpha=0.3, color=color6, label='IQR')
    ax.plot(T, dh_point6, color=color6, linestyle='-', linewidth=2, label='Point estimate')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('dh/dT', fontsize=10)
    ax.set_title('Approach 6: LOESS quadratic', fontsize=11)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    # Top-right: Approach 8 (piecewise quadratic derivative)
    ax = axes[0, 1]
    result8 = results['Approach PL']
    color8 = get_color('Approach PL', 'magenta')

    dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
        result8, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach PL'
    )
    dh_point8 = _compute_derivative_point_estimate(result8, T, 'Approach PL', None)

    ax.fill_between(T, dh_p5, dh_p95, alpha=0.2, color=color8, label='90% CI')
    ax.fill_between(T, dh_p25, dh_p75, alpha=0.3, color=color8, label='IQR')
    ax.plot(T, dh_point8, color=color8, linestyle='-', linewidth=2, label='Point estimate')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('dh/dT', fontsize=10)
    ax.set_title('Approach 8: Piecewise quadratic', fontsize=11)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    # Bottom-left: Approach 4 derivative (persistence decay)
    ax = axes[1, 0]
    result4 = results['Approach DL']
    color4 = get_color('Approach DL', 'cyan')

    dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
        result4, T, percentiles=(5, 25, 50, 75, 95), approach_key='Approach DL'
    )
    dh_point4 = _compute_derivative_point_estimate(result4, T, 'Approach DL', None)

    ax.fill_between(T, dh_p5, dh_p95, alpha=0.2, color=color4, label='90% CI')
    ax.fill_between(T, dh_p25, dh_p75, alpha=0.3, color=color4, label='IQR')
    ax.plot(T, dh_point4, color=color4, linestyle='-', linewidth=2, label='Point estimate')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('dh/dT', fontsize=10)
    ax.set_title('Approach 4: Persistence decay', fontsize=11)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    # Bottom-right: Approach 4 persistence parameter distribution
    ax = axes[1, 1]
    h4_samples = getattr(result4, 'h4_samples', None)
    h4_point = getattr(result4, 'h4_point', None)

    if h4_samples is not None and h4_point is not None:
        h4_valid = h4_samples[~np.isnan(h4_samples)]
        # Plot histogram of h4 (persistence parameter)
        ax.hist(h4_valid, bins=30, density=True, color=color4, alpha=0.6, edgecolor='black')
        ax.axvline(h4_point, color='red', linestyle='-', linewidth=2, label=f'Point est: {h4_point:.3f}')
        ax.axvline(np.median(h4_valid), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(h4_valid):.3f}')
        ax.axvline(np.percentile(h4_valid, 5), color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(np.percentile(h4_valid, 95), color='gray', linestyle=':', linewidth=1.5)
        ax.set_xlabel('h₄ (persistence parameter)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Approach 4: Persistence parameter distribution', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    else:
        ax.text(0.5, 0.5, 'No h4 samples available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Approach 4: Persistence parameter', fontsize=11)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ==============================================================================
# 3x3 Grid Plotting Functions
# ==============================================================================

# Systematic 3x3 grid layout:
# - Rows = Climate response functions (1=quadratic, 2=piecewise, 3=persistence)
# - Columns = Solution methods (J=Joint, P=Polynomial, L=LOESS)
#
RESPONSE_TYPE_ORDER = ['Q', 'P', 'S', 'T', 'D', 'L']
TREND_TYPE_ORDER = ['J', 'P', 'L']

RESPONSE_TYPE_LABELS = {
    'Q': 'Quadratic (Q)',
    'P': 'Piecewise (P)',
    'S': 'Segmented (S)',
    'T': 'Three-Interval (T)',
    'D': 'Decay (D)',
    'L': 'Level (L)',
}
TREND_TYPE_LABELS = {
    'J': 'Joint (J)',
    'P': 'Polynomial (P)',
    'L': 'LOESS (L)',
}


def build_approach_grid(approach_keys):
    """Build NxM grid from approach names, ordered by canonical type ordering.

    Excludes N-type (null model) approaches — they have no climate response
    to plot on temperature response, derivative, or coefficient figures.

    Returns (grid, response_types, trend_types) where grid[i][j] is an
    approach name or None.
    """
    codes = {}
    for key in approach_keys:
        code = key.split()[-1]  # "Approach QJ" → "QJ"
        if code[0] != 'N':  # Exclude null models
            codes[code] = key
    response_types = [r for r in RESPONSE_TYPE_ORDER if any(c[0] == r for c in codes)]
    trend_types = [t for t in TREND_TYPE_ORDER if any(c[1] == t for c in codes)]
    grid = []
    for r in response_types:
        row = []
        for t in trend_types:
            code = r + t
            row.append(codes.get(code))  # None if not in results
        grid.append(row)
    return grid, response_types, trend_types


def plot_temperature_response_3x3(
    results: Dict[str, "BootstrapResult"],
    data: AnalysisData,
    output_dir: Path,
    filename: str = None,
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot NxM temperature response figure (h(T) - h(T_opt)).

    Grid dimensions are determined dynamically from the approaches present
    in results.

    Args:
        results: Dict of BootstrapResult for each approach
        data: AnalysisData for temperature histogram
        output_dir: Directory to save the plot
        filename: Output filename (default: auto-sized, e.g. fig_temperature_response_3x3.pdf)
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    grid, response_types, trend_types = build_approach_grid(results.keys())
    nrows = len(grid)
    ncols = len(grid[0]) if grid else 0
    if nrows == 0 or ncols == 0:
        return

    if filename is None:
        filename = f'fig_temperature_response_{nrows}x{ncols}.pdf'

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    # Temperature array for response plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Get temperature data from most recent year for histogram
    temp_recent = None
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        temp_recent = data.temp[mask_recent]

    # Pre-compute uncertainty bands and point estimates for all approaches
    precomputed = {}
    iqr_extremes = []
    for row in range(nrows):
        for col in range(ncols):
            name = grid[row][col]
            if name is None or name not in results:
                continue
            result = results[name]
            h_p5, h_p25, h_p50, h_p75, h_p95 = compute_h_response_uncertainty_bands(
                result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
            )
            h_point, T_opt = _compute_point_estimate_response(result, T, name, None)
            precomputed[name] = (h_p5, h_p25, h_p50, h_p75, h_p95, h_point, T_opt)
            iqr_extremes.extend([np.nanmin(h_p25), np.nanmax(h_p75)])

    # Compute y-axis bounds from IQR extremes across all approaches
    if iqr_extremes:
        (y_min, y_max), y_ticks = get_axis_bounds_and_ticks(iqr_extremes)
    else:
        y_min, y_max = -0.15, 0.00
        y_ticks = np.arange(-0.15, 0.01, 0.03)

    # Plot each panel
    for row in range(nrows):
        for col in range(ncols):
            name = grid[row][col]
            ax = axes[row, col]
            display_name = name if name is not None else f'Approach {response_types[row]}{trend_types[col]}'

            if name is None or name not in precomputed:
                ax.text(0.5, 0.5, f'{display_name}\n(not available)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.set_xlim(T_range)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('Temperature (°C)', fontsize=10)
                ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
                ax.grid(True, alpha=0.3)
                continue

            result = results[name]
            color = get_color(name, 'steelblue')

            # Retrieve pre-computed bands and point estimate
            h_p5, h_p25, h_p50, h_p75, h_p95, h_point, T_opt = precomputed[name]

            # Add temperature histogram on secondary y-axis
            if temp_recent is not None:
                ax2 = ax.twinx()
                bins = np.linspace(T_range[0], T_range[1], 30)
                ax2.hist(temp_recent, bins=bins, color='gray', alpha=0.3, density=True)
                ax2.set_ylabel('Data density', fontsize=8, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
                ax2.set_ylim(bottom=0)
                ax2.set_zorder(ax.get_zorder() - 1)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)

            # Plot 90% CI band
            ax.fill_between(T, h_p5, h_p95, alpha=0.2, color=color, label='90% CI')
            # Plot IQR band
            ax.fill_between(T, h_p25, h_p75, alpha=0.3, color=color, label='IQR')
            # Plot point estimate
            ax.plot(T, h_point, color=color, linestyle='-', linewidth=2, label='Point estimate')

            # Mark optimal temperature
            if T_opt is not None and not np.isnan(T_opt):
                ax.axvline(T_opt, color=color, linestyle=':', alpha=0.7, label=f'T_opt = {T_opt:.1f}°C')

            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_xlabel('Temperature (°C)', fontsize=10)
            ax.set_ylabel('h(T) - h(T_opt)', fontsize=10)
            ax.set_title(result.approach, fontsize=11)
            ax.set_xlim(T_range)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_temperature_derivative_3x3(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = None,
    T_range: tuple = (0, 30),
    input_file: str = None,
) -> None:
    """Plot NxM temperature derivative figure (dh/dT).

    Grid dimensions are determined dynamically from the approaches present
    in results.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename (default: auto-sized)
        T_range: Temperature range for x-axis (default: (0, 30))
        input_file: Optional input file path for annotation
    """
    grid, response_types, trend_types = build_approach_grid(results.keys())
    nrows = len(grid)
    ncols = len(grid[0]) if grid else 0
    if nrows == 0 or ncols == 0:
        return

    if filename is None:
        filename = f'fig_temperature_derivative_{nrows}x{ncols}.pdf'

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    # Temperature array for derivative plots
    T = np.linspace(T_range[0], T_range[1], 200)

    # Pre-compute uncertainty bands and point estimates for all approaches
    precomputed = {}
    iqr_extremes = []
    for row in range(nrows):
        for col in range(ncols):
            name = grid[row][col]
            if name is None or name not in results:
                continue
            result = results[name]
            dh_p5, dh_p25, dh_p50, dh_p75, dh_p95 = compute_derivative_uncertainty_bands(
                result, T, percentiles=(5, 25, 50, 75, 95), approach_key=name
            )
            dh_point = _compute_derivative_point_estimate(result, T, name, None)
            precomputed[name] = (dh_p5, dh_p25, dh_p50, dh_p75, dh_p95, dh_point)
            iqr_extremes.extend([np.nanmin(dh_p25), np.nanmax(dh_p75)])

    # Compute y-axis bounds from IQR extremes across all approaches
    if iqr_extremes:
        (y_min, y_max), y_ticks = get_axis_bounds_and_ticks(iqr_extremes)
    else:
        y_min, y_max = -0.025, 0.015
        y_ticks = np.arange(-0.025, 0.016, 0.005)

    # Plot each panel
    for row in range(nrows):
        for col in range(ncols):
            name = grid[row][col]
            ax = axes[row, col]
            display_name = name if name is not None else f'Approach {response_types[row]}{trend_types[col]}'

            if name is None or name not in precomputed:
                ax.text(0.5, 0.5, f'{display_name}\n(not available)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.set_xlim(T_range)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('Temperature (°C)', fontsize=10)
                ax.set_ylabel('dh/dT', fontsize=10)
                ax.grid(True, alpha=0.3)
                continue

            result = results[name]
            color = get_color(name, 'steelblue')

            # Retrieve pre-computed bands and point estimate
            dh_p5, dh_p25, dh_p50, dh_p75, dh_p95, dh_point = precomputed[name]

            # Plot 90% CI band
            ax.fill_between(T, dh_p5, dh_p95, alpha=0.2, color=color, label='90% CI')
            # Plot IQR band
            ax.fill_between(T, dh_p25, dh_p75, alpha=0.3, color=color, label='IQR')
            # Plot point estimate
            ax.plot(T, dh_point, color=color, linestyle='-', linewidth=2, label='Point estimate')

            # Mark zero line
            ax.axhline(0, color='gray', linewidth=0.5)

            ax.set_xlabel('Temperature (°C)', fontsize=10)
            ax.set_ylabel('dh/dT', fontsize=10)
            ax.set_title(result.approach, fontsize=11)
            ax.set_xlim(T_range)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_T_optimal_histogram_3x3(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = None,
    input_file: str = None,
) -> None:
    """Plot NxM T_optimal bootstrap histogram figure.

    Grid dimensions are determined dynamically from the approaches present
    in results.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename (default: auto-sized)
        input_file: Optional input file path for annotation
    """
    grid, response_types, trend_types = build_approach_grid(results.keys())
    nrows = len(grid)
    ncols = len(grid[0]) if grid else 0
    if nrows == 0 or ncols == 0:
        return

    if filename is None:
        filename = f'fig_T_optimal_histogram_{nrows}x{ncols}.pdf'

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    # Fixed x-axis range (same as temperature response/derivative plots)
    x_min, x_max = 0, 30
    # Bins with 0.5°C width
    bins = np.arange(x_min, x_max + 0.5, 0.5)

    # Plot each panel
    for row in range(nrows):
        for col in range(ncols):
            name = grid[row][col]
            ax = axes[row, col]
            display_name = name if name is not None else f'Approach {response_types[row]}{trend_types[col]}'

            if name is None or name not in results:
                ax.text(0.5, 0.5, f'{display_name}\n(not available)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.set_xlim(x_min, x_max)
                ax.set_xlabel('Optimal Temperature (°C)', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.grid(True, alpha=0.3)
                continue

            result = results[name]
            color = get_color(name, 'steelblue')

            # Get valid samples
            valid_samples = result.T_opt_samples[~np.isnan(result.T_opt_samples)]

            if len(valid_samples) == 0:
                ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                continue

            # Compute statistics
            point_est = result.T_opt_point
            p5 = np.percentile(valid_samples, 5)
            p25 = np.percentile(valid_samples, 25)
            p75 = np.percentile(valid_samples, 75)
            p95 = np.percentile(valid_samples, 95)

            # Plot bands and lines first (bottom layer)
            # 90% CI vertical band
            ax.axvspan(p5, p95, alpha=0.2, color=color,
                       label=f'90% CI: [{p5:.1f}, {p95:.1f}]°C')

            # IQR vertical band
            ax.axvspan(p25, p75, alpha=0.3, color=color,
                       label=f'IQR: [{p25:.1f}, {p75:.1f}]°C')

            # Point estimate (solid line, behind histogram)
            ax.axvline(x=point_est, color='black', linestyle='-', linewidth=2,
                       label=f'Point: {point_est:.1f}°C')

            # Plot histogram on top (fully saturated)
            n, _, _ = ax.hist(valid_samples, bins=bins, density=True, alpha=1.0, color=color,
                              edgecolor=color, linewidth=0.5)

            (y_lo, y_hi), y_ticks = get_axis_bounds_and_ticks([0, np.max(n)], 0.1)
            ax.set_ylim(y_lo, y_hi)
            ax.set_yticks(y_ticks)

            ax.set_xlabel('Optimal Temperature (°C)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(result.approach, fontsize=11)
            ax.set_xlim(x_min, x_max)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_h2_histogram_4x3(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = None,
    input_file: str = None,
) -> None:
    """Plot h2 coefficient histogram figure with dynamic grid.

    Dynamically builds rows based on which non-N response types are present.
    One row per response type for h2, plus an extra row for Piecewise h4
    (only if P-type approaches are present).

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename (default: auto-sized)
        input_file: Optional input file path for annotation
    """
    grid, response_types, trend_types = build_approach_grid(results.keys())
    ncols = len(trend_types)
    if not response_types or ncols == 0:
        return

    # Build row definitions: each is (row_label, approach_names, sample_attr, point_attr, x_label)
    row_defs = []
    for r in response_types:
        cols = [f'Approach {r}{t}' for t in trend_types]
        row_defs.append((r, cols, 'h2_samples', 'h2_point', 'h₂ Coefficient', None))
        if r == 'P':
            # Extra row for piecewise h4
            row_defs.append(('P_h4', cols, 'h4_samples', 'h4_point', 'h₄ Coefficient', 'h₄ (T > T_opt)'))

    nrows = len(row_defs)

    if filename is None:
        filename = f'fig_h2_histogram_{nrows}x{ncols}.pdf'

    # Collect all sample values to determine dynamic x-axis bounds
    all_samples = []
    for _, cols, sample_attr, _, _, _ in row_defs:
        for name in cols:
            if name in results:
                samples = getattr(results[name], sample_attr, None)
                if samples is not None:
                    all_samples.append(samples[~np.isnan(samples)])
    all_vals = np.concatenate(all_samples) if all_samples else np.array([-0.001, 0.0001])
    (x_min, x_max), _ = get_axis_bounds_and_ticks([all_vals.min(), all_vals.max()], padding=0.05)
    bin_width = (x_max - x_min) / 50
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    def plot_histogram_panel(ax, samples, point_est, color, title, x_label='h₂ Coefficient'):
        """Helper to plot a single histogram panel."""
        if samples is None:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            return

        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            return

        # Compute statistics
        p5 = np.percentile(valid_samples, 5)
        p25 = np.percentile(valid_samples, 25)
        p75 = np.percentile(valid_samples, 75)
        p95 = np.percentile(valid_samples, 95)

        # Plot bands and lines first (bottom layer)
        ax.axvspan(p5, p95, alpha=0.2, color=color,
                   label=f'90% CI: [{p5:.5f}, {p95:.5f}]')
        ax.axvspan(p25, p75, alpha=0.3, color=color,
                   label=f'IQR: [{p25:.5f}, {p75:.5f}]')
        if point_est is not None:
            ax.axvline(x=point_est, color='black', linestyle='-', linewidth=2,
                       label=f'Point: {point_est:.5f}')

        # Plot histogram on top
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            n, _, _ = ax.hist(valid_samples, bins=bins, density=True, alpha=1.0, color=color,
                              edgecolor=color, linewidth=0.5)

        # Compute y-axis bounds from histogram data
        n_max = np.nanmax(n) if len(n) > 0 and np.any(~np.isnan(n)) else 1.0
        if np.isnan(n_max) or n_max <= 0:
            n_max = 1.0
        (y_lo, y_hi), y_ticks = get_axis_bounds_and_ticks([0, n_max], 0.1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y_ticks)

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    for row_idx, (row_type, cols, sample_attr, point_attr, x_label, title_suffix) in enumerate(row_defs):
        for col_idx, name in enumerate(cols):
            ax = axes[row_idx, col_idx]
            if name not in results:
                ax.text(0.5, 0.5, f'{name}\n(not available)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.set_xlim(x_min, x_max)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.grid(True, alpha=0.3)
                continue

            result = results[name]
            color = get_color(name, 'steelblue')
            samples = getattr(result, sample_attr, None)
            point_est = getattr(result, point_attr, None)
            title = f'{result.approach}: {title_suffix}' if title_suffix else result.approach
            plot_histogram_panel(ax, samples, point_est, color, title, x_label)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_h4_histogram_1x3(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    filename: str = None,
    input_file: str = None,
) -> None:
    """Plot 1xM h4 (persistence parameter) histogram figure.

    Dynamically finds all D-type approaches in results and plots 1×M
    where M = number of trend types with D approaches.

    Args:
        results: Dict of BootstrapResult for each approach
        output_dir: Directory to save the plot
        filename: Output filename (default: auto-sized)
        input_file: Optional input file path for annotation
    """
    # Find D-type approaches present in results
    trend_types = [t for t in TREND_TYPE_ORDER
                   if f'Approach D{t}' in results]
    if not trend_types:
        return

    approaches = [f'Approach D{t}' for t in trend_types]
    ncols = len(approaches)

    if filename is None:
        filename = f'fig_h4_histogram_1x{ncols}.pdf'

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)

    # Plot each panel
    for col, name in enumerate(approaches):
        ax = axes[0, col]

        result = results[name]
        color = get_color(name, 'steelblue')

        # Get h4 samples
        h4_samples = getattr(result, 'h4_samples', None)
        h4_point = getattr(result, 'h4_point', None)

        if h4_samples is None:
            ax.text(0.5, 0.5, 'No h4 samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(result.approach if hasattr(result, 'approach') else name, fontsize=11)
            continue

        valid_samples = h4_samples[~np.isnan(h4_samples)]
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            continue

        # Compute statistics
        p5 = np.percentile(valid_samples, 5)
        p25 = np.percentile(valid_samples, 25)
        p75 = np.percentile(valid_samples, 75)
        p95 = np.percentile(valid_samples, 95)

        # Plot bands and lines first (bottom layer)
        ax.axvspan(p5, p95, alpha=0.2, color=color,
                   label=f'90% CI: [{p5:.3f}, {p95:.3f}]')
        ax.axvspan(p25, p75, alpha=0.3, color=color,
                   label=f'IQR: [{p25:.3f}, {p75:.3f}]')
        if h4_point is not None:
            ax.axvline(x=h4_point, color='black', linestyle='-', linewidth=2,
                       label=f'Point: {h4_point:.3f}')

        # Plot histogram on top
        n, _, _ = ax.hist(valid_samples, bins=30, density=True, alpha=1.0, color=color,
                          edgecolor=color, linewidth=0.5)

        (y_lo, y_hi), y_ticks = get_axis_bounds_and_ticks([0, np.max(n)], 0.1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y_ticks)

        ax.set_xlabel('h₄ (persistence parameter)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(result.approach, fontsize=11)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_input_file_annotation(fig, input_file)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")
