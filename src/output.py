"""Output and visualization for detrended response analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict
from .data_loader import AnalysisData
from .detrending import CountryTrends
from .fitting import FitResult

# Import for type hints - bootstrap module imported at end to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bootstrap import BootstrapResult


def create_output_dir(base_dir: str = "data/output") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_summary_table(results: Dict[str, FitResult], output_dir: Path) -> None:
    """Save comparison table of all approaches."""
    rows = []
    for name, r in results.items():
        rows.append({
            'Approach': r.approach,
            'h1': r.h1,
            'h1_SE': r.h1_se,
            'h2': r.h2,
            'h2_SE': r.h2_se,
            'T_optimal': r.T_optimal,
            'R_squared': r.r_squared,
            'Total_R_squared': r.total_r_squared,
            'Adj_R_squared': r.adj_r_squared,
            'RMSE': r.rmse,
            'n_obs': r.n_obs,
            'n_params': r.n_params,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'comparison_table.csv', index=False)
    df.to_excel(output_dir / 'comparison_table.xlsx', index=False)

    # Also save as formatted text
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("Detrended Response Analysis - Comparison of Approaches\n")
        f.write("=" * 70 + "\n\n")

        for name, r in results.items():
            f.write(f"{r.approach}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  h1 = {r.h1:12.6f}  (SE: {r.h1_se:.6f})\n")
            f.write(f"  h2 = {r.h2:12.6f}  (SE: {r.h2_se:.6f})\n")
            f.write(f"  T_optimal = {r.T_optimal:.2f} C\n")
            f.write(f"  R² = {r.r_squared:.4f}\n")
            f.write(f"  Total R² = {r.total_r_squared:.4f}\n")
            f.write(f"  Adjusted R² = {r.adj_r_squared:.4f}\n")
            f.write(f"  RMSE = {r.rmse:.6f}\n")
            f.write(f"  Observations: {r.n_obs}\n")
            f.write(f"  Parameters: {r.n_params}\n")
            f.write("\n")


def save_country_trends(
    data: AnalysisData, trends: CountryTrends, output_dir: Path
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
    df.to_csv(output_dir / 'country_trends.csv', index=False)
    df.to_excel(output_dir / 'country_trends.xlsx', index=False)


def _plot_temperature_response_subset(
    results: Dict[str, FitResult], output_dir: Path,
    approaches: list, filename: str, title_suffix: str = "",
    T_range: tuple = (0, 30)
) -> None:
    """Plot h(T) - h(T*) for a subset of approaches.

    This shows the temperature response relative to the optimal temperature,
    so the maximum is at y=0 for each curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.linspace(T_range[0], T_range[1], 200)

    # Color scheme (degree of detrending):
    # - No detrending: black
    # - Mixed (linear T + quadratic GDP): red
    # - Linear: green
    # - Quadratic: blue
    colors = {
        'approach0': 'black',
        'approach1': 'green',      # Linear Temperature Detrending
        'approach2': 'blue',       # Quadratic GDP Growth Detrending
        'approach3': 'red',        # Combined Detrending (Mixed)
        'approach4': 'green',      # Combined Linear Detrending
        'approach5': 'blue',       # Combined Quadratic Detrending
        'approach6': 'green',      # Precomputed k Linear
        'approach7': 'blue',       # Precomputed k Quadratic
    }
    # Line style scheme (what's being detrended):
    # - No detrending or combined (both): solid
    # - GDP growth detrending only: dashed
    # - Temperature detrending only: dotted
    # - Precomputed k approaches: dash-dot
    linestyles = {
        'approach0': '-',
        'approach1': ':',
        'approach2': '--',
        'approach3': '-',
        'approach4': '-',
        'approach5': '-',
        'approach6': '-.',
        'approach7': '-.',
    }

    for name in approaches:
        if name not in results:
            continue
        r = results[name]
        # h(T) = h1*T + h2*T²
        h_T = r.h1 * T + r.h2 * T ** 2

        # h(T*) = h1*T* + h2*T*² = -h1²/(4*h2) when T* = -h1/(2*h2)
        if r.h2 != 0:
            h_T_opt = -r.h1 ** 2 / (4 * r.h2)
        else:
            h_T_opt = 0

        # Plot h(T) - h(T*)
        h_relative = h_T - h_T_opt

        label = f"{r.approach} (T_opt = {r.T_optimal:.1f}°C)"
        ax.plot(T, h_relative, color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'), label=label, linewidth=2)

        # Mark optimal temperature
        ax.axvline(r.T_optimal, color=colors.get(name, 'gray'),
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
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_temperature_response(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30)
) -> None:
    """Plot h(T) - h(T*) for approaches, generating two separate plots."""
    # Plot 1: Approaches 0-5 (all original approaches)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3', 'approach4', 'approach5'],
        filename='temperature_response_all.png',
        title_suffix='Approaches 0-5',
        T_range=T_range
    )
    # Plot 2: Approaches 0, 6, 7 (precomputed k approaches)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach0', 'approach6', 'approach7'],
        filename='temperature_response_precomputed_k.png',
        title_suffix='Approaches 0, 6, 7',
        T_range=T_range
    )


def _plot_temperature_derivative_subset(
    results: Dict[str, FitResult], output_dir: Path,
    approaches: list, filename: str, title_suffix: str = "",
    T_range: tuple = (0, 30)
) -> None:
    """Plot dh/dT = h1 + 2*h2*T for a subset of approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.linspace(T_range[0], T_range[1], 200)

    # Color scheme (degree of detrending):
    # - No detrending: black
    # - Mixed (linear T + quadratic GDP): red
    # - Linear: green
    # - Quadratic: blue
    colors = {
        'approach0': 'black',
        'approach1': 'green',      # Linear Temperature Detrending
        'approach2': 'blue',       # Quadratic GDP Growth Detrending
        'approach3': 'red',        # Combined Detrending (Mixed)
        'approach4': 'green',      # Combined Linear Detrending
        'approach5': 'blue',       # Combined Quadratic Detrending
        'approach6': 'green',      # Precomputed k Linear
        'approach7': 'blue',       # Precomputed k Quadratic
    }
    # Line style scheme (what's being detrended):
    # - No detrending or combined (both): solid
    # - GDP growth detrending only: dashed
    # - Temperature detrending only: dotted
    # - Precomputed k approaches: dash-dot
    linestyles = {
        'approach0': '-',
        'approach1': ':',
        'approach2': '--',
        'approach3': '-',
        'approach4': '-',
        'approach5': '-',
        'approach6': '-.',
        'approach7': '-.',
    }

    for name in approaches:
        if name not in results:
            continue
        r = results[name]
        dh_dT = r.h1 + 2 * r.h2 * T
        label = f"{r.approach}"
        ax.plot(T, dh_dT, color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'), label=label, linewidth=2)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('dh/dT = h₁ + 2h₂T', fontsize=12)
    title = 'Temperature Derivative by Approach'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(T_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_temperature_derivative(
    results: Dict[str, FitResult], output_dir: Path, T_range: tuple = (0, 30)
) -> None:
    """Plot dh/dT for approaches, generating two separate plots."""
    # Plot 1: Approaches 0-5 (all original approaches)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3', 'approach4', 'approach5'],
        filename='temperature_derivative_all.png',
        title_suffix='Approaches 0-5',
        T_range=T_range
    )
    # Plot 2: Approaches 0, 6, 7 (precomputed k approaches)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach0', 'approach6', 'approach7'],
        filename='temperature_derivative_precomputed_k.png',
        title_suffix='Approaches 0, 6, 7',
        T_range=T_range
    )


def plot_coefficient_comparison(results: Dict[str, FitResult], output_dir: Path) -> None:
    """Plot T_opt and h2 coefficients for each approach."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    approaches = list(results.keys())
    labels = [results[a].approach for a in approaches]
    x = np.arange(len(approaches))

    # T_optimal values
    T_opt_vals = [results[a].T_optimal for a in approaches]

    axes[0].bar(x, T_opt_vals, color='steelblue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Optimal Temperature (°C)')
    axes[0].set_title('Optimal Temperature (T_opt)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, val in enumerate(T_opt_vals):
        axes[0].text(i, val + 0.3, f'{val:.1f}°C', ha='center', va='bottom', fontsize=9)

    # h2 coefficients
    h2_vals = [results[a].h2 for a in approaches]
    h2_errs = [results[a].h2_se * 1.96 for a in approaches]  # 95% CI

    axes[1].bar(x, h2_vals, yerr=h2_errs, capsize=5, color='coral', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('h₂ coefficient')
    axes[1].set_title('Quadratic Temperature Coefficient (h₂)')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_comparison.png', dpi=150)
    plt.close()


def plot_optimal_temperature_comparison(
    results: Dict[str, FitResult], output_dir: Path
) -> None:
    """Plot optimal temperature comparison across approaches."""
    fig, ax = plt.subplots(figsize=(8, 5))

    approaches = list(results.keys())
    labels = [results[a].approach for a in approaches]
    T_opt = [results[a].T_optimal for a in approaches]

    # Colors mapping for approaches
    color_map = {
        'approach0': 'black',
        'approach1': 'green',
        'approach2': 'blue',
        'approach3': 'red',
        'approach4': 'green',
        'approach5': 'blue',
        'approach6': 'green',
        'approach7': 'blue',
    }
    colors = [color_map.get(a, 'gray') for a in approaches]
    x = np.arange(len(approaches))

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
    plt.savefig(output_dir / 'optimal_temperature_comparison.png', dpi=150)
    plt.close()


def plot_year_effects(
    results: Dict[str, FitResult], data: AnalysisData, output_dir: Path
) -> None:
    """Plot year fixed effects k(t) for all approaches.

    All approaches now use year fixed effects k_t.

    For Approach 0 (no detrending), we subtract a least-squares best-fit quadratic
    from k_t. This shows what the year effects would look like if the quadratic
    trend were absorbed into the country-specific j_i(t) terms. The subtracted
    quadratic is what would be added to all j_i(t) under an alternative
    identifiability constraint.
    """
    # Get unique years from data
    unique_years = sorted(set(data.year))
    years_array = np.array(unique_years)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color scheme (same as other plots, except approach6 is black here)
    colors = {
        'approach0': 'black',
        'approach1': 'green',
        'approach2': 'blue',
        'approach3': 'red',
        'approach4': 'green',
        'approach5': 'blue',
        'approach6': 'black',  # Black for precomputed k on this plot
    }
    linestyles = {
        'approach0': '-',
        'approach1': ':',
        'approach2': '--',
        'approach3': '-',
        'approach4': '-',
        'approach5': '-',
        'approach6': '-.',
    }

    for name, r in results.items():
        # Skip approach7 - it has the same k values as approach6 (both are precomputed year means)
        if name == 'approach7':
            continue

        # k is stored with actual year as key
        k_values = np.array([r.k[yr] for yr in unique_years])

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
            label = "No Detrending (minus best-fit quadratic)"
        elif name == 'approach6':
            k_values_plot = k_values
            label = "Precomputed k (year means)"
        else:
            k_values_plot = k_values
            label = f"{r.approach}"

        ax.plot(unique_years, k_values_plot, color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'), linewidth=1.5,
                label=label)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('k(t) - Year Fixed Effect', fontsize=12)
    ax.set_title('Year Fixed Effects by Approach', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'year_effects.png', dpi=150)
    plt.close()


def plot_residual_diagnostics(
    results: Dict[str, FitResult], data: AnalysisData, output_dir: Path
) -> None:
    """Plot residual diagnostics for each approach."""
    for name, r in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        residuals = r.residuals

        # Histogram of residuals
        axes[0, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue')
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

        fig.suptitle(f'Residual Diagnostics: {r.approach}', fontsize=14)
        plt.tight_layout()

        # Safe filename
        safe_name = name.replace(' ', '_').lower()
        plt.savefig(output_dir / f'residuals_{safe_name}.png', dpi=150)
        plt.close()


def save_all_outputs(
    data: AnalysisData,
    trends: CountryTrends,
    results: Dict[str, FitResult],
    output_dir: Path = None,
) -> Path:
    """Save all outputs to the specified directory.

    Args:
        data: AnalysisData object
        trends: CountryTrends object
        results: Dictionary of FitResult objects
        output_dir: Output directory (created if None)

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = create_output_dir()

    print(f"Saving outputs to: {output_dir}")

    # Save tables
    save_summary_table(results, output_dir)
    save_country_trends(data, trends, output_dir)

    # Generate plots
    plot_temperature_response(results, output_dir)
    plot_temperature_derivative(results, output_dir)
    plot_coefficient_comparison(results, output_dir)
    plot_optimal_temperature_comparison(results, output_dir)
    plot_year_effects(results, data, output_dir)
    plot_residual_diagnostics(results, data, output_dir)

    print("All outputs saved.")
    return output_dir


def save_bootstrap_coefficients_csv(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path
) -> None:
    """Save bootstrap samples to CSV for each approach.

    Creates: bootstrap_coefficients.csv with columns:
    - iteration
    - approach
    - h1, h2, T_optimal, r_squared, total_r_squared
    """
    rows = []
    for name, result in results.items():
        for i in range(result.n_bootstrap):
            rows.append({
                'iteration': i,
                'approach': name,
                'approach_name': result.approach,
                'h1': result.h1_samples[i],
                'h2': result.h2_samples[i],
                'T_optimal': result.T_optimal_samples[i],
                'r_squared': result.r_squared_samples[i],
                'total_r_squared': result.total_r_squared_samples[i],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'bootstrap_coefficients.csv', index=False)
    print(f"  Saved bootstrap_coefficients.csv ({len(df)} rows)")


def save_bootstrap_summary_txt(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path
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

            f.write("\n")

    print(f"  Saved bootstrap_summary.txt")
