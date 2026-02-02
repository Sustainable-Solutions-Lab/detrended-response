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
        row = {
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
        }
        # Add beta for Approach 8
        if hasattr(r, 'beta'):
            row['beta'] = r.beta
            row['beta_SE'] = r.beta_se
            row['Y_ref'] = r.Y_ref
        rows.append(row)

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
            # Add beta for Approach 8
            if hasattr(r, 'beta'):
                f.write(f"  beta = {r.beta:10.4f}  (SE: {r.beta_se:.4f})\n")
                f.write(f"  Y_ref = {r.Y_ref:.2f}\n")
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
        'approach8': 'purple',     # GDP-dependent Response
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
        'approach8': '-.',
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
    # Plot 2: Approaches 0, 6, 7, 8 (precomputed k approaches)
    _plot_temperature_response_subset(
        results, output_dir,
        approaches=['approach0', 'approach6', 'approach7', 'approach8'],
        filename='temperature_response_precomputed_k.png',
        title_suffix='Approaches 0, 6, 7, 8',
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
        'approach8': 'purple',     # GDP-dependent Response
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
        'approach8': '-.',
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
    # Plot 2: Approaches 0, 6, 7, 8 (precomputed k approaches)
    _plot_temperature_derivative_subset(
        results, output_dir,
        approaches=['approach0', 'approach6', 'approach7', 'approach8'],
        filename='temperature_derivative_precomputed_k.png',
        title_suffix='Approaches 0, 6, 7, 8',
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
        'approach8': 'purple',
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
        'approach8': 'purple',
    }
    linestyles = {
        'approach0': '-',
        'approach1': ':',
        'approach2': '--',
        'approach3': '-',
        'approach4': '-',
        'approach5': '-',
        'approach6': '-.',
        'approach8': '-.',
    }

    for name, r in results.items():
        # Skip approach7 and approach8 - they use the same k values as approach6 (precomputed year means)
        if name in ('approach7', 'approach8'):
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


def plot_gdp_scaling_factor(
    results: Dict[str, FitResult],
    output_dir: Path,
    data: AnalysisData = None,
    Y_range: tuple = None,
) -> None:
    """Plot the GDP scaling factor (Y/Y_ref)^(-beta) for Approach 8.

    This shows how the temperature response is scaled by per capita GDP level.
    Countries with lower GDP have larger scaling factors (more affected).

    Args:
        results: Dictionary of FitResult objects (must include 'approach8')
        output_dir: Output directory
        data: AnalysisData for adding GDP histogram (optional)
        Y_range: GDP range for x-axis (default: from data min to max)
    """
    if 'approach8' not in results:
        return

    r = results['approach8']
    if not hasattr(r, 'beta') or not hasattr(r, 'Y_ref'):
        return

    beta = r.beta
    Y_ref = r.Y_ref

    # Default Y range: from ~500 to ~100000 (covers most country GDPs)
    if Y_range is None:
        Y_range = (500, 100000)

    # Create GDP array (log-spaced for better visualization)
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    # Compute scaling factor
    g = (Y / Y_ref) ** (-beta)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Add GDP histogram on secondary y-axis (if data provided)
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        gdp_recent = data.pcGDP[mask_recent]

        ax2 = ax.twinx()
        # Create histogram with log-spaced bins
        bins = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 30)
        ax2.hist(gdp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax2.set_ylabel(f'Data density ({max_year})', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
        ax2.set_ylim(bottom=0)
        # Ensure histogram is behind main plot
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.plot(Y, g, 'purple', linewidth=2, label=f'β = {beta:.3f}')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='g = 1 (at Y = Y_ref)')
    ax.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref = ${Y_ref:,.0f}')

    ax.set_xscale('log')
    ax.set_xlabel('Per Capita GDP ($)', fontsize=12)
    ax.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-β)', fontsize=12)
    ax.set_title('GDP-Dependent Temperature Response Scaling (Approach 8)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotations for interpretation
    ax.annotate('Poorer countries:\nmore affected',
                xy=(Y_range[0] * 2, g[0] * 0.9),
                fontsize=10, color='darkred')
    ax.annotate('Richer countries:\nless affected',
                xy=(Y_range[1] * 0.3, g[-1] * 1.1),
                fontsize=10, color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_dir / 'gdp_scaling_factor.png', dpi=150)
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

    # Plot GDP scaling factor for Approach 8
    if 'approach8' in results:
        plot_gdp_scaling_factor(results, output_dir, data=data)

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


def compute_h_response_uncertainty_bands(
    result: "BootstrapResult",
    T_range: np.ndarray,
    percentiles: tuple = (5, 50, 95)
) -> tuple:
    """Compute h(T) - h(T*) uncertainty bands from bootstrap samples.

    For each bootstrap sample, computes h(T) - h(T*) over the temperature range.
    Returns percentile bands across all bootstrap samples.

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)

    Returns:
        Tuple of arrays (h_lower, h_median, h_upper) each with shape (len(T_range),)
    """
    # Get valid bootstrap samples (exclude NaN)
    valid_mask = ~np.isnan(result.h1_samples) & ~np.isnan(result.h2_samples)
    h1_valid = result.h1_samples[valid_mask]
    h2_valid = result.h2_samples[valid_mask]

    if len(h1_valid) == 0:
        return (np.full_like(T_range, np.nan),
                np.full_like(T_range, np.nan),
                np.full_like(T_range, np.nan))

    # Compute h(T) - h(T*) for each bootstrap sample
    # h(T) = h1*T + h2*T^2
    # h(T*) = -h1^2 / (4*h2) when T* = -h1/(2*h2)
    n_samples = len(h1_valid)
    n_T = len(T_range)
    h_relative_samples = np.zeros((n_samples, n_T))

    for i in range(n_samples):
        h1 = h1_valid[i]
        h2 = h2_valid[i]
        h_T = h1 * T_range + h2 * T_range ** 2
        if h2 != 0:
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    params = [
        ('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁ (Linear Coefficient)'),
        ('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (Quadratic Coefficient)'),
        ('T_optimal', result.T_optimal_samples, result.T_optimal_point, stats['T_optimal'], 'T_optimal (°C)'),
    ]

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
    plt.savefig(output_dir / f'bootstrap_distributions_{approach_key}.png', dpi=150)
    plt.close()


def plot_all_bootstrap_distributions(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    filename: str = "bootstrap_distributions.pdf"
) -> None:
    """Plot h1, h2, T_optimal distributions for all approaches in a single PDF.

    Creates a multi-panel figure with one row per approach and 3 columns
    (h1, h2, T_optimal).

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save the plot
        filename: Output filename (should end in .pdf)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    approach_names = list(results.keys())
    n_approaches = len(approach_names)

    with PdfPages(output_dir / filename) as pdf:
        # Create a figure with all approaches - one row per approach, 3 columns
        fig, axes = plt.subplots(n_approaches, 3, figsize=(14, 4 * n_approaches))

        if n_approaches == 1:
            axes = axes.reshape(1, -1)

        # First pass: determine x-axis ranges for each column
        col_ranges = {0: [], 1: [], 2: []}  # h1, h2, T_optimal
        for name in approach_names:
            result = results[name]
            for col_idx, samples in enumerate([result.h1_samples, result.h2_samples, result.T_optimal_samples]):
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
                ('h1', result.h1_samples, result.h1_point, stats['h1'], 'h₁ (Linear Coefficient)'),
                ('h2', result.h2_samples, result.h2_point, stats['h2'], 'h₂ (Quadratic Coefficient)'),
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
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close()


def plot_bootstrap_temperature_response(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    approaches: list = None,
    filename: str = "bootstrap_temperature_response.pdf",
    T_range: tuple = (0, 30),
    data: AnalysisData = None,
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

    # Color scheme (same as existing plots)
    colors = {
        'approach0': 'black',
        'approach1': 'green',
        'approach2': 'blue',
        'approach3': 'red',
        'approach4': 'green',
        'approach5': 'blue',
        'approach6': 'green',
        'approach7': 'blue',
        'approach8': 'purple',
    }

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for name in approaches:
        result = results[name]

        # Compute uncertainty bands
        h_lower, h_median, h_upper = compute_h_response_uncertainty_bands(result, T)

        # Compute point estimate response
        h1_point = result.h1_point
        h2_point = result.h2_point
        h_T_point = h1_point * T + h2_point * T ** 2
        if h2_point != 0:
            h_T_opt_point = -h1_point ** 2 / (4 * h2_point)
        else:
            h_T_opt_point = 0
        h_point = h_T_point - h_T_opt_point

        plot_data[name] = {
            'h_lower': h_lower,
            'h_upper': h_upper,
            'h_point': h_point,
        }

        # Update global y range
        y_min = min(y_min, np.nanmin(h_lower), np.nanmin(h_point))
        y_max = max(y_max, np.nanmax(h_upper), np.nanmax(h_point))

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
    else:
        n_rows, n_cols = 3, 3

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
        color = colors.get(name, 'steelblue')
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

        # Plot CI band
        ax.fill_between(T, pdata['h_lower'], pdata['h_upper'], alpha=0.3, color=color, label='90% CI')

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

    fig.suptitle('Temperature Response with Bootstrap 90% CI', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bootstrap_T_optimal_comparison(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Horizontal error bar plot: point estimate + 90% CI + IQR for each approach.

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color scheme
    colors = {
        'approach0': 'black',
        'approach1': 'green',
        'approach2': 'blue',
        'approach3': 'red',
        'approach4': 'green',
        'approach5': 'blue',
        'approach6': 'green',
        'approach7': 'blue',
        'approach8': 'purple',
    }

    approach_names = list(results.keys())
    n_approaches = len(approach_names)
    y_positions = np.arange(n_approaches)

    for i, name in enumerate(approach_names):
        result = results[name]
        stats = all_stats[name]['T_optimal']
        color = colors.get(name, 'gray')

        point_est = result.T_optimal_point
        p5, p25, p50, p75, p95 = stats['p5'], stats['p25'], stats['p50'], stats['p75'], stats['p95']

        # Plot 90% CI as error bar
        ci_lower = point_est - p5
        ci_upper = p95 - point_est
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
    plt.savefig(output_dir / 'bootstrap_T_optimal_comparison.png', dpi=150)
    plt.close()


def compute_derivative_uncertainty_bands(
    result: "BootstrapResult",
    T_range: np.ndarray,
    percentiles: tuple = (5, 50, 95)
) -> tuple:
    """Compute dh/dT = h1 + 2*h2*T uncertainty bands from bootstrap samples.

    Args:
        result: BootstrapResult containing h1_samples and h2_samples
        T_range: Array of temperature values
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)

    Returns:
        Tuple of arrays (dh_lower, dh_median, dh_upper) each with shape (len(T_range),)
    """
    # Get valid bootstrap samples (exclude NaN)
    valid_mask = ~np.isnan(result.h1_samples) & ~np.isnan(result.h2_samples)
    h1_valid = result.h1_samples[valid_mask]
    h2_valid = result.h2_samples[valid_mask]

    if len(h1_valid) == 0:
        return (np.full_like(T_range, np.nan),
                np.full_like(T_range, np.nan),
                np.full_like(T_range, np.nan))

    # Compute dh/dT for each bootstrap sample
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
    T_range: tuple = (0, 30)
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
    """
    T = np.linspace(T_range[0], T_range[1], 200)

    if approaches is None:
        approaches = list(results.keys())

    # Filter to only approaches that exist in results
    approaches = [name for name in approaches if name in results]
    n_approaches = len(approaches)

    if n_approaches == 0:
        return

    # Color scheme (same as existing plots)
    colors = {
        'approach0': 'black',
        'approach1': 'green',
        'approach2': 'blue',
        'approach3': 'red',
        'approach4': 'green',
        'approach5': 'blue',
        'approach6': 'green',
        'approach7': 'blue',
        'approach8': 'purple',
    }

    # First pass: compute all data and find global y-axis range
    plot_data = {}
    y_min, y_max = np.inf, -np.inf

    for name in approaches:
        result = results[name]

        # Compute uncertainty bands
        dh_lower, dh_median, dh_upper = compute_derivative_uncertainty_bands(result, T)

        # Compute point estimate derivative
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
    else:
        n_rows, n_cols = 3, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_approaches == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Second pass: create the plots
    for idx, name in enumerate(approaches):
        ax = axes[idx]
        result = results[name]
        color = colors.get(name, 'steelblue')
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
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bootstrap_gdp_scaling(
    results: Dict[str, "BootstrapResult"],
    output_dir: Path,
    Y_ref: float,
    Y_range: tuple = None,
    filename: str = 'bootstrap_gdp_scaling.png',
    data: AnalysisData = None,
) -> None:
    """Plot GDP scaling factor with bootstrap uncertainty bands for Approach 8.

    Shows the spread of (Y/Y_ref)^(-beta) curves across bootstrap samples.

    Args:
        results: Dict of BootstrapResult (must include 'approach8')
        output_dir: Directory to save the plot
        Y_ref: Reference GDP value (same as used in fitting)
        Y_range: GDP range for x-axis (default: 500 to 100000)
        filename: Output filename
        data: AnalysisData for adding GDP histogram (optional)
    """
    if 'approach8' not in results:
        return

    result = results['approach8']
    if result.beta_point is None or result.beta_samples is None:
        return

    beta_point = result.beta_point
    beta_samples = result.beta_samples

    # Filter out NaN values
    valid_betas = beta_samples[~np.isnan(beta_samples)]
    if len(valid_betas) == 0:
        return

    # Default Y range
    if Y_range is None:
        Y_range = (500, 100000)

    # Create GDP array (log-spaced)
    Y = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual bootstrap samples (thin gray lines)
    n_samples_to_plot = min(100, len(valid_betas))  # Limit for clarity
    sample_indices = np.linspace(0, len(valid_betas) - 1, n_samples_to_plot, dtype=int)

    for idx in sample_indices:
        beta_b = valid_betas[idx]
        g_b = (Y / Y_ref) ** (-beta_b)
        ax.plot(Y, g_b, color='purple', alpha=0.05, linewidth=0.5)

    # Compute percentile bands
    g_samples = np.zeros((len(valid_betas), len(Y)))
    for i, beta_b in enumerate(valid_betas):
        g_samples[i, :] = (Y / Y_ref) ** (-beta_b)

    g_p5 = np.percentile(g_samples, 5, axis=0)
    g_p25 = np.percentile(g_samples, 25, axis=0)
    g_p50 = np.percentile(g_samples, 50, axis=0)
    g_p75 = np.percentile(g_samples, 75, axis=0)
    g_p95 = np.percentile(g_samples, 95, axis=0)

    # Add GDP histogram on secondary y-axis (if data provided)
    if data is not None:
        max_year = data.year_range[1]
        mask_recent = data.year == max_year
        gdp_recent = data.pcGDP[mask_recent]

        ax2 = ax.twinx()
        # Create histogram with log-spaced bins
        bins = np.logspace(np.log10(Y_range[0]), np.log10(Y_range[1]), 30)
        ax2.hist(gdp_recent, bins=bins, color='gray', alpha=0.3, density=True)
        ax2.set_ylabel(f'Data density ({max_year})', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
        ax2.set_ylim(bottom=0)
        # Ensure histogram is behind main plot
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    # Plot uncertainty bands
    ax.fill_between(Y, g_p5, g_p95, color='purple', alpha=0.2, label='90% CI')
    ax.fill_between(Y, g_p25, g_p75, color='purple', alpha=0.3, label='IQR')

    # Plot point estimate
    g_point = (Y / Y_ref) ** (-beta_point)
    ax.plot(Y, g_point, 'purple', linewidth=2.5, label=f'Point estimate (β = {beta_point:.3f})')

    # Reference lines
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(Y_ref, color='gray', linestyle=':', alpha=0.5, label=f'Y_ref ≈ ${Y_ref:,.0f}')

    ax.set_xscale('log')
    ax.set_xlabel('Per Capita GDP ($)', fontsize=12)
    ax.set_ylabel('GDP Scaling Factor g = (Y/Y_ref)^(-β)', fontsize=12)
    ax.set_title('GDP-Dependent Temperature Response Scaling with Bootstrap Uncertainty', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add beta distribution inset
    ax_inset = ax.inset_axes([0.02, 0.55, 0.25, 0.35])
    ax_inset.hist(valid_betas, bins=30, color='purple', alpha=0.7, density=True)
    ax_inset.axvline(beta_point, color='red', linewidth=1.5, label='Point est.')
    ax_inset.set_xlabel('β', fontsize=9)
    ax_inset.set_ylabel('Density', fontsize=9)
    ax_inset.set_title('Bootstrap β distribution', fontsize=9)
    ax_inset.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def save_all_bootstrap_plots(
    results: Dict[str, "BootstrapResult"],
    all_stats: Dict[str, Dict],
    output_dir: Path,
    T_range: tuple = (0, 30),
    Y_ref: float = None,
    data: AnalysisData = None,
) -> None:
    """Generate all bootstrap plots.

    Calls:
    - plot_all_bootstrap_distributions() for all approaches in single PDF
    - plot_bootstrap_temperature_response() for approaches 0-5 and 0,6,7
    - plot_bootstrap_temperature_derivative() for approaches 0-5 and 0,6,7
    - plot_bootstrap_T_optimal_comparison() for all approaches
    - plot_bootstrap_gdp_scaling() for Approach 8 (if Y_ref provided)

    Args:
        results: Dict of BootstrapResult for each approach
        all_stats: Dict mapping approach key to statistics dict
        output_dir: Directory to save plots
        T_range: Temperature range for response plots
        Y_ref: Reference GDP for Approach 8 GDP scaling plot
        data: AnalysisData for adding data density histograms (optional)
    """
    # Generate combined distribution plot for all approaches
    plot_all_bootstrap_distributions(results, all_stats, output_dir)
    print("      Saved bootstrap_distributions.pdf")

    # Temperature response plot - all 9 approaches in one PDF
    plot_bootstrap_temperature_response(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3',
                    'approach4', 'approach5', 'approach6', 'approach7', 'approach8'],
        filename='bootstrap_temperature_response.pdf',
        T_range=T_range,
        data=data
    )
    print("      Saved bootstrap_temperature_response.pdf")

    # Temperature derivative plot - all 9 approaches in one PDF
    plot_bootstrap_temperature_derivative(
        results, output_dir,
        approaches=['approach0', 'approach1', 'approach2', 'approach3',
                    'approach4', 'approach5', 'approach6', 'approach7', 'approach8'],
        filename='bootstrap_temperature_derivative.pdf',
        T_range=T_range
    )
    print("      Saved bootstrap_temperature_derivative.pdf")

    # T_optimal comparison across all approaches
    plot_bootstrap_T_optimal_comparison(results, all_stats, output_dir)
    print("      Saved bootstrap_T_optimal_comparison.png")

    # GDP scaling factor with bootstrap uncertainty (Approach 8)
    if 'approach8' in results and Y_ref is not None:
        plot_bootstrap_gdp_scaling(results, output_dir, Y_ref, data=data)
        print("      Saved bootstrap_gdp_scaling.png")
