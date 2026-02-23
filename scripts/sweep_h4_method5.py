#!/usr/bin/env python3
"""Sweep h4 (persistence decay) parameter for Approach3L.

This script evaluates Approach3L at fixed h4 values across a range,
computing and reporting metrics at each value.

- h4 = 0: Full persistence (accumulated temperature effects persist)
- h4 = 1: No persistence (first-difference behavior)

Usage:
    python scripts/sweep_h4_method5.py
    python scripts/sweep_h4_method5.py --mean-weight-distance 10
    python scripts/sweep_h4_method5.py --h4-min 0 --h4-max 0.5 --h4-steps 51
    python scripts/sweep_h4_method5.py --output-csv data/output/h4_sweep.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data, load_data_from_csv
from src.detrending import (
    compute_year_means,
    compute_country_trends_loess,
)
from src.fitting import (
    compute_persistence_accumulators,
    compute_persistence_accumulators_at_T,
    compute_T_optimal,
)


def fit_Approach3L_at_h4(data, trends_loess, year_means, h4):
    """Fit Approach3L at a fixed h4 value and return metrics.

    Returns dict with h4, h1, h2, T_optimal, SSE, RMSE, r_squared, total_r_squared.
    """
    # Compute dependent variable: dy - k[t] - j_i[t]
    y = np.zeros(data.n_obs)
    for i in range(data.n_obs):
        yr = data.year[i]
        y[i] = data.growth_pcGDP[i] - year_means[yr] - trends_loess.y_loess[i]

    T = data.temp
    T_trend = trends_loess.T_loess

    # Compute accumulators for observed and trend temperatures
    A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
    A_T_trend_lag, A_T2_trend_lag = compute_persistence_accumulators_at_T(
        data, h4, T_trend
    )

    # Modified regressors with detrending
    X1 = (T - h4 * A_T_lag) - (T_trend - h4 * A_T_trend_lag)
    X2 = (T**2 - h4 * A_T2_lag) - (T_trend**2 - h4 * A_T2_trend_lag)
    X = np.column_stack([X1, X2])

    # Solve OLS
    beta_ols, _, _, _ = linalg.lstsq(X, y)
    h1 = beta_ols[0]
    h2 = beta_ols[1]

    y_pred = X @ beta_ols
    residuals = y - y_pred

    # Metrics
    sse = np.sum(residuals**2)
    rmse = np.sqrt(sse / data.n_obs)

    # R² on detrended y
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - sse / ss_tot

    # Total R² on original dy
    ss_tot_dy = np.sum((data.growth_pcGDP - np.mean(data.growth_pcGDP))**2)
    total_r_squared = 1 - sse / ss_tot_dy

    # Optimal temperature
    T_optimal = compute_T_optimal(h1, h2)

    return {
        'h4': h4,
        'h1': h1,
        'h2': h2,
        'T_optimal': T_optimal,
        'SSE': sse,
        'RMSE': rmse,
        'r_squared': r_squared,
        'total_r_squared': total_r_squared,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep h4 (persistence decay) parameter for Approach3L"
    )
    parser.add_argument(
        "--h4-min", type=float, default=0.0, help="Minimum h4 value"
    )
    parser.add_argument(
        "--h4-max", type=float, default=1.0, help="Maximum h4 value"
    )
    parser.add_argument(
        "--h4-steps", type=int, default=21, help="Number of h4 values to evaluate"
    )
    parser.add_argument(
        "--mean-weight-distance", type=float, default=None,
        help="Mean weighting distance in years for LOESS. Window = 44/7 * this value."
    )
    parser.add_argument(
        "--loess-window", type=int, default=25,
        help="LOESS window in years (if not using mean-weight-distance)"
    )
    parser.add_argument(
        "--use-csv", type=str, default="data/input/Maddison_CRU_dataset.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--year-min", type=int, default=None, help="Minimum year to include"
    )
    parser.add_argument(
        "--year-max", type=int, default=None, help="Maximum year to include"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None, help="Output CSV file path"
    )

    args = parser.parse_args()

    # Compute LOESS window
    if args.mean_weight_distance is not None:
        loess_window = (44 / 7) * args.mean_weight_distance
    else:
        loess_window = args.loess_window

    # Load data
    if args.use_csv and args.use_csv.strip():
        csv_path = Path(args.use_csv).expanduser()
        print(f"Loading data from {csv_path}...")
        data = load_data_from_csv(
            str(csv_path),
            year_min=args.year_min,
            year_max=args.year_max
        )
    else:
        year_min = args.year_min if args.year_min is not None else 1960
        year_max = args.year_max if args.year_max is not None else 2022
        print(f"Loading data from default sources (years {year_min}-{year_max})...")
        data = load_data(
            "data/input/mpd2023_web.xlsx",
            "data/input/cru_climate_data.csv",
            year_min=year_min,
            year_max=year_max
        )

    print(f"  Observations: {data.n_obs}")
    print(f"  Countries: {data.n_countries}")
    print(f"  Years: {data.year_range[0]} - {data.year_range[1]}")

    # Compute trends
    print(f"Computing LOESS trends (window={loess_window:.1f} years)...")
    year_means = compute_year_means(data)
    trends_loess = compute_country_trends_loess(data, year_means, loess_window)

    # Sweep h4 values
    h4_values = np.linspace(args.h4_min, args.h4_max, args.h4_steps)
    results = []

    print(f"\nSweeping h4 from {args.h4_min} to {args.h4_max} ({args.h4_steps} steps)...\n")

    for h4 in h4_values:
        result = fit_Approach3L_at_h4(data, trends_loess, year_means, h4)
        results.append(result)

    # Print results table
    header = f"{'h4':>8} {'h1':>12} {'h2':>12} {'T_optimal':>10} {'SSE':>14} {'RMSE':>10} {'r_squared':>10} {'total_R2':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['h4']:>8.4f} "
            f"{r['h1']:>12.6f} "
            f"{r['h2']:>12.6f} "
            f"{r['T_optimal']:>10.2f} "
            f"{r['SSE']:>14.4f} "
            f"{r['RMSE']:>10.6f} "
            f"{r['r_squared']:>10.4f} "
            f"{r['total_r_squared']:>10.4f}"
        )

    # Find optimal h4 (minimum SSE)
    best_idx = np.argmin([r['SSE'] for r in results])
    best = results[best_idx]
    print(f"\nMinimum SSE at h4 = {best['h4']:.4f}")
    print(f"  h1 = {best['h1']:.6f}, h2 = {best['h2']:.6f}")
    print(f"  T_optimal = {best['T_optimal']:.2f} C")
    print(f"  total_R² = {best['total_r_squared']:.4f}")

    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("h4,h1,h2,T_optimal,SSE,RMSE,r_squared,total_r_squared\n")
            for r in results:
                f.write(
                    f"{r['h4']},{r['h1']},{r['h2']},{r['T_optimal']},"
                    f"{r['SSE']},{r['RMSE']},{r['r_squared']},{r['total_r_squared']}\n"
                )
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
