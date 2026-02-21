#!/usr/bin/env python3
"""Find LOESS window that makes method2h0 Total R² equal to method1h0 Total R².

This script uses Brent's method to find the LOESS smoothing window width
where the null models (no climate response) have equal explanatory power.

Usage:
    python scripts/find_matching_loess_window.py [--tol TOL]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
)
from src.fitting import fit_method1h0_precomputed_k, fit_method2h0_precomputed_k_loess


def main():
    parser = argparse.ArgumentParser(
        description="Find LOESS window matching method1h0 and method2h0 Total R²"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Convergence tolerance (default: 1e-12)",
    )
    parser.add_argument(
        "--use-csv",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Path to input CSV file",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Finding LOESS Window for Matching Total R²")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_data_from_csv(args.use_csv)
    print(f"  Countries: {data.n_countries}")
    print(f"  Observations: {data.n_obs}")
    print(f"  Year range: {data.year_range[0]} - {data.year_range[1]}")

    # Compute trends for method1h0 (independent of LOESS window)
    print("\nComputing method1h0 (quadratic trends)...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)

    # Fit method1h0
    result1h0 = fit_method1h0_precomputed_k(data, trends_with_k, year_means)
    target_r2 = result1h0.total_r_squared
    print(f"  method1h0 Total R² = {target_r2:.15f}")

    # Define objective function: difference between method2h0 and method1h0 Total R²
    def objective(window):
        trends_loess = compute_country_trends_loess(data, year_means, window)
        result2h0 = fit_method2h0_precomputed_k_loess(data, trends_loess, year_means)
        return result2h0.total_r_squared - target_r2

    # Find bracket - LOESS with very small window overfits (higher R²),
    # LOESS with very large window underfits (lower R²)
    print("\nFinding bracket...")
    window_low, window_high = 5.0, 100.0

    r2_low = objective(window_low)
    r2_high = objective(window_high)
    print(f"  Window {window_low:.1f}: diff = {r2_low:+.10f}")
    print(f"  Window {window_high:.1f}: diff = {r2_high:+.10f}")

    if r2_low * r2_high > 0:
        print("\nERROR: No sign change in bracket. Adjusting...")
        # Try wider range
        for w in [2.0, 3.0, 150.0, 200.0]:
            r2_w = objective(w)
            print(f"  Window {w:.1f}: diff = {r2_w:+.10f}")

    # Use Brent's method to find root
    print(f"\nFinding root with tolerance {args.tol}...")
    optimal_window = brentq(objective, window_low, window_high, xtol=args.tol)

    # Verify result
    trends_loess = compute_country_trends_loess(data, year_means, optimal_window)
    result2h0 = fit_method2h0_precomputed_k_loess(data, trends_loess, year_means)
    final_diff = result2h0.total_r_squared - target_r2

    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    print(f"\n  Optimal LOESS window: {optimal_window:.15f} years")
    print(f"\n  method1h0 Total R²:   {target_r2:.15f}")
    print(f"  method2h0 Total R²:   {result2h0.total_r_squared:.15f}")
    print(f"  Difference:           {final_diff:+.2e}")

    # Also compute mean weight distance (inverse of the 44/7 factor)
    mean_weight_distance = optimal_window * 7 / 44
    print(f"\n  Mean weight distance: {mean_weight_distance:.15f} years")
    print("=" * 70)


if __name__ == "__main__":
    main()
