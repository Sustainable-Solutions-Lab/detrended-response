#!/usr/bin/env python3
"""Run bootstrap analysis for uncertainty quantification.

This script performs country-level cluster bootstrap resampling to compute
confidence intervals for h1, h2, and T_optimal across all 8 approaches.

Usage:
    python scripts/run_bootstrap.py [--n-bootstrap N] [--random-seed SEED] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data, load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
)
from src.fitting import fit_all_approaches
from src.bootstrap import run_bootstrap, compute_bootstrap_statistics
from src.output import (
    create_output_dir,
    save_bootstrap_coefficients_csv,
    save_bootstrap_k_samples_csv,
    save_bootstrap_var_attrib_csv,
    save_bootstrap_country_samples_csv,
    save_bootstrap_summary_txt,
    save_bootstrap_summary_table,
    save_all_bootstrap_plots,
)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap uncertainty analysis for detrended response"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use-csv",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Use pre-processed CSV file (default: data/input/Maddison_CRU_dataset.csv). "
             "Set to empty string to use Maddison/CRU instead.",
    )
    parser.add_argument(
        "--maddison",
        default="data/input/mpd2023_web.xlsx",
        help="Path to Maddison GDP Excel file (ignored if --use-csv is set)",
    )
    parser.add_argument(
        "--cru",
        default="data/input/cru_climate_data.csv",
        help="Path to CRU temperature CSV file (ignored if --use-csv is set)",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Minimum year to include (default: 1960 for Maddison/CRU, all years for CSV)",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Maximum year to include (default: 2022 for Maddison/CRU, all years for CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--loess-window",
        type=int,
        default=25,
        help="Window size in years for LOESS smoothing (default: 25)",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 70)
    print("Bootstrap Uncertainty Analysis")
    print("=" * 70)

    # Load data
    input_file = None  # Track which input file was used
    if args.use_csv and args.use_csv.strip():
        csv_path = Path(args.use_csv).expanduser()
        input_file = str(csv_path)
        print(f"\n[1/7] Loading data from {csv_path}...")
        data = load_data_from_csv(
            str(csv_path),
            year_min=args.year_min,
            year_max=args.year_max
        )
    else:
        year_min = args.year_min if args.year_min is not None else 1960
        year_max = args.year_max if args.year_max is not None else 2022
        input_file = f"{args.maddison} + {args.cru}"
        print(f"\n[1/7] Loading data from {args.maddison} and {args.cru}...")
        print(f"      Year range: {year_min} - {year_max}")
        data = load_data(
            args.maddison, args.cru,
            year_min=year_min, year_max=year_max
        )

    print(f"      Observations: {data.n_obs}")
    print(f"      Countries: {data.n_countries}")
    print(f"      Years: {data.n_years}")

    # Compute country-level trends
    print("\n[2/7] Computing country-level trends...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    # Compute LOESS trends for approaches 9 and 10
    trends_loess = compute_country_trends_loess(data, year_means, args.loess_window)
    # Compute Y_ref based on most recent year (used for all bootstrap iterations)
    max_year = data.year_range[1]
    mask_recent = data.year == max_year
    Y_ref = np.mean(data.pcGDP[mask_recent])
    print(f"      Y_ref (mean pcGDP in {max_year}): {Y_ref:.2f}")
    print(f"      LOESS window: {args.loess_window} years")
    print("      Done.")

    # Fit original model (point estimates)
    print("\n[3/7] Fitting original model (point estimates)...")
    original_results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        Y_ref=Y_ref,
        trends_loess=trends_loess
    )
    print("      Done.")

    # Run bootstrap
    print(f"\n[4/7] Running bootstrap ({args.n_bootstrap} iterations, seed={args.random_seed})...")
    bootstrap_results, country_samples = run_bootstrap(
        data=data,
        trends=trends,
        original_results=original_results,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        verbose=verbose,
        Y_ref=Y_ref,
        loess_window=args.loess_window,
    )
    print("      Done.")

    # Compute summary statistics
    print("\n[5/7] Computing bootstrap statistics...")
    all_stats = {}
    for name, result in bootstrap_results.items():
        all_stats[name] = compute_bootstrap_statistics(result)
    print("      Done.")

    # Save outputs
    print("\n[6/7] Saving outputs...")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="bootstrap_")

    save_bootstrap_coefficients_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_k_samples_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_var_attrib_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_country_samples_csv(country_samples, data, output_dir, input_file=input_file)
    save_bootstrap_summary_txt(bootstrap_results, all_stats, output_dir, input_file=input_file)
    save_bootstrap_summary_table(bootstrap_results, all_stats, output_dir, input_file=input_file)

    # Generate bootstrap plots
    print("\n[7/7] Generating bootstrap plots...")
    save_all_bootstrap_plots(bootstrap_results, all_stats, output_dir, Y_ref=Y_ref, data=data, input_file=input_file)
    print("      Done.")

    print(f"      Output saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("Bootstrap Summary")
    print("=" * 70)

    for name, result in bootstrap_results.items():
        stats = all_stats[name]
        print(f"\n{result.approach}")
        print("-" * 50)
        print(f"  T_optimal: {result.T_optimal_point:.2f} C")
        print(f"    90% CI: [{stats['T_optimal']['p5']:.2f}, {stats['T_optimal']['p95']:.2f}]")
        print(f"    IQR:    [{stats['T_optimal']['p25']:.2f}, {stats['T_optimal']['p75']:.2f}]")
        print(f"  h1: {result.h1_point:.6f}")
        print(f"    90% CI: [{stats['h1']['p5']:.6f}, {stats['h1']['p95']:.6f}]")
        print(f"  h2: {result.h2_point:.6f}")
        print(f"    90% CI: [{stats['h2']['p5']:.6f}, {stats['h2']['p95']:.6f}]")
        # Print beta for Approach 7
        if result.beta_point is not None and 'beta' in stats:
            print(f"  beta: {result.beta_point:.4f}")
            print(f"    90% CI: [{stats['beta']['p5']:.4f}, {stats['beta']['p95']:.4f}]")
        # Print h2_low and h2_high for Approach 8 (piecewise quadratic)
        if result.h2_low_point is not None and 'h2_low' in stats:
            print(f"  h2_low: {result.h2_low_point:.6f}")
            print(f"    90% CI: [{stats['h2_low']['p5']:.6f}, {stats['h2_low']['p95']:.6f}]")
        if result.h2_high_point is not None and 'h2_high' in stats:
            print(f"  h2_high: {result.h2_high_point:.6f}")
            print(f"    90% CI: [{stats['h2_high']['p5']:.6f}, {stats['h2_high']['p95']:.6f}]")
        print(f"  Successful iterations: {result.n_successful}/{result.n_bootstrap}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
