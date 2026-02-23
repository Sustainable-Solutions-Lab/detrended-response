#!/usr/bin/env python3
"""Run bootstrap analysis for uncertainty quantification.

This script performs country-level cluster bootstrap resampling to compute
confidence intervals for h1, h2, and T_opt across all methods.

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
    save_bootstrap_h_values,
    save_bootstrap_h_baselines,
    save_bootstrap_summary_txt,
    save_bootstrap_summary_table,
    save_variance_decomposition_table,
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
        type=float,
        default=42.447947771790915,
        help="Window size in years for LOESS smoothing (default: 42.45)",
    )
    parser.add_argument(
        "--mean-weight-distance",
        type=float,
        default=None,
        help="Mean weighting distance in years for LOESS. Window = 44/7 * this value. "
             "If specified, adds '_mwXX' suffix to output directory.",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Compute LOESS window from mean weight distance if specified
    if args.mean_weight_distance is not None:
        loess_window = (44 / 7) * args.mean_weight_distance
        # Format suffix: integer -> mwNN, non-integer -> mwNN.NNNN
        if args.mean_weight_distance == int(args.mean_weight_distance):
            mw_suffix = f"mw{int(args.mean_weight_distance):02d}"
        else:
            mw_suffix = f"mw{args.mean_weight_distance:.4f}"
    else:
        loess_window = args.loess_window
        mw_suffix = ""

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
    # Compute LOESS trends for methods 2-4
    trends_loess = compute_country_trends_loess(data, year_means, loess_window)
    print(f"      LOESS window: {loess_window:.1f} years")
    print("      Done.")

    # Fit original model (point estimates)
    print("\n[3/7] Fitting original model (point estimates)...")
    original_results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess
    )
    print("      Done.")

    # Run bootstrap
    print(f"\n[4/7] Running bootstrap ({args.n_bootstrap} iterations, seed={args.random_seed})...")
    # Specify methods for h(T) computation
    h_T_approaches = ['Approach1J', 'Approach1P', 'Approach1L', 'Approach2L', 'Approach3L', 'Approach2J', 'Approach3J', 'Approach2P', 'Approach3P']
    bootstrap_results, country_samples, h_T_samples = run_bootstrap(
        data=data,
        trends=trends,
        original_results=original_results,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        verbose=verbose,
        loess_window=loess_window,
        h_T_approaches=h_T_approaches,
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
        output_dir = create_output_dir(prefix="bootstrap_", suffix=mw_suffix)

    save_bootstrap_coefficients_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_k_samples_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_var_attrib_csv(bootstrap_results, output_dir, input_file=input_file)
    save_bootstrap_country_samples_csv(country_samples, data, output_dir, input_file=input_file)
    if h_T_samples:
        save_bootstrap_h_values(
            h_T_samples, data, output_dir,
            input_file=input_file,
            original_results=original_results,
            trends_loess=trends_loess
        )
        save_bootstrap_h_baselines(
            bootstrap_results, data, trends_loess, output_dir,
            input_file=input_file,
            original_results=original_results
        )
    save_bootstrap_summary_txt(bootstrap_results, all_stats, output_dir, input_file=input_file)
    save_bootstrap_summary_table(bootstrap_results, all_stats, output_dir, input_file=input_file)
    save_variance_decomposition_table(bootstrap_results, output_dir, input_file=input_file)

    # Generate bootstrap plots
    print("\n[7/7] Generating bootstrap plots...")
    save_all_bootstrap_plots(bootstrap_results, all_stats, output_dir, data=data, input_file=input_file)
    print("      Done.")

    # Save run metadata for post-processing scripts
    import json
    metadata = {
        'loess_window': loess_window,
        'mean_weight_distance': args.mean_weight_distance,
        'input_file': input_file,
        'year_min': data.year_range[0],
        'year_max': data.year_range[1],
        'n_countries': data.n_countries,
        'n_obs': data.n_obs,
        'n_bootstrap': args.n_bootstrap,
        'random_seed': args.random_seed,
    }
    metadata_path = output_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved: {metadata_path}")

    print(f"      Output saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("Bootstrap Summary")
    print("=" * 70)

    for name, result in bootstrap_results.items():
        stats = all_stats[name]
        print(f"\n{result.approach}")
        print("-" * 50)

        # Approach2L/Approach2J/Approach2P: h2 (below T_opt), h4 (above T_opt), T_opt (piecewise quadratic)
        if name in ['Approach2L', 'Approach2J', 'Approach2P'] and result.h4_point is not None:
            print(f"  h2 (below T_opt): {result.h2_point:.6f}")
            print(f"    90% CI: [{stats['h2']['p5']:.6f}, {stats['h2']['p95']:.6f}]")
            print(f"  h4 (above T_opt): {result.h4_point:.6f}")
            print(f"    90% CI: [{stats['h4']['p5']:.6f}, {stats['h4']['p95']:.6f}]")
            print(f"  T_opt: {result.T_opt_point:.2f} C")
            print(f"    90% CI: [{stats['T_opt']['p5']:.2f}, {stats['T_opt']['p95']:.2f}]")

        # Approach3L/Approach3J/Approach3P: h1, h2, h4 (persistence decay), T_opt
        elif name in ['Approach3L', 'Approach3J', 'Approach3P'] and result.h4_point is not None:
            print(f"  h1: {result.h1_point:.6f}")
            print(f"    90% CI: [{stats['h1']['p5']:.6f}, {stats['h1']['p95']:.6f}]")
            print(f"  h2: {result.h2_point:.6f}")
            print(f"    90% CI: [{stats['h2']['p5']:.6f}, {stats['h2']['p95']:.6f}]")
            print(f"  h4 (persistence decay): {result.h4_point:.6f}")
            print(f"    90% CI: [{stats['h4']['p5']:.6f}, {stats['h4']['p95']:.6f}]")
            if result.T_opt_point is not None and not np.isnan(result.T_opt_point):
                print(f"  T_opt: {result.T_opt_point:.2f} C")
                print(f"    90% CI: [{stats['T_opt']['p5']:.2f}, {stats['T_opt']['p95']:.2f}]")
            else:
                print(f"  T_opt: N/A")

            # Add filtered statistics for Approach3L/Approach3J/Approach3P
            from src.bootstrap import compute_Approach3L_filtered_statistics
            filtered_stats = compute_Approach3L_filtered_statistics(result)
            n_filtered = int(filtered_stats['n_filtered'])
            filter_frac = filtered_stats['filter_fraction']

            print(f"\n{result.approach} (Filtered: h4 > 0.001)")
            print("-" * 50)
            print(f"  Note: {n_filtered} samples ({100*filter_frac:.1f}%) with h4 > 0.001")
            print(f"  h1: {result.h1_point:.6f}")
            print(f"    90% CI: [{filtered_stats['h1']['p5']:.6f}, {filtered_stats['h1']['p95']:.6f}]")
            print(f"  h2: {result.h2_point:.6f}")
            print(f"    90% CI: [{filtered_stats['h2']['p5']:.6f}, {filtered_stats['h2']['p95']:.6f}]")
            if 'h4' in filtered_stats:
                print(f"  h4 (persistence decay): {result.h4_point:.6f}")
                print(f"    90% CI: [{filtered_stats['h4']['p5']:.6f}, {filtered_stats['h4']['p95']:.6f}]")
            if result.T_opt_point is not None and not np.isnan(result.T_opt_point):
                print(f"  T_opt: {result.T_opt_point:.2f} C")
                print(f"    90% CI: [{filtered_stats['T_opt']['p5']:.2f}, {filtered_stats['T_opt']['p95']:.2f}]")

        else:
            # Standard approaches (Approach1J, Approach1P, Approach1L, null models)
            if result.T_opt_point is not None and not np.isnan(result.T_opt_point):
                print(f"  T_opt: {result.T_opt_point:.2f} C")
                print(f"    90% CI: [{stats['T_opt']['p5']:.2f}, {stats['T_opt']['p95']:.2f}]")
                print(f"    IQR:    [{stats['T_opt']['p25']:.2f}, {stats['T_opt']['p75']:.2f}]")
            else:
                print(f"  T_opt: N/A")
            print(f"  h1: {result.h1_point:.6f}")
            print(f"    90% CI: [{stats['h1']['p5']:.6f}, {stats['h1']['p95']:.6f}]")
            print(f"  h2: {result.h2_point:.6f}")
            print(f"    90% CI: [{stats['h2']['p5']:.6f}, {stats['h2']['p95']:.6f}]")

        print(f"  Successful iterations: {result.n_successful}/{result.n_bootstrap}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
