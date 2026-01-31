#!/usr/bin/env python3
"""Main script for detrended response analysis.

This script implements and compares three approaches to making explicit
the time trend terms in the Burke et al. (2015) climate-economy relationship:

1. Linear temperature detrending
2. Quadratic GDP growth detrending
3. Combined detrending

Usage:
    python scripts/run_analysis.py [--year-min YEAR] [--year-max YEAR] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data, load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
)
from src.fitting import (
    fit_approach0_no_detrending,
    fit_approach1_temperature_detrending,
    fit_approach2_growth_detrending,
    fit_approach3_combined_detrending,
    fit_approach4_combined_linear_detrending,
    fit_approach5_combined_quadratic_detrending,
    fit_approach6_precomputed_k_linear,
    fit_approach7_precomputed_k_quadratic,
)
from src.output import save_all_outputs, create_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Detrended response analysis of climate-economy relationship"
    )
    parser.add_argument(
        "--use-csv",
        type=str,
        default="data/input/df_base_withPop.csv",
        help="Use pre-processed CSV file (default: data/input/df_base_withPop.csv). "
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

    args = parser.parse_args()

    print("=" * 70)
    print("Detrended Response Analysis")
    print("=" * 70)

    # Load data
    if args.use_csv and args.use_csv.strip():
        # Load from pre-processed CSV file
        csv_path = Path(args.use_csv).expanduser()
        print(f"\n[1/10] Loading data from {csv_path}...")
        data = load_data_from_csv(
            str(csv_path),
            year_min=args.year_min,
            year_max=args.year_max
        )
    else:
        # Load from Maddison/CRU files
        year_min = args.year_min if args.year_min is not None else 1960
        year_max = args.year_max if args.year_max is not None else 2022
        print(f"\n[1/10] Loading data from {args.maddison} and {args.cru}...")
        print(f"      Year range: {year_min} - {year_max}")
        data = load_data(
            args.maddison, args.cru,
            year_min=year_min, year_max=year_max
        )

    print(f"      Observations: {data.n_obs}")
    print(f"      Countries: {data.n_countries}")
    print(f"      Years: {data.n_years}")
    print(f"      Year range: {data.year_range[0]} - {data.year_range[1]}")

    # Compute country-level trends
    print("\n[2/10] Computing country-level trends...")
    trends = compute_country_trends(data)
    print("      Done.")

    # Compute year means and country trends with k for Approaches 6 and 7
    print("\n[3/10] Computing year means k[t] and adjusted country trends...")
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    print("      Done.")

    # Fit all approaches
    results = {}

    print("\n[4/10] Fitting Approach 0: No detrending...")
    results['approach0'] = fit_approach0_no_detrending(data)
    print("      Done.")

    print("\n[5/10] Fitting Approach 1: Temperature detrending...")
    results['approach1'] = fit_approach1_temperature_detrending(data, trends)
    print("      Done.")

    print("\n[6/10] Fitting Approach 2: GDP growth detrending...")
    results['approach2'] = fit_approach2_growth_detrending(data, trends)
    print("      Done.")

    print("\n[7/10] Fitting Approach 3: Combined detrending (quadratic GDP, linear T)...")
    results['approach3'] = fit_approach3_combined_detrending(data, trends)
    print("      Done.")

    print("\n[8/10] Fitting Approach 4: Combined detrending (linear GDP, linear T)...")
    results['approach4'] = fit_approach4_combined_linear_detrending(data, trends)
    print("      Done.")

    print("\n[9/10] Fitting Approach 5: Combined detrending (quadratic GDP, quadratic T)...")
    results['approach5'] = fit_approach5_combined_quadratic_detrending(data, trends)
    print("      Done.")

    print("\n[10/10] Fitting Approaches 6 & 7: Precomputed k with linear/quadratic trends...")
    results['approach6'] = fit_approach6_precomputed_k_linear(data, trends_with_k, year_means)
    results['approach7'] = fit_approach7_precomputed_k_quadratic(data, trends_with_k, year_means)
    print("      Done.")

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for name, r in results.items():
        print(f"\n{r.approach}")
        print("-" * 50)
        print(f"  h1 = {r.h1:12.6f}  (SE: {r.h1_se:.6f})")
        print(f"  h2 = {r.h2:12.6f}  (SE: {r.h2_se:.6f})")
        print(f"  T_optimal = {r.T_optimal:.2f} C")
        print(f"  R² = {r.r_squared:.4f}")
        print(f"  Adjusted R² = {r.adj_r_squared:.4f}")
        print(f"  RMSE = {r.rmse:.6f}")

    # Save outputs
    print("\n" + "=" * 70)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir()

    save_all_outputs(data, trends, results, output_dir)

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
