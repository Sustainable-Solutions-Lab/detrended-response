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
    compute_country_trends_loess,
)
import numpy as np
from src.fitting import (
    fit_approach0_no_detrending,
    fit_approach2_temperature_detrending,
    fit_approach3_growth_detrending,
    fit_approach1_combined_detrending,
    fit_approach4_combined_quadratic_detrending,
    fit_approach5_precomputed_k_quadratic,
    fit_approach5a_precomputed_k_linear_temp,
    fit_approach5b_precomputed_k_gdp_only,
    fit_approach5c_precomputed_k_combined,
    fit_approach5d_precomputed_k_gdp_response,
    fit_approach6_precomputed_k_loess,
    fit_approach6a_separate_high_low_loess,
    fit_approach6b_low_only_loess,
    fit_approach6c_departure_trend_loess,
    fit_approach8_gaussian_loess,
    fit_approach8a_shared_Topt_loess,
    fit_approach8b_modulated_loess,
    fit_nocr0_joint,
    fit_nocr5_precomputed_k,
)
from src.output import save_all_outputs, create_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Detrended response analysis of climate-economy relationship"
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
        "--loess-window",
        type=int,
        default=25,
        help="Window size in years for LOESS smoothing (default: 25)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Detrended Response Analysis")
    print("=" * 70)

    # Load data
    input_file = None  # Track which input file was used
    if args.use_csv and args.use_csv.strip():
        # Load from pre-processed CSV file
        csv_path = Path(args.use_csv).expanduser()
        input_file = str(csv_path)
        print(f"\n[1/11] Loading data from {csv_path}...")
        data = load_data_from_csv(
            str(csv_path),
            year_min=args.year_min,
            year_max=args.year_max
        )
    else:
        # Load from Maddison/CRU files
        year_min = args.year_min if args.year_min is not None else 1960
        year_max = args.year_max if args.year_max is not None else 2022
        input_file = f"{args.maddison} + {args.cru}"
        print(f"\n[1/11] Loading data from {args.maddison} and {args.cru}...")
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
    print("\n[2/11] Computing country-level trends...")
    trends = compute_country_trends(data)
    print("      Done.")

    # Compute year means and country trends with k for Approach 5
    print("\n[3/11] Computing year means k[t] and adjusted country trends...")
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    print("      Done.")

    # Compute LOESS trends for Approaches 6 and 7
    print(f"\n[4/11] Computing LOESS trends (window={args.loess_window} years)...")
    trends_loess = compute_country_trends_loess(data, year_means, args.loess_window)
    print("      Done.")

    # Compute Y_ref for Approach 7 (based on most recent year)
    max_year = data.year_range[1]
    mask_recent = data.year == max_year
    Y_ref = np.mean(data.pcGDP[mask_recent])
    print(f"      Y_ref (mean pcGDP in {max_year}): {Y_ref:.2f}")

    # Fit all approaches
    results = {}

    print("\n[5/11] Fitting Approach 0: Conjoined OLS fit...")
    results['approach0'] = fit_approach0_no_detrending(data)
    print("      Done.")

    print("\n[6/11] Fitting Approaches 1-4: Detrending approaches...")
    results['approach1'] = fit_approach1_combined_detrending(data, trends)
    results['approach2'] = fit_approach2_temperature_detrending(data, trends)
    results['approach3'] = fit_approach3_growth_detrending(data, trends)
    results['approach4'] = fit_approach4_combined_quadratic_detrending(data, trends)
    print("      Done.")

    print("\n[7/11] Fitting Approach 5: Precomputed k (quadratic)...")
    results['approach5'] = fit_approach5_precomputed_k_quadratic(data, trends_with_k, year_means)
    print("      Done.")

    print("\n[8/11] Fitting Approaches 5a, 5b, 5c, 5d: Precomputed k variants...")
    results['approach5a'] = fit_approach5a_precomputed_k_linear_temp(data, trends_with_k, year_means)
    results['approach5b'] = fit_approach5b_precomputed_k_gdp_only(data, trends_with_k, year_means)
    results['approach5c'] = fit_approach5c_precomputed_k_combined(data, trends_with_k, year_means)
    results['approach5d'] = fit_approach5d_precomputed_k_gdp_response(data, trends_with_k, year_means, Y_ref)
    print("      Done.")

    print("\n[9/11] Fitting Approaches 6, 6a, 6b, 6c, 8, 8a, 8b: LOESS detrending...")
    results['approach6'] = fit_approach6_precomputed_k_loess(data, trends_loess, year_means)
    results['approach6a'] = fit_approach6a_separate_high_low_loess(data, trends_loess, year_means)
    results['approach6b'] = fit_approach6b_low_only_loess(data, trends_loess, year_means)
    results['approach6c'] = fit_approach6c_departure_trend_loess(data, trends_loess, year_means)
    results['approach8'] = fit_approach8_gaussian_loess(data, trends_loess, year_means)
    results['approach8a'] = fit_approach8a_shared_Topt_loess(data, trends_loess, year_means)
    results['approach8b'] = fit_approach8b_modulated_loess(data, trends_loess, year_means)
    print("      Done.")

    print("\n[10/11] Fitting null models (no climate response)...")
    results['nocr0'] = fit_nocr0_joint(data)
    results['nocr5'] = fit_nocr5_precomputed_k(data, trends_with_k, year_means)
    print("      Done.")

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for name, r in results.items():
        print(f"\n{r.approach}")
        print("-" * 50)
        # Special handling for Approach 6a/6b (separate total/trend frequency)
        if hasattr(r, 'h1_total') and hasattr(r, 'h1_trend'):
            print(f"  h1_total = {r.h1_total:.6f}  (SE: {r.h1_total_se:.6f})")
            print(f"  h2_total = {r.h2_total:.6f}  (SE: {r.h2_total_se:.6f})")
            print(f"  h1_trend = {r.h1_trend:.6f}  (SE: {r.h1_trend_se:.6f})")
            print(f"  h2_trend = {r.h2_trend:.6f}  (SE: {r.h2_trend_se:.6f})")
            if not np.isnan(r.T_optimal_total):
                print(f"  T_optimal_total = {r.T_optimal_total:.2f} C")
            else:
                print(f"  T_optimal_total = N/A")
            if not np.isnan(r.T_optimal_trend):
                print(f"  T_optimal_trend = {r.T_optimal_trend:.2f} C")
            else:
                print(f"  T_optimal_trend = N/A")
        # Special handling for Approach 6c (departure/trend decomposition)
        elif hasattr(r, 'h1_dep') and hasattr(r, 'h1_trend'):
            print(f"  h1_dep = {r.h1_dep:.6f}  (SE: {r.h1_dep_se:.6f})")
            print(f"  h2_dep = {r.h2_dep:.6f}  (SE: {r.h2_dep_se:.6f})")
            print(f"  h1_trend = {r.h1_trend:.6f}  (SE: {r.h1_trend_se:.6f})")
            print(f"  h2_trend = {r.h2_trend:.6f}  (SE: {r.h2_trend_se:.6f})")
            if not np.isnan(r.T_optimal_dep):
                print(f"  T_optimal_dep = {r.T_optimal_dep:.2f} C")
            else:
                print(f"  T_optimal_dep = N/A")
            if not np.isnan(r.T_optimal_trend):
                print(f"  T_optimal_trend = {r.T_optimal_trend:.2f} C")
            else:
                print(f"  T_optimal_trend = N/A")
        # Special handling for Approach 8a (total/trend with shared T_opt)
        elif hasattr(r, 'h2_total') and hasattr(r, 'h2_trend') and hasattr(r, 'T_opt'):
            print(f"  h2_total = {r.h2_total:.6f}  (SE: {r.h2_total_se:.6f})")
            print(f"  h2_trend = {r.h2_trend:.6f}  (SE: {r.h2_trend_se:.6f})")
            print(f"  T_opt = {r.T_opt:.4f}  (SE: {r.T_opt_se:.4f})")
        # Special handling for Approach 8 (Piecewise Quadratic, temperature regions)
        elif hasattr(r, 'h2_low') and hasattr(r, 'h2_high') and hasattr(r, 'T_opt'):
            print(f"  h2_low = {r.h2_low:.6f}  (SE: {r.h2_low_se:.6f})")
            print(f"  h2_high = {r.h2_high:.6f}  (SE: {r.h2_high_se:.6f})")
            print(f"  T_opt = {r.T_opt:.4f}  (SE: {r.T_opt_se:.4f})")
        else:
            # Print h0 for Approach 8b (modulated response)
            if hasattr(r, 'h0') and r.h0 is not None:
                print(f"  h0 = {r.h0:12.6f}  (SE: {r.h0_se:.6f})")
            print(f"  h1 = {r.h1:12.6f}  (SE: {r.h1_se:.6f})")
            print(f"  h2 = {r.h2:12.6f}  (SE: {r.h2_se:.6f})")
            # Print beta for Approach 7
            if hasattr(r, 'beta') and r.beta is not None:
                print(f"  beta = {r.beta:10.4f}  (SE: {r.beta_se:.4f})")
                print(f"  Y_ref = {r.Y_ref:.2f}")
            if np.isnan(r.T_optimal):
                print(f"  T_optimal = N/A")
            else:
                print(f"  T_optimal = {r.T_optimal:.2f} C")
        print(f"  R² = {r.r_squared:.4f}")
        print(f"  Total R² = {r.total_r_squared:.4f}")
        print(f"  Adjusted R² = {r.adj_r_squared:.4f}")
        print(f"  RMSE = {r.rmse:.6f}")
        if r.rms_imbalance is not None:
            print(f"  RMS Imbalance = {r.rms_imbalance:.6f}")
        if r.rms_h is not None:
            print(f"  RMS h(T) = {r.rms_h:.6f}")
        if r.imbalance_ratio is not None:
            print(f"  Imbalance Ratio = {r.imbalance_ratio:.4f}")

    # Save outputs
    print("\n[11/11] Saving outputs...")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="analysis_")

    save_all_outputs(data, trends, results, output_dir, input_file=input_file)

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
