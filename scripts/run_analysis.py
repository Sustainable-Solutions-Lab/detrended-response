#!/usr/bin/env python3
"""Main script for detrended response analysis.

This script implements and compares multiple methods for analyzing
the climate-economy relationship with explicit time trend detrending.

Methods:
    method0: Conjoined OLS with country time trends and year fixed effects
    method1: Pre-computed k with linear T + quadratic GDP detrending
    method2: Pre-computed k with LOESS trends
    method3: Piecewise quadratic response with LOESS
    method4: T response with quadratic departure term
    method5: Persistence decay model with LOESS
    method0h0: Null model (h1=h2=0) for method0
    method1h0: Null model (h1=h2=0) for method1

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
from src.fitting import fit_all_approaches
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
    parser.add_argument(
        "--mean-weight-distance",
        type=float,
        default=None,
        help="Mean weighting distance in years for LOESS. Window = 44/7 * this value. "
             "If specified, adds '_mwXX' suffix to output directory.",
    )

    args = parser.parse_args()

    # Compute LOESS window from mean weight distance if specified
    if args.mean_weight_distance is not None:
        loess_window = (44 / 7) * args.mean_weight_distance
        mw_suffix = f"mw{int(args.mean_weight_distance):02d}"
    else:
        loess_window = args.loess_window
        mw_suffix = ""

    print("=" * 70)
    print("Detrended Response Analysis")
    print("=" * 70)

    # Load data
    input_file = None  # Track which input file was used
    if args.use_csv and args.use_csv.strip():
        # Load from pre-processed CSV file
        csv_path = Path(args.use_csv).expanduser()
        input_file = str(csv_path)
        print(f"\n[1/5] Loading data from {csv_path}...")
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
        print(f"\n[1/5] Loading data from {args.maddison} and {args.cru}...")
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
    print("\n[2/5] Computing country-level trends...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    print("      Done.")

    # Compute LOESS trends
    print(f"\n[3/5] Computing LOESS trends (window={loess_window:.1f} years)...")
    trends_loess = compute_country_trends_loess(data, year_means, loess_window)
    print("      Done.")

    # Fit all methods
    print("\n[4/5] Fitting all methods...")
    results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess
    )
    print("      Done.")

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for name, r in results.items():
        print(f"\n{r.approach}")
        print("-" * 50)

        # method4: h1,h2 (actual T), h4 (departure), T_opt, T_dep_opt
        if name == 'method4' and hasattr(r, 'h4'):
            print(f"  h1 (T) = {r.h1:.6f}  (SE: {r.h1_se:.6f})")
            print(f"  h2 (T) = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (departure) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            if not np.isnan(r.T_opt):
                print(f"  T_opt = {r.T_opt:.2f} C")
            else:
                print(f"  T_opt = N/A")
            if hasattr(r, 'T_dep_opt') and not np.isnan(r.T_dep_opt):
                print(f"  T_dep_opt (departure opt) = {r.T_dep_opt:.2f} C")
            else:
                print(f"  T_dep_opt (departure opt) = N/A")

        # method3: h2 (below T_opt), h4 (above T_opt), T_opt
        elif name == 'method3' and hasattr(r, 'h4'):
            print(f"  h2 (below T_opt) = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (above T_opt) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            print(f"  T_opt = {r.T_opt:.4f}  (SE: {r.T_opt_se:.4f})")

        # method5: h1, h2, h4 (persistence decay), T_opt
        elif name == 'method5' and hasattr(r, 'h4'):
            print(f"  h1 = {r.h1:.6f}  (SE: {r.h1_se:.6f})")
            print(f"  h2 = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (persistence decay) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            if not np.isnan(r.T_opt):
                print(f"  T_opt = {r.T_opt:.2f} C")
            else:
                print(f"  T_opt = N/A")

        else:
            # Standard methods (method0, method1, method2, null models)
            print(f"  h1 = {r.h1:12.6f}  (SE: {r.h1_se:.6f})")
            print(f"  h2 = {r.h2:12.6f}  (SE: {r.h2_se:.6f})")
            if hasattr(r, 'T_opt') and not np.isnan(r.T_opt):
                print(f"  T_opt = {r.T_opt:.2f} C")
            else:
                print(f"  T_opt = N/A")
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
    print("\n[5/5] Saving outputs...")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="analysis_", suffix=mw_suffix)

    save_all_outputs(data, trends, results, output_dir, input_file=input_file)

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
