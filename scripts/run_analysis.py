#!/usr/bin/env python3
"""Main script for detrended response analysis.

This script implements and compares multiple approaches for analyzing
the climate-economy relationship with explicit time trend detrending.

Approaches (publication-ready):
    Approach NJ: Null model (h1=h2=0) with joint OLS (country trends + year effects only)
    Approach NP: Null model (h1=h2=0) with polynomial trend identification
    Approach NL: Null model (h1=h2=0) with LOESS trend identification
    Approach QJ: Quadratic response with joint OLS (country time trends and year fixed effects)
    Approach QP: Quadratic response with polynomial trend identification (linear T + quadratic GDP)
    Approach QL: Quadratic response with LOESS trend identification
    Approach PJ: Piecewise quadratic response with joint OLS
    Approach PP: Piecewise quadratic response with polynomial trend identification
    Approach PL: Piecewise quadratic response with LOESS trend identification
    Approach DJ: Persistence decay model with joint OLS
    Approach DP: Persistence decay model with polynomial trend identification
    Approach DL: Persistence decay model with LOESS trend identification

Usage:
    python scripts/run_analysis.py [--year-min YEAR] [--year-max YEAR] [--output-dir DIR]
"""

import argparse
import sys
import time
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


def main(argv=None):
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
        "--start-year",
        type=int,
        default=None,
        help="Exclude years before this from analysis (applied after growth computation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped)",
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
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=None,
        help="Two-letter approach codes to fit (e.g., QJ PL DJ). "
             "First letter: N/Q/P/S/T/D/L (response type). "
             "Second letter: J/P/L (trend method). "
             "Default: fit all approaches.",
    )

    args = parser.parse_args(argv)

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
    print("Detrended Response Analysis")
    print("=" * 70)

    # Load data
    input_file = None  # Track which input file was used
    t_start = time.perf_counter()
    if args.use_csv and args.use_csv.strip():
        # Load from pre-processed CSV file
        csv_path = Path(args.use_csv).expanduser()
        input_file = str(csv_path)
        print(f"\n[1/5] Loading data from {csv_path}...")
        data = load_data_from_csv(
            str(csv_path),
            year_min=args.year_min,
            year_max=args.year_max,
            start_year=args.start_year,
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
            year_min=year_min, year_max=year_max,
            start_year=args.start_year,
        )
    t_load = time.perf_counter() - t_start

    print(f"      Observations: {data.n_obs}")
    print(f"      Countries: {data.n_countries}")
    print(f"      Years: {data.n_years}")
    print(f"      Year range: {data.year_range[0]} - {data.year_range[1]}")
    print(f"      Time: {t_load:.3f}s")
    print(f"      Start year: {data.year_range[0]}")
    print(f"      End year: {data.year_range[1]}")
    print(f"      Mean GDP growth rate: {np.mean(data.growth_pcGDP):.16e}")
    print(f"      Mean temperature: {np.mean(data.temp):.16e} C")

    # Compute country-level trends
    print("\n[2/5] Computing country-level trends...")
    t_start = time.perf_counter()
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    t_trends = time.perf_counter() - t_start
    print(f"      Time: {t_trends:.3f}s")

    # Compute LOESS trends
    print(f"\n[3/5] Computing LOESS trends (window={loess_window:.1f} years)...")
    t_start = time.perf_counter()
    trends_loess = compute_country_trends_loess(data, year_means, loess_window)
    t_loess = time.perf_counter() - t_start
    print(f"      Time: {t_loess:.3f}s")

    # Fit all methods
    approach_names = [f"Approach {code}" for code in args.approaches] if args.approaches else None
    print("\n[4/5] Fitting all methods...")
    t_start = time.perf_counter()
    results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess,
        approaches=approach_names,
    )
    t_fit = time.perf_counter() - t_start

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for name, r in results.items():
        print(f"\n{r.approach}")
        print("-" * 50)

        # Approach PL/Approach PJ/Approach PP: h2 (below T_opt), h4 (above T_opt), T_opt (piecewise)
        if name in ['Approach PL', 'Approach PJ', 'Approach PP'] and hasattr(r, 'h4'):
            print(f"  h2 (below T_opt) = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (above T_opt) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            print(f"  T_opt = {r.T_opt:.4f}  (SE: {r.T_opt_se:.4f})")

        # Three-interval approaches: h2, h4, T_crit_low, T_crit_high, T_opt
        elif name in ['Approach TL', 'Approach TJ', 'Approach TP'] and hasattr(r, 'T_crit_low'):
            print(f"  h2 (slope below T_crit_low) = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (slope above T_crit_high) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            print(f"  T_crit_low = {r.T_crit_low:.4f}")
            print(f"  T_crit_high = {r.T_crit_high:.4f}")
            T_opt_str = f"{r.T_opt:.4f}" if not np.isnan(r.T_opt) else "N/A"
            print(f"  T_opt = {T_opt_str}")

        # Segmented linear approaches: h2, h4 (slopes), T_opt
        elif name in ['Approach SL', 'Approach SJ', 'Approach SP'] and hasattr(r, 'h4'):
            print(f"  h2 (slope below T_opt) = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (slope above T_opt) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            print(f"  T_opt = {r.T_opt:.4f}  (SE: {r.T_opt_se:.4f})")

        # Approach DL/Approach DJ/Approach DP: h1, h2, h4 (persistence decay), T_opt
        elif name in ['Approach DL', 'Approach DJ', 'Approach DP'] and hasattr(r, 'h4'):
            print(f"  h1 = {r.h1:.6f}  (SE: {r.h1_se:.6f})")
            print(f"  h2 = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
            print(f"  h4 (persistence decay) = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
            if not np.isnan(r.T_opt):
                print(f"  T_opt = {r.T_opt:.2f} C")
            else:
                print(f"  T_opt = N/A")

        else:
            # Standard approaches (Approach QJ, Approach QP, Approach QL, null models)
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

    # Save outputs
    print("\n[5/5] Saving outputs...")
    t_start = time.perf_counter()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(prefix="analysis_", suffix=mw_suffix)

    # Define approach order for output tables (filter to only fitted approaches)
    all_approach_order = [
        'Approach NJ', 'Approach NP', 'Approach NL',
        'Approach QJ', 'Approach QP', 'Approach QL',
        'Approach PJ', 'Approach PP', 'Approach PL',
        'Approach SJ', 'Approach SP', 'Approach SL',
        'Approach TJ', 'Approach TP', 'Approach TL',
        'Approach DJ', 'Approach DP', 'Approach DL',
        'Approach LJ', 'Approach LL',
    ]
    approach_order = [a for a in all_approach_order if a in results]
    save_all_outputs(data, trends, results, output_dir, input_file=input_file, approaches=approach_order)
    t_output = time.perf_counter() - t_start

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
    }
    metadata_path = output_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved: {metadata_path}")

    print(f"\nOutput saved to: {output_dir}")
    print(f"      Time: {t_output:.3f}s")

    # Print timing summary
    total_time = t_load + t_trends + t_loess + t_fit + t_output
    print("\n" + "-" * 70)
    print("Timing Summary")
    print("-" * 70)
    print(f"  Data loading:     {t_load:7.3f}s")
    print(f"  Country trends:   {t_trends:7.3f}s")
    print(f"  LOESS trends:     {t_loess:7.3f}s")
    print(f"  Model fitting:    {t_fit:7.3f}s")
    print(f"  Output/plots:     {t_output:7.3f}s")
    print(f"  Total:            {total_time:7.3f}s")
    print("=" * 70)

    return output_dir


if __name__ == "__main__":
    main()
