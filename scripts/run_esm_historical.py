#!/usr/bin/env python3
"""Run ESM historical analysis for multiple Earth System Models.

Processes ESM historical CSV files through 4 approaches:
  Approach QJ: Quadratic response with joint OLS
  Approach QL: Quadratic response with LOESS detrending
  Approach DL: Persistence decay model with LOESS detrending
  Approach LL: Level effect model (h4=1) with LOESS detrending

Usage:
    python scripts/run_esm_historical.py                        # All 3 models
    python scripts/run_esm_historical.py --models ACCESS-ESM1-5 # One model
    python scripts/run_esm_historical.py --n-bootstrap 10       # Quick test
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
)
from src.fitting import fit_all_approaches
from src.bootstrap import run_bootstrap, compute_bootstrap_statistics, compute_ApproachDL_filtered_statistics
from src.output import save_all_outputs, save_bootstrap_coefficients_csv, save_bootstrap_k_samples_csv, save_bootstrap_h_values, save_all_bootstrap_plots
import numpy as np

# The 6 approaches to fit
ESM_APPROACHES = ['Approach QJ', 'Approach DJ', 'Approach LJ', 'Approach QL', 'Approach DL', 'Approach LL']

# Available ESM models
AVAILABLE_MODELS = ['ACCESS-ESM1-5', 'CNRM-ESM2-1', 'MIROC-ES2L']


def run_single_model(model_name, csv_path, output_dir, args):
    """Run analysis and bootstrap for a single ESM model."""
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    analysis_dir = output_dir / "analysis"
    bootstrap_dir = output_dir / "bootstrap"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    t_start = time.perf_counter()
    print(f"\n[1/5] Loading data from {csv_path}...")
    data = load_data_from_csv(
        str(csv_path),
        start_year=args.start_year,
    )
    t_load = time.perf_counter() - t_start
    print(f"      Observations: {data.n_obs}")
    print(f"      Countries/Regions: {data.n_countries}")
    print(f"      Years: {data.n_years}")
    print(f"      Year range: {data.year_range[0]} - {data.year_range[1]}")
    print(f"      Time: {t_load:.3f}s")

    # Step 2: Compute trends
    print("\n[2/5] Computing country-level trends...")
    t_start = time.perf_counter()
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    t_trends = time.perf_counter() - t_start
    print(f"      Time: {t_trends:.3f}s")

    # Step 3: Compute LOESS trends
    loess_window = args.loess_window
    print(f"\n[3/5] Computing LOESS trends (window={loess_window:.1f} years)...")
    t_start = time.perf_counter()
    trends_loess = compute_country_trends_loess(data, year_means, loess_window)
    t_loess = time.perf_counter() - t_start
    print(f"      Time: {t_loess:.3f}s")

    # Step 4: Fit approaches (point estimates)
    print(f"\n[4/5] Fitting {len(ESM_APPROACHES)} approaches...")
    t_start = time.perf_counter()
    results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess,
        approaches=ESM_APPROACHES,
    )
    t_fit = time.perf_counter() - t_start

    # Print summary
    print("\n" + "-" * 50)
    print("Point Estimates")
    print("-" * 50)
    for name, r in results.items():
        print(f"\n  {r.approach}")
        print(f"    h1 = {r.h1:.6f}  (SE: {r.h1_se:.6f})")
        print(f"    h2 = {r.h2:.6f}  (SE: {r.h2_se:.6f})")
        if hasattr(r, 'h4') and r.h4 is not None:
            print(f"    h4 = {r.h4:.6f}  (SE: {r.h4_se:.6f})")
        if not np.isnan(r.T_opt):
            print(f"    T_opt = {r.T_opt:.2f} C")
        print(f"    R² = {r.r_squared:.4f}, Total R² = {r.total_r_squared:.4f}")

    # Save analysis outputs
    save_all_outputs(data, trends, results, analysis_dir,
                     input_file=str(csv_path), approaches=ESM_APPROACHES)

    # Save run metadata
    import json
    metadata = {
        'model': model_name,
        'loess_window': loess_window,
        'input_file': str(csv_path),
        'year_min': data.year_range[0],
        'year_max': data.year_range[1],
        'n_countries': data.n_countries,
        'n_obs': data.n_obs,
        'approaches': ESM_APPROACHES,
    }
    with open(analysis_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 5: Bootstrap
    print(f"\n[5/5] Running bootstrap ({args.n_bootstrap} iterations)...")
    t_start = time.perf_counter()

    bootstrap_results, country_samples, h_T_samples, year_samples = run_bootstrap(
        data, trends, results,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        verbose=not args.quiet,
        loess_window=loess_window,
        h_T_approaches=ESM_APPROACHES,
        sample_years=args.sample_years,
        approaches=ESM_APPROACHES,
    )
    t_bootstrap = time.perf_counter() - t_start

    # Compute statistics
    all_stats = {}
    for name, br in bootstrap_results.items():
        if name == 'Approach DL':
            all_stats[name] = compute_ApproachDL_filtered_statistics(br)
        else:
            all_stats[name] = compute_bootstrap_statistics(br)

    # Save bootstrap outputs
    save_bootstrap_coefficients_csv(bootstrap_results, bootstrap_dir)
    save_bootstrap_k_samples_csv(bootstrap_results, bootstrap_dir)
    save_bootstrap_h_values(h_T_samples, data, bootstrap_dir)
    save_all_bootstrap_plots(bootstrap_results, all_stats, bootstrap_dir)

    # Save bootstrap metadata
    bootstrap_metadata = {
        'model': model_name,
        'n_bootstrap': args.n_bootstrap,
        'random_seed': args.random_seed,
        'sample_years': args.sample_years,
        'approaches': ESM_APPROACHES,
    }
    with open(bootstrap_dir / 'bootstrap_metadata.json', 'w') as f:
        json.dump(bootstrap_metadata, f, indent=2)

    # Timing summary
    total_time = t_load + t_trends + t_loess + t_fit + t_bootstrap
    print(f"\n  {model_name} complete in {total_time:.1f}s")

    return results, bootstrap_results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run ESM historical analysis for Earth System Models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=AVAILABLE_MODELS,
        help=f"Models to process (default: all). Available: {', '.join(AVAILABLE_MODELS)}",
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
        "--sample-years",
        action="store_true",
        default=True,
        help="Sample years with replacement (default: True for ESM data)",
    )
    parser.add_argument(
        "--no-sample-years",
        action="store_true",
        help="Disable year sampling",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1960,
        help="Exclude years before this (default: 1960)",
    )
    parser.add_argument(
        "--loess-window",
        type=float,
        default=42.447947771790915,
        help="Window size in years for LOESS smoothing (default: 42.45)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent output directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args(argv)

    if args.no_sample_years:
        args.sample_years = False

    # Set up output directory
    if args.output_dir:
        parent_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = Path("data/output") / f"esm_historical_{timestamp}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()

    print("=" * 70)
    print("ESM HISTORICAL ANALYSIS")
    print("=" * 70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Approaches: {', '.join(ESM_APPROACHES)}")
    print(f"Bootstrap: {args.n_bootstrap} iterations")
    print(f"Sample years: {args.sample_years}")
    print(f"Output: {parent_dir}")

    # Process each model
    for model_name in args.models:
        csv_path = Path("data/input") / f"{model_name}_historical.csv"
        model_output_dir = parent_dir / model_name
        run_single_model(model_name, csv_path, model_output_dir, args)

    total_time = time.perf_counter() - t_total_start
    print("\n" + "=" * 70)
    print("ESM HISTORICAL ANALYSIS COMPLETE")
    print(f"Total time: {total_time:.1f}s")
    print(f"All outputs in: {parent_dir}")
    print("=" * 70)

    return parent_dir


if __name__ == "__main__":
    main()
