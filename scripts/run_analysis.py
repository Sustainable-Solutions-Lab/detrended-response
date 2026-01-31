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

from src.data_loader import load_data
from src.detrending import compute_country_trends
from src.fitting import fit_all_approaches
from src.output import save_all_outputs, create_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Detrended response analysis of climate-economy relationship"
    )
    parser.add_argument(
        "--maddison",
        default="data/input/mpd2023_web.xlsx",
        help="Path to Maddison GDP Excel file",
    )
    parser.add_argument(
        "--cru",
        default="data/input/cru_climate_data.csv",
        help="Path to CRU temperature CSV file",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=1960,
        help="Minimum year to include (default: 1960)",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=2022,
        help="Maximum year to include (default: 2022)",
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
    print(f"\nLoading data from {args.maddison} and {args.cru}...")
    print(f"Year range: {args.year_min} - {args.year_max}")

    data = load_data(
        args.maddison, args.cru,
        year_min=args.year_min, year_max=args.year_max
    )

    print(f"  Observations: {data.n_obs}")
    print(f"  Countries: {data.n_countries}")
    print(f"  Years: {data.n_years}")

    # Compute country-level trends
    print("\nComputing country-level trends...")
    trends = compute_country_trends(data)

    # Fit all approaches
    print("\nFitting models...")
    results = fit_all_approaches(data, trends)

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
