#!/usr/bin/env python3
"""Fix missing point estimates in bootstrap_h_values.csv.

This script adds the missing point estimates (iteration=-1) for approaches
that were skipped due to a bug: Approach2J, Approach2P, Approach3J, Approach3P.

Usage:
    python scripts/fix_missing_point_estimates.py --bootstrap-dir data/output/reference/bootstrap_*
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.fitting import fit_all_approaches
from src.detrending import (
    compute_country_trends,
    compute_country_trends_with_k,
    compute_country_trends_loess,
    compute_year_means,
)
from src.output import compute_persistence_accumulators, compute_pre_first_year_correction, _get_T_loess_at_base_year


def main():
    parser = argparse.ArgumentParser(description="Fix missing point estimates in bootstrap_h_values.csv")
    parser.add_argument(
        "--bootstrap-dir",
        type=str,
        required=True,
        help="Path to bootstrap output directory",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Input data file",
    )
    parser.add_argument(
        "--loess-window",
        type=int,
        default=21,
        help="LOESS window size in years",
    )
    args = parser.parse_args()

    bootstrap_dir = Path(args.bootstrap_dir)
    h_values_path = bootstrap_dir / "bootstrap_h_values.csv"

    if not h_values_path.exists():
        print(f"ERROR: {h_values_path} not found")
        return 1

    print(f"Fixing missing point estimates in: {h_values_path}")

    # Load data and fit approaches to get point estimates
    print("\n[1/4] Loading data...")
    data = load_data_from_csv(args.data_file)
    input_file = args.data_file

    print("\n[2/4] Computing trends...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    trends_loess = compute_country_trends_loess(data, year_means, args.loess_window)

    print("\n[3/4] Fitting approaches...")
    results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess
    )

    # Check which approaches are missing point estimates
    print("\n[4/4] Computing missing point estimates...")

    # Read existing point estimates
    existing_approaches = set()
    for chunk in pd.read_csv(h_values_path, comment='#', chunksize=100000):
        point_est = chunk[chunk['iteration'] == -1]
        existing_approaches.update(point_est['approach'].unique())

    print(f"  Existing point estimates: {sorted(existing_approaches)}")

    # Approaches that need point estimates
    missing_approaches = []
    for name in ['Approach2J', 'Approach2P', 'Approach3J', 'Approach3P']:
        if name not in existing_approaches and name in results:
            missing_approaches.append(name)

    if not missing_approaches:
        print("  No missing point estimates found!")
        return 0

    print(f"  Missing point estimates: {missing_approaches}")

    # Prepare data arrays
    iso3_arr = np.array([data.idx_to_iso[idx] for idx in data.country_idx])
    year_arr = data.year.astype(int)
    temp_arr = data.temp
    n_obs = data.n_obs

    # Compute and append missing point estimates
    new_lines = []

    for name in missing_approaches:
        r = results[name]
        print(f"  Computing h(T) for {name}...")

        if name in ['Approach2J', 'Approach2P']:
            # Piecewise quadratic
            below = temp_arr <= r.T_opt
            h_T_point = np.where(below, r.h2 * (temp_arr - r.T_opt)**2, r.h4 * (temp_arr - r.T_opt)**2)
        elif name in ['Approach3J', 'Approach3P']:
            # Persistence decay
            h4 = r.h4
            A_T_lag, A_T2_lag = compute_persistence_accumulators(data, h4)
            T_loess_base = _get_T_loess_at_base_year(data, trends_loess, base_year=1961)
            correction_T, correction_T2 = compute_pre_first_year_correction(data, h4, T_loess_base)
            X1 = temp_arr - h4 * A_T_lag - correction_T
            X2 = temp_arr**2 - h4 * A_T2_lag - correction_T2
            h_T_point = r.h1 * X1 + r.h2 * X2
        else:
            continue

        # Format lines
        for i in range(n_obs):
            new_lines.append(f'-1,{name},{iso3_arr[i]},{year_arr[i]},{temp_arr[i]:.4f},{h_T_point[i]:.8f}\n')

    # Append to file
    print(f"\n  Appending {len(new_lines):,} rows to {h_values_path}...")
    with open(h_values_path, 'a') as f:
        f.writelines(new_lines)

    print("  Done!")
    return 0


if __name__ == "__main__":
    exit(main())
