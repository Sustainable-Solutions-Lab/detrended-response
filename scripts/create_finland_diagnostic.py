#!/usr/bin/env python3
"""Create diagnostic CSV for Finland's Approach3L cumulative effects calculation.

This script generates a CSV file showing all intermediate values for Finland's
Approach3L forward simulation (point estimate only), allowing step-by-step
verification that the calculation is correct.

Usage:
    python scripts/create_finland_diagnostic.py [--output OUTPUT_FILE]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data_from_csv
from src.detrending import (
    compute_country_trends,
    compute_year_means,
    compute_country_trends_with_k,
    compute_country_trends_loess,
)
from src.fitting import (
    fit_all_approaches,
    compute_persistence_accumulators,
    compute_pre_first_year_correction,
)


def get_finland_indices(data):
    """Get observation indices for Finland."""
    finland_idx = data.iso_to_idx.get('FIN')
    mask = data.country_idx == finland_idx
    return np.where(mask)[0]


def get_T_loess_at_base_year_for_country(data, trends_loess, country_idx, base_year=1961):
    """Get T_loess at base year for a specific country.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess object
        country_idx: Index of the country
        base_year: Base year (default: 1961)

    Returns:
        T_loess value at base year for the country
    """
    country_mask = data.country_idx == country_idx
    country_indices = np.where(country_mask)[0]
    years_for_country = data.year[country_indices].astype(int)

    # Find T_loess at base_year
    base_year_mask = years_for_country == base_year
    if base_year_mask.any():
        base_idx = country_indices[np.where(base_year_mask)[0][0]]
        return trends_loess.T_loess[base_idx]
    else:
        # If no observation at base year, use earliest year
        earliest_idx = country_indices[np.argmin(years_for_country)]
        return trends_loess.T_loess[earliest_idx]


def compute_finland_diagnostic(data, trends_loess, h1, h2, h4):
    """Compute all diagnostic values for Finland.

    Args:
        data: AnalysisData object
        trends_loess: CountryTrendsLoess with T_loess values
        h1, h2, h4: Approach4 coefficients

    Returns:
        DataFrame with diagnostic values for each year
    """
    finland_country_idx = data.iso_to_idx.get('FIN')
    finland_obs_indices = get_finland_indices(data)

    # Get T_loess at base year 1961
    T_loess_1961 = get_T_loess_at_base_year_for_country(
        data, trends_loess, finland_country_idx, base_year=1961
    )

    # Compute persistence accumulators for all observations
    A_T_lag_all, A_T2_lag_all = compute_persistence_accumulators(data, h4)

    # Compute pre-first-year corrections using T_loess at base year
    # Create an array with T_loess_1961 for all observations of this country
    T_loess_base_array = np.zeros(data.n_obs)
    T_loess_base_array[finland_obs_indices] = T_loess_1961
    # For the correction, we need to use T_loess_1961 (not actual T)
    correction_T_all, correction_T2_all = compute_pre_first_year_correction(
        data, h4, T_loess_base_array
    )

    # Sort Finland observations by year
    years = data.year[finland_obs_indices]
    sort_order = np.argsort(years)
    sorted_indices = finland_obs_indices[sort_order]

    rows = []
    h_T_delta_cum = 0.0

    for i, idx in enumerate(sorted_indices):
        year = int(data.year[idx])
        T_actual = data.temp[idx]
        T_loess = trends_loess.T_loess[idx]

        # Get accumulators and corrections for this observation
        A_T_lag = A_T_lag_all[idx]
        A_T2_lag = A_T2_lag_all[idx]
        correction_T = correction_T_all[idx]
        correction_T2 = correction_T2_all[idx]

        # Compute adjusted regressors (before detrending, for clarity)
        # X1 = T - h4*A_T_lag - correction_T
        # X2 = T^2 - h4*A_T2_lag - correction_T2
        X1 = T_actual - h4 * A_T_lag - correction_T
        X2 = T_actual**2 - h4 * A_T2_lag - correction_T2

        # Climate response: h_T = h1*X1 + h2*X2
        h_T = h1 * X1 + h2 * X2

        # Baseline for Approach3L:
        # If h4 > 0: baseline = 0 (constant temperature gives X1=X2=0 due to persistence)
        # If h4 = 0: baseline = h(T_loess_1961) = h1*T_loess_1961 + h2*T_loess_1961^2
        if h4 > 0:
            h_T_baseline = 0.0
        else:
            h_T_baseline = h1 * T_loess_1961 + h2 * T_loess_1961**2

        # Annual delta
        h_T_delta = h_T - h_T_baseline

        # Cumulative sum
        h_T_delta_cum += h_T_delta

        rows.append({
            'year': year,
            'T_actual': T_actual,
            'T_loess': T_loess,
            'T_loess_1961': T_loess_1961,
            'h1': h1,
            'h2': h2,
            'h4': h4,
            'A_T_lag': A_T_lag,
            'A_T2_lag': A_T2_lag,
            'correction_T': correction_T,
            'correction_T2': correction_T2,
            'X1': X1,
            'X2': X2,
            'h_T': h_T,
            'h_T_baseline': h_T_baseline,
            'h_T_delta': h_T_delta,
            'h_T_delta_cum': h_T_delta_cum,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create diagnostic CSV for Finland's Approach3L calculation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finland_Approach3L_diagnostic.csv",
        help="Output CSV file path (default: finland_Approach3L_diagnostic.csv)",
    )
    parser.add_argument(
        "--use-csv",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Input CSV file (default: data/input/Maddison_CRU_dataset.csv)",
    )
    parser.add_argument(
        "--loess-window",
        type=float,
        default=42.447947771790915,
        help="Window size in years for LOESS smoothing (default: 42.45)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Finland Approach4 Diagnostic")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    csv_path = Path(args.use_csv).expanduser()
    data = load_data_from_csv(str(csv_path))
    print(f"      Observations: {data.n_obs}")
    print(f"      Countries: {data.n_countries}")

    # Compute trends
    print("\n[2/4] Computing trends...")
    trends = compute_country_trends(data)
    year_means = compute_year_means(data)
    trends_with_k = compute_country_trends_with_k(data, year_means)
    trends_loess = compute_country_trends_loess(data, year_means, args.loess_window)
    print(f"      LOESS window: {args.loess_window:.1f} years")

    # Fit all approaches to get coefficients
    print("\n[3/4] Fitting Approach3L...")
    results = fit_all_approaches(
        data, trends,
        trends_with_k=trends_with_k,
        year_means=year_means,
        trends_loess=trends_loess
    )

    # Get Approach3L coefficients
    Approach3L_result = results['Approach3L']
    h1 = Approach3L_result.h1
    h2 = Approach3L_result.h2
    h4 = Approach3L_result.h4

    print(f"      h1 = {h1:.10f}")
    print(f"      h2 = {h2:.10f}")
    print(f"      h4 = {h4:.10f}")
    print(f"      T_opt = {Approach3L_result.T_opt:.4f}")

    # Compute diagnostic values for Finland
    print("\n[4/4] Computing Finland diagnostic values...")
    df = compute_finland_diagnostic(data, trends_loess, h1, h2, h4)

    # Save to CSV with full precision
    output_path = Path(args.output)
    df.to_csv(output_path, index=False, float_format='%.15g')
    print(f"      Saved: {output_path}")
    print(f"      Rows: {len(df)}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nFirst year (1961):")
    row_1961 = df[df['year'] == 1961].iloc[0]
    print(f"  T_actual      = {row_1961['T_actual']:.6f}")
    print(f"  T_loess       = {row_1961['T_loess']:.6f}")
    print(f"  T_loess_1961  = {row_1961['T_loess_1961']:.6f}")
    print(f"  A_T_lag       = {row_1961['A_T_lag']:.6f} (should be 0)")
    print(f"  correction_T  = {row_1961['correction_T']:.6f}")
    print(f"  X1            = {row_1961['X1']:.6f}")
    print(f"  h_T           = {row_1961['h_T']:.10f}")
    print(f"  h_T_baseline  = {row_1961['h_T_baseline']:.10f}")
    print(f"  h_T_delta     = {row_1961['h_T_delta']:.10f}")
    print(f"  h_T_delta_cum = {row_1961['h_T_delta_cum']:.10f}")

    print(f"\nLast year (2022):")
    row_2022 = df[df['year'] == 2022].iloc[0]
    print(f"  T_actual      = {row_2022['T_actual']:.6f}")
    print(f"  T_loess       = {row_2022['T_loess']:.6f}")
    print(f"  h_T           = {row_2022['h_T']:.10f}")
    print(f"  h_T_delta     = {row_2022['h_T_delta']:.10f}")
    print(f"  h_T_delta_cum = {row_2022['h_T_delta_cum']:.10f}")

    # Verification checks
    print("\n" + "=" * 70)
    print("Verification Checks")
    print("=" * 70)

    # Check 1: A_T_lag[1961] should be 0
    check1_pass = abs(row_1961['A_T_lag']) < 1e-10
    print(f"\n1. A_T_lag[1961] = 0: {'PASS' if check1_pass else 'FAIL'}")

    # Check 2: correction_T uses T_loess_1961
    # correction_T[1961] = (1-h4)^0 * T_loess_1961 = T_loess_1961
    expected_correction_1961 = row_1961['T_loess_1961']
    check2_pass = abs(row_1961['correction_T'] - expected_correction_1961) < 1e-10
    print(f"2. correction_T[1961] = T_loess_1961: {'PASS' if check2_pass else 'FAIL'}")
    if not check2_pass:
        print(f"   Expected: {expected_correction_1961:.10f}")
        print(f"   Got: {row_1961['correction_T']:.10f}")

    # Check 3: h_T_delta formula
    expected_delta = row_1961['h_T'] - row_1961['h_T_baseline']
    check3_pass = abs(row_1961['h_T_delta'] - expected_delta) < 1e-10
    print(f"3. h_T_delta = h_T - h_T_baseline: {'PASS' if check3_pass else 'FAIL'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
