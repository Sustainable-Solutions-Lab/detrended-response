#!/usr/bin/env python3
"""Compare bootstrap uncertainty widths across trend identification methods.

For each (response type, coefficient) cell in the 3x3 grid, tests whether
methods P or L produce significantly different uncertainty widths compared
to method J, using bootstrap-of-bootstrap resampling.

Usage:
    python scripts/compare_bootstrap_widths.py
    python scripts/compare_bootstrap_widths.py --bootstrap-dir data/output/pipeline_years-countries_2026-03-03/bootstrap
    python scripts/compare_bootstrap_widths.py --n-resample 5000
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_most_recent_dir(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def compute_width(samples, range_type):
    """Compute IQR or 90% CI width from bootstrap samples."""
    if range_type == 'iqr':
        return np.percentile(samples, 75) - np.percentile(samples, 25)
    return np.percentile(samples, 95) - np.percentile(samples, 5)


def compare_widths(samples_x, samples_y, range_type, n_resample, rng):
    """Bootstrap-of-bootstrap test for width difference.

    Returns observed difference, 95% CI of difference, and p-value.
    Negative difference means X is narrower than Y.
    """
    n_x = len(samples_x)
    n_y = len(samples_y)
    observed_width_x = compute_width(samples_x, range_type)
    observed_width_y = compute_width(samples_y, range_type)
    observed_diff = observed_width_x - observed_width_y

    diffs = np.empty(n_resample)
    for k in range(n_resample):
        resampled_x = rng.choice(samples_x, size=n_x, replace=True)
        resampled_y = rng.choice(samples_y, size=n_y, replace=True)
        diffs[k] = compute_width(resampled_x, range_type) - compute_width(resampled_y, range_type)

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    # One-sided p-values
    p_alt_wider = np.mean(diffs <= 0)   # small => evidence alt is wider than J
    p_alt_narrower = np.mean(diffs >= 0)  # small => evidence alt is narrower than J

    return observed_width_x, observed_width_y, observed_diff, ci_low, ci_high, p_alt_wider, p_alt_narrower


def compute_width_comparisons(bootstrap_coeff_df, n_resample=10000, seed=42):
    """Compute bootstrap-of-bootstrap width comparisons for P and L vs J.

    Parameters
    ----------
    bootstrap_coeff_df : pd.DataFrame
        Bootstrap coefficients, already filtered to iteration >= 0.
    n_resample : int
        Number of meta-bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Comparison results with columns: response, coefficient, comparison,
        range_type, width_alt, width_J, diff, ci_low, ci_high, p_alt_wider,
        p_alt_narrower.
    """
    rng = np.random.default_rng(seed)

    comparisons = [
        ('Q', 'h2'), ('Q', 'T_opt'),
        ('P', 'h2'), ('P', 'T_opt'), ('P', 'h4'),
        ('D', 'h2'), ('D', 'T_opt'), ('D', 'h4'),
    ]

    results = []

    for response_type, coeff in comparisons:
        j_approach = f'Approach {response_type}J'
        j_samples = bootstrap_coeff_df.loc[bootstrap_coeff_df['approach'] == j_approach, coeff].dropna().values

        if len(j_samples) == 0:
            continue

        for alt_method in ['P', 'L']:
            alt_approach = f'Approach {response_type}{alt_method}'
            alt_samples = bootstrap_coeff_df.loc[bootstrap_coeff_df['approach'] == alt_approach, coeff].dropna().values

            if len(alt_samples) == 0:
                continue

            for range_type in ['iqr', '90ci']:
                width_alt, width_j, diff, ci_low, ci_high, p_alt_wider, p_alt_narrower = compare_widths(
                    alt_samples, j_samples, range_type, n_resample, rng
                )
                results.append({
                    'response': response_type,
                    'coefficient': coeff,
                    'comparison': f'{alt_method} vs J',
                    'range_type': range_type,
                    'width_alt': width_alt,
                    'width_J': width_j,
                    'diff': diff,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_alt_wider': p_alt_wider,
                    'p_alt_narrower': p_alt_narrower,
                })

    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare bootstrap uncertainty widths across trend methods"
    )
    parser.add_argument(
        "--bootstrap-dir", type=str, default=None,
        help="Path to bootstrap output directory",
    )
    parser.add_argument(
        "--n-resample", type=int, default=10000,
        help="Number of meta-bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for CSV (default: same as bootstrap directory)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args(argv)

    # Find bootstrap directory
    if args.bootstrap_dir:
        bootstrap_dir = Path(args.bootstrap_dir)
    else:
        bootstrap_dir_str = find_most_recent_dir("data/output/reference/bootstrap_*")
        if bootstrap_dir_str is None:
            print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
            sys.exit(1)
        bootstrap_dir = Path(bootstrap_dir_str)

    coeff_path = bootstrap_dir / "bootstrap_coefficients.csv"
    if not coeff_path.exists():
        print(f"ERROR: bootstrap_coefficients.csv not found at {coeff_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else bootstrap_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Bootstrap Width Comparison Statistical Test")
    print("=" * 90)
    print(f"Bootstrap dir: {bootstrap_dir}")
    print(f"Meta-bootstrap resamples: {args.n_resample}")

    # Load data and filter to bootstrap samples only (iteration >= 0)
    df = pd.read_csv(coeff_path, comment='#')
    df = df[df['iteration'] >= 0]

    results_df = compute_width_comparisons(df, args.n_resample, args.seed)

    # Print formatted table
    print(f"\n{'Resp':>4s} {'Coeff':>6s} {'Comp':>7s} {'Range':>5s}  "
          f"{'W_alt':>8s} {'W_J':>8s} {'Diff':>8s} {'CI_low':>8s} {'CI_high':>8s} {'p(a>J)':>8s} {'p(a<J)':>8s}")
    print("-" * 100)
    for _, r in results_df.iterrows():
        print(f"{r['response']:>4s} {r['coefficient']:>6s} {r['comparison']:>7s} {r['range_type']:>5s}  "
              f"{r['width_alt']:>8.5f} {r['width_J']:>8.5f} {r['diff']:>8.5f} "
              f"{r['ci_low']:>8.5f} {r['ci_high']:>8.5f} {r['p_alt_wider']:>8.4f} {r['p_alt_narrower']:>8.4f}")

    # Save CSV
    csv_path = output_dir / "bootstrap_width_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
