#!/usr/bin/env python3
"""Analyze which countries systematically skew bootstrap coefficient estimates.

This script identifies countries that, when included more frequently in bootstrap
resamples, are associated with higher or lower coefficient estimates.

Usage:
    python scripts/run_influence_analysis.py [options]

Options:
    --bootstrap-dir DIR    Bootstrap output directory
                           (default: most recent data/output/reference/bootstrap_*)
    --output-dir DIR       Output directory
                           (default: data/output/influence_{timestamp})
    --approaches LIST      Approaches to analyze (default: all)
    --coefficients LIST    Coefficients to analyze (default: approach-specific)
    --percentiles LIST     Percentile thresholds (default: 5 25 75 95)
    --regression-type      "linear" or "logistic" (default: linear)
    --n-top N              Number of top/bottom countries to report (default: 10)
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.influence import (
    run_influence_analysis,
    save_influence_results,
    APPROACH_COEFFICIENTS,
)


def find_most_recent_dir(pattern: str) -> str:
    """Find the most recent directory matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern to match directories (e.g., 'data/output/reference/bootstrap_*')

    Returns
    -------
    str
        Path to the most recent matching directory, or None if no match found
    """
    matches = sorted(glob.glob(pattern))
    if matches:
        # Return the last one (most recent by timestamp in name)
        return matches[-1]
    return None


def parse_list_arg(value: str) -> list:
    """Parse a comma or space-separated list argument."""
    if not value:
        return []
    # Split on comma or whitespace
    items = value.replace(',', ' ').split()
    return [item.strip() for item in items if item.strip()]


def parse_int_list(value: str) -> tuple:
    """Parse a list of integers."""
    items = parse_list_arg(value)
    return tuple(int(x) for x in items)


def main():
    # Find most recent bootstrap directory for default
    default_bootstrap_dir = find_most_recent_dir("data/output/reference/bootstrap_*")

    parser = argparse.ArgumentParser(
        description="Analyze country influence on bootstrap coefficient estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (most recent bootstrap output)
    python scripts/run_influence_analysis.py

    # Analyze specific approaches
    python scripts/run_influence_analysis.py --approaches "approach5 method2"

    # Use custom percentiles
    python scripts/run_influence_analysis.py --percentiles "10 50 90"

        """,
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=str,
        default=default_bootstrap_dir,
        help="Path to bootstrap output directory (default: most recent in data/output/reference/bootstrap_*)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-timestamped under data/output/)",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=None,
        help="Space or comma-separated list of approaches to analyze (default: all)",
    )
    parser.add_argument(
        "--coefficients",
        type=str,
        default=None,
        help="Space or comma-separated list of coefficients to analyze (default: approach-specific)",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="5 25 75 95",
        help="Space or comma-separated list of percentile thresholds (default: 5 25 75 95)",
    )
    parser.add_argument(
        "--regression-type",
        type=str,
        choices=["linear"],
        default="linear",
        help="Regression type: 'linear' (Linear Probability Model) (default: linear)",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=10,
        help="Number of top/bottom countries to report in summary (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Country Influence Analysis")
    print("=" * 70)

    # Validate bootstrap directory
    print(f"\n[1/4] Validating input directory...")

    if args.bootstrap_dir is None:
        print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
        print("       Please specify --bootstrap-dir explicitly")
        sys.exit(1)

    bootstrap_dir = Path(args.bootstrap_dir)
    print(f"      Bootstrap dir: {bootstrap_dir}")

    if not bootstrap_dir.exists():
        print(f"ERROR: Bootstrap directory does not exist: {bootstrap_dir}")
        sys.exit(1)

    # Check required files
    coef_path = bootstrap_dir / "bootstrap_coefficients.csv"
    samples_path = bootstrap_dir / "bootstrap_country_samples.csv"

    if not coef_path.exists():
        print(f"ERROR: bootstrap_coefficients.csv not found in {bootstrap_dir}")
        sys.exit(1)
    if not samples_path.exists():
        print(f"ERROR: bootstrap_country_samples.csv not found in {bootstrap_dir}")
        sys.exit(1)

    print("      Required files found.")

    # Create output directory
    print(f"\n[2/4] Creating output directory...")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/output") / f"influence_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"      Output dir: {output_dir}")

    # Parse arguments
    approaches = parse_list_arg(args.approaches) if args.approaches else None
    percentiles = parse_int_list(args.percentiles)

    # Handle coefficients argument
    coefficients_dict = None
    if args.coefficients:
        coef_list = parse_list_arg(args.coefficients)
        # Apply same coefficients to all approaches
        coefficients_dict = {approach: coef_list for approach in APPROACH_COEFFICIENTS.keys()}

    print(f"\n[3/4] Running influence analysis...")
    print(f"      Approaches: {approaches if approaches else 'all'}")
    print(f"      Coefficients: {args.coefficients if args.coefficients else 'approach-specific defaults'}")
    print(f"      Percentiles: {percentiles}")
    print(f"      Regression: {args.regression_type}")

    # Run analysis
    results = run_influence_analysis(
        bootstrap_dir=bootstrap_dir,
        approaches=approaches,
        coefficients=coefficients_dict,
        percentiles=percentiles,
        regression_type=args.regression_type,
    )

    print(f"      Completed {len(results)} analyses")

    # Save results
    print(f"\n[4/4] Saving results...")
    save_influence_results(
        results=results,
        output_dir=output_dir,
        n_top=args.n_top,
    )

    print("\n" + "=" * 70)
    print(f"Influence analysis complete. Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
