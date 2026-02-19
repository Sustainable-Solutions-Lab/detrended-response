#!/usr/bin/env python3
"""Generate publication-quality tables and figures from analysis outputs.

This script creates publication-ready tables and figures by loading pre-computed
results from run_analysis.py and run_bootstrap.py output directories.

Usage:
    python scripts/make_tables_and_figures.py [--analysis-dir DIR] [--bootstrap-dir DIR] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_loader import load_data_from_csv
from src.publication import (
    generate_tables,
    generate_figures,
)


def find_most_recent_dir(pattern: str) -> str:
    """Find the most recent directory matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern to match directories (e.g., 'data/output/reference/analysis_*')

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


def load_analysis_results(analysis_dir: Path) -> dict:
    """Load pre-computed results from analysis output directory.

    Parameters
    ----------
    analysis_dir : Path
        Path to analysis output directory (from run_analysis.py)

    Returns
    -------
    dict
        Dictionary containing:
        - 'comparison_table': DataFrame with main results (h1, h2, T_opt, R-squared, etc.)
        - 'country_trends': DataFrame with country-level trend estimates
    """
    results = {}

    # Load comparison table
    comparison_path = analysis_dir / "comparison_table.csv"
    if comparison_path.exists():
        results['comparison_table'] = pd.read_csv(comparison_path, comment='#')
        print(f"      Loaded comparison_table.csv: {len(results['comparison_table'])} approaches")
    else:
        print(f"      WARNING: comparison_table.csv not found at {comparison_path}")
        results['comparison_table'] = None

    # Load country trends
    country_trends_path = analysis_dir / "country_trends.csv"
    if country_trends_path.exists():
        results['country_trends'] = pd.read_csv(country_trends_path, comment='#')
        print(f"      Loaded country_trends.csv: {len(results['country_trends'])} countries")
    else:
        print(f"      WARNING: country_trends.csv not found at {country_trends_path}")
        results['country_trends'] = None

    return results


def load_bootstrap_results(bootstrap_dir: Path) -> dict:
    """Load pre-computed results from bootstrap output directory.

    Parameters
    ----------
    bootstrap_dir : Path
        Path to bootstrap output directory (from run_bootstrap.py)

    Returns
    -------
    dict
        Dictionary containing:
        - 'bootstrap_coefficients': DataFrame with all bootstrap samples
        - 'bootstrap_summary': DataFrame with pre-computed percentiles (p5, p25, p50, p75, p95)
    """
    results = {}

    # Load bootstrap coefficients (all samples)
    coefficients_path = bootstrap_dir / "bootstrap_coefficients.csv"
    if coefficients_path.exists():
        results['bootstrap_coefficients'] = pd.read_csv(coefficients_path, comment='#')
        n_samples = len(results['bootstrap_coefficients'])
        n_approaches = results['bootstrap_coefficients']['approach'].nunique()
        print(f"      Loaded bootstrap_coefficients.csv: {n_samples} samples across {n_approaches} approaches")
    else:
        print(f"      WARNING: bootstrap_coefficients.csv not found at {coefficients_path}")
        results['bootstrap_coefficients'] = None

    # Load bootstrap summary table (percentiles)
    summary_path = bootstrap_dir / "bootstrap_summary_table.csv"
    if summary_path.exists():
        results['bootstrap_summary'] = pd.read_csv(summary_path, comment='#')
        print(f"      Loaded bootstrap_summary_table.csv: {len(results['bootstrap_summary'])} approaches")
    else:
        print(f"      WARNING: bootstrap_summary_table.csv not found at {summary_path}")
        results['bootstrap_summary'] = None

    # Load bootstrap var_attrib samples
    var_attrib_path = bootstrap_dir / "bootstrap_var_attrib_samples.csv"
    if var_attrib_path.exists():
        results['bootstrap_var_attrib'] = pd.read_csv(var_attrib_path, comment='#')
        n_samples = len(results['bootstrap_var_attrib'])
        n_approaches = results['bootstrap_var_attrib']['approach'].nunique()
        print(f"      Loaded bootstrap_var_attrib_samples.csv: {n_samples} samples across {n_approaches} approaches")
    else:
        print(f"      WARNING: bootstrap_var_attrib_samples.csv not found at {var_attrib_path}")
        results['bootstrap_var_attrib'] = None

    # Load bootstrap k_samples (year fixed effects)
    k_samples_path = bootstrap_dir / "bootstrap_k_samples.csv"
    if k_samples_path.exists():
        results['bootstrap_k_samples'] = pd.read_csv(k_samples_path, comment='#')
        n_samples = len(results['bootstrap_k_samples'])
        n_approaches = results['bootstrap_k_samples']['approach'].nunique()
        print(f"      Loaded bootstrap_k_samples.csv: {n_samples} samples across {n_approaches} approaches")
    else:
        print(f"      WARNING: bootstrap_k_samples.csv not found at {k_samples_path}")
        results['bootstrap_k_samples'] = None

    return results


def create_publication_output_dir(base_dir: str = None) -> Path:
    """Create timestamped output directory for publication outputs.

    Parameters
    ----------
    base_dir : str, optional
        Base directory path. If None, creates timestamped directory under data/output/

    Returns
    -------
    Path
        Path to created output directory
    """
    if base_dir:
        output_dir = Path(base_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/output") / f"publication_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality tables and figures from analysis outputs"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Parent directory containing analysis_* and bootstrap_* subdirectories (e.g., data/output/reference_mw10)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=None,
        help="Path to analysis output directory (default: most recent in reference-dir/analysis_* or data/output/reference/analysis_*)",
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=str,
        default=None,
        help="Path to bootstrap output directory (default: most recent in reference-dir/bootstrap_* or data/output/reference/bootstrap_*)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for publication files (default: auto-timestamped under data/output/)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Path to input data CSV for temperature histogram (default: data/input/Maddison_CRU_dataset.csv)",
    )

    args = parser.parse_args()

    # Resolve directories: explicit args > --reference-dir > default
    if args.analysis_dir is not None:
        analysis_dir = args.analysis_dir
    elif args.reference_dir is not None:
        analysis_dir = find_most_recent_dir(f"{args.reference_dir}/analysis_*")
    else:
        analysis_dir = find_most_recent_dir("data/output/reference/analysis_*")

    if args.bootstrap_dir is not None:
        bootstrap_dir = args.bootstrap_dir
    elif args.reference_dir is not None:
        bootstrap_dir = find_most_recent_dir(f"{args.reference_dir}/bootstrap_*")
    else:
        bootstrap_dir = find_most_recent_dir("data/output/reference/bootstrap_*")

    print("=" * 70)
    print("Publication Tables and Figures Generator")
    print("=" * 70)

    # Validate input directories
    print(f"\n[1/6] Validating input directories...")

    if analysis_dir is None:
        if args.reference_dir:
            print(f"ERROR: No analysis directory found matching {args.reference_dir}/analysis_*")
        else:
            print("ERROR: No analysis directory found matching data/output/reference/analysis_*")
        print("       Please specify --analysis-dir or --reference-dir explicitly")
        sys.exit(1)
    if bootstrap_dir is None:
        if args.reference_dir:
            print(f"ERROR: No bootstrap directory found matching {args.reference_dir}/bootstrap_*")
        else:
            print("ERROR: No bootstrap directory found matching data/output/reference/bootstrap_*")
        print("       Please specify --bootstrap-dir or --reference-dir explicitly")
        sys.exit(1)

    analysis_dir = Path(analysis_dir)
    bootstrap_dir = Path(bootstrap_dir)

    print(f"      Analysis dir: {analysis_dir}")
    print(f"      Bootstrap dir: {bootstrap_dir}")

    if not analysis_dir.exists():
        print(f"ERROR: Analysis directory does not exist: {analysis_dir}")
        sys.exit(1)
    if not bootstrap_dir.exists():
        print(f"ERROR: Bootstrap directory does not exist: {bootstrap_dir}")
        sys.exit(1)

    print("      Both directories exist.")

    # Load analysis results
    print(f"\n[2/6] Loading analysis results...")
    analysis_results = load_analysis_results(analysis_dir)

    # Load bootstrap results
    print(f"\n[3/6] Loading bootstrap results...")
    bootstrap_results = load_bootstrap_results(bootstrap_dir)

    # Load input data for temperature histogram
    print(f"\n[4/6] Loading input data for histogram...")
    data_path = Path(args.data_file)
    if data_path.exists():
        data = load_data_from_csv(str(data_path))
        print(f"      Loaded {data.n_obs} observations from {data_path.name}")
    else:
        print(f"      WARNING: Data file not found at {data_path}")
        print(f"      Temperature histogram will not be included")
        data = None

    # Create output directory
    print(f"\n[5/6] Creating output directory...")
    output_dir = create_publication_output_dir(args.output_dir)
    print(f"      Output directory: {output_dir}")

    # Generate tables and figures
    print(f"\n[6/6] Generating publication outputs...")

    generate_tables(
        analysis_results=analysis_results,
        bootstrap_results=bootstrap_results,
        output_dir=output_dir,
    )

    generate_figures(
        analysis_results=analysis_results,
        bootstrap_results=bootstrap_results,
        output_dir=output_dir,
        data=data,
    )

    print("\n" + "=" * 70)
    print(f"Publication outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
