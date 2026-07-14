"""Pipeline orchestrator for the detrended response analysis.

Runs analysis steps in sequence, automatically passing output directories
between dependent steps. Each step can also be run standalone via its own script.

Usage:
    python scripts/main.py                          # Full pipeline, 1000 bootstrap iterations
    python scripts/main.py --n-bootstrap 5          # Quick test run
    python scripts/main.py --pipeline-dir data/output/pipeline_years-countries_2026-03-03
        # Skip analysis+bootstrap, rerun steps 3-7 from existing pipeline
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_analysis import main as run_analysis_main
from run_bootstrap import main as run_bootstrap_main
from make_tables_and_figures import main as run_publication_main
from summarize_bootstrap_uncertainty import main as run_uncertainty_bars_main
from compare_Approach1J_Approach1P import main as run_comparison_main
from calculate_cumulative_effects import main as run_cumulative_main
from run_influence_analysis import main as run_influence_main


def main():
    parser = argparse.ArgumentParser(
        description="Run the full detrended response analysis pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent output directory (default: data/output/pipeline_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--use-csv",
        type=str,
        default="data/input/Maddison_CRU_dataset.csv",
        help="Input CSV file (default: data/input/Maddison_CRU_dataset.csv)",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Minimum year to include",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Maximum year to include",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Exclude years before this from analysis (applied after growth computation)",
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
        help="Mean weighting distance in years for LOESS. Window = 44/7 * this value.",
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
        "--sample-countries-only",
        action="store_true",
        help="Only sample countries (skip year resampling in bootstrap)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages in bootstrap",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=None,
        help="Two-letter approach codes to fit (e.g., QJ PL DJ SJ TJ). "
             "First letter: N/Q/P/S/T/D/L/G/C/I (response type; "
             "G/C = power-law GDP-scaled variants; I = log-linear GDP-dependent). "
             "Second letter: J/P/L (trend method). "
             "Default: fit all approaches.",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default=None,
        help="Existing pipeline directory with analysis/ and bootstrap/ subdirs. "
             "Skips steps 1-2 and reruns steps 3-7 from existing results.",
    )

    args = parser.parse_args()

    t_pipeline_start = time.perf_counter()

    if args.pipeline_dir:
        # Reuse existing analysis and bootstrap results
        pipeline_dir = Path(args.pipeline_dir)
        analysis_dir = pipeline_dir / "analysis"
        bootstrap_dir = pipeline_dir / "bootstrap"

        if not analysis_dir.exists():
            print(f"ERROR: analysis directory not found: {analysis_dir}")
            sys.exit(1)
        if not bootstrap_dir.exists():
            print(f"ERROR: bootstrap directory not found: {bootstrap_dir}")
            sys.exit(1)

        print("=" * 70)
        print("DETRENDED RESPONSE ANALYSIS PIPELINE (post-processing only)")
        print("=" * 70)
        print(f"Pipeline directory: {pipeline_dir}")
        print(f"Skipping steps 1-2 (using existing analysis and bootstrap)")
        print()
    else:
        # Full pipeline: create output directory and run analysis + bootstrap
        if args.output_dir:
            pipeline_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_dir = Path("data/output") / f"pipeline_{timestamp}"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("DETRENDED RESPONSE ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Output directory: {pipeline_dir}")
        print(f"Bootstrap iterations: {args.n_bootstrap}")
        print()

        # Build shared args for analysis and bootstrap
        shared_args = ["--use-csv", args.use_csv, "--loess-window", str(args.loess_window)]
        if args.year_min is not None:
            shared_args += ["--year-min", str(args.year_min)]
        if args.year_max is not None:
            shared_args += ["--year-max", str(args.year_max)]
        if args.mean_weight_distance is not None:
            shared_args += ["--mean-weight-distance", str(args.mean_weight_distance)]
        if args.start_year is not None:
            shared_args += ["--start-year", str(args.start_year)]
        if args.approaches is not None:
            shared_args += ["--approaches"] + args.approaches

        # Step 1: Analysis
        print("\n" + "#" * 70)
        print("# STEP 1/7: Analysis")
        print("#" * 70)
        analysis_dir = run_analysis_main([
            "--output-dir", str(pipeline_dir / "analysis"),
            *shared_args,
        ])

        # Step 2: Bootstrap
        print("\n" + "#" * 70)
        print("# STEP 2/7: Bootstrap")
        print("#" * 70)
        bootstrap_args = [
            "--output-dir", str(pipeline_dir / "bootstrap"),
            "--n-bootstrap", str(args.n_bootstrap),
            "--random-seed", str(args.random_seed),
            *shared_args,
        ]
        if args.sample_countries_only:
            bootstrap_args.append("--sample-countries-only")
        if args.quiet:
            bootstrap_args.append("--quiet")
        bootstrap_dir = run_bootstrap_main(bootstrap_args)

    # Step 3: Publication tables and figures
    print("\n" + "#" * 70)
    print("# STEP 3/7: Publication tables and figures")
    print("#" * 70)
    run_publication_main([
        "--analysis-dir", str(analysis_dir),
        "--bootstrap-dir", str(bootstrap_dir),
        "--output-dir", str(pipeline_dir / "publication"),
        "--data-file", args.use_csv,
    ])

    # Step 4: Uncertainty bar plots with width comparisons
    print("\n" + "#" * 70)
    print("# STEP 4/7: Uncertainty bar plots with width comparisons")
    print("#" * 70)
    run_uncertainty_bars_main([
        "--bootstrap-dir", str(bootstrap_dir),
        "--output-dir", str(pipeline_dir / "publication"),
    ])

    # Step 5: QJ vs QP comparison
    print("\n" + "#" * 70)
    print("# STEP 5/7: Approach QJ vs QP comparison")
    print("#" * 70)
    run_comparison_main([
        "--data-file", args.use_csv,
        "--output-dir", str(pipeline_dir / "comparison"),
    ])

    # Step 6: Cumulative effects
    print("\n" + "#" * 70)
    print("# STEP 6/7: Cumulative effects")
    print("#" * 70)
    run_cumulative_main([
        "--input-dir", str(bootstrap_dir),
        "--output-dir", str(pipeline_dir / "cumulative"),
    ])

    # Step 7: Influence analysis
    print("\n" + "#" * 70)
    print("# STEP 7/7: Influence analysis")
    print("#" * 70)
    run_influence_main([
        "--bootstrap-dir", str(bootstrap_dir),
        "--output-dir", str(pipeline_dir / "influence"),
    ])

    total_time = time.perf_counter() - t_pipeline_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_time:.1f}s")
    print(f"All outputs in: {pipeline_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
