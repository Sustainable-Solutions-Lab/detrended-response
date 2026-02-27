#!/usr/bin/env python3
"""Create Maddison_CRU_dataset.csv from Maddison and CRU data.

This script reads Maddison GDP/population data and CRU climate data,
performs gap-filling for missing GDP years, and outputs a merged dataset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import CRU_COUNTRY_OVERRIDES, map_cru_country_to_iso3


def load_maddison_gdp_wide(excel_path: str) -> pd.DataFrame:
    """Load Maddison GDP data from GDPpc sheet (wide format).

    The GDPpc sheet structure:
    - Row 0: Header label and country names
    - Row 1: 'Region' and region names
    - Row 2: 'year' and ISO codes
    - Rows 3+: Year values in column 0, GDP values in columns 1+

    Returns:
        DataFrame with columns: iso_id, year, gdppc
    """
    # Read without header to handle the complex structure
    df = pd.read_excel(excel_path, sheet_name='GDPpc', header=None)

    # Extract ISO codes from row 2 (columns 1+)
    iso_codes = df.iloc[2, 1:].tolist()

    # Extract data starting from row 3
    data_df = df.iloc[3:].copy()
    data_df.columns = ['year'] + iso_codes

    # Filter to numeric years only (skip any header rows mixed in)
    data_df = data_df[pd.to_numeric(data_df['year'], errors='coerce').notna()].copy()
    data_df['year'] = data_df['year'].astype(int)

    # Convert from wide to long format
    df_long = pd.melt(
        data_df,
        id_vars=['year'],
        var_name='iso_id',
        value_name='gdppc'
    )

    # Drop rows with missing GDP values
    df_long = df_long.dropna(subset=['gdppc'])

    # Convert GDP to float
    df_long['gdppc'] = df_long['gdppc'].astype(float)

    return df_long


def load_maddison_population_wide(excel_path: str) -> pd.DataFrame:
    """Load Maddison population data from Population sheet (wide format).

    Population values are in thousands, so multiply by 1000.

    Returns:
        DataFrame with columns: iso_id, year, pop
    """
    # Read without header to handle the complex structure
    df = pd.read_excel(excel_path, sheet_name='Population', header=None)

    # Extract ISO codes from row 2 (columns 1+)
    iso_codes = df.iloc[2, 1:].tolist()

    # Extract data starting from row 3
    data_df = df.iloc[3:].copy()
    data_df.columns = ['year'] + iso_codes

    # Filter to numeric years only
    data_df = data_df[pd.to_numeric(data_df['year'], errors='coerce').notna()].copy()
    data_df['year'] = data_df['year'].astype(int)

    # Convert from wide to long format
    df_long = pd.melt(
        data_df,
        id_vars=['year'],
        var_name='iso_id',
        value_name='pop'
    )

    # Drop rows with missing population values
    df_long = df_long.dropna(subset=['pop'])

    # Convert population to actual counts (from thousands) and to float
    df_long['pop'] = df_long['pop'].astype(float) * 1000

    return df_long


def infill_gdp_gaps(df: pd.DataFrame, max_gap: int = 4) -> pd.DataFrame:
    """Infill GDP gaps of at most max_gap missing years using constant growth rate.

    For a gap from year t1 to t2 where (t2 - t1 - 1) <= max_gap:
    1. Compute growth rate: r = (GDP_t2 / GDP_t1)^(1/(t2-t1))
    2. Fill each missing year y: GDP_y = GDP_t1 * r^(y-t1)

    Args:
        df: DataFrame with columns iso_id, year, gdppc
        max_gap: Maximum number of missing years to fill (default 4)

    Returns:
        DataFrame with gaps filled
    """
    filled_rows = []

    for iso_id, group in df.groupby('iso_id'):
        group = group.sort_values('year').copy()
        years = group['year'].values
        gdppc = group['gdppc'].values

        # Add original rows
        for _, row in group.iterrows():
            filled_rows.append(row.to_dict())

        # Find and fill gaps
        for i in range(len(years) - 1):
            t1 = years[i]
            t2 = years[i + 1]
            gap_size = t2 - t1 - 1

            if 0 < gap_size <= max_gap:
                gdp_t1 = gdppc[i]
                gdp_t2 = gdppc[i + 1]

                # Compute constant growth rate
                r = (gdp_t2 / gdp_t1) ** (1 / (t2 - t1))

                # Fill missing years
                for y in range(t1 + 1, t2):
                    gdp_y = gdp_t1 * (r ** (y - t1))
                    filled_rows.append({
                        'iso_id': iso_id,
                        'year': y,
                        'gdppc': gdp_y
                    })

    result = pd.DataFrame(filled_rows)
    result = result.sort_values(['iso_id', 'year']).reset_index(drop=True)
    return result


def load_cru_data(csv_path: str) -> pd.DataFrame:
    """Load CRU climate data and map country names to ISO3 codes.

    Applies log transformation to precipitation.
    Aggregates by iso_id and year (some countries like Hawaii map to USA).

    Returns:
        DataFrame with columns: iso_id, year, temp, precp
    """
    df = pd.read_csv(csv_path)

    # Map country names to ISO3 codes
    df['iso_id'] = df['Country'].apply(map_cru_country_to_iso3)

    # Drop rows where mapping failed
    df = df[df['iso_id'] != ''].copy()

    # Rename columns before aggregation
    df = df.rename(columns={'Year': 'year', 'Temperature': 'temp', 'Precipitation': 'precipitation'})

    # Aggregate by iso_id and year (some regions like Hawaii map to same country)
    # Take mean of temperature and precipitation
    df = df.groupby(['iso_id', 'year']).agg({
        'temp': 'mean',
        'precipitation': 'mean'
    }).reset_index()

    # Log-transform precipitation relative to country mean
    # precp = log(P_t) - log(mean(P)) = log(P_t / mean(P))
    df['precp'] = np.log(df['precipitation']) - np.log(df.groupby('iso_id')['precipitation'].transform('mean'))

    # Drop the raw precipitation column
    df = df.drop(columns=['precipitation'])

    return df


def compute_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GDP growth rate as log difference.

    growth_pcGDP = log(GDP_t) - log(GDP_{t-1})

    First observation per country is dropped (no lagged value).
    """
    df = df.sort_values(['iso_id', 'year']).copy()

    # Compute log GDP
    df['log_gdppc'] = np.log(df['pcGDP'])

    # Compute growth as first difference of log GDP within each country
    df['growth_pcGDP'] = df.groupby('iso_id')['log_gdppc'].diff()

    # Drop first observation per country (where growth is NaN)
    df = df.dropna(subset=['growth_pcGDP'])

    # Drop temporary column
    df = df.drop(columns=['log_gdppc'])

    return df


def permute_climate_countries(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """Randomly permute country assignments for climate data.

    Creates a random one-to-one mapping between countries, so country A's
    temperature time series is assigned to country B, etc.

    Args:
        df: DataFrame with columns iso_id, year, temp, precp
        seed: Random seed for reproducibility

    Returns:
        DataFrame with iso_id values permuted
    """
    rng = np.random.default_rng(seed)

    # Get unique countries
    countries = df['iso_id'].unique()

    # Create random permutation
    permuted_countries = rng.permutation(countries)

    # Create mapping from original to permuted
    country_map = dict(zip(countries, permuted_countries))

    # Apply mapping
    df = df.copy()
    df['iso_id'] = df['iso_id'].map(country_map)

    return df


def compute_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-related variables.

    time = year - 1960
    time2 = time^2
    """
    df = df.copy()
    df['time'] = df['year'] - 1960
    df['time2'] = df['time'] ** 2
    return df


def validate_output(df: pd.DataFrame, reference_path: str = None) -> bool:
    """Validate the output DataFrame.

    Returns True if validation passes, False otherwise.
    """
    required_columns = ['iso_id', 'year', 'pcGDP', 'growth_pcGDP', 'temp', 'precp', 'time', 'time2', 'Pop']

    print("\n=== Validation Results ===")

    # Check all required columns present
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"FAIL: Missing columns: {missing_cols}")
        return False
    print("PASS: All required columns present")

    # Check year range
    year_min, year_max = df['year'].min(), df['year'].max()
    print(f"Year range: {year_min} - {year_max}")
    if year_min < 1961 or year_max > 2022:
        print(f"WARNING: Year range outside expected 1961-2022")

    # Verify time = year - 1960
    time_check = (df['time'] == df['year'] - 1960).all()
    if not time_check:
        print("FAIL: time != year - 1960")
        return False
    print("PASS: time = year - 1960")

    # Verify time2 = time^2
    time2_check = (df['time2'] == df['time'] ** 2).all()
    if not time2_check:
        print("FAIL: time2 != time^2")
        return False
    print("PASS: time2 = time^2")

    # Check for NaN values in key columns
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"WARNING: NaN values found:\n{nan_counts[nan_counts > 0]}")
    else:
        print("PASS: No NaN values in key columns")

    # Sample check that growth rate = log difference
    sample = df.groupby('iso_id').apply(lambda g: g.sort_values('year').head(3), include_groups=False).reset_index()
    if len(sample) > 0:
        sample_country = sample['iso_id'].iloc[0]
        country_data = df[df['iso_id'] == sample_country].sort_values('year')
        if len(country_data) >= 2:
            idx = country_data.index[1]
            expected_growth = np.log(country_data['pcGDP'].iloc[1]) - np.log(country_data['pcGDP'].iloc[0])
            actual_growth = country_data['growth_pcGDP'].iloc[1]
            if np.isclose(expected_growth, actual_growth, rtol=1e-6):
                print(f"PASS: Growth rate sample check for {sample_country}")
            else:
                print(f"WARNING: Growth rate mismatch for {sample_country}: expected {expected_growth:.6f}, got {actual_growth:.6f}")

    # Count countries
    n_countries = df['iso_id'].nunique()
    print(f"Country count: {n_countries}")
    if n_countries < 150:
        print(f"WARNING: Expected ~170 countries, found {n_countries}")

    # Total observations
    print(f"Total observations: {len(df)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create df_base_withPop.csv from Maddison and CRU data'
    )
    parser.add_argument(
        '--maddison',
        default='data/input/mpd2023_web.xlsx',
        help='Path to Maddison Excel file (default: data/input/mpd2023_web.xlsx)'
    )
    parser.add_argument(
        '--cru',
        default='data/input/cru_climate_data.csv',
        help='Path to CRU CSV file (default: data/input/cru_climate_data.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/input/Maddison_CRU_dataset.csv',
        help='Output CSV path (default: data/input/Maddison_CRU_dataset.csv)'
    )
    parser.add_argument(
        '--max-gap',
        type=int,
        default=4,
        help='Maximum GDP gap size to fill (default: 4)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation checks on output'
    )
    parser.add_argument(
        '--randomT',
        action='store_true',
        help='Randomly permute temperature data across countries (for synthetic null dataset)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for temperature permutation (default: random)'
    )

    args = parser.parse_args()

    # Use different default output filename for randomT
    if args.randomT and args.output == 'data/input/Maddison_CRU_dataset.csv':
        args.output = 'data/input/Maddison_CRU_dataset_randomT.csv'
        print(f"Using randomT default output: {args.output}")

    print("Loading Maddison GDP data from GDPpc sheet...")
    df_gdp = load_maddison_gdp_wide(args.maddison)
    print(f"  Loaded {len(df_gdp)} GDP observations for {df_gdp['iso_id'].nunique()} countries")

    print(f"Filling GDP gaps (max {args.max_gap} missing years)...")
    df_gdp = infill_gdp_gaps(df_gdp, max_gap=args.max_gap)
    print(f"  After gap filling: {len(df_gdp)} observations")

    print("Loading Maddison population data...")
    df_pop = load_maddison_population_wide(args.maddison)
    print(f"  Loaded {len(df_pop)} population observations")

    print("Loading CRU climate data...")
    df_cru = load_cru_data(args.cru)
    print(f"  Loaded {len(df_cru)} climate observations for {df_cru['iso_id'].nunique()} countries")

    if args.randomT:
        print(f"Permuting temperature data across countries (seed={args.seed})...")
        df_cru = permute_climate_countries(df_cru, seed=args.seed)
        print("  Temperature and precipitation data now randomly assigned to different countries")

    print("Merging GDP and population data...")
    df = pd.merge(df_gdp, df_pop, on=['iso_id', 'year'], how='inner')
    df = df.rename(columns={'gdppc': 'pcGDP', 'pop': 'Pop'})
    print(f"  After GDP+Pop merge: {len(df)} observations")

    print("Computing GDP growth rate...")
    df = compute_growth_rate(df)
    print(f"  After growth calculation: {len(df)} observations")

    print("Merging with CRU climate data...")
    df = pd.merge(df, df_cru, on=['iso_id', 'year'], how='inner')
    print(f"  After climate merge: {len(df)} observations for {df['iso_id'].nunique()} countries")

    print("Computing time variables...")
    df = compute_time_variables(df)

    # Filter to years 1961-2022 (as specified in output format)
    df = df[(df['year'] >= 1961) & (df['year'] <= 2022)].copy()
    print(f"  After year filter (1961-2022): {len(df)} observations for {df['iso_id'].nunique()} countries")

    # Select and order columns
    output_columns = ['iso_id', 'year', 'pcGDP', 'growth_pcGDP', 'temp', 'precp', 'time', 'time2', 'Pop']
    df = df[output_columns]

    # Sort by iso_id and year
    df = df.sort_values(['iso_id', 'year']).reset_index(drop=True)

    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=True)
    print(f"  Saved {len(df)} observations")

    if args.validate:
        validate_output(df)

    print("\nDone!")


if __name__ == '__main__':
    main()
