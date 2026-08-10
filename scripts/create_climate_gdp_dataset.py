#!/usr/bin/env python3
"""Create a merged climate-GDP dataset (Maddison_CRU or WorldBank_CRU).

Reads GDP + population data from a chosen source (Maddison Project or World Bank),
merges with CRU climate data, gap-fills missing GDP years, optionally filters
countries, and writes a merged panel.

The retained year window is set by --year-min/--year-max (default 1960-2022).

GDP source selected with --gdp-source {maddison,worldbank}:
  maddison  -> Maddison GDPpc + Population sheets (per-capita GDP directly); GDP reaches far
               enough back that the panel can start at 1960 or earlier.
  worldbank -> World Bank wide CSVs (total GDP NY.GDP.MKTP.PP.KD / population
               SP.POP.TOTL); pcGDP = GDP / Pop; PPP data begins ~1990.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import CRU_COUNTRY_OVERRIDES, map_cru_country_to_iso3

# World Bank real-country codes that pycountry does not recognize (mirrors CRU_COUNTRY_OVERRIDES).
WB_COUNTRY_OVERRIDES = {'XKX'}  # Kosovo


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


def _is_real_country(iso3) -> bool:
    """True for genuine ISO3 country codes; False for WB aggregates (WLD/HIC/EUU/…) and junk."""
    if not isinstance(iso3, str) or iso3 == '':
        return False
    if iso3 in WB_COUNTRY_OVERRIDES:
        return True
    try:
        pycountry.countries.lookup(iso3)
        return True
    except LookupError:
        return False


def _load_worldbank_wide(csv_path: str, value_name: str) -> pd.DataFrame:
    """Melt a World Bank DataBank wide CSV to long (iso_id, year, <value_name>).

    Wide columns are 'Series Name, Series Code, Country Name, Country Code, "1960 [YR1960]", …';
    missing values are the string '..'. Aggregate rows (World, income groups, regions) and the
    trailing junk rows are dropped by keeping only real ISO3 country codes.
    """
    df = pd.read_csv(csv_path)
    year_cols = [c for c in df.columns if c[:4].isdigit()]
    long = df.melt(id_vars=['Country Code'], value_vars=year_cols,
                   var_name='year_label', value_name=value_name)
    long['iso_id'] = long['Country Code']
    long['year'] = long['year_label'].str.slice(0, 4).astype(int)
    long[value_name] = pd.to_numeric(long[value_name].replace('..', np.nan), errors='coerce')
    long = long.dropna(subset=[value_name])
    long = long[long['iso_id'].map(_is_real_country)]
    return long[['iso_id', 'year', value_name]].reset_index(drop=True)


def load_worldbank_gdp_wide(csv_path: str) -> pd.DataFrame:
    """World Bank total GDP (PPP, constant int'l $) → columns iso_id, year, gdppc.

    The 'gdppc' column carries TOTAL GDP here (per-capita is formed later via divide-by-pop),
    matching the intermediate schema of the Maddison GDP loader.
    """
    return _load_worldbank_wide(csv_path, 'gdppc')


def load_worldbank_pop_wide(csv_path: str) -> pd.DataFrame:
    """World Bank population (SP.POP.TOTL) → columns iso_id, year, pop (head counts, no *1000)."""
    return _load_worldbank_wide(csv_path, 'pop')


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


def _longest_contiguous_mask(years: np.ndarray) -> np.ndarray:
    """Boolean mask over one country's sorted years selecting the longest run of consecutive
    years (ties broken toward the earliest run)."""
    breaks = np.diff(years) != 1                       # gap boundary between adjacent years
    group = np.concatenate(([0], np.cumsum(breaks)))   # run label per year
    _, inverse, lengths = np.unique(group, return_inverse=True, return_counts=True)
    run_len = lengths[inverse]
    best_group = group[run_len == run_len.max()].min()  # earliest run among longest
    return group == best_group


def _filter_contiguous(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each country's longest contiguous stretch of years (run on infilled data)."""
    df = df.sort_values(['iso_id', 'year'])
    keep = df.groupby('iso_id')['year'].transform(
        lambda s: _longest_contiguous_mask(s.to_numpy()))
    return df[keep.to_numpy()].reset_index(drop=True)


def _filter_nearly_all(df: pd.DataFrame, min_years: int, min_frac: float) -> pd.DataFrame:
    """Keep only countries with data in at least `min_years` years (or `min_frac` of the span)."""
    threshold = (int(np.ceil(min_frac * (df['year'].max() - df['year'].min() + 1)))
                 if min_frac is not None else min_years)
    counts = df.groupby('iso_id')['year'].transform('count')
    return df[counts >= threshold].reset_index(drop=True)


COUNTRY_FILTERS = {
    'none':       lambda df, **k: df,
    'nearly-all': lambda df, **k: _filter_nearly_all(df, k['min_years'], k['min_frac']),
    'contiguous': lambda df, **k: _filter_contiguous(df),
}


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

    # Log-transform precipitation (raw levels, no country-mean normalization)
    # precp = log(P_t)  — retains cross-country level differences
    df['precp'] = np.log(df['precipitation'])

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


def validate_output(df: pd.DataFrame, reference_path: str = None,
                    expected_year_min: int = 1960, expected_year_max: int = 2022) -> bool:
    """Validate the output DataFrame.

    Returns True if validation passes, False otherwise.
    """
    required_columns = ['iso_id', 'year', 'pcGDP', 'growth_pcGDP', 'temp', 'precp', 'Pop']

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
    if year_min < expected_year_min or year_max > expected_year_max:
        print(f"WARNING: Year range outside expected {expected_year_min}-{expected_year_max}")

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


# Per-source defaults: output path and randomT output path.
SOURCE_DEFAULTS = {
    'maddison':  {'output': 'data/input/Maddison_CRU_dataset.csv',
                  'randomT': 'data/input/Maddison_CRU_dataset_randomT.csv'},
    'worldbank': {'output': 'data/input/WorldBank_CRU_dataset.csv',
                  'randomT': 'data/input/WorldBank_CRU_dataset_randomT.csv'},
}


def _wb_series_code(csv_path: str) -> str:
    """The (single) World Bank series code in a DataBank wide CSV, e.g. NY.GDP.PCAP.KD."""
    return pd.read_csv(csv_path, usecols=['Series Code'])['Series Code'].dropna().iloc[0]


def _resolve_wb_files(args) -> tuple:
    """Resolve (gdp_path, pop_path) for the World Bank source. Auto-detects among
    data/input/WB_*_Data.csv by series code (NY.GDP.* = GDP, SP.POP.* = population),
    so the exact download filename does not matter; overridable via --wb-gdp/--wb-pop."""
    gdp_path, pop_path = args.wb_gdp, args.wb_pop
    if gdp_path is None or pop_path is None:
        gdp_files, pop_files = [], []
        for c in sorted(glob.glob('data/input/WB_*_Data.csv')):
            code = _wb_series_code(c)
            (gdp_files if code.startswith('NY.GDP') else
             pop_files if code.startswith('SP.POP') else []).append(c)
        if gdp_path is None:
            if len(gdp_files) != 1:
                raise FileNotFoundError(f"Expected exactly one WB GDP (NY.GDP.*) file, found {gdp_files}")
            gdp_path = gdp_files[0]
        if pop_path is None:
            if len(pop_files) != 1:
                raise FileNotFoundError(f"Expected exactly one WB population (SP.POP.*) file, found {pop_files}")
            pop_path = pop_files[0]
    return gdp_path, pop_path


def load_gdp_pop(gdp_source: str, args) -> tuple:
    """Dispatch GDP + population loading by source. Returns (df_gdp, df_pop, divide_by_pop) with
    the shared intermediate schema (iso_id, year, gdppc) and (iso_id, year, pop). divide_by_pop is
    True only for World Bank *total*-GDP series (NY.GDP.MKTP.*); per-capita series (NY.GDP.PCAP.*)
    and Maddison are already per-capita."""
    if gdp_source == 'maddison':
        return (load_maddison_gdp_wide(args.maddison),
                load_maddison_population_wide(args.maddison), False)
    wb_gdp, wb_pop = _resolve_wb_files(args)
    code = _wb_series_code(wb_gdp)
    divide_by_pop = 'MKTP' in code   # total GDP -> per-capita; PCAP series already per-capita
    print(f"  WB GDP file: {wb_gdp}  (series {code}, {'total -> /pop' if divide_by_pop else 'per-capita'})")
    print(f"  WB Pop file: {wb_pop}")
    return load_worldbank_gdp_wide(wb_gdp), load_worldbank_pop_wide(wb_pop), divide_by_pop


def main():
    parser = argparse.ArgumentParser(
        description='Create a merged climate-GDP dataset (Maddison_CRU or WorldBank_CRU)'
    )
    parser.add_argument(
        '--gdp-source', choices=['maddison', 'worldbank'], default='maddison',
        help='GDP/population source (default: maddison)'
    )
    parser.add_argument(
        '--maddison',
        default='data/input/mpd2023_web.xlsx',
        help='Path to Maddison Excel file (default: data/input/mpd2023_web.xlsx)'
    )
    parser.add_argument(
        '--wb-gdp', default=None,
        help='Path to World Bank GDP wide CSV (default: auto-detect the NY.GDP.* file in '
             'data/input/WB_*_Data.csv). Total-GDP (MKTP) series are divided by population; '
             'per-capita (PCAP) series are used as-is.'
    )
    parser.add_argument(
        '--wb-pop', default=None,
        help='Path to World Bank population wide CSV (default: auto-detect the SP.POP.* file)'
    )
    parser.add_argument(
        '--cru',
        default='data/input/cru_climate_data.csv',
        help='Path to CRU CSV file (default: data/input/cru_climate_data.csv)'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output CSV path (default: per source, e.g. data/input/Maddison_CRU_dataset.csv)'
    )
    parser.add_argument(
        '--year-min', type=int, default=1960,
        help='First year retained in the panel (default: 1960). Growth for year y needs GDP at y-1, '
             'so Maddison supports 1960 (and earlier) while World Bank NY.GDP.PCAP.KD starts at 1960 '
             'and therefore has no growth before 1961.'
    )
    parser.add_argument(
        '--year-max', type=int, default=2022,
        help='Last year retained in the panel (default: 2022)'
    )
    parser.add_argument(
        '--country-filter', choices=['none', 'nearly-all', 'contiguous', 'endpoints'], default='none',
        help="Country retention: none (unbalanced, keep all), nearly-all (>= --min-years/--min-frac), "
             "contiguous (longest consecutive-year stretch per country), or endpoints (present in both "
             "the first and last panel year, i.e. the balanced-panel rule). Default: none"
    )
    parser.add_argument(
        '--min-years', type=int, default=30,
        help='nearly-all filter: minimum years of data per country (default: 30)'
    )
    parser.add_argument(
        '--min-frac', type=float, default=None,
        help='nearly-all filter: minimum fraction of the year span per country (overrides --min-years)'
    )
    parser.add_argument(
        '--exclude-iso', nargs='*', default=[],
        help='ISO3 codes to exclude from the panel. Default: exclude none (dependent territories '
             'such as BMU/GRL/HKG are kept). Example: --exclude-iso BMU GRL HKG'
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

    defaults = SOURCE_DEFAULTS[args.gdp_source]

    # Resolve output path from source default, then apply the randomT rename if the user did not
    # override --output.
    if args.output is None:
        args.output = defaults['output']
    if args.randomT and args.output == defaults['output']:
        args.output = defaults['randomT']
        print(f"Using randomT default output: {args.output}")

    print(f"Loading {args.gdp_source} GDP + population data...")
    df_gdp, df_pop, divide_by_pop = load_gdp_pop(args.gdp_source, args)
    print(f"  Loaded {len(df_gdp)} GDP observations for {df_gdp['iso_id'].nunique()} countries")
    print(f"  Loaded {len(df_pop)} population observations")

    print(f"Filling GDP gaps (max {args.max_gap} missing years)...")
    df_gdp = infill_gdp_gaps(df_gdp, max_gap=args.max_gap)
    print(f"  After gap filling: {len(df_gdp)} observations")

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
    if divide_by_pop:
        df['pcGDP'] = df['pcGDP'] / df['Pop']   # total-GDP series -> per-capita
    print(f"  After GDP+Pop merge: {len(df)} observations")

    if args.exclude_iso:
        df = df[~df['iso_id'].isin(args.exclude_iso)].copy()
        print(f"  Excluded ISO {args.exclude_iso}: {df['iso_id'].nunique()} countries remain")

    if args.country_filter in ('nearly-all', 'contiguous'):
        df = COUNTRY_FILTERS[args.country_filter](df, min_years=args.min_years, min_frac=args.min_frac)
        print(f"  After country-filter '{args.country_filter}': {len(df)} observations for "
              f"{df['iso_id'].nunique()} countries")

    print("Computing GDP growth rate...")
    df = compute_growth_rate(df)
    print(f"  After growth calculation: {len(df)} observations")

    print("Merging with CRU climate data...")
    df = pd.merge(df, df_cru, on=['iso_id', 'year'], how='inner')
    print(f"  After climate merge: {len(df)} observations for {df['iso_id'].nunique()} countries")

    # Restrict to the requested year window. The start year is bounded by the GDP source, not by
    # CRU (which covers 1901-2024): the first growth year is one after the source's first GDP year.
    df = df[(df['year'] >= args.year_min) & (df['year'] <= args.year_max)].copy()
    print(f"  After year filter ({args.year_min}-{args.year_max}): {len(df)} observations for "
          f"{df['iso_id'].nunique()} countries")

    # 'endpoints' filter: keep only countries with data in both the first and last panel year
    # (the balanced-panel rule), applied after the year filter.
    if args.country_filter == 'endpoints':
        y0, y1 = int(df['year'].min()), int(df['year'].max())
        keep = set(df.loc[df['year'] == y0, 'iso_id']) & set(df.loc[df['year'] == y1, 'iso_id'])
        df = df[df['iso_id'].isin(keep)].copy()
        print(f"  After endpoints filter (present in {y0} and {y1}): {len(df)} observations for "
              f"{df['iso_id'].nunique()} countries")

    # Select and order columns
    output_columns = ['iso_id', 'year', 'pcGDP', 'growth_pcGDP', 'temp', 'precp', 'Pop']
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
