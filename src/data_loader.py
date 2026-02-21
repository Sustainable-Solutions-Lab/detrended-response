"""Data loading and preprocessing for detrended response analysis."""

import pandas as pd
import numpy as np
import pycountry
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Manual country name overrides for CRU data -> ISO3 codes
CRU_COUNTRY_OVERRIDES = {
    'Bolivia': 'BOL',
    'Brunei': 'BRN',
    'Cape Verde': 'CPV',
    'Congo': 'COG',
    'Cote d\'Ivoire': 'CIV',
    'Czech Republic': 'CZE',
    'Democratic Republic of the Congo': 'COD',
    'East Timor': 'TLS',
    'Falkland Isl': 'FLK',
    'Gambia': 'GMB',
    'Iran': 'IRN',
    'Ivory Coast': 'CIV',
    'Korea, Democratic People\'s Republic of': 'PRK',
    'Korea, Republic of': 'KOR',
    'Laos': 'LAO',
    'Libya': 'LBY',
    'Micronesia': 'FSM',
    'Moldova': 'MDA',
    'North Korea': 'PRK',
    'Palestine': 'PSE',
    'Reunion': 'REU',
    'Russia': 'RUS',
    'South Korea': 'KOR',
    'Swaziland': 'SWZ',
    'Syria': 'SYR',
    'Taiwan': 'TWN',
    'Tanzania': 'TZA',
    'The Gambia': 'GMB',
    'UK': 'GBR',
    'USA': 'USA',
    'United States': 'USA',
    'Vatican': 'VAT',
    'Venezuela': 'VEN',
    'Vietnam': 'VNM',
    'Virgin Islands, British': 'VGB',
    'Virgin Islands, U.S.': 'VIR',
}


@dataclass
class AnalysisData:
    """Container for data needed by the analysis."""
    # Observation arrays (length N)
    growth_pcGDP: np.ndarray      # Target variable: GDP growth rate
    pcGDP: np.ndarray             # Per capita GDP level
    temp: np.ndarray              # Temperature (Celsius)
    time: np.ndarray              # Time index (centered)
    country_idx: np.ndarray       # Country index for each observation
    year: np.ndarray              # Calendar year

    # Mappings
    iso_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_iso: Dict[int, str] = field(default_factory=dict)

    # Dimensions
    n_obs: int = 0
    n_countries: int = 0
    n_years: int = 0

    # Year range
    year_range: Tuple[int, int] = (0, 0)
    time_offset: float = 0.0  # Subtracted from year to get time


def map_cru_country_to_iso3(country_name: str) -> str:
    """Map CRU country name to ISO3 code.

    Returns empty string if no mapping found.
    """
    # Check manual overrides first
    if country_name in CRU_COUNTRY_OVERRIDES:
        return CRU_COUNTRY_OVERRIDES[country_name]

    # Try pycountry lookup
    try:
        country = pycountry.countries.lookup(country_name)
        return country.alpha_3
    except LookupError:
        pass

    # Try fuzzy search
    try:
        results = pycountry.countries.search_fuzzy(country_name)
        if results:
            return results[0].alpha_3
    except LookupError:
        pass

    return ''


def load_maddison_data(excel_path: str) -> pd.DataFrame:
    """Load Maddison Project Database GDP data.

    Returns DataFrame with columns: iso_id, year, gdppc, pop
    """
    df = pd.read_excel(excel_path, sheet_name='Full data')

    # Select and rename columns
    df = df[['countrycode', 'year', 'gdppc', 'pop']].copy()
    df = df.rename(columns={'countrycode': 'iso_id'})

    # Drop rows with missing GDP
    df = df.dropna(subset=['gdppc'])

    return df


def load_cru_temperature(csv_path: str) -> pd.DataFrame:
    """Load CRU temperature data and map to ISO3 codes.

    Returns DataFrame with columns: iso_id, year, temp
    """
    df = pd.read_csv(csv_path)

    # Map country names to ISO3 codes
    df['iso_id'] = df['Country'].apply(map_cru_country_to_iso3)

    # Drop rows where mapping failed
    df = df[df['iso_id'] != ''].copy()

    # Select and rename columns
    df = df[['iso_id', 'Year', 'Temperature']].copy()
    df = df.rename(columns={'Year': 'year', 'Temperature': 'temp'})

    return df


def compute_gdp_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per capita GDP growth rate.

    Growth rate is computed as: log(gdppc_t) - log(gdppc_{t-1})

    First observation per country is dropped (no lagged value).
    """
    df = df.sort_values(['iso_id', 'year']).copy()

    # Compute log GDP
    df['log_gdppc'] = np.log(df['gdppc'])

    # Compute growth as first difference of log GDP within each country
    df['growth_pcGDP'] = df.groupby('iso_id')['log_gdppc'].diff()

    # Drop first observation per country (where growth is NaN)
    df = df.dropna(subset=['growth_pcGDP'])

    return df


def load_data(maddison_path: str, cru_path: str,
              year_min: int = 1960, year_max: int = 2022) -> AnalysisData:
    """Load and merge Maddison GDP and CRU temperature data.

    Args:
        maddison_path: Path to Maddison Excel file
        cru_path: Path to CRU CSV file
        year_min: Minimum year to include
        year_max: Maximum year to include

    Returns:
        AnalysisData object containing all arrays and mappings
    """
    # Load datasets
    df_gdp = load_maddison_data(maddison_path)
    df_temp = load_cru_temperature(cru_path)

    # Merge on iso_id and year
    df = pd.merge(df_gdp, df_temp, on=['iso_id', 'year'], how='inner')

    # Compute GDP growth rate BEFORE filtering by year range
    # This allows us to use year_min-1 data to compute growth for year_min
    df = compute_gdp_growth(df)

    # NOW filter by year range (after growth is computed)
    df = df[(df['year'] >= year_min) & (df['year'] <= year_max)].copy()

    # Filter to countries that have data in both the first and last years
    countries_with_first_year = set(df[df['year'] == year_min]['iso_id'])
    countries_with_last_year = set(df[df['year'] == year_max]['iso_id'])
    countries_with_both = countries_with_first_year & countries_with_last_year
    df = df[df['iso_id'].isin(countries_with_both)].copy()

    # Create country index mapping (sorted for reproducibility)
    unique_countries = sorted(df['iso_id'].unique())
    iso_to_idx = {iso: i for i, iso in enumerate(unique_countries)}
    idx_to_iso = {i: iso for iso, i in iso_to_idx.items()}

    # Compute centered time index
    year_mid = (year_min + year_max) / 2
    df['time'] = df['year'] - year_mid

    # Extract arrays
    growth_pcGDP = df['growth_pcGDP'].values.astype(np.float64)
    pcGDP = df['gdppc'].values.astype(np.float64)
    temp = df['temp'].values.astype(np.float64)
    time = df['time'].values.astype(np.float64)
    year = df['year'].values.astype(np.int32)
    country_idx = df['iso_id'].map(iso_to_idx).values.astype(np.int32)

    return AnalysisData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        time=time,
        country_idx=country_idx,
        year=year,
        iso_to_idx=iso_to_idx,
        idx_to_iso=idx_to_iso,
        n_obs=len(growth_pcGDP),
        n_countries=len(unique_countries),
        n_years=len(df['year'].unique()),
        year_range=(year_min, year_max),
        time_offset=year_mid,
    )


def load_data_from_csv(csv_path: str,
                       year_min: int = None, year_max: int = None) -> AnalysisData:
    """Load pre-processed data from CSV file (e.g., df_base_withPop.csv).

    Expected columns: iso_id, year, pcGDP, growth_pcGDP, temp, time, Pop

    Args:
        csv_path: Path to the CSV file
        year_min: Minimum year to include (default: use all years in data)
        year_max: Maximum year to include (default: use all years in data)

    Returns:
        AnalysisData object containing all arrays and mappings
    """
    df = pd.read_csv(csv_path)

    # Filter by year range if specified
    if year_min is not None:
        df = df[df['year'] >= year_min]
    if year_max is not None:
        df = df[df['year'] <= year_max]

    df = df.copy()

    # Determine actual year range from data
    actual_year_min = int(df['year'].min())
    actual_year_max = int(df['year'].max())

    # Filter to countries that have data in both the first and last years
    countries_with_first_year = set(df[df['year'] == actual_year_min]['iso_id'])
    countries_with_last_year = set(df[df['year'] == actual_year_max]['iso_id'])
    countries_with_both = countries_with_first_year & countries_with_last_year
    df = df[df['iso_id'].isin(countries_with_both)].copy()

    # Create country index mapping (sorted for reproducibility)
    unique_countries = sorted(df['iso_id'].unique())
    iso_to_idx = {iso: i for i, iso in enumerate(unique_countries)}
    idx_to_iso = {i: iso for iso, i in iso_to_idx.items()}

    # Compute centered time index
    year_mid = (actual_year_min + actual_year_max) / 2
    df['time_centered'] = df['year'] - year_mid

    # Extract arrays
    growth_pcGDP = df['growth_pcGDP'].values.astype(np.float64)
    pcGDP = df['pcGDP'].values.astype(np.float64)
    temp = df['temp'].values.astype(np.float64)
    time = df['time_centered'].values.astype(np.float64)
    year = df['year'].values.astype(np.int32)
    country_idx = df['iso_id'].map(iso_to_idx).values.astype(np.int32)

    return AnalysisData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        time=time,
        country_idx=country_idx,
        year=year,
        iso_to_idx=iso_to_idx,
        idx_to_iso=idx_to_iso,
        n_obs=len(growth_pcGDP),
        n_countries=len(unique_countries),
        n_years=len(df['year'].unique()),
        year_range=(actual_year_min, actual_year_max),
        time_offset=year_mid,
    )
