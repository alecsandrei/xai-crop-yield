from __future__ import annotations

import geopandas as gpd
import pandas as pd

from xai_crop_yield.config import RAW_DATA_DIR


def get_county_data() -> gpd.GeoDataFrame:
    links = {
        'state': RAW_DATA_DIR / 'us_states.geojson',
        'county': RAW_DATA_DIR / 'us_counties.geojson',
    }
    state = gpd.read_file(links['state'])[['STUSPS', 'STATEFP', 'NAME']]
    state.rename(columns={'NAME': 'STATE_NAME'}, inplace=True)
    county = gpd.read_file(links['county'])
    county = county.merge(state, on='STATEFP')
    county['NAME_LOWERCASE'] = county['NAME'].str.lower()
    return county


def get_data_split() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / 'splits.csv')
