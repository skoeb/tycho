"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

 # --- Python Batteries Included---
import os

# --- External Libraries ---
import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal

# --- Module Imports ---
import tycho
from tycho.config import *


def test_wind_spd():
    df = pd.DataFrame({
        'u':[0,0,1],
        'v':[1,-1,1]
    })

    spd = np.vectorize(fetcher.calculate_wind_spd)(df['u'], df['v'])

    assert spd == np.array([0, 0, 0])

def test_wind_deg():
    df = pd.DataFrame({
        'u':[0,0,1],
        'v':[1,-1,1]
    })

    spd = np.vectorize(fetcher.calculate_wind_deg)(df['u'], df['v'])

    assert spd == np.array([180, 0, 225])

#TODO: test WRI distance matching? 

def test_duplicates(ts_frequency=TS_FREQUENCY):
    # --- establish SQL Connection ---
    SQL = tycho.PostgreSQLCon(schema='test')
    SQL.make_con()

    # --- Read in ETL Pickle ---
    merged = SQL.sql_to_pandas('etl_L3')

    # --- count samples for each plant_id_wri ---
    counts = merged.groupby('plant_id_wri', as_index=False)['datetime_utc'].count()
    
    # --- test counts ---
    counts['not_valid'] = 0
    counts['upper_limit'] = int(TS_DIVISOR * 1.1)
    counts.loc[counts['datetime_utc'] > counts['upper_limit'], 'not_valid'] = 1
    assert counts['not_valid'].sum() == 0


def test_CEMS(test_n=10):
    """Compare merged dataframe CEMS to the raw data."""
    
    # --- establish SQL Connection ---
    SQL = tycho.PostgreSQLCon(schema='test')
    SQL.make_con()

    # --- Read in ETL Pickle ---
    merged = SQL.sql_to_pandas('etl_L3')

    # --- subset to n ---
    keep_plants = merged.sample(frac=1).iloc[0:test_n]['plant_id_eia'].tolist()
    merged = merged.loc[merged['plant_id_eia'].isin(keep_plants)]

    # --- fetch cems ---
    # ASSUMING FETCHING WORKS

    # --- load cems ---
    loader = tycho.CEMSLoader(ts_frequency=TS_FREQUENCY, years=[2019], clean_on_load=True, use_pickle=False)
    loader._read_csvs()
    loader._clean_cems()
    cems = loader.cems

    # --- pivot onto index ---
    index_df = merged[['plant_id_eia', 'datetime_utc']]
    cems = cems.merge(index_df, on=['plant_id_eia','datetime_utc'], how='right')
    
    # --- check for missing dates from merged ---
    cems_subset = cems.loc[cems['plant_id_eia'].isin(keep_plants)]
    missing = cems_subset.merge(index_df, on=['plant_id_eia','datetime_utc'], how='left')
    missing = missing.loc[missing.isnull().sum(axis=1) > 0]
    assert len(missing) == 0

    # --- clean up ---
    keep_cols = ['datetime_utc', 'plant_id_eia', 'gross_load_mw', 'so2_lbs', 'nox_lbs', 'co2_lbs', 'operational_time']
    cems = cems[keep_cols]
    merged = merged[keep_cols]

    # --- sort ---
    cems.sort_values(['datetime_utc','plant_id_eia'], ascending=True, inplace=True)
    cems.reset_index(drop=True, inplace=True)
    merged.sort_values(['datetime_utc', 'plant_id_eia'], ascending=True, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # --- compare ---
    assert_frame_equal(cems, merged, check_less_precise=True, check_dtype=False)


def test_earthengine(test_n=5):
    """Compare merged dataframe earth engine to the raw data."""

    # --- establish SQL Connection ---
    SQL = tycho.PostgreSQLCon(schema='test')
    SQL.make_con()

    # --- Read in ETL Pickle ---
    merged = SQL.sql_to_pandas('etl_L3')

    # --- subset to n ---
    keep_plants = merged.sample(frac=1).iloc[0:test_n]['plant_id_eia'].tolist()
    merged = merged.loc[merged['plant_id_eia'].isin(keep_plants)]

    # --- fetch earth engine ---
    fetcher = tycho.EarthEngineFetcherLite(read_cache=False, use_cache=False)
    earthengine = fetcher.fetch(merged)

    # --- pivot onto index ---
    index_df = merged[['plant_id_wri', 'datetime_utc']]
    
    merger = tycho.RemoteDataMerger
    merger.earthengine = earthengine #override pickle reading
    merger._pivot_buffers()
    merger._merge_pivot(index_df)
    merger._clean()
    earthengine_out = merger.merged

    # --- filter ---
    merged = merged[list(earthengine_out.columns)]

    # --- sort ---
    earthengine_out.sort_values(['datetime_utc','plant_id_eia'], ascending=True, inplace=True)
    earthengine_out.reset_index(drop=True, inplace=True)
    merged.sort_values(['datetime_utc', 'plant_id_eia'], ascending=True, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # --- compare ---
    assert_frame_equal(earthengine_out, merged)
