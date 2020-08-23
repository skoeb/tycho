
# --- Python Batteries Included---
<<<<<<< HEAD
=======
import sqlite3
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
import os
import ftplib
import concurrent.futures as cf
import time
import json
import itertools
import random
import pickle

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# --- Module Imports ---
import tycho.config as config
import tycho

import logging
log = logging.getLogger("tycho")

def apply_date_range_to_gppd(gppd,
                            start_date=config.PREDICT_START_DATE,
                            end_date=config.PREDICT_END_DATE,
                            ts_frequency=config.TS_FREQUENCY):

    # --- Initialize date range ---
    date_range = pd.date_range(start=start_date, end=end_date, freq=ts_frequency)

    # --- Permute ---
    date_dfs = []
    for d in date_range:
        date_df = gppd.copy()
        date_df['datetime_utc'] = d
        date_dfs.append(date_df)

    # --- Concat ---
    df = pd.concat(date_dfs, axis='rows', sort=False)

    return df


<<<<<<< HEAD
def predict(plot=True):

    # --- establish SQL Connection ---
    SQL = tycho.PostgreSQLCon()
=======
def predict(sql_db='tycho_production', plot=True):

    # --- establish SQLite Connection ---
    SQL = tycho.SQLiteCon(sql_db)
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
    SQL.make_con()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~ Read GPPD ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    GPPDLoad = tycho.GPPDLoader(countries=config.PREDICT_COUNTRIES)
    GPPDLoad.load() #loads countries 
    gppd = GPPDLoad.gppd
    
    # --- Repeat rows for observations at TS_FREQUENCY ---
    df = apply_date_range_to_gppd(gppd)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~ Download Earth Engine Data ~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--- Load Google Earth Engine Data (such as weather and population) using df for dates ---
    EarthEngineFetch = tycho.EarthEngineFetcherLite()
    EarthEngineFetch.fetch(df)

    # --- Merge Remote Sensing (Earth Engine) Data onto df ---
    EarthEngineMerge = tycho.EarthEngineDataMergerLite()
    df = EarthEngineMerge.merge(df)

    if config.FETCH_S3:
        # --- fetch S3 data ---
        SentinelFetcher = tycho.S3Fetcher()
        SentinelFetcher.fetch(df)

    # --- aggregate and merge onto df ---
    SentinelCalculator = tycho.L3Loader()
    merged = SentinelCalculator.calculate(df)

    # --- Sanitize ---
    ColumnSanitize = tycho.ColumnSanitizer()
    clean = ColumnSanitize.sanitize(merged)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ Predict ~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    load_path = os.path.join('models', config.PREDICT_MODEL)

    with open(os.path.join(load_path, 'pipe.pkl'), 'rb') as handle:
        preprocess_pipe = pickle.load(handle)

    clean = preprocess_pipe.transform(clean)

    # --- Iterate through each target variable ---
    pred_out_df = merged[['plant_id_wri', 'datetime_utc', 'country', 'latitude', 'longitude', 'primary_fuel', 'estimated_generation_gwh', 'wri_capacity_mw']]

    log.info('....starting predictions')
    for y_col in config.ML_Y_COLS:

        with open(os.path.join(load_path, f'model_{y_col}_{config.PREDICT_MODEL}.pkl'), 'rb') as handle:
            model = pickle.load(handle)
        
        log.info(f'........predicting for {y_col}')
        y_pred = model.predict(clean)

        # --- Cap gross_load_mw by feasible capacity factor ---
        if y_col == 'gross_load_mw':
            estimated_cf = merged['estimated_generation_gwh'] / (merged['wri_capacity_mw'] * 365 * 24 / 1000)
            max_feasible_cf = (estimated_cf * 1.25).clip(upper=0.95)
            max_feasible_mwh = merged['wri_capacity_mw'] * max_feasible_cf * (365 / config.TS_DIVISOR) * 24
            assert len(y_pred) == len(max_feasible_mwh)
            y_pred = y_pred.clip(min=0, max=max_feasible_mwh)
        
        # --- Append prediction to dataframe ---
        pred_out_df[f'pred_{y_col}'] = y_pred

    # --- Write out dfs ---
    log.info('....writing out predictions')
<<<<<<< HEAD
    SQL.pandas_to_sql(pred_out_df, 'predictions')
=======
    SQL.pandas_to_sql('predictions.csv')
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
    
    return pred_out_df

if __name__ == '__main__':
    predict()