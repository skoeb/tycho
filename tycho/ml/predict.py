
# --- Python Batteries Included---
import sqlite3
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


def predict(plot=True):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~ Read GPPD ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    GPPDLoad = tycho.GPPDLoader()
    GPPDLoad.load() #loads countries 
    gppd = GPPDLoad.gppd

    # --- Subset to PREDICT_COUNTRIES ---
    for country in config.PREDICT_COUNTRIES:
        assert country in list(gppd['country_long'])

    gppd = gppd.loc[gppd['country_long'].isin(config.PREDICT_COUNTRIES)]

    # --- Repeat rows for observations at TS_FREQUENCY ---
    df = apply_date_range_to_gppd(gppd)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~ Download Earth Engine Data ~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    EeFetch = tycho.EarthEngineFetcher()
    EeFetch.fetch(df)

    # --- Merge Remote Sensing (Earth Engine) Data onto df ---
    RemoteMerge = tycho.RemoteDataMerger()
    merged = RemoteMerge.merge(df)

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
    pred_out_df = merged[['plant_id_wri', 'datetime_utc', 'country', 'latitude', 'longitude', 'primary_fuel', 'estimated_generation_gwh']]

    log.info('....starting predictions')
    for y_col in config.ML_Y_COLS:

        with open(os.path.join(load_path, f'model_{y_col}_{config.PREDICT_MODEL}.pkl'), 'rb') as handle:
            model = pickle.load(handle)
        
        log.info(f'........predicting for {y_col}')
        y_pred = model.predict(clean)
        
        # --- Append prediction to dataframe ---
        pred_out_df[f'pred_{y_col}'] = y_pred

    # --- Write out dfs ---
    log.info('....writing out predictions')
    pred_out_df.to_csv(os.path.join('processed','predictions','predictions.csv'), index=False)
    
    return pred_out_df

if __name__ == '__main__':
    predict()