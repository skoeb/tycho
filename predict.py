
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
from tycho.config import *
import tycho

import logging
log = logging.getLogger("tycho")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ Read GPPD ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GPPDLoad = tycho.GPPDLoader()
GPPDLoad.load() #loads countries 
gppd = GPPDLoad.gppd

# --- Subset to PREDICT_COUNTRIES ---
for country in PREDICT_COUNTRIES:
    assert country in list(gppd['country_long'])

gppd = gppd.loc[gppd['country_long'].isin(PREDICT_COUNTRIES)]

# --- Repeat rows for observations at TS_FREQUENCY ---
df = tycho.apply_date_range_to_gppd(gppd)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ Download Earth Engine Data ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for earthengine_db in EARTHENGINE_DBS:
    EeFetch = tycho.EarthEngineFetcher(earthengine_db, buffers=BUFFERS)
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
load_path = os.path.join('models', PREDICT_MODEL)

with open(os.path.join(load_path, 'pipe.pkl'), 'rb') as handle:
    preprocess_pipe = pickle.load(handle)

clean = preprocess_pipe.transform(clean)

# --- Iterate through each target variable ---
pred_out_df = merged[['plant_id_wri', 'datetime_utc', 'country_long', 'primary_fuel', 'estimated_generation_gwh']]
for y_col in ML_Y_COLS:

    with open(os.path.join(load_path, f'model_{y_col}_{PREDICT_MODEL}.pkl'), 'rb') as handle:
        model = pickle.load(handle)
    
    y_pred = model.predict(clean)
    
    # --- Append prediction to dataframe ---
    pred_out_df[f'pred_{y_col}'] = y_pred

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ Calculate Emission Factor ~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for y_col in ML_Y_COLS:

    # --- Exogenous emission factors based on WRI annual estimation---
    if y_col != 'gross_load_mw': 
        if TS_FREQUENCY in ['W','W-SUN']:
            divisor = 52
        elif TS_FREQUENCY in ['MS']:
            divisor = 12
        elif TS_FREQUENCY in ['D']:
            divisor = 365
        elif TS_FREQUENCY in ['A','AS']:
            divisior = 1
        elif TS_FREQUENCY == '3D':
            divisor = int(365 / 3)
        else:
            raise NotImplementedError(f'Please write a wrapper for {TS_FREQUENCY}!')
        pred_out_df[f"exo_{y_col}_factor_mwh"] = pred_out_df[f"pred_{y_col}"] / (pred_out_df['estimated_generation_gwh'] * 1000 / divisor)

    # --- Endogenous emission factors based on gross_load_mw prediction ---
    if 'gross_load_mw' in ML_Y_COLS:
        pred_out_df[f"endo_{y_col}_factor_mwh"] = pred_out_df[f"pred_{y_col}"] / pred_out_df['pred_gross_load_mw']

# --- Write out dfs ---
pred_out_df.to_csv(os.path.join('processed','predictions','pred_emission_factors.csv'), index=False)

# --- Plot Prediction ---
tycho.plot_emission_factor(data_type='pred', country=PREDICT_COUNTRIES[0])