
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
gppd = WRILoad.gppd

# --- Subset to PREDICT_COUNTRIES ---
for country in PREDICT_COUNTRIES:
    assert country in list(gppd['country_long'])

gppd = gppd.loc[gppd['country_long'].isin(PREDICT_COUNTRIES)]

# --- Sanitize ---
ColumnSanitize = tycho.ColumnSanitizer()
clean = ColumnSanitize.sanitize(gppd)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~ Predict ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Preprocess ---
load_path = os.path.join('models', PREDICT_MODEL)

with open(os.path.join(load_path, 'pipe.pkl'), 'wb') as handle:
    preprocess_pipe = pickle.load(handle)

clean = preprocess_pipe.transform(clean)

# --- Iterate through each target variable ---
preds = []
for y_col in ML_Y_COLS:

    with open(os.path.join(out_path, f'model_{y_col}.pkl'), 'wb') as handle:
        model = pickle.load(handle)
    
    y_pred = model.predict(clean)
    
    y_pred = pd.DataFrame(y_pred, columns=[y_col])
    preds.append(y_pred)

# --- Package and output ---
pred_df = pd.concat(preds, axis='columns')
pred_df[['plant_id_wri', 'datetime_utc','country_long','primary_fuel']] = gppd[['plant_id_wri', 'datetime_utc','country_long','primary_fuel']]
pred_df.to_pickle(os.path.join('processed','pred.pkl'))

