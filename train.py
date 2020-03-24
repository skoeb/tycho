
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, SelectFwe, f_regression
import tpot

# --- Module Imports ---
import tycho
from tycho.config import *

import logging
log = logging.getLogger("tycho")

# --- Dumb but clean, sklearn is throwing a few unavoidable warnings ---
import warnings
warnings.filterwarnings("ignore")

# --- Define custom metrics ---
def mean_average_percent_error(y_true, y_pred): 
    np.seterr(divide='ignore')
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pct = np.abs((y_true - y_pred) / (y_true))
    pct = np.nan_to_num(pct, 0, 0, 0)
    mape = np.mean(pct)
    return mape

mape = mean_average_percent_error
mae = mean_absolute_error

# --- Read in ETL Pickle ---
merged = pd.read_pickle(os.path.join('processed', 'merged_df.pkl'))

# --- Sanitize ---
ColumnSanitize = tycho.ColumnSanitizer()
clean = ColumnSanitize.sanitize(merged)

# --- Create average lookup tables ---
avg_table = tycho.calc_average_y_vals_per_MW(clean)

# --- Split ---
SPLITTER = tycho.FourWaySplit()
X_train_df, X_test_df, y_train_all, y_test_all = SPLITTER.split(clean)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ Pipeline ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pandas_pipe = Pipeline(steps=[
                        ('capacity', tycho.CapacityFeatures()),
                        ('avg_values', tycho.ApplyAvgY(avg_table)),
                        ('date', tycho.DateFeatures()),
                        ('dropnull', tycho.DropNullColumns()),
                        ('onehot', tycho.OneHotEncodeWithThresh()),
                        
])

numpy_pipe = Pipeline(steps=[
                        ('imputer', SimpleImputer()),
                        ('scaler', MinMaxScaler()),
])

preprocess_pipe = Pipeline(steps=[
                        ('pd', pandas_pipe),
                        ('np', numpy_pipe),
])

# --- Fit/transform ---
X_train = preprocess_pipe.fit_transform(X_train_df)
X_test = preprocess_pipe.transform(X_test_df)

# --- output preprocessing pipe ---
out_path = os.path.join('models', TRAIN_MODEL)
with open(os.path.join(out_path, 'pipe.pkl'), 'wb') as handle:
    pickle.dump(preprocess_pipe, handle)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ Train Model ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if TRAIN_MODEL == 'lr':
    model = LinearRegression(
                fit_intercept=True, 
                normalize=False,
                n_jobs=-1
            )

elif TRAIN_MODEL == 'xgb':
    model = Pipeline([
        # ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ('fwe', SelectFwe(score_func=f_regression, alpha=0.1)),
        ('xgb', RandomizedSearchCV(
                XGBRegressor(objective='reg:squarederror'),
                param_distributions=XGB_PARAMS, 
                n_iter=RANDOMSEARCH_ITER,
                scoring='neg_mean_squared_error',
                cv=CV_FOLDS, verbose=1,
                refit=True, n_jobs=-1))
    ])

elif TRAIN_MODEL == 'tpot':

    model = tpot.TPOTRegressor(
        generations=TPOT_GENERATIONS,
        population_size=TPOT_POPULATION_SIZE,
        use_dask=True,
        verbosity=2,
        n_jobs=-1,
        cv=CV_FOLDS,
        config_dict=TPOT_CONFIG_DICT,
        max_time_mins=TPOT_TIMEOUT_MINS,
        max_eval_time_mins=3,
        # warm_start=True
    )

# --- Create dfs for output ---
train_out_df = X_train_df[['datetime_utc','plant_id_wri', 'estimated_generation_gwh','primary_fuel']]
train_out_df = pd.concat([train_out_df, y_train_all], axis='columns')

test_out_df = X_test_df[['datetime_utc','plant_id_wri', 'estimated_generation_gwh','primary_fuel']]
test_out_df = pd.concat([test_out_df, y_test_all], axis='columns')

for y_col in ML_Y_COLS:
    log.info('\n')
    log.info(f'....beginning fit for {y_col} using {TRAIN_MODEL}')

    # --- Subset y ---
    y_train = np.array(y_train_all[y_col])
    y_test = np.array(y_test_all[y_col])

    # --- Fit  ---
    model.fit(X_train, y_train)

    # --- Get best estimator ---
    y_train_pred = model.predict(X_train)
    log.info(f'........best train MAE for {y_col}: {mae(y_train, y_train_pred)}')
    log.info(f'........best train mape for {y_col}: {mape(y_train, y_train_pred)}')

    # --- Predict on test ---
    y_pred = model.predict(X_test)
    log.info(f'........best test mae for {y_col}: {mae(y_test, y_pred)}')
    log.info(f'........best test mape for {y_col}: {mape(y_test, y_pred)}')

    if TRAIN_MODEL == 'tpot':
        model.export(os.path.join(out_path, f'tpot_best_pipe_{y_col}.py'))
        best = model.fitted_pipeline_
        with open(os.path.join(out_path, f'model_{y_col}_{TRAIN_MODEL}.pkl'), 'wb') as handle:
            pickle.dump(best, handle)

    elif TRAIN_MODEL == 'xgb':
        log.info('....best params:')
        log.info(f"{model['xgb'].best_params_}")
        
        # --- Output model for y_col ---
        with open(os.path.join(out_path, f'model_{y_col}_{TRAIN_MODEL}.pkl'), 'wb') as handle:
            pickle.dump(model, handle)

    else:
        # --- Output model for y_col ---
        with open(os.path.join(out_path, f'model_{y_col}_{TRAIN_MODEL}.pkl'), 'wb') as handle:
            pickle.dump(model, handle)

    # --- save to out dfs ---
    train_out_df[f'pred_{y_col}'] = y_train_pred
    test_out_df[f'pred_{y_col}'] = y_pred

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
        else:
            raise NotImplementedError(f'Please write a wrapper for {TS_FREQUENCY}!')
        train_out_df[f"exo_{y_col}_factor_mwh"] = train_out_df[f"pred_{y_col}"] / (train_out_df['estimated_generation_gwh'] * 1000 / divisor)
        test_out_df[f"exo_{y_col}_factor_mwh"] = test_out_df[f"pred_{y_col}"] / (test_out_df['estimated_generation_gwh'] * 1000 / divisor)

    # --- Endogenous emission factors based on gross_load_mw prediction ---
    if 'gross_load_mw' in ML_Y_COLS:
        train_out_df[f"endo_{y_col}_factor_mwh"] = train_out_df[f"pred_{y_col}"] / train_out_df['pred_gross_load_mw']
        test_out_df[f"endo_{y_col}_factor_mwh"] = test_out_df[f"pred_{y_col}"] / test_out_df['pred_gross_load_mw']

# --- Write out dfs ---
train_out_df.to_csv(os.path.join('processed','predictions','train_emission_factors.csv'), index=False)
test_out_df.to_csv(os.path.join('processed','predictions','test_emission_factors.csv'), index=False)