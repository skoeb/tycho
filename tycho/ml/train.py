
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
from lightgbm import LGBMRegressor

# --- Module Imports ---
import tycho
import tycho.config as config

# --- logging ---
import logging
log = logging.getLogger("tycho")

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

def train(save_pickles=True, sql_db='tycho_production'):

    # --- establish SQLite Connection ---
    SQL = tycho.SQLiteCon(sql_db)
    SQL.make_con()

    # --- Read in ETL Pickle ---
    merged = SQL.sql_to_pandas('etl_L3')

    # --- Sanitize ---
    ColumnSanitize = tycho.ColumnSanitizer()
    clean = ColumnSanitize.sanitize(merged)

    # --- Create average lookup tables ---
    avg_table = tycho.calc_average_y_vals_per_MW(clean)

    # --- Split ---
    Splitter = tycho.FourWaySplit()
    X_train_df, X_test_df, y_train_all, y_test_all = Splitter.split(clean)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~ Pipeline ~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pandas_pipe = Pipeline(steps=[
                            ('capacity', tycho.CapacityFeatures()),
                            ('date', tycho.DateFeatures()),
                            ('avg_values', tycho.ApplyAvgY(avg_table)),
                            ('dropnull', tycho.DropNullColumns()),
                            ('onehot', tycho.OneHotEncodeWithThresh()),                 
    ])

    numpy_pipe = Pipeline(steps=[
                            ('imputer', SimpleImputer()),
                            ('scaler', tycho.LowMemoryMinMaxScaler()),
    ])

    preprocess_pipe = Pipeline(steps=[
                            ('pd', pandas_pipe),
                            ('np', numpy_pipe),
    ])

    # --- Fit/transform ---
    X_train = preprocess_pipe.fit_transform(X_train_df)
    X_test = preprocess_pipe.transform(X_test_df)

    # --- Create complete dfs for output ---
    train_out_df = X_train_df[['datetime_utc','plant_id_wri', 'estimated_generation_gwh','primary_fuel']]
    train_out_df = pd.concat([train_out_df, y_train_all], axis='columns')

    test_out_df = X_test_df[['datetime_utc','plant_id_wri', 'estimated_generation_gwh','primary_fuel']]
    test_out_df = pd.concat([test_out_df, y_test_all], axis='columns')

    # --- output preprocessing pipe ---
    if save_pickles:
        out_path = os.path.join('models', config.TRAIN_MODEL)
        with open(os.path.join(out_path, 'pipe.pkl'), 'wb') as handle:
            pickle.dump(preprocess_pipe, handle)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~ Train Model ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for y_col in config.ML_Y_COLS:
        log.info('\n')
        log.info(f'....beginning fit for {y_col} using {config.TRAIN_MODEL}')

        # --- Subset y ---
        y_train = np.array(y_train_all[y_col])
        y_test = np.array(y_test_all[y_col])

        # --- Initialize Model ---
        if config.TRAIN_MODEL == 'lr':
            model = LinearRegression(
                        fit_intercept=True, 
                        normalize=False,
                        n_jobs=-1
                    )
        
        elif config.TRAIN_MODEL == 'bayes-lgbm':

            estimator = LGBMRegressor(random_state=1, n_jobs=12, verbose=-1,
                                      num_iterations=1000, boosting_type=None,
                                      learning_rate=0.03, subsample=0.7,
                                      boosting='dart',
                                      )
            
            lgbm_pbounds = {
                # 'boosting':['gbdt','dart'],
                # 'learning_rate': (0.01, 1.),
                # 'n_estimators': (2, 2000),
                'max_depth': (3, 12),
                # 'min_child_weight': (0., 100.),
                # 'min_data_in_leaf' : (1, 40),
                'num_leaves': (2, 2000), # large num_leaves helps improve accuracy but might lead to over-fitting
                # 'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
                'objective' : ['rmse','mae','tweedie'],
                'max_bin': (128, 10000), # large max_bin helps improve accuracy but might slow down training progress
                # 'colsample_bytree' : (0.3,1),
                # 'subsample' : (0.3, 1.),
                # 'reg_alpha' : (0., 300.),
                # 'reg_lambda' : (0., 300.),
            }

            model = tycho.BayesRegressor(estimator=estimator, pbounds=lgbm_pbounds)

        elif config.TRAIN_MODEL == 'bayes-xgb':

            estimator = XGBRegressor(random_state=1, nthread=12, tree_method='gpu_hist', single_precision_histogram=True, validate_paramters=True)

            xgb_pbounds = {
                'booster':['dart','gbtree','gblinear'],
                'max_depth': (3, 11),
                # 'learning_rate': (0.1, 0.5),
                'subsample': (0.1, 1.),
                # 'sampling_metod':['uniform','gradient_based'],
                'colsample_bytree': (0.1, 1.),
                # 'colsample_bylevel': (0.1, 1.),
                'max_bin': (2, 10000), # large max_bin helps improve accuracy but might slow down training progress
                # 'grow_policy':['depthwise','lossguide'],
                # 'min_child_weight': (0., 100),
                'reg_alpha' : (0., 250.),
                'reg_lambda' : (0., 250.),
                'gamma': (0., 10.),
                # 'objective': ['reg:tweedie'],
            }

            model = tycho.BayesRegressor(estimator=estimator, pbounds=xgb_pbounds)

        # --- Fit  ---
        model.fit(X_train, y_train)

        # --- Get best estimator ---
        y_train_pred = model.predict(X_train)
        log.info(f'........best train MAE for {y_col}: {mae(y_train, y_train_pred)}')
        log.info(f'........best train mape for {y_col}: {mape(y_train, y_train_pred)}')
        log.info(f'........average value for {y_col} is {y_train.mean()}, MAE as a percent is {mae(y_train, y_train_pred) / y_train.mean()}')

        # --- Predict on test ---
        y_pred = model.predict(X_test)
        log.info(f'........best test mae for {y_col}: {mae(y_test, y_pred)}')
        log.info(f'........best test mape for {y_col}: {mape(y_test, y_pred)}')
        log.info(f'........average value for {y_col} is {y_test.mean()}, MAE as a percent is {mae(y_test, y_pred) / y_test.mean()}')

        if save_pickles:
            if config.TRAIN_MODEL == 'tpot':
                # --- Output model pipeline ---
                model.export(os.path.join(out_path, f'tpot_best_pipe_{y_col}.py'))
                best = model.fitted_pipeline_
                with open(os.path.join(out_path, f'model_{y_col}_{config.TRAIN_MODEL}.pkl'), 'wb') as handle:
                    pickle.dump(best, handle)

            else:
                # --- Output model ---
                with open(os.path.join(out_path, f'model_{y_col}_{config.TRAIN_MODEL}.pkl'), 'wb') as handle:
                    pickle.dump(model, handle)

        # --- save predictions to out dfs ---
        train_out_df[f'pred_{y_col}'] = y_train_pred
        test_out_df[f'pred_{y_col}'] = y_pred
    
    return train_out_df, test_out_df

if __name__ == '__main__':
    train()