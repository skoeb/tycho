
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Module Imports ---
import tycho.config as config
import tycho.helper as helper

import logging
log = logging.getLogger("tycho")

class ColumnSanitizer():
    """
    Drop columns that should not be used for ML pipelines. 
    """
    def __init__(self):
        pass
    
    def _drop(self, df):
        """
        Drop columns that would not be available for test set
            (i.e. those from eia data not available internationally).
        """
        drop_cols = [
            'plant_id_eia','report_year',
            'capacity_mw','summer_capacity_mw',
            'winter_capacity_mw','minimum_load_mw',
            'fuel_type_code_pudl','multiple_fuels',
            'planned_retirement_year','plant_name_eia',
            'city','county','latitude','longitude',
            'timezone','geometry','index',
        ]

        df = df.drop(drop_cols, axis='columns', errors='ignore')
        return df

    def sanitize(self, X):
        # --- drop columns ---
        log.info(f'....starting ColumnSanitizer, shape {X.shape}')
        Xt = self._drop(X) 
        log.info(f'........finished ColumnSanitizer, shape {Xt.shape}')
        return Xt

class OneHotEncodeWithThresh(TransformerMixin):
    def __init__(self, n_unique=12):
        self.n_unique = n_unique

    def _find_categories(self, X):
        
        # --- Get columns with fewer than n_unique ---
        self.categorical_cols = [c for c in X.columns if len(set(X[c])) <= self.n_unique]
        self.numerical_cols = [c for c in X.columns if c not in self.categorical_cols]

        # --- Force some columns to be numeric ---
        force_num_cols = ['estimated_generation_gwh','planned_retirement_year','country']
        self.categorical_cols = [c for c in self.categorical_cols if c not in force_num_cols]

        # --- Permute categorical cols ---
        self.dummy_cols = []
        for c in self.categorical_cols:
            col_vals = list(set(X[c]))
            for v in col_vals:
                self.dummy_cols.append(f"{c}_{v}")
            
        return self
    
    def _save_feature_names(self, X):
        self.feature_names = list(X.columns)
        return self
    
    def fit(self, X, y=None):
        self._find_categories(X)
        return self
    
    def transform(self, X):

        log.info(f'....starting OneHotEncodeWithThresh, shape {X.shape}')
        Xt = pd.get_dummies(X, columns=self.categorical_cols)

        # --- add in 0 dummy_cols if not in (i.e. test set without a type of fuel) ---
        for c in self.dummy_cols:
            if c not in Xt.columns:
                Xt[c] = 0
        
        # --- drop any extra columns ---
        Xt = Xt[self.dummy_cols + self.numerical_cols]
        
        # --- force column order ---
        Xt= Xt.reindex(sorted(Xt.columns), axis=1)

        # --- save feature names ---
        self._save_feature_names(Xt)

        log.info(f'........finished OneHotEncodeWithThresh, shape {Xt.shape}')
        return Xt
            

class DropNullColumns(TransformerMixin):
    def __init__(self, null_frac=0.8):
        self.null_frac = null_frac

    def _find_nulls(self, X):
        
        # --- Get columns with more null % than null_frac ---
        self.null_cols = [c for c in X.columns if (X[c].isnull().sum() / len(X[c])) > self.null_frac]
        return self
    
    def fit(self, X, y=None):
        self._find_nulls(X)
        return self
    
    def transform(self, X):
        log.info(f'....starting DropNullColumns, shape {X.shape}')
        # --- drop null cols ---
        Xt = X.drop(self.null_cols, axis='columns')

        # --- drop identifier cols ---
        Xt = Xt.drop(['datetime_utc', 'plant_id_wri','index'], axis='columns', errors='ignore')
        log.info(f'........finished DropNullColumns, shape {Xt.shape}')
        return Xt

class LowMemoryMinMaxScaler(TransformerMixin):
    def __init__(self, dtype='uint16'):
        self.dtype = dtype
        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max
        self.MinMax = MinMaxScaler(feature_range=(self.min_val, self.max_val))

    def fit(self, X, y=None):
        self.MinMax.fit(X, y)
        return self

    def transform(self, X, y=None):
        Xt = self.MinMax.transform(X)
        Xt = Xt.astype(self.dtype)
        return Xt