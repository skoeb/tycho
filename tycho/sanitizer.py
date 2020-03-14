
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

# --- Module Imports ---
import tycho.config as config
import tycho.helper as helper

import logging
log = logging.getLogger("tycho")

class ColumnSanitizer():
    """
    Drop columns that should not be used for ML pipelines. 

    Inputs
    ------
    geographic_scope (str) : domestic models (for us) provide more training data 
        from EIA and PUDL sources, international models are more generalizable with
        data from global sources (remote sensing, WRI database, etc.)


    Methods
    -------

    Attributes
    ----------

    Assumptions
    -----------

    """
    def __init__(self, geographic_scope=config.GEOGRAPHIC_SCOPE):
        assert geographic_scope in ['domestic','international']
        self.geographic_scope=geographic_scope
    
    def _international(self, df):
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
            'city','county','latitude','longitude','state',
            'timezone','geometry'
        ]

        df = df.drop(drop_cols, axis='columns', errors='ignore')
        return df
    
    def _domestic(self, df):

        drop_cols = [
            'plant_id_eia','report_year',
            'plant_name_eia','city','county',
            'latitude','longitude',
            'timezone','geometry'
        ]

        df = df.drop(drop_cols, axis='columns', errors='ignore')
        return df

    def sanitize(self, df):

        # --- drop columns ---
        if self.geographic_scope == 'domestic':
            df = self._domestic(df)
        elif self.geographic_scope == 'international':
            df = self._international(df)

        # --- set multiindex ---
        df = df.set_index(['plant_id_wri', 'datetime_utc'], drop=True)
        return df

class OneHotEncodeWithThresh(TransformerMixin):
    def __init__(self, n_unique=15):
        self.n_unique = n_unique

    def _find_categories(self, X):
        
        # --- Get columns with fewer than n_unique ---
        self.categorical_cols = [c for c in X.columns if len(set(X[c])) < self.n_unique]

        # --- Force some columns to be numeric ---
        force_num_cols = ['estimated_generation_gwh','planned_retirement_year','country']
        self.categorical_cols = [c for c in self.categorical_cols if c not in force_num_cols]
        return self
    
    def fit(self, X, y=None):
        self._find_categories(X)
        return self
    
    def transform(self, X):
        Xt = pd.get_dummies(X, columns=self.categorical_cols)
        return Xt

class DropNullColumns(TransformerMixin):
    def __init__(self, null_frac=0.8):
        self.null_frac = null_frac

    def _find_nulls(self, X):
        
        # --- Get columns with more null % than null_frac ---
        self.null_cols = [c for c in X.columns if (X[c].isnull().sum() / len(X[c])) > self.null_frac]
        return self
    
    def fit(self, X, y=None):
        self._find_categories(X)
        return self
    
    def transform(self, X):
        Xt = X.drop(self.null_cols, axis='columns')
        return Xt
            
#keep wri plant id
# merge EE onto merged_df
# get CEMS columns
# set multiindex with datetime_utc and wri plant id
# write clean_df
# preprocess pipeline:
    # one_hot_encoding
    # imputation
    # scaling