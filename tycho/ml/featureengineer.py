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

class WindFeatures(TransformerMixin):

    def __init__(self):
        pass
    
    def _calc(self, df):
        for db in config.S3_DBS:
            df[f"marginal_{db}"] = df[f"up_wind_{db}"] - df[f"down_wind_{db}"]
        return df
    
    def fit(self, X, y=None):
        return self._calc(X)
    
    def predict(self, X, y=None):
        return self._calc(X)


class CapacityFeatures(TransformerMixin):
    def __init__(self):
        self.region_fuel_table = None
        self.region_table = None


    def _calc(self, X, update_table=False):
        
        Xt = X.copy()

        # --- turn state into region ---
        if 'state' in Xt.columns:
            Xt['region'] = Xt['state']
        else:
            Xt['region'] = Xt['country']

        # --- fill region nan with country (for non-US) ---
        Xt['region'] = Xt['region'].fillna(Xt['country'])
        
        # --- calc total capacity by region ---
        Xt = Xt.merge(self.region_table, on='region', how='left')
        
        # --- calc total capcity from fuel source by region ---
        Xt = Xt.merge(self.region_fuel_table, on=['region','primary_fuel'], how='left')
        
        # --- calc plant % of total capcity by region ---
        Xt['plant_region_pct'] = Xt['wri_capacity_mw'] / Xt['region_mw']
        
        # --- calc plant % of fuel source capacity by region ---
        Xt['plant_region_fuel_pct'] = Xt['wri_capacity_mw'] / Xt['region_mw']

        # --- calc plant avg capacity factor ---
        Xt['capacity_factor'] = (Xt['generation_gwh_2017'] * 1000) / (Xt['wri_capacity_mw'] * 8760)
        
        # --- calc plant age ---
        Xt['plant_age'] = 2020 - Xt['commissioning_year']

        # --- Drop country and state columns ---
        Xt = Xt.drop(['region','country','state','commissioning_year'], axis='columns', errors='ignore')

        return Xt
    

    def _update_table(self, X):

        Xt = X.copy()

        # --- turn state into region ---
        if 'state' in Xt.columns:
            Xt['region'] = Xt['state']
        else:
            Xt['region'] = Xt['country']

        train_data = Xt[['region','primary_fuel','wri_capacity_mw']]

        # --- create lookup tables ---
        new_region_fuel_table = train_data.groupby(['region','primary_fuel'], as_index=False)['wri_capacity_mw'].sum()
        new_region_fuel_table = new_region_fuel_table.rename({'wri_capacity_mw':'region_fuel_mw'}, axis='columns')
        
        new_region_table = train_data.groupby(['region'], as_index=False)['wri_capacity_mw'].sum()
        new_region_table = new_region_table.rename({'wri_capacity_mw':'region_mw'}, axis='columns')

        # --- write new tables ---
        if isinstance(self.region_fuel_table, type(None)):
            self.region_fuel_table = new_region_fuel_table
            self.region_table = new_region_table

        # --- update existing tables --- 
        else:
            region_fuel_merged = self.region_fuel_table.merge(new_region_fuel_table, on=['region','primary_fuel'])
            region_merged = self.region_table.merge(new_region_table, on=['region'])

            region_fuel_merged['region_fuel_mw'] = region_fuel_merged[['region_fuel_mw_x', 'region_fuel_mw_y']].sum(axis=1)
            region_merged['region_mw'] = region_merged[['region_mw_x','region_mw_y']].sum(axis=1)

            region_fuel_merged = region_fuel_merged[['region','primary_fuel','region_fuel_mw']]
            region_merged = region_merged[['region','region_mw']]

            self.region_fuel_table = region_fuel_merged
            self.region_table = region_merged

        return self


    def fit(self, X, y=None):
        self._update_table(X)
        return self


    def transform(self, X, y=None):
        self._update_table(X)
        log.info(f'....starting CapacityFeatures, shape {X.shape}')
        Xt = self._calc(X, update_table=True)
        log.info(f'........finished CapacityFeatures, shape {Xt.shape}')
        return Xt
    
    def get_feature_names(self):
        return X.columns.tolist()


class DateFeatures(TransformerMixin):
    def __init__(self):
        pass

    def _calc(self, X):
        
        Xt = X.copy()

        # --- get day of week ---
        Xt['dow'] = [i.dayofweek for i in Xt['datetime_utc']]
        Xt['month'] = [i.month for i in Xt['datetime_utc']]
        return Xt
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        log.info(f'....starting DateFeatures, shape {X.shape}')
        Xt = self._calc(X)
        log.info(f'........finished DateFeatures, shape {Xt.shape}')
        return Xt

def calc_average_y_vals_per_MW(df,
                               groupbycols=['month','primary_fuel'],
                               y_cols=config.CEMS_Y_COLS):
    
    X = df.copy()
    
    # --- drop gross_load_mw from y_cols ---
    y_cols = [c for c in y_cols if c not in 'gross_load_mw']

    # --- Create needed cols ---
    X['month'] = [i.month for i in X['datetime_utc']]
    
    for y in y_cols:
        X[f'{y}_per_MW'] = X[y] / X['wri_capacity_mw']
    
    # --- group training data by groupbycols ---
    per_MW_cols = [f'{y}_per_MW' for y in y_cols]
    grouped_avg = X.groupby(groupbycols, as_index=False)[per_MW_cols].mean()
    grouped_std = X.groupby(groupbycols, as_index=False)[per_MW_cols].agg(np.std, ddof=0)

    # --- merge grouped ---
    grouped = grouped_avg.merge(grouped_std, on=groupbycols, how='inner', suffixes=('_mean', '_std'))

    # --- output ---
    grouped.to_pickle(os.path.join('models', 'avg_y_table.pkl'))

    return grouped

class ApplyAvgY(TransformerMixin):
    def __init__(self, avg_table=None):
        if isinstance(avg_table, type(None)):
            self.avg_table = pd.read_pickle(os.path.join('models', 'avg_y_table.pkl'))
        else:
            self.avg_table = avg_table

    def _merge(self, X):
        Xt = X.merge(self.avg_table, on=['month','primary_fuel'], how='left')
        return Xt
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        log.info(f'....starting ApplyAvgY, shape {X.shape}')
        Xt = self._merge(X)
        log.info(f'........finished ApplyAvgY, shape {Xt.shape}')
        return Xt
        
