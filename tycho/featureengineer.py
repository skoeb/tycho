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

class CapacityFeatures(TransformerMixin):
    # def __init__(self):

    def _calc(self, X):
        
        Xt = X.copy()

        # --- turn state into region ---
        if 'state' in Xt.columns:
            Xt['region'] = Xt['state']
        else:
            Xt['region'] = Xt['country']

        # --- fill region nan with country (for non-US) ---
        Xt['region'] = Xt['region'].fillna(Xt['country'])
        
        # --- calc total capacity by region ---
        Xt['region_mw'] = Xt.groupby('region')['wri_capacity_mw'].transform(lambda x: x.sum())
        
        # --- calc total capcity from fuel source by region ---
        Xt['region_fuel_mw'] = Xt.groupby(['region','fuel_type_code_pudl'])['wri_capacity_mw'].transform(lambda x: x.sum())
        
        # --- calc plant % of total capcity by region ---
        Xt['plant_region_pct'] = Xt['wri_capacity_mw'] / Xt['region_mw']
        
        # --- calc plant % of fuel source capacity by region ---
        Xt['plant_region_fuel_pct'] = Xt['wri_capacity_mw'] / Xt['region_mw']

        # --- calc plant avg capacity factor ---
        Xt['capacity_factor'] = Xt['generation_gwh_2017'] / (Xt['wri_capacity_mw'] * 8760)
        
        # --- calc fuel avg capacity factor for region ---
        Xt['avg_region_fuel_capacity_factor'] = Xt.groupby(['region','fuel_type_code_pudl'])['capacity_factor'].transform(lambda x: x.mean())
        
        # --- calc difference between plant cf and region class avg ---
        Xt['capacity_factor_diff'] = Xt['capacity_factor'] - Xt['avg_region_fuel_capacity_factor']

        # --- Drop country and state columns ---
        Xt = Xt.drop(['region','country','state'], axis='columns')

        return Xt
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xt = self._calc(X)
        return Xt


class DateFeatures(TransformerMixin):
    # def __init__(self):

    def _calc(self, X):
        
        Xt = X.copy()

        # --- get day of week ---
        Xt['dow'] = [i]

        return Xt
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xt = self._calc(X)
        return Xt