"""
Create a long df with the following columns for CEMS and predictions:
    source, country, lat/lon, date, fuel, plant_id_wri, variable, value

Used for dashboard.
"""

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
import hashlib

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee
import pycountry_convert as pcc

# --- Module Imports ---
import tycho.config as config
import tycho.helper as helper

import logging
log = logging.getLogger("tycho")

def calc_emission_factor(df):
    """Calculate endogenous and exogenous emission factor."""
    
    # --- create exogenous df ---
    exo = df.copy()
    exo['type'] = 'Exogenous'

    for y in config.ML_Y_COLS:

        # --- Exogenous emission factors based on WRI annual estimation---
        if y != 'gross_load_mw': 
            exo[f"{y}_ef_mwh"] = exo[y] / (exo['estimated_generation_gwh'] * 1000 / config.TS_DIVISOR)

    # --- Endogenous emission factors based on gross_load_mw prediction ---
    if 'gross_load_mw' in df.columns:
            
        # --- create endogenous df ---
        endo = df.copy()
        endo['type'] = 'Endogenous'

        for y in config.ML_Y_COLS:
            if y != 'gross_load_mw': 
                endo[f"{y}_ef_mwh"] = endo[y] / endo['gross_load_mw']

        # --- merge exo and endo ---
        out = pd.concat([exo, endo], axis='rows')
        
    else:
        out = exo

    return out


def country_to_continent(country_name):
    country_alpha2 = pcc.country_name_to_country_alpha2(country_name)
    country_continent_code = pcc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pcc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

def package():
    """Create the output dataframe for dashboard visualization. 
        long df with the following columns:
            - lat/lon, date, fuel, plant_id_wri, source (CEMS or Tycho Prediction)
            - variable (co2_lbs, nox_lbs, so2_lbs, gross_load_mw, emission factors)
            -values 
    """
    # --- Load ground truth data ---
    merged = pd.read_pickle(os.path.join('processed', 'merged_df.pkl'))

    # --- Load predicted data ---
    predicted = pd.read_csv(os.path.join('processed','predictions','predictions.csv'))

    # --- rename ---
    pred_rename_dict = {
        'pred_co2_lbs':'co2_lbs',
        'pred_so2_lbs':'so2_lbs',
        'pred_nox_lbs':'nox_lbs',
        'pred_gross_load_mw':'gross_load_mw'
    }
    predicted.rename(pred_rename_dict, inplace=True, axis='columns')

    # --- Assign Source ---
    merged['source'] = 'EPA CEMS Ground Truth'
    predicted['source'] = 'Tycho Prediction'

    # --- Calc EFs ---
    log.info('....calculating emission factors')
    merged = calc_emission_factor(merged)
    predicted = calc_emission_factor(predicted)

    # --- add continents ---
    merged['continent'] = merged['country'].apply(country_to_continent)
    predicted['continent'] = predicted['country'].apply(country_to_continent)

    # --- subset ---
    id_vars = ['plant_id_wri','datetime_utc','latitude','longitude','estimated_generation_gwh','wri_capacity_mw','country','continent','primary_fuel','source','type']
    var_cols = ['co2_lbs', 'so2_lbs','nox_lbs','gross_load_mw', 'co2_lbs_ef_mwh', 'so2_lbs_ef_mwh', 'nox_lbs_ef_mwh']
    merged = merged[id_vars + var_cols]
    predicted = predicted[id_vars + var_cols]

    # --- merge ---
    df = pd.concat([merged, predicted], axis='rows')

    # --- melt ---
    long_df = df.melt(id_vars=id_vars, value_vars=var_cols)

    # --- clean up ---
    log.info(f'....dropping nans and inf, shape before: {long_df.shape}')
    long_df = long_df.replace([np.inf, -np.inf], np.nan)
    long_df.dropna(inplace=True)
    log.info(f'........shape after: {long_df.shape}')

    # --- output ---
    log.info('....outputting melted df')
    out_path = os.path.join('data','dashboard','dashboard_df.pkl')
    long_df.to_pickle(out_path)

    return long_df

if __name__ == "__main__":
    package()


