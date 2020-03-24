"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

import tycho.tpot_custom_configs as tpot_configs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Number of generators for training set ---
#   - most downloads are cached, so if you set a higher number, 
#     and then a lower, you don't lose data.
N_GENERATORS = None

# --- Multiprocessing settings ---
MULTIPROCESSING = True
WORKERS = 12
THREADS = 6 #ThreadPoolExecutor is failing for Earth Engine queries, so this is still using ProcessPool

# --- Bool for detailed output ---
VERBOSE = False

# --- Frequency of observations (i.e. 'D' for daily, 'W', for weekly, 'M' for monthly, 'A' for annual) --- 
TS_FREQUENCY = 'W-SUN'

TRAIN_COUNTRIES = ['United States of America']
PREDICT_COUNTRIES = ['Puerto Rico']

PREDICT_START_DATE = '01-01-2019'
PREDICT_END_DATE = '03-01-2020'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ EARTH ENGINE SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Sources ---
EARTHENGINE_DBS = [
    "COPERNICUS/S5P/OFFL/L3_NO2",
    "COPERNICUS/S5P/OFFL/L3_SO2",
    # "COPERNICUS/S5P/OFFL/L3_CO",
    # "COPERNICUS/S5P/OFFL/L3_HCHO",
    # "COPERNICUS/S5P/OFFL/L3_O3",
    "CIESIN/GPWv411/GPW_Population_Count",
    "ECMWF/ERA5/DAILY"
]

# --- API Query settings ---
EE_TIMEOUT = 10 #forced timeout, overriding exponential backoff before calling a API query nans
RETRY_EE_NANS = True #after loading cache, retry queries that returned nans in last call

# --- Scale (in meters) to query ---
BUFFERS = [1e3, 1e4, 1e5] #1e2, 1e4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ SANITIZE/SPLIT SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- US y (target) variables ---
CEMS_Y_COLS = [ #LEAKAGE WARNING! Removing something from this list will cause it to persist in the X sets
    'gross_load_mw', 
    'so2_lbs', 
    'nox_lbs',
    'operational_time',
    'co2_lbs'
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ ML SETTINGS ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HOW_TO_SPLIT = 0.25 #train/test split by fraction or list of countries/states
TRAIN_MODEL = 'tpot'
PREDICT_MODEL = 'tpot'
ML_Y_COLS = ['gross_load_mw','so2_lbs','nox_lbs','co2_lbs']
# ML_Y_COLS = ['so2_lbs','nox_lbs']
CV_FOLDS = 3

# --- Grid search params for XGB Training ---
XGB_PARAMS = {
    'learning_rate': [0.1],
    'estimators':[100],
    'min_child_weight': [0, 1, 6, 8, 10],
    'subsample': [0.6],
    'max_depth': [8,9,10],
    'gamma': [0.5, 1, 2, 5],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'tree_method':['gpu_hist'],
    # 'gpu_hist':[0],
}

RANDOMSEARCH_ITER = 50

# --- Params for TPOT Training ---
TPOT_GENERATIONS = 10
TPOT_POPULATION_SIZE = 10
TPOT_TIMEOUT_MINS = 240
TPOT_CONFIG_DICT = tpot_configs.regressor_config_decision_tree #'TPOT Light'