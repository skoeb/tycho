"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

import os
from pathlib import Path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Number of generators for training set ---
#   - most downloads are cached, so if you set a higher number, 
#     and then a lower, you don't lose data.
RUN_PRE_EE = True
FETCH_S3 = True
RUN_BAYES_OPT = True
N_GENERATORS = None
MAX_YEAR = 2019
<<<<<<< HEAD
SCHEMA = 'production'
=======
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28

# --- Multiprocessing settings ---
MULTIPROCESSING = True
WORKERS = 12
THREADS = 6 #ThreadPoolExecutor is failing for Earth Engine queries, so this is still using ProcessPool

# --- Bool for detailed output ---
VERBOSE = False

# --- Frequency of observations (i.e. 'D' for daily, 'W', for weekly, 'M' for monthly, 'A' for annual) --- 
TS_FREQUENCY = '1W'
if TS_FREQUENCY in ['1W']:
    TS_DIVISOR = 52
elif TS_FREQUENCY in ['MS']:
    TS_DIVISOR = 12
elif TS_FREQUENCY in ['D', '1D']:
    TS_DIVISOR = 365
elif TS_FREQUENCY in ['A','AS']:
    TS_DIVISOR = 1
elif TS_FREQUENCY == '3D':
    TS_DIVISOR = int(365 / 3)
elif TS_FREQUENCY == '2D':
    TS_DIVISOR = int(365 / 2)
else:
    raise NotImplementedError(f'Please write a wrapper for {TS_FREQUENCY}!')

TRAIN_COUNTRIES = ['United States of America']
<<<<<<< HEAD
PREDICT_COUNTRIES = ['WORLD'] #['Colombia', 'Venezuela', 'Costa Rica', 'Guatemala', 'Jamaica', 'Puerto Rico']
=======
PREDICT_COUNTRIES = None
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28

PREDICT_START_DATE = '01-01-2019'
PREDICT_END_DATE = '12-31-2019'

<<<<<<< HEAD
CEMS_MEASUREMENT_FLAGS = [] #blank for include everything
=======
CEMS_MEASUREMENT_FLAGS = ['Measured'] #blank for include everything
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28

FETCH_SENTINEL_FROM = 'S3'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ EARTH ENGINE SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Sources ---
EARTHENGINE_DBS = [
    "CIESIN/GPWv411/GPW_Population_Count",
    "ECMWF/ERA5/DAILY"
]

# --- API Query settings ---
EE_TIMEOUT = 120 #forced timeout, overriding exponential backoff before calling a API query nans
RETRY_EE_NANS = False #after loading cache, retry queries that returned nans in last call

# --- Scale (in meters) to query ---
BUFFERS = [20000] #1e2, 1e4

# --- Hour to grab after --- (i.e. don't consider an observation unless it is after this hour)
LEFT_WINDOW_HOUR = 20

# --- Degrees to match GPPD and EIA dataframes ---
DEGREES_DISTANCE_MATCH = 0.005

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ L3 SETTINGS ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RAW_S3_DIR = os.path.join(str(Path.home()),'HDD','tycho_raw')

S3_DBS = [
        'COGT/OFFL/L2__SO2___/',
        # 'COGT/OFFL/L2__O3____/', #available 2019-04 on
        'COGT/OFFL/L2__NO2___/',
        'COGT/OFFL/L2__HCHO__/',
        'COGT/OFFL/L2__CH4___/',
        'COGT/OFFL/L2__CO____/',
        # 'COGT/OFFL/L2__CLOUD_/' #available 2019-04 on 
        'COGT/OFFL/L2__AER_AI/'
        ]

S3_BUCKET = 'meeo-s5p'

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
HOW_TO_SPLIT = 0.1 #test frac
TRAIN_MODEL = 'bayes-xgb'
PREDICT_MODEL = 'bayes-xgb'
ML_Y_COLS = ['gross_load_mw','so2_lbs','nox_lbs','co2_lbs']
CV_FOLDS = 3

RANDOMSEARCH_ITER = 50

# --- Params for Bayes training ---
BAYES_N_ITER = 50
BAYES_INIT_POINTS = 100
BAYES_ACQ = 'ucb'

# --- Params for TPOT Training ---
TPOT_GENERATIONS = 100
TPOT_POPULATION_SIZE = 100
TPOT_TIMEOUT_MINS = 60*24
# TPOT_CONFIG_DICT = tpot_configs.lgbm_config_dict
