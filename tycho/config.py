"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

#TODO:
# drop index column from train.py
# plot training data against actual
# 1 day earth engine, with 8PM offset?
# try with CEMS observed plants only
# test corr for lots of buffers for small number of plants (25?)
# integrate linear regression
# store data on google cloud and write function to pull?
# make bokeh map

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Number of generators for training set ---
#   - most downloads are cached, so if you set a higher number, 
#     and then a lower, you don't lose data.
RUN_PRE_EE = True
N_GENERATORS = None

# --- Multiprocessing settings ---
MULTIPROCESSING = True
WORKERS = 12
THREADS = 12 #ThreadPoolExecutor is failing for Earth Engine queries, so this is still using ProcessPool

# --- Bool for detailed output ---
VERBOSE = False

# --- Frequency of observations (i.e. 'D' for daily, 'W', for weekly, 'M' for monthly, 'A' for annual) --- 
TS_FREQUENCY = 'MS'
if TS_FREQUENCY in ['W','W-SUN']:
    TS_DIVISOR = 52
elif TS_FREQUENCY in ['MS']:
    TS_DIVISOR = 12
elif TS_FREQUENCY in ['D']:
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
PREDICT_COUNTRIES = ['Puerto Rico', 'Cuba', 'Dominican Republic', 'Jamaica', 'Colombia', 'Venezuela']

PREDICT_START_DATE = '01-01-2019'
PREDICT_END_DATE = '03-01-2020'

CEMS_MEASUREMENT_FLAGS = ['Measured'] #blank for include everything

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ EARTH ENGINE SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Sources ---
EARTHENGINE_DBS = [
    "COPERNICUS/S5P/OFFL/L3_CLOUD",
    "COPERNICUS/S5P/OFFL/L3_NO2",
    "COPERNICUS/S5P/OFFL/L3_SO2",
    "COPERNICUS/S5P/OFFL/L3_CO",
    "COPERNICUS/S5P/OFFL/L3_HCHO",
    "COPERNICUS/S5P/OFFL/L3_O3",
    "CIESIN/GPWv411/GPW_Population_Count",
    "ECMWF/ERA5/DAILY"
]

# --- API Query settings ---
EE_TIMEOUT = 120 #forced timeout, overriding exponential backoff before calling a API query nans
RETRY_EE_NANS = False #after loading cache, retry queries that returned nans in last call

# --- Scale (in meters) to query ---
BUFFERS = [10000, 20000] #1e2, 1e4

# --- Hour to grab after --- (i.e. don't consider an observation unless it is after this hour)
LEFT_WINDOW_HOUR = 20

# --- Degrees to match GPPD and EIA dataframes ---
DEGREES_DISTANCE_MATCH = 0.005

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
BAYES_N_ITER = 5
BAYES_INIT_POINTS = 3
BAYES_ACQ = 'ucb'

# --- Params for TPOT Training ---
TPOT_GENERATIONS = 100
TPOT_POPULATION_SIZE = 100
TPOT_TIMEOUT_MINS = 60*24
# TPOT_CONFIG_DICT = tpot_configs.lgbm_config_dict
