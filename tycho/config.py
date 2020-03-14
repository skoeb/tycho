"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Number of Generators to aggregate ---
#   - most downloads are cached, so if you set a higher number, 
#     and then a lower, you don't lose data.
N_GENERATORS = 100

# --- Multiprocessing settings ---
MULTIPROCESSING = True
WORKERS = 6
THREADS = 12 #ThreadPoolExecutor is failing for Earth Engine queries, so this is still using ProcessPool

# --- Bool for detailed output ---
VERBOSE = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ EARTH ENGINE SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Sources ---
EARTHENGINE_DBS = [
    "COPERNICUS/S5P/OFFL/L3_NO2",
    "COPERNICUS/S5P/OFFL/L3_SO2",
    "COPERNICUS/S5P/OFFL/L3_HCHO",
    # "COPERNICUS/S5P/OFFL/L3_CO",
    # "COPERNICUS/S5P/OFFL/L3_O3",
    "CIESIN/GPWv411/GPW_Population_Count",
    "ECMWF/ERA5/DAILY"
] #53,125,404,151

# --- API Query settings ---
EE_TIMEOUT = 4 #forced timeout, overriding exponential backoff before calling a API query nans
RETRY_EE_NANS = True #after loading cache, retry queries that returned nans in last call

# --- Scale (in meters) to query ---
BUFFERS = [1e3, 1e5] #1e2, 1e4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ SANITIZE/SPLIT SETTINGS ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Whether to include some US only PUDL/EIA data in training ---
GEOGRAPHIC_SCOPE = 'domestic' # 'domestic' or 'international'


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

# --- Grid search params for XGB Training ---
XGB_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators':[500],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}