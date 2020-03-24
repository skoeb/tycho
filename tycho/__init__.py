"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

# --- Python Batteries Included---
import os
import logging

# --- External Libraries ---
import pandas as pd

# --- Module Imports ---
from tycho.config import *
import tycho.helper
from tycho.fetcher import EPACEMSFetcher, EarthEngineFetcher
from tycho.loader import PUDLLoader, CEMSLoader, GPPDLoader
from tycho.merger import TrainingDataMerger, RemoteDataMerger
from tycho.splitter import FourWaySplit
from tycho.sanitizer import ColumnSanitizer, OneHotEncodeWithThresh, DropNullColumns, apply_date_range_to_gppd
from tycho.featureengineer import CapacityFeatures, DateFeatures, calc_average_y_vals_per_MW, ApplyAvgY
from tycho.visualizer import plot_cems_emissions, plot_corr_heatmap, plot_eda_pair, plot_map_plants, plot_emission_factor

# --- Initialize Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs","tycho.txt")),
        logging.StreamHandler()
    ])

log = logging.getLogger("tycho")

# --- Hide Pandas Warning ---
pd.options.mode.chained_assignment = None
