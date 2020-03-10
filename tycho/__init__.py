"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

# --- Python Batteries Included---
import os
import logging

# --- Module Imports ---
from tycho.config import *
import tycho.helper
from tycho.fetcher import EPACEMSFetcher, EarthEngineFetcher
from tycho.loader import PUDLLoader, CEMSLoader, GPPDLoader, EarthEngineLoader
from tycho.merger import TrainingDataMerger
from tycho.splitter import FourWaySplit


# --- Initialize Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs","sklearn_spoilage_log.txt")),
        logging.StreamHandler()
    ])