"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

# --- Python Batteries Included---
import os
import logging

# --- Module Imports ---
import tycho.config as config
from tycho.etl import *
from tycho.ml import *
from tycho.output import *
from tycho.helper import *
from tycho.database import *

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
import pandas as pd
pd.options.mode.chained_assignment = None

# --- Dumb but clean, sklearn is throwing a few unavoidable warnings ---
import warnings
warnings.filterwarnings("ignore")