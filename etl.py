"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
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
import logging

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee

# --- Module Imports ---
import tycho
from tycho.config import *

log = logging.getLogger("train")

def main():

    # --- Fetch EPA CEMS data if not present in 'data/CEMS/csvs' (as zip files) ---
    CEMSFETCHER = tycho.EPACEMSFetcher()
    CEMSFETCHER.fetch()

    # --- Load EIA 860/923 data from PUDL ---
    pudlloader = tycho.PUDLLoader()
    pudlloader.load()
    eightsixty = pudlloader.eightsixty

    # --- load CEMS data from pickle, or construct dataframe from csvs ---
    CEMS = tycho.CEMSLoader()
    CEMS.load()
    cems = CEMS.cems

    # --- Load WRI Global Power Plant Database data from csv ---
    GPPD = tycho.GPPDLoader() 
    GPPD.load()
    gppd = GPPD.gppd

    # --- Merge eightsixty, gppd, cems together into a long_df ---
    MERGER = tycho.TrainingDataMerger(eightsixty, gppd, cems)
    MERGER.merge()
    df = MERGER.df

    # --- Only keep n_generators worth of plants ---
    if tycho.N_GENERATORS != None:
        log.info(f'....reducing n_generators down to {N_GENERATORS} from config')
        _df = df.sample(1, random_state=42)
        plant_ids = list(set(_df['plant_id_eia']))
        keep = plant_ids[0:N_GENERATORS]
        df = df.loc[df['plant_id_eia'].isin(keep)]

    # --- Load Google Earth Engine Data using db for dates ---
    for earthengine_db in EARTHENGINE_DBS:
        EESCRAPER = tycho.EarthEngineFetcher(earthengine_db, buffers=BUFFERS)
        EESCRAPER.fetch(df)

    # --- Save to Pickle ---
    with open(os.path.join('processed','clean_df.pkl'), 'wb') as handle:
        pickle.dump(df, handle)

if __name__ == '__main__':
    main()

"""
TODO:
- Try on weekly basis
"""