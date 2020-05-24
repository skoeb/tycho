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


# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee

# --- Module Imports ---
import tycho
import tycho.config as config

import logging
log = logging.getLogger("tycho")

def etl():
    
    if config.RUN_PRE_EE:
        # --- Fetch EPA CEMS data if not present in 'data/CEMS/csvs' (as zip files) ---
        CemsFetch = tycho.EPACEMSFetcher()
        CemsFetch.fetch()

        # --- Load EIA 860/923 data from PUDL ---
        PudlLoad = tycho.PUDLLoader()
        PudlLoad.load()
        eightsixty = PudlLoad.eightsixty

        # --- load CEMS data from pickle, or construct dataframe from csvs ---
        CemsLoad = tycho.CEMSLoader()
        CemsLoad.load()
        cems = CemsLoad.cems

        # --- Load WRI Global Power Plant Database data from csv ---
        GppdLoad = tycho.GPPDLoader() 
        GppdLoad.load()
        gppd = GppdLoad.gppd
        
        # --- Merge eightsixty, gppd, cems together into a long_df ---
        TrainingMerge = tycho.TrainingDataMerger(eightsixty, gppd, cems)
        TrainingMerge.merge()
        df = TrainingMerge.df
        
        # --- Output pickle ---
        df.to_pickle(os.path.join('processed','pre_ee.pkl'), protocol=4)
    
    else:
        df = pd.read_pickle(os.path.join('processed','pre_ee.pkl'))
        df = df.loc[df['datetime_utc'] < pd.datetime(config.MAX_YEAR+1 ,1, 1)]

    # --- Only keep n_generators worth of plants ---
    if tycho.config.N_GENERATORS != None:
        # --- Create list of plant_id_wri ---
        plant_ids = list(set(df['plant_id_wri']))
        plant_ids = sorted(plant_ids)

        # --- Shuffle ---
        random.Random(42).shuffle(plant_ids)

        # --- Subset list ---
        log.info(f"....reducing n_generators down from {len(plant_ids)} from config")
        keep = plant_ids[0:config.N_GENERATORS]

        # --- Subset df ---
        df = df.loc[df['plant_id_wri'].isin(keep)]
        log.info(f"....n_generators now {len(list(set(df['plant_id_wri'])))} from config")

    #--- Load Google Earth Engine Data (such as weather and population) using df for dates ---
    EarthEngineFetch = tycho.EarthEngineFetcherLite()
    EarthEngineFetch.fetch(df)

    # --- Merge Remote Sensing (Earth Engine) Data onto df ---
    EarthEngineMerge = tycho.EarthEngineDataMergerLite()
    df = EarthEngineMerge.merge(df)

    # --- fetch S3 data ---
    # SentinelFetcher = tycho.S3Fetcher()
    # SentinelFetcher.fetch(df)

    # --- merge S3 data together ---
    SentinelMerger = tycho.L3Merger()
    SentinelMerger.merge(df)

    if config.RUN_BAYES_OPT:
        # --- Optimize L3 Conical Parameters (distance and angle of emission measurement) ---
        SentinelOptimzier = tycho.L3Optimizer()
        SentinelOptimzier.optimize(df)

    # --- aggregate and merge onto df ---
    SentinelCalculator = tycho.L3Loader()
    df = SentinelCalculator.serial_calculate(df)
    
    # --- Save to Pickle ---
    with open(os.path.join('processed','merged_df.pkl'), 'wb') as handle:
        pickle.dump(df, handle)

    # --- Generate Plots ---
    tycho.plot_cems_emissions(df)
    tycho.plot_corr_heatmap(df)
    tycho.plot_eda_pair(df)
    tycho.plot_map_plants(gppd)

if __name__ == '__main__':
    etl()