"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

# --- Python Batteries Included---
<<<<<<< HEAD
=======
import sqlite3
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
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

<<<<<<< HEAD
def etl():

    # --- establish SQL Connection ---
    SQL = tycho.PostgreSQLCon()
=======
def etl(sql_db='tycho_production'):

    # --- establish SQLite Connection ---
    SQL = tycho.SQLiteCon(sql_db)
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
    SQL.make_con()
    
    if config.RUN_PRE_EE:
        # --- Fetch EPA CEMS data if not present in 'data/CEMS/csvs' (as zip files) ---
        CemsFetch = tycho.EPACEMSFetcher()
        CemsFetch.fetch()

        # --- Load EIA 860/923 data from PUDL ---
        PudlLoad = tycho.PUDLLoader(SQL=SQL)
        PudlLoad.load()
        eightsixty = PudlLoad.eightsixty

        # --- load CEMS data from pickle, or construct dataframe from csvs ---
        CemsLoad = tycho.CEMSLoader(SQL=SQL)
        CemsLoad.load()
        cems = CemsLoad.cems

        # --- Load WRI Global Power Plant Database data from csv ---
        GppdLoad = tycho.GPPDLoader(SQL=SQL) 
        GppdLoad.load()
        gppd = GppdLoad.gppd
        
        # --- Merge eightsixty, gppd, cems together into a long_df ---
        TrainingMerge = tycho.TrainingDataMerger(eightsixty, gppd, cems)
        TrainingMerge.merge()
        df = TrainingMerge.df
        
<<<<<<< HEAD
        # --- Output to SQL ---
=======
        # --- Output to SQLite ---
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
        SQL.pandas_to_sql(df, 'etl_pre_L3')
    
    else:
        df = SQL.sql_to_pandas('etl_pre_L3')
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
    
    if config.FETCH_S3:
        # --- fetch S3 data ---
        SentinelFetcher = tycho.S3Fetcher()
        SentinelFetcher.fetch(df)

    # --- merge S3 data together ---
    SentinelMerger = tycho.L3Merger()
    SentinelMerger.merge(df)

    if config.RUN_BAYES_OPT:
        # --- Optimize L3 Conical Parameters (distance and angle of emission measurement) ---
        SentinelOptimzier = tycho.L3Optimizer()
        SentinelOptimzier.optimize(df)

    # --- aggregate and merge onto df ---
    SentinelCalculator = tycho.L3Loader()
    df = SentinelCalculator.calculate(df)
    
    # --- Save to SQL ---
    SQL.pandas_to_sql(df, 'etl_L3')

if __name__ == '__main__':
    etl()