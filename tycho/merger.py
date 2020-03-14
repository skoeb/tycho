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
import tycho.config as config
import tycho.helper as helper

import logging
log = logging.getLogger("tycho")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ MERGE DATA TOGETHER ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class TrainingDataMerger():
    """
    Merge all possible cached data for training, including:
        - EIA 860/923 data returned from PUDLLoader()
        - WRI Global Powerplant Database Data returned from GPPDLoader()
        - EPA Continuous Emission Monitoring System Target Data 

    Inputs
    ------
    eightsixty (GeoDataFrame) - returned from loader.PUDLLoader()
    gppd (GeoDataFrame) - returned from loader.GPPDLoader()
    cems (DataFrame) - returned from CEMSLoader()
    match_distance_thresh (float) - distance (in degrees) to match powerplant lat/lons
        for instance if gppd says a plant is at (-50.01, 39.27) and EIA says a plant is at (-50.0, 39.272),
        they are probably the same and should be matched

    Methods
    -------
    merge()

    Attributes
    ----------
    self.df - merged GeoDataFrame
    """
    def __init__(self, eightsixty, gppd, cems,
                 match_distance_thresh=0.01):

        log.info('\n')
        log.info('Initializing TrainingDataMerger')

        self.eightsixty = eightsixty
        self.gppd = gppd
        self.cems = cems

        self.match_distance_thresh = match_distance_thresh
        
        
    def _make_df_points(self):
        # --- Drop duplicates (as strings for performance) ---
        _df = self.df.copy()
        _df['wkt'] = self.df['geometry'].apply(lambda x: x.wkt).values
        self.unique = _df.drop_duplicates(subset=['wkt'])

        # --- Calc unions of points ---
        self.df_points = self.unique.unary_union
        return self

    
    def _nearest_point_worker(self, gppd_point):
        # add thresh for max distance
        _, match = nearest_points(gppd_point, self.df_points)
        dist = gppd_point.distance(match)
        
        if dist < self.match_distance_thresh:
            matched_plant_id = self.unique.loc[self.unique['geometry'] == match, 'plant_id_eia'].values[0]
            return (gppd_point, matched_plant_id)
        
        else:
            return (gppd_point, np.nan)

    
    def _duplicate_for_dates(self, df, dt_range):
        to_concat = []
        for d in dt_range:
            
            # --- Subset report year df ---
            y = pd.Timestamp(d).year
            _df = df.copy()
            _df['date'] = d
            to_concat.append(_df)

        long_df = pd.concat(to_concat, axis='rows')
        return long_df
    
    
    def merge(self):

        # --- Merge cems and eightsixty on eia plant id ---
        log.info("....beginning merge process between cems and eightsixty")
        log.info(f"........pre-merge generator count in eightsixty: {len(set(self.eightsixty['plant_id_eia']))}")
        log.info(f"........pre-merge generator count in cems: {len(set(self.cems['plant_id_eia']))}")
        self.df = self.eightsixty.merge(self.cems, on=['plant_id_eia'], how='right')
        log.info(f"........post-merge generator count: {len(set(self.df['plant_id_eia']))}")
        
        # --- Drop CEMS data not in eightsixty ---
        log.info(f"........pre-drop generator count: {len(set(self.df['plant_id_eia']))}")
        self.df = self.df.dropna(subset=['plant_name_eia'])
        log.info(f"........post-drop generator count: {len(set(self.df['plant_id_eia']))}")

        # --- Create list of known points in self.df ---
        log.info('....making df points list.')
        self._make_df_points()
        
        # --- Find nearest df plant for plants in gppd ---
        log.info('....finding nearest neighboring plants between gppd and df.')

        jobs = list(self.gppd['geometry'].unique())
        if config.MULTIPROCESSING:
            results_list = []
            with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as executor:
                ten_percent = max(1, int(len(jobs) * 0.1))

                # --- Submit to worker ---
                futures = [executor.submit(self._nearest_point_worker, job) for job in jobs]
                for f in cf.as_completed(futures):
                    results_list.append(f.result())
                    if len(results_list) % ten_percent == 0:
                        log.info('........finished point matching job {} / {}'.format(len(results_list), len(jobs)))
        else:
            results_list = [self._nearest_point_worker(job) for job in jobs]
        
        results_dict = {k.wkt :v for k,v in results_list}
        self.gppd['wkt'] = self.gppd['geometry'].apply(lambda x: x.wkt).values
        self.gppd['plant_id_eia'] = self.gppd['wkt'].map(results_dict)

        # --- Filter out plants that no match was found ---
        log.info(f"........pre-drop generator count in gppd: {len(self.gppd)}")
        self.gppd = self.gppd.dropna(subset=['plant_id_eia'])
        log.info(f"........post-drop generator count in gppd: {len(self.gppd)}")

        # --- Drop geometry from gppd to avoid duplicate columns ---
        keep_cols = [
            'wri_capacity_mw',
            'primary_fuel',
            'commissioning_year',
            'generation_gwh_2013',
            'generation_gwh_2014',
            'generation_gwh_2015',
            'generation_gwh_2016',
            'generation_gwh_2017',
            'estimated_generation_gwh',
            'plant_id_eia',
            'plant_id_wri',
            'country',
        ]

        # --- Drop unneeded cols ---
        self.gppd = self.gppd[keep_cols]

        # --- gppd groupby plant_id_eia to aggregate generators that were matched together ---
        agg_dict = { 
            'wri_capacity_mw':'sum',
            'primary_fuel':'first',
            'commissioning_year':'mean',
            'generation_gwh_2013':'sum',
            'generation_gwh_2014':'sum',
            'generation_gwh_2015':'sum',
            'generation_gwh_2016':'sum',
            'generation_gwh_2017':'sum',
            'estimated_generation_gwh':'sum',
        }
        self.gppd = self.gppd.sort_values('wri_capacity_mw', ascending=False)
        self.gppd = self.gppd.groupby(['plant_id_eia','plant_id_wri','country'], as_index=False).agg(agg_dict)

        #TODO: consider just dropping these rather than the groupby above
        # # --- Filter out plants that duplicate eightsixty was found ---
        # log.info(f"........pre-drop generator count in df: {len(set(self.df['plant_id_eia']))}")
        # self.gppd = self.gppd.drop_duplicates(subset=['plant_id_eia'], keep='first')
        # log.info(f"........post-drop generator count in df: {len(set(self.df['plant_id_eia']))}")

        # --- Merge on plant_id_eia ---
        log.info(f"........pre-merge generator count in df: {len(set(self.df['plant_id_eia']))}")
        self.df = self.df.merge(self.gppd, on='plant_id_eia', how='inner')
        log.info(f"........post-merge generator count in df: {len(set(self.df['plant_id_eia']))}")
        
        # --- Drop plants where difference between WRI capacity and EIA capacity is greater than 40% of WRI capacity ---
        log.info(f"........pre-drop generator count in df: {len(set(self.df['plant_id_eia']))}")
        self.df['diff'] = self.df['wri_capacity_mw'] - self.df['capacity_mw']
        self.df['diff'] = self.df['diff'].abs()
        self.df['thresh'] = self.df['wri_capacity_mw'] * 0.40
        self.df = self.df[self.df['diff'] <= self.df['thresh']]
        self.df = self.df.drop(['diff','thresh'], axis='columns')
        log.info(f"........post-merge generator count in df: {len(set(self.df['plant_id_eia']))}")

        # --- convert back to DataFrame ---
        self.df = pd.DataFrame(self.df)
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ STITCH EARTH ENGINE ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RemoteDataMerger():
    """
    Convert pickled monthly long_dfs containing earth engine data as 'value' and 'variable'
    columns into a wide_df, grouped by datetime_utc and plant_id_eia, 
    with multiple columns per buffer. 

    Inputs
    ------
    earthengine_dbs (list) - the list of earthengine databases to fetch (i.e. "COPERNICUS/S5P/OFFL/L3_NO2").
    buffers (list) - list of buffer sizes (in meters) to load. Each pickle has multiple buffers stored within it. 
    ts_frequency (str) - the desired pandas frequency of resampling, hardcoded into pickle names. 

    Methods
    -------
    merge() - perform pickle loading, merging, cleaning, and pivoting

    Attributes
    -------
    self.clean_files (list) - the list of files that have been loaded. Each formatted like:
        <EARTHENGINE_DB>&agg<TS_FREQUENCY>&<MONTH>.pkl
    self.merged (DataFrame) - merged and cleaned long_df
    self.pivot (DataFrame) - merged, pivoted into a wide_df

    Assumptions
    -----------
    - all of the requested buffers and earthengine_dbs have already been run thourh EarthEngineFetcher()
    - plant_id_eia is not purposefully duplicated
    """
    def __init__(self,
                earthengine_dbs=config.EARTHENGINE_DBS,
                buffers=config.BUFFERS,
                ts_frequency='D'):
        self.earthengine_dbs = [i.replace('/','-') for i in earthengine_dbs]
        self.buffers = buffers
        self.ts_frequency = ts_frequency

        self.pickle_path = os.path.join('data','earthengine')
    
    def _read_pickles(self):
        files = os.listdir(self.pickle_path)  
        self.clean_files = []
        # --- find out which files to read in --
        for f in files:
            if '&' in f:
                db, ts, m = f.split('&')
                ts = ts.replace('agg', '')
                if (db in self.earthengine_dbs) & (ts == self.ts_frequency):
                    self.clean_files.append(f) 

        # --- read files and concat ---
        dfs = []
        for f in self.clean_files:
            dfs.append(pd.read_pickle(os.path.join(self.pickle_path, f)))
        
        # --- concat dfs into long earthengine df ---
        self.earthengine = pd.concat(dfs, axis='rows', sort=False)

        return self

    def _pivot_buffers(self, buffers=config.BUFFERS):
        """Pivot multiple buffers into a wider df."""

        # --- Filter out the buffers provided ---
        self.earthengine = self.earthengine.loc[self.earthengine['buffer'].isin(buffers)]

        # --- Pivot ---
        self.earthengine['buffer_variable'] = self.earthengine['buffer'].astype(int).astype(str) + "/" + self.earthengine['variable']
        self.pivot = self.earthengine.pivot_table(index=['plant_id_wri','datetime_utc'], columns='buffer_variable',values='value')
        self.pivot.reset_index(drop=False, inplace=True)
        return self

    def _merge_pivot(self, df):
        """Merge pivot onto df (continaing generators and CEMS if training)."""
        self.merged = df.merge(self.pivot, on=['plant_id_wri', 'datetime_utc'])
        return self

    def _clean(self):

        # --- Drop any duplicates ---
        self.merged = self.merged.drop_duplicates(subset=['plant_id_wri','datetime_utc'])

        # --- Drop any nans ---
        self.merged = self.merged.dropna(subset=['plant_id_wri','datetime_utc'])

        return self
    
    def merge(self, df):
        self._read_pickles()
        self._pivot_buffers()
        self._merge_pivot(df)
        self._clean()
        return self.merged