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
import hashlib
from pathlib import Path
import math
from datetime import timedelta 

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee
import shapely
from shapely.geometry import Point, Polygon
from rasterio.mask import mask
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
                 match_distance_thresh=config.DEGREES_DISTANCE_MATCH):

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
            'country_long',
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
        self.gppd = self.gppd.groupby(['plant_id_eia','plant_id_wri','country','country_long'], as_index=False).agg(agg_dict)

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

        # --- Drop plants with no emissions ---
        grouped_emissions = self.df.groupby('plant_id_wri')[['co2_lbs','nox_lbs','so2_lbs']].sum().sum(axis=1)
        nonzero_emissions = grouped_emissions.loc[grouped_emissions > 0]
        nonzero_emission_ids = list(nonzero_emissions.index)
        self.df = self.df.loc[self.df['plant_id_wri'].isin(nonzero_emission_ids)]

        # --- Drop plants outside of 5-95% operational time ---
        grouped_time = self.df.groupby('plant_id_wri')['operational_time'].sum()
        nonzero_time = grouped_time.loc[grouped_time > 0]
        nonzero_time_ids = list(nonzero_time.index)
        self.df = self.df.loc[self.df['plant_id_wri'].isin(nonzero_time_ids)]

        # --- convert back to DataFrame ---
        self.df = pd.DataFrame(self.df)
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ STITCH EARTH ENGINE ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EarthEngineDataMerger():
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
                ts_frequency=config.TS_FREQUENCY):
        self.earthengine_dbs = earthengine_dbs
        self.buffers = buffers
        self.ts_frequency = ts_frequency

        self.pickle_path = os.path.join('data','earthengine')
    
    def _read_pickles(self):

        # --- create hash of dbs to query ---
        db_string = ''.join(list(sorted(self.earthengine_dbs)))
        db_hash = int(hashlib.sha1(str.encode(db_string)).hexdigest(), 16) % (10 ** 8)
        db_hash = 'hash' + str(db_hash)

        files = os.listdir(self.pickle_path)  
        self.clean_files = []
        # --- find out which files to read in --
        for f in files:
            if '#' in f:
                h, ts, m = f.split('#')
                ts = ts.replace('agg', '')
                if ts == self.ts_frequency:
                    if h == db_hash:
                        self.clean_files.append(f) 

        # --- read files and concat ---
        dfs = []
        for f in self.clean_files:
            dfs.append(pd.read_pickle(os.path.join(self.pickle_path, f)))
        
        # --- concat dfs into long earthengine df ---
        if len(dfs) > 0:
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

        # --- convert emissions to lbs ---
        convert_columns = [c for c in self.merged.columns if 'column_number_density' in c]
        self.merged[convert_columns] = self.merged[convert_columns] * 24500000 * 46 / 454
        new_columns = [c.replace('column_number_density','sentinel_lbs') for c in self.merged.columns]
        self.merged.columns = new_columns

        return self
    
    def merge(self, df):
        self._read_pickles()
        self._pivot_buffers()
        self._merge_pivot(df)
        self._clean()
        return self.merged

 
class EarthEngineDataMergerLite():
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
                ts_frequency=config.TS_FREQUENCY):
        self.earthengine_dbs = earthengine_dbs
        self.buffers = buffers
        self.ts_frequency = ts_frequency

        self.pickle_path = os.path.join('data','earthengine')
    
    def _read_pickles(self):

        # --- create hash of dbs to query ---
        db_string = ''.join(list(sorted(self.earthengine_dbs)))
        db_hash = int(hashlib.sha1(str.encode(db_string)).hexdigest(), 16) % (10 ** 8)
        db_hash = 'hash' + str(db_hash)

        files = os.listdir(self.pickle_path)  
        self.clean_files = []
        # --- find out which files to read in --
        for f in files:
            if '#' in f:
                h, ts = f.split('#')
                ts = ts.replace('agg', '').replace('.pkl','')
                if ts == self.ts_frequency:
                    if h == db_hash:
                        self.clean_files.append(f) 

        # --- read files and concat ---
        dfs = []
        for f in self.clean_files:
            dfs.append(pd.read_pickle(os.path.join(self.pickle_path, f)))
        
        # --- concat dfs into long earthengine df ---
        if len(dfs) > 0:
            self.earthengine = pd.concat(dfs, axis='rows', sort=False)

        return self

    def _pivot_buffers(self, buffers=config.BUFFERS):
        """Pivot multiple buffers into a wider df."""

        # --- Filter out the buffers provided ---
        self.earthengine = self.earthengine.loc[self.earthengine['buffer'].isin(buffers)]

        # --- Pivot ---
        self.pivot = self.earthengine.pivot_table(index=['plant_id_wri','datetime_utc'], columns='variable',values='value')
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

        # --- convert emissions to lbs ---
        convert_columns = [c for c in self.merged.columns if 'column_number_density' in c]
        self.merged[convert_columns] = self.merged[convert_columns] * 24500000 * 46 / 454
        new_columns = [c.replace('column_number_density','sentinel_lbs') for c in self.merged.columns]
        self.merged.columns = new_columns

        return self
    
    def merge(self, df):
        self._read_pickles()
        self._pivot_buffers()
        self._merge_pivot(df)
        self._clean()
        return self.merged
  
class L3Merger():
    def __init__(self,
                 s3_dbs=config.S3_DBS,
                 raw_s3_local_path=config.RAW_S3_DIR, 
                 checkpoint_freq=0.05,
                 if_already_exist='skip',
                 resample_shape=(5143,10286),
                 downcast_dtype='uint8',
                 ):
        
        assert if_already_exist in ['skip', 'replace']
        
        self.s3_dbs = s3_dbs
        self.raw_s3_local_path = raw_s3_local_path
        self.checkpoint_freq = checkpoint_freq
        self.if_already_exist = if_already_exist
        self.resample_shape = resample_shape
        self.downcast_dtype = downcast_dtype
        
        self.db_mosaics = {}
        self.db_rasters = {}

        self.out_file_dir = os.path.join('data','s3','clean')
    
    def _load_basemap(self):
        """Read in blank basemap, to merge satelite images with."""
        self.blank_world = rasterio.open(os.path.join('data','s3','blankCOGT.tiff'))
        return self
    
    def _worker(self, job):
        """Load rasters, merge, and save to disk for a given date and db."""

        # --- unpack job ---
        sanitized_db, date = job
        
        # --- list all satellite image COGT fragments for date ---
        y = str(date.year).zfill(2)
        m = str(date.month).zfill(2)
        d = str(date.day).zfill(2)

        db_dir = os.path.join(self.raw_s3_local_path, sanitized_db)
        date_dir = os.path.join(db_dir, str(y), str(m), str(d))
        files = os.listdir(date_dir)
        
        # --- filter out inapplicable files ---
        files = [f for f in files if '.tif' in f]
        files = [f for f in files if 'PRODUCT_qa_value' not in f]
        
        # --- open files with rasterio ---
        src_files = []
        for f in files:
            
            # --- read rasters ---
            srcr = rasterio.open(os.path.join(date_dir, f))         
            src_files.append(srcr)
        
        # --- merge into a single raster --- #TODO: Confirm rasterio behavior for overlaps
        mosaic, mosaic_transform = merge(src_files + [self.blank_world]) #https://gis.stackexchange.com/questions/360685/creating-daily-mosaics-from-orbiting-satellite-imagery
        
        # --- replace null values ---
        null_val = srcr.nodata
        mosaic[mosaic==null_val] = np.nan
        
        # --- copy meta data ---
        profile = srcr.meta.copy()
        profile.update(
                    width=mosaic.shape[2],
                    height=mosaic.shape[1],
                    transform=mosaic_transform)
        
        # --- resample in memory ---
        if self.resample_shape != None:

            with MemoryFile() as memfile:
                with memfile.open(**profile) as dataset:
                    dataset.write(mosaic)
                
                dataset = memfile.open()
                
            mosaic = dataset.read(
                out_shape=(
                    dataset.count,
                        self.resample_shape[0],
                        self.resample_shape[1]
                    # int(dataset.height * self.resample_scale),
                    # int(dataset.width * self.resample_scale)
                ),
                resampling=Resampling.bilinear
            )
        
            # --- scale image transform ---
            mosaic_transform = dataset.transform * dataset.transform.scale(
                                    (dataset.width / mosaic.shape[-1]),
                                    (dataset.height / mosaic.shape[-2])
                                )
            
            del dataset
        
        if self.downcast_dtype != 'float64':

            # --- clip ---
            mosaic = np.clip(mosaic, np.percentile(mosaic, 1), np.percentile(mosaic, 99))
            
            # --- normalize ---
            downcast_min_val = np.iinfo(self.downcast_dtype).min
            downcast_max_val = np.iinfo(self.downcast_dtype).max
            Scaler = MinMaxScaler(feature_range=(downcast_min_val, downcast_max_val))
            mosaic = np.squeeze(mosaic)
            mosaic = Scaler.fit_transform(mosaic)
            
            # --- downcast ---
            mosaic = mosaic.astype(self.downcast_dtype)
            
            # --- Make 3D as rasterio expects ---
            mosaic = np.expand_dims(mosaic, axis=0)
            
            # --- update profile again ---
            profile.update(
                width=mosaic.shape[2],
                height=mosaic.shape[1],
                transform=mosaic_transform,
                dtype=self.downcast_dtype,
                nodata=0)
    
        # --- save to disk ---
        out_db = sanitized_db.split(os.path.sep)[-2].replace('L2','L3')
        out_filename = f"{out_db}{date.strftime('%Y-%m-%d')}.tif"
        out_fp = os.path.join(self.out_file_dir, out_filename)
        
        with rasterio.open(out_fp, "w", **profile) as fp:
            fp.write(mosaic)
        
        return self
        
    
    def merge(self, df):
        """Main wrapper."""

        dates = pd.date_range(df['datetime_utc'].min(), df['datetime_utc'].max())
        
        # --- load basemap ---
        self._load_basemap()
        
        # --- Construct list of jobs ---
        jobs = []
        for db in self.s3_dbs:
            
            # --- clean db name ---
            sanitized_db = db.replace('/', os.path.sep)
            out_db = db.split('/')[-2].replace('L2','L3')
        
            # --- for each date ---
            for date in dates:
        
                # --- check if file already exists ---
                out_filename = f"{out_db}{date.strftime('%Y-%m-%d')}.tif"
                out_fp = os.path.join(self.out_file_dir, out_filename)

                if os.path.exists(out_fp):
                    if self.if_already_exist == 'skip':
                        continue
                    elif self.if_already_exist == 'replace':
                        os.remove(out_fp)
                
                # --- add to jobs ---
                job = (sanitized_db, date)
                jobs.append(job)
    
        # --- run jobs ---
        log.info(f"....{len(jobs)} jobs queued for merging")
        if config.MULTIPROCESSING:
            
            start = time.time()
            completed = 0
            checkpoint = max(1, int(len(jobs) * self.checkpoint_freq))
            
            with cf.ThreadPoolExecutor(max_workers=config.THREADS) as executor: 
                
                # --- Submit to worker ---
                futures = [executor.submit(self._worker, job) for job in jobs]
                
                for f in cf.as_completed(futures):
                    completed += 1
                    
                    if completed % checkpoint == 0:
                        per_download = round((time.time() - start) / completed, 3)
                        eta = round((len(jobs) - completed) * per_download / 3600, 3)
                        log.info(f"........finished job {completed} / {len(jobs)}  ETA: {eta} hours")
        else:
            for job in jobs:
                self._worker(job)

        return self