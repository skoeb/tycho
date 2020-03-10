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
import tycho.config as config
import tycho.helper as helper

log = logging.getLogger("fetcher")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ FETCH EPA CEMS DATA ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EPACEMSFetcher():
    """
    The EPA requires certain thremal generators within the U.S. to report
    certain emission outputs on an hourly basis, including NOx, SO2, and CO2.

    Many of these values are measured, however some are calculated or imputed based
    on known emission outputs and MWh of production (also reported on an hourly basis).

    This data will serve as the training target (y values) for our supervised learning model.

    We will build an ML pipeline that trains on this data, and can predict for the rest
    of the world (where we don't have EPA CEMS data). 

    EPACEMSScraper scrapes .csvs from an FTP server hosted by the EIA.
    csvs are segmented by state and month. This script will download any
    files that are not present, and skip files that are already downloaded. 

    If the EPA ever updates files (which they often do around the September/October)
    timeframe for the previous year, it is responsible to delete the files locally
    and rerun the scraper to download new files. 

    Inputs
    ------
    server (str) - EPA FTP server
    server_dir (str) - Directory on FTP server to scrape all files in
    download_path (str) - Where to download files to locally
    years (list) - List of years to scrape data for (which is stored monthly)

    Methods
    -------
    fetch() - Connect to FTP and loop through years, downloading data with caching

    Attributes
    ----------
    None, use CEMSLoader() to load the downloaded files

    Assumptions
    -----------
    - if a file name matches locally (cached) and on the FTP, it's identical
    """

    def __init__(self, server='newftp.epa.gov',
                 server_dir='DMDnLoad/emissions/hourly/monthly/',
                 download_path=os.path.join('data','CEMS','csvs'),
                 years=[2019]):
        
        log.info('\n')
        log.info('Initializing EPACEMSScraper')

        self.server=server
        self.server_dir = server_dir
        self.download_path = download_path
        self.years=years
    
    def _connect_to_ftp(self):
        self.ftp = ftplib.FTP(self.server)
        self.ftp.login()
        log.info('....connected to EPA CEMS FTP Server')
        return self
    
    def _cwd_annual(self, year):
        year_server_dir = self.server_dir + str(year) + '/'
        self.ftp.cwd(year_server_dir)
        files = self.ftp.nlst()
        log.info('........downloaded file list from EPA CEMS FTP Server')
        return files
    
    def _already_downloaded(self, files):
        """Check what is already downloaded and skip it."""
        downloaded = os.listdir(self.download_path)
        needed = [f for f in files if f not in downloaded]
        log.info(f"....{len(needed)} files needed, {len(downloaded) - len(needed)} files already downloaded")
        return needed
        
    
    def _worker(self, f):
        self.ftp.retrbinary("RETR " + f, open(os.path.join(self.download_path, f), "wb").write)
        time.sleep(0.5)
        return
    
    def fetch(self):
        # --- Connect to FTP ---
        self._connect_to_ftp()
        
        # --- Loop through years ---
        for y in self.years:
            log.info(f"....working on {y}")
            jobs = self._cwd_annual(y)
            jobs = self._already_downloaded(jobs) # see what is already downloaded and update jobs
        
            # --- Download monthly/state files ---
            jobs_complete = 0 
            ten_percent = max(1, int(len(jobs) * 0.1))
            
            for job in jobs: #FTP limits connections, so multiprocessing doesn't work
                self._worker(job)
                jobs_complete += 1
                    
                if jobs_complete % ten_percent == 0:
                    log.info('........finished EPA CEMS download {} / {}'.format(jobs_complete, len(jobs)))
        log.info(f"....finished all downloads")
        return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ FETCH EARTH ENGINE DATA ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EarthEngineFetcher():
    """
    Fetch Earth Engine data for a geography.
    
    Inputs
    ------
    df (string) - From https://developers.google.com/earth-engine/datasets/catalog
    scale (int) - Granularity of calculating average values within geography
    buffers (list) - size in meters to return data for (i.e. 1e3 for 1km)
    days_combine (int) - Number of days to combine when calculating aggregated sattelite data
        agg_func is applied to the returned collection of images.
    agg_func (string) - How to calculate the daily data when aggregating multiple
        satelitte swaths together. 

    Methods
    -------
    fetch() - 
        accepts a df (geopandas.GeoDataFrame) - Dataframe with 'generator_id', 'geometry', and 'date' columns.

    Attributes
    ----------
    """
    
    def __init__(self, earthengine_db='COPERNICUS/S5P/OFFL/L3_NO2',
                 agg_func='median',
                 scale=50, tilescale=16,
                 buffers=[1e2, 1e3, 1e4, 1e5],
                 id_col='plant_id_eia',
                 geo_col='geometry',
                 date_col='datetime_utc',
                 read_cache=True, use_cache=True):

        log.info('\n')
        log.info('Initializing GetDailyEarthEngineData')

        try:
            ee.Initialize()
        except Exception as e:
            ee.Authenticate()
            ee.Initialize()

        self.earthengine_db = earthengine_db
        self.agg_func = agg_func
        self.tilescale = tilescale
        self.scale = scale
        self.buffers=buffers
        self.read_cache = read_cache
        self.use_cache = use_cache
        
        self.id_col=id_col
        self.geo_col=geo_col
        self.date_col=date_col

        
    def _load_image(self, date):
        """Load Earth Edge image."""

        # --- Make dates strings so google is happy ---
        start_date = date #.strftime('%m-%d-%Y')
        next_date = (pd.Timestamp(date) + pd.tseries.offsets.DateOffset(days=1)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 

        # --- Load Image and add as layer ---
        imagecollection = ee.ImageCollection(self.earthengine_db)
        date_agg = imagecollection.filterDate(start_date, next_date)

        if self.agg_func == 'median':
            image = date_agg.median()
        else:
            raise NotImplementedError(f"please write a wrapper for {self.agg_func}!")
        return image


    def _load_geometry(self, geometry, buffer):
        """Return geometry point object."""
        lon = geometry.x
        lat = geometry.y
        geometry = ee.Geometry.Point(lon, lat).buffer(buffer)
        return geometry

    
    def _calc_geography_mean(self, date_agg, geometry, buffer):
        """
        Compute Aggregation.

        Inputs
        ------
        date_agg (ee.Image) - Image object that has been filtered for a date period. 
        geometry (ee.Geometry) - Geometry object that indicates the area of interest.
        scale (int) - Integer representing the level of the detail, higher represents lower
            representation.

        Returns
        -------
        dict - keys are bands, values are the mean for that buffer size. 
        """

        average_dict = date_agg.reduceRegion(**{
          'reducer': ee.Reducer.mean(),
          'geometry':geometry,
          'scale': self.scale,
          'bestEffort':True,
          'tileScale':self.tilescale
        })
        average_dict = average_dict.getInfo()
        return average_dict
    
    
    def _worker(self, row):
        """Returns a dict with keys as buffer size, and values of dicts of band values."""
        try:
            geometry = self._load_geometry(row[self.geo_col], row['buffer'])
            date_agg = self._load_image(row[self.date_col])
            
            # --- Download results and add identifier features ---
            result = self._calc_geography_mean(date_agg, geometry, row['buffer'])
            result[self.id_col] = row[self.id_col]
            result[self.geo_col] = row[self.geo_col]
            result[self.date_col] = row[self.date_col]
            result['buffer'] = row['buffer']

        except Exception as e:
            log.info(f"WARNING! Failed on {row['datetime_utc']}, {e} \n")
            return {}

        return result
        

    def _run_jobs(self, jobs_df, month):
        log.info(f'....starting earth engine jobs')
        results = []
        if config.MULTIPROCESSING:
            ten_percent = max(1, int(len(jobs_df) * 0.1))
            with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as executor:
                # --- Submit to worker ---
                futures = [executor.submit(self._worker, row) for _, row in jobs_df.iterrows()]
                for f in cf.as_completed(futures):
                    results.append(f.result())
                    if len(results) % ten_percent == 0:
                        log.info(f"........finished earth engine job {len(results)} / {len(jobs_df)} month: {month}")
        else:
            results = [self._worker(row) for _, row in jobs_df.iterrows()]
            
        return results
    
    def _clean_results(self, results):

        # --- Read list of dicts into DataFrame ---
        results = pd.DataFrame(results)

        # --- Make long ---
        id_vars = [self.id_col, self.date_col, self.geo_col, 'buffer']
        results = pd.melt(results, id_vars)

        # --- Drop duplicates ---
        results = results.drop_duplicates(subset=['plant_id_eia','datetime_utc','buffer','variable'])

        # --- Drop nans ---
        results = results.dropna(subset=['variable','value'], how='any')

        # --- Make small ---
        results = helper.memory_downcaster(results)

        return results

        
    def fetch(self, df):
        """Top-level function."""

        # --- Construct output path ---
        infered_freq = pd.infer_freq(pd.to_datetime(pd.Series(list(set(df[self.date_col])))).sort_values())
        db_clean = self.earthengine_db.replace('/','-')
        buffers_clean = '-'.join([str(i) for i in self.buffers])
        self.cache_dir = os.path.join('data', 'earthengine')
        
        # --- break up by month ---
        df['month'] = [i.month for i in df['datetime_utc']]
        months = list(set(df['month']))
        sorted(months)
        
        for m in months:
            cache_month_path =  os.path.join(self.cache_dir, f"{db_clean}&agg{infered_freq}&{m}.pkl")
            
            # --- See what is requested for the given month ---
            month_df = df.loc[df['month'] == m]

            # --- duplicate rows for each buffer ---
            month_df_list = []
            for b in self.buffers:
                _month_df = month_df.copy()
                _month_df['buffer'] = b
                month_df_list.append(_month_df)
            requested = pd.concat(month_df_list, axis='rows', sort=False)

            # --- try to load from cache ---
            if self.use_cache:
                if os.path.exists(cache_month_path):
                    with open(cache_month_path, 'rb') as handle:
                        log.info(f'....loading cache for month {m}')
                        cache = pickle.load(handle)
                else:
                    log.info(f'....no cache found for month {m}')
                    cache = pd.DataFrame({
                        'plant_id_eia':[],
                        'datetime_utc':[],
                        'buffer':[],
                        'variable':[],
                        'value':[]
                    })

                # --- drop nans from cache ---
                keys = ['plant_id_eia','datetime_utc','buffer']
                cache = cache.dropna(subset=keys+['variable','value'], how='any')
                
                # --- figure out what we already have and what we need ---
                cache.set_index(keys, inplace=True, drop=True)
                requested.set_index(keys, inplace=True, drop=True)
                downloaded = cache.loc[cache.index.isin(requested.index)]
                needed = requested.loc[~requested.index.isin(cache.index)]
                log.info(f'........{len(downloaded)} queries loaded from cache')
                log.info(f'........{len(needed)} queries still needed')

                downloaded.reset_index(inplace=True, drop=False)
                needed.reset_index(inplace=True, drop=False)
                cache.reset_index(inplace=True, drop=False)
                
                # --- get what we need ---
                results = self._run_jobs(needed, m)
                results = self._clean_results(results)

                # --- concat everything together and save ---
                out = pd.concat([downloaded, results], axis='rows', sort=False)
                out = out[keys + ['variable', 'value']]

                # --- rewrite new cache ---
                cache = pd.concat([out, cache], axis='rows', sort=False)
                cache = cache.drop_duplicates(subset=keys, keep='first')
                
                # --- save to pickle ---
                with open(cache_month_path, 'wb') as handle:
                    pickle.dump(cache, handle)
        
            else:
                results = self._run_jobs(needed, m)
                results = self._clean_results(results)
                
        return self