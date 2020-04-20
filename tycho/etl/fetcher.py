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
import hashlib

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
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
log = logging.getLogger("tycho")

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
        log.info(f"....{len(needed)} files needed, {len(downloaded)} files already downloaded")
        return needed
        
    
    def _worker(self, f):
        self.ftp.retrbinary("RETR " + f, open(os.path.join(self.download_path, f), "wb").write)
        time.sleep(0.5)
        return
    

    def fetch(self):
        # --- Connect to FTP ---
        self._connect_to_ftp()
        
        try:
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
        
        except Exception as e:
            log.warning('....warning Could not connect to CEMS FTP server, assuming that csvs are already downloaded.')
            log.exception(e)
            log.info('\n')
        
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
    
    def __init__(self, earthengine_dbs=config.EARTHENGINE_DBS,
                 agg_func='sum',
                 scale=1113, tilescale=16,
                 buffers=config.BUFFERS,
                 id_col='plant_id_wri',
                 geo_col='geometry',
                 date_col='datetime_utc',
                 ts_frequency=config.TS_FREQUENCY,
                 percentiles = [10,25,50,75,90],
                 left_window_hour = config.LEFT_WINDOW_HOUR,
                 read_cache=True, use_cache=True):

        log.info('\n')
        log.info('Initializing EarthEngineFetcher')

        self.earthengine_dbs = earthengine_dbs
        self.agg_func = agg_func
        self.tilescale = tilescale
        self.scale = scale
        self.buffers=buffers
        self.read_cache = read_cache
        self.use_cache = use_cache
        self.ts_frequency = ts_frequency
        self.percentiles = percentiles
        self.left_window_hour = left_window_hour
        
        self.id_col=id_col
        self.geo_col=geo_col
        self.date_col=date_col

    
    def _calc_geography(self, row):
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

        # --- get buffer as max_distance ---
        buffer = float(row['buffer'])

        # --- calculate weight band ---
        lon = row.geometry.x
        lat = row.geometry.y
        geometry = ee.Geometry.Point(lon=lon, lat=lat).buffer(buffer)

        start_date = row[self.date_col] #.strftime('%m-%d-%Y')
        start_date = start_date.replace(hour=self.left_window_hour)

        if self.ts_frequency == 'D':
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(days=1)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        elif self.ts_frequency == 'W-SUN':
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(weeks=1)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        elif self.ts_frequency in ['A','AS']:
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(years=1)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        elif self.ts_frequency in ['MS']:
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(months=1)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        elif self.ts_frequency == '3D':
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(days=3)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        elif self.ts_frequency == '2D':
            next_date = (pd.Timestamp(start_date) + pd.tseries.offsets.DateOffset(days=2)) #.strftime('%m-%d-%Y') #TODO: implement frequency keyword here 

        # -- Make a date filter to get images in this date range. --
        dateFilter = ee.Filter.date(start_date, next_date)

        # --- Specify an equals filter for image timestamps. ---
        filterTimeEq = ee.Filter.equals(** {
            'leftField': 'system:time_start',
            'rightField': 'system:time_start'
            })
        
        for ix, db in enumerate(self.earthengine_dbs):

            if db == "CIESIN/GPWv411/GPW_Population_Count":
                pass

            elif db == "ECMWF/ERA5/DAILY":
                pass
            
            elif ix == 0:
                ic = ee.ImageCollection(db)
            
            else:
                # --- Define the new data ---
                _new = ee.ImageCollection(db).filter(dateFilter)

                # --- Define an inner join. ---
                innerJoin = ee.Join.inner()

                # --- Apply the join. ---
                ic = innerJoin.apply(ic, _new, filterTimeEq)
                
                # --- flatten ---
                ic = ic.map(lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
                ic = ee.ImageCollection(ic)

        # --- select only the bands we want---
        keep = []
        if "COPERNICUS/S5P/OFFL/L3_CLOUD" in self.earthengine_dbs:
            keep += ['cloud_fraction','cloud_top_pressure']
                     #'cloud_base_height','cloud_top_height',
                     #'cloud_optical_depth','surface_albedo']

        if "COPERNICUS/S5P/OFFL/L3_NO2" in self.earthengine_dbs:
            keep += ['tropospheric_NO2_column_number_density',
                     'tropopause_pressure', 'absorbing_aerosol_index']
                     #'NO2_column_number_density',
                    # 'stratospheric_NO2_column_number_density','NO2_slant_column_number_density',
                    

        if "COPERNICUS/S5P/OFFL/L3_SO2" in self.earthengine_dbs:
            keep += ['SO2_column_number_density']#,'SO2_column_number_density_amf']
                    #  'SO2_slant_column_number_density']

        if "COPERNICUS/S5P/OFFL/L3_CO" in self.earthengine_dbs:
            keep += ['CO_column_number_density','H2O_column_number_density']

        if "COPERNICUS/S5P/OFFL/L3_HCHO" in self.earthengine_dbs:
            keep += ['tropospheric_HCHO_column_number_density']
                     #'tropospheric_HCHO_column_number_density_amf']
                     #'HCHO_slant_column_number_density']

        if "COPERNICUS/S5P/OFFL/L3_O3" in self.earthengine_dbs:
            keep += ['O3_column_number_density','O3_effective_temperature']

        if "COPERNICUS/S5P/OFFL/L3_CH4" in self.earthengine_dbs:
            keep += ['CH4_column_volume_mixing_ratio_dry_air']
                     #'aerosol_height',
                     #'aerosol_optical_depth']

        ic = ic.select(keep)

        # --- Aggregate image collection into an image ---
        if self.agg_func == 'median':
            image = ic.median()
        elif self.agg_func == 'mean':
            image = ic.mean()
        elif self.agg_func == 'sum':
            image = ic.sum()
        else:
            raise NotImplementedError(f"please write a wrapper for {self.agg_func}!")
        
        # --- Load GPPD on earth engine ---
        wri_powerplants = ee.FeatureCollection("WRI/GPPD/power_plants")
        country_filter = ee.Filter.eq('country', 'USA')
        fuel_filter = ee.Filter.inList('fuel1', ['Coal','Gas','Oil','Petcoke'])
        wri_powerplants = wri_powerplants.filter(ee.Filter.And(country_filter, fuel_filter))

        # --- add weighted distance between powerplants as mask ---
        distance = wri_powerplants.distance(searchRadius=buffer, maxError=100)
        weight = distance.subtract(buffer).abs().divide(buffer)

        # --- Reduce ---
        # weight_sum_dict = {}
        # for c in keep:

        #     # --- select the relevant column ---
        #     _img = image.select(c)

        #     # --- add the weight ---
        #     _img = _img.addBands(weight)

        #     query = _img.reduceRegion(**{
        #         'reducer': ee.Reducer.sum().splitWeights(),
        #         'geometry':geometry,
        #         'scale': self.scale,
        #         'bestEffort':True,
        #         'tileScale':self.tilescale
        #     })
        #     weight_sum_dict[c] = query.getInfo()['sum']

        # sum_dict = image.reduceRegion(**{
        #     'reducer': ee.Reducer.sum(),
        #     'geometry': geometry,
        #     'scale': self.scale,
        #     'bestEffort': True,
        #     'tileScale': self.tilescale
        # })

        sum_dict = image.reduceRegion(**{
            'reducer': ee.Reducer.sum(),
            'geometry':geometry,
            'scale': self.scale,
            'bestEffort':True,
            'tileScale':self.tilescale
        })

        mean_dict = image.reduceRegion(**{
            'reducer': ee.Reducer.mean(),
            'geometry':geometry,
            'scale': self.scale,
            'bestEffort':True,
            'tileScale':self.tilescale
        })

        std_dict = image.reduceRegion(**{
          'reducer': ee.Reducer.stdDev(),
          'geometry':geometry,
          'scale': self.scale,
          'bestEffort':True,
          'tileScale':self.tilescale
        })

        # --- make queries ---
        mean_dict = mean_dict.getInfo()
        std_dict = std_dict.getInfo()
        sum_dict = sum_dict.getInfo()

        # --- Get population attributes seperately ---
        if "CIESIN/GPWv411/GPW_Population_Count" in self.earthengine_dbs:
            pop_ic = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count")
            pop_ic = pop_ic.select('population_count')
            pop_ic = pop_ic.mean()
            pop_dict = pop_ic.reduceRegion(**{
                'reducer': ee.Reducer.sum(),
                'geometry':geometry,
                'scale': self.scale,
                'bestEffort':True,
                # 'tileScale':self.tilescale
                })
            pop_dict = pop_dict.getInfo()
        else:
            pop_dict = {}

        # --- Get weather seperately ---
        if "ECMWF/ERA5/DAILY" in self.earthengine_dbs:
            weath_ic = ee.ImageCollection("ECMWF/ERA5/DAILY")
            weath_ic = weath_ic.filter(dateFilter)
            weath_image = weath_ic.sum()
            weath_dict = weath_image.reduceRegion(**{
                'reducer': ee.Reducer.mean(), 
                'geometry':geometry,
                'scale': self.scale,
                'bestEffort':True,
                # 'tileScale':self.tilescale
                })
            weath_dict = weath_dict.getInfo()
        else:
            weath_dict = {}

        # --- rename features ---
        mean_dict = {f"{k}_mean":v for k,v in mean_dict.items()}
        std_dict = {f"{k}_std":v for k,v in std_dict.items()}
        sum_dict = {f"{k}_sum":v for k,v in sum_dict.items()}

        # weight_sum_dict = {f"{k}_weighed_sum":v for k,v in weight_sum_dict.items()}

        # --- merge ---
        out_dict = {**mean_dict, **sum_dict, **std_dict, **pop_dict, **weath_dict}
        # out_dict = {**pct_dict, **pop_dict, **weath_dict}

        return out_dict
    
    
    def _worker(self, row):
        """Returns a dict with keys as buffer size, and values of dicts of band values."""
        try:
            # with helper.timeout(config.EE_TIMEOUT): #wrapped in a timeout as I can't figure out how to modify ee's exponential backoff
            ee.Initialize()

            # --- Download results and add identifier features ---
            result = self._calc_geography(row)
            result[self.id_col] = row[self.id_col]
            result[self.geo_col] = row[self.geo_col]
            result[self.date_col] = row[self.date_col]
            result['buffer'] = row['buffer']

        except Exception as e:
            if config.VERBOSE:
                log.info(f"WARNING! Failed on {row['datetime_utc']}")

            result = {
                self.id_col:row[self.id_col],
                self.geo_col:row[self.geo_col],
                self.date_col:row[self.date_col],
                'buffer':row['buffer']
            }

            return result
        
        # --- Sleep to avoid upsetting google ---
        time.sleep(0.1)

        return result
        

    def _run_jobs(self, jobs_df, month):
        log.info(f'....starting earth engine jobs')
        results = []
        jobs_df.to_pickle('jobs_df.pkl')
        if config.MULTIPROCESSING:
            ten_percent = max(1, int(len(jobs_df) * 0.1))
            with cf.ProcessPoolExecutor(max_workers=config.THREADS) as executor: #TODO: Not sure why thread pool is dumping core here? 
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
        long_results = pd.melt(results, id_vars)

        # --- Make small ---
        long_results = helper.memory_downcaster(long_results)

        return long_results

        
    def fetch(self, df):
        """Top-level function."""

        # --- Construct output path ---
        self.cache_dir = os.path.join('data', 'earthengine')
        
        # --- break up by month ---
        df['month'] = [i.month for i in df['datetime_utc']]
        months = list(set(df['month']))
        sorted(months)

        # --- create hash of dbs to query ---
        db_string = ''.join(list(sorted(self.earthengine_dbs)))
        db_hash = int(hashlib.sha1(str.encode(db_string)).hexdigest(), 16) % (10 ** 8)
        
        self.monthly_results = []
        for m in months:
            cache_month_path =  os.path.join(self.cache_dir, f"hash{db_hash}#agg{self.ts_frequency}#{m}.pkl")
            
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
                        self.id_col:[],
                        self.date_col:[],
                        'buffer':[],
                        'variable':[],
                        'value':[]
                    })

                # --- drop nans from cache ---
                keys = ['plant_id_wri','datetime_utc','buffer']
                
                # --- Some nans are unavoidable, retrying can take time ---
                if config.RETRY_EE_NANS:
                    cache = cache.dropna(subset=keys+['variable','value'], how='any')
                
                # --- figure out what we already have and what we need ---
                cache.set_index(keys, inplace=True, drop=True)
                requested.set_index(keys, inplace=True, drop=True)
                downloaded = cache.loc[cache.index.isin(requested.index)]
                needed = requested.loc[~requested.index.isin(cache.index)]
                n_features_per_query = max(1, len(set(downloaded['variable'])))
                log.info(f'........{int(len(downloaded) / n_features_per_query)} queries loaded from cache')
                log.info(f'........{len(needed)} queries still needed')

                downloaded.reset_index(inplace=True, drop=False)
                needed.reset_index(inplace=True, drop=False)
                cache.reset_index(inplace=True, drop=False)
                
                # --- get what we need ---
                if len(needed) > 0:
                    results = self._run_jobs(needed, m)
                    results = self._clean_results(results)
                else:
                    results = pd.DataFrame({
                        self.id_col:[],
                        self.date_col:[],
                        'buffer':[],
                        'variable':[],
                        'value':[]
                    })

                # --- concat everything together and save ---
                out = pd.concat([downloaded, results], axis='rows', sort=False)
                out = out[keys + ['variable', 'value']]

                # --- rewrite new cache ---
                cache = pd.concat([out, cache], axis='rows', sort=False)
                cache = cache.sort_values('value')
                cache = cache.drop_duplicates(subset=keys + ['variable'], keep='first')
                
                # --- save to pickle ---
                cache.to_pickle(cache_month_path)
                self.cache = cache
        
            else: # mostly for testing
                results = self._run_jobs(requested, m)
                results = self._clean_results(results)
                self.monthly_results.append(results)
        
        if not self.use_cache: # mostly for testing
            all_results = pd.concat(self.monthly_results, axis='columns')
            return all_results
        
        else:  
            return self
