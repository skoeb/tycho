#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""


import ee
import folium
import geehydro

import sqlite3
import os
import ftplib
import concurrent.futures as cf
import time
import json
import itertools

import swifter
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points

import config
import helper

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ LOAD PUDL 860/923 ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PUDLLoader():
    def __init__(self, years=[2018],
                 round_coords_at=3,
                 ts_frequency='D'):
        
        print('\n')
        print('Initializing PUDLLoader')

        assert ts_frequency in ['D','H']
        
        self.years = years
        self.round_coords_at = round_coords_at
        self.ts_frequency = ts_frequency
        
    def _connect_to_sqlite(self):
        print('....connecting to sqlite for 860/923 data')
        self.db_path = os.path.join('data','pudl-work','sqlite','pudl.sqlite')
        self.engine = sqlite3.connect(self.db_path)
        return self
    
    
    def _load_plants_entity_eia(self):
        print('....loading plant level data')
        self.plants = pd.read_sql_query("SELECT * FROM plants_entity_eia", self.engine)
        self.plants = helper.memory_downcaster(self.plants)
        return self
    
    
    def _load_generators_eia860(self):
        print('....loading generator eightsixty data')
        self.eightsixty = pd.read_sql_query("SELECT * FROM generators_eia860", self.engine)
        self.eightsixty = helper.memory_downcaster(self.eightsixty)
        return self
    
    
    def _clean_eightsixty(self):
        keep = [
            'plant_id_eia',
            'report_year',
            'operational_status',
            'capacity_mw',
            'summer_capacity_mw',
            'winter_capacity_mw',
            'fuel_type_code_pudl',
            'multiple_fuels',
            'planned_retirement_year',
            'minimum_load_mw',
        ]

        agg_dict = {
            'capacity_mw':'sum',
            'summer_capacity_mw':'sum',
            'winter_capacity_mw':'sum',
            'minimum_load_mw':'sum',
            'fuel_type_code_pudl':'first',
            'multiple_fuels':'max',
            'planned_retirement_year':'max',
        }

        # --- convert to datetime ---
        self.eightsixty['report_date'] = pd.to_datetime(self.eightsixty['report_date'])
        self.eightsixty['planned_retirement_date'] = pd.to_datetime(self.eightsixty['planned_retirement_date'])
        self.eightsixty['report_year'] = [i.year for i in self.eightsixty['report_date']]
        self.eightsixty['planned_retirement_year'] = [i.year for i in self.eightsixty['planned_retirement_date']]

        # --- only take input year ---
        self.eightsixty = self.eightsixty.loc[self.eightsixty['report_year'].isin(self.years)]
        print(f"........filtering to report years: len {len(self.eightsixty)}")

        # --- take out possible retirements within next two years ---
        self.eightsixty['planned_retirement_year'].fillna(2099, inplace=True) #fill in nans for plants with no planned retirement
        self.eightsixty = self.eightsixty.loc[self.eightsixty['planned_retirement_year'] > self.eightsixty['report_year'] + 2]
        print(f"........filtering out retirements in next year: len {len(self.eightsixty)}")

        # --- only take operational assets ---
        self.eightsixty = self.eightsixty.loc[self.eightsixty['operational_status'] == 'existing']
        print(f"........filtering out non-operational assets: len {len(self.eightsixty)}")

        # --- only take fossil generators ---
        self.eightsixty = self.eightsixty.loc[self.eightsixty['fuel_type_code_pudl'].isin(['coal','gas','oil'])]
        print(f"........filtering out non-fossil generators: len {len(self.eightsixty)}")
        
        # --- filter out columns ---
        self.eightsixty = self.eightsixty[keep]
        
        # --- groupby to reduce multiple generators at one plant ---
        self.eightsixty = self.eightsixty.groupby(['plant_id_eia','report_year'], as_index=False).agg(agg_dict)
        print(f"........reducing generators to plant level: len {len(self.eightsixty)}")

        # --- make small ---
        self.eightsixty = helper.memory_downcaster(self.eightsixty)
        return self
    
    def _clean_plants(self):
        
        keep = [
            'plant_id_eia',
            'plant_name_eia',
            'city',
            'county', 
            'latitude',
            'longitude',
            'state',
            'timezone'
        ]
        
        # --- Round coordinates ---
        self.plants[['latitude','longitude']] = self.plants[['latitude','longitude']].round(self.round_coords_at)
        
        # --- Filter out unnecessary columns ---
        self.plants = self.plants[keep]

        # --- Only plants in EIA 860 ---
        self.plants = self.plants.loc[self.plants['plant_id_eia'].isin(list(set(self.eightsixty['plant_id_eia'])))]
        
        return self
    
    def _make_geopandas(self):
        self.eightsixty = gpd.GeoDataFrame(
            self.eightsixty, geometry=gpd.points_from_xy(self.eightsixty['longitude'], self.eightsixty['latitude']))
        self.eightsixty.crs = "EPSG:4326"
        return self
        
    
    def _merge_dbs(self):
        print('....merging dbs')
        # --- Merge plant data with generator data ---
        self.eightsixty = self.eightsixty.merge(self.plants, on='plant_id_eia', how='inner')
        print(f"........merging eightsixty with plants on plant_id_eia: len {len(self.eightsixty)}")
        return self
    
    
    def load(self):
        
        # --- Grab SQLite data ---
        self._connect_to_sqlite()
        self._load_plants_entity_eia()
        self._load_generators_eia860()
        
        # --- Clean ---
        self._clean_eightsixty()
        self._clean_plants()
        
        # --- Merge ---
        self._merge_dbs() #merge eightsixty and plants
        
        # --- Make Geopandas ---
        self._make_geopandas()
        
        return self
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ LOAD EPA CEMS DATA ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EPACEMSScraper():
    """
    The EPA requires certain thremal generators within the U.S. to report
    certain emission outputs on an hourly basis, including NOx, SO2, and CO2.

    Many of these values are measured, however some are calculated or imputed based
    on known emission outputs and MWh of production (also reported on an hourly basis).

    This data will serve as the training target (y_train values) for our supervised learning model.

    We will build an ML pipeline that trains on this data, and can predict for the rest
    of the world (where we don't have EPA CEMS data). 

    EPACEMSScraper scrapes .csvs from an FTP server hosted by the EIA.
    csvs are segmented by state and month. This script will download any
    files that are not present, and skip files that are already downloaded. 

    If the EPA ever updates files (which they often do around the September/October)
    timeframe for the previous year, it is responsible to delete the files locally
    and rerun the scraper to download new files. 
    """

    def __init__(self, server='newftp.epa.gov',
                 server_dir='DMDnLoad/emissions/hourly/monthly/',
                 download_path=os.path.join('data','CEMS','csvs'),
                 years=[2019]):
        
        print('\n')
        print('Initializing EPACEMSScraper')

        self.server=server
        self.server_dir = server_dir
        self.download_path = download_path
        self.years=years
    
    def _connect_to_ftp(self):
        self.ftp = ftplib.FTP(self.server)
        self.ftp.login()
        print('....connected to EPA CEMS FTP Server')
        return self
    
    def _cwd_annual(self, year):
        year_server_dir = self.server_dir + str(year) + '/'
        self.ftp.cwd(year_server_dir)
        files = self.ftp.nlst()
        print('........downloaded file list from EPA CEMS FTP Server')
        return files
    
    def _already_downloaded(self, files):
        """Check what is already downloaded and skip it."""
        downloaded = os.listdir(self.download_path)
        needed = [f for f in files if f not in downloaded]
        print(f"....{len(needed)} files needed, {len(downloaded) - len(needed)} files already downloaded")
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
            print(f"....working on {y}")
            jobs = self._cwd_annual(y)
            jobs = self._already_downloaded(jobs) # see what is already downloaded and update jobs
        
            # --- Download monthly/state files ---
            jobs_complete = 0 
            ten_percent = max(1, int(len(jobs) * 0.1))
            
            for job in jobs: #FTP limits connections, so multiprocessing doesn't work
                self._worker(job)
                jobs_complete += 1
                    
                if jobs_complete % ten_percent == 0:
                    print('........finished EPA CEMS download {} / {}'.format(jobs_complete, len(jobs)))
        print(f"....finished all downloads")
        return self


class CEMSLoader():
    """
    CEMSLoader loads the csvs returned by EPACEMSScraper() and performs sanitizing functions.
    This includes:
        - aggregating multiple units into a single plant
        - resampling hourly data to the desired resoultion ('D' for daily by default)
        - dropping plants without a full year of data
        - dropping plants with a significant amount of missing or nan values
        - dropping plants without any reported emissions
        - converting emission units from tons to lbs
        - cleaning up column names
        - reducing memory consumption by converting dtypes where possible

    This script can be time consuming, and if sub-daily resolution is used can result in very large
    file sizes. However if daily resoultion is used, the resulting files is only ~ 12 MB pickled,
    and saved to a 'data/CEMS/processed' file. If an identical query (year/ts_frequency) is performed,
    the processed pickle will be loaded to save time. 
    """
    def __init__(self, ts_frequency='D', years=[2019], clean_on_load=True,
                 use_pickle=True):
        
        print('\n')
        print('Initializing CEMSLoader')

        self.ts_frequency = ts_frequency
        self.years = years
        self.dir_path = os.path.join('data','CEMS','csvs')
        self.clean_on_load = clean_on_load
        
        
        self.use_pickle = use_pickle
        years_clean = [str(i) for i in years]
        years_clean = '-'.join(years_clean) #save as seperate caches
        self.pkl_path = os.path.join('data','CEMS','processed',f"cems_{ts_frequency}_{years_clean}.pkl")
        
        self.cems = None

    
    def _read_csvs(self):
        print('....reading CEMS csvs')
        # --- Get file paths ---
        files = os.listdir(self.dir_path)
        files = [f for f in files if not f.startswith('.')]
        
        year_files = []
        for y in self.years:
            year_files += [i for i in files if str(y) in i]
        
        to_concat = []
        ten_percent = max(1, int(len(files)*0.1))
        done = 0
        for f in files:
            _df = pd.read_csv(os.path.join(self.dir_path, f))
            if self.clean_on_load:
                _df = self._clean_cems(_df)

            to_concat.append(_df)
            done +=1
            if done % ten_percent == 0:
                print(f"........finished loading {done}/{len(files)} csvs")
            
        # --- Convert to dataframe ---
        print('....concatenating CEMS csvs')
        self.cems = pd.concat(to_concat, axis='rows', sort=False)
            
        return self
    
    def _clean_cems(self, df):
        
        rename_dict = {
            'STATE':'state',
            'ORISPL_CODE':'plant_id_eia',
            'UNIT_ID':'unit',
            'OP_DATE':'date',
            'OP_HOUR':'hour',
            'OP_TIME':'operational_time',
            'GLOAD (MW)':'gross_load_mw',
            'SO2_MASS (lbs)':'so2_lbs',
            'NOX_MASS (lbs)':'nox_lbs',
            'CO2_MASS (tons)':'co2_tons',
        }

        # --- Rename columns ---
        df = df.rename(rename_dict, axis='columns')

        # --- Convert to datetime ---
        if self.ts_frequency != 'D':
            df['hour'] = [str(i)+':00:00' for i in df['hour']]
            df['datetime_utc'] = pd.to_datetime(df['date'] + ' ' + df['hour'])
        elif self.ts_frequency == 'D':
            df['datetime_utc'] = pd.to_datetime(df['date'])
            
         # --- Aggregate by unit ---
        agg_dict = {
            'gross_load_mw':'sum',
            'so2_lbs':'sum',
            'nox_lbs':'sum',
            'co2_tons':'sum',
            'operational_time':'mean',
        }
        df = df.groupby(['plant_id_eia','datetime_utc'], as_index=False).agg(agg_dict)
        
        # --- Aggregate by ts_frequency ---
        df = df.groupby('plant_id_eia').resample(self.ts_frequency, on='datetime_utc').sum()
        df.drop(['plant_id_eia'], axis='columns', inplace=True, errors='ignore') #duplicated by resample
        df.reset_index(inplace=True, drop=False)
        
                
        # --- fill nans with zeros---
        df = df.fillna(0)
        
        # --- drop plants with a large number of zeros ---
        df = df.loc[df.groupby('plant_id_eia')['nox_lbs'].filter(lambda x: len(x[x > 0]) > 0).index]
        df = df.loc[df.groupby('plant_id_eia')['so2_lbs'].filter(lambda x: len(x[x > 0]) > 0).index]
        df = df.loc[df.groupby('plant_id_eia')['co2_tons'].filter(lambda x: len(x[x > 0]) > 0).index]
        
        # --- Drop unnecessary columns ---
        keep = ['datetime_utc','plant_id_eia',
                'gross_load_mw','so2_lbs','nox_lbs','co2_tons','operational_time']
        df = df[keep]

        # --- convert co2 from tons to lbs ---
        df['co2_lbs'] = df['co2_tons'] / 2000
        df = df.drop(['co2_tons'], axis='columns')
        
        # --- reduce size ---
        df = helper.memory_downcaster(df)
        
        return df
    
    
    def _post_load_clean(self):
        
        print(f"........postprocessing CEMS, len: {len(self.cems)}")
        
        # --- drop plants without a full year of data ---
        plant_id_eias_keep = list(set(self.cems.groupby('plant_id_eia', as_index=False)['plant_id_eia'].filter(lambda x: x.count() == 365)))
        self.cems = self.cems.loc[self.cems['plant_id_eia'].isin(plant_id_eias_keep)]
        print(f"........droping generators without a full year of data, len: {len(self.cems)}")
        
        # --- reset index ---
        self.cems.reset_index(drop=True, inplace=True)
        
        return self
    
    def fetch(self):
        
        # --- Try to load aggregated pickle ---
        if self.use_pickle:
            if os.path.exists(self.pkl_path):
                print('....reading CEMS from pickle')
                self.cems = pd.read_pickle(self.pkl_path)
        
        # --- Calculate aggregate df from csvs ---
        if not isinstance(self.cems, pd.DataFrame):
            self._read_csvs()
            self._post_load_clean()
            
            # --- Save pickle ---
            if self.use_pickle:
                print('....saving CEMS to pickle')
                self.cems.to_pickle(self.pkl_path)
                
        return self
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ LOAD WRI GPPD DATA ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GPPDLoader():
    
    def __init__(self, ts_frequency='D', 
                 round_coords_at=3, countries=['United States of America']):

        print('\n')
        print('Initializing GPPDLoader')

        self.pdir = os.path.join(os.getcwd())#, os.pardir)
        self.ts_frequency = ts_frequency
        
        self.round_coords_at = round_coords_at #.01 degrees = 1 km
        self.countries = countries

        
    def _load_csv(self):
        self.gppd = pd.read_csv(os.path.join(self.pdir, 'data','wri','global_power_plant_database.csv'))
        return self
    
    
    def _make_geopandas(self):
        self.gppd = gpd.GeoDataFrame(
            self.gppd, geometry=gpd.points_from_xy(self.gppd['longitude'], self.gppd['latitude']))
        self.gppd.crs = "EPSG:4326"
        return self
    
    
    def _clean_gppd(self):
        
        keep = [
            'country_long',
            'name', 
            'wri_capacity_mw',
            'latitude',
            'longitude',
            'primary_fuel', 
            'commissioning_year',
            'generation_gwh_2013',
            'generation_gwh_2014',
            'generation_gwh_2015',
            'generation_gwh_2016',
            'generation_gwh_2017',
            'estimated_generation_gwh',
        ]

        # --- Round lat lon ---
        self.gppd[['latitude','longitude']] = self.gppd[['latitude','longitude']].round(self.round_coords_at)
        
        # --- Filter country ---
        print(f"........filtering gppd to include {self.countries}")
        if 'all' not in self.countries: #include all countries
            for country in self.countries:
                assert country in set(self.gppd['country_long'])
                self.gppd = self.gppd.loc[self.gppd['country_long'].isin(self.countries)]
        
        # --- Drop non fossil fuels ---
        self.gppd = self.gppd.loc[self.gppd['primary_fuel'].isin(['Coal','Oil','Gas','Petcoke','Cogeneration'])]
        
        # --- Rename columns ---
        self.gppd.rename({'capacity_mw':'wri_capacity_mw'}, axis='columns', inplace=True)
        
        # --- filter columns we want ---
        self.gppd = self.gppd[keep]
        
        return self
    
    
    def load(self):
        print(f"....Loading gppd from csv")
        # --- Read csv ---
        self._load_csv()
        
        # --- Clean df ---
        self._clean_gppd()
        
        # --- Make Geopandas ---
        self._make_geopandas()
        
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ MERGE DATA TOGETHER ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
class TrainingDataMerger():
    """
    Merge all possible cached data for training, including:
        - EIA 860/923 data returned from PUDLLoader()
        - WRI Global Powerplant Database Data returned from GPPDLoader()
        - EPA Continuous Emission Monitoring System Target Data 
    """
    def __init__(self, eightsixty, gppd, cems,
                 match_distance_thresh=0.01):

        print('\n')
        print('Initializing TrainingDataMerger')

        self.eightsixty = eightsixty
        self.gppd = gppd
        self.cems = cems

        self.match_distance_thresh = match_distance_thresh
        
        
    def _make_db_points(self):
        # --- Drop duplicates (as strings for performance) ---
        _db = self.db.copy()
        _db['wkt'] = self.db['geometry'].apply(lambda x: x.wkt).values
        self.unique = _db.drop_duplicates(subset=['wkt'])

        # --- Calc unions of points ---
        self.db_points = self.unique.unary_union

        return self

    
    def _nearest_point_worker(self, gppd_point):
        # add thresh for max distance
        _, match = nearest_points(gppd_point, self.db_points)
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
        print("....beginning merge process between cems and eightsixty")
        print(f"........pre-merge generator count in eightsixty: {len(set(self.eightsixty['plant_id_eia']))}")
        print(f"........pre-merge generator count in cems: {len(set(self.cems['plant_id_eia']))}")
        self.db = self.eightsixty.merge(self.cems, on=['plant_id_eia'], how='right')
        print(f"........post-merge generator count: {len(set(self.db['plant_id_eia']))}")
        
        # --- Drop CEMS data not in eightsixty ---
        print(f"........pre-drop generator count: {len(set(self.db['plant_id_eia']))}")
        self.db = self.db.dropna(subset=['plant_name_eia'])
        print(f"........post-drop generator count: {len(set(self.db['plant_id_eia']))}")

        # --- Create list of known points in self.db ---
        print('....making db points list.')
        self._make_db_points()
        
        # --- Find nearest db plant for plants in gppd ---
        print('....finding nearest neighboring plants between gppd and db.')

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
                        print('....finished point matching job {} / {}'.format(len(results_list), len(jobs)))
        else:
            results_list = [self._nearest_point_worker(job) for job in jobs]
        
        results_dict = {k.wkt :v for k,v in results_list}
        self.gppd['wkt'] = self.gppd['geometry'].apply(lambda x: x.wkt).values
        self.gppd['plant_id_eia'] = self.gppd['wkt'].map(results_dict)

        # --- Filter out plants that no match was found ---
        print(f"........pre-drop generator count in gppd: {len(self.gppd)}")
        self.gppd = self.gppd.dropna(subset=['plant_id_eia'])
        print(f"........post-drop generator count in gppd: {len(self.gppd)}")

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
            'plant_id_eia'
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
        self.gppd = self.gppd.groupby('plant_id_eia', as_index=False).agg(agg_dict)

        #TODO: consider just dropping these rather than the groupby above
        # # --- Filter out plants that duplicate eightsixty was found ---
        # print(f"........pre-drop generator count in db: {len(set(self.db['plant_id_eia']))}")
        # self.gppd = self.gppd.drop_duplicates(subset=['plant_id_eia'], keep='first')
        # print(f"........post-drop generator count in db: {len(set(self.db['plant_id_eia']))}")

        # --- Merge on plant_id_eia ---
        print(f"........pre-merge generator count in db: {len(set(self.db['plant_id_eia']))}")
        self.db = self.db.merge(self.gppd, on='plant_id_eia', how='inner')
        print(f"........post-merge generator count in db: {len(set(self.db['plant_id_eia']))}")
        
        # --- Drop plants where difference between WRI capacity and EIA capacity is greater than 40% of WRI capacity ---
        print(f"........pre-drop generator count in db: {len(set(self.db['plant_id_eia']))}")
        self.db['diff'] = self.db['wri_capacity_mw'] - self.db['capacity_mw']
        self.db['diff'] = self.db['diff'].abs()
        self.db['thresh'] = self.db['wri_capacity_mw'] * 0.40
        self.db = self.db[self.db['diff'] <= self.db['thresh']]
        self.db = self.db.drop(['diff','thresh'], axis='columns')
        print(f"........post-merge generator count in db: {len(set(self.db['plant_id_eia']))}")
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~ LOAD EARTH ENGINE DATA ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GetDailyEarthEngineData():
    """
    Fetch Earth Engine data for a geography.
    
    Inputs
    ------
    df (geopandas.GeoDataFrame) - Dataframe with 'generator_id', 'geometry', and 'date' columns.
    db (string) - From https://developers.google.com/earth-engine/datasets/catalog
    scale (int) - Granularity of calculating average values within geography
    buffers (list) - size in meters to return data for (i.e. 1e3 for 1km)
    days_combine (int) - Number of days to combine when calculating aggregated sattelite data
        agg_func is applied to the returned collection of images.
    agg_func (string) - How to calculate the daily data when aggregating multiple
        satelitte swaths together. 
    
    Returns
    -------
    if pandas_out (bool):
        long DataFrame
    else:
        list of dicts containing variables as keys
    
    """
    
    def __init__(self, db='COPERNICUS/S5P/OFFL/L3_NO2',
                 agg_func='median',
                 scale=10, buffers=[1e3, 5e3],
                 id_col='plant_id_eia',
                 geo_col='geometry',
                 date_col='date',
                 pandas_out=True,
                 read_cache=True, to_cache=True):

        print('\n')
        print('Initializing GetDailyEarthEngineData')

        self.db = db
        self.agg_func = agg_func
        self.scale = scale
        self.buffers=buffers
        self.pandas_out = pandas_out
        self.read_cache = read_cache
        self.to_cache = to_cache
        
        self.id_col=id_col
        self.geo_col=geo_col
        self.date_col=date_col

        
    def _load_image(self, date):
        """Load Earth Edge image."""
        # --- Make dates strings so google is happy ---
        start_date = date
        next_date = (pd.Timestamp(date) + pd.tseries.offsets.DateOffset(days=1)).strftime('%m-%d-%Y') #TODO: implement frequency keyword here 
        import pdb; pdb.set_trace()

        # --- Load Image and add as layer ---
        imagecollection = ee.ImageCollection(self.db)
        date_agg = collection.filterDate(start_date, next_date)

        if self.agg_func == 'median':
            image = date_agg.median()
        else:
            raise NotImplementedError(f"please write a wrapper for {self.agg_func}!")

        return image

    
    def _load_geometry(self, geometry):
        """Return geometry point object."""
        lon = geometry.x
        lat = geometry.y
        geometry = ee.Geometry.Point(lon, lat)
        return geometry

    
    def _calc_geography_mean(self, date_agg, geometry):
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
        dict - keys are bands, values are 
        """
        average_dict = date_agg.reduceRegion(**{
          'reducer': ee.Reducer.mean(),
          'geometry':geometry,
          'scale': self.scale,
        })
        return average_dict
    
    
    def _worker(self, row):
        """Returns a dict with keys as buffer size, and values of dicts of band values."""
        geometry = self._load_geometry(row[self.geo_col])
        date_agg = self._load_image(row[self.date_col])
        
        # --- List of results, with one per buffer in self.buffers ---
        _results = []
        
        for b in self.buffers:
            _b_result = self._calc_geography_mean(date_agg, geometry)
            _b_result[self.id_col] = self.id_col
            _b_result[self.geo_col] = self.geo_col
            _b_results[self.date_col] = self.date_col
            _b_results['buffer'] = b
            _results.append(_b_results)
            
        return _results
        
        
    def _chunkify(self, df, n):
        """Break df (df) into chunks of size (n)"""
        for i in range(0, len(df), n):
            yield df[i:i+n]
    
    
    def _run_jobs(self, jobs_df):
        
        if config.MULTIPROCESSING:
            results = []
            with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as executor:
                ten_percent = max(1, int(len(jobs_df) * 0.1))

                # --- Chunk jobs ---
                chunks = self._chunkify(jobs_df, config.WORKERS)
                for chunk in chunks:
                    
                    # --- Submit to worker ---
                    futures = [executor.submit(self._worker, row) for _, row in chunk.iterrows()]
                    for f in cf.as_completed(futures):
                        results.append(f.result())
                        if len(results) % config.WORKERS == 0:
                            print('Finished Earth Engine Job {} / {}'.format(len(results), len(jobs)))
        else:
            results = [self._worker(row) for _, row in jobs_df.iterrows()]
            
        return results
    
    
    def _load_cache(self):
        
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as handle:
                cache = pickle.load(handle)
        
        else:
            cache = pd.DataFrame({
                self.id_col:[np.nan],
                self.geo_col:[np.nan],
                self.date_col:[np.nan]
            })
       
        return cache
    
    
    def _dump_cache(self, results):
        with open(self.cache_path, 'wb') as handle:
            pickle.dump(results, handle)
        
        return self
             
        
    def _unpack_results(self, results):
        
        # --- Unpack 2d list to flat ---
        results = list(itertools.chain(*results))
        
        if self.pandas_out:
            
            # --- Read list of dicts into DataFrame ---
            results = pd.DataFrame(results)
            
            # --- Make long ---
            id_vars = [self.id_col, self.date_col, self.geo_col, 'buffer']
            results = pd.melt(results, id_vars)
        
        return results   
        
        
    def fetch(self, df):
        """Top-level function."""
        
        
        # --- Construct output path ---
        self.infered_freq = pd.infer_freq(pd.to_datetime(pd.Series(list(set(df['date'])))).sort_values())
        db_clean = self.db.replace('/','-')
        buffers_clean = [str(i) for i in self.buffers]
        buffers_clean = '-'.join(buffers_clean) #save as seperate caches
        self.pdir = os.path.join(os.getcwd(), os.pardir)
        self.cache_path = os.path.join(self.pdir, 'cache', f"{db_clean}_agg{self.infered_freq}_{buffers_clean}.pkl")
        
        results = self._run_jobs(df)
        results = self.unpack_results(results)
        
        if self.to_cache:
            self._dump_cache(results)
        
        return results
    
def main():

    # --- Load EIA 860/923 data from PUDL ---
    pudlloader = PUDLLoader()
    pudlloader.load()
    eightsixty = pudlloader.eightsixty
    eightsixty = eightsixty

    # --- scrape EPA CEMS data if not present in 'data/CEMS/csvs' (as zip files) ---
    # scraper = EPACEMSScraper()
    # scraper.fetch()

    # --- load CEMS data from pickle, or construct dataframe from csvs ---
    CEMS = CEMSLoader()
    CEMS.fetch()
    cems = CEMS.cems
    cems

    # --- Load WRI Global Power Plant Database data from csv ---
    GPPD = GPPDLoader() 
    GPPD.load()
    gppd = GPPD.gppd
    gppd

    # --- Merge eightsixty, gppd, cems together into a long_df ---
    MERGER = TrainingDataMerger(eightsixty, gppd, cems)
    MERGER.merge()
    db = MERGER.db

    import pdb; pdb.set_trace()
    # --- Load Google Earth Engine Data using db for dates ---
    eeloader = GetDailyEarthEngineData()
    nox = eeloader.fetch(db)

if __name__ == '__main__':
    main()