"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""
# --- Python Batteries Included---
import os
import sqlite3
import ftplib
import concurrent.futures as cf
import time
import json
import itertools
import random
import pickle
import zipfile
from pathlib import Path
import math
from datetime import timedelta 
import functools

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
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

# --- Module Imports ---
import tycho.config as config
import tycho.helper as helper

import logging
log = logging.getLogger("tycho")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ LOAD EPA CEMS DATA ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CEMSLoader():
    """
    CEMSLoader loads the csvs returned by EPACEMSScraper() and performs sanitizing functions.


    This script can be time consuming, and if sub-daily resolution is used can result in very large
    file sizes. However if daily resoultion is used, the resulting files is only ~ 12 MB pickled,
    and saved to a 'data/CEMS/processed' file. If an identical query (year/ts_frequency) is performed,
    the processed pickle will be loaded to save time. 

    Inputs
    ------
    ts_frequency (str) - Pandas frequency code (i.e. 'D','H','W') to resample CEMS data into
    years (list) - int representation of years to load.
    clean_on_load (bool) - Whether to execute cleaning functions or return the raw data. 
    use_pickle (bool) - Load processed pickles of CEMS data, rather than reprocessing them

    Methods
    -------
    load() - load pickles, process them as needed
        This includes:
            - aggregating multiple units into a single plant
            - resampling hourly data to the desired resoultion ('D' for daily by default)
            - dropping plants without a full year of data
            - dropping plants with a significant amount of missing or nan values
            - dropping plants without any reported emissions
            - converting emission units from tons to lbs
            - cleaning up column names
            - reducing memory consumption by converting dtypes where possible
    
    Attributes
    ----------
    self.cems - the santized long_df of CEMS data

    Assumptions
    -----------
    - see load() method description
    - underlying data (.csvs from FTP) have not changed if use_pickle is True
    """
    def __init__(self, ts_frequency=config.TS_FREQUENCY,
                 years=[2019], clean_on_load=True,
                 use_pickle=True, SQL=None,
                 measurement_flags=config.CEMS_MEASUREMENT_FLAGS):
        
        log.info('\n')
        log.info('Initializing CEMSLoader')

        self.ts_frequency = ts_frequency
        self.years = years
        self.dir_path = os.path.join('data','CEMS','csvs')
        self.clean_on_load = clean_on_load
        self.use_pickle = use_pickle
        years_clean = [str(i) for i in years]
        years_clean = '-'.join(years_clean) #save as seperate caches
        self.pkl_path = os.path.join('data','CEMS','processed',f"cems_{ts_frequency}_{years_clean}.pkl")
        self.SQL = SQL
        self.measurement_flags = measurement_flags

        self.cems = None

    
    def _read_csvs(self):
        log.info('....reading CEMS csvs')
        # --- Get file paths ---
        files = os.listdir(self.dir_path)
        files = [f for f in files if not f.startswith('.')]
        
        year_files = []
        for y in self.years:
            year_files += [i for i in files if str(y) in i]
        
        to_concat = []
        checkpoint = max(1, int(len(files)*0.1))
        done = 0
        for f in files:
            _df = pd.read_csv(os.path.join(self.dir_path, f), low_memory=False)
            to_concat.append(_df)
            done +=1
            if done % checkpoint == 0:
                log.info(f"........finished loading {done}/{len(files)} csvs")
            
        # --- Convert to dataframe ---
        log.info('....concatenating CEMS csvs')
        self.cems = pd.concat(to_concat, axis='rows', sort=False)
        del to_concat
            
        return self
    
    def _clean_cems(self):
        log.info(f"........postprocessing CEMS, len: {len(self.cems)}")
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

        # ---Only take observed CEMS ---
        if len(self.measurement_flags) > 0:
            for c in ['NOX']: # CO2 and SO2 are all calculated or nans
                log.info(f'....removing {c} that are not in {self.measurement_flags}')
                self.cems = self.cems[self.cems[f"{c}_RATE_MEASURE_FLG"].isin(self.measurement_flags)]
                log.info(f'........len after removing: {len(self.cems)}')

        # --- Rename columns ---
        self.cems.rename(rename_dict, axis='columns', inplace=True)

        # --- drop nans ---
        log.info(f"....dropping observations with nans")
        self.cems.dropna(subset=['so2_lbs', 'nox_lbs', 'co2_tons'], inplace=True)
        log.info(f"........len after drop: {len(self.cems)}")

        # --- Convert to datetime ---
        if self.ts_frequency == 'H':
            self.cems['hour'] = [str(i)+':00:00' for i in self.cems['hour']]
            self.cems['datetime_utc'] = pd.to_datetime(self.cems['date'] + ' ' + self.cems['hour'])
        else:
            self.cems['datetime_utc'] = pd.to_datetime(self.cems['date'])

        # --- drop plants without 24 entries in a date ---
        log.info(f'....dropping observations without a full 24 hours of data, len before: {len(self.cems)}')
        self.cems['count'] = self.cems.groupby(['date', 'plant_id_eia','unit'])['plant_id_eia'].transform(lambda x: x.count())
        self.cems = self.cems.loc[self.cems['count'] == 24]
        self.cems.drop(['count', 'date'], axis='columns', inplace=True)
        log.info(f'........len after drop: {len(self.cems)}')
            
         # --- Aggregate by unit ---
        agg_dict = {
            'gross_load_mw':'sum',
            'so2_lbs':'sum',
            'nox_lbs':'sum',
            'co2_tons':'sum',
            'operational_time':'mean',
        }
        self.cems = self.cems.groupby(['plant_id_eia','datetime_utc'], as_index=False).agg(agg_dict)

        # --- Aggregate by ts_frequency ---
        self.cems = self.cems.groupby('plant_id_eia').resample(self.ts_frequency, on='datetime_utc').sum() #TODO: check how resampling works
        self.cems.drop(['plant_id_eia'], axis='columns', inplace=True, errors='ignore') #duplicated by resample
        self.cems.reset_index(inplace=True, drop=False)

        # --- Drop rows with no gross load (causing division by 0 error) ---
        self.cems = self.cems.loc[self.cems['gross_load_mw'] > 0]
        
        # --- Drop unnecessary columns ---
        keep = ['datetime_utc','plant_id_eia',
                'gross_load_mw','so2_lbs','nox_lbs',
                'co2_tons','operational_time']
        self.cems = self.cems[keep]

        # --- convert co2 from tons to lbs ---
        self.cems['co2_lbs'] = self.cems['co2_tons'] * 2000
        self.cems = self.cems.drop(['co2_tons'], axis='columns')
        
        # --- reduce size ---
        self.cems = helper.memory_downcaster(self.cems)

        # --- reset index ---
        self.cems.reset_index(drop=True, inplace=True)
        
        return self
    

    def load(self):
        # --- Try to load aggregated pickle ---
        if self.use_pickle:
            if os.path.exists(self.pkl_path):
                log.info('....reading CEMS from pickle')
                self.cems = pd.read_pickle(self.pkl_path)
        
        # --- Calculate aggregate df from csvs ---
        if not isinstance(self.cems, pd.DataFrame):
            self._read_csvs()
            self._clean_cems()
            
            # --- Save pickle ---
            if self.use_pickle:
                log.info('....saving CEMS to pickle')
                self.cems.to_pickle(self.pkl_path, protocol=4)
                # --- Out ---
        
        if self.SQL is not None:
            self.SQL.pandas_to_sql(self.cems, 'cems_merged')
                
        return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ LOAD WRI GPPD DATA ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GPPDLoader():
    """
    Load the Global Power Plant Database from WRI. 

    The GPPD is saved locally as a .csv, this performs some simple cleaning and
    loads it as a geopandas GeoDataFrame with a 'geometry' column holding points.

    Inputs
    ------
    round_coords_at (int) - round lat/lon coords at a decimal point to simplify future queries,
        and potentially merge generators located extremely close together as one. 
    countries (list) - List of countries to include from the GPPD

    Methods
    -------
    load() - Load and clean the GPPD from .csv 

    Attributes
    ----------
    self.gppd (GeoDataFrame)

    Assumptions
    -----------
    - 'Coal','Oil','Gas','Petcoke','Cogeneration' resources are the only generator types with emissions. 
    """

    def __init__(self,
                 round_coords_at=3,
                 countries=config.TRAIN_COUNTRIES,
                 SQL=None):

        log.info('\n')
        log.info('Initializing GPPDLoader')
        
        self.round_coords_at = round_coords_at #.01 degrees = 1 km
        self.countries = set(countries)
        self.SQL = SQL

        
    def _load_csv(self):
        csv_path = os.path.join('data','wri')
        self.gppd = pd.read_csv(os.path.join(csv_path,'global_power_plant_database.csv'), low_memory=False)
        return self
    
    
    def _make_geopandas(self):
        
        self.gppd = gpd.GeoDataFrame(
            self.gppd, geometry=gpd.points_from_xy(self.gppd['longitude'], self.gppd['latitude']))
        self.gppd.crs = "EPSG:4326"
        
        return self
    
    
    def _clean_gppd(self):
        
        keep = [
            'country',
            'country_long',
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
            'plant_id_wri'
        ]

        # --- change puerto rico to its own country ---
        pr = pd.read_csv(os.path.join('data', 'wri', 'gppd_120_pr.csv'))
        pr_ids = list(set(pr['gppd_idnr']))
        self.gppd.loc[self.gppd['gppd_idnr'].isin(pr_ids), 'country_long'] = 'Puerto Rico'
        self.gppd.loc[self.gppd['gppd_idnr'].isin(pr_ids), 'country'] = 'PRI'

        # --- Subset to countries ---
        if 'WORLD' in self.countries: #all countries!
            log.info('....including all countries in GPPD!')
            pass

        else:
            for country in self.countries:
                try:
                    assert country in list(self.gppd['country_long'])
                except AssertionError:
                    others = [c for c in list(set(self.gppd['country_long'])) if c[0:2] == country[0:2]]
                    log.error(f"{country} not in gppd, did you mean {others}?")

            log.info(f'....subsetting gppd to include {self.countries}')
            self.gppd = self.gppd.loc[self.gppd['country_long'].isin(self.countries)]
        
        # --- Round lat lon ---
        self.gppd[['latitude','longitude']] = self.gppd[['latitude','longitude']].round(self.round_coords_at)
        
        # --- Drop non fossil fuels ---
        self.gppd = self.gppd.loc[self.gppd['primary_fuel'].isin(['Coal','Oil','Gas','Petcoke'])]
        
        # --- Rename columns ---
        self.gppd.rename({ 
                            'capacity_mw':'wri_capacity_mw',
                            'gppd_idnr':'plant_id_wri',
                        }, axis='columns', inplace=True)

        # --- fill generation nans ---
        gen_cols = [
            'generation_gwh_2013',
            'generation_gwh_2014',
            'generation_gwh_2015',
            'generation_gwh_2016',
            'generation_gwh_2017',
            'estimated_generation_gwh'
        ]

        self.gppd[gen_cols] = self.gppd[gen_cols].replace(0, np.nan) \
                                    .interpolate(method='linear', axis='columns') \
                                    .fillna(method='backfill', axis='columns') \
                                    .fillna(method='ffill', axis='columns')
        
        # --- filter columns we want ---
        self.gppd = self.gppd[keep]

        # --- drop rows without a WRI id ---
        self.gppd = self.gppd.dropna(subset=['plant_id_wri'])
        
        return self
    
    
    def load(self):
        log.info(f"....Loading gppd from csv")
        # --- Read csv ---
        self._load_csv()
        
        # --- Clean df ---
        self._clean_gppd()
        
        # --- Make Geopandas ---
        self._make_geopandas()

        # --- Out ---
        if self.SQL is not None:
            self.SQL.pandas_to_sql(self.gppd, 'gppd_merged')
        
        return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ LOAD PUDL 860/923 ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PUDLLoader():
    """
    Load plant/generator level data from the Public Utility Data Liberation (PUDL)
    project, which includes EIA 860 and 923 forms for the US, listing all MW scale
    generators along with data surronding their capcaity, location, ownership, etc. 

    Credit goes to PUDL for organizing and preprocessingall of this data,
    which is located within the 'data' folder for the purposes of this project.

    Inputs
    ------
    years (list): list of years to grab data for. EIA 860 is released in October for the previous year.
        Only grabbing data for one year for this project. 
    round_coords_at (int): round lat/lon coords at a decimal point to simplify future queries,
        and potentially merge generators located extremely close together as one. 

    Methods
    -------
    load() - initializes a load, clean merge process, and returns a geopandas GeoDataFrame with a
        geometry point as self.eightsixty. 
    
    Attributes
    ----------
    self.eightsixty - main output dataframe
    self.plants - Form 923 data, primarily used for lat/lons. Not used. 

    Assumptions
    ---------        
    - retirements within next two years are removed from the report_year
    - only operational generators (those deemed 'existing') are taken
    - only outputs fossil generators with the 'fuel_type_code_pudl' of 'coal','gas', or 'oil'
    - multiple units at a single plant are aggregated together
    """

    def __init__(self, years=[2018],
                 round_coords_at=3,
                 SQL=None):
        
        log.info('\n')
        log.info('Initializing PUDLLoader')

        self.years = years
        self.round_coords_at = round_coords_at
        self.SQL = SQL

        self.db_path = os.path.join('data','pudl-work','sqlite','pudl.sqlite')
    
    def _unzip(self):
        log.info('........unzipping sqlite db')
        db_folder_path = os.path.join('data','pudl-work','sqlite')
        with zipfile.ZipFile(os.path.join(db_folder_path, 'pudl.sqlite.zip'), 'r') as zip_ref:
            zip_ref.extractall(db_folder_path)
        return self

    def _connect_to_sqlite(self):
        log.info('....connecting to sqlite for 860/923 data')
        if os.path.exists(self.db_path):
            pass
        else:
            self._unzip()

        self.engine = sqlite3.connect(self.db_path)
        return self
    
    def _load_plants_entity_eia(self):
        log.info('....loading plant level data')
        self.plants = pd.read_sql_query("SELECT * FROM plants_entity_eia", self.engine)
        self.plants = helper.memory_downcaster(self.plants)
        return self
    
    
    def _load_generators_eia860(self):
        log.info('....loading generator eightsixty data')
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
        log.info(f"........filtering to report years: len {len(self.eightsixty)}")

        # --- take out possible retirements within next two years ---
        self.eightsixty['planned_retirement_year'].fillna(2099, inplace=True) #fill in nans for plants with no planned retirement
        self.eightsixty = self.eightsixty.loc[self.eightsixty['planned_retirement_year'] > self.eightsixty['report_year'] + 2]
        log.info(f"........filtering out retirements in next year: len {len(self.eightsixty)}")

        # --- only take operational assets ---
        self.eightsixty = self.eightsixty.loc[self.eightsixty['operational_status'] == 'existing']
        log.info(f"........filtering out non-operational assets: len {len(self.eightsixty)}")

        # --- only take fossil generators ---
        self.eightsixty = self.eightsixty.loc[self.eightsixty['fuel_type_code_pudl'].isin(['coal','gas','oil'])]
        log.info(f"........filtering out non-fossil generators: len {len(self.eightsixty)}")
        
        # --- filter out columns ---
        self.eightsixty = self.eightsixty[keep]
        
        # --- groupby to reduce multiple generators at one plant ---
        self.eightsixty = self.eightsixty.groupby(['plant_id_eia','report_year'], as_index=False).agg(agg_dict)
        log.info(f"........reducing generators to plant level: len {len(self.eightsixty)}")

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
        
    
    def _merge_dfs(self):
        log.info('....merging dfs')
        # --- Merge plant data with generator data ---
        self.eightsixty = self.eightsixty.merge(self.plants, on='plant_id_eia', how='inner')
        log.info(f"........merging eightsixty with plants on plant_id_eia: len {len(self.eightsixty)}")
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
        self._merge_dfs() #merge eightsixty and plants
        
        # --- Make Geopandas ---
        self._make_geopandas()

        # --- Out ---
        if self.SQL is not None:
            self.SQL.pandas_to_sql(self.eightsixty, 'eightsixty_merged')
        
        return self

class L3Loader():
    def __init__(self,
                 s3_dbs=config.S3_DBS,
                 checkpoint_freq=0.05,
                 agg_func = np.nansum): #lat/lon degrees 0.1 ~ 11 km. TODO: calculate this more accurately as this changes with latitude):
        
        self.s3_dbs = s3_dbs
        self.checkpoint_freq = checkpoint_freq
        
        self.agg_func = agg_func

        with open(os.path.join('models','bayes_buffer_params.pkl'), 'rb') as handle:
            self.bayes_param_dict = pickle.load(handle)
        
        self.read_file_dir = os.path.join('data','s3','clean')
        
    def _loader(self, start_date, db):
        """
        Load multiple days worth of arrays, and return them in a list.
        
        Inputs
        ------
        start_date (str) - YYYY-MM-DD representation of date
        TS_FREQUENCY (pd code) from config
        """
        
        # --- find time delta --- 
        delta = pd.to_timedelta(config.TS_FREQUENCY)
        
        # --- find dates to aggregate ---
        dates_to_agg = []
        for i in range(delta.days):
            day = pd.Timestamp(start_date) + timedelta(days=i)
            if day.year <= config.MAX_YEAR:
                day = day.strftime('%Y-%m-%d')
                dates_to_agg.append(day)
        
        # --- convert L2 db name to L3 file convention ---
        read_db = db.split(os.path.sep)[-2].replace('L2','L3')
        
        # --- associate dates with file paths ---
        files = [f"{read_db}{day}.tif" for day in dates_to_agg]
        
        # --- open files with rasterio ---
        arrays = []
        for f in files:  
            try:
                # --- read as numpy array ---
                srcr = rasterio.open(os.path.join(self.read_file_dir, f))      
                array = srcr.read()
                arrays.append(array)
                profile = srcr.profile
                srcr.close()
            except rasterio.errors.RasterioIOError:
                breakpoint()
                # log.warning(f'Warining, could not find {f}') #TODO FIX THIS for 12-30 and 12-31 missing!
                pass

        return arrays, profile

    def _sector_of_circle_on_row(self, row, direction):
        """Helper function to apply sector_of_circle on a dataframe."""
        if direction == 'up':
            return self._sector_of_circle(row['center'], row['up_wind_start_angle'], row['up_wind_end_angle'], row['radius'])
        elif direction == 'down':
            return self._sector_of_circle(row['center'], row['down_wind_start_angle'], row['down_wind_end_angle'], row['radius'])


    def _sector_of_circle(self, center, start_angle, end_angle, radius, steps=50):
        """Helper function to calculate a mask comprising a segment of a circle."""
        #https://stackoverflow.com/questions/54284984/sectors-representing-and-intersections-in-shapely
        
        def polar_point(origin_point, angle, distance):
            return [origin_point.x + math.sin(math.radians(angle)) * distance, origin_point.y + math.cos(math.radians(angle)) * distance]

        if start_angle > end_angle:
            start_angle = start_angle - 360
        else:
            pass
        step_angle_width = (end_angle-start_angle) / steps
        sector_width = (end_angle-start_angle) 
        segment_vertices = []
    
        segment_vertices.append(polar_point(center, 0,0))
        segment_vertices.append(polar_point(center, start_angle,radius))
    
        for z in range(1, steps):
            segment_vertices.append((polar_point(center, start_angle + z * step_angle_width,radius)))
        segment_vertices.append(polar_point(center, start_angle+sector_width,radius))
        segment_vertices.append(polar_point(center, 0,0))
        return Polygon(segment_vertices)

    def db_worker(self, db_df, db, buffer_angle=None, buffer_radius=None):

        db_df['db'] = db
        db_df = db_df[['db','datetime_utc','plant_id_wri', 'wind_deg_from','longitude','latitude']]

        start = time.time()

        # --- fetch params from bayes dict ---
        if buffer_angle == None:
            buffer_angle = self.bayes_param_dict[db]['params']['angle']
        if buffer_radius == None:
            buffer_radius = self.bayes_param_dict[db]['params']['radius']
        db_df['radius'] = buffer_radius
        
        # --- construct start and end angles for buffers ---
        db_df['center'] = [Point(row['longitude'], row['latitude']) for _, row in db_df.iterrows()]#TODO: reverse these and see effect
        db_df.drop(['longitude','latitude'], axis='columns', inplace=True)
        db_df['down_wind_start_angle'] = 180 + db_df['wind_deg_from'] - (buffer_angle / 2)
        db_df['down_wind_end_angle'] = 180 + db_df['wind_deg_from'] + (buffer_angle / 2)
        db_df['up_wind_start_angle'] =  db_df['wind_deg_from'] - (buffer_angle / 2)
        db_df['up_wind_end_angle'] = db_df['wind_deg_from'] + (buffer_angle / 2)

        # --- deal with angles greater than 360 or negative ---
        def clean_angle(angle):
            if angle > 360:
                return angle - 360
            elif angle < 0:
                return 360 + angle
            else:
                return angle

        db_df['down_wind_start_angle'] = db_df['down_wind_start_angle'].apply(clean_angle)
        db_df['up_wind_end_angle'] = db_df['up_wind_end_angle'].apply(clean_angle)
        db_df['down_wind_start_angle'] = db_df['down_wind_start_angle'].apply(clean_angle)
        db_df['down_wind_end_angle'] = db_df['down_wind_end_angle'].apply(clean_angle)

        # --- create masks of buffers ---
        db_df['up_wind_mask'] = db_df.apply(self._sector_of_circle_on_row, direction='up', axis=1)
        db_df['down_wind_mask'] = db_df.apply(self._sector_of_circle_on_row, direction='down', axis=1)

        checkpoint = max(int(len(set(db_df['datetime_utc'])) * 0.05), 1)
        dates_done = 0
        for date in set(db_df['datetime_utc']):

            # --- load arrays ---
            try:
                arrays, profile = self._loader(date, db)
            except Exception as e:
                log.warning(f'Failed on self._loader for {date}')
                continue

            # --- aggregate to timeslice size (mosaic) ---
            profile['dtype'] = 'uint32'
            agg = np.sum(arrays, axis=0, dtype='uint32')

            with MemoryFile() as memfile:
                with memfile.open(**profile) as dataset: # Open as DatasetWriter
                    dataset.write(agg)
                
                with memfile.open() as dataset:  # Reopen as DatasetReader

                    for ix, row in db_df.loc[db_df['datetime_utc'] == date].iterrows():
        
                        # --- apply masks ---
                        up_wind_masked, up_masked_transform = mask(dataset, [row['up_wind_mask']], crop=True)
                        down_wind_masked, down_masked_transform = mask(dataset, [row['down_wind_mask']], crop=True)

                        # --- replace null values ---
                        up_wind_masked[up_wind_masked==np.nan] = 0
                        down_wind_masked[down_wind_masked==np.nan] = 0
                                
                        # --- aggregate ---
                        up_wind_val = self.agg_func(up_wind_masked)
                        down_wind_val = self.agg_func(down_wind_masked)

                        db_df.at[ix, 'up_wind_val'] = up_wind_val
                        db_df.at[ix, 'down_wind_val'] = down_wind_val

        return db_df[['plant_id_wri','datetime_utc','db','up_wind_val','down_wind_val']]


    def calculate(self, df):
        """Components or self.worker that can be vectorize."""

        # --- Calculate Wind Speed ---
        df['wind_spd'] = np.vectorize(helper.calc_wind_speed)(df['u_component_of_wind_10m'], df['v_component_of_wind_10m'])
        df['wind_deg_from'] = np.vectorize(helper.calc_wind_deg)(df['u_component_of_wind_10m'], df['v_component_of_wind_10m'])

        if config.MULTIPROCESSING:
            # --- for each db ---
            result_db_dfs = []
            with cf.ProcessPoolExecutor(max_workers=min(config.WORKERS, len(self.s3_dbs))) as executor:

                futures = [executor.submit(self.db_worker, df.copy(), db) for db in self.s3_dbs]

                for f in cf.as_completed(futures):
                    result = f.result()
                    result_db_dfs.append(result)
                    log.info(f"........finished db {result['db'][0]}")
        
        else:
            result_db_dfs = [self.db_worker(df.copy(), db) for db in self.s3_dbs]


        # --- concat db_dfs ---
        long_df = pd.concat(result_db_dfs, axis='rows')
        up_long_df = long_df[['plant_id_wri','datetime_utc','db','up_wind_val']]
        down_long_df = long_df[['plant_id_wri','datetime_utc','db','down_wind_val']]

        # --- make wide df ---
        up_wide_df = pd.pivot_table(up_long_df, index=['plant_id_wri', 'datetime_utc'], columns='db', values='up_wind_val')
        down_wide_df = pd.pivot_table(down_long_df, index=['plant_id_wri', 'datetime_utc'], columns='db', values='down_wind_val')
        up_wide_df.columns = [f'up_wind_{c}' for c in up_wide_df.columns]
        down_wide_df.columns = [f'down_wind_{c}' for c in down_wide_df.columns]
        wide_df = pd.concat([up_wide_df, down_wide_df], axis='columns')
        wide_df.reset_index(inplace=True)

        # --- merge on df ---
        df = df.merge(wide_df, on=['datetime_utc','plant_id_wri'], how='left')

        return df

class L3Optimizer(L3Loader):

    def __init__(self, bayes_params={'angle':(1, 90), 'radius':(0.05, 0.6)}, n_samples=100,
                 init_points=config.BAYES_INIT_POINTS, n_iter=config.BAYES_N_ITER):
        
        L3Loader.__init__(self)

        self.col_map = {
                  'COGT/OFFL/L2__NO2___/':'nox_lbs',
                  'COGT/OFFL/L2__SO2___/':'so2_lbs',
                  'COGT/OFFL/L2__HCHO__/':'co2_lbs',
                  'COGT/OFFL/L2__CH4___/':'co2_lbs',
                  'COGT/OFFL/L2__CO____/':'co2_lbs',
                  'COGT/OFFL/L2__AER_AI/':'co2_lbs',

        }
        self.bayes_params = bayes_params
        self.n_samples = n_samples
        self.init_points = init_points
        self.n_iter = n_iter

    def _bayes_marginal_worker(self, sample, db, angle, radius):
        """Wrapper around db_worker to return pearsonr correlation score from average."""
        db_df = self.db_worker(db_df=sample, db=db, buffer_angle=angle, buffer_radius=radius)

        # --- calculate marginal ---
        db_df[f'marginal'] = db_df[f'down_wind_val'] - db_df[f'up_wind_val']

        # --- merge onto sample ---
        merged = sample.copy()
        merged = merged.merge(db_df, on=['datetime_utc','plant_id_wri'], how='left')

        # --- return pearson r score ---
        true_col = self.col_map[db]
        score = merged[true_col].corr(merged[f'marginal'], method='pearson')
        return score

    def bayes_db_worker(self, sample, db):
        """Given a set of bayes params, calculate the emissions on row."""

        log.info(f'....starting bayesian optimization for {db}')
        func = functools.partial(L3Optimizer._bayes_marginal_worker, self=self, sample=sample, db=db)

        bounds_transformer = SequentialDomainReductionTransformer()
        optimizer = BayesianOptimization(
                        f=func,
                        pbounds=self.bayes_params,
                        random_state=1,
                        bounds_transformer=bounds_transformer,
        )

        # --- Optimize ---
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )

        return (db, optimizer.max)

    def optimize(self, df):

        log.info(f'starting bayesian optimization of buffer size using {self.n_samples}')
        
        # --- subset df into sample size ---
        plant_ids = list(set(df['plant_id_wri']))
        plant_ids = sorted(plant_ids)
        random.Random(42).shuffle(plant_ids)
        keep = plant_ids[0:self.n_samples]
        sample = df.loc[df['plant_id_wri'].isin(keep)]

        # --- Calculate Wind Speed ---
        sample['wind_spd'] = np.vectorize(helper.calc_wind_speed)(sample['u_component_of_wind_10m'], sample['v_component_of_wind_10m'])
        sample['wind_deg_from'] = np.vectorize(helper.calc_wind_deg)(sample['u_component_of_wind_10m'], sample['v_component_of_wind_10m'])
            
         # --- for each db ---
        bayes_best_results = {}
        if config.MULTIPROCESSING:
            with cf.ProcessPoolExecutor(max_workers=min(config.WORKERS, len(self.s3_dbs))) as executor:

                futures = [executor.submit(self.bayes_db_worker, sample.copy(), db) for db in self.s3_dbs]

                for f in cf.as_completed(futures):
                    db, best_params = f.result()
                    bayes_best_results[db] = best_params
                    log.info(f"........finished optimizing {db}, best params: {best_params}")
        else:
            for db in self.s3_dbs:
                bayes_best_results[db] = self.bayes_db_worker(sample.copy(), db)

        # --- Save best params to pickle ---
        if self.n_samples > 10: #ignore if testing
            with open(os.path.join('models', 'bayes_buffer_params.pkl'), 'wb') as handle:
                pickle.dump(bayes_best_results, handle)

        return self
