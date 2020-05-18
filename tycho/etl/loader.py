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
import zipfile
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
                 use_pickle=True, save_pickle=True,
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
        self.save_pickle = save_pickle
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
        
        if self.save_pickle:
            self.cems.to_pickle(os.path.join('processed','cems_clean.pkl'))
                
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
                 save_pickle=True):

        log.info('\n')
        log.info('Initializing GPPDLoader')
        
        self.round_coords_at = round_coords_at #.01 degrees = 1 km
        self.countries = set(countries)
        self.save_pickle = save_pickle

        
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
        if None in self.countries: #all countries!
            log.info('....including all countries in GPPD!')
            pass

        else:
            for country in self.countries:
                try:
                    assert country in list(self.gppd['country_long'])
                except AssertionError:
                    others = [c for c in list(set(self.gppd['country_long'])) if c[0:2] == country[0:2]]
                    log.error(f"{country} not in gppd, did you mean {others}?")

            log.info('....subsetting gppd to include {countries}')
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
        if self.save_pickle:
            self.gppd.to_pickle(os.path.join('processed','gppd_clean.pkl'))
        
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
                 save_pickle=True):
        
        log.info('\n')
        log.info('Initializing PUDLLoader')

        self.years = years
        self.round_coords_at = round_coords_at
        self.save_pickle = save_pickle

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
        if self.save_pickle:
            self.eightsixty.to_pickle(os.path.join('processed','eightsixty_clean.pkl'))
        
        return self

class L3Loader():
    def __init__(self,
                 s3_dbs=config.S3_DBS,
                 checkpoint_freq=0.05,
                 if_already_exist='replace',
                 agg_func = np.nansum,
                 buffer_angle = 60,
                 buffer_radius = 0.2): #lat/lon degrees 0.1 ~ 11 km. TODO: calculate this more accurately as this changes with latitude):
        
        assert if_already_exist in ['skip', 'replace']
        
        self.s3_dbs = s3_dbs
        self.checkpoint_freq = checkpoint_freq
        self.if_already_exist = if_already_exist
        
        self.agg_func = agg_func
        self.buffer_angle = buffer_angle
        self.buffer_radius = buffer_radius
        
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
            day = day.strftime('%Y-%m-%d')
            dates_to_agg.append(day)
        
        # --- convert L2 db name to L3 file convention ---
        read_db = db.split(os.path.sep)[-2].replace('L2','L3')
        
        # --- associate dates with file paths ---
        files = [f"{read_db}{day}.tif" for day in dates_to_agg]
        
        # --- open files with rasterio ---
        arrays = []
        for f in files:
            
            # --- read as numpy array ---
            srcr = rasterio.open(os.path.join(self.read_file_dir, f))      
            array = srcr.read()
            arrays.append(array)
            
        # --- get metadata ---
        profile = srcr.profile
        
        return arrays, profile
    
    def _sector_of_circle(self, center, start_angle, end_angle, radius, steps=200):
        """Helper function to calculate a mask comprising a segment of a circle."""
        #https://stackoverflow.com/questions/54284984/sectors-representing-and-intersections-in-shapely
        
        def polar_point(origin_point, angle,  distance):
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
        
    def _aggregate(self, arrays, profile):
        """
        Aggregate list of numpy arrays into a single array (as a sum)
        
        Inputs
        ------
        arrays (list) - list of numpy arrays
        """
        
        # --- aggregate to timeslice size (mosaic) ---
        dtype = profile['dtype']
        agg = np.sum(arrays, axis=0, dtype=dtype)
        
        # --- convert agg from array to raster ---
        # https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
        # with MemoryFile() as memfile:
        memfile = MemoryFile()
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(agg)
                # del data
        
        dataset = memfile.open()
        return dataset
        # with memfile.open() as dataset:  # Reopen as DatasetReader
        #     yield dataset  # Note yield not return    
              
    def _worker(self, job, dataset):
        """
        Given a job (consisting of a lat, lon, and winddirection) and a dataset, calculate the upwind and downwind emissions.
        
        Inputs
        ------
        job (tuple) - (lat, lon, wind direction degrees)
        dataset (rasterio dataset) - result of self._aggregate()
        """

        # --- unpack job ---
        lat = job['latitude']
        lon = job['longitude']
        wind_deg = job['wind_deg_from']
        
        # --- construct start and end angles for buffers ---
        center = Point(lon, lat)
        down_wind_start_angle = 180 + wind_deg - (self.buffer_angle / 2)
        down_wind_end_angle = 180 + wind_deg + (self.buffer_angle / 2)
        up_wind_start_angle =  wind_deg - (self.buffer_angle / 2)
        up_wind_end_angle = wind_deg + (self.buffer_angle / 2)

        # --- deal with angles greater than 360 or negative ---
        def clean_angle(angle):
            if angle > 360:
                return angle - 360
            elif angle < 0:
                return 360 + angle
            else:
                return angle
        
        up_wind_start_angle = clean_angle(up_wind_start_angle)
        up_wind_end_angle = clean_angle(up_wind_end_angle)
        down_wind_start_angle = clean_angle(down_wind_start_angle)
        down_wind_end_angle = clean_angle(down_wind_end_angle)
        
        # --- create masks of buffers ---
        up_wind = self._sector_of_circle(center, up_wind_start_angle, up_wind_end_angle, self.buffer_radius)
        down_wind = self._sector_of_circle(center, down_wind_start_angle, down_wind_end_angle, self.buffer_radius)
        
        # --- apply masks ---
        up_wind_masked, up_masked_transform = mask(dataset, [up_wind], crop=True)
        down_wind_masked, down_masked_transform = mask(dataset, [down_wind], crop=True)
        
        # --- replace null values ---
        up_wind_masked[up_wind_masked==np.nan] = 0
        down_wind_masked[down_wind_masked==np.nan] = 0
        
        # --- aggregate ---
        up_wind_val = self.agg_func(up_wind_masked)
        down_wind_val = self.agg_func(down_wind_masked)
        
        return (job['plant_id_wri'], up_wind_val, down_wind_val)
    
    def calculate(self, df):

        # --- Calculate Wind Speed ---
        df['wind_spd'] = np.vectorize(helper.calc_wind_speed)(df['u_component_of_wind_10m'], df['v_component_of_wind_10m'])
        df['wind_deg_from'] = np.vectorize(helper.calc_wind_deg)(df['u_component_of_wind_10m'], df['v_component_of_wind_10m'])

        # --- for each date ---
        date_dfs = []
        for start_date in set(df['datetime_utc']):
            
            # --- for each db ---
            for db in self.s3_dbs:
            
                # --- subset date ---
                date_df = df.loc[df['datetime_utc'] == start_date]
                
                # --- load arrays ---
                arrays, profile = self._loader(start_date, db)
                
                # --- aggregate into single array ---
                dataset = self._aggregate(arrays, profile)
                
                # --- run df as jobs ---
                log.info(f"....{len(df)} jobs queued for merging")
                if config.MULTIPROCESSING:
                    results = []
                    
                    start = time.time()
                    completed = 0
                    checkpoint = max(1, int(len(date_df) * self.checkpoint_freq))
                    
                    with cf.ThreadPoolExecutor(max_workers=config.THREADS) as executor: 
                        
                        # --- Submit to worker ---
                        futures = [executor.submit(self._worker, job, dataset) for _, job in date_df.iterrows()]
                        
                        for f in cf.as_completed(futures):
                            results.append(f.result())
                            completed += 1
                            
                            if completed % checkpoint == 0:
                                per_download = round((time.time() - start) / completed, 3)
                                eta = round((len(date_df) - completed) * per_download / 3600, 3)
                                log.info(f"........finished job {completed} / {len(date_df)}  ETA: {eta} hours")
                else:
                    results = [self._worker(job, dataset) for _, job in df.iterrows()]
                        
                # --- assemble results ---
                results_df = pd.DataFrame(results, columns=['plant_id_wri',f'{db}_up_wind',f'{db}_down_wind'])
                results_df['datetime_utc'] = start_date
                
                # --- merge results ---
                date_df = df.merge(results_df, on=['plant_id_wri', 'datetime_utc'], how='left')
                
                # --- append to date_dfs ---
                date_dfs.append(date_df)
            
            df = pd.concat(date_dfs, axis='rows')
            
            return df