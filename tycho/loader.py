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
                 use_pickle=True, save_pickle=True):
        
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

        # --- Rename columns ---
        self.cems.rename(rename_dict, axis='columns', inplace=True)

        # --- Convert to datetime ---
        if self.ts_frequency == 'H':
            self.cems['hour'] = [str(i)+':00:00' for i in self.cems['hour']]
            self.cems['datetime_utc'] = pd.to_datetime(self.cems['date'] + ' ' + self.cems['hour'])
        else:
            self.cems['datetime_utc'] = pd.to_datetime(self.cems['date'])
            
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
                self.cems.to_pickle(self.pkl_path)
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
                 countries=config.TRAIN_COUNTRIES + config.PREDICT_COUNTRIES,
                 save_pickle=True):

        log.info('\n')
        log.info('Initializing GPPDLoader')
        
        self.round_coords_at = round_coords_at #.01 degrees = 1 km
        self.countries = countries
        self.save_pickle = save_pickle

        
    def _load_csv(self):
        csv_path = os.path.join('data','wri')
        self.gppd = pd.read_csv(os.path.join(csv_path,'global_power_plant_database.csv'), low_memory=False)
        self.pr = pd.read_csv(os.path.join(csv_path, 'gppd_120_pr.csv'))
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
        pr_ids = list(set(self.pr['gppd_idnr']))
        self.gppd.loc[self.gppd['gppd_idnr'].isin(pr_ids), 'country_long'] = 'Puerto Rico'
        self.gppd.loc[self.gppd['gppd_idnr'].isin(pr_ids), 'country'] = 'PRI'

        # --- Filter country ---
        log.info(f"........filtering gppd to include {self.countries}")
        if 'all' not in self.countries: #include all countries
            for country in self.countries:
                assert country in set(self.gppd['country_long'])
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