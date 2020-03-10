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

log = logging.getLogger("loader")

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
    def __init__(self, ts_frequency='D', years=[2019], clean_on_load=True,
                 use_pickle=True):
        
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
        ten_percent = max(1, int(len(files)*0.1))
        done = 0
        for f in files:
            _df = pd.read_csv(os.path.join(self.dir_path, f))
            if self.clean_on_load:
                _df = self._clean_cems(_df)

            to_concat.append(_df)
            done +=1
            if done % ten_percent == 0:
                log.info(f"........finished loading {done}/{len(files)} csvs")
            
        # --- Convert to dataframe ---
        log.info('....concatenating CEMS csvs')
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
        
        log.info(f"........postprocessing CEMS, len: {len(self.cems)}")
        
        # --- drop plants without a full year of data ---
        plant_id_eias_keep = list(set(self.cems.groupby('plant_id_eia', as_index=False)['plant_id_eia'].filter(lambda x: x.count() == 365)))
        self.cems = self.cems.loc[self.cems['plant_id_eia'].isin(plant_id_eias_keep)]
        log.info(f"........droping generators without a full year of data, len: {len(self.cems)}")
        
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
            self._post_load_clean()
            
            # --- Save pickle ---
            if self.use_pickle:
                log.info('....saving CEMS to pickle')
                self.cems.to_pickle(self.pkl_path)
                
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

    def __init__(self,round_coords_at=3, countries=['United States of America']):

        log.info('\n')
        log.info('Initializing GPPDLoader')
        
        self.round_coords_at = round_coords_at #.01 degrees = 1 km
        self.countries = countries

        
    def _load_csv(self):
        self.gppd = pd.read_csv(os.path.join('data','wri','global_power_plant_database.csv'))
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
        log.info(f"........filtering gppd to include {self.countries}")
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
        log.info(f"....Loading gppd from csv")
        # --- Read csv ---
        self._load_csv()
        
        # --- Clean df ---
        self._clean_gppd()
        
        # --- Make Geopandas ---
        self._make_geopandas()
        
        return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ STITCH EARTH ENGINE ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EarthEngineLoader():
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
    def __init__(self, earthengine_dbs, buffers, ts_frequency='D'):
        self.earthengine_dbs = earthengine_dbs
        self.buffers = buffers
        self.ts_frequency = ts_frequency

        self.pickle_path = os.path.join('data','earthengine')
    
    def _read_pickles(self, files):
        files = os.listdir(self.pickle_path)  
        self.clean_files = []
        # --- find out which files to read in --
        for f in files:
            if '&' in f:
                db, ts, m = f.split('&')
                if (db in self.earthengine_dbs) & (ts == self.ts_frequency):
                    self.clean_files.append(f) 
        import pdb; pdb.set_trace()

        # --- read files and concat ---
        dfs = []
        for f in self.clea_files:
            dfs.append(pd.read_pickle(os.path.join(self.pickle_path, f)))
        
        # --- concat dfs into long earthengine df ---
        self.earthengine = pd.concat(dfs, axis='rows', sort=False)

        return self

    def _pivot_buffers(self):
        """Pivot multiple buffers into a wider df."""
        self.earthengine['buffer_variable'] = self.earthengine['buffer'].astype(int).astype(str) + "/" + self.earthengine['variable']
        self.pivot = self.earthengine.pivot_table(index=['plant_id_eia','datetime_utc'], columns='buffer_variable',values='value')
        self.pivot.reset_index(drop=False, inplace=True)
        return self

    def _merge_pivot(self, df):
        """Merge pivot onto df (continaing generators and CEMS if training)."""
        self.merged = df.merge(self.pivot, on=['plant_id_eia', 'datetime_utc'])
        return self

    def _clean(self):

        # --- Drop any duplicates ---
        self.merged = self.merged.drop_duplicates(subset=['plant_id_eia','datetime_utc'])

        # --- Drop any nans ---
        self.merged = self.merged.dropna(subset=['plant_id_eia','datetime_utc'])

        return self
    
    def merge(self, df):
        self._read_pickles()
        self._pivot_buffers()
        self._merge_pivot(df)
        self._clean()
        return self.merged


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
                 round_coords_at=3):
        
        log.info('\n')
        log.info('Initializing PUDLLoader')

        self.years = years
        self.round_coords_at = round_coords_at
        
    def _connect_to_sqlite(self):
        log.info('....connecting to sqlite for 860/923 data')
        self.db_path = os.path.join('data','pudl-work','sqlite','pudl.sqlite')
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
        
        return self