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
import signal

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
import ee

# --- Module Imports ---
import tycho.config
import tycho.helper

def memory_downcaster(df):

    assert isinstance(df, pd.DataFrame) | isinstance(df, pd.Series)

    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.select_dtypes(include=[np.number]).columns:
        # make variables for Int, max and min
        IsInt = False
        mx = df[col].max()
        mn = df[col].min()
        
        # Integer does not support NA, therefore, NA needs to be filled
        if not np.isfinite(df[col]).all(): 
            NAlist.append(col)
            df[col].fillna(mn-1,inplace=True)  
                
        # test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = (df[col] - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True

        
        # Make Integer/unsigned Integer datatypes
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)    
        
        # Make float datatypes 32 bit
        else:
            df[col] = df[col].astype(np.float32)
            
    return df

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)