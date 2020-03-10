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

log = logging.getLogger("splitter")

class FourWaySplit():
    """
    X/y - Train/test split similar to sklearn's implementation, but keeping a holdout
    set of generators with all dataes consistent in test.

    Inputs
    ------
    y_columns (list) - during the merges, multiple y columns (from CEMS) are processed. 
        hence, y is potentially returned as a dataframe with multiple columns.
    testfrac (float) - the fraction of generators to split into the test arrays
    seed (int) - seed for shuffle

    Methods
    -------
    split() - accepts a dataframe

    Attributes
    ----------
    self.X_train, self.X_test, self.y_train, self.y_test (DataFrames) - Split

    Assumptions
    -----------
    - a single random shuffle is done (seeded) to the rows of the dataframe
    - that the input dataframe is geographically balanced
    """

    def __init__(self, y_columns, testfrac=0.2, seed=42):
        log.info('\n')
        log.info('Initializing TrainTestSplit')

        self.y_columns = y_columns
        self.testfrac = testfracv
        self.seed = seed
    
    def _train_test_split(self, df):
        """Keep consistant group of generators in test, with size equal to len*testfrac."""
        # --- shuffle ---
        _df = df.sample(frac=1, random_state=self.seed)

        # --- split generator list ---
        gens = list(set(_df['plant_id_eia']))
        train_len = int((1-self.testfrac) * len(gens))
        train_gens = gens[0:train_len]
        test_gens = gens[train_len:]
        
        # --- split data frame ---
        self.train = _df.loc[_df['plant_id_eia'].isin(train_gens)].reset_index(drop=True)
        self.test = _df.loc[_df['plant_id_eia'].isin(test_gens)].reset_index(drop=True)
        return self

    def _X_y_split(self):
        all_columns = self.train.columns
        self.X_columns = [c for c in all_columns if c not in self.y_columns]

        self.X_train = self.train[self.X_columns]
        self.X_test = self.test[self.X_columns]
        self.y_train = self.train[self.y_columns]
        self.y_test = self.test[self.y_columns]
        return self
    
    def split(self, df):
        self._train_test_split(df)
        self._X_y_split()
        return self.X_train, self.X_test, self. y_train, self.y_test