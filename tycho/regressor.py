import os
import pickle
import requests
import concurrent.futures as cf
import itertools

import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from tslearn.metrics import gamma_soft_dtw, sigma_gak, cdist_gak, cdist_dtw, cdist_soft_dtw

from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np

import config
import helper
import backtester

pd.options.mode.chained_assignment = None

# --- Logging ---
import logging
log = logging.getLogger("tycho")

class XGBRegressor(BaseEstimator):
    def __init__(self, slice_start, slice_end):

        self.slice_start = slice_start
        self.slice_end = slice_end

        log.info('\n')
        log.info('Initializing XGBRegressor')
    

    def load(self):
        log.info(f'........Loading regression model pickle')
        in_path = os.path.join(pdir, 'models','regressor',f'xgb_{self.slice_start}_{self.slice_end}.pkl')
        with open(in_path, 'rb') as handle:
            self.model = pickle.load(handle)
        return self
    

    def predict(self, X, y=None):
        log.info('....predicting with XGBRegressor')
        dmatrix = xgb.DMatrix(X)
        pred = self.model.predict(dmatrix)
        return pred
    

    def _mae_score(self, y, prediction):
        mae = mean_absolute_error(y, prediction)
        log.info(f'....Calculating MAE score {round(mae, 4)}')
        return mae


class GridSearchXGB():
    def __init__(self, slice_start, slice_end,
                 grid_search_params, constant_params,
                 early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, num_boost_round=config.NUM_BOOST_ROUNDS,
                 eval_metric='mae',
                 selection_metric='mae',
                 verbose=True, gpu=False):
        
        log.info('\n')
        log.info('Initializing GridSearchXGB.')

        self.slice_start = slice_start
        self.slice_end = slice_end
        self.selection_metric = selection_metric
        self.grid_search_params = grid_search_params
        self.constant_params = constant_params
        self.num_boost_round=3000
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.gpu = gpu
        
        # --- Handle Eval Metric Optimization ---
        if self.eval_metric in ['auc']:
            self.biggest = True
            self.best_score = 0
        elif self.eval_metric in ['mae']:
            self.biggest = False
            self.best_score = 1e10
        else:
            raise NotImplementedError(f'Please write a wrapper for {eval_metric}!')
    

    def _parameterize(self):
        log.info('....parameterizing grid search')
        # --- Create Products of all Parameters ---
        iterables = [i.tolist() for i in self.grid_search_params.values()]
        products = list(itertools.product(*iterables))

        # --- Create Dicts ---
        products = [dict(zip(self.grid_search_params.keys(), p)) for p in products]
        self.jobs = []
        for p in products:
            d = self.constant_params.copy() #copy default dict
            for k,v in p.items(): #update values for parameters in given product
                d[k] = v
            self.jobs.append(d)
        
        return self
    

    def _xgb_worker(self, _job):
        _clf = xgb.train(
             params=_job,
             dtrain=self.dmatrix_train,
             num_boost_round=self.num_boost_round,
             evals=[(self.dmatrix_test, self.eval_metric)],
             early_stopping_rounds=self.early_stopping_rounds,
             verbose_eval=False)
        return _clf
        
    
    def _xgb(self, X_train, y_train, X_test, y_test):
        log.info('....begininng grid search')
        # --- Run XGBoost on GPU ---
        completed = []
        count = 1
        ten_percent = max(int(len(self.jobs) * 0.1), 1)
        
        
        # --- Initialize DMatrixes ---
        self.dmatrix_train = xgb.DMatrix(X_train, label=y_train)
        self.dmatrix_test = xgb.DMatrix(X_test, label=y_test)
                
        for job in self.jobs:
            
            if self.gpu:
                try: #on gpu
                    gjob = jobs.copy()
                    gjob['tree_method'] = 'gpu_hist'
                    _clf = self._xgb_worker(gjob)

                except Exception as e:
                    log.info('GPU ERROR!')
                    _clf = self._xgb_worker(job)
            else:
                _clf = self._xgb_worker(job)
                
            y_pred = _clf.predict(self.dmatrix_test)
            
            if self.selection_metric == 'mae':
                selection_score = mean_absolute_error(y_test, y_pred)
            else:
                raise NotImplementedError(f'Please write a wrapper for {self.selection_metric}')
                
            # --- Reporting ---
            clean_job = {k:v for k,v in job.items() if k in self.grid_search_params.keys()}
            if self.biggest:
                if selection_score > self.best_score:
                    self.best_score = selection_score
                    log.info(f'........New best selection score! job {count}/{len(self.jobs)}: {self.best_score}')
                    if self.verbose:
                        log.info(clean_job)

            else:
                if selection_score < self.best_score:
                    self.best_score = selection_score
                    log.info(f'........New best selection score! job {count}/{len(self.jobs)}: {self.best_score}')
                    if self.verbose:
                        log.info(clean_job)
            
            if count % ten_percent == 0:
                log.info(f'........on job {count}/{len(self.jobs)}, current best selection score: {self.best_score}')
                    
            # --- Package Score ---
            job[f'best_{self.eval_metric}'] = _clf.best_score
            job['test_selection_score'] = selection_score
            completed.append(job)
            count += 1
            
        # --- Package Completed ---
        self.results_df = pd.DataFrame(completed)
        
        return self
    

    def _rerun_best(self, X_train, y_train, X_test, y_test):
        log.info('....rerunning best XGB with sklearn wrapper')
        # --- Make params complete ---
        _clf = xgb.train(
                 params=self.best_params,
                 dtrain=self.dmatrix_train,
                 num_boost_round=self.num_boost_round,
                 evals=[(self.dmatrix_test, self.eval_metric)],
                 early_stopping_rounds=self.early_stopping_rounds,
                 verbose_eval=False)
        
        self.best_model = _clf
        return self


    def _find_best(self):
        log.info('....finding best XGB model')

        optimize_for = 'test_selection_score'
        
        if self.biggest:
            best = self.results_df.loc[self.results_df[optimize_for] == self.results_df[optimize_for].max()].head(1)
        else:
            best = self.results_df.loc[self.results_df[optimize_for] == self.results_df[optimize_for].min()].head(1)
        
        self.best_params = best.to_dict('records')[0]
        return self


    def save_best(self):
        # --- Save model ---
        out_path = os.path.join(pdir, 'models','regressor',f'xgb_{self.slice_start}_{self.slice_end}.pkl')

        with open(out_path, 'wb') as handle:
            pickle.dump(self.best_model, handle)
    

    def fit(self, X_train, y_train, X_test, y_test):
        self._parameterize()
        self._xgb(X_train, y_train, X_test, y_test)
        self._find_best()
        self._rerun_best(X_train, y_train, X_test, y_test)
        return self

    def score(self, X_train, y_train, X_test, y_test):
            log.info('\n')
            log.info(f'....Best score from grid search: {self.best_score}')
            log.info(f'....Best hyperparameters from grid search: {self.best_params}')

            train_pred = self.best_model.predict(self.dmatrix_train)
            test_pred = self.best_model.predict(self.dmatrix_test)
            test_results = pd.DataFrame({'knn':X_test['pred'].values, 'reg':test_pred.ravel(), 'y':y_test.ravel()}, index=X_test.index)

            train_rmse = mean_squared_error(y_train, train_pred, squared=False)
            test_rmse = mean_squared_error(y_test, test_pred, squared=False)
            knn_rmse = mean_squared_error(y_test, test_results['knn'], squared=False)
            train_mae = mean_absolute_error(y_train, train_pred)

            test_mae = mean_absolute_error(y_test, test_pred)
            knn_mae = mean_absolute_error(y_test, test_results['knn'])

            log.info(f'....Train RMSE: {train_rmse}')
            log.info(f'....Test RMSE: {test_rmse}')
            log.info(f'....Train MAE: {train_mae}')
            log.info(f'....Test MAE: {test_mae}')

            log.info(f'....kNN RMSE: {knn_rmse}')
            log.info(f'....kNN MAE: {knn_mae}')

            log.info(f'....Delta Train MAE: {train_mae - knn_mae}')
            log.info(f'....Delta Train RMSE: {train_rmse - knn_rmse}')

def load_knn_pickles(ts):
    """Accepts a time slice, and loads all kNN outputs for symbols in config."""
    
    # --- Initialize values ---
    slice_start, slice_end = ts
    X_trains = []; X_tests=[]; y_trains = []; y_tests = []

    # --- Load knn outputs ---
    load_folder = os.path.join(pdir, 'data', 'knn_output')
    for symbol in config.SYMBOL_LIST:
        X_trains.append(pd.read_pickle(os.path.join(load_folder, f'X_train_{symbol}_{slice_start}_{slice_end}.pkl')))
        X_tests.append(pd.read_pickle(os.path.join(load_folder, f'X_test_{symbol}_{slice_start}_{slice_end}.pkl')))
        y_trains.append(pd.read_pickle(os.path.join(load_folder, f'y_train_{symbol}_{slice_start}_{slice_end}.pkl')))
        y_tests.append(pd.read_pickle(os.path.join(load_folder, f'y_test_{symbol}_{slice_start}_{slice_end}.pkl')))

    X_train = pd.concat(X_trains, axis='rows', sort=False)
    y_train = pd.concat(y_trains, axis='rows', sort=False)
    X_test = pd.concat(X_tests, axis='rows', sort=False)
    y_test = pd.concat(y_tests, axis='rows', sort=False)

    return X_train, y_train, X_test, y_test

def save_regressor_pickles(ts, X_train, y_train, X_test, y_test):
    slice_start, slice_end = ts
    out_folder = os.path.join(pdir, 'data', 'regressor_output')
    X_train.to_pickle(os.path.join(out_folder, f'X_train_{slice_start}_{slice_end}.pkl'))
    y_train.to_pickle(os.path.join(out_folder, f'y_train_{slice_start}_{slice_end}.pkl'))
    X_test.to_pickle(os.path.join(out_folder, f'X_test_{slice_start}_{slice_end}.pkl'))
    y_test.to_pickle(os.path.join(out_folder, f'y_test_{slice_start}_{slice_end}.pkl'))

def main():
    log.info('Starting main function')

    slice_ends = range(config.OPEN_MINUTE_INT + (config.PERIOD * 3), config.CLOSE_MINUTE_INT - config.PERIOD, config.PERIOD)
    slice_starts = [config.OPEN_MINUTE_INT for i in slice_ends]
    time_slices = list(zip(slice_starts, slice_ends))
    
    for ts in time_slices:
        log.info('....Loading training pickles')
        X_train, y_train, X_test, y_test = load_knn_pickles(ts)

        if config.GRIDSEARCHXGB:
            # ---run grid search ---

            gs = GridSearchXGB(grid_search_params=config.XGB_GS_PARAMS, constant_params=config.XGB_CONSTANT_PARAMS,
                               eval_metric='mae', early_stopping_rounds=25,
                               slice_start=ts[0], slice_end=ts[1])
            gs.fit(X_train, y_train, X_test, y_test)
            gs.score(X_train, y_train, X_test, y_test)
            gs.save_best()

        # --- run regressor ---
        log.info('....Starting Regressor from main function')
        reg = XGBRegressor(slice_start=ts[0], slice_end=ts[1])
        reg.load()

        train_reg = reg.predict(X=X_train)
        test_reg = reg.predict(X=X_test)
        X_train['reg_pred'] = train_reg
        X_test['reg_pred'] = test_reg

        log.info(f'Main train mae for {ts[0]}/{ts[1]}: {mean_absolute_error(y_train, train_reg)}')
        log.info(f'Main test mae for {ts[0]}/{ts[1]}: {mean_absolute_error(y_test, test_reg)}')

        save_regressor_pickles(ts, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    log.info('Starting main')
    main()
