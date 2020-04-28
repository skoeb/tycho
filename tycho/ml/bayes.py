import copy

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

import tycho.config as config

import logging
log = logging.getLogger("tycho")


class BayesRegressor(RegressorMixin):
    def __init__(self,
                estimator, pbounds,
                cv_opt_func='neg_mean_absolute_error', cv_folds=config.CV_FOLDS,
                bayes_n_iter=config.BAYES_N_ITER, bayes_init_points=config.BAYES_INIT_POINTS,
                bayes_kappa=1, bayes_acq=config.BAYES_ACQ
            ):
        self.estimator = estimator
        self.pbounds = pbounds
        
        self.cv_folds = cv_folds
        self.cv_opt_func = cv_opt_func

        self.bayes_n_iter = bayes_n_iter
        self.bayes_init_points = bayes_init_points
        self.bayes_kappa = bayes_kappa
        self.bayes_acq = bayes_acq

    def _cast_discrete(self):
        """Convert string and discrete vars (represeted by lists in dict) into ints."""
        self.discrete_var_lookup = {}
        for k, v in self.pbounds.items():
            if isinstance(v, list):
                self.discrete_var_lookup[k] = dict(zip(range(0, len(v)), v))
                self.pbounds[k] = (0, len(v) - 1)
        return self
    
    def _save_ints(self):
        """Ensure that params that should be ints (such as n_estimators) remain so."""
        self.int_var_list = []
        for k, v in self.pbounds.items():
            if isinstance(v[0], int):
                self.int_var_list.append(k)
        return self

    def _cv_worker(self, **kwargs):
        """Initiate a regressor and run through CV."""

        # --- clean up params ---
        for k,v in kwargs.items():
            if k in self.int_var_list: #cast ints
                kwargs[k] = int(v)
            if k in self.discrete_var_lookup.keys(): #cast disrete back
                kwargs[k] = self.discrete_var_lookup[k][int(v)]

        # --- copy initial estimator and update param arguments ---
        est = copy.deepcopy(self.estimator)
        est.set_params(**kwargs)

        # --- fit w/ cv ---
        cv = cross_val_score(est, self.X, self.y, scoring=self.cv_opt_func, cv=self.cv_folds)
        return cv.mean() / 1000
    
    def _fit_best_model(self):
        # --- copy initial estimator and update param arguments ---
        est = copy.deepcopy(self.estimator)
        est.set_params(**self.best_params)
        est.fit(self.X, self.y)
        return est

    def fit(self, X, y):
        
        # --- Save X/ys as attributes so worker can access ---
        self.X = X; self.y = y

        # --- convert discrete vars ---
        self._cast_discrete()
        self._save_ints()

        # --- run bayes ---
        optimizer = BayesianOptimization(
            f=self._cv_worker,
            pbounds=self.pbounds,
            random_state=1234,
            verbose=2
        )

        optimizer.maximize(
            init_points=self.bayes_init_points,
            n_iter=self.bayes_n_iter,
            kappa=self.bayes_kappa,
            acq=self.bayes_acq
        )
        
        # --- expose predictions ---
        self.best_score = optimizer.max['target']
        self.best_params = optimizer.max['params']
        self.evaluated_models = optimizer.res

        # --- clean up params ---
        for k,v in self.best_params.items():
            if k in self.int_var_list: #cast ints
                self.best_params[k] = int(v)
            if k in self.discrete_var_lookup.keys(): #cast disrete back
                self.best_params[k] = self.discrete_var_lookup[k][int(v)]

        log.info(f'....best score from BayesOptimizer: {self.best_score}')
        log.info(f'....best params: {self.best_params}')

        # --- rerun for best ---
        self.best_model = self._fit_best_model()
    
    def predict(self, X):
        pred = self.best_model.predict(X)
        pred = np.clip(pred, 0, None)
        return pred

"""
TODO:
    -add to train.py
    -run w/ eval set as init arg? 
    -inherit for lgbm
    -add parametric / select percentile to preprocessing pipe
    -run within TPOT???
"""