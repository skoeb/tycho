import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -35170754.58232267
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        SelectPercentile(score_func=f_regression, percentile=61)
    ),
    SelectFwe(score_func=f_regression, alpha=0.004),
    XGBRegressor(colsample_bytree=0.43, gamma=0.14, learning_rate=0.2, max_depth=8, min_child_weight=7, n_estimators=200, nthread=12, objective="reg:squarederror", reg_alpha=149, reg_lambda=141, subsample=0.87)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
