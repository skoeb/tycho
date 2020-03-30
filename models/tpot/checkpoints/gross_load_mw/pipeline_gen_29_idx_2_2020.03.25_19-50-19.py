import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -360641458.43230104
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.037),
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.7999999999999999, gamma=0.6000000000000001, learning_rate=0.5, max_depth=1, min_child_weight=15, n_estimators=450, nthread=1, objective="reg:squarederror", reg_alpha=116, reg_lambda=26, subsample=0.8500000000000001)),
    XGBRegressor(colsample_bytree=0.7999999999999999, gamma=0.9, learning_rate=0.1, max_depth=10, min_child_weight=5, n_estimators=400, nthread=1, objective="reg:squarederror", reg_alpha=6, reg_lambda=21, subsample=0.7000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
