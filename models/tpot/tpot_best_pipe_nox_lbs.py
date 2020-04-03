import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -16061314.2
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.01),
    VarianceThreshold(threshold=0.01),
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.51, gamma=0.67, learning_rate=0.4, max_depth=6, min_child_weight=8, n_estimators=100, nthread=12, objective="reg:squarederror", reg_alpha=83, reg_lambda=91, subsample=0.3)),
    XGBRegressor(colsample_bytree=0.28, gamma=0.77, learning_rate=0.2, max_depth=4, min_child_weight=9, n_estimators=75, nthread=12, objective="reg:squarederror", reg_alpha=60, reg_lambda=33, subsample=0.88)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
