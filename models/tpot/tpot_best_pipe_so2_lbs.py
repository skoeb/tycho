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

# Average CV score on the training set was: -23398886.4
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.007),
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.47000000000000003, gamma=0.17, learning_rate=0.30000000000000004, max_depth=6, min_child_weight=15, n_estimators=50, nthread=12, objective="reg:squarederror", reg_alpha=47, reg_lambda=19, subsample=0.51)),
    XGBRegressor(colsample_bytree=0.43, gamma=0.39, learning_rate=0.1, max_depth=1, min_child_weight=16, n_estimators=200, nthread=12, objective="reg:squarederror", reg_alpha=25, reg_lambda=20, subsample=0.28)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
