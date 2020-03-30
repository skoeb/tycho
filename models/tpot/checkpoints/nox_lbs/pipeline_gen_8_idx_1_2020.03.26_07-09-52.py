import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -608684838.4
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.013000000000000001),
    XGBRegressor(colsample_bytree=0.8999999999999999, gamma=0.5, learning_rate=0.1, max_depth=7, min_child_weight=2, n_estimators=500, nthread=1, objective="reg:squarederror", reg_alpha=146, reg_lambda=101, subsample=0.9500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
