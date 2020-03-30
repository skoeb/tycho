import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -432028623.0225336
exported_pipeline = XGBRegressor(colsample_bytree=0.8999999999999999, gamma=0.30000000000000004, gpu_id=0, learning_rate=0.5, max_depth=4, min_child_weight=17, n_estimators=500, nthread=1, objective="reg:squarederror", reg_alpha=11, reg_lambda=121, subsample=0.6000000000000001, tree_method="gpu_hist")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
