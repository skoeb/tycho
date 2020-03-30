import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -704891648.0
exported_pipeline = XGBRegressor(colsample_bytree=0.6, gamma=0.4, learning_rate=0.5, max_depth=6, min_child_weight=2, n_estimators=350, nthread=1, objective="reg:squarederror", reg_alpha=131, reg_lambda=131, subsample=0.8)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
