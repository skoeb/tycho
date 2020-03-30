import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -504695525.24499655
exported_pipeline = XGBRegressor(colsample_bytree=0.5, gamma=0.4, learning_rate=0.1, max_depth=5, min_child_weight=18, n_estimators=350, nthread=1, objective="reg:squarederror", reg_alpha=1, reg_lambda=146, subsample=0.6500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
