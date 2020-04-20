import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -88621.96588556988
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    LGBMRegressor(boosting_type="gbdt", colsample_bytree=0.8, learning_rate=0.03, max_bin=256, max_depth=11, n_estimators=512, num_leaves=64, objective="mae", reg_alpha=8, reg_lambda=2, subsample=0.4)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
