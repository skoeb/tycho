import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler
from tpot.builtins import OneHotEncoder

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -26224.71210269184
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    MinMaxScaler(),
    Binarizer(threshold=0.9),
    LGBMRegressor(boosting_type="gbdt", colsample_bytree=0.8, learning_rate=0.03, max_bin=64, max_depth=7, n_estimators=128, num_leaves=64, objective="mae", reg_alpha=2, reg_lambda=32, subsample=0.6)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
