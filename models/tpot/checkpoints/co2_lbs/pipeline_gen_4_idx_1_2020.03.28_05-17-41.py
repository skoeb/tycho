import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import OneHotEncoder

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -1440018623015317.2
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    MinMaxScaler(),
    DecisionTreeRegressor(ccp_alpha=0, criterion="mse", max_depth=10, max_features=32, min_samples_leaf=4, min_samples_split=11)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
