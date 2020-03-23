import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -121988816.8449157
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    SelectFwe(score_func=f_regression, alpha=0.019),
    MaxAbsScaler(),
    ZeroCount(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.30000000000000004, tol=0.001)),
    DecisionTreeRegressor(max_depth=9, min_samples_leaf=5, min_samples_split=4)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
