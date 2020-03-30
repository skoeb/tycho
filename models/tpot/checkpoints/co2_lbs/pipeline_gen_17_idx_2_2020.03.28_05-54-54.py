import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import OneHotEncoder, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -1263938262620632.5
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.01),
    VarianceThreshold(threshold=0.0005),
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    DecisionTreeRegressor(ccp_alpha=0.1, criterion="mse", max_depth=9, max_features=32, min_samples_leaf=11, min_samples_split=11)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
