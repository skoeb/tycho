import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -1448965423.0265622
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=15),
    StackingEstimator(estimator=DecisionTreeRegressor(ccp_alpha=0.01, criterion="mse", max_depth=10, max_features=4, min_samples_leaf=6, min_samples_split=10)),
    DecisionTreeRegressor(ccp_alpha=0.01, criterion="mse", max_depth=10, max_features=16, min_samples_leaf=1, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
