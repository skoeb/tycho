import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -1354737740.8
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    SelectPercentile(score_func=f_regression, percentile=34),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    XGBRegressor(colsample_bytree=0.5, gamma=0.4, learning_rate=0.1, max_depth=8, min_child_weight=3, n_estimators=200, nthread=1, objective="reg:squarederror", reg_alpha=46, reg_lambda=31, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
