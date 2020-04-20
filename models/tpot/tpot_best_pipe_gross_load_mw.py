import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -54491.758412349656
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=12),
    StackingEstimator(estimator=LGBMRegressor(boosting_type="dart", colsample_bytree=0.8, learning_rate=0.3, max_bin=512, max_depth=8, n_estimators=1200, num_leaves=20, objective="mae", reg_alpha=0, reg_lambda=2, subsample=0.4)),
    RobustScaler(),
    LGBMRegressor(boosting_type="gbdt", colsample_bytree=0.6, learning_rate=0.03, max_bin=128, max_depth=8, n_estimators=400, num_leaves=128, objective="mae", reg_alpha=1, reg_lambda=64, subsample=0.6)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
