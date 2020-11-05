import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=232)

# Average CV score on the training set was: 0.3211803357979871
exported_pipeline = make_pipeline(
    ZeroCount(),
    StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    MinMaxScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=18, min_samples_split=16)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=11)),
    ZeroCount(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, min_samples_split=3)),
    SelectFwe(score_func=f_regression, alpha=0.006),
    SelectPercentile(score_func=f_regression, percentile=68),
    AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 232)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
