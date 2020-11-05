import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.inspection import permutation_importance
from sklearn import metrics

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('capitalism.csv', dtype=np.float64)
features = tpot_data.drop('extentcapitalism', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['extentcapitalism'], random_state=53)

# Average CV score on the training set was: 0.15931468524082187
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=26),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=1, min_samples_split=6)),
    DecisionTreeRegressor(max_depth=1, min_samples_leaf=13, min_samples_split=20)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 53)

model = exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print("r2:",metrics.r2_score(testing_target, results))


r = permutation_importance(model, training_features, training_target,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{tpot_data.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

