import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn import metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('articulator.csv', dtype=np.float64)
features = tpot_data.drop('extentarticulator', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['extentarticulator'], random_state=49)

# Average CV score on the training set was: 0.10221262902540063
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=54),
    FeatureAgglomeration(affinity="manhattan", linkage="complete"),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=1.0, loss="ls", max_depth=6, max_features=0.5, min_samples_leaf=18, min_samples_split=7, n_estimators=100, subsample=0.05)),
    KNeighborsRegressor(n_neighbors=10, p=2, weights="uniform")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 49)

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

# features = [0, 1, (0, 1)]
# plot_partial_dependence(model, training_features, features)
# fig = plt.gcf()
# fig.savefig('demo.pdf', bbox_inches='tight')