import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.inspection import permutation_importance
from sklearn import metrics

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('spiritual.csv', dtype=np.float64)
features = tpot_data.drop('extentspiritual', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['extentspiritual'], random_state=26)

# Average CV score on the training set was: 0.11861258284551135
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=make_pipeline(
            make_union(
                make_union(
                    FunctionTransformer(copy),
                    FunctionTransformer(copy)
                ),
                StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=20, min_samples_split=11, n_estimators=100))
            ),
            ElasticNetCV(l1_ratio=0.4, tol=0.01)
        ))
    ),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.01, fit_intercept=True, l1_ratio=0.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=50.0)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.1, loss="square", n_estimators=100)),
    FeatureAgglomeration(affinity="cosine", linkage="complete"),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=45, p=1, weights="uniform")),
    ElasticNetCV(l1_ratio=1.0, tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 26)

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