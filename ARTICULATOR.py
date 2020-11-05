import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.2178961519064649
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.1, loss="epsilon_insensitive", tol=0.001)),
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=1.0)),
            FeatureAgglomeration(affinity="manhattan", linkage="complete"),
            StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="exponential", n_estimators=100)),
            StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=13, p=2, weights="distance")),
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.4, tol=1e-05)),
            KNeighborsRegressor(n_neighbors=13, p=1, weights="distance")
        ))
    ),
    DecisionTreeRegressor(max_depth=2, min_samples_leaf=20, min_samples_split=2)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
