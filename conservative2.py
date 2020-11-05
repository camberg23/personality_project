import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.1398423136070082
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.01)),
    FeatureAgglomeration(affinity="cosine", linkage="complete"),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=35, p=2, weights="uniform")),
    PCA(iterated_power=6, svd_solver="randomized"),
    ZeroCount(),
    MinMaxScaler(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=10, max_features=0.9000000000000001, min_samples_leaf=5, min_samples_split=5, n_estimators=100, subsample=0.15000000000000002)),
    MinMaxScaler(),
    LinearSVR(C=25.0, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
