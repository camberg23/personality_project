import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=232)

# Average CV score on the training set was: 0.26666028551716264
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9, tol=0.1)),
    PCA(iterated_power=4, svd_solver="randomized"),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=19, min_samples_split=4)),
    AdaBoostRegressor(learning_rate=0.001, loss="exponential", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 232)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
